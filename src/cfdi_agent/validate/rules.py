"""Deterministic validation and anomaly detection.

Every function here is pure: no database, no network, no model. Historical
context arrives as a `HistoryContext` the caller assembles. That is what lets
the whole detector suite run in a unit test in milliseconds, and it is why the
tier-0 pipeline needs no credentials.

Accept vs. reject
-----------------
An invoice is *rejected* (routed to `review_queue`, never written to
`invoices`) only when we cannot trust what we extracted, or the document is not
ours. An invoice whose arithmetic is wrong is **accepted and flagged**: the
whole point is to record it, alert on it, and chase the supplier. Dropping it
into a review queue would lose the structured data and turn a detected problem
into an invisible one.

That is a deliberate narrowing of "nothing is persisted without balancing."
The rule that actually holds is: *nothing is persisted that we have not fully
validated* — where "validated" means we know precisely what is wrong with it,
not that nothing is wrong.

Detector #2 (semantic duplicate) lives in `enrich.anomalies`, not here: it
needs vector similarity, so it cannot be a pure function over one invoice.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from cfdi_agent.schemas import RFC_RE, ParsedInvoice
from cfdi_agent.validate import catalogs

Severity = Literal["info", "warn", "critical"]

# CFDI amounts are cent-precision; a one-cent gap is rounding, not fraud.
MONEY_TOLERANCE = Decimal("0.01")

# Robust z-score cutoff for the price outlier detector.
PRICE_Z_THRESHOLD = Decimal("3.5")
PRICE_MIN_SAMPLES = 5

# Two guards, both learned from measuring the detector rather than reasoning
# about it. Measured on a 120-invoice corpus, the z-score alone gave recall
# 1.00 but precision 0.22 — it fired on 7 of 90 clean invoices.
#
# 1. MAD floor. With only 5-8 samples the MAD is unstable: a run that happens
#    to cluster tightly produces a near-zero MAD, and then ordinary ±3% price
#    drift sits many MADs from the median. Flooring the MAD at a fraction of
#    the median stops a lucky-tight sample from making everything an outlier.
MAD_FLOOR_RATIO = Decimal("0.02")
#
# 2. Materiality gate. A price must be *statistically* unusual AND actually
#    far from the historical price. Nobody wants an alert because a supplier
#    raised a unit price 4%, however clean the statistics look.
PRICE_MIN_RATIO_DEVIATION = Decimal("0.5")  # at least ±50% off the median

# Fallback when every historical price is identical (MAD == 0): a bare ratio.
PRICE_RATIO_THRESHOLD = Decimal("3")

MAD_TO_SIGMA = Decimal("0.6745")


@dataclass(frozen=True, slots=True)
class Anomaly:
    kind: str
    severity: Severity
    detail: dict
    # The exact values that fired the rule. Written to `anomalies.evidence`;
    # the LLM may summarize this but never invent an anomaly without it.
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PriceStats:
    """Historical unit prices for one (supplier, product) pair."""

    median: Decimal
    mad: Decimal
    n: int
    recent: tuple[Decimal, ...] = ()


@dataclass(frozen=True, slots=True)
class HistoryContext:
    """Everything the detectors need to know about the past.

    Assembled by the persistence layer from the database. Empty by default so
    a caller can validate a single invoice in isolation — the history-dependent
    detectors simply stay quiet rather than firing spuriously.
    """

    # Whether this context was read from the ledger at all.
    #
    # An empty history and an unread history look identical from the fields
    # alone, and for most detectors that does not matter: no known folio means
    # no gap either way. `new_supplier` inverts it — with a genuinely empty
    # ledger, every supplier is new. Without this flag it stayed silent on the
    # first invoice ever loaded, which is exactly the invoice a first-day
    # operator is watching.
    #
    # Third time this project has needed the distinction between "not recorded"
    # and "recorded as nothing" (see `seen_folios` and `processed_files`). It is
    # the shape of bug this domain keeps producing.
    loaded: bool = False

    known_uuids: frozenset[str] = frozenset()
    known_rfcs: frozenset[str] = frozenset()
    # (rfc_emisor, serie) -> highest folio seen so far
    last_folio: Mapping[tuple[str, str | None], int] = field(default_factory=dict)
    # (rfc_emisor, clave_prod_serv) -> price history
    price_stats: Mapping[tuple[str, str], PriceStats] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    accepted: bool
    reject_reason: str | None
    anomalies: tuple[Anomaly, ...]

    @property
    def worst_severity(self) -> Severity | None:
        for level in ("critical", "warn", "info"):
            if any(a.severity == level for a in self.anomalies):
                return level  # type: ignore[return-value]
        return None


# --------------------------------------------------------------------------
# Individual detectors
# --------------------------------------------------------------------------


def detect_duplicate_uuid(inv: ParsedInvoice, ctx: HistoryContext) -> Anomaly | None:
    """#1 — the same fiscal document submitted twice.

    A UUID is unique per stamped CFDI nationwide, so a repeat is unambiguous:
    either a resend or a double-billing attempt. The DB unique constraint is
    the real backstop; this detector exists so the reason is recorded rather
    than surfacing as an integrity error.
    """
    if inv.uuid in ctx.known_uuids:
        return Anomaly(
            kind="duplicate_uuid",
            severity="critical",
            detail={"uuid": inv.uuid, "rfc_emisor": inv.rfc_emisor},
            evidence={"uuid": inv.uuid, "already_recorded": True},
        )
    return None


def detect_invalid_rfc(inv: ParsedInvoice) -> list[Anomaly]:
    """#5 — structurally malformed RFC.

    Structure only. The RFC check digit is *not* verified: the algorithm is
    fiddly, an incorrect implementation would reject valid taxpayers, and
    getting it wrong is worse than not having it. Documented as a gap in the
    README rather than shipped unverified.
    """
    out: list[Anomaly] = []
    for role, rfc in (("emisor", inv.rfc_emisor), ("receptor", inv.rfc_receptor)):
        if not RFC_RE.match(rfc):
            out.append(
                Anomaly(
                    kind="invalid_rfc",
                    severity="critical",
                    detail={"role": role, "rfc": rfc},
                    evidence={"rfc": rfc, "pattern": RFC_RE.pattern},
                )
            )
    return out


def detect_arithmetic(inv: ParsedInvoice) -> list[Anomaly]:
    """#4 — the invoice does not add up.

    Three independent checks, because they fail for different reasons:
      * a line whose importe != cantidad * valor_unitario
      * a subtotal that is not the sum of the lines
      * a total that is not subtotal - descuento + traslados - retenciones
    """
    out: list[Anomaly] = []

    bad_lines = [
        {
            "line_no": c.line_no,
            "descripcion": c.descripcion,
            "cantidad": str(c.cantidad),
            "valor_unitario": str(c.valor_unitario),
            "importe_reportado": str(c.importe),
            "importe_calculado": str(c.importe_esperado),
            "diff": str(c.importe - c.importe_esperado),
        }
        for c in inv.conceptos
        if abs(c.importe - c.importe_esperado) > MONEY_TOLERANCE
    ]
    if bad_lines:
        out.append(
            Anomaly(
                kind="line_math_mismatch",
                severity="critical",
                detail={"n_lines": len(bad_lines)},
                evidence={"lines": bad_lines},
            )
        )

    if inv.conceptos and abs(inv.subtotal - inv.subtotal_esperado) > MONEY_TOLERANCE:
        out.append(
            Anomaly(
                kind="subtotal_mismatch",
                severity="critical",
                detail={
                    "subtotal_reportado": str(inv.subtotal),
                    "subtotal_calculado": str(inv.subtotal_esperado),
                },
                evidence={
                    "suma_importes": str(inv.subtotal_esperado),
                    "diff": str(inv.subtotal - inv.subtotal_esperado),
                },
            )
        )

    if abs(inv.total - inv.total_esperado) > MONEY_TOLERANCE:
        out.append(
            Anomaly(
                kind="total_mismatch",
                severity="critical",
                detail={
                    "total_reportado": str(inv.total),
                    "total_calculado": str(inv.total_esperado),
                },
                evidence={
                    "subtotal": str(inv.subtotal),
                    "descuento": str(inv.descuento),
                    "traslados": str(inv.traslados),
                    "retenciones": str(inv.retenciones),
                    "diff": str(inv.total - inv.total_esperado),
                },
            )
        )
    return out


def detect_new_supplier(inv: ParsedInvoice, ctx: HistoryContext) -> Anomaly | None:
    """#6 — first invoice ever from this RFC.

    Informational on its own. It matters as corroboration: a new supplier plus
    an odd amount is a very different signal from either alone.
    """
    if (ctx.loaded or ctx.known_rfcs) and inv.rfc_emisor not in ctx.known_rfcs:
        return Anomaly(
            kind="new_supplier",
            severity="info",
            detail={"rfc_emisor": inv.rfc_emisor, "nombre": inv.nombre_emisor},
            evidence={"known_supplier_count": len(ctx.known_rfcs)},
        )
    return None


def detect_folio_gap(inv: ParsedInvoice, ctx: HistoryContext) -> Anomaly | None:
    """#7 — a jump in the supplier's folio sequence.

    Usually benign (the supplier invoices other customers too), which is why
    this is `warn` and not `critical`. It earns its place because a gap that
    coincides with a duplicate or a price spike is a much stronger signal.

    This is the least precise detector in the suite and the number is published
    rather than buried. Measured on a 300-invoice corpus: 28 firings against 13
    injected gaps, precision ~0.43. Every remaining false positive traces to an
    invoice whose issuer RFC was malformed — it files under a different RFC, so
    the real supplier's sequence shows a hole.

    That is arguably the correct verdict (an unreadable RFC genuinely breaks
    supplier identity) and it is inflated here by the corpus, which corrupts
    RFCs at ~4%; real inboxes are far cleaner. An earlier version was much
    worse at precision 0.30, because the watermark was read from `invoices` and
    so counted documents *we* declined to insert as supplier gaps. See
    `seen_folios` in schema.sql.
    """
    if inv.folio is None:
        return None
    try:
        folio = int(inv.folio)
    except ValueError:
        # Alphanumeric folios exist and are legal; sequence checking simply
        # does not apply to them.
        return None

    previous = ctx.last_folio.get((inv.rfc_emisor, inv.serie))
    if previous is None or folio <= previous + 1:
        return None
    return Anomaly(
        kind="folio_gap",
        severity="warn",
        detail={
            "rfc_emisor": inv.rfc_emisor,
            "serie": inv.serie,
            "folio_anterior": previous,
            "folio_actual": folio,
            "faltantes": folio - previous - 1,
        },
        evidence={"expected_next": previous + 1, "got": folio},
    )


def robust_z(value: Decimal, stats: PriceStats) -> Decimal | None:
    """MAD-based z-score, with the MAD floored. None if the sample is too thin.

    The floor is what makes this usable on the 5-20 sample histories a real
    supplier relationship produces; see `MAD_FLOOR_RATIO`.
    """
    if stats.n < PRICE_MIN_SAMPLES:
        return None
    mad = max(stats.mad, abs(stats.median) * MAD_FLOOR_RATIO)
    if mad == 0:
        return None
    return (MAD_TO_SIGMA * (value - stats.median) / mad).quantize(Decimal("0.01"))


def detect_price_outlier(inv: ParsedInvoice, ctx: HistoryContext) -> list[Anomaly]:
    """#3 — a unit price far outside what this supplier has charged before.

    Median + MAD rather than mean + standard deviation: with a handful of
    samples, a single previous overcharge would inflate the standard deviation
    enough to hide the next one. The median is unmoved by it.
    """
    out: list[Anomaly] = []
    for c in inv.conceptos:
        if not c.clave_prod_serv:
            continue
        stats = ctx.price_stats.get((inv.rfc_emisor, c.clave_prod_serv))
        if stats is None or stats.n < PRICE_MIN_SAMPLES:
            continue

        z = robust_z(c.valor_unitario, stats)
        ratio = (c.valor_unitario / stats.median) if stats.median > 0 else None

        # Materiality gate, applied before anything else: a price within ±50%
        # of the historical median is not an outlier no matter what the
        # statistics say. This is the guard that took precision from 0.22 to
        # 1.00 on the corpus without costing any recall.
        material = ratio is not None and abs(ratio - 1) >= PRICE_MIN_RATIO_DEVIATION
        if not material:
            continue

        triggered = False
        if z is not None and abs(z) > PRICE_Z_THRESHOLD:
            triggered = True
        elif stats.mad == 0 and ratio is not None and ratio > PRICE_RATIO_THRESHOLD:
            # Perfectly stable price history — any large multiple is notable.
            triggered = True

        if triggered:
            out.append(
                Anomaly(
                    kind="price_outlier",
                    severity="warn",
                    detail={
                        "line_no": c.line_no,
                        "descripcion": c.descripcion,
                        "clave_prod_serv": c.clave_prod_serv,
                        "valor_unitario": str(c.valor_unitario),
                        "mediana_historica": str(stats.median),
                        "ratio": str(ratio.quantize(Decimal("0.01"))) if ratio else None,
                    },
                    evidence={
                        "robust_z": str(z) if z is not None else None,
                        "mad": str(stats.mad),
                        "n_samples": stats.n,
                        "recent_prices": [str(p) for p in stats.recent],
                    },
                )
            )
    return out


def detect_catalog_issues(inv: ParsedInvoice) -> list[Anomaly]:
    """Advisory notes for codes outside the bundled catalog subset.

    Always `info`: the subset is incomplete by design (see `catalogs`), so an
    unknown code means "not in our list", never "invalid".
    """
    unknown: list[dict] = []
    if inv.uso_cfdi and not catalogs.is_known(catalogs.USOS_CFDI, inv.uso_cfdi):
        unknown.append({"field": "uso_cfdi", "value": inv.uso_cfdi})
    if inv.forma_pago and not catalogs.is_known(catalogs.FORMAS_PAGO, inv.forma_pago):
        unknown.append({"field": "forma_pago", "value": inv.forma_pago})
    if inv.metodo_pago and not catalogs.is_known(catalogs.METODOS_PAGO, inv.metodo_pago):
        unknown.append({"field": "metodo_pago", "value": inv.metodo_pago})
    for c in inv.conceptos:
        if c.clave_unidad and not catalogs.is_known(catalogs.CLAVES_UNIDAD, c.clave_unidad):
            unknown.append(
                {"field": "clave_unidad", "value": c.clave_unidad, "line_no": c.line_no}
            )
    if not unknown:
        return []
    return [
        Anomaly(
            kind="unknown_catalog_code",
            severity="info",
            detail={"count": len(unknown)},
            evidence={"codes": unknown, "note": "bundled SAT catalog subset is partial"},
        )
    ]


def detect_stale_stamp(inv: ParsedInvoice) -> Anomaly | None:
    """Stamped far later than issued.

    The SAT requires stamping within 72 hours of issue. A large gap usually
    means a backdated invoice.
    """
    if inv.fecha_timbrado is None:
        return None
    emision = inv.fecha_emision
    timbrado = inv.fecha_timbrado
    # Mixed awareness happens: Fecha is naive local, FechaTimbrado may carry Z.
    if (emision.tzinfo is None) != (timbrado.tzinfo is None):
        return None
    hours = (timbrado - emision).total_seconds() / 3600
    if hours <= 72:
        return None
    return Anomaly(
        kind="stale_stamp",
        severity="warn",
        detail={
            "fecha_emision": emision.isoformat(),
            "fecha_timbrado": timbrado.isoformat(),
            "horas": round(hours, 1),
        },
        evidence={"limite_horas": 72, "excedente_horas": round(hours - 72, 1)},
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def validate_invoice(
    inv: ParsedInvoice,
    ctx: HistoryContext | None = None,
    *,
    company_rfc: str | None = None,
) -> ValidationResult:
    """Run every pure detector and decide accept vs. reject."""
    ctx = ctx or HistoryContext()

    # Rejection is about trust and ownership, not correctness.
    if company_rfc and inv.rfc_receptor.upper() != company_rfc.upper():
        return ValidationResult(
            accepted=False,
            reject_reason=(
                f"invoice is addressed to {inv.rfc_receptor}, not to {company_rfc}"
            ),
            anomalies=(),
        )

    anomalies: list[Anomaly] = []
    if a := detect_duplicate_uuid(inv, ctx):
        anomalies.append(a)
    anomalies.extend(detect_invalid_rfc(inv))
    anomalies.extend(detect_arithmetic(inv))
    if a := detect_new_supplier(inv, ctx):
        anomalies.append(a)
    if a := detect_folio_gap(inv, ctx):
        anomalies.append(a)
    anomalies.extend(detect_price_outlier(inv, ctx))
    if a := detect_stale_stamp(inv):
        anomalies.append(a)
    anomalies.extend(detect_catalog_issues(inv))

    return ValidationResult(accepted=True, reject_reason=None, anomalies=tuple(anomalies))


def price_stats_from_samples(samples: Sequence[Decimal]) -> PriceStats:
    """Build `PriceStats` from a list of historical unit prices."""
    if not samples:
        return PriceStats(median=Decimal("0"), mad=Decimal("0"), n=0)
    ordered = sorted(samples)
    median = Decimal(str(statistics.median(ordered)))
    deviations = sorted(abs(s - median) for s in ordered)
    mad = Decimal(str(statistics.median(deviations)))
    return PriceStats(
        median=median,
        mad=mad,
        n=len(ordered),
        recent=tuple(samples[-5:]),
    )
