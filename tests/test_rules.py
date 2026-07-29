"""Detector tests.

Two layers. Unit tests pin each detector's behaviour on a hand-built invoice.
The corpus test then replays the whole generated dataset chronologically —
accumulating history exactly as the ingest pipeline does — and checks that the
injected defects are actually caught and that clean invoices stay quiet.

That second layer is the one that matters. A detector that fires on its unit
fixture but drowns clean invoices in false positives is worthless, and only a
corpus-level precision check will tell you.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from cfdi_agent.extract.xml_parser import parse_cfdi_file
from cfdi_agent.schemas import Concepto, ParsedInvoice
from cfdi_agent.validate.rules import (
    HistoryContext,
    detect_arithmetic,
    detect_duplicate_uuid,
    detect_folio_gap,
    detect_invalid_rfc,
    detect_new_supplier,
    detect_price_outlier,
    detect_stale_stamp,
    price_stats_from_samples,
    validate_invoice,
)
from synth.generate_cfdi import generate

UUID_A = "A1B2C3D4-E5F6-4718-9A0B-1C2D3E4F5061"


def make_invoice(**overrides) -> ParsedInvoice:
    """A clean, balanced invoice: 2 x 100.00 + 16% IVA = 232.00."""
    base = {
        "uuid": UUID_A,
        "serie": "A",
        "folio": "100",
        "fecha_emision": datetime(2026, 3, 1, 10, 0, 0),
        "fecha_timbrado": datetime(2026, 3, 1, 10, 30, 0),
        "rfc_emisor": "AAA010101AAA",
        "rfc_receptor": "XAXX010101000",
        "subtotal": "200.00",
        "total": "232.00",
        "conceptos": [
            Concepto(
                line_no=1,
                clave_prod_serv="01010101",
                clave_unidad="H87",
                descripcion="Servicio",
                cantidad="2",
                valor_unitario="100",
                importe="200.00",
            )
        ],
        "impuestos": [
            {"tipo": "traslado", "impuesto": "002", "base": "200.00",
             "tasa": "0.16", "importe": "32.00"}
        ],
        "uso_cfdi": "G03",
        "forma_pago": "03",
        "metodo_pago": "PUE",
    }
    base.update(overrides)
    return ParsedInvoice(**base)


# ----------------------------------------------------------------- baseline


def test_clean_invoice_produces_no_anomalies() -> None:
    result = validate_invoice(make_invoice(), company_rfc="XAXX010101000")
    assert result.accepted
    assert result.anomalies == ()


def test_invoice_for_another_company_is_rejected() -> None:
    """Not an anomaly — simply not our document."""
    result = validate_invoice(
        make_invoice(rfc_receptor="BBB020202BB2"), company_rfc="XAXX010101000"
    )
    assert not result.accepted
    assert "not to XAXX010101000" in result.reject_reason


# ------------------------------------------------------------ #1 duplicates


def test_duplicate_uuid_detected() -> None:
    ctx = HistoryContext(known_uuids=frozenset({UUID_A}))
    assert detect_duplicate_uuid(make_invoice(), ctx).kind == "duplicate_uuid"


def test_unseen_uuid_is_quiet() -> None:
    ctx = HistoryContext(known_uuids=frozenset({"FFFFFFFF-0000-4000-8000-000000000000"}))
    assert detect_duplicate_uuid(make_invoice(), ctx) is None


# ------------------------------------------------------------ #4 arithmetic


def test_total_mismatch_detected() -> None:
    kinds = {a.kind for a in detect_arithmetic(make_invoice(total="999.00"))}
    assert "total_mismatch" in kinds


def test_line_math_mismatch_detected() -> None:
    inv = make_invoice(
        conceptos=[
            Concepto(
                line_no=1, descripcion="X", cantidad="2",
                valor_unitario="100", importe="250.00",  # 2*100 != 250
            )
        ],
        subtotal="250.00",
        total="290.00",
        impuestos=[{"tipo": "traslado", "impuesto": "002", "importe": "40.00"}],
    )
    kinds = {a.kind for a in detect_arithmetic(inv)}
    assert "line_math_mismatch" in kinds


def test_one_cent_rounding_is_tolerated() -> None:
    """Cent-level noise is rounding, not fraud — must not fire."""
    assert detect_arithmetic(make_invoice(total="232.01")) == []


def test_arithmetic_evidence_carries_the_numbers() -> None:
    """Evidence must let a human verify without re-opening the XML."""
    anomaly = next(
        a for a in detect_arithmetic(make_invoice(total="999.00"))
        if a.kind == "total_mismatch"
    )
    assert anomaly.evidence["subtotal"] == "200.00"
    assert anomaly.evidence["traslados"] == "32.00"
    assert Decimal(anomaly.evidence["diff"]) == Decimal("767.00")


# -------------------------------------------------------------------- #5 RFC


@pytest.mark.parametrize(
    "bad",
    [
        "AAA010101AA-",   # invalid character
        "TOOSHORT",
        "AAA0101",
        "aaa010101!!!",
        "AAA011301AA1",   # month 13
        "AAA010132AA1",   # day 32
        "AAA010101AAZ",   # check digit must be [0-9A]; Z cannot occur
    ],
)
def test_malformed_rfc_detected(bad: str) -> None:
    anomalies = detect_invalid_rfc(make_invoice(rfc_emisor=bad))
    assert any(a.kind == "invalid_rfc" for a in anomalies), bad


@pytest.mark.parametrize(
    "good",
    [
        "AAA010101AAA",   # persona moral, check digit 'A'
        "CABL850315HN3",  # persona física, 13 chars
        "AÑE900101XY9",   # Ñ is legal in the name block
    ],
)
def test_valid_rfc_shapes_accepted(good: str) -> None:
    assert detect_invalid_rfc(make_invoice(rfc_emisor=good)) == [], good


# ---------------------------------------------------------- #6 new supplier


def test_new_supplier_flagged_as_info() -> None:
    ctx = HistoryContext(known_rfcs=frozenset({"ZZZ990909ZZ9"}))
    anomaly = detect_new_supplier(make_invoice(), ctx)
    assert anomaly.severity == "info"


def test_an_unread_history_does_not_flag_every_supplier() -> None:
    """A default context means nobody looked, not that nobody was there.

    This test used to read "on a cold database everyone is new; that is noise",
    and that reasoning was wrong in a way that cost the first invoice of every
    ledger. On a genuinely cold database everyone *is* new, and saying so is
    the correct `info` — see `TestNewSupplierOnAColdLedger`. What must stay
    silent is a context nobody populated, which is what `loaded` now separates.
    """
    assert detect_new_supplier(make_invoice(), HistoryContext()) is None


# ------------------------------------------------------------- #7 folio gap


def test_folio_gap_detected() -> None:
    ctx = HistoryContext(last_folio={("AAA010101AAA", "A"): 95})
    anomaly = detect_folio_gap(make_invoice(folio="100"), ctx)
    assert anomaly.detail["faltantes"] == 4


def test_consecutive_folio_is_quiet() -> None:
    ctx = HistoryContext(last_folio={("AAA010101AAA", "A"): 99})
    assert detect_folio_gap(make_invoice(folio="100"), ctx) is None


def test_alphanumeric_folio_is_skipped() -> None:
    """Alphanumeric folios are legal; sequence checking does not apply."""
    ctx = HistoryContext(last_folio={("AAA010101AAA", "A"): 95})
    assert detect_folio_gap(make_invoice(folio="F-2026-X"), ctx) is None


# --------------------------------------------------------- #3 price outlier


def test_price_outlier_detected() -> None:
    stats = price_stats_from_samples([Decimal(x) for x in ("98", "101", "99", "100", "102")])
    ctx = HistoryContext(price_stats={("AAA010101AAA", "01010101"): stats})
    inv = make_invoice(
        conceptos=[
            Concepto(
                line_no=1, clave_prod_serv="01010101", descripcion="Servicio",
                cantidad="1", valor_unitario="400", importe="400.00",
            )
        ]
    )
    anomalies = detect_price_outlier(inv, ctx)
    assert [a.kind for a in anomalies] == ["price_outlier"]
    assert anomalies[0].evidence["n_samples"] == 5


def test_normal_price_drift_is_quiet() -> None:
    """Ordinary movement must not fire, or precision collapses."""
    stats = price_stats_from_samples([Decimal(x) for x in ("98", "101", "99", "100", "102")])
    ctx = HistoryContext(price_stats={("AAA010101AAA", "01010101"): stats})
    assert detect_price_outlier(make_invoice(), ctx) == []


def test_tight_history_does_not_make_small_drift_an_outlier() -> None:
    """Regression: the false-positive case that gave precision 0.22.

    A history that happens to cluster tightly produces a near-zero MAD, so a
    4% price increase scored a huge robust-z and fired. Both the MAD floor and
    the materiality gate exist to stop exactly this.
    """
    stats = price_stats_from_samples(
        [Decimal(x) for x in ("100.00", "100.10", "99.95", "100.05", "100.02", "99.98")]
    )
    ctx = HistoryContext(price_stats={("AAA010101AAA", "01010101"): stats})
    inv = make_invoice(
        conceptos=[
            Concepto(
                line_no=1, clave_prod_serv="01010101", descripcion="S",
                cantidad="1", valor_unitario="104.00", importe="104.00",
            )
        ]
    )
    assert detect_price_outlier(inv, ctx) == []


def test_material_spike_on_tight_history_still_fires() -> None:
    """The guard must not have bought precision by giving up recall."""
    stats = price_stats_from_samples(
        [Decimal(x) for x in ("100.00", "100.10", "99.95", "100.05", "100.02", "99.98")]
    )
    ctx = HistoryContext(price_stats={("AAA010101AAA", "01010101"): stats})
    inv = make_invoice(
        conceptos=[
            Concepto(
                line_no=1, clave_prod_serv="01010101", descripcion="S",
                cantidad="1", valor_unitario="380.00", importe="380.00",
            )
        ]
    )
    assert [a.kind for a in detect_price_outlier(inv, ctx)] == ["price_outlier"]


def test_thin_history_suppresses_the_detector() -> None:
    """Fewer than 5 samples cannot support an outlier verdict."""
    stats = price_stats_from_samples([Decimal("100"), Decimal("101")])
    ctx = HistoryContext(price_stats={("AAA010101AAA", "01010101"): stats})
    inv = make_invoice(
        conceptos=[
            Concepto(
                line_no=1, clave_prod_serv="01010101", descripcion="S",
                cantidad="1", valor_unitario="9999", importe="9999.00",
            )
        ]
    )
    assert detect_price_outlier(inv, ctx) == []


def test_flat_price_history_uses_ratio_fallback() -> None:
    """MAD == 0 would divide by zero; the ratio path must still catch it."""
    stats = price_stats_from_samples([Decimal("100")] * 6)
    assert stats.mad == 0
    ctx = HistoryContext(price_stats={("AAA010101AAA", "01010101"): stats})
    inv = make_invoice(
        conceptos=[
            Concepto(
                line_no=1, clave_prod_serv="01010101", descripcion="S",
                cantidad="1", valor_unitario="500", importe="500.00",
            )
        ]
    )
    assert [a.kind for a in detect_price_outlier(inv, ctx)] == ["price_outlier"]


# ------------------------------------------------------------- stale stamp


def test_stale_stamp_detected() -> None:
    inv = make_invoice(fecha_timbrado=datetime(2026, 3, 1, 10, 0) + timedelta(hours=100))
    assert detect_stale_stamp(inv).kind == "stale_stamp"


def test_prompt_stamp_is_quiet() -> None:
    assert detect_stale_stamp(make_invoice()) is None


# --------------------------------------------------------------------------
# Corpus-level: does the suite actually work end to end?
# --------------------------------------------------------------------------


def _replay(out: Path, labels: list[dict]) -> list[tuple[dict, set[str]]]:
    """Replay the corpus chronologically, accumulating history as we go.

    Mirrors what the ingest pipeline does: each invoice is validated against
    the state produced by every invoice before it.
    """
    known_uuids: set[str] = set()
    known_rfcs: set[str] = set()
    last_folio: dict[tuple[str, str | None], int] = {}
    price_samples: dict[tuple[str, str], list[Decimal]] = {}
    results: list[tuple[dict, set[str]]] = []

    for label in labels:
        inv = parse_cfdi_file(out / label["file"])
        ctx = HistoryContext(
            known_uuids=frozenset(known_uuids),
            known_rfcs=frozenset(known_rfcs),
            last_folio=dict(last_folio),
            price_stats={
                key: price_stats_from_samples(samples)
                for key, samples in price_samples.items()
            },
        )
        result = validate_invoice(inv, ctx, company_rfc="XAXX010101000")
        results.append((label, {a.kind for a in result.anomalies}))

        # Update history *after* validating, exactly as ingest must.
        known_uuids.add(inv.uuid)
        known_rfcs.add(inv.rfc_emisor)
        if inv.folio and inv.folio.isdigit():
            key = (inv.rfc_emisor, inv.serie)
            last_folio[key] = max(last_folio.get(key, 0), int(inv.folio))
        for c in inv.conceptos:
            if c.clave_prod_serv:
                price_samples.setdefault((inv.rfc_emisor, c.clave_prod_serv), []).append(
                    c.valor_unitario
                )
    return results


@pytest.fixture(scope="module")
def replayed(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("corpus")
    labels = generate(
        n=120,
        defect_rate=0.35,
        out_dir=out,
        labels_path=out / "labeled.jsonl",
        seed=99,
        n_suppliers=6,
        receptor_rfc="XAXX010101000",
        receptor_nombre="Mi Empresa SA de CV",
    )
    return _replay(out, labels)


# Which detector kinds count as catching which injected defect.
DEFECT_TO_KINDS = {
    "total_mismatch": {"total_mismatch"},
    "line_math": {"line_math_mismatch", "subtotal_mismatch", "total_mismatch"},
    "bad_rfc": {"invalid_rfc"},
    "dup_uuid": {"duplicate_uuid"},
    "folio_gap": {"folio_gap"},
    "price_spike": {"price_outlier"},
}


@pytest.mark.parametrize("defect", sorted(DEFECT_TO_KINDS))
def test_injected_defects_are_caught(replayed, defect: str) -> None:
    affected = [(lb, kinds) for lb, kinds in replayed if defect in lb["defects"]]
    if not affected:
        pytest.skip(f"no {defect} injected at this seed")
    missed = [lb["file"] for lb, kinds in affected if not (kinds & DEFECT_TO_KINDS[defect])]
    recall = 1 - len(missed) / len(affected)
    assert recall >= 0.95, f"{defect}: recall {recall:.2f}, missed {missed}"


def test_clean_invoices_raise_no_critical_anomaly(replayed) -> None:
    """Precision guard: a defect-free invoice must never look critical.

    `info` (new supplier, unknown catalog code) and `warn` are acceptable on
    clean invoices — they are context, not accusations. A `critical` on a clean
    invoice is a false positive that would erode trust in every alert.
    """
    critical = {
        "total_mismatch", "subtotal_mismatch", "line_math_mismatch",
        "invalid_rfc", "duplicate_uuid",
    }
    offenders = [
        lb["file"] for lb, kinds in replayed if not lb["defects"] and (kinds & critical)
    ]
    assert not offenders, f"false positives on clean invoices: {offenders}"


class TestNewSupplierOnAColdLedger:
    """The distinction between an unread history and an empty one.

    `new_supplier` is the only detector where the two invert: with a genuinely
    empty ledger every supplier is new, while an unread history must stay
    silent. Guarding on `known_rfcs` alone conflates them, and the invoice that
    goes missing is the first one ever processed — the one a first-day operator
    is watching.
    """

    def test_the_very_first_invoice_reports_a_new_supplier(self) -> None:
        from cfdi_agent.validate.rules import HistoryContext, detect_new_supplier

        inv = make_invoice()
        found = detect_new_supplier(inv, HistoryContext(loaded=True))
        assert found is not None
        assert found.kind == "new_supplier"
        assert found.evidence["known_supplier_count"] == 0

    def test_an_unread_history_stays_silent(self) -> None:
        """Validating one invoice in isolation must not accuse anybody."""
        from cfdi_agent.validate.rules import HistoryContext, detect_new_supplier

        assert detect_new_supplier(make_invoice(), HistoryContext()) is None

    def test_a_known_supplier_is_not_new(self) -> None:
        from cfdi_agent.validate.rules import HistoryContext, detect_new_supplier

        inv = make_invoice()
        ctx = HistoryContext(loaded=True, known_rfcs=frozenset({inv.rfc_emisor}))
        assert detect_new_supplier(inv, ctx) is None
