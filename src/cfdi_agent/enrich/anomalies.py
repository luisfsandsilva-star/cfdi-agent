"""Detector #2: the semantic duplicate.

The one detector that cannot be a pure function, which is why it lives here
rather than in `validate.rules`: it needs vector similarity over the whole
ledger.

What it catches that #1 (duplicate UUID) does not: the same work billed twice
under two legitimately distinct invoices. Different UUID, different folio,
maybe a reworded description — same supplier, same amount, same week. That is
the expensive kind of double-billing, and a UUID check is blind to it.

Deliberately conservative, in this order:

1. Cheap SQL filter first — same issuer, total within 1%, date within 7 days.
   Running a vector search across every invoice to find candidates would be
   both slow and needlessly broad.
2. Cosine similarity over line-item centroids, and only then a verdict.

An invoice legitimately repeats: a monthly retainer is the same supplier, the
same amount, the same description. The 7-day window is what separates "billed
twice this week" from "billed again this month", and it is the parameter to
tune first if this ever gets noisy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Any

from psycopg import Connection

from cfdi_agent.validate.rules import Anomaly

# Same supplier, near-identical amount, close in time.
TOTAL_TOLERANCE = Decimal("0.01")  # 1%
DATE_WINDOW = timedelta(days=7)
# Cosine similarity over line-item centroids, measured rather than guessed.
#
# This was 0.93, picked from intuition, and it caught nothing: bge-m3 puts a
# reworded line item between 0.715 and 0.910, so every real duplicate fell
# under the bar. The stub embedder in the tests passed because it was built to
# pass — a threshold is not tested by a fixture you designed around it.
#
# Measured on the product catalog (10 rewordings, 135 unrelated pairs):
#
#     same product, reworded    0.715 .. 0.910
#     different products        0.253 .. 0.684
#
# 0.70 separates both sets completely. The margin is only 0.031, so on a wider
# vocabulary the sets would likely overlap — this number is not load-bearing on
# its own. The SQL pre-filter does the heavy lifting: same issuer, total within
# 1%, date within 7 days. Cosine is the last confirmation, not the first cut.
SIMILARITY_THRESHOLD = 0.70


@dataclass(frozen=True, slots=True)
class DuplicateCandidate:
    invoice_id: int
    uuid: str
    folio: str | None
    fecha_emision: str
    total: Decimal
    similarity: float


def find_semantic_duplicates(
    conn: Connection[Any], invoice_id: int, *, threshold: float = SIMILARITY_THRESHOLD
) -> list[DuplicateCandidate]:
    """Invoices that look like the same billing event as `invoice_id`."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT rfc_emisor, total, fecha_emision
              FROM invoices WHERE id = %s
            """,
            (invoice_id,),
        )
        subject = cur.fetchone()
        if subject is None:
            return []

        # The centroid of an invoice's line-item embeddings. NULL when the
        # backfill has not run, in which case the detector stays silent rather
        # than guessing.
        cur.execute(
            """
            WITH subject AS (
                SELECT avg(embedding)::vector AS centroid
                  FROM line_items
                 WHERE invoice_id = %(id)s AND embedding IS NOT NULL
            ),
            candidates AS (
                SELECT i.id, i.uuid::text AS uuid, i.folio, i.fecha_emision, i.total,
                       avg(li.embedding)::vector AS centroid
                  FROM invoices i
                  JOIN line_items li ON li.invoice_id = i.id
                 WHERE i.id <> %(id)s
                   AND i.rfc_emisor = %(rfc)s
                   AND abs(i.total - %(total)s) <= %(total)s * %(tol)s
                   AND i.fecha_emision BETWEEN %(from)s AND %(to)s
                   AND li.embedding IS NOT NULL
                 GROUP BY i.id, i.uuid, i.folio, i.fecha_emision, i.total
            )
            SELECT c.id, c.uuid, c.folio, c.fecha_emision, c.total,
                   1 - (c.centroid <=> s.centroid) AS similarity
              FROM candidates c, subject s
             WHERE s.centroid IS NOT NULL
               AND 1 - (c.centroid <=> s.centroid) >= %(threshold)s
             ORDER BY similarity DESC
             LIMIT 10
            """,
            {
                "id": invoice_id,
                "rfc": subject["rfc_emisor"],
                "total": subject["total"],
                "tol": TOTAL_TOLERANCE,
                "from": subject["fecha_emision"] - DATE_WINDOW,
                "to": subject["fecha_emision"] + DATE_WINDOW,
                "threshold": threshold,
            },
        )
        return [
            DuplicateCandidate(
                invoice_id=r["id"],
                uuid=r["uuid"].upper(),
                folio=r["folio"],
                fecha_emision=r["fecha_emision"].isoformat(),
                total=r["total"],
                similarity=round(float(r["similarity"]), 4),
            )
            for r in cur.fetchall()
        ]


def detect_semantic_duplicate(
    conn: Connection[Any], invoice_id: int
) -> Anomaly | None:
    """Wrap the search as an `Anomaly`, matching the pure detectors' shape."""
    matches = find_semantic_duplicates(conn, invoice_id)
    if not matches:
        return None
    best = matches[0]
    return Anomaly(
        kind="semantic_duplicate",
        severity="critical",
        detail={
            "match_uuid": best.uuid,
            "match_folio": best.folio,
            "similitud": best.similarity,
            "n_coincidencias": len(matches),
        },
        evidence={
            "threshold": SIMILARITY_THRESHOLD,
            "ventana_dias": DATE_WINDOW.days,
            "tolerancia_total": str(TOTAL_TOLERANCE),
            "coincidencias": [
                {
                    "uuid": m.uuid,
                    "folio": m.folio,
                    "fecha": m.fecha_emision,
                    "total": str(m.total),
                    "similitud": m.similarity,
                }
                for m in matches
            ],
        },
    )


# Set once the shared embedding backend has failed repeatedly, so a batch of
# 300 invoices does not wait on the same dead host 300 times. Ingesting the
# corpus with EMBED_BASE_URL pointing at an offline machine went from 9 seconds
# to unbounded before this existed.
_BACKEND_DOWN: str | None = None
_CONSECUTIVE_FAILURES = 0

# Tripping on the first failure was wrong, and a real run showed why: the
# vision model was still resident in GPU memory when an eval started, the
# embedding model could not load for a few seconds, and one transient error
# silenced the detector for all 300 documents. The report said so rather than
# printing a zero, which is the only reason it was caught.
#
# Three keeps the original property — a genuinely dead host costs 3 × 15 s
# instead of 300 × 15 s — while surviving contention that resolves itself.
FAILURES_BEFORE_TRIPPING = 3


def reset_backend_state() -> None:
    """Forget a previous failure. For tests and long-lived processes."""
    global _BACKEND_DOWN, _CONSECUTIVE_FAILURES
    _BACKEND_DOWN = None
    _CONSECUTIVE_FAILURES = 0


def run_vector_stage(
    conn: Connection[Any], invoice_id: int, *, provider: Any = None
) -> tuple[list[Anomaly], str | None]:
    """Embed this invoice's line items, then look for a semantic duplicate.

    Runs *after* the invoice is persisted, because the comparison is against
    the ledger and the embeddings have to exist before the vector search can
    use them.

    Returns the findings and, when the stage could not run, the reason. A
    missing embedding backend is a normal operating state, not an error: the
    rest of the pipeline is deterministic and does not need one. Reporting the
    reason keeps that visible instead of letting the detector look silently
    healthy — which is exactly how this detector spent its first week, wired to
    nothing and documented as working.
    """
    from cfdi_agent.extract.providers.base import ProviderError

    global _BACKEND_DOWN, _CONSECUTIVE_FAILURES
    if provider is None and _BACKEND_DOWN is not None:
        return [], _BACKEND_DOWN

    shared = provider is None
    try:
        if shared:
            from cfdi_agent.enrich.embeddings import embedding_provider

            provider = embedding_provider()
        embed_invoice(conn, invoice_id, provider)
    except ProviderError as exc:
        # The breaker covers the shared backend and nothing else, on both
        # sides: an injected provider is neither counted against it nor
        # stopped by it. Counting one but not the other let a caller's own
        # provider disable the shared path it never used.
        if shared:
            _CONSECUTIVE_FAILURES += 1
            if _CONSECUTIVE_FAILURES >= FAILURES_BEFORE_TRIPPING:
                _BACKEND_DOWN = str(exc)
        return [], str(exc)

    if shared:
        _CONSECUTIVE_FAILURES = 0
    anomaly = detect_semantic_duplicate(conn, invoice_id)
    return ([anomaly] if anomaly else []), None


def embed_invoice(conn: Connection[Any], invoice_id: int, provider: Any) -> int:
    """Fill in the embeddings for one invoice's line items."""
    from cfdi_agent.config import get_config

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, descripcion FROM line_items "
            "WHERE invoice_id = %s AND embedding IS NULL ORDER BY line_no",
            (invoice_id,),
        )
        rows = cur.fetchall()
    if not rows:
        return 0

    vectors = provider.embed([r["descripcion"] for r in rows])
    expected = get_config().embed_dim
    for row, vector in zip(rows, vectors, strict=True):
        if len(vector) != expected:
            from cfdi_agent.extract.providers.base import ProviderError

            raise ProviderError(
                f"embedding model returned {len(vector)} dims, schema expects "
                f"{expected}. Check EMBED_MODEL / EMBED_DIM."
            )
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE line_items SET embedding = %s WHERE id = %s",
                (str(vector), row["id"]),
            )
    return len(rows)
