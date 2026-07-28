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
# Cosine similarity over line-item centroids. 0.93 is deliberately high: the
# cost of a false accusation against a supplier is a phone call and lost trust
# in the alert channel.
SIMILARITY_THRESHOLD = 0.93


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
