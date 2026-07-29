"""Persistence: build detector history from the database, then write results.

Two responsibilities, deliberately kept apart from `validate.rules`:

`build_history_context` turns database state into the plain `HistoryContext`
the pure detectors consume. It is **scoped to the invoice being validated** —
it loads the price history for this supplier's products, the folio watermark
for this series, and whether this exact UUID exists. Loading the whole table
would work on a demo and fall over on a real ledger.

`persist_invoice` writes the invoice, its lines, its taxes and its anomalies in
a single transaction. Either all of it lands or none of it does; a half-written
invoice with no line items would quietly corrupt the price history that every
later outlier verdict depends on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from psycopg import Connection
from psycopg.types.json import Jsonb

from cfdi_agent.schemas import ParsedInvoice
from cfdi_agent.validate.rules import (
    Anomaly,
    HistoryContext,
    ValidationResult,
    price_stats_from_samples,
)

IngestStatus = Literal["ok", "anomaly", "needs_review", "duplicate_file"]

# How many historical prices to pull per (supplier, product). The detector only
# needs a median and a MAD; an unbounded window would make an old price war
# dominate the baseline forever.
PRICE_WINDOW = 50


@dataclass(frozen=True, slots=True)
class PersistResult:
    status: IngestStatus
    invoice_id: int | None
    uuid: str | None
    anomalies: tuple[Anomaly, ...] = ()
    note: str | None = None


# --------------------------------------------------------------------------
# Reading history
# --------------------------------------------------------------------------


def _norm_uuid(value: Any) -> str:
    """Normalize a UUID to the uppercase string form used across the codebase.

    psycopg maps a Postgres `uuid` column to a Python `uuid.UUID`, and Postgres
    itself stores the canonical lowercase form. `ParsedInvoice.uuid` is an
    uppercase string. Left unconverted, `"EAC0..." in {UUID("eac0...")}` is
    silently False — the duplicate detector never fires and the insert dies on
    the unique constraint instead. Normalize at the database boundary, once.
    """
    return str(value).upper()


def file_already_processed(conn: Connection[Any], file_hash: str) -> dict | None:
    """Same bytes seen before, whatever happened to them? Bumps the counter.

    The retry guard, and it reads `processed_files` rather than
    `invoices.file_hash` on purpose. A duplicate-UUID submission is never
    inserted into `invoices`, so an invoices-only check never recognized it and
    every redelivery produced another critical anomaly — 16 more per pass over
    the same corpus, unbounded. In production that is a webhook retry spamming
    the alert channel.

    Returns the recorded outcome so a redelivery replays the original verdict
    instead of re-deriving it.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE processed_files
               SET seen_count = seen_count + 1, last_seen = now()
             WHERE file_hash = %s
         RETURNING status, invoice_uuid, summary, seen_count
            """,
            (file_hash,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "status": row["status"],
        "uuid": _norm_uuid(row["invoice_uuid"]) if row["invoice_uuid"] else None,
        "summary": row["summary"],
        "seen_count": row["seen_count"],
    }


def record_processed_file(
    conn: Connection[Any],
    *,
    file_hash: str,
    file_path: str,
    status: str,
    invoice_uuid: str | None,
    summary: str,
) -> None:
    """Remember this document was handled, so a retry is a no-op."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO processed_files
                (file_hash, file_path, status, invoice_uuid, summary)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (file_hash) DO UPDATE
               SET last_seen = now(), seen_count = processed_files.seen_count + 1
            """,
            (file_hash, file_path, status, invoice_uuid, summary),
        )


def build_history_context(conn: Connection[Any], inv: ParsedInvoice) -> HistoryContext:
    """Assemble exactly the history the detectors need for *this* invoice."""
    with conn.cursor() as cur:
        # Detector #1: does this specific UUID already exist? A single-row
        # probe, not a table scan.
        cur.execute("SELECT uuid FROM invoices WHERE uuid = %s", (inv.uuid,))
        known_uuids = frozenset(_norm_uuid(row["uuid"]) for row in cur.fetchall())

        # Detector #6: the full supplier list. A company has hundreds of
        # suppliers, not millions, and the detector reports the count as
        # evidence — so it genuinely needs the set, not a membership test.
        cur.execute("SELECT rfc FROM suppliers")
        known_rfcs = frozenset(row["rfc"] for row in cur.fetchall())

        # Detector #7: the folio watermark, read from `seen_folios` rather than
        # `invoices`. The question is what the supplier issued, not what we
        # filed — see the table's comment in schema.sql.
        last_folio: dict[tuple[str, str | None], int] = {}
        cur.execute(
            """
            SELECT max(folio) AS last
              FROM seen_folios
             WHERE rfc_emisor = %s AND serie = %s
            """,
            (inv.rfc_emisor, inv.serie or ""),
        )
        row = cur.fetchone()
        if row and row["last"] is not None:
            last_folio[(inv.rfc_emisor, inv.serie)] = int(row["last"])

        # Detector #3: unit-price history for the products on this invoice.
        claves = sorted({c.clave_prod_serv for c in inv.conceptos if c.clave_prod_serv})
        price_stats = {}
        for clave in claves:
            cur.execute(
                """
                SELECT li.valor_unitario
                  FROM line_items li
                  JOIN invoices i ON i.id = li.invoice_id
                 WHERE i.rfc_emisor = %s
                   AND li.clave_prod_serv = %s
                 ORDER BY i.fecha_emision DESC
                 LIMIT %s
                """,
                (inv.rfc_emisor, clave, PRICE_WINDOW),
            )
            samples = [r["valor_unitario"] for r in cur.fetchall()]
            if samples:
                # Restore chronological order: `recent` in the evidence should
                # mean recent, and the query pulled newest-first.
                price_stats[(inv.rfc_emisor, clave)] = price_stats_from_samples(
                    list(reversed(samples))
                )

    return HistoryContext(
        loaded=True,
        known_uuids=known_uuids,
        known_rfcs=known_rfcs,
        last_folio=last_folio,
        price_stats=price_stats,
    )


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def record_seen_folio(conn: Connection[Any], inv: ParsedInvoice) -> None:
    """Note that this folio was observed, whatever we go on to do with it.

    Called for every invoice that parses, *before* the persistence decision, so
    a document we decline to insert still advances the supplier's watermark.
    Alphanumeric folios are skipped — sequence checking does not apply to them.
    """
    if not inv.folio or not inv.folio.isdigit():
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO seen_folios (rfc_emisor, serie, folio)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (inv.rfc_emisor, inv.serie or "", int(inv.folio)),
        )


def _upsert_supplier(cur, inv: ParsedInvoice) -> None:
    cur.execute(
        """
        INSERT INTO suppliers (rfc, nombre, invoice_count)
        VALUES (%s, %s, 1)
        ON CONFLICT (rfc) DO UPDATE
           SET invoice_count = suppliers.invoice_count + 1,
               nombre = COALESCE(EXCLUDED.nombre, suppliers.nombre)
        """,
        (inv.rfc_emisor, inv.nombre_emisor),
    )


def _insert_anomalies(cur, invoice_id: int | None, anomalies: tuple[Anomaly, ...]) -> None:
    for a in anomalies:
        cur.execute(
            """
            INSERT INTO anomalies (invoice_id, kind, severity, detail, evidence)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (invoice_id, a.kind, a.severity, Jsonb(a.detail), Jsonb(a.evidence)),
        )


def record_anomalies(
    conn: Connection[Any], invoice_id: int, anomalies: Sequence[Anomaly]
) -> None:
    """Attach findings to an invoice after it was persisted.

    The vector stage runs after the row exists, because a semantic duplicate is
    found by comparing against the ledger — including, now, this invoice.
    """
    if not anomalies:
        return
    with conn.cursor() as cur:
        _insert_anomalies(cur, invoice_id, tuple(anomalies))


def persist_invoice(
    conn: Connection[Any],
    inv: ParsedInvoice,
    result: ValidationResult,
    *,
    file_hash: str,
    file_path: str,
) -> PersistResult:
    """Write an accepted invoice and its findings in one transaction."""
    if not result.accepted:
        enqueue_review(
            conn,
            file_hash=file_hash,
            file_path=file_path,
            reason=result.reject_reason or "rejected",
            payload={"uuid": inv.uuid, "rfc_receptor": inv.rfc_receptor},
        )
        return PersistResult(
            status="needs_review",
            invoice_id=None,
            uuid=inv.uuid,
            note=result.reject_reason,
        )

    duplicate_uuid = any(a.kind == "duplicate_uuid" for a in result.anomalies)

    with conn.cursor() as cur:
        if duplicate_uuid:
            # The invoice is already in the ledger under this UUID and the
            # column is UNIQUE. Attach the finding to the *existing* row rather
            # than failing the insert: the alert belongs on the invoice that
            # exists, and the operator needs to see both submissions.
            cur.execute("SELECT id FROM invoices WHERE uuid = %s", (inv.uuid,))
            row = cur.fetchone()
            existing_id = row["id"] if row else None
            _insert_anomalies(cur, existing_id, result.anomalies)
            enqueue_review(
                conn,
                file_hash=file_hash,
                file_path=file_path,
                reason=f"UUID {inv.uuid} already recorded",
                payload={"existing_invoice_id": existing_id},
            )
            return PersistResult(
                status="anomaly",
                invoice_id=existing_id,
                uuid=inv.uuid,
                anomalies=result.anomalies,
                note="duplicate UUID; finding attached to the existing invoice",
            )

        _upsert_supplier(cur, inv)

        cur.execute(
            """
            INSERT INTO invoices (
                uuid, serie, folio, fecha_emision, fecha_timbrado,
                rfc_emisor, rfc_receptor, subtotal, descuento, total,
                moneda, tipo_cambio, metodo_pago, forma_pago, uso_cfdi,
                source, file_hash, file_path, sat_status
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING id
            """,
            (
                inv.uuid, inv.serie, inv.folio, inv.fecha_emision, inv.fecha_timbrado,
                inv.rfc_emisor, inv.rfc_receptor, inv.subtotal, inv.descuento, inv.total,
                inv.moneda, inv.tipo_cambio, inv.metodo_pago, inv.forma_pago, inv.uso_cfdi,
                inv.source, file_hash, file_path, None,
            ),
        )
        invoice_id = cur.fetchone()["id"]

        for c in inv.conceptos:
            cur.execute(
                """
                INSERT INTO line_items (
                    invoice_id, line_no, clave_prod_serv, clave_unidad, descripcion,
                    cantidad, valor_unitario, importe, descuento, objeto_imp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    invoice_id, c.line_no, c.clave_prod_serv, c.clave_unidad,
                    c.descripcion, c.cantidad, c.valor_unitario, c.importe,
                    c.descuento, c.objeto_imp,
                ),
            )

        for t in inv.impuestos:
            cur.execute(
                """
                INSERT INTO taxes (invoice_id, tipo, impuesto, base, tasa, importe)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (invoice_id, t.tipo, t.impuesto, t.base, t.tasa, t.importe),
            )

        _insert_anomalies(cur, invoice_id, result.anomalies)

    status: IngestStatus = "anomaly" if result.anomalies else "ok"
    return PersistResult(
        status=status,
        invoice_id=invoice_id,
        uuid=inv.uuid,
        anomalies=result.anomalies,
    )


def enqueue_review(
    conn: Connection[Any],
    *,
    file_hash: str,
    file_path: str,
    reason: str,
    payload: dict | None = None,
) -> int:
    """Park a document a human has to look at."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO review_queue (file_hash, file_path, reason, payload)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (file_hash, file_path, reason, Jsonb(payload or {})),
        )
        return cur.fetchone()["id"]


def record_extraction_run(
    conn: Connection[Any],
    *,
    file_hash: str,
    tier: int,
    provider: str,
    model: str | None,
    latency_ms: int,
    ok: bool,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    cost_usd: Decimal | None = None,
    error: str | None = None,
) -> None:
    """Log one pass through the router.

    Tier-0 runs are logged too (`provider='none'`), so cost per invoice is
    computed over every document rather than only the ones that reached a
    model — which is the difference between an honest number and a flattering
    one.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO extraction_runs (
                file_hash, tier, provider, model, latency_ms,
                tokens_in, tokens_out, cost_usd, ok, error
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                file_hash, tier, provider, model, latency_ms,
                tokens_in, tokens_out, cost_usd, ok, error,
            ),
        )


# --------------------------------------------------------------------------
# Queries for the API and dashboard
# --------------------------------------------------------------------------


def open_anomalies(conn: Connection[Any], limit: int = 100) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, invoice_uuid, rfc_emisor, total, kind, severity,
                   detail, explanation, created_at
              FROM v_anomalies
             WHERE NOT resolved
             ORDER BY CASE severity
                        WHEN 'critical' THEN 0 WHEN 'warn' THEN 1 ELSE 2
                      END,
                      created_at DESC
             LIMIT %s
            """,
            (limit,),
        )
        return cur.fetchall()


def pending_review(conn: Connection[Any], limit: int = 100) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, file_path, reason, payload, created_at
              FROM review_queue
             WHERE status = 'pending'
             ORDER BY created_at DESC
             LIMIT %s
            """,
            (limit,),
        )
        return cur.fetchall()
