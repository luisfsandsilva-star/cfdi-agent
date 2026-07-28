"""The ingest path: bytes in, verdict out.

One function so the HTTP endpoint, the watch directory and the eval harness all
exercise the same code. A pipeline that behaves differently depending on how
the document arrived is a pipeline whose eval numbers mean nothing.

Everything here is tier 0 — no model is called. The summary handed to Slack is
assembled from the detectors' own evidence, not written by an LLM. That is what
makes it impossible for the alert text to describe a finding that did not
happen.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from psycopg import Connection

from cfdi_agent.config import get_config
from cfdi_agent.db import repo
from cfdi_agent.extract.xml_parser import CfdiParseError, parse_cfdi_bytes
from cfdi_agent.ingest.dedupe import sha256_bytes
from cfdi_agent.validate.rules import Anomaly, validate_invoice

# Human-readable, in Spanish: these strings end up in a Slack channel read by
# the accounting team, not by developers.
KIND_LABELS = {
    "duplicate_uuid": "UUID duplicado",
    "invalid_rfc": "RFC inválido",
    "line_math_mismatch": "una línea no multiplica",
    "subtotal_mismatch": "el subtotal no cuadra",
    "total_mismatch": "el total no cuadra",
    "price_outlier": "precio fuera de rango",
    "new_supplier": "proveedor nuevo",
    "folio_gap": "salto de folio",
    "stale_stamp": "timbrado fuera de plazo",
    "unknown_catalog_code": "código de catálogo desconocido",
}

SEVERITY_ICON = {"critical": "🔴", "warn": "🟡", "info": "🔵"}


@dataclass(frozen=True, slots=True)
class IngestOutcome:
    """The contract n8n consumes. Keep it stable — a flow branches on it."""

    status: str  # ok | anomaly | needs_review | duplicate_file
    uuid: str | None
    invoice_id: int | None
    summary: str
    anomalies: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "uuid": self.uuid,
            "invoice_id": self.invoice_id,
            "summary": self.summary,
            "anomalies": self.anomalies,
        }


def _anomaly_payload(anomalies: tuple[Anomaly, ...]) -> list[dict]:
    return [
        {
            "kind": a.kind,
            "label": KIND_LABELS.get(a.kind, a.kind),
            "severity": a.severity,
            "detail": a.detail,
            "evidence": a.evidence,
        }
        for a in anomalies
    ]


def _summarize(inv, anomalies: tuple[Anomaly, ...]) -> str:
    """Build the alert text straight from the detectors' evidence."""
    who = inv.nombre_emisor or inv.rfc_emisor
    head = f"{who} · {inv.moneda} {inv.total} · {inv.fecha_emision:%Y-%m-%d}"
    if not anomalies:
        return f"{head} — sin hallazgos"
    parts = [
        f"{SEVERITY_ICON.get(a.severity, '•')} {KIND_LABELS.get(a.kind, a.kind)}"
        for a in anomalies
    ]
    return f"{head} — " + ", ".join(parts)


def ingest_bytes(
    conn: Connection[Any],
    data: bytes,
    *,
    file_path: str,
    company_rfc: str | None = None,
) -> IngestOutcome:
    """Parse, validate and persist one document."""
    started = time.perf_counter()
    company_rfc = company_rfc or get_config().company_rfc
    file_hash = sha256_bytes(data)

    # Retry guard, before anything else. The same bytes arriving twice is an
    # n8n redelivery, not a fiscal duplicate, and must not raise an alert.
    # Reads `processed_files`, which records every outcome — an invoices-only
    # check misses the documents we deliberately do not insert.
    seen = repo.file_already_processed(conn, file_hash)
    if seen:
        return IngestOutcome(
            status="duplicate_file",
            uuid=seen["uuid"],
            invoice_id=None,
            summary=(
                f"Archivo ya procesado (visto {seen['seen_count']} veces, "
                f"resultado original: {seen['status']}); sin cambios."
            ),
        )

    try:
        inv = parse_cfdi_bytes(data)
    except CfdiParseError as exc:
        repo.enqueue_review(
            conn,
            file_hash=file_hash,
            file_path=file_path,
            reason=f"no se pudo parsear: {exc}",
            payload={"bytes": len(data)},
        )
        repo.record_extraction_run(
            conn,
            file_hash=file_hash,
            tier=0,
            provider="none",
            model=None,
            latency_ms=int((time.perf_counter() - started) * 1000),
            ok=False,
            error=str(exc),
        )
        summary = f"No se pudo leer el documento: {exc}"
        repo.record_processed_file(
            conn,
            file_hash=file_hash,
            file_path=file_path,
            status="needs_review",
            invoice_uuid=None,
            summary=summary,
        )
        return IngestOutcome(
            status="needs_review", uuid=None, invoice_id=None, summary=summary
        )

    ctx = repo.build_history_context(conn, inv)
    result = validate_invoice(inv, ctx, company_rfc=company_rfc)

    # Record the folio *after* reading history (so this invoice is not compared
    # against itself) but *before* persisting (so a document we decline to
    # insert still advances the supplier's watermark).
    repo.record_seen_folio(conn, inv)

    persisted = repo.persist_invoice(
        conn, inv, result, file_hash=file_hash, file_path=file_path
    )

    repo.record_extraction_run(
        conn,
        file_hash=file_hash,
        tier=0,
        provider="none",
        model=None,
        latency_ms=int((time.perf_counter() - started) * 1000),
        ok=True,
    )

    summary = (
        _summarize(inv, result.anomalies)
        if result.accepted
        else f"Rechazada: {result.reject_reason}"
    )
    repo.record_processed_file(
        conn,
        file_hash=file_hash,
        file_path=file_path,
        status=persisted.status,
        invoice_uuid=inv.uuid,
        summary=summary,
    )
    return IngestOutcome(
        status=persisted.status,
        uuid=persisted.uuid,
        invoice_id=persisted.invoice_id,
        summary=summary,
        anomalies=_anomaly_payload(result.anomalies),
    )


def ingest_file(
    conn: Connection[Any], path: str | Path, *, company_rfc: str | None = None
) -> IngestOutcome:
    p = Path(path)
    return ingest_bytes(conn, p.read_bytes(), file_path=str(p), company_rfc=company_rfc)
