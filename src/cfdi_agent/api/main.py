"""HTTP surface. This is the contract n8n consumes.

The division of labour matters: n8n owns orchestration (triggers, retries,
cron, Slack, credentials, and a canvas the accounting team can read), this app
owns the domain. Tax rules do not belong in canvas nodes — they belong in
tested Python, which is why the flow's only job is to POST a file here and
branch on `status`.

`POST /ingest` returns a deliberately small, stable payload:

    {"status": "ok" | "anomaly" | "needs_review" | "duplicate_file",
     "uuid": ..., "invoice_id": ..., "summary": ..., "anomalies": [...]}

An n8n `If` node switches on `status`; `summary` goes straight into the Slack
message. Adding fields is safe, renaming these is not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from cfdi_agent.config import get_config
from cfdi_agent.db import repo
from cfdi_agent.db.conn import connect
from cfdi_agent.ingest.pipeline import ingest_bytes

# 8 MB. A CFDI XML is a few kilobytes; anything approaching this is not an
# invoice, and an unbounded upload endpoint is a denial-of-service invitation.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

app = FastAPI(
    title="CFDI agent",
    version="0.1.0",
    summary="Deterministic CFDI ingest, validation and anomaly detection.",
)


def db():
    with connect() as conn:
        yield conn


DbConn = Annotated[Any, Depends(db)]


class IngestPathRequest(BaseModel):
    """For documents already on a shared volume, rather than uploaded."""

    path: str = Field(description="Absolute path readable by this process")


@app.get("/health")
def health() -> dict:
    cfg = get_config()
    try:
        with connect(autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 AS ok")
            db_ok = cur.fetchone()["ok"] == 1
    except Exception as exc:  # noqa: BLE001 - health must report, never raise
        return {"status": "degraded", "database": False, "error": str(exc)}
    return {
        "status": "ok",
        "database": db_ok,
        "company_rfc": cfg.company_rfc,
        # Useful in a smoke test: tells you whether tier 2 is even reachable
        # without exposing whether a key is set to anyone who can read a log.
        "llm_provider": cfg.llm_provider,
    }


@app.post("/ingest")
async def ingest(conn: DbConn, file: Annotated[UploadFile, File()]) -> dict:
    """Ingest one uploaded CFDI (XML today; PDF once tier 2 lands)."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds {MAX_UPLOAD_BYTES} bytes",
        )
    outcome = ingest_bytes(
        conn, data, file_path=file.filename or "<upload>"
    )
    return outcome.to_dict()


@app.post("/ingest/path")
def ingest_path(conn: DbConn, body: IngestPathRequest) -> dict:
    """Ingest a document already present on disk.

    Convenience for local runs and for a shared volume between containers.
    Note this reads any path the process can reach — keep the API off the
    public internet, or drop this route in a deployment where that matters.
    """
    path = Path(body.path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"not a file: {path}")
    outcome = ingest_bytes(
        conn, path.read_bytes(), file_path=str(path)
    )
    return outcome.to_dict()


@app.get("/anomalies/open")
def anomalies_open(conn: DbConn, limit: int = 100) -> dict:
    """Unresolved findings, most severe first. Drives the daily digest flow."""
    rows = repo.open_anomalies(conn, limit=limit)
    return {"count": len(rows), "anomalies": rows}


@app.get("/review/pending")
def review_pending(conn: DbConn, limit: int = 100) -> dict:
    """Documents that could not be ingested confidently."""
    rows = repo.pending_review(conn, limit=limit)
    return {"count": len(rows), "items": rows}
