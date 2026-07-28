"""End-to-end ingest against a real Postgres.

Uses a dedicated `cfdi_test` database, created on demand and truncated between
tests, so running the suite never destroys the corpus in the development
database. Skipped entirely when Postgres is unreachable — the deterministic
core must stay testable with no services running.

What these cover that the pure-function tests cannot: the boundary. The
duplicate-UUID bug that shipped in the first version of `repo` was invisible to
every unit test, because psycopg returns `uuid.UUID` objects and the in-memory
`HistoryContext` holds strings — a comparison that is silently always False.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

TEST_DB = "cfdi_test"


def _admin_url() -> str | None:
    """Connection URL for the maintenance database, or None if unreachable."""
    base = os.environ.get("DATABASE_URL", "postgresql://cfdi:cfdi@localhost:5432/cfdi")
    return base.rsplit("/", 1)[0] + "/postgres"


def _postgres_available() -> bool:
    try:
        import psycopg

        with psycopg.connect(_admin_url(), connect_timeout=3):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_available(), reason="Postgres not reachable; `docker compose up -d db`"
)


@pytest.fixture(scope="module", autouse=True)
def test_database():
    """Create `cfdi_test`, point the config at it, apply the schema."""
    import psycopg

    with psycopg.connect(_admin_url(), autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,))
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE "{TEST_DB}"')

    base = os.environ.get("DATABASE_URL", "postgresql://cfdi:cfdi@localhost:5432/cfdi")
    os.environ["DATABASE_URL"] = base.rsplit("/", 1)[0] + f"/{TEST_DB}"

    from cfdi_agent.config import get_config

    get_config.cache_clear()

    from cfdi_agent.db.init import apply_schema

    apply_schema()
    yield
    get_config.cache_clear()


@pytest.fixture(autouse=True)
def clean_tables():
    from cfdi_agent.db.conn import connect

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "TRUNCATE invoices, suppliers, line_items, taxes, anomalies, "
            "review_queue, extraction_runs, seen_folios, processed_files "
            "RESTART IDENTITY CASCADE"
        )
    yield


@pytest.fixture
def corpus(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict]]:
    from synth.generate_cfdi import generate

    out = tmp_path_factory.mktemp("db_corpus")
    labels = generate(
        n=40,
        defect_rate=0.0,
        out_dir=out,
        labels_path=out / "labeled.jsonl",
        seed=17,
        n_suppliers=3,
        receptor_rfc="XAXX010101000",
        receptor_nombre="Mi Empresa SA de CV",
    )
    return out, labels


def _ingest(path: Path):
    from cfdi_agent.db.conn import connect
    from cfdi_agent.ingest.pipeline import ingest_file

    with connect() as conn:
        return ingest_file(conn, path, company_rfc="XAXX010101000")


def test_clean_invoice_is_persisted_whole(corpus) -> None:
    """Invoice, lines and taxes all land, or none of them do."""
    from cfdi_agent.db.conn import fetch_one

    out, labels = corpus
    outcome = _ingest(out / labels[0]["file"])
    assert outcome.status == "ok", outcome.summary

    row = fetch_one(
        """
        SELECT i.uuid::text AS uuid,
               (SELECT count(*) FROM line_items WHERE invoice_id = i.id) AS lines,
               (SELECT count(*) FROM taxes      WHERE invoice_id = i.id) AS taxes
          FROM invoices i WHERE i.id = %s
        """,
        (outcome.invoice_id,),
    )
    assert row["uuid"].upper() == labels[0]["uuid"]
    assert row["lines"] == labels[0]["expected"]["n_conceptos"]
    assert row["taxes"] >= 1


def test_same_file_twice_is_idempotent(corpus) -> None:
    """An n8n retry must not raise an alert or double-insert."""
    from cfdi_agent.db.conn import fetch_one

    out, labels = corpus
    first = _ingest(out / labels[0]["file"])
    second = _ingest(out / labels[0]["file"])

    assert second.status == "duplicate_file"
    assert second.uuid == first.uuid
    assert second.anomalies == []
    assert fetch_one("SELECT count(*) AS n FROM invoices")["n"] == 1
    assert fetch_one("SELECT count(*) AS n FROM anomalies")["n"] == 0


def test_duplicate_uuid_attaches_to_the_existing_invoice(corpus, tmp_path) -> None:
    """Different bytes, same UUID: the fiscal duplicate.

    This is the regression guard for the psycopg UUID-vs-str comparison bug:
    without normalization the detector never fires and the insert dies on the
    unique constraint instead.
    """
    from cfdi_agent.db.conn import fetch_one

    out, labels = corpus
    first = _ingest(out / labels[0]["file"])

    # Same UUID, different bytes — a whitespace tweak is enough to change the
    # file hash without touching any field.
    original = (out / labels[0]["file"]).read_bytes()
    twin = tmp_path / "twin.xml"
    twin.write_bytes(original.replace(b"</cfdi:Comprobante>", b"\n</cfdi:Comprobante>"))

    second = _ingest(twin)
    assert second.status == "anomaly"
    assert second.invoice_id == first.invoice_id
    assert any(a["kind"] == "duplicate_uuid" for a in second.anomalies)
    assert fetch_one("SELECT count(*) AS n FROM invoices")["n"] == 1
    assert fetch_one(
        "SELECT count(*) AS n FROM anomalies WHERE kind = 'duplicate_uuid'"
    )["n"] == 1


def test_invoice_for_another_company_goes_to_review(corpus) -> None:
    from cfdi_agent.db.conn import fetch_one

    out, labels = corpus
    outcome = _ingest_as(out / labels[0]["file"], company_rfc="BBB020202BB2")
    assert outcome.status == "needs_review"
    assert fetch_one("SELECT count(*) AS n FROM invoices")["n"] == 0
    assert fetch_one("SELECT count(*) AS n FROM review_queue")["n"] == 1


def _ingest_as(path: Path, company_rfc: str):
    from cfdi_agent.db.conn import connect
    from cfdi_agent.ingest.pipeline import ingest_file

    with connect() as conn:
        return ingest_file(conn, path, company_rfc=company_rfc)


def test_unparseable_document_goes_to_review(tmp_path) -> None:
    from cfdi_agent.db.conn import fetch_one

    junk = tmp_path / "not-an-invoice.xml"
    junk.write_bytes(b"<html><body>Su factura adjunta</body></html>")

    outcome = _ingest(junk)
    assert outcome.status == "needs_review"
    row = fetch_one("SELECT reason FROM review_queue")
    assert "no se pudo parsear" in row["reason"]


def test_every_document_logs_an_extraction_run(corpus) -> None:
    """Tier-0 runs are logged too, so cost per invoice has an honest denominator."""
    from cfdi_agent.db.conn import fetch_all

    out, labels = corpus
    for label in labels[:5]:
        _ingest(out / label["file"])

    runs = fetch_all("SELECT tier, provider, ok, latency_ms FROM extraction_runs")
    assert len(runs) == 5
    assert all(r["tier"] == 0 and r["provider"] == "none" and r["ok"] for r in runs)
    assert all(r["latency_ms"] >= 0 for r in runs)


def test_seen_folios_advances_even_when_not_persisted(corpus, tmp_path) -> None:
    """A declined document must still advance the supplier's watermark.

    Otherwise the next invoice from that supplier is reported as a folio gap we
    manufactured ourselves — which is what took folio_gap precision to 0.30.
    """
    from cfdi_agent.db.conn import fetch_one

    out, labels = corpus
    original = (out / labels[0]["file"]).read_bytes()
    twin = tmp_path / "twin.xml"
    twin.write_bytes(original.replace(b"</cfdi:Comprobante>", b"\n</cfdi:Comprobante>"))

    _ingest(out / labels[0]["file"])
    _ingest(twin)  # duplicate UUID: deliberately not inserted into `invoices`

    assert fetch_one("SELECT count(*) AS n FROM invoices")["n"] == 1
    # ...but the folio was observed, and the watermark knows it.
    assert fetch_one("SELECT count(*) AS n FROM seen_folios")["n"] == 1


def test_history_context_is_scoped_to_the_invoice(corpus) -> None:
    """Price history must load only the products on the invoice at hand."""
    from cfdi_agent.db.conn import connect
    from cfdi_agent.db.repo import build_history_context
    from cfdi_agent.extract.xml_parser import parse_cfdi_file

    out, labels = corpus
    for label in labels[:12]:
        _ingest(out / label["file"])

    inv = parse_cfdi_file(out / labels[12]["file"])
    with connect() as conn:
        ctx = build_history_context(conn, inv)

    claves_on_invoice = {c.clave_prod_serv for c in inv.conceptos}
    for _rfc, clave in ctx.price_stats:
        assert clave in claves_on_invoice


def test_a_declined_document_is_also_idempotent(corpus, tmp_path) -> None:
    """Retries must be a no-op for *every* outcome, not just accepted invoices.

    Regression guard. The retry guard used to read `invoices.file_hash`, and a
    duplicate-UUID submission is deliberately never inserted there — so every
    redelivery re-derived the verdict and wrote another critical anomaly.
    Re-running one 300-document corpus grew duplicate_uuid anomalies by 16 per
    pass, unbounded; in production that is a webhook retry spamming Slack.
    """
    from cfdi_agent.db.conn import fetch_one

    out, labels = corpus
    original = (out / labels[0]["file"]).read_bytes()
    twin = tmp_path / "twin.xml"
    twin.write_bytes(original.replace(b"</cfdi:Comprobante>", b"\n</cfdi:Comprobante>"))

    _ingest(out / labels[0]["file"])
    first = _ingest(twin)
    assert first.status == "anomaly"
    after_first = fetch_one("SELECT count(*) AS n FROM anomalies")["n"]

    for _ in range(3):
        again = _ingest(twin)
        assert again.status == "duplicate_file"

    assert fetch_one("SELECT count(*) AS n FROM anomalies")["n"] == after_first
    assert fetch_one(
        "SELECT seen_count AS n FROM processed_files WHERE status = 'anomaly'"
    )["n"] == 4


def test_an_unreadable_document_is_remembered(tmp_path) -> None:
    """Junk must not be re-queued for review on every retry either."""
    from cfdi_agent.db.conn import fetch_one

    junk = tmp_path / "junk.xml"
    junk.write_bytes(b"<html><body>Su factura adjunta</body></html>")

    assert _ingest(junk).status == "needs_review"
    assert _ingest(junk).status == "duplicate_file"
    assert fetch_one("SELECT count(*) AS n FROM review_queue")["n"] == 1
