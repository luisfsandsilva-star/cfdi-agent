"""Detector #2: the semantic duplicate, against a real Postgres and pgvector.

An honest limit up front. These tests drive the detector with a **stub
embedder** that maps each description to a controlled vector, so they verify
the plumbing and the thresholds: that the stage runs, that embeddings land in
`line_items`, that the SQL window and the cosine cutoff behave, and that a
missing backend degrades to a recorded reason rather than a silent zero.

They do **not** verify how close a real model puts "Servicio de limpieza de
oficina" and "Limpieza de oficinas". That question was answered by measuring
bge-m3 directly, and the answer moved the threshold from 0.93 to 0.70 — the
original value caught nothing, while these same stub tests passed, because the
stub was built to clear whatever bar was set. A fixture you design around a
constant cannot validate that constant.
"""

from __future__ import annotations

import os
from decimal import Decimal

import pytest

TEST_DB = "cfdi_test"
DIM = 1024


def _admin_url() -> str:
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


class StubEmbedder:
    """Maps a description to a vector this test controls.

    Descriptions sharing a `group` get near-identical vectors, which is what a
    real multilingual model does for a reworded line item. Anything unknown
    gets an orthogonal vector.
    """

    def __init__(self, groups: dict[str, str]) -> None:
        self.groups = groups
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out = []
        for text in texts:
            group = self.groups.get(text, text)
            axis = abs(hash(group)) % DIM
            # Unit vector on one axis, with a small tilt so two members of the
            # same group are close but not byte-identical.
            v = [0.0] * DIM
            v[axis] = 1.0
            v[(axis + 1) % DIM] = 0.05 if text == group else 0.12
            out.append(v)
        return out


@pytest.fixture(scope="module", autouse=True)
def test_database():
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
    from cfdi_agent.enrich.anomalies import reset_backend_state

    # The circuit breaker is process-level state; without this, the first test
    # that trips it silences the stage for every test after it.
    reset_backend_state()

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "TRUNCATE invoices, suppliers, line_items, taxes, anomalies, "
            "review_queue, extraction_runs, seen_folios, processed_files "
            "RESTART IDENTITY CASCADE"
        )
    yield


def _insert(uuid: str, folio: str, descripcion: str, *, total="1160.00", day=14) -> int:
    """Put one invoice in the ledger directly, bypassing the pipeline."""
    from cfdi_agent.db.conn import connect

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO suppliers (rfc, nombre) VALUES ('AAA010101AAA', 'Prov SA') "
            "ON CONFLICT (rfc) DO NOTHING"
        )
        cur.execute(
            """
            INSERT INTO invoices (uuid, serie, folio, fecha_emision, rfc_emisor,
                                  rfc_receptor, subtotal, total, source,
                                  file_hash, file_path)
            VALUES (%s, 'A', %s, %s, 'AAA010101AAA', 'XAXX010101000',
                    '1000.00', %s, 'xml', %s, 'x.xml')
            RETURNING id
            """,
            (uuid, folio, f"2026-03-{day:02d}T10:00:00", total, f"hash-{folio}"),
        )
        invoice_id = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO line_items (invoice_id, line_no, clave_prod_serv, "
            "descripcion, cantidad, valor_unitario, importe) "
            "VALUES (%s, 1, '80161501', %s, 1, 1000, 1000.00)",
            (invoice_id, descripcion),
        )
    return invoice_id


def _run(invoice_id: int, embedder):
    from cfdi_agent.db.conn import connect
    from cfdi_agent.enrich.anomalies import run_vector_stage

    with connect() as conn:
        return run_vector_stage(conn, invoice_id, provider=embedder)


SAME = {
    "Servicio de limpieza de oficina": "limpieza",
    "Limpieza de oficinas": "limpieza",
}


# ------------------------------------------------------------------ plumbing


def test_the_stage_writes_embeddings() -> None:
    from cfdi_agent.db.conn import fetch_one

    invoice_id = _insert("A1B2C3D4-0000-4000-8000-000000000001", "1", "Servicio")
    _run(invoice_id, StubEmbedder({}))
    assert fetch_one(
        "SELECT count(*) AS n FROM line_items WHERE embedding IS NOT NULL"
    )["n"] == 1


def test_no_backend_reports_a_reason_instead_of_a_clean_zero(monkeypatch) -> None:
    """A skipped stage must not look like a stage that found nothing.

    This detector spent its first week wired to nothing while the README
    described it as working. Silence is the failure mode to design against.
    """
    from cfdi_agent.config import get_config

    invoice_id = _insert("A1B2C3D4-0000-4000-8000-000000000002", "2", "Servicio")
    monkeypatch.delenv("EMBED_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    get_config.cache_clear()
    try:
        anomalies, reason = _run(invoice_id, None)
    finally:
        get_config.cache_clear()

    assert anomalies == []
    assert reason and "EMBED_BASE_URL" in reason


def test_a_wrong_dimension_is_refused() -> None:
    """A dimension mismatch means the wrong model is being served.

    Writing those vectors would corrupt the index and every later verdict.
    """

    class WrongDim:
        def embed(self, texts):
            return [[0.1] * 384 for _ in texts]

    invoice_id = _insert("A1B2C3D4-0000-4000-8000-000000000003", "3", "Servicio")
    anomalies, reason = _run(invoice_id, WrongDim())
    assert anomalies == []
    assert reason and "384" in reason


# ------------------------------------------------------------------ detection


def test_reworded_duplicate_is_caught() -> None:
    """Same supplier, same amount, same week, different wording and folio.

    Detector #1 is blind to this: the UUID and the folio are legitimately
    different. This is the expensive kind of double billing.
    """
    first = _insert(
        "A1B2C3D4-0000-4000-8000-00000000000A", "10",
        "Servicio de limpieza de oficina", day=14,
    )
    _run(first, StubEmbedder(SAME))

    second = _insert(
        "A1B2C3D4-0000-4000-8000-00000000000B", "11",
        "Limpieza de oficinas", day=17,
    )
    anomalies, reason = _run(second, StubEmbedder(SAME))

    assert reason is None
    assert [a.kind for a in anomalies] == ["semantic_duplicate"]
    assert anomalies[0].severity == "critical"
    assert anomalies[0].detail["match_uuid"].startswith("A1B2C3D4")
    assert anomalies[0].evidence["coincidencias"]


def test_unrelated_purchases_are_not_flagged() -> None:
    """Precision guard: two different things at the same price are not a duplicate."""
    first = _insert(
        "A1B2C3D4-0000-4000-8000-00000000000C", "20", "Cartucho de tóner", day=14
    )
    _run(first, StubEmbedder({}))

    second = _insert(
        "A1B2C3D4-0000-4000-8000-00000000000D", "21", "Combustible diésel", day=15
    )
    anomalies, _ = _run(second, StubEmbedder({}))
    assert anomalies == []


def test_the_same_bill_a_month_later_is_not_a_duplicate() -> None:
    """A monthly retainer repeats by design. The 7-day window is the separator."""
    first = _insert(
        "A1B2C3D4-0000-4000-8000-00000000000E", "30",
        "Servicio de limpieza de oficina", day=1,
    )
    _run(first, StubEmbedder(SAME))

    second = _insert(
        "A1B2C3D4-0000-4000-8000-00000000000F", "31",
        "Limpieza de oficinas", day=28,
    )
    anomalies, _ = _run(second, StubEmbedder(SAME))
    assert anomalies == []


def test_a_different_total_is_not_a_duplicate() -> None:
    """Outside the 1% window the invoices are for different work."""
    first = _insert(
        "A1B2C3D4-0000-4000-8000-000000000010", "40",
        "Servicio de limpieza de oficina", total="1160.00", day=14,
    )
    _run(first, StubEmbedder(SAME))

    second = _insert(
        "A1B2C3D4-0000-4000-8000-000000000011", "41",
        "Limpieza de oficinas", total="2400.00", day=15,
    )
    anomalies, _ = _run(second, StubEmbedder(SAME))
    assert anomalies == []


def test_a_different_supplier_is_not_a_duplicate() -> None:
    from cfdi_agent.db.conn import connect

    first = _insert(
        "A1B2C3D4-0000-4000-8000-000000000012", "50",
        "Servicio de limpieza de oficina", day=14,
    )
    _run(first, StubEmbedder(SAME))

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO suppliers (rfc, nombre) VALUES ('BBB020202BB2', 'Otro SA')"
        )
        cur.execute(
            """
            INSERT INTO invoices (uuid, serie, folio, fecha_emision, rfc_emisor,
                                  rfc_receptor, subtotal, total, source,
                                  file_hash, file_path)
            VALUES ('A1B2C3D4-0000-4000-8000-000000000013', 'B', '51',
                    '2026-03-15T10:00:00', 'BBB020202BB2', 'XAXX010101000',
                    '1000.00', '1160.00', 'xml', 'hash-51', 'y.xml')
            RETURNING id
            """
        )
        other = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO line_items (invoice_id, line_no, clave_prod_serv, "
            "descripcion, cantidad, valor_unitario, importe) "
            "VALUES (%s, 1, '80161501', 'Limpieza de oficinas', 1, 1000, 1000.00)",
            (other,),
        )

    anomalies, _ = _run(other, StubEmbedder(SAME))
    assert anomalies == []


def test_the_finding_carries_verifiable_evidence() -> None:
    """A person must be able to check the claim without re-running anything."""
    first = _insert(
        "A1B2C3D4-0000-4000-8000-000000000020", "60",
        "Servicio de limpieza de oficina", day=14,
    )
    _run(first, StubEmbedder(SAME))
    second = _insert(
        "A1B2C3D4-0000-4000-8000-000000000021", "61",
        "Limpieza de oficinas", day=16,
    )
    anomalies, _ = _run(second, StubEmbedder(SAME))

    evidence = anomalies[0].evidence
    assert evidence["ventana_dias"] == 7
    assert evidence["threshold"] == 0.70
    match = evidence["coincidencias"][0]
    assert match["folio"] == "60"
    assert Decimal(match["total"]) == Decimal("1160.00")
    assert match["similitud"] >= 0.93


# ------------------------------------------------------------ circuit breaker


class FlakyBackend:
    """Fails the first `n` calls, then works. Named `local` so it is treated
    as the shared backend, which is the one the breaker guards."""

    name = "local"

    def __init__(self, failures: int) -> None:
        self.remaining = failures
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        from cfdi_agent.extract.providers.base import ProviderError

        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise ProviderError("model is loading")
        return [[0.0] * DIM for _ in texts]


def test_a_transient_backend_failure_does_not_silence_the_detector(monkeypatch) -> None:
    """One blip must not disable the stage for the rest of the process.

    This tripped on the first failure once. A real eval hit it: the vision
    model still held the GPU when the run started, the embedding model could
    not load for a few seconds, and all 300 documents came back unembedded.
    """
    from cfdi_agent.enrich import embeddings
    from cfdi_agent.enrich.anomalies import FAILURES_BEFORE_TRIPPING

    backend = FlakyBackend(FAILURES_BEFORE_TRIPPING - 1)
    monkeypatch.setattr(embeddings, "embedding_provider", lambda: backend)
    for index in range(FAILURES_BEFORE_TRIPPING + 1):
        invoice_id = _insert(
            f"A1B2C3D4-0000-4000-8000-0000000001{index:02d}", f"7{index}", "Servicio"
        )
        anomalies, reason = _run(invoice_id, None)
    assert reason is None
    assert backend.calls == FAILURES_BEFORE_TRIPPING + 1


def test_a_dead_backend_still_stops_being_called(monkeypatch) -> None:
    """The property the breaker exists for, unchanged.

    Driven through the shared path, because that is the only path the breaker
    covers — an injected provider is the caller's business in both directions.
    """
    from cfdi_agent.enrich import anomalies, embeddings

    backend = FlakyBackend(999)
    monkeypatch.setattr(embeddings, "embedding_provider", lambda: backend)
    for index in range(10):
        invoice_id = _insert(
            f"A1B2C3D4-0000-4000-8000-0000000002{index:02d}", f"8{index}", "Servicio"
        )
        _run(invoice_id, None)
    assert backend.calls == anomalies.FAILURES_BEFORE_TRIPPING


def test_an_injected_provider_never_trips_the_shared_breaker() -> None:
    """A caller's own failing provider must not disable a backend it never used."""
    from cfdi_agent.enrich import anomalies

    backend = FlakyBackend(999)
    for index in range(6):
        invoice_id = _insert(
            f"A1B2C3D4-0000-4000-8000-0000000003{index:02d}", f"9{index}", "Servicio"
        )
        _run(invoice_id, backend)
    assert backend.calls == 6
    assert anomalies._BACKEND_DOWN is None
