-- CFDI agent schema.
--
-- Naming convention: generic infrastructure columns are English; columns that
-- carry a CFDI 4.0 field keep the SAT's own Spanish name (rfc_emisor,
-- clave_prod_serv, uso_cfdi, ...). Those are proper nouns from the standard —
-- translating them would break traceability with the XSD.
--
-- Idempotent: safe to re-run via `python -m cfdi_agent.db.init`.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- suppliers
CREATE TABLE IF NOT EXISTS suppliers (
    id            BIGSERIAL PRIMARY KEY,
    rfc           TEXT UNIQUE NOT NULL,
    nombre        TEXT,
    first_seen    TIMESTAMPTZ NOT NULL DEFAULT now(),
    invoice_count INT NOT NULL DEFAULT 0
);

-- ----------------------------------------------------------------- invoices
CREATE TABLE IF NOT EXISTS invoices (
    id             BIGSERIAL PRIMARY KEY,
    uuid           UUID UNIQUE NOT NULL,          -- TimbreFiscalDigital/@UUID
    serie          TEXT,
    folio          TEXT,
    fecha_emision  TIMESTAMPTZ NOT NULL,
    fecha_timbrado TIMESTAMPTZ,
    rfc_emisor     TEXT NOT NULL REFERENCES suppliers (rfc),
    rfc_receptor   TEXT NOT NULL,
    subtotal       NUMERIC(14, 2) NOT NULL,
    descuento      NUMERIC(14, 2) NOT NULL DEFAULT 0,
    total          NUMERIC(14, 2) NOT NULL,
    moneda         TEXT NOT NULL DEFAULT 'MXN',
    tipo_cambio    NUMERIC(14, 6),
    metodo_pago    TEXT,
    forma_pago     TEXT,
    uso_cfdi       TEXT,
    source         TEXT NOT NULL CHECK (source IN ('xml', 'pdf')),
    file_hash      TEXT UNIQUE NOT NULL,
    file_path      TEXT NOT NULL,
    sat_status     TEXT,                          -- vigente | cancelado | desconocido
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_invoices_emisor_fecha
    ON invoices (rfc_emisor, fecha_emision DESC);
CREATE INDEX IF NOT EXISTS idx_invoices_fecha
    ON invoices (fecha_emision DESC);
-- Supports detector #7 (folio gap) without a sequential scan.
CREATE INDEX IF NOT EXISTS idx_invoices_serie_folio
    ON invoices (rfc_emisor, serie, folio);

-- --------------------------------------------------------------- line_items
CREATE TABLE IF NOT EXISTS line_items (
    id              BIGSERIAL PRIMARY KEY,
    invoice_id      BIGINT NOT NULL REFERENCES invoices (id) ON DELETE CASCADE,
    line_no         INT NOT NULL,
    clave_prod_serv TEXT,
    clave_unidad    TEXT,
    descripcion     TEXT NOT NULL,
    cantidad        NUMERIC(14, 6) NOT NULL,
    valor_unitario  NUMERIC(14, 6) NOT NULL,
    importe         NUMERIC(14, 2) NOT NULL,
    descuento       NUMERIC(14, 2) NOT NULL DEFAULT 0,
    objeto_imp      TEXT,
    category        TEXT,                          -- Tier 1 enrichment
    embedding       vector(1024),                  -- bge-m3
    UNIQUE (invoice_id, line_no)
);

-- Supports detector #3 (price outlier) — the MAD baseline per (supplier, product).
CREATE INDEX IF NOT EXISTS idx_line_items_prodserv
    ON line_items (clave_prod_serv);
-- Built even while empty; pgvector handles NULL embeddings fine.
CREATE INDEX IF NOT EXISTS idx_line_items_embedding
    ON line_items USING hnsw (embedding vector_cosine_ops);

-- -------------------------------------------------------------------- taxes
CREATE TABLE IF NOT EXISTS taxes (
    id         BIGSERIAL PRIMARY KEY,
    invoice_id BIGINT NOT NULL REFERENCES invoices (id) ON DELETE CASCADE,
    tipo       TEXT NOT NULL CHECK (tipo IN ('traslado', 'retencion')),
    impuesto   TEXT,                               -- 001 ISR | 002 IVA | 003 IEPS
    base       NUMERIC(14, 2),
    tasa       NUMERIC(10, 6),
    importe    NUMERIC(14, 2) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_taxes_invoice ON taxes (invoice_id);

-- --------------------------------------------------------- processed_files
-- Every document that has been through the pipeline, whatever the outcome.
--
-- The retry guard cannot read `invoices.file_hash` alone: a duplicate-UUID
-- submission is deliberately never inserted there, so it was never recognized
-- on redelivery and produced a fresh critical anomaly on every retry. Measured
-- by re-running the same 300-document corpus: duplicate_uuid anomalies grew
-- 16 per pass, unbounded. In production that is an n8n redelivery spamming the
-- alert channel until someone mutes it.
--
-- Same lesson as `seen_folios`: record what was *seen*, not what was *stored*.
CREATE TABLE IF NOT EXISTS processed_files (
    file_hash    TEXT PRIMARY KEY,
    file_path    TEXT NOT NULL,
    status       TEXT NOT NULL,
    invoice_uuid UUID,
    summary      TEXT,
    first_seen   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen    TIMESTAMPTZ NOT NULL DEFAULT now(),
    seen_count   INT NOT NULL DEFAULT 1
);

-- ------------------------------------------------------------- seen_folios
-- Every folio we have *observed*, written before any persistence decision.
--
-- Detector #7 asks "did the supplier skip a folio?", which is a question about
-- what the supplier issued — not about what our ledger happens to contain. The
-- two diverge constantly: a duplicate-UUID submission is deliberately not
-- inserted into `invoices`, so its folio would leave a hole and the next
-- invoice from that supplier would be reported as a gap we created ourselves.
--
-- Measured on a 300-invoice corpus, reading the watermark from `invoices`
-- produced 43 folio_gap alerts against 13 real ones: 74% false positives, which
-- is well past the point where an alert channel stops being read.
CREATE TABLE IF NOT EXISTS seen_folios (
    rfc_emisor TEXT NOT NULL,
    -- '' rather than NULL: a folio series with no Serie is a real case, and
    -- NULL in a primary key does not compare the way this needs to.
    serie      TEXT NOT NULL DEFAULT '',
    folio      BIGINT NOT NULL,
    first_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (rfc_emisor, serie, folio)
);

-- ---------------------------------------------------------------- anomalies
CREATE TABLE IF NOT EXISTS anomalies (
    id          BIGSERIAL PRIMARY KEY,
    invoice_id  BIGINT REFERENCES invoices (id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    severity    TEXT NOT NULL CHECK (severity IN ('info', 'warn', 'critical')),
    detail      JSONB NOT NULL,
    -- The exact rows/values that fired the detector. The LLM may only write
    -- `explanation` on top of this — never invent an anomaly without evidence.
    evidence    JSONB,
    explanation TEXT,
    resolved    BOOLEAN NOT NULL DEFAULT false,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_anomalies_open
    ON anomalies (created_at DESC) WHERE NOT resolved;
CREATE INDEX IF NOT EXISTS idx_anomalies_invoice ON anomalies (invoice_id);

-- ------------------------------------------------------------- review_queue
-- Anything that failed deterministic validation. Nothing reaches `invoices`
-- without balancing, so this is the human fallback — not a dead-letter bin.
CREATE TABLE IF NOT EXISTS review_queue (
    id         BIGSERIAL PRIMARY KEY,
    file_hash  TEXT NOT NULL,
    file_path  TEXT NOT NULL,
    reason     TEXT NOT NULL,
    payload    JSONB,
    status     TEXT NOT NULL DEFAULT 'pending'
               CHECK (status IN ('pending', 'accepted', 'rejected')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_review_queue_pending
    ON review_queue (created_at DESC) WHERE status = 'pending';

-- ---------------------------------------------------------- extraction_runs
-- The eval harness and the cost dashboard both read from this table. Every
-- document that passes through the router writes exactly one row, including
-- tier-0 runs (provider='none') so cost-per-invoice is computed over the real
-- denominator, not just the LLM subset.
CREATE TABLE IF NOT EXISTS extraction_runs (
    id         BIGSERIAL PRIMARY KEY,
    file_hash  TEXT NOT NULL,
    tier       SMALLINT NOT NULL CHECK (tier IN (0, 1, 2)),
    provider   TEXT NOT NULL,                      -- none | anthropic | local
    model      TEXT,
    latency_ms INT NOT NULL,
    tokens_in  INT,
    tokens_out INT,
    cost_usd   NUMERIC(10, 6),
    ok         BOOLEAN NOT NULL,
    error      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_extraction_runs_model
    ON extraction_runs (model, created_at DESC);

-- ------------------------------------------------------------------- views
-- The only surface the agent's query_sql tool is allowed to touch. Keeping the
-- agent off the base tables means a prompt injection cannot reach file paths,
-- hashes, or the review queue.
CREATE OR REPLACE VIEW v_invoices AS
SELECT
    i.uuid,
    i.serie,
    i.folio,
    i.fecha_emision,
    i.rfc_emisor,
    s.nombre AS proveedor,
    i.rfc_receptor,
    i.subtotal,
    i.descuento,
    i.total,
    i.moneda,
    i.uso_cfdi,
    i.sat_status
FROM invoices i
JOIN suppliers s ON s.rfc = i.rfc_emisor;

CREATE OR REPLACE VIEW v_line_items AS
SELECT
    i.uuid AS invoice_uuid,
    i.fecha_emision,
    i.rfc_emisor,
    li.line_no,
    li.clave_prod_serv,
    li.clave_unidad,
    li.descripcion,
    li.cantidad,
    li.valor_unitario,
    li.importe,
    li.category
FROM line_items li
JOIN invoices i ON i.id = li.invoice_id;

CREATE OR REPLACE VIEW v_anomalies AS
SELECT
    a.id,
    i.uuid AS invoice_uuid,
    i.rfc_emisor,
    i.total,
    a.kind,
    a.severity,
    a.detail,
    a.explanation,
    a.resolved,
    a.created_at
FROM anomalies a
LEFT JOIN invoices i ON i.id = a.invoice_id;
