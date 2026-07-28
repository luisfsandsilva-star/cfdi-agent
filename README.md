# CFDI agent

Accounts-payable automation for Mexican invoices. Reads the billing inbox,
extracts and validates CFDIs, detects anomalies, and answers questions about
spending in natural language.

*[Léeme en español](README.es.md)*

---

## The design decision this project is about

> **A CFDI is already structured data with a published schema. Running it
> through an LLM costs money per document and introduces a failure mode —
> transcription error — that does not otherwise exist.**

So every document is routed to the cheapest layer that can answer it correctly:

| Tier | Handles | Cost | Deterministic |
|---|---|---|---|
| **0 — code** | CFDI 4.0/3.3 XML parsing, arithmetic, RFC and catalog validation | $0 | yes |
| **1 — local** | Embeddings, semantic duplicate search, scanned-page transcription | $0 marginal | no |
| **2 — API** | The agent loop, anomaly explanations, hard PDFs | measured per call | no |

**The LLM never writes to the database.** Everything it produces passes the same
deterministic validation as the XML path. A hallucinated total arrives as a
well-formed string, fails arithmetic, and lands in the review queue.

Orchestration is n8n; domain logic is tested Python behind HTTP. A tax rule
expressed as canvas nodes is a rule nothing can test and nobody can review.

---

## Measured results

From `python -m evals.run_eval` — 300 synthetic invoices, seed 1312, against a
dedicated database. Regenerate with one command; nothing below is estimated.

**Field accuracy, tier 0** — 100% on `uuid`, `rfc_emisor`, `rfc_receptor`,
`subtotal`, `total`, `moneda`, `n_conceptos` (287/287 each).

**Anomaly detection**

| defect | injected | recall | precision | F1 |
|---|---:|---:|---:|---:|
| `bad_rfc` | 16 | 1.00 | 1.00 | 1.00 |
| `dup_uuid` | 13 | 1.00 | 1.00 | 1.00 |
| `line_math` | 12 | 1.00 | 1.00 | 1.00 |
| `price_spike` | 9 | 1.00 | 1.00 | 1.00 |
| `total_mismatch` | 11 | 1.00 | 1.00 | 1.00 |
| `folio_gap` | 12 | 1.00 | **0.48** | 0.65 |

**Throughput** 33 documents/s · p50 15 ms · p95 21 ms
**Cost** $0.00 per invoice on the XML path — no model is involved
**Schema** 284/300 validate against the SAT's official XSD chain; the 16
failures are exactly the deliberately malformed RFCs

`folio_gap` is the weak one and the number is published rather than buried.
Every false positive traces to an invoice whose issuer RFC was malformed, which
files it under a different supplier and leaves a hole in the real one's
sequence. Arguably the correct verdict, and inflated here by a corpus that
corrupts RFCs at ~5%.

**Of six injected defect kinds, XSD validation catches one.** Duplicates,
inflated prices and totals that do not add up are all perfectly schema-valid.
Passing schema validation says nothing about whether an invoice should be paid —
which is the entire argument for the detector suite.

---

## Architecture

```
  ┌──────────── n8n — orchestration, editable in the canvas ────────────┐
  │  Webhook / Gmail ─► POST /ingest ─► If status == "anomaly" ─► Slack │
  │  Cron 09:00 ──────► GET /anomalies/open ─► format ─────────► Slack  │
  └────────────────────────────┬───────────────────────────────────────┘
                               │ HTTP
  ┌────────────────────────────▼─── FastAPI — domain, Python ──────────┐
  │  dedupe by file hash ─► router ─┬─ .xml ─► lxml parser    (tier 0)  │
  │                                 └─ .pdf ─► vision        (tier 1/2) │
  │                                     │                               │
  │  validate ─┬─ rejected ─► review_queue                              │
  │            └─ accepted ─► Postgres + pgvector                       │
  │                              │                                      │
  │                     embeddings ─► 8 anomaly detectors               │
  └────────────────────────────┬───────────────────────────────────────┘
                               ▼
                    NL agent (read-only, views only)
```

Every document writes a row to `extraction_runs` — model, latency, tokens,
cost. That table *is* the eval harness and the cost dashboard.

---

## Detectors

| # | Detector | Method | Severity |
|---|---|---|---|
| 1 | `duplicate_uuid` | UNIQUE constraint plus explicit check | critical |
| 2 | `semantic_duplicate` | Same issuer, total ±1%, date ±7d, cosine > 0.93 over line-item centroids | critical |
| 3 | `price_outlier` | MAD-based robust z per (supplier, product), with a floor and a materiality gate | warn |
| 4 | `total_mismatch` / `subtotal_mismatch` / `line_math_mismatch` | Re-adds the invoice | critical |
| 5 | `invalid_rfc` | The SAT's own `t_RFC` pattern | critical |
| 6 | `new_supplier` | First invoice from this RFC | info |
| 7 | `folio_gap` | Sequence break per (issuer, series) | warn |
| 8 | `stale_stamp` | Stamped more than 72h after issue | warn |

Each emits `evidence` as JSONB — the exact values that fired the rule. The LLM
may summarize that evidence; it cannot invent a finding without it.

---

## Quickstart

```bash
cp .env.example .env
docker compose up -d db                      # Postgres 17 + pgvector
python -m venv .venv && ./.venv/bin/pip install -e '.[api,llm,synth,dev]'

python -m scripts.fetch_xsd                  # vendor the SAT schema chain
python -m synth.generate_cfdi --out data/synth
pytest                                       # no API key, no network

python -m cfdi_agent.ingest.watch_dir --once data/synth
python -m evals.run_eval                     # writes evals/report.md
```

Serve the API and the flows:

```bash
uvicorn cfdi_agent.api.main:app --port 8000
docker compose up -d n8n
python -m flows.sync build                   # compile, no n8n needed
python -m flows.sync push                    # needs N8N_API_KEY
```

Ask a question (needs `ANTHROPIC_API_KEY`):

```bash
python -m cfdi_agent.agent.loop -v \
  "¿Cuánto gasté con ACME en Q2 y hubo algo raro?"
```

Run against a local model instead — no code changes:

```bash
LLM_PROVIDER=local LLM_BASE_URL=http://orin.local:8080/v1 \
  python -m evals.run_eval
```

---

## n8n as code

Workflows are defined in Python, compiled to n8n JSON, pushed over the Public
API, and exported back. Both directions are the same file.

```python
wf = Workflow("cfdi-invoice-intake")
trigger = wf.add(webhook("Factura recibida", path="cfdi"))
ingest  = wf.add(http_request("Ingestar CFDI", url=f"{api}/ingest", method="POST"))
branch  = wf.add(if_equals("¿Hay anomalía?", left="={{ $json.status }}", right="anomaly"))
wf.chain(trigger, ingest, branch)
wf.connect(branch, alert, port=0)   # true
```

```bash
python -m flows.sync push      # define here, run there
# ...someone rearranges the canvas...
python -m flows.sync export    # and the diff is readable
```

The export normalizer is what makes that last step real: a raw n8n export
carries server-assigned ids, `updatedAt`, a per-node random `webhookId` and
pixel-exact positions. Commit that and every diff is noise, so nobody reads
them, so the round-trip is abandoned within a week. Normalizing drops volatile
state, recomputes node ids deterministically, snaps positions to a grid, and
sorts. A test simulates a canvas edit and asserts the normalized form is
byte-identical — with a companion test that a *real* edit still shows up.

Node `typeVersion` values were read out of the running n8n's type catalog, not
written from memory. A wrong one produces a workflow that imports without
complaint and then behaves differently.

---

## Security

Line-item descriptions are written by suppliers and end up in the agent's
context. "Ignore previous instructions" is something a vendor can bill you for,
so the agent's safety properties are structural rather than prompt-based:

- `SET TRANSACTION READ ONLY`, enforced by Postgres
- A view allowlist — file paths, hashes and the review queue are unreachable
- Single statement only, forced `LIMIT`, statement timeout
- Keyword filtering as defence in depth, explicitly *not* the backstop:
  regex-based SQL filtering is a losing game

The agent can read anomalies but cannot create them. Detection already happened,
deterministically, before it saw anything.

Invoice XML is untrusted input: the parser resolves no external entities.

---

## What this does not do

Stated plainly, because a demo that overclaims is worse than one with a short
honest list.

- **The corpus is synthetic.** Generated from the CFDI 4.0 structure and
  validated against the official XSD, but no real invoices are included — real
  ones carry third-party RFCs and traceable UUIDs that do not belong in a public
  repository.
- **Digital seals are not verified.** `Sello`, `Certificado` and `SelloSAT` are
  present but never checked; doing it properly needs the PAC's certificate
  chain.
- **SAT status is mocked by default.** The live `ConsultaCFDIService` lookup is
  behind a flag.
- **The SAT catalogs are a subset.** `c_ClaveProdServ` alone is ~52,000 rows. An
  unknown code produces an `info` note, never a rejection.
- **The RFC check digit is not computed.** Adopting the SAT's own regex recovers
  part of it — the final character can only be `[0-9A]` — but the full algorithm
  is not implemented, because getting it wrong would reject valid taxpayers.
- **Tier 2 is unmeasured.** The vision path and the API-vs-local comparison need
  credentials; `evals/report.md` says so rather than leaving a blank.

---

## Failure path

What happens when the model gets it wrong:

1. Vision output is validated against a Pydantic schema. Malformed → review queue.
2. It is then re-added: line items, subtotal, taxes, total. Any mismatch → a
   critical anomaly.
3. Nothing reaches `invoices` that has not been fully validated — where
   "validated" means *we know precisely what is wrong with it*, not that nothing
   is wrong. An invoice that fails arithmetic is persisted **and** flagged,
   because recording it is the entire point.
4. `review_queue` holds documents we could not extract confidently, or that are
   addressed to another company.

---

## Layout

```
src/cfdi_agent/
  schemas.py          canonical Decimal model + constraint-free LLM schema
  extract/            xml_parser (tier 0), pdf_vision, providers/
  validate/           rules (pure), catalogs, xsd_validator
  enrich/             embeddings, semantic duplicates
  db/                 schema.sql, repo, connection
  ingest/             pipeline, watch_dir, dedupe
  agent/              tools (security boundary), loop
  api/                FastAPI — the contract n8n consumes
flows/                n8n as code: builder, definitions, exported JSON
synth/                corpus generator with free ground truth
evals/                harness and report
xsd/                  vendored SAT schema chain + provenance manifest
```

---

## Stack

Python 3.12 · Postgres 17 + pgvector · FastAPI · n8n · Anthropic API
(`claude-opus-5`) · llama.cpp for the local path · lxml · Pydantic v2 · Docker
Compose
