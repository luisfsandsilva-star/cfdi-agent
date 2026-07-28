# CFDI agent

This software automates accounts-payable work for Mexican invoices (CFDI).
It reads the invoice inbox. It extracts and validates each invoice. It finds
anomalies. It answers questions about expenses in natural language.

*[Léeme en español](README.es.md)*

---

## The main design decision

A CFDI invoice is already structured data. It has a published schema. If you
send the invoice to a large language model, you pay for each document. You also
add a new type of failure: the model can transcribe a value incorrectly. This
failure does not occur when you parse the XML.

Therefore the software sends each document to the lowest-cost layer that gives
a correct result.

| Layer | Function | Cost | Deterministic |
|---|---|---|---|
| **0 — code** | Parse CFDI 4.0 and 3.3 XML. Check arithmetic, RFC values and catalog codes. | $0 | yes |
| **1 — local** | Calculate embeddings. Find semantic duplicates. Read scanned pages. | $0 for each call | no |
| **2 — API** | Run the agent. Explain anomalies. Read difficult PDF files. | measured for each call | no |

The language model does not write to the database. All model output goes
through the same deterministic checks as the XML path. If the model invents a
total, the arithmetic check finds the error. The software then puts the
document in the review queue.

The software uses n8n for orchestration. It keeps the business logic in Python
behind an HTTP interface. A tax rule in a canvas node is a rule that you cannot
test and cannot review.

---

## Measured results

The command `python -m evals.run_eval` calculates these numbers. The test uses
300 synthetic invoices and seed 1312. The test uses a separate database. You
can calculate the numbers again with one command. This section contains no
estimates.

**Field accuracy, layer 0.** The parser is correct for 287 of 287 invoices for
each of these fields: `uuid`, `rfc_emisor`, `rfc_receptor`, `subtotal`,
`total`, `moneda` and `n_conceptos`. This is 100%.

**Anomaly detection**

| Defect | Injected | Recall | Precision | F1 |
|---|---:|---:|---:|---:|
| `bad_rfc` | 16 | 1.00 | 1.00 | 1.00 |
| `dup_uuid` | 13 | 1.00 | 1.00 | 1.00 |
| `line_math` | 12 | 1.00 | 1.00 | 1.00 |
| `price_spike` | 9 | 1.00 | 1.00 | 1.00 |
| `total_mismatch` | 11 | 1.00 | 1.00 | 1.00 |
| `folio_gap` | 12 | 1.00 | **0.48** | 0.65 |

**Speed.** The software processes 33 documents each second. The p50 latency is
15 ms. The p95 latency is 21 ms.

**Cost.** The XML path costs $0.00 for each invoice, because it uses no model.

**Schema.** 284 of 300 invoices are valid against the official SAT schema. The
16 invalid invoices are the invoices with an incorrect RFC. The test corpus
makes these RFC values incorrect on purpose.

### About the folio_gap detector

The `folio_gap` detector has the lowest precision. This README shows the
number. It does not hide it.

Each false positive comes from an invoice with an incorrect issuer RFC. The
software files that invoice under a different supplier. This action makes a gap
in the folio sequence of the correct supplier. This result is possibly correct.
The test corpus makes approximately 5% of the RFC values incorrect, so this
number is higher than in production.

### About schema validation

The XSD validation finds only one of the six defect types. Duplicate invoices,
high prices and incorrect totals are all valid against the schema.

A valid schema does not tell you if you must pay an invoice. This is the reason
for the detectors.

---

## Architecture

```
  ┌──────────── n8n — orchestration, editable in the canvas ────────────┐
  │  Webhook / Gmail ─► POST /ingest ─► If status == "anomaly" ─► Slack │
  │  Cron 09:00 ──────► GET /anomalies/open ─► format ─────────► Slack  │
  └────────────────────────────┬───────────────────────────────────────┘
                               │ HTTP
  ┌────────────────────────────▼─── FastAPI — business logic, Python ──┐
  │  find duplicate file ─► router ─┬─ .xml ─► lxml parser   (layer 0)  │
  │                                 └─ .pdf ─► vision      (layer 1/2)  │
  │                                     │                               │
  │  validate ─┬─ rejected ─► review_queue                              │
  │            └─ accepted ─► Postgres + pgvector                       │
  │                              │                                      │
  │                     embeddings ─► 8 anomaly detectors               │
  └────────────────────────────┬───────────────────────────────────────┘
                               ▼
                 NL agent (read-only access, views only)
```

The software writes one row to the `extraction_runs` table for each document.
The row contains the model name, the latency, the token counts and the cost.
This table supplies the data for the evaluation report and the cost report.

---

## Detectors

| # | Detector | Method | Severity |
|---|---|---|---|
| 1 | `duplicate_uuid` | Find the UUID in the database. The UNIQUE constraint gives a second check. | critical |
| 2 | `semantic_duplicate` | Find an invoice with the same issuer, a total within 1%, and a date within 7 days. Then compare the line-item vectors. Report if the cosine value is more than 0.93. | critical |
| 3 | `price_outlier` | Calculate a robust z-score with the MAD for each supplier and product. Apply a minimum MAD and a minimum difference. | warn |
| 4 | `total_mismatch`, `subtotal_mismatch`, `line_math_mismatch` | Calculate the invoice totals again and compare. | critical |
| 5 | `invalid_rfc` | Apply the RFC pattern from the SAT schema. | critical |
| 6 | `new_supplier` | Report the first invoice from an RFC. | info |
| 7 | `folio_gap` | Find a gap in the folio sequence for each issuer and series. | warn |
| 8 | `stale_stamp` | Report a stamp more than 72 hours after the issue date. | warn |

Each detector writes an `evidence` field in JSONB. This field contains the
values that caused the detector to report. The language model can summarize
this evidence. The language model cannot report an anomaly without evidence.

---

## How to start

```bash
cp .env.example .env
docker compose up -d db                      # Postgres 17 and pgvector
python -m venv .venv && ./.venv/bin/pip install -e '.[api,llm,synth,dev]'

python -m scripts.fetch_xsd                  # download the SAT schemas
python -m synth.generate_cfdi --out data/synth
pytest                                       # no API key, no network
```

Start the API and the workflows:

```bash
uvicorn cfdi_agent.api.main:app --port 8000
docker compose up -d n8n
python -m flows.sync build                   # compile only, n8n not necessary
python -m flows.sync push                    # N8N_API_KEY is necessary
```

Process the invoices and calculate the report:

```bash
python -m cfdi_agent.ingest.watch_dir --once data/synth
python -m evals.run_eval                     # writes evals/report.md
```

Ask a question. This command needs `ANTHROPIC_API_KEY`:

```bash
python -m cfdi_agent.agent.loop -v \
  "¿Cuánto gasté con ACME en Q2 y hubo algo raro?"
```

Use a local model. You do not change the code:

```bash
LLM_PROVIDER=local LLM_BASE_URL=http://orin.local:8080/v1 \
  python -m evals.run_eval
```

---

## n8n workflows as code

You define the workflows in Python. The software compiles them to n8n JSON. It
sends them to n8n through the public API. It also reads them back. Both
directions use the same file.

```python
wf = Workflow("cfdi-invoice-intake")
trigger = wf.add(webhook("Factura recibida", path="cfdi"))
ingest  = wf.add(http_request("Ingestar CFDI", url=f"{api}/ingest", method="POST"))
branch  = wf.add(if_equals("¿Hay anomalía?", left="={{ $json.status }}", right="anomaly"))
wf.chain(trigger, ingest, branch)
wf.connect(branch, alert, port=0)   # the true output
```

```bash
python -m flows.sync push      # define here, run in n8n
# a person changes the canvas
python -m flows.sync export    # then you can read the difference
```

### The export normalizer

An n8n export contains data that changes at each save: identifiers from the
server, the `updatedAt` value, a random `webhookId` for each node, and exact
node positions. If you commit this data, each difference contains much noise.

The normalizer removes this data. It calculates the node identifiers again from
the workflow name and the node name. It moves the positions to a grid. It sorts
the nodes and the keys.

A test simulates a change in the canvas. The test then checks that the
normalized file is the same. A second test checks that a real change is still
visible.

### About typeVersion values

The `typeVersion` value of each node comes from the type catalog of the n8n
installation. Do not write these values from memory. An incorrect
`typeVersion` gives a workflow that n8n imports without an error but that
operates differently.

---

## Security

Suppliers write the line-item descriptions. These descriptions go into the
agent context. A supplier can put an instruction in a description. Therefore
the safety controls are structural. They do not depend on the prompt.

- The agent uses `SET TRANSACTION READ ONLY`. Postgres applies this control.
- The agent reads only three views. It cannot read file paths, file hashes or
  the review queue.
- The agent sends one statement for each query. The software adds a `LIMIT`
  clause and a statement timeout.
- The software also filters SQL keywords. This filter is an additional control.
  It is not the primary control. A regular expression cannot filter SQL
  correctly.

The agent can read the anomalies. The agent cannot create an anomaly. The
detectors find the anomalies before the agent starts.

Invoice XML comes from an external source. The parser does not read external
entities.

---

## Limits of this software

- **The test corpus is synthetic.** The generator makes the invoices from the
  CFDI 4.0 structure. The software validates them against the official schema.
  The repository contains no real invoices. Real invoices contain the RFC
  values of other companies and UUID values that identify real documents.
- **The software does not verify the digital signature.** The `Sello`,
  `Certificado` and `SelloSAT` fields are present but the software does not
  check them. This check needs the certificate chain of the PAC.
- **The SAT status check uses a simulation.** A flag enables the connection to
  the `ConsultaCFDIService` web service.
- **The SAT catalogs are incomplete.** The `c_ClaveProdServ` catalog contains
  approximately 52,000 rows. This repository contains a subset. An unknown code
  gives an `info` message. It does not cause a rejection.
- **The software does not calculate the RFC check digit.** The pattern from the
  SAT schema gives part of this check: the last character can only be a digit or
  the letter `A`. The full algorithm is not implemented. An incorrect
  implementation rejects correct taxpayers.
- **Layer 2 has no measurements.** The vision path and the comparison between
  the API and a local model need credentials. The file `evals/report.md` shows
  this status.

---

## What occurs after a model error

1. The software validates the vision output against a Pydantic schema. If the
   output is incorrect, the document goes to the review queue.
2. The software then calculates the totals again: line items, subtotal, taxes
   and total. A difference gives a critical anomaly.
3. The software writes an invoice to the `invoices` table only after full
   validation. Full validation means that the software knows each error in the
   invoice. It does not mean that the invoice has no errors. An invoice with
   incorrect arithmetic goes into the table **and** gets an anomaly. The
   purpose of the software is to record this invoice.
4. The `review_queue` table contains the documents that the software cannot
   extract, and the documents for a different company.

---

## Repository structure

```
src/cfdi_agent/
  schemas.py          Decimal model, and a separate schema for the model API
  extract/            xml_parser (layer 0), pdf_vision, providers/
  validate/           rules (pure functions), catalogs, xsd_validator
  enrich/             embeddings, semantic duplicates
  db/                 schema.sql, repo, connection
  ingest/             pipeline, watch_dir, dedupe
  agent/              tools (security controls), loop
  api/                FastAPI, the interface for n8n
flows/                n8n as code: builder, definitions, exported JSON
synth/                corpus generator with labels
evals/                measurement and report
xsd/                  SAT schemas and a manifest file
```

---

## Software

Python 3.12 · Postgres 17 with pgvector · FastAPI · n8n · Anthropic API
(`claude-opus-5`) · llama.cpp for the local path · lxml · Pydantic v2 · Docker
Compose
