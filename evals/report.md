# Eval report

Generated 2026-07-28T04:49:10+00:00 · corpus 300 · seed 1312 · defect rate 25%

Every figure below is produced by `python -m evals.run_eval` against a dedicated `cfdi_eval` database. Nothing here is estimated.

## Ingest

300 documents in 8.68s (35/s)

| status | n |
|---|---:|
| `ok` | 210 |
| `anomaly` | 90 |

## Field accuracy — tier 0 (deterministic XML)

Exact match against the generator's ground truth, read back out of Postgres. Duplicate-UUID submissions are excluded: they are deliberately not inserted, and counting them as misses would penalize correct behaviour.

| field | correct | total | accuracy |
|---|---:|---:|---:|
| `uuid` | 287 | 287 | 100.0% |
| `rfc_emisor` | 287 | 287 | 100.0% |
| `rfc_receptor` | 287 | 287 | 100.0% |
| `subtotal` | 287 | 287 | 100.0% |
| `total` | 287 | 287 | 100.0% |
| `moneda` | 287 | 287 | 100.0% |
| `n_conceptos` | 287 | 287 | 100.0% |

## Anomaly detectors

Scored against injected defects. A firing counts as a false positive only on an invoice carrying no defect at all.

| defect | injected | caught | recall | fired | FP | precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bad_rfc` | 16 | 16 | 1.00 | 16 | 0 | 1.00 | 1.00 |
| `dup_uuid` | 13 | 13 | 1.00 | 13 | 0 | 1.00 | 1.00 |
| `folio_gap` | 12 | 12 | 1.00 | 36 | 13 | 0.48 | 0.65 |
| `line_math` | 12 | 12 | 1.00 | 23 | 0 | 1.00 | 1.00 |
| `price_spike` | 9 | 9 | 1.00 | 9 | 0 | 1.00 | 1.00 |
| `total_mismatch` | 11 | 11 | 1.00 | 11 | 0 | 1.00 | 1.00 |

### Contextual detectors

These describe an invoice rather than accuse it, so they have no injected ground truth and are reported as counts. `new_supplier` on a first invoice is correct, not a false positive.

| detector | invoices |
|---|---:|
| `new_supplier` | 10 |
| `stale_stamp` | 0 |
| `unknown_catalog_code` | 0 |

## Latency

| percentile | ms |
|---|---:|
| p50 | 14 |
| p95 | 21 |
| max | 25 |

## Cost

- documents processed: **300**
- calls that reached a model: **0**
- total: **$0**
- per invoice: **$0.000000**
- per 1,000 invoices: **$0.00**

The denominator is every document, not only the ones that reached a model. An XML invoice costs nothing because no model is involved — that is the tier-0 argument, stated as a measurement.

## XSD conformance

284/300 validate against the SAT's official CFDI 4.0 schema chain.

Every invalid document is one of the 16 deliberately malformed RFCs. **Of six injected defect kinds, the schema catches one** — duplicates, inflated prices and totals that do not add up are all perfectly schema-valid. Schema conformance says nothing about whether an invoice should be paid.

## Tier 2 — vision path

**not run** — no LLM credentials configured (ANTHROPIC_API_KEY unset and LLM_PROVIDER is not a reachable local server)

Requested models: `claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5`

This section is left visible on purpose. The comparison between the API and a local model on the Orin is the point of the provider seam, and an empty table is an honest statement that it has not been measured yet.

