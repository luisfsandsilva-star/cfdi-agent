# Eval report

Generated 2026-07-28T21:38:55+00:00 · corpus 300 · seed 1312 · defect rate 25%

Every figure below is produced by `python -m evals.run_eval` against a dedicated `cfdi_eval` database. Nothing here is estimated.

## Ingest

300 documents in 14.06s (21/s)

| status | n |
|---|---:|
| `ok` | 231 |
| `anomaly` | 69 |

## Field accuracy — tier 0 (deterministic XML)

Exact match against the generator's ground truth, read back out of Postgres. Duplicate-UUID submissions are excluded: they are deliberately not inserted, and counting them as misses would penalize correct behaviour.

| field | correct | total | accuracy |
|---|---:|---:|---:|
| `uuid` | 286 | 286 | 100.0% |
| `rfc_emisor` | 286 | 286 | 100.0% |
| `rfc_receptor` | 286 | 286 | 100.0% |
| `subtotal` | 286 | 286 | 100.0% |
| `total` | 286 | 286 | 100.0% |
| `moneda` | 286 | 286 | 100.0% |
| `n_conceptos` | 286 | 286 | 100.0% |

## Anomaly detectors

Scored against injected defects. A firing counts as a false positive only on an invoice carrying no defect at all.

| defect | injected | caught | recall | fired | FP | precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bad_rfc` | 10 | 10 | 1.00 | 10 | 0 | 1.00 | 1.00 |
| `dup_uuid` | 14 | 14 | 1.00 | 14 | 0 | 1.00 | 1.00 |
| `folio_gap` | 12 | 12 | 1.00 | 27 | 8 | 0.60 | 0.75 |
| `line_math` | 4 | 4 | 1.00 | 10 | 0 | 1.00 | 1.00 |
| `price_spike` | 10 | 9 | 0.90 | 9 | 0 | 1.00 | 0.95 |
| `semantic_dup` | 15 | 0 | 0.00 | 0 | 0 | — | — |
| `total_mismatch` | 6 | 6 | 1.00 | 6 | 0 | 1.00 | 1.00 |

### Contextual detectors

These describe an invoice rather than accuse it, so they have no injected ground truth and are reported as counts. `new_supplier` on a first invoice is correct, not a false positive.

| detector | invoices |
|---|---:|
| `new_supplier` | 10 |
| `stale_stamp` | 0 |
| `unknown_catalog_code` | 0 |

### Detectors with no opportunity to fire

These did not report, and that is not the same as reporting nothing. The corpus or the configuration gave them no case to react to.

| detector | why |
|---|---|
| `semantic_duplicate` | no line item was embedded, so the vector stage never ran; EMBED_BASE_URL is http://orin.local:8082/v1 |
| `stale_stamp` | the generator stamps every invoice within 4 hours, so no invoice exceeds the 72-hour limit |
| `unknown_catalog_code` | the generator only emits catalog codes that are in the bundled subset |

## Latency

| percentile | ms |
|---|---:|
| p50 | 15 |
| p95 | 22 |
| max | 25 |

## Cost

- documents processed: **300**
- calls that reached a model: **0**
- total: **$0**
- per invoice: **$0.000000**
- per 1,000 invoices: **$0.00**

The denominator is every document, not only the ones that reached a model. An XML invoice costs nothing because no model is involved — that is the tier-0 argument, stated as a measurement.

## XSD conformance

290/300 validate against the SAT's official CFDI 4.0 schema chain.

Every invalid document is one of the 10 deliberately malformed RFCs. **Of six injected defect kinds, the schema catches one** — duplicates, inflated prices and totals that do not add up are all perfectly schema-valid. Schema conformance says nothing about whether an invoice should be paid.

## Tier 2 — vision path

**not run** — no LLM credentials configured (ANTHROPIC_API_KEY unset and LLM_PROVIDER is not a reachable local server)

Requested models: `claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5`

This section is left visible on purpose. The comparison between the API and a local model on the Orin is the point of the provider seam, and an empty table is an honest statement that it has not been measured yet.

