"""Render an `EvalReport` as Markdown.

Kept separate from the measurement so the report can be regenerated from
stored numbers, and so nothing here can accidentally influence what was
measured.

The formatting rule throughout: a metric that was not measured is printed as
"not run" with the reason, never omitted and never shown as zero. A blank in a
results table reads as "nothing to report"; an explicit "not run — no
credentials" reads as what it is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from evals.run_eval import EvalReport


def _pct(numerator: int, denominator: int) -> str:
    if not denominator:
        return "—"
    return f"{numerator / denominator:.1%}"


def _num(value: float) -> str:
    return "—" if value != value else f"{value:.2f}"


def render_markdown(r: EvalReport) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Eval report")
    add("")
    add(f"Generated {r.generated_at} · corpus {r.corpus_size} · seed {r.seed} · "
        f"defect rate {r.defect_rate:.0%}")
    add("")
    add("Every figure below is produced by `python -m evals.run_eval` against a "
        "dedicated `cfdi_eval` database. Nothing here is estimated.")
    add("")

    # ---------------------------------------------------------------- ingest
    add("## Ingest")
    add("")
    rate = r.corpus_size / r.ingest_seconds if r.ingest_seconds else 0
    add(f"{r.corpus_size} documents in {r.ingest_seconds}s ({rate:.0f}/s)")
    add("")
    add("| status | n |")
    add("|---|---:|")
    for status, n in sorted(r.status_counts.items(), key=lambda kv: -kv[1]):
        add(f"| `{status}` | {n} |")
    add("")

    # -------------------------------------------------------- field accuracy
    add("## Field accuracy — tier 0 (deterministic XML)")
    add("")
    add("Exact match against the generator's ground truth, read back out of "
        "Postgres. Duplicate-UUID submissions are excluded: they are "
        "deliberately not inserted, and counting them as misses would penalize "
        "correct behaviour.")
    add("")
    add("| field | correct | total | accuracy |")
    add("|---|---:|---:|---:|")
    for name, (correct, total) in r.field_accuracy.items():
        add(f"| `{name}` | {correct} | {total} | {_pct(correct, total)} |")
    add("")

    # ------------------------------------------------------------- detectors
    add("## Anomaly detectors")
    add("")
    add("Scored against injected defects. A firing counts as a false positive "
        "only on an invoice carrying no defect at all.")
    add("")
    add("| defect | injected | caught | recall | fired | FP | precision | F1 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for d in r.detectors:
        add(
            f"| `{d.kind}` | {d.injected} | {d.caught} | {_num(d.recall)} | "
            f"{d.fired} | {d.false_positives} | {_num(d.precision)} | {_num(d.f1)} |"
        )
    add("")

    if r.contextual_counts:
        add("### Contextual detectors")
        add("")
        add("These describe an invoice rather than accuse it, so they have no "
            "injected ground truth and are reported as counts. `new_supplier` "
            "on a first invoice is correct, not a false positive.")
        add("")
        add("| detector | invoices |")
        add("|---|---:|")
        for kind, n in r.contextual_counts.items():
            add(f"| `{kind}` | {n} |")
        add("")

    # --------------------------------------------------------------- latency
    add("## Latency")
    add("")
    if r.latency_ms:
        add("| percentile | ms |")
        add("|---|---:|")
        for label in ("p50", "p95", "max"):
            if label in r.latency_ms:
                add(f"| {label} | {r.latency_ms[label]:.0f} |")
    else:
        add("not run — no extraction runs recorded")
    add("")

    # ------------------------------------------------------------------ cost
    add("## Cost")
    add("")
    add(f"- documents processed: **{r.cost.get('documents')}**")
    add(f"- calls that reached a model: **{r.cost.get('llm_calls')}**")
    add(f"- total: **${r.cost.get('total_usd')}**")
    add(f"- per invoice: **${r.cost.get('per_invoice_usd')}**")
    add(f"- per 1,000 invoices: **${r.cost.get('per_1000_usd')}**")
    add("")
    add("The denominator is every document, not only the ones that reached a "
        "model. An XML invoice costs nothing because no model is involved — "
        "that is the tier-0 argument, stated as a measurement.")
    add("")

    # ------------------------------------------------------------------- xsd
    add("## XSD conformance")
    add("")
    if not r.xsd.get("ran"):
        add(f"not run — {r.xsd.get('reason')}")
    else:
        valid = r.xsd["total"] - r.xsd["invalid"]
        add(f"{valid}/{r.xsd['total']} validate against the SAT's official "
            f"CFDI 4.0 schema chain.")
        add("")
        if r.xsd.get("invalid_are_all_injected_defects"):
            add(f"Every invalid document is one of the {r.xsd['schema_breaking_defects']} "
                "deliberately malformed RFCs. **Of six injected defect kinds, the "
                "schema catches one** — duplicates, inflated prices and totals that "
                "do not add up are all perfectly schema-valid. Schema conformance "
                "says nothing about whether an invoice should be paid.")
        else:
            add("Some invalid documents are **not** accounted for by injected "
                "defects — the generator has drifted from the schema.")
    add("")

    # ---------------------------------------------------------------- tier 2
    add("## Tier 2 — vision path")
    add("")
    if r.tier2.get("ran"):
        add("| model | field accuracy | p95 ms | $/invoice |")
        add("|---|---:|---:|---:|")
        for row in r.tier2.get("rows", []):
            add(f"| `{row['model']}` | {row['accuracy']} | {row['p95']} | {row['cost']} |")
    else:
        add(f"**not run** — {r.tier2.get('reason')}")
        add("")
        add("Requested models: "
            + ", ".join(f"`{m}`" for m in r.tier2.get("requested_models", [])))
        add("")
        add("This section is left visible on purpose. The comparison between the "
            "API and a local model on the Orin is the point of the provider "
            "seam, and an empty table is an honest statement that it has not "
            "been measured yet.")
    add("")

    return "\n".join(lines) + "\n"


def to_json(r: EvalReport) -> dict:
    return {
        "generated_at": r.generated_at,
        "corpus_size": r.corpus_size,
        "seed": r.seed,
        "defect_rate": r.defect_rate,
        "ingest_seconds": r.ingest_seconds,
        "status_counts": r.status_counts,
        "field_accuracy": {
            k: {"correct": c, "total": t} for k, (c, t) in r.field_accuracy.items()
        },
        "detectors": [
            {
                "kind": d.kind,
                "injected": d.injected,
                "caught": d.caught,
                "fired": d.fired,
                "false_positives": d.false_positives,
                "recall": None if d.recall != d.recall else round(d.recall, 4),
                "precision": None if d.precision != d.precision else round(d.precision, 4),
                "f1": None if d.f1 != d.f1 else round(d.f1, 4),
            }
            for d in r.detectors
        ],
        "contextual_counts": r.contextual_counts,
        "latency_ms": r.latency_ms,
        "cost": r.cost,
        "xsd": r.xsd,
        "tier2": r.tier2,
    }
