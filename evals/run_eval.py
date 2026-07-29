"""Measure the pipeline and write `evals/report.md`.

    python -m evals.run_eval
    python -m evals.run_eval --n 500 --seed 1312

The rule this exists to enforce: no number reaches the README unless it came
out of here. Estimates presented as measurements are the fastest way to lose a
reader who knows the domain.

Runs against a dedicated `cfdi_eval` database so a measurement never depends on
what happens to be sitting in the development ledger — and never destroys it.

Tier 0 (deterministic XML) always runs. Tier 2 (vision) runs only when
credentials are present; when they are not, the report says so explicitly
rather than quietly omitting the section and leaving the reader to assume the
path was measured.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

EVAL_DB = "cfdi_eval"

# Which detector kinds count as catching which injected defect. `line_math`
# maps to three because a wrong line importe legitimately breaks the subtotal
# and the total as well — any of them is a catch, not a false positive.
DEFECT_TO_KINDS = {
    "total_mismatch": {"total_mismatch"},
    "line_math": {"line_math_mismatch", "subtotal_mismatch", "total_mismatch"},
    "bad_rfc": {"invalid_rfc"},
    "dup_uuid": {"duplicate_uuid"},
    "folio_gap": {"folio_gap"},
    "price_spike": {"price_outlier"},
    "semantic_dup": {"semantic_duplicate"},
}

# Detectors that fire on context rather than on a defect. Scoring them against
# injected labels would report a false positive for every correct observation.
CONTEXTUAL_KINDS = {"new_supplier", "unknown_catalog_code", "stale_stamp"}

COMPARED_FIELDS = (
    "uuid",
    "rfc_emisor",
    "rfc_receptor",
    "subtotal",
    "total",
    "moneda",
    "n_conceptos",
)


@dataclass
class DetectorScore:
    kind: str
    injected: int
    caught: int
    fired: int
    false_positives: int

    @property
    def recall(self) -> float:
        return self.caught / self.injected if self.injected else float("nan")

    @property
    def precision(self) -> float:
        total = self.caught + self.false_positives
        return self.caught / total if total else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if not (p == p and r == r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)


@dataclass
class EvalReport:
    generated_at: str
    corpus_size: int
    seed: int
    defect_rate: float
    ingest_seconds: float
    status_counts: dict[str, int] = field(default_factory=dict)
    field_accuracy: dict[str, tuple[int, int]] = field(default_factory=dict)
    detectors: list[DetectorScore] = field(default_factory=list)
    contextual_counts: dict[str, int] = field(default_factory=dict)
    unexercised: dict[str, str] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    cost: dict[str, object] = field(default_factory=dict)
    xsd: dict[str, object] = field(default_factory=dict)
    tier2: dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------


def _use_eval_database() -> None:
    import psycopg

    base = os.environ.get("DATABASE_URL", "postgresql://cfdi:cfdi@localhost:5432/cfdi")
    admin = base.rsplit("/", 1)[0] + "/postgres"
    with psycopg.connect(admin, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (EVAL_DB,))
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE "{EVAL_DB}"')

    os.environ["DATABASE_URL"] = base.rsplit("/", 1)[0] + f"/{EVAL_DB}"
    from cfdi_agent.config import get_config

    get_config.cache_clear()

    from cfdi_agent.db.conn import connect
    from cfdi_agent.db.init import apply_schema

    apply_schema()
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "TRUNCATE invoices, suppliers, line_items, taxes, anomalies, "
            "review_queue, extraction_runs, seen_folios, processed_files "
            "RESTART IDENTITY CASCADE"
        )


# The corpus is generated addressed to this RFC, so the harness states it
# rather than reading COMPANY_RFC out of the environment.
#
# It used to inherit the ambient config, and a measurement that depends on
# deployment settings is not a measurement. Pointing COMPANY_RFC at a real
# company — the ordinary thing to do when running against real invoices — sent
# all 300 synthetic documents to `needs_review` and zeroed every number in the
# report, including the field accuracies that are exact by construction.
EVAL_RECEPTOR_RFC = "XAXX010101000"


def _ingest_corpus(corpus_dir: Path, labels: list[dict]) -> tuple[dict[str, int], float]:
    from cfdi_agent.db.conn import connect
    from cfdi_agent.ingest.pipeline import ingest_file

    counts: Counter[str] = Counter()
    started = time.perf_counter()
    for label in labels:
        with connect() as conn:
            outcome = ingest_file(
                conn, corpus_dir / label["file"], company_rfc=EVAL_RECEPTOR_RFC
            )
        counts[outcome.status] += 1
    if counts.get("needs_review", 0) == len(labels):
        # Fail loudly instead of writing a report full of zeros that looks like
        # a catastrophic regression.
        raise SystemExit(
            f"every document was refused. The corpus is addressed to "
            f"{EVAL_RECEPTOR_RFC}; check that the pipeline is being told so."
        )
    return dict(counts), time.perf_counter() - started


def _score_fields(labels: list[dict]) -> dict[str, tuple[int, int]]:
    """Exact-match accuracy per field, against what actually landed in the DB."""
    from cfdi_agent.db.conn import fetch_all

    rows = {
        r["uuid"].upper(): r
        for r in fetch_all(
            """
            SELECT i.uuid::text AS uuid, i.rfc_emisor, i.rfc_receptor,
                   i.subtotal, i.total, i.moneda,
                   (SELECT count(*) FROM line_items l WHERE l.invoice_id = i.id)
                     AS n_conceptos
              FROM invoices i
            """
        )
    }

    scores: dict[str, list[int]] = {f: [0, 0] for f in COMPARED_FIELDS}
    for label in labels:
        # A duplicate-UUID submission is deliberately not inserted; scoring it
        # as a field-accuracy miss would penalize correct behaviour.
        if "dup_uuid" in label["defects"]:
            continue
        row = rows.get(label["uuid"])
        expected = label["expected"]
        for f in COMPARED_FIELDS:
            scores[f][1] += 1
            if row is None:
                continue
            got, want = row[f], expected[f]
            if f in ("subtotal", "total"):
                ok = Decimal(str(got)) == Decimal(str(want))
            elif f == "n_conceptos":
                ok = int(got) == int(want)
            else:
                ok = str(got).upper() == str(want).upper()
            if ok:
                scores[f][0] += 1
    return {f: (c, t) for f, (c, t) in scores.items()}


def _score_detectors(labels: list[dict]) -> tuple[list[DetectorScore], dict[str, int]]:
    from cfdi_agent.db.conn import fetch_all

    fired: dict[str, set[str]] = {}
    for row in fetch_all(
        """
        SELECT a.kind, i.uuid::text AS uuid
          FROM anomalies a JOIN invoices i ON i.id = a.invoice_id
        """
    ):
        fired.setdefault(row["kind"], set()).add(row["uuid"].upper())

    by_uuid = {lb["uuid"]: lb for lb in labels}
    scores: list[DetectorScore] = []
    for defect, kinds in sorted(DEFECT_TO_KINDS.items()):
        injected_uuids = {lb["uuid"] for lb in labels if defect in lb["defects"]}
        hit_uuids: set[str] = set()
        for kind in kinds:
            hit_uuids |= fired.get(kind, set())

        caught = len(injected_uuids & hit_uuids)
        # A firing is a false positive only when the invoice carries no defect
        # this detector family could legitimately be reacting to.
        false_positives = sum(
            1
            for u in hit_uuids - injected_uuids
            if u in by_uuid and not (set(by_uuid[u]["defects"]) & set(DEFECT_TO_KINDS))
        )
        scores.append(
            DetectorScore(
                kind=defect,
                injected=len(injected_uuids),
                caught=caught,
                fired=len(hit_uuids),
                false_positives=false_positives,
            )
        )

    contextual = {k: len(fired.get(k, set())) for k in sorted(CONTEXTUAL_KINDS)}
    return scores, contextual


def _score_runs() -> tuple[dict[str, float], dict[str, object]]:
    from cfdi_agent.db.conn import fetch_all

    rows = fetch_all(
        "SELECT tier, provider, model, latency_ms, cost_usd, ok FROM extraction_runs"
    )
    latencies = sorted(r["latency_ms"] for r in rows)
    quantiles = {}
    if latencies:
        quantiles = {
            "p50": float(statistics.median(latencies)),
            "p95": float(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]),
            "max": float(latencies[-1]),
        }

    priced = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    total = sum(priced, Decimal("0")) if priced else Decimal("0")
    cost = {
        "documents": len(rows),
        "llm_calls": sum(1 for r in rows if r["tier"] != 0),
        "total_usd": str(total),
        "per_invoice_usd": str(
            (total / len(rows)).quantize(Decimal("0.000001")) if rows else Decimal("0")
        ),
        "per_1000_usd": str(
            (total / len(rows) * 1000).quantize(Decimal("0.01")) if rows else Decimal("0")
        ),
    }
    return quantiles, cost


def _score_xsd(corpus_dir: Path, labels: list[dict]) -> dict[str, object]:
    from cfdi_agent.validate.xsd_validator import schemas_available, validate_file

    if not schemas_available():
        return {"ran": False, "reason": "SAT schemas not vendored (scripts/fetch_xsd)"}
    invalid = [lb["file"] for lb in labels if validate_file(corpus_dir / lb["file"])]
    schema_breaking = {lb["file"] for lb in labels if "bad_rfc" in lb["defects"]}
    return {
        "ran": True,
        "total": len(labels),
        "invalid": len(invalid),
        "invalid_are_all_injected_defects": set(invalid) == schema_breaking,
        "schema_breaking_defects": len(schema_breaking),
    }


def _tier2_status(models: list[str]) -> dict[str, object]:
    from cfdi_agent.config import get_config

    cfg = get_config()
    if not cfg.llm_enabled:
        return {
            "ran": False,
            "reason": (
                "no LLM credentials configured (ANTHROPIC_API_KEY unset and "
                "LLM_PROVIDER is not a reachable local server)"
            ),
            "requested_models": models,
        }
    return {
        "ran": False,
        "reason": (
            "credentials are present, but the corpus is XML only. The vision "
            "path needs PDF or image invoices, which this generator does not "
            "produce yet"
        ),
        "requested_models": models,
    }


# --------------------------------------------------------------------------


def _unexercised(labels: list[dict]) -> dict[str, str]:
    """Detectors the corpus gave no opportunity to fire, and why.

    A zero in a results table reads as "this ran and found nothing". For a
    detector that was never reachable it means something else entirely, and the
    difference is exactly what hid detector 2 for a week.
    """
    from cfdi_agent.config import get_config
    from cfdi_agent.db.conn import fetch_one

    out: dict[str, str] = {}
    injected = {d for lb in labels for d in lb["defects"]}

    # Ask the ledger, not the configuration. EMBED_BASE_URL can point at a host
    # that is simply not running, which the config cannot know and which looks
    # identical to a detector that found nothing.
    embedded = fetch_one(
        "SELECT count(*) AS n FROM line_items WHERE embedding IS NOT NULL"
    )["n"]
    if embedded == 0:
        configured = get_config().embed_base_url or "(unset)"
        out["semantic_duplicate"] = (
            f"no line item was embedded, so the vector stage never ran; "
            f"EMBED_BASE_URL is {configured}"
        )
    elif "semantic_dup" not in injected:
        out["semantic_duplicate"] = "no semantic duplicates injected at this seed"

    if fetch_one("SELECT count(*) AS n FROM anomalies WHERE kind = 'stale_stamp'")["n"] == 0:
        out["stale_stamp"] = (
            "the generator stamps every invoice within 4 hours, so no invoice "
            "exceeds the 72-hour limit"
        )
    if fetch_one(
        "SELECT count(*) AS n FROM anomalies WHERE kind = 'unknown_catalog_code'"
    )["n"] == 0:
        out["unknown_catalog_code"] = (
            "the generator only emits catalog codes that are in the bundled subset"
        )
    return out


def run(n: int, seed: int, defect_rate: float, models: list[str]) -> EvalReport:
    from synth.generate_cfdi import generate

    corpus_dir = Path("data/eval")
    labels = generate(
        n=n,
        defect_rate=defect_rate,
        out_dir=corpus_dir,
        labels_path=Path("evals/datasets/eval_labels.jsonl"),
        seed=seed,
        n_suppliers=6,
        receptor_rfc=EVAL_RECEPTOR_RFC,
        receptor_nombre="Mi Empresa SA de CV",
    )

    _use_eval_database()
    status_counts, elapsed = _ingest_corpus(corpus_dir, labels)
    detectors, contextual = _score_detectors(labels)
    latency, cost = _score_runs()

    unexercised = _unexercised(labels)

    return EvalReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        corpus_size=len(labels),
        seed=seed,
        defect_rate=defect_rate,
        ingest_seconds=round(elapsed, 2),
        status_counts=status_counts,
        field_accuracy=_score_fields(labels),
        detectors=detectors,
        contextual_counts=contextual,
        latency_ms=latency,
        cost=cost,
        xsd=_score_xsd(corpus_dir, labels),
        tier2=_tier2_status(models),
        unexercised=unexercised,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1312)
    ap.add_argument("--defect-rate", type=float, default=0.25)
    ap.add_argument(
        "--models",
        default="claude-opus-5,claude-sonnet-5,claude-haiku-4-5",
        help="tier-2 models to compare, when credentials allow",
    )
    ap.add_argument("--out", type=Path, default=Path("evals/report.md"))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    try:
        report = run(args.n, args.seed, args.defect_rate, args.models.split(","))
    except Exception as exc:  # noqa: BLE001 - the operator needs the real cause
        print(f"eval failed: {exc}", file=sys.stderr)
        return 1

    from evals.report import render_markdown, to_json

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {args.out}")
    if args.json:
        args.json.write_text(json.dumps(to_json(report), indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
