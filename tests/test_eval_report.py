"""Tests for the eval harness's scoring and reporting.

This module produces the numbers that end up in the README, so its own failure
modes matter more than most. Two in particular:

  * a metric that was not measured must be printed as "not run" with a reason,
    never omitted and never rendered as zero — a blank cell reads as "nothing to
    report", which is a different claim entirely
  * NaN (no samples) must not render as a number
"""

from __future__ import annotations

from evals.report import render_markdown, to_json
from evals.run_eval import DetectorScore, EvalReport


def _report(**overrides) -> EvalReport:
    base = {
        "generated_at": "2026-07-28T02:00:00+00:00",
        "corpus_size": 300,
        "seed": 1312,
        "defect_rate": 0.25,
        "ingest_seconds": 9.0,
        "status_counts": {"ok": 210, "anomaly": 90},
        "field_accuracy": {"uuid": (287, 287), "total": (280, 287)},
        "detectors": [
            DetectorScore("total_mismatch", injected=11, caught=11, fired=11,
                          false_positives=0),
            DetectorScore("folio_gap", injected=12, caught=12, fired=36,
                          false_positives=13),
        ],
        "contextual_counts": {"new_supplier": 10},
        "latency_ms": {"p50": 15.0, "p95": 21.0, "max": 25.0},
        "cost": {
            "documents": 300, "llm_calls": 0, "total_usd": "0",
            "per_invoice_usd": "0.000000", "per_1000_usd": "0.00",
        },
        "xsd": {"ran": True, "total": 300, "invalid": 16,
                "invalid_are_all_injected_defects": True,
                "schema_breaking_defects": 16},
        "tier2": {"ran": False, "reason": "no LLM credentials configured",
                  "requested_models": ["claude-opus-5"]},
    }
    base.update(overrides)
    return EvalReport(**base)


# ------------------------------------------------------------------ scoring


def test_detector_scores_are_arithmetic() -> None:
    d = DetectorScore("x", injected=10, caught=8, fired=12, false_positives=4)
    assert d.recall == 0.8
    assert d.precision == 8 / 12
    assert abs(d.f1 - 2 * (8 / 12) * 0.8 / ((8 / 12) + 0.8)) < 1e-9


def test_a_detector_with_no_samples_is_nan_not_zero() -> None:
    """Zero would read as 'it failed'; NaN reads as 'nothing was tested'."""
    d = DetectorScore("x", injected=0, caught=0, fired=0, false_positives=0)
    assert d.recall != d.recall
    assert d.f1 != d.f1


def test_perfect_detector_scores_one() -> None:
    d = DetectorScore("x", injected=5, caught=5, fired=5, false_positives=0)
    assert d.recall == 1.0 and d.precision == 1.0 and d.f1 == 1.0


# ---------------------------------------------------------------- rendering


def test_report_renders_every_section() -> None:
    md = render_markdown(_report())
    for heading in (
        "# Eval report",
        "## Ingest",
        "## Field accuracy",
        "## Anomaly detectors",
        "## Latency",
        "## Cost",
        "## XSD conformance",
        "## Tier 2",
    ):
        assert heading in md, heading


def test_unmeasured_tier2_says_not_run_with_a_reason() -> None:
    """The load-bearing honesty property of the whole report."""
    md = render_markdown(_report())
    assert "**not run**" in md
    assert "no LLM credentials configured" in md
    # And it must not be quietly dropped instead.
    assert "## Tier 2" in md


def test_unmeasured_xsd_says_not_run() -> None:
    md = render_markdown(
        _report(xsd={"ran": False, "reason": "SAT schemas not vendored"})
    )
    assert "not run — SAT schemas not vendored" in md


def test_nan_renders_as_a_dash_not_a_number() -> None:
    md = render_markdown(
        _report(
            detectors=[
                DetectorScore("never_injected", injected=0, caught=0, fired=0,
                              false_positives=0)
            ]
        )
    )
    row = next(line for line in md.splitlines() if "never_injected" in line)
    assert "—" in row
    assert "nan" not in row.lower()


def test_a_weak_detector_is_reported_not_hidden() -> None:
    """folio_gap's precision is published; a report that buries it is worthless."""
    md = render_markdown(_report())
    row = next(line for line in md.splitlines() if "folio_gap" in line)
    assert "0.48" in row


def test_xsd_section_states_the_schema_catches_little() -> None:
    md = render_markdown(_report())
    assert "the schema catches one" in md


def test_field_accuracy_shows_a_shortfall() -> None:
    md = render_markdown(_report())
    total_row = next(line for line in md.splitlines() if "`total`" in line)
    assert "97.6%" in total_row


# --------------------------------------------------------------------- json


def test_json_export_round_trips_the_numbers() -> None:
    payload = to_json(_report())
    assert payload["corpus_size"] == 300
    assert payload["detectors"][0]["recall"] == 1.0
    assert payload["field_accuracy"]["uuid"] == {"correct": 287, "total": 287}
    assert payload["tier2"]["ran"] is False


def test_json_export_uses_null_for_undefined_metrics() -> None:
    """NaN is not valid JSON; None is the honest encoding of 'no samples'."""
    payload = to_json(
        _report(
            detectors=[
                DetectorScore("x", injected=0, caught=0, fired=0, false_positives=0)
            ]
        )
    )
    assert payload["detectors"][0]["recall"] is None
    assert payload["detectors"][0]["f1"] is None
