"""Conformance of the synthetic corpus against the SAT's official XSD.

This suite exists because "structurally faithful to CFDI 4.0" was, until it
was written, an unverified claim about a hand-written Jinja template. On first
run it failed 60/60 and surfaced three real bugs:

  * `NoCertificado` and `NoCertificadoSAT` are exactly 20 digits; the generator
    emitted 17.
  * The RFC pattern's final character is `[0-9A]`, not `[0-9A-Z]` — it is the
    check digit, and the algorithm can only produce a digit or 'A'. Roughly two
    thirds of generated RFCs were invalid, and the project's own `RFC_RE` was
    accepting RFCs the SAT rejects.
  * Month and day inside the RFC are range-checked by the schema.

The second test is the one worth reading twice: schema validity and business
correctness are almost disjoint. Of the six defect kinds the generator injects,
the XSD catches exactly one. A duplicated invoice, an inflated price, a total
that does not add up — all perfectly schema-valid. Passing XSD validation says
nothing about whether an invoice should be paid, which is the entire reason the
detector suite exists.

Requires `python -m scripts.fetch_xsd`. Skipped, not failed, when the schemas
have not been vendored — the deterministic core must stay runnable offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cfdi_agent.validate.xsd_validator import (
    schemas_available,
    validate_bytes,
    validate_file,
)
from synth.generate_cfdi import generate

pytestmark = pytest.mark.skipif(
    not schemas_available(),
    reason="SAT XSD chain not vendored; run `python -m scripts.fetch_xsd`",
)

# The only injected defect that is also a schema violation.
SCHEMA_BREAKING_DEFECTS = {"bad_rfc"}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict]]:
    out = tmp_path_factory.mktemp("xsd_corpus")
    labels = generate(
        n=60,
        defect_rate=0.3,
        out_dir=out,
        labels_path=out / "labeled.jsonl",
        seed=4242,
        n_suppliers=6,
        receptor_rfc="XAXX010101000",
        receptor_nombre="Mi Empresa SA de CV",
    )
    return out, labels


def test_generated_invoices_are_xsd_valid(corpus: tuple[Path, list[dict]]) -> None:
    """Every invoice without a schema-breaking defect must validate."""
    out, labels = corpus
    failures: list[str] = []
    for label in labels:
        if SCHEMA_BREAKING_DEFECTS & set(label["defects"]):
            continue
        errors = validate_file(out / label["file"])
        if errors:
            failures.append(f"{label['file']}: {errors[0]}")
    assert not failures, "schema violations:\n" + "\n".join(failures[:10])


def test_injected_bad_rfc_really_violates_the_schema(
    corpus: tuple[Path, list[dict]],
) -> None:
    """The defect must be real, not cosmetic.

    If a corrupted RFC still validated, the detector would be scored against a
    defect that is not actually present.
    """
    out, labels = corpus
    affected = [lb for lb in labels if "bad_rfc" in lb["defects"]]
    if not affected:
        pytest.skip("no bad_rfc injected at this seed")
    for label in affected:
        errors = validate_file(out / label["file"])
        assert any("Rfc" in e for e in errors), label["file"]


def test_xsd_does_not_catch_business_defects(corpus: tuple[Path, list[dict]]) -> None:
    """Schema validity and business correctness are near-disjoint.

    Documents the justification for the whole detector suite: an invoice can be
    a flawless CFDI 4.0 document and still be a duplicate, inflated, or simply
    not add up.
    """
    out, labels = corpus
    business_only = [
        lb
        for lb in labels
        if lb["defects"] and not (SCHEMA_BREAKING_DEFECTS & set(lb["defects"]))
    ]
    if not business_only:
        pytest.skip("no business-only defects at this seed")
    for label in business_only:
        assert validate_file(out / label["file"]) == [], (
            f"{label['file']} carries {label['defects']} and was expected to be "
            "schema-valid; if the XSD now catches it, the docs above are stale"
        )


def test_rejects_non_cfdi_namespace() -> None:
    errors = validate_bytes(b'<?xml version="1.0"?><invoice><total>1</total></invoice>')
    assert errors and "not a CFDI version" in errors[0]


def test_reports_malformed_xml() -> None:
    errors = validate_bytes(b"<cfdi:Comprobante>unclosed")
    assert errors and "malformed XML" in errors[0]
