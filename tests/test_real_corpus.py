"""The confidentiality boundary of the real-invoice smoke test.

`evals.real_corpus` runs somebody else's invoices through the pipeline, so the
only thing standing between a validation error and a printed RFC is `_scrub`.
That makes it worth testing directly, and it is the one part of that module a
test can cover without a corpus.

The first implementation scrubbed by value pattern and leaked: a *malformed*
RFC does not match the RFC regex, and a malformed RFC is exactly what a
validation error contains. These tests encode that case.
"""

from __future__ import annotations

from evals.real_corpus import _scrub, survey_xml


class TestScrub:
    def test_a_valid_rfc_is_redacted(self) -> None:
        assert "AAA010101AAA" not in _scrub(
            "Element 'Emisor', attribute 'Rfc': The value 'AAA010101AAA' is not valid."
        )

    def test_a_malformed_rfc_is_redacted(self) -> None:
        """The case that broke pattern-based scrubbing.

        `BPZ161106UL-` fails the RFC pattern, which is why it reached an error
        message, which is why a pattern-based scrubber let it through.
        """
        out = _scrub(
            "line 12: Element '{http://www.sat.gob.mx/cfd/4}Emisor', attribute "
            "'Rfc': [facet 'pattern'] The value 'BPZ161106UL-' is not accepted."
        )
        assert "BPZ161106UL-" not in out
        assert "<redacted>" in out

    def test_schema_vocabulary_survives(self) -> None:
        """A scrubbed message still has to be actionable.

        Element and attribute names come out of the SAT's public XSD. Redacting
        them would leave a message that proves nothing was leaked and tells the
        reader nothing about what to fix.
        """
        out = _scrub(
            "Element '{http://www.sat.gob.mx/cfd/4}Concepto', attribute "
            "'ValorUnitario': The value '18432.50' is not a valid decimal."
        )
        assert "Concepto" in out
        assert "attribute 'ValorUnitario'" in out
        assert "18432.50" not in out

    def test_unknown_quoted_content_is_assumed_to_be_data(self) -> None:
        """An addenda or a PAC message can quote anything. Default to redacting."""
        out = _scrub("Addenda rejected: 'Cliente Ejemplo SA de CV / Pedido 8891'")
        assert "Cliente Ejemplo" not in out

    def test_uuids_amounts_and_dates_are_redacted_anywhere(self) -> None:
        out = _scrub(
            "duplicate key 3F2504E0-4F89-41D3-9A0C-0305E82C3301 on 2026-03-14 "
            "for 14250.00"
        )
        assert "3F2504E0" not in out
        assert "2026-03-14" not in out
        assert "14250.00" not in out

    def test_variants_of_one_failure_collapse_to_one_line(self) -> None:
        """Scrubbing is also what makes the output readable.

        Seven documents failing the same facet on seven different RFCs should
        report as one row with a count, not seven rows of noise.
        """
        template = (
            "Element 'Emisor', attribute 'Rfc': [facet 'pattern'] "
            "The value '{}' is not accepted."
        )
        seen = {_scrub(template.format(rfc)) for rfc in ("BPZ161106UL-", "URB130226B5-")}
        assert len(seen) == 1

    def test_it_is_bounded(self) -> None:
        assert len(_scrub("x" * 5000)) <= 200


class TestSurvey:
    """The corpus shape survey reports tag names, never values."""

    XML = b"""<?xml version="1.0" encoding="UTF-8"?>
    <cfdi:Comprobante xmlns:cfdi="http://www.sat.gob.mx/cfd/4"
                      xmlns:tfd="http://www.sat.gob.mx/TimbreFiscalDigital"
                      Version="4.0" Moneda="USD" TipoDeComprobante="I">
      <cfdi:Complemento>
        <tfd:TimbreFiscalDigital Version="1.1" UUID="3F2504E0-4F89-41D3-9A0C-0305E82C3301"/>
      </cfdi:Complemento>
      <cfdi:Addenda><Pedido num="8891" cliente="Ejemplo SA"/></cfdi:Addenda>
    </cfdi:Comprobante>"""

    def test_it_reports_structure(self) -> None:
        out = survey_xml(self.XML)
        assert out["version"] == ["4.0"]
        assert out["moneda"] == ["USD"]
        assert out["tipo_comprobante"] == ["I"]
        assert out["complementos"] == ["TimbreFiscalDigital"]
        assert out["addenda"] == ["present"]

    def test_it_reports_no_values(self) -> None:
        """An addenda is arbitrary vendor XML. Only its presence is reported."""
        flat = " ".join(v for values in survey_xml(self.XML).values() for v in values)
        assert "3F2504E0" not in flat
        assert "Ejemplo" not in flat
        assert "8891" not in flat

    def test_a_cfdi_33_document_is_surveyed_too(self) -> None:
        """Version spread is the point: the generator only emits 4.0."""
        older = self.XML.replace(b"cfd/4", b"cfd/3").replace(b'Version="4.0"', b'Version="3.3"')
        out = survey_xml(older)
        assert out["version"] == ["3.3"]
        assert out["complementos"] == ["TimbreFiscalDigital"]


class TestScrubOnOwnMessages:
    """Rejection reasons this project writes name the RFC they rejected.

    Those are unquoted, so the positional scrub cannot see them and the loose
    fallback pattern is what has to catch them.
    """

    def test_a_rejection_reason_does_not_leak_either_rfc(self) -> None:
        out = _scrub("invoice is addressed to API120327LD6, not to XAXX010101000")
        assert "API120327LD6" not in out
        assert "XAXX010101000" not in out
        assert "addressed to" in out

    def test_a_sat_catalog_code_is_not_mistaken_for_an_rfc(self) -> None:
        """Over-redaction is the safe direction, but not to the point of noise."""
        out = _scrub("unknown ClaveProdServ 80161501 and ClaveUnidad H87")
        assert "80161501" in out
        assert "H87" in out


class TestPairing:
    """A PDF and its XML twin, as a real delivery names them."""

    def test_pairs_match_across_case(self, tmp_path) -> None:
        """The batch this was written against names PDFs with a lowercase UUID
        and XMLs with the same UUID uppercased. Exact-stem matching found 4
        pairs in it instead of 93."""
        from evals.vision_accuracy import find_pairs

        (tmp_path / "a1b2c3d4-0000-4000-8000-000000000001.pdf").write_bytes(b"%PDF-")
        (tmp_path / "A1B2C3D4-0000-4000-8000-000000000001.xml").write_bytes(b"<x/>")
        assert len(find_pairs(tmp_path)) == 1

    def test_a_pdf_without_its_xml_is_skipped(self, tmp_path) -> None:
        """There would be nothing to score it against."""
        from evals.vision_accuracy import find_pairs

        (tmp_path / "orphan.pdf").write_bytes(b"%PDF-")
        assert find_pairs(tmp_path) == []
