"""Routing tests: which layer reads which document.

No API key and no local server. The vision path is exercised with a stub
provider, which is possible because `extract_from_document` accepts an injected
provider. What is tested here is the decision, not the model.

The case worth reading twice is `test_pdf_without_credentials_is_unroutable`.
Before the router existed, a PDF failed XML parsing and landed in the review
queue with "no se pudo parsear" — technically true, useless to the person
reading the queue, and it hid the fact that the vision path was never wired in.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from cfdi_agent.extract.providers.base import LLMProvider, LLMResult, ProviderError
from cfdi_agent.extract.router import (
    PDF,
    XML,
    RoutedDocument,
    UnroutableDocument,
    detect_media_type,
    route_document,
)
from cfdi_agent.schemas import InvoiceExtraction

MINIMAL_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<cfdi:Comprobante xmlns:cfdi="http://www.sat.gob.mx/cfd/4"
  Version="4.0" Fecha="2026-02-01T09:00:00" SubTotal="100.00" Total="116.00"
  Moneda="MXN" TipoDeComprobante="I" Exportacion="01" LugarExpedicion="64000">
  <cfdi:Emisor Rfc="AAA010101AAA" Nombre="Proveedor SA" RegimenFiscal="601"/>
  <cfdi:Receptor Rfc="XAXX010101000" Nombre="Cliente SA"
    DomicilioFiscalReceptor="64000" RegimenFiscalReceptor="601" UsoCFDI="G03"/>
  <cfdi:Conceptos>
    <cfdi:Concepto ClaveProdServ="01010101" Cantidad="1" ClaveUnidad="H87"
      Descripcion="Servicio" ValorUnitario="100.00" Importe="100.00" ObjetoImp="02">
      <cfdi:Impuestos><cfdi:Traslados>
        <cfdi:Traslado Base="100.00" Impuesto="002" TipoFactor="Tasa"
          TasaOCuota="0.160000" Importe="16.00"/>
      </cfdi:Traslados></cfdi:Impuestos>
    </cfdi:Concepto>
  </cfdi:Conceptos>
  <cfdi:Complemento>
    <tfd:TimbreFiscalDigital xmlns:tfd="http://www.sat.gob.mx/TimbreFiscalDigital"
      Version="1.1" UUID="a1b2c3d4-e5f6-4718-9a0b-1c2d3e4f5061"
      FechaTimbrado="2026-02-01T09:05:00"/>
  </cfdi:Complemento>
</cfdi:Comprobante>
"""

EXTRACTION = {
    "uuid": "a1b2c3d4-e5f6-4718-9a0b-1c2d3e4f5061",
    "fecha_emision": "2026-02-01T09:00:00",
    "rfc_emisor": "AAA010101AAA",
    "rfc_receptor": "XAXX010101000",
    "subtotal": "100.00",
    "total": "116.00",
    "moneda": "MXN",
    "conceptos": [
        {
            "descripcion": "Servicio",
            "cantidad": "1",
            "valor_unitario": "100.00",
            "importe": "100.00",
        }
    ],
    "impuestos": [
        {"tipo": "traslado", "impuesto": "002", "base": "100.00",
         "tasa": "0.160000", "importe": "16.00"}
    ],
}


class StubProvider(LLMProvider):
    """A provider that transcribes without a model."""

    supported_media = frozenset({PDF, "image/png", "image/jpeg", "image/webp"})

    def __init__(self, name: str = "anthropic", *, fail: bool = False) -> None:
        self.name = name
        self.model = "stub-model"
        self._fail = fail
        self.calls: list[str] = []

    def extract_invoice(self, data: bytes, *, media_type: str) -> LLMResult:
        self.calls.append(media_type)
        if self._fail:
            raise ProviderError("the model declined the request")
        return LLMResult(
            content=InvoiceExtraction.model_validate(EXTRACTION),
            provider=self.name,
            model=self.model,
            latency_ms=1234,
            tokens_in=2400,
            tokens_out=380,
            cost_usd=Decimal("0.021500"),
        )

    def complete(self, system: str, user: str, *, max_tokens: int = 1024):
        raise NotImplementedError

    def embed(self, texts: list[str]):
        raise NotImplementedError


# ------------------------------------------------------------------ detection


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"%PDF-1.7\n%...", PDF),
        (b"\x89PNG\r\n\x1a\n\x00\x00", "image/png"),
        (b"\xff\xd8\xff\xe0\x00\x10JFIF", "image/jpeg"),
        (b"RIFF\x24\x00\x00\x00WEBPVP8 ", "image/webp"),
        (MINIMAL_XML, XML),
    ],
)
def test_magic_bytes_identify_the_format(data: bytes, expected: str) -> None:
    """Attachments get renamed in transit; the first bytes do not lie."""
    assert detect_media_type(data) == expected


def test_xml_is_found_behind_a_byte_order_mark() -> None:
    assert detect_media_type(b"\xef\xbb\xbf" + MINIMAL_XML) == XML


def test_xml_is_found_behind_leading_whitespace() -> None:
    assert detect_media_type(b"\n\n  " + MINIMAL_XML) == XML


def test_extension_is_the_fallback_when_magic_fails() -> None:
    assert detect_media_type(b"\x00\x01\x02\x03", "factura.pdf") == PDF


def test_magic_bytes_beat_a_misleading_extension() -> None:
    """A PDF renamed to .xml is still a PDF."""
    assert detect_media_type(b"%PDF-1.7 content", "factura.xml") == PDF


def test_unknown_bytes_with_no_extension_are_unidentified() -> None:
    assert detect_media_type(b"\x00\x01\x02\x03") is None


# -------------------------------------------------------------------- routing


def test_xml_goes_to_the_free_deterministic_path() -> None:
    routed = route_document(MINIMAL_XML, filename="factura.xml")
    assert isinstance(routed, RoutedDocument)
    assert routed.tier == 0
    assert routed.provider == "none"
    assert routed.cost_usd is None
    assert routed.invoice.rfc_emisor == "AAA010101AAA"


def test_xml_never_reaches_a_provider() -> None:
    """The whole thesis: a CFDI with its XML must not cost anything."""
    stub = StubProvider()
    route_document(MINIMAL_XML, filename="factura.xml", provider=stub)
    assert stub.calls == []


def test_pdf_goes_to_the_vision_path() -> None:
    stub = StubProvider()
    routed = route_document(b"%PDF-1.7 fake", filename="scan.pdf", provider=stub)
    assert stub.calls == [PDF]
    assert routed.tier == 2
    assert routed.provider == "anthropic"
    assert routed.invoice.source == "pdf"
    assert routed.invoice.total == Decimal("116.00")


def test_vision_usage_is_carried_for_the_cost_report() -> None:
    """These values become an extraction_runs row and then a cost per invoice."""
    routed = route_document(b"%PDF fake", filename="s.pdf", provider=StubProvider())
    assert routed.tokens_in == 2400
    assert routed.tokens_out == 380
    assert routed.cost_usd == Decimal("0.021500")
    assert routed.latency_ms == 1234


def test_local_backend_is_reported_as_tier_1() -> None:
    """The tier records where the work ran, not where it was configured to run."""
    routed = route_document(
        b"\x89PNG\r\n\x1a\n", filename="s.png", provider=StubProvider("local")
    )
    assert routed.tier == 1
    assert routed.provider == "local"


def test_images_route_to_vision() -> None:
    stub = StubProvider()
    route_document(b"\xff\xd8\xff fake", filename="foto.jpg", provider=stub)
    assert stub.calls == ["image/jpeg"]


# ------------------------------------------------------------- unroutable


def test_pdf_without_credentials_is_unroutable(monkeypatch) -> None:
    """The message must name the fix, not just report a parse failure.

    Before the router existed, this document failed XML parsing and reached the
    review queue as "no se pudo parsear" — true, unhelpful, and it concealed
    that the vision path was never connected.
    """
    from cfdi_agent.config import get_config

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    get_config.cache_clear()
    try:
        with pytest.raises(UnroutableDocument, match="ANTHROPIC_API_KEY"):
            route_document(b"%PDF-1.7 fake", filename="scan.pdf")
    finally:
        get_config.cache_clear()


def test_malformed_xml_carries_the_parse_error() -> None:
    with pytest.raises(UnroutableDocument, match="no se pudo parsear"):
        route_document(b"<cfdi:Comprobante>unclosed", filename="roto.xml")


def test_unstamped_invoice_is_unroutable() -> None:
    start = MINIMAL_XML.index(b"<cfdi:Complemento>")
    end = MINIMAL_XML.index(b"</cfdi:Complemento>") + len(b"</cfdi:Complemento>")
    with pytest.raises(UnroutableDocument, match="not stamped"):
        route_document(MINIMAL_XML[:start] + MINIMAL_XML[end:], filename="x.xml")


def test_unknown_format_is_unroutable() -> None:
    with pytest.raises(UnroutableDocument, match="formato no reconocido"):
        route_document(b"\x00\x01\x02\x03", filename="misterio.bin")


def test_a_provider_failure_becomes_a_review_reason() -> None:
    """A refusal or a timeout is a document for a human, not a crash."""
    with pytest.raises(UnroutableDocument, match="visión falló"):
        route_document(
            b"%PDF fake", filename="s.pdf", provider=StubProvider(fail=True)
        )


def test_office_documents_are_refused_clearly() -> None:
    """A .docx is a ZIP. It must not be mistaken for anything readable."""
    with pytest.raises(UnroutableDocument):
        route_document(b"PK\x03\x04\x14\x00", filename="factura.docx")
