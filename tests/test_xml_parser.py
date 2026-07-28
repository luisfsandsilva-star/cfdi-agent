"""Tier-0 parser tests.

These must pass with no API key, no database, and no network. If the
deterministic path ever needs any of those, the design is wrong.

The main test parses the generated corpus and asserts field-by-field equality
against the labels the generator emitted. That is a genuine round-trip check:
the generator writes XML from a set of values, the parser reads those values
back, and any drift in either direction fails here rather than silently
skewing the eval numbers later.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from cfdi_agent.extract.xml_parser import (
    CfdiParseError,
    parse_cfdi_bytes,
    parse_cfdi_file,
)
from synth.generate_cfdi import generate

CORPUS_SIZE = 40


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict]]:
    """A fresh labelled corpus, generated with a fixed seed."""
    out = tmp_path_factory.mktemp("synth")
    labels = generate(
        n=CORPUS_SIZE,
        defect_rate=0.3,
        out_dir=out,
        labels_path=out / "labeled.jsonl",
        seed=7,
        n_suppliers=6,
        receptor_rfc="XAXX010101000",
        receptor_nombre="Mi Empresa SA de CV",
    )
    return out, labels


def test_every_generated_invoice_parses(corpus: tuple[Path, list[dict]]) -> None:
    out, labels = corpus
    # Defective invoices must parse too — a bad total is an anomaly to report,
    # not a document to drop on the floor.
    for label in labels:
        parse_cfdi_file(out / label["file"])


def test_parsed_fields_match_labels(corpus: tuple[Path, list[dict]]) -> None:
    out, labels = corpus
    for label in labels:
        inv = parse_cfdi_file(out / label["file"])
        exp = label["expected"]

        assert inv.uuid == exp["uuid"].upper()
        assert inv.serie == exp["serie"]
        assert inv.folio == exp["folio"]
        assert inv.rfc_emisor == exp["rfc_emisor"].upper()
        assert inv.rfc_receptor == exp["rfc_receptor"].upper()
        assert inv.subtotal == Decimal(exp["subtotal"])
        assert inv.total == Decimal(exp["total"])
        assert inv.descuento == Decimal(exp["descuento"])
        assert inv.moneda == exp["moneda"]
        assert len(inv.conceptos) == exp["n_conceptos"]
        assert inv.fecha_emision.strftime("%Y-%m-%dT%H:%M:%S") == exp["fecha_emision"]


def test_amounts_are_decimal_not_float(corpus: tuple[Path, list[dict]]) -> None:
    """Money must never round-trip through binary floating point."""
    out, labels = corpus
    inv = parse_cfdi_file(out / labels[0]["file"])
    assert isinstance(inv.total, Decimal)
    assert isinstance(inv.conceptos[0].importe, Decimal)
    # Two decimal places exactly — quantized, not merely close.
    assert inv.total == inv.total.quantize(Decimal("0.01"))


def test_clean_invoices_balance(corpus: tuple[Path, list[dict]]) -> None:
    """A defect-free invoice must satisfy total == subtotal - desc + taxes.

    This is what gives detector #4 a meaningful baseline: if clean invoices did
    not balance, every document would be flagged and precision would be zero.
    """
    out, labels = corpus
    clean = [lb for lb in labels if not lb["defects"]]
    assert clean, "fixture produced no clean invoices"
    for label in clean:
        inv = parse_cfdi_file(out / label["file"])
        assert inv.total == inv.total_esperado, label["file"]
        assert inv.subtotal == inv.subtotal_esperado, label["file"]


def test_injected_total_mismatch_is_visible(corpus: tuple[Path, list[dict]]) -> None:
    """The generator's defect must actually be present in the parsed output."""
    out, labels = corpus
    broken = [lb for lb in labels if "total_mismatch" in lb["defects"]]
    if not broken:
        pytest.skip("no total_mismatch injected at this seed")
    for label in broken:
        inv = parse_cfdi_file(out / label["file"])
        assert inv.total != inv.total_esperado, label["file"]


def test_injected_line_math_is_visible(corpus: tuple[Path, list[dict]]) -> None:
    out, labels = corpus
    broken = [lb for lb in labels if "line_math" in lb["defects"]]
    if not broken:
        pytest.skip("no line_math injected at this seed")
    for label in broken:
        inv = parse_cfdi_file(out / label["file"])
        assert any(
            c.importe != c.importe_esperado for c in inv.conceptos
        ), label["file"]


# --------------------------------------------------------------------------
# Edge cases, as inline fixtures
# --------------------------------------------------------------------------

MINIMAL_40 = b"""<?xml version="1.0" encoding="UTF-8"?>
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


def test_falls_back_to_line_level_taxes() -> None:
    """No invoice-level cfdi:Impuestos block — aggregate from the lines.

    Some PACs omit it. Without the fallback, `total_esperado` would compute
    with zero tax and detector #4 would flag every such invoice.
    """
    inv = parse_cfdi_bytes(MINIMAL_40)
    assert inv.traslados == Decimal("16.00")
    assert inv.total == inv.total_esperado


def test_accepts_cfdi_33_namespace() -> None:
    """Pre-2023 documents still arrive. They carry readable data.

    Built by stripping the 4.0-only attributes rather than only swapping the
    namespace, so this is a document a 3.3 PAC would actually have produced.
    """
    xml = (
        MINIMAL_40.replace(b"cfd/4", b"cfd/3")
        .replace(b'Version="4.0"', b'Version="3.3"')
        .replace(b' ObjetoImp="02"', b"")
        .replace(b' Exportacion="01"', b"")
        .replace(b' DomicilioFiscalReceptor="64000"', b"")
        .replace(b' RegimenFiscalReceptor="601"', b"")
    )
    inv = parse_cfdi_bytes(xml)
    assert inv.rfc_emisor == "AAA010101AAA"
    assert inv.total == Decimal("116.00")
    assert inv.uso_cfdi == "G03"  # present in both versions
    assert inv.conceptos[0].objeto_imp is None  # 4.0-only attribute


def test_rejects_unstamped_invoice() -> None:
    """No TimbreFiscalDigital means no UUID: no primary key, no dedupe."""
    start = MINIMAL_40.index(b"<cfdi:Complemento>")
    end = MINIMAL_40.index(b"</cfdi:Complemento>") + len(b"</cfdi:Complemento>")
    xml = MINIMAL_40[:start] + MINIMAL_40[end:]
    with pytest.raises(CfdiParseError, match="not stamped"):
        parse_cfdi_bytes(xml)


def test_rejects_malformed_xml() -> None:
    with pytest.raises(CfdiParseError, match="malformed XML"):
        parse_cfdi_bytes(b"<cfdi:Comprobante>unclosed")


def test_rejects_non_cfdi_root() -> None:
    with pytest.raises(CfdiParseError, match="expected a cfdi:Comprobante"):
        parse_cfdi_bytes(b'<?xml version="1.0"?><invoice><total>10</total></invoice>')


def test_blocks_xxe_entity_expansion(tmp_path: Path) -> None:
    """External entities must not resolve.

    Invoice XML arrives from suppliers by email — it is untrusted input, and a
    parser that resolves entities would happily read /etc/passwd into a field.
    """
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET", encoding="utf-8")
    xml = f"""<?xml version="1.0"?>
<!DOCTYPE foo [ <!ENTITY xxe SYSTEM "file://{secret}"> ]>
<cfdi:Comprobante xmlns:cfdi="http://www.sat.gob.mx/cfd/4" Version="4.0"
  Fecha="2026-02-01T09:00:00" SubTotal="1.00" Total="1.00" Moneda="MXN"
  TipoDeComprobante="I" Exportacion="01" LugarExpedicion="64000">
  <cfdi:Emisor Rfc="AAA010101AAA" Nombre="&xxe;" RegimenFiscal="601"/>
  <cfdi:Receptor Rfc="XAXX010101000" Nombre="C" DomicilioFiscalReceptor="64000"
    RegimenFiscalReceptor="601" UsoCFDI="G03"/>
  <cfdi:Conceptos/>
  <cfdi:Complemento>
    <tfd:TimbreFiscalDigital xmlns:tfd="http://www.sat.gob.mx/TimbreFiscalDigital"
      Version="1.1" UUID="a1b2c3d4-e5f6-4718-9a0b-1c2d3e4f5061"
      FechaTimbrado="2026-02-01T09:05:00"/>
  </cfdi:Complemento>
</cfdi:Comprobante>
""".encode()
    try:
        inv = parse_cfdi_bytes(xml)
    except CfdiParseError:
        return  # rejecting outright is also an acceptable outcome
    assert "TOP-SECRET" not in (inv.nombre_emisor or "")


def test_uuid_is_normalized_uppercase() -> None:
    inv = parse_cfdi_bytes(MINIMAL_40)
    assert inv.uuid == "A1B2C3D4-E5F6-4718-9A0B-1C2D3E4F5061"


def test_line_numbers_follow_document_order() -> None:
    inv = parse_cfdi_bytes(MINIMAL_40)
    assert [c.line_no for c in inv.conceptos] == [1]
