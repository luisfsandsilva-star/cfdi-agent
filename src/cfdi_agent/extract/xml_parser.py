"""Tier 0: deterministic CFDI XML → `ParsedInvoice`.

This is the path that handles the overwhelming majority of real invoices, and
it never calls a model. A CFDI is already structured data with a published
schema; running it through an LLM would cost money per document and introduce
a failure mode (transcription error) that simply does not exist here.

Both CFDI 4.0 and 3.3 are accepted. Real accounting inboxes still carry 3.3
documents from before the 2023 cutover, and the attributes this parser reads
are common to both versions — supporting the older namespace costs one extra
entry in a tuple, and refusing it would push perfectly machine-readable
invoices into the expensive vision path.

What this module does NOT do: verify the digital seal, check arithmetic, or
validate against the SAT. Sello/Certificado are RSA signatures over the
original string and verifying them properly requires the PAC's certificate
chain; `README` states plainly that this is out of scope rather than implying
the pipeline authenticates the document. Arithmetic is `validate.rules`' job —
a malformed invoice must still parse, or the detectors never get to see it.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from lxml import etree

from cfdi_agent.schemas import Concepto, Impuesto, ParsedInvoice

# Namespace URIs, newest first. tfd 1.1 is shared by CFDI 3.3 and 4.0.
CFDI_NAMESPACES = (
    "http://www.sat.gob.mx/cfd/4",
    "http://www.sat.gob.mx/cfd/3",
)
TFD_NAMESPACE = "http://www.sat.gob.mx/TimbreFiscalDigital"
# State-level taxes (lodging tax and similar). Not federal, so they are absent
# from cfdi:Impuestos, but they are part of the total the issuer charges.
IMPLOCAL_NAMESPACE = "http://www.sat.gob.mx/implocal"


class CfdiParseError(ValueError):
    """The document is not a CFDI we can read at all.

    Distinct from "the CFDI is wrong": that produces a `ParsedInvoice` plus
    anomalies. This exception means there is nothing to hand downstream, and
    the document belongs in the review queue.
    """


def _qname(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


def _detect_namespace(root: etree._Element) -> str:
    for ns in CFDI_NAMESPACES:
        if root.tag == _qname(ns, "Comprobante"):
            return ns
    raise CfdiParseError(
        f"root element is {root.tag!r}, expected a cfdi:Comprobante in one of "
        f"{CFDI_NAMESPACES}"
    )


def _parse_dt(value: str | None) -> datetime | None:
    """CFDI datetimes are local time, no offset. FechaTimbrado may carry 'Z'."""
    if not value:
        return None
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def parse_cfdi_bytes(data: bytes) -> ParsedInvoice:
    """Parse CFDI XML bytes into the canonical model."""
    parser = etree.XMLParser(
        resolve_entities=False,  # no XXE
        no_network=True,
        huge_tree=False,
        recover=False,
    )
    try:
        root = etree.fromstring(data, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise CfdiParseError(f"malformed XML: {exc}") from exc

    ns = _detect_namespace(root)
    cfdi = lambda tag: _qname(ns, tag)  # noqa: E731 - local shorthand, reads better here

    emisor = root.find(cfdi("Emisor"))
    receptor = root.find(cfdi("Receptor"))
    if emisor is None or receptor is None:
        raise CfdiParseError("missing cfdi:Emisor or cfdi:Receptor")

    tfd = root.find(f'{cfdi("Complemento")}/{_qname(TFD_NAMESPACE, "TimbreFiscalDigital")}')
    if tfd is None:
        # An un-stamped CFDI has no UUID, which means no primary key and no
        # duplicate detection. It is not a valid tax document either.
        raise CfdiParseError("no TimbreFiscalDigital complement: invoice is not stamped")
    invoice_uuid = tfd.get("UUID")
    if not invoice_uuid:
        raise CfdiParseError("TimbreFiscalDigital carries no UUID attribute")

    implocal = root.find(
        f'{cfdi("Complemento")}/{_qname(IMPLOCAL_NAMESPACE, "ImpuestosLocales")}'
    )
    # Attribute strings go straight in: the model's `Money` type coerces them,
    # the same way SubTotal and Total already do.
    traslados_locales = implocal.get("TotaldeTraslados") if implocal is not None else "0"
    retenciones_locales = implocal.get("TotaldeRetenciones") if implocal is not None else "0"

    conceptos = _parse_conceptos(root, cfdi)
    impuestos = _parse_impuestos(root, cfdi, conceptos_node=root.find(cfdi("Conceptos")))

    return ParsedInvoice(
        uuid=invoice_uuid,
        serie=root.get("Serie"),
        folio=root.get("Folio"),
        fecha_emision=_parse_dt(root.get("Fecha")),
        fecha_timbrado=_parse_dt(tfd.get("FechaTimbrado")),
        rfc_emisor=emisor.get("Rfc", ""),
        nombre_emisor=emisor.get("Nombre"),
        rfc_receptor=receptor.get("Rfc", ""),
        nombre_receptor=receptor.get("Nombre"),
        subtotal=root.get("SubTotal", "0"),
        descuento=root.get("Descuento") or "0",
        total=root.get("Total", "0"),
        moneda=root.get("Moneda", "MXN"),
        tipo_cambio=root.get("TipoCambio"),
        metodo_pago=root.get("MetodoPago"),
        forma_pago=root.get("FormaPago"),
        uso_cfdi=receptor.get("UsoCFDI"),
        conceptos=conceptos,
        impuestos=impuestos,
        traslados_locales=traslados_locales or "0",
        retenciones_locales=retenciones_locales or "0",
        source="xml",
    )


def _parse_conceptos(root: etree._Element, cfdi) -> list[Concepto]:
    node = root.find(cfdi("Conceptos"))
    if node is None:
        return []
    out: list[Concepto] = []
    # CFDI carries no line number; document order is the line number.
    for i, c in enumerate(node.findall(cfdi("Concepto")), start=1):
        out.append(
            Concepto(
                line_no=i,
                clave_prod_serv=c.get("ClaveProdServ"),
                clave_unidad=c.get("ClaveUnidad"),
                descripcion=c.get("Descripcion", ""),
                cantidad=c.get("Cantidad", "0"),
                valor_unitario=c.get("ValorUnitario", "0"),
                importe=c.get("Importe", "0"),
                descuento=c.get("Descuento") or "0",
                objeto_imp=c.get("ObjetoImp"),  # 4.0 only; None on 3.3
            )
        )
    return out


def _parse_impuestos(
    root: etree._Element, cfdi, conceptos_node: etree._Element | None
) -> list[Impuesto]:
    """Read invoice-level taxes, falling back to aggregating line-level ones.

    The invoice-level `cfdi:Impuestos` block is optional. When a PAC omits it,
    the tax data lives only inside each `cfdi:Concepto`. Falling back keeps the
    total check (detector #4) meaningful instead of computing an expected total
    with zero tax and flagging every such invoice as broken.
    """
    node = root.find(cfdi("Impuestos"))
    out: list[Impuesto] = []

    if node is not None:
        for t in node.findall(f'{cfdi("Traslados")}/{cfdi("Traslado")}'):
            out.append(
                Impuesto(
                    tipo="traslado",
                    impuesto=t.get("Impuesto"),
                    base=t.get("Base"),
                    tasa=t.get("TasaOCuota"),
                    importe=t.get("Importe", "0"),
                )
            )
        for r in node.findall(f'{cfdi("Retenciones")}/{cfdi("Retencion")}'):
            out.append(
                Impuesto(
                    tipo="retencion",
                    impuesto=r.get("Impuesto"),
                    base=r.get("Base"),
                    tasa=r.get("TasaOCuota"),
                    importe=r.get("Importe", "0"),
                )
            )
        if out:
            return out

    if conceptos_node is None:
        return out

    for c in conceptos_node.findall(cfdi("Concepto")):
        imp = c.find(cfdi("Impuestos"))
        if imp is None:
            continue
        for t in imp.findall(f'{cfdi("Traslados")}/{cfdi("Traslado")}'):
            out.append(
                Impuesto(
                    tipo="traslado",
                    impuesto=t.get("Impuesto"),
                    base=t.get("Base"),
                    tasa=t.get("TasaOCuota"),
                    importe=t.get("Importe", "0"),
                )
            )
        for r in imp.findall(f'{cfdi("Retenciones")}/{cfdi("Retencion")}'):
            out.append(
                Impuesto(
                    tipo="retencion",
                    impuesto=r.get("Impuesto"),
                    base=r.get("Base"),
                    tasa=r.get("TasaOCuota"),
                    importe=r.get("Importe", "0"),
                )
            )
    return out


def parse_cfdi_file(path: str | Path) -> ParsedInvoice:
    p = Path(path)
    try:
        return parse_cfdi_bytes(p.read_bytes())
    except CfdiParseError as exc:
        # Re-raise with the filename attached: when a batch of 200 documents
        # lands in the review queue, "which file" is the first question.
        raise CfdiParseError(f"{p.name}: {exc}") from exc
