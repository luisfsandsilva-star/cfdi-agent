"""Canonical CFDI 4.0 data model.

Two families of models live here, and the split is deliberate:

`ParsedInvoice` & friends
    The canonical internal model. Money is `Decimal`, quantized to the
    precision the SAT uses. Every extractor — the deterministic XML parser
    (tier 0) and the vision path (tier 2) — must produce this exact type.
    That is what makes the two paths comparable in the eval harness and
    interchangeable at runtime.

`InvoiceExtraction` & friends
    The LLM-facing schema, used with `client.messages.parse()`. Structured
    outputs do not support `Decimal`, numeric constraints (`minimum`,
    `multipleOf`), or string-length constraints, so amounts cross the wire as
    plain strings and get converted here. Asking a model for a float and then
    doing money arithmetic on it is a real source of one-cent errors; strings
    avoid the round trip through binary floating point entirely.

Field names follow the SAT's own Spanish vocabulary (`rfc_emisor`,
`clave_prod_serv`, `uso_cfdi`) because those are proper nouns from the CFDI 4.0
standard. Everything else is English.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The SAT's own RFC pattern, lifted verbatim from `t_RFC` in tdCFDI.xsd rather
# than approximated. Personas morales use 3 leading letters, personas físicas 4.
#
# Two details a hand-rolled regex gets wrong, and this one does not:
#   * the month and day are range-checked (01-12, 01-31), not just \d{6}
#   * the final character is [0-9A], NOT [0-9A-Z]
#
# That last character is the RFC check digit, and its algorithm can only ever
# produce a digit or the letter A. Adopting the SAT's pattern therefore
# recovers part of the check-digit validation for free — without implementing
# the algorithm from memory and risking rejection of valid taxpayers.
RFC_RE = re.compile(
    r"^[A-ZÑ&]{3,4}\d{2}(0[1-9]|1[012])(0[1-9]|[12]\d|3[01])[A-Z0-9]{2}[0-9A]$"
)
UUID_RE = re.compile(
    r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)

CENTS = Decimal("0.01")
UNIT_PRECISION = Decimal("0.000001")

TaxKind = Literal["traslado", "retencion"]
Source = Literal["xml", "pdf"]


def to_decimal(value: object, exp: Decimal = CENTS) -> Decimal:
    """Coerce a string/number to Decimal at the given precision.

    Goes through `str()` rather than `Decimal(float)` so that a float that
    slipped in upstream does not carry its binary representation error into
    the ledger.
    """
    if isinstance(value, Decimal):
        d = value
    else:
        try:
            d = Decimal(str(value).strip().replace(",", ""))
        except (InvalidOperation, AttributeError) as exc:
            raise ValueError(f"not a valid amount: {value!r}") from exc
    return d.quantize(exp)


Money = Annotated[Decimal, Field(description="Amount in the invoice currency")]


# --------------------------------------------------------------------------
# Canonical model
# --------------------------------------------------------------------------


class Concepto(BaseModel):
    """A single line item (cfdi:Concepto)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    line_no: int
    clave_prod_serv: str | None = None
    clave_unidad: str | None = None
    descripcion: str
    cantidad: Decimal
    valor_unitario: Decimal
    importe: Money
    descuento: Money = Decimal("0.00")
    objeto_imp: str | None = None

    @field_validator("cantidad", "valor_unitario", mode="before")
    @classmethod
    def _unit_precision(cls, v: object) -> Decimal:
        return to_decimal(v, UNIT_PRECISION)

    @field_validator("importe", "descuento", mode="before")
    @classmethod
    def _cents(cls, v: object) -> Decimal:
        return to_decimal(v)

    @property
    def importe_esperado(self) -> Decimal:
        """cantidad * valor_unitario, at cent precision.

        Detector #4 compares this against the reported `importe`; a mismatch
        means the line does not multiply out.
        """
        return (self.cantidad * self.valor_unitario).quantize(CENTS)


class Impuesto(BaseModel):
    """A tax entry (cfdi:Traslado or cfdi:Retencion)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    tipo: TaxKind
    impuesto: str | None = None  # 001 ISR, 002 IVA, 003 IEPS
    base: Decimal | None = None
    tasa: Decimal | None = None
    importe: Money

    @field_validator("importe", "base", mode="before")
    @classmethod
    def _cents(cls, v: object) -> Decimal | None:
        return None if v is None else to_decimal(v)

    @field_validator("tasa", mode="before")
    @classmethod
    def _rate(cls, v: object) -> Decimal | None:
        return None if v is None else to_decimal(v, UNIT_PRECISION)


class ParsedInvoice(BaseModel):
    """A CFDI 4.0 invoice, however it was extracted.

    Note what is *not* validated here: the arithmetic. A malformed invoice must
    still parse — otherwise the anomaly detectors would never see it and the
    document would vanish instead of landing in the review queue. Structural
    validity is this model's job; correctness is `validate.rules`'.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    uuid: str
    serie: str | None = None
    folio: str | None = None
    fecha_emision: datetime
    fecha_timbrado: datetime | None = None

    rfc_emisor: str
    nombre_emisor: str | None = None
    rfc_receptor: str
    nombre_receptor: str | None = None

    subtotal: Money
    descuento: Money = Decimal("0.00")
    total: Money
    moneda: str = "MXN"
    tipo_cambio: Decimal | None = None

    metodo_pago: str | None = None
    forma_pago: str | None = None
    uso_cfdi: str | None = None

    conceptos: list[Concepto] = Field(default_factory=list)
    impuestos: list[Impuesto] = Field(default_factory=list)

    # State-level taxes, from the `implocal` complement. Separate from
    # `impuestos` because they are not federal and do not appear in
    # cfdi:Impuestos, but they do move the total: two invoices in a 95-document
    # real batch reported a total higher than subtotal + traslados by exactly
    # their TotaldeTraslados, and were flagged critical for arithmetic that was
    # in fact correct.
    traslados_locales: Money = Decimal("0.00")
    retenciones_locales: Money = Decimal("0.00")

    source: Source = "xml"

    @field_validator("rfc_emisor", "rfc_receptor", mode="before")
    @classmethod
    def _upper_rfc(cls, v: object) -> str:
        # Uppercase but do NOT reject a malformed RFC — detector #5 owns that
        # verdict, and it needs to see the bad value to report it.
        return str(v).strip().upper()

    @field_validator("subtotal", "descuento", "total", mode="before")
    @classmethod
    def _cents(cls, v: object) -> Decimal:
        return to_decimal(v)

    @field_validator("tipo_cambio", mode="before")
    @classmethod
    def _rate(cls, v: object) -> Decimal | None:
        return None if v is None else to_decimal(v, UNIT_PRECISION)

    @field_validator("uuid")
    @classmethod
    def _uuid_shape(cls, v: str) -> str:
        if not UUID_RE.match(v):
            raise ValueError(f"UUID has invalid shape: {v!r}")
        return v.upper()

    # -- derived ----------------------------------------------------------

    @property
    def traslados(self) -> Decimal:
        return sum(
            (i.importe for i in self.impuestos if i.tipo == "traslado"), Decimal("0.00")
        ).quantize(CENTS)

    @property
    def retenciones(self) -> Decimal:
        return sum(
            (i.importe for i in self.impuestos if i.tipo == "retencion"), Decimal("0.00")
        ).quantize(CENTS)

    @property
    def subtotal_esperado(self) -> Decimal:
        """Sum of line-item `importe` (before line discounts)."""
        return sum((c.importe for c in self.conceptos), Decimal("0.00")).quantize(CENTS)

    @property
    def descuento_esperado(self) -> Decimal:
        return sum((c.descuento for c in self.conceptos), Decimal("0.00")).quantize(CENTS)

    @property
    def total_esperado(self) -> Decimal:
        """subtotal - descuento + traslados - retenciones, federal and local."""
        return (
            self.subtotal
            - self.descuento
            + self.traslados
            - self.retenciones
            + self.traslados_locales
            - self.retenciones_locales
        ).quantize(CENTS)


# --------------------------------------------------------------------------
# LLM-facing extraction schema (tier 2)
# --------------------------------------------------------------------------


class ConceptoExtraction(BaseModel):
    """Line item as returned by a vision model. Strings only for amounts."""

    model_config = ConfigDict(extra="forbid")

    clave_prod_serv: str | None = Field(
        default=None, description="SAT ClaveProdServ code, e.g. 01010101"
    )
    clave_unidad: str | None = Field(default=None, description="SAT ClaveUnidad, e.g. H87")
    descripcion: str = Field(description="Line item description as printed")
    cantidad: str = Field(description="Quantity, decimal as a plain string, e.g. '2.5'")
    valor_unitario: str = Field(description="Unit price, decimal string, no currency symbol")
    importe: str = Field(description="Line total, decimal string, no currency symbol")
    descuento: str | None = Field(default=None, description="Line discount, decimal string")


class ImpuestoExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tipo: TaxKind = Field(description="'traslado' for charged tax, 'retencion' for withheld")
    impuesto: str | None = Field(default=None, description="001 ISR, 002 IVA, or 003 IEPS")
    base: str | None = Field(default=None, description="Taxable base, decimal string")
    tasa: str | None = Field(default=None, description="Rate, e.g. '0.160000'")
    importe: str = Field(description="Tax amount, decimal string")


# Kept flat and constraint-free on purpose: structured outputs reject Decimal,
# numeric constraints (minimum/maximum/multipleOf), string-length constraints,
# and recursive schemas. Every value is validated downstream by `to_parsed()`
# plus the deterministic rules, so a hallucinated total cannot reach the
# database — it fails arithmetic validation and lands in the review queue.
#
# Pydantic lifts a class docstring into the schema's `description`, which then
# ships to the model on every single call. Keep the docstrings below short and
# written *for the model*; explanation for maintainers belongs in comments.
class InvoiceExtraction(BaseModel):
    """A Mexican CFDI 4.0 invoice transcribed from a document."""

    model_config = ConfigDict(extra="forbid")

    uuid: str = Field(description="UUID from the Timbre Fiscal Digital, with hyphens")
    serie: str | None = Field(default=None, description="Serie, if printed")
    folio: str | None = Field(default=None, description="Folio, if printed")
    fecha_emision: str = Field(description="Issue datetime, ISO 8601, e.g. 2026-03-14T10:22:05")
    fecha_timbrado: str | None = Field(default=None, description="Stamping datetime, ISO 8601")

    rfc_emisor: str = Field(description="Issuer RFC")
    nombre_emisor: str | None = Field(default=None, description="Issuer legal name")
    rfc_receptor: str = Field(description="Recipient RFC")
    nombre_receptor: str | None = Field(default=None, description="Recipient legal name")

    subtotal: str = Field(description="Subtotal, decimal string, no currency symbol")
    descuento: str | None = Field(default=None, description="Invoice-level discount")
    total: str = Field(description="Grand total, decimal string, no currency symbol")
    moneda: str = Field(default="MXN", description="ISO currency code")
    tipo_cambio: str | None = Field(default=None, description="FX rate if moneda is not MXN")

    metodo_pago: str | None = Field(default=None, description="PUE or PPD")
    forma_pago: str | None = Field(default=None, description="SAT FormaPago code, e.g. 03")
    uso_cfdi: str | None = Field(default=None, description="SAT UsoCFDI code, e.g. G03")

    conceptos: list[ConceptoExtraction] = Field(description="Every line item on the invoice")
    impuestos: list[ImpuestoExtraction] = Field(
        default_factory=list, description="Tax entries, if broken out"
    )

    def to_parsed(self) -> ParsedInvoice:
        """Convert to the canonical model. Raises on anything unusable."""
        return ParsedInvoice(
            uuid=self.uuid,
            serie=self.serie,
            folio=self.folio,
            fecha_emision=_parse_dt(self.fecha_emision),
            fecha_timbrado=_parse_dt(self.fecha_timbrado) if self.fecha_timbrado else None,
            rfc_emisor=self.rfc_emisor,
            nombre_emisor=self.nombre_emisor,
            rfc_receptor=self.rfc_receptor,
            nombre_receptor=self.nombre_receptor,
            subtotal=self.subtotal,
            descuento=self.descuento or "0",
            total=self.total,
            moneda=self.moneda,
            tipo_cambio=self.tipo_cambio,
            metodo_pago=self.metodo_pago,
            forma_pago=self.forma_pago,
            uso_cfdi=self.uso_cfdi,
            conceptos=[
                Concepto(
                    line_no=i,
                    clave_prod_serv=c.clave_prod_serv,
                    clave_unidad=c.clave_unidad,
                    descripcion=c.descripcion,
                    cantidad=c.cantidad,
                    valor_unitario=c.valor_unitario,
                    importe=c.importe,
                    descuento=c.descuento or "0",
                )
                for i, c in enumerate(self.conceptos, start=1)
            ],
            impuestos=[
                Impuesto(
                    tipo=t.tipo,
                    impuesto=t.impuesto,
                    base=t.base,
                    tasa=t.tasa,
                    importe=t.importe,
                )
                for t in self.impuestos
            ],
            source="pdf",
        )


def _parse_dt(value: str) -> datetime:
    """Parse a CFDI datetime.

    CFDI 4.0 issue dates are local time with no offset; stamping dates may
    carry a 'Z'. `fromisoformat` handles both on 3.11+.
    """
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
