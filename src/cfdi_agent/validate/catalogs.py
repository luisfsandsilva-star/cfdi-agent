"""Subsets of the SAT catalogs, as plain constants.

The real catalogs are large — `c_ClaveProdServ` alone is roughly 52,000 rows —
and shipping them would mean a download step, a parsing step, and a staleness
problem, for a check that is advisory anyway. What is here covers the codes
that appear on ordinary supplier invoices.

Consequence, stated plainly rather than hidden: an unknown code produces an
`info`-level note, never a rejection. Treating "not in my subset" as "invalid"
would flag legitimate invoices constantly. The README repeats this so nobody
reads a green run as full SAT catalog conformance.
"""

from __future__ import annotations

from types import MappingProxyType

# c_RegimenFiscal
REGIMENES_FISCALES = MappingProxyType(
    {
        "601": "General de Ley Personas Morales",
        "603": "Personas Morales con Fines no Lucrativos",
        "605": "Sueldos y Salarios e Ingresos Asimilados a Salarios",
        "606": "Arrendamiento",
        "607": "Régimen de Enajenación o Adquisición de Bienes",
        "608": "Demás ingresos",
        "610": "Residentes en el Extranjero sin Establecimiento Permanente en México",
        "611": "Ingresos por Dividendos (socios y accionistas)",
        "612": "Personas Físicas con Actividades Empresariales y Profesionales",
        "614": "Ingresos por intereses",
        "615": "Régimen de los ingresos por obtención de premios",
        "616": "Sin obligaciones fiscales",
        "620": "Sociedades Cooperativas de Producción",
        "621": "Incorporación Fiscal",
        "622": "Actividades Agrícolas, Ganaderas, Silvícolas y Pesqueras",
        "623": "Opcional para Grupos de Sociedades",
        "624": "Coordinados",
        "625": "Actividades Empresariales con ingresos a través de Plataformas Tecnológicas",
        "626": "Régimen Simplificado de Confianza",
    }
)

# c_UsoCFDI
USOS_CFDI = MappingProxyType(
    {
        "G01": "Adquisición de mercancías",
        "G02": "Devoluciones, descuentos o bonificaciones",
        "G03": "Gastos en general",
        "I01": "Construcciones",
        "I02": "Mobiliario y equipo de oficina por inversiones",
        "I03": "Equipo de transporte",
        "I04": "Equipo de cómputo y accesorios",
        "I05": "Dados, troqueles, moldes, matrices y herramental",
        "I06": "Comunicaciones telefónicas",
        "I07": "Comunicaciones satelitales",
        "I08": "Otra maquinaria y equipo",
        "D01": "Honorarios médicos, dentales y gastos hospitalarios",
        "D10": "Pagos por servicios educativos (colegiaturas)",
        "P01": "Por definir",
        "S01": "Sin efectos fiscales",
        "CP01": "Pagos",
        "CN01": "Nómina",
    }
)

# c_FormaPago
FORMAS_PAGO = MappingProxyType(
    {
        "01": "Efectivo",
        "02": "Cheque nominativo",
        "03": "Transferencia electrónica de fondos",
        "04": "Tarjeta de crédito",
        "05": "Monedero electrónico",
        "06": "Dinero electrónico",
        "08": "Vales de despensa",
        "12": "Dación en pago",
        "17": "Compensación",
        "23": "Novación",
        "28": "Tarjeta de débito",
        "29": "Tarjeta de servicios",
        "30": "Aplicación de anticipos",
        "31": "Intermediario pagos",
        "99": "Por definir",
    }
)

# c_MetodoPago
METODOS_PAGO = MappingProxyType(
    {
        "PUE": "Pago en una sola exhibición",
        "PPD": "Pago en parcialidades o diferido",
    }
)

# c_Impuesto
IMPUESTOS = MappingProxyType({"001": "ISR", "002": "IVA", "003": "IEPS"})

# c_ClaveUnidad — the handful that cover most goods and services invoices.
CLAVES_UNIDAD = MappingProxyType(
    {
        "H87": "Pieza",
        "E48": "Unidad de servicio",
        "EA": "Elemento",
        "KGM": "Kilogramo",
        "LTR": "Litro",
        "MTR": "Metro",
        "MTK": "Metro cuadrado",
        "XBX": "Caja",
        "GRM": "Gramo",
        "HUR": "Hora",
        "DAY": "Día",
        "MON": "Mes",
        "ACT": "Actividad",
        "E51": "Trabajo",
        "A9": "Tarifa",
    }
)

# c_TipoDeComprobante
TIPOS_COMPROBANTE = MappingProxyType(
    {
        "I": "Ingreso",
        "E": "Egreso",
        "T": "Traslado",
        "N": "Nómina",
        "P": "Pago",
    }
)


def describe_uso_cfdi(code: str | None) -> str | None:
    return USOS_CFDI.get(code) if code else None


def describe_forma_pago(code: str | None) -> str | None:
    return FORMAS_PAGO.get(code) if code else None


def is_known(catalog: MappingProxyType, code: str | None) -> bool:
    """False for an unknown *or missing* code. Callers treat this as advisory."""
    return bool(code) and code in catalog
