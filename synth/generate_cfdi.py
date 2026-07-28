"""Generate synthetic CFDI 4.0 invoices together with their ground truth.

The point of generating rather than collecting: because we construct each
invoice, we already know every field and every defect we injected. The labels
come out for free, which is what makes a real eval harness affordable in a
weekend. Collecting and hand-labelling 60 invoices costs a day and gets you a
noisier answer.

Two properties this generator takes seriously:

*Chronological, stateful emission.* Invoices are produced in date order and
each supplier carries its own folio counter and price history. Without that,
"price outlier" and "folio gap" have no baseline to deviate from and the
corresponding detectors would be untestable.

*Only injectable defects are labelled.* A price spike needs prior invoices for
the same (supplier, product) before it means anything. When a precondition is
not met the defect is skipped, and the label records what was **actually**
injected — never what was intended. A label that claims a defect the file does
not contain would silently cap the measured recall below 1.0 and send you
debugging a detector that is working fine.

Usage:
    python -m synth.generate_cfdi --n 60 --defect-rate 0.25 --out data/synth
"""

from __future__ import annotations

import argparse
import json
import random
import uuid as uuidlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

CENTS = Decimal("0.01")
IVA = Decimal("0.160000")
# Must match validate.rules.PRICE_MIN_SAMPLES: below this the outlier
# detector has no baseline, so a spike here would be unfindable by design.
PRICE_HISTORY_FLOOR = 5
TEMPLATE_DIR = Path(__file__).parent / "templates"

DEFECT_KINDS = (
    "total_mismatch",
    "bad_rfc",
    "dup_uuid",
    "price_spike",
    "folio_gap",
    "line_math",
    "semantic_dup",
)

# A small, realistic product catalog. ClaveProdServ and ClaveUnidad values are
# real SAT catalog codes; the price is a plausible MXN unit price.
CATALOG = [
    ("01010101", "H87", "Servicio de consultoría en TI", 850, 2500),
    ("44121618", "H87", "Papel bond carta (paquete 500 hojas)", 95, 180),
    ("81112501", "E48", "Licencia de software mensual", 300, 1200),
    ("43211508", "H87", "Laptop empresarial", 14000, 32000),
    ("78181500", "E48", "Mantenimiento preventivo de flotilla", 1200, 4500),
    ("14111704", "H87", "Cartucho de tóner", 480, 1400),
    ("80161501", "E48", "Servicio de limpieza de oficina", 2000, 6000),
    ("72101511", "E48", "Reparación eléctrica", 900, 3800),
    ("15101506", "LTR", "Combustible diésel", 22, 28),
    ("47131700", "H87", "Insumos de limpieza", 60, 320),
]

REWORDS = {
    "Servicio de consultoría en TI": "Consultoría en tecnologías de información",
    "Papel bond carta (paquete 500 hojas)": "Paquete papel bond tamaño carta 500 h",
    "Licencia de software mensual": "Suscripción mensual de software",
    "Laptop empresarial": "Computadora portátil para empresa",
    "Mantenimiento preventivo de flotilla": "Servicio preventivo a flotilla vehicular",
    "Cartucho de tóner": "Tóner de repuesto",
    "Servicio de limpieza de oficina": "Limpieza de oficinas",
    "Reparación eléctrica": "Servicio de reparación eléctrica",
    "Combustible diésel": "Diésel",
    "Insumos de limpieza": "Artículos de limpieza",
}

REGIMENES = ["601", "603", "605", "612", "626"]
USOS_CFDI = ["G01", "G03", "I04", "P01"]
FORMAS_PAGO = ["01", "03", "04", "28", "99"]
METODOS_PAGO = ["PUE", "PPD"]


def d(value: object, exp: Decimal = CENTS) -> Decimal:
    return Decimal(str(value)).quantize(exp)


def _rand_rfc(rng: random.Random, moral: bool = True) -> str:
    """Generate an RFC that satisfies the SAT's `t_RFC` pattern.

    The final character is drawn from `[0-9A]`, not the full alphabet: it is
    the RFC check digit, whose algorithm can only produce a digit or 'A'. The
    official XSD encodes that, and generating a plain alphanumeric there made
    roughly two thirds of the corpus schema-invalid.
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    alnum = letters + "0123456789"
    check_digit_alphabet = "0123456789A"
    n = 3 if moral else 4
    prefix = "".join(rng.choice(letters) for _ in range(n))
    # The RFC date is the incorporation date, YYMMDD. Two-digit years wrap, so
    # pick the century first rather than trying to span 1985..2024 in one call.
    year = rng.randint(85, 99) if rng.random() < 0.4 else rng.randint(0, 24)
    date = f"{year:02d}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}"
    homoclave = "".join(rng.choice(alnum) for _ in range(2)) + rng.choice(
        check_digit_alphabet
    )
    return f"{prefix}{date}{homoclave}"


def _fake_sello(rng: random.Random, length: int = 172) -> str:
    """Placeholder for the PAC's RSA signature. Not verified anywhere."""
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    return "".join(rng.choice(chars) for _ in range(length)) + "=="


@dataclass
class Product:
    clave_prod_serv: str
    clave_unidad: str
    descripcion: str
    unit_price: Decimal


@dataclass
class Supplier:
    rfc: str
    nombre: str
    regimen: str
    serie: str
    products: list[Product]
    folio: int = 1
    invoices_emitted: int = 0
    # (product, quantity) of the most recent invoice, so a semantic duplicate
    # can re-bill exactly the same work under different wording.
    # (clave_prod_serv, cantidad, valor_unitario) of the most recent invoice.
    # A re-bill has to reproduce the price too, not just the quantity: letting
    # the price drift moves the total outside the detector's 1% window, which
    # made 10 of 15 injected duplicates structurally undetectable.
    last_lines: list = field(default_factory=list)
    # (rfc_as_written, clave_prod_serv) -> prior invoices carrying that product.
    #
    # Keyed by the RFC actually written into the XML, not by `self.rfc`, so it
    # matches the key the pipeline builds its price history under. They diverge
    # whenever a `bad_rfc` defect corrupts the issuer RFC: that invoice's prices
    # land in a different bucket downstream. Counting them here would make the
    # generator believe a product has 5 samples when the detector only sees 4,
    # and it would then label an undetectable price spike.
    product_history: dict[tuple[str, str], int] = field(default_factory=dict)

    def next_folio(self, skip: int = 0) -> int:
        self.folio += skip
        current = self.folio
        self.folio += 1
        return current


def build_world(rng: random.Random, n_suppliers: int) -> list[Supplier]:
    suppliers: list[Supplier] = []
    suffixes = ["SA de CV", "S de RL de CV", "SAPI de CV", "SC"]
    stems = [
        "Suministros del Norte", "Tecnología Regia", "Papelería Cumbres",
        "Servicios Integrales MTY", "Logística Aztlán", "Soluciones Obispado",
        "Comercializadora San Pedro", "Insumos Industriales Apodaca",
        "Consultores Valle Oriente", "Distribuidora Guadalupe",
    ]
    rng.shuffle(stems)
    for i in range(n_suppliers):
        stem = stems[i % len(stems)]
        products = [
            Product(cps, cu, desc, d(rng.uniform(lo, hi)))
            for cps, cu, desc, lo, hi in rng.sample(CATALOG, k=rng.randint(2, 4))
        ]
        suppliers.append(
            Supplier(
                rfc=_rand_rfc(rng),
                nombre=f"{stem} {rng.choice(suffixes)}",
                regimen=rng.choice(REGIMENES),
                serie=chr(ord("A") + i % 26),
                products=products,
            )
        )
    return suppliers


@dataclass
class GeneratedInvoice:
    filename: str
    xml: str
    label: dict


def _render_invoice(
    env: Environment,
    rng: random.Random,
    supplier: Supplier,
    receptor_rfc: str,
    receptor_nombre: str,
    fecha: datetime,
    defects: list[str],
    reused_uuid: str | None,
) -> GeneratedInvoice:
    """Build one invoice, applying whichever defects were selected for it."""
    n_lines = rng.randint(1, 4)
    chosen = rng.sample(supplier.products, k=min(n_lines, len(supplier.products)))

    # A semantic duplicate re-bills the previous invoice from this supplier:
    # same products, same quantities, reworded descriptions, a near-identical
    # total a few days later. Detector #1 cannot see it — the UUID and folio
    # are legitimately different.
    twin = getattr(supplier, "last_lines", None)
    if "semantic_dup" in defects:
        if twin:
            keys = {k for k, _, _ in twin}
            chosen = [p for p in supplier.products if p.clave_prod_serv in keys]
        else:
            defects = [d for d in defects if d != "semantic_dup"]

    # `defects` is what the caller asked for; `applied` is what this invoice
    # actually ends up carrying. They differ when a precondition fails, and the
    # label must record `applied` — see the module docstring.
    applied = list(defects)

    conceptos: list[dict] = []

    # Decide the issuer RFC first: it is part of the price-history key, so the
    # spike-eligibility check below has to know whether this invoice's prices
    # will be filed under the real RFC or a corrupted one.
    rfc_emisor = supplier.rfc
    if "bad_rfc" in applied:
        rfc_emisor = supplier.rfc[:-1] + "-"  # invalid character in the homoclave

    # A price spike is only detectable on a product that already has enough
    # price history to establish a baseline. Picking a random line and hoping
    # it qualifies labels undetectable defects and silently caps recall: on one
    # seed this alone dragged price_spike recall to 0.75.
    spike_line = -1
    if "price_spike" in applied:
        eligible = [
            i
            for i, p in enumerate(chosen)
            if supplier.product_history.get((rfc_emisor, p.clave_prod_serv), 0)
            >= PRICE_HISTORY_FLOOR
        ]
        if eligible:
            spike_line = rng.choice(eligible)
        else:
            applied.remove("price_spike")

    math_line = rng.randrange(len(chosen)) if "line_math" in applied else -1

    # Keyed by ClaveProdServ: Product is a mutable dataclass and not hashable.
    twin_lines = (
        {k: (q, u) for k, q, u in twin} if ("semantic_dup" in applied and twin) else {}
    )

    for idx, product in enumerate(chosen):
        reuse = twin_lines.get(product.clave_prod_serv)
        if reuse:
            cantidad, forced_unit = reuse
        else:
            cantidad, forced_unit = d(rng.randint(1, 12), Decimal("0.000001")), None
        if forced_unit is not None:
            # A re-bill charges the same price. Any drift here would move the
            # total out of the detector's window and make the defect unfindable.
            unit = forced_unit
        else:
            unit = product.unit_price
            # Normal drift so a real baseline has variance; the outlier detector
            # must survive ordinary price movement without firing.
            unit = d(unit * Decimal(str(rng.uniform(0.97, 1.03))), Decimal("0.000001"))
        if idx == spike_line:
            unit = d(unit * Decimal(str(rng.uniform(3.0, 5.0))), Decimal("0.000001"))

        importe = d(cantidad * unit)
        if idx == math_line:
            # Line does not multiply out — importe is wrong, not the inputs.
            importe = d(importe * Decimal(str(rng.uniform(1.08, 1.35))))

        base = importe
        traslado = {
            "base": f"{base}",
            "impuesto": "002",
            "tasa": f"{IVA}",
            "importe": f"{d(base * IVA)}",
        }
        descripcion = product.descripcion
        if "semantic_dup" in applied and twin_lines:
            descripcion = REWORDS.get(descripcion, descripcion)

        conceptos.append(
            {
                "clave_prod_serv": product.clave_prod_serv,
                "clave_unidad": product.clave_unidad,
                "descripcion": descripcion,
                "cantidad": f"{cantidad}",
                "valor_unitario": f"{unit}",
                "importe": f"{importe}",
                "descuento": "0.00",
                "objeto_imp": "02",
                "traslado": traslado,
            }
        )
        history_key = (rfc_emisor, product.clave_prod_serv)
        supplier.product_history[history_key] = (
            supplier.product_history.get(history_key, 0) + 1
        )

    subtotal = d(sum(Decimal(c["importe"]) for c in conceptos))
    total_traslados = d(sum(Decimal(c["traslado"]["importe"]) for c in conceptos))
    total = d(subtotal + total_traslados)

    if "total_mismatch" in applied:
        # Perturb the reported total while leaving the parts consistent, so the
        # only way to catch it is to actually re-add the invoice.
        delta = d(max(Decimal("1.00"), total * Decimal(str(rng.uniform(0.005, 0.05)))))
        total = d(total + (delta if rng.random() < 0.5 else -delta))

    folio_skip = rng.randint(2, 7) if "folio_gap" in applied else 0
    folio = supplier.next_folio(skip=folio_skip)

    inv_uuid = reused_uuid or str(uuidlib.UUID(int=rng.getrandbits(128), version=4)).upper()
    fecha_timbrado = fecha + timedelta(minutes=rng.randint(2, 240))

    ctx = {
        "serie": supplier.serie,
        "folio": str(folio),
        "fecha_emision": fecha.strftime("%Y-%m-%dT%H:%M:%S"),
        "fecha_timbrado": fecha_timbrado.strftime("%Y-%m-%dT%H:%M:%S"),
        "uuid": inv_uuid,
        "sello": _fake_sello(rng),
        "sello_sat": _fake_sello(rng),
        "certificado": _fake_sello(rng, 300),
        # Certificate serials are exactly 20 digits ([0-9]{20} in the XSD).
        "no_certificado": f"300010000005{rng.randrange(10**8):08d}",
        "no_certificado_sat": f"300010000004{rng.randrange(10**8):08d}",
        "rfc_pac": _rand_rfc(rng),
        "subtotal": f"{subtotal}",
        "descuento": "0.00",
        "total": f"{total}",
        "moneda": "MXN",
        "tipo_cambio": None,
        "metodo_pago": rng.choice(METODOS_PAGO),
        "forma_pago": rng.choice(FORMAS_PAGO),
        "lugar_expedicion": rng.choice(["64000", "66220", "64610", "67100"]),
        "rfc_emisor": rfc_emisor,
        "nombre_emisor": supplier.nombre,
        "regimen_emisor": supplier.regimen,
        "rfc_receptor": receptor_rfc,
        "nombre_receptor": receptor_nombre,
        "domicilio_receptor": "64000",
        "regimen_receptor": "601",
        "uso_cfdi": rng.choice(USOS_CFDI),
        "conceptos": conceptos,
        "traslados": [
            {
                "base": f"{subtotal}",
                "impuesto": "002",
                "tasa": f"{IVA}",
                "importe": f"{total_traslados}",
            }
        ],
        "retenciones": [],
        "total_traslados": f"{total_traslados}",
        "total_retenciones": "0.00",
    }

    xml = env.get_template("cfdi40.xml.j2").render(**ctx)
    supplier.invoices_emitted += 1
    supplier.last_lines = [
        (p.clave_prod_serv, Decimal(c["cantidad"]), Decimal(c["valor_unitario"]))
        for p, c in zip(chosen, conceptos, strict=True)
    ]

    label = {
        "file": f"{supplier.serie}-{folio:06d}.xml",
        "uuid": inv_uuid,
        "defects": sorted(applied),
        "expected": {
            "uuid": inv_uuid,
            "serie": supplier.serie,
            "folio": str(folio),
            "fecha_emision": ctx["fecha_emision"],
            "rfc_emisor": rfc_emisor,
            "rfc_receptor": receptor_rfc,
            "subtotal": f"{subtotal}",
            "descuento": "0.00",
            "total": f"{total}",
            "moneda": "MXN",
            "n_conceptos": len(conceptos),
        },
    }
    return GeneratedInvoice(filename=label["file"], xml=xml, label=label)


def generate(
    n: int,
    defect_rate: float,
    out_dir: Path,
    labels_path: Path,
    seed: int,
    n_suppliers: int,
    receptor_rfc: str,
    receptor_nombre: str,
    clean: bool = True,
) -> list[dict]:
    rng = random.Random(seed)
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filenames are derived from serie+folio, so a different seed (or any
    # change to how the RNG is consumed) produces a different set of names and
    # the previous run's files survive. Left in place they silently inflate the
    # corpus with invoices no label describes — which corrupts every eval
    # computed over the directory. Scoped to *.xml in the target directory.
    if clean:
        stale = list(out_dir.glob("*.xml"))
        for path in stale:
            path.unlink()
        if stale:
            print(f"removed {len(stale)} stale invoice(s) from {out_dir}")

    suppliers = build_world(rng, n_suppliers)
    start = datetime(2026, 1, 5, 9, 0, 0)
    emitted_uuids: list[str] = []
    labels: list[dict] = []
    counts: dict[str, int] = dict.fromkeys(DEFECT_KINDS, 0)

    # A supplier held back until the last 15% of the run, so "first invoice
    # from an unknown supplier" is a genuine late arrival rather than just the
    # first row in the table.
    late_supplier = suppliers[-1]
    regular = suppliers[:-1]

    for i in range(n):
        progress = i / max(n - 1, 1)
        supplier = late_supplier if progress > 0.85 and rng.random() < 0.3 else rng.choice(regular)
        fecha = start + timedelta(
            days=int(progress * 120), hours=rng.randint(0, 9), minutes=rng.randint(0, 59)
        )

        defects: list[str] = []
        reused_uuid: str | None = None
        if rng.random() < defect_rate:
            candidate = rng.choice(DEFECT_KINDS)
            # Preconditions: only label a defect the file can actually exhibit.
            if candidate == "dup_uuid":
                if emitted_uuids:
                    reused_uuid = rng.choice(emitted_uuids)
                    defects.append(candidate)
            elif candidate == "semantic_dup":
                # Needs a previous invoice from this supplier to re-bill, and
                # the pair has to fall inside the detector's 7-day window.
                if supplier.last_lines:
                    defects.append(candidate)
            elif candidate == "price_spike":
                if any(v >= 5 for v in supplier.product_history.values()):
                    defects.append(candidate)
            elif candidate == "folio_gap":
                if supplier.invoices_emitted >= 3:
                    defects.append(candidate)
            else:
                defects.append(candidate)

        inv = _render_invoice(
            env, rng, supplier, receptor_rfc, receptor_nombre, fecha, defects, reused_uuid
        )
        if reused_uuid is None:
            emitted_uuids.append(inv.label["uuid"])
        for kind in inv.label["defects"]:
            counts[kind] += 1

        (out_dir / inv.filename).write_text(inv.xml, encoding="utf-8")
        labels.append(inv.label)

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", encoding="utf-8") as fh:
        for label in labels:
            fh.write(json.dumps(label, ensure_ascii=False) + "\n")

    clean = sum(1 for lb in labels if not lb["defects"])
    print(f"wrote {len(labels)} invoices to {out_dir}")
    print(f"labels: {labels_path}")
    print(f"  clean:    {clean}")
    for kind, count in sorted(counts.items()):
        if count:
            print(f"  {kind:16s} {count}")
    skipped = [k for k, v in counts.items() if v == 0]
    if skipped:
        print(f"  (never injected, preconditions unmet: {', '.join(skipped)})")
    return labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # 300 rather than a token 60: the price-outlier detector needs 5 prior
    # invoices for the same (supplier, product) before a spike is even
    # injectable, and a 60-invoice corpus spread over 8 suppliers never gets
    # there — the generator honestly reports "price_spike never injected" and
    # the detector goes completely unexercised. A default that silently skips
    # a detector is a footgun; generation still takes under two seconds.
    ap.add_argument("--n", type=int, default=300, help="number of invoices")
    ap.add_argument("--defect-rate", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=Path("data/synth"))
    ap.add_argument("--labels", type=Path, default=Path("evals/datasets/labeled.jsonl"))
    ap.add_argument("--seed", type=int, default=1312, help="fixed for reproducible evals")
    ap.add_argument("--suppliers", type=int, default=8)
    ap.add_argument("--receptor-rfc", default="XAXX010101000")
    ap.add_argument("--receptor-nombre", default="Mi Empresa SA de CV")
    ap.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        help="keep existing *.xml in --out (they will not match the labels)",
    )
    args = ap.parse_args()

    generate(
        n=args.n,
        defect_rate=args.defect_rate,
        out_dir=args.out,
        labels_path=args.labels,
        seed=args.seed,
        n_suppliers=args.suppliers,
        receptor_rfc=args.receptor_rfc,
        receptor_nombre=args.receptor_nombre,
        clean=args.clean,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
