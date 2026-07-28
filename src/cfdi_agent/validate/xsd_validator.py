"""Validate a CFDI against the SAT's official XSD.

Deliberately **not** wired into the ingest path. A structurally invalid invoice
must still be parsed and reported — routing it to the review queue on a schema
error would hide exactly the documents worth looking at, and the deterministic
detectors already catch the failures that cost money.

Where this earns its keep is the generator. `synth/generate_cfdi.py` writes XML
from a hand-written Jinja template; "structurally faithful to CFDI 4.0" was an
unverified claim until this module existed. Validating the generated corpus
against the real schema turns it into a checkable one, and any drift in the
template shows up as a test failure rather than as quietly unrealistic
fixtures.

Note that `catCFDI.xsd` enumerates the SAT catalogs, so this validates catalog
*codes* too — a ClaveProdServ or ClaveUnidad that does not exist fails here
even though the document is well-formed. That is a strictly stronger check than
the partial catalog subset in `validate.catalogs`.

Requires the vendored schemas: `python -m scripts.fetch_xsd`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from lxml import etree

XSD_DIR = Path(__file__).resolve().parents[3] / "xsd"

# Root namespace -> wrapper schema that imports it together with the Timbre
# Fiscal Digital namespace.
WRAPPER_FOR_NAMESPACE = {
    "http://www.sat.gob.mx/cfd/4": "cfdi40.xsd",
    "http://www.sat.gob.mx/cfd/3": "cfdi33.xsd",
}


class SchemasUnavailableError(RuntimeError):
    """The vendored XSD chain is missing."""


@lru_cache(maxsize=4)
def _load_schema(wrapper: str) -> etree.XMLSchema:
    """Compile a wrapper schema. Cached — catCFDI.xsd is ~6 MB to parse."""
    path = XSD_DIR / wrapper
    if not path.exists():
        raise SchemasUnavailableError(
            f"{path} not found. Run `python -m scripts.fetch_xsd` to vendor the "
            "SAT schema chain."
        )
    return etree.XMLSchema(etree.parse(str(path)))


def schemas_available() -> bool:
    return all((XSD_DIR / w).exists() for w in WRAPPER_FOR_NAMESPACE.values())


def validate_bytes(data: bytes) -> list[str]:
    """Return a list of schema violations. Empty means valid.

    Returns errors rather than raising: a caller usually wants to report every
    problem in a document, not just the first.
    """
    try:
        doc = etree.fromstring(
            data, parser=etree.XMLParser(resolve_entities=False, no_network=True)
        )
    except etree.XMLSyntaxError as exc:
        return [f"malformed XML: {exc}"]

    namespace = etree.QName(doc).namespace
    wrapper = WRAPPER_FOR_NAMESPACE.get(namespace)
    if wrapper is None:
        return [
            f"root namespace {namespace!r} is not a CFDI version this project "
            f"validates (known: {sorted(WRAPPER_FOR_NAMESPACE)})"
        ]

    schema = _load_schema(wrapper)
    if schema.validate(doc.getroottree()):
        return []
    return [f"line {e.line}: {e.message}" for e in schema.error_log]


def validate_file(path: str | Path) -> list[str]:
    return validate_bytes(Path(path).read_bytes())
