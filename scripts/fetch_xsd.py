"""Download the SAT's XSD chain and rewrite it for offline validation.

Why this exists: `cfdv40.xsd` imports two more schemas by absolute URL, and
those import more. Handing the remote URL straight to lxml would mean every
test run hits sat.gob.mx — slow, flaky, and it turns an offline test suite into
one that fails when the SAT has a bad afternoon. So the whole chain is fetched
once, `schemaLocation` attributes are rewritten to relative local paths, and
the result is committed.

A `manifest.json` records the source URL, sha256 and byte size of every file.
Vendored third-party artifacts with no provenance are how supply chains rot;
anyone can re-run this script and diff the manifest to confirm nothing drifted.

    python -m scripts.fetch_xsd
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from lxml import etree

XS = "http://www.w3.org/2001/XMLSchema"

# CFDI 4.0 plus the Timbre Fiscal Digital complement. TFD is *not* imported by
# cfdv40.xsd — the Complemento element is an `xs:any`, so the stamp schema has
# to be loaded alongside rather than pulled in transitively.
#
# 3.3 is here because the parser accepts it: validating only the 4.0 corpus
# would leave the 3.3 path unverified.
DEFAULT_ROOTS = (
    "http://www.sat.gob.mx/sitio_internet/cfd/4/cfdv40.xsd",
    "http://www.sat.gob.mx/sitio_internet/cfd/3/cfdv33.xsd",
    "http://www.sat.gob.mx/sitio_internet/cfd/TimbreFiscalDigital/TimbreFiscalDigitalv11.xsd",
)

OUT_DIR = Path(__file__).resolve().parents[1] / "xsd"


def local_path_for(url: str, out_dir: Path) -> Path:
    """Mirror the URL's host and path under `out_dir`.

    Keeping the full path prevents collisions — several SAT schemas share a
    basename across directories.
    """
    parsed = urlparse(url)
    return out_dir / parsed.netloc / parsed.path.lstrip("/")


def discover_refs(xsd_bytes: bytes, base_url: str) -> list[str]:
    """Absolute URLs of every xs:import / xs:include in this schema."""
    root = etree.fromstring(
        xsd_bytes, parser=etree.XMLParser(resolve_entities=False, no_network=True)
    )
    out: list[str] = []
    for tag in ("import", "include", "redefine"):
        for node in root.iter(f"{{{XS}}}{tag}"):
            loc = node.get("schemaLocation")
            if loc:
                out.append(urljoin(base_url, loc))
    return out


def rewrite_locations(xsd_bytes: bytes, base_url: str, out_dir: Path) -> bytes:
    """Point every schemaLocation at the local copy, relative to this file."""
    root = etree.fromstring(
        xsd_bytes, parser=etree.XMLParser(resolve_entities=False, no_network=True)
    )
    self_path = local_path_for(base_url, out_dir)
    changed = False
    for tag in ("import", "include", "redefine"):
        for node in root.iter(f"{{{XS}}}{tag}"):
            loc = node.get("schemaLocation")
            if not loc:
                continue
            target = local_path_for(urljoin(base_url, loc), out_dir)
            relative = Path(
                __import__("os").path.relpath(target, start=self_path.parent)
            ).as_posix()
            node.set("schemaLocation", relative)
            changed = True
    if not changed:
        return xsd_bytes
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8")


def fetch_all(roots: tuple[str, ...], out_dir: Path, timeout: float) -> dict:
    seen: dict[str, dict] = {}
    queue = list(roots)

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        while queue:
            url = queue.pop(0)
            if url in seen:
                continue

            resp = client.get(url)
            resp.raise_for_status()
            raw = resp.content

            # Guard against a captive portal or error page returning 200 with
            # HTML. Writing that into the schema cache would produce a baffling
            # validation error later.
            if b"<xs:schema" not in raw and b"<schema" not in raw:
                raise RuntimeError(
                    f"{url} returned {len(raw)} bytes that are not an XSD "
                    f"(starts with {raw[:80]!r})"
                )

            for ref in discover_refs(raw, url):
                if ref not in seen:
                    queue.append(ref)

            rewritten = rewrite_locations(raw, url, out_dir)
            dest = local_path_for(url, out_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(rewritten)

            seen[url] = {
                "url": url,
                "path": dest.relative_to(out_dir).as_posix(),
                "sha256_upstream": hashlib.sha256(raw).hexdigest(),
                "bytes_upstream": len(raw),
                "rewritten": rewritten != raw,
            }
            print(f"  {len(raw):>7d}b  {dest.relative_to(out_dir)}")

    return {
        "fetched_at": datetime.now(UTC).isoformat(),
        "roots": list(roots),
        "note": (
            "schemaLocation attributes are rewritten to relative local paths; "
            "sha256_upstream is of the ORIGINAL bytes as served by the SAT."
        ),
        "files": sorted(seen.values(), key=lambda f: f["path"]),
    }


# A CFDI's `cfdi:Complemento` is an `xs:any namespace="##other"
# processContents="strict"`, which means the validator must already know the
# Timbre Fiscal Digital schema — it is never reached by following imports from
# cfdv40.xsd. A wrapper schema that imports both namespaces is the standard way
# to hand lxml one schema object covering the whole document.
WRAPPERS = {
    "cfdi40.xsd": (
        ("http://www.sat.gob.mx/cfd/4", "www.sat.gob.mx/sitio_internet/cfd/4/cfdv40.xsd"),
        (
            "http://www.sat.gob.mx/TimbreFiscalDigital",
            "www.sat.gob.mx/sitio_internet/cfd/TimbreFiscalDigital/TimbreFiscalDigitalv11.xsd",
        ),
    ),
    "cfdi33.xsd": (
        ("http://www.sat.gob.mx/cfd/3", "www.sat.gob.mx/sitio_internet/cfd/3/cfdv33.xsd"),
        (
            "http://www.sat.gob.mx/TimbreFiscalDigital",
            "www.sat.gob.mx/sitio_internet/cfd/TimbreFiscalDigital/TimbreFiscalDigitalv11.xsd",
        ),
    ),
}


def write_wrappers(out_dir: Path) -> list[str]:
    written: list[str] = []
    for name, imports in WRAPPERS.items():
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<!-- Generated by scripts/fetch_xsd.py. Do not edit by hand. -->",
            '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">',
        ]
        for namespace, location in imports:
            if not (out_dir / location).exists():
                raise RuntimeError(f"wrapper {name} references missing {location}")
            lines.append(f'  <xs:import namespace="{namespace}" schemaLocation="{location}"/>')
        lines.append("</xs:schema>")
        (out_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(name)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--root", action="append", dest="roots", default=None)
    args = ap.parse_args()

    roots = tuple(args.roots) if args.roots else DEFAULT_ROOTS
    print(f"fetching {len(roots)} root schema(s) into {args.out}")
    try:
        manifest = fetch_all(roots, args.out, args.timeout)
    except (httpx.HTTPError, RuntimeError) as exc:
        print(f"\nfetch failed: {exc}", file=sys.stderr)
        return 1

    written = write_wrappers(args.out)
    for name in written:
        print(f"  wrapper   {name}")

    manifest["wrappers"] = written
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n{len(manifest['files'])} files + {len(written)} wrappers + manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
