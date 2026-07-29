"""Run real invoices through the pipeline and report only aggregates.

    python -m evals.real_corpus data/real

Different question from `run_eval`, and the difference is the whole point.

`run_eval` measures **accuracy**: synthetic invoices come with ground truth, so
recall and precision are computable. That is impossible here — nobody labelled
a real inbox, and a real inbox is roughly 99% clean anyway, so even a perfect
detector would have almost nothing to be right about.

This measures **robustness**: does the parser survive documents it did not
generate? Real CFDIs carry addendas, CFDI 3.3 alongside 4.0, complementos this
project has never seen, retenciones, multi-currency, twelve-decimal unit
prices, and PAC quirks. The generator produces none of that, so every number
in `report.md` is measured on a corpus written by the same hand that wrote the
parser. This is the check on that.

The two outputs stay separate on purpose. `evals/report.md` is published;
this one is not, and neither is its input.

Confidentiality
---------------
Real invoices carry RFCs, legal names, addresses, amounts, and SAT-traceable
UUIDs — personal data under the LFPDPPP once a persona física is involved.

So nothing this module prints is derived from the *content* of a document.
Everything is a count, a rate, a percentile, or a schema-defined tag name.
Distinct issuers is a `count(distinct)`; it never names one. Parse failures
report the exception class and a scrubbed message, because a failure you
cannot see is a failure you cannot fix — `_scrub` removes RFCs, UUIDs, dates
and digit runs from that message before it reaches the terminal.

The corpus lives under `data/`, which is gitignored, and lands in its own
`cfdi_real` database so it can be dropped in one statement and can never be
confused with the eval or development ledger.

    dropdb cfdi_real     # that is the whole cleanup

One thing this cannot hide: a PDF has to reach a vision model to be read, and
tier 2 means the Anthropic API. The bytes of that invoice leave the building.
For someone else's invoices that is a decision to make deliberately and out
loud, not a default — `--skip-pdf` refuses the whole PDF path, and the tier 1
seam (a local VLM) is what makes the on-premise answer possible.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

REAL_DB = "cfdi_real"

XML_SUFFIXES = {".xml"}
PDF_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}

# Scrubbing by value pattern was the first attempt and it does not work. The
# rule `[A-ZÑ&]{3,4}\d{6}[A-Z0-9]{3}` misses a *malformed* RFC — and a
# malformed RFC is precisely what appears in a validation error, so the one
# case that reaches the terminal is the one case the pattern cannot catch.
#
# So scrub by position instead. lxml states the offending value in exactly one
# shape, `The value 'X'`, and everything else it quotes is schema vocabulary:
# element names, attribute names, facet names, the regex the facet enforces.
# Those are public — they come out of the SAT's own XSD — and they are the
# entire diagnostic value of the message.
_VALUE = re.compile(r"(The value |El valor )'[^']*'")
_KEEP_QUOTED = re.compile(r"(Element|element|attribute|atributo|facet) '[^']{0,120}'")
_FALLBACK = (
    (re.compile(r"\b[0-9A-Fa-f]{8}(-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}\b"), "<uuid>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}(T[\d:.]+)?"), "<date>"),
    (re.compile(r"\d[\d,]*\.\d{2,}"), "<amount>"),
    # Deliberately looser than the real RFC rule. This one runs over messages
    # this project writes itself — a rejection reason names the RFC it rejected
    # — where the value is unquoted and the positional scrub above cannot see
    # it. Being loose is correct here: over-redacting a catalog code costs a
    # little clarity, under-redacting costs somebody's tax ID.
    (re.compile(r"\b[A-ZÑ&]{3,4}\d{4,6}[A-Z0-9]{0,3}\b"), "<rfc>"),
)


def _scrub(text: str, limit: int = 200) -> str:
    """Keep the shape of an error, drop every value inside it.

    Scrubbing collapses the variants too: seven documents failing the same
    facet on seven different RFCs become one line with a count, which is the
    reading you want anyway.
    """
    text = _VALUE.sub(r"\1'<redacted>'", text)
    # Anything still quoted that is not schema vocabulary is unknown provenance
    # — a PAC's custom message, an addenda's contents. Assume it is data.
    kept: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        kept.append(match.group(0))
        return f"\x00{len(kept) - 1}\x00"

    text = _KEEP_QUOTED.sub(_stash, text)
    text = re.sub(r"'[^']*'", "'<redacted>'", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: kept[int(m.group(1))], text)

    for pattern, replacement in _FALLBACK:
        text = pattern.sub(replacement, text)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ----------------------------------------------------------------- structure


def survey_xml(data: bytes) -> dict[str, list[str]]:
    """What this document *contains*, independent of whether we parsed it.

    Runs on raw bytes rather than on a `ParsedInvoice`, so a document the
    parser rejects still reports its shape. That is the more useful direction:
    the failures are exactly the ones worth characterising.

    Every value returned is a tag name or a schema-defined attribute value —
    version numbers, complemento names, tax rates. No amounts, no identifiers.
    """
    from lxml import etree

    from cfdi_agent.extract.xml_parser import CFDI_NAMESPACES

    out: dict[str, list[str]] = {
        "version": [],
        "complementos": [],
        "addenda": [],
        "moneda": [],
        "tipo_comprobante": [],
    }
    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
    root = etree.fromstring(data, parser=parser)

    ns = next((n for n in CFDI_NAMESPACES if root.tag.startswith(f"{{{n}}}")), None)
    out["version"].append(root.get("Version") or root.get("version") or "unknown")
    for attr, key in (("Moneda", "moneda"), ("TipoDeComprobante", "tipo_comprobante")):
        value = root.get(attr)
        if value:
            out[key].append(value)

    if ns:
        for complemento in root.findall(f"{{{ns}}}Complemento"):
            # Tag name only. `{...TimbreFiscalDigital}TimbreFiscalDigital` says
            # the document is stamped; it says nothing about who stamped it.
            for child in complemento:
                out["complementos"].append(etree.QName(child).localname)
        if root.find(f"{{{ns}}}Addenda") is not None:
            # The free-for-all element. Every PAC and every large buyer puts
            # something different in here, and it is the usual reason a parser
            # written against the XSD alone falls over on real documents.
            out["addenda"].append("present")
    return out


# ------------------------------------------------------------------- running


def _prepare_database() -> None:
    import psycopg

    base = os.environ.get("DATABASE_URL", "postgresql://cfdi:cfdi@localhost:5432/cfdi")
    admin = base.rsplit("/", 1)[0] + "/postgres"
    with psycopg.connect(admin, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (REAL_DB,))
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE "{REAL_DB}"')

    os.environ["DATABASE_URL"] = base.rsplit("/", 1)[0] + f"/{REAL_DB}"
    from cfdi_agent.config import get_config

    get_config.cache_clear()
    from cfdi_agent.db.init import apply_schema

    apply_schema()


def run(directory: Path, *, skip_pdf: bool, company_rfc: str | None) -> int:
    from cfdi_agent.db.conn import connect, fetch_all, fetch_one
    from cfdi_agent.ingest.pipeline import ingest_bytes
    from cfdi_agent.validate.xsd_validator import schemas_available, validate_bytes

    files = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in XML_SUFFIXES | PDF_SUFFIXES
    )
    if not files:
        print(f"no documents under {directory}", file=sys.stderr)
        return 1

    by_suffix = Counter(p.suffix.lower() for p in files)
    if skip_pdf:
        files = [p for p in files if p.suffix.lower() in XML_SUFFIXES]

    statuses: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    structure: dict[str, Counter[str]] = {
        k: Counter() for k in ("version", "complementos", "addenda", "moneda",
                               "tipo_comprobante")
    }
    survey_failures: Counter[str] = Counter()
    xsd_invalid: Counter[str] = Counter()
    xsd_checked = 0
    latencies: list[float] = []

    xsd_ready = schemas_available()
    started = time.perf_counter()

    for path in files:
        data = path.read_bytes()
        is_xml = path.suffix.lower() in XML_SUFFIXES

        if is_xml:
            try:
                for key, values in survey_xml(data).items():
                    structure[key].update(values)
            except Exception as exc:  # noqa: BLE001 - characterising the corpus
                survey_failures[type(exc).__name__] += 1
            if xsd_ready:
                xsd_checked += 1
                for error in validate_bytes(data)[:1]:
                    xsd_invalid[_scrub(error)] += 1

        t0 = time.perf_counter()
        try:
            with connect() as conn:
                outcome = ingest_bytes(
                    conn, data, file_path=str(path), company_rfc=company_rfc
                )
            statuses[outcome.status] += 1
        except Exception as exc:  # noqa: BLE001 - the measurement is the crash
            statuses["failed"] += 1
            failures[f"{type(exc).__name__}: {_scrub(str(exc))}"] += 1
        latencies.append((time.perf_counter() - t0) * 1000)

    elapsed = time.perf_counter() - started
    report(
        directory=directory,
        by_suffix=by_suffix,
        processed=len(files),
        elapsed=elapsed,
        statuses=statuses,
        failures=failures,
        structure=structure,
        survey_failures=survey_failures,
        xsd_checked=xsd_checked,
        xsd_invalid=xsd_invalid,
        latencies=latencies,
        fetch_all=fetch_all,
        fetch_one=fetch_one,
        skip_pdf=skip_pdf,
    )
    return 0


# ------------------------------------------------------------------ reporting


def _bar(label: str, n: int, total: int, width: int = 24) -> str:
    filled = round(width * n / total) if total else 0
    return f"  {label:<28} {n:>5}  {'█' * filled}{'·' * (width - filled)}"


def report(**kw) -> None:
    total = kw["processed"]
    print()
    print(f"Real corpus · {kw['directory']} · {total} documents · "
          f"{kw['elapsed']:.1f}s")
    print("Aggregates only. No field of any document is printed.")
    if kw["skip_pdf"]:
        print("PDF path skipped (--skip-pdf): no document left this machine.")
    print()

    print("Documents")
    for suffix, n in kw["by_suffix"].most_common():
        print(f"  {suffix:<28} {n:>5}")
    print()

    print("Outcome")
    for status, n in kw["statuses"].most_common():
        print(_bar(status, n, total))
    print()

    # Without this, `needs_review` is an unexplained bar and reads like a
    # parser failure. The first real run put all four invoices here, and the
    # reason was that COMPANY_RFC still held the synthetic default — the guard
    # working correctly against the wrong configuration. A refusal has to say
    # which refusal it was.
    refused = kw["fetch_all"](
        "SELECT reason, count(*) AS n FROM review_queue GROUP BY reason ORDER BY n DESC"
    )
    if refused:
        print("Held for review — why the pipeline declined to persist")
        for row in refused[:12]:
            print(f"  {row['n']:>3}×  {_scrub(row['reason'])}")
        print()

    if kw["failures"]:
        print("Failures — the reason the corpus is worth running")
        for message, n in kw["failures"].most_common(12):
            print(f"  {n:>3}×  {message}")
        print()
    if kw["survey_failures"]:
        print("Documents that are not parseable XML at all")
        for name, n in kw["survey_failures"].most_common():
            print(f"  {n:>3}×  {name}")
        print()

    print("Corpus shape — what the generator does not produce")
    for key in ("version", "tipo_comprobante", "moneda", "addenda", "complementos"):
        counter = kw["structure"][key]
        if not counter:
            continue
        print(f"  {key}")
        for value, n in counter.most_common(10):
            print(f"    {value:<26} {n:>5}")
    print()

    if kw["xsd_checked"]:
        bad = sum(kw["xsd_invalid"].values())
        print(f"XSD conformance — {kw['xsd_checked'] - bad}/{kw['xsd_checked']} "
              "validate against the SAT schema chain")
        for message, n in kw["xsd_invalid"].most_common(8):
            print(f"  {n:>3}×  {message}")
        print()

    lat = sorted(kw["latencies"])
    if lat:
        print("Latency (ms)")
        print(f"  p50 {statistics.median(lat):.0f} · "
              f"p95 {lat[min(len(lat) - 1, int(len(lat) * 0.95))]:.0f} · "
              f"max {lat[-1]:.0f}")
        print()

    ledger = kw["fetch_one"](
        "SELECT count(*) AS invoices, count(DISTINCT rfc_emisor) AS issuers "
        "FROM invoices"
    )
    print("Ledger")
    print(f"  invoices persisted           {ledger['invoices']:>5}")
    print(f"  distinct issuers             {ledger['issuers']:>5}")
    print()

    # The one precision signal a real corpus can give. Ground truth is absent,
    # but the prior is strong: a real accounts-payable inbox is overwhelmingly
    # clean, so a detector firing on a large share of it is reporting noise.
    # This is a smell test, not a measurement, and is labelled as such.
    fired = kw["fetch_all"](
        "SELECT kind, severity, count(*) AS n FROM anomalies "
        "GROUP BY kind, severity ORDER BY n DESC"
    )
    print("Anomalies fired — no ground truth, read as a smell test")
    if not fired:
        print("  none")
    for row in fired:
        rate = 100 * row["n"] / ledger["invoices"] if ledger["invoices"] else 0
        print(f"  {row['kind']:<28} {row['n']:>5}  {rate:5.1f}% of invoices "
              f"({row['severity']})")
    print()

    cost = kw["fetch_one"](
        "SELECT count(*) AS runs, "
        "       count(*) FILTER (WHERE tier = 2) AS tier2, "
        "       coalesce(sum(cost_usd), 0) AS usd FROM extraction_runs"
    )
    print("Cost")
    print(f"  documents processed          {cost['runs']:>5}")
    print(f"  reached a vision model       {cost['tier2']:>5}")
    print(f"  total                       ${cost['usd']:.4f}")
    if cost["tier2"]:
        print(f"  per PDF                     ${cost['usd'] / cost['tier2']:.4f}")
        print()
        print("  Those PDFs were sent to the Anthropic API. That is what tier 2")
        print("  means. Use --skip-pdf, or a local VLM through the tier 1 seam,")
        print("  when the documents belong to somebody else.")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Robustness check against real invoices. Prints aggregates only."
    )
    ap.add_argument("directory", type=Path, help="folder of real .xml / .pdf documents")
    ap.add_argument(
        "--skip-pdf",
        action="store_true",
        help="XML only. No document reaches a model or leaves this machine.",
    )
    ap.add_argument(
        "--company-rfc",
        default=None,
        help="receiver RFC to treat as the company (defaults to COMPANY_RFC)",
    )
    ap.add_argument(
        "--keep",
        action="store_true",
        help=f"do not drop the {REAL_DB} tables first",
    )
    args = ap.parse_args(argv)

    if not args.directory.is_dir():
        print(f"not a directory: {args.directory}", file=sys.stderr)
        return 1

    _prepare_database()
    if not args.keep:
        from cfdi_agent.db.conn import connect

        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                "TRUNCATE invoices, suppliers, line_items, taxes, anomalies, "
                "review_queue, extraction_runs, seen_folios, processed_files "
                "RESTART IDENTITY CASCADE"
            )
    return run(
        args.directory, skip_pdf=args.skip_pdf, company_rfc=args.company_rfc
    )


if __name__ == "__main__":
    raise SystemExit(main())
