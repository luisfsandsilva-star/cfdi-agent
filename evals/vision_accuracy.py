"""Score the vision path against the deterministic parse of the same invoice.

    python -m evals.vision_accuracy data/real --provider local --model qwen2.5vl:3b

The measurement this project was missing, and it was sitting in the corpus the
whole time: **a real CFDI arrives as a PDF and an XML of the same invoice.**
The XML parses deterministically, so it is ground truth — free, exact, and not
written by the same hand as the thing being scored.

That is a materially better test than the synthetic corpus can give. A rendered
synthetic invoice is a PDF this project laid out itself, in one template, with
one font. These are real supplier PDFs: different layouts, different logos,
different places to hide the UUID, and whatever the issuer's PAC decided a
receipt should look like.

It also answers the tier question with numbers instead of an argument. Tier 0
reads the XML for free and is exact by construction. Tier 1 and tier 2 read the
picture of that same invoice, and this reports what that costs in accuracy,
seconds, and dollars. When a supplier sends only a PDF — which happens — that
difference is the whole decision.

Confidentiality is the same rule as `evals.real_corpus`: counts and rates only.
A field is right or wrong; neither the expected nor the extracted value is ever
printed. That is enough to act on, because the fix for a wrong `uuid` is never
"look at this particular UUID" — it is more resolution, a better model, or a
different tier.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

# Ordered by what an accounts-payable process actually depends on. `uuid` is
# first because it is the fiscal identity of the document: a wrong total is a
# wrong number, a wrong UUID is a different invoice.
CRITICAL = ("uuid", "rfc_emisor", "total")
COMPARED = (
    "uuid",
    "rfc_emisor",
    "rfc_receptor",
    "subtotal",
    "total",
    "moneda",
    "n_conceptos",
)


@dataclass
class FieldScore:
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else float("nan")


@dataclass
class Run:
    scores: dict[str, FieldScore] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)
    cost_usd: Decimal = Decimal(0)
    tokens_in: int = 0
    tokens_out: int = 0
    documents: int = 0
    exact_documents: int = 0
    failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- comparison


def _money(value) -> Decimal | None:
    try:
        return Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, AttributeError, TypeError):
        return None


def _same(field_name: str, expected, actual) -> bool:
    """Compare one field the way the domain compares it, not the way Python does.

    `1160` and `1160.00` are the same amount, and a UUID is case-insensitive by
    RFC 4122. Scoring those as misses would report a transcription problem that
    is really a formatting difference, and would hide the real ones underneath.
    """
    if actual is None:
        return False
    if field_name == "n_conceptos":
        return int(expected) == int(actual)
    if field_name in ("subtotal", "total"):
        left, right = _money(expected), _money(actual)
        return left is not None and right is not None and left == right
    return str(expected).strip().upper() == str(actual).strip().upper()


def _expected(inv) -> dict:
    return {
        "uuid": inv.uuid,
        "rfc_emisor": inv.rfc_emisor,
        "rfc_receptor": inv.rfc_receptor,
        "subtotal": inv.subtotal,
        "total": inv.total,
        "moneda": inv.moneda,
        "n_conceptos": len(inv.conceptos),
    }


def _actual(extraction) -> dict:
    return {
        "uuid": extraction.uuid,
        "rfc_emisor": extraction.rfc_emisor,
        "rfc_receptor": extraction.rfc_receptor,
        "subtotal": extraction.subtotal,
        "total": extraction.total,
        "moneda": extraction.moneda,
        "n_conceptos": len(extraction.conceptos),
    }


# ------------------------------------------------------------------- pairing


def find_pairs(directory: Path) -> list[tuple[Path, Path]]:
    """PDFs that have an XML of the same invoice beside them.

    Matched on filename stem, which is how a PAC delivers them and how an
    accounting inbox stores them. A PDF without its XML is skipped rather than
    guessed at — there would be nothing to score it against.

    Case-insensitively, because a real delivery does not agree with itself: the
    batch this was written against names every PDF with a lowercase UUID and
    every XML with the same UUID in uppercase. An exact-stem match found 4
    pairs in it instead of 93.
    """
    xmls = {
        p.stem.lower(): p for p in directory.rglob("*") if p.suffix.lower() == ".xml"
    }
    return sorted(
        (p, xmls[p.stem.lower()])
        for p in directory.rglob("*")
        if p.suffix.lower() == ".pdf" and p.stem.lower() in xmls
    )


# ------------------------------------------------------------------- running


def score(pairs: list[tuple[Path, Path]], provider) -> Run:
    from cfdi_agent.extract.xml_parser import parse_cfdi_bytes

    run = Run(scores={name: FieldScore() for name in COMPARED})
    for pdf_path, xml_path in pairs:
        truth = _expected(parse_cfdi_bytes(xml_path.read_bytes()))
        started = time.perf_counter()
        try:
            result = provider.extract_invoice(
                pdf_path.read_bytes(), media_type="application/pdf"
            )
        except Exception as exc:  # noqa: BLE001 - a failure is a measurement
            run.failures.append(type(exc).__name__)
            run.documents += 1
            continue
        run.latencies.append((time.perf_counter() - started) * 1000)

        got = _actual(result.content)
        hits = 0
        for name in COMPARED:
            run.scores[name].total += 1
            if _same(name, truth[name], got.get(name)):
                run.scores[name].correct += 1
                hits += 1
        run.documents += 1
        run.exact_documents += hits == len(COMPARED)
        run.tokens_in += result.tokens_in or 0
        run.tokens_out += result.tokens_out or 0
        if result.cost_usd:
            run.cost_usd += Decimal(str(result.cost_usd))
    return run


def render(run: Run, *, provider_name: str, model: str, dpi: int) -> None:
    print()
    print(f"Vision accuracy · {provider_name}/{model} · {dpi} DPI · "
          f"{run.documents} invoices")
    print("Ground truth is the deterministic parse of each invoice's own XML.")
    print("Counts only. No expected or extracted value is printed.")
    print()

    if run.failures:
        print("Extraction failures")
        for name in sorted(set(run.failures)):
            print(f"  {run.failures.count(name):>3}×  {name}")
        print()

    scored = run.documents - len(run.failures)
    if not scored:
        print("Nothing was scored — every document failed to extract.")
        return

    print("| field | correct | of | accuracy |")
    print("|---|---:|---:|---:|")
    for name in COMPARED:
        s = run.scores[name]
        mark = " *" if name in CRITICAL else ""
        print(f"| `{name}`{mark} | {s.correct} | {s.total} | {s.accuracy:.0%} |")
    print()
    print("* the three fields a payment decision cannot be made without.")
    print()

    critical_ok = sum(
        1 for name in CRITICAL if run.scores[name].correct == run.scores[name].total
    )
    print(f"Fully correct invoices        {run.exact_documents}/{scored}")
    print(f"Critical fields at 100%       {critical_ok}/{len(CRITICAL)}")
    print()

    if run.latencies:
        lat = sorted(run.latencies)
        print("Latency (ms)")
        print(f"  p50 {statistics.median(lat):.0f} · "
              f"p95 {lat[min(len(lat) - 1, int(len(lat) * 0.95))]:.0f} · "
              f"max {lat[-1]:.0f}")
        print()

    print("Cost")
    print(f"  tokens in / out             {run.tokens_in} / {run.tokens_out}")
    if run.cost_usd:
        print(f"  total                      ${run.cost_usd:.4f}")
        print(f"  per invoice                ${run.cost_usd / scored:.4f}")
    else:
        # Not the same as $0.00 for an unpriced route. Local inference has no
        # per-token price at all; saying "no token cost" keeps that distinct
        # from "we failed to price it".
        print("  no token cost — local inference")
    print()
    print("Tier 0 reads the XML of these same invoices with no transcription")
    print("step at all, so it has no error of this kind to measure. This table")
    print("is what reading the picture instead costs, when only a PDF arrives.")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directory", type=Path)
    ap.add_argument("--provider", default="local", choices=("local", "anthropic"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args(argv)

    pairs = find_pairs(args.directory)
    if not pairs:
        print(f"no PDF/XML pairs under {args.directory}", file=sys.stderr)
        return 1

    if args.provider == "local":
        from cfdi_agent.extract.providers.openai_compat import OpenAICompatProvider

        model = args.model or "qwen2.5vl:3b"
        provider = OpenAICompatProvider(
            base_url=args.base_url,
            model=model,
            timeout=args.timeout,
            raster_dpi=args.dpi,
        )
    else:
        from cfdi_agent.extract.providers.anthropic_provider import AnthropicProvider

        model = args.model or "claude-opus-5"
        provider = AnthropicProvider(model=model)

    run = score(pairs, provider)
    render(run, provider_name=args.provider, model=model, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
