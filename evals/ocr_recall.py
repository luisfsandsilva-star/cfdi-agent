"""Is the field present in the OCR text at all?

    python -m evals.ocr_recall data/real --model ibm/granite-docling:258m

A weaker and earlier question than `evals.vision_accuracy`, and the one that
decides whether a **two-stage** extraction pipeline is worth building:

    PDF → OCR/layout model → text and tables → extraction → JSON

instead of the single shot the vision path uses today:

    PDF → general VLM → JSON

The split is attractive because a document-parsing model is small — 258M to
900M against 3B — and because the second stage on a CFDI is nearly
deterministic: the printed labels are fixed by the SAT, so "Folio Fiscal" is
followed by the UUID on every invoice from every issuer. That would push work
*down* the tiers rather than reaching for a bigger model, which is the same
argument the project makes about XML.

The whole idea rests on one assumption, which is what this measures: **if the
value is in the text, a second stage can extract it. If it is absent, nothing
downstream can recover it.** Stage-1 recall is therefore the ceiling on
whatever the full pipeline could ever achieve.

Money is compared numerically rather than by substring. The first version of
this reported `total 0/4` while `subtotal` was 3/4, which is not a plausible
OCR failure pattern — the XML holds `6863.40` and the page prints `6,863.4`,
so containment fails on a trailing zero. That turned out to hide a real
finding underneath, but only after the comparison stopped producing a false
one.

Same confidentiality rule as the rest of `evals/`: counts only. Whether a
field was found is printed; the value never is. OCR output is cached to
`.ocr_cache/`, which lives under a gitignored path — it holds full invoice
text and must never be committed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import statistics
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx

# The default is the docling task prompt. A document-parsing model is usually
# trained on a small set of instructions and drifts badly outside them, so this
# is a flag rather than a constant.
DEFAULT_PROMPT = "Convert this page to docling."
CACHE_DIR = Path("data/.ocr_cache")

NUMBER = re.compile(r"\d[\d,]*\.?\d*")
CENTS = Decimal("0.01")


class TranscriptionFailed(RuntimeError):
    """One document did not come back. The other 92 still count."""


def _error_message(response, limit: int = 200) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:limit].strip()
    error = body.get("error", body) if isinstance(body, dict) else body
    if isinstance(error, dict):
        error = error.get("message", error)
    return str(error)[:limit].strip()
FIELDS = ("uuid", "rfc_emisor", "rfc_receptor", "subtotal", "total", "line_amounts")


def _numbers(text: str) -> set[Decimal]:
    """Every number on the page, as a number.

    Comparing money as text fails on formatting that no reader would notice:
    `6863.40` in the XML against `6,863.4` on the page.
    """
    out: set[Decimal] = set()
    for token in NUMBER.findall(text):
        try:
            out.add(Decimal(token.replace(",", "")))
        except InvalidOperation:
            pass
    return out


def _has_money(target: Decimal, found: set[Decimal]) -> bool:
    """Whether an amount is on the page, at the precision the XML stores it.

    Exact set membership was wrong, and it reported six invoices as missing
    their line amounts when every one of them was there. The page prints the
    unrounded product -- `515.517242` -- while the XML's `Importe` attribute is
    that value rounded to cents, `515.52`. A downstream stage can round; it
    cannot invent. So the field is present.
    """
    for value in found:
        if value == target:
            return True
        try:
            if value.quantize(CENTS) == target:
                return True
        except InvalidOperation:
            # A certificate serial is a 20-digit integer on the same page.
            # Quantizing it overflows, and it was never a candidate anyway.
            continue
    return False


def _present(inv, text: str) -> dict[str, bool]:
    flat = re.sub(r"\s", "", text).upper()
    nums = _numbers(text)
    return {
        "uuid": inv.uuid.upper() in flat,
        "rfc_emisor": inv.rfc_emisor.upper() in flat,
        "rfc_receptor": inv.rfc_receptor.upper() in flat,
        "subtotal": _has_money(inv.subtotal, nums),
        "total": _has_money(inv.total, nums),
        "line_amounts": all(_has_money(c.importe, nums) for c in inv.conceptos),
    }


# Not a model. Reads the PDF's own text layer, which most CFDIs carry because
# they are generated rather than scanned. Included here so it is scored by the
# same harness on the same corpus as the models it is competing with -- an
# alternative measured a different way is not a comparison.
TEXT_LAYER = "textlayer"


def transcribe(
    pdf: Path, *, model: str, prompt: str, base_url: str, dpi: int, timeout: float
) -> tuple[str, float]:
    """One page through the OCR model, cached by (model, prompt, file)."""
    from cfdi_agent.extract.pdf_text import extract_text_layer
    from cfdi_agent.extract.providers.openai_compat import rasterize_pdf

    if model == TEXT_LAYER:
        started = time.perf_counter()
        text = extract_text_layer(pdf.read_bytes())
        return text, time.perf_counter() - started

    key = hashlib.sha256(
        f"{model}|{prompt}|{dpi}|{pdf.name}".encode()
    ).hexdigest()[:16]
    cached = CACHE_DIR / f"{key}.json"
    if cached.exists():
        blob = json.loads(cached.read_text())
        return blob["text"], blob["seconds"]

    image = rasterize_pdf(pdf.read_bytes(), dpi=dpi)[0]
    payload = {
        "model": model,
        "max_tokens": 8192,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.standard_b64encode(image).decode("ascii")
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    started = time.perf_counter()
    response = httpx.post(f"{base_url}/chat/completions", json=payload, timeout=timeout)
    seconds = time.perf_counter() - started
    if response.status_code >= 400:
        # The status line alone is useless. llama-server explains itself in the
        # body -- one document here failed with "the model produced output that
        # does not match the expected peg-native format", which is its response
        # parser rejecting a transcription it had already produced. Nothing
        # about that is visible from "500".
        raise TranscriptionFailed(
            f"{response.status_code} {_error_message(response)}"
        )
    text = response.json()["choices"][0]["message"]["content"] or ""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps({"text": text, "seconds": seconds}))
    return text, seconds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directory", type=Path)
    ap.add_argument("--model", default="ibm/granite-docling:258m")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args(argv)

    from cfdi_agent.extract.xml_parser import parse_cfdi_bytes
    from evals.vision_accuracy import find_pairs

    pairs = find_pairs(args.directory)
    if not pairs:
        print(f"no PDF/XML pairs under {args.directory}", file=sys.stderr)
        return 1

    results: list[dict[str, bool]] = []
    seconds: list[float] = []
    failures: list[str] = []
    for pdf_path, xml_path in pairs:
        inv = parse_cfdi_bytes(xml_path.read_bytes())
        try:
            text, elapsed = transcribe(
                pdf_path,
                model=args.model,
                prompt=args.prompt,
                base_url=args.base_url,
                dpi=args.dpi,
                timeout=args.timeout,
            )
        except (TranscriptionFailed, httpx.HTTPError) as exc:
            # A batch of 93 must not be lost to one document. A model that
            # cannot read a page is a result about that model, and scoring it
            # as "found nothing" is both true and the honest thing to report.
            failures.append(str(exc))
            results.append(dict.fromkeys(FIELDS, False))
            print(f"      —  FAILED  {exc}", flush=True)
            continue
        found = _present(inv, text)
        results.append(found)
        seconds.append(elapsed)
        marks = " ".join(f"{k}={'ok' if v else '--'}" for k, v in found.items())
        print(f"  {elapsed:5.1f}s  {marks}", flush=True)

    n = len(results)
    print()
    print(f"Stage-1 recall · {args.model} · {n} invoices · {args.dpi} DPI")
    print(f"prompt: {args.prompt!r}")
    print("Counts only. Whether a field was found, never the value.")
    if failures:
        print(f"{len(failures)} document(s) failed to transcribe and score as "
              "found-nothing:")
        for message in sorted(set(failures))[:5]:
            print(f"  {failures.count(message):>3}×  {message}")
    print()
    print("| field | found | of |")
    print("|---|---:|---:|")
    for name in FIELDS:
        print(f"| `{name}` | {sum(r[name] for r in results)} | {n} |")
    print()
    print(f"median latency {statistics.median(seconds):.1f}s")
    print()
    print("This is the ceiling on any two-stage pipeline built on this model.")
    print("A field missing here cannot be recovered by a later stage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
