"""Tier 2: invoices that arrive without their XML.

Reached only when there is nothing better. A CFDI with its XML attached is
parsed deterministically for free and cannot be transcribed wrong; sending it
to a model would pay per document to *introduce* a failure mode.

The output goes through exactly the same `validate.rules` as the XML path.
That is the whole safety argument: a hallucinated total arrives as a
well-formed string, fails arithmetic, and lands in the review queue. Nothing a
model produces reaches `invoices` without balancing first.
"""

from __future__ import annotations

import time
from pathlib import Path

from cfdi_agent.extract.providers.base import (
    LLMProvider,
    LLMResult,
    ProviderError,
    get_provider,
)
from cfdi_agent.schemas import ParsedInvoice

MEDIA_BY_SUFFIX = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def media_type_for(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    media = MEDIA_BY_SUFFIX.get(suffix)
    if media is None:
        raise ProviderError(f"no known media type for {suffix!r}")
    return media


def extract_from_document(
    data: bytes,
    *,
    media_type: str,
    provider: LLMProvider | None = None,
) -> tuple[ParsedInvoice, LLMResult]:
    """Transcribe a document into the canonical model.

    Returns the invoice *and* the raw `LLMResult`, because the caller has to
    write latency, tokens and cost to `extraction_runs`. Dropping the result
    would leave the cost report with an incomplete denominator.
    """
    provider = provider or get_provider()
    started = time.perf_counter()
    result = provider.extract_invoice(data, media_type=media_type)

    try:
        invoice = result.content.to_parsed()
    except Exception as exc:  # noqa: BLE001 - a bad transcription is a normal outcome
        raise ProviderError(
            f"model returned a document that could not be interpreted as a CFDI: {exc}"
        ) from exc

    # The provider may not have timed the conversion; make sure the caller sees
    # wall-clock time for the whole tier-2 step, not just the HTTP call.
    if result.latency_ms == 0:
        object.__setattr__(
            result, "latency_ms", int((time.perf_counter() - started) * 1000)
        )
    return invoice, result


def extract_from_file(
    path: str | Path, *, provider: LLMProvider | None = None
) -> tuple[ParsedInvoice, LLMResult]:
    p = Path(path)
    return extract_from_document(
        p.read_bytes(), media_type=media_type_for(p), provider=provider
    )
