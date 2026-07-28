"""Decide which extraction layer handles a document.

This is the component the architecture diagram has always shown and that the
code did not have: `pdf_vision` was written, correct, and imported by nothing.
A PDF sent to `/ingest` failed XML parsing and landed in the review queue,
never reaching the vision path.

The routing rule is the project's whole thesis in one function. An XML invoice
goes to the deterministic parser: free, instant, and unable to transcribe a
value incorrectly. Everything else costs money and can be read wrong, so it
only goes to a model when there is no better option.

Detection uses magic bytes before the file extension. Invoice attachments
arrive from suppliers by email and get renamed constantly; the first bytes of
the file are the honest signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from cfdi_agent.config import get_config
from cfdi_agent.extract.providers.base import LLMProvider, ProviderError
from cfdi_agent.extract.xml_parser import CfdiParseError, parse_cfdi_bytes
from cfdi_agent.schemas import ParsedInvoice

XML = "application/xml"
PDF = "application/pdf"

MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", PDF),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)

EXTENSIONS = {
    ".xml": XML,
    ".pdf": PDF,
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

# How much of the file to inspect when looking for an XML declaration. A CFDI
# can carry a byte-order mark and leading whitespace before the first tag.
SNIFF = 512


class UnroutableDocument(ValueError):
    """These bytes cannot become an invoice.

    The message goes into `review_queue.reason` and is read by a person, so it
    says what the document was and why it stopped here.
    """


@dataclass(frozen=True, slots=True)
class RoutedDocument:
    """An extracted invoice plus the accounting for how it was extracted."""

    invoice: ParsedInvoice
    media_type: str
    tier: int
    provider: str
    model: str | None = None
    latency_ms: int = 0
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: Decimal | None = None


def detect_media_type(data: bytes, filename: str | None = None) -> str | None:
    """Identify the document. Magic bytes first, extension as a fallback."""
    for signature, media in MAGIC:
        if data.startswith(signature):
            return media

    # WebP is RIFF....WEBP, so the marker is not at offset 0.
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"

    head = data[:SNIFF].lstrip(b"\xef\xbb\xbf").lstrip()
    if head.startswith(b"<?xml") or head.startswith(b"<"):
        return XML

    if filename:
        return EXTENSIONS.get(Path(filename).suffix.lower())
    return None


def route_document(
    data: bytes,
    *,
    filename: str | None = None,
    provider: LLMProvider | None = None,
) -> RoutedDocument:
    """Extract an invoice using the cheapest layer that can read this document."""
    media = detect_media_type(data, filename)
    if media is None:
        raise UnroutableDocument(
            f"formato no reconocido ({len(data)} bytes); "
            f"se esperaba XML, PDF o imagen"
        )

    if media == XML:
        try:
            invoice = parse_cfdi_bytes(data)
        except CfdiParseError as exc:
            raise UnroutableDocument(f"no se pudo parsear: {exc}") from exc
        return RoutedDocument(
            invoice=invoice, media_type=media, tier=0, provider="none"
        )

    # Everything below here costs money or local compute.
    if provider is None:
        cfg = get_config()
        if not cfg.llm_enabled:
            raise UnroutableDocument(
                f"documento {media} requiere el modelo de visión, y no hay "
                f"credenciales configuradas (ANTHROPIC_API_KEY, o "
                f"LLM_PROVIDER=local con LLM_BASE_URL)"
            )

    from cfdi_agent.extract.pdf_vision import extract_from_document

    try:
        invoice, result = extract_from_document(
            data, media_type=media, provider=provider
        )
    except ProviderError as exc:
        raise UnroutableDocument(f"extracción por visión falló: {exc}") from exc

    # The tier reports where the work actually ran, not where it was configured
    # to run. A local backend is tier 1; the API is tier 2. Cost per invoice is
    # computed from these rows, so they have to reflect reality.
    tier = 1 if result.provider == "local" else 2
    return RoutedDocument(
        invoice=invoice,
        media_type=media,
        tier=tier,
        provider=result.provider,
        model=result.model,
        latency_ms=result.latency_ms,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        cost_usd=result.cost_usd,
    )
