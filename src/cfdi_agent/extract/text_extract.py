"""Extract an invoice from a PDF's text layer rather than from its picture.

The measurement that produced this module: on 93 real PDF/XML pairs, the text
layer contains 551 of 558 checked fields (99%) in under 0.1 seconds, against
466 (83%) at 10 seconds for the best local vision model. Both RFCs, the
subtotal and the total are at 100%.

So the text layer does not replace the model — **it replaces the image.** The
model still maps text to fields, which is a real task with no deterministic
answer across every PAC's layout. What disappears is the transcription step,
and with it the failure mode that step introduced:

*   A vision model can read `6,863.40` as `6,863.48`. Reading the text layer
    cannot: the digits are in the file.
*   A rasterized page at 200 DPI costs ~4,200 image tokens. The same invoice's
    text is ~800 tokens. The cheaper input is also the more accurate one.
*   Any text model will do. No VLM, no mmproj, no vision encoder — which is
    what makes this path viable on hardware where a 7B VLM does not fit.

`pdf_text.has_text_layer` decides between this and the vision path. A scan or a
photo yields almost no characters and goes to `pdf_vision`, which is what that
module is for.

One field is taken before the model sees anything. The UUID is recovered by
regex when the text layer has it, because it is the field a model is likeliest
to corrupt — a long hex string with no linguistic redundancy to fall back on —
and the field that decides *which invoice* a document is. Letting a model
re-type a value already present verbatim is the exact mistake this project
argues against.
"""

from __future__ import annotations

import time

from cfdi_agent.extract.pdf_text import extract_text_layer, find_uuid
from cfdi_agent.extract.providers.base import (
    LLMProvider,
    LLMResult,
    ProviderError,
    get_provider,
    validate_extraction,
)
from cfdi_agent.extract.providers.openai_compat import _parse_json_object
from cfdi_agent.schemas import InvoiceExtraction, ParsedInvoice

# Enough for a long multi-page invoice's text without being able to blow up a
# context window. ~800 tokens is typical for a single page; this is generous.
MAX_TEXT_CHARS = 40_000

# Headroom for a reasoning model, not for the answer. A CFDI's JSON is ~2,600
# characters; qwen3:4b spends ~10,000 tokens reaching it, almost all of them
# deliberating over what is a copy. At 4,096 every one of 95 real invoices came
# back empty with `finish_reason: length`.
#
# The cost of that reasoning is real and not tunable on Ollama (see
# `OpenAICompatProvider.complete`), which makes a *non-reasoning* model the
# right tool for this path rather than a bigger budget.
MAX_OUTPUT_TOKENS = 16384

TEXT_EXTRACTION_PROMPT = """\
A continuación está el texto extraído de una factura CFDI en PDF. El texto es \
exacto: está tomado de la capa de texto del archivo, no de un OCR. No corrijas \
ni recalcules ningún valor — transcríbelos tal como aparecen.

El orden de lectura puede estar mezclado, porque el PDF posiciona el texto por \
coordenadas. Las etiquetas y sus valores pueden aparecer separados.

--- TEXTO ---
{text}
--- FIN ---

Devuelve el JSON de la factura, con todas las líneas de conceptos."""


def extract_from_text_layer(
    data: bytes, *, provider: LLMProvider | None = None
) -> tuple[ParsedInvoice, LLMResult]:
    """Read a PDF's text layer and map it to the canonical model.

    Returns the invoice and the raw `LLMResult`, like `pdf_vision`, because the
    caller writes latency, tokens and cost to `extraction_runs`.
    """
    from cfdi_agent.extract.providers.anthropic_provider import EXTRACTION_SYSTEM

    text = extract_text_layer(data)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]

    provider = provider or get_provider()
    started = time.perf_counter()
    prompt = TEXT_EXTRACTION_PROMPT.format(text=text)
    try:
        # The values are in the text. Deliberating about them costs tokens and
        # buys nothing; see `OpenAICompatProvider.complete`.
        result = provider.complete(
            EXTRACTION_SYSTEM,
            prompt,
            max_tokens=MAX_OUTPUT_TOKENS,
            thinking=False,
            # The field names are not guessable and a small model does not
            # guess them: without the schema a 3B answered in invented keys.
            json_schema=InvoiceExtraction.model_json_schema(),
        )
    except TypeError:
        # A provider that does not accept the hint. The Anthropic path manages
        # its own thinking budget and has no such parameter.
        result = provider.complete(EXTRACTION_SYSTEM, prompt, max_tokens=MAX_OUTPUT_TOKENS)

    try:
        extraction = validate_extraction(_parse_json_object(result.content))
    except ProviderError:
        raise
    except Exception as exc:  # noqa: BLE001 - a bad answer is a normal outcome
        raise ProviderError(
            f"model returned no usable invoice for the text layer: {exc}"
        ) from exc

    # The UUID from the file wins over the UUID from the model. They should
    # agree; when they do not, the file is right, because the model retyped a
    # string it was handed.
    printed_uuid = find_uuid(text)
    if printed_uuid and printed_uuid != extraction.uuid.upper():
        extraction = extraction.model_copy(update={"uuid": printed_uuid})

    try:
        invoice = extraction.to_parsed()
    except Exception as exc:  # noqa: BLE001
        raise ProviderError(
            f"model returned a document that could not be interpreted as a CFDI: {exc}"
        ) from exc

    if result.latency_ms == 0:
        object.__setattr__(
            result, "latency_ms", int((time.perf_counter() - started) * 1000)
        )
    return invoice, result
