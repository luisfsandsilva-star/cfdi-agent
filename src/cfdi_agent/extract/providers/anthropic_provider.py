"""Tier 2: Claude via the Anthropic API.

Reached only by invoices that arrive without their XML — a scanned PDF, a
photo of a printed invoice. Everything with an XML attachment is handled by the
deterministic parser for free.

Three choices worth stating:

*Structured outputs, not prompt-and-parse.* `messages.parse` with a Pydantic
schema means the response is validated before it is returned. `InvoiceExtraction`
is deliberately constraint-free (see `schemas`) because structured outputs
reject `Decimal` and numeric constraints, and amounts cross as strings so money
never round-trips through binary floating point.

*The system prompt is cached.* It plus the schema is a stable prefix on every
call; Claude Opus 5's minimum cacheable prefix is 512 tokens, which this clears.
Cache reads bill at roughly a tenth of the input rate.

*Validation still owns the verdict.* A hallucinated total arrives here as a
well-typed string and then fails `validate.rules` arithmetic, landing the
document in the review queue. The model is never trusted, only used.
"""

from __future__ import annotations

import base64
import time
from typing import Any

from cfdi_agent.extract.providers.base import (
    LLMProvider,
    LLMResult,
    ProviderError,
    validate_extraction,
)
from cfdi_agent.extract.providers.pricing import estimate_cost
from cfdi_agent.schemas import InvoiceExtraction

DEFAULT_MODEL = "claude-opus-5"

EXTRACTION_SYSTEM = """\
Eres un asistente de cuentas por pagar. Transcribes facturas mexicanas (CFDI) \
a datos estructurados.

Reglas:
- Transcribe únicamente lo que está impreso en el documento. No calcules, no \
completes, no corrijas.
- Si un campo no aparece, déjalo nulo. Un campo faltante es información; un \
campo inventado es un error que se propaga a la contabilidad.
- Los montos van como cadenas decimales sin símbolo de moneda ni separadores \
de miles: "1234.56", no "$1,234.56".
- El UUID está en el Timbre Fiscal Digital, con guiones.
- Copia los importes exactamente como aparecen, aunque no cuadren entre sí. \
Otro sistema verifica la aritmética; tu trabajo es la transcripción fiel.
"""

EXTRACTION_PROMPT = (
    "Transcribe esta factura CFDI. Incluye todas las líneas de conceptos."
)


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    supported_media = frozenset(
        {"application/pdf", "image/png", "image/jpeg", "image/webp", "image/gif"}
    )

    def __init__(self, model: str = DEFAULT_MODEL, *, client: Any = None) -> None:
        self.model = model
        self._client = client

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise ProviderError(
                    "the `anthropic` package is not installed: pip install -e '.[llm]'"
                ) from exc
            # Zero-arg constructor on purpose: it resolves ANTHROPIC_API_KEY, an
            # ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile. Reading the
            # key ourselves would break the profile path for no benefit.
            self._client = anthropic.Anthropic()
        return self._client

    # ------------------------------------------------------------------ api

    def extract_invoice(self, data: bytes, *, media_type: str) -> LLMResult:
        self.check_media(media_type)
        encoded = base64.standard_b64encode(data).decode("ascii")
        block_type = "document" if media_type == "application/pdf" else "image"

        started = time.perf_counter()
        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=8192,
                # Adaptive lets Claude decide how much to think per document; a
                # clean one-page invoice should not cost what a blurry scan does.
                thinking={"type": "adaptive"},
                system=[
                    {
                        "type": "text",
                        "text": EXTRACTION_SYSTEM,
                        # Stable across every call — the whole point of a cache
                        # breakpoint. Volatile content goes after it.
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": block_type,
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": encoded,
                                },
                            },
                            {"type": "text", "text": EXTRACTION_PROMPT},
                        ],
                    }
                ],
                output_format=InvoiceExtraction,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as-is
            raise ProviderError(f"anthropic extract failed: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        # Safety classifiers can decline with HTTP 200. Reading content[0]
        # unconditionally would raise something unrelated and hide the reason.
        if getattr(response, "stop_reason", None) == "refusal":
            raise ProviderError(
                f"model declined the request (stop_reason=refusal, "
                f"category={getattr(response.stop_details, 'category', None)})"
            )

        return self._to_result(
            validate_extraction(response.parsed_output), response, latency_ms
        )

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> LLMResult:
        started = time.perf_counter()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                thinking={"type": "adaptive"},
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"anthropic complete failed: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        if getattr(response, "stop_reason", None) == "refusal":
            raise ProviderError("model declined the request (stop_reason=refusal)")

        text = "".join(b.text for b in response.content if b.type == "text")
        return self._to_result(text, response, latency_ms)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Not offered by the Anthropic API.

        Embeddings run locally (bge-m3) regardless of which backend handles
        extraction — they are cheap, multilingual, and there is no reason to
        send line-item descriptions over the network to compute them.
        """
        raise ProviderError(
            "the Anthropic API does not serve embeddings; use the local "
            "embedding backend (EMBED_BASE_URL) for enrich.embeddings"
        )

    # -------------------------------------------------------------- helpers

    def _to_result(self, content: Any, response: Any, latency_ms: int) -> LLMResult:
        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "input_tokens", 0) or 0
        tokens_out = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        return LLMResult(
            content=content,
            provider=self.name,
            model=getattr(response, "model", self.model),
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            cost_usd=estimate_cost(
                self.model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
            ),
        )
