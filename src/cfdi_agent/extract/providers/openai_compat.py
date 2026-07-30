"""Tier 1: a local model behind an OpenAI-compatible server.

Targets `llama-server` from llama.cpp on the AGX Orin 64 GB, but works against
anything speaking `/v1/chat/completions` and `/v1/embeddings` — Ollama, vLLM, a
second machine on the LAN.

The point is not that local is better. It is that with one interface and two
backends, "does this route need a frontier model?" becomes a measurement
instead of an opinion. Some routes plainly do not: embeddings for line-item
similarity are a solved problem at 1 GB.

*PDF is rasterized here, not upstream.* Claude reads `application/pdf`
natively; a local VLM takes images. That difference is a property of this
backend, so the adaptation belongs at this boundary — nothing above it should
have to know which tier can read which container. `pypdfium2` is an optional
extra (`pip install -e '.[local]'`); without it a PDF still raises
`UnsupportedMediaError` naming the install rather than half-working.

*Structured output is a request, not a guarantee.* llama.cpp honours a JSON
schema via `response_format`, but coverage varies by build and model. The
response is parsed defensively and validated by the same `InvoiceExtraction`
schema the API path uses, so a malformed answer fails here rather than
downstream.

Decode speed on the Orin is bound by memory bandwidth (204.8 GB/s), not
compute. A dense 14B at Q4 reads ~9 GB per token and lands around 10-16 tok/s;
a 30B MoE with ~3B active reads far less and goes several times faster. Those
are derivations, not measurements — the harness replaces them with real numbers
once the hardware is free.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from typing import Any

import httpx

from cfdi_agent.extract.providers.anthropic_provider import (
    EXTRACTION_PROMPT,
    EXTRACTION_SYSTEM,
)
from cfdi_agent.extract.providers.base import (
    LLMProvider,
    LLMResult,
    ProviderError,
    UnsupportedMediaError,
    validate_extraction,
)
from cfdi_agent.schemas import InvoiceExtraction

# Local inference costs no tokens. Zero here is a fact, not a missing price —
# `estimate_cost` returns None for the unknown case, and the two must not be
# confused when the harness reports cost per invoice.
LOCAL_COST = None

# Generous for a batch of descriptions, short enough that an unreachable host
# fails fast instead of stalling ingest.
EMBED_TIMEOUT = 15.0

# Rasterization settings for the PDF path.
#
# Swept, because the first version of this comment asserted that 200 was "the
# lowest that keeps the smallest print legible" and that was a guess.
#
# Measured on four real supplier invoices with qwen2.5vl:3b, scored against the
# deterministic parse of each invoice's own XML:
#
#     150 DPI    one document failed to extract at all    44 s
#     200 DPI    all four extracted                       68 s
#     250 DPI    all four extracted                       76 s
#
# Field accuracy did not improve with resolution — `total` was wrong in three
# of four at every setting. Resolution is not the binding constraint; model
# capacity is. 200 stays the default as the cheapest point where every document
# produced an answer, not because it fixes anything.
RASTER_DPI = 200
# A CFDI is nearly always one page and the fields this extracts live on the
# first. The cap bounds the work a malformed or padded PDF can cause; pages
# past it are dropped, and the drop is reported in the result rather than
# passed over.
MAX_PAGES = 3


class OpenAICompatProvider(LLMProvider):
    name = "local"
    supported_media = frozenset({"image/png", "image/jpeg", "image/webp"})

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        embed_base_url: str = "",
        embed_model: str = "bge-m3",
        timeout: float = 300.0,
        raster_dpi: int = RASTER_DPI,
    ) -> None:
        if not base_url:
            raise ProviderError("LLM_BASE_URL is required for the local provider")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_base_url = (embed_base_url or base_url).rstrip("/")
        self.embed_model = embed_model
        # Generous: a 7B VLM on Jetson-class hardware takes seconds per page,
        # and a timeout that fires mid-generation looks like a model failure.
        self.timeout = timeout
        # Per-instance rather than a module constant the caller reaches in and
        # rewrites: image tokens scale with the square of this, so it is the
        # first knob a harness wants to sweep.
        self.raster_dpi = raster_dpi

    # ------------------------------------------------------------------ api

    def extract_invoice(self, data: bytes, *, media_type: str) -> LLMResult:
        if media_type == "application/pdf":
            pages = rasterize_pdf(data, dpi=self.raster_dpi)
            images = [("image/png", page) for page in pages]
        else:
            self.check_media(media_type)
            images = [(media_type, data)]

        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mt};base64,"
                    + base64.standard_b64encode(blob).decode("ascii")
                },
            }
            for mt, blob in images
        ]
        # Text after the images. A VLM conditions its answer on what it has
        # already read, and putting the instruction last is what makes the
        # schema apply to the page rather than the page interrupt the schema.
        content.append({"type": "text", "text": EXTRACTION_PROMPT})

        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "invoice_extraction",
                    "schema": InvoiceExtraction.model_json_schema(),
                },
            },
        }
        body, latency_ms = self._post("/chat/completions", payload)
        text = self._first_message(body)
        return self._to_result(
            validate_extraction(_parse_json_object(text)), body, latency_ms
        )

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        thinking: bool = True,
        json_schema: dict | None = None,
    ) -> LLMResult:
        """One completion. `thinking=False` asks the server to skip reasoning.

        A reasoning model spends its budget before it answers, and qwen3 spends
        ~3,800 tokens per invoice deliberating over what is, at bottom, a
        copy: the values are already in the text handed to it. So the request
        carries the standard hint.

        **Measured caveat: Ollama ignores it.** Four ways of asking were tried
        against `qwen3:4b` on Ollama's OpenAI-compatible endpoint, and all four
        produced the same token count:

            baseline            346
            think: false        346
            /no_think in prompt 337
            reasoning_effort    346

        So on that server the reasoning cost is not optional, and the ~40 s per
        invoice it produces is a real property of that configuration rather
        than a setting left wrong. vLLM and llama.cpp do honour
        `chat_template_kwargs`; the hint is kept for them, and a server that
        ignores it degrades to the same answer more slowly, which is the right
        failure mode for a hint.

        The agent loop leaves reasoning on deliberately, since chaining queries
        is the whole job there.
        """
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if not thinking:
            # Ollama and vLLM both read this; a server that does not simply
            # ignores it, which is the right failure mode for a hint.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if json_schema is not None:
            # Without this the model invents field names. A 3B handed the same
            # prompt and no schema answered with `NombreEmisor` and
            # `RfcReceptorCFDI` -- plausible Spanish, wrong keys, 31 validation
            # errors. `extract_invoice` always constrained its output; the text
            # path was built on `complete`, which did not, and inherited a bug
            # the vision path never had.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "invoice_extraction", "schema": json_schema},
            }
        body, latency_ms = self._post("/chat/completions", payload)
        return self._to_result(self._first_message(body), body, latency_ms)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        url = f"{self.embed_base_url}/embeddings"
        try:
            resp = httpx.post(
                url,
                json={"model": self.embed_model, "input": texts},
                # Not `self.timeout`: that budget exists for a vision model
                # generating tokens. An embedding call is milliseconds of
                # compute, so a long wait here only means the host is gone.
                timeout=EMBED_TIMEOUT,
            )
            resp.raise_for_status()
            body = resp.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"embedding request to {url} failed: {exc}") from exc

        rows = body.get("data") or []
        if len(rows) != len(texts):
            raise ProviderError(
                f"embedding server returned {len(rows)} vectors for {len(texts)} inputs"
            )
        # Sort by index: the OpenAI schema does not promise input order.
        rows.sort(key=lambda r: r.get("index", 0))
        return [r["embedding"] for r in rows]

    # -------------------------------------------------------------- helpers

    def _post(self, path: str, payload: dict) -> tuple[dict, int]:
        url = f"{self.base_url}{path}"
        started = time.perf_counter()
        try:
            resp = httpx.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            body = resp.json()
        except httpx.HTTPStatusError as exc:
            # The status line alone is close to useless here. A local server
            # explains itself in the body — "exceeds the available context
            # size", "cudaMalloc failed: out of memory" — and those two have
            # completely different fixes. Dropping the body cost an hour of
            # bisecting by hand.
            raise ProviderError(
                f"local inference at {url} failed: {exc.response.status_code} "
                f"{_error_message(exc.response)}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"local inference at {url} failed: {exc}") from exc
        return body, int((time.perf_counter() - started) * 1000)

    @staticmethod
    def _first_message(body: dict) -> str:
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {str(body)[:200]}") from exc

        # Empty content with `finish_reason: length` is a token budget that ran
        # out, and saying so is the difference between a one-line fix and an
        # afternoon. A reasoning model makes this the *common* failure: qwen3
        # spent all 4,096 tokens deliberating and never began the answer, and
        # the error read "no JSON object in model output: ''" -- a description
        # of the symptom that points nowhere near the cause.
        if not content.strip() and choice.get("finish_reason") == "length":
            used = (body.get("usage") or {}).get("completion_tokens", "?")
            raise ProviderError(
                f"the model used its whole {used}-token budget without "
                f"producing an answer. Raise max_tokens; a reasoning model "
                f"needs room for the reasoning *and* the reply."
            )
        return content

    def _to_result(self, content: Any, body: dict, latency_ms: int) -> LLMResult:
        usage = body.get("usage") or {}
        return LLMResult(
            content=content,
            provider=self.name,
            model=body.get("model") or self.model,
            latency_ms=latency_ms,
            tokens_in=usage.get("prompt_tokens", 0) or 0,
            tokens_out=usage.get("completion_tokens", 0) or 0,
            cost_usd=LOCAL_COST,
            raw={"finish_reason": (body.get("choices") or [{}])[0].get("finish_reason")},
        )


def rasterize_pdf(
    data: bytes, *, dpi: int = RASTER_DPI, max_pages: int = MAX_PAGES
) -> list[bytes]:
    """Render a PDF to PNG pages for a backend that cannot read PDF.

    Kept a module-level function rather than a method so the eval harness can
    measure rasterization separately from inference. Measured on this laptop:
    115-187 ms and ~600 KB of PNG per page, against seconds of decode. That
    ratio is the answer to whether it belongs in the hot path.
    """
    try:
        import pypdfium2
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise UnsupportedMediaError(
            "the local backend takes images, not PDF, and the rasterizer is "
            "not installed. Run `pip install -e '.[local]'`, or route PDFs to "
            "the anthropic provider."
        ) from exc

    try:
        pdf = pypdfium2.PdfDocument(data)
    except Exception as exc:  # noqa: BLE001 - pdfium raises its own types
        raise ProviderError(f"could not open PDF: {exc}") from exc

    try:
        pages: list[bytes] = []
        # pdfium's scale is relative to 72 dpi, its own base unit.
        scale = dpi / 72
        for index in range(min(len(pdf), max_pages)):
            page = pdf[index]
            image = page.render(scale=scale).to_pil()
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            pages.append(buffer.getvalue())
        return pages
    finally:
        pdf.close()


def _error_message(response: httpx.Response, limit: int = 300) -> str:
    """Pull the useful sentence out of an error body.

    Ollama, llama.cpp and vLLM all nest it differently, and all three fall back
    to plain text on some paths.
    """
    try:
        body = response.json()
    except ValueError:
        return response.text[:limit].strip()
    error = body.get("error", body) if isinstance(body, dict) else body
    if isinstance(error, dict):
        error = error.get("message", error)
    return str(error)[:limit].strip()


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_object(text: str) -> dict:
    """Pull a JSON object out of a completion.

    Smaller models wrap JSON in prose or a ```json fence even when handed a
    schema. Trying the whole string first and falling back to the outermost
    braces recovers those without pretending the model complied.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ProviderError(f"no JSON object in model output: {text[:200]!r}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ProviderError(f"malformed JSON in model output: {exc}") from exc
