"""Tier 1: a local model behind an OpenAI-compatible server.

Targets `llama-server` from llama.cpp on the AGX Orin 64 GB, but works against
anything speaking `/v1/chat/completions` and `/v1/embeddings` — Ollama, vLLM, a
second machine on the LAN.

The point is not that local is better. It is that with one interface and two
backends, "does this route need a frontier model?" becomes a measurement
instead of an opinion. Some routes plainly do not: embeddings for line-item
similarity are a solved problem at 1 GB.

Two honest limits, both surfaced as errors rather than silent degradation:

*No PDF support.* Claude reads `application/pdf` natively; a local VLM takes
images. Rasterizing here would mean a PDF renderer dependency in the hot path,
so a PDF sent to this backend raises `UnsupportedMediaError` naming the fix
rather than half-working.

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

    # ------------------------------------------------------------------ api

    def extract_invoice(self, data: bytes, *, media_type: str) -> LLMResult:
        if media_type == "application/pdf":
            raise UnsupportedMediaError(
                "the local backend takes images, not PDF. Rasterize first "
                "(pypdfium2 or pdftoppm) and pass image/png pages, or route "
                "PDFs to the anthropic provider."
            )
        self.check_media(media_type)

        encoded = base64.standard_b64encode(data).decode("ascii")
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                        },
                        {"type": "text", "text": EXTRACTION_PROMPT},
                    ],
                },
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

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> LLMResult:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
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
                timeout=self.timeout,
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
        except httpx.HTTPError as exc:
            raise ProviderError(f"local inference at {url} failed: {exc}") from exc
        return body, int((time.perf_counter() - started) * 1000)

    @staticmethod
    def _first_message(body: dict) -> str:
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {str(body)[:200]}") from exc

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
