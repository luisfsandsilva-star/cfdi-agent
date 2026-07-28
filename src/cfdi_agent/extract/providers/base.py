"""The seam between the API path and the local path.

One interface, two backends, one harness that compares them. That is cheaper to
build than two implementations and produces something neither gives you alone:
a measured answer to "which routes actually need a frontier model?".

Concretely, this is what makes the AGX Orin a configuration change rather than
a rewrite for the extraction path:

    LLM_PROVIDER=local LLM_BASE_URL=http://orin.local:8080/v1

Where the seam does *not* reach, stated plainly because the README used to
overclaim it: `agent/loop.py` imports the Anthropic SDK directly and is not
routed through `get_provider`. It uses the SDK's tool runner, and an
OpenAI-compatible server has no equivalent — that server exposes a `tools`
parameter on `/chat/completions` and leaves the caller to drive the loop.
Sending the agent to a local model means writing that loop in
`openai_compat.py` first. Until then the seam covers document extraction and
embeddings only.

Every call returns an `LLMResult` carrying latency, tokens and cost, which the
caller writes to `extraction_runs`. Usage accounting is not optional bookkeeping
here — it is the deliverable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from cfdi_agent.schemas import InvoiceExtraction


class ProviderError(RuntimeError):
    """The backend could not be reached, or refused the request."""


class UnsupportedMediaError(ProviderError):
    """This backend cannot read this kind of document."""


@dataclass(frozen=True, slots=True)
class LLMResult:
    """One call's output plus everything the eval harness needs to score it."""

    content: Any
    provider: str
    model: str
    latency_ms: int
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    # None means "no price known", which is not the same as free. A locally
    # hosted model is genuinely 0; an unpriced model id is unknown.
    cost_usd: Decimal | None = None
    raw: dict = field(default_factory=dict)


class LLMProvider(ABC):
    """Minimum surface the pipeline needs. Deliberately small.

    Kept to three verbs so a second backend stays a weekend's work rather than
    a port. Anything richer belongs in the caller.
    """

    name: str = "base"
    model: str = ""
    #: Media types this backend can read directly.
    supported_media: frozenset[str] = frozenset()

    @abstractmethod
    def extract_invoice(self, data: bytes, *, media_type: str) -> LLMResult:
        """Transcribe a document into an `InvoiceExtraction`.

        Only used for invoices that arrive without XML. A CFDI with its XML
        never reaches this method — parsing it deterministically is free and
        cannot hallucinate.
        """

    @abstractmethod
    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> LLMResult:
        """Plain text in, plain text out. Used for anomaly explanations."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Vectors for line-item descriptions, for the similarity detector."""

    def check_media(self, media_type: str) -> None:
        if media_type not in self.supported_media:
            raise UnsupportedMediaError(
                f"{self.name} cannot read {media_type!r}; supports "
                f"{sorted(self.supported_media)}"
            )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} model={self.model!r}>"


def validate_extraction(payload: Any) -> InvoiceExtraction:
    """Coerce whatever the model returned into the extraction schema.

    A separate function so both backends validate identically — and so the
    boundary is explicit: past this point the data is typed, and it still has
    to survive the deterministic rules before it can reach the database.
    """
    if isinstance(payload, InvoiceExtraction):
        return payload
    if isinstance(payload, dict):
        return InvoiceExtraction.model_validate(payload)
    raise ProviderError(f"cannot interpret model output of type {type(payload).__name__}")


def get_provider(
    provider: str | None = None, model: str | None = None
) -> LLMProvider:
    """Build the configured backend. Imports lazily so neither SDK is required."""
    from cfdi_agent.config import get_config

    cfg = get_config()
    kind = (provider or cfg.llm_provider).lower()

    if kind == "anthropic":
        from cfdi_agent.extract.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=model or cfg.llm_model)
    if kind == "local":
        from cfdi_agent.extract.providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(
            base_url=cfg.llm_base_url,
            model=model or cfg.llm_model,
            embed_base_url=cfg.embed_base_url,
            embed_model=cfg.embed_model,
        )
    raise ProviderError(f"unknown LLM_PROVIDER {kind!r}; expected 'anthropic' or 'local'")
