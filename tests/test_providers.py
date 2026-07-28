"""Provider layer tests. No API key, no network, no local server.

Both backends are exercised against fakes: a stub Anthropic client and an
httpx mock transport. That covers the parts that are actually ours — request
shape, usage accounting, cost arithmetic, error surfacing, and the guarantee
that both backends produce the same validated type — without pretending to
test Anthropic's servers.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest

from cfdi_agent.extract.providers.anthropic_provider import AnthropicProvider
from cfdi_agent.extract.providers.base import (
    LLMProvider,
    ProviderError,
    UnsupportedMediaError,
    get_provider,
    validate_extraction,
)
from cfdi_agent.extract.providers.openai_compat import (
    OpenAICompatProvider,
    _parse_json_object,
)
from cfdi_agent.extract.providers.pricing import PRICES, estimate_cost
from cfdi_agent.schemas import InvoiceExtraction

EXTRACTION_PAYLOAD = {
    "uuid": "a1b2c3d4-e5f6-4718-9a0b-1c2d3e4f5061",
    "fecha_emision": "2026-03-14T10:22:05",
    "rfc_emisor": "AAA010101AAA",
    "rfc_receptor": "XAXX010101000",
    "subtotal": "1000.00",
    "total": "1160.00",
    "moneda": "MXN",
    "conceptos": [
        {
            "descripcion": "Servicio",
            "cantidad": "1",
            "valor_unitario": "1000.00",
            "importe": "1000.00",
        }
    ],
    "impuestos": [],
}


# --------------------------------------------------------------- pricing


def test_cost_is_computed_from_measured_tokens() -> None:
    # 1M in + 1M out on Opus 5 = $5 + $25.
    cost = estimate_cost("claude-opus-5", tokens_in=1_000_000, tokens_out=1_000_000)
    assert cost == Decimal("30.000000")


def test_cache_reads_are_cheaper_than_fresh_input() -> None:
    fresh = estimate_cost("claude-opus-5", tokens_in=100_000)
    cached = estimate_cost("claude-opus-5", cache_read_tokens=100_000)
    assert cached == fresh / 10


def test_promotional_price_expires() -> None:
    """A table that keeps applying a lapsed intro rate reports precise nonsense."""
    during = estimate_cost(
        "claude-sonnet-5", tokens_in=1_000_000, on=date(2026, 8, 1)
    )
    after = estimate_cost("claude-sonnet-5", tokens_in=1_000_000, on=date(2026, 9, 1))
    assert during == Decimal("2.000000")
    assert after == Decimal("3.000000")
    assert PRICES["claude-sonnet-5"].promo_until == date(2026, 8, 31)


def test_unknown_model_costs_none_not_zero() -> None:
    """'Unpriced' and 'free' must never be conflated in the cost report."""
    assert estimate_cost("some-local-gguf", tokens_in=1000) is None


# ------------------------------------------------------------- anthropic


class _StubMessages:
    def __init__(self, response, recorder: dict) -> None:
        self._response = response
        self._recorder = recorder

    def parse(self, **kwargs):
        self._recorder.update(kwargs)
        return self._response

    def create(self, **kwargs):
        self._recorder.update(kwargs)
        return self._response


def _stub_client(response, recorder: dict):
    return SimpleNamespace(messages=_StubMessages(response, recorder))


def _anthropic_response(**overrides):
    base = {
        "parsed_output": InvoiceExtraction.model_validate(EXTRACTION_PAYLOAD),
        "model": "claude-opus-5",
        "stop_reason": "end_turn",
        "usage": SimpleNamespace(
            input_tokens=2400,
            output_tokens=380,
            cache_read_input_tokens=1800,
            cache_creation_input_tokens=0,
        ),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_anthropic_extract_returns_validated_extraction() -> None:
    recorder: dict = {}
    provider = AnthropicProvider(client=_stub_client(_anthropic_response(), recorder))
    result = provider.extract_invoice(b"%PDF-1.7 fake", media_type="application/pdf")

    assert isinstance(result.content, InvoiceExtraction)
    assert result.content.uuid.lower().startswith("a1b2c3d4")
    assert result.provider == "anthropic"
    assert result.tokens_in == 2400 and result.tokens_out == 380
    assert result.cache_read_tokens == 1800
    assert result.cost_usd is not None and result.cost_usd > 0


def test_anthropic_sends_a_document_block_for_pdf() -> None:
    recorder: dict = {}
    provider = AnthropicProvider(client=_stub_client(_anthropic_response(), recorder))
    provider.extract_invoice(b"%PDF-1.7 fake", media_type="application/pdf")

    content = recorder["messages"][0]["content"]
    assert content[0]["type"] == "document"
    assert content[0]["source"]["media_type"] == "application/pdf"


def test_anthropic_sends_an_image_block_for_png() -> None:
    recorder: dict = {}
    provider = AnthropicProvider(client=_stub_client(_anthropic_response(), recorder))
    provider.extract_invoice(b"\x89PNG fake", media_type="image/png")
    assert recorder["messages"][0]["content"][0]["type"] == "image"


def test_anthropic_marks_the_system_prompt_cacheable() -> None:
    """The system prompt plus schema is the stable prefix on every call."""
    recorder: dict = {}
    provider = AnthropicProvider(client=_stub_client(_anthropic_response(), recorder))
    provider.extract_invoice(b"%PDF fake", media_type="application/pdf")
    assert recorder["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_uses_adaptive_thinking() -> None:
    recorder: dict = {}
    provider = AnthropicProvider(client=_stub_client(_anthropic_response(), recorder))
    provider.extract_invoice(b"%PDF fake", media_type="application/pdf")
    assert recorder["thinking"] == {"type": "adaptive"}


def test_anthropic_surfaces_a_refusal_instead_of_crashing() -> None:
    """Classifiers decline with HTTP 200; reading content blindly hides why."""
    response = _anthropic_response(
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber", explanation="nope"),
    )
    provider = AnthropicProvider(client=_stub_client(response, {}))
    with pytest.raises(ProviderError, match="declined"):
        provider.extract_invoice(b"%PDF fake", media_type="application/pdf")


def test_anthropic_rejects_unsupported_media() -> None:
    provider = AnthropicProvider(client=_stub_client(_anthropic_response(), {}))
    with pytest.raises(UnsupportedMediaError, match="text/csv"):
        provider.extract_invoice(b"a,b,c", media_type="text/csv")


def test_anthropic_does_not_pretend_to_serve_embeddings() -> None:
    provider = AnthropicProvider(client=_stub_client(_anthropic_response(), {}))
    with pytest.raises(ProviderError, match="does not serve embeddings"):
        provider.embed(["papel bond"])


# ------------------------------------------------------------------ local


@pytest.fixture
def mock_httpx(monkeypatch):
    def install(handler):
        transport = httpx.MockTransport(handler)

        def fake_post(url, **kwargs):
            with httpx.Client(transport=transport) as client:
                return client.post(url, **kwargs)

        monkeypatch.setattr(httpx, "post", fake_post)

    return install


def test_local_extract_parses_and_validates(mock_httpx) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5-vl-7b",
                "choices": [
                    {
                        "message": {"content": json.dumps(EXTRACTION_PAYLOAD)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1500, "completion_tokens": 300},
            },
        )

    mock_httpx(handler)
    provider = OpenAICompatProvider(base_url="http://orin:8080/v1", model="qwen2.5-vl-7b")
    result = provider.extract_invoice(b"\x89PNG fake", media_type="image/png")

    assert isinstance(result.content, InvoiceExtraction)
    assert result.provider == "local"
    assert result.tokens_in == 1500
    # Local inference is genuinely free per token, and says so.
    assert result.cost_usd is None


def test_local_refuses_pdf_with_an_actionable_message() -> None:
    provider = OpenAICompatProvider(base_url="http://orin:8080/v1", model="qwen")
    with pytest.raises(UnsupportedMediaError, match="Rasterize"):
        provider.extract_invoice(b"%PDF-1.7", media_type="application/pdf")


def test_local_embeddings_are_reordered_by_index(mock_httpx) -> None:
    """The OpenAI schema does not promise input order; embeddings must not scramble."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.2, 0.2]},
                    {"index": 0, "embedding": [0.1, 0.1]},
                ]
            },
        )

    mock_httpx(handler)
    provider = OpenAICompatProvider(base_url="http://orin:8080/v1", model="q")
    vectors = provider.embed(["primero", "segundo"])
    assert vectors == [[0.1, 0.1], [0.2, 0.2]]


def test_local_embedding_count_mismatch_is_an_error(mock_httpx) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1]}]})

    mock_httpx(handler)
    provider = OpenAICompatProvider(base_url="http://orin:8080/v1", model="q")
    with pytest.raises(ProviderError, match="1 vectors for 2 inputs"):
        provider.embed(["a", "b"])


def test_local_server_error_is_wrapped(mock_httpx) -> None:
    mock_httpx(lambda request: httpx.Response(500, text="out of memory"))
    provider = OpenAICompatProvider(base_url="http://orin:8080/v1", model="q")
    with pytest.raises(ProviderError, match="local inference"):
        provider.complete("s", "u")


def test_local_provider_requires_a_base_url() -> None:
    with pytest.raises(ProviderError, match="LLM_BASE_URL"):
        OpenAICompatProvider(base_url="", model="q")


@pytest.mark.parametrize(
    "text",
    [
        json.dumps(EXTRACTION_PAYLOAD),
        "```json\n" + json.dumps(EXTRACTION_PAYLOAD) + "\n```",
        "Aquí está la factura:\n" + json.dumps(EXTRACTION_PAYLOAD) + "\nEspero sirva.",
    ],
)
def test_json_is_recovered_from_chatty_output(text: str) -> None:
    """Smaller models fence or narrate their JSON even when given a schema."""
    assert _parse_json_object(text)["uuid"].startswith("a1b2")


def test_unparseable_output_is_an_error() -> None:
    with pytest.raises(ProviderError, match="no JSON object"):
        _parse_json_object("lo siento, no puedo leer esta imagen")


# --------------------------------------------------------------- the seam


def test_both_backends_produce_the_same_type() -> None:
    """The point of the ABC: nothing above it can tell the two apart."""
    assert issubclass(AnthropicProvider, LLMProvider)
    assert issubclass(OpenAICompatProvider, LLMProvider)
    parsed = validate_extraction(EXTRACTION_PAYLOAD)
    assert isinstance(parsed, InvoiceExtraction)
    assert isinstance(parsed.to_parsed().total, Decimal)


def test_unknown_provider_name_is_rejected(monkeypatch) -> None:
    from cfdi_agent.config import get_config

    monkeypatch.setenv("LLM_PROVIDER", "gpt-at-home")
    get_config.cache_clear()
    try:
        with pytest.raises(ProviderError, match="unknown LLM_PROVIDER"):
            get_provider()
    finally:
        get_config.cache_clear()


def test_hallucinated_amounts_still_have_to_survive_validation() -> None:
    """A well-typed lie is still caught downstream, by arithmetic.

    This is the guarantee that makes using a model here acceptable at all.
    """
    from cfdi_agent.validate.rules import validate_invoice

    payload = dict(EXTRACTION_PAYLOAD, total="9999.00")
    inv = validate_extraction(payload).to_parsed()
    result = validate_invoice(inv, company_rfc="XAXX010101000")
    assert any(a.kind == "total_mismatch" for a in result.anomalies)
