"""The agent against a local model, over the OpenAI tools API.

Two things are worth testing here and neither needs a model running.

The first is drift. There are now two definitions of the same five tools —
`@beta_tool` decorators for the Anthropic SDK and JSON schemas for the OpenAI
shape — and a change to one that never reaches the other produces an agent
that can quietly do less on one path than the other. That is a failure no
runtime error announces.

The second is the message protocol. The OpenAI shape makes the caller own the
loop: the assistant message has to be appended *with* its `tool_calls`, then
one `role: "tool"` message per call keyed by `tool_call_id`. Getting that wrong
makes a model stop using tools rather than raise, which is exactly the kind of
bug that survives manual testing because the answer still looks plausible.
"""

from __future__ import annotations

import contextlib
import json

import httpx
import pytest

from cfdi_agent.agent.local_loop import MAX_ITERATIONS, _dispatch, tool_specs


class TestToolParity:
    def test_both_paths_expose_the_same_tools(self) -> None:
        """Anthropic and local must offer the same five names.

        Read from the Anthropic path's own definitions rather than a hardcoded
        list, so adding a tool to one side and not the other fails here.
        """
        from cfdi_agent.agent.loop import build_tools

        anthropic_names = {t.name for t in build_tools(conn=None)}
        local_names = {s["function"]["name"] for s in tool_specs()}
        assert local_names == anthropic_names

    def test_every_spec_is_a_usable_schema(self) -> None:
        for spec in tool_specs():
            function = spec["function"]
            assert spec["type"] == "function"
            assert function["description"].strip()
            params = function["parameters"]
            assert params["type"] == "object"
            # Every declared required argument must exist in properties, or the
            # server rejects the request and the agent has no tools at all.
            for name in params["required"]:
                assert name in params["properties"], (function["name"], name)

    def test_the_sql_tool_documents_the_views(self) -> None:
        """The description is the model's only schema documentation. Without the
        column names it guesses, writes failing SQL, and burns iterations."""
        sql = next(
            s for s in tool_specs() if s["function"]["name"] == "consultar_sql"
        )
        description = sql["function"]["description"]
        for view in ("v_invoices", "v_line_items", "v_anomalies"):
            assert view in description
        assert "solo lectura" in description


class TestDispatch:
    def test_an_unknown_tool_is_answered_not_raised(self) -> None:
        """The model chose the name. A tool result saying it is wrong lets it
        correct itself; raising ends the conversation over a typo."""
        out = json.loads(_dispatch(None, "herramienta_inventada", {}))
        assert "desconocida" in out["error"]

    def test_a_rejected_query_comes_back_as_a_tool_result(self) -> None:
        """The read-only boundary lives in tools.py and raises. That rejection
        is information the model should see and work around, not a crash."""
        out = json.loads(_dispatch(None, "consultar_sql", {"sql": "DROP TABLE x"}))
        assert "error" in out

    def test_a_missing_argument_is_reported_as_an_error(self) -> None:
        out = json.loads(_dispatch(None, "detalle_factura", {}))
        assert "KeyError" in out["error"]


class TestMessageProtocol:
    """Drives `ask_local` against a stub server to pin the wire protocol."""

    def _run(self, monkeypatch, responses: list[dict], question="¿cuánto gasté?"):
        from cfdi_agent.agent import local_loop

        sent: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(200, json=responses[len(sent) - 1])

        transport = httpx.MockTransport(handler)
        real_post = httpx.post

        def post(url, **kwargs):
            with httpx.Client(transport=transport) as client:
                return client.post(url, **kwargs)

        monkeypatch.setattr(httpx, "post", post)
        monkeypatch.setattr("cfdi_agent.db.conn.connect", _no_database)
        try:
            answer = local_loop.ask_local(question)
        finally:
            monkeypatch.setattr(httpx, "post", real_post)
        return answer, sent

    def test_a_plain_answer_ends_the_loop(self, monkeypatch) -> None:
        answer, sent = self._run(
            monkeypatch,
            [{"choices": [{"message": {"content": "Gastaste 1000 pesos."}}]}],
        )
        assert answer == "Gastaste 1000 pesos."
        assert len(sent) == 1
        assert sent[0]["tools"]

    def test_a_tool_call_is_executed_and_fed_back(self, monkeypatch) -> None:
        """The assistant message must be replayed with its tool_calls, and the
        result must carry the matching tool_call_id."""
        answer, sent = self._run(
            monkeypatch,
            [
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "herramienta_inventada",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
                {"choices": [{"message": {"content": "Listo."}}]},
            ],
        )
        assert answer == "Listo."
        second = sent[1]["messages"]
        assert second[-2]["tool_calls"][0]["id"] == "call_1"
        assert second[-1]["role"] == "tool"
        assert second[-1]["tool_call_id"] == "call_1"

    def test_malformed_arguments_do_not_crash_the_loop(self, monkeypatch) -> None:
        """A small model emits invalid JSON arguments. The tool should report a
        problem the model can react to, not end the run."""
        answer, _ = self._run(
            monkeypatch,
            [
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "c",
                                        "function": {
                                            "name": "detalle_factura",
                                            "arguments": "{not json",
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
                {"choices": [{"message": {"content": "Corregido."}}]},
            ],
        )
        assert answer == "Corregido."

    def test_an_endless_tool_loop_is_capped(self, monkeypatch) -> None:
        """A loop that will not terminate is worse than a wrong answer, and a
        small model is likelier to circle."""
        forever = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "c",
                                "function": {"name": "x", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        }
        answer, sent = self._run(monkeypatch, [forever] * (MAX_ITERATIONS + 2))
        assert str(MAX_ITERATIONS) in answer
        assert len(sent) == MAX_ITERATIONS

    def test_an_unreachable_host_is_reported_not_raised(self, monkeypatch) -> None:
        from cfdi_agent.agent import local_loop

        def boom(url, **kwargs):
            raise httpx.ConnectError("no route to host")

        monkeypatch.setattr(httpx, "post", boom)
        monkeypatch.setattr("cfdi_agent.db.conn.connect", _no_database)
        assert "falló" in local_loop.ask_local("hola")


@contextlib.contextmanager
def _no_database():
    """These tests pin the wire protocol, not the queries. No tool that needs a
    connection is reached: the stub server only ever names tools that fail
    before touching one."""
    yield None


@pytest.fixture(autouse=True)
def _local_provider(monkeypatch):
    """The loop reads LLM_BASE_URL and the model name from config."""
    from cfdi_agent.config import get_config

    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen3:4b")
    get_config.cache_clear()
    yield
    get_config.cache_clear()
