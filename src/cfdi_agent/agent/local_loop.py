"""The agent loop against a local model, over the OpenAI tools API.

The provider seam covered extraction from the start and stopped short of the
agent: `openai_compat` could complete and embed, but the agent loop only ever
ran through `anthropic.beta.messages.tool_runner`. So "does this need a
frontier model?" was answerable for reading an invoice and not for reasoning
over the ledger — which is the more interesting half of the question.

This closes it. Same tools, same read-only SQL boundary, same iteration cap,
driven by whatever speaks `/v1/chat/completions` with tool calling. Ollama with
`qwen3` does; so does llama.cpp's server and vLLM.

Why a separate module rather than a branch inside `loop.py`: the two APIs
disagree on more than field names. Anthropic's SDK runs the loop for you and
hands back messages; the OpenAI shape makes the caller own the cycle — append
the assistant message *with* its `tool_calls`, then one `role: "tool"` message
per call, keyed by `tool_call_id`. Getting that wrong produces a model that
silently stops using tools rather than an error.

The risk this introduces is drift: two definitions of the same five tools, and
a change to one that never reaches the other. `tests/test_local_agent.py`
asserts both paths expose the same names, so drift fails a test instead of
quietly shipping an agent that can do less than the other one.

A small local model will chain fewer queries and write worse SQL than a
frontier model. That is the measurement, not a defect to hide: the point of the
seam is that the comparison is a number rather than an opinion.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import httpx

from cfdi_agent.agent import tools
from cfdi_agent.config import get_config

# Same cap as the Anthropic path. A loop that will not terminate is worse than
# a wrong answer, and a small model is likelier to circle.
MAX_ITERATIONS = 12

# Long enough for a 4B model to think through a multi-step question on CPU or a
# small GPU, short enough that a dead host is obvious.
TIMEOUT = 300.0


def tool_specs() -> list[dict[str, Any]]:
    """The five tools in OpenAI function-calling form.

    Descriptions are the model's only documentation of the schema, so the view
    columns are spelled out here exactly as they are in the Anthropic path. A
    model that has to guess a column name writes SQL that fails, retries, and
    burns iterations.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "consultar_sql",
                "description": (
                    "Ejecuta un SELECT de solo lectura sobre las vistas de "
                    "reporte. Una sola sentencia, sin punto y coma, sin "
                    "escrituras.\n"
                    "v_invoices(uuid, serie, folio, fecha_emision, rfc_emisor, "
                    "proveedor, rfc_receptor, subtotal, descuento, total, "
                    "moneda, uso_cfdi)\n"
                    "v_line_items(invoice_uuid, fecha_emision, rfc_emisor, "
                    "line_no, clave_prod_serv, descripcion, cantidad, "
                    "valor_unitario, importe, category)\n"
                    "v_anomalies(id, invoice_uuid, rfc_emisor, total, kind, "
                    "severity, detail, explanation, resolved, created_at)"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "Un SELECT."}
                    },
                    "required": ["sql"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "detalle_factura",
                "description": "Una factura con sus conceptos y hallazgos.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "uuid": {
                            "type": "string",
                            "description": "UUID fiscal, con guiones.",
                        }
                    },
                    "required": ["uuid"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "historial_proveedor",
                "description": (
                    "Gasto por mes y hallazgos acumulados de un proveedor."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rfc": {"type": "string", "description": "RFC del emisor."},
                        "meses": {"type": "integer", "description": "Meses atrás."},
                    },
                    "required": ["rfc"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "buscar_conceptos_similares",
                "description": (
                    "Busca conceptos parecidos por significado, no por texto "
                    "exacto. Útil cuando distintos proveedores describen lo "
                    "mismo de otra forma."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "texto": {"type": "string", "description": "Descripción."},
                        "k": {"type": "integer", "description": "Cuántos resultados."},
                    },
                    "required": ["texto"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "enviar_alerta",
                "description": (
                    "Publica un mensaje en el canal de facturación de Slack. "
                    "Úsala solo si el usuario lo pide explícitamente."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mensaje": {"type": "string", "description": "Texto."}
                    },
                    "required": ["mensaje"],
                },
            },
        },
    ]


def _dispatch(conn: Any, name: str, args: dict[str, Any]) -> str:
    """Run one tool call. Same functions the Anthropic path calls.

    An unknown name is answered, not raised: the model chose it, and a tool
    result saying so lets it correct itself. Raising would end the conversation
    over a typo.
    """
    try:
        if name == "consultar_sql":
            payload = tools.query_sql(conn, args["sql"])
        elif name == "detalle_factura":
            payload = tools.get_invoice(conn, args["uuid"])
        elif name == "historial_proveedor":
            payload = tools.supplier_history(
                conn, args["rfc"], months=int(args.get("meses", 12))
            )
        elif name == "buscar_conceptos_similares":
            payload = tools.find_similar_line_items(
                conn, args["texto"], k=int(args.get("k", 10))
            )
        elif name == "enviar_alerta":
            payload = tools.send_alert(args["mensaje"])
        else:
            return json.dumps({"error": f"herramienta desconocida: {name}"})
    except Exception as exc:  # noqa: BLE001 - a tool error is a tool result
        # The security boundary lives in `tools.py` and raises on a rejected
        # query. That rejection is information the model should see and work
        # around, not a crash.
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)
    return json.dumps(payload, ensure_ascii=False, default=str)


def ask_local(question: str, *, verbose: bool = False) -> str:
    """Run the agent against the configured local model until it stops."""
    from cfdi_agent.agent.loop import SYSTEM_PROMPT
    from cfdi_agent.db.conn import connect

    cfg = get_config()
    base_url = (cfg.llm_base_url or "").rstrip("/")
    if not base_url:
        return "LLM_BASE_URL no está configurado; la ruta local no puede correr."

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                company=cfg.company_name, rfc=cfg.company_rfc
            ),
        },
        {"role": "user", "content": question},
    ]
    specs = tool_specs()

    with connect() as conn:
        for _ in range(MAX_ITERATIONS):
            try:
                response = httpx.post(
                    f"{base_url}/chat/completions",
                    json={
                        "model": cfg.llm_model,
                        "messages": messages,
                        "tools": specs,
                        "temperature": 0,
                        "max_tokens": 4096,
                    },
                    timeout=TIMEOUT,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                return f"la inferencia local falló: {exc}"

            message = response.json()["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            if not calls:
                return (message.get("content") or "").strip() or "(sin respuesta)"

            # The assistant message must be appended *with* its tool_calls
            # before the results, or the server sees results for a request it
            # has no record of and the model stops calling tools.
            messages.append(message)
            for call in calls:
                function = call.get("function", {})
                name = function.get("name", "")
                raw = function.get("arguments") or "{}"
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    args = {}
                if verbose:
                    print(f"  → {name}({raw[:160]})", file=sys.stderr)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", name),
                        "content": _dispatch(conn, name, args),
                    }
                )

    return (
        f"el modelo local no terminó en {MAX_ITERATIONS} iteraciones. "
        "Un modelo chico encadena peor; eso es la medición, no un error."
    )
