"""Typed node constructors for n8n workflows.

An n8n workflow is JSON: a `nodes[]` array and a `connections{}` object. The
canvas and the file are the same thing, which is what makes "define in Python,
edit in the canvas, diff in git" possible at all.

`typeVersion` values here were read out of the running n8n's own
`types/nodes.json` (see `scripts/` usage in the README), not written from
memory. A wrong typeVersion produces a workflow that imports without complaint
and then behaves subtly differently — the worst failure mode available.

Verified against n8n 2.31.7. When upgrading n8n, re-read the type catalog:

    docker compose exec -T n8n cat /home/node/.cache/n8n/public/types/nodes.json

There is no published JSON Schema for a node's `parameters`. The shapes below
come from that same catalog's declared defaults; when adding a node type, build
it once in the UI, export it, and turn the result into a constructor here
rather than guessing.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

# Stable namespace for deterministic node ids. Recomputing a workflow must
# produce byte-identical JSON, or every push shows up as a diff.
_NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


@dataclass
class Node:
    """One node. `name` is the identity: n8n keys connections by name, not id."""

    name: str
    type: str
    type_version: float | int
    parameters: dict[str, Any] = field(default_factory=dict)
    position: tuple[int, int] = (0, 0)
    credentials: dict[str, Any] | None = None
    always_output_data: bool = False

    def node_id(self, workflow_name: str) -> str:
        """Deterministic id derived from workflow + node name.

        n8n assigns random UUIDs. Deriving them instead keeps a recompiled
        workflow byte-identical, so a git diff shows what a human changed
        rather than which ids got regenerated.
        """
        return str(uuid.uuid5(_NS, f"{workflow_name}/{self.name}"))

    def to_dict(self, workflow_name: str) -> dict:
        out: dict[str, Any] = {
            "parameters": self.parameters,
            "id": self.node_id(workflow_name),
            "name": self.name,
            "type": self.type,
            "typeVersion": self.type_version,
            "position": list(self.position),
        }
        if self.credentials:
            out["credentials"] = self.credentials
        if self.always_output_data:
            out["alwaysOutputData"] = True
        return out


# --------------------------------------------------------------------------
# Triggers
# --------------------------------------------------------------------------


def webhook(
    name: str = "Webhook",
    *,
    path: str,
    method: str = "POST",
    response_mode: str = "lastNode",
) -> Node:
    """HTTP entry point. Reachable at /webhook/<path> once the flow is active."""
    return Node(
        name=name,
        type="n8n-nodes-base.webhook",
        type_version=2.1,
        parameters={
            "httpMethod": method,
            "path": path,
            "responseMode": response_mode,
            "options": {},
        },
    )


def schedule_cron(name: str = "Schedule", *, expression: str) -> Node:
    """Cron trigger. `expression` is standard 5-field cron.

    Timezone comes from the container's GENERIC_TIMEZONE (America/Monterrey in
    this compose file) — a cron expression with no timezone is a bug waiting
    for the next DST change.
    """
    return Node(
        name=name,
        type="n8n-nodes-base.scheduleTrigger",
        type_version=1.3,
        parameters={
            "rule": {"interval": [{"field": "cronExpression", "expression": expression}]}
        },
    )


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------


def http_request(
    name: str,
    *,
    url: str,
    method: str = "GET",
    send_body: bool = False,
    body_parameters: list[dict] | None = None,
    content_type: str | None = None,
    options: dict | None = None,
) -> Node:
    params: dict[str, Any] = {
        "method": method,
        "url": url,
        "options": options or {},
    }
    if send_body:
        params["sendBody"] = True
        if content_type:
            params["contentType"] = content_type
        if body_parameters:
            params["bodyParameters"] = {"parameters": body_parameters}
    return Node(
        name=name,
        type="n8n-nodes-base.httpRequest",
        type_version=4.4,
        parameters=params,
    )


def if_equals(name: str, *, left: str, right: str) -> Node:
    """Two-output branch: port 0 is true, port 1 is false.

    v2 of the If node takes a `filter`-typed `conditions` object rather than
    the flat left/operation/right of v1. The condition `id` is derived from the
    node name so recompiling stays byte-stable.
    """
    return Node(
        name=name,
        type="n8n-nodes-base.if",
        type_version=2.3,
        parameters={
            "conditions": {
                "options": {
                    "caseSensitive": True,
                    "leftValue": "",
                    "typeValidation": "strict",
                    "version": 2,
                },
                "conditions": [
                    {
                        "id": str(uuid.uuid5(_NS, f"cond/{name}")),
                        "leftValue": left,
                        "rightValue": right,
                        "operator": {"type": "string", "operation": "equals"},
                    }
                ],
                "combinator": "and",
            },
            "looseTypeValidation": False,
            "options": {},
        },
    )


def code(name: str, *, js: str) -> Node:
    """Run JavaScript over the incoming items.

    Kept for formatting only. Domain logic belongs in the Python service, not
    in a canvas node that nothing tests.
    """
    return Node(
        name=name,
        type="n8n-nodes-base.code",
        type_version=2,
        parameters={"jsCode": js},
    )


def slack_message(
    name: str,
    *,
    channel: str,
    text: str,
    credential_name: str = "Slack account",
) -> Node:
    """Post to Slack.

    The credential is referenced by name and lives in n8n's own encrypted
    store; nothing secret is ever written into the workflow JSON, which is what
    makes these files safe to commit.
    """
    return Node(
        name=name,
        type="n8n-nodes-base.slack",
        type_version=2.5,
        parameters={
            "resource": "message",
            "operation": "post",
            "select": "channel",
            "channelId": {"__rl": True, "value": channel, "mode": "name"},
            "text": text,
            "otherOptions": {},
        },
        credentials={"slackApi": {"name": credential_name}},
    )


def no_op(name: str = "Done") -> Node:
    return Node(name=name, type="n8n-nodes-base.noOp", type_version=1)
