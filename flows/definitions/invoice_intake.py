"""Flow: a CFDI arrives, gets ingested, anomalies reach Slack.

    Webhook ─► POST /ingest ─► If status == "anomaly" ─┬─true─► Slack
                                                       └─false─► Done

Note what is *not* here. No tax rules, no arithmetic, no RFC validation — the
canvas holds triggers, a call, a branch and a notification. Domain logic lives
in tested Python behind `/ingest`, because a fiscal rule expressed as canvas
nodes is a rule nothing can test and nobody can review.

The webhook is the development entry point. Swapping it for a Gmail trigger is
a one-node change and needs no downstream edits, which is the reason ingestion
was not built around Gmail from the start.
"""

from __future__ import annotations

import os

from flows.builder.nodes import http_request, if_equals, no_op, slack_message, webhook
from flows.builder.workflow import Workflow

NAME = "cfdi-invoice-intake"


def build(api_base_url: str | None = None, slack_channel: str | None = None) -> Workflow:
    api = (api_base_url or os.environ.get("API_BASE_URL", "http://host.docker.internal:8000")).rstrip("/")
    channel = slack_channel or os.environ.get("SLACK_CHANNEL", "#facturacion")

    wf = Workflow(NAME)

    trigger = wf.add(webhook("Factura recibida", path="cfdi", method="POST"))
    ingest = wf.add(
        http_request(
            "Ingestar CFDI",
            url=f"{api}/ingest",
            method="POST",
            send_body=True,
            content_type="multipart-form-data",
            body_parameters=[
                {
                    "parameterType": "formBinaryData",
                    "name": "file",
                    "inputDataFieldName": "data",
                }
            ],
        )
    )
    branch = wf.add(
        if_equals("¿Hay anomalía?", left="={{ $json.status }}", right="anomaly")
    )
    alert = wf.add(
        slack_message(
            "Avisar en Slack",
            channel=channel,
            # `summary` is assembled from the detectors' own evidence, so the
            # alert text cannot describe a finding that did not happen.
            text=(
                "={{ $json.summary }}\n"
                "UUID: {{ $json.uuid }} · "
                "hallazgos: {{ $json.anomalies.map(a => a.label).join(', ') }}"
            ),
        )
    )
    done = wf.add(no_op("Sin hallazgos"))

    wf.chain(trigger, ingest, branch)
    wf.connect(branch, alert, port=0)  # true
    wf.connect(branch, done, port=1)  # false
    return wf
