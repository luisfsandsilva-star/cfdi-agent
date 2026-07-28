"""Flow: a daily digest of unresolved findings.

    Cron 09:00 ─► GET /anomalies/open ─► format ─► Slack

Separate from the per-invoice alert on purpose. Critical findings interrupt
immediately; everything else accumulates and arrives once a morning. A channel
that pings on every `info` gets muted within a week, and then the critical
alerts are muted too.

Cron runs in the container's GENERIC_TIMEZONE (America/Monterrey here), so
"09:00" means local time and survives DST.
"""

from __future__ import annotations

import os

from flows.builder.nodes import code, http_request, schedule_cron, slack_message
from flows.builder.workflow import Workflow

NAME = "cfdi-anomaly-digest"

# Formatting only. Anything resembling a decision belongs behind the API.
FORMAT_JS = """
const payload = $input.first().json;
const rows = payload.anomalies || [];
const icon = { critical: '🔴', warn: '🟡', info: '🔵' };

if (rows.length === 0) {
  return [{ json: { text: '✅ Sin anomalías abiertas.' } }];
}

const bySeverity = {};
for (const r of rows) (bySeverity[r.severity] ||= []).push(r);

const lines = [`*Anomalías abiertas: ${payload.count}*`];
for (const sev of ['critical', 'warn', 'info']) {
  for (const r of bySeverity[sev] || []) {
    lines.push(`${icon[sev]} ${r.kind} · ${r.rfc_emisor} · ${r.total} · ${r.invoice_uuid}`);
  }
}
return [{ json: { text: lines.join('\\n') } }];
""".strip()


def build(api_base_url: str | None = None, slack_channel: str | None = None) -> Workflow:
    api = (api_base_url or os.environ.get("API_BASE_URL", "http://host.docker.internal:8000")).rstrip("/")
    channel = slack_channel or os.environ.get("SLACK_CHANNEL", "#facturacion")

    wf = Workflow(NAME)

    trigger = wf.add(schedule_cron("Cada mañana 09:00", expression="0 9 * * *"))
    fetch = wf.add(
        http_request("Anomalías abiertas", url=f"{api}/anomalies/open", method="GET")
    )
    fmt = wf.add(code("Formatear digest", js=FORMAT_JS))
    post = wf.add(
        slack_message("Publicar digest", channel=channel, text="={{ $json.text }}")
    )

    wf.chain(trigger, fetch, fmt, post)
    return wf
