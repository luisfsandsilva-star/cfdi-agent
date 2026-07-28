"""Thin client for the n8n Public API.

Verified against n8n 2.31.7: `GET /api/v1/workflows` answers 401 without a key,
which is how you confirm the Public API is enabled at all (a build gate worth
running before writing anything on top of it).

Two shapes that bite:

*Create rejects read-only fields.* `POST /api/v1/workflows` accepts only
`name`, `nodes`, `connections` and `settings`. Passing back `id`, `active`,
`createdAt` or `tags` — the fields a `GET` hands you — returns 400. `_writable`
strips them, so an export can be re-pushed unchanged.

*Activation is a separate endpoint.* Setting `"active": true` in the body does
nothing; you POST to `/activate` afterwards.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

WRITABLE_FIELDS = ("name", "nodes", "connections", "settings")


class N8nError(RuntimeError):
    pass


class N8nAuthError(N8nError):
    """No API key, or the key was rejected."""


@dataclass
class N8nClient:
    base_url: str
    api_key: str
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> N8nClient:
        base = os.environ.get("N8N_BASE_URL", "http://localhost:5678").rstrip("/")
        key = os.environ.get("N8N_API_KEY", "").strip()
        if not key:
            raise N8nAuthError(
                "N8N_API_KEY is not set. Create one in the n8n UI under "
                "Settings > n8n API, then put it in .env."
            )
        return cls(base_url=base, api_key=key)

    # ------------------------------------------------------------- plumbing

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}/api/v1{path}"
        headers = {"X-N8N-API-KEY": self.api_key, "Accept": "application/json"}
        try:
            resp = httpx.request(
                method, url, headers=headers, timeout=self.timeout, **kwargs
            )
        except httpx.HTTPError as exc:
            raise N8nError(f"{method} {url} failed: {exc}") from exc
        if resp.status_code == 401:
            raise N8nAuthError(f"n8n rejected the API key ({url})")
        if resp.status_code >= 400:
            raise N8nError(f"{method} {path} -> {resp.status_code}: {resp.text[:400]}")
        return resp.json() if resp.content else None

    @staticmethod
    def _writable(workflow: dict) -> dict:
        """Keep only the fields the create/update endpoints accept."""
        return {k: workflow[k] for k in WRITABLE_FIELDS if k in workflow}

    # ---------------------------------------------------------------- verbs

    def ping(self) -> bool:
        self._request("GET", "/workflows", params={"limit": 1})
        return True

    def list_workflows(self) -> list[dict]:
        data = self._request("GET", "/workflows", params={"limit": 250})
        return data.get("data", []) if isinstance(data, dict) else []

    def find_by_name(self, name: str) -> dict | None:
        for wf in self.list_workflows():
            if wf.get("name") == name:
                return wf
        return None

    def get(self, workflow_id: str) -> dict:
        return self._request("GET", f"/workflows/{workflow_id}")

    def create(self, workflow: dict) -> dict:
        return self._request("POST", "/workflows", json=self._writable(workflow))

    def update(self, workflow_id: str, workflow: dict) -> dict:
        return self._request(
            "PUT", f"/workflows/{workflow_id}", json=self._writable(workflow)
        )

    def upsert(self, workflow: dict) -> tuple[dict, str]:
        """Create or update by name. Returns (workflow, 'created'|'updated')."""
        existing = self.find_by_name(workflow["name"])
        if existing is None:
            return self.create(workflow), "created"
        return self.update(existing["id"], workflow), "updated"

    def activate(self, workflow_id: str) -> dict:
        return self._request("POST", f"/workflows/{workflow_id}/activate")

    def deactivate(self, workflow_id: str) -> dict:
        return self._request("POST", f"/workflows/{workflow_id}/deactivate")
