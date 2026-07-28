"""Runtime configuration, loaded from the environment (and .env if present).

Deliberately a plain dataclass rather than pydantic-settings: the deterministic
core (parser, rules, detectors) must import cleanly with zero optional
dependencies installed and no credentials configured. `pytest` runs against
this module with an empty environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:  # optional; the core works without it
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:  # pragma: no cover
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


@dataclass(frozen=True, slots=True)
class Config:
    # Database
    database_url: str

    # The entity receiving the invoices. An invoice addressed to a different
    # RFC is not an anomaly — it is not ours, and goes straight to review.
    company_rfc: str
    company_name: str

    # LLM
    llm_provider: str  # anthropic | local
    llm_model: str
    llm_base_url: str
    anthropic_api_key: str
    embed_base_url: str
    embed_model: str
    embed_dim: int

    # n8n
    n8n_base_url: str
    n8n_api_key: str
    api_base_url: str

    # SAT status web service: mock | live
    sat_status_mode: str

    # Slack
    slack_webhook_url: str
    slack_channel: str

    @property
    def llm_enabled(self) -> bool:
        """True when tier-2 (API) calls are actually possible.

        The tier-0 pipeline never consults this — it must run with no key.
        """
        if self.llm_provider == "local":
            return bool(self.llm_base_url)
        return bool(self.anthropic_api_key)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config(
        database_url=_env(
            "DATABASE_URL", "postgresql://cfdi:cfdi@localhost:5432/cfdi"
        ),
        company_rfc=_env("COMPANY_RFC", "XAXX010101000").upper(),
        company_name=_env("COMPANY_NAME", "Mi Empresa SA de CV"),
        llm_provider=_env("LLM_PROVIDER", "anthropic").lower(),
        llm_model=_env("LLM_MODEL", "claude-opus-5"),
        llm_base_url=_env("LLM_BASE_URL"),
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        embed_base_url=_env("EMBED_BASE_URL"),
        embed_model=_env("EMBED_MODEL", "bge-m3"),
        embed_dim=int(_env("EMBED_DIM", "1024")),
        n8n_base_url=_env("N8N_BASE_URL", "http://localhost:5678").rstrip("/"),
        n8n_api_key=_env("N8N_API_KEY"),
        api_base_url=_env("API_BASE_URL", "http://host.docker.internal:8000").rstrip("/"),
        sat_status_mode=_env("SAT_STATUS_MODE", "mock").lower(),
        slack_webhook_url=_env("SLACK_WEBHOOK_URL"),
        slack_channel=_env("SLACK_CHANNEL", "#facturacion"),
    )
