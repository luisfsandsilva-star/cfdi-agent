"""Embed line-item descriptions so near-duplicates can be found.

Runs locally regardless of which backend handles extraction. bge-m3 is
multilingual, strong on Spanish, and about 1 GB — there is no argument for
sending every line of every invoice over the network to compute a vector this
cheap. It is the clearest example of the tier-0/1/2 split paying off.

Backfill in batches:

    python -m cfdi_agent.enrich.embeddings --limit 2000
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from psycopg import Connection

from cfdi_agent.config import get_config
from cfdi_agent.db.conn import connect
from cfdi_agent.extract.providers.base import LLMProvider, ProviderError

BATCH = 128


def embedding_provider() -> LLMProvider:
    """Always the local backend: the Anthropic API does not serve embeddings."""
    from cfdi_agent.extract.providers.openai_compat import OpenAICompatProvider

    cfg = get_config()
    base = cfg.embed_base_url or cfg.llm_base_url
    if not base:
        raise ProviderError(
            "EMBED_BASE_URL is not set. Point it at a llama.cpp/Ollama server "
            "serving bge-m3, e.g. http://orin.local:8082/v1"
        )
    return OpenAICompatProvider(
        base_url=base,
        model=cfg.embed_model,
        embed_base_url=base,
        embed_model=cfg.embed_model,
    )


def pending_count(conn: Connection[Any]) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM line_items WHERE embedding IS NULL")
        return cur.fetchone()["n"]


def embed_pending(
    conn: Connection[Any], *, limit: int = 1000, provider: LLMProvider | None = None
) -> int:
    """Fill in missing embeddings. Returns how many rows were written."""
    provider = provider or embedding_provider()
    cfg = get_config()
    written = 0

    while written < limit:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, descripcion
                  FROM line_items
                 WHERE embedding IS NULL
                 ORDER BY id
                 LIMIT %s
                """,
                (min(BATCH, limit - written),),
            )
            rows = cur.fetchall()
        if not rows:
            break

        vectors = provider.embed([r["descripcion"] for r in rows])
        for row, vector in zip(rows, vectors, strict=True):
            if len(vector) != cfg.embed_dim:
                # A dimension mismatch means the wrong model is being served.
                # Writing it would corrupt the index silently and every
                # similarity verdict after it.
                raise ProviderError(
                    f"embedding model returned {len(vector)} dims, schema expects "
                    f"{cfg.embed_dim}. Check EMBED_MODEL / EMBED_DIM."
                )
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE line_items SET embedding = %s WHERE id = %s",
                    (str(vector), row["id"]),
                )
        written += len(rows)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()

    with connect() as conn:
        pending = pending_count(conn)
        print(f"{pending} line items without an embedding")
        if not pending:
            return 0
        try:
            written = embed_pending(conn, limit=args.limit)
        except ProviderError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 1
    print(f"embedded {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
