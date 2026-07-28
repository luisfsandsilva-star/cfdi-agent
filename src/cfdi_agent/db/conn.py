"""Postgres connection helpers.

Import is lazy on psycopg so the deterministic core can be imported (and
tested) on a machine with no database driver installed.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from cfdi_agent.config import get_config

if TYPE_CHECKING:  # pragma: no cover
    from psycopg import Connection


@contextmanager
def connect(autocommit: bool = False) -> Iterator[Connection[Any]]:
    """Yield a connection. Commits on clean exit, rolls back on exception."""
    import psycopg
    from psycopg.rows import dict_row

    conn = psycopg.connect(
        get_config().database_url, row_factory=dict_row, autocommit=autocommit
    )
    try:
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()


def fetch_all(sql: str, params: tuple[Any, ...] | dict[str, Any] | None = None) -> list[dict]:
    with connect(autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def fetch_one(sql: str, params: tuple[Any, ...] | dict[str, Any] | None = None) -> dict | None:
    with connect(autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()
