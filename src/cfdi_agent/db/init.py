"""Apply schema.sql to the configured database.

docker-compose mounts the same file into /docker-entrypoint-initdb.d, so a
fresh volume is initialized automatically. This module exists for the other
case: re-applying after a schema edit without wiping the volume. schema.sql is
written to be idempotent.

    python -m cfdi_agent.db.init
"""

from __future__ import annotations

import sys
from pathlib import Path

from cfdi_agent.config import get_config
from cfdi_agent.db.conn import connect

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def apply_schema() -> None:
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql)


def main() -> int:
    cfg = get_config()
    # Never print the password back out.
    safe_url = cfg.database_url.split("@")[-1]
    try:
        apply_schema()
    except Exception as exc:  # noqa: BLE001 - surface the real cause to the operator
        print(f"failed to apply schema to {safe_url}: {exc}", file=sys.stderr)
        return 1
    print(f"schema applied to {safe_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
