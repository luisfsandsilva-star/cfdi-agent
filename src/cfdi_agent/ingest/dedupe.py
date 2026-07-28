"""Content hashing for idempotent ingest.

Two different kinds of "duplicate" live in this pipeline and conflating them
produces false alarms:

*Same bytes.* The identical file arrived twice — an n8n retry, a re-forwarded
email, a re-run of the watch directory. This is a no-op, not a finding. Nobody
wants a critical alert every time a webhook retries.

*Same UUID, different bytes.* The same fiscal document submitted twice in
different form, or a supplier re-issuing under a UUID already used. That is
detector #1 and it is genuinely critical.

The file hash answers the first question; the UUID answers the second.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
