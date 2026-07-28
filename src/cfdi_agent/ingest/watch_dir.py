"""Ingest a directory of CFDIs.

The zero-dependency entry point: no n8n, no webhook, no Gmail OAuth. Useful for
bulk-loading a backlog, for the eval harness, and as the thing that still works
when the orchestration layer is down.

    python -m cfdi_agent.ingest.watch_dir --once data/synth
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

from cfdi_agent.db.conn import connect
from cfdi_agent.ingest.pipeline import ingest_file

STATUS_ORDER = ("ok", "anomaly", "duplicate_file", "needs_review")


def run_once(directory: Path, pattern: str, quiet: bool) -> dict[str, int]:
    files = sorted(directory.glob(pattern))
    if not files:
        print(f"no files matching {pattern!r} in {directory}", file=sys.stderr)
        return {}

    counts: collections.Counter[str] = collections.Counter()
    findings: collections.Counter[str] = collections.Counter()
    started = time.perf_counter()

    # One connection, one transaction per file: a failure on invoice 47 must
    # not roll back the 46 already ingested.
    for path in files:
        with connect() as conn:
            outcome = ingest_file(conn, path)
        counts[outcome.status] += 1
        for a in outcome.anomalies:
            findings[a["kind"]] += 1
        if not quiet and outcome.status != "ok":
            print(f"  {outcome.status:<14} {path.name:<16} {outcome.summary}")

    elapsed = time.perf_counter() - started
    print(f"\n{len(files)} documentos en {elapsed:.1f}s ({len(files)/elapsed:.0f}/s)")
    for status in STATUS_ORDER:
        if counts[status]:
            print(f"  {status:<14} {counts[status]}")
    if findings:
        print("\nhallazgos:")
        for kind, n in findings.most_common():
            print(f"  {kind:<24} {n}")
    return dict(counts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directory", type=Path)
    ap.add_argument("--once", action="store_true", help="process and exit (default)")
    ap.add_argument("--pattern", default="*.xml")
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    args = ap.parse_args()

    if not args.directory.is_dir():
        print(f"not a directory: {args.directory}", file=sys.stderr)
        return 1
    run_once(args.directory, args.pattern, args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
