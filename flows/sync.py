"""Move workflows between Python, n8n and git.

    python -m flows.sync build     # compile to flows/exported/, touch nothing remote
    python -m flows.sync push      # compile and upsert into n8n
    python -m flows.sync export    # pull from n8n, normalize, write to flows/exported/
    python -m flows.sync diff      # show what changed in n8n vs the committed files

The loop this enables: define a flow in Python, `push`, hand the canvas to
whoever owns the process, and when they rearrange it, `export` and read the
diff. Both directions are the same file.

`build` needs no n8n at all, which keeps the compiler testable in CI.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

from flows.builder.client import N8nAuthError, N8nClient, N8nError
from flows.builder.normalize import dumps, normalize
from flows.definitions import anomaly_digest, invoice_intake

EXPORT_DIR = Path(__file__).parent / "exported"

DEFINITIONS = {
    invoice_intake.NAME: invoice_intake.build,
    anomaly_digest.NAME: anomaly_digest.build,
}


def _path_for(name: str) -> Path:
    return EXPORT_DIR / f"{name}.json"


def _display(path: Path) -> str:
    cwd = Path.cwd()
    return str(path.relative_to(cwd)) if path.is_relative_to(cwd) else str(path)


def cmd_build() -> int:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in DEFINITIONS.items():
        payload = dumps(build().to_dict())
        path = _path_for(name)
        changed = not path.exists() or path.read_text(encoding="utf-8") != payload
        path.write_text(payload, encoding="utf-8")
        print(f"  {'updated' if changed else 'unchanged':<10} {_display(path)}")
    return 0


def cmd_push(activate: bool) -> int:
    client = N8nClient.from_env()
    client.ping()
    for name, build in DEFINITIONS.items():
        workflow = normalize(build().to_dict())
        remote, action = client.upsert(workflow)
        note = ""
        if activate:
            try:
                client.activate(remote["id"])
                note = " (activo)"
            except N8nError as exc:
                # Activation fails when a credential is missing — the flow is
                # still pushed and editable, so this is a warning, not a failure.
                note = f" (no activado: {str(exc)[:80]})"
        print(f"  {action:<8} {name}  id={remote['id']}{note}")
    return 0


def cmd_export() -> int:
    client = N8nClient.from_env()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for name in DEFINITIONS:
        remote = client.find_by_name(name)
        if remote is None:
            print(f"  missing   {name} (not in n8n; push it first)", file=sys.stderr)
            continue
        full = client.get(remote["id"])
        payload = dumps(full)
        path = _path_for(name)
        changed = not path.exists() or path.read_text(encoding="utf-8") != payload
        path.write_text(payload, encoding="utf-8")
        print(f"  {'changed':<8} {name}" if changed else f"  {'same':<8} {name}")
        written += 1
    return 0 if written else 1


def cmd_diff() -> int:
    """What has n8n got that the committed files do not?"""
    client = N8nClient.from_env()
    dirty = False
    for name in DEFINITIONS:
        remote = client.find_by_name(name)
        path = _path_for(name)
        local = path.read_text(encoding="utf-8") if path.exists() else ""
        if remote is None:
            print(f"--- {name}: not present in n8n")
            continue
        current = dumps(client.get(remote["id"]))
        if current == local:
            print(f"    {name}: in sync")
            continue
        dirty = True
        print(f"--- {name}: n8n differs from {path.name}")
        for line in difflib.unified_diff(
            local.splitlines(keepends=True),
            current.splitlines(keepends=True),
            fromfile=f"git/{path.name}",
            tofile=f"n8n/{name}",
        ):
            sys.stdout.write(line)
    return 1 if dirty else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("build", "push", "export", "diff"))
    ap.add_argument(
        "--activate", action="store_true", help="activate after push"
    )
    args = ap.parse_args()

    try:
        if args.command == "build":
            return cmd_build()
        if args.command == "push":
            return cmd_push(args.activate)
        if args.command == "export":
            return cmd_export()
        return cmd_diff()
    except N8nAuthError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2
    except N8nError as exc:
        print(f"\nn8n error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
