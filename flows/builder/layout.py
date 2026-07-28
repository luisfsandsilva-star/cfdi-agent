"""Place nodes on the canvas.

`position` is cosmetic to n8n — it will run a workflow whose nodes are all at
[0, 0]. It is not cosmetic to the person who opens the canvas: a pile of
overlapping boxes is unreadable, and the whole point of pushing to n8n is that
a non-developer can read the flow.

Layered left-to-right by graph depth, siblings stacked vertically. Positions
snap to a grid so a human nudging a node produces a small, legible diff rather
than a wall of one-pixel changes.
"""

from __future__ import annotations

from collections import defaultdict

X_STEP = 260
Y_STEP = 150
GRID = 20
ORIGIN = (0, 0)


def _snap(value: int) -> int:
    return round(value / GRID) * GRID


def auto_layout(
    node_names: list[str], edges: dict[tuple[str, int], list[str]]
) -> dict[str, tuple[int, int]]:
    """Assign a canvas position to each node.

    Depth is the longest path from a root, so a node that is reachable both
    directly and via a longer branch sits to the right of everything feeding
    it rather than overlapping its own predecessor.
    """
    successors: dict[str, list[str]] = defaultdict(list)
    has_incoming: set[str] = set()
    for (src, _port), targets in edges.items():
        for t in targets:
            successors[src].append(t)
            has_incoming.add(t)

    roots = [n for n in node_names if n not in has_incoming] or node_names[:1]

    depth: dict[str, int] = {r: 0 for r in roots}
    # Relax depths until stable. Graphs here are tiny (tens of nodes); the
    # iteration cap just guarantees termination if a cycle sneaks in.
    for _ in range(len(node_names) + 1):
        changed = False
        for src, targets in successors.items():
            if src not in depth:
                continue
            for t in targets:
                candidate = depth[src] + 1
                if depth.get(t, -1) < candidate:
                    depth[t] = candidate
                    changed = True
        if not changed:
            break

    # Anything unreachable still needs a home rather than stacking on the origin.
    for name in node_names:
        depth.setdefault(name, 0)

    by_depth: dict[int, list[str]] = defaultdict(list)
    for name in node_names:
        by_depth[depth[name]].append(name)

    positions: dict[str, tuple[int, int]] = {}
    for d, names in by_depth.items():
        # Centre each column vertically so branches fan out symmetrically.
        offset = -(len(names) - 1) * Y_STEP // 2
        for i, name in enumerate(names):
            positions[name] = (
                _snap(ORIGIN[0] + d * X_STEP),
                _snap(ORIGIN[1] + offset + i * Y_STEP),
            )
    return positions
