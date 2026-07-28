"""Compile a node graph into n8n workflow JSON.

The two things that make hand-written n8n JSON go wrong, both handled here:

*Connections are keyed by node name, not id.* Rename a node and every edge
pointing at it silently detaches. `Workflow.add` rejects duplicate names, and
`connect` refuses to reference a node that is not in the graph — so a broken
edge is an exception at build time rather than a flow that quietly does nothing
at 3am.

*`main` is an array of arrays.* The outer index is the output port (an `If`
node has 0 = true, 1 = false), the inner list is fan-out to several nodes.
Collapsing that to a flat list is the classic mistake, and it produces a
workflow where the false branch silently never fires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flows.builder.layout import auto_layout
from flows.builder.nodes import Node


class WorkflowError(ValueError):
    """The graph is malformed. Raised at build time, never at runtime."""


@dataclass
class Workflow:
    name: str
    nodes: list[Node] = field(default_factory=list)
    # (source name, output port) -> [target names]
    edges: dict[tuple[str, int], list[str]] = field(default_factory=dict)
    settings: dict[str, Any] = field(
        default_factory=lambda: {"executionOrder": "v1"}
    )

    def add(self, node: Node) -> Node:
        if any(n.name == node.name for n in self.nodes):
            raise WorkflowError(
                f"duplicate node name {node.name!r}: n8n keys connections by name, "
                "so two nodes sharing one would merge their edges"
            )
        self.nodes.append(node)
        return node

    def _require(self, node: Node) -> str:
        if node not in self.nodes:
            raise WorkflowError(f"node {node.name!r} was never added to the workflow")
        return node.name

    def connect(self, source: Node, target: Node, *, port: int = 0) -> None:
        """Wire source's output `port` to target's input."""
        src, dst = self._require(source), self._require(target)
        self.edges.setdefault((src, port), []).append(dst)

    def chain(self, *nodes: Node) -> None:
        """Connect a linear run of nodes through their default output."""
        for a, b in zip(nodes, nodes[1:], strict=False):
            self.connect(a, b)

    # ----------------------------------------------------------------- build

    def _connections(self) -> dict:
        out: dict[str, dict[str, list[list[dict]]]] = {}
        for (src, port), targets in self.edges.items():
            main = out.setdefault(src, {"main": []})["main"]
            # Pad so the port index is meaningful. An If node wired only on its
            # false branch still needs an empty slot at index 0.
            while len(main) <= port:
                main.append([])
            main[port] = [
                {"node": t, "type": "main", "index": 0} for t in targets
            ]
        return out

    def to_dict(self) -> dict:
        positions = auto_layout(
            [n.name for n in self.nodes],
            {k: list(v) for k, v in self.edges.items()},
        )
        for node in self.nodes:
            node.position = positions[node.name]
        return {
            "name": self.name,
            "nodes": [n.to_dict(self.name) for n in self.nodes],
            "connections": self._connections(),
            "settings": self.settings,
        }
