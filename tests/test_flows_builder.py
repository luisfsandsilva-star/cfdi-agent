"""Tests for the n8n workflow compiler and the round-trip normalizer.

No n8n instance required. The compiler and the normalizer are where the value
is; `client.py` is a thin HTTP wrapper over four endpoints.

The test that matters most is `test_round_trip_survives_a_canvas_edit`: it
fakes what n8n does to a workflow on save — random ids, a fresh webhookId, a
new updatedAt, jittered drag positions, reordered nodes — and asserts the
normalized form is unchanged. If that ever fails, every canvas save produces a
noisy diff, nobody reads them, and the whole "edit in the canvas, review in
git" story quietly dies.
"""

from __future__ import annotations

import json
import random

import pytest

from flows.builder import Workflow, WorkflowError, no_op, webhook
from flows.builder.client import N8nClient
from flows.builder.layout import GRID
from flows.builder.normalize import VOLATILE_WORKFLOW_KEYS, dumps, normalize
from flows.definitions import anomaly_digest, invoice_intake

ALL_DEFINITIONS = (invoice_intake, anomaly_digest)


@pytest.fixture(params=ALL_DEFINITIONS, ids=lambda m: m.NAME)
def definition(request):
    return request.param


# ------------------------------------------------------------------ compile


def test_compiles_to_valid_json(definition) -> None:
    payload = json.loads(dumps(definition.build().to_dict()))
    assert payload["name"] == definition.NAME
    assert payload["nodes"]
    assert payload["settings"]["executionOrder"] == "v1"


def test_build_is_deterministic(definition) -> None:
    """Two compilations must be byte-identical.

    Random node ids would make every push look like a change.
    """
    assert dumps(definition.build().to_dict()) == dumps(definition.build().to_dict())


def test_every_connection_targets_a_real_node(definition) -> None:
    payload = definition.build().to_dict()
    names = {n["name"] for n in payload["nodes"]}
    for source, ports in payload["connections"].items():
        assert source in names, f"edge from unknown node {source!r}"
        for port in ports["main"]:
            for edge in port:
                assert edge["node"] in names, f"edge to unknown node {edge['node']!r}"


def test_node_type_versions_are_pinned(definition) -> None:
    """A missing typeVersion makes n8n guess, and it guesses v1."""
    for node in definition.build().to_dict()["nodes"]:
        assert node.get("typeVersion"), node["name"]


# ------------------------------------------------------------------- graph


def test_duplicate_node_name_is_rejected() -> None:
    """n8n keys connections by name; two nodes sharing one merge their edges."""
    wf = Workflow("dup")
    wf.add(no_op("Same"))
    with pytest.raises(WorkflowError, match="duplicate node name"):
        wf.add(no_op("Same"))


def test_connecting_an_unadded_node_is_rejected() -> None:
    wf = Workflow("dangling")
    a = wf.add(no_op("A"))
    orphan = no_op("B")  # never added
    with pytest.raises(WorkflowError, match="never added"):
        wf.connect(a, orphan)


def test_if_node_ports_are_array_of_arrays() -> None:
    """The classic mistake: flattening `main` so the false branch never fires."""
    payload = invoice_intake.build().to_dict()
    branch = payload["connections"]["¿Hay anomalía?"]["main"]
    assert len(branch) == 2, "If node must expose both true and false ports"
    assert [e["node"] for e in branch[0]] == ["Avisar en Slack"]
    assert [e["node"] for e in branch[1]] == ["Sin hallazgos"]


def test_a_gap_in_ports_is_padded_not_collapsed() -> None:
    """Wiring only the false branch must still leave port 0 present but empty."""
    wf = Workflow("only-false")
    trigger = wf.add(webhook("T", path="t"))
    branch = wf.add(no_op("Branch"))
    target = wf.add(no_op("Target"))
    wf.chain(trigger, branch)
    wf.connect(branch, target, port=1)

    main = wf.to_dict()["connections"]["Branch"]["main"]
    assert main[0] == []
    assert [e["node"] for e in main[1]] == ["Target"]


# ------------------------------------------------------------------ layout


def test_nodes_do_not_overlap(definition) -> None:
    """A pile of boxes at [0,0] runs fine and is unreadable to a human."""
    positions = [tuple(n["position"]) for n in definition.build().to_dict()["nodes"]]
    assert len(set(positions)) == len(positions)


def test_positions_snap_to_the_grid(definition) -> None:
    for node in definition.build().to_dict()["nodes"]:
        x, y = node["position"]
        assert x % GRID == 0 and y % GRID == 0, node["name"]


def test_layout_flows_left_to_right() -> None:
    payload = invoice_intake.build().to_dict()
    by_name = {n["name"]: n["position"][0] for n in payload["nodes"]}
    assert by_name["Factura recibida"] < by_name["Ingestar CFDI"]
    assert by_name["Ingestar CFDI"] < by_name["¿Hay anomalía?"]
    assert by_name["¿Hay anomalía?"] < by_name["Avisar en Slack"]


# --------------------------------------------------------------- normalize


def test_normalize_strips_volatile_server_state(definition) -> None:
    raw = definition.build().to_dict()
    raw.update(
        {
            "id": "wf_abc123",
            "versionId": "ver_xyz",
            "createdAt": "2026-01-01T00:00:00.000Z",
            "updatedAt": "2026-07-27T12:00:00.000Z",
            "active": True,
            "triggerCount": 7,
            "tags": [{"id": "1", "name": "prod"}],
        }
    )
    cleaned = normalize(raw)
    assert not (VOLATILE_WORKFLOW_KEYS & set(cleaned))


def test_normalize_is_idempotent(definition) -> None:
    once = normalize(definition.build().to_dict())
    assert normalize(once) == once


def _simulate_n8n_export(workflow: dict, seed: int = 5) -> dict:
    """What a workflow looks like after n8n has stored and re-served it."""
    rng = random.Random(seed)
    out = json.loads(json.dumps(workflow))
    out["id"] = "wf_" + str(rng.getrandbits(48))
    out["versionId"] = str(rng.getrandbits(64))
    out["createdAt"] = "2026-01-01T00:00:00.000Z"
    out["updatedAt"] = "2026-07-27T18:30:00.000Z"
    out["active"] = True
    out["triggerCount"] = 1
    out["pinData"] = {}
    for node in out["nodes"]:
        node["id"] = str(rng.getrandbits(64))  # n8n reassigns ids
        if node["type"].endswith("webhook"):
            node["webhookId"] = str(rng.getrandbits(64))
        # A human nudging a box a few pixels
        node["position"] = [
            node["position"][0] + rng.randint(-8, 8),
            node["position"][1] + rng.randint(-8, 8),
        ]
    rng.shuffle(out["nodes"])  # the UI does not preserve array order
    return out


def test_round_trip_survives_a_canvas_edit(definition) -> None:
    """Build, push, drag things around, export: the diff must be empty.

    This is the load-bearing test for "define in Python, edit in the canvas,
    review in git".
    """
    built = definition.build().to_dict()
    exported = _simulate_n8n_export(built)
    assert dumps(exported) == dumps(built)


def test_a_real_edit_does_show_up(definition) -> None:
    """The normalizer must not be so aggressive it hides actual changes."""
    built = definition.build().to_dict()
    edited = _simulate_n8n_export(built)
    edited["nodes"][0]["parameters"]["__added_by_a_human"] = "yes"
    assert dumps(edited) != dumps(built)


def test_node_ids_are_stable_across_rebuilds(definition) -> None:
    a = {n["name"]: n["id"] for n in normalize(definition.build().to_dict())["nodes"]}
    b = {n["name"]: n["id"] for n in normalize(definition.build().to_dict())["nodes"]}
    assert a == b


# ------------------------------------------------------------------ client


def test_writable_strips_read_only_fields() -> None:
    """POST /workflows returns 400 if handed back the fields GET provides."""
    payload = normalize(invoice_intake.build().to_dict())
    payload.update({"id": "x", "active": True, "createdAt": "...", "tags": []})
    writable = N8nClient._writable(payload)
    assert set(writable) <= {"name", "nodes", "connections", "settings"}
    assert "name" in writable and "nodes" in writable


def test_from_env_refuses_to_run_without_a_key(monkeypatch) -> None:
    from flows.builder.client import N8nAuthError

    monkeypatch.delenv("N8N_API_KEY", raising=False)
    with pytest.raises(N8nAuthError, match="N8N_API_KEY"):
        N8nClient.from_env()


# --------------------------------------------------------- committed files


def test_exported_files_match_the_definitions(definition) -> None:
    """`flows/exported/*.json` must be what `python -m flows.sync build` writes.

    Guards against committing a hand-edited export that no longer matches the
    Python it claims to come from.
    """
    from flows.sync import _path_for

    path = _path_for(definition.NAME)
    if not path.exists():
        pytest.skip(f"{path.name} not built yet; run `python -m flows.sync build`")
    assert path.read_text(encoding="utf-8") == dumps(definition.build().to_dict())
