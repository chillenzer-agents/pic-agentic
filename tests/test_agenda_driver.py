# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for the agenda live-test driver helpers (replica tagging)."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

_DRIVER = Path(__file__).resolve().parent.parent / "scripts" / "local_mcp_check.py"
_SPEC = importlib.util.spec_from_file_location("local_mcp_check", _DRIVER)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - loader always present
    msg = f"cannot load driver at {_DRIVER}"
    raise RuntimeError(msg)
local_mcp_check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(local_mcp_check)


def test_replica_tag_makes_otherwise_identical_specs_distinct() -> None:
    base = {"sim": {"time_steps": 4}}
    a = local_mcp_check._tag_replica(base, 0)
    b = local_mcp_check._tag_replica(base, 1)
    assert a != b
    assert a["sim"]["customuserinput"]["pic_agentic_replica"] == 0
    assert b["sim"]["customuserinput"]["pic_agentic_replica"] == 1
    # The base spec is not mutated.
    assert "customuserinput" not in base["sim"]


def test_replica_tag_preserves_existing_custom_input() -> None:
    base = {"sim": {"customuserinput": {"tags": ["user"], "user_key": 7}}}
    tagged = local_mcp_check._tag_replica(base, 2)
    custom = tagged["sim"]["customuserinput"]
    assert custom["user_key"] == 7
    assert "user" in custom["tags"]
    assert "pic_agentic_replica" in custom["tags"]
    assert custom["pic_agentic_replica"] == 2


def test_replica_tag_json_round_trips() -> None:
    tagged = local_mcp_check._tag_replica({"sim": {"time_steps": 4}}, 3)
    assert json.loads(json.dumps(tagged)) == tagged


def test_agenda_init_records_sweep_points(tmp_path: Path) -> None:
    """--agenda-init must record each leaf's sweep ``point``.

    Regression: the builder created leaves without ``point``, so
    ``--agenda-refinement`` scored nothing (``analysed: 0``) even after an
    analysis was recorded, because the refiner falls back to the leaf's point.
    """
    agenda_file = tmp_path / "campaign.json"
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps({"sim": {"time_steps": 4}}))
    args = argparse.Namespace(
        agenda_file=agenda_file,
        agenda_script=str(spec_file),
        agenda_name="campaign",
        agenda_replicas=1,
        agenda_patch="sim.time_steps",
        agenda_values="50,100,200",
    )
    assert local_mcp_check.cmd_agenda_init(args) == 0
    campaign = json.loads(agenda_file.read_text())
    points = {name: leaf["point"] for name, leaf in campaign["agenda"]["entries"].items()}
    assert points == {
        "leaf000": {"time_steps": 50},
        "leaf001": {"time_steps": 100},
        "leaf002": {"time_steps": 200},
    }
    # The patched spec still carries the value, so spec and point agree.
    for leaf in campaign["agenda"]["entries"].values():
        assert leaf["spec"]["sim"]["time_steps"] == leaf["point"]["time_steps"]


def test_patch_spec_indexes_lists_by_decimal_segment() -> None:
    """A numeric segment descends a list; a non-numeric segment descends a dict."""
    base = {"sim": {"laser": [{"focus_pos_si": [{"component": 0.0}, {"component": 4.6e-5}]}]}}
    patched = local_mcp_check._patch_spec(base, "sim.laser.0.focus_pos_si.1.component", 4.4e-5)
    assert patched["sim"]["laser"][0]["focus_pos_si"][1]["component"] == pytest.approx(4.4e-5)
    assert patched["sim"]["laser"][0]["focus_pos_si"][0]["component"] == pytest.approx(0.0)
    # A negative index counts from the end (Python list indexing).
    tail = local_mcp_check._patch_spec(base, "sim.laser.-1.focus_pos_si.0.component", 9.9)
    assert tail["sim"]["laser"][0]["focus_pos_si"][0]["component"] == pytest.approx(9.9)
    # The base spec is not mutated.
    assert base["sim"]["laser"][0]["focus_pos_si"][1]["component"] == pytest.approx(4.6e-5)


def test_patch_spec_rejects_an_out_of_range_list_index() -> None:
    """A decimal segment beyond the list length is a soft ``SystemExit``."""
    base = {"sim": {"laser": [{"focus_pos_si": [{"component": 0.0}]}]}}
    with pytest.raises(SystemExit):
        local_mcp_check._patch_spec(base, "sim.laser.3.focus_pos_si.0.component", 1.0)


def test_agenda_init_omits_invalid_points() -> None:
    """A non-scalar/bool sweep value yields ``point=None`` instead of crashing."""
    assert local_mcp_check._point_for("p", 5) == {"p": 5}
    assert local_mcp_check._point_for("p", 1.5) == {"p": 1.5}
    assert local_mcp_check._point_for("p", "x") == {"p": "x"}
    truthy = True
    assert local_mcp_check._point_for("p", truthy) is None  # bool is not a point
    assert local_mcp_check._point_for("p", None) is None
    assert local_mcp_check._point_for("p", [1, 2]) is None


def test_agenda_init_with_a_non_scalar_value_does_not_raise(tmp_path: Path) -> None:
    """--agenda-init must not abort when a value cannot be a leaf point."""
    agenda_file = tmp_path / "campaign.json"
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps({"sim": {"time_steps": 4}}))
    args = argparse.Namespace(
        agenda_file=agenda_file,
        agenda_script=str(spec_file),
        agenda_name="campaign",
        agenda_replicas=1,
        agenda_patch="sim.time_steps",
        agenda_values="[1],4",
    )
    assert local_mcp_check.cmd_agenda_init(args) == 0
    campaign = json.loads(agenda_file.read_text())
    points = {name: leaf["point"] for name, leaf in campaign["agenda"]["entries"].items()}
    assert points == {"leaf000": None, "leaf001": {"time_steps": 4}}
