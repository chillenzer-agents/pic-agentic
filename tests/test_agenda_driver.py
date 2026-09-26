# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for the agenda live-test driver helpers (replica tagging)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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
