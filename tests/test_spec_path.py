# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Offline tests for path-referenced campaign specs (``base_spec_path``).

A staged JSON spec lets an agent pass a multi-KiB build through a file instead
of re-typing it into the LLM.  These tests cover the safe-path discipline (the
staging root is the only readable location), the exactly-one-of contract, the
size cap, and the ``build_spec(write_to=...)`` -> ``create_campaign(
base_spec_path=...)`` round trip.
"""

from __future__ import annotations

import json
from pathlib import Path

from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.store import AgendaStore
from pic_agentic.config import Config
from pic_agentic.rcp import new_secret_hex
from pic_agentic.server.app import build_server
from pic_agentic.server.spec_files import MAX_SPEC_FILE_BYTES, read_spec_file, write_spec_file
from pic_agentic.simclient.safety import UnsafePathError
from pic_agentic.simulation_build import BuiltSimulation

SECRET = new_secret_hex()
SIM = "7f3a2b1c"
FIXTURE = Path(__file__).parent / "fixtures" / "pypicongpu_runner.json"


def _runner_dump() -> dict:
    return json.loads(FIXTURE.read_text())


class _StubBuilder:
    def __init__(self, result: BuiltSimulation) -> None:
        self.result = result

    async def __call__(self, *, script_path, **_kw: object) -> BuiltSimulation:
        del script_path
        return self.result


def _built(runner: dict | None = None) -> BuiltSimulation:
    return BuiltSimulation(
        runner=runner or _runner_dump(),
        picongpu_version="0.9.0-dev",
        picongpu_revision="122160f6125eccd0606bb8ceb11c815c88ffaf2d",
        schema_hash="f6471fe1244f6a9d819951e090a281862b58b5b7170b5ae209b8841c3d94e4d9",
    )


def _config(tmp_path: Path) -> Config:
    return Config(
        rcp_secret=SECRET,
        agenda_file=str(tmp_path / "campaign.json"),
        spec_dir=str(tmp_path / "specs"),
    )


async def _call(config: Config, name: str, arguments: dict) -> dict:
    server, _runtime = build_server(config, SIM)
    return (await server.call_tool(name, arguments)).structured_content


async def test_read_write_spec_file_round_trip(tmp_path) -> None:
    config = _config(tmp_path)
    spec = {"sim": {"time_steps": 4}}
    written = write_spec_file(config, str(tmp_path / "specs" / "base.json"), spec)
    assert written.read_bytes().endswith(b"\n")
    assert read_spec_file(config, str(written)) == spec
    # A root-relative path resolves under the same staging root.
    assert read_spec_file(config, "base.json") == spec


def test_validate_spec_path_refuses_escape_and_charset(tmp_path) -> None:
    config = _config(tmp_path)
    for bad in ("/etc/passwd", str(tmp_path / "outside.json"), "../escape.json", " pad.json", "", "a b.json"):
        try:
            read_spec_file(config, bad)
        except UnsafePathError:
            continue
        msg = f"expected UnsafePathError for {bad!r}"
        raise AssertionError(msg)


async def test_create_campaign_from_a_staged_path(tmp_path) -> None:
    config = _config(tmp_path)
    base = _runner_dump()
    path = write_spec_file(config, "scan.json", base)
    result = await _call(
        config,
        "create_campaign",
        {"name": "scan", "base_spec_path": str(path), "patch_path": "sim.time_steps", "values": [50, 100]},
    )
    assert result == {"ok": True, "name": "scan", "leaves": ["leaf000", "leaf001"]}
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    assert campaign.agenda.entries["leaf000"].spec["sim"]["time_steps"] == 50
    # The inline base was never touched (there was none).
    assert json.loads(path.read_text()) == base


async def test_create_campaign_refuses_a_path_outside_the_root(tmp_path) -> None:
    config = _config(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"sim": {"time_steps": 4}}))
    config.agenda_file = str(tmp_path / "campaign.json")
    result = await _call(
        config,
        "create_campaign",
        {"name": "bad", "base_spec_path": str(outside), "patch_path": "sim.time_steps", "values": [1]},
    )
    assert result["ok"] is False
    assert result["error"] == "invalid_spec_path"
    assert not (tmp_path / "campaign.json").exists()


async def test_create_campaign_requires_exactly_one_base_form(tmp_path) -> None:
    config = _config(tmp_path)
    base = {"sim": {"time_steps": 4}}
    neither = await _call(
        config,
        "create_campaign",
        {"name": "x", "patch_path": "sim.time_steps", "values": [1]},
    )
    assert neither["ok"] is False
    assert neither["error"] == "base_spec_required"

    both = await _call(
        config,
        "create_campaign",
        {
            "name": "x",
            "base_spec": base,
            "base_spec_path": "scan.json",
            "patch_path": "sim.time_steps",
            "values": [1],
        },
    )
    assert both["ok"] is False
    assert both["error"] == "base_spec_required"
    assert not (tmp_path / "campaign.json").exists()


async def test_create_campaign_refuses_an_oversized_staged_file(tmp_path) -> None:
    config = _config(tmp_path)
    root = tmp_path / "specs"
    root.mkdir()
    oversized = root / "big.json"
    oversized.write_text(json.dumps({"sim": {"blob": "a" * (MAX_SPEC_FILE_BYTES + 1)}}))
    result = await _call(
        config,
        "create_campaign",
        {"name": "big", "base_spec_path": str(oversized), "patch_path": "sim.blob", "values": [1]},
    )
    assert result["ok"] is False
    assert result["error"] == "invalid_spec_path"
    assert not (tmp_path / "campaign.json").exists()


async def test_build_spec_stages_a_path_create_campaign_reads(tmp_path) -> None:
    """The whole point: build_spec writes -> create_campaign reads by path."""
    config = _config(tmp_path)
    server, runtime = build_server(config, SIM)
    runtime.submit_service.runner_dump_builder = _StubBuilder(_built())

    built = (
        await server.call_tool("build_spec", {"picmi_script": "# picmi\n", "write_to": "base.json"})
    ).structured_content
    assert built["ok"] is True
    assert Path(built["spec_path"]).exists()
    # ``write_to`` stages the spec, so the inline copy is omitted by default.
    assert "spec" not in built
    assert json.loads(Path(built["spec_path"]).read_text(encoding="utf-8")) == {"sim": _runner_dump()["sim"]}

    created = (
        await server.call_tool(
            "create_campaign",
            {"name": "scan", "base_spec_path": built["spec_path"], "patch_path": "sim.time_steps", "values": [1, 2]},
        )
    ).structured_content
    assert created == {"ok": True, "name": "scan", "leaves": ["leaf000", "leaf001"]}
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    assert campaign.agenda.entries["leaf000"].spec["sim"]["time_steps"] == 1


async def test_build_spec_without_write_to_has_no_spec_path(tmp_path) -> None:
    config = _config(tmp_path)
    server, runtime = build_server(config, SIM)
    runtime.submit_service.runner_dump_builder = _StubBuilder(_built())
    built = (await server.call_tool("build_spec", {"picmi_script": "# picmi\n"})).structured_content
    assert "spec_path" not in built


def test_build_spec_write_to_outside_root_is_an_error(tmp_path) -> None:
    config = _config(tmp_path)
    try:
        write_spec_file(config, "/etc/passwd", {"sim": {}})
    except UnsafePathError:
        return
    msg = "expected UnsafePathError"
    raise AssertionError(msg)


async def test_add_agenda_leaf_from_a_staged_path(tmp_path) -> None:
    """A leaf can carry its own whole spec by reference (multi-node studies)."""
    config = _config(tmp_path)
    base = _runner_dump()
    write_spec_file(config, "base.json", base)
    created = await _call(
        config,
        "create_campaign",
        {"name": "scan", "base_spec_path": "base.json", "patch_path": "sim.time_steps", "values": [50]},
    )
    assert created["ok"] is True

    # A different whole spec, staged under the same root, becomes the leaf spec
    # without re-typing or the inline cap applying.
    alt = _runner_dump()
    alt["sim"]["time_steps"] = 7
    leaf_path = write_spec_file(config, "leaf.json", alt)
    result = await _call(
        config,
        "add_agenda_leaf",
        {"name": "extra", "spec_path": str(leaf_path), "point": {"time_steps": 7}, "parameter": "steps"},
    )
    assert result == {"ok": True, "path": "extra"}
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    assert campaign.agenda.entries["extra"].spec["sim"]["time_steps"] == 7
    assert campaign.agenda.entries["extra"].sweep_parameter == "steps"


async def test_add_agenda_leaf_requires_exactly_one_spec_form(tmp_path) -> None:
    config = _config(tmp_path)
    write_spec_file(config, "base.json", _runner_dump())
    await _call(
        config,
        "create_campaign",
        {"name": "scan", "base_spec_path": "base.json", "patch_path": "sim.time_steps", "values": [50]},
    )
    neither = await _call(config, "add_agenda_leaf", {"name": "x"})
    assert neither["ok"] is False
    assert neither["error"] == "spec_required"
    assert "spec_path" in neither["detail"]

    both = await _call(config, "add_agenda_leaf", {"name": "x", "spec": {"sim": {}}, "spec_path": "base.json"})
    assert both["ok"] is False
    assert both["error"] == "spec_required"


async def test_add_agenda_leaf_refuses_a_path_outside_the_root(tmp_path) -> None:
    config = _config(tmp_path)
    write_spec_file(config, "base.json", _runner_dump())
    await _call(
        config,
        "create_campaign",
        {"name": "scan", "base_spec_path": "base.json", "patch_path": "sim.time_steps", "values": [50]},
    )
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"sim": {}}))
    result = await _call(config, "add_agenda_leaf", {"name": "x", "spec_path": str(outside)})
    assert result["ok"] is False
    assert result["error"] == "invalid_spec_path"


async def test_add_agenda_leaf_refuses_a_non_sim_spec(tmp_path) -> None:
    """A leaf that is not a Runner spec must be refused at add time.

    ``create_campaign`` already validates through the submission path; the
    by-reference leaf path must not be the one hole that persists a spec every
    later ``advance_agenda`` tick can only fail on.
    """
    config = _config(tmp_path)
    write_spec_file(config, "base.json", _runner_dump())
    await _call(
        config,
        "create_campaign",
        {"name": "scan", "base_spec_path": "base.json", "patch_path": "sim.time_steps", "values": [50]},
    )
    result = await _call(config, "add_agenda_leaf", {"name": "bad", "spec": {"not_sim": 1}})
    assert result["ok"] is False
    assert result["error"] == "invalid_campaign_spec"
    # Nothing invalid was persisted: the campaign still holds only its leaf000.
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    assert "bad" not in campaign.agenda.entries


async def test_add_agenda_leaf_refuses_an_over_cap_staged_spec(tmp_path, monkeypatch) -> None:
    """A staged leaf over the 48 KiB wire cap is refused at add time.

    Staging lifts the input file cap (4 MiB), not the escaped wire cap every
    ``submit_spec`` still meets; the leaf path must surface that early rather
    than at the next tick.  The pinned-schema round-trip is stubbed out so the
    size check is exercised deterministically with or without the pin.
    """
    from pic_agentic.server import agenda as agenda_module

    monkeypatch.setattr(agenda_module, "check_spec_round_trip", lambda _dump: None)
    config = _config(tmp_path)
    base = _runner_dump()
    write_spec_file(config, "base.json", base)
    await _call(
        config,
        "create_campaign",
        {"name": "scan", "base_spec_path": "base.json", "patch_path": "sim.time_steps", "values": [50]},
    )
    oversized = {"sim": {**base["sim"], "walltime": "a" * 60000}}
    leaf_path = write_spec_file(config, "big.json", oversized)
    result = await _call(config, "add_agenda_leaf", {"name": "big", "spec_path": str(leaf_path)})
    assert result["ok"] is False
    assert result["error"] == "spec_exceeds_inline_limit"
    assert result["inline_limit_bytes"] == 48 * 1024
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    assert "big" not in campaign.agenda.entries
