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
