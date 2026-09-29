# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Offline tests for the ``build_spec`` dry-run tool (S2).

The PICMI child is stubbed at the ``runner_dump_builder`` seam, so the tool
exercises the real payload validation/size path without a cluster or a
PIConGPU install.  The point of ``build_spec`` is that it never submits: these
tests assert the builder is called and no simulation is registered.
"""

from __future__ import annotations

import json
from pathlib import Path

from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.store import AgendaStore
from pic_agentic.config import Config
from pic_agentic.protocol.simulation import MAX_INLINE_PAYLOAD_BYTES
from pic_agentic.rcp import new_secret_hex
from pic_agentic.server.app import build_server
from pic_agentic.simulation_build import BuiltSimulation, SimulationBuildError

SECRET = new_secret_hex()
SIM = "7f3a2b1c"
FIXTURE = Path(__file__).parent / "fixtures" / "pypicongpu_runner.json"


def _runner_dump() -> dict:
    return json.loads(FIXTURE.read_text())


def _built(runner: dict | None = None) -> BuiltSimulation:
    return BuiltSimulation(
        runner=runner or _runner_dump(),
        picongpu_version="0.9.0-dev",
        picongpu_revision="667c537620e685486aceeaa77deb6550ac9972cf",
        schema_hash="f6471fe1244f6a9d819951e090a281862b58b5b7170b5ae209b8841c3d94e4d9",
    )


class _StubBuilder:
    """Record calls and return a canned build (or raise)."""

    def __init__(self, result: BuiltSimulation, calls: list[object] | None = None) -> None:
        self.result = result
        self.calls = calls if calls is not None else []

    async def __call__(self, *, script_path, **_kw: object) -> BuiltSimulation:
        self.calls.append(script_path)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _server_with_builder(builder):
    server, runtime = build_server(Config(rcp_secret=SECRET), SIM)
    runtime.submit_service.runner_dump_builder = builder
    return server, runtime


async def test_build_spec_tool_registration_and_annotations() -> None:
    server, _runtime = build_server(Config(rcp_secret=SECRET), SIM)
    tools = {tool.name: tool for tool in await server.list_tools()}
    assert "build_spec" in tools
    annotations = tools["build_spec"].annotations
    assert annotations is not None
    assert annotations.read_only_hint is True
    assert annotations.destructive_hint is False
    assert set(tools["build_spec"].input_schema["properties"]) == {"picmi_script"}


async def test_build_spec_returns_the_runner_spec_without_submitting() -> None:
    builder = _StubBuilder(_built())
    server, runtime = _server_with_builder(builder)
    payload = (await server.call_tool("build_spec", {"picmi_script": "# picmi\n"})).structured_content

    assert payload["ok"] is True
    assert payload["spec"] == {"sim": _runner_dump()["sim"]}
    assert payload["wire_bytes"] > 0
    assert payload["wire_bytes"] <= MAX_INLINE_PAYLOAD_BYTES
    assert payload["inline_limit_bytes"] == MAX_INLINE_PAYLOAD_BYTES
    assert payload["schema_hash"]
    # The builder ran and nothing was registered/submitted (no transport here).
    assert len(builder.calls) == 1
    assert runtime.submit_service.registry == {}


async def test_build_spec_spec_is_accepted_by_submit_spec() -> None:
    """The returned spec feeds straight into the submit/agenda wire path."""
    builder = _StubBuilder(_built())
    server, runtime = _server_with_builder(builder)
    payload = (await server.call_tool("build_spec", {"picmi_script": "# picmi\n"})).structured_content

    _cmd_id, _sim_payload, command = runtime.submit_service._build_spec_payload(payload["spec"])
    assert command.payload["header"]["sim_id"]
    body = json.loads(command.payload["payload"])
    assert body["simulation"] == payload["spec"]


async def test_build_spec_over_inline_limit_is_an_explicit_soft_error() -> None:
    oversized = {"sim": {"blob": "a" * (MAX_INLINE_PAYLOAD_BYTES * 2)}}
    builder = _StubBuilder(_built(oversized))
    server, _runtime = _server_with_builder(builder)
    payload = (await server.call_tool("build_spec", {"picmi_script": "# picmi\n"})).structured_content

    assert payload["ok"] is False
    assert payload["error"] == "spec_exceeds_inline_limit"
    assert payload["wire_bytes"] > MAX_INLINE_PAYLOAD_BYTES
    assert "spec" not in payload
    # It was still only a dry run.
    assert len(builder.calls) == 1


async def test_build_spec_build_failure_is_a_soft_error() -> None:
    builder = _StubBuilder(SimulationBuildError("PICMI script failed"))
    server, _runtime = _server_with_builder(builder)
    payload = (await server.call_tool("build_spec", {"picmi_script": "# bad\n"})).structured_content

    assert payload["ok"] is False
    assert payload["state"] == "error"
    assert "PICMI script failed" in payload["error"]


async def test_build_spec_then_create_campaign_round_trip(tmp_path) -> None:
    """S2 + S1: the built spec is a valid create_campaign base_spec."""
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    builder = _StubBuilder(_built())
    server, runtime = build_server(config, SIM)
    runtime.submit_service.runner_dump_builder = builder

    built = (await server.call_tool("build_spec", {"picmi_script": "# picmi\n"})).structured_content
    created = (
        await server.call_tool(
            "create_campaign",
            {
                "name": "scan",
                "base_spec": built["spec"],
                "patch_path": "sim.time_steps",
                "values": [1, 2],
            },
        )
    ).structured_content
    assert created["ok"] is True
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    assert campaign.agenda.entries["leaf000"].point == {"time_steps": 1}
    assert campaign.agenda.entries["leaf000"].spec["sim"]["time_steps"] == 1
