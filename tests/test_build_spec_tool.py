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
        picongpu_revision="122160f6125eccd0606bb8ceb11c815c88ffaf2d",
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
    assert set(tools["build_spec"].input_schema["properties"]) == {"picmi_script", "write_to", "include_spec"}


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


async def test_build_spec_deep_legitimate_spec_round_trips_exactly() -> None:
    """A depth>=8 legitimate spec is returned verbatim, never redacted (B1).

    The old path routed the built spec through ``_redact_dict``, whose
    ``_REDACT_MAX_DEPTH`` cutoff replaced deeply nested (but entirely
    legitimate) config with a marker while still reporting ``ok=True``.
    """
    runner = _runner_dump()
    # Push a real, allow-listed branch deeper: depth(sim) becomes 7 (>= 8 when
    # counted through the {"sim": ...} wrapper + payload).
    runner["sim"]["output"][0]["config"]["radiation"]["frequencies"] = {
        "type_linear_frequencies": {"omega_min": 1.0, "omega_max": 2.0},
    }
    builder = _StubBuilder(_built(runner))
    server, _runtime = _server_with_builder(builder)
    payload = (await server.call_tool("build_spec", {"picmi_script": "# picmi\n"})).structured_content

    assert payload["ok"] is True
    assert payload["spec"] == {"sim": runner["sim"]}
    assert "nesting too deep" not in json.dumps(payload["spec"])


async def test_build_spec_wire_bytes_matches_the_escaped_submission_measure() -> None:
    """``wire_bytes`` is the escaped size ``submit_spec`` enforces (n2)."""
    from pic_agentic.protocol.simulation import SimulationPayload, payload_wire_size

    builder = _StubBuilder(_built())
    server, _runtime = _server_with_builder(builder)
    payload = (await server.call_tool("build_spec", {"picmi_script": "# picmi\n"})).structured_content

    runner = _runner_dump()
    expected = SimulationPayload.build(
        picongpu_version="0.9.0-dev",
        picongpu_revision="122160f6125eccd0606bb8ceb11c815c88ffaf2d",
        schema_hash="f6471fe1244f6a9d819951e090a281862b58b5b7170b5ae209b8841c3d94e4d9",
        runner_dump=runner,
    )
    assert payload["wire_bytes"] == payload_wire_size(expected)
    # The escaped measure is genuinely larger than the raw inner object.
    assert payload["wire_bytes"] > len(json.dumps({"sim": runner["sim"]}))


async def test_build_spec_cap_boundary_is_inclusive() -> None:
    """Exactly the cap passes; one byte more fails (n2)."""
    from pic_agentic.protocol.simulation import (
        _ENVELOPE_ALLOWANCE_BYTES,
        SimulationPayload,
        payload_wire_size,
    )

    runner = _runner_dump()
    base_payload = SimulationPayload.build(
        picongpu_version="0.9.0-dev",
        picongpu_revision="122160f6125eccd0606bb8ceb11c815c88ffaf2d",
        schema_hash="f6471fe1244f6a9d819951e090a281862b58b5b7170b5ae209b8841c3d94e4d9",
        runner_dump=runner,
    )

    def sized(blob: int) -> dict:
        candidate = json.loads(json.dumps(runner))
        candidate["sim"]["blob"] = "a" * blob
        return candidate

    # Grow the raw blob until the escaped size is exactly the cap.
    lo, hi = 0, MAX_INLINE_PAYLOAD_BYTES
    while lo < hi:
        mid = (lo + hi) // 2
        probe = SimulationPayload.build(
            picongpu_version="0.9.0-dev",
            picongpu_revision="122160f6125eccd0606bb8ceb11c815c88ffaf2d",
            schema_hash="f6471fe1244f6a9d819951e090a281862b58b5b7170b5ae209b8841c3d94e4d9",
            runner_dump=sized(mid),
        )
        if payload_wire_size(probe) < MAX_INLINE_PAYLOAD_BYTES:
            lo = mid + 1
        else:
            hi = mid
    # ``lo`` yields size >= cap; step back to land exactly on the cap.
    exact = None
    for blob in (lo, lo - 1, lo + 1):
        if blob < 0:
            continue
        probe = SimulationPayload.build(
            picongpu_version="0.9.0-dev",
            picongpu_revision="122160f6125eccd0606bb8ceb11c815c88ffaf2d",
            schema_hash="f6471fe1244f6a9d819951e090a281862b58b5b7170b5ae209b8841c3d94e4d9",
            runner_dump=sized(blob),
        )
        if payload_wire_size(probe) == MAX_INLINE_PAYLOAD_BYTES:
            exact = blob
            break
    assert exact is not None, "no raw blob lands exactly on the escaped cap"
    assert payload_wire_size(base_payload) < MAX_INLINE_PAYLOAD_BYTES

    # Exactly the cap: accepted.
    builder = _StubBuilder(_built(sized(exact)))
    server, _runtime = _server_with_builder(builder)
    ok = (await server.call_tool("build_spec", {"picmi_script": "# picmi\n"})).structured_content
    assert ok["ok"] is True
    assert ok["wire_bytes"] == MAX_INLINE_PAYLOAD_BYTES
    assert "spec" in ok

    # One raw byte more: rejected.
    builder = _StubBuilder(_built(sized(exact + 1)))
    server, _runtime = _server_with_builder(builder)
    over = (await server.call_tool("build_spec", {"picmi_script": "# picmi\n"})).structured_content
    assert over["ok"] is False
    assert over["error"] == "spec_exceeds_inline_limit"
    assert over["wire_bytes"] > MAX_INLINE_PAYLOAD_BYTES
    assert _ENVELOPE_ALLOWANCE_BYTES > 0  # sanity: the allowance is part of the measure


async def test_build_spec_write_to_omits_the_inline_spec(tmp_path) -> None:
    """``write_to`` stages the spec, so it is not echoed inline by default."""
    config = Config(
        rcp_secret=SECRET,
        agenda_file=str(tmp_path / "campaign.json"),
        spec_dir=str(tmp_path / "specs"),
    )
    builder = _StubBuilder(_built())
    server, runtime = build_server(config, SIM)
    runtime.submit_service.runner_dump_builder = builder

    staged = (
        await server.call_tool("build_spec", {"picmi_script": "# picmi\n", "write_to": "base.json"})
    ).structured_content
    assert staged["ok"] is True
    assert "spec" not in staged
    assert staged["spec_path"]
    assert staged["wire_bytes"] > 0
    # The staged file still carries the full spec.
    assert json.loads(Path(staged["spec_path"]).read_text(encoding="utf-8")) == {"sim": _runner_dump()["sim"]}

    inline = (
        await server.call_tool(
            "build_spec",
            {"picmi_script": "# picmi\n", "write_to": "base2.json", "include_spec": True},
        )
    ).structured_content
    assert inline["spec"] == {"sim": _runner_dump()["sim"]}


async def test_build_spec_over_cap_with_write_to_reports_spec_path(tmp_path) -> None:
    """An over-cap spec still surfaces the staged path and honours ``include_spec``.

    The staging use case is exactly the spec too big to hand back inline, so the
    soft error must not drop ``spec_path`` (the runtime already wrote the file);
    an explicit ``include_spec=true`` returns the inline copy alongside the size
    warning.
    """
    config = Config(
        rcp_secret=SECRET,
        agenda_file=str(tmp_path / "campaign.json"),
        spec_dir=str(tmp_path / "specs"),
    )
    oversized = {"sim": {"blob": "a" * (MAX_INLINE_PAYLOAD_BYTES * 2)}}
    builder = _StubBuilder(_built(oversized))
    server, runtime = build_server(config, SIM)
    runtime.submit_service.runner_dump_builder = builder

    staged = (
        await server.call_tool("build_spec", {"picmi_script": "# picmi\n", "write_to": "base.json"})
    ).structured_content
    assert staged["ok"] is False
    assert staged["error"] == "spec_exceeds_inline_limit"
    assert staged["wire_bytes"] > MAX_INLINE_PAYLOAD_BYTES
    # The staged path is reported even though the inline body is not.
    assert "spec" not in staged
    assert staged["spec_path"]
    assert Path(staged["spec_path"]).exists()
    assert json.loads(Path(staged["spec_path"]).read_text(encoding="utf-8")) == oversized

    forced = (
        await server.call_tool(
            "build_spec",
            {"picmi_script": "# picmi\n", "write_to": "base2.json", "include_spec": True},
        )
    ).structured_content
    assert forced["ok"] is False
    assert forced["error"] == "spec_exceeds_inline_limit"
    assert forced["spec"] == oversized
    assert forced["spec_path"]


async def test_build_spec_include_spec_false_omits_without_write_to() -> None:
    builder = _StubBuilder(_built())
    server, _runtime = _server_with_builder(builder)
    payload = (
        await server.call_tool("build_spec", {"picmi_script": "# picmi\n", "include_spec": False})
    ).structured_content
    assert payload["ok"] is True
    assert "spec" not in payload
    assert "spec_path" not in payload


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
