# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Offline tests for the M3 server-side control and results tools.

A tiny fake responder answers ``control_request`` / ``result_request`` over
``MemoryTransport``; it deliberately does not depend on the real simclient or
the results engine.  The contract-4 local-mirror readability check is exercised
against a temporary ``results_root``.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from pic_agentic.config import Config
from pic_agentic.protocol.simulation import (
    ResultOp,
    SimulationOp,
    SimulationType,
    build_control_ack,
    build_result_ack,
)
from pic_agentic.rcp import new_secret_hex
from pic_agentic.server.app import build_server
from pic_agentic.server.simulation import _PULL_ACK_FOR_REQUEST, SubmitService
from pic_agentic.transport.memory import MemoryTransport

SECRET = new_secret_hex()
SIM = "7f3a2b1c"
SIM_ID = "abcd1234"
JOB_ID = 4242

#: The result file the fake describe/export acks reference, relative to
#: ``<results_root>/<sim_id>/simOutput``.
_REL_PATH = "diag/fields_00000100.h5"


def _service(*, ack_timeout_s: float = 2.0, results_root: str = "") -> SubmitService:
    return SubmitService(sim=SIM, secret=SECRET, ack_timeout_s=ack_timeout_s, results_root=results_root)


def _ack_for(command: SimulationType, payload: dict[str, object], *, event_id: str | None) -> object:
    """Build the ack a healthy simclient would answer with for ``command``."""
    cmd_id = str(payload.get("cmd_id", ""))
    sim_id = str(payload.get("sim_id", ""))
    if command == SimulationType.CONTROL_COMMAND:
        return build_control_ack(
            sim=SIM,
            seq=1,
            cmd_id=cmd_id,
            sim_id=sim_id,
            op=SimulationOp(str(payload["op"])),
            ok=True,
            in_reply_to=event_id,
            job_id=JOB_ID,
            signal="USR1",
            state="simulation.checkpoint",
        ).sign(SECRET)
    op = ResultOp(str(payload["op"]))
    if op is ResultOp.DESCRIBE:
        return build_result_ack(
            sim=SIM,
            seq=2,
            cmd_id=cmd_id,
            sim_id=sim_id,
            op=op,
            in_reply_to=event_id,
            manifest={
                "sim_id": sim_id,
                "run_dir": "/cluster/run",
                "output_dir": "/cluster/run/simOutput",
                "total_bytes": 12,
                "reader": "openpmd",
                "readable_local": False,
                "files": [
                    {
                        "path": _REL_PATH,
                        "uri": f"file:///cluster/run/simOutput/{_REL_PATH}",
                        "format": "openpmd-hdf5",
                        "size_bytes": 12,
                    }
                ],
            },
        ).sign(SECRET)
    if op is ResultOp.EXPORT:
        return build_result_ack(
            sim=SIM,
            seq=3,
            cmd_id=cmd_id,
            sim_id=sim_id,
            op=op,
            in_reply_to=event_id,
            result={
                "path": _REL_PATH,
                "uri": f"file:///cluster/run/simOutput/{_REL_PATH}",
                "format": "openpmd-hdf5",
                "size_bytes": 12,
                "readable": False,
            },
        ).sign(SECRET)
    if op is ResultOp.SLICE:
        return build_result_ack(
            sim=SIM,
            seq=4,
            cmd_id=cmd_id,
            sim_id=sim_id,
            op=op,
            in_reply_to=event_id,
            data=[1.0, 2.0, 3.0],
            data_encoding="float",
            n_points=3,
        ).sign(SECRET)
    if op is ResultOp.IMAGE:
        return build_result_ack(
            sim=SIM,
            seq=6,
            cmd_id=cmd_id,
            sim_id=sim_id,
            op=op,
            in_reply_to=event_id,
            data="aGVsbG8=",
            data_encoding="png",
            n_points=1,
        ).sign(SECRET)
    # READ: a small text tail (or a shaped reader error).
    return build_result_ack(
        sim=SIM,
        seq=5,
        cmd_id=cmd_id,
        sim_id=sim_id,
        op=op,
        in_reply_to=event_id,
        data=["line-1", "line-2"],
        data_encoding="text",
        n_points=2,
    ).sign(SECRET)


async def _serve(sim_transport: MemoryTransport, *, error_code: str | None = None) -> asyncio.Task:
    """Answer control/result pulls on the sim side until cancelled."""

    async def responder() -> None:
        async for command in sim_transport.receive():
            if command.type not in {SimulationType.CONTROL_COMMAND, SimulationType.RESULT_COMMAND}:
                continue
            if error_code is not None and command.type == SimulationType.RESULT_COMMAND:
                ack = build_result_ack(
                    sim=SIM,
                    seq=1,
                    cmd_id=str(command.payload.get("cmd_id", "")),
                    sim_id=str(command.payload.get("sim_id", "")),
                    op=ResultOp(str(command.payload["op"])),
                    in_reply_to=command.transport_event_id,
                    error="reader unavailable",
                    error_code=error_code,
                ).sign(SECRET)
            else:
                ack = _ack_for(command.type, command.payload, event_id=command.transport_event_id)
            await sim_transport.send(ack)

    return asyncio.create_task(responder())


async def _pump(mcp_transport: MemoryTransport, service: SubmitService) -> asyncio.Task:
    async def pump() -> None:
        async for message in mcp_transport.receive():
            service.on_message(message)

    return asyncio.create_task(pump())


async def _call_tool(
    config: Config,
    name: str,
    arguments: dict[str, object],
    *,
    error_code: str | None = None,
):
    """Build a server, wire a fake responder and call one tool."""
    mcp_t, sim_t = MemoryTransport.create_pair()
    server, runtime = build_server(config, SIM)
    runtime._transport = mcp_t
    tasks = [await _serve(sim_t, error_code=error_code), await _pump(mcp_t, runtime.submit_service)]
    try:
        result = await server.call_tool(name, arguments)
        return result.structured_content
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()


def test_pull_ack_map_covers_control_and_result() -> None:
    assert _PULL_ACK_FOR_REQUEST[SimulationType.CONTROL_COMMAND] is SimulationType.CONTROL_ACK
    assert _PULL_ACK_FOR_REQUEST[SimulationType.RESULT_COMMAND] is SimulationType.RESULT_ACK


async def test_tool_registration_shape() -> None:
    server, _runtime = build_server(Config(rcp_secret=SECRET), SIM)
    tools = {tool.name: tool for tool in await server.list_tools()}
    expected = {
        "checkpoint_simulation",
        "stop_simulation",
        "cancel_simulation",
        "checkpoint_and_stop_simulation",
        "describe_results",
        "get_result_slice",
        "get_result_image",
        "read_result",
        "read_plugin_result",
        "export_results",
    }
    assert expected <= set(tools)
    # The frozen tool surface the LLM sees: one new tool must not go missing.
    assert len(tools) == 35
    for name in (
        "checkpoint_simulation",
        "stop_simulation",
        "checkpoint_and_stop_simulation",
    ):
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.read_only_hint is False
        assert annotations.destructive_hint is False
        assert annotations.idempotent_hint is False
    # A hard cancel kills the job outright (no clean shutdown), so it is
    # destructive, unlike the signal-delivering control verbs above.
    cancel = tools["cancel_simulation"].annotations
    assert cancel is not None
    assert cancel.read_only_hint is False
    assert cancel.destructive_hint is True
    for name in (
        "describe_results",
        "get_result_slice",
        "get_result_image",
        "read_result",
        "read_plugin_result",
        "export_results",
    ):
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.read_only_hint is True
    # The frozen signatures the LLM sees.
    assert set(tools["get_result_slice"].input_schema["properties"]) == {
        "sim_id",
        "record",
        "path",
        "component",
        "iteration",
        "axis",
        "index",
        "downsample",
    }
    assert set(tools["get_result_image"].input_schema["properties"]) == {
        "sim_id",
        "record",
        "path",
        "component",
        "iteration",
    }
    assert set(tools["read_result"].input_schema["properties"]) == {"sim_id", "path", "stream", "tail"}
    assert set(tools["read_plugin_result"].input_schema["properties"]) == {
        "sim_id",
        "reader",
        "species",
        "species_filter",
        "iteration",
        "path",
        "min_kev",
        "max_kev",
    }


async def test_read_plugin_result_tool_forwards_the_energy_window() -> None:
    """The ``read_plugin_result`` tool forwards ``min_kev``/``max_kev`` (F2 Nit).

    Closes the ``read_plugin_result -> _result_tool -> ResultParams ->
    build_result_command`` leg that the direct-service round-trips do not
    exercise.
    """
    mcp_t, sim_t = MemoryTransport.create_pair()
    server, runtime = build_server(Config(rcp_secret=SECRET), SIM)
    runtime._transport = mcp_t
    captured: list[dict[str, object]] = []

    async def responder() -> None:
        async for command in sim_t.receive():
            if command.type == SimulationType.RESULT_COMMAND:
                captured.append(dict(command.payload))
                await sim_t.send(_ack_for(command.type, command.payload, event_id=command.transport_event_id))

    tasks = [asyncio.create_task(responder()), await _pump(mcp_t, runtime.submit_service)]
    try:
        await server.call_tool(
            "read_plugin_result",
            {
                "sim_id": SIM_ID,
                "reader": "energy_histogram",
                "species": "e",
                "min_kev": 2500.0,
                "max_kev": 20000.0,
            },
        )
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()

    assert captured, "the tool emitted no result_request"
    payload = captured[-1]
    assert payload["reader"] == "energy_histogram"
    assert payload["min_kev"] == pytest.approx(2500.0)
    assert payload["max_kev"] == pytest.approx(20000.0)


async def test_direct_service_control_round_trip() -> None:
    mcp_t, sim_t = MemoryTransport.create_pair()
    service = _service()
    tasks = [await _serve(sim_t), await _pump(mcp_t, service)]
    try:
        payload = await service.control(mcp_t.send, SIM_ID, SimulationOp.CHECKPOINT)
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()

    assert payload["sim_id"] == SIM_ID
    assert payload["op"] == SimulationOp.CHECKPOINT.value
    assert payload["ok"] is True
    assert payload["signal"] == "USR1"


async def test_direct_service_fetch_result_round_trip() -> None:
    from pic_agentic.protocol.simulation import ResultParams

    mcp_t, sim_t = MemoryTransport.create_pair()
    service = _service()
    tasks = [await _serve(sim_t), await _pump(mcp_t, service)]
    try:
        params = ResultParams(sim_id=SIM_ID, op=ResultOp.SLICE, record="E", component="z", iteration="last")
        payload = await service.fetch_result(mcp_t.send, params)
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()

    assert payload["op"] == ResultOp.SLICE.value
    assert payload["data"] == [1.0, 2.0, 3.0]
    assert payload["n_points"] == 3


async def test_control_tool_via_server() -> None:
    payload = await _call_tool(Config(rcp_secret=SECRET), "stop_simulation", {"sim_id": SIM_ID})
    assert payload["ok"] is True
    assert payload["op"] == SimulationOp.STOP.value
    assert payload["sim_id"] == SIM_ID


async def test_result_tools_via_server() -> None:
    describe = await _call_tool(Config(rcp_secret=SECRET), "describe_results", {"sim_id": SIM_ID})
    assert describe["op"] == ResultOp.DESCRIBE.value
    assert describe["manifest"]["files"][0]["path"] == _REL_PATH
    assert describe["manifest"]["readable_local"] is False

    slice_payload = await _call_tool(
        Config(rcp_secret=SECRET),
        "get_result_slice",
        {"sim_id": SIM_ID, "record": "E", "iteration": "last"},
    )
    assert slice_payload["data"] == [1.0, 2.0, 3.0]

    image_payload = await _call_tool(
        Config(rcp_secret=SECRET),
        "get_result_image",
        {"sim_id": SIM_ID, "record": "E", "iteration": "last"},
    )
    assert image_payload["data"] == "aGVsbG8="
    assert image_payload["data_encoding"] == "png"

    read = await _call_tool(Config(rcp_secret=SECRET), "read_result", {"sim_id": SIM_ID, "path": _REL_PATH})
    assert read["data"] == ["line-1", "line-2"]
    assert read["data_encoding"] == "text"


async def test_control_timeout_is_a_soft_error() -> None:
    mcp_t, _sim_t = MemoryTransport.create_pair()
    server, runtime = build_server(Config(rcp_secret=SECRET), SIM)
    runtime._transport = mcp_t
    runtime.submit_service.ack_timeout_s = 0.05
    try:
        payload = (await server.call_tool("checkpoint_simulation", {"sim_id": SIM_ID})).structured_content
    finally:
        await mcp_t.close()
    assert payload["ok"] is False
    assert payload["error"] == "timeout"
    assert payload["op"] == SimulationOp.CHECKPOINT.value


async def test_result_timeout_is_a_soft_error() -> None:
    mcp_t, _sim_t = MemoryTransport.create_pair()
    server, runtime = build_server(Config(rcp_secret=SECRET), SIM)
    runtime._transport = mcp_t
    runtime.submit_service.ack_timeout_s = 0.05
    try:
        payload = (await server.call_tool("describe_results", {"sim_id": SIM_ID})).structured_content
    finally:
        await mcp_t.close()
    assert payload["ok"] is False
    assert payload["error"] == "timeout"


async def test_reader_unavailable_is_a_soft_error_with_code() -> None:
    payload = await _call_tool(
        Config(rcp_secret=SECRET),
        "get_result_slice",
        {"sim_id": SIM_ID, "record": "E"},
        error_code="reader_unavailable",
    )
    assert payload["ok"] is False
    assert payload["error_code"] == "reader_unavailable"


@pytest.mark.parametrize(
    "arguments",
    [
        {"sim_id": SIM_ID, "path": _REL_PATH, "stream": "nope"},
        {"sim_id": SIM_ID, "path": "../escape.txt"},
        {"sim_id": SIM_ID, "path": "/etc/passwd"},
    ],
)
async def test_result_invalid_args_are_soft_errors(arguments: dict[str, object]) -> None:
    # No responder needed: local ResultParams validation fails before any send.
    server, runtime = build_server(Config(rcp_secret=SECRET), SIM)
    runtime._transport = MemoryTransport()
    payload = (await server.call_tool("read_result", arguments)).structured_content
    assert payload["ok"] is False
    assert "error" in payload


async def test_unexpected_value_error_is_a_soft_error() -> None:
    # A ValueError raised inside a pull (not by MCP schema validation) must be
    # caught by the tool rather than escaping.
    server, runtime = build_server(Config(rcp_secret=SECRET), SIM)

    async def boom(_params):  # type: ignore[no-untyped-def]
        msg = "boom from the engine"
        raise ValueError(msg)

    runtime.fetch_result = boom  # type: ignore[method-assign]
    payload = (await server.call_tool("describe_results", {"sim_id": SIM_ID})).structured_content
    assert payload["ok"] is False
    assert "boom from the engine" in payload["error"]


async def test_result_unsafe_path_is_a_soft_error() -> None:
    server, runtime = build_server(Config(rcp_secret=SECRET), SIM)
    runtime._transport = MemoryTransport()
    payload = (await server.call_tool("read_result", {"sim_id": SIM_ID, "path": "../escape.txt"})).structured_content
    assert payload["ok"] is False
    assert "unsafe result path" in payload["error"]


def test_local_mirror_marks_readable_with_explicit_root() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / SIM_ID / "simOutput"
        (base / "diag").mkdir(parents=True)
        (base / _REL_PATH).write_text("data")
        service = _service(results_root=tmp)
        payload = {
            "manifest": {
                "sim_id": SIM_ID,
                "readable_local": False,
                "files": [{"path": _REL_PATH, "readable": False}, {"path": "missing.h5", "readable": False}],
            },
            "result": {"path": _REL_PATH, "readable": False},
        }
        service._mark_readable(SIM_ID, payload)

    assert payload["manifest"]["readable_local"] is True
    assert payload["manifest"]["files"][0]["readable"] is True
    assert payload["manifest"]["files"][1]["readable"] is False
    assert payload["result"]["readable"] is True


def test_local_mirror_is_a_noop_without_root() -> None:
    service = _service(results_root="")
    payload = {"manifest": {"readable_local": False, "files": [{"path": _REL_PATH, "readable": False}]}}
    service._mark_readable(SIM_ID, payload)
    assert payload["manifest"]["readable_local"] is False
    assert payload["manifest"]["files"][0]["readable"] is False


async def test_describe_readable_toggles_with_temp_results_root() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / SIM_ID / "simOutput"
        (base / "diag").mkdir(parents=True)
        (base / _REL_PATH).write_text("data")
        payload = await _call_tool(Config(rcp_secret=SECRET, results_root=tmp), "describe_results", {"sim_id": SIM_ID})
    assert payload["manifest"]["readable_local"] is True
    assert payload["manifest"]["files"][0]["readable"] is True


async def test_export_readable_toggles_with_temp_results_root() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / SIM_ID / "simOutput"
        (base / "diag").mkdir(parents=True)
        (base / _REL_PATH).write_text("data")
        payload = await _call_tool(Config(rcp_secret=SECRET, results_root=tmp), "export_results", {"sim_id": SIM_ID})
    assert payload["result"]["readable"] is True


async def test_misrouted_ack_does_not_resolve_a_pending_control_pull() -> None:
    service = _service(ack_timeout_s=0.05)
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    service._pending_pull["shared-cmd"] = future
    service._pending_pull_kind["shared-cmd"] = SimulationType.CONTROL_COMMAND
    # A results ack carrying the control request's cmd_id must not resolve it.
    misrouted = build_result_ack(
        sim=SIM,
        seq=1,
        cmd_id="shared-cmd",
        sim_id=SIM_ID,
        op=ResultOp.DESCRIBE,
        in_reply_to=None,
        error="nope",
        error_code="no_results",
    ).sign(SECRET)
    service.on_message(misrouted)
    assert not future.done()


async def test_control_and_result_acks_are_not_projected_into_the_registry() -> None:
    service = _service()
    ack = build_control_ack(
        sim=SIM,
        seq=1,
        cmd_id="c1",
        sim_id=SIM_ID,
        op=SimulationOp.CHECKPOINT,
        ok=True,
        in_reply_to=None,
        state="simulation.checkpoint",
    ).sign(SECRET)
    service.on_message(ack)
    result = build_result_ack(
        sim=SIM,
        seq=2,
        cmd_id="c2",
        sim_id=SIM_ID,
        op=ResultOp.DESCRIBE,
        in_reply_to=None,
        manifest={"sim_id": SIM_ID, "run_dir": "/r", "total_bytes": 0, "readable_local": False, "files": []},
    ).sign(SECRET)
    service.on_message(result)
    assert service.registry == {}
    assert service.event_log == []


def test_local_mirror_rejects_sim_id_traversal() -> None:
    """A ``../`` sim_id must not turn the mirror flag into an existence oracle."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "mirror"
        victim = Path(tmp) / "victim" / "simOutput"
        victim.mkdir(parents=True)
        (victim / "secret.h5").write_text("data")
        service = _service(results_root=str(root))
        payload = {
            "manifest": {
                "sim_id": "../../victim",
                "readable_local": False,
                "files": [{"path": "secret.h5", "readable": False}],
            },
            "result": {"path": "secret.h5", "readable": False},
        }
        service._mark_readable("../../victim", payload)

    assert payload["manifest"]["readable_local"] is False
    assert payload["manifest"]["files"][0]["readable"] is False
    assert payload["result"]["readable"] is False


def test_mirror_has_rejects_path_escape() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / SIM_ID / "simOutput"
        base.mkdir(parents=True)
        (Path(tmp) / "outside").write_text("x")
        assert SubmitService._mirror_has(base, "../../outside") is False


async def test_transport_failure_is_a_soft_error() -> None:
    """A transport error must never escape the tool as an exception."""
    server, runtime = build_server(Config(rcp_secret=SECRET), SIM)

    class _Boom:
        @staticmethod
        async def send(_message: object) -> str:
            msg = "transport down"
            raise RuntimeError(msg)

    runtime._transport = _Boom()
    control = await server.call_tool("checkpoint_simulation", {"sim_id": SIM_ID})
    assert control.structured_content["ok"] is False
    result = await server.call_tool("describe_results", {"sim_id": SIM_ID})
    assert result.structured_content["ok"] is False
