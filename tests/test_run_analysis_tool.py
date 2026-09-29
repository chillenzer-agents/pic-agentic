# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Offline tests for the ``run_analysis`` server tool over MemoryTransport."""

from __future__ import annotations

import asyncio
import json

from pic_agentic.analysis_program import AnalysisProgram
from pic_agentic.config import Config
from pic_agentic.protocol.simulation import ResultOp, SimulationType, build_result_ack
from pic_agentic.rcp import new_secret_hex
from pic_agentic.server.app import build_server
from pic_agentic.transport.memory import MemoryTransport

SECRET = new_secret_hex()
SIM = "7f3a2b1c"
SIM_ID = "abcd1234"


def _spectrum_program() -> dict:
    var = lambda name, comp: {"kind": "var", "name": name, "record": "E", "component": comp}  # ruff: ignore[lambda-assignment]
    return {
        "selectors": [var("px", "x"), var("py", "y")],
        "output": {
            "kind": "reduce",
            "op": "histogram",
            "bins": 4,
            "operand": {
                "kind": "binop",
                "op": "add",
                "left": {"kind": "binop", "op": "mul", "left": var("px", "x"), "right": var("px", "x")},
                "right": {"kind": "binop", "op": "mul", "left": var("py", "y"), "right": var("py", "y")},
            },
        },
    }


async def _serve_compute(sim_transport: MemoryTransport) -> asyncio.Task:
    """Echo a computed histogram for a COMPUTE request."""

    async def responder() -> None:
        async for command in sim_transport.receive():
            if command.type != SimulationType.RESULT_COMMAND:
                continue
            op = ResultOp(str(command.payload["op"]))
            ack = build_result_ack(
                sim=SIM,
                seq=1,
                cmd_id=str(command.payload.get("cmd_id", "")),
                sim_id=str(command.payload.get("sim_id", "")),
                op=op,
                in_reply_to=command.transport_event_id,
                result={"result_kind": "array"},
                data=[3.0, 1.0, 0.0, 0.0],
                data_encoding="float",
                n_points=4,
            ).sign(SECRET)
            await sim_transport.send(ack)

    return asyncio.create_task(responder())


async def _pump(mcp_transport, service) -> asyncio.Task:
    async def pump() -> None:
        async for message in mcp_transport.receive():
            service.on_message(message)

    return asyncio.create_task(pump())


async def _call(config: Config, name: str, arguments: dict):
    mcp_t, sim_t = MemoryTransport.create_pair()
    server, runtime = build_server(config, SIM)
    runtime._transport = mcp_t
    tasks = [await _serve_compute(sim_t), await _pump(mcp_t, runtime.submit_service)]
    try:
        return (await server.call_tool(name, arguments)).structured_content
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()


async def test_run_analysis_tool_is_registered() -> None:
    server, _runtime = build_server(Config(rcp_secret=SECRET), SIM)
    tools = {tool.name: tool for tool in await server.list_tools()}
    assert "run_analysis" in tools
    annotations = tools["run_analysis"].annotations
    assert annotations is not None
    assert annotations.read_only_hint is False
    assert annotations.destructive_hint is False


async def test_run_analysis_description_documents_the_program() -> None:
    server, _runtime = build_server(Config(rcp_secret=SECRET), SIM)
    tools = {tool.name: tool for tool in await server.list_tools()}
    description = tools["run_analysis"].description or ""
    for keyword in ("kind: 'var'", "kind: 'binop'", "kind: 'unop'", "kind: 'reduce'", "selectors", "output"):
        assert keyword in description, keyword
    assert '"op": "histogram"' in description
    assert description.index("Worked example") < description.index('"selectors"')


async def test_run_analysis_worked_example_is_valid_and_parses() -> None:
    """The documented example must be copy-pasteable: valid JSON that validates.

    The example is extracted verbatim from the live tool description, parsed
    with ``json`` and checked against the real ``AnalysisProgram`` model -- a
    substring grep cannot catch a missing brace.
    """
    server, _runtime = build_server(Config(rcp_secret=SECRET), SIM)
    tools = {tool.name: tool for tool in await server.list_tools()}
    description = tools["run_analysis"].description or ""
    fragment = description[description.index("Worked example") :]
    start = fragment.index("{")
    end = fragment.rindex("}") + 1
    example = fragment[start:end]

    parsed = json.loads(example)
    program = AnalysisProgram.model_validate(parsed)
    # The parsed program round-trips through the wire form the tool accepts.
    assert program.model_dump(mode="json", exclude_none=True)["output"]["kind"] == "reduce"


async def test_run_analysis_returns_the_spectrum() -> None:
    config = Config(rcp_secret=SECRET)
    payload = await _call(config, "run_analysis", {"sim_id": SIM_ID, "program": _spectrum_program()})
    assert payload["data_encoding"] == "float"
    assert payload["n_points"] == 4
    assert payload["data"] == [3.0, 1.0, 0.0, 0.0]


async def test_run_analysis_rejects_an_invalid_program_before_sending() -> None:
    # The tool's ResultParams validation rejects an unsafe program, so no
    # command reaches the simclient.
    config = Config(rcp_secret=SECRET)
    payload = await _call(
        config,
        "run_analysis",
        {
            "sim_id": SIM_ID,
            "program": {"output": {"kind": "unop", "op": "exec", "operand": {"kind": "const", "value": 1.0}}},
        },
    )
    assert payload["ok"] is False
