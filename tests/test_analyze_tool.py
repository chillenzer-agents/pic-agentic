# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Server-tool tests for ``analyze_output`` (milestone A).

The tool routes through the existing result pull; a fake responder answers the
``ANALYZE`` result request, so this exercises registration, the query
re-synthesis, redaction and the error paths without a cluster.
"""

from __future__ import annotations

import asyncio

from pic_agentic.config import Config
from pic_agentic.protocol.simulation import ResultOp, SimulationType, build_result_ack
from pic_agentic.rcp import new_secret_hex
from pic_agentic.server.app import build_server
from pic_agentic.transport.memory import MemoryTransport

SECRET = new_secret_hex()
SIM = "7f3a2b1c"
SIM_ID = "abcd1234"

_SECTIONS = {
    "rocrate": {"name": "Laser sweep", "software": {"name": "PIConGPU", "id": "#p"}},
    "metadata": {"runner": {"run_dir": "/x/run"}, "rc_params": {}, "rendering_context": {}},
    "openpmd": {"iterations": [0, 10], "latest_step": 10, "backends": ["openpmd-adios2"]},
    "answer": "Analysis summary: experiment name is Laser sweep.",
}


async def _responder(sim_t: MemoryTransport, *, sections: dict | None, error_code: str | None = None) -> asyncio.Task:
    async def loop() -> None:
        async for msg in sim_t.receive():
            # ``msg.type`` is a plain str on the wire, so use equality
            # (StrEnum compares by value), never identity.
            if msg.type != SimulationType.RESULT_COMMAND:
                continue
            kwargs: dict = {"in_reply_to": msg.transport_event_id}
            if error_code:
                kwargs.update({"error": "nope", "error_code": error_code})
            else:
                kwargs.update({"result": sections})
            ack = build_result_ack(
                sim=SIM,
                seq=1,
                cmd_id=str(msg.payload["cmd_id"]),
                sim_id=str(msg.payload["sim_id"]),
                op=ResultOp(str(msg.payload["op"])),
                **kwargs,
            )
            await sim_t.send(ack.sign(SECRET))

    return asyncio.create_task(loop())


async def _pump(mcp_t: MemoryTransport, runtime) -> asyncio.Task:
    async def loop() -> None:
        async for message in mcp_t.receive():
            runtime.submit_service.on_message(message)

    return asyncio.create_task(loop())


async def _call_tool(
    config: Config,
    name: str,
    arguments: dict,
    *,
    sections: dict | None = _SECTIONS,
    error_code: str | None = None,
) -> dict | None:
    mcp_t, sim_t = MemoryTransport.create_pair()
    server, runtime = build_server(config, SIM)
    runtime._transport = mcp_t
    tasks = [await _responder(sim_t, sections=sections, error_code=error_code), await _pump(mcp_t, runtime)]
    try:
        return (await server.call_tool(name, arguments)).structured_content
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await mcp_t.close()
        await sim_t.close()


async def test_analyze_tool_is_registered_read_only() -> None:
    server, _runtime = build_server(Config(rcp_secret=SECRET), SIM)
    tools = {tool.name: tool for tool in await server.list_tools()}
    assert "analyze_output" in tools
    assert tools["analyze_output"].annotations.read_only_hint is True
    assert set(tools["analyze_output"].input_schema["properties"]) >= {"sim_id", "query"}


async def test_analyze_tool_returns_sections() -> None:
    payload = await _call_tool(Config(rcp_secret=SECRET), "analyze_output", {"sim_id": SIM_ID})
    assert payload["ok"] is True
    assert payload["rocrate"]["name"] == "Laser sweep"
    assert payload["openpmd"]["latest_step"] == 10
    assert "Analysis summary" in payload["answer"]


async def test_analyze_tool_query_resynthesizes_answer() -> None:
    payload = await _call_tool(
        Config(rcp_secret=SECRET),
        "analyze_output",
        {"sim_id": SIM_ID, "query": "openpmd iterations"},
    )
    assert "Matched the query" in payload["answer"]
    assert "openPMD iterations" in payload["answer"]


async def test_analyze_tool_error_is_soft() -> None:
    payload = await _call_tool(Config(rcp_secret=SECRET), "analyze_output", {"sim_id": SIM_ID}, error_code="no_results")
    assert payload["ok"] is False
    assert payload["error_code"] == "no_results"


async def test_analyze_tool_redacts_secrets() -> None:
    sections = dict(_SECTIONS)
    sections["metadata"] = {"runner": {"api_token": SECRET}, "rc_params": {}, "rendering_context": {}}
    payload = await _call_tool(Config(rcp_secret=SECRET), "analyze_output", {"sim_id": SIM_ID}, sections=sections)
    assert SECRET not in str(payload)
