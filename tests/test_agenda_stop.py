# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Server-level tests for the campaign kill-switch (stop_agenda)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.model import AgendaGroup, AgendaSim
from pic_agentic.agenda.store import AgendaStore
from pic_agentic.config import Config
from pic_agentic.protocol.simulation import (
    SimulationOp,
    SimulationState,
    SimulationType,
    build_control_ack,
    build_submit_ack,
)
from pic_agentic.rcp import new_secret_hex
from pic_agentic.server.agenda import NO_CAMPAIGN_MESSAGE
from pic_agentic.server.app import build_server
from pic_agentic.transport.memory import MemoryTransport

SECRET = new_secret_hex()
SIM = "7f3a2b1c"


def _campaign_file(tmp_path: Path, *, leaves: int = 2) -> str:
    agenda = AgendaGroup(name="group")
    for index in range(leaves):
        leaf = f"leaf{index}"
        agenda = agenda.add(**{leaf: AgendaSim(name=leaf, spec={"sim": {"replica": index}})})
    AgendaStore(tmp_path, filename="campaign.json").save(Campaign(name="c", agenda=agenda))
    return str(tmp_path / "campaign.json")


async def _serve(sim_transport: MemoryTransport) -> tuple[asyncio.Task, list[dict]]:
    """Ack submits and controls; record every control op seen."""
    seen: list[dict] = []

    async def responder() -> None:
        counter = 0
        async for command in sim_transport.receive():
            if command.type == SimulationType.COMMAND:
                counter += 1
                ack = build_submit_ack(
                    sim=SIM,
                    seq=counter,
                    cmd_id=str(command.payload.get("cmd_id", "")),
                    sim_id=f"sim{counter:04d}",
                    state=SimulationState.ACCEPTED,
                    in_reply_to=command.transport_event_id,
                ).sign(SECRET)
                await sim_transport.send(ack)
            elif command.type == SimulationType.CONTROL_COMMAND:
                counter += 1
                payload = command.payload
                seen.append({"sim_id": payload.get("sim_id"), "op": payload.get("op")})
                ack = build_control_ack(
                    sim=SIM,
                    seq=counter,
                    cmd_id=str(payload.get("cmd_id", "")),
                    sim_id=str(payload.get("sim_id", "")),
                    op=SimulationOp(payload["op"]),
                    ok=True,
                    in_reply_to=command.transport_event_id,
                ).sign(SECRET)
                await sim_transport.send(ack)

    return asyncio.create_task(responder()), seen


async def _pump(mcp_transport: MemoryTransport, service) -> asyncio.Task:
    async def pump() -> None:
        async for message in mcp_transport.receive():
            service.on_message(message)

    return asyncio.create_task(pump())


def _runtime(config: Config):
    mcp_t, sim_t = MemoryTransport.create_pair()
    server, runtime = build_server(config, SIM)
    runtime._transport = mcp_t
    return mcp_t, sim_t, server, runtime


async def test_stop_cancels_in_flight_leaves(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    mcp_t, sim_t, server, runtime = _runtime(config)
    responder, seen = await _serve(sim_t)
    pump = await _pump(mcp_t, runtime.submit_service)
    try:
        tick = (await server.call_tool("advance_agenda", {})).structured_content
        assert sorted(tick["submitted"]) == ["leaf0", "leaf1"]

        result = (await server.call_tool("stop_agenda", {})).structured_content
        assert result["ok"] is True
        assert result["state"] == "stopped"
        # Exactly the two in-flight leaves were cancelled (by sim_id), with the
        # cancel op.
        assert len(result["cancelled"]) == 2
        assert set(result["cancelled"]) == {f"sim{i:04d}" for i in (1, 2)}
        assert len(seen) == 2
        assert all(op["op"] == SimulationOp.CANCEL.value for op in seen)
        assert {op["sim_id"] for op in seen} == set(result["cancelled"])

        # A stopped campaign holds everything.
        after = (await server.call_tool("advance_agenda", {})).structured_content
        assert after["submitted"] == []
        assert after["state"] == "stopped"
    finally:
        for task in (responder, pump):
            task.cancel()
        await mcp_t.close()
        await sim_t.close()


async def test_stop_without_campaign_is_a_soft_error(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "missing.json"))
    mcp_t, sim_t, server, _ = _runtime(config)
    try:
        result = (await server.call_tool("stop_agenda", {})).structured_content
        assert result == {"ok": False, "error": "no_campaign", "message": NO_CAMPAIGN_MESSAGE}
    finally:
        await mcp_t.close()
        await sim_t.close()


async def test_stop_without_transport_is_a_soft_error(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    server, _ = build_server(config, SIM)
    result = (await server.call_tool("stop_agenda", {})).structured_content
    assert result == {"ok": False, "error": "unavailable"}
