# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Server-level tests for lost-ack deferral and reconcile (C2/H1).

Drives the real ``AgendaService`` submit callable over a ``MemoryTransport``:
an outcome-unknown ack (a pending idempotency record) must defer the leaf and
retry it next tick under the same ``cmd_id``, while a genuine rejection must
still fail it terminally.  No cluster is involved.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.model import AgendaGroup, AgendaSim
from pic_agentic.agenda.store import AgendaStore
from pic_agentic.config import Config
from pic_agentic.protocol.simulation import SimulationState, SimulationType, build_submit_ack
from pic_agentic.rcp import new_secret_hex
from pic_agentic.server.app import build_server
from pic_agentic.server.simulation import SubmitService
from pic_agentic.transport.memory import MemoryTransport

SECRET = new_secret_hex()
SIM = "7f3a2b1c"


def _campaign_file(tmp_path: Path, *, leaves: int = 1) -> str:
    agenda = AgendaGroup(name="group")
    for index in range(leaves):
        leaf = f"leaf{index}"
        agenda = agenda.add(**{leaf: AgendaSim(name=leaf, spec={"sim": {"time_steps": 4, "replica": index}})})
    AgendaStore(tmp_path, filename="campaign.json").save(Campaign(name="scan", agenda=agenda))
    return str(tmp_path / "campaign.json")


async def _pump(mcp_transport: MemoryTransport, service: SubmitService) -> asyncio.Task:
    async def pump() -> None:
        async for message in mcp_transport.receive():
            service.on_message(message)

    return asyncio.create_task(pump())


def _runtime(config: Config):
    mcp_t, sim_t = MemoryTransport.create_pair()
    server, runtime = build_server(config, SIM)
    runtime._transport = mcp_t
    return mcp_t, sim_t, server, runtime


async def test_outcome_unknown_defers_then_reconciles(tmp_path) -> None:
    """A lost ack leaves the leaf planned; the retry re-acks and links it (C2)."""
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    mcp_t, sim_t, server, runtime = _runtime(config)

    seen_cmd_ids: list[str] = []

    async def responder() -> None:
        counter = 0
        async for command in sim_t.receive():
            if command.type != SimulationType.COMMAND:
                continue
            cmd_id = str(command.payload.get("cmd_id", ""))
            seen_cmd_ids.append(cmd_id)
            counter += 1
            # First response for the cmd_id is a pending idempotency record: the
            # command was accepted but the outcome is unknown (a lost ack).  The
            # ack names the sim even though the outcome is unknown.  The retry
            # (same cmd_id) is the completed record re-acking the running job.
            if seen_cmd_ids.count(cmd_id) == 1:
                ack = build_submit_ack(
                    sim=SIM,
                    seq=counter,
                    cmd_id=cmd_id,
                    sim_id="stranded1",
                    state=SimulationState.FAILED,
                    error="already_submitted:outcome_unknown",
                    error_code="outcome_unknown",
                    in_reply_to=command.transport_event_id,
                )
            else:
                ack = build_submit_ack(
                    sim=SIM,
                    seq=counter,
                    cmd_id=cmd_id,
                    sim_id="stranded1",
                    state=SimulationState.ACCEPTED,
                    in_reply_to=command.transport_event_id,
                )
            await sim_t.send(ack.sign(SECRET))

    responder_task = asyncio.create_task(responder())
    pump_task = await _pump(mcp_t, runtime.submit_service)
    try:
        first = (await server.call_tool("advance_agenda", {})).structured_content
        # Deferred, not failed: the lost ack is not a physics failure.
        assert first["submitted"] == []
        assert first["failed"] == []
        assert first["deferred"] == ["leaf0"]
        status = (await server.call_tool("agenda_status", {})).structured_content
        assert status["deferred"] == ["leaf0"]
        (leaf,) = status["leaves"]
        assert leaf["status"] == "planned"
        assert leaf["deferred"] is True
        # The ack named the sim, so cleanup can still reach it.
        assert leaf["sim_id"] == "stranded1"

        second = (await server.call_tool("advance_agenda", {})).structured_content
        assert second["submitted"] == ["leaf0"]
        assert second["deferred"] == []
        assert second["failed"] == []
        # Both ticks used the same exactly-once command id.
        assert len(seen_cmd_ids) == 2
        assert seen_cmd_ids[0] == seen_cmd_ids[1]
    finally:
        for task in (responder_task, pump_task):
            task.cancel()
        await mcp_t.close()
        await sim_t.close()


async def test_genuine_rejection_still_fails_terminally(tmp_path) -> None:
    """A policy rejection is terminal and carries its code (not treated as transient)."""
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    mcp_t, sim_t, server, runtime = _runtime(config)

    async def responder() -> None:
        counter = 0
        async for command in sim_t.receive():
            if command.type != SimulationType.COMMAND:
                continue
            counter += 1
            ack = build_submit_ack(
                sim=SIM,
                seq=counter,
                cmd_id=str(command.payload.get("cmd_id", "")),
                sim_id="ignored01",
                state=SimulationState.FAILED,
                error="rejected_by_policy",
                error_code="rejected_by_policy",
                in_reply_to=command.transport_event_id,
            )
            await sim_t.send(ack.sign(SECRET))

    responder_task = asyncio.create_task(responder())
    pump_task = await _pump(mcp_t, runtime.submit_service)
    try:
        result = (await server.call_tool("advance_agenda", {})).structured_content
        assert result["deferred"] == []
        assert result["failed"] == ["leaf0"]
        status = (await server.call_tool("agenda_status", {})).structured_content
        (leaf,) = status["leaves"]
        assert leaf["status"] == "failed"
        assert leaf["error_code"] == "rejected_by_policy"
    finally:
        for task in (responder_task, pump_task):
            task.cancel()
        await mcp_t.close()
        await sim_t.close()
