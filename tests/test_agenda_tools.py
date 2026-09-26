# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Offline tests for the server-side campaign-agenda tools.

A tiny fake responder acks ``submit_simulation`` commands over
``MemoryTransport``; the campaign is a real file produced by ``AgendaStore``, so
the tools run the full durable engine path without a cluster.
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

#: A minimal (allow-list-clean) Runner spec: the wire payload is ``{"sim": ...}``.
_SPEC = {"sim": {"time_steps": 4}}


def _campaign_file(tmp_path: Path, *, name: str = "campaign", leaves: int = 1) -> str:
    """Write a campaign file with ``leaves`` specs and return its path."""
    agenda = AgendaGroup(name="group")
    for index in range(leaves):
        leaf = f"leaf{index}"
        agenda = agenda.add(**{leaf: AgendaSim(name=leaf, spec=_SPEC)})
    path = tmp_path / "campaign.json"
    AgendaStore(tmp_path, filename="campaign.json").save(Campaign(name=name, agenda=agenda))
    return str(path)


async def _serve(sim_transport: MemoryTransport) -> asyncio.Task:
    """Ack each submit command with a fresh ``sim_id`` until cancelled."""

    async def responder() -> None:
        counter = 0
        async for command in sim_transport.receive():
            if command.type != SimulationType.COMMAND:
                continue
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

    return asyncio.create_task(responder())


async def _pump(mcp_transport: MemoryTransport, service: SubmitService) -> asyncio.Task:
    async def pump() -> None:
        async for message in mcp_transport.receive():
            service.on_message(message)

    return asyncio.create_task(pump())


async def _call(config: Config, name: str, arguments: dict[str, object]):
    mcp_t, sim_t = MemoryTransport.create_pair()
    server, runtime = build_server(config, SIM)
    runtime._transport = mcp_t
    tasks = [await _serve(sim_t), await _pump(mcp_t, runtime.submit_service)]
    try:
        return (await server.call_tool(name, arguments)).structured_content
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()


async def test_tool_registration_and_annotations() -> None:
    server, _runtime = build_server(Config(rcp_secret=SECRET), SIM)
    tools = {tool.name: tool for tool in await server.list_tools()}
    assert {"advance_agenda", "agenda_status"} <= set(tools)

    advance = tools["advance_agenda"].annotations
    assert advance is not None
    assert advance.read_only_hint is False
    assert advance.destructive_hint is False
    assert advance.idempotent_hint is False

    status = tools["agenda_status"].annotations
    assert status is not None
    assert status.read_only_hint is True


async def test_advance_agenda_submits_and_returns_a_tick(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path, leaves=2))
    payload = await _call(config, "advance_agenda", {})
    assert payload["submitted"] == ["leaf0", "leaf1"]
    assert payload["waiting"] == []
    assert payload["complete"] is False
    assert payload["usage"]["jobs_submitted"] == 2

    # The tick is durable: a second call must not resubmit the recorded leaves.
    second = await _call(config, "advance_agenda", {})
    assert second["submitted"] == []


async def test_agenda_status_shape(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path, name="scan", leaves=2))
    await _call(config, "advance_agenda", {})
    status = await _call(config, "agenda_status", {})
    assert status["name"] == "scan"
    assert status["complete"] is False
    assert status["counts"]["submitted"] == 2
    assert {leaf["path"] for leaf in status["leaves"]} == {"leaf0", "leaf1"}
    assert "usage" in status


async def test_no_campaign_is_a_soft_error(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "missing.json"))
    advance = await _call(config, "advance_agenda", {})
    assert advance == {"ok": False, "error": "no_campaign"}
    status = await _call(config, "agenda_status", {})
    assert status == {"ok": False, "error": "no_campaign"}


async def test_agenda_status_redacts_secrets(tmp_path) -> None:
    secret = "topsecret-token"
    config = Config(rcp_secret=secret, agenda_file=_campaign_file(tmp_path, name=f"campaign-{secret}"))
    status = await _call(config, "agenda_status", {})
    assert secret not in status["name"]
    assert config.redact(secret) in status["name"]


async def test_advance_without_transport_is_a_soft_error(tmp_path) -> None:
    # advance needs the transport to submit; it must degrade, not raise.
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    server, _runtime = build_server(config, SIM)
    payload = (await server.call_tool("advance_agenda", {})).structured_content
    assert payload == {"ok": False, "error": "unavailable"}


async def test_status_works_without_a_transport(tmp_path) -> None:
    # agenda_status reads the local campaign file only, so it needs no transport.
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    server, _runtime = build_server(config, SIM)
    payload = (await server.call_tool("agenda_status", {})).structured_content
    assert payload["name"] == "campaign"
    assert payload["counts"]["planned"] == 1
