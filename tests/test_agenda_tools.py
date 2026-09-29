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
from pic_agentic.server.agenda import NO_CAMPAIGN_MESSAGE
from pic_agentic.server.app import build_server
from pic_agentic.server.simulation import SubmitService
from pic_agentic.transport.memory import MemoryTransport

SECRET = new_secret_hex()
SIM = "7f3a2b1c"

#: The actionable ``no_campaign`` soft error every campaign tool returns.
NO_CAMPAIGN = {"ok": False, "error": "no_campaign", "message": NO_CAMPAIGN_MESSAGE}

#: A minimal (allow-list-clean) Runner spec: the wire payload is ``{"sim": ...}``.
_SPEC = {"sim": {"time_steps": 4}}


def _campaign_file(tmp_path: Path, *, name: str = "campaign", leaves: int = 1) -> str:
    """Write a campaign file with ``leaves`` specs and return its path."""
    agenda = AgendaGroup(name="group")
    for index in range(leaves):
        leaf = f"leaf{index}"
        # Distinct specs: identical payloads map to the same sim_id.
        spec = {"sim": {**_SPEC["sim"], "replica": index}}
        agenda = agenda.add(**{leaf: AgendaSim(name=leaf, spec=spec)})
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
    assert {
        "advance_agenda",
        "agenda_status",
        "approve_agenda_leaf",
        "take_agenda_callbacks",
        "add_agenda_leaf",
    } <= set(tools)

    advance = tools["advance_agenda"].annotations
    assert advance is not None
    assert advance.read_only_hint is False
    assert advance.destructive_hint is False
    assert advance.idempotent_hint is False

    status = tools["agenda_status"].annotations
    assert status is not None
    assert status.read_only_hint is True

    approve = tools["approve_agenda_leaf"].annotations
    assert approve is not None
    assert approve.read_only_hint is False
    assert approve.destructive_hint is False

    callbacks = tools["take_agenda_callbacks"].annotations
    assert callbacks is not None
    assert callbacks.read_only_hint is False


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
    assert advance == NO_CAMPAIGN
    status = await _call(config, "agenda_status", {})
    assert status == NO_CAMPAIGN


async def test_no_campaign_error_is_actionable(tmp_path) -> None:
    """Every campaign tool's ``no_campaign`` error tells the caller how to recover."""
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "missing.json"))
    for tool, arguments in (
        ("advance_agenda", {}),
        ("agenda_status", {}),
        ("pause_agenda", {}),
        ("add_agenda_leaf", {"name": "x", "spec": {"sim": {"replica": 0}}}),
        ("take_agenda_callbacks", {}),
    ):
        result = await _call(config, tool, arguments)
        assert result["error"] == "no_campaign", tool
        assert result["message"] == NO_CAMPAIGN_MESSAGE, tool
        assert "create_campaign" in result["message"], tool
        assert "create a campaign first" in result["message"], tool


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


async def test_require_approval_gate_is_wired_from_config(tmp_path) -> None:
    """``agenda_require_approval`` holds leaves until ``approve_agenda_leaf``."""
    config = Config(
        rcp_secret=SECRET,
        agenda_file=_campaign_file(tmp_path, leaves=1),
        agenda_require_approval=True,
    )
    gated = await _call(config, "advance_agenda", {})
    assert gated["submitted"] == []
    assert gated["pending_approval"] == ["leaf0"]

    approved = await _call(config, "approve_agenda_leaf", {"path": "leaf0"})
    assert approved == {"ok": True, "path": "leaf0", "approved": True}

    released = await _call(config, "advance_agenda", {})
    assert released["submitted"] == ["leaf0"]


async def test_approve_unknown_leaf_is_a_soft_error(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    result = await _call(config, "approve_agenda_leaf", {"path": "nope"})
    assert result == {"ok": False, "error": "no_such_leaf", "path": "nope"}


async def test_approve_without_campaign_is_a_soft_error(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "missing.json"))
    result = await _call(config, "approve_agenda_leaf", {"path": "leaf0"})
    assert result == NO_CAMPAIGN


async def test_agenda_status_reports_approval_flags(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    await _call(config, "approve_agenda_leaf", {"path": "leaf0"})
    status = await _call(config, "agenda_status", {})
    leaf = status["leaves"][0]
    assert leaf["approved"] is True
    assert leaf["requires_approval"] is False


async def test_concurrent_advance_does_not_duplicate_submissions(tmp_path) -> None:
    """Two overlapping ticks must not both submit the same planned leaves.

    ``AgendaService.advance`` serialises on a lock: without it, two ticks could
    load the same campaign and each submit every planned leaf.
    """
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path, leaves=2))
    mcp_t, sim_t = MemoryTransport.create_pair()
    server, runtime = build_server(config, SIM)
    runtime._transport = mcp_t
    tasks = [await _serve(sim_t), await _pump(mcp_t, runtime.submit_service)]
    try:
        first, second = await asyncio.gather(
            server.call_tool("advance_agenda", {}),
            server.call_tool("advance_agenda", {}),
        )
        submitted = first.structured_content["submitted"] + second.structured_content["submitted"]
        assert len(submitted) == 2  # each leaf exactly once, not four
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()


async def test_take_callbacks_drains_durably(tmp_path) -> None:
    """Callbacks emitted by a tick are returned once, then cleared on disk."""
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    mcp_t, sim_t = MemoryTransport.create_pair()
    server, runtime = build_server(config, SIM)
    runtime._transport = mcp_t
    tasks = [await _serve(sim_t), await _pump(mcp_t, runtime.submit_service)]
    try:
        await server.call_tool("advance_agenda", {})
        # The fake responder acked sim0001; drive it terminal so the next tick
        # folds a done transition and emits a callback.
        sim_id = next(iter(runtime.submit_service.registry))
        runtime.submit_service.registry[sim_id].state = "results.ready"
        runtime.submit_service.registry[sim_id].active = False

        tick = (await server.call_tool("advance_agenda", {})).structured_content
        assert [c["path"] for c in tick["callbacks"]] == ["leaf0"]

        drained = (await server.call_tool("take_agenda_callbacks", {})).structured_content
        assert [c["path"] for c in drained["callbacks"]] == ["leaf0"]
        # A second call is empty: the clear persisted.
        again = (await server.call_tool("take_agenda_callbacks", {})).structured_content
        assert again == {"ok": True, "callbacks": []}
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()


async def test_add_leaf_is_submitted_by_the_next_tick(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    added = await _call(config, "add_agenda_leaf", {"name": "refined", "spec": {"sim": {"replica": 9}}})
    assert added == {"ok": True, "path": "refined"}
    tick = await _call(config, "advance_agenda", {})
    assert "refined" in tick["submitted"]


async def test_add_duplicate_leaf_is_a_soft_error(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    result = await _call(config, "add_agenda_leaf", {"name": "leaf0", "spec": {"sim": {"replica": 1}}})
    assert result["ok"] is False
    assert result["error"] == "duplicate_leaf"


async def test_add_leaf_without_campaign_is_a_soft_error(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "missing.json"))
    result = await _call(config, "add_agenda_leaf", {"name": "x", "spec": {"sim": {"replica": 0}}})
    assert result == NO_CAMPAIGN


async def test_pause_and_resume_tools(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    assert await _call(config, "pause_agenda", {}) == {"ok": True, "state": "paused"}
    # A paused tick holds the ready leaf.
    held = await _call(config, "advance_agenda", {})
    assert held["submitted"] == []
    assert held["held"] == ["leaf0"]
    assert held["state"] == "paused"
    # Resume and it submits.
    assert await _call(config, "resume_agenda", {}) == {"ok": True, "state": "running"}
    tick = await _call(config, "advance_agenda", {})
    assert tick["submitted"] == ["leaf0"]


async def test_pause_without_campaign_is_a_soft_error(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "missing.json"))
    assert await _call(config, "pause_agenda", {}) == NO_CAMPAIGN
