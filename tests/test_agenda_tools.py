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
import json
from pathlib import Path

import pytest

from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.model import AgendaGroup, AgendaSim
from pic_agentic.agenda.reuse import ReuseRegistry
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

#: Full ``Runner`` dumps a beta agent passed to ``create_campaign``: indices 0
#: and 1 store the ``num_tmp_field_slots`` computed field one level too deep,
#: index 2 is a known round-tripping (valid) spec.  ``create_campaign`` replays
#: the simclient's round-trip gate, so a leaf must validate against the pinned
#: schema; the minimal ``_SPEC`` above is allow-list-clean but not a valid
#: ``Runner`` and is only for the store-level tests.
CAMPAIGN_SPECS = Path(__file__).parent / "fixtures" / "campaign_specs.json"


def _campaign_spec(index: int) -> dict:
    return json.loads(CAMPAIGN_SPECS.read_text(encoding="utf-8"))[index]


def _valid_spec() -> dict:
    """Return a deep copy of the known round-tripping (valid) campaign spec."""
    return _campaign_spec(2)


def _add_spec() -> dict:
    """Return a valid leaf spec with a distinct payload for ``add_agenda_leaf``.

    ``add_agenda_leaf`` validates through the submission path (allow-list +
    pinned-schema round-trip + inline cap), so the minimal ``_SPEC`` used by the
    store-level tests is rejected under the real pin.  Bumping ``time_steps``
    keeps the payload distinct from the seeded ``leaf0`` while remaining a valid
    ``Runner`` dump.
    """
    base = _valid_spec()
    base["sim"]["time_steps"] += 1
    return base


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
        "create_campaign",
        "delete_campaign",
    } <= set(tools)

    delete = tools["delete_campaign"].annotations
    assert delete is not None
    assert delete.read_only_hint is False
    assert delete.destructive_hint is True
    assert delete.idempotent_hint is False
    assert set(tools["delete_campaign"].input_schema["properties"]) == {"force"}

    create = tools["create_campaign"].annotations
    assert create is not None
    assert create.read_only_hint is False
    assert create.destructive_hint is False
    # The frozen signature the LLM sees.
    assert set(tools["create_campaign"].input_schema["properties"]) == {
        "name",
        "base_spec",
        "base_spec_path",
        "patch_path",
        "values",
        "parameter",
        "point_key",
    }
    # ``patch_path``/``values`` are optional: omitting both creates an empty
    # campaign to be populated with whole-spec leaves.
    assert set(tools["create_campaign"].input_schema["required"]) == {"name"}

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


async def test_agenda_status_flags_a_successfully_empty_leaf(tmp_path) -> None:
    """A done leaf whose output is all-zero is flagged, not reported clean (F4)."""
    warning = "energy_histogram is all zeros; the run may have no particles in range"
    agenda = AgendaGroup(name="group").add(
        leaf=AgendaSim(name="leaf", spec={"sim": {"replica": 0}}, status="done", sim_id="sim0001", suspect=warning)
    )
    AgendaStore(tmp_path, filename="campaign.json").save(Campaign(name="scan", agenda=agenda))
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))

    status = await _call(config, "agenda_status", {})
    assert status["suspect_count"] == 1
    assert status["suspects"] == {"leaf": warning}
    (leaf,) = status["leaves"]
    assert leaf["suspect"] == warning


async def test_agenda_status_does_not_flag_a_populated_leaf(tmp_path) -> None:
    agenda = AgendaGroup(name="group").add(
        leaf=AgendaSim(name="leaf", spec={"sim": {"replica": 0}}, status="done", sim_id="sim0001")
    )
    AgendaStore(tmp_path, filename="campaign.json").save(Campaign(name="scan", agenda=agenda))
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))

    status = await _call(config, "agenda_status", {})
    assert status["suspect_count"] == 0
    assert status["leaves"][0]["suspect"] is None


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


async def test_failed_leaf_surfaces_the_simclient_reason(tmp_path) -> None:
    """A rejected submission exposes error/error_code in callbacks and status (P2a).

    The offline responder rejects the submit with the simclient's
    ``unsupported`` ack; the campaign must surface *why* the leaf failed, not a
    bare ``failed`` with a null sim_id.
    """
    agenda = AgendaGroup(name="g")
    agenda = agenda.add(leaf0=AgendaSim(name="leaf0", spec={"sim": {"replica": 0}}))
    AgendaStore(tmp_path, filename="campaign.json").save(Campaign(name="c", agenda=agenda))
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    mcp_t, sim_t = MemoryTransport.create_pair()
    server, runtime = build_server(config, SIM)
    runtime._transport = mcp_t

    async def rejecting() -> None:
        counter = 0
        async for command in sim_t.receive():
            if command.type != SimulationType.COMMAND:
                continue
            counter += 1
            ack = build_submit_ack(
                sim=SIM,
                seq=counter,
                cmd_id=str(command.payload.get("cmd_id", "")),
                sim_id="sim0001",
                state=SimulationState.FAILED,
                in_reply_to=command.transport_event_id,
                error="simulation carries fields outside the pinned pypicongpu schema",
                error_code="unsupported",
            ).sign(SECRET)
            await sim_t.send(ack)

    tasks = [asyncio.create_task(rejecting()), await _pump(mcp_t, runtime.submit_service)]
    try:
        tick = (await server.call_tool("advance_agenda", {})).structured_content
        assert tick["failed"] == ["leaf0"]
        (callback,) = tick["callbacks"]
        assert callback["error"] == "simulation carries fields outside the pinned pypicongpu schema"
        assert callback["error_code"] == "unsupported"

        status = (await server.call_tool("agenda_status", {})).structured_content
        leaf = next(item for item in status["leaves"] if item["path"] == "leaf0")
        assert leaf["status"] == "failed"
        assert leaf["error_code"] == "unsupported"

        drained = (await server.call_tool("take_agenda_callbacks", {})).structured_content
        assert drained["callbacks"][0]["error_code"] == "unsupported"
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()


async def test_identical_leaf_failures_are_grouped_and_truncated(tmp_path) -> None:
    """Several leaves failing the same way yield one bounded group + digest.

    The regression: three leaves each carried an identical multi-KB validation
    dump inline.  The tick must name the shared reason once (bounded) and list
    the affected paths, while the full text stays on the persisted callback.
    """
    agenda = AgendaGroup(name="g")
    for index in range(3):
        leaf = f"leaf{index}"
        agenda = agenda.add(**{leaf: AgendaSim(name=leaf, spec={"sim": {"replica": index}})})
    AgendaStore(tmp_path, filename="campaign.json").save(Campaign(name="c", agenda=agenda))
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    long_reason = "rejected field: " + "x" * 4000
    mcp_t, sim_t = MemoryTransport.create_pair()
    server, runtime = build_server(config, SIM)
    runtime._transport = mcp_t

    async def rejecting() -> None:
        counter = 0
        async for command in sim_t.receive():
            if command.type != SimulationType.COMMAND:
                continue
            counter += 1
            ack = build_submit_ack(
                sim=SIM,
                seq=counter,
                cmd_id=str(command.payload.get("cmd_id", "")),
                sim_id=f"sim{counter:04d}",
                state=SimulationState.FAILED,
                in_reply_to=command.transport_event_id,
                error=long_reason,
                error_code="unsupported",
            ).sign(SECRET)
            await sim_t.send(ack)

    tasks = [asyncio.create_task(rejecting()), await _pump(mcp_t, runtime.submit_service)]
    try:
        tick = (await server.call_tool("advance_agenda", {})).structured_content
        assert tick["failed"] == ["leaf0", "leaf1", "leaf2"]
        assert len(tick["callbacks"]) == 3
        expected = long_reason[:500] + "...(truncated)"
        for callback in tick["callbacks"]:
            assert callback["error"] == expected
        (group,) = tick["failure_groups"]
        assert group["error_code"] == "unsupported"
        assert group["paths"] == ["leaf0", "leaf1", "leaf2"]
        assert group["message"].endswith("...(truncated)")
        assert len(group["message"]) <= 500 + len("...(truncated)")
        assert tick["failure_summary"].startswith("3 leaves failed: unsupported: rejected field:")

        # The full reason survives on the persisted callback, not just the tick.
        drained = (await server.call_tool("take_agenda_callbacks", {})).structured_content
        assert drained["callbacks"][0]["error"] == long_reason
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()


async def test_advance_flags_a_successfully_empty_campaign(tmp_path) -> None:
    """The beta-4 scenario: a done leaf with zero physics is reported, not clean.

    Drives one submitted leaf to ``results.ready`` with the registry carrying the
    all-zero health flag, then asserts the tick result, its done callback,
    ``agenda_status`` and ``fleet_status`` all surface it.
    """
    warning = "energy_histogram is all zeros; the run may have no particles in range"
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    mcp_t, sim_t = MemoryTransport.create_pair()
    server, runtime = build_server(config, SIM)
    runtime._transport = mcp_t
    tasks = [await _serve(sim_t), await _pump(mcp_t, runtime.submit_service)]
    try:
        await server.call_tool("advance_agenda", {})
        sim_id = next(iter(runtime.submit_service.registry))
        record = runtime.submit_service.registry[sim_id]
        record.state = "results.ready"
        record.active = False
        record.suspect = warning

        tick = (await server.call_tool("advance_agenda", {})).structured_content
        assert tick["done"] == ["leaf0"]
        assert tick["suspects"] == {"leaf0": warning}
        (callback,) = tick["callbacks"]
        assert callback["kind"] == "done"
        assert callback["suspect"] == warning
        assert callback["error"] is None

        status = (await server.call_tool("agenda_status", {})).structured_content
        assert status["suspect_count"] == 1
        assert status["suspects"] == {"leaf0": warning}
        assert status["leaves"][0]["suspect"] == warning

        fleet = (await server.call_tool("fleet_status", {})).structured_content
        assert fleet["summary"]["suspect"] == 1
        assert [alert["kind"] for alert in fleet["alerts"]] == ["suspect"]
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


async def test_advance_agenda_reports_complete_state_when_finished(tmp_path) -> None:
    """A completed tick says ``state: "complete"``, not ``"running"``."""
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    mcp_t, sim_t = MemoryTransport.create_pair()
    server, runtime = build_server(config, SIM)
    runtime._transport = mcp_t
    tasks = [await _serve(sim_t), await _pump(mcp_t, runtime.submit_service)]
    try:
        first = (await server.call_tool("advance_agenda", {})).structured_content
        assert first["complete"] is False
        assert first["state"] == "running"
        assert first["lifecycle"] == "running"

        sim_id = next(iter(runtime.submit_service.registry))
        runtime.submit_service.registry[sim_id].state = "results.ready"
        runtime.submit_service.registry[sim_id].active = False

        done = (await server.call_tool("advance_agenda", {})).structured_content
        assert done["complete"] is True
        assert done["state"] == "complete"
        # The stored lifecycle is still reported, so completion is not lossy.
        assert done["lifecycle"] == "running"
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()


async def test_callback_tool_descriptions_explain_the_double_exposure() -> None:
    """The inline vs. drained callback contract is documented on the tools."""
    server, _runtime = build_server(Config(rcp_secret=SECRET), SIM)
    tools = {tool.name: tool for tool in await server.list_tools()}

    advance = tools["advance_agenda"].description
    assert "take_agenda_callbacks" in advance
    assert "drain" in advance

    drain = tools["take_agenda_callbacks"].description
    assert "advance_agenda" in drain
    assert "destructive" in drain
    assert "empty list" in drain


async def test_delete_campaign_description_notes_the_registry_survives() -> None:
    """Deleting a campaign must not be read as forgetting its simulations."""
    server, _runtime = build_server(Config(rcp_secret=SECRET), SIM)
    tools = {tool.name: tool for tool in await server.list_tools()}
    description = tools["delete_campaign"].description
    assert "fleet registry" in description
    assert "list_simulations" in description
    assert "registry" in tools["list_simulations"].description


async def test_add_leaf_is_submitted_by_the_next_tick(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    added = await _call(config, "add_agenda_leaf", {"name": "refined", "spec": _add_spec()})
    assert added == {"ok": True, "path": "refined"}
    tick = await _call(config, "advance_agenda", {})
    assert "refined" in tick["submitted"]


async def test_add_duplicate_leaf_is_a_soft_error(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    result = await _call(config, "add_agenda_leaf", {"name": "leaf0", "spec": _add_spec()})
    assert result["ok"] is False
    assert result["error"] == "duplicate_leaf"


async def test_add_duplicate_leaf_wins_over_an_invalid_spec(tmp_path) -> None:
    """A duplicate name is reported as ``duplicate_leaf`` even if the spec is invalid."""
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    result = await _call(config, "add_agenda_leaf", {"name": "leaf0", "spec": {"sim": {"replica": 9}}})
    assert result["ok"] is False
    assert result["error"] == "duplicate_leaf"


async def test_add_leaf_refuses_an_invalid_spec(tmp_path) -> None:
    """``add_agenda_leaf`` validates through the submission path at add time.

    A spec that is not an allow-listed ``{"sim": ...}`` wire is refused in both
    the offline and real-pin venvs (the pinned-schema round-trip is a no-op
    without the pin), so it must never be persisted and only fail at submit.
    """
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    result = await _call(config, "add_agenda_leaf", {"name": "bad", "spec": {"not_sim": 1}})
    assert result["ok"] is False
    assert result["error"] == "invalid_campaign_spec"
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    assert "bad" not in campaign.agenda.entries


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


async def test_create_campaign_builds_leaves_with_points_and_values(tmp_path) -> None:
    """create_campaign mirrors the driver: one patched leaf + point per value."""
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    base = _valid_spec()
    grid = base["sim"]["grid"]
    original_time_steps = base["sim"]["time_steps"]
    result = await _call(
        config,
        "create_campaign",
        {"name": "scan", "base_spec": base, "patch_path": "sim.time_steps", "values": [50, 100, 200]},
    )
    assert result == {"ok": True, "name": "scan", "leaves": ["leaf000", "leaf001", "leaf002"]}

    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    assert campaign.name == "scan"
    assert campaign.created_ts is not None
    points = {name: leaf.point for name, leaf in campaign.agenda.entries.items()}
    assert points == {
        "leaf000": {"time_steps": 50},
        "leaf001": {"time_steps": 100},
        "leaf002": {"time_steps": 200},
    }
    for leaf in campaign.agenda.entries.values():
        # The nested siblings survive and the patch path value agrees with point.
        assert leaf.spec["sim"]["time_steps"] == leaf.point["time_steps"]
        assert leaf.spec["sim"]["grid"] == grid
    # The base spec is not mutated by the per-leaf patch.
    assert base["sim"]["time_steps"] == original_time_steps


async def test_create_campaign_creates_a_loadable_campaign_the_tick_submits(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    base = _valid_spec()
    await _call(
        config,
        "create_campaign",
        {"name": "scan", "base_spec": base, "patch_path": "sim.time_steps", "values": [50, 100]},
    )
    tick = await _call(config, "advance_agenda", {})
    assert tick["submitted"] == ["leaf000", "leaf001"]
    status = await _call(config, "agenda_status", {})
    assert status["name"] == "scan"
    assert {leaf["path"] for leaf in status["leaves"]} == {"leaf000", "leaf001"}


async def test_create_campaign_without_values_is_a_soft_error(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    result = await _call(
        config,
        "create_campaign",
        {"name": "scan", "base_spec": {"sim": {"time_steps": 4}}, "patch_path": "sim.time_steps", "values": []},
    )
    assert result == {"ok": False, "error": "no_values"}


async def test_create_empty_campaign_is_not_complete_and_takes_whole_spec_leaves(tmp_path) -> None:
    """A whole-spec-only study: create empty, then add whole-spec leaves (F7).

    This is the clean path the beta-7 agent had to fake with an identity patch.
    The empty campaign persists with no leaves and reports ``empty: true`` /
    ``complete: false`` (so "no work yet" is not mistaken for "finished"), then
    ``add_agenda_leaf`` populates it with independent whole specs, and the next
    tick submits them.
    """
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    created = await _call(config, "create_campaign", {"name": "whole-specs"})
    assert created == {"ok": True, "name": "whole-specs", "leaves": [], "empty": True}

    # Before any leaf exists, neither status nor a tick claims the work is done.
    status = await _call(config, "agenda_status", {})
    assert status["empty"] is True
    assert status["complete"] is False
    assert status["counts"] == {"planned": 0, "submitted": 0, "running": 0, "done": 0, "failed": 0}
    tick = await _call(config, "advance_agenda", {})
    assert tick["empty"] is True
    assert tick["complete"] is False
    # ``state`` must not contradict the forced ``complete: false``.
    assert tick["state"] == "running"
    assert tick["submitted"] == []

    # Populate with whole-spec leaves that vary as many nodes as they like.
    for name, spec in (("s1", _valid_spec()), ("s2", _add_spec())):
        added = await _call(config, "add_agenda_leaf", {"name": name, "spec": spec})
        assert added == {"ok": True, "path": name}

    after = await _call(config, "agenda_status", {})
    assert after["complete"] is False
    assert "empty" not in after
    assert {leaf["path"] for leaf in after["leaves"]} == {"s1", "s2"}

    submitted = await _call(config, "advance_agenda", {})
    assert submitted["submitted"] == ["s1", "s2"]
    assert submitted["complete"] is False


async def test_added_whole_spec_leaf_carries_no_derived_point(tmp_path) -> None:
    """A whole-spec leaf records exactly the point given, never a derived one.

    A misleading point derived from an unrelated patch must not appear (F7/A2).
    """
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    await _call(config, "create_campaign", {"name": "whole-specs"})
    # No point: the leaf must stay point-less and label-less.
    await _call(config, "add_agenda_leaf", {"name": "bare", "spec": _valid_spec()})
    # Explicit point/parameter: recorded verbatim.
    await _call(
        config,
        "add_agenda_leaf",
        {"name": "labelled", "spec": _add_spec(), "point": {"resolution_scale": 2}, "parameter": "resolution scale"},
    )
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    bare = campaign.agenda.entries["bare"]
    assert bare.point is None
    assert bare.sweep_parameter is None
    labelled = campaign.agenda.entries["labelled"]
    assert labelled.point == {"resolution_scale": 2}
    assert labelled.sweep_parameter == "resolution scale"


async def test_create_empty_campaign_rejects_stray_values_or_base(tmp_path) -> None:
    """An empty campaign is created bare: values or a base spec are refused."""
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    stray_values = await _call(config, "create_campaign", {"name": "x", "values": [1]})
    assert stray_values["ok"] is False
    assert stray_values["error"] == "invalid_campaign"
    assert not (tmp_path / "campaign.json").exists()

    with_patch_no_values = await _call(config, "create_campaign", {"name": "x", "patch_path": "sim.time_steps"})
    assert with_patch_no_values["ok"] is False
    assert with_patch_no_values["error"] == "base_spec_required"

    with_base = await _call(config, "create_campaign", {"name": "x", "base_spec": _valid_spec()})
    assert with_base["ok"] is False
    assert with_base["error"] == "invalid_campaign"
    assert not (tmp_path / "campaign.json").exists()


async def test_create_campaign_refuses_to_clobber_an_existing_campaign(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    result = await _call(
        config,
        "create_campaign",
        {"name": "other", "base_spec": {"sim": {"time_steps": 4}}, "patch_path": "sim.time_steps", "values": [1]},
    )
    assert result == {"ok": False, "error": "campaign_exists"}


async def test_create_campaign_patches_a_list_indexed_path(tmp_path) -> None:
    """A numeric patch-path segment indexes a list (real specs are list-shaped).

    The focal-position sweep patches ``sim.laser.0.focus_pos_si.1.component``:
    ``sim.laser`` and ``focus_pos_si`` are lists, so the patcher must descend
    through list indices and leave the nested siblings intact.
    """
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    base = _valid_spec()
    focus = base["sim"]["laser"][0]["focus_pos_si"]
    original = focus[1]["component"]
    siblings = (focus[0]["component"], focus[2]["component"])
    result = await _call(
        config,
        "create_campaign",
        {
            "name": "focal",
            "base_spec": base,
            "patch_path": "sim.laser.0.focus_pos_si.1.component",
            "values": [4.4e-5, 4.8e-5],
        },
    )
    assert result == {"ok": True, "name": "focal", "leaves": ["leaf000", "leaf001"]}

    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    expected = {"leaf000": 4.4e-5, "leaf001": 4.8e-5}
    for name, value in expected.items():
        leaf = campaign.agenda.entries[name]
        focus = leaf.spec["sim"]["laser"][0]["focus_pos_si"]
        assert focus[1]["component"] == pytest.approx(value)
        # The point names the last path segment and agrees with the patched spec.
        assert leaf.point == {"component": value}
        # But the sweep is self-describing: the readable name drops the list
        # indices while keeping every field name.
        assert leaf.sweep_parameter == "sim.laser.focus_pos_si.component"
        # The nested siblings survive untouched.
        assert focus[0]["component"] == pytest.approx(siblings[0])
        assert focus[2]["component"] == pytest.approx(siblings[1])
    # The base spec is not mutated by the per-leaf patch.
    assert base["sim"]["laser"][0]["focus_pos_si"][1]["component"] == pytest.approx(original)


async def test_create_campaign_explicit_parameter_overrides_the_derived_name(tmp_path) -> None:
    """An explicit ``parameter`` label wins over the patch-path derivation."""
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    base = _valid_spec()
    result = await _call(
        config,
        "create_campaign",
        {
            "name": "focal",
            "base_spec": base,
            "patch_path": "sim.laser.0.focus_pos_si.1.component",
            "values": [4.4e-5],
            "parameter": "focal y [m]",
        },
    )
    assert result["ok"] is True
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    leaf = campaign.agenda.entries["leaf000"]
    assert leaf.sweep_parameter == "focal y [m]"
    # The point key is unchanged: the label is recorded alongside, not instead.
    assert leaf.point == {"component": 4.4e-5}


async def test_create_campaign_derives_a_readable_name_for_a_dict_key(tmp_path) -> None:
    """A numeric dict key is kept in the readable name, not mistaken for an index."""
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    base = _valid_spec()
    base["sim"]["customuserinput"] = {"tags": ["t"], "0": "periodic"}
    result = await _call(
        config,
        "create_campaign",
        {
            "name": "bc",
            "base_spec": base,
            "patch_path": "sim.customuserinput.0",
            "values": ["open"],
        },
    )
    assert result["ok"] is True
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    assert campaign.agenda.entries["leaf000"].sweep_parameter == "sim.customuserinput.0"


async def test_create_campaign_sanitises_a_derived_name(tmp_path) -> None:
    """A caller-influenced spec key cannot smuggle a control char into a report."""
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    base = _valid_spec()
    base["sim"]["customuserinput"] = {"tags": ["t"], "bad\x00key": 1}
    result = await _call(
        config,
        "create_campaign",
        {
            "name": "c",
            "base_spec": base,
            "patch_path": "sim.customuserinput.bad\x00key",
            "values": [1],
        },
    )
    assert result["ok"] is True
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    assert campaign.agenda.entries["leaf000"].sweep_parameter == "sim.customuserinput.bad key"


async def test_create_campaign_preserves_a_non_ascii_label(tmp_path) -> None:
    """The label sanitiser keeps Unicode units instead of mangling them."""
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    result = await _call(
        config,
        "create_campaign",
        {
            "name": "focal",
            "base_spec": _valid_spec(),
            "patch_path": "sim.laser.0.focus_pos_si.1.component",
            "values": [4.4e-5],
            "parameter": "focus y [μm]",
        },
    )
    assert result["ok"] is True
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    assert campaign.agenda.entries["leaf000"].sweep_parameter == "focus y [μm]"


async def test_add_agenda_leaf_records_the_parameter_label(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    result = await _call(
        config,
        "add_agenda_leaf",
        {"name": "refined", "spec": _add_spec(), "point": {"component": 5.0}, "parameter": "focal y"},
    )
    assert result == {"ok": True, "path": "refined"}
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    leaf = campaign.agenda.entries["refined"]
    assert leaf.sweep_parameter == "focal y"
    assert leaf.point == {"component": 5.0}


async def test_old_campaign_without_sweep_parameter_still_loads(tmp_path) -> None:
    """A campaign serialised before ``sweep_parameter`` existed loads unchanged."""
    legacy = {
        "name": "old",
        "agenda": {
            "kind": "group",
            "name": "group",
            "entries": {
                "leaf0": {
                    "kind": "sim",
                    "name": "leaf0",
                    "spec": {"sim": {"time_steps": 4}},
                    "point": {"time_steps": 4},
                    "status": "planned",
                }
            },
        },
    }
    (tmp_path / "campaign.json").write_text(json.dumps(legacy), encoding="utf-8")
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    status = await _call(config, "agenda_status", {})
    assert status["leaves"][0]["point"] == {"time_steps": 4}
    assert status["leaves"][0]["sweep_parameter"] is None
    # The old campaign is still advanceable.
    tick = await _call(config, "advance_agenda", {})
    assert tick["submitted"] == ["leaf0"]


async def test_create_campaign_bad_patch_path_is_a_soft_error(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    result = await _call(
        config,
        "create_campaign",
        {"name": "scan", "base_spec": {"sim": {"time_steps": 4}}, "patch_path": "sim.missing.deeper", "values": [1]},
    )
    assert result["ok"] is False
    assert result["error"] == "invalid_campaign"
    # The failed creation persisted nothing.
    assert not (tmp_path / "campaign.json").exists()


async def test_create_campaign_out_of_range_list_index_is_a_soft_error(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    result = await _call(
        config,
        "create_campaign",
        {
            "name": "scan",
            "base_spec": {"sim": {"laser": [{"focus_pos_si": [{"component": 0.0}]}]}},
            "patch_path": "sim.laser.3.focus_pos_si.0.component",
            "values": [1.0],
        },
    )
    assert result["ok"] is False
    assert result["error"] == "invalid_campaign"
    # The failed creation persisted nothing.
    assert not (tmp_path / "campaign.json").exists()


async def test_create_campaign_omits_non_scalar_points(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    base = _valid_spec()
    species = base["sim"]["species"]
    # ``species`` is a list field, so both sweep values are non-scalar: they are
    # still applied to the spec but cannot become leaf points.
    result = await _call(
        config,
        "create_campaign",
        {
            "name": "scan",
            "base_spec": base,
            "patch_path": "sim.species",
            "values": [[species[0]], species],
        },
    )
    assert result["ok"] is True
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    assert campaign.agenda.entries["leaf000"].point is None
    assert campaign.agenda.entries["leaf000"].spec["sim"]["species"] == [species[0]]
    assert campaign.agenda.entries["leaf001"].point is None
    assert campaign.agenda.entries["leaf001"].spec["sim"]["species"] == species


async def test_create_campaign_rejects_duplicate_values(tmp_path) -> None:
    """Identical patched specs are refused up front (B2).

    They would map to one ``sim_id`` and the engine would later abort the whole
    tick with ``duplicate payload ...``; the agent must not be able to wedge a
    campaign that can never advance.
    """
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    base = _valid_spec()
    time_steps = base["sim"]["time_steps"]
    result = await _call(
        config,
        "create_campaign",
        {"name": "d", "base_spec": base, "patch_path": "sim.time_steps", "values": [time_steps, time_steps]},
    )
    assert result["ok"] is False
    assert result["error"] == "duplicate_campaign_spec"
    # Nothing was persisted, so the agent can retry with distinct values.
    assert not (tmp_path / "campaign.json").exists()


async def test_create_campaign_accepts_a_cell_cnt_only_box_size_sweep(tmp_path) -> None:
    """A lone ``cell_cnt`` patch is a legitimate box-size sweep, and warns (A1).

    The maintainer directive is explicit: "Box-size sweeps are allowed."
    Patching only ``sim.grid.cell_cnt`` holds ``cell_size`` (the resolution)
    fixed and varies the physical box -- a valid study that must be accepted and
    produce a working campaign.  Because a Runner spec is a rendered snapshot
    with denormalised fields, the result carries a non-blocking ``warnings``
    entry naming the effect, so an agent that meant a *fixed-box* resolution
    sweep is not misled by the beta-6 trap of silent box variation.
    """
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    base = _valid_spec()
    result = await _call(
        config,
        "create_campaign",
        {
            "name": "warm_conv",
            "base_spec": base,
            "patch_path": "sim.grid.cell_cnt",
            "values": [{"x": 48, "y": 48, "z": 48}, {"x": 64, "y": 64, "z": 64}],
        },
    )
    assert result["ok"] is True
    assert result["leaves"] == ["leaf000", "leaf001"]
    assert result["warnings"]
    assert "box" in result["warnings"][0].lower()
    assert "add_agenda_leaf" in result["warnings"][0]
    # The campaign really persisted with the patched cell counts.
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    assert campaign.agenda.entries["leaf000"].spec["sim"]["grid"]["cell_cnt"] == {"x": 48, "y": 48, "z": 48}
    assert campaign.agenda.entries["leaf001"].spec["sim"]["grid"]["cell_cnt"] == {"x": 64, "y": 64, "z": 64}
    # cell_size (the resolution) is untouched by a cell_cnt-only patch.
    assert campaign.agenda.entries["leaf000"].spec["sim"]["grid"]["cell_size"] == base["sim"]["grid"]["cell_size"]


async def test_create_campaign_accepts_a_sub_axis_grid_patch(tmp_path) -> None:
    """A sub-axis ``cell_cnt.x`` patch is accepted with the box-size advisory (A1).

    The sub-axis form is the same box-size sweep as the whole node: it varies one
    axis's cell count at the fixed cell size, so it must be accepted (the z
    ``cell_depth`` arithmetic invariant is untouched by an x patch).
    """
    for patch_path, value in (
        ("sim.grid.cell_cnt.x", 48),
        ("sim.grid.cell_cnt.z", 48),
    ):
        config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / f"{patch_path.replace('.', '_')}.json"))
        base = _valid_spec()
        result = await _call(
            config,
            "create_campaign",
            {"name": "subaxis", "base_spec": base, "patch_path": patch_path, "values": [value]},
        )
        assert result["ok"] is True, (patch_path, result)
        assert "warnings" in result, patch_path
        assert (tmp_path / f"{patch_path.replace('.', '_')}.json").exists(), patch_path


async def test_create_campaign_accepts_a_cell_size_only_resolution_change_with_advice(tmp_path) -> None:
    """A lone ``cell_size`` patch at a consistent ``cell_depth`` is accepted (A1).

    This is the per-axis resolution change form of a box-size sweep: here the
    base ``cell_depth`` already mirrors the patched ``cell_size.z``, so the
    arithmetic invariant holds and the change is accepted with the advisory.
    """
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    base = _valid_spec()
    # Leave cell_size.z (== the base cell_depth, 1.772e-7) unchanged so the
    # arithmetically-required cell_depth stays consistent, and enlarge the y cell
    # so the base delta_t_si remains within the Yee CFL limit.
    result = await _call(
        config,
        "create_campaign",
        {
            "name": "res",
            "base_spec": base,
            "patch_path": "sim.grid.cell_size",
            "values": [{"x": 1.4e-7, "y": 8e-8, "z": 1.772e-7}],
        },
    )
    assert result["ok"] is True, result
    assert result["warnings"]
    assert "resolution" in result["warnings"][0].lower()


async def test_create_campaign_still_refuses_a_stale_cell_depth(tmp_path) -> None:
    """A ``cell_size`` patch that leaves ``cell_depth`` stale is still refused (A1).

    Allowing box-size sweeps does not loosen the genuine arithmetic invariant:
    for a 3D grid ``cell_depth`` is the z cell length, so changing
    ``cell_size.z`` without moving ``cell_depth`` is arithmetically
    inconsistent and must be refused, not warned.  With the pin importable the
    pin's own ``cell_depth`` computed field rejects it at the round-trip gate;
    offline :func:`~pic_agentic.simulation_build.check_spec_consistency` reports
    the same stale ``cell_depth``.  Both are hard refusals.
    """
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    base = _valid_spec()
    result = await _call(
        config,
        "create_campaign",
        {
            "name": "stale_depth",
            "base_spec": base,
            "patch_path": "sim.grid.cell_size",
            "values": [{"x": 1.4e-7, "y": 8e-8, "z": 1.5e-7}],
        },
    )
    assert result["ok"] is False
    assert result["error"] == "invalid_campaign_spec"
    assert not (tmp_path / "campaign.json").exists()


async def test_create_campaign_still_refuses_a_cfl_violation(tmp_path) -> None:
    """A CFL-violating ``delta_t_si`` patch is still refused (A1).

    The CFL limit is arithmetic, not intent, so it stays a hard rejection.
    """
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    base = _valid_spec()
    result = await _call(
        config,
        "create_campaign",
        {"name": "cfl", "base_spec": base, "patch_path": "sim.delta_t_si", "values": [5.0e-15]},
    )
    assert result["ok"] is False
    assert result["error"] == "invalid_campaign_spec"
    assert "CFL" in result["detail"]
    assert not (tmp_path / "campaign.json").exists()


async def test_create_campaign_accepts_a_whole_grid_patch(tmp_path) -> None:
    """A whole ``sim.grid`` node patch is the supported single-path form (A1).

    Replacing the entire grid object moves ``cell_size``, ``cell_cnt`` and
    ``cell_depth`` together, so all derived invariants hold and the campaign is
    accepted.
    """
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    base = _valid_spec()
    coarse = dict(base["sim"]["grid"])
    coarse["cell_cnt"] = {"x": 48, "y": 48, "z": 48}
    coarse["cell_size"] = {"x": 1.25e-7, "y": 1.25e-7, "z": 1.25e-7}
    coarse["cell_depth"] = 1.25e-7
    result = await _call(
        config,
        "create_campaign",
        {"name": "fixedbox", "base_spec": base, "patch_path": "sim.grid", "values": [coarse]},
    )
    assert result == {"ok": True, "name": "fixedbox", "leaves": ["leaf000"]}
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    assert campaign.agenda.entries["leaf000"].spec["sim"]["grid"]["cell_cnt"] == {"x": 48, "y": 48, "z": 48}


async def test_create_campaign_accepts_a_consistent_multi_field_leaf(tmp_path) -> None:
    """A whole-spec leaf with co-varied box/dt/steps passes ``add_agenda_leaf`` (A1).

    This is the empirical-correct route from the beta-6 run: build the base
    campaign, then add one leaf per resolution carrying every co-varied node.
    """
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    base = _valid_spec()
    await _call(
        config,
        "create_campaign",
        {"name": "scan", "base_spec": base, "patch_path": "sim.time_steps", "values": [base["sim"]["time_steps"]]},
    )
    leaf = _valid_spec()
    leaf["sim"]["grid"]["cell_cnt"] = {"x": 48, "y": 48, "z": 48}
    leaf["sim"]["grid"]["cell_size"] = {"x": 1.25e-7, "y": 1.25e-7, "z": 1.25e-7}
    leaf["sim"]["grid"]["cell_depth"] = 1.25e-7
    # CFL: c*dt <= 1/sqrt(3)/dx for a cube; 1/sqrt(3)/1.25e-7 / c with ~0.95 margin.
    leaf["sim"]["delta_t_si"] = 2.2869269268364335e-16
    leaf["sim"]["time_steps"] = 110
    result = await _call(config, "add_agenda_leaf", {"name": "leaf_n48", "spec": leaf})
    assert result == {"ok": True, "path": "leaf_n48"}


async def test_create_campaign_accepts_an_unrelated_patch(tmp_path) -> None:
    """A normal non-grid patch is untouched by the guard (no regression) (A1).

    The P1 focal-position sweep patches a laser component; the grid, dt and
    solver are all untouched, so neither the consistency guard nor the box-size
    advisory fires.
    """
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    base = _valid_spec()
    result = await _call(
        config,
        "create_campaign",
        {
            "name": "focal",
            "base_spec": base,
            "patch_path": "sim.laser.0.focus_pos_si.1.component",
            "values": [4.4e-5, 4.8e-5],
        },
    )
    assert result["ok"] is True
    assert "warnings" not in result


async def test_create_campaign_explicit_point_key_overrides_the_derived_point(tmp_path) -> None:
    """A seed-only campaign can name its point instead of the patch field (A2).

    ``create_campaign(patch_path="sim.time_steps", values=[...])`` used only to
    seed a campaign otherwise records ``point={"time_steps": ...}`` even though
    the leaf represents a different quantity (N, resolution, ...); ``point_key``
    overrides that mislabel.
    """
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    base = _valid_spec()
    result = await _call(
        config,
        "create_campaign",
        {
            "name": "seed",
            "base_spec": base,
            "patch_path": "sim.time_steps",
            "values": [146],
            "point_key": "N",
            "parameter": "grid resolution N",
        },
    )
    assert result["ok"] is True
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    leaf = campaign.agenda.entries["leaf000"]
    assert leaf.point == {"N": 146}
    assert leaf.sweep_parameter == "grid resolution N"


async def test_create_campaign_without_point_key_keeps_the_derived_point(tmp_path) -> None:
    """``point_key`` is opt-in: the historic last-segment point is unchanged (A2)."""
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    base = _valid_spec()
    result = await _call(
        config,
        "create_campaign",
        {"name": "scan", "base_spec": base, "patch_path": "sim.time_steps", "values": [146]},
    )
    assert result["ok"] is True
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    assert campaign.agenda.entries["leaf000"].point == {"time_steps": 146}


async def test_create_campaign_rejects_a_misplaced_computed_field(tmp_path) -> None:
    """A spec whose computed field is nested fails at creation, actionably (P2b).

    ``num_tmp_field_slots`` is a ``@computed_field`` on ``collisional_physics``,
    so storing it under ``numerics_config`` is dropped on re-validation and the
    simclient rejects the leaf later as ``unsupported``.  The create-time gate
    must name the offending path instead of letting the campaign fail silently.
    """
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    result = await _call(
        config,
        "create_campaign",
        {
            "name": "bad",
            "base_spec": _campaign_spec(0),
            "patch_path": "sim.time_steps",
            "values": [100, 200],
        },
    )
    assert result["ok"] is False
    assert result["error"] == "invalid_campaign_spec"
    assert "collisional_physics.numerics_config.num_tmp_field_slots" in result["detail"]
    assert "collisional_physics.num_tmp_field_slots" in result["detail"]
    # The malformed campaign was not persisted.
    assert not (tmp_path / "campaign.json").exists()


async def test_create_campaign_accepts_a_build_spec_shaped_spec(tmp_path) -> None:
    """A spec whose computed field is at the correct level works unchanged (P2b)."""
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    result = await _call(
        config,
        "create_campaign",
        {
            "name": "good",
            "base_spec": _campaign_spec(2),
            "patch_path": "sim.time_steps",
            "values": [100, 200],
        },
    )
    assert result == {"ok": True, "name": "good", "leaves": ["leaf000", "leaf001"]}
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    assert campaign.agenda.entries["leaf000"].spec["sim"]["time_steps"] == 100


async def test_create_campaign_rejects_a_non_wire_spec(tmp_path) -> None:
    """A base_spec that is not an allow-listed ``{"sim": ...}`` wire is refused (B2)."""
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    result = await _call(
        config,
        "create_campaign",
        {
            "name": "bad",
            "base_spec": {"not_sim": {"time_steps": 4}},
            "patch_path": "not_sim.time_steps",
            "values": [1, 2],
        },
    )
    assert result["ok"] is False
    assert result["error"] == "invalid_campaign_spec"
    assert not (tmp_path / "campaign.json").exists()


async def test_create_campaign_rejects_an_over_cap_spec(tmp_path) -> None:
    """A leaf over the escaped inline cap is refused at creation (B2)."""
    from pic_agentic.protocol.simulation import MAX_INLINE_PAYLOAD_BYTES

    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    base = _valid_spec()
    # A field the pin preserves verbatim; ``solver.name``/``species[].name``
    # are normalised and would (correctly) trip the exact round-trip gate
    # before the size cap is reached.
    base["sim"]["customuserinput"] = {"tags": ["a" * (MAX_INLINE_PAYLOAD_BYTES * 2)]}
    result = await _call(
        config,
        "create_campaign",
        {"name": "big", "base_spec": base, "patch_path": "sim.time_steps", "values": [5]},
    )
    assert result["ok"] is False
    assert result["error"] == "spec_exceeds_inline_limit"
    assert result["wire_bytes"] > MAX_INLINE_PAYLOAD_BYTES
    assert not (tmp_path / "campaign.json").exists()


async def test_create_campaign_reaches_a_numeric_dict_key(tmp_path) -> None:
    """A numeric-looking dict key is addressed as a key, not as a list index (M2)."""
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    base = _valid_spec()
    base["sim"]["customuserinput"] = {"tags": ["t"], "0": "periodic"}
    result = await _call(
        config,
        "create_campaign",
        {
            "name": "bc",
            "base_spec": base,
            "patch_path": "sim.customuserinput.0",
            "values": ["open"],
        },
    )
    assert result["ok"] is True
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    assert campaign.agenda.entries["leaf000"].spec["sim"]["customuserinput"]["0"] == "open"


async def test_create_campaign_rejects_a_typo_field(tmp_path) -> None:
    """A path whose final segment does not exist is refused, not silently added (M3)."""
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "campaign.json"))
    result = await _call(
        config,
        "create_campaign",
        {
            "name": "typo",
            "base_spec": {"sim": {"time_steps": 4}},
            "patch_path": "sim.time_step",
            "values": [50],
        },
    )
    assert result["ok"] is False
    assert result["error"] == "invalid_campaign"
    assert not (tmp_path / "campaign.json").exists()


async def test_delete_campaign_removes_both_files_then_no_campaign(tmp_path) -> None:
    campaign = Path(_campaign_file(tmp_path))
    reuse = tmp_path / "reuse-registry.json"
    AgendaStore(tmp_path, filename="reuse-registry.json").save(ReuseRegistry())
    assert campaign.exists()
    assert reuse.exists()

    config = Config(rcp_secret=SECRET, agenda_file=str(campaign))
    result = await _call(config, "delete_campaign", {})
    assert result["ok"] is True
    assert set(result["deleted"]) == {str(campaign), str(reuse)}
    assert not campaign.exists()
    assert not reuse.exists()

    # With the state gone, status reports the actionable no_campaign error.
    status = await _call(config, "agenda_status", {})
    assert status == NO_CAMPAIGN


async def test_delete_campaign_without_a_campaign_is_a_soft_error(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "missing.json"))
    result = await _call(config, "delete_campaign", {})
    assert result["error"] == "no_campaign"
    assert result["message"] == NO_CAMPAIGN_MESSAGE
    assert "create_campaign" in result["message"]


def _campaign_file_with_status(tmp_path: Path, status: str) -> str:
    agenda = AgendaGroup(name="group").add(
        leaf=AgendaSim(name="leaf", spec={"sim": {"replica": 0}}, status=status, sim_id="sim0001")
    )
    AgendaStore(tmp_path, filename="campaign.json").save(Campaign(name="c", agenda=agenda))
    return str(tmp_path / "campaign.json")


async def test_delete_campaign_refuses_while_a_leaf_is_in_flight(tmp_path) -> None:
    campaign = Path(_campaign_file_with_status(tmp_path, "running"))
    config = Config(rcp_secret=SECRET, agenda_file=str(campaign))

    refused = await _call(config, "delete_campaign", {})
    assert refused["ok"] is False
    assert refused["error"] == "campaign_in_flight"
    assert refused["in_flight"] == ["leaf"]
    assert "force" in refused["message"]
    assert campaign.exists()


async def test_delete_campaign_force_overrides_the_in_flight_guard(tmp_path) -> None:
    campaign = Path(_campaign_file_with_status(tmp_path, "submitted"))
    config = Config(rcp_secret=SECRET, agenda_file=str(campaign))

    result = await _call(config, "delete_campaign", {"force": True})
    assert result["ok"] is True
    assert not campaign.exists()

    # A finished leaf is not in flight, so the default delete succeeds.
    done = Path(_campaign_file_with_status(tmp_path, "done"))
    done_result = await _call(config, "delete_campaign", {})
    assert done_result["ok"] is True
    assert not done.exists()


async def test_delete_campaign_clears_a_corrupt_campaign(tmp_path) -> None:
    """A corrupt campaign is removable: the reset primitive is the recovery path (B2).

    ``create_campaign`` refuses while the file exists and the in-flight guard
    loads the campaign, so without this the agent could never clear an
    unparseable file through any tool.
    """
    campaign = tmp_path / "campaign.json"
    campaign.write_text("{ this is not json", encoding="utf-8")
    config = Config(rcp_secret=SECRET, agenda_file=str(campaign))

    # A fresh campaign is blocked while the file exists.
    blocked = await _call(
        config,
        "create_campaign",
        {"name": "x", "base_spec": {"sim": {"time_steps": 4}}, "patch_path": "sim.time_steps", "values": [1]},
    )
    assert blocked == {"ok": False, "error": "campaign_exists"}

    result = await _call(config, "delete_campaign", {})
    assert result["ok"] is True
    assert str(campaign) in result["deleted"]
    assert not campaign.exists()

    # With the broken state gone, status reports the actionable no_campaign.
    status = await _call(config, "agenda_status", {})
    assert status == NO_CAMPAIGN


async def test_delete_campaign_guards_a_planned_leaf_with_a_sim_id(tmp_path) -> None:
    """A lost-ack leaf (planned + sim_id) still counts as in flight (m1)."""
    campaign = Path(_campaign_file_with_status(tmp_path, "planned"))
    config = Config(rcp_secret=SECRET, agenda_file=str(campaign))

    refused = await _call(config, "delete_campaign", {})
    assert refused["ok"] is False
    assert refused["error"] == "campaign_in_flight"
    assert refused["in_flight"] == ["leaf"]
    assert campaign.exists()

    forced = await _call(config, "delete_campaign", {"force": True})
    assert forced["ok"] is True
    assert not campaign.exists()
