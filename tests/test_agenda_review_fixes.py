# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Regression tests for the independent-review findings (lifecycle round)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pic_agentic.agenda.budget import Budget, BudgetUsage
from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.engine import AgendaEngine, EnginePolicy
from pic_agentic.agenda.model import AgendaGroup
from pic_agentic.agenda.planner import next_actions
from pic_agentic.agenda.store import DEFAULT_CAMPAIGN_FILE, AgendaStore
from pic_agentic.config import Config
from pic_agentic.protocol.simulation import (
    SimulationOp,
    SimulationState,
    SimulationType,
    build_control_ack,
    build_submit_ack,
)
from pic_agentic.rcp import new_secret_hex
from pic_agentic.server.app import build_server
from pic_agentic.transport.memory import MemoryTransport

SECRET = new_secret_hex()
SIM = "7f3a2b1c"


def _store(tmp_path) -> AgendaStore:
    return AgendaStore(tmp_path, filename=DEFAULT_CAMPAIGN_FILE)


def _agenda(n: int) -> AgendaGroup:
    group = AgendaGroup(name="sweep")
    for i in range(n):
        group = group.add_sim(name=f"a{i}", spec={"replica": i})
    return group


async def test_callback_survives_a_submit_failure_in_the_same_tick(tmp_path) -> None:
    """H2: a completion observed in a tick that later fails must not lose its callback."""
    store = _store(tmp_path)
    group = _agenda(2)
    group = group.add_sim(name="a2", spec={"replica": 2})
    store.save(Campaign(name="c", agenda=group))
    state: dict[str, str] = {}
    calls: list[str] = []
    fail_a2 = False

    async def submit(spec: dict, key: str) -> str:
        calls.append(key)
        if spec.get("replica") == 2 and fail_a2:
            msg = "permanent failure"
            raise RuntimeError(msg)
        sim_id = f"sim{spec['replica']}"
        state[sim_id] = "simulation.job_running"
        return sim_id

    engine = AgendaEngine(
        store=store,
        submit=submit,
        observe=lambda: dict(state),
        policy=EnginePolicy(max_submits_per_tick=2),
    )
    await engine.tick()  # submits a0, a1 (a2 waits on the per-tick cap)
    # a0 finishes; a2 then fails in the same next tick.
    state["sim0"] = "results.ready"
    fail_a2 = True
    result = await engine.tick()
    kinds = {c.path: c.kind for c in result.callbacks}
    assert kinds.get("a0") == "done"
    assert kinds.get("a2") == "failed"
    # Both callbacks are on disk, so a later drain cannot miss them.
    on_disk = {c.path for c in store.load(Campaign).callbacks}
    assert {"a0", "a2"} <= on_disk


async def test_planner_enforces_concurrency_cap_in_one_batch() -> None:
    """H3(planner): a single next_actions batch must respect max_concurrent_jobs."""
    agenda = AgendaGroup(name="g")
    for i in range(4):
        agenda = agenda.add_sim(name=f"j{i}", spec={"replica": i})
    steps = next_actions(agenda, {}, budget=Budget(max_concurrent_jobs=1), usage=BudgetUsage())
    assert sum(s.action == "submit" for s in steps) == 1


async def test_stop_waits_for_an_inflight_tick_and_wins(tmp_path) -> None:
    """H1: a stop issued during a tick must win, not be overwritten."""
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path, leaves=3))
    mcp_t, sim_t = MemoryTransport.create_pair()
    server, runtime = build_server(config, SIM)
    runtime._transport = mcp_t
    started = asyncio.Event()
    release = asyncio.Event()

    async def responder() -> None:
        counter = 0
        async for command in sim_t.receive():
            if command.type == SimulationType.COMMAND:
                counter += 1
                started.set()
                await release.wait()
                ack = build_submit_ack(
                    sim=SIM,
                    seq=counter,
                    cmd_id=str(command.payload.get("cmd_id", "")),
                    sim_id=f"sim{counter:04d}",
                    state=SimulationState.ACCEPTED,
                    in_reply_to=command.transport_event_id,
                ).sign(SECRET)
                await sim_t.send(ack)
            elif command.type == SimulationType.CONTROL_COMMAND:
                counter += 1
                ack = build_control_ack(
                    sim=SIM,
                    seq=counter,
                    cmd_id=str(command.payload.get("cmd_id", "")),
                    sim_id=str(command.payload.get("sim_id", "")),
                    op=SimulationOp(command.payload["op"]),
                    ok=True,
                    in_reply_to=command.transport_event_id,
                ).sign(SECRET)
                await sim_t.send(ack)

    async def pump() -> None:
        async for message in mcp_t.receive():
            runtime.submit_service.on_message(message)

    tasks = [asyncio.create_task(responder()), asyncio.create_task(pump())]
    try:
        tick_task = asyncio.create_task(server.call_tool("advance_agenda", {}))
        await started.wait()
        stop_task = asyncio.create_task(server.call_tool("stop_agenda", {}))
        await asyncio.sleep(0)  # let stop block on the lock
        release.set()
        await tick_task
        stop = (await stop_task).structured_content
        assert stop["state"] == "stopped"
        # The stop is the last writer: the campaign must not be resurrected.
        assert store_state(config) == "stopped"
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()


async def test_stop_does_not_report_a_failed_cancel_as_cancelled(tmp_path) -> None:
    """Security-H2: a rejected/timeout cancel is an error, not a success."""
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path, leaves=1))
    mcp_t, sim_t = MemoryTransport.create_pair()
    server, runtime = build_server(config, SIM)
    runtime._transport = mcp_t

    async def responder() -> None:
        counter = 0
        async for command in sim_t.receive():
            counter += 1
            if command.type == SimulationType.COMMAND:
                ack = build_submit_ack(
                    sim=SIM,
                    seq=counter,
                    cmd_id=str(command.payload.get("cmd_id", "")),
                    sim_id=f"sim{counter:04d}",
                    state=SimulationState.ACCEPTED,
                    in_reply_to=command.transport_event_id,
                ).sign(SECRET)
            elif command.type == SimulationType.CONTROL_COMMAND:
                ack = build_control_ack(
                    sim=SIM,
                    seq=counter,
                    cmd_id=str(command.payload.get("cmd_id", "")),
                    sim_id=str(command.payload.get("sim_id", "")),
                    op=SimulationOp(command.payload["op"]),
                    ok=False,
                    error="rejected",
                    in_reply_to=command.transport_event_id,
                ).sign(SECRET)
            else:
                continue
            await sim_t.send(ack)

    async def pump() -> None:
        async for message in mcp_t.receive():
            runtime.submit_service.on_message(message)

    tasks = [asyncio.create_task(responder()), asyncio.create_task(pump())]
    try:
        await server.call_tool("advance_agenda", {})
        result = (await server.call_tool("stop_agenda", {})).structured_content
        assert result["cancelled"] == []
        assert len(result["errors"]) == 1
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()


async def test_add_leaf_with_bad_dependency_does_not_brick(tmp_path) -> None:
    """M5: an invalid depends_on must be rejected, not persisted."""
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path, leaves=1))
    mcp_t, sim_t = MemoryTransport.create_pair()
    server, runtime = build_server(config, SIM)
    runtime._transport = mcp_t
    try:
        result = (
            await server.call_tool(
                "add_agenda_leaf",
                {"name": "bad", "spec": {"sim": {"replica": 9}}, "depends_on": ["x/x"]},
            )
        ).structured_content
        assert result["ok"] is False
        # The campaign is still loadable: the bad mutation was not persisted.
        status = (await server.call_tool("agenda_status", {})).structured_content
        assert status["name"] == "campaign"
    finally:
        await mcp_t.close()
        await sim_t.close()


async def test_deeply_nested_secret_is_redacted_not_passed_raw() -> None:
    """Security-M3: redaction must not be bypassed by deep nesting."""
    from pic_agentic.server.app import _redact_dict

    secret = "syt_deep_secret"
    config = Config(rcp_secret=SECRET, access_token=secret)
    _server, runtime = build_server(config, SIM)
    # A 40-deep nested payload carrying the secret past _REDACT_MAX_DEPTH.
    payload: dict = {"level": "x"}
    node = payload
    for _ in range(40):
        node["nested"] = {}
        node = node["nested"]
    node["secret"] = secret
    result = _redact_dict(runtime, payload)
    assert secret not in str(result)


async def test_read_tool_transport_error_is_a_soft_error(tmp_path, monkeypatch) -> None:
    """M6: a read tool must not let a transport exception escape."""
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    server, runtime = build_server(config, SIM)
    runtime.submit_service.registry["s"] = runtime.submit_service._record_for("s", cmd_id="c")
    runtime.submit_service.registry["s"].state = "simulation.job_running"
    runtime.submit_service.registry["s"].active = True

    async def boom(*_a: object, **_k: object) -> None:
        msg = "transport down"
        raise RuntimeError(msg)

    monkeypatch.setattr(runtime, "fetch_status", boom)
    result = (await server.call_tool("get_status", {"sim_id": "s"})).structured_content
    assert "transport down" in str(result)  # captured as data, not raised


def _campaign_file(tmp_path: Path, *, leaves: int = 1) -> str:
    from pic_agentic.agenda.model import AgendaSim

    agenda = AgendaGroup(name="group")
    for i in range(leaves):
        agenda = agenda.add(**{f"leaf{i}": AgendaSim(name=f"leaf{i}", spec={"sim": {"replica": i}})})
    AgendaStore(tmp_path, filename="campaign.json").save(Campaign(name="campaign", agenda=agenda))
    return str(tmp_path / "campaign.json")


def store_state(config: Config) -> str:
    store = AgendaStore(Path(config.agenda_file).parent, filename=Path(config.agenda_file).name)
    return store.load(Campaign).state
