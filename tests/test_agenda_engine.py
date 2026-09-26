# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for the campaign engine (durability, idempotency, gating)."""

from __future__ import annotations

import pytest

from pic_agentic.agenda.budget import Budget
from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.engine import AgendaEngine, EnginePolicy
from pic_agentic.agenda.model import AgendaGroup
from pic_agentic.agenda.store import DEFAULT_CAMPAIGN_FILE, AgendaStore


def _store(tmp_path) -> AgendaStore:
    return AgendaStore(tmp_path, filename=DEFAULT_CAMPAIGN_FILE)


def _agenda(n: int, *, core_hours: float = 1.0) -> AgendaGroup:
    group = AgendaGroup(name="sweep")
    for i in range(n):
        group = group.add_sim(name=f"a{i}", spec={"resources": {"est_core_hours": core_hours}})
    return group


class _Submitter:
    def __init__(self) -> None:
        self.specs: dict[str, dict] = {}
        self.calls = 0

    async def __call__(self, spec: dict) -> str:
        self.calls += 1
        sim_id = f"sim{self.calls:03d}"
        self.specs[sim_id] = spec
        return sim_id


async def test_submit_once_and_budget_across_ticks(tmp_path) -> None:
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(3), budget=Budget(max_core_hours=2.0)))
    submit = _Submitter()
    engine = AgendaEngine(store=store, submit=submit, observe=dict)

    first = await engine.tick()
    assert first.submitted == ["a0", "a1"]  # 2 core-hours cap
    assert first.usage.core_hours == pytest.approx(2.0)
    second = await engine.tick()
    assert second.submitted == []  # cap exhausted, nothing resubmitted
    assert submit.calls == 2


async def test_restart_does_not_resubmit(tmp_path) -> None:
    """A fresh engine over the same store must not resubmit recorded leaves."""
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(4), budget=Budget(max_core_hours=10.0)))
    submit = _Submitter()
    first_engine = AgendaEngine(store=store, submit=submit, observe=dict)
    await first_engine.tick()
    calls_after_first = submit.calls
    assert calls_after_first == 4

    # Simulate a restart: brand-new engine + submitter, same store.
    submit2 = _Submitter()
    engine2 = AgendaEngine(store=store, submit=submit2, observe=dict)
    result = await engine2.tick()
    assert result.submitted == []
    assert submit2.calls == 0


async def test_dependency_ordering_and_completion(tmp_path) -> None:
    store = _store(tmp_path)
    agenda = _agenda(2).model_copy(deep=True)
    agenda.entries["a1"] = agenda.entries["a1"].model_copy(update={"depends_on": ["a0"]})
    store.save(Campaign(name="c", agenda=agenda))

    state: dict[str, str] = {}
    counter = [0]

    async def submit(spec: dict) -> str:
        sim_id = f"sim{counter[0]}"
        counter[0] += 1
        state[sim_id] = "simulation.job_running"
        return sim_id

    engine = AgendaEngine(store=store, submit=submit, observe=lambda: dict(state))
    first = await engine.tick()
    assert first.submitted == ["a0"]
    assert first.waiting == ["a1"]

    for sim_id in state:
        state[sim_id] = "results.ready"
    second = await engine.tick()
    assert second.submitted == ["a1"]
    assert second.done == ["a0"]
    assert second.complete is False

    for sim_id in state:
        state[sim_id] = "results.ready"
    third = await engine.tick()
    assert third.complete is True
    assert set(third.done) == {"a0", "a1"}


async def test_approval_gate(tmp_path) -> None:
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(1, core_hours=500.0)))
    submit = _Submitter()
    gate = EnginePolicy(approve_over_est_core_hours=100.0)

    engine = AgendaEngine(store=store, submit=submit, observe=dict, policy=gate)
    result = await engine.tick()
    assert result.submitted == []
    assert result.pending_approval == ["a0"]
    assert submit.calls == 0

    # With pre-approval the same engine submits it.
    approver = AgendaEngine(store=store, submit=submit, observe=dict, policy=gate, approve=lambda _path: True)
    approved = await approver.tick()
    assert approved.submitted == ["a0"]


async def test_require_approval_gate(tmp_path) -> None:
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(1)))
    submit = _Submitter()
    engine = AgendaEngine(
        store=store,
        submit=submit,
        observe=dict,
        policy=EnginePolicy(require_approval=True),
    )
    result = await engine.tick()
    assert result.pending_approval == ["a0"]
    assert submit.calls == 0


async def test_max_submits_per_tick(tmp_path) -> None:
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(5), budget=Budget()))
    submit = _Submitter()
    engine = AgendaEngine(store=store, submit=submit, observe=dict, policy=EnginePolicy(max_submits_per_tick=2))
    result = await engine.tick()
    assert len(result.submitted) == 2
    assert len(result.waiting) == 3


async def test_failed_dependency_blocks_successor(tmp_path) -> None:
    store = _store(tmp_path)
    agenda = _agenda(2).model_copy(deep=True)
    agenda.entries["a1"] = agenda.entries["a1"].model_copy(update={"depends_on": ["a0"]})
    store.save(Campaign(name="c", agenda=agenda))

    state: dict[str, str] = {}
    counter = [0]

    async def submit(spec: dict) -> str:
        sim_id = f"sim{counter[0]}"
        counter[0] += 1
        state[sim_id] = "simulation.job_running"
        return sim_id

    engine = AgendaEngine(store=store, submit=submit, observe=lambda: dict(state))
    await engine.tick()
    for sim_id in state:
        state[sim_id] = "simulation.job_failed"
    result = await engine.tick()
    assert result.failed == ["a0", "a1"]  # successor is blocked, not waiting forever
    assert result.complete is True


async def test_status_and_report_shape(tmp_path) -> None:
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(2)))
    submit = _Submitter()
    engine = AgendaEngine(store=store, submit=submit, observe=dict)
    await engine.tick()

    status = engine.status()
    assert status["name"] == "c"
    assert status["counts"]["submitted"] == 2
    assert {leaf["path"] for leaf in status["leaves"]} == {"a0", "a1"}

    report = engine.campaign_report()
    assert report["created_ts"]
    assert all(len(leaf["spec_hash"]) == 64 for leaf in report["leaves"])
    # The spec hash is stable across calls.
    assert report["leaves"][0]["spec_hash"] == engine.campaign_report()["leaves"][0]["spec_hash"]


async def test_complete_with_empty_state_persists_progress(tmp_path) -> None:
    """A tick that only observes progress (no submission) still persists it."""
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(1)))
    state: dict[str, str] = {}
    counter = [0]

    async def submit(spec: dict) -> str:
        sim_id = f"sim{counter[0]}"
        counter[0] += 1
        state[sim_id] = "simulation.job_running"
        return sim_id

    engine = AgendaEngine(store=store, submit=submit, observe=lambda: dict(state))
    await engine.tick()
    for sim_id in state:
        state[sim_id] = "results.ready"
    await engine.tick()
    # Reload from disk: the done status must have been persisted.
    reloaded = store.load(Campaign)
    assert reloaded.agenda.simulations()[0][1].status == "done"


def test_campaign_round_trip(tmp_path) -> None:
    store = _store(tmp_path)
    campaign = Campaign(name="c", agenda=_agenda(2), budget=Budget(max_gpu_hours=5.0)).with_created_ts()
    store.save(campaign)
    loaded = store.load(Campaign)
    assert loaded.model_dump_json() == campaign.model_dump_json()
