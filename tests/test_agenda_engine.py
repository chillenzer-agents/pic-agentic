# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for the campaign engine (durability, idempotency, gating)."""

from __future__ import annotations

import pytest

from pic_agentic.agenda.budget import Budget
from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.engine import AgendaEngine, DuplicateSpecError, EnginePolicy
from pic_agentic.agenda.model import AgendaGroup
from pic_agentic.agenda.store import DEFAULT_CAMPAIGN_FILE, AgendaStore


def _store(tmp_path) -> AgendaStore:
    return AgendaStore(tmp_path, filename=DEFAULT_CAMPAIGN_FILE)


def _agenda(n: int, *, core_hours: float = 1.0) -> AgendaGroup:
    group = AgendaGroup(name="sweep")
    for i in range(n):
        # Distinct specs: identical payloads map to the same sim_id.
        group = group.add_sim(name=f"a{i}", spec={"resources": {"est_core_hours": core_hours}, "replica": i})
    return group


class _Submitter:
    def __init__(self) -> None:
        self.specs: dict[str, dict] = {}
        self.calls = 0
        self.keys: list[str] = []

    async def __call__(self, spec: dict, key: str) -> str:
        self.calls += 1
        self.keys.append(key)
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


async def test_idempotency_key_is_stable_across_restarts(tmp_path) -> None:
    """The command id for a leaf is derived from persisted data, not randomness.

    A retry after a lost ack must re-use the same cluster-side command id so the
    simclient replays its recorded ack instead of running the job twice.
    """
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(1)))
    submit = _Submitter()
    engine = AgendaEngine(store=store, submit=submit, observe=dict)
    await engine.tick()
    key_one = submit.keys[0]

    # Crash before persistence would normally lose the sim_id; emulate by
    # clearing it, then re-running with a fresh engine over the same campaign.
    campaign = store.load(Campaign)
    campaign.agenda.entries["a0"].sim_id = None
    campaign.agenda.entries["a0"].status = "planned"
    store.save(campaign)
    submit2 = _Submitter()
    engine2 = AgendaEngine(store=store, submit=submit2, observe=dict)
    await engine2.tick()
    assert submit2.keys[0] == key_one


async def test_lost_ack_does_not_duplicate_after_incremental_save(tmp_path) -> None:
    """A crash after an accepted submit must not resubmit that leaf."""
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(2)))

    class _BoomError(Exception):
        pass

    calls: list[str] = []

    async def submit(spec: dict, key: str) -> str:
        calls.append(key)
        if len(calls) == 2:
            raise _BoomError
        return "sim000"

    engine = AgendaEngine(store=store, submit=submit, observe=dict)
    with pytest.raises(_BoomError):
        await engine.tick()
    # The first leaf was persisted incrementally and must not be resubmitted.
    reloaded = store.load(Campaign)
    assert reloaded.agenda.entries["a0"].sim_id == "sim000"
    assert reloaded.agenda.entries["a0"].status == "submitted"

    resumed_calls: list[str] = []

    async def submit2(spec: dict, key: str) -> str:
        resumed_calls.append(key)
        return "sim001"

    engine2 = AgendaEngine(store=store, submit=submit2, observe=dict)
    result = await engine2.tick()
    assert result.submitted == ["a1"]
    assert len(resumed_calls) == 1


async def test_duplicate_spec_is_refused(tmp_path) -> None:
    """Identical specs would collide on sim_id; the engine refuses them."""
    store = _store(tmp_path)
    group = AgendaGroup(name="g")
    group = group.add_sim(name="a0", spec={"sim": {"time_steps": 4}})
    group = group.add_sim(name="a1", spec={"sim": {"time_steps": 4}})
    store.save(Campaign(name="c", agenda=group))

    async def submit(spec: dict, key: str) -> str:
        return "sim000"

    engine = AgendaEngine(store=store, submit=submit, observe=dict)
    with pytest.raises(DuplicateSpecError):
        await engine.tick()


async def test_concurrency_cap_bounds_one_tick(tmp_path) -> None:
    """A single tick must not submit past ``max_concurrent_jobs``."""
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(5), budget=Budget(max_concurrent_jobs=2)))
    submit = _Submitter()
    engine = AgendaEngine(store=store, submit=submit, observe=dict)
    result = await engine.tick()
    assert len(result.submitted) == 2
    assert len(result.waiting) == 3
    assert submit.calls == 2


async def test_jobs_running_is_recomputed_from_observations(tmp_path) -> None:
    """``jobs_running`` reflects observed in-flight leaves, not just accounting."""
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(3), budget=Budget(max_concurrent_jobs=2)))
    state: dict[str, str] = {}
    counter = [0]

    async def submit(spec: dict, key: str) -> str:
        counter[0] += 1
        sim_id = f"sim{counter[0]}"
        state[sim_id] = "simulation.job_running"
        return sim_id

    engine = AgendaEngine(store=store, submit=submit, observe=lambda: dict(state))
    first = await engine.tick()
    assert len(first.submitted) == 2
    assert store.load(Campaign).usage.jobs_running == 2

    # Finish one run; the next tick may now submit one more.
    state["sim1"] = "results.ready"
    second = await engine.tick()
    assert second.submitted == ["a2"]


async def test_dependency_ordering_and_completion(tmp_path) -> None:
    store = _store(tmp_path)
    agenda = _agenda(2).model_copy(deep=True)
    agenda.entries["a1"] = agenda.entries["a1"].model_copy(update={"depends_on": ["a0"]})
    store.save(Campaign(name="c", agenda=agenda))

    state: dict[str, str] = {}
    counter = [0]

    async def submit(spec: dict, key: str) -> str:
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


async def test_leaf_requires_approval_is_honoured(tmp_path) -> None:
    """A per-leaf ``requires_approval`` gates submission until approved."""
    store = _store(tmp_path)
    group = AgendaGroup(name="g").add_sim(name="a0", spec={"replica": 0})
    group.entries["a0"].requires_approval = True
    store.save(Campaign(name="c", agenda=group))
    submit = _Submitter()

    engine = AgendaEngine(store=store, submit=submit, observe=dict)
    result = await engine.tick()
    assert result.pending_approval == ["a0"]
    assert submit.calls == 0

    # Persisted approval (as the approve tool sets it) unblocks the leaf.
    campaign = store.load(Campaign)
    campaign.agenda.entries["a0"].approved = True
    store.save(campaign)
    approved = await engine.tick()
    assert approved.submitted == ["a0"]


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

    async def submit(spec: dict, key: str) -> str:
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
    assert all(leaf["requires_approval"] is False for leaf in status["leaves"])

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

    async def submit(spec: dict, key: str) -> str:
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
