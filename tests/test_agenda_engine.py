# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for the campaign engine (durability, idempotency, gating)."""

from __future__ import annotations

import pytest

from pic_agentic.agenda.budget import Budget
from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.engine import (
    MAX_DEFERRED_SUBMIT_ATTEMPTS,
    AgendaEngine,
    DuplicateSpecError,
    EnginePolicy,
    TransientSubmitError,
)
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
    """A lost ack leaves the leaf planned; the retry reuses its key."""
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(2)))

    calls: list[str] = []

    async def submit(spec: dict, key: str) -> str:
        calls.append(key)
        if len(calls) == 2:
            msg = "lost ack"
            raise TransientSubmitError(msg)
        return "sim000"

    engine = AgendaEngine(store=store, submit=submit, observe=dict)
    # A lost ack does not fail the tick: the first leaf is recorded, the second
    # is deferred and stays planned.
    first = await engine.tick()
    assert first.submitted == ["a0"]
    assert first.failed == []
    reloaded = store.load(Campaign)
    assert reloaded.agenda.entries["a0"].sim_id == "sim000"
    assert reloaded.agenda.entries["a1"].status == "planned"
    deferred_key = calls[1]

    resumed_calls: list[str] = []

    async def submit2(spec: dict, key: str) -> str:
        resumed_calls.append(key)
        return "sim001"

    engine2 = AgendaEngine(store=store, submit=submit2, observe=dict)
    result = await engine2.tick()
    assert result.submitted == ["a1"]
    assert len(resumed_calls) == 1
    # The retry reuses the same idempotency key, so the cluster replays.
    assert resumed_calls[0] == deferred_key


async def test_lost_ack_is_deferred_not_failed_and_reconciles(tmp_path) -> None:
    """A lost ack stays planned and is retried, never presented as a failure.

    This is the C2 regression: the server's ``submit`` callable raises
    ``TransientSubmitError`` for an outcome-unknown ack (a pending idempotency
    record), and the engine must (a) surface the leaf under ``deferred`` rather
    than ``failed``, (b) keep it ``planned``, and (c) retry it under the same
    exactly-once key until the record completes.
    """
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(1)))

    keys: list[str] = []

    async def submit_lost(spec: dict, key: str) -> str:
        keys.append(key)
        msg = "already_submitted:outcome_unknown"
        raise TransientSubmitError(msg, error_code="outcome_unknown", sim_id="sim-str")

    engine = AgendaEngine(store=store, submit=submit_lost, observe=dict)
    result = await engine.tick()
    assert result.submitted == []
    assert result.failed == []
    assert result.deferred == ["a0"]
    leaf = store.load(Campaign).agenda.entries["a0"]
    assert leaf.status == "planned"
    assert leaf.deferred_attempts == 1
    # The pending record names the job, so cleanup can still reach it.
    assert leaf.sim_id == "sim-str"

    # The retry (same key) now re-acks the run: the leaf links and the deferral
    # counter resets.
    async def submit_ok(spec: dict, key: str) -> str:
        keys.append(key)
        return "sim-str"

    engine2 = AgendaEngine(store=store, submit=submit_ok, observe=dict)
    result2 = await engine2.tick()
    assert result2.submitted == ["a0"]
    assert result2.deferred == []
    reloaded = store.load(Campaign)
    assert reloaded.agenda.entries["a0"].status == "submitted"
    assert reloaded.agenda.entries["a0"].deferred_attempts == 0
    # Both ticks used the same idempotency key (exactly-once retry).
    assert keys[0] == keys[1]


async def test_deferred_submission_gives_up_after_bounded_attempts(tmp_path) -> None:
    """A never-resolving pending record must not defer forever.

    After ``MAX_DEFERRED_SUBMIT_ATTEMPTS`` consecutive deferrals the leaf is
    failed with an actionable ``outcome_unknown`` code (not a policy rejection),
    so the campaign can complete and the agent can clean up the maybe-job.
    """
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(1)))

    async def submit(spec: dict, key: str) -> str:
        msg = "already_submitted:outcome_unknown"
        raise TransientSubmitError(msg, error_code="outcome_unknown", sim_id="sim-str")

    engine = AgendaEngine(store=store, submit=submit, observe=dict)
    for attempt in range(1, MAX_DEFERRED_SUBMIT_ATTEMPTS + 1):
        result = await engine.tick()
        if attempt < MAX_DEFERRED_SUBMIT_ATTEMPTS:
            assert result.deferred == ["a0"]
            assert result.failed == []
            assert store.load(Campaign).agenda.entries["a0"].status == "planned"
        else:
            # The final attempt exhausts the budget: now (and only now) terminal.
            assert result.deferred == ["a0"]
            assert result.failed == ["a0"]
            assert [c.error_code for c in result.callbacks] == ["outcome_unknown"]
    leaf = store.load(Campaign).agenda.entries["a0"]
    assert leaf.status == "failed"
    assert leaf.error_code == "outcome_unknown"
    assert "cancel_simulation" in (leaf.error or "")


async def test_permanent_submit_failure_marks_leaf_failed(tmp_path) -> None:
    """A non-transient submit failure fails the leaf, not the whole tick."""
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(2)))
    calls: list[str] = []

    async def submit(spec: dict, key: str) -> str:
        calls.append(key)
        if len(calls) == 1:
            msg = "payload rejected"
            raise RuntimeError(msg)
        return "sim001"

    engine = AgendaEngine(store=store, submit=submit, observe=dict)
    result = await engine.tick()
    assert result.failed == ["a0"]
    assert result.submitted == ["a1"]  # the later leaf is not blocked
    assert [c.kind for c in result.callbacks] == ["failed"]
    reloaded = store.load(Campaign)
    assert reloaded.agenda.entries["a0"].status == "failed"


async def test_duplicate_spec_is_refused(tmp_path) -> None:
    """Identical payloads would collide on sim_id; the engine refuses them."""
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


async def test_duplicate_payload_with_different_resources_is_refused(tmp_path) -> None:
    """The guard hashes the wire payload (``sim``), like ``sim_id`` does."""
    store = _store(tmp_path)
    group = AgendaGroup(name="g")
    group = group.add_sim(name="a0", spec={"sim": {"t": 4}, "resources": {"est_core_hours": 1.0}})
    group = group.add_sim(name="a1", spec={"sim": {"t": 4}, "resources": {"est_core_hours": 2.0}})
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
    # A completed tick reports ``complete`` rather than the stored ``running``
    # lifecycle state, so state and complete never contradict.
    assert third.state == "complete"
    # The stored lifecycle is preserved separately, not folded into ``state``.
    assert third.lifecycle == "running"
    assert set(third.done) == {"a0", "a1"}


async def test_complete_tick_on_paused_campaign_keeps_lifecycle(tmp_path) -> None:
    """A terminal tick on a non-running campaign is not lossy.

    ``state`` reports ``complete`` (all leaves terminal) while ``lifecycle``
    keeps the real pause/stop, so the two never collapse into one field.
    """
    for lifecycle in ("paused", "stopped"):
        root = tmp_path / lifecycle
        root.mkdir()
        store = _store(root)
        agenda = _agenda(1)
        agenda.entries["a0"].status = "done"
        store.save(Campaign(name="c", agenda=agenda, state=lifecycle))

        result = await AgendaEngine(store=store, submit=_Submitter(), observe=dict).tick()

        assert result.complete is True
        assert result.state == "complete"
        assert result.lifecycle == lifecycle


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


def test_failure_grouping_key_separates_stages() -> None:
    """Leaves failing identically at different stages stay distinct groups.

    ``FailureGroup`` carries ``stage``, so the dedupe key must include it: a
    ``build`` failure and a ``run`` failure with the same code/message are not
    the same group.
    """
    from pic_agentic.agenda.campaign import Callback
    from pic_agentic.agenda.engine import TickResult, _compact_failures

    result = TickResult(
        failed=["a", "b"],
        callbacks=[
            Callback(path="a", kind="failed", error="boom", error_code="unsupported", stage="build"),
            Callback(path="b", kind="failed", error="boom", error_code="unsupported", stage="run"),
        ],
    )
    _compact_failures(result)
    assert len(result.failure_groups) == 2
    assert {group.stage for group in result.failure_groups} == {"build", "run"}
    assert [group.paths for group in result.failure_groups] == [["a"], ["b"]]
