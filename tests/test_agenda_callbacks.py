# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for agent callbacks and agenda mutation (milestone F)."""

from __future__ import annotations

from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.engine import AgendaEngine, FailureInfo
from pic_agentic.agenda.model import AgendaGroup
from pic_agentic.agenda.store import DEFAULT_CAMPAIGN_FILE, AgendaStore


def _store(tmp_path) -> AgendaStore:
    return AgendaStore(tmp_path, filename=DEFAULT_CAMPAIGN_FILE)


def _agenda(n: int) -> AgendaGroup:
    group = AgendaGroup(name="sweep")
    for i in range(n):
        group = group.add_sim(name=f"a{i}", spec={"replica": i})
    return group


class _Runner:
    """A submitter whose sim states live in a shared dict."""

    def __init__(self, state: dict[str, str]) -> None:
        self.state = state
        self.calls = 0

    async def __call__(self, _spec: dict, _key: str) -> str:
        self.calls += 1
        sim_id = f"sim{self.calls:03d}"
        self.state[sim_id] = "simulation.job_running"
        return sim_id


async def test_done_leaf_emits_exactly_one_callback(tmp_path) -> None:
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(1)))
    state: dict[str, str] = {}
    engine = AgendaEngine(store=store, submit=_Runner(state), observe=lambda: dict(state))
    first = await engine.tick()
    assert first.callbacks == []

    for sim_id in state:
        state[sim_id] = "results.ready"
    second = await engine.tick()
    assert [(c.path, c.kind) for c in second.callbacks] == [("a0", "done")]

    # A further tick must not re-emit (the leaf is already terminal on disk).
    third = await engine.tick()
    assert third.callbacks == []


async def test_failed_leaf_emits_failed_callback(tmp_path) -> None:
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(1)))
    state: dict[str, str] = {}
    engine = AgendaEngine(store=store, submit=_Runner(state), observe=lambda: dict(state))
    await engine.tick()
    for sim_id in state:
        state[sim_id] = "simulation.job_failed"
    result = await engine.tick()
    assert [(c.path, c.kind) for c in result.callbacks] == [("a0", "failed")]


async def test_restart_does_not_duplicate_callbacks(tmp_path) -> None:
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(1)))
    state: dict[str, str] = {}
    engine = AgendaEngine(store=store, submit=_Runner(state), observe=lambda: dict(state))
    await engine.tick()
    for sim_id in state:
        state[sim_id] = "results.ready"
    await engine.tick()

    # Fresh engine over the same store: the dedup is against persisted data.
    fresh = AgendaEngine(store=store, submit=_Runner(state), observe=lambda: dict(state))
    result = await fresh.tick()
    assert result.callbacks == []


async def test_pending_callbacks_survive_a_restart(tmp_path) -> None:
    """A transition before the poll is not lost: the callback is on disk."""
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(1)))
    state: dict[str, str] = {}
    engine = AgendaEngine(store=store, submit=_Runner(state), observe=lambda: dict(state))
    await engine.tick()
    for sim_id in state:
        state[sim_id] = "results.ready"
    await engine.tick()
    assert [c.kind for c in store.load(Campaign).callbacks] == ["done"]


async def test_failed_leaf_callback_carries_the_observed_reason(tmp_path) -> None:
    """A failed transition stamps the simclient's reason onto leaf and callback."""
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(1)))
    state: dict[str, str] = {}
    engine = AgendaEngine(store=store, submit=_Runner(state), observe=lambda: dict(state))
    await engine.tick()
    sim_id = next(iter(state))
    state[sim_id] = "simulation.failed"
    failures = {sim_id: FailureInfo(error="bad field", error_code="unsupported", stage="prepare")}
    engine = AgendaEngine(
        store=store,
        submit=_Runner(state),
        observe=lambda: dict(state),
        failures=lambda: failures,
    )
    result = await engine.tick()
    (callback,) = result.callbacks
    assert (callback.path, callback.kind) == ("a0", "failed")
    assert callback.error == "bad field"
    assert callback.error_code == "unsupported"
    assert callback.stage == "prepare"
    # The reason is persisted on the leaf and surfaced by status.
    leaf = store.load(Campaign).agenda.entries["a0"]
    assert (leaf.error, leaf.error_code, leaf.stage) == ("bad field", "unsupported", "prepare")
    status_leaf = next(leaf for leaf in engine.status()["leaves"] if leaf["path"] == "a0")
    assert status_leaf["error"] == "bad field"
    assert status_leaf["error_code"] == "unsupported"


async def test_submit_rejection_reason_flows_onto_the_leaf(tmp_path) -> None:
    """A submit callable raising SubmitFailureError surfaces the reason."""
    from pic_agentic.agenda.engine import SubmitFailureError

    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(1)))

    async def rejecting(_spec: dict, _key: str) -> str:
        msg = "payload malformed"
        raise SubmitFailureError(msg, error_code="payload_invalid", stage="prepare")

    engine = AgendaEngine(store=store, submit=rejecting, observe=dict)
    result = await engine.tick()
    (callback,) = result.callbacks
    assert callback.kind == "failed"
    assert callback.error == "payload malformed"
    assert callback.error_code == "payload_invalid"
    assert callback.stage == "prepare"
    leaf = store.load(Campaign).agenda.entries["a0"]
    assert leaf.error_code == "payload_invalid"


async def test_dependency_failed_callback_for_blocked_successor(tmp_path) -> None:
    """A blocked successor terminal-persisted by the planner also emits."""
    store = _store(tmp_path)
    agenda = _agenda(2).model_copy(deep=True)
    agenda.entries["a1"] = agenda.entries["a1"].model_copy(update={"depends_on": ["a0"]})
    store.save(Campaign(name="c", agenda=agenda))
    state: dict[str, str] = {}
    engine = AgendaEngine(store=store, submit=_Runner(state), observe=lambda: dict(state))
    await engine.tick()
    for sim_id in state:
        state[sim_id] = "simulation.job_failed"
    result = await engine.tick()
    assert sorted((c.path, c.kind) for c in result.callbacks) == [("a0", "failed"), ("a1", "failed")]
    # The blocked successor names the dependency that failed it.
    blocked = next(c for c in result.callbacks if c.path == "a1")
    assert blocked.error is not None
    assert "a0" in blocked.error


async def test_end_to_end_refine_loop(tmp_path) -> None:
    """Sweep -> callbacks -> add a refined leaf -> the next tick submits it."""
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(1)))
    state: dict[str, str] = {}
    engine = AgendaEngine(store=store, submit=_Runner(state), observe=lambda: dict(state))
    await engine.tick()
    for sim_id in state:
        state[sim_id] = "results.ready"
    callbacks = (await engine.tick()).callbacks
    assert [c.path for c in callbacks] == ["a0"]

    campaign = store.load(Campaign)
    agenda = campaign.agenda.add_sim(name="refined", spec={"replica": 99})
    store.save(campaign.model_copy(update={"agenda": agenda}))

    # Clear the callbacks the way the drain tool does, then submit the new leaf.
    campaign = store.load(Campaign)
    store.save(campaign.model_copy(update={"callbacks": []}))
    result = await engine.tick()
    assert result.submitted == ["refined"]
