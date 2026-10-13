# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for the campaign lifecycle: pause / resume / kill-switch (B + H)."""

from __future__ import annotations

from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.engine import AgendaEngine
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
    def __init__(self, state: dict[str, str]) -> None:
        self.state = state
        self.calls = 0

    async def __call__(self, _spec: dict, _key: str) -> str:
        self.calls += 1
        sim_id = f"sim{self.calls:03d}"
        self.state[sim_id] = "simulation.job_running"
        return sim_id


async def test_paused_campaign_holds_submissions(tmp_path) -> None:
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(2), state="paused"))
    state: dict[str, str] = {}
    runner = _Runner(state)
    engine = AgendaEngine(store=store, submit=runner, observe=lambda: dict(state))
    result = await engine.tick()
    assert result.submitted == []
    assert result.held == ["a0", "a1"]
    assert result.state == "paused"
    assert runner.calls == 0
    # The leaves stay planned, so they are submittable after a resume.
    assert all(sim.status == "planned" for _, sim in store.load(Campaign).agenda.simulations())


async def test_resume_submits_held_leaves(tmp_path) -> None:
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(2), state="paused"))
    state: dict[str, str] = {}
    runner = _Runner(state)
    engine = AgendaEngine(store=store, submit=runner, observe=lambda: dict(state))
    await engine.tick()

    campaign = store.load(Campaign)
    store.save(campaign.model_copy(update={"state": "running"}))
    result = await engine.tick()
    assert result.submitted == ["a0", "a1"]
    assert result.state == "running"


async def test_paused_campaign_still_folds_and_emits_callbacks(tmp_path) -> None:
    """A pause must not stall observation: transitions still surface."""
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(1)))
    state: dict[str, str] = {}
    engine = AgendaEngine(store=store, submit=_Runner(state), observe=lambda: dict(state))
    await engine.tick()
    for sim_id in state:
        state[sim_id] = "results.ready"
    campaign = store.load(Campaign)
    store.save(campaign.model_copy(update={"state": "paused"}))
    result = await engine.tick()
    assert [c.path for c in result.callbacks] == ["a0"]
    assert result.done == ["a0"]
    assert result.submitted == []


async def test_stopped_campaign_stays_held_across_restarts(tmp_path) -> None:
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(3), state="stopped"))
    state: dict[str, str] = {}
    for _ in range(2):
        engine = AgendaEngine(store=store, submit=_Runner(state), observe=lambda: dict(state))
        result = await engine.tick()
        assert result.submitted == []
        assert result.held == ["a0", "a1", "a2"]


async def test_lifecycle_state_round_trips(tmp_path) -> None:
    store = _store(tmp_path)
    for state in ("paused", "stopped", "running"):
        store.save(Campaign(name="c", agenda=_agenda(1), state=state))
        assert store.load(Campaign).state == state


async def test_status_reports_state(tmp_path) -> None:
    store = _store(tmp_path)
    store.save(Campaign(name="c", agenda=_agenda(1), state="paused"))
    engine = AgendaEngine(store=store, submit=_Runner({}), observe=dict)
    assert engine.status()["state"] == "paused"
