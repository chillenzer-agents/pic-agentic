# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Offline end-to-end proof of the first acceptance test (north star).

Drives the campaign engine through the acceptance scenario without a cluster:
a laser-intensity sweep, an analysis-driven refinement around the optimum, a
declared convergence, and a campaign-level provenance record -- while surviving
**two engine restarts** and **one job failure** with **no duplicate
submissions**, under a **hard budget** and behind an **approval gate** before
the sweep.

The "cluster" is a deterministic fake: a submit callable assigns a sim_id per
distinct spec, records the payload under the *stable idempotency key* (so a
retry after a simulated lost ack replays the same sim_id, exactly as the real
simclient does), and an observe callable reports lifecycle states from a script.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pic_agentic.agenda.budget import Budget
from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.engine import AgendaEngine, EnginePolicy
from pic_agentic.agenda.model import AgendaGroup
from pic_agentic.agenda.provenance import RO_CRATE_METADATA_FILE, campaign_rocrate
from pic_agentic.agenda.store import DEFAULT_CAMPAIGN_FILE, AgendaStore

#: The laser intensities the campaign sweeps (the conceptual "parameter study").
INTENSITIES = [1e18, 2e18, 3e18, 4e18]


class FakeCluster:
    """A deterministic stand-in for the simclient + SLURM.

    A submission is recorded under its idempotency *key* (not its call ordinal),
    so a retried submission of the same key replays the same sim_id -- the
    exactly-once contract the real simclient provides.
    """

    def __init__(self, *, fail_once: set[str] | None = None) -> None:
        self.key_to_sim: dict[str, str] = {}
        self.states: dict[str, str] = {}
        self.sim_to_key: dict[str, str] = {}
        self.submits: list[str] = []
        self.specs: dict[str, dict] = {}
        #: Keys whose *first* submit raises, simulating a lost ack / crash.
        self.fail_once = set(fail_once or ())

    async def submit(self, spec: dict, key: str) -> str:
        self.submits.append(key)
        self.specs[key] = spec
        if key in self.key_to_sim:
            # A retry: the simclient would replay its recorded ack.
            return self.key_to_sim[key]
        if key in self.fail_once:
            self.fail_once.discard(key)
            msg = "simulated lost ack"
            raise RuntimeError(msg)
        sim_id = f"sim{len(self.key_to_sim):08x}"
        self.key_to_sim[key] = sim_id
        self.sim_to_key[sim_id] = key
        self.states[sim_id] = "simulation.job_running"
        return sim_id

    def observe(self) -> dict[str, str]:
        return dict(self.states)

    def finish(self, sim_id: str, *, failed: bool = False) -> None:
        self.states[sim_id] = "simulation.job_failed" if failed else "results.ready"


def _sweep_agenda() -> AgendaGroup:
    group = AgendaGroup(name="intensity-sweep")
    for intensity in INTENSITIES:
        name = f"i={intensity:g}"
        group = group.add_sim(name=name, spec={"sim": {"laser_intensity": intensity}})
    return group


async def _tick_fresh(store: AgendaStore, cluster: FakeCluster, **policy: object) -> object:
    """Run one tick with a brand-new engine (a simulated MCP-server restart)."""
    engine = AgendaEngine(store=store, submit=cluster.submit, observe=cluster.observe, policy=EnginePolicy(**policy))
    return await engine.tick()


def _approve_all(store: AgendaStore) -> None:
    """Approve every leaf (the human gate) and persist."""
    campaign = store.load(Campaign)
    agenda = campaign.agenda.model_copy(deep=True)
    for _, leaf in agenda.simulations():
        leaf.approved = True
    store.save(campaign.model_copy(update={"agenda": agenda}))


def _drain_and_refine(store: AgendaStore) -> None:
    """Drain the done callbacks and add a refined sweep point.

    A real campaign would analyse the finished run's spectrum here; the fake
    refinement stands in for that decision.
    """
    campaign = store.load(Campaign)
    drained = [c for c in campaign.callbacks if c.kind == "done"]
    assert drained, "the finished leaf must have raised a done callback"
    agenda = campaign.agenda.add_sim(name="refine_2.5e18", spec={"sim": {"laser_intensity": 2.5e18}})
    store.save(campaign.model_copy(update={"agenda": agenda, "callbacks": []}))


def _assert_provenance(campaign: Campaign) -> None:
    """Assert the campaign RO-Crate links every leaf and the software entity."""
    crate = campaign_rocrate(campaign)
    ids = {entity["@id"] for entity in crate["@graph"]}
    assert RO_CRATE_METADATA_FILE in ids
    assert "./" in ids
    assert "#refine_2.5e18" in ids
    assert "#picongpu" in ids
    root = next(entity for entity in crate["@graph"] if entity["@id"] == "./")
    assert len(root["hasPart"]) == len(INTENSITIES) + 1  # the sweep + the refinement


async def _run_to_completion(store: AgendaStore, cluster: FakeCluster) -> object:
    """Tick until complete, finishing every still-running (non-failed) sim.

    Returns:
        The final tick result.

    """
    tick: object = None
    for _ in range(10):
        tick = await _tick_fresh(store, cluster)
        for sim_id, state in list(cluster.states.items()):
            if state == "simulation.job_running":
                cluster.finish(sim_id)
        if tick.complete:  # type: ignore[attr-defined]
            break
    return tick


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_first_acceptance_test_offline(tmp_path: Path) -> None:
    store = AgendaStore(tmp_path, filename=DEFAULT_CAMPAIGN_FILE)
    cluster = FakeCluster()
    # Budget: 4 jobs total, 2 concurrent, and an approval gate over 1 core-hour.
    store.save(
        Campaign(
            name="laser-intensity-study",
            agenda=_sweep_agenda(),
            budget=Budget(max_total_jobs=6, max_concurrent_jobs=2),
        ),
    )

    # ---- Phase 0: the approval gate holds the expensive sweep -------------
    gated = await _tick_fresh(store, cluster, require_approval=True)
    assert gated.submitted == []
    # The concurrency cap bounds the batch, so only the first two leaves reach
    # the approval gate this tick; the rest are still waiting.
    assert sorted(gated.pending_approval) == sorted(f"i={intensity:g}" for intensity in INTENSITIES)[:2]

    # Approve every leaf (the human gate), then advance.
    _approve_all(store)

    # ---- Phase 1: submit the sweep, concurrency-capped --------------------
    tick1 = await _tick_fresh(store, cluster)
    assert len(tick1.submitted) == 2  # max_concurrent_jobs
    assert len(cluster.submits) == 2

    # A restart must not resubmit the two accepted leaves.
    tick1b = await _tick_fresh(store, cluster)
    assert tick1b.submitted == []
    assert len(cluster.submits) == 2

    # ---- Phase 2: one job fails, the rest finish --------------------------
    sims = sorted(cluster.key_to_sim.values())
    cluster.finish(sims[0], failed=True)
    cluster.finish(sims[1])
    tick2 = await _tick_fresh(store, cluster)
    # The failure frees a slot; the failure callback is recorded.
    assert any(c.kind == "failed" for c in tick2.callbacks)
    # Both slots are free (one failed, one done), so two more leaves go out.
    assert len(cluster.submits) == 4

    # ---- Phase 3: drain callbacks, analyse, refine around the optimum -----
    _drain_and_refine(store)

    # ---- Phase 3b: pause holds the refined leaf, resume releases it --------
    # Pause first, then finish the in-flight jobs: the pause still folds the
    # progress, but the now-ready refined leaf is held rather than submitted.
    paused = store.load(Campaign)
    store.save(paused.model_copy(update={"state": "paused"}))
    for sim_id, sim_state in list(cluster.states.items()):
        if sim_state == "simulation.job_running":
            cluster.finish(sim_id)
    held_tick = await _tick_fresh(store, cluster)
    assert held_tick.submitted == []
    assert "refine_2.5e18" in held_tick.held
    submits_before_resume = len(cluster.submits)
    resumed = store.load(Campaign)
    store.save(resumed.model_copy(update={"state": "running"}))
    released = await _tick_fresh(store, cluster)
    assert len(cluster.submits) > submits_before_resume
    assert released.state == "running"

    # ---- Phase 4: a second restart, then run to completion ----------------
    tick = await _run_to_completion(store, cluster)

    # ---- Assertions: budget, no duplicates, convergence, provenance -------
    final = store.load(Campaign)
    # No duplicate submissions: every recorded key is unique.
    assert len(cluster.submits) == len(set(cluster.submits))
    # Budget respected.
    assert final.usage.jobs_submitted <= final.budget.max_total_jobs
    # Every leaf but the one deliberately failed is done; the campaign is complete.
    assert tick.complete is True
    statuses = {path: sim.status for path, sim in final.agenda.simulations()}
    assert statuses["refine_2.5e18"] == "done"
    assert sum(1 for s in statuses.values() if s == "failed") == 1

    # Campaign-level provenance: an RO-Crate linking every leaf + the software.
    _assert_provenance(final)
