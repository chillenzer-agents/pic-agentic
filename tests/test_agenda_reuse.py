# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for content-addressed reuse of identical simulations."""

from __future__ import annotations

from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.engine import AgendaEngine
from pic_agentic.agenda.engine import _wire_hash as engine_key
from pic_agentic.agenda.model import AgendaGroup
from pic_agentic.agenda.reuse import ReuseRecord, ReuseRegistry
from pic_agentic.agenda.store import DEFAULT_CAMPAIGN_FILE, AgendaStore


def _store(tmp_path) -> AgendaStore:
    return AgendaStore(tmp_path, filename=DEFAULT_CAMPAIGN_FILE)


class _Submitter:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, spec: dict, key: str) -> str:
        _ = (spec, key)
        self.calls += 1
        return f"sim{self.calls:03d}"


def _campaign(specs: list[dict]) -> Campaign:
    group = AgendaGroup(name="sweep")
    for index, spec in enumerate(specs):
        group = group.add_sim(name=f"a{index}", spec=spec)
    return Campaign(name="c", agenda=group).with_created_ts()


def test_registry_lookup_only_returns_reusable_states() -> None:
    registry = ReuseRegistry()
    registry = registry.remember(ReuseRecord(wire_hash="h1", sim_id="s1", state="done"))
    registry = registry.remember(ReuseRecord(wire_hash="h2", sim_id="s2", state="failed"))
    assert registry.lookup("h1") is not None
    assert registry.lookup("h2") is None  # failed runs are not reusable
    assert registry.lookup("missing") is None


def test_registry_roundtrips_through_the_store(tmp_path) -> None:
    store = AgendaStore(tmp_path, filename="reuse.json")
    registry = ReuseRegistry().remember(ReuseRecord(wire_hash="h1", sim_id="s1", state="done", run_dir="/run"))
    store.save(registry)
    loaded = ReuseRegistry.model_validate_json(store.path.read_text(encoding="utf-8"))
    record = loaded.lookup("h1")
    assert record is not None
    assert record.sim_id == "s1"
    assert record.run_dir == "/run"


def _engine(store: AgendaStore, submit: _Submitter, registry: dict[str, ReuseRecord], observed: dict[str, str]):
    def lookup(key: str) -> ReuseRecord | None:
        record = registry.get(key)
        return record if record is not None and record.state == "done" else None

    def record(key: str, sim_id: str, state: str) -> None:
        registry[key] = ReuseRecord(wire_hash=key, sim_id=sim_id, state=state)

    return AgendaEngine(
        store=store,
        submit=submit,
        observe=lambda: dict(observed),
        reuse_lookup=lookup,
        reuse_record=record,
    )


async def test_engine_records_a_result_ready_run_then_reuses_it(tmp_path) -> None:
    """A completed identical run is recorded and then linked, not re-submitted."""
    store = _store(tmp_path)
    spec = {"replica": 0}
    store.save(_campaign([spec]))
    registry: dict[str, ReuseRecord] = {}
    observed: dict[str, str] = {}
    submit = _Submitter()

    engine = _engine(store, submit, registry, observed)
    first = await engine.tick()
    assert first.submitted == ["a0"]
    assert first.reused == []
    assert submit.calls == 1

    # The run reaches results.ready: the next tick records it as reusable.
    observed["sim001"] = "results.ready"
    await engine.tick()
    assert registry, "a results.ready run should be recorded for reuse"

    # A fresh campaign with the *same* spec must be reused, not submitted -- and
    # only ONE tick is needed (no waiting behind admission/gates).
    store.save(_campaign([spec]))
    fresh = _engine(store, submit, registry, observed)
    tick = await fresh.tick()
    assert tick.reused == ["a0"]
    assert tick.submitted == []
    assert submit.calls == 1  # no second cluster job
    leaf = store.load(Campaign).agenda.entries["a0"]
    assert leaf.status == "done"
    assert leaf.sim_id == "sim001"
    assert leaf.reused is True
    # A reuse is a decision point: a done callback is emitted.
    assert any(call.kind == "done" for call in tick.callbacks)


async def test_reuse_happens_before_the_budget_and_concurrency_gates(tmp_path) -> None:
    """A reusable leaf is not blocked by an exhausted budget/concurrency cap.

    Regression: reuse used to run *after* admission, so a reusable leaf waited
    forever once ``max_total_jobs``/concurrency was reached, wedging completion.
    """
    from pic_agentic.agenda.budget import Budget

    store = _store(tmp_path)
    spec = {"replica": 0, "resources": {"est_core_hours": 1.0}}
    store.save(_campaign([spec]))
    registry: dict[str, ReuseRecord] = {}
    observed: dict[str, str] = {}
    submit = _Submitter()
    engine = _engine(store, submit, registry, observed)

    # First campaign runs the spec to results.ready and records it.
    await engine.tick()
    observed["sim001"] = "results.ready"
    await engine.tick()
    assert registry

    # A fresh campaign with the same spec, but a budget already exhausted: the
    # reusable leaf must still be satisfied (no submission, no wait-for-budget).
    fresh_campaign = _campaign([spec])
    exhausted = fresh_campaign.usage.model_copy(update={"jobs_submitted": 9})
    fresh_campaign = fresh_campaign.model_copy(update={"budget": Budget(max_total_jobs=0), "usage": exhausted})
    store.save(fresh_campaign)
    fresh = _engine(store, submit, registry, observed)
    tick = await fresh.tick()
    assert tick.reused == ["a0"]
    assert tick.waiting == []
    assert tick.submitted == []
    assert tick.complete is True
    assert submit.calls == 1


async def test_reuse_is_not_blocked_by_the_approval_gate(tmp_path) -> None:
    """Reuse spends no resources, so require_approval must not hold it."""
    from pic_agentic.agenda.engine import EnginePolicy

    store = _store(tmp_path)
    spec = {"replica": 0}
    store.save(_campaign([spec]))
    registry = {engine_key(spec): ReuseRecord(wire_hash=engine_key(spec), sim_id="simExisting", state="done")}
    submit = _Submitter()

    def lookup(key: str) -> ReuseRecord | None:
        record = registry.get(key)
        return record if record is not None and record.state == "done" else None

    engine = AgendaEngine(
        store=store,
        submit=submit,
        observe=dict,
        reuse_lookup=lookup,
        policy=EnginePolicy(require_approval=True),
    )
    tick = await engine.tick()
    assert tick.reused == ["a0"]
    assert tick.pending_approval == []
    assert submit.calls == 0


async def test_failed_run_is_not_reused(tmp_path) -> None:
    """A recorded failed state is not reusable; the leaf is submitted normally."""
    store = _store(tmp_path)
    spec = {"replica": 0}
    store.save(_campaign([spec]))
    registry = {engine_key(spec): ReuseRecord(wire_hash=engine_key(spec), sim_id="simOld", state="failed")}
    submit = _Submitter()

    def lookup(key: str) -> ReuseRecord | None:
        record = registry.get(key)
        return record if record is not None and record.state == "done" else None

    engine = AgendaEngine(store=store, submit=submit, observe=dict, reuse_lookup=lookup)
    tick = await engine.tick()
    assert tick.reused == []
    assert tick.submitted == ["a0"]
    assert submit.calls == 1


async def test_job_finished_without_results_is_not_recorded(tmp_path) -> None:
    """Only results.ready is reusable; job_finished alone may lack simOutput."""
    store = _store(tmp_path)
    store.save(_campaign([{"replica": 0}]))
    registry: dict[str, ReuseRecord] = {}
    observed = {"sim001": "simulation.job_finished"}
    submit = _Submitter()
    engine = _engine(store, submit, registry, observed)
    await engine.tick()  # submits
    await engine.tick()  # observes job_finished
    assert registry == {}


async def test_two_identical_leaves_with_a_registry_hit_both_reuse(tmp_path) -> None:
    """Both identical leaves are satisfied by reuse (no DuplicateSpecError).

    Regression: the duplicate-spec guard ran before reuse, so two identical
    planned leaves raised instead of both being satisfied by the same record.
    """
    store = _store(tmp_path)
    spec = {"replica": 0}
    store.save(_campaign([spec, spec]))
    key = engine_key(spec)
    registry = {key: ReuseRecord(wire_hash=key, sim_id="simExisting", state="done")}
    submit = _Submitter()

    def lookup(k: str) -> ReuseRecord | None:
        record = registry.get(k)
        return record if record is not None and record.state == "done" else None

    engine = AgendaEngine(store=store, submit=submit, observe=dict, reuse_lookup=lookup)
    tick = await engine.tick()
    assert tick.reused == ["a0", "a1"]
    assert tick.submitted == []
    assert submit.calls == 0
    assert tick.complete is True


async def test_engine_without_reuse_callables_behaves_as_before(tmp_path) -> None:
    """Omitting the reuse hooks disables reuse entirely (offline default)."""
    store = _store(tmp_path)
    store.save(_campaign([{"replica": 0}]))
    submit = _Submitter()
    engine = AgendaEngine(store=store, submit=submit, observe=dict)
    tick = await engine.tick()
    assert tick.submitted == ["a0"]
    assert tick.reused == []
