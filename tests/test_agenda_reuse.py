# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for content-addressed reuse of identical simulations."""

from __future__ import annotations

from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.engine import AgendaEngine
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


async def test_engine_reuses_a_completed_identical_spec(tmp_path) -> None:
    """A leaf whose spec already completed is linked, not re-submitted."""
    store = _store(tmp_path)
    spec = {"replica": 0}
    store.save(_campaign([spec]))

    completions: dict[str, ReuseRecord] = {}

    def lookup(wire_hash: str) -> ReuseRecord | None:
        record = completions.get(wire_hash)
        return record if record is not None and record.state == "done" else None

    def record(wire_hash: str, sim_id: str, state: str, run_dir: str | None) -> None:
        completions[wire_hash] = ReuseRecord(wire_hash=wire_hash, sim_id=sim_id, state=state, run_dir=run_dir)

    submit = _Submitter()
    engine = AgendaEngine(store=store, submit=submit, observe=dict, reuse_lookup=lookup, reuse_record=record)

    first = await engine.tick()
    assert first.submitted == ["a0"]
    assert first.reused == []
    assert submit.calls == 1

    # Simulate the run finishing: observe it done, tick once to fold + record.
    saved = store.load(Campaign)
    saved.agenda.entries["a0"].status = "done"
    saved.agenda.entries["a0"].sim_id = "sim001"
    store.save(saved)
    await engine.tick()
    assert completions, "the completed run should be recorded for reuse"

    # A fresh campaign with the *same* spec must be reused, not submitted.
    store.save(_campaign([spec]))
    fresh = AgendaEngine(store=store, submit=submit, observe=dict, reuse_lookup=lookup, reuse_record=record)
    tick = await fresh.tick()
    assert tick.reused == ["a0"]
    assert tick.submitted == []
    assert submit.calls == 1  # no second cluster job
    reused_leaf = store.load(Campaign).agenda.entries["a0"]
    assert reused_leaf.status == "done"
    assert reused_leaf.sim_id == "sim001"
    # A reuse is a decision point: a done callback is emitted.
    assert any(call.kind == "done" for call in tick.callbacks)


async def test_engine_without_reuse_callables_behaves_as_before(tmp_path) -> None:
    """Omitting the reuse hooks disables reuse entirely (offline default)."""
    store = _store(tmp_path)
    store.save(_campaign([{"replica": 0}]))
    submit = _Submitter()
    engine = AgendaEngine(store=store, submit=submit, observe=dict)
    tick = await engine.tick()
    assert tick.submitted == ["a0"]
    assert tick.reused == []
