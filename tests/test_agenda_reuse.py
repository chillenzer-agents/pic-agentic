# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for content-addressed reuse of identical simulations."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.engine import AgendaEngine
from pic_agentic.agenda.engine import _wire_hash as engine_key
from pic_agentic.agenda.model import AgendaGroup, AgendaSim
from pic_agentic.agenda.reuse import PENDING_STATE, ReuseRecord, ReuseRegistry
from pic_agentic.agenda.store import DEFAULT_CAMPAIGN_FILE, AgendaStore
from pic_agentic.config import Config
from pic_agentic.protocol.simulation import (
    SimulationState,
    SimulationType,
    build_submit_ack,
    build_submit_event,
)
from pic_agentic.rcp import new_secret_hex
from pic_agentic.server.agenda import AgendaService
from pic_agentic.server.simulation import SubmitService
from pic_agentic.simulation_build import BuiltSimulation
from pic_agentic.transport.memory import MemoryTransport

_SECRET = new_secret_hex()
_SIM = "7f3a2b1c"
_RUNNER = Path(__file__).parent / "fixtures" / "pypicongpu_runner.json"


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

    def record(key: str, sim_id: str, state: str, run_id: str | None = None) -> None:
        registry[key] = ReuseRecord(wire_hash=key, sim_id=sim_id, state=state, run_id=run_id)

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


def test_registry_promote_flips_a_pending_run_to_reusable() -> None:
    """A pending run (recorded on acceptance) becomes reusable on results.ready."""
    key = engine_key({"replica": 0})
    registry = ReuseRegistry().remember(
        ReuseRecord(wire_hash=key, sim_id="s1", state=PENDING_STATE, run_id="run-1"),
    )
    assert registry.lookup(key) is None  # a pending run is not reusable
    promoted = registry.promote("run-1")
    assert promoted.lookup(key) is not None
    assert promoted.lookup(key).run_id == "run-1"  # type: ignore[union-attr]
    # A replayed results.ready (or a non-matching run id) is idempotent.
    assert promoted.promote("run-1").lookup(key).state == "done"  # type: ignore[union-attr]
    assert promoted.promote("other").lookup(key) is not None  # already done, unchanged
    assert registry.promote("unknown") is registry  # no match: self returned unchanged


async def test_direct_submission_run_is_recorded_with_its_run_batch(tmp_path) -> None:
    """A completed engine run records the run batch alongside the spec label.

    The registry must carry the run-batch identity (the engine's stable command
    id) so a reused leaf can name the run it linked to, distinct from the spec
    label ``sim_id``.
    """
    store = _store(tmp_path)
    spec = {"replica": 7}
    store.save(_campaign([spec]))
    registry: dict[str, ReuseRecord] = {}
    observed: dict[str, str] = {}
    submit = _Submitter()
    engine = _engine(store, submit, registry, observed)
    await engine.tick()
    observed["sim001"] = "results.ready"
    await engine.tick()
    record = registry[engine_key(spec)]
    assert record.sim_id == "sim001"
    assert record.run_id  # the run-batch identity was recorded
    assert record.state == "done"


# --- server wiring: a bare submit_simulation feeds the registry (H7) ---------


class _DirectResponder:
    """Ack each submit command; never emit lifecycle events (tests drive those)."""

    def __init__(self, transport: MemoryTransport, sim_id: str = "abcd1234") -> None:
        self.transport = transport
        self.sim_id = sim_id
        self.commands: list[str] = []

    async def run(self) -> None:
        counter = 0
        async for command in self.transport.receive():
            if command.type != SimulationType.COMMAND:
                continue
            counter += 1
            cmd_id = str(command.payload.get("cmd_id", ""))
            self.commands.append(cmd_id)
            ack = build_submit_ack(
                sim=_SIM,
                seq=counter,
                cmd_id=cmd_id,
                sim_id=self.sim_id,
                state=SimulationState.ACCEPTED,
                in_reply_to=command.transport_event_id,
            ).sign(_SECRET)
            await self.transport.send(ack)


async def _pump_into(transport: MemoryTransport, service: SubmitService) -> asyncio.Task:
    async def pump() -> None:
        async for message in transport.receive():
            service.on_message(message)

    return asyncio.create_task(pump())


def _direct_service(
    tmp_path: Path, *, picongpu_revision: str = "rev-test"
) -> tuple[AgendaService, SubmitService, BuiltSimulation]:
    runner = json.loads(_RUNNER.read_text())
    built = BuiltSimulation(
        runner=runner,
        picongpu_version="0.9.0-dev",
        picongpu_revision=picongpu_revision,
        schema_hash="schema-test",
    )
    service = SubmitService(sim=_SIM, secret=_SECRET, picongpu_revision=picongpu_revision)

    async def builder(*, script_path, **_kw):  # ruff: ignore[missing-type-kwargs]
        return built

    service.runner_dump_builder = builder
    config = Config(rcp_secret=_SECRET, agenda_file=str(tmp_path / "campaign.json"))
    agenda = AgendaService(config, service)
    service.on_direct_submission = agenda.remember_direct_spec
    service.on_run_ready = agenda.promote_reuse
    return agenda, service, built


async def test_direct_submission_is_reused_by_an_identical_campaign_leaf(tmp_path) -> None:
    """H7: a completed ad-hoc submit_simulation is reused, not re-run.

    The bare submission never went through the engine, so its result was absent
    from the registry and an identical campaign leaf re-ran it.  It is now
    recorded (pending) on acceptance and promoted on results.ready, so the
    campaign leaf links to it with no second cluster job.
    """
    from pic_agentic.server.agenda import _reuse_key

    agenda, service, built = _direct_service(tmp_path)
    mcp_t, sim_t = MemoryTransport.create_pair()
    responder = _DirectResponder(sim_t)
    tasks = [asyncio.create_task(responder.run()), await _pump_into(mcp_t, service)]
    try:
        outcome = await service.submit(mcp_t.send, Path("/tmp/anything.py"))
        assert outcome.ok
        assert outcome.run_id == outcome.cmd_id
        # The result arrives: promote the pending record deterministically.
        service.on_message(
            build_submit_event(
                sim=_SIM,
                seq=999,
                cmd_id=outcome.cmd_id,
                sim_id=outcome.sim_id,
                state=SimulationState.RESULTS_READY,
            ).sign(_SECRET),
        )
        record = agenda._load_reuse().lookup(_reuse_key({"sim": built.runner["sim"]}, "rev-test"))
        assert record is not None
        assert record.run_id == outcome.cmd_id

        # A campaign whose only leaf is byte-identical must reuse, not resubmit.
        group = AgendaGroup(name="g").add(leaf=AgendaSim(name="leaf", spec={"sim": built.runner["sim"]}))
        AgendaStore(tmp_path, filename="campaign.json").save(Campaign(name="camp", agenda=group))
        tick = await agenda.advance(mcp_t.send)
        assert tick["reused"] == ["leaf"]
        assert tick["submitted"] == []
        assert len(responder.commands) == 1  # no second cluster job
        leaf = agenda.store.load(Campaign).agenda.entries["leaf"]
        assert leaf.reused is True
        assert leaf.run_id == outcome.cmd_id
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()


async def test_direct_submission_pending_promotes_across_a_restart(tmp_path) -> None:
    """The in-memory pending-direct guard must survive a restart.

    ``promote_reuse`` now short-circuits a run with no in-memory pending entry,
    so the set must be hydrated from the persisted registry at construction:
    a fresh server that only replays the signed ``results.ready`` event (the
    command id is all the event carries) must still promote the pending record.
    """
    from pic_agentic.server.agenda import _reuse_key

    agenda, service, built = _direct_service(tmp_path)
    mcp_t, sim_t = MemoryTransport.create_pair()
    responder = _DirectResponder(sim_t)
    tasks = [asyncio.create_task(responder.run()), await _pump_into(mcp_t, service)]
    try:
        outcome = await service.submit(mcp_t.send, Path("/tmp/anything.py"))
        assert agenda._load_reuse().lookup(_reuse_key({"sim": built.runner["sim"]}, "rev-test")) is None
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()

    # A new process over the same files: only the signed event is replayed.
    fresh_service = SubmitService(sim=_SIM, secret=_SECRET, picongpu_revision="rev-test")
    fresh = AgendaService(Config(rcp_secret=_SECRET, agenda_file=str(tmp_path / "campaign.json")), fresh_service)
    fresh_service.on_run_ready = fresh.promote_reuse
    fresh_service.ingest_backfill(
        [
            build_submit_event(
                sim=_SIM,
                seq=999,
                cmd_id=outcome.cmd_id,
                sim_id=outcome.sim_id,
                state=SimulationState.RESULTS_READY,
            ).sign(_SECRET),
        ],
    )
    record = fresh._load_reuse().lookup(_reuse_key({"sim": built.runner["sim"]}, "rev-test"))
    assert record is not None
    assert record.run_id == outcome.cmd_id


async def test_direct_submission_under_a_different_revision_is_not_reused(tmp_path) -> None:
    """A different provenance tuple must not match: reuse stays attributable.

    With no configured revision the key falls back to the spec-carried
    provenance, so a leaf whose physics is attributed to another PIConGPU
    revision does not reuse the direct run and is submitted normally.
    """
    agenda, service, built = _direct_service(tmp_path, picongpu_revision="")
    mcp_t, sim_t = MemoryTransport.create_pair()
    responder = _DirectResponder(sim_t)
    tasks = [asyncio.create_task(responder.run()), await _pump_into(mcp_t, service)]
    try:
        outcome = await service.submit(mcp_t.send, Path("/tmp/anything.py"))
        service.on_message(
            build_submit_event(
                sim=_SIM,
                seq=999,
                cmd_id=outcome.cmd_id,
                sim_id=outcome.sim_id,
                state=SimulationState.RESULTS_READY,
            ).sign(_SECRET),
        )
        # The leaf's spec carries a different revision: it must not hit.
        leaf_spec = {"sim": built.runner["sim"], "provenance": {"picongpu_revision": "other-rev"}}
        group = AgendaGroup(name="g").add(leaf=AgendaSim(name="leaf", spec=leaf_spec))
        AgendaStore(tmp_path, filename="campaign.json").save(Campaign(name="camp", agenda=group))
        tick = await agenda.advance(mcp_t.send)
        assert tick["reused"] == []
        assert tick["submitted"] == ["leaf"]
        assert len(responder.commands) == 2
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()


async def test_two_direct_submissions_do_not_orphan_the_first_result(tmp_path) -> None:
    """A re-submit of an identical spec must not clobber a completed record.

    The ``submit -> inspect -> re-submit same script -> then campaign`` iteration
    submits the same spec twice.  The second (still-unfinished) run must not
    repoint the record's ``run_id`` away from the first, completed run -- doing
    so would orphan its result (its ``results.ready`` could no longer promote
    the record) and block the campaign leaf from reusing it.
    """
    from pic_agentic.server.agenda import _reuse_key

    agenda, service, built = _direct_service(tmp_path)
    mcp_t, sim_t = MemoryTransport.create_pair()
    responder = _DirectResponder(sim_t)
    tasks = [asyncio.create_task(responder.run()), await _pump_into(mcp_t, service)]
    try:
        run1 = await service.submit(mcp_t.send, Path("/tmp/anything.py"))
        run2 = await service.submit(mcp_t.send, Path("/tmp/anything.py"))
        assert run1.sim_id == run2.sim_id
        assert run1.cmd_id != run2.cmd_id

        # The first run's result arrives *after* the second was accepted.
        service.on_message(
            build_submit_event(
                sim=_SIM,
                seq=999,
                cmd_id=run1.cmd_id,
                sim_id=run1.sim_id,
                state=SimulationState.RESULTS_READY,
            ).sign(_SECRET),
        )
        record = agenda._load_reuse().lookup(_reuse_key({"sim": built.runner["sim"]}, "rev-test"))
        assert record is not None, "first completed run must remain reusable"
        assert record.run_id == run1.cmd_id  # the run that produced the result

        # The campaign leaf reuses the completed run; no third job is launched.
        group = AgendaGroup(name="g").add(leaf=AgendaSim(name="leaf", spec={"sim": built.runner["sim"]}))
        AgendaStore(tmp_path, filename="campaign.json").save(Campaign(name="camp", agenda=group))
        tick = await agenda.advance(mcp_t.send)
        assert tick["reused"] == ["leaf"]
        assert tick["submitted"] == []
        assert len(responder.commands) == 2  # only the two direct submits
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()


def test_record_direct_never_demotes_a_done_record() -> None:
    """Unit-level pin: a later submission of a done content key is a no-op."""
    key = engine_key({"replica": 0})
    done = ReuseRecord(wire_hash=key, sim_id="s1", state="done", run_id="run-1")
    registry = ReuseRegistry().remember(done)
    incoming = ReuseRecord(wire_hash=key, sim_id="s1", state=PENDING_STATE, run_id="run-2")
    assert registry.record_direct(incoming) is registry

    # A second pending run accumulates so *either* result can promote it.
    pending = ReuseRegistry().remember(
        ReuseRecord(wire_hash=key, sim_id="s1", state=PENDING_STATE, run_id="run-1"),
    )
    merged = pending.record_direct(incoming)
    assert merged.records[key].pending_run_ids == ["run-1", "run-2"]
    assert merged.records[key].run_id == "run-1"
    # Either run promotes; the promoting run becomes the surfaced identity.
    promoted = merged.promote("run-2")
    assert promoted.lookup(key).run_id == "run-2"  # type: ignore[union-attr]
    assert promoted.records[key].pending_run_ids == []


async def test_reused_leaf_records_the_run_it_was_linked_to(tmp_path) -> None:
    """A reused leaf carries the linked run's run_id, not just the sim label."""
    store = _store(tmp_path)
    spec = {"replica": 3}
    store.save(_campaign([spec]))
    key = engine_key(spec)
    registry = {
        key: ReuseRecord(wire_hash=key, sim_id="simExisting", state="done", run_id="run-original"),
    }

    def lookup(k: str) -> ReuseRecord | None:
        record = registry.get(k)
        return record if record is not None and record.state == "done" else None

    engine = AgendaEngine(store=store, submit=_Submitter(), observe=dict, reuse_lookup=lookup)
    tick = await engine.tick()
    assert tick.reused == ["a0"]
    leaf = store.load(Campaign).agenda.entries["a0"]
    assert leaf.reused is True
    assert leaf.sim_id == "simExisting"
    assert leaf.run_id == "run-original"
    # The run identity surfaces in status and the campaign report.
    status_leaf = engine.status()["leaves"][0]
    assert status_leaf["reused"] is True
    assert status_leaf["run_id"] == "run-original"
    assert engine.campaign_report()["leaves"][0]["run_id"] == "run-original"


# --- run identity in the server status surface (N1) -------------------------


async def test_status_tools_expose_run_id_beside_the_spec_label() -> None:
    """get_status/list_simulations carry run_id, distinct from the spec label.

    ``sim_id`` alone is ambiguous: re-runs share it.  Surface the run-batch
    identity so an operator can tell a re-run from a distinct study point.
    """
    from pic_agentic.server.app import build_server

    server, runtime = build_server(Config(rcp_secret=_SECRET), _SIM)
    record = runtime.submit_service._record_for("e8484fcd", cmd_id="run-abc")
    record.state = SimulationState.ACCEPTED.value
    runtime.submit_service.registry["e8484fcd"] = record

    status = (await server.call_tool("get_status", {"sim_id": "e8484fcd"})).structured_content
    assert status["sim_id"] == "e8484fcd"
    assert status["run_id"] == "run-abc"
    rows = (await server.call_tool("list_simulations", {})).structured_content["simulations"]
    assert rows[0]["sim_id"] == "e8484fcd"
    assert rows[0]["run_id"] == "run-abc"
