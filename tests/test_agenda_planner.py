# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Unit tests for the agenda execution planner."""

from __future__ import annotations

import pytest

from pic_agentic.agenda.budget import Budget, BudgetUsage, ResourceRequest
from pic_agentic.agenda.model import AgendaGroup, AgendaSim, AgendaSweep
from pic_agentic.agenda.planner import (
    SIMULATION_STATE_STATUS,
    PlanStep,
    account,
    apply_states,
    next_actions,
    resource_request_from_spec,
)

#: Every ``SimulationState`` value (kept as plain strings on purpose: the
#: planner is protocol-decoupled and must not import the protocol module).
ALL_STATES = [
    "accepted",
    "simulation.submitted",
    "workflow.finished",
    "simulation.job_running",
    "simulation.job_finished",
    "simulation.job_failed",
    "simulation.step_finished",
    "results.ready",
    "simulation.failed",
    "simulation.checkpoint",
    "simulation.cancelled",
]


def _leaf(name: str, *, status: str = "planned", depends_on: list[str] | None = None, **spec: object) -> AgendaSim:
    return AgendaSim(
        name=name,
        spec={"sim": {"time_steps": 10}, **spec},
        status=status,
        depends_on=depends_on or [],
    )


def _actions(steps: list[PlanStep]) -> list[tuple[str, str]]:
    return [(step.path, step.action) for step in steps]


def _scan_group() -> AgendaGroup:
    scan = AgendaGroup(name="scan", sweep=AgendaSweep(parameter="intensity", values=[1.0, 2.0]))
    return scan.expand(scan.sweep, lambda point: {"sim": {"intensity": point["intensity"]}})


# --------------------------------------------------------------------------
# next_actions: dependency ordering
# --------------------------------------------------------------------------


def test_independent_leaves_are_submitted_in_path_order() -> None:
    agenda = AgendaGroup(name="study").add(a=_leaf("a"), b=_leaf("b"))
    steps = next_actions(agenda, {}, budget=Budget(), usage=BudgetUsage())
    assert _actions(steps) == [("a", "submit"), ("b", "submit")]


def test_dependency_makes_a_leaf_wait_until_done() -> None:
    agenda = AgendaGroup(name="study").add(first=_leaf("first"), second=_leaf("second", depends_on=["first"]))
    steps = next_actions(agenda, {}, budget=Budget(), usage=BudgetUsage())
    assert _actions(steps) == [("first", "submit"), ("second", "wait")]
    assert "first" in steps[1].reason

    # Once the dependency is observed done, the successor becomes submit.
    steps = next_actions(agenda, {"first": "results.ready"}, budget=Budget(), usage=BudgetUsage())
    assert _actions(steps) == [("first", "done"), ("second", "submit")]


def test_dependency_on_a_failed_leaf_never_submits_the_successor() -> None:
    agenda = AgendaGroup(name="study").add(first=_leaf("first"), second=_leaf("second", depends_on=["first"]))
    steps = next_actions(agenda, {"first": "simulation.job_failed"}, budget=Budget(), usage=BudgetUsage())
    # The successor is terminal (failed) with a distinct reason, not a
    # perpetual wait: a failed dependency can never become done.
    assert _actions(steps) == [("first", "failed"), ("second", "failed")]
    assert "dependency failed" in steps[1].reason


def test_dependency_as_a_nested_group_waits_for_all_leaves() -> None:
    agenda = AgendaGroup(name="study").add(scan=_scan_group(), summary=_leaf("summary", depends_on=["scan"]))
    paths = [path for path, _ in agenda.simulations()]

    # The group dependency is unmet while any leaf is not done.
    steps = next_actions(agenda, {}, budget=Budget(), usage=BudgetUsage())
    assert _actions(steps) == [
        ("scan/scan__intensity=1.0", "submit"),
        ("scan/scan__intensity=2.0", "submit"),
        ("summary", "wait"),
    ]

    half = {paths[0]: "results.ready"}
    steps = next_actions(agenda, half, budget=Budget(), usage=BudgetUsage())
    assert steps[-1].action == "wait"

    full = {paths[0]: "results.ready", paths[1]: "simulation.job_finished"}
    steps = next_actions(agenda, full, budget=Budget(), usage=BudgetUsage())
    assert _actions(steps)[-1] == ("summary", "submit")


def test_unresolvable_dependency_is_treated_as_not_done() -> None:
    agenda = AgendaGroup(name="study").add(leaf=_leaf("leaf", depends_on=["ghost"]))
    steps = next_actions(agenda, {}, budget=Budget(), usage=BudgetUsage())
    assert steps[0].action == "wait"
    assert "ghost" in steps[0].reason


def test_inflight_leaf_waits() -> None:
    for status in ("submitted", "running"):
        leaf = _leaf("leaf", status=status)
        steps = next_actions(AgendaGroup(name="g").add(leaf=leaf), {}, budget=Budget(), usage=BudgetUsage())
        assert steps[0].action == "wait"
        assert status in steps[0].reason


# --------------------------------------------------------------------------
# next_actions: budget gate
# --------------------------------------------------------------------------


def test_budget_blocked_submit_becomes_wait_with_reason_and_does_not_raise() -> None:
    agenda = AgendaGroup(name="study").add(leaf=_leaf("leaf"))
    budget = Budget(max_total_jobs=1)
    usage = BudgetUsage(jobs_submitted=1)
    steps = next_actions(agenda, {}, budget=budget, usage=usage)
    assert steps[0].action == "wait"
    assert steps[0].reason is not None
    assert "job budget" in steps[0].reason


def test_core_hour_cap_blocks_submit() -> None:
    agenda = AgendaGroup(name="study").add(leaf=_leaf("leaf", resources={"est_core_hours": 10.0}))
    steps = next_actions(agenda, {}, budget=Budget(max_core_hours=5.0), usage=BudgetUsage())
    assert steps[0].action == "wait"
    assert "core-hour" in steps[0].reason


# --------------------------------------------------------------------------
# apply_states: every SimulationState, plus absent/unknown
# --------------------------------------------------------------------------


@pytest.mark.parametrize("state", ALL_STATES)
def test_apply_states_maps_every_known_state(state: str) -> None:
    agenda = AgendaGroup(name="g").add(leaf=_leaf("leaf"))
    updated = apply_states(agenda, {"leaf": state})
    assert updated.entries["leaf"].status == SIMULATION_STATE_STATUS[state]


def test_apply_states_map_is_total_and_sensible() -> None:
    assert SIMULATION_STATE_STATUS == {
        "accepted": "submitted",
        "simulation.submitted": "submitted",
        "simulation.job_running": "running",
        "simulation.step_finished": "running",
        "workflow.finished": "running",
        "simulation.checkpoint": "running",
        "results.ready": "done",
        "simulation.job_finished": "done",
        "simulation.job_failed": "failed",
        "simulation.failed": "failed",
        "simulation.cancelled": "failed",
    }


def test_apply_states_absent_and_unknown_leave_status_untouched() -> None:
    agenda = AgendaGroup(name="g").add(leaf=_leaf("leaf", status="running"))
    assert apply_states(agenda, {}).entries["leaf"].status == "running"
    assert apply_states(agenda, {"leaf": "not-a-state"}).entries["leaf"].status == "running"
    assert apply_states(agenda, {"other": "results.ready"}).entries["leaf"].status == "running"


def test_next_actions_applies_states_itself() -> None:
    agenda = AgendaGroup(name="g").add(leaf=_leaf("leaf"))
    direct = next_actions(agenda, {"leaf": "results.ready"}, budget=Budget(), usage=BudgetUsage())
    staged = next_actions(apply_states(agenda, {"leaf": "results.ready"}), {}, budget=Budget(), usage=BudgetUsage())
    assert direct == staged
    assert direct[0].action == "done"


# --------------------------------------------------------------------------
# purity
# --------------------------------------------------------------------------


def test_apply_states_does_not_mutate_the_source() -> None:
    agenda = AgendaGroup(name="g").add(leaf=_leaf("leaf"))
    before = agenda.model_dump_json()
    apply_states(agenda, {"leaf": "results.ready"})
    assert agenda.model_dump_json() == before


def test_next_actions_does_not_mutate_the_source() -> None:
    agenda = AgendaGroup(name="g").add(leaf=_leaf("leaf"))
    before = agenda.model_dump_json()
    next_actions(agenda, {"leaf": "results.ready"}, budget=Budget(), usage=BudgetUsage())
    assert agenda.model_dump_json() == before


def test_account_is_pure_and_adds_cost_and_one_job() -> None:
    usage = BudgetUsage(core_hours=1.0, gpu_hours=2.0, jobs_submitted=3, jobs_running=1)
    step = PlanStep(name="leaf", path="leaf", spec={}, action="submit")
    out = account(usage, step, core_hours=4.0, gpu_hours=8.0, is_gpu=True)
    assert out == BudgetUsage(core_hours=5.0, gpu_hours=10.0, jobs_submitted=4, jobs_running=1)
    # The input is untouched: both Pydantic models declare (immutable) floats,
    # so compare the full serialised value.
    assert usage.model_dump() == {"core_hours": 1.0, "gpu_hours": 2.0, "jobs_submitted": 3, "jobs_running": 1}


def test_account_ignores_gpu_cost_for_non_gpu_steps() -> None:
    step = PlanStep(name="leaf", path="leaf", spec={}, action="submit")
    out = account(BudgetUsage(), step, core_hours=2.0, gpu_hours=9.0, is_gpu=False)
    assert out == BudgetUsage(core_hours=2.0, gpu_hours=0.0, jobs_submitted=1)


# --------------------------------------------------------------------------
# resource_request_from_spec
# --------------------------------------------------------------------------


def test_resource_request_reads_nested_then_top_level() -> None:
    nested = resource_request_from_spec({"resources": {"est_core_hours": 3.0, "is_gpu": True}})
    assert nested == ResourceRequest(est_core_hours=3.0, is_gpu=True)
    # Top-level keys are a fallback; nested wins on conflict.
    mixed = resource_request_from_spec({"est_core_hours": 1.0, "resources": {"est_core_hours": 2.0}})
    assert mixed == ResourceRequest(est_core_hours=2.0)
    assert resource_request_from_spec({}) == ResourceRequest()


# --------------------------------------------------------------------------
# end-to-end cycle: sweep -> plan -> apply -> account
# --------------------------------------------------------------------------


def test_full_sweep_plan_apply_account_cycle() -> None:
    agenda = AgendaGroup(name="study").add(scan=_scan_group(), summary=_leaf("summary", depends_on=["scan"]))
    budget = Budget(max_total_jobs=3)
    usage = BudgetUsage()

    # 1. Plan the fresh agenda: both sweep leaves submit, the summary waits.
    steps = next_actions(agenda, {}, budget=budget, usage=usage)
    assert _actions(steps) == [
        ("scan/scan__intensity=1.0", "submit"),
        ("scan/scan__intensity=2.0", "submit"),
        ("summary", "wait"),
    ]

    # 2. Account the two admitted submissions.
    paths = [step.path for step in steps]
    usage = account(usage, steps[0], core_hours=10.0)
    usage = account(usage, steps[1], core_hours=20.0)
    assert usage == BudgetUsage(core_hours=30.0, gpu_hours=0.0, jobs_submitted=2, jobs_running=0)

    # 3. Observe the sweep done, plan again: the summary is now submittable.
    observed = {paths[0]: "results.ready", paths[1]: "simulation.job_finished"}
    updated = apply_states(agenda, observed)
    statuses = {path: sim.status for path, sim in updated.simulations()}
    assert statuses[paths[0]] == "done"
    assert statuses[paths[1]] == "done"
    assert statuses["summary"] == "planned"

    steps = next_actions(updated, {}, budget=budget, usage=usage)
    assert _actions(steps) == [
        ("scan/scan__intensity=1.0", "done"),
        ("scan/scan__intensity=2.0", "done"),
        ("summary", "submit"),
    ]

    # 4. Account the summary; the 3-job cap is now reached exactly.
    summary_step = steps[-1]
    usage = account(usage, summary_step, gpu_hours=5.0, is_gpu=True)
    assert usage == BudgetUsage(core_hours=30.0, gpu_hours=5.0, jobs_submitted=3, jobs_running=0)

    # 5. A further plan cannot submit the summary: the job cap blocks it.
    steps = next_actions(agenda, observed, budget=budget, usage=usage)
    summary = next(step for step in steps if step.path == "summary")
    assert summary.action == "wait"
    assert summary.reason is not None
    assert "job budget" in summary.reason


def test_batch_plan_never_overcommits_the_budget() -> None:
    """H2: a batch of ready leaves must not plan past the caps."""
    agenda = AgendaGroup(name="r")
    for i in range(5):
        agenda = agenda.add_sim(name=f"j{i}", spec={"resources": {"est_core_hours": 1.0}})

    steps = next_actions(agenda, {}, budget=Budget(max_core_hours=3.0), usage=BudgetUsage())
    submits = [s for s in steps if s.action == "submit"]
    waits = [s for s in steps if s.action == "wait"]
    assert len(submits) == 3
    assert len(waits) == 2
    assert all("core-hour" in s.reason for s in waits)


def test_batch_plan_respects_total_job_cap() -> None:
    agenda = AgendaGroup(name="r")
    for i in range(4):
        agenda = agenda.add_sim(name=f"j{i}", spec={})
    steps = next_actions(agenda, {}, budget=Budget(max_total_jobs=2), usage=BudgetUsage())
    assert sum(s.action == "submit" for s in steps) == 2


def test_batch_plan_accounts_gpu_hours() -> None:
    agenda = AgendaGroup(name="r")
    for i in range(3):
        agenda = agenda.add_sim(name=f"g{i}", spec={"resources": {"est_gpu_hours": 2.0, "is_gpu": True}})
    steps = next_actions(agenda, {}, budget=Budget(max_gpu_hours=5.0), usage=BudgetUsage())
    assert sum(s.action == "submit" for s in steps) == 2
