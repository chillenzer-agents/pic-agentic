# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""The deterministic agenda execution planner (milestone C-core).

Given a persisted :class:`~pic_agentic.agenda.model.AgendaGroup`, the observed
lifecycle state of each simulation and the agenda's resource budget, this module
decides the *next actionable step* for every leaf.  It is the offline-testable
decision core an execution engine drives; it never submits, waits or mutates
anything itself.

The module is deliberately **decoupled**: it imports only the standard library,
``pydantic`` and the agenda model/budget layers.  Observed states arrive as
plain strings (the ``SimulationState`` values) rather than as protocol objects,
so this core can move to the pypicongpu layer without a protocol dependency.

Three pure operations compose the planner:

- :func:`apply_states` folds an observed ``path -> state`` map onto the agenda,
  producing a new agenda; the source is never mutated.
- :func:`next_actions` turns that (or any) agenda into a list of
  :class:`PlanStep`s in deterministic path order, honouring ``depends_on`` and
  the budget (a submission the budget would reject becomes ``wait`` with a
  reason instead of raising -- planning is advisory).
- :func:`account` returns a new :class:`~pic_agentic.agenda.budget.BudgetUsage`
  with the actual cost and one submitted job added; the input is untouched.

Resource estimates for a prospective submission are read from the leaf's opaque
``spec`` using the documented convention in
:func:`resource_request_from_spec`: an optional ``spec["resources"]`` mapping
with ``est_core_hours``/``est_gpu_hours``/``partition``/``is_gpu`` keys, with
those same keys accepted at the top level as a fallback.  Anything absent
defaults to :class:`~pic_agentic.agenda.budget.ResourceRequest`'s own defaults.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from pic_agentic.agenda.budget import (
    Budget,
    BudgetExceededError,
    BudgetUsage,
    ResourceRequest,
    check_admission,
)
from pic_agentic.agenda.model import AgendaGroup, AgendaSim

#: Coarse leaf status derived from an observed ``SimulationState`` value.
#:
#: ``accepted``/``simulation.submitted`` mean the command was received but the
#: job has not started; ``job_running``/``step_finished``/``workflow.finished``
#: and the non-terminal ``checkpoint`` mean work is in flight; ``results.ready``
#: and ``job_finished`` are terminal success; ``job_failed``/``failed``/
#: ``cancelled`` are terminal failure.  A value absent from this map leaves the
#: leaf's existing status unchanged.
SIMULATION_STATE_STATUS: dict[str, Literal["submitted", "running", "done", "failed"]] = {
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

#: Leaf ``spec`` keys promoted into a :class:`ResourceRequest`.
_RESOURCE_KEYS = ("est_core_hours", "est_gpu_hours", "partition", "is_gpu")

PlanAction = Literal["submit", "wait", "done", "failed"]


class PlanStep(BaseModel):
    """The planner's decision for one agenda leaf.

    ``action`` is the next thing to do: ``submit`` the job, ``wait`` for it (or
    for a dependency, or for budget headroom), or record that the leaf is
    already ``done``/``failed``.  A leaf whose *dependency* failed is reported
    as ``failed`` with a ``reason`` naming the dependency (a failed dependency
    can never become done, so the successor is terminal rather than waiting
    forever).  ``reason`` explains a non-obvious decision and is ``None``
    otherwise.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    path: str
    spec: dict[str, Any]
    point: dict[str, float | int | str] | None = None
    action: PlanAction
    reason: str | None = None


def resource_request_from_spec(spec: Mapping[str, Any]) -> ResourceRequest:
    """Derive the prospective resource request from a leaf's opaque spec.

    Reads ``spec["resources"]`` (a mapping) first, then the same keys at the top
    level as a fallback.  Unknown keys are ignored so a spec may carry unrelated
    data; missing keys fall back to :class:`ResourceRequest`'s defaults.

    Args:
        spec: The leaf's opaque execution payload.

    Returns:
        The resource request the budget gate should check.

    """
    data: dict[str, Any] = {}
    nested = spec.get("resources")
    if isinstance(nested, Mapping):
        data.update({key: nested[key] for key in _RESOURCE_KEYS if key in nested})
    for key in _RESOURCE_KEYS:
        if key in spec:
            data.setdefault(key, spec[key])
    return ResourceRequest(**data)


def _index_entries(agenda: AgendaGroup) -> tuple[dict[str, AgendaSim | AgendaGroup], dict[str, AgendaGroup]]:
    """Index an agenda's entries by path plus each entry's containing group.

    Returns:
        A ``(paths, parents)`` pair: ``paths`` maps every entry's full path to
        the entry, ``parents`` maps it to the :class:`AgendaGroup` that holds it.

    """
    paths: dict[str, AgendaSim | AgendaGroup] = {}
    parents: dict[str, AgendaGroup] = {}

    def walk(group: AgendaGroup, prefix: str) -> None:
        for name, entry in group.entries.items():
            path = f"{prefix}/{name}" if prefix else name
            paths[path] = entry
            parents[path] = group
            if isinstance(entry, AgendaGroup):
                walk(entry, path)

    walk(agenda, "")
    return paths, parents


def _subtree_done(entry: AgendaSim | AgendaGroup) -> bool:
    """Return whether every leaf below ``entry`` has status ``done``.

    Returns:
        ``True`` for a done leaf or a group all of whose descendants are done.

    """
    if isinstance(entry, AgendaGroup):
        return all(_subtree_done(child) for child in entry.entries.values())
    return entry.status == "done"


def _subtree_failed(entry: AgendaSim | AgendaGroup) -> bool:
    """Whether a dependency subtree contains a failed leaf.

    A dependency containing any failed leaf can never become done, so its
    successors are blocked rather than perpetually "waiting".

    Returns:
        True when the entry or any descendant is a failed leaf.

    """
    if isinstance(entry, AgendaGroup):
        return any(_subtree_failed(child) for child in entry.entries.values())
    return entry.status == "failed"


def _resolve_dependency(
    dep: str,
    parent: AgendaGroup | None,
    paths: Mapping[str, AgendaSim | AgendaGroup],
) -> AgendaSim | AgendaGroup | None:
    """Resolve one ``depends_on`` name to the entry it refers to.

    Resolution is sibling-first (matching the CWL emitter's semantics): a bare
    name is looked up among the leaf's siblings, then as a full path, then by
    its last path segment.  An unresolvable name returns ``None`` and is treated
    as not-done by the caller.

    Returns:
        The referenced entry, or ``None`` when it cannot be resolved.

    """
    if parent is not None and dep in parent.entries:
        return parent.entries[dep]
    if dep in paths:
        return paths[dep]
    if parent is not None:
        last = dep.rsplit("/", maxsplit=1)[-1]
        if last in parent.entries:
            return parent.entries[last]
    return None


def apply_states(agenda: AgendaGroup, states: Mapping[str, str]) -> AgendaGroup:
    """Return a new agenda with leaf statuses folded from observed states.

    Pure: the source agenda is not mutated.  Each leaf's full path is looked up
    in ``states``; a recognised state maps through :data:`SIMULATION_STATE_STATUS`
    to a coarse status.  An absent or unknown state leaves the leaf's existing
    status unchanged.

    Args:
        agenda: The agenda to update.
        states: Observed ``leaf path -> SimulationState value`` mapping.

    Returns:
        The updated agenda (a deep copy).

    """
    updated = agenda.model_copy(deep=True)
    for path, sim in updated.simulations():
        mapped = SIMULATION_STATE_STATUS.get(states.get(path, ""))
        if mapped is not None:
            sim.status = mapped
    return updated


def next_actions(
    agenda: AgendaGroup,
    states: Mapping[str, str],
    *,
    budget: Budget,
    usage: BudgetUsage,
) -> list[PlanStep]:
    """Decide the next action for every leaf, in deterministic path order.

    Observed ``states`` are folded onto the agenda first (see
    :func:`apply_states`), so a caller may pass a persisted agenda plus fresh
    states without pre-applying them.  A leaf is then planned as follows:

    - already ``done``/``failed`` -> mirror that action;
    - already ``submitted``/``running`` -> ``wait`` (it is in flight);
    - ``planned`` with every dependency done -> ``submit``, unless
      :func:`~pic_agentic.agenda.budget.check_admission` would reject it, in
      which case it becomes ``wait`` with the budget reason (never raises);
    - ``planned`` with an unmet dependency -> ``wait`` naming the dependency.

    Args:
        agenda: The persisted agenda.
        states: Observed ``leaf path -> SimulationState value`` mapping.
        budget: The agenda's hard resource caps.
        usage: Resources already consumed.

    Returns:
        One :class:`PlanStep` per leaf, leaves first per group in insertion
        order (the same order as :meth:`AgendaGroup.simulations`).

    """
    effective = apply_states(agenda, states)
    paths, parents = _index_entries(effective)
    steps: list[PlanStep] = []
    # Thread a *running* usage through the plan: each emitted ``submit``
    # consumes its estimated budget immediately, so a batch can never plan more
    # concurrent work than the caps allow (per-leaf checks alone would).
    running = usage
    for path, sim in effective.simulations():
        step = _plan_leaf(sim, path, parents.get(path), paths, budget=budget, usage=running)
        if step.action == "submit":
            request = resource_request_from_spec(sim.spec)
            running = account(
                running,
                step,
                core_hours=request.est_core_hours,
                gpu_hours=request.est_gpu_hours,
                is_gpu=request.is_gpu,
            )
        steps.append(step)
    return steps


def _plan_leaf(
    sim: AgendaSim,
    path: str,
    parent: AgendaGroup | None,
    paths: Mapping[str, AgendaSim | AgendaGroup],
    *,
    budget: Budget,
    usage: BudgetUsage,
) -> PlanStep:
    """Plan a single leaf (the branch table of :func:`next_actions`).

    Returns:
        The leaf's plan step.

    """
    base = PlanStep(name=sim.name, path=path, spec=dict(sim.spec), point=sim.point, action="wait")
    if sim.status in {"done", "failed"}:
        return base.model_copy(update={"action": sim.status})
    if sim.status in {"submitted", "running"}:
        return base.model_copy(update={"reason": f"already {sim.status}"})
    unmet: list[str] = []
    blocked: list[str] = []
    for dep in sim.depends_on:
        resolved = _resolve_dependency(dep, parent, paths)
        if resolved is None:
            unmet.append(dep)
        elif _subtree_failed(resolved):
            # A failed dependency will never become done: propagate the failure
            # distinctly so the engine can branch instead of waiting forever.
            blocked.append(dep)
        elif not _subtree_done(resolved):
            unmet.append(dep)
    if blocked:
        return base.model_copy(update={"action": "failed", "reason": f"dependency failed: {', '.join(blocked)}"})
    if unmet:
        return base.model_copy(update={"action": "wait", "reason": f"dependency not done: {', '.join(unmet)}"})
    try:
        check_admission(budget, usage, resource_request_from_spec(sim.spec))
    except BudgetExceededError as exc:
        return base.model_copy(update={"action": "wait", "reason": str(exc)})
    return base.model_copy(update={"action": "submit"})


def account(
    usage: BudgetUsage,
    step: PlanStep,
    *,
    core_hours: float = 0.0,
    gpu_hours: float = 0.0,
    is_gpu: bool = False,
) -> BudgetUsage:
    """Return ``usage`` incremented by one admitted submission's actual cost.

    Pure: the input usage is not mutated.  ``core_hours`` is always added;
    ``gpu_hours`` is added only when ``is_gpu`` is set, mirroring
    :func:`~pic_agentic.agenda.budget.check_admission`'s GPU accounting.
    ``jobs_submitted`` is incremented by one; ``jobs_running`` is left to the
    caller (a submission is not yet a running job).

    Args:
        usage: Resources consumed so far.
        step: The admitted plan step (advisory; carried for context).
        core_hours: Actual core-hours the step consumed.
        gpu_hours: Actual GPU-hours the step consumed.
        is_gpu: Whether the step ran on a GPU.

    Returns:
        A new usage value with the cost and one job added.

    """
    _ = step
    return usage.model_copy(
        update={
            "core_hours": usage.core_hours + core_hours,
            "gpu_hours": usage.gpu_hours + (gpu_hours if is_gpu else 0.0),
            "jobs_submitted": usage.jobs_submitted + 1,
        },
    )


__all__ = [
    "SIMULATION_STATE_STATUS",
    "PlanAction",
    "PlanStep",
    "account",
    "apply_states",
    "next_actions",
    "resource_request_from_spec",
]
