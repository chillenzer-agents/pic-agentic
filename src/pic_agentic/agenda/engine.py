# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""The campaign engine: drive an agenda to completion, durably and idempotently.

The engine owns no authoritative state: every :meth:`AgendaEngine.tick` loads
the :class:`~pic_agentic.agenda.campaign.Campaign` from its store, folds in the
observed simulation states, asks the pure planner what to do next, submits what
it may, and saves the campaign back.  A fresh engine over the same store
therefore resumes exactly where the last tick stopped -- that is what lets a
campaign survive MCP-server restarts, and what makes a tick idempotent (a leaf
that already carries a ``sim_id`` is never submitted again).

This module is transport- and protocol-agnostic: submission and observation are
injected callables.  It imports only the agenda model/budget/planner/campaign
layers plus stdlib and pydantic.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from pic_agentic.agenda.budget import Budget, BudgetUsage
from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.planner import PlanStep, account, apply_states, next_actions

if TYPE_CHECKING:
    from pic_agentic.agenda.model import AgendaGroup
    from pic_agentic.agenda.store import AgendaStore

#: Async ``spec -> sim_id`` submission callable.
SubmitFn = Callable[[dict[str, Any]], Awaitable[str]]

#: ``() -> {sim_id: state}`` observation callable.
ObserveFn = Callable[[], Mapping[str, str]]

log = logging.getLogger(__name__)

#: A leaf status that still requires engine action.
_ACTIVE_STATUSES = frozenset({"planned", "submitted", "running"})


class EnginePolicy(BaseModel):
    """Limits and gates the engine applies on top of the budget."""

    model_config = ConfigDict(extra="forbid")

    #: Maximum number of new submissions per tick (fleet-size control).
    max_submits_per_tick: int = 8
    #: Gate every submission behind pre-approval.
    require_approval: bool = False
    #: Gate submissions whose estimated core-hours exceed this threshold.
    approve_over_est_core_hours: float | None = None


class TickResult(BaseModel):
    """The outcome of one engine tick."""

    model_config = ConfigDict(extra="forbid")

    submitted: list[str] = Field(default_factory=list)
    pending_approval: list[str] = Field(default_factory=list)
    waiting: list[str] = Field(default_factory=list)
    done: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    complete: bool = False
    usage: BudgetUsage = BudgetUsage()


class AgendaEngine:
    """Advance one campaign from a store, one durable tick at a time.

    Plain class (it holds callables and a store, i.e. runtime resources).
    """

    def __init__(
        self,
        *,
        store: AgendaStore,
        budget: Budget | None = None,
        submit: SubmitFn,
        observe: ObserveFn,
        policy: EnginePolicy | None = None,
        approve: Callable[[str], bool] | None = None,
    ) -> None:
        """Create an engine.

        Args:
            store: The campaign store (its ``filename`` must be the campaign
                file; callers use :data:`~pic_agentic.agenda.store.
                DEFAULT_CAMPAIGN_FILE`).
            budget: Hard caps overriding the campaign's stored budget.
            submit: Async ``spec -> sim_id`` submission callable.
            observe: ``() -> {sim_id: state}`` observation callable.
            policy: Engine limits/gates.
            approve: Optional ``path -> bool`` pre-approval predicate.

        """
        self.store = store
        self.budget_override = budget
        self.submit = submit
        self.observe = observe
        self.policy = policy or EnginePolicy()
        self.approve = approve

    async def tick(self) -> TickResult:
        """Run one durable, idempotent engine tick.

        Returns:
            The tick's outcome (and the campaign is persisted).

        """
        campaign = self.store.load(Campaign).with_created_ts()
        budget = self.budget_override or campaign.budget
        campaign, steps = self._plan(campaign, budget)
        attempted = 0
        result = TickResult()
        for path, step in steps:
            if step.action == "submit":
                if attempted >= self.policy.max_submits_per_tick:
                    result.waiting.append(path)
                    continue
                if not self._may_submit(step):
                    result.pending_approval.append(path)
                    continue
                campaign = await self._do_submit(campaign, path, step)
                result.submitted.append(path)
                attempted += 1
            else:
                campaign = self._persist_terminal(campaign, path, step.action)
                _bucket(result, path, step.action)
        result.complete = _is_complete(campaign.agenda)
        result.usage = campaign.usage
        self.store.save(campaign)
        return result

    @staticmethod
    def _persist_terminal(campaign: Campaign, path: str, action: str) -> Campaign:
        """Persist a planner-terminal decision the observation did not cover.

        A leaf blocked by a failed dependency is reported ``failed`` by the
        planner but has no observed state of its own, so its folded status would
        stay ``planned`` and the campaign could never be ``complete``.  Writing
        the terminal status back closes that gap.

        Returns:
            The campaign, with the leaf's status updated when terminal.

        """
        if action not in {"failed", "done"}:
            return campaign
        agenda = campaign.agenda.model_copy(deep=True)
        leaf = _leaf_by_path(agenda, path)
        if leaf is None or leaf.status in {"failed", "done"}:
            return campaign
        leaf.status = action
        return campaign.model_copy(update={"agenda": agenda})

    def _plan(self, campaign: Campaign, budget: Budget) -> tuple[Campaign, list[tuple[str, PlanStep]]]:
        """Fold observed states into the campaign and compute next actions.

        The folded agenda is written back into the returned campaign so that a
        tick persists observed progress (not only new submissions); otherwise
        ``complete``/``done`` could never be reached.

        Returns:
            The campaign with the folded agenda, and ``(path, step)`` pairs in
            planner order.

        """
        observed = self.observe()
        by_sim = {sim.sim_id: path for path, sim in campaign.agenda.simulations() if sim.sim_id}
        path_states = {by_sim[sim_id]: state for sim_id, state in observed.items() if sim_id in by_sim}
        agenda = apply_states(campaign.agenda, path_states)
        steps = next_actions(agenda, {}, budget=budget, usage=campaign.usage)
        return campaign.model_copy(update={"agenda": agenda}), [(s.path, s) for s in steps]

    def _may_submit(self, step: PlanStep) -> bool:
        """Whether the policy gates allow submitting ``step`` right now.

        Returns:
            True when the step may be submitted.

        """
        if self.approve is not None and self.approve(step.path):
            return True
        if self.policy.require_approval:
            return False
        threshold = self.policy.approve_over_est_core_hours
        if threshold is not None:
            est = float(step.spec.get("resources", {}).get("est_core_hours", 0.0)) if step.spec else 0.0
            if est > threshold:
                return False
        return True

    async def _do_submit(self, campaign: Campaign, path: str, step: PlanStep) -> Campaign:
        """Submit one leaf, record its sim_id and accrue usage.

        Returns:
            The updated campaign (a new object; the input is not mutated).

        """
        sim_id = await self.submit(step.spec)
        agenda = campaign.agenda.model_copy(deep=True)
        leaf = _leaf_by_path(agenda, path)
        if leaf is not None:
            leaf.sim_id = sim_id
            leaf.status = "submitted"
        resources = step.spec.get("resources", {}) if step.spec else {}
        usage = account(
            campaign.usage,
            step,
            core_hours=float(resources.get("est_core_hours", 0.0)),
            gpu_hours=float(resources.get("est_gpu_hours", 0.0)),
            is_gpu=bool(resources.get("is_gpu", False)),
        )
        return campaign.model_copy(update={"agenda": agenda, "usage": usage})

    def status(self) -> dict[str, Any]:
        """Return an aggregate, serialisable campaign status.

        Returns:
            Counts, usage and the per-leaf view.

        """
        campaign = self.store.load(Campaign)
        counts = {"planned": 0, "submitted": 0, "running": 0, "done": 0, "failed": 0}
        leaves: list[dict[str, Any]] = []
        for path, sim in campaign.agenda.simulations():
            counts[sim.status] = counts.get(sim.status, 0) + 1
            leaves.append(
                {"path": path, "status": sim.status, "sim_id": sim.sim_id, "point": sim.point},
            )
        return {
            "name": campaign.name,
            "complete": _is_complete(campaign.agenda),
            "counts": counts,
            "usage": campaign.usage.model_dump(),
            "leaves": leaves,
        }

    def campaign_report(self) -> dict[str, Any]:
        """Return the campaign-level provenance record (minimal milestone E).

        Returns:
            The lineage record: each leaf's path, sim_id, status, point and a
            content hash of its spec.

        """
        campaign = self.store.load(Campaign)
        leaves = [
            {
                "path": path,
                "sim_id": sim.sim_id,
                "status": sim.status,
                "point": sim.point,
                "spec_hash": _spec_hash(sim.spec),
            }
            for path, sim in campaign.agenda.simulations()
        ]
        return {
            "name": campaign.name,
            "created_ts": campaign.created_ts,
            "complete": _is_complete(campaign.agenda),
            "usage": campaign.usage.model_dump(),
            "leaves": leaves,
        }


def _bucket(result: TickResult, path: str, action: str) -> None:
    """Append ``path`` to the result bucket matching ``action``.

    Args:
        result: The tick result, updated in place.
        path: The leaf path.
        action: One of ``wait``/``done``/``failed``.

    """
    target = {"wait": result.waiting, "done": result.done, "failed": result.failed}.get(action)
    if target is not None:
        target.append(path)


def _is_complete(agenda: AgendaGroup) -> bool:
    """Whether every leaf has reached a terminal status.

    Returns:
        True when no leaf is still planned/submitted/running.

    """
    return all(sim.status not in _ACTIVE_STATUSES for _, sim in agenda.simulations())


def _leaf_by_path(agenda: AgendaGroup, path: str) -> Any:
    """Return the leaf at ``path`` (``a/b/c``), or None.

    Returns:
        The :class:`~pic_agentic.agenda.model.AgendaSim`, or None when absent.

    """
    node: Any = agenda
    parts = path.split("/")
    for name in parts[:-1]:
        node = node.entries.get(name)
        if node is None:
            return None
    return node.entries.get(parts[-1])


def _spec_hash(spec: Mapping[str, Any]) -> str:
    """Return a stable content hash of a leaf spec.

    Returns:
        The sha256 hex digest of the canonical JSON encoding.

    """
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["AgendaEngine", "EnginePolicy", "ObserveFn", "SubmitFn", "TickResult"]
