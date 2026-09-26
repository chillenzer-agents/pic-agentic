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

Submission is **exactly-once even across a lost acknowledgement**: each leaf is
submitted under a deterministic idempotency key derived from the campaign name,
the leaf path and a hash of its spec, so a retry after a crash re-uses the same
cluster-side command id and the simclient replays the original ack instead of
running a second job.  The campaign is also saved *incrementally* (right after
each submission), so a crash mid-tick never loses the leaves already accepted.

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
from pic_agentic.agenda.campaign import Callback, Campaign, utc_now_iso
from pic_agentic.agenda.model import AgendaGroup, AgendaSim
from pic_agentic.agenda.planner import (
    PlanStep,
    account,
    apply_states,
    next_actions,
    resource_request_from_spec,
)

if TYPE_CHECKING:
    from pic_agentic.agenda.store import AgendaStore

#: Async ``(spec, idempotency_key) -> sim_id`` submission callable.  The key is
#: stable across retries so a lost ack re-uses the same cluster-side command id
#: (exactly-once at the cluster, not merely at-most-once here).
SubmitFn = Callable[[dict[str, Any], str], Awaitable[str]]

#: ``() -> {sim_id: state}`` observation callable.
ObserveFn = Callable[[], Mapping[str, str]]

log = logging.getLogger(__name__)

#: A leaf status that still requires engine action.
_ACTIVE_STATUSES = frozenset({"planned", "submitted", "running"})

#: Leaf statuses that count as an in-flight job for the concurrency cap.
_RUNNING_STATUSES = frozenset({"submitted", "running"})


class DuplicateSpecError(RuntimeError):
    """Raised when two leaves carry identical specs (they would collide).

    A simulation's ``sim_id`` is derived from the payload hash, so two leaves
    with byte-identical specs collapse into one cluster job and one registry
    record -- the observation of the second silently drives the first.  Replicas
    must therefore be made distinct (the driver injects a per-replica tag);
    submitting an identical spec is refused loudly instead.
    """


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
    #: Decision-point callbacks emitted this tick (newly done/failed leaves).
    callbacks: list[Callback] = Field(default_factory=list)


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
            submit: Async ``(spec, idempotency_key) -> sim_id`` callable.
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
        # Edge-triggered callbacks: capture the pre-tick statuses (persisted by
        # the last tick) so only genuine transitions to done/failed emit.
        before = {path: sim.status for path, sim in campaign.agenda.simulations()}
        budget = self.budget_override or campaign.budget
        campaign, steps = self._plan(campaign, budget)
        # Persist the folded observation before submitting anything: progress
        # observed in this tick survives a crash before the first submission,
        # and the incremental saves below lock in each accepted leaf.
        self.store.save(campaign)
        result = TickResult()
        in_flight = campaign.usage.jobs_running
        concurrency_cap = budget.max_concurrent_jobs
        for path, step in steps:
            if step.action != "submit":
                campaign = self._persist_terminal(campaign, path, step.action)
                _bucket(result, path, step.action)
                continue
            if len(result.submitted) >= self.policy.max_submits_per_tick:
                result.waiting.append(path)
                continue
            if concurrency_cap is not None and in_flight >= concurrency_cap:
                # The planner threads a *running* usage through the batch, but
                # it does not accrue jobs_running per emitted submit; enforce
                # the concurrency headroom here so one tick cannot over-fill.
                result.waiting.append(path)
                continue
            leaf = leaf_at(campaign.agenda, path)
            if not self._may_submit(leaf, step):
                result.pending_approval.append(path)
                continue
            campaign = await self._do_submit(campaign, path, step)
            # Incremental persistence: a crash after this point must not
            # resubmit the leaf (the next tick sees its recorded sim_id).
            self.store.save(campaign)
            result.submitted.append(path)
            in_flight += 1
        campaign, result.callbacks = self._emit_callbacks(campaign, before)
        result.complete = _is_complete(campaign.agenda)
        result.usage = campaign.usage
        self.store.save(campaign)
        return result

    @staticmethod
    def _emit_callbacks(campaign: Campaign, before: Mapping[str, str]) -> tuple[Campaign, list[Callback]]:
        """Append a callback for every leaf that newly reached done/failed.

        Edge-triggered against the *persisted* pre-tick statuses, so a restart
        between a transition and the agent's poll never re-emits: once the leaf
        is terminal on disk, a later tick sees no transition.  The callbacks are
        accumulated on the campaign (and drained by ``take_agenda_callbacks``),
        so they survive the restart that follows the transition.

        Returns:
            The campaign with the new callbacks appended, and the new callbacks.

        """
        emitted: list[Callback] = []
        for path, sim in campaign.agenda.simulations():
            if sim.status not in {"done", "failed"} or before.get(path) == sim.status:
                continue
            emitted.append(Callback(path=path, kind=sim.status, sim_id=sim.sim_id, ts=utc_now_iso()))
        if not emitted:
            return campaign, emitted
        return campaign.model_copy(update={"callbacks": [*campaign.callbacks, *emitted]}), emitted

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
        leaf = leaf_at(agenda, path)
        if leaf is None or leaf.status in {"failed", "done"}:
            return campaign
        leaf.status = action
        return campaign.model_copy(update={"agenda": agenda})

    def _plan(self, campaign: Campaign, budget: Budget) -> tuple[Campaign, list[tuple[str, PlanStep]]]:
        """Fold observed states into the campaign and compute next actions.

        The folded agenda is written back into the returned campaign so that a
        tick persists observed progress (not only new submissions); otherwise
        ``complete``/``done`` could never be reached.  ``jobs_running`` is
        recomputed from the observed in-flight leaves so the concurrency cap
        reflects reality across ticks.

        Returns:
            The campaign with the folded agenda, and ``(path, step)`` pairs in
            planner order.

        """
        observed = self.observe()
        by_sim = {sim.sim_id: path for path, sim in campaign.agenda.simulations() if sim.sim_id}
        path_states = {by_sim[sim_id]: state for sim_id, state in observed.items() if sim_id in by_sim}
        agenda = apply_states(campaign.agenda, path_states)
        in_flight = sum(1 for _, sim in agenda.simulations() if sim.status in _RUNNING_STATUSES)
        usage = campaign.usage.model_copy(update={"jobs_running": in_flight})
        steps = next_actions(agenda, {}, budget=budget, usage=usage)
        return campaign.model_copy(update={"agenda": agenda, "usage": usage}), [(s.path, s) for s in steps]

    def _may_submit(self, leaf: AgendaSim | None, step: PlanStep) -> bool:
        """Whether the policy gates allow submitting ``step`` right now.

        A leaf pre-approved either through the injected predicate or through its
        persisted ``approved`` flag (set by the ``approve_agenda_leaf`` tool) is
        always allowed; otherwise ``require_approval``, a per-leaf
        ``requires_approval`` and the estimated-core-hours threshold each gate.

        Returns:
            True when the step may be submitted.

        """
        if self.approve is not None and self.approve(step.path):
            return True
        if leaf is not None and leaf.approved:
            return True
        if (leaf is not None and leaf.requires_approval) or self.policy.require_approval:
            return False
        threshold = self.policy.approve_over_est_core_hours
        return threshold is None or resource_request_from_spec(step.spec).est_core_hours <= threshold

    async def _do_submit(self, campaign: Campaign, path: str, step: PlanStep) -> Campaign:
        """Submit one leaf, record its sim_id and accrue usage.

        The submission carries a deterministic idempotency key so a retried
        submit after a lost ack re-uses the same cluster-side command id and the
        simclient replays the original ack (exactly-once).

        Returns:
            The updated campaign (a new object; the input is not mutated).

        """
        _reject_duplicate_spec(campaign, path, step)
        key = _idempotency_key(campaign, path, step)
        sim_id = await self.submit(step.spec, key)
        agenda = campaign.agenda.model_copy(deep=True)
        leaf = leaf_at(agenda, path)
        if leaf is not None:
            leaf.sim_id = sim_id
            leaf.status = "submitted"
        request = resource_request_from_spec(step.spec)
        usage = account(
            campaign.usage,
            step,
            core_hours=request.est_core_hours,
            gpu_hours=request.est_gpu_hours,
            is_gpu=request.is_gpu,
        )
        # A freshly accepted job is in flight until an observation says
        # otherwise; the next tick recomputes jobs_running from the registry
        # (the authoritative source), so this is only the between-ticks view.
        usage = usage.model_copy(update={"jobs_running": usage.jobs_running + 1})
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
                {
                    "path": path,
                    "status": sim.status,
                    "sim_id": sim.sim_id,
                    "point": sim.point,
                    "requires_approval": sim.requires_approval,
                    "approved": sim.approved,
                },
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
            "revision": _revision(campaign),
            "usage": campaign.usage.model_dump(),
            "leaves": leaves,
        }


def leaf_at(agenda: AgendaGroup, path: str) -> AgendaSim | None:
    """Return the leaf at ``path`` (``a/b/c``), or None.

    Returns:
        The :class:`~pic_agentic.agenda.model.AgendaSim`, or None when absent
        (or when ``path`` names a group).

    """
    node: Any = agenda
    parts = path.split("/")
    for name in parts[:-1]:
        node = getattr(node, "entries", {}).get(name)
        if not isinstance(node, AgendaGroup):
            return None
    leaf = node.entries.get(parts[-1]) if isinstance(node, AgendaGroup) else None
    return leaf if isinstance(leaf, AgendaSim) else None


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


def _reject_duplicate_spec(campaign: Campaign, path: str, step: PlanStep) -> None:
    """Refuse an already-submitted identical spec (sim_id collision).

    Raises:
        DuplicateSpecError: If ``step``'s spec matches an already-submitted
            leaf's spec.

    """
    digest = _spec_hash(step.spec)
    for other_path, sim in campaign.agenda.simulations():
        if other_path == path or not sim.sim_id:
            continue
        if _spec_hash(sim.spec) == digest:
            msg = (
                f"duplicate spec at {path!r} and {other_path!r}: identical payloads map to the same "
                "sim_id and would collapse into one job; make replicas distinct (e.g. per-replica tag)"
            )
            raise DuplicateSpecError(msg)


def _idempotency_key(campaign: Campaign, path: str, step: PlanStep) -> str:
    """Return the stable, collision-free command id for one leaf submission.

    The key is a pure function of the persisted campaign data, so a fresh engine
    after a restart derives the same key and the simclient replays its recorded
    ack.  The spec hash is folded in so a *changed* spec is a genuinely new
    command (and therefore a new job), while a retry of the same spec is not.

    Returns:
        A 32-character lowercase hex command id.

    """
    seed = f"{campaign.name}\x00{path}\x00{_spec_hash(step.spec)}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def _spec_hash(spec: Mapping[str, Any]) -> str:
    """Return a stable content hash of a leaf spec.

    Returns:
        The sha256 hex digest of the canonical JSON encoding.

    """
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _revision(campaign: Campaign) -> str | None:
    """Return the pinned revision carried by the campaign's specs, if uniform.

    A spec may carry a ``provenance`` mapping (the server injects it into the
    wire payload); the campaign report surfaces a single revision when every
    leaf agrees on one, so the lineage record names the revision of the runs.

    Returns:
        The shared ``picongpu_revision``, or None when absent or mixed.

    """
    revisions = {
        str(sim.spec["provenance"]["picongpu_revision"])
        for _, sim in campaign.agenda.simulations()
        if isinstance(sim.spec.get("provenance"), Mapping) and sim.spec["provenance"].get("picongpu_revision")
    }
    return next(iter(revisions)) if len(revisions) == 1 else None


__all__ = [
    "AgendaEngine",
    "DuplicateSpecError",
    "EnginePolicy",
    "ObserveFn",
    "SubmitFn",
    "TickResult",
    "leaf_at",
]
