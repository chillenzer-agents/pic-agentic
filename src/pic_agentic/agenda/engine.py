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
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from pic_agentic.agenda.budget import Budget, BudgetUsage
from pic_agentic.agenda.campaign import Callback, Campaign, CampaignState, TickState, utc_now_iso
from pic_agentic.agenda.model import AgendaGroup, AgendaSim
from pic_agentic.agenda.planner import (
    PlanStep,
    account,
    apply_states,
    next_actions,
    reconcile,
    resource_request_from_spec,
)

if TYPE_CHECKING:
    from pic_agentic.agenda.reuse import ReuseRecord
    from pic_agentic.agenda.store import AgendaStore

#: Async ``(spec, idempotency_key) -> sim_id`` submission callable.  The key is
#: stable across retries so a lost ack re-uses the same cluster-side command id
#: (exactly-once at the cluster, not merely at-most-once here).
SubmitFn = Callable[[dict[str, Any], str], Awaitable[str]]

#: ``() -> {sim_id: state}`` observation callable.
ObserveFn = Callable[[], Mapping[str, str]]

#: ``(key) -> ReuseRecord | None``: lookup a completed run to reuse.
ReuseLookupFn = Callable[[str], "ReuseRecord | None"]
#: ``(key, sim_id, state) -> None``: record a completed run.
ReuseRecordFn = Callable[[str, str, str], None]

#: ``(spec) -> key``: the content key for reuse.  Defaults to the wire hash
#: (``{"sim": ...}``); the server overrides it to fold in the provenance tuple,
#: so a result produced under a different PIConGPU revision is not reused.
ReuseKeyFn = Callable[[Mapping[str, Any]], str]

#: ``() -> {sim_id: ActualUsage}`` actual-cost observation callable (gap 4).
ActualsFn = Callable[[], Mapping[str, "ActualUsage"]]

#: ``() -> {sim_id: FailureInfo}`` observation callable for failure reasons.
FailuresFn = Callable[[], Mapping[str, "FailureInfo"]]

#: ``() -> {sim_id: SuspectInfo}`` observation callable for the F4
#: "successful-but-empty" health flag.
SuspectsFn = Callable[[], Mapping[str, "SuspectInfo"]]

log = logging.getLogger(__name__)


class ActualUsage(BaseModel):
    """Actual resource usage the cluster reported for one finished run."""

    model_config = ConfigDict(extra="forbid")

    core_hours: float
    gpu_hours: float = 0.0
    is_gpu: bool = False


class FailureInfo(BaseModel):
    """The reason a simulation failed, as reported by the simclient."""

    model_config = ConfigDict(extra="forbid")

    error: str | None = None
    error_code: str | None = None
    stage: str | None = None


class SuspectInfo(BaseModel):
    """The "successful-but-empty" health detail for one finished run (F4)."""

    model_config = ConfigDict(extra="forbid")

    #: The all-zero warning text (reused from the plugin reader path).
    warning: str


#: A leaf status that still requires engine action.
_ACTIVE_STATUSES = frozenset({"planned", "submitted", "running"})

#: Leaf statuses that count as an in-flight job for the concurrency cap.
_RUNNING_STATUSES = frozenset({"submitted", "running"})

#: Observed ``SimulationState`` values whose run has linked results on disk and
#: is therefore safe to record as reusable.  ``results.ready`` is the state that
#: guarantees ``run_dir/simOutput`` exists (``job_finished`` alone may not).
_REUSABLE_OBSERVED_STATES = frozenset({"results.ready"})

#: A failed leaf's reason is truncated to this many characters in the
#: ``advance_agenda`` tick result.  Several leaves failing identically would
#: otherwise replay the same multi-KB validation dump once per leaf; the full
#: text stays on the persisted callback (``take_agenda_callbacks``) and in
#: ``agenda_status``.
FAILURE_MESSAGE_MAX_CHARS = 500
FAILURE_TRUNCATION_MARKER = "...(truncated)"


class DuplicateSpecError(RuntimeError):
    """Raised when two leaves carry identical specs (they would collide).

    A simulation's ``sim_id`` is derived from the payload hash, so two leaves
    with byte-identical specs collapse into one cluster job and one registry
    record -- the observation of the second silently drives the first.  Replicas
    must therefore be made distinct (the driver injects a per-replica tag);
    submitting an identical spec is refused loudly instead.
    """


#: Wall-clock window (seconds) a leaf may stay deferred before the engine stops
#: retrying and marks it failed with an actionable reason.  A pending
#: idempotency record (a simclient that died mid-build) leaves the leaf
#: ``planned`` so the retry can re-ack once the outcome is known, but a record
#: that *never* resolves must not retry forever and hold the campaign open.
#:
#: The bound is **elapsed time, not a tick count**: the pending record only
#: becomes ``completed`` when the simclient *finishes the build*, which can take
#: minutes (``execute_submit`` runs ``generate()`` + the CWL workflow), and an
#: agent can trigger ``advance_agenda`` several times in seconds.  Counting ticks
#: would therefore fail a leaf whose job is still building -- the very harm the
#: deferral exists to avoid.  Default 900 s (15 min), overridable per campaign
#: via :attr:`EnginePolicy.deferred_outcome_timeout_s`.
DEFAULT_DEFERRED_OUTCOME_TIMEOUT_S = 900.0

#: The leaf error code stamped when a deferred submission exhausts
#: :data:`DEFAULT_DEFERRED_OUTCOME_TIMEOUT_S`.  Distinct from a policy rejection
#: so an agent can tell "the ack was lost and never reconciled" from "the
#: simclient refused the spec".
OUTCOME_UNKNOWN_ERROR_CODE = "outcome_unknown"


class TransientSubmitError(RuntimeError):
    """A submission failure that may succeed on retry (e.g. a lost ack).

    A lost acknowledgement does not mean the job failed -- it may be running --
    and the engine's exactly-once idempotency key makes retrying it safe.  The
    engine therefore leaves the leaf ``planned`` (to retry next tick) for a
    transient error, but marks it ``failed`` for any other submission error (a
    rejected payload, a build failure), which a retry would only repeat.

    ``error_code`` names why the retry is worthwhile (e.g. ``outcome_unknown``)
    and is surfaced with any bounded-retry failure so the distinction from a
    terminal rejection survives into ``agenda_status``.  ``sim_id`` carries the
    simclient's declared simulation id when it is known despite the lost ack
    (the pending record was written before execution), so the deferred leaf can
    still be cancelled/cleaned up by the kill-switch rather than orphaned.
    """

    def __init__(self, message: str, *, error_code: str | None = None, sim_id: str | None = None) -> None:
        """Create the transient failure.

        Args:
            message: Human-readable reason (e.g. the lost-ack detail).
            error_code: Stable code for the transient condition, if any.
            sim_id: The simulation id the lost ack named, when known.

        """
        super().__init__(message)
        self.error_code = error_code
        self.sim_id = sim_id


class SubmitFailureError(RuntimeError):
    """A terminal submission rejection carrying the simclient's reason.

    The submit callable raises this (instead of a bare ``RuntimeError``) when
    the simclient rejected the spec: the ``error``/``error_code``/``stage`` are
    persisted onto the leaf and its failed callback so the campaign can surface
    *why* a leaf failed rather than only that it did.
    """

    def __init__(self, message: str, *, error_code: str | None = None, stage: str | None = None) -> None:
        """Create the failure.

        Args:
            message: The human-readable rejection reason.
            error_code: The simclient's stable machine-readable code, if any.
            stage: The pipeline stage the failure occurred in, if reported.

        """
        super().__init__(message)
        self.error_code = error_code
        self.stage = stage


class EnginePolicy(BaseModel):
    """Limits and gates the engine applies on top of the budget."""

    model_config = ConfigDict(extra="forbid")

    #: Maximum number of new submissions per tick (fleet-size control).
    max_submits_per_tick: int = 8
    #: Gate every submission behind pre-approval.
    require_approval: bool = False
    #: Gate submissions whose estimated core-hours exceed this threshold.
    approve_over_est_core_hours: float | None = None
    #: Wall-clock seconds a leaf may stay deferred (outcome-unknown) before the
    #: engine stops retrying and marks it failed with an ``outcome_unknown``
    #: code.  A pending idempotency record only resolves when the simclient
    #: finishes the build, so an operator whose builds take longer than the
    #: default should raise this.  See :data:`DEFAULT_DEFERRED_OUTCOME_TIMEOUT_S`.
    deferred_outcome_timeout_s: float = DEFAULT_DEFERRED_OUTCOME_TIMEOUT_S


class FailureGroup(BaseModel):
    """Failed leaves of one tick sharing an identical reason.

    Several leaves failing the same way (e.g. one rejected field) would
    otherwise repeat the same multi-KB validation dump once per leaf.  The group
    names the reason once and lists every affected leaf path.
    """

    model_config = ConfigDict(extra="forbid")

    error_code: str | None = None
    stage: str | None = None
    #: The shared reason, bounded to
    #: :data:`FAILURE_MESSAGE_MAX_CHARS` characters (the full text stays on the
    #: persisted callback drained by ``take_agenda_callbacks``).
    message: str
    #: Every failed leaf path sharing this reason (in first-seen order).
    paths: list[str] = Field(default_factory=list)


class TickResult(BaseModel):
    """The outcome of one engine tick."""

    model_config = ConfigDict(extra="forbid")

    submitted: list[str] = Field(default_factory=list)
    pending_approval: list[str] = Field(default_factory=list)
    waiting: list[str] = Field(default_factory=list)
    #: Leaves whose submission outcome is unknown (a lost ack or a pending
    #: idempotency record).  They stay ``planned`` and are retried next tick
    #: under the same exactly-once ``cmd_id``.  Kept distinct from ``waiting``
    #: (a resource/dependency hold) and from ``failed`` so a lost ack is never
    #: presented as a physics failure.
    deferred: list[str] = Field(default_factory=list)
    done: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    #: Leaves linked to an already-completed identical run instead of being
    #: submitted again (content-addressed reuse).
    reused: list[str] = Field(default_factory=list)
    complete: bool = False
    usage: BudgetUsage = BudgetUsage()
    #: Decision-point callbacks emitted this tick (newly done/failed leaves).
    #: A failed callback's ``error`` is truncated to
    #: :data:`FAILURE_MESSAGE_MAX_CHARS`; the full reason remains on the
    #: persisted callback and in ``agenda_status``.
    callbacks: list[Callback] = Field(default_factory=list)
    #: This tick's failures grouped by identical reason, so a repeated dump is
    #: emitted once with the affected paths rather than once per leaf.
    failure_groups: list[FailureGroup] = Field(default_factory=list)
    #: A one-line digest of this tick's failures (one clause per distinct
    #: reason), so the common "N leaves failed identically" case reads as a
    #: sentence rather than N repeated payloads.  None when nothing failed.
    failure_summary: str | None = None
    #: This tick's "successful-but-empty" walks, keyed by leaf path (F4).  A
    #: ``done`` leaf that is nonetheless suspect is listed here so an agent can
    #: tell "physics ran and succeeded" from "physics ran and was empty".
    suspects: dict[str, str] = Field(default_factory=dict)
    #: The campaign's lifecycle state after this tick, or ``"complete"`` once
    #: every leaf is terminal (so ``state`` never reads ``"running"`` next to
    #: ``complete: true``).  ``lifecycle`` always carries the stored
    #: :data:`~pic_agentic.agenda.campaign.CampaignState`, so a terminal tick on
    #: a paused/stopped campaign is not lossy: ``state == "complete"`` reports
    #: that all work is finished while ``lifecycle`` keeps the pause/stop.
    state: TickState = "running"
    #: The stored lifecycle state after this tick, unchanged by completion, so
    #: ``state == "complete"`` does not hide a paused/stopped campaign.  Equals
    #: ``state`` whenever the campaign is not terminal.
    lifecycle: CampaignState = "running"
    #: Leaves the planner would have submitted but the lifecycle held back.
    held: list[str] = Field(default_factory=list)


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
        actuals: ActualsFn | None = None,
        failures: FailuresFn | None = None,
        suspects: SuspectsFn | None = None,
        reuse_lookup: ReuseLookupFn | None = None,
        reuse_record: ReuseRecordFn | None = None,
        reuse_key: ReuseKeyFn | None = None,
        clock: Callable[[], datetime] | None = None,
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
            actuals: Optional ``() -> {sim_id: ActualUsage}`` callable supplying
                the cluster's actual cost for finished runs (gap 4); when given,
                a finished leaf's usage estimate is corrected to the actual.
            failures: Optional ``() -> {sim_id: FailureInfo}`` callable supplying
                the simclient's reason for a failed run.  When given, a leaf that
                reaches a failed state has the reason stamped onto its callback
                and persisted status.
            suspects: Optional ``() -> {sim_id: SuspectInfo}`` callable supplying
                the "successful-but-empty" health flag for a finished run (F4).
                When given, a leaf that newly reaches ``done`` has the warning
                stamped onto its callback and persisted status, so a zero-physics
                run is no longer indistinguishable from a real success.
            reuse_lookup: Optional ``(key) -> ReuseRecord | None`` that returns
                a completed run whose key matches, so the leaf is linked to it
                instead of being submitted again.  Without it (and without
                ``reuse_record``) no reuse is attempted.
            reuse_record: Optional ``(key, sim_id, state)`` called when a leaf
                finishes successfully, so a later identical spec can reuse it.
            reuse_key: Optional ``(spec) -> key`` content key.  Defaults to the
                wire hash; the server folds in the provenance tuple so a result
                from a different PIConGPU revision is not reused.
            clock: Optional ``() -> datetime`` (UTC) used to time the deferred
                outcome window.  Defaults to the real clock; injectable so a
                test can advance time without sleeping.

        """
        self.store = store
        self.budget_override = budget
        self.submit = submit
        self.observe = observe
        self.policy = policy or EnginePolicy()
        self.approve = approve
        self.actuals = actuals
        self.failures = failures
        self.suspects = suspects
        self.reuse_lookup = reuse_lookup
        self.reuse_record = reuse_record
        self.reuse_key = reuse_key or _wire_hash
        self.clock = clock or (lambda: datetime.now(UTC))

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
        observed = self.observe()
        failures = self._observe_failures()
        suspects = self._observe_suspects()
        campaign, steps, reused = self._plan(campaign, budget, observed)
        # Refuse a campaign whose submissions would collide *before* submitting
        # anything: a duplicate payload maps two leaves to one sim_id, so a
        # partial tick would leave a leaf running against a corrupted
        # observation.  Raising up-front keeps the tick atomic and tells the
        # agent to fix the campaign (tag the replicas distinctly).
        _reject_duplicate_specs(campaign, [path for path, step in steps if step.action == "submit"])
        # Persist the folded observation *together with its callbacks* before
        # submitting anything.  Doing it in one save is what makes the
        # transition durable: if a later submit raises or the process crashes,
        # the callback is already on disk rather than lost (the next tick would
        # see the leaf already terminal and emit nothing).
        campaign, emitted, suspects = self._emit_callbacks(campaign, before, failures, suspects)
        result = TickResult(
            state=campaign.state,
            lifecycle=campaign.state,
            callbacks=list(emitted),
            reused=list(reused),
            suspects=dict(suspects),
        )
        self.store.save(campaign)
        campaign = await self._run_steps(campaign, steps, result, budget)
        result.complete = _is_complete(campaign.agenda)
        # A completed campaign reports ``state: "complete"`` rather than the
        # stored ``"running"`` lifecycle state, so a tick can never contradict
        # itself with ``state: "running"`` next to ``complete: true``.  The
        # stored lifecycle is preserved in ``lifecycle`` so a terminal tick on a
        # paused/stopped campaign is not lossy.
        if result.complete:
            result.state = "complete"
        result.usage = campaign.usage
        self._record_reuse(campaign, observed, before)
        self.store.save(campaign)
        _compact_failures(result)
        return result

    async def _run_steps(
        self,
        campaign: Campaign,
        steps: list[tuple[str, PlanStep]],
        result: TickResult,
        budget: Budget,
    ) -> Campaign:
        """Execute the planner's steps in order, updating ``result`` in place.

        Returns:
            The campaign after all steps (incrementally persisted).

        """
        in_flight = campaign.usage.jobs_running
        concurrency_cap = budget.max_concurrent_jobs
        for path, step in steps:
            if step.action != "submit":
                campaign, terminal = self._persist_terminal(campaign, path, step.action, reason=step.reason)
                if terminal is not None:
                    result.callbacks.append(terminal)
                _bucket(result, path, step.action)
                continue
            blocked = self._submission_block(campaign, path, step, result, in_flight=in_flight, cap=concurrency_cap)
            if blocked is not None:
                {"held": result.held, "waiting": result.waiting, "pending_approval": result.pending_approval}[
                    blocked
                ].append(path)
                continue
            campaign, failure, deferred = await self._submit_one(campaign, path, step)
            if deferred:
                # A lost ack: the leaf stays planned and is retried next tick
                # (under the same idempotency key).  It is bucketed as
                # ``deferred`` -- *not* ``waiting`` (a resource/dependency hold)
                # and *not* ``failed`` -- so the tick never presents a lost ack
                # as a physics failure.  ``failure`` is non-None only when the
                # retry budget was exhausted (a bounded terminal failure).
                result.deferred.append(path)
                if failure is not None:
                    result.failed.append(path)
                    result.callbacks.append(failure)
            elif failure is not None:
                # A single leaf's submission failure must not abort the whole
                # tick (which would wedge every later leaf behind it): mark it
                # failed, record the callback and carry on.
                result.failed.append(path)
                result.callbacks.append(failure)
            else:
                result.submitted.append(path)
                in_flight += 1
            # Incremental persistence: a crash after this point must not
            # resubmit the leaf (the next tick sees its recorded sim_id).
            self.store.save(campaign)
        return campaign

    def _submission_block(
        self,
        campaign: Campaign,
        path: str,
        step: PlanStep,
        result: TickResult,
        *,
        in_flight: int,
        cap: int | None,
    ) -> str | None:
        """Return the bucket a step is held in, or None when it may proceed.

        Encodes the lifecycle, per-tick, concurrency and approval gates in one
        place so the submission loop stays flat.  ``result`` is only read (for
        the per-tick submission count).

        Returns:
            ``"held"``, ``"waiting"`` or ``"pending_approval"`` when blocked,
            else None.

        """
        if campaign.state != "running":
            # Paused/stopped: observe, fold and emit callbacks as usual, but
            # hold every submission (no gate, no usage).  Resuming simply lets
            # the next tick submit the still-planned leaves.
            return "held"
        if len(result.submitted) >= self.policy.max_submits_per_tick:
            return "waiting"
        if cap is not None and in_flight >= cap:
            # The concurrency headroom is enforced here so one tick cannot
            # over-fill (the planner also caps, but the observed in-flight count
            # is authoritative).
            return "waiting"
        if not self._may_submit(leaf_at(campaign.agenda, path), step):
            return "pending_approval"
        return None

    async def _submit_one(
        self,
        campaign: Campaign,
        path: str,
        step: PlanStep,
    ) -> tuple[Campaign, Callback | None, bool]:
        """Submit one leaf, classifying a failure as deferred or terminal.

        A transient failure (a lost ack) leaves the leaf ``planned`` for a retry.
        The deferral is bounded by *elapsed wall-clock time* (see
        :data:`DEFAULT_DEFERRED_OUTCOME_TIMEOUT_S`, overridable via
        :attr:`EnginePolicy.deferred_outcome_timeout_s`): once a leaf has been
        deferred longer than that window with the outcome still unknown it is
        marked ``failed`` with an actionable ``outcome_unknown`` reason rather
        than retried forever.  Bounding ticks would fail a still-building job
        after a few fast ``advance_agenda`` calls, which is the very harm the
        deferral exists to avoid.  Any other submission failure is terminal
        immediately.  Neither raises out of :meth:`tick`, so one bad leaf cannot
        block every leaf after it.

        Returns:
            ``(campaign, callback, deferred)``: on success ``(campaign, None,
            False)``; on a deferred submission ``(campaign, None, True)`` while
            within the timeout, or ``(campaign, callback, True)`` once the
            timeout has elapsed (the callback is the terminal ``outcome_unknown``
            failure); on an immediate terminal failure
            ``(campaign, callback, False)``.

        """
        try:
            return await self._do_submit(campaign, path, step), None, False
        except TransientSubmitError as exc:
            updated, callback = self._record_deferral(campaign, path, exc, now=self.clock())
            if callback is not None:
                return updated, callback, True
            return updated, None, True
        except Exception as exc:  # ruff: ignore[blind-except] - a leaf failure is data, not a fault
            log.warning("agenda submit failed for %s: %s", path, exc)
            agenda = campaign.agenda.model_copy(deep=True)
            leaf = leaf_at(agenda, path)
            if leaf is None:
                return campaign, None, False
            leaf.status = "failed"
            error_code = getattr(exc, "error_code", None)
            stage = getattr(exc, "stage", None)
            leaf.error = str(exc)
            leaf.error_code = error_code
            leaf.stage = stage
            callback = Callback(
                path=path,
                kind="failed",
                sim_id=leaf.sim_id,
                ts=utc_now_iso(),
                error=str(exc),
                error_code=error_code,
                stage=stage,
            )
            updated = campaign.model_copy(update={"agenda": agenda, "callbacks": [*campaign.callbacks, callback]})
            return updated, callback, False

    def _record_deferral(
        self,
        campaign: Campaign,
        path: str,
        exc: TransientSubmitError,
        *,
        now: datetime,
    ) -> tuple[Campaign, Callback | None]:
        """Persist one deferred (outcome-unknown) submission attempt.

        The leaf stays ``planned`` so the next tick retries it under the same
        exactly-once ``cmd_id``; the attempt counter, the wall-clock start of the
        deferral and the reason are updated.  Once the leaf has been deferred for
        longer than :attr:`EnginePolicy.deferred_outcome_timeout_s` with the
        outcome still unknown, it is marked ``failed`` with an
        ``outcome_unknown`` error code and a callback is emitted -- a bounded
        escape hatch so a never-resolving pending record cannot defer forever and
        hold the campaign open.

        Args:
            campaign: The campaign to update.
            path: The leaf path.
            exc: The transient failure that deferred the leaf.
            now: The current UTC time (injected so tests can advance it).

        Returns:
            ``(campaign, callback)``: the callback is non-``None`` only on the
            terminal (timeout-exhausted) attempt.

        """
        agenda = campaign.agenda.model_copy(deep=True)
        leaf = leaf_at(agenda, path)
        if leaf is None:
            return campaign, None
        leaf.deferred_attempts += 1
        # The pending record names the sim even though the ack was lost; keep it
        # so ``stop_agenda``/``cancel_simulation`` can still reach the job
        # instead of reporting an unknown sim for a real (possibly running) one.
        if exc.sim_id and not leaf.sim_id:
            leaf.sim_id = exc.sim_id
        if leaf.deferred_since is None:
            leaf.deferred_since = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        error_code = exc.error_code or OUTCOME_UNKNOWN_ERROR_CODE
        elapsed_s = _elapsed_seconds(leaf.deferred_since, now)
        if elapsed_s < self.policy.deferred_outcome_timeout_s:
            # Within the window: leave the leaf planned and record only why it is
            # deferred.  No callback: this is not a decision point yet.  A few
            # quick ticks therefore never fail a still-building job.
            leaf.error = str(exc)
            leaf.error_code = error_code
            log.warning(
                "agenda submit deferred for %s (attempt %d, %.0fs/%gs): %s",
                path,
                leaf.deferred_attempts,
                elapsed_s,
                self.policy.deferred_outcome_timeout_s,
                exc,
            )
            return campaign.model_copy(update={"agenda": agenda}), None
        # Window exhausted: the job may still exist (the pending record was
        # written before execution), so fail with an actionable code rather than
        # pretending the physics failed.  The leaf keeps its sim_id (if any).
        leaf.status = "failed"
        message = (
            f"submission outcome still unknown after {elapsed_s:.0f}s and "
            f"{leaf.deferred_attempts} deferred attempts ({error_code}); the job may have been "
            f"accepted -- check list_simulations/get_status and cancel_simulation before "
            f"resubmitting: {exc}"
        )
        leaf.error = message
        leaf.error_code = error_code
        callback = Callback(
            path=path,
            kind="failed",
            sim_id=leaf.sim_id,
            ts=utc_now_iso(),
            error=message,
            error_code=error_code,
        )
        log.warning(
            "agenda submit gave up for %s after %.0fs (%d deferrals)",
            path,
            elapsed_s,
            leaf.deferred_attempts,
        )
        updated = campaign.model_copy(update={"agenda": agenda, "callbacks": [*campaign.callbacks, callback]})
        return updated, callback

    def _record_reuse(self, campaign: Campaign, observed: Mapping[str, str], before: Mapping[str, str]) -> None:
        """Offer leaves that newly reached a *reusable* state to the registry.

        Only leaves that (a) transitioned since the pre-tick statuses, (b) are
        observed ``results.ready`` (the state that guarantees the run's
        ``simOutput`` exists -- ``job_finished`` alone may lack results), and
        (c) were not themselves reused are offered.  Recording every done leaf
        each tick would rewrite the registry needlessly; keying on the transition
        keeps it to one write per newly-finished run.  Best-effort: a registry
        write must never break a tick.

        """
        if self.reuse_record is None:
            return
        by_sim = {sim.sim_id: (path, sim) for path, sim in campaign.agenda.simulations() if sim.sim_id}
        for sim_id, state in observed.items():
            if state not in _REUSABLE_OBSERVED_STATES or sim_id not in by_sim:
                continue
            path, sim = by_sim[sim_id]
            if sim.reused or before.get(path) == "done":
                # Already terminal before this tick (or itself reused): no new
                # run to record.
                continue
            try:
                self.reuse_record(self.reuse_key(sim.spec), sim.sim_id or sim_id, "done")
            except Exception:  # ruff: ignore[blind-except] - recording is best-effort
                log.warning("reuse record failed for sim %s", sim.sim_id)

    @staticmethod
    def _emit_callbacks(
        campaign: Campaign,
        before: Mapping[str, str],
        failures: Mapping[str, FailureInfo],
        observed_suspects: Mapping[str, SuspectInfo],
    ) -> tuple[Campaign, list[Callback], dict[str, str]]:
        """Append a callback for every leaf that newly reached done/failed.

        Edge-triggered against the *persisted* pre-tick statuses, so a restart
        between a transition and the agent's poll never re-emits: once the leaf
        is terminal on disk, a later tick sees no transition.  The callbacks are
        accumulated on the campaign (and drained by ``take_agenda_callbacks``),
        so they survive the restart that follows the transition.  A failed
        transition also stamps the observed reason (``error``/``error_code``/
        ``stage``) onto the leaf and its callback, so the campaign can report
        *why* it failed.  A ``done`` transition whose run is "successfully empty"
        stamps the all-zero warning onto the leaf and callback instead (F4), so
        the decision point itself carries the health flag.

        A suspect flag that arrives *after* the done transition (e.g. a later
        status pull promoting a linked run) is still written onto the already
        done leaf - it just does not emit a second callback.

        Returns:
            The campaign, the new callbacks, and ``{path: warning}`` for every
            done leaf currently known to be suspect.

        """
        emitted: list[Callback] = []
        suspects: dict[str, str] = {}
        agenda = campaign.agenda.model_copy(deep=True)
        for path, sim in agenda.simulations():
            warning = observed_suspects.get(sim.sim_id).warning if sim.sim_id in observed_suspects else None
            if sim.status == "done" and warning is not None:
                sim.suspect = warning
                suspects[path] = warning
            if sim.status not in {"done", "failed"} or before.get(path) == sim.status:
                continue
            failure = failures.get(sim.sim_id) if sim.sim_id else None
            callback = Callback(path=path, kind=sim.status, sim_id=sim.sim_id, ts=utc_now_iso())
            if sim.status == "failed" and failure is not None:
                sim.error = failure.error
                sim.error_code = failure.error_code
                sim.stage = failure.stage
                callback = callback.model_copy(
                    update={"error": failure.error, "error_code": failure.error_code, "stage": failure.stage}
                )
            if sim.status == "done" and warning is not None:
                callback = callback.model_copy(update={"suspect": warning})
            emitted.append(callback)
        if not emitted:
            return campaign.model_copy(update={"agenda": agenda}), emitted, suspects
        updated = campaign.model_copy(update={"agenda": agenda, "callbacks": [*campaign.callbacks, *emitted]})
        return updated, emitted, suspects

    def _observe_failures(self) -> Mapping[str, FailureInfo]:
        """Return sim_id -> failure reason for the campaign's known sims.

        Best-effort: a lookup that raises yields no reason rather than breaking
        the tick.  Returns an empty mapping when no ``failures`` callable was
        injected.

        Returns:
            The observed failure reasons keyed by ``sim_id``.

        """
        if self.failures is None:
            return {}
        try:
            return self.failures()
        except Exception as exc:  # ruff: ignore[blind-except] - failure lookup must never break a tick
            log.warning("agenda failures lookup failed: %s", exc)
            return {}

    def _observe_suspects(self) -> Mapping[str, SuspectInfo]:
        """Return sim_id -> health flag for the campaign's known sims (F4).

        Best-effort: a lookup that raises yields no flag rather than breaking the
        tick.  Returns an empty mapping when no ``suspects`` callable was
        injected.

        Returns:
            The observed "successful-but-empty" flags keyed by ``sim_id``.

        """
        if self.suspects is None:
            return {}
        try:
            return self.suspects()
        except Exception as exc:  # ruff: ignore[blind-except] - health lookup must never break a tick
            log.warning("agenda suspects lookup failed: %s", exc)
            return {}

    @staticmethod
    def _persist_terminal(
        campaign: Campaign,
        path: str,
        action: str,
        *,
        reason: str | None = None,
    ) -> tuple[Campaign, Callback | None]:
        """Persist a planner-terminal decision the observation did not cover.

        A leaf blocked by a failed dependency is reported ``failed`` by the
        planner but has no observed state of its own, so its folded status would
        stay ``planned`` and the campaign could never be ``complete``.  Writing
        the terminal status back closes that gap, and a callback is emitted for
        the transition (it is a decision point like any other).  The planner's
        ``reason`` (e.g. which dependency failed) is stamped on the callback so
        the campaign can report *why* the successor was failed.

        Returns:
            The campaign (with the leaf's status updated when terminal) and the
            emitted callback, if any.

        """
        if action not in {"failed", "done"}:
            return campaign, None
        agenda = campaign.agenda.model_copy(deep=True)
        leaf = leaf_at(agenda, path)
        if leaf is None or leaf.status in {"failed", "done"}:
            return campaign, None
        leaf.status = action
        callback = Callback(path=path, kind=action, sim_id=leaf.sim_id, ts=utc_now_iso())
        if action == "failed" and reason:
            leaf.error = reason
            callback = callback.model_copy(update={"error": reason})
        updated = campaign.model_copy(update={"agenda": agenda, "callbacks": [*campaign.callbacks, callback]})
        return updated, callback

    def _plan(
        self,
        campaign: Campaign,
        budget: Budget,
        observed: Mapping[str, str],
    ) -> tuple[Campaign, list[tuple[str, PlanStep]], list[str]]:
        """Fold observed states into the campaign and compute next actions.

        The folded agenda is written back into the returned campaign so that a
        tick persists observed progress (not only new submissions); otherwise
        ``complete``/``done`` could never be reached.  ``jobs_running`` is
        recomputed from the observed in-flight leaves so the concurrency cap
        reflects reality across ticks.

        Returns:
            The campaign with the folded agenda, ``(path, step)`` pairs in
            planner order, and the paths satisfied by content-addressed reuse
            this tick.

        """
        by_sim = {sim.sim_id: path for path, sim in campaign.agenda.simulations() if sim.sim_id}
        path_states = {by_sim[sim_id]: state for sim_id, state in observed.items() if sim_id in by_sim}
        agenda = apply_states(campaign.agenda, path_states)
        # Content-addressed reuse runs *before* admission and the gates: a leaf
        # satisfied by an earlier identical run spends no cluster resources, so
        # it must not be blocked by the budget/concurrency/approval that guard
        # real submissions (and it must not wedge completion when a cap is hit).
        agenda, reused = self._apply_reuse(agenda)
        usage = campaign.usage.model_copy(
            update={"jobs_running": sum(1 for _, sim in agenda.simulations() if sim.status in _RUNNING_STATUSES)},
        )
        agenda, usage = self._reconcile_actuals(agenda, usage)
        steps = next_actions(agenda, {}, budget=budget, usage=usage)
        return campaign.model_copy(update={"agenda": agenda, "usage": usage}), [(s.path, s) for s in steps], reused

    def _apply_reuse(self, agenda: AgendaGroup) -> tuple[AgendaGroup, list[str]]:
        """Mark every planned leaf with a registry hit as ``done`` (reused).

        Pure: returns a new agenda.  A reused leaf records the earlier run's
        ``sim_id`` and the ``reused`` flag, so it neither accrues estimated usage
        nor is reconciled against the cluster's actuals (the cost belonged to the
        run that first executed it).  Doing this before planning means the leaf
        is never a submission step, so it cannot be blocked by admission/gates.

        Returns:
            The agenda with reused leaves folded to ``done``, and their paths.

        """
        if self.reuse_lookup is None:
            return agenda, []
        updated = agenda.model_copy(deep=True)
        reused: list[str] = []
        for path, sim in updated.simulations():
            if sim.status != "planned":
                continue
            record = self.reuse_lookup(self.reuse_key(sim.spec))
            if record is None:
                continue
            sim.sim_id = record.sim_id
            sim.status = "done"
            sim.reused = True
            reused.append(path)
        return updated, reused

    def _reconcile_actuals(self, agenda: AgendaGroup, usage: BudgetUsage) -> tuple[AgendaGroup, BudgetUsage]:
        """Replace finished leaves' estimated usage with the cluster's actuals.

        For every leaf that has finished and whose actual cost is known, the
        estimate accrued at submission is corrected (once -- the actual is
        stamped on the leaf, so a second tick does not re-apply it).  Pure: the
        input agenda/usage are not mutated.

        Returns:
            The agenda with actuals stamped on reconciled leaves, and the
            corrected usage.

        """
        if self.actuals is None:
            return agenda, usage
        try:
            actuals = self.actuals()
        except Exception as exc:  # ruff: ignore[blind-except] - reconciliation must never break a tick
            log.warning("agenda actuals lookup failed: %s", exc)
            return agenda, usage
        updated = agenda.model_copy(deep=True)
        for _, sim in updated.simulations():
            if sim.reused:
                # A reused leaf was never run by this campaign; its cost was
                # accounted where it first executed.  Re-charging it here would
                # retroactively consume this campaign's budget for nothing.
                continue
            if sim.status not in {"done", "failed"} or sim.actual_core_hours is not None or not sim.sim_id:
                continue
            actual = actuals.get(sim.sim_id)
            if actual is None:
                continue
            usage = reconcile(
                usage,
                estimated_core_hours=sim.estimated_core_hours,
                actual_core_hours=actual.core_hours,
                estimated_gpu_hours=sim.estimated_gpu_hours,
                actual_gpu_hours=actual.gpu_hours,
                is_gpu=sim.is_gpu,
            )
            sim.actual_core_hours = actual.core_hours
            sim.actual_gpu_hours = actual.gpu_hours
        return updated, usage

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
        request = resource_request_from_spec(step.spec)
        agenda = campaign.agenda.model_copy(deep=True)
        leaf = leaf_at(agenda, path)
        if leaf is not None:
            leaf.sim_id = sim_id
            leaf.status = "submitted"
            # A clean ack resolves any prior deferral: the retry window resets so
            # a later, unrelated lost ack gets a fresh allowance, and the stale
            # stage/error of the lost-ack attempt is cleared (it described the
            # deferral, not this accepted submission).
            leaf.deferred_attempts = 0
            leaf.deferred_since = None
            leaf.error = None
            leaf.error_code = None
            leaf.stage = None
            # Stamp what we reserved so a later reconciliation can correct it.
            leaf.estimated_core_hours = request.est_core_hours
            leaf.estimated_gpu_hours = request.est_gpu_hours
            leaf.is_gpu = request.is_gpu
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
        observed = self._observe_suspects()
        counts = {"planned": 0, "submitted": 0, "running": 0, "done": 0, "failed": 0}
        leaves: list[dict[str, Any]] = []
        suspects: dict[str, str] = {}
        # Leaves whose last submission outcome was unknown (a lost ack or a
        # pending idempotency record): still ``planned`` but not merely waiting.
        # Kept a separate list so a caller can tell "deferred, outcome unknown"
        # from an ordinary resource/dependency hold and from a failure.
        deferred: list[str] = []
        for path, sim in campaign.agenda.simulations():
            counts[sim.status] = counts.get(sim.status, 0) + 1
            if sim.status == "planned" and sim.deferred_attempts > 0:
                deferred.append(path)
            # Prefer the live probe over the persisted flag, so a leaf already
            # observed done by the registry is suspect even before a tick stamps
            # it; fall back to the durable flag otherwise.
            warning = observed[sim.sim_id].warning if sim.sim_id in observed else sim.suspect
            if sim.status == "done" and warning:
                suspects[path] = warning
            leaves.append(
                {
                    "path": path,
                    "status": sim.status,
                    "sim_id": sim.sim_id,
                    "point": sim.point,
                    "sweep_parameter": sim.sweep_parameter,
                    "requires_approval": sim.requires_approval,
                    "approved": sim.approved,
                    "error": sim.error,
                    "error_code": sim.error_code,
                    "stage": sim.stage,
                    # Whether this leaf's last submission outcome is unknown and
                    # it is being retried (distinct from a terminal failure).
                    "deferred": sim.status == "planned" and sim.deferred_attempts > 0,
                    "deferred_attempts": sim.deferred_attempts,
                    "deferred_since": sim.deferred_since,
                    "suspect": warning if sim.status == "done" else None,
                },
            )
        return {
            "name": campaign.name,
            "state": campaign.state,
            "complete": _is_complete(campaign.agenda),
            "counts": counts,
            "usage": campaign.usage.model_dump(),
            "leaves": leaves,
            "deferred": deferred,
            "suspects": suspects,
            "suspect_count": len(suspects),
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
                "sweep_parameter": sim.sweep_parameter,
                "spec_hash": _spec_hash(sim.spec),
                "error": sim.error,
                "error_code": sim.error_code,
                "stage": sim.stage,
                "deferred": sim.status == "planned" and sim.deferred_attempts > 0,
                "deferred_attempts": sim.deferred_attempts,
                "deferred_since": sim.deferred_since,
                "suspect": sim.suspect if sim.status == "done" else None,
                "estimated_core_hours": sim.estimated_core_hours,
                "actual_core_hours": sim.actual_core_hours,
                "actual_gpu_hours": sim.actual_gpu_hours,
            }
            for path, sim in campaign.agenda.simulations()
        ]
        return {
            "name": campaign.name,
            "created_ts": campaign.created_ts,
            "complete": _is_complete(campaign.agenda),
            "revision": _revision(campaign),
            "conclusion": campaign.conclusion,
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


def _elapsed_seconds(since: str | None, now: datetime) -> float:
    """Return whole seconds between an ISO-8601 ``Z`` timestamp and ``now``.

    A missing/unparseable timestamp means "just started deferring" (0 s), so a
    legacy campaign persisted before ``deferred_since`` existed is retried
    rather than failed on sight.

    Returns:
        The elapsed seconds (never negative).

    """
    if not since:
        return 0.0
    try:
        started = datetime.strptime(since, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return 0.0
    return max(0.0, (now - started).total_seconds())


def _truncate(text: str, limit: int = FAILURE_MESSAGE_MAX_CHARS) -> str:
    """Bound ``text`` to ``limit`` characters, marking a cut tail.

    Returns:
        The text unchanged when short enough, else its head plus
        :data:`FAILURE_TRUNCATION_MARKER`.

    """
    if len(text) <= limit:
        return text
    return text[:limit] + FAILURE_TRUNCATION_MARKER


def _compact_failures(result: TickResult) -> None:
    """Deduplicate and bound the failed callbacks of one tick, in place.

    The per-leaf callbacks stay (so each affected path remains addressable) but
    their ``error`` is truncated, and identical reasons are collapsed into
    :attr:`TickResult.failure_groups` so a repeated validation dump is spelled
    out once.  The full reason is preserved on the persisted campaign callback
    and in ``agenda_status``.

    Args:
        result: The tick result, updated in place.

    """
    order: list[tuple[str | None, str | None, str]] = []
    groups: dict[tuple[str | None, str | None, str], FailureGroup] = {}
    for callback in result.callbacks:
        if callback.kind != "failed":
            continue
        reason = callback.error or "the failure reason was not reported"
        # ``stage`` is part of the key so two leaves failing with the same
        # code/message at different pipeline stages (e.g. build vs run) stay
        # distinct groups, matching the advertised grouping semantics.
        key = (callback.error_code, callback.stage, reason)
        group = groups.get(key)
        if group is None:
            group = FailureGroup(error_code=callback.error_code, stage=callback.stage, message=_truncate(reason))
            groups[key] = group
            order.append(key)
        group.paths.append(callback.path)
        if callback.error is not None:
            callback.error = _truncate(callback.error)
    result.failure_groups = [groups[key] for key in order]
    result.failure_summary = _failure_summary(result.failure_groups)


def _failure_summary(groups: list[FailureGroup]) -> str | None:
    """Compose the one-line failure digest for a tick result.

    Returns:
        ``"N leaf/leaves failed: <code>: <message>; ..."``, or None when there
        are no failure groups.

    """
    if not groups:
        return None
    total = sum(len(group.paths) for group in groups)
    noun = "leaf" if total == 1 else "leaves"
    clauses = ", ".join(
        f"{group.error_code}: {group.message}" if group.error_code else group.message for group in groups
    )
    return f"{total} {noun} failed: {clauses}"


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


def _reject_duplicate_specs(campaign: Campaign, submit_paths: list[str]) -> None:
    """Refuse a tick whose *submitting* leaves would collide on ``sim_id``.

    Checks the pending submissions against each other and against every
    already-submitted leaf, using the wire payload (``spec["sim"]``) -- the same
    bytes the cluster hashes into ``sim_id``.  Done up-front so a collision
    aborts the tick before any job is launched.

    Raises:
        DuplicateSpecError: On the first colliding pair (deterministic order).

    """
    submitted = [(path, sim) for path, sim in campaign.agenda.simulations() if sim.sim_id]
    by_path = dict(campaign.agenda.simulations())
    pending = [(path, by_path[path]) for path in submit_paths if path in by_path]
    for index, (path, sim) in enumerate(pending):
        digest = _wire_hash(sim.spec)
        for other_path, other in [*submitted, *pending[:index]]:
            # A deferred (lost-ack) leaf carries a sim_id *and* is still
            # planned, so it appears in both ``submitted`` and ``pending``;
            # resubmitting it under its own id is a retry, not a collision.
            # This guard is live (not dead): the deferred leaf is the only
            # entry in ``submitted`` for its own path, so without it the
            # retry tick would collide with itself.  ``pending[:index]``
            # excludes the current index but *not* the same path from
            # ``submitted``, which the standalone case exercises.
            if other_path == path:
                continue
            if _wire_hash(other.spec) == digest:
                msg = (
                    f"duplicate payload at {path!r} and {other_path!r}: identical simulations map to the same "
                    "sim_id and would collapse into one job; make replicas distinct (e.g. per-replica tag)"
                )
                raise DuplicateSpecError(msg)


def _reject_duplicate_spec(campaign: Campaign, path: str, step: PlanStep) -> None:
    """Refuse a spec whose wire payload collides with an already-submitted leaf.

    The per-submit guard, kept as a defensive check inside :meth:`_do_submit`;
    the up-front :func:`_reject_duplicate_specs` does the authoritative scan.

    Raises:
        DuplicateSpecError: If ``step``'s wire payload matches an
            already-submitted leaf's.

    """
    digest = _wire_hash(step.spec)
    for other_path, sim in campaign.agenda.simulations():
        if other_path == path or not sim.sim_id:
            continue
        if _wire_hash(sim.spec) == digest:
            msg = (
                f"duplicate payload at {path!r} and {other_path!r}: identical simulations map to the same "
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


def _wire_hash(spec: Mapping[str, Any]) -> str:
    """Return the hash of the *wire payload* a leaf submission produces.

    Mirrors :func:`pic_agentic.protocol.simulation.
    simulation_spec_from_runner_dump`: the payload is ``{"sim": spec["sim"]}``,
    and the cluster's ``sim_id`` is the payload-hash prefix.  Kept local (stdlib
    only) so this module stays extraction-ready, and used so duplicate detection
    matches exactly the bytes the ``sim_id`` uses.

    Returns:
        The sha256 hex digest of the canonical ``{"sim": ...}`` encoding.

    """
    sim = spec.get("sim")
    payload = {"sim": sim} if sim is not None else dict(spec)
    return _spec_hash(payload)


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
    "ActualUsage",
    "ActualsFn",
    "AgendaEngine",
    "DuplicateSpecError",
    "EnginePolicy",
    "FailureGroup",
    "FailureInfo",
    "FailuresFn",
    "ObserveFn",
    "SubmitFailureError",
    "SubmitFn",
    "TickResult",
    "leaf_at",
]
