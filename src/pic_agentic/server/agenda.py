# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Server-side agenda service: advance a persisted campaign by one durable tick.

This is the bridge between the transport-agnostic campaign engine
(:mod:`pic_agentic.agenda.engine`) and the MCP server: it owns the
:class:`~pic_agentic.agenda.store.AgendaStore` holding the campaign, wraps the
shared :class:`~pic_agentic.server.simulation.SubmitService` as the engine's
``submit`` callable (submitting a leaf's already-built Runner spec directly) and
projects the server's sim registry into the engine's ``observe`` mapping.  The
engine itself keeps no authoritative state, so an MCP-server restart resumes
exactly where the last tick stopped.

The module deliberately does not import cluster-side code: the campaign is a
local file and submission is delegated to the injected service.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pic_agentic.agenda.campaign import Campaign, CampaignState
from pic_agentic.agenda.engine import (
    OUTCOME_UNKNOWN_ERROR_CODE,
    ActualUsage,
    AgendaEngine,
    EnginePolicy,
    FailureInfo,
    SubmitFailureError,
    SuspectInfo,
    TransientSubmitError,
    leaf_at,
)
from pic_agentic.agenda.model import AgendaGroup, AgendaSim, readable_label
from pic_agentic.agenda.refine import summary as refine_summary
from pic_agentic.agenda.reuse import DEFAULT_REUSE_FILE, PENDING_STATE, ReuseRecord, ReuseRegistry
from pic_agentic.agenda.store import DEFAULT_CAMPAIGN_FILE, AgendaStore
from pic_agentic.protocol.simulation import (
    MAX_INLINE_PAYLOAD_BYTES,
    SimulationOp,
    UnsupportedPayloadError,
    payload_wire_size,
)
from pic_agentic.server.hello import AckTimeoutError
from pic_agentic.server.simulation import _spec_provenance
from pic_agentic.simclient.simulation import SimulationErrorCode
from pic_agentic.simulation_build import check_spec_consistency, check_spec_round_trip

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pic_agentic.config import Config
    from pic_agentic.server.hello import SendFn
    from pic_agentic.server.simulation import SubmitService

log = logging.getLogger(__name__)


#: Actionable text attached to every ``no_campaign`` soft error, naming the tool
#: that creates a campaign so the caller knows how to recover.
NO_CAMPAIGN_MESSAGE = "no campaign is persisted; create a campaign first, e.g. with create_campaign, then retry."

#: Actionable text attached to the ``campaign_in_flight`` soft error returned when
#: a reset is asked for while leaves may still have a live cluster job: it names
#: the escape hatch (``force``) rather than silently leaving orphaned jobs.
IN_FLIGHT_MESSAGE = (
    "campaign has leaves still submitted/running (or a non-terminal leaf with a "
    "sim_id, e.g. a lost-ack submission); stop or cancel them first, or pass "
    "force=true to delete the campaign anyway."
)

#: Leaf statuses that are not terminal, so a leaf carrying one *and* a ``sim_id``
#: may have a live job: a lost-ack submission stays ``planned`` but already
#: stamped a ``sim_id`` (the engine submits before it flips the status), and
#: deleting then would orphan that job exactly like a ``submitted`` leaf.
_IN_FLIGHT_STATUSES = frozenset({"planned", "submitted", "running"})

#: A dotted patch-path segment that indexes a list rather than a dict key.
_LIST_INDEX_RE = re.compile(r"-?\d+")

#: Control-ack codes that mean "the sim is known but has no live job to kill":
#: the requested cancellation is satisfied by clearing it, not an error.
_NOT_SIGNALABLE_CODE = SimulationErrorCode.NOT_SIGNALABLE.value
_NOT_TERMINAL_CODE = SimulationErrorCode.NOT_TERMINAL.value


def no_campaign_error() -> dict[str, Any]:
    """Return the actionable ``no_campaign`` soft error.

    Every campaign-scoped operation that finds no persisted campaign returns
    this, so the error carries a recovery hint (``message``) alongside the
    stable machine-readable ``error`` code.

    Returns:
        ``{"ok": False, "error": "no_campaign", "message": <actionable text>}``.

    """
    return {"ok": False, "error": "no_campaign", "message": NO_CAMPAIGN_MESSAGE}


def _store_for(config: Config) -> AgendaStore:
    """Build the campaign store for ``config``.

    ``config.agenda_file`` names the campaign file directly (its directory is
    the store root, its file name the store file).  When unset, the default
    :data:`~pic_agentic.agenda.store.DEFAULT_CAMPAIGN_FILE` under the message
    directory is used.

    Args:
        config: The resolved server configuration.

    Returns:
        The campaign store.

    """
    target = config.agenda_file
    if target:
        path = Path(target).expanduser()
        return AgendaStore(path.parent, filename=path.name)
    return AgendaStore(config.message_dir or ".", filename=DEFAULT_CAMPAIGN_FILE)


class AgendaService:
    """Own one campaign store and advance it against the live sim registry.

    Plain class (it holds a store and a submit service, i.e. runtime
    resources), not a pydantic model.
    """

    def __init__(self, config: Config, submit_service: SubmitService, *, policy: EnginePolicy | None = None) -> None:
        """Create the service.

        Args:
            config: The resolved server configuration (store path, redaction).
            submit_service: The shared submit service whose registry is the
                engine's observation source and whose ``submit_spec`` performs
                the leaf submissions.
            policy: Optional engine limits/gates; defaults to the policy derived
                from :class:`~pic_agentic.config.Config` (``agenda_require_
                approval`` and ``agenda_approve_over_est_core_hours``).

        """
        self.config = config
        self.submit_service = submit_service
        self.policy = policy or _policy_from_config(config)
        self.store = _store_for(config)
        #: Content-addressed reuse registry beside the campaign, so a spec that
        #: already completed can be linked instead of re-submitted.
        self.reuse_store = AgendaStore(self.store.root, filename=DEFAULT_REUSE_FILE)
        #: Run ids of *direct* submissions still awaiting their ``results.ready``.
        #: ``on_run_ready`` fires for every completing run (including campaign
        #: runs whose engine writes the ``done`` record directly), so this set
        #: lets :meth:`promote_reuse` skip the whole-registry disk read for a run
        #: that has no pending direct entry.  Hydrated once from the registry so
        #: a post-restart backfilled ``results.ready`` still promotes.
        self._pending_direct_run_ids: set[str] = set()
        self._hydrate_pending_direct_run_ids()
        #: Serialise every campaign read-modify-write (advance and the lifecycle
        #: mutators) on one lock.  Without this, a tick's incremental save can
        #: clobber a concurrent add_leaf/approve/drain, and -- worst -- a stop
        #: issued mid-tick is overwritten by the tick's stale in-memory campaign,
        #: resurrecting the campaign and letting it keep submitting.
        self._lock = asyncio.Lock()

    def _hydrate_pending_direct_run_ids(self) -> None:
        """Seed the in-memory pending-direct set from the persisted registry.

        Called once at construction so a post-restart backfilled
        ``results.ready`` (whose run id is only known from the persisted
        ``pending`` record) is still promoted.  Best-effort.

        """
        try:
            registry = self._load_reuse()
        except Exception as exc:  # ruff: ignore[blind-except] - hydration must never block startup
            log.warning("reuse hydration failed: %s", exc)
            return
        ids: set[str] = set()
        for record in registry.records.values():
            if record.state != PENDING_STATE:
                continue
            if record.run_id:
                ids.add(record.run_id)
            ids.update(record.pending_run_ids)
        self._pending_direct_run_ids = ids

    def _load_reuse(self) -> ReuseRegistry:
        """Load the reuse registry, or an empty one when absent/corrupt.

        A malformed registry is treated as empty rather than fatal: reuse is an
        optimisation, and a corrupt cache must never block a tick.

        Returns:
            The persisted registry, or a fresh empty one.

        """
        if not self.reuse_store.exists():
            return ReuseRegistry()
        try:
            return ReuseRegistry.model_validate_json(self.reuse_store.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("ignoring unreadable reuse registry: %s", exc)
            return ReuseRegistry()

    def remember_direct_spec(self, spec: dict[str, Any], *, sim_id: str, run_id: str) -> None:
        """Record an accepted direct ``submit_simulation`` as a pending reuse.

        A bare ``submit_simulation`` never went through the campaign engine, so
        its completed result was invisible to the registry and a later identical
        campaign leaf re-ran it (H7).  This stores a ``pending`` entry the
        moment the simclient accepts the submission; the result is not ready
        yet, so it is not reusable, and the entry is attributed by ``run_id``
        (the submission's stable command id) so the later ``results.ready``
        event can promote it **even across a server restart** (the signed-room
        replay carries the command id, not the spec).  A direct submission has
        no sweep point, so the record is keyed under ``point=None``: it only
        ever reuses against another point-less run, never against a
        point-carrying campaign leaf (a different simulation under option A).
        Best-effort: registry bookkeeping must never fail a submission.

        Args:
            spec: The submitted wire spec (``{"sim": ...}``).
            sim_id: The spec's payload label.
            run_id: The submission's stable command id (the run identity).

        """
        try:
            self._remember_direct(spec, sim_id=sim_id, run_id=run_id)
        except Exception as exc:  # ruff: ignore[blind-except] - registry bookkeeping is best-effort
            log.warning("reuse remember failed for sim %s: %s", sim_id, exc)

    def _remember_direct(self, spec: dict[str, Any], *, sim_id: str, run_id: str) -> None:
        """Persist one direct submission's reuse record and sync the pending set.

        ``_load_reuse`` is called once here (the submit-path parse the review
        flagged), and the record is merged with :meth:`ReuseRegistry.record_direct`
        so a second identical submission never demotes a completed entry.

        """
        registry = self._load_reuse()
        record = self._direct_reuse_record(registry, spec, sim_id=sim_id, run_id=run_id)
        updated = registry.record_direct(record)
        if updated is not registry:
            self.reuse_store.save(updated)
        merged = updated.records.get(record.wire_hash, record)
        if merged.state == PENDING_STATE:
            self._pending_direct_run_ids.add(run_id)
        else:
            self._pending_direct_run_ids.discard(run_id)

    def _direct_reuse_record(
        self,
        registry: ReuseRegistry,
        spec: dict[str, Any],
        *,
        sim_id: str,
        run_id: str,
    ) -> ReuseRecord:
        """Build the registry entry for an accepted direct submission.

        Args:
            registry: The already-loaded reuse registry (single parse on the
                submit path).
            spec: The submitted wire spec (``{"sim": ...}``).
            sim_id: The spec's payload label.
            run_id: The submission's stable command id (the run identity).

        Returns:
            The entry: ``pending`` normally, or ``done`` when the run's
            ``results.ready`` was already projected (the ack and the event can
            arrive back to back, so the result may beat this bookkeeping).

        """
        # A direct submission has no sweep point, so it is keyed under the
        # point-less key (``point=None``); a point-carrying campaign leaf is a
        # *different* simulation under option A and never shares this record.
        # This is deliberate: ``remember_direct_spec``/``_direct_reuse_record``
        # are the point-less path, and the campaign engine passes the leaf's
        # actual point through :func:`_reuse_key`.
        key = _reuse_key(spec, self.submit_service.picongpu_revision, None)
        existing = registry.records.get(key)
        # A replayed submission (e.g. ingest_backfill after a restart) must not
        # demote an already-promoted record back to pending.
        state = existing.state if existing is not None and existing.state != PENDING_STATE else PENDING_STATE
        run = self.submit_service.registry.get(sim_id)
        if state == PENDING_STATE and run is not None and run.cmd_id == run_id and run.state == "results.ready":
            state = "done"
        return ReuseRecord(wire_hash=key, sim_id=sim_id, state=state, run_id=run_id)

    def promote_reuse(self, run_id: str) -> None:
        """Promote the pending direct-submission record accepted by ``run_id``.

        Called when the simclient reports a run's results are ready: it is the
        one event that guarantees ``run_dir/simOutput`` exists, so it is the
        point at which the run becomes reusable (matching the engine's own
        ``results.ready``-only rule).  The event carries the command id, so the
        record is found by identity even after a restart.  This hook fires for
        *every* completing run -- including campaign runs whose engine already
        wrote the ``done`` record -- so the in-memory pending-direct set short-
        circuits a run that has no pending direct entry, avoiding a whole-file
        registry read (and reload) per completion.  Best-effort.

        Args:
            run_id: The submission's stable command id.

        """
        if run_id not in self._pending_direct_run_ids:
            return
        try:
            registry = self._load_reuse()
            promoted = registry.promote(run_id)
            if promoted is not registry:
                self.reuse_store.save(promoted)
                self._pending_direct_run_ids.discard(run_id)
        except Exception as exc:  # ruff: ignore[blind-except] - registry bookkeeping is best-effort
            log.warning("reuse promote failed for run %s: %s", run_id, exc)

    async def advance(self, send: SendFn) -> dict[str, Any]:
        """Run one durable engine tick over the persisted campaign.

        The engine's ``submit`` wraps :meth:`SubmitService.submit_spec` for the
        leaf's Runner spec; its ``observe`` projects the sim registry (one
        ``state`` per ``sim_id``).  Any failure is returned as a soft
        ``{"ok": False, "error": ...}`` dict (redacted), never raised.

        Args:
            send: Async RCP sender from the running transport.

        Returns:
            The tick result as a dict, or ``{"ok": False, "error": ...}`` when
            there is no campaign or a tick step fails.

        """
        if not self.store.exists():
            return no_campaign_error()
        engine = self._build_engine(send)
        async with self._lock:
            try:
                result = await engine.tick()
            except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
                log.warning("agenda tick failed: %s", exc)
                return {"ok": False, "error": self.config.redact(str(exc))}
        return result.model_dump()

    def _build_engine(self, send: SendFn) -> AgendaEngine:
        """Assemble the engine with its live observation/submission callables.

        Kept out of :meth:`advance` so the tick loop reads as one straight line;
        every callable closes over ``self`` and the running transport's ``send``.

        Returns:
            The engine wired to the shared submit service and reuse registry.

        """
        registry = self.submit_service.registry

        def observe() -> Mapping[str, str]:
            return {sim_id: record.state for sim_id, record in registry.items()}

        def actuals() -> Mapping[str, ActualUsage]:
            # The registry carries the actual cost reported on a run's terminal
            # event (gap 4); only records that have it are offered.
            return {
                sim_id: ActualUsage(core_hours=record.core_hours, gpu_hours=record.gpu_hours or 0.0)
                for sim_id, record in registry.items()
                if record.core_hours is not None
            }

        def failures() -> Mapping[str, FailureInfo]:
            # The registry carries the reason a run failed, projected from the
            # submit ack or a lifecycle event; only records that have one are
            # offered.  Redaction is applied at the tool boundary.
            return {
                sim_id: FailureInfo(error=record.error, error_code=record.error_code, stage=record.stage)
                for sim_id, record in registry.items()
                if record.error or record.error_code
            }

        async def submit(spec: dict[str, Any], key: str) -> str:
            try:
                outcome = await self.submit_service.submit_spec(send, spec, cmd_id=key)
            except AckTimeoutError as exc:
                # The ack was lost: the job may well be running.  Defer to the
                # next tick (the stable cmd_id makes the retry exactly-once).
                msg = f"lost ack: {exc}"
                raise TransientSubmitError(msg, error_code=OUTCOME_UNKNOWN_ERROR_CODE) from exc
            return _classify_submit(outcome)

        def reuse_key(spec: dict[str, Any], point: dict[str, Any] | None) -> str:
            return _reuse_key(spec, self.submit_service.picongpu_revision, point)

        def reuse_lookup(key: str) -> ReuseRecord | None:
            return self._load_reuse().lookup(key)

        def reuse_record(key: str, sim_id: str, state: str, run_id: str | None) -> None:
            updated = self._load_reuse().remember(
                ReuseRecord(wire_hash=key, sim_id=sim_id, state=state, run_id=run_id),
            )
            self.reuse_store.save(updated)

        return AgendaEngine(
            store=self.store,
            submit=submit,
            observe=observe,
            policy=self.policy,
            actuals=actuals,
            failures=failures,
            suspects=self._registry_suspects,
            reuse_lookup=reuse_lookup,
            reuse_record=reuse_record,
            reuse_key=reuse_key,
        )

    def _registry_suspects(self) -> Mapping[str, SuspectInfo]:
        """Project the registry's "successful-but-empty" flags (F4).

        The engine's :meth:`AgendaEngine.status` probe needs no transport, but
        the registry only knows the flag the simclient pushed on the
        ``results.ready`` event.  Passing it here keeps ``agenda_status`` able
        to flag an already-done empty leaf before the next tick stamps it.

        Returns:
            ``{sim_id: SuspectInfo}`` for records carrying the flag.

        """
        return {
            sim_id: SuspectInfo(warning=record.suspect)
            for sim_id, record in self.submit_service.registry.items()
            if record.suspect
        }

    def status(self) -> dict[str, Any]:
        """Return the aggregate campaign status (redacted-safe).

        Returns:
            The engine's status dict, or ``{"ok": False, "error": ...}`` when
            there is no campaign or it cannot be read.

        """
        if not self.store.exists():
            return no_campaign_error()
        engine = AgendaEngine(
            store=self.store,
            submit=_never_submit,
            observe=dict,
            policy=self.policy,
            suspects=self._registry_suspects,
        )
        try:
            return engine.status()
        except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
            log.warning("agenda status failed: %s", exc)
            return {"ok": False, "error": self.config.redact(str(exc))}

    async def approve(self, path: str) -> dict[str, Any]:
        """Mark one campaign leaf as approved so the next tick may submit it.

        The leaf's persisted ``approved`` flag is what the engine's submission
        gate consults, so approval survives a restart.  A missing campaign or
        leaf is a soft error.

        Args:
            path: The dotted agenda path (``group/leaf``) to approve.

        Returns:
            ``{"ok": True, "path": ..., "approved": True}``, or a soft error.

        """
        if not self.store.exists():
            return no_campaign_error()
        async with self._lock:
            try:
                return self._approve_leaf(path)
            except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
                log.warning("agenda approve failed: %s", exc)
                return {"ok": False, "error": self.config.redact(str(exc))}

    def _approve_leaf(self, path: str) -> dict[str, Any]:
        """Load, set the approval flag on one leaf and save (may raise).

        Returns:
            ``{"ok": True, ...}`` on success, or a ``no_such_leaf`` soft error.

        """
        campaign = self.store.load(Campaign)
        agenda = campaign.agenda.model_copy(deep=True)
        leaf = leaf_at(agenda, path)
        if leaf is None:
            return {"ok": False, "error": "no_such_leaf", "path": path}
        leaf.approved = True
        self.store.save(campaign.model_copy(update={"agenda": agenda}))
        return {"ok": True, "path": path, "approved": True}

    async def record_analysis(self, path: str, analysis: dict[str, Any]) -> dict[str, Any]:
        """Record one leaf's analysis on the campaign (research-loop gap 2).

        The recorded sections travel into the campaign RO-Crate, closing the
        inputs -> runs -> analyses -> conclusion lineage.

        Args:
            path: The analysed leaf's path.
            analysis: The ``analyze_output`` sections (or any JSON object).  To be
                ranked by :meth:`suggest_refinement`, put the scalar objective
                under a top-level numeric ``score`` (or ``value``/``peak``) key;
                sections without one are recorded but not ranked.  Retain the
                sections themselves (and/or the chosen key) so the provenance
                stays self-describing.

        Returns:
            ``{"ok": True, "path": path}``, or a soft error.

        """
        return await self._mutate_campaign(lambda campaign: self._record_analysis(campaign, path, analysis))

    def _record_analysis(self, campaign: Campaign, path: str, analysis: dict[str, Any]) -> dict[str, Any]:
        """Set one analysis and return the success dict (may raise).

        Returns:
            ``{"ok": True, "path": path}``.

        """
        if leaf_at(campaign.agenda, path) is None:
            return {"ok": False, "error": "no_such_leaf", "path": path}
        analyses = {**campaign.analyses, path: analysis}
        self.store.save(campaign.model_copy(update={"analyses": analyses}))
        return {"ok": True, "path": path}

    async def record_conclusion(self, conclusion: str) -> dict[str, Any]:
        """Record the campaign's declared conclusion.

        Args:
            conclusion: The agent's conclusion text (must be non-empty).

        Returns:
            ``{"ok": True, "conclusion": conclusion}``, or a soft error.

        """
        return await self._mutate_campaign(lambda campaign: self._record_conclusion(campaign, conclusion))

    def _record_conclusion(self, campaign: Campaign, conclusion: str) -> dict[str, Any]:
        """Set the conclusion and return the success dict (may raise).

        Returns:
            ``{"ok": True, "conclusion": conclusion}``.

        """
        if not conclusion.strip():
            return {"ok": False, "error": "empty_conclusion"}
        self.store.save(campaign.model_copy(update={"conclusion": conclusion}))
        return {"ok": True, "conclusion": conclusion}

    def suggest_refinement(self, *, rel_tol: float = 0.05) -> dict[str, Any]:
        """Suggest refinement points from the recorded analyses (gap 2).

        Scores each leaf by its recorded analysis's numeric score.  The sweep
        point is deliberately *never* used as a score: a value that was merely
        simulated (not analysed) must not be mistaken for an optimum.  When no
        analysis is recorded the summary reports ``best: null``, ``analysed: 0``
        and an actionable message instead of inventing a ranking; when analyses
        are recorded but none carries a recognised score the message says so.

        Args:
            rel_tol: Relative tolerance for the convergence check.

        Returns:
            The refinement summary, or a soft error.

        """
        if not self.store.exists():
            return no_campaign_error()
        try:
            campaign = self.store.load(Campaign)
        except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
            return {"ok": False, "error": self.config.redact(str(exc))}
        result = refine_summary(_analysed_points(campaign), rel_tol=rel_tol)
        response: dict[str, Any] = {"ok": True, **result}
        if result["analysed"] == 0:
            if campaign.analyses:
                response["message"] = (
                    "No analysable score is available yet: analyses are recorded, but none "
                    "exposes a numeric top-level `score` (or `value`/`peak`). Re-record each "
                    "leaf's analysis with the scalar objective under one of those keys "
                    "(`record_agenda_analysis`)."
                )
            else:
                response["message"] = (
                    "No analyses are recorded yet, so there is no best point to report. "
                    "Record analyses with `record_agenda_analysis` first."
                )
        return response

    async def _mutate_campaign(self, mutate: Callable[[Campaign], dict[str, Any]]) -> dict[str, Any]:
        """Run a locked load -> mutate -> save, degrading failures to data.

        Returns:
            The mutation's result dict, or a soft error.

        """
        if not self.store.exists():
            return no_campaign_error()
        async with self._lock:
            try:
                campaign = self.store.load(Campaign)
                return mutate(campaign)
            except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
                log.warning("agenda mutation failed: %s", exc)
                return {"ok": False, "error": self.config.redact(str(exc))}

    async def set_state(self, state: CampaignState) -> dict[str, Any]:
        """Persist the campaign lifecycle state (running/paused/stopped).

        Args:
            state: The new state.

        Returns:
            ``{"ok": True, "state": state}``, or a soft error.

        """
        if not self.store.exists():
            return no_campaign_error()
        async with self._lock:
            try:
                campaign = self.store.load(Campaign)
                # Re-validate the whole model: ``model_copy(update=...)`` does
                # not validate, so an invalid state would otherwise be persisted
                # and only fail on the next load.
                updated = Campaign.model_validate({**campaign.model_dump(), "state": state})
                self.store.save(updated)
            except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
                log.warning("agenda set_state failed: %s", exc)
                return {"ok": False, "error": self.config.redact(str(exc))}
        return {"ok": True, "state": state}

    async def stop(self, send: SendFn) -> dict[str, Any]:
        """Kill-switch: stop the campaign and cancel its in-flight jobs.

        The lock is held for the whole operation, so a stop waits for any
        in-flight tick to finish (whose final save would otherwise overwrite the
        stop) and then blocks every later tick.  The state is set to ``stopped``
        and persisted *before* any cancellation, so a crash mid-cancellation
        still leaves the campaign stopped.

        Every non-terminal leaf carrying a ``sim_id`` is targeted, not only
        ``submitted``/``running`` ones: a lost-ack submission stays ``planned``
        but has already been stamped with a ``sim_id``, so leaving it out would
        orphan a job that may well be running.  A cancellation the simclient
        confirmed (``ok`` and no error) is reported as ``cancelled``; a
        ``not_signalable``/``not_terminal`` answer means the sim is known but has
        no live job, so it is reported as ``cleared`` (resolved, nothing to
        kill); everything else -- a timeout, a rejection, an exception -- is
        collected in ``errors`` with its ``error_code`` and message (never a
        bare ``rejected``) and never raised.

        Args:
            send: Async RCP sender from the running transport.

        Returns:
            ``{"ok": True, "state": "stopped", "cancelled": [...],
            "cleared": [...], "errors": [...]}``, or a soft error when no
            transport/campaign.

        """
        if not self.store.exists():
            return no_campaign_error()
        async with self._lock:
            try:
                campaign = self.store.load(Campaign)
                self.store.save(campaign.model_copy(update={"state": "stopped"}))
            except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
                log.warning("agenda stop failed: %s", exc)
                return {"ok": False, "error": self.config.redact(str(exc))}
            # Any non-terminal leaf with a sim_id may have a live job: include
            # ``planned`` so a deferred/lost-ack leaf is not stranded.
            in_flight = [
                sim.sim_id
                for _, sim in campaign.agenda.simulations()
                if sim.status in _IN_FLIGHT_STATUSES and sim.sim_id
            ]
            cancelled: list[str] = []
            cleared: list[str] = []
            errors: list[dict[str, str]] = []
            for sim_id in in_flight:
                try:
                    ack = await self.submit_service.control(send, sim_id, SimulationOp.CANCEL)
                except Exception as exc:  # ruff: ignore[blind-except] - collect, never raise
                    errors.append(
                        {
                            "sim_id": sim_id,
                            "error": "cancellation request failed",
                            "detail": self.config.redact(str(exc)),
                        },
                    )
                    continue
                if ack.get("ok") and not ack.get("error"):
                    cancelled.append(sim_id)
                    continue
                code = ack.get("error_code")
                if code in {_NOT_SIGNALABLE_CODE, _NOT_TERMINAL_CODE}:
                    # Known sim, but no live job to cancel (never started, or
                    # already finished): the orphan is resolved by recording it,
                    # not an error.  No longer leave it lingering in ``errors``.
                    cleared.append(sim_id)
                    continue
                errors.append(
                    {
                        "sim_id": sim_id,
                        "error": "cancellation rejected",
                        "detail": self.config.redact(str(ack.get("error") or "the simclient reported no reason")),
                        "error_code": self.config.redact(str(code)) if code else "",
                    },
                )
        return {
            "ok": True,
            "state": "stopped",
            "cancelled": cancelled,
            "cleared": cleared,
            "errors": errors,
        }

    async def take_callbacks(self) -> dict[str, Any]:
        """Return and durably clear the pending decision-point callbacks.

        An MCP server cannot call the LLM, so "callbacks" are pollable records:
        the agent drains them here and decides what to do next.  Draining is
        idempotent in effect -- a second call returns an empty list, and the
        clear is persisted atomically so a restart cannot resurrect them.

        Returns:
            ``{"ok": True, "callbacks": [...]}``, or a soft error.

        """
        if not self.store.exists():
            return no_campaign_error()
        async with self._lock:
            try:
                campaign = self.store.load(Campaign)
                drained = list(campaign.callbacks)
                if drained:
                    self.store.save(campaign.model_copy(update={"callbacks": []}))
            except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
                log.warning("agenda take_callbacks failed: %s", exc)
                return {"ok": False, "error": self.config.redact(str(exc))}
        return {"ok": True, "callbacks": [callback.model_dump() for callback in drained]}

    async def add_leaf(
        self,
        name: str,
        spec: dict[str, Any],
        *,
        point: dict[str, float | int | str] | None = None,
        depends_on: list[str] | None = None,
        parameter: str | None = None,
    ) -> dict[str, Any]:
        """Add one leaf to the persisted campaign's root group (agent expansion).

        This is the agent's refinement primitive: after draining a ``done``
        callback it may add a new simulation (e.g. a refined sweep point) that
        the next tick will submit.  A duplicate name is a soft error.

        Args:
            name: The new leaf's entry name.
            spec: The leaf's Runner spec.
            point: Optional sweep point recorded on the leaf.
            depends_on: Optional sibling dependencies.
            parameter: Optional human-readable name for the swept quantity
                (stored as ``sweep_parameter``).  It is dropped when no
                ``point`` is given, since there is then no sweep to label.

        Returns:
            ``{"ok": True, "path": name}``, or a soft error.

        """
        if not self.store.exists():
            return no_campaign_error()
        sweep_parameter = readable_label(parameter) if parameter and point else None
        async with self._lock:
            try:
                return self._add_leaf(
                    name,
                    spec,
                    point=point,
                    depends_on=depends_on,
                    sweep_parameter=sweep_parameter,
                )
            except ValueError as exc:
                # A duplicate/illegal name or an invalid dependency is a
                # model-level ValueError: report it as data, not a tool
                # exception, so a bad mutation is never persisted.
                code = self._add_leaf_error_code(name)
                return {"ok": False, "error": code, "detail": self.config.redact(str(exc)), "path": name}
            except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
                log.warning("agenda add_leaf failed: %s", exc)
                return {"ok": False, "error": self.config.redact(str(exc))}

    def _add_leaf_error_code(self, name: str) -> str:
        """Classify a rejected ``add_leaf`` (duplicate name vs invalid leaf).

        Returns:
            ``duplicate_leaf`` when the name already exists, else
            ``invalid_leaf``.

        """
        try:
            return "duplicate_leaf" if name in self.store.load(Campaign).agenda.entries else "invalid_leaf"
        except Exception:  # ruff: ignore[blind-except] - classification must never mask the error
            return "invalid_leaf"

    async def create_campaign(
        self,
        name: str,
        base_spec: dict[str, Any],
        patch_path: str,
        values: list[Any],
        *,
        parameter: str | None = None,
        point_key: str | None = None,
    ) -> dict[str, Any]:
        """Create and persist a campaign with one leaf per sweep value.

        The server-side equivalent of the driver's ``--agenda-init``: a fresh
        campaign is created with one leaf per entry in ``values``, each holding
        a deep copy of ``base_spec`` with the dotted Runner-spec path
        ``patch_path`` set to that value.  Each leaf records
        ``point={last path segment: value}`` exactly as :meth:`AgendaSim` and
        the driver do, preserving the sweep assignment for provenance, and a
        human-readable ``sweep_parameter`` (the dotted path with list indices
        dropped, e.g. ``sim.laser.focus_pos_si.component``) so the point key is
        not opaque; an explicit ``parameter`` overrides the derived name and an
        explicit ``point_key`` overrides the point key itself.  The
        campaign is written through the same
        :class:`~pic_agentic.agenda.store.AgendaStore` the other agenda tools
        read, so ``advance_agenda`` picks it up on the next tick.

        Creating over an existing campaign would clobber its state, so it is
        refused rather than overwritten.

        Every patched leaf is built and validated through
        :meth:`SubmitService.prepare_spec` and :func:`payload_wire_size` -- the
        *same* allow-list and escaped inline-size checks a submission runs -- so
        a campaign that could never be submitted is refused at creation time
        rather than surfacing a bare error at ``advance_agenda``.  The derived
        invariants a single-node patch can leave *arithmetically* inconsistent
        (a stale ``cell_depth``, a CFL violation, a ``grid_dist``/``cell_cnt``
        mismatch) are checked too and refused with an actionable reason.  A lone
        ``sim.grid.cell_cnt`` (or ``cell_size``) patch is **not** refused: a
        box-size sweep is legitimate.  It is instead recorded as a non-blocking
        ``warnings`` entry on the result, because a fixed-box resolution sweep
        must co-vary cell_size/cell_cnt(/cell_depth) and delta_t_si/time_steps.
        Duplicate
        patched specs (e.g. ``values=[4, 4]``) are rejected up front too: they
        map to one ``sim_id`` and the engine would later refuse the tick.
        Nothing is persisted unless every leaf validates.

        Args:
            name: The campaign's display name.
            base_spec: The base Runner spec (a wire spec carrying ``sim``).
            patch_path: A dotted path into ``base_spec`` (e.g.
                ``sim.time_steps``); the final segment must already exist.
            values: One value per leaf; each patches ``patch_path``.
            parameter: Optional human-readable name for the swept quantity,
                stored on each leaf as ``sweep_parameter``.  When omitted it is
                derived from ``patch_path``.
            point_key: Optional override for the ``point`` key, which otherwise
                defaults to the last ``patch_path`` segment.  Use it when the
                campaign is seeded with an unrelated patch (e.g. a single
                ``sim.time_steps`` value) and the recorded point should name the
                quantity actually being studied.

        Returns:
            ``{"ok": True, "name": name, "leaves": [<paths>]}`` -- with an
            advisory ``warnings`` list for a lone ``cell_cnt``/``cell_size``
            box-size sweep -- or a soft error (``campaign_exists``,
            ``no_values``, ``invalid_campaign``, ``invalid_campaign_spec``,
            ``spec_exceeds_inline_limit``, ``duplicate_campaign_spec``).

        """
        async with self._lock:
            try:
                return self._create_campaign(
                    name, base_spec, patch_path, values, parameter=parameter, point_key=point_key
                )
            except (TypeError, ValueError, IndexError) as exc:
                # A bad patch path (including an out-of-range list index) or an
                # invalid leaf name/value is a model-level error: report it as
                # data, not a tool exception.
                return {"ok": False, "error": "invalid_campaign", "detail": self.config.redact(str(exc))}
            except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
                log.warning("agenda create_campaign failed: %s", exc)
                return {"ok": False, "error": self.config.redact(str(exc))}

    def _create_campaign(
        self,
        name: str,
        base_spec: dict[str, Any],
        patch_path: str,
        values: list[Any],
        *,
        parameter: str | None = None,
        point_key: str | None = None,
    ) -> dict[str, Any]:
        """Build the campaign and save it (may raise).

        Returns:
            ``{"ok": True, "name": name, "leaves": [<paths>]}`` (plus a
            non-blocking ``warnings`` list for a lone ``cell_cnt``/``cell_size``
            box-size sweep), or a soft error (``campaign_exists``,
            ``no_values``, ``invalid_campaign``, ``invalid_campaign_spec``,
            ``spec_exceeds_inline_limit``, ``duplicate_campaign_spec``).  No file
            is written on any error.

        """
        if self.store.exists():
            return {"ok": False, "error": "campaign_exists"}
        if not values:
            return {"ok": False, "error": "no_values"}
        # An explicit ``point_key`` (sanitised) overrides the last-patch-segment
        # derivation, so a seed-only campaign is not mislabelled by an unrelated
        # patch field (A2); the readable ``sweep_parameter`` stays separate.
        point_key = (readable_label(point_key) if point_key else None) or _parameter_for(patch_path)
        derived = sweep_parameter_for(patch_path, base_spec)
        sweep_parameter = readable_label(parameter) if parameter else readable_label(derived)
        if not _leaf_target_exists(base_spec, patch_path):
            return {
                "ok": False,
                "error": "invalid_campaign",
                "detail": (
                    f"patch path {patch_path!r} does not address an existing field; "
                    "refusing to create it (check the Runner-spec field name)"
                ),
            }
        patched: list[dict[str, Any]] = [_patch_spec(base_spec, patch_path, value) for value in values]
        for spec in patched:
            invalid = self._validate_leaf_spec(spec, patch_path)
            if invalid is not None:
                return invalid
        collision = _duplicate_leaf_error(patched)
        if collision is not None:
            return collision

        agenda = AgendaGroup(name="campaign")
        leaves: list[str] = []
        for index, spec in enumerate(patched):
            leaf_name = f"leaf{index:03d}"
            leaf = AgendaSim(
                name=leaf_name,
                spec=spec,
                point=_point_for(point_key, values[index]),
                sweep_parameter=sweep_parameter,
            )
            agenda = agenda.add(**{leaf_name: leaf})
            leaves.append(leaf_name)
        self.store.save(Campaign(name=name, agenda=agenda).with_created_ts())
        result: dict[str, Any] = {"ok": True, "name": name, "leaves": leaves}
        # Advisory only (never a refusal): flag a lone cell_cnt/cell_size patch
        # as a box-size/resolution sweep so an agent that meant to hold the box
        # fixed is not misled by the beta-6 trap of silent box variation.
        advisory = _box_size_sweep_warning(patch_path)
        if advisory is not None:
            result["warnings"] = [advisory]
        return result

    async def delete_campaign(self, *, force: bool = False) -> dict[str, Any]:
        """Remove the persisted campaign and its sibling reuse registry.

        This is the reset primitive: ``create_campaign`` refuses to overwrite an
        existing campaign, so an agent that wants a fresh study must be able to
        clear the old one through a tool rather than deleting files by hand.

        Deleting while a leaf is still ``submitted``/``running`` would leave the
        cluster job orphaned (no campaign state to observe or cancel it), so it
        is refused by default with an actionable ``campaign_in_flight`` error;
        ``force=True`` overrides the guard and deletes anyway.  The guard holds
        the lock together with the delete, so a concurrent ``advance_agenda``
        cannot submit between the check and the removal.

        A campaign file that cannot be parsed is removed without the guard: the
        reset primitive is the recovery path for broken state, so a corrupt
        ``campaign.json`` must be clearable through a tool rather than trapping
        the agent (``create_campaign`` refuses while the file exists).  The
        in-flight guard can only run when the file parses; an unreadable file
        has no observable live leaves.

        Args:
            force: Delete even when leaves may still have a live cluster job.

        Returns:
            ``{"ok": True, "deleted": [...paths...]}`` on success, the
            actionable ``no_campaign`` soft error when there is nothing to
            delete, or ``campaign_in_flight`` when the guard trips.

        """
        if not self.store.exists():
            return no_campaign_error()
        async with self._lock:
            try:
                campaign = self.store.load(Campaign)
            except Exception as exc:  # ruff: ignore[blind-except] - unparseable state is removable
                # Corrupt/old campaign: no live leaves can be observed, so drop
                # the file(s) rather than blocking the only reset primitive.
                log.warning("agenda delete_campaign: removing unparseable campaign: %s", exc)
                campaign = None
            try:
                if campaign is not None:
                    in_flight = [
                        path
                        for path, sim in campaign.agenda.simulations()
                        if sim.status in _IN_FLIGHT_STATUSES and sim.sim_id
                    ]
                    if in_flight and not force:
                        return {
                            "ok": False,
                            "error": "campaign_in_flight",
                            "message": IN_FLIGHT_MESSAGE,
                            "in_flight": in_flight,
                        }
                deleted = self._remove_campaign_files()
            except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
                log.warning("agenda delete_campaign failed: %s", exc)
                return {"ok": False, "error": self.config.redact(str(exc))}
        return {"ok": True, "deleted": deleted}

    def _remove_campaign_files(self) -> list[str]:
        """Delete the campaign file and its sibling reuse registry (may raise).

        Returns:
            The paths removed, campaign first then reuse registry (each only
            when it existed).

        """
        removed: list[str] = []
        for store in (self.store, self.reuse_store):
            if store.exists():
                store.path.unlink()
                removed.append(str(store.path))
        return removed

    def _validate_leaf_spec(self, spec: dict[str, Any], patch_path: str = "") -> dict[str, Any] | None:
        """Validate one patched leaf through the submission path.

        Runs the same allow-list check, the pinned-schema round-trip gate the
        simclient applies (via :func:`~pic_agentic.simulation_build.
        check_spec_round_trip`), the derived-invariant check scoped to
        ``patch_path`` (via :func:`~pic_agentic.simulation_build.
        check_spec_consistency`) and the escaped inline-size cap a ``submit_spec``
        would, so a leaf that could never be submitted is rejected at creation
        with an actionable reason.  A whole-spec ``add_agenda_leaf`` passes no
        ``patch_path`` and is not consistency-checked (it owns every node).

        This is synchronous and runs under the agenda lock.  With the pin
        importable the round-trip is ~7 ms per leaf, so a 200-leaf campaign
        blocks the event loop for ~1.5 s; acceptable for now, but if campaigns
        grow this belongs on a worker thread (the same seam as the build).

        Returns:
            ``None`` when the leaf is a valid, in-cap wire spec, else the soft
            error dict to return.

        """
        try:
            payload = self.submit_service.prepare_spec(spec)
        except UnsupportedPayloadError as exc:
            return {"ok": False, "error": "invalid_campaign_spec", "detail": self.config.redact(str(exc))}
        round_trip = check_spec_round_trip(spec)
        if round_trip is not None:
            return {"ok": False, "error": "invalid_campaign_spec", "detail": self.config.redact(round_trip)}
        inconsistent = check_spec_consistency(spec, patch_path)
        if inconsistent is not None:
            return {"ok": False, "error": "invalid_campaign_spec", "detail": self.config.redact(inconsistent)}
        size = payload_wire_size(payload)
        if size > MAX_INLINE_PAYLOAD_BYTES:
            return {
                "ok": False,
                "error": "spec_exceeds_inline_limit",
                "wire_bytes": size,
                "inline_limit_bytes": MAX_INLINE_PAYLOAD_BYTES,
            }
        return None

    def _add_leaf(
        self,
        name: str,
        spec: dict[str, Any],
        *,
        point: dict[str, float | int | str] | None,
        depends_on: list[str] | None,
        sweep_parameter: str | None = None,
    ) -> dict[str, Any]:
        """Load, add the leaf and save (may raise).

        The leaf spec is validated through the *same* submission path
        ``create_campaign`` and ``submit_spec`` use
        (:meth:`_validate_leaf_spec`: allow-list, pinned-schema round-trip and
        the escaped inline-size cap), so a leaf that could never be submitted is
        refused here rather than surfacing a bare error at ``advance_agenda``.
        This matters most for the by-reference form: staging relaxes only the
        *input* file cap (4 MiB), never the 48 KiB wire cap every submission
        still meets.

        The name collision is checked *first*: a duplicate name is the caller's
        most direct error and must be reported as ``duplicate_leaf`` even when
        the supplied spec is also invalid, preserving the long-standing contract
        that an existing entry is never silently re-validated or replaced.

        The new leaf is built as an :class:`AgendaSim` and validated *before* it
        is inserted, so its ``depends_on`` goes through the model validator
        (a duplicate or path-style dependency is rejected rather than persisted
        into an unloadable campaign).

        Returns:
            ``{"ok": True, "path": name}``, the ``duplicate_leaf`` soft error, or
            the ``_validate_leaf_spec`` soft error when the spec could never be
            submitted.

        """
        campaign = self.store.load(Campaign)
        if name in campaign.agenda.entries:
            return {"ok": False, "error": "duplicate_leaf", "path": name}
        invalid = self._validate_leaf_spec(spec)
        if invalid is not None:
            return invalid
        leaf = AgendaSim(
            name=name,
            spec=spec,
            point=point,
            sweep_parameter=sweep_parameter,
            depends_on=list(depends_on or []),
        )
        agenda = campaign.agenda.add(**{name: leaf})
        self.store.save(campaign.model_copy(update={"agenda": agenda}))
        return {"ok": True, "path": name}


#: Dotted patch-path prefixes whose single-node patch changes one half of the
#: box-size/resolution pair while holding the other fixed.  A ``cell_cnt``-only
#: patch keeps the base ``cell_size`` (and the CFL-consistent ``delta_t_si``) and
#: so varies the physical box at constant resolution; a ``cell_size``-only patch
#: holds the cell count and so varies the resolution at constant box.  Both are
#: **allowed** -- a box-size sweep is a legitimate study -- but the beta-6 harm
#: was *silent* box variation when a fixed-box resolution sweep was intended, so
#: these are flagged with an advisory (never a refusal).
_BOX_OR_RESOLUTION_PATCH_PREFIXES = ("sim.grid.cell_cnt", "sim.grid.cell_size")


def _box_size_sweep_warning(patch_path: str) -> str | None:
    """Return an advisory when a lone grid patch changes the box/resolution pair.

    ``create_campaign`` patches exactly one node, so a ``cell_cnt``-only patch
    cannot also move ``cell_size`` (and its derived ``cell_depth``): the physical
    box changes while the cell size -- the resolution -- stays fixed.  This is a
    legitimate **box-size sweep** and is accepted; the same is true of a
    ``cell_size``-only patch, which changes the resolution at a constant cell
    count.  The advisory exists only because the beta-6 trap was that an agent
    intending a *fixed-box resolution sweep* got the box varied silently; it is
    never a refusal and never blocks submission.

    Returns:
        The advisory text for a box-size/resolution single-node patch, else
        ``None``.

    """
    if not patch_path.startswith(_BOX_OR_RESOLUTION_PATCH_PREFIXES):
        return None
    if patch_path.startswith("sim.grid.cell_cnt"):
        effect = "changes the physical box size at constant cell_size (resolution)"
    else:
        effect = "changes the cell_size (resolution) at constant cell count"
    return (
        f"patch_path {patch_path!r} {effect} -- a legitimate box-size sweep; accepted. "
        "If you intended a fixed-box resolution sweep, this does not hold the box fixed: co-vary "
        "`sim.grid.cell_size`, `sim.grid.cell_cnt` (and 3D `sim.grid.cell_depth`) plus a "
        "CFL-consistent `sim.delta_t_si`/`sim.time_steps` (e.g. via add_agenda_leaf with whole specs)."
    )


def _parameter_for(patch_path: str) -> str:
    """Return the sweep parameter key encoded in a dotted patch path.

    Mirrors the driver: the leaf's ``point`` key is the last segment of the
    dotted Runner-spec path (``sim.time_steps`` -> ``time_steps``).  Kept as the
    point key for backward compatibility; the human-readable name lives
    alongside it in ``AgendaSim.sweep_parameter`` (see
    :func:`sweep_parameter_for`).

    Returns:
        The final path segment.

    """
    return patch_path.rsplit(".", 1)[-1]


def sweep_parameter_for(patch_path: str, spec: dict[str, Any]) -> str:
    """Derive a human-readable name for a sweep's dotted patch path.

    The leaf's ``point`` key is only the last dotted segment, which is
    meaningless on its own: the focal scan
    ``sim.laser.0.focus_pos_si.1.component`` would otherwise only record
    ``point={"component": ...}``.  This drops list indices while keeping every
    field name, yielding ``sim.laser.focus_pos_si.component``.  A numeric
    segment that indexes a *dict key* (``sim.bc.0`` for ``{"0": ...}``) is
    kept; the base ``spec`` tells the two apart since a list node is only
    indexed numerically.

    Args:
        patch_path: The dotted Runner-spec path.
        spec: The base spec, used to tell a list index from a numeric dict key.

    Returns:
        The readable parameter name; never empty for a non-empty path.

    """
    kept: list[str] = []
    node: Any = spec
    for part in patch_path.split("."):
        if isinstance(node, list) and _LIST_INDEX_RE.fullmatch(part):
            index = int(part)
            node = node[index] if -len(node) <= index < len(node) else None
            continue
        kept.append(part)
        node = node.get(part) if isinstance(node, dict) else None
    return ".".join(kept) or _parameter_for(patch_path)


def _patch_spec(spec: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    r"""Return a deep copy of ``spec`` with the dotted JSON path set to ``value``.

    Behaviourally equivalent to the driver's ``_patch_spec``
    (``scripts/local_mcp_check.py``): the two agree on every valid path.  They
    deliberately diverge only in *error convention* -- the server raises
    ``TypeError``/``IndexError`` (surfaced as the ``invalid_campaign`` soft
    error) while the CLI raises ``SystemExit`` -- so they are not byte-identical
    and need not be.

    The rule for a segment is decided by the *current node*, not by the
    segment's spelling: a dict node is indexed by its key (so a numeric-looking
    dict key such as a boundary-condition map ``{"0": "periodic"}`` is
    reachable via ``sim.bc.0``), and a list node is indexed by the integer the
    segment spells (negative indices count from the end).  This is the
    least-surprising rule and fixes the ``sim.bc.0`` ambiguity; list-shaped
    Runner specs (``sim.laser.0.focus_pos_si.1.component``) still work because
    the nodes there really are lists.

    Args:
        spec: The base Runner spec (a deep copy is patched, the base is not
            mutated).
        dotted: A dotted path such as ``sim.time_steps``.
        value: The JSON value to set.

    Returns:
        The patched deep copy.

    Raises:
        TypeError: If a dict segment is missing, or when a list node is
            addressed by a non-numeric segment.

    A list index out of range raises ``IndexError`` implicitly; both it and the
    ``TypeError`` surface as the ``invalid_campaign`` soft error.

    """
    patched = json.loads(json.dumps(spec))
    node: Any = patched
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = _patch_child(node, dotted, part)
    last = parts[-1]
    if isinstance(node, list):
        if not _LIST_INDEX_RE.fullmatch(last):
            msg = f"patch path {dotted!r} has no numeric index at {last!r}"
            raise TypeError(msg)
        node[int(last)] = value
    elif isinstance(node, dict):
        node[last] = value
    else:
        msg = f"patch path {dotted!r} has no object at {last!r}"
        raise TypeError(msg)
    return patched


def _patch_child(node: Any, dotted: str, part: str) -> Any:
    """Return the child of ``node`` addressed by one intermediate path segment.

    A dict node is indexed by ``part`` as a key; a list node is indexed by the
    integer ``part`` spells (Python indexing, so ``-1`` is the last element).
    Mirrors the driver's ``_patch_child`` behaviourally for valid paths.

    Args:
        node: The current dict or list node.
        dotted: The whole dotted path, used in the error message.
        part: The segment to descend through.

    Returns:
        The addressed child node (dict or list).

    Raises:
        TypeError: If a dict segment is missing, a list node is addressed by a
            non-numeric segment, or an intermediate node is neither.

    """
    if isinstance(node, dict):
        child = node.get(part)
        if not isinstance(child, (dict, list)):
            msg = f"patch path {dotted!r} has no object at {part!r}"
            raise TypeError(msg)
        return child
    if isinstance(node, list):
        if not _LIST_INDEX_RE.fullmatch(part):
            msg = f"patch path {dotted!r} has no numeric index at {part!r}"
            raise TypeError(msg)
        return node[int(part)]
    msg = f"patch path {dotted!r} has no object at {part!r}"
    raise TypeError(msg)


def _leaf_target_exists(spec: dict[str, Any], dotted: str) -> bool:
    """Whether the final segment of ``dotted`` already addresses a real field.

    ``create_campaign`` uses this to reject a typo (``sim.time_step`` for
    ``sim.time_steps``) instead of silently adding an unknown key that the
    Runner ignores.  The walk follows the same dict-key/list-index rule as
    :func:`_patch_spec`; a dict key must be present (any value, including
    ``None``) and a list index must be in range.

    Returns:
        True when the final segment addresses an existing dict key or an
        in-range list index, else False.

    """
    node: Any = spec
    parts = dotted.split(".")
    for part in parts[:-1]:
        if isinstance(node, dict):
            child = node.get(part)
        elif isinstance(node, list) and _LIST_INDEX_RE.fullmatch(part):
            index = int(part)
            child = node[index] if -len(node) <= index < len(node) else None
        else:
            return False
        if not isinstance(child, (dict, list)):
            return False
        node = child
    last = parts[-1]
    if isinstance(node, dict):
        return last in node
    if isinstance(node, list):
        if not _LIST_INDEX_RE.fullmatch(last):
            return False
        index = int(last)
        return -len(node) <= index < len(node)
    return False


def _duplicate_leaf_error(specs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return a soft error when two patched leaves share a wire payload.

    Identical specs map to one ``sim_id`` and the engine later refuses the whole
    tick (``duplicate payload ...``), so the collision is caught here, at
    creation.  The comparison uses the ``{"sim": ...}`` wire form, the same
    bytes :func:`~pic_agentic.agenda.engine._wire_hash` hashes.

    Returns:
        ``{"ok": False, "error": "duplicate_campaign_spec", ...}`` on the first
        colliding pair, else None.

    """
    seen: dict[str, int] = {}
    for index, spec in enumerate(specs):
        sim = spec.get("sim") if isinstance(spec, dict) else None
        key = json.dumps({"sim": sim}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if key in seen:
            return {
                "ok": False,
                "error": "duplicate_campaign_spec",
                "detail": (
                    f"leaves leaf{seen[key]:03d} and leaf{index:03d} have identical specs; "
                    "identical simulations map to the same sim_id and would collapse into one job; "
                    "make the values distinct"
                ),
            }
        seen[key] = index
    return None


def _point_for(parameter: str, value: Any) -> dict[str, float | int | str] | None:
    """Return the leaf ``point`` for one sweep value, or None when unusable.

    ``AgendaSim.point`` accepts only ``float | int | str`` (and rejects bools via
    pydantic).  A value that is not one of those (a list/dict/null, or a bool)
    cannot be a valid point, so it is omitted rather than aborting the whole
    creation.  Identical to the driver's ``_point_for``.

    Returns:
        ``{parameter: value}`` when the value is a valid point, else None.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    return {parameter: value}


def _analysed_points(campaign: Campaign) -> dict[str, float | None]:
    """Extract ``leaf path -> analysed score`` from a campaign's analyses.

    A leaf's score is the numeric score field of its *recorded analysis* only.
    The sweep ``point`` is deliberately not consulted: it is an input, not a
    measured outcome, and ranking on it points users at a wrong optimum.  A
    leaf with no recorded analysis (or one without a usable score) maps to
    ``None``, which the refine helpers ignore and do not count.

    Returns:
        The ``path -> score`` sample map (``None`` for unanalysed leaves).

    """
    points: dict[str, float | None] = {}
    for path, _sim in campaign.agenda.simulations():
        points[path] = _leaf_score(campaign.analyses.get(path))
    return points


#: Analysis keys holding a scalar objective that refinement can rank.  The
#: writer-side contract (``record_agenda_analysis`` and ``record_analysis``):
#: one scalar objective must sit under one of these top-level keys.
_SCORE_KEYS = ("score", "value", "peak")


def _leaf_score(analysis: dict[str, Any] | None) -> float | None:
    """Return a single numeric score for one recorded analysis, or None.

    Only the scalar objective keys in :data:`_SCORE_KEYS` count.  A canonical
    ``analyze_output`` payload (e.g. ``focal_position_m`` / ``total_electrons`` /
    ``high_energy_tail``) carries none of them, so it is recorded but not ranked.

    Returns:
        The analysis ``score`` (or ``value``/``peak``), else None.

    """
    if not isinstance(analysis, dict):
        return None
    for key in _SCORE_KEYS:
        candidate = analysis.get(key)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return float(candidate)
    return None


def _policy_from_config(config: Config) -> EnginePolicy:
    """Derive the engine policy from the server configuration.

    Returns:
        The policy, honouring the approval gates when configured.

    """
    return EnginePolicy(
        require_approval=config.agenda_require_approval,
        approve_over_est_core_hours=config.agenda_approve_over_est_core_hours,
        deferred_outcome_timeout_s=config.agenda_deferred_outcome_timeout_s,
    )


def _classify_submit(outcome: Any) -> str:
    """Turn one submit ack into a sim_id, a deferral, or a terminal failure.

    Kept out of the ``submit`` closure so the outcome classification reads in
    one place and the closure stays branch-light.

    Returns:
        The accepted ``sim_id``.

    Raises:
        TransientSubmitError: When the outcome is unknown (a lost ack or a
            pending idempotency record): the job may exist, so the retry under
            the same ``cmd_id`` is safe and the leaf must stay planned.
        SubmitFailureError: When the simclient genuinely rejected the spec (a
            retry would only repeat the rejection).

    """
    if outcome.outcome_unknown:
        # The simclient found a pending idempotency record for this exactly-once
        # cmd_id: it accepted the command but died before recording an outcome,
        # so the job may exist or be running.  A *terminal* failure here would
        # strand real work and present a lost ack as a physics failure; defer
        # and retry instead, under the same cmd_id, so the record's eventual
        # completion re-acks.
        msg = outcome.error or "submission outcome unknown"
        # Force the stable code rather than trusting the ack's ``error_code``: a
        # pre-fix client still returns ``error_code: rejected_by_policy`` with
        # the sentinel string (that is exactly the C2 confusion).  During a
        # rolling upgrade the server must not persist that misleading code on
        # the deferred leaf or its eventual give-up callback.
        raise TransientSubmitError(
            msg,
            error_code=OUTCOME_UNKNOWN_ERROR_CODE,
            sim_id=outcome.sim_id or None,
        )
    if not outcome.ok or not outcome.sim_id:
        # A genuine rejection (bad payload, policy, version drift): a retry
        # would only repeat it, so this stays terminal.
        msg = outcome.error or "submit failed"
        raise SubmitFailureError(msg, error_code=outcome.error_code, stage=outcome.stage)
    return outcome.sim_id


def _reuse_key(spec: dict[str, Any], fallback_revision: str, point: Mapping[str, Any] | None) -> str:
    """Return the content key under which a spec's completed result is reused.

    The key folds the **sweep point** in with the provenanced wire payload, so
    reuse requires identical content, provenance *and* point: two byte-identical
    specs at different sweep points (e.g. ``point={"x": 1.0}`` and
    ``point={"x": 2.0}``) are *different simulations* under option A and never
    share a record.  Without this, two identical specs at different points
    collapse onto one run and a leaf's provenance can attribute its result to a
    point that never ran.  Finer-grained reuse (e.g. point-insensitive reuse of
    the same physics) is deliberately out of scope here and left to future work.

    The provenance tuple is still folded in, so identical physics authored for a
    different PIConGPU revision/schema is not reused (the result would not be
    attributable to the campaign's revision).  Shared by the campaign engine's
    lookup/record hooks and by the direct ``submit_simulation`` record path.
    A direct submission has no sweep point and passes ``point=None``: it only
    reuses against another point-less run, never against a point-carrying
    campaign leaf.

    Args:
        spec: A wire spec (``{"sim": ...}``, optionally carrying ``provenance``).
        fallback_revision: The server's configured ``picongpu_revision``.
        point: The leaf's sweep point (``{parameter: value}``), or None for a
            point-less direct submission.

    Returns:
        The sha256 hex digest of the canonical ``{sim, provenance, point}``
        payload.

    """
    provenance = _spec_provenance(spec, fallback_revision)
    payload = {"sim": spec.get("sim"), "provenance": provenance, "point": _canonical_point(point)}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _canonical_point(point: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a JSON-normalised copy of ``point`` for the reuse payload, or None.

    The point may arrive from pydantic (``dict[str, float | int | str]``) or a
    plain mapping; a shallow dict copy is enough to serialise it canonically
    (``json.dumps(..., sort_keys=True)`` orders the keys), and None stays None so
    a point-less direct run keys distinctly from every point-carrying leaf.

    Returns:
        A plain ``dict`` copy of the point, or None when no point is given.

    """
    return None if point is None else dict(point)


async def _never_submit(_spec: dict[str, Any], _key: str) -> str:  # ruff: ignore[unused-async] - matches SubmitFn
    """Refuse any submission; the status path never submits.

    Raises:
        RuntimeError: Always (the status engine must not submit).

    """
    msg = "agenda status does not submit"
    raise RuntimeError(msg)


__all__ = ["NO_CAMPAIGN_MESSAGE", "AgendaService", "no_campaign_error"]
