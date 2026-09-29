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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pic_agentic.agenda.campaign import Campaign, CampaignState
from pic_agentic.agenda.engine import ActualUsage, AgendaEngine, EnginePolicy, TransientSubmitError, leaf_at
from pic_agentic.agenda.model import AgendaSim
from pic_agentic.agenda.refine import summary as refine_summary
from pic_agentic.agenda.reuse import DEFAULT_REUSE_FILE, ReuseRecord, ReuseRegistry
from pic_agentic.agenda.store import DEFAULT_CAMPAIGN_FILE, AgendaStore
from pic_agentic.protocol.simulation import SimulationOp
from pic_agentic.server.hello import AckTimeoutError
from pic_agentic.server.simulation import _spec_provenance

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pic_agentic.config import Config
    from pic_agentic.server.hello import SendFn
    from pic_agentic.server.simulation import SubmitService

log = logging.getLogger(__name__)


#: Actionable text attached to every ``no_campaign`` soft error, naming the tool
#: that creates a campaign so the caller knows how to recover.
NO_CAMPAIGN_MESSAGE = "no campaign is persisted; create a campaign first, e.g. with create_campaign, then retry."


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
        #: Serialise every campaign read-modify-write (advance and the lifecycle
        #: mutators) on one lock.  Without this, a tick's incremental save can
        #: clobber a concurrent add_leaf/approve/drain, and -- worst -- a stop
        #: issued mid-tick is overwritten by the tick's stale in-memory campaign,
        #: resurrecting the campaign and letting it keep submitting.
        self._lock = asyncio.Lock()

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

        async def submit(spec: dict[str, Any], key: str) -> str:
            try:
                outcome = await self.submit_service.submit_spec(send, spec, cmd_id=key)
            except AckTimeoutError as exc:
                # The ack was lost: the job may well be running.  Defer to the
                # next tick (the stable cmd_id makes the retry exactly-once).
                msg = f"lost ack: {exc}"
                raise TransientSubmitError(msg) from exc
            if not outcome.ok or not outcome.sim_id:
                msg = outcome.error or "submit failed"
                raise RuntimeError(msg)
            return outcome.sim_id

        def reuse_key(spec: dict[str, Any]) -> str:
            # Fold the provenance tuple into the reuse key, so identical physics
            # authored for a different PIConGPU revision/schema is *not* reused
            # (the result would not be attributable to the campaign's revision).
            provenance = _spec_provenance(spec, self.submit_service.picongpu_revision)
            payload = {"sim": spec.get("sim"), "provenance": provenance}
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            return hashlib.sha256(canonical.encode("ascii")).hexdigest()

        def reuse_lookup(key: str) -> ReuseRecord | None:
            return self._load_reuse().lookup(key)

        def reuse_record(key: str, sim_id: str, state: str) -> None:
            updated = self._load_reuse().remember(ReuseRecord(wire_hash=key, sim_id=sim_id, state=state))
            self.reuse_store.save(updated)

        return AgendaEngine(
            store=self.store,
            submit=submit,
            observe=observe,
            policy=self.policy,
            actuals=actuals,
            reuse_lookup=reuse_lookup,
            reuse_record=reuse_record,
            reuse_key=reuse_key,
        )

    def status(self) -> dict[str, Any]:
        """Return the aggregate campaign status (redacted-safe).

        Returns:
            The engine's status dict, or ``{"ok": False, "error": ...}`` when
            there is no campaign or it cannot be read.

        """
        if not self.store.exists():
            return no_campaign_error()
        engine = AgendaEngine(store=self.store, submit=_never_submit, observe=dict, policy=self.policy)
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
            analysis: The ``analyze_output`` sections (or any JSON object).

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

        Scores each leaf by its recorded analysis's numeric ``score`` (falling
        back to the sweep point's single numeric value), then applies the
        deterministic :mod:`pic_agentic.agenda.refine` helpers.

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
        points = _analysis_points(campaign)
        return {"ok": True, **refine_summary(points, rel_tol=rel_tol)}

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
        still leaves the campaign stopped.  Only cancellations the simclient
        confirmed (``ok`` and no error) are reported as ``cancelled``;
        everything else -- a timeout, a rejection, an exception -- is collected
        in ``errors`` and never raised.

        Args:
            send: Async RCP sender from the running transport.

        Returns:
            ``{"ok": True, "state": "stopped", "cancelled": [...],
            "errors": [...]}``, or a soft error when no transport/campaign.

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
            in_flight = [
                sim.sim_id
                for _, sim in campaign.agenda.simulations()
                if sim.status in {"submitted", "running"} and sim.sim_id
            ]
            cancelled: list[str] = []
            errors: list[dict[str, str]] = []
            for sim_id in in_flight:
                try:
                    ack = await self.submit_service.control(send, sim_id, SimulationOp.CANCEL)
                except Exception as exc:  # ruff: ignore[blind-except] - collect, never raise
                    errors.append({"sim_id": sim_id, "error": self.config.redact(str(exc))})
                    continue
                if ack.get("ok") and not ack.get("error"):
                    cancelled.append(sim_id)
                else:
                    errors.append({"sim_id": sim_id, "error": self.config.redact(str(ack.get("error", "rejected")))})
        return {"ok": True, "state": "stopped", "cancelled": cancelled, "errors": errors}

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

        Returns:
            ``{"ok": True, "path": name}``, or a soft error.

        """
        if not self.store.exists():
            return no_campaign_error()
        async with self._lock:
            try:
                return self._add_leaf(name, spec, point=point, depends_on=depends_on)
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

    def _add_leaf(
        self,
        name: str,
        spec: dict[str, Any],
        *,
        point: dict[str, float | int | str] | None,
        depends_on: list[str] | None,
    ) -> dict[str, Any]:
        """Load, add the leaf and save (may raise).

        The new leaf is built as an :class:`AgendaSim` and validated *before* it
        is inserted, so its ``depends_on`` goes through the model validator
        (a duplicate or path-style dependency is rejected rather than persisted
        into an unloadable campaign).

        Returns:
            ``{"ok": True, "path": name}``.

        """
        campaign = self.store.load(Campaign)
        leaf = AgendaSim(name=name, spec=spec, point=point, depends_on=list(depends_on or []))
        agenda = campaign.agenda.add(**{name: leaf})
        self.store.save(campaign.model_copy(update={"agenda": agenda}))
        return {"ok": True, "path": name}


def _analysis_points(campaign: Campaign) -> dict[str, float | None]:
    """Extract ``leaf path -> score`` from a campaign's recorded analyses.

    A leaf's score is the numeric ``score`` field of its recorded analysis, or
    its sweep point's single numeric value.  A leaf with neither (or with no
    recorded analysis/point) maps to ``None``, which the refine helpers ignore.

    Returns:
        The ``path -> value`` sample map.

    """
    points: dict[str, float | None] = {}
    for path, sim in campaign.agenda.simulations():
        points[path] = _leaf_score(campaign.analyses.get(path), sim)
    return points


def _leaf_score(analysis: dict[str, Any] | None, sim: AgendaSim) -> float | None:
    """Return a single numeric score for one leaf, or None.

    Returns:
        The analysis ``score`` (or ``value``/``peak``), else the sweep point's
        single numeric value, else None.

    """
    for key in ("score", "value", "peak"):
        if isinstance(analysis, dict):
            candidate = analysis.get(key)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                return float(candidate)
    if isinstance(sim.point, dict) and len(sim.point) == 1:
        value = next(iter(sim.point.values()))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _policy_from_config(config: Config) -> EnginePolicy:
    """Derive the engine policy from the server configuration.

    Returns:
        The policy, honouring the approval gates when configured.

    """
    return EnginePolicy(
        require_approval=config.agenda_require_approval,
        approve_over_est_core_hours=config.agenda_approve_over_est_core_hours,
    )


async def _never_submit(_spec: dict[str, Any], _key: str) -> str:  # ruff: ignore[unused-async] - matches SubmitFn
    """Refuse any submission; the status path never submits.

    Raises:
        RuntimeError: Always (the status engine must not submit).

    """
    msg = "agenda status does not submit"
    raise RuntimeError(msg)


__all__ = ["NO_CAMPAIGN_MESSAGE", "AgendaService", "no_campaign_error"]
