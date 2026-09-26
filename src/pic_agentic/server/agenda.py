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
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.engine import AgendaEngine, EnginePolicy, leaf_at
from pic_agentic.agenda.store import DEFAULT_CAMPAIGN_FILE, AgendaStore

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pic_agentic.config import Config
    from pic_agentic.server.hello import SendFn
    from pic_agentic.server.simulation import SubmitService

log = logging.getLogger(__name__)


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
        #: Serialise ticks: ``advance`` may be called concurrently (two tool
        #: calls, or a tool call racing a reconnect); without this, two ticks
        #: could load the same campaign and both submit its planned leaves.
        self._lock = asyncio.Lock()

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
            return {"ok": False, "error": "no_campaign"}

        registry = self.submit_service.registry

        def observe() -> Mapping[str, str]:
            return {sim_id: record.state for sim_id, record in registry.items()}

        async def submit(spec: dict[str, Any], key: str) -> str:
            outcome = await self.submit_service.submit_spec(send, spec, cmd_id=key)
            if not outcome.ok or not outcome.sim_id:
                msg = outcome.error or "submit failed"
                raise RuntimeError(msg)
            return outcome.sim_id

        engine = AgendaEngine(store=self.store, submit=submit, observe=observe, policy=self.policy)
        async with self._lock:
            try:
                result = await engine.tick()
            except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
                log.warning("agenda tick failed: %s", exc)
                return {"ok": False, "error": self.config.redact(str(exc))}
        return result.model_dump()

    def status(self) -> dict[str, Any]:
        """Return the aggregate campaign status (redacted-safe).

        Returns:
            The engine's status dict, or ``{"ok": False, "error": ...}`` when
            there is no campaign or it cannot be read.

        """
        if not self.store.exists():
            return {"ok": False, "error": "no_campaign"}
        engine = AgendaEngine(store=self.store, submit=_never_submit, observe=dict, policy=self.policy)
        try:
            return engine.status()
        except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
            log.warning("agenda status failed: %s", exc)
            return {"ok": False, "error": self.config.redact(str(exc))}

    def approve(self, path: str) -> dict[str, Any]:
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
            return {"ok": False, "error": "no_campaign"}
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


__all__ = ["AgendaService"]
