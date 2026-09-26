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

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from pic_agentic.agenda.engine import AgendaEngine, EnginePolicy
from pic_agentic.agenda.store import DEFAULT_CAMPAIGN_FILE, AgendaStore

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pic_agentic.config import Config
    from pic_agentic.server.hello import SendFn
    from pic_agentic.server.simulation import SubmitService

log = logging.getLogger(__name__)


def _never_submit(_spec: dict[str, Any]) -> NoReturn:
    """Refuse any submission; the status path never submits.

    Raises:
        RuntimeError: Always (the status engine must not submit).

    """
    msg = "agenda status does not submit"
    raise RuntimeError(msg)


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
            policy: Optional engine limits/gates; defaults to
                :class:`~pic_agentic.agenda.engine.EnginePolicy`.

        """
        self.config = config
        self.submit_service = submit_service
        self.policy = policy or EnginePolicy()
        self.store = _store_for(config)

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

        async def submit(spec: dict[str, Any]) -> str:
            outcome = await self.submit_service.submit_spec(send, spec)
            if not outcome.ok or not outcome.sim_id:
                msg = outcome.error or "submit failed"
                raise RuntimeError(msg)
            return outcome.sim_id

        engine = AgendaEngine(store=self.store, submit=submit, observe=observe, policy=self.policy)
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


__all__ = ["AgendaService"]
