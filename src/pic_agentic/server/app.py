# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""MCP stdio server exposing the M1 ``hello`` tool (design sections 4, 8.1)."""

from __future__ import annotations

import asyncio
import base64
import logging
import tempfile
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from pic_agentic.agenda.campaign import Campaign, CampaignState
from pic_agentic.agenda.cwl_exec import manifest_for
from pic_agentic.agenda.provenance import campaign_rocrate
from pic_agentic.auth import MasTokenStore
from pic_agentic.fleet import fleet_view
from pic_agentic.human import (
    HumanAction,
    HumanCommand,
    format_fleet,
    format_leaves,
    format_png_caption,
    format_status,
    help_text,
    notification_text,
    parse_command,
)
from pic_agentic.protocol.simulation import (
    LOG_STREAMS,
    PayloadTooLargeError,
    ResultOp,
    ResultParams,
    SimulationOp,
    SubmitParams,
    UnsupportedPayloadError,
    simulation_phase,
)
from pic_agentic.server.agenda import AgendaService, no_campaign_error
from pic_agentic.server.hello import AckTimeoutError, HelloOutcome, HelloService
from pic_agentic.server.simulation import (
    BuiltSpec,
    SimRecord,
    SubmitOutcome,
    SubmitService,
    condense_events,
    resolve_script,
)
from pic_agentic.server.spec_files import read_spec_file, write_spec_file
from pic_agentic.simclient.safety import UnsafePathError
from pic_agentic.simulation_build import SimulationBuildError
from pic_agentic.transport.matrix import MatrixTransport

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pic_agentic.config import Config
    from pic_agentic.rcp.envelope import RcpMessage

log = logging.getLogger(__name__)

#: The canonical RCP timestamp format.
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

#: Errors the ``submit_simulation`` tool turns into a soft ``{"ok": false}``
#: result rather than letting them escape as an unhandled tool exception.
#: Includes the protocol ``ValueError``s raised before/while sending
#: (``PayloadTooLargeError``, ``UnsupportedPayloadError``) and pydantic's
#: ``ValidationError`` for bad params; no ``ack`` reaches the LLM otherwise.
_SUBMIT_TOOL_ERRORS: tuple[type[BaseException], ...] = (
    AckTimeoutError,
    SimulationBuildError,
    UnsafePathError,
    PayloadTooLargeError,
    UnsupportedPayloadError,
    OSError,
    ValueError,
)

#: Read-tier annotations shared by the M2b reporting tools.
_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)

#: Write-tier annotations for the M3 control verbs: they consume no new
#: resources and are not destructive, but each call sends a fresh signal (not
#: idempotent in the MCP sense).
_CONTROL_ANNOTATIONS = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)

#: Cap on the number of log lines a single ``get_logs`` call may return.
_MAX_LOG_TAIL = 10_000

#: Recursion guard for :func:`_redact_dict`.
_REDACT_MAX_DEPTH = 8


class HelloRuntime:
    """Owns the Matrix transport and the async RCP service."""

    def __init__(self, config: Config, sim: str) -> None:
        """Create the runtime (the transport starts in :meth:`start`).

        Args:
            config: Resolved configuration.
            sim: Simulation id the server operates under.

        """
        self.config = config
        self.sim = sim
        message_dir = Path(config.message_dir)
        self.service = HelloService(
            sim=sim,
            secret=config.rcp_secret,
            message_dir=message_dir,
            ack_timeout_s=config.ack_timeout_s,
        )
        self.submit_service = SubmitService(
            sim=sim,
            secret=config.rcp_secret,
            picongpu_python=config.picongpu_python,
            picongpu_revision=config.picongpu_revision,
            ack_timeout_s=config.ack_timeout_s,
            results_root=config.results_root,
        )
        self.agenda_service = AgendaService(config, self.submit_service)
        self._transport: MatrixTransport | None = None
        self._pump: asyncio.Task | None = None

    async def start(self) -> None:
        """Open the Matrix transport and start pumping inbound messages."""
        config = self.config
        config.require("homeserver", "user_id", "access_token", "room_id", "rcp_secret", "message_dir")
        token_provider = None
        if config.has_refresh_chain():
            store = MasTokenStore.from_config(config)
            token_provider = store.access_token
        self._transport = MatrixTransport(
            config.homeserver,
            config.user_id,
            config.access_token,
            config.room_id,
            store_path=config.nio_store_dir or None,
            token_provider=token_provider,
        )
        # Drain anything already in the room before we start awaiting.  The
        # hello service only needs acks matching a live exchange; the submit
        # service rebuilds its registry from the signed-room replay.
        backfilled = await self._transport.backfill()
        for message in backfilled:
            self.service.on_message(message)
        # A replayed ``hello_ack`` re-establishes the capability handshake across
        # a server restart.
        self.submit_service.set_client_capabilities(self.service.capabilities)
        self.submit_service.ingest_backfill(backfilled)
        # With a human room configured, one pump loop serves both the signed RCP
        # room and the human chat (they share the sync position); otherwise the
        # plain RCP pump is enough.
        pump = self._pump_with_human if self.config.human_room_id else self._pump_forever
        self._pump = asyncio.create_task(pump())

    def _route(self, message: RcpMessage) -> None:
        # Both services filter by envelope kind/type/sim/signature, so feeding
        # every message to both is safe and keeps the routing trivial.
        self.service.on_message(message)
        # Propagate the capability handshake learned from a ``hello`` ack so the
        # submit service can warn about a version drift before sending an op the
        # older client cannot handle.
        self.submit_service.set_client_capabilities(self.service.capabilities)
        self.submit_service.on_message(message)

    async def _pump_forever(self) -> None:
        if self._transport is None:
            msg = "runtime is not started"
            raise RuntimeError(msg)
        async for message in self._transport.receive():
            self._route(message)

    async def _pump_with_human(self) -> None:
        """Pump RCP messages and answer human chat in one sync loop.

        Raises:
            RuntimeError: If the runtime has not been started.

        """
        if self._transport is None:
            msg = "runtime is not started"
            raise RuntimeError(msg)
        while True:
            rcp, human = await self._transport.drain()
            for message in rcp:
                self._route(message)
            for room_id, sender, body in human:
                await self._handle_human(room_id, sender, body)

    async def _handle_human(self, room_id: str, sender: str, body: str) -> None:
        """Answer one human room message (best-effort; never raises).

        Only messages in the configured human room from a sender that is not
        the bot itself are acted on.

        Args:
            room_id: The room the message arrived in.
            sender: The sender's Matrix id.
            body: The message body.

        """
        if self._transport is None:
            return
        if not self.config.human_room_id or room_id != self.config.human_room_id:
            return
        if sender == self.config.user_id:
            return  # never answer ourselves
        try:
            action = await self.dispatch_human(body)
            await self._send_human_action(room_id, action)
        except Exception as exc:  # ruff: ignore[blind-except] - a bot must never crash the pump
            log.warning("human command failed: %s", exc)

    async def _send_human_action(self, room_id: str, action: HumanAction) -> None:
        """Send a :class:`HumanAction` (text or PNG) to ``room_id``.

        Args:
            room_id: The target room.
            action: The action produced by :meth:`dispatch_human`.

        """
        if self._transport is None:
            return
        if action.kind == "image" and action.png_base64:
            await self._transport.send_image(room_id, base64.b64decode(action.png_base64), body=action.text)
        else:
            await self._transport.send_text(room_id, action.text)

    async def dispatch_human(self, body: str) -> HumanAction:
        """Map one human message to the reply the agent should send.

        Args:
            body: The raw message body.

        Returns:
            The text or image action; never raises (an unexpected failure is
            returned as help text).

        """
        command = parse_command(body)
        try:
            return await self._dispatch_command(command)
        except Exception as exc:  # ruff: ignore[blind-except] - a reply must never raise
            return HumanAction(text=f"Could not handle {command.verb}: {self.config.redact(str(exc))}")

    async def _dispatch_command(self, command: HumanCommand) -> HumanAction:
        """Handle a parsed command (may raise; wrapped by :meth:`dispatch_human`).

        Returns:
            The reply action.

        """
        simple = {
            "help": lambda: HumanAction(text=help_text()),
            "status": lambda: HumanAction(text=format_status(self.agenda_status(), self.fleet_status())),
            "fleet": lambda: HumanAction(text=format_fleet(self.fleet_status())),
            "leaves": lambda: HumanAction(text=format_leaves(self.agenda_status())),
        }
        if command.verb in simple:
            return simple[command.verb]()
        if command.verb == "pause":
            return await self._lifecycle_action("paused")
        if command.verb == "resume":
            return await self._lifecycle_action("running")
        if command.verb == "stop":
            result = await self.stop_agenda()
            return HumanAction(
                text=f"Stopped: {len(result.get('cancelled', []))} cancelled, {len(result.get('errors', []))} errors.",
            )
        if command.verb == "png":
            return await self._human_png(command)
        return HumanAction(text=help_text())

    async def _lifecycle_action(self, state: CampaignState) -> HumanAction:
        """Set the campaign state and phrase the reply.

        Returns:
            The reply action.

        """
        result = await self.set_agenda_state(state)
        return HumanAction(text=f"Campaign {result.get('state', result.get('error'))}.")

    async def _human_png(self, command: HumanCommand) -> HumanAction:
        """Render ``!png <sim_id> [record] [component]`` as an image reply.

        Returns:
            The image action, or a text error when unavailable.

        """
        sim_id = command.arg
        if not sim_id:
            return HumanAction(text="Usage: !png <sim_id> [record] [component]")
        if self._transport is None:
            return HumanAction(text="Transport is not started.")
        record = command.extras[0] if command.extras else None
        component = command.extras[1] if len(command.extras) > 1 else None
        params = ResultParams(sim_id=sim_id, op=ResultOp.IMAGE, record=record, component=component)
        payload = await self.submit_service.fetch_result(self._transport.send, params)
        if not payload or payload.get("error"):
            error = self.config.redact(str(payload.get("error", "unavailable")))
            return HumanAction(text=f"No image for {sim_id}: {error}")
        data = payload.get("data")
        if not isinstance(data, str):
            return HumanAction(text=f"No image data for {sim_id}.")
        return HumanAction(
            kind="image",
            text=format_png_caption(sim_id, record, component),
            png_base64=data,
        )

    async def notify_human(self, callbacks: list[dict[str, Any]], alerts: list[dict[str, Any]]) -> None:
        """Push a one-line notification to the human room (best-effort).

        Args:
            callbacks: Callbacks emitted by the last tick.
            alerts: Fleet alerts at that point.

        """
        if self._transport is None or not self.config.human_room_id or not self.config.notify:
            return
        text = notification_text(callbacks, alerts)
        if text is None:
            return
        try:
            await self._transport.send_text(self.config.human_room_id, text)
        except Exception as exc:  # ruff: ignore[blind-except] - notifications are best-effort
            log.warning("human notification failed: %s", exc)

    async def hello(self, message: str) -> HelloOutcome:
        """Run one ``hello`` exchange.

        Args:
            message: The LLM-supplied message text.

        Returns:
            The outcome of the exchange.

        Raises:
            RuntimeError: If the runtime has not been started.

        """
        if self._transport is None:
            msg = "runtime is not started"
            raise RuntimeError(msg)
        return await self.service.hello(self._transport.send, message)

    async def submit(
        self,
        picmi_script: str,
        *,
        params: SubmitParams | None = None,
    ) -> SubmitOutcome:
        """Run one ``submit_simulation`` exchange.

        Args:
            picmi_script: A path to a PICMI script or inline PICMI code.
            params: Optional build/run flags.

        Returns:
            The outcome; ``state`` is the simclient's first ack state.

        Raises:
            RuntimeError: If the runtime has not been started.

        """
        if self._transport is None:
            msg = "runtime is not started"
            raise RuntimeError(msg)
        script_path = resolve_script(picmi_script, workdir=Path(tempfile.gettempdir()) / "pic-agentic")
        return await self.submit_service.submit(self._transport.send, script_path, params=params)

    async def build_spec(self, picmi_script: str, *, write_to: str | None = None) -> tuple[BuiltSpec, str | None]:
        """Build one PICMI script into a Runner spec without submitting it.

        Args:
            picmi_script: A path to a PICMI script or inline PICMI code.
            write_to: Optional staging path; when set the built spec is written
                there as JSON so a later tool can consume it by reference.

        Returns:
            The built wire spec plus the absolute staged path (or ``None``).

        """
        script_path = resolve_script(picmi_script, workdir=Path(tempfile.gettempdir()) / "pic-agentic")
        built = await self.submit_service.build_spec(script_path)
        staged: str | None = None
        if write_to is not None:
            staged = str(write_spec_file(self.config, write_to, built.spec))
        return built, staged

    def registry(self) -> dict[str, SimRecord]:
        """Return the submit service's sim_id-keyed registry.

        Returns:
            The registry mapping (mutated in place by the service).

        """
        return self.submit_service.registry

    def get_sim(self, sim_id: str) -> SimRecord | None:
        """Return the registry record for ``sim_id``, if known.

        Returns:
            The record, or None.

        """
        return self.submit_service.get(sim_id)

    def list_sims(self, *, active_only: bool = False) -> list[SimRecord]:
        """Return the registry records.

        Returns:
            The selected records.

        """
        return self.submit_service.list(active_only=active_only)

    async def fetch_status(self, sim_id: str) -> dict[str, Any]:
        """Run a live-status pull, if the transport is started.

        Returns:
            The ``status_ack`` payload, or an empty dict when not started.

        """
        if self._transport is None:
            return {}
        return await self.submit_service.fetch_status(self._transport.send, sim_id)

    async def fetch_logs(self, sim_id: str, *, stream: str = "stdout", tail: int = 100) -> dict[str, Any]:
        """Run a log pull, if the transport is started.

        Returns:
            The ``logs_ack`` payload, or an empty dict when not started.

        """
        if self._transport is None:
            return {}
        return await self.submit_service.fetch_logs(self._transport.send, sim_id, stream=stream, tail=tail)

    async def control(self, sim_id: str, op: SimulationOp) -> dict[str, Any]:
        """Run a control pull, if the transport is started.

        Returns:
            The ``control_ack`` payload, or an empty dict when not started.

        """
        if self._transport is None:
            return {}
        return await self.submit_service.control(self._transport.send, sim_id, op)

    async def fetch_result(self, params: ResultParams) -> dict[str, Any]:
        """Run a results pull, if the transport is started.

        Returns:
            The ``result_ack`` payload, or an empty dict when not started.

        """
        if self._transport is None:
            return {}
        return await self.submit_service.fetch_result(self._transport.send, params)

    async def advance_agenda(self) -> dict[str, Any]:
        """Advance the persisted campaign by one engine tick, if started.

        Returns:
            The tick result, or ``{"ok": False, "error": "unavailable"}`` when
            the transport has not been started.

        """
        if self._transport is None:
            return {"ok": False, "error": "unavailable"}
        result = await self.agenda_service.advance(self._transport.send)
        # Best-effort human notification when a tick produced decision points; a
        # notification failure must never fail the tick.
        await self.notify_human(result.get("callbacks", []), self.fleet_status().get("alerts", []))
        return result

    def agenda_status(self) -> dict[str, Any]:
        """Return the aggregate campaign status.

        Returns:
            The status dict, or ``{"ok": False, "error": ...}``.

        """
        return self.agenda_service.status()

    async def approve_agenda_leaf(self, path: str) -> dict[str, Any]:
        """Approve one gated campaign leaf so a later tick may submit it.

        Returns:
            ``{"ok": True, "path": ..., "approved": True}``, or a soft error.

        """
        return await self.agenda_service.approve(path)

    async def take_agenda_callbacks(self) -> dict[str, Any]:
        """Drain the pending decision-point callbacks.

        Returns:
            ``{"ok": True, "callbacks": [...]}``, or a soft error.

        """
        return await self.agenda_service.take_callbacks()

    async def add_agenda_leaf(
        self,
        name: str,
        spec: dict[str, Any],
        *,
        point: dict[str, float | int | str] | None = None,
        depends_on: list[str] | None = None,
        parameter: str | None = None,
    ) -> dict[str, Any]:
        """Add one leaf to the campaign's root group (agent expansion).

        Returns:
            ``{"ok": True, "path": name}``, or a soft error.

        """
        return await self.agenda_service.add_leaf(
            name,
            spec,
            point=point,
            depends_on=depends_on,
            parameter=parameter,
        )

    async def create_campaign(
        self,
        name: str,
        base_spec: dict[str, Any] | None,
        patch_path: str,
        values: list[Any],
        *,
        base_spec_path: str | None = None,
        parameter: str | None = None,
    ) -> dict[str, Any]:
        """Create and persist a campaign with one leaf per sweep value.

        The base spec comes either inline (``base_spec``) or from a staged JSON
        file (``base_spec_path``); exactly one must be given.

        Returns:
            ``{"ok": True, "name": name, "leaves": [...]}``, or a soft error.

        """
        resolved, error = self._resolve_base_spec(base_spec, base_spec_path)
        if error is not None:
            return error
        return await self.agenda_service.create_campaign(name, resolved, patch_path, values, parameter=parameter)

    def _resolve_base_spec(
        self,
        base_spec: dict[str, Any] | None,
        base_spec_path: str | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Pick and load the base spec from the inline or path form.

        Exactly one of the two forms is required.  A path is read only inside
        the staging root (:mod:`~pic_agentic.server.spec_files`), so an
        LLM-supplied path can never reach outside it.

        Returns:
            ``(spec, None)`` on success, else ``(None, soft_error)``.

        """
        if (base_spec is None) == (base_spec_path is None):
            return None, {
                "ok": False,
                "error": "base_spec_required",
                "detail": "provide exactly one of base_spec (inline) or base_spec_path (staged file)",
            }
        if base_spec_path is not None:
            try:
                return read_spec_file(self.config, base_spec_path), None
            except UnsafePathError as exc:
                return None, {"ok": False, "error": "invalid_spec_path", "detail": self.config.redact(str(exc))}
        return base_spec, None

    async def delete_campaign(self, *, force: bool = False) -> dict[str, Any]:
        """Remove the persisted campaign and its reuse registry (reset).

        Returns:
            ``{"ok": True, "deleted": [...]}``, or a soft error
            (``no_campaign``, ``campaign_in_flight``).

        """
        return await self.agenda_service.delete_campaign(force=force)

    async def record_analysis(self, path: str, analysis: dict[str, Any]) -> dict[str, Any]:
        """Record one leaf's analysis on the campaign.

        Returns:
            ``{"ok": True, "path": path}``, or a soft error.

        """
        return await self.agenda_service.record_analysis(path, analysis)

    async def record_conclusion(self, conclusion: str) -> dict[str, Any]:
        """Record the campaign's declared conclusion.

        Returns:
            ``{"ok": True, "conclusion": conclusion}``, or a soft error.

        """
        return await self.agenda_service.record_conclusion(conclusion)

    def suggest_refinement(self, *, rel_tol: float = 0.05) -> dict[str, Any]:
        """Suggest refinement points from the recorded analyses.

        Returns:
            The refinement summary, or a soft error.

        """
        return self.agenda_service.suggest_refinement(rel_tol=rel_tol)

    def export_agenda_cwl(self) -> dict[str, Any]:
        """Return the campaign agenda as a CWL workflow document.

        Makes the emitted CWL reachable (it is the execution substrate the
        queue-based engine stands in for): the workflow YAML plus the leaf
        paths, or a soft error when there is no campaign.

        Returns:
            ``{"ok": True, "workflow": <yaml>, "leaves": [...]}`` or a soft
            error.

        """
        if not self.agenda_service.store.exists():
            return no_campaign_error()
        try:
            campaign = self.agenda_service.store.load(Campaign)
        except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
            return {"ok": False, "error": self.config.redact(str(exc))}
        return {"ok": True, **manifest_for(campaign.agenda)}

    async def set_agenda_state(self, state: CampaignState) -> dict[str, Any]:
        """Persist the campaign lifecycle state.

        Returns:
            ``{"ok": True, "state": state}``, or a soft error.

        """
        return await self.agenda_service.set_state(state)

    async def stop_agenda(self) -> dict[str, Any]:
        """Kill-switch: stop the campaign and cancel its in-flight jobs.

        Returns:
            The stop result, or ``{"ok": False, "error": "unavailable"}`` when
            the transport has not been started.

        """
        if self._transport is None:
            return {"ok": False, "error": "unavailable"}
        return await self.agenda_service.stop(self._transport.send)

    def fleet_status(self) -> dict[str, Any]:
        """Return the aggregate fleet view (summary + alerts).

        Returns:
            ``{"summary": {...}, "alerts": [...]}``; never raises.

        """
        return fleet_view(
            self.submit_service.list(),
            now=datetime.now(UTC),
            stall_after_s=self.config.fleet_stall_after_s,
        )

    def campaign_provenance(self) -> dict[str, Any]:
        """Return the campaign's RO-Crate provenance document.

        Returns:
            The ``ro-crate-metadata.json`` dict, or the actionable
            ``{"ok": False, "error": "no_campaign", "message": <recovery hint>}``
            soft error when no campaign is persisted.

        """
        if not self.agenda_service.store.exists():
            return no_campaign_error()
        try:
            campaign = self.agenda_service.store.load(Campaign)
        except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
            return {"ok": False, "error": self.config.redact(str(exc))}
        crate = campaign_rocrate(
            campaign,
            analyses=campaign.analyses,
            revision=self.config.picongpu_revision or None,
        )
        return {"ok": True, **crate}

    def condensed_events(
        self,
        sim_id: str,
        *,
        since: str | None = None,
        types: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return the condensed event list for ``sim_id``.

        Returns:
            The condensed payload dicts, oldest first.

        """
        return condense_events(self.submit_service.event_log, sim_id=sim_id, since=since, types=types, limit=limit)

    async def stop(self) -> None:
        """Cancel the pump and close the transport."""
        if self._pump:
            self._pump.cancel()
        if self._transport:
            await self._transport.close()


#: Seeded MCP instructions.  They point the agent at the PIConGPU documentation
#: and examples that ship with the pinned ``picongpu`` package, so an agent that
#: does not know PICMI/PyPIConGPU can find how to define a simulation and how to
#: scan parameters without us bundling an example script.
SERVER_INSTRUCTIONS = (
    "Submit and follow PIConGPU simulations on a remote SLURM cluster. "
    "The 'hello' tool performs an end-to-end connectivity check. "
    "A simulation is defined either as a PICMI Python script (submit_simulation) "
    "or, for parameter studies, as a pypicongpu Runner spec "
    "(build_spec to obtain one from a PICMI script, then create_campaign to scan "
    "a spec field across values). "
    "A large spec should be passed by reference, not re-typed: call "
    "build_spec(picmi_script, write_to=...) and hand the returned spec_path to "
    "create_campaign(base_spec_path=...). "
    "create_campaign's patch_path is a dotted Runner-spec path addressed node "
    "by node: a numeric segment is a list index on a list or a dict key on a "
    "dict, so a nested list element is reachable too "
    "(e.g. sim.laser.0.focus_pos_si.1.component for the laser's focal-position "
    "component), not only a top-level field such as sim.time_steps. "
    "create_campaign refuses to overwrite an existing campaign; to start a fresh "
    "one, remove the old state first with delete_campaign (reset), optionally "
    "after stop_agenda to cancel in-flight jobs. "
    "For how to write a PICMI input file and how to define or scan multiple "
    "simulations, see the PyPIConGPU documentation: the page 'Defining Your "
    "Simulation' under python_package/foundations/defining_simulation "
    "(published at https://picongpu.readthedocs.io/en/latest/python_package/foundations/) "
    "covers simulation definition and static/dynamic parameter scans; the "
    "tutorial and the examples under lib/python/examples/ in the picongpu "
    "source tree show complete setups. Note: the focal example on the "
    "'Defining Your Simulation' page defines no plasma species, so it yields an "
    "empty spectrum as written; fold in the LWFA tutorial's plasma species for "
    "a non-empty result. A run that finishes but whose only numeric plugin "
    "artifact reads all-zero is reported as `suspect` in advance_agenda, "
    "agenda_status and fleet_status (and on its done callback); treat such a "
    "run as inconclusive physics, not as a valid result."
)


def build_server(config: Config, sim: str) -> tuple[MCPServer, HelloRuntime]:
    """Create the MCP server and its runtime, wired together via lifespan.

    Returns:
        The ``(server, runtime)`` pair.

    """
    runtime = HelloRuntime(config, sim)

    @asynccontextmanager
    async def lifespan(_server: MCPServer) -> AsyncIterator[None]:
        await runtime.start()
        try:
            yield
        finally:
            await runtime.stop()

    server = MCPServer(
        "pic-agentic",
        instructions=SERVER_INSTRUCTIONS,
        lifespan=lifespan,
    )

    @server.tool(
        title="Hello World connectivity check",
        description=(
            "Send a short message through the Matrix control channel to the "
            "simulation-side client, which runs a trivial SLURM job that "
            "prints it back. Returns the SLURM job id and captured output."
        ),
        # write/resource tier: consumes cluster resources, not destructive
        # (design section 6.2).  Not read-only, not idempotent (each call
        # submits a fresh job).
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
    )
    async def hello(message: str = "Hello World") -> dict[str, Any]:
        try:
            outcome = await runtime.hello(message)
        except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
            return {"ok": False, "state": "error", "error": runtime.config.redact(str(exc))}
        return _outcome_dict(runtime, outcome)

    @server.tool(
        title="Submit a PIConGPU simulation",
        description=(
            "Build a PICMI simulation script into a PyPIConGPU runner, send it "
            "through the Matrix control channel to the simulation-side client "
            "and submit it to the remote SLURM cluster. Returns the simulation "
            "id and the coarse accepted/submitted state. The script should "
            "define a single picmi.Simulation; a trailing sim.run(...) is "
            "tolerated and ignored (the tool never runs it here)."
        ),
        # write/resource tier: consumes cluster resources, not destructive
        # (design section 6.2).  The server-side MCP client prompts for human
        # confirmation; each call is a fresh submission.
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
    )
    async def submit_simulation(
        picmi_script: str,
        *,
        build_jobs: int | None = None,
        build_cmake: str | None = None,
        build_preset: int | None = None,
        build_force: bool = False,
        cfg_file: str | None = None,
        overwrite_vars: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            params = SubmitParams(
                build_jobs=build_jobs,
                build_cmake=build_cmake,
                build_preset=build_preset,
                build_force=build_force,
                cfg_file=cfg_file,
                overwrite_vars=overwrite_vars,
            )
            outcome = await runtime.submit(picmi_script, params=params)
        except _SUBMIT_TOOL_ERRORS as exc:
            return {"ok": False, "state": "error", "error": runtime.config.redact(str(exc))}
        return _submit_outcome_dict(runtime, outcome)

    @server.tool(
        title="Build a simulation spec without submitting",
        description=(
            "Build a PICMI simulation script into a PyPIConGPU Runner spec and "
            "return it without sending anything to the cluster. Use it to obtain "
            "a base spec for create_campaign or add_agenda_leaf. The returned "
            "`spec` is the inline `{sim: ...}` wire object accepted by those "
            "tools; the result also reports the encoded wire size and whether it "
            "fits the 48 KiB inline submission limit. Pass `write_to` to stage "
            "the spec as JSON under the server's spec directory (a "
            "relative path is resolved there and a missing parent is created; a "
            "path outside that root is refused): the "
            "returned `spec_path` can then be handed to "
            "create_campaign(base_spec_path=...) without re-typing the spec. When "
            "`write_to` is set the inline `spec` is omitted by default so the "
            "tens-of-KiB body is not echoed; set `include_spec=true` to force it "
            "back (or `include_spec=false` to omit it without staging). An "
            "over-cap spec is reported as `ok: false` with "
            "`error='spec_exceeds_inline_limit'` and cannot be submitted inline; "
            "`write_to` still stages it and the result still reports `spec_path`, "
            "and `include_spec=true` returns the inline copy anyway. The "
            "script should define a single picmi.Simulation; a trailing "
            "sim.run(...) is tolerated and ignored (the tool never runs it here). "
            "Note: the pinned pypicongpu always adds a default `type_radiation` "
            "output block (empty species/period) even when the script requests no "
            "radiation; this is the schema's own default and is required for the "
            "spec to validate, so forward the `spec` verbatim rather than editing "
            "it out."
        ),
        # read-tier: it builds locally and starts no cluster work.
        annotations=_READ_ONLY,
    )
    async def build_spec(
        picmi_script: str,
        *,
        write_to: str | None = None,
        include_spec: bool | None = None,
    ) -> dict[str, Any]:
        try:
            built, spec_path = await runtime.build_spec(picmi_script, write_to=write_to)
        except _SUBMIT_TOOL_ERRORS as exc:
            return {"ok": False, "state": "error", "error": runtime.config.redact(str(exc))}
        # A staged spec (``write_to``) is consumed by reference, so echoing the
        # full tens-of-KiB inline copy would only bloat the tool result; an
        # over-cap spec is likewise omitted by default (the soft error reports
        # why), while an explicit ``include_spec`` overrides either way.
        if include_spec is None:
            include_spec = spec_path is None and built.within_inline_limit
        return _built_spec_dict(runtime, built, spec_path=spec_path, include_spec=include_spec)

    _register_reporting_tools(server, runtime)
    _register_control_result_tools(server, runtime)
    _register_agenda_tools(server, runtime)
    _register_lifecycle_tools(server, runtime)
    return server, runtime


def _register_reporting_tools(server: MCPServer, runtime: HelloRuntime) -> None:
    """Register the M2b read-tier reporting tools on ``server``.

    Args:
        server: The MCP server to add the tools to.
        runtime: The runtime the tools delegate to.

    """

    @server.tool(
        title="Get simulation status",
        description=(
            "Report the lifecycle state of one simulation. For a known, "
            "non-terminal simulation a live scontrol view is fetched from the "
            "cluster and merged over the last-event projection; otherwise the "
            "signed-room projection is returned. Progress (step, percent, "
            "walltime, avg_per_step, eta_s) is populated from the run's "
            "step_finished events while it is running, not only after it finishes. "
            "`phase` is the coarse build-vs-queue-vs-run substate "
            "(building/queued/running/done/failed/cancelled): while `job_id` is "
            "null the run is `building`, which can take 15-20 minutes before the "
            "SLURM job id appears, so a null `job_id` is not a fault. "
            "For a completed run, `suspect` carries the all-zero health warning "
            "when its numeric diagnostics are all empty. A status is available "
            "for any simulation the signed room records, including runs whose "
            "campaign was since deleted with delete_campaign; such a run is "
            "history, not live campaign state."
        ),
        annotations=_READ_ONLY,
    )
    async def get_status(sim_id: str) -> dict[str, Any]:
        return await _status_tool(runtime, sim_id)

    @server.tool(
        title="List simulations",
        description=(
            "List the simulations the server knows about, optionally only the "
            "still-active ones. Each row carries `phase` (building/queued/"
            "running/done/failed/cancelled), so a run with `job_id: null` in the "
            "`building` phase is visibly mid-build rather than missing. Each row "
            "also carries `suspect`, the all-zero health "
            "warning for a completed empty run. This is the fleet registry (a "
            "replay of the signed room), so it is independent of the campaign "
            "file: deleting a campaign with delete_campaign does not remove its "
            "already-run simulations from this list, and their results stay "
            "reachable. Use active_only=true to hide terminal history."
        ),
        annotations=_READ_ONLY,
    )
    def list_simulations(*, active_only: bool = False) -> dict[str, Any]:
        rows = [
            {
                "sim_id": record.sim_id,
                "cmd_id": record.cmd_id,
                "state": record.state,
                "phase": record.phase,
                "job_id": record.job_id,
                "suspect": record.suspect,
                "last_event_type": record.last_event_type,
                "last_event_ts": record.last_event_ts,
                "active": record.active,
            }
            for record in runtime.list_sims(active_only=active_only)
        ]
        return _redact_dict(runtime, {"simulations": rows})

    @server.tool(
        title="Get simulation events",
        description=(
            "Return the condensed lifecycle-event history of one simulation "
            "(consecutive duplicate states collapse). Optionally filter by an "
            "ISO timestamp lower bound and by state type. An empty `events` "
            "list is not an error: between `accepted` and the SLURM job the "
            "simclient reports no lifecycle event for the multi-minute local "
            "build, so read `phase` from get_status/list_simulations to tell "
            "`building`/`queued` apart; `note` explains the empty result "
            "(build window, finished run, or excluding filter)."
        ),
        annotations=_READ_ONLY,
    )
    def get_events(
        sim_id: str,
        *,
        since: str | None = None,
        types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        events = runtime.condensed_events(sim_id, since=since, types=types, limit=limit)
        result: dict[str, Any] = {"sim_id": sim_id, "events": events, "count": len(events)}
        record = runtime.get_sim(sim_id)
        if record is not None:
            result["phase"] = record.phase
        if not events:
            result["note"] = _empty_events_note(record, filtered=bool(since) or bool(types))
        return _redact_dict(runtime, result)

    @server.tool(
        title="Get simulation logs",
        description="Return up to `tail` lines of a simulation's stdout, stderr or workflow log stream.",
        annotations=_READ_ONLY,
    )
    async def get_logs(sim_id: str, *, stream: str = "stdout", tail: int = 100) -> dict[str, Any]:
        return await _logs_tool(runtime, sim_id, stream=stream, tail=tail)


async def _logs_tool(runtime: HelloRuntime, sim_id: str, *, stream: str, tail: int) -> dict[str, Any]:
    """Fetch a simulation's log tail, degrading every failure to data.

    Returns:
        The redacted log payload, or a soft ``error`` dict.

    """
    if stream not in LOG_STREAMS:
        return {"sim_id": sim_id, "stream": stream, "error": "unknown_stream"}
    tail = max(0, min(tail, _MAX_LOG_TAIL))
    try:
        payload = await runtime.fetch_logs(sim_id, stream=stream, tail=tail)
    except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
        return {
            "sim_id": sim_id,
            "stream": stream,
            "lines": [],
            "total_lines": 0,
            "error": runtime.config.redact(str(exc)),
        }
    if not payload:
        return {"sim_id": sim_id, "stream": stream, "lines": [], "total_lines": 0, "error": "unavailable"}
    return _redact_dict(runtime, payload)


def _register_control_result_tools(  # ruff: ignore[complex-structure] - one registration block per verb
    server: MCPServer,
    runtime: HelloRuntime,
) -> None:
    """Register the M3 control and results tools on ``server``.

    Control verbs are write-tier pulls (not destructive, not idempotent); the
    result verbs are read-tier pulls.  Every soft failure (a bad argument, an
    ack timeout, an unreadable file) comes back as ``{"ok": False, ...}`` rather
    than raising, so one failing tool never kills the MCP session.

    Args:
        server: The MCP server to add the tools to.
        runtime: The runtime the tools delegate to.

    """

    @server.tool(
        title="Checkpoint a simulation",
        description=(
            "Ask a running simulation to write a checkpoint at the next step "
            "and keep running (SIGUSR1 via scancel --signal)."
        ),
        annotations=_CONTROL_ANNOTATIONS,
    )
    async def checkpoint_simulation(sim_id: str) -> dict[str, Any]:
        return await _control_tool(runtime, sim_id, SimulationOp.CHECKPOINT)

    @server.tool(
        title="Stop a simulation",
        description=(
            "Ask a running simulation to stop cleanly at the next step (SIGTERM "
            "via scancel --signal), leaving any checkpoint it has already written."
        ),
        annotations=_CONTROL_ANNOTATIONS,
    )
    async def stop_simulation(sim_id: str) -> dict[str, Any]:
        return await _control_tool(runtime, sim_id, SimulationOp.STOP)

    @server.tool(
        title="Cancel a simulation",
        description=(
            "Cancel a simulation's SLURM job immediately (plain scancel). For a "
            "known sim that has not reached a live job yet (accepted but still "
            "building, or a lost-ack record) the client answers "
            "`not_signalable` with the state rather than a bare `unknown_sim`, "
            "so a stranded sim is diagnosable; `unknown_sim` still means the id "
            "was never submitted."
        ),
        # destructive: it kills the job outright -- no clean shutdown, and any
        # output since the last checkpoint is lost.
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False),
    )
    async def cancel_simulation(sim_id: str) -> dict[str, Any]:
        return await _control_tool(runtime, sim_id, SimulationOp.CANCEL)

    @server.tool(
        title="Checkpoint and stop a simulation",
        description=(
            "Ask a running simulation to write a checkpoint at the next step "
            "and then stop cleanly (SIGALRM via scancel --signal) -- the atomic "
            "'save state and finish' control."
        ),
        annotations=_CONTROL_ANNOTATIONS,
    )
    async def checkpoint_and_stop_simulation(sim_id: str) -> dict[str, Any]:
        return await _control_tool(runtime, sim_id, SimulationOp.CHECKPOINT_AND_STOP)

    @server.tool(
        title="Describe a simulation's results",
        description=(
            "Return the light manifest (files, formats, sizes, records) of a "
            "simulation's linked simOutput directory. `format` names the likely "
            "reader for each file: `openpmd-adios2`/`openpmd-hdf5` for field "
            "series, a plugin reader name (`energy_histogram`, `emittance`, "
            "`transition_radiation`, `phase_space`, `radiation`, `calorimeter`, "
            "`png`) for plugin output, or `text`/`dir`/`binary`. The scan never "
            "opens a file and needs no openPMD reader."
        ),
        annotations=_READ_ONLY,
    )
    async def describe_results(sim_id: str) -> dict[str, Any]:
        return await _result_tool(runtime, ResultOp.DESCRIBE, sim_id=sim_id)

    @server.tool(
        title="Read a slice of simulation results",
        description=(
            "Reduce one openPMD record/component of a simulation to a bounded "
            "1D slice (axis/index/downsample optional; iteration 'last' by "
            "default). The openPMD series is discovered under the run's "
            "simOutput (optionally narrowed by `path`). Requires the openPMD "
            "reader on the cluster."
        ),
        annotations=_READ_ONLY,
    )
    async def get_result_slice(
        sim_id: str,
        record: str,
        *,
        path: str | None = None,
        component: str | None = None,
        iteration: int | str = "last",
        axis: int = 0,
        index: int | None = None,
        downsample: int | None = None,
    ) -> dict[str, Any]:
        return await _result_tool(
            runtime,
            ResultOp.SLICE,
            sim_id=sim_id,
            path=path,
            record=record,
            component=component,
            iteration=iteration,
            axis=axis,
            index=index,
            downsample=downsample,
        )

    @server.tool(
        title="Read a small result text stream",
        description=(
            "Return a small text tail of a result file, or of the captured "
            "stdout/stderr stream, under a simulation's simOutput directory."
        ),
        annotations=_READ_ONLY,
    )
    async def read_result(
        sim_id: str,
        path: str | None = None,
        *,
        stream: str | None = None,
        tail: int | None = None,
    ) -> dict[str, Any]:
        return await _result_tool(runtime, ResultOp.READ, sim_id=sim_id, path=path, stream=stream, tail=tail)

    @server.tool(
        title="Render a result image",
        description=(
            "Render one openPMD record/component of a simulation as a bounded "
            "base64 PNG thumbnail (iteration 'last' by default). The openPMD "
            "series is discovered under the run's simOutput (optionally narrowed "
            "by `path`). Requires the openPMD and Pillow readers on the cluster."
        ),
        annotations=_READ_ONLY,
    )
    async def get_result_image(
        sim_id: str,
        record: str,
        *,
        path: str | None = None,
        component: str | None = None,
        iteration: int | str = "last",
    ) -> dict[str, Any]:
        return await _result_tool(
            runtime,
            ResultOp.IMAGE,
            sim_id=sim_id,
            path=path,
            record=record,
            component=component,
            iteration=iteration,
        )

    @server.tool(
        title="Read a PIConGPU plugin result",
        description=(
            "Run one of PIConGPU's shipped plugin readers over a simulation's "
            "simOutput and return a bounded numeric summary. `reader` is one of "
            "'energy_histogram', 'emittance', 'transition_radiation', "
            "'phase_space', 'radiation', 'calorimeter' or 'png'; `species` and "
            "`species_filter` select the output, and `iteration` picks a step "
            "('last' by default). For `energy_histogram`, `min_kev`/`max_kev` "
            "set the `count_in_window` energy window; when omitted it is derived "
            "from the populated bins, and the summary always reports the window, "
            "`n_nonzero_bins` and the populated min/max so a mismatched window is "
            "obvious. The PNG reader returns image metadata only "
            "(dimensions, iteration, path); fetch the image with "
            "`export_results`. Use `describe_results` to see which plugin files "
            "exist. Requires the picongpu readers (and, for the openPMD/image "
            "readers, openpmd_api/imageio) on the cluster."
        ),
        annotations=_READ_ONLY,
    )
    async def read_plugin_result(
        sim_id: str,
        reader: str,
        *,
        species: str | None = None,
        species_filter: str | None = None,
        iteration: int | str | None = None,
        path: str | None = None,
        min_kev: float | None = None,
        max_kev: float | None = None,
    ) -> dict[str, Any]:
        return await _result_tool(
            runtime,
            ResultOp.PLUGIN,
            sim_id=sim_id,
            reader=reader,
            species=species,
            species_filter=species_filter,
            iteration=iteration,
            path=path,
            min_kev=min_kev,
            max_kev=max_kev,
        )

    @server.tool(
        title="Export simulation results",
        description=(
            "Return a transfer ticket (a ResultRef plus an rsync command, or a "
            "resolved local path when the server mirrors the output). The server "
            "never moves the bulk data itself."
        ),
        annotations=_READ_ONLY,
    )
    async def export_results(sim_id: str) -> dict[str, Any]:
        return await _result_tool(runtime, ResultOp.EXPORT, sim_id=sim_id)

    @server.tool(
        title="Analyze a simulation's output",
        description=(
            "Analyze a simulation's output: reuses the shipped PIConGPU plugin "
            "readers to summarize the physics (energy histogram, phase space, "
            "radiation, ...), plus the RO-Crate experiment metadata, the "
            "(redacted) pypicongpu run metadata and an openPMD output summary. "
            "The deterministic natural-language answer is physics-first: a "
            "`query` about the spectrum or energy is answered from the plugin "
            "summary values and never from bookkeeping that merely shares a "
            "word, and when the run has no openPMD output or no plugin "
            "histogram the answer says so explicitly instead of returning "
            "metadata only. For a multi-species run each reader's configured "
            "species is resolved from the pypicongpu metadata. No LLM is "
            "called; missing inputs degrade to empty sections."
        ),
        annotations=_READ_ONLY,
    )
    async def analyze_output(sim_id: str, *, query: str | None = None) -> dict[str, Any]:
        return await _analyze_tool(runtime, sim_id, query=query)


def _register_agenda_tools(server: MCPServer, runtime: HelloRuntime) -> None:
    """Register the campaign-agenda tools on ``server``.

    ``advance_agenda`` is a write-tier call (it may submit new simulations, so
    not read-only and not idempotent, but it is not destructive); it returns the
    :class:`~pic_agentic.agenda.engine.TickResult` dict.  ``agenda_status`` is a
    read-tier view of the aggregate campaign.  Both degrade to a soft
    ``{"ok": False, ...}`` dict rather than raising.

    Args:
        server: The MCP server to add the tools to.
        runtime: The runtime the tools delegate to.

    """

    @server.tool(
        title="Advance the simulation campaign",
        description=(
            "Run one durable tick of the campaign engine stored on the server: "
            "observe the known simulations, plan the next actions and submit "
            "what the budget and the policy allow. Returns the tick result "
            "(submitted/waiting/deferred/done/failed paths and usage). `deferred` "
            "lists leaves whose submission outcome is unknown (a lost ack or a "
            "pending cluster record): they stay planned and are retried next "
            "tick under the same exactly-once command id, so a lost ack is never "
            "reported as a physics failure. The retry is bounded by elapsed "
            "wall-clock time (configurable), not by a tick count, so quick "
            "successive ticks do not fail a still-building job; after the window "
            "the leaf is failed with an `outcome_unknown` code (the job may still "
            "exist -- check list_simulations and cancel it). `state` is the "
            "campaign's lifecycle state (running/paused/stopped), or `complete` "
            "once `complete` is true and every leaf has finished; `lifecycle` "
            "always carries the stored lifecycle state (running/paused/stopped), "
            "so a finished paused/stopped campaign reads `state: complete` with "
            "`lifecycle: paused`/`stopped`. Failures are "
            "summarised in `failure_summary` and grouped by identical reason in "
            "`failure_groups` (with the full, untruncated text available via "
            "take_agenda_callbacks). A leaf that finished but whose only numeric "
            "artifact reads all-zero is 'successfully empty': it appears in "
            "`suspects` (path -> warning) and its done callback carries a "
            "`suspect` warning, so a zero-physics run is never mistaken for a "
            "real result. `callbacks` holds only the decision points emitted by "
            "*this* tick; they are also persisted, so `take_agenda_callbacks` "
            "(which drains the durable store and is the recovery path after a "
            "restart) returns them too until drained. React to the inline "
            "`callbacks` for the tick you just ran; call `take_agenda_callbacks` "
            "only to recover callbacks from earlier ticks, since draining clears "
            "the persisted copy."
        ),
        # write/resource tier: a tick may submit new cluster jobs, so it is not
        # read-only and not idempotent, but it is not destructive.
        annotations=_CONTROL_ANNOTATIONS,
    )
    async def advance_agenda() -> dict[str, Any]:
        result = await runtime.advance_agenda()
        return _redact_dict(runtime, result)

    @server.tool(
        title="Get the simulation campaign status",
        description=(
            "Report the aggregate status of the persisted campaign: its name, "
            "completion flag, per-status counts, accumulated usage and the "
            "per-leaf view (path, status, sim_id, sweep point and its readable "
            "sweep_parameter). A leaf whose submission outcome is unknown (a "
            "lost ack, still retried) is flagged `deferred` with its "
            "`deferred_attempts` and `deferred_since`, distinct from a terminal "
            "`failed`; retrying is bounded by wall-clock time, not by a tick "
            "count, so quick successive ticks do not fail a still-building job. A ``done`` "
            "leaf whose only numeric artifact reads all-zero is flagged `suspect` "
            "(and counted in `suspect_count`/`suspects`), so an empty run is not "
            "reported as a clean success."
        ),
        annotations=_READ_ONLY,
    )
    def agenda_status() -> dict[str, Any]:
        return _redact_dict(runtime, runtime.agenda_status())

    @server.tool(
        title="Approve a gated campaign leaf",
        description=(
            "Mark one leaf of the persisted campaign as approved, so the next "
            "advance_agenda tick may submit it despite an approval gate. The "
            "flag is persisted, so approval survives a server restart."
        ),
        # write/resource tier: it changes persisted campaign state but starts no
        # work itself; it is not destructive.
        annotations=_CONTROL_ANNOTATIONS,
    )
    async def approve_agenda_leaf(path: str) -> dict[str, Any]:
        return _redact_dict(runtime, await runtime.approve_agenda_leaf(path))

    @server.tool(
        title="Drain the campaign decision-point callbacks",
        description=(
            "Return and clear the pending callbacks for newly finished or failed "
            "campaign leaves. An MCP server cannot call the agent, so these are "
            "pollable records: drain them, analyse the run or refine the agenda, "
            "then advance_agenda again. `advance_agenda` also returns each "
            "tick's new callbacks inline, and that inline copy does **not** "
            "consume the stored ones: react to the inline list for the tick you "
            "just ran, and use this tool only to recover callbacks from earlier "
            "ticks or after a restart. Draining is destructive -- it clears the "
            "persisted copy, so a repeat call returns an empty list, and after "
            "draining the same callbacks are no longer available at all."
        ),
        annotations=_CONTROL_ANNOTATIONS,
    )
    async def take_agenda_callbacks() -> dict[str, Any]:
        return _redact_dict(runtime, await runtime.take_agenda_callbacks())

    @server.tool(
        title="Create a simulation campaign",
        description=(
            "Create and persist a campaign with one leaf per value: each leaf is "
            "`base_spec` with the dotted Runner-spec path `patch_path` set to "
            "that value (e.g. patch_path='sim.time_steps'), and records "
            "point={last path segment: value} plus a human-readable "
            "sweep_parameter (the path with list indices dropped, e.g. "
            "'sim.laser.focus_pos_si.component'; override it with `parameter`). "
            "A numeric path segment is interpreted by the node it addresses: a "
            "list index on a list, or a dict key on a dict, so a nested list "
            "element is reachable (e.g. "
            "patch_path='sim.laser.0.focus_pos_si.1.component' for the laser's "
            "focal-position component); the final segment must already address "
            "an existing field or in-range list index. This is the entry point for the "
            "research loop -- call build_spec first to get base_spec, then "
            "advance_agenda. Provide the base spec exactly one way: inline as "
            "`base_spec`, or by reference as `base_spec_path` (the `spec_path` "
            "build_spec(write_to=...) returned, a JSON file under the server's "
            "spec directory) so the spec never has to be re-typed. Every patched "
            "leaf is validated against the pinned pypicongpu schema before "
            "anything is persisted, so a malformed base_spec is refused up front "
            "with an `invalid_campaign_spec` error naming the offending field(s) "
            "instead of failing at each submission. Refuses to overwrite an "
            "existing campaign."
        ),
        # write/resource tier: it creates persisted campaign state but starts no
        # cluster work itself; it is not destructive.
        annotations=_CONTROL_ANNOTATIONS,
    )
    async def create_campaign(
        name: str,
        patch_path: str,
        values: list[Any],
        *,
        base_spec: dict[str, Any] | None = None,
        base_spec_path: str | None = None,
        parameter: str | None = None,
    ) -> dict[str, Any]:
        result = await runtime.create_campaign(
            name,
            base_spec,
            patch_path,
            values,
            base_spec_path=base_spec_path,
            parameter=parameter,
        )
        return _redact_dict(runtime, result)

    @server.tool(
        title="Delete/reset the campaign",
        description=(
            "Remove the persisted campaign and its sibling reuse registry, so a "
            "fresh campaign can be created (create_campaign refuses to overwrite "
            "an existing one). Refused with an actionable `campaign_in_flight` "
            "error while any leaf may still have a live cluster job (submitted/"
            "running, or a non-terminal leaf carrying a sim_id such as a lost-ack "
            "submission), since deleting then would orphan the jobs; pass "
            "force=true to delete anyway, or stop_agenda first to cancel them. A "
            "corrupt campaign file is removed without the guard. This only "
            "clears the campaign: the simulations that already ran stay in the "
            "fleet registry (list_simulations/get_status/fleet_status), because "
            "the registry is a replay of the signed room, not the campaign file, "
            "and their results remain reachable. Treat those entries as the "
            "history of runs that actually happened, not as live campaign "
            "state."
        ),
        # destructive: it irreversibly removes the persisted campaign state.
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False),
    )
    async def delete_campaign(*, force: bool = False) -> dict[str, Any]:
        result = await runtime.delete_campaign(force=force)
        return _redact_dict(runtime, result)

    @server.tool(
        title="Add a simulation leaf to the campaign",
        description=(
            "Add one simulation leaf to the persisted campaign's root group, so "
            "the next advance_agenda tick can submit it. Used by the agent to "
            "refine a sweep (e.g. add points around an optimum). `point` is the "
            "sweep assignment to record; pass `parameter` to label the swept "
            "quantity in human-readable form (stored as sweep_parameter)."
        ),
        annotations=_CONTROL_ANNOTATIONS,
    )
    async def add_agenda_leaf(
        name: str,
        spec: dict[str, Any],
        point: dict[str, Any] | None = None,
        depends_on: list[str] | None = None,
        parameter: str | None = None,
    ) -> dict[str, Any]:
        result = await runtime.add_agenda_leaf(name, spec, point=point, depends_on=depends_on, parameter=parameter)
        return _redact_dict(runtime, result)

    _register_research_tools(server, runtime)


def _register_research_tools(server: MCPServer, runtime: HelloRuntime) -> None:
    """Register the research-loop tools (analysis, refinement, conclusion, CWL).

    Args:
        server: The MCP server to add the tools to.
        runtime: The runtime the tools delegate to.

    """

    @server.tool(
        title="Record an analysis on the campaign",
        description=(
            "Attach one leaf's analysis (e.g. the analyze_output sections) to the "
            "campaign, so the provenance record links inputs -> runs -> analyses. "
            "Only a top-level numeric `score` (or `value`/`peak`) key is ranked by "
            "`suggest_agenda_refinement`; an analysis without one is recorded but "
            "not ranked, so keep the scalar objective under such a key."
        ),
        annotations=_CONTROL_ANNOTATIONS,
    )
    async def record_agenda_analysis(path: str, analysis: dict[str, Any]) -> dict[str, Any]:
        return _redact_dict(runtime, await runtime.record_analysis(path, analysis))

    @server.tool(
        title="Suggest a refinement from the recorded analyses",
        description=(
            "Score the *recorded analyses* and report the best point, whether the "
            "sweep has converged, and deterministic refinement points to add "
            "around the optimum. Only leaves analysed with "
            "`record_agenda_analysis` are ranked; with no analyses it returns "
            "`best: null` and `analysed: 0` (the sweep point is never a score). "
            "Convergence needs at least two ranked analyses, so a single analysis "
            "reports `converged: false`."
        ),
        annotations=_READ_ONLY,
    )
    def suggest_agenda_refinement(rel_tol: float = 0.05) -> dict[str, Any]:
        return _redact_dict(runtime, runtime.suggest_refinement(rel_tol=rel_tol))

    @server.tool(
        title="Declare the campaign conclusion",
        description=(
            "Record the campaign's conclusion text; it is emitted in the campaign "
            "provenance report, closing the lineage."
        ),
        annotations=_CONTROL_ANNOTATIONS,
    )
    async def conclude_agenda(conclusion: str) -> dict[str, Any]:
        return _redact_dict(runtime, await runtime.record_conclusion(conclusion))

    @server.tool(
        title="Export the campaign agenda as CWL",
        description=(
            "Render the campaign's agenda as a CWL Workflow document (the "
            "execution substrate): the workflow YAML plus the leaf paths. "
            "Inspect it, or hand it to cwltool."
        ),
        annotations=_READ_ONLY,
    )
    def export_agenda_cwl() -> dict[str, Any]:
        return _redact_dict(runtime, runtime.export_agenda_cwl())

    @server.tool(
        title="Run a tailored analysis program on a simulation",
        description=(
            "Evaluate a declarative analysis program on the cluster next to the "
            "data, without executing any code: select openPMD mesh components by "
            "name and combine them with a bounded expression tree (arithmetic, "
            "abs/sqrt/log/exp/sin/cos/tanh and reductions such as sum/mean/std/"
            "min/max/median/quantile/histogram/fft_peak). Returns a scalar, a "
            "bounded numeric array, or a histogram/spectrum. Invalid or oversized "
            "programs are rejected; the output is capped to the ack budget.\n"
            "The program is a fully validated expression tree (never eval'd, "
            "never a serialised sympy object); unknown fields are rejected. "
            "Schema:\n"
            "  program = {selectors?: [var, ...], output: expr, points?: expr}\n"
            "  var     = {kind: 'var', name: str, record?: str, component?: str, iteration?: int|str}\n"
            "  expr    = const | var | binop | unop | reduce\n"
            "    const  = {kind: 'const', value: number}\n"
            "    binop  = {kind: 'binop', op: 'add'|'sub'|'mul'|'div'|'pow', left: expr, right: expr}\n"
            "    unop   = {kind: 'unop', op: 'neg'|'abs'|'sqrt'|'log'|'exp'|'sin'|'cos'|'tanh'|'sign', operand: expr}\n"
            "    reduce = {kind: 'reduce', op: 'sum'|'mean'|'std'|'min'|'max'|'median'|"
            "'argmax'|'argmin'|'quantile'|'histogram'|'fft_peak'|'fft_freq', operand: expr, "
            "q?: 0..1, bins?: int}\n"
            "Selector precedence: the `selectors` list is documentation, not a "
            "gate - a `var` may appear bare and is resolved by `name`. When both "
            "are present, the attributes on the `var` node itself are "
            "authoritative: `var.record`/`var.component`/`var.iteration` win, and "
            "each falls back to the matching declared selector's value, then to "
            "the request defaults, when the node omits it. A node attribute that "
            "contradicts its declared selector is rejected as ambiguous, declared "
            "selector names must be unique, and one name may not be referenced by "
            "two nodes whose attributes differ. "
            "`output` is the returned expression; `points` is an optional "
            "parallel expression (e.g. an FFT frequency axis).\n"
            "Worked example - the transverse energy spectrum of the E field:\n"
            '{"selectors": [{"kind": "var", "name": "px", "record": "E", "component": "x"}, '
            '{"kind": "var", "name": "py", "record": "E", "component": "y"}], '
            '"output": {"kind": "reduce", "op": "histogram", "bins": 4, "operand": '
            '{"kind": "binop", "op": "add", '
            '"left": {"kind": "binop", "op": "mul", "left": {"kind": "var", "name": "px"}, '
            '"right": {"kind": "var", "name": "px"}}, '
            '"right": {"kind": "binop", "op": "mul", "left": {"kind": "var", "name": "py"}, '
            '"right": {"kind": "var", "name": "py"}}}}}'
        ),
        annotations=_CONTROL_ANNOTATIONS,
    )
    async def run_analysis(sim_id: str, program: dict[str, Any]) -> dict[str, Any]:
        return await _result_tool(runtime, ResultOp.COMPUTE, sim_id=sim_id, program=program)

    @server.tool(
        title="Get the aggregate fleet status",
        description=(
            "Report the whole fleet at a glance: total/active/terminal counts, "
            "per-state counts, a mean progress percentage, per-phase counts "
            "(building/queued/running/done/failed/cancelled), and actionable "
            "alerts (failed, cancelled, non-zero exit, stalled, suspect). A "
            "`building`/`queued` run is never reported as `stalled` even if it "
            "has been silent for a long time: only a `running` run that stops "
            "emitting progress is flagged. The "
            "`suspect` count/alert marks a done run whose only numeric artifact "
            "reads all-zero - a successful-but-empty run, not a failure."
        ),
        annotations=_READ_ONLY,
    )
    def fleet_status() -> dict[str, Any]:
        return _redact_dict(runtime, runtime.fleet_status())

    @server.tool(
        title="Export the campaign provenance (RO-Crate)",
        description=(
            "Render the persisted campaign's lineage as a minimal RO-Crate "
            "JSON-LD document: one entity per simulation linked to the "
            "PIConGPU software entity at a pinned revision, plus its runs."
        ),
        annotations=_READ_ONLY,
    )
    def campaign_provenance() -> dict[str, Any]:
        return _redact_dict(runtime, runtime.campaign_provenance())


def _register_lifecycle_tools(server: MCPServer, runtime: HelloRuntime) -> None:
    """Register the campaign lifecycle tools (pause / resume / kill-switch).

    These control whether ``advance_agenda`` may submit: ``pause_agenda`` holds
    submissions (resumable), ``resume_agenda`` releases them, and ``stop_agenda``
    is the destructive kill-switch that flips the state and cancels in-flight
    jobs.  All degrade to a soft ``{"ok": False, ...}`` dict.

    Args:
        server: The MCP server to add the tools to.
        runtime: The runtime the tools delegate to.

    """

    @server.tool(
        title="Pause the campaign",
        description=(
            "Pause the campaign: advance_agenda still observes runs and records "
            "callbacks, but submits nothing until it is resumed. Persisted, so "
            "the pause survives a server restart."
        ),
        annotations=_CONTROL_ANNOTATIONS,
    )
    async def pause_agenda() -> dict[str, Any]:
        return _redact_dict(runtime, await runtime.set_agenda_state("paused"))

    @server.tool(
        title="Resume the campaign",
        description="Resume a paused campaign so advance_agenda may submit again.",
        annotations=_CONTROL_ANNOTATIONS,
    )
    async def resume_agenda() -> dict[str, Any]:
        return _redact_dict(runtime, await runtime.set_agenda_state("running"))

    @server.tool(
        title="Stop the campaign (kill-switch)",
        description=(
            "Stop the campaign outright and cancel every job that may still be "
            "live -- including a lost-ack/deferred leaf, which stays planned but "
            "already carries a sim_id. The state is flipped before cancellation, "
            "so no tick can submit after the switch. Returns `cancelled` (jobs "
            "the client confirmed killed), `cleared` (leaves that were known but "
            "had no live job, e.g. an accepted-but-unsubmitted stranded sim: "
            "resolved, nothing to kill) and `errors` (each with the sim_id and "
            "the client's reason/code, never a bare 'rejected'). This is "
            "terminal; it is not resumable."
        ),
        # destructive: it cancels running cluster jobs.
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False),
    )
    async def stop_agenda() -> dict[str, Any]:
        return _redact_dict(runtime, await runtime.stop_agenda())


async def _analyze_tool(runtime: HelloRuntime, sim_id: str, *, query: str | None = None) -> dict[str, Any]:
    """Run one ``analyze`` pull and shape it into the design's section dict.

    The simclient composes the RO-Crate / metadata / openPMD sections and a
    default answer.  The frozen result command carries no query field, so when
    a query is given the answer is re-synthesized here -- ``synthesize_answer``
    is a pure function of the sections, so this is exact.

    Returns:
        ``{"ok": True, rocrate, metadata, openpmd, answer}``, else
        ``{"ok": False, "sim_id", "op", "error"}``.

    """
    op = ResultOp.ANALYZE
    try:
        params = ResultParams(sim_id=sim_id, op=op)
        payload = await runtime.fetch_result(params)
    except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
        return _soft_error(runtime, sim_id, op.value, exc)
    if not payload:
        return {"ok": False, "sim_id": sim_id, "op": op.value, "error": "unavailable"}
    if payload.get("error"):
        return {"ok": False, **_redact_dict(runtime, payload)}
    sections = payload.get("result")
    if not isinstance(sections, dict):
        return {"ok": False, "sim_id": sim_id, "op": op.value, "error": "unavailable"}
    rocrate = sections.get("rocrate") or {}
    metadata = sections.get("metadata") or {}
    openpmd = sections.get("openpmd") or {}
    plugins = sections.get("plugins") or {}
    answer = sections.get("answer")
    if query:
        try:
            from pic_agentic import analysis  # ruff: ignore[import-outside-top-level] - lazy optional engine

            answer = analysis.synthesize_answer(query, rocrate, metadata, openpmd, plugins)
        except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
            return _soft_error(runtime, sim_id, op.value, exc)
    result = {
        "ok": True,
        "sim_id": sim_id,
        "rocrate": rocrate,
        "metadata": metadata,
        "openpmd": openpmd,
        "plugins": plugins,
        "answer": answer,
    }
    return _redact_dict(runtime, result)


async def _control_tool(runtime: HelloRuntime, sim_id: str, op: SimulationOp) -> dict[str, Any]:
    """Run one control pull and shape it into a redacted outcome dict.

    Returns:
        ``{"ok": True, ...}`` on success, else ``{"ok": False, "error": ...}``.

    """
    op = SimulationOp(op)
    try:
        payload = await runtime.control(sim_id, op)
    except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
        return _soft_error(runtime, sim_id, op.value, exc)
    if not payload:
        return {"ok": False, "sim_id": sim_id, "op": op.value, "error": "unavailable"}
    if "ok" not in payload:
        # The transport returns a bare ``{"sim_id", "error": "timeout"}`` dict
        # (never raises), which is still a soft error to the LLM.
        return _soft_error(runtime, sim_id, op.value, payload.get("error", "unavailable"))
    return _redact_dict(runtime, {"sim_id": sim_id, "op": op.value, **payload})


async def _result_tool(
    runtime: HelloRuntime,
    op: ResultOp,
    *,
    sim_id: str,
    **knobs: Any,
) -> dict[str, Any]:
    """Build and run one results pull, returning a redacted outcome dict.

    Returns:
        The ``result_ack`` payload (redacted), or ``{"ok": False, "error"}``.

    """
    try:
        params = ResultParams(sim_id=sim_id, op=op, **knobs)
        payload = await runtime.fetch_result(params)
    except Exception as exc:  # ruff: ignore[blind-except] - a tool must never raise
        return _soft_error(runtime, sim_id, op.value, exc)
    if not payload:
        return {"ok": False, "sim_id": sim_id, "op": op.value, "error": "unavailable"}
    if payload.get("error"):
        # The simclient answered with a shaped error (e.g. READER_UNAVAILABLE);
        # keep its error_code and present the same soft-error shape.
        redacted = _redact_dict(runtime, payload)
        return {"ok": False, **redacted}
    return _redact_dict(runtime, payload)


def _soft_error(runtime: HelloRuntime, sim_id: str, op: str, exc: str | BaseException) -> dict[str, Any]:
    """Shape a caught tool failure into the M2b soft-error convention.

    Returns:
        ``{"ok": False, "sim_id", "op", "error"}`` with the message redacted.

    """
    message = exc if isinstance(exc, str) else str(exc)
    return {"ok": False, "sim_id": sim_id, "op": op, "error": runtime.config.redact(message)}


async def _status_tool(runtime: HelloRuntime, sim_id: str) -> dict[str, Any]:
    """Return one simulation's status, merging a live view when available.

    Returns:
        The redacted status dict; a live-fetch failure is captured as
        ``live_error`` rather than raised.

    """
    record = runtime.get_sim(sim_id)
    if record is None:
        return {"sim_id": sim_id, "known": False, "error": "unknown_sim"}
    projection = _status_dict(record)
    if record.active:
        try:
            live = await runtime.fetch_status(sim_id)
        except Exception as exc:  # ruff: ignore[blind-except] - fall back to the projection
            return _redact_dict(runtime, {**projection, "live_error": str(exc)})
        if live and not live.get("error"):
            _merge_status(projection, live)
    return _redact_dict(runtime, projection)


def _status_dict(record: SimRecord) -> dict[str, Any]:
    """Build the registry projection of one simulation's status.

    Returns:
        The fields shared by the live and projected status responses.

    """
    return {
        "sim_id": record.sim_id,
        "state": record.state,
        "phase": record.phase,
        "slurm_state": record.slurm_state,
        "job_id": record.job_id,
        "step": record.step,
        "percent": record.percent,
        "walltime": record.walltime,
        "eta_s": record.eta_s,
        "suspect": record.suspect,
        "since_last_event_s": _since_last_event_s(record.last_event_ts),
    }


def _merge_status(projection: dict[str, Any], live: dict[str, Any]) -> None:
    """Overlay the live ``status_ack`` fields on the registry projection.

    Only fields the ack actually carries override the projection, so a live view
    missing a field does not erase a value known from the event log.

    Args:
        projection: The registry projection, updated in place.
        live: The live ``status_ack`` payload (without ids).

    """
    for field in (
        "state",
        "slurm_state",
        "job_id",
        "step",
        "percent",
        "walltime",
        "avg_per_step",
        "eta_s",
        "exit_code",
        "suspect",
    ):
        if live.get(field) is not None:
            projection[field] = live[field]
    # ``phase`` is derived from the (possibly updated) state/job_id, so it must
    # be recomputed after the overlay rather than merged as a raw field.
    projection["phase"] = simulation_phase(str(projection.get("state", "")), projection.get("job_id"))


def _empty_events_note(record: SimRecord | None, *, filtered: bool) -> str:
    """Explain an empty ``get_events`` result (H4: latency opacity).

    An empty list is normal either because the simclient reports no lifecycle
    event during the multi-minute local build/queue window, because the run has
    already finished, or because a ``since``/``types`` filter excluded every
    row.  The note names which case applies so an empty result is not mistaken
    for "started but silent"; ``filtered`` records whether the caller actually
    passed a filter, so a terminal result is not dishonestly blamed on one.

    Returns:
        A short human-readable explanation.

    """
    if record is None:
        return "no lifecycle event matches; the simulation is not in the registry (it may never have been submitted)"
    phase = record.phase
    if phase in {"building", "queued"}:
        return (
            f"no lifecycle event yet: the run is in the {phase} phase, where the simclient "
            "builds locally and waits for the SLURM job id; this can take 15-20 minutes and "
            "logs are not written yet, so there is nothing to report until the job starts"
        )
    if phase == "running":
        return (
            "no lifecycle event recorded yet for this running run; progress events are emitted "
            "only every 25% of the run, so a long-running job can be between events"
        )
    if phase in {"done", "failed", "cancelled"}:
        if filtered:
            return (
                f"the run is finished ({phase}); no retained lifecycle event matches the requested "
                "filters, so widen `since`/`types` or omit them"
            )
        return (
            f"the run is finished ({phase}) and no lifecycle event is retained for it; there is nothing more to report"
        )
    return (
        "no lifecycle event matches the requested filters; widen `since`/`types` or omit them"
        if filtered
        else "no lifecycle event recorded for this simulation"
    )


def _since_last_event_s(ts: str | None) -> int | None:
    """Seconds elapsed since ``ts``, or None when unparsable.

    Returns:
        The whole seconds since the timestamp, or None.

    """
    if not ts:
        return None
    try:
        when = datetime.strptime(ts, _TS_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return None
    return max(0, int((datetime.now(UTC) - when).total_seconds()))


def _redact_dict(runtime: HelloRuntime, payload: Any, _depth: int = 0) -> Any:
    """Redact every string in an outbound payload.

    Args:
        runtime: The runtime whose config holds the secrets.
        payload: The value about to leave toward the LLM.
        _depth: Recursion guard for nested containers.

    Returns:
        The payload with all strings redacted.

    """
    redact = runtime.config.redact
    # Redact strings at *any* depth first: a deeply nested secret must never
    # survive just because its container sits past the structural cap.
    if isinstance(payload, str):
        return redact(payload)
    if _depth > _REDACT_MAX_DEPTH:
        # Beyond the cap the shape is no longer trusted, so the subtree is
        # dropped rather than passed through raw (a raw nested value could carry
        # a secret the recursion never reached).  Dropping is safe; leaking is
        # not.
        return "[REDACTED: nesting too deep]"
    if isinstance(payload, dict):
        # Redact keys as well: a state string can itself be a dict key (e.g.
        # ``fleet_status``'s ``by_state``), so a secret embedded in a key must
        # not slip through.
        return {redact(str(key)): _redact_dict(runtime, value, _depth + 1) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_redact_dict(runtime, value, _depth + 1) for value in payload]
    return payload


def _built_spec_dict(
    runtime: HelloRuntime,
    built: BuiltSpec,
    *,
    spec_path: str | None = None,
    include_spec: bool = True,
) -> dict[str, Any]:
    """Shape a dry-run build result for the ``build_spec`` tool.

    An over-cap spec is reported as a soft error (``ok: False`` naming
    ``spec_exceeds_inline_limit``): it can never be submitted, so the failure
    mode is made explicit instead of handing the agent a spec that would fail
    later at ``advance_agenda``.  The staged path, however, is *still* surfaced
    when ``write_to`` was given -- the runtime has already written the file, so
    dropping ``spec_path`` would hide the one output the caller asked for (the
    over-cap + ``write_to`` case has no other use for ``spec``).  An explicit
    ``include_spec=True`` is likewise honoured over the cap: the inline copy
    travels over MCP, not the 48 KiB-limited homeserver path, so a caller that
    asks for it gets it alongside the size warning.

    The built spec is trusted builder output (it passed ``check_allowlist``),
    not untrusted free text, so it is returned **verbatim**: routing it through
    :func:`_redact_dict` would hit the ``_REDACT_MAX_DEPTH`` structural cap on
    any legitimate spec whose ``sim`` nests deeply (the shipped
    rotation/frequency config does), silently replacing real config with a
    ``"[REDACTED: nesting too deep]"`` marker while still reporting ``ok=True``.
    The scalar metadata (version/revision/schema hash) is redacted individually,
    matching the submit path, and the cap path is the same shape (no structural
    redaction on either branch).

    When ``spec_path`` is set the caller asked for ``write_to`` and the runtime
    has already written the spec, so ``spec_path`` names the staged file (the
    reference form of ``create_campaign``'s ``base_spec``).  The path is
    server-controlled config output, not free text.

    Args:
        runtime: The runtime whose config holds the secrets.
        built: The built wire spec plus its provenance and wire size.
        spec_path: The staged file path when ``write_to`` was given, else None.
        include_spec: Whether to include the inline ``spec`` in the result.
            ``build_spec`` defaults this to False whenever the spec is over-cap
            or a ``write_to`` path was staged (the caller consumes either by
            reference), so a matching ``spec_path`` is still reported while the
            tens-of-KiB body is not echoed back.

    Returns:
        ``{"ok": True, "spec", "wire_bytes", ...}``, or
        ``{"ok": False, "error": "spec_exceeds_inline_limit", ...}``.

    """
    redact = runtime.config.redact
    payload: dict[str, Any] = {
        "ok": built.within_inline_limit,
        "wire_bytes": built.wire_bytes,
        "inline_limit_bytes": built.inline_limit_bytes,
        "picongpu_version": redact(built.picongpu_version),
        "picongpu_revision": redact(built.picongpu_revision),
        "schema_hash": built.schema_hash,
    }
    if not built.within_inline_limit:
        payload["error"] = "spec_exceeds_inline_limit"
    if include_spec:
        payload["spec"] = built.spec
    if spec_path is not None:
        payload["spec_path"] = spec_path
    return payload


def _submit_outcome_dict(runtime: HelloRuntime, outcome: SubmitOutcome) -> dict[str, Any]:
    redact = runtime.config.redact
    payload: dict[str, Any] = {
        "ok": outcome.ok,
        "sim": outcome.sim,
        "sim_id": outcome.sim_id,
        "state": outcome.state,
        "job_id": outcome.job_id,
        "acked": outcome.acked,
        # Surface an unknown outcome explicitly: the job may exist, so a caller
        # must not read a bare ``ok: false`` as a physics/policy failure.
        "outcome_unknown": outcome.outcome_unknown,
    }
    if outcome.error:
        payload["error"] = redact(outcome.error)
    if outcome.error_code:
        payload["error_code"] = outcome.error_code
    return payload


def _outcome_dict(runtime: HelloRuntime, outcome: HelloOutcome) -> dict[str, Any]:
    redact = runtime.config.redact
    payload: dict[str, Any] = {
        "ok": outcome.ok,
        "sim": outcome.sim,
        "job_id": outcome.job_id,
        "acked": outcome.acked,
        "cluster_output": redact(outcome.cluster_output) if outcome.cluster_output else None,
    }
    if outcome.capabilities is not None:
        # Surface the handshake so a version drift is visible to the agent up
        # front: an older client advertises fewer ops, and a partial advert
        # leaves a set as ``None`` (unknown).
        payload["client_version"] = redact(outcome.capabilities.client_version)
        result_ops = outcome.capabilities.result_ops
        payload["client_result_ops"] = sorted(result_ops) if result_ops is not None else None
    if outcome.error:
        payload["error"] = redact(outcome.error)
    if outcome.error_code:
        # Symmetry with the submit path: a ``hello`` replay of a pending
        # idempotency record carries the stable ``outcome_unknown`` code, so the
        # N3 path is not left with only the confusing sentinel string.
        payload["error_code"] = outcome.error_code
    return payload
