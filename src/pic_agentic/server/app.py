# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""MCP stdio server exposing the M1 ``hello`` tool (design sections 4, 8.1)."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from pic_agentic.auth import MasTokenStore
from pic_agentic.protocol.simulation import SubmitParams
from pic_agentic.server.hello import AckTimeoutError, HelloOutcome, HelloService
from pic_agentic.server.simulation import SubmitOutcome, SubmitService, resolve_script
from pic_agentic.simclient.safety import UnsafePathError
from pic_agentic.simulation_build import SimulationBuildError
from pic_agentic.transport.matrix import MatrixTransport

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pic_agentic.config import Config
    from pic_agentic.rcp.envelope import RcpMessage

log = logging.getLogger(__name__)


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
        )
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
        # Drain anything already in the room before we start awaiting.
        for message in await self._transport.backfill():
            self._route(message)
        self._pump = asyncio.create_task(self._pump_forever())

    def _route(self, message: RcpMessage) -> None:
        # Both services filter by envelope kind/type/sim/signature, so feeding
        # every message to both is safe and keeps the routing trivial.
        self.service.on_message(message)
        self.submit_service.on_message(message)

    async def _pump_forever(self) -> None:
        if self._transport is None:
            msg = "runtime is not started"
            raise RuntimeError(msg)
        async for message in self._transport.receive():
            self._route(message)

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

    async def stop(self) -> None:
        """Cancel the pump and close the transport."""
        if self._pump:
            self._pump.cancel()
        if self._transport:
            await self._transport.close()


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
        instructions=(
            "Submit and follow PIConGPU simulations on a remote SLURM cluster. "
            "The 'hello' tool performs an end-to-end connectivity check."
        ),
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
        outcome = await runtime.hello(message)
        return _outcome_dict(runtime, outcome)

    @server.tool(
        title="Submit a PIConGPU simulation",
        description=(
            "Build a PICMI simulation script into a PyPIConGPU runner, send it "
            "through the Matrix control channel to the simulation-side client "
            "and submit it to the remote SLURM cluster. Returns the simulation "
            "id and the coarse accepted/submitted state."
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
        params = SubmitParams(
            build_jobs=build_jobs,
            build_cmake=build_cmake,
            build_preset=build_preset,
            build_force=build_force,
            cfg_file=cfg_file,
            overwrite_vars=overwrite_vars,
        )
        try:
            outcome = await runtime.submit(picmi_script, params=params)
        except (AckTimeoutError, SimulationBuildError, UnsafePathError, OSError) as exc:
            return {"ok": False, "state": "error", "error": runtime.config.redact(str(exc))}
        return _submit_outcome_dict(runtime, outcome)

    return server, runtime


def _submit_outcome_dict(runtime: HelloRuntime, outcome: SubmitOutcome) -> dict[str, Any]:
    redact = runtime.config.redact
    payload: dict[str, Any] = {
        "ok": outcome.ok,
        "sim": outcome.sim,
        "sim_id": outcome.sim_id,
        "state": outcome.state,
        "job_id": outcome.job_id,
        "acked": outcome.acked,
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
    if outcome.error:
        payload["error"] = redact(outcome.error)
    return payload
