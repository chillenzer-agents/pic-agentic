# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""MCP stdio server exposing the M1 ``hello`` tool (design sections 4, 8.1)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from pic_agentic.server.hello import HelloOutcome, HelloService
from pic_agentic.transport.matrix import MatrixTransport

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pic_agentic.config import Config

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
        self.service = HelloService(
            sim=sim,
            secret=config.rcp_secret,
            message_dir=Path(config.message_dir),
            ack_timeout_s=config.ack_timeout_s,
        )
        self._transport: MatrixTransport | None = None
        self._pump: asyncio.Task | None = None

    async def start(self) -> None:
        """Open the Matrix transport and start pumping inbound messages."""
        config = self.config
        config.require("homeserver", "user_id", "access_token", "room_id", "rcp_secret", "message_dir")
        self._transport = MatrixTransport(
            config.homeserver,
            config.user_id,
            config.access_token,
            config.room_id,
            store_path=config.nio_store_dir or None,
        )
        # Drain anything already in the room before we start awaiting.
        for message in await self._transport.backfill():
            self.service.on_message(message)
        self._pump = asyncio.create_task(self._pump_forever())

    async def _pump_forever(self) -> None:
        if self._transport is None:
            msg = "runtime is not started"
            raise RuntimeError(msg)
        async for message in self._transport.receive():
            self.service.on_message(message)

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

    return server, runtime


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
