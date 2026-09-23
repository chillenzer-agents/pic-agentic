# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""MCP-side ``hello`` orchestration (design sections 4.1, 8.1).

The service is transport-agnostic so it can be driven by Matrix in production
and by the in-memory transport in tests.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import BaseModel, computed_field

from pic_agentic.protocol.hello import HELLO_ACK, build_hello_command
from pic_agentic.rcp import Kind, RcpMessage, SenderRole, SequenceState, new_cmd_id

#: Async sender signature used to dispatch one RCP message.
SendFn = Callable[[RcpMessage], Awaitable[str]]


class HelloOutcome(BaseModel):
    """The MCP-side result of one ``hello`` exchange."""

    sim: str
    cmd_id: str
    job_id: int | None
    cluster_output: str | None
    acked: bool
    error: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ok(self) -> bool:
        """Whether the exchange completed without an error."""
        return not self.error


class AckTimeoutError(RuntimeError):
    """Raised when no ack arrives within the configured wait."""


class HelloService:
    """Issue ``hello`` commands and await their acks."""

    def __init__(
        self,
        sim: str,
        secret: str,
        message_dir: str,
        *,
        ack_timeout_s: float = 90.0,
        resend_once: bool = True,
    ) -> None:
        """Create a service for one simulation.

        Args:
            sim: Simulation id.
            secret: Shared per-simulation RCP secret.
            message_dir: Shared-filesystem base directory for payload files.
            ack_timeout_s: Maximum wait for an ack per attempt.
            resend_once: Whether to re-send the command once on timeout.

        """
        self.sim = sim
        self.secret = secret
        self.message_dir = Path(message_dir)
        self.ack_timeout_s = ack_timeout_s
        self.resend_once = resend_once
        self.sequences = SequenceState()
        self._pending: dict[str, asyncio.Future[RcpMessage]] = {}

    def message_path_for(self, cmd_id: str) -> str:
        """Return the server-generated path for a command's message file.

        Args:
            cmd_id: The command id.

        Returns:
            An absolute path with a safe charset (design section 6.4).

        """
        return str(self.message_dir / "msg" / f"{self.sim}-{cmd_id}.txt")

    def build_command(self, message: str, *, cmd_id: str | None = None) -> RcpMessage:
        """Build and sign a ``hello`` command.

        Args:
            message: The LLM-supplied message text.
            cmd_id: Optional command id (generated when omitted).

        Returns:
            The signed ``rcp.hello`` command.

        """
        command_id = cmd_id or new_cmd_id()
        seq = self.sequences.next_seq(self.sim, SenderRole.MCP_SERVER)
        return build_hello_command(
            sim=self.sim,
            seq=seq,
            message=message,
            message_path=self.message_path_for(command_id),
            cmd_id=command_id,
        ).sign(self.secret)

    def on_message(self, message: RcpMessage) -> None:
        """Feed an inbound message; resolve its pending ack future.

        Only signed acks from the simulation-side client count.  In particular
        the Matrix transport echoes our own outbound commands back through
        ``receive()``, and those must never resolve the future.

        Args:
            message: An inbound RCP message.

        """
        if message.kind is not Kind.ACK:
            return
        if message.type != HELLO_ACK:
            return
        if message.sender_role is not SenderRole.SIMCLIENT:
            return
        if message.sim != self.sim or not message.verify(self.secret):
            return
        cmd_id = str(message.payload.get("cmd_id", ""))
        future = self._pending.get(cmd_id)
        if future is not None and not future.done():
            future.set_result(message)

    async def hello(self, send: SendFn, message: str = "Hello World") -> HelloOutcome:
        """Send a ``hello`` command and wait for its ack.

        Args:
            send: Async callable ``send(RcpMessage) -> event_id``.
            message: The LLM-supplied message text.

        Returns:
            The outcome of the exchange.

        Raises:
            AckTimeoutError: If no ack arrives within the configured wait.

        """
        command = self.build_command(message)
        cmd_id = str(command.payload["cmd_id"])
        attempts = 2 if self.resend_once else 1
        ack: RcpMessage | None = None
        try:
            for attempt in range(attempts):
                # A fresh future per attempt: wait_for cancels the future it
                # waits on, so it must not be reused across a re-send.  The
                # pending map is updated so a late ack can still resolve the
                # current attempt.
                future: asyncio.Future[RcpMessage] = asyncio.get_running_loop().create_future()
                self._pending[cmd_id] = future
                await send(command)
                try:
                    ack = await asyncio.wait_for(future, timeout=self.ack_timeout_s)
                    break
                except TimeoutError:
                    if attempt == attempts - 1:
                        msg = f"no ack for command {cmd_id} within {self.ack_timeout_s}s"
                        raise AckTimeoutError(msg) from None
        finally:
            self._pending.pop(cmd_id, None)
        assert ack is not None
        return HelloOutcome(
            sim=self.sim,
            cmd_id=cmd_id,
            job_id=ack.payload.get("job_id"),
            cluster_output=ack.payload.get("cluster_output"),
            acked=True,
            error=ack.payload.get("error"),
        )


def default_message_dir() -> str:
    """Return the default shared-filesystem message directory.

    Returns:
        ``$PIC_AGENTIC_MESSAGE_DIR`` or a per-user default below ``$HOME``.

    """
    return os.environ.get("PIC_AGENTIC_MESSAGE_DIR", str(Path.home() / ".local/share/pic-agentic/shared"))
