"""MCP-side ``hello`` orchestration (design sections 4.1, 8.1).

The service is transport-agnostic so it can be driven by Matrix in production
and by the in-memory transport in tests.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

from pic_agentic.protocol.hello import HELLO_ACK, build_hello_command
from pic_agentic.rcp import Kind, RcpMessage, SenderRole, SequenceState


@dataclass
class HelloOutcome:
    ok: bool
    sim: str
    cmd_id: str
    job_id: int | None
    cluster_output: str | None
    acked: bool
    error: str | None = None


class AckTimeout(RuntimeError):
    pass


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
        self.sim = sim
        self.secret = secret
        self.message_dir = Path(message_dir)
        self.ack_timeout_s = ack_timeout_s
        self.resend_once = resend_once
        self.sequences = SequenceState()
        self._pending: dict[str, asyncio.Future[RcpMessage]] = {}

    def message_path_for(self, cmd_id: str) -> str:
        """Server-generated absolute path with a safe charset (section 6.4)."""
        return str(self.message_dir / "msg" / f"{self.sim}-{cmd_id}.txt")

    def build_command(self, message: str, *, cmd_id: str | None = None) -> RcpMessage:
        from pic_agentic.rcp import new_cmd_id

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
        """Feed inbound messages here; resolves a pending ack future.

        Only signed acks from the simulation-side client count.  In particular
        the Matrix transport echoes our own outbound commands back through
        ``receive()``, and those must never resolve the future.
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

    async def hello(self, send, message: str = "Hello World") -> HelloOutcome:
        """Send a ``hello`` command via ``send`` and wait for its ack.

        ``send`` is an async callable ``send(RcpMessage) -> event_id``.
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
                except (asyncio.TimeoutError, TimeoutError):
                    if attempt == attempts - 1:
                        raise AckTimeout(f"no ack for command {cmd_id} within {self.ack_timeout_s}s") from None
        finally:
            self._pending.pop(cmd_id, None)
        assert ack is not None
        return HelloOutcome(
            ok=not ack.payload.get("error"),
            sim=self.sim,
            cmd_id=cmd_id,
            job_id=ack.payload.get("job_id"),
            cluster_output=ack.payload.get("cluster_output"),
            acked=True,
            error=ack.payload.get("error"),
        )


def default_message_dir() -> str:
    return os.environ.get("PIC_AGENTIC_MESSAGE_DIR", str(Path.home() / ".local/share/pic-agentic/shared"))
