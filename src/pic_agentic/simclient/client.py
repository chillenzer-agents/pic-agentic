# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""The simulation-side RCP client.

Runs on the submission node, holds the cluster session, verifies inbound
commands (HMAC + sender allow-list) and executes only the fixed M1 ``hello``
command.  It has no arbitrary-shell surface.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from pic_agentic.protocol.hello import HELLO, build_hello_ack
from pic_agentic.rcp import DedupStore, Kind, RcpMessage, SenderRole, SequenceState
from pic_agentic.simclient.safety import safe_write_message
from pic_agentic.slurm import JobInfo, SlurmClient, SlurmError

if TYPE_CHECKING:
    from pic_agentic.transport.base import Transport

log = logging.getLogger(__name__)


class HelloResult(BaseModel):
    """Outcome of one ``hello`` command execution."""

    job_id: int | None
    cluster_output: str | None
    error: str | None = None


class SimClient:
    """Handle ``rcp.hello`` commands for one simulation on one transport."""

    def __init__(
        self,
        *,
        sim: str,
        secret: str,
        transport: Transport,
        slurm: SlurmClient,
        message_dir: str,
        job_wait_timeout_s: float = 60.0,
        poll_interval_s: float = 0.2,
        allowed_sender_user_id: str | None = None,
    ) -> None:
        """Create a simulation-side client.

        Args:
            sim: Simulation id this client answers for.
            secret: Shared per-simulation RCP secret.
            transport: The RCP transport carrying commands and acks.
            slurm: The SLURM command layer.
            message_dir: Shared-filesystem base directory for payload files.
            job_wait_timeout_s: Maximum wait for a submitted job.
            poll_interval_s: Delay between ``scontrol`` polls.
            allowed_sender_user_id: Optional expected MCP-server identity.

        """
        self.sim = sim
        self.secret = secret
        self.transport = transport
        self.slurm = slurm
        self.message_dir = Path(message_dir)
        self.job_wait_timeout_s = job_wait_timeout_s
        self.poll_interval_s = poll_interval_s
        self.allowed_sender_user_id = allowed_sender_user_id
        self.sequences = SequenceState()
        self.seen = DedupStore()
        self._processed_commands: set[str] = set()

    def _accepts(self, message: RcpMessage) -> bool:
        if message.sim != self.sim or message.kind is not Kind.COMMAND:
            return False
        if not message.verify(self.secret):
            log.warning("rejecting RCP message with bad signature: %s", message.type)
            return False
        # Defence in depth (design section 6.4): the registered MCP server
        # identity must match, when configured.
        if (
            self.allowed_sender_user_id
            and message.transport_sender
            and message.transport_sender != self.allowed_sender_user_id
        ):
            log.warning("rejecting command from unexpected sender %s", message.transport_sender)
            return False
        return self.seen.seen(message)

    async def handle(self, message: RcpMessage) -> RcpMessage | None:
        """Validate and dispatch one inbound message.

        Args:
            message: The inbound message.

        Returns:
            The reply ack that was sent, or None if the message was ignored.

        """
        if not self._accepts(message):
            return None
        self.sequences.observe(message.sim, message.sender_role, message.seq)
        if message.type != HELLO:
            await self._ack(message, cmd_id=message.payload.get("cmd_id"), error="rejected_by_policy")
            return None
        return await self._handle_hello(message)

    @staticmethod
    def _read_outcome(info: JobInfo, outfile: str) -> tuple[str | None, str | None]:
        """Map a terminal job state to a result pair.

        Returns:
            The ``(error, cluster_output)`` pair for the finished job.

        """
        if info.state.value == "COMPLETED":
            return None, Path(outfile).read_text(encoding="utf-8", errors="replace")
        if info.state.terminal:
            return f"job_failed:{info.state.value}", None
        return f"job_timeout:{info.state.value}", None

    async def _execute_hello(self, message: RcpMessage, cmd_id: str) -> HelloResult:
        message_path = str(message.payload.get("message_path", ""))
        content = str(message.payload.get("message", "Hello World"))
        outfile = str(self._outfile_path(cmd_id))
        result = HelloResult(job_id=None, cluster_output=None)
        try:
            safe_write_message(message_path, self.message_dir, default=content)
            result.job_id = await self.slurm.submit_wrap_cat(message_path, outfile)
            info = await self.slurm.wait_for_job(
                result.job_id,
                timeout_s=self.job_wait_timeout_s,
                interval_s=self.poll_interval_s,
            )
        except SlurmError as exc:
            result.error = f"signal_failed:{exc}" if result.job_id else f"submit_failed:{exc}"
            return result
        except Exception as exc:  # ruff: ignore[blind-except] - surfaced verbatim to the LLM
            result.error = f"unexpected:{exc}"
            return result
        result.error, result.cluster_output = self._read_outcome(info, outfile)
        return result

    async def _handle_hello(self, message: RcpMessage) -> RcpMessage | None:
        cmd_id = str(message.payload.get("cmd_id", ""))
        # Idempotency: a re-sent command carries the same cmd_id; do not
        # submit a second job.
        if cmd_id in self._processed_commands:
            log.info("ignoring re-sent hello command %s", cmd_id)
            return None
        self._processed_commands.add(cmd_id)
        result = await self._execute_hello(message, cmd_id)
        ack = self._build_ack(message, cmd_id=cmd_id, result=result)
        await self.transport.send(ack)
        return ack

    async def _ack(self, message: RcpMessage, *, cmd_id: object, error: str) -> None:
        ack = self._build_ack(
            message,
            cmd_id=str(cmd_id or ""),
            result=HelloResult(job_id=None, cluster_output=None, error=error),
        )
        await self.transport.send(ack)

    def _build_ack(self, message: RcpMessage, *, cmd_id: str, result: HelloResult) -> RcpMessage:
        return build_hello_ack(
            sim=self.sim,
            seq=self.sequences.next_seq(self.sim, SenderRole.SIMCLIENT),
            cmd_id=cmd_id,
            in_reply_to=message.transport_event_id,
            job_id=result.job_id,
            cluster_output=result.cluster_output,
            error=result.error,
        ).sign(self.secret)

    def _outfile_path(self, cmd_id: str) -> Path:
        outdir = self.message_dir / "out"
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir / f"hello-{cmd_id}.out"

    async def serve(self) -> None:
        """Consume inbound messages until the transport closes.

        Raises:
            asyncio.CancelledError: If the serving task is cancelled.

        """
        async for message in self.transport.receive():
            try:
                await self.handle(message)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("error handling inbound RCP message")
