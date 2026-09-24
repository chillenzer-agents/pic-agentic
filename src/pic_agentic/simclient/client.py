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
import os
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from pic_agentic.protocol.hello import HelloType, build_hello_ack
from pic_agentic.rcp import DedupStore, Kind, RcpMessage, SenderRole, SequenceState
from pic_agentic.simclient.safety import safe_write_message
from pic_agentic.slurm import JobInfo, SlurmClient, SlurmError

if TYPE_CHECKING:
    from pic_agentic.transport.base import Transport

log = logging.getLogger(__name__)

#: Upper bound on retained idempotency records, so the durable file cannot grow
#: without limit on a long-lived message directory.
_MAX_PROCESSED = 4096


class HelloResult(BaseModel):
    """Outcome of one ``hello`` command execution."""

    job_id: int | None
    cluster_output: str | None
    error: str | None = None


class ProcessedCommand(BaseModel):
    """Durable idempotency record for one command.

    The record is written *before* execution (so a crash mid-job cannot cause a
    re-submission) and updated with the result afterwards.  Storing the result
    lets a replay re-send the same ack instead of leaving the MCP sender to time
    out on a command that already ran.
    """

    cmd_id: str
    #: False while the job is still running (or the process died mid-execution).
    completed: bool = False
    job_id: int | None = None
    cluster_output: str | None = None
    error: str | None = None

    def to_result(self) -> HelloResult:
        """Return the execution result to replay in an ack.

        Returns:
            The stored result, or a sentinel error when the outcome is unknown
            (the job started but no result was recorded before a restart).

        """
        if self.completed:
            return HelloResult(job_id=self.job_id, cluster_output=self.cluster_output, error=self.error)
        return HelloResult(job_id=self.job_id, cluster_output=None, error="already_submitted:outcome_unknown")


class SimClient:
    """Handle ``rcp.hello`` commands for one simulation on one transport."""

    def __init__(
        self,
        *,
        sim: str,
        secret: str,
        transport: Transport,
        slurm: SlurmClient,
        message_dir: Path,
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
        self.message_dir = message_dir
        self.job_wait_timeout_s = job_wait_timeout_s
        self.poll_interval_s = poll_interval_s
        self.allowed_sender_user_id = allowed_sender_user_id
        self.sequences = SequenceState()
        self.seen = DedupStore()
        #: Command ids already executed, mapped to their result.  Persisted
        #: under ``message_dir`` so a restart that backfills the room does not
        #: re-submit cluster jobs for commands it already ran, and so a replay
        #: can re-send the original ack (the transport replays the whole
        #: timeline on reconnect).  The store assumes one simclient per
        #: ``message_dir`` (the supported topology); it is not cross-process
        #: locked.  ``cluster_output`` is small for the M1 ``hello`` job and the
        #: file is capped at :data:`_MAX_PROCESSED` records.
        self._processed: dict[str, ProcessedCommand] = {}
        self._processed_path = message_dir / "processed-cmds.jsonl"
        self._load_processed()

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
        if message.type != HelloType.COMMAND:
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

    def _load_processed(self) -> None:
        """Load persisted command records, ignoring a missing or unreadable file."""
        try:
            text = self._processed_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError as exc:
            log.warning("cannot read %s: %s", self._processed_path, exc)
            return
        records: dict[str, ProcessedCommand] = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                record = ProcessedCommand.model_validate_json(line)
            except ValueError as exc:
                log.warning(
                    "ignoring malformed processed-command line in %s (%s)",
                    self._processed_path,
                    type(exc).__name__,
                )
                continue
            records[record.cmd_id] = record
        self._processed = records

    def _persist_processed(self, record: ProcessedCommand) -> None:
        """Append one record to the durable idempotency file.

        Records are append-only and last-write-wins on load, so updating an
        execution's outcome is just another append.  The file is compacted once
        it holds more than :data:`_MAX_PROCESSED` records.

        Args:
            record: The command record (pending or completed) to persist.

        """
        try:
            self._processed_path.parent.mkdir(parents=True, exist_ok=True)
            with self._processed_path.open("a", encoding="utf-8") as handle:
                handle.write(record.model_dump_json() + "\n")
        except OSError as exc:
            log.warning("cannot persist processed id %s: %s", record.cmd_id, exc)
            return
        self._processed[record.cmd_id] = record
        if len(self._processed) > _MAX_PROCESSED:
            self._rewrite_processed()

    def _rewrite_processed(self) -> None:
        """Rewrite the idempotency file with only the most recent records."""
        recent = list(self._processed.values())[-_MAX_PROCESSED:]
        self._processed = {record.cmd_id: record for record in recent}
        tmp = self._processed_path.with_suffix(self._processed_path.suffix + ".tmp")
        try:
            with os.fdopen(os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600), "w", encoding="utf-8") as handle:
                for record in recent:
                    handle.write(record.model_dump_json() + "\n")
            tmp.replace(self._processed_path)
        except OSError as exc:
            log.warning("cannot compact %s: %s", self._processed_path, exc)

    async def _handle_hello(self, message: RcpMessage) -> RcpMessage | None:
        cmd_id = str(message.payload.get("cmd_id", ""))
        # Idempotency: a re-sent or backfilled command carries the same cmd_id;
        # do not submit a second job.  Re-ack the stored result so a sender
        # whose original ack was lost does not block until its timeout.
        if cmd_id and cmd_id in self._processed:
            log.info("re-acking already-processed hello command %s", cmd_id)
            ack = self._build_ack(message, cmd_id=cmd_id, result=self._processed[cmd_id].to_result())
            await self.transport.send(ack)
            return ack
        if cmd_id:
            # Persist *before* executing so a crash mid-job cannot resubmit.
            self._persist_processed(ProcessedCommand(cmd_id=cmd_id))
        result = await self._execute_hello(message, cmd_id)
        ack = self._build_ack(message, cmd_id=cmd_id, result=result)
        if cmd_id:
            self._persist_processed(
                ProcessedCommand(
                    cmd_id=cmd_id,
                    completed=True,
                    job_id=result.job_id,
                    cluster_output=result.cluster_output,
                    error=result.error,
                )
            )
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
