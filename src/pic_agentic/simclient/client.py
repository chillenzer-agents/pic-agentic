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
import contextlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from pic_agentic import __version__ as _package_version
from pic_agentic.protocol.hello import HelloType, build_hello_ack
from pic_agentic.protocol.simulation import (
    CONTROL_REQUIRES_RUNNING,
    CONTROL_SIGNAL,
    PAYLOAD_KEY,
    ClientCapabilities,
    ControlParams,
    ResultOp,
    ResultParams,
    SimulationOp,
    SimulationStage,
    SimulationState,
    SimulationType,
    build_control_ack,
    build_logs_ack,
    build_result_ack,
    build_status_ack,
    build_submit_ack,
    build_submit_event,
    client_capability_mismatch,
)
from pic_agentic.rcp import DedupStore, Kind, RcpMessage, SenderRole, SequenceState
from pic_agentic.simclient.follow import JobFollower, TrackedSim
from pic_agentic.simclient.safety import safe_write_message
from pic_agentic.simclient.simulation import (
    PreparedSubmit,
    SimulationErrorCode,
    SimulationExecutionError,
    SubmitConfig,
    execute_submit,
    find_stdout_path,
    parse_payload,
    prepare_submit,
)
from pic_agentic.slurm import JobInfo, SlurmClient, SlurmError, SlurmJobState
from pic_agentic.version import local_provenance as _local_provenance

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pic_agentic.transport.base import Transport

log = logging.getLogger(__name__)

#: Upper bound on retained idempotency records, so the durable file cannot grow
#: without limit on a long-lived message directory.
_MAX_PROCESSED = 4096

#: M2 commands gated by the presence of a cluster-local ``submit_config``.
_M2_COMMANDS = frozenset(
    {
        SimulationType.COMMAND,
        SimulationType.STATUS_COMMAND,
        SimulationType.LOGS_COMMAND,
        SimulationType.CONTROL_COMMAND,
        SimulationType.RESULT_COMMAND,
    },
)

#: Default first (and reset) poll interval for the job-follow watcher, per the
#: M2b plan (30 s -> 5 min adaptive backoff); overridable per instance and via
#: ``PIC_AGENTIC_POLL_INTERVAL_S``.
DEFAULT_POLL_INTERVAL_S = 30.0

#: Default cap for the watcher's backing-off poll interval (the plan's 5 min).
DEFAULT_POLL_MAX_INTERVAL_S = 300.0

#: Default number of build/run workflows a simclient executes concurrently.  One
#: keeps the shared login node from being stampeded by simultaneous builds;
#: override with ``PIC_AGENTIC_BUILD_CONCURRENCY``.  This is an explicit gate:
#: without it the non-blocking control plane would start every build at once.
DEFAULT_BUILD_CONCURRENCY = 1

#: Environment variable naming the build concurrency (see
#: :data:`DEFAULT_BUILD_CONCURRENCY`).
BUILD_CONCURRENCY_ENV = "PIC_AGENTIC_BUILD_CONCURRENCY"

#: Default grace period (seconds) :meth:`SimClient.serve` grants an in-flight
#: build to drain on shutdown before it logs and returns.  A build runs in
#: ``asyncio.to_thread`` and cannot be cancelled mid-compile, so shutdown waits
#: for it rather than abandoning the worker thread; override with
#: ``PIC_AGENTIC_SHUTDOWN_GRACE_S`` (``0`` = do not wait at all).
DEFAULT_SHUTDOWN_GRACE_S = 120.0

#: Environment variable naming the shutdown build-drain grace (see
#: :data:`DEFAULT_SHUTDOWN_GRACE_S`).
SHUTDOWN_GRACE_ENV = "PIC_AGENTIC_SHUTDOWN_GRACE_S"

#: Assume a log line is at most this long when sizing the tail read window; a
#: longer line is still returned in full once the window is aligned to a
#: newline, but may cover fewer than ``tail`` lines.
_ASSUMED_MAX_LINE_BYTES = 4096

#: Hard cap on the bytes a single ``get_logs`` reads from the end of a stream,
#: so an unbounded PIConGPU ``stdout`` cannot OOM the simclient.
_MAX_LOG_READ_BYTES = 8 * 1024 * 1024


def derive_setup_dir(run_dir: Path | str) -> Path | None:
    """Derive a tracked sim's ``setup_dir`` from its ``run_dir``.

    ``Runner.generate`` lays a run out as ``<base>/input`` (setup) and
    ``<base>/run`` (run), so the setup directory is the run directory's sibling
    ``input``.  The derived path is only accepted when it resolves to a direct
    child of the run's parent, i.e. it can never escape ``<base>`` via a crafted
    or symlinked ``run_dir``.

    Args:
        run_dir: The tracked run directory (``<base>/run``).

    Returns:
        The resolved ``<base>/input`` path, or None when it is not a direct
        sibling of the run directory.

    """
    run = Path(run_dir)
    try:
        base = run.parent.resolve()
        setup = (run.parent / "input").resolve()
    except OSError:
        return None
    if setup.parent != base:
        return None
    return setup


def resolve_build_concurrency(raw: str | None = None) -> int:
    """Resolve the build concurrency from an argument or the environment.

    Args:
        raw: Optional explicit value (the environment is read when omitted).

    Returns:
        The configured concurrency, clamped to at least one.

    """
    value = raw if raw is not None else os.environ.get(BUILD_CONCURRENCY_ENV)
    if value is None or not str(value).strip():
        return DEFAULT_BUILD_CONCURRENCY
    try:
        parsed = int(value)
    except ValueError:
        log.warning("ignoring invalid %s=%r; using %d", BUILD_CONCURRENCY_ENV, value, DEFAULT_BUILD_CONCURRENCY)
        return DEFAULT_BUILD_CONCURRENCY
    return max(1, parsed)


def resolve_shutdown_grace(raw: str | None = None) -> float:
    """Resolve the shutdown build-drain grace from an argument or the environment.

    Args:
        raw: Optional explicit value (the environment is read when omitted).

    Returns:
        The grace period in seconds, clamped to non-negative.

    """
    value = raw if raw is not None else os.environ.get(SHUTDOWN_GRACE_ENV)
    if value is None or not str(value).strip():
        return DEFAULT_SHUTDOWN_GRACE_S
    try:
        parsed = float(value)
    except ValueError:
        log.warning("ignoring invalid %s=%r; using %.0fs", SHUTDOWN_GRACE_ENV, value, DEFAULT_SHUTDOWN_GRACE_S)
        return DEFAULT_SHUTDOWN_GRACE_S
    return max(0.0, parsed)


class HelloResult(BaseModel):
    """Outcome of one ``hello`` command execution."""

    job_id: int | None
    cluster_output: str | None
    error: str | None = None
    #: Stable machine-readable code for a non-``None`` ``error`` (e.g. the
    #: ``outcome_unknown`` of a pending idempotency record), so a caller does
    #: not have to match on the human-readable string.
    error_code: str | None = None


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
    #: For ``submit_simulation``: the payload hash this cmd_id was executed
    #: with, so an identical resend is re-acked while a *changed* payload under
    #: the same cmd_id is treated as a new simulation.
    payload_hash: str | None = None
    sim_id: str | None = None
    state: str | None = None
    #: Machine-readable code of a recorded failure (``hello`` leaves it unset).
    error_code: str | None = None

    def to_result(self) -> HelloResult:
        """Return the execution result to replay in an ack.

        Returns:
            The stored result, or a sentinel error when the outcome is unknown
            (the job started but no result was recorded before a restart).

        """
        if self.completed:
            return HelloResult(
                job_id=self.job_id,
                cluster_output=self.cluster_output,
                error=self.error,
                error_code=self.error_code,
            )
        return HelloResult(
            job_id=self.job_id,
            cluster_output=None,
            error="already_submitted:outcome_unknown",
            error_code=SimulationErrorCode.OUTCOME_UNKNOWN,
        )


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
        poll_max_interval_s: float = 300.0,
        allowed_sender_user_id: str | None = None,
        submit_config: SubmitConfig | None = None,
        control_fn: Callable[[SimulationOp, TrackedSim], Awaitable[str]] | None = None,
        build_concurrency: int | None = None,
        shutdown_grace_s: float | None = None,
    ) -> None:
        """Create a simulation-side client.

        Args:
            sim: Simulation id this client answers for.
            secret: Shared per-simulation RCP secret.
            transport: The RCP transport carrying commands and acks.
            slurm: The SLURM command layer.
            message_dir: Shared-filesystem base directory for payload files.
            job_wait_timeout_s: Maximum wait for a submitted job.
            poll_interval_s: Initial (and reset) interval between ``scontrol``
                polls in the job-follow watcher; also the ``hello`` wait loop.
            poll_max_interval_s: Cap for the watcher's backing-off interval.
            allowed_sender_user_id: Optional expected MCP-server identity.
            submit_config: Cluster-local policy for ``submit_simulation``;
                when omitted the M2 handler is disabled.
            control_fn: Optional M3 control translation ``(op, tracked) -> str``
                returning SLURM's output; ``None`` disables the control handler.
            build_concurrency: Maximum number of build/run workflows executed
                concurrently; defaults to
                :data:`DEFAULT_BUILD_CONCURRENCY` (1), overridable via
                ``PIC_AGENTIC_BUILD_CONCURRENCY``.
            shutdown_grace_s: Seconds :meth:`serve` waits for an in-flight
                build to drain on shutdown; defaults to
                :data:`DEFAULT_SHUTDOWN_GRACE_S`, overridable via
                ``PIC_AGENTIC_SHUTDOWN_GRACE_S``.

        """
        self.sim = sim
        self.secret = secret
        self.transport = transport
        self.slurm = slurm
        self.message_dir = message_dir
        self.job_wait_timeout_s = job_wait_timeout_s
        self.poll_interval_s = poll_interval_s
        self.poll_max_interval_s = poll_max_interval_s
        self.allowed_sender_user_id = allowed_sender_user_id
        self.submit_config = submit_config
        self.control_fn = control_fn
        self.sequences = SequenceState()
        self.seen = DedupStore()
        #: Capability set advertised in the ``hello`` ack (version handshake).
        #: Derived from the compiled enums, so an older deployed client simply
        #: omits a newly added op and the server can name the drift.
        self.capabilities = ClientCapabilities.current(client_version=_package_version)
        #: Per-sim follow-state and detached watcher tasks, keyed by ``sim_id``.
        self._tracked: dict[str, TrackedSim] = {}
        #: Current watcher per ``sim_id`` (the latest run).
        self._follow_tasks: dict[str, asyncio.Task[None]] = {}
        #: Every live watcher task, including a superseded one still winding
        #: down, so shutdown reaps orphans the dict slot no longer points at.
        self._follow_tasks_all: set[asyncio.Task[None]] = set()
        #: Per-``sim_id`` lock serialising the cancel-previous/replace sequence
        #: in :meth:`_start_follower`, so two concurrent completions for one
        #: ``sim_id`` (possible with build concurrency > 1) cannot interleave
        #: at the cancel ``await`` and leave two live watchers for one sim.
        self._follow_locks: dict[str, asyncio.Lock] = {}
        #: Explicit gate on concurrent build/run workflows (see
        #: :data:`DEFAULT_BUILD_CONCURRENCY`); acquired in :meth:`_run_submit`.
        self.build_concurrency = resolve_build_concurrency(
            str(build_concurrency) if build_concurrency is not None else None,
        )
        self._build_semaphore = asyncio.Semaphore(self.build_concurrency)
        #: Grace period for draining an in-flight build on shutdown (a build in
        #: ``asyncio.to_thread`` cannot be cancelled mid-compile).
        self.shutdown_grace_s = resolve_shutdown_grace(
            str(shutdown_grace_s) if shutdown_grace_s is not None else None,
        )
        #: Background tasks running a whole build+workflow+follow for one
        #: submission, keyed by ``cmd_id``; reaped on shutdown.
        self._submit_tasks: dict[str, asyncio.Task[None]] = {}
        #: Every background control-plane task (submissions and per-message
        #: dispatches), so shutdown can reap them all.
        self._background_tasks: set[asyncio.Task[None]] = set()
        #: Detached watchers are started only while :meth:`serve` owns the event
        #: loop, so a direct ``handle`` call in a test does not leave a task
        #: (and its subprocesses) running past the test.
        self._serving = False
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
        #: Bounded replay cache of control acks, keyed by
        #: ``(cmd_id, sim_id, op)``.  A control command must not be re-executed
        #: (re-signal/re-cancel) when the transport redelivers it with a fresh
        #: event id inside one process; the original ack is re-sent instead.
        #: The full key (not just cmd_id) prevents a reused id for a different
        #: simulation/op from replaying the wrong ack.  Results are read-only
        #: and therefore not cached.
        self._control_acks: dict[tuple[str, str, str], RcpMessage] = {}

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
        if message.type == HelloType.COMMAND:
            return await self._handle_hello(message)
        if message.type in _M2_COMMANDS:
            if self.submit_config is None:
                return await self._reject_m2(message, error="rejected_by_policy")
            return await self._dispatch_m2(message)
        # Unknown request type: the deployed client predates this operation.
        return await self._reject_unsupported(message, request_type=str(message.type))

    async def _dispatch_m2(self, message: RcpMessage) -> RcpMessage:
        """Dispatch one of the M2 commands to its handler.

        Returns:
            The signed acknowledgement that was sent.

        """
        if message.type == SimulationType.COMMAND:
            return await self._handle_submit(message)
        if message.type == SimulationType.STATUS_COMMAND:
            return await self._handle_status(message)
        if message.type == SimulationType.CONTROL_COMMAND:
            return await self._handle_control(message)
        if message.type == SimulationType.RESULT_COMMAND:
            return await self._handle_result(message)
        if message.type == SimulationType.LOGS_COMMAND:
            return await self._handle_logs(message)
        # A member of ``_M2_COMMANDS`` with no handler: a newer server sent a
        # command this client does not implement.
        return await self._reject_unsupported(message, request_type=str(message.type))

    async def _reject_unsupported(self, message: RcpMessage, *, op: str = "", request_type: str = "") -> RcpMessage:
        """Answer a request whose op/type this client does not implement.

        The cluster client is older than the MCP server: name the missing
        capability and answer in the request's own ack shape so the server
        reports the drift rather than an opaque policy rejection.

        Returns:
            The signed acknowledgement that was sent.

        """
        error = client_capability_mismatch(self.capabilities, op=op, request_type=request_type)
        code = SimulationErrorCode.UNSUPPORTED_BY_CLIENT
        if message.type == SimulationType.RESULT_COMMAND:
            return await self._send(self._build_result_rejection(message, error=error, error_code=code))
        if message.type == SimulationType.CONTROL_COMMAND:
            return await self._send(self._build_control_rejection(message, error=error, error_code=code))
        return await self._reject_pull(message, error=error, error_code=code)

    async def _send(self, ack: RcpMessage) -> RcpMessage:
        """Send one ack and return it.

        Returns:
            The ack that was sent.

        """
        await self.transport.send(ack)
        return ack

    async def _reject_m2(
        self,
        message: RcpMessage,
        *,
        error: str,
        error_code: SimulationErrorCode = SimulationErrorCode.REJECTED,
    ) -> RcpMessage:
        """Reject an M2 command when the cluster-local handler is disabled.

        Returns:
            The signed acknowledgement that was sent.

        """
        if message.type == SimulationType.COMMAND:
            return await self._reject_submit(message, error=error, error_code=error_code)
        return await self._reject_pull(message, error=error, error_code=error_code)

    async def _reject_pull(
        self,
        message: RcpMessage,
        *,
        error: str,
        error_code: SimulationErrorCode = SimulationErrorCode.REJECTED,
    ) -> RcpMessage:
        """Send a status/logs/control-shaped rejection.

        Returns:
            The signed acknowledgement that was sent.

        """
        if message.type == SimulationType.LOGS_COMMAND:
            ack = self._build_logs_ack(
                message,
                cmd_id=str(message.payload.get("cmd_id", "")),
                sim_id=str(message.payload.get("sim_id", "")),
                stream=str(message.payload.get("stream", "stdout")),
                lines=[],
                total_lines=0,
                error=error,
                error_code=error_code,
            )
        elif message.type == SimulationType.CONTROL_COMMAND:
            ack = self._build_control_rejection(message, error=error, error_code=error_code)
        elif message.type == SimulationType.RESULT_COMMAND:
            ack = self._build_result_rejection(message, error=error, error_code=error_code)
        else:
            ack = self._build_status_ack(
                message,
                cmd_id=str(message.payload.get("cmd_id", "")),
                sim_id=str(message.payload.get("sim_id", "")),
                state=SimulationState.FAILED.value,
                error=error,
                error_code=error_code,
            )
        await self.transport.send(ack)
        return ack

    def _build_control_rejection(
        self,
        message: RcpMessage,
        *,
        error: str,
        error_code: SimulationErrorCode = SimulationErrorCode.REJECTED,
    ) -> RcpMessage:
        """Build a control-shaped rejection from a (possibly invalid) payload.

        The payload may not parse as :class:`ControlParams`; the fields are read
        defensively so a malformed request still gets a control ack rather than a
        hello ack.

        Returns:
            The signed ``control_ack``.

        """
        try:
            op = SimulationOp(str(message.payload.get("op", "")))
        except ValueError:
            op = SimulationOp.CHECKPOINT
        return self._build_control_ack(
            message,
            cmd_id=str(message.payload.get("cmd_id", "")),
            sim_id=str(message.payload.get("sim_id", "")),
            op=op,
            ok=False,
            error=error,
            error_code=error_code,
        )

    def _build_result_rejection(
        self,
        message: RcpMessage,
        *,
        error: str,
        error_code: SimulationErrorCode = SimulationErrorCode.REJECTED,
    ) -> RcpMessage:
        """Build a result-shaped rejection from a (possibly invalid) payload.

        The payload may not parse as :class:`ResultParams`; the fields are read
        defensively so a malformed request still gets a ``result_ack`` rather
        than a hello ack.

        Returns:
            The signed ``result_ack``.

        """
        try:
            op = ResultOp(str(message.payload.get("op", "")))
        except ValueError:
            op = ResultOp.DESCRIBE
        return self._build_result_ack(
            message,
            cmd_id=str(message.payload.get("cmd_id", "")),
            sim_id=str(message.payload.get("sim_id", "")),
            op=op,
            error=error,
            error_code=error_code,
        )

    async def _reject_submit(
        self,
        message: RcpMessage,
        *,
        error: str,
        sim_id: str = "",
        cmd_id: str = "",
        error_code: SimulationErrorCode = SimulationErrorCode.REJECTED,
    ) -> RcpMessage:
        """Send a submit-shaped rejection (never a ``hello_ack``).

        Returns:
            The signed acknowledgement that was sent.

        """
        header = message.payload.get("header")
        header = header if isinstance(header, dict) else {}
        ack = self._build_submit_ack(
            message,
            cmd_id=cmd_id or str(message.payload.get("cmd_id", "")),
            sim_id=sim_id or str(header.get("sim_id", "")),
            state=SimulationState.FAILED.value,
            job_id=None,
            error=error,
            error_code=error_code,
        )
        await self.transport.send(ack)
        return ack

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

    async def _submit_replay_ack(
        self,
        message: RcpMessage,
        existing: ProcessedCommand,
        *,
        cmd_id: str,
        sim_id: str,
        payload_hash: str,
    ) -> RcpMessage | None:
        """Handle an already-seen ``cmd_id``.

        Returns:
            The signed ack to re-send, or None when the command is genuinely new
            and should be executed.

        """
        if existing.payload_hash and payload_hash and existing.payload_hash != payload_hash:
            # A *different* payload under an already-executed cmd_id is a sender
            # bug or an attack: executing it would also overwrite the stored
            # record, so the original could later be re-run.  Reject and keep
            # the original record; a real resubmission always gets a new cmd_id.
            log.warning("rejecting submit command %s with a changed payload", cmd_id)
            ack = self._build_submit_ack(
                message,
                cmd_id=cmd_id,
                sim_id=sim_id,
                state=SimulationState.FAILED.value,
                job_id=None,
                error="cmd_id_conflict: a different payload was already submitted under this cmd_id",
                error_code=SimulationErrorCode.REJECTED,
            )
        elif existing.payload_hash == payload_hash and payload_hash:
            log.info("re-acking already-processed submit command %s", cmd_id)
            # A pending record means the process died mid-build: report that
            # honestly instead of claiming acceptance (mirrors the hello path's
            # already_submitted:outcome_unknown).  A finished record re-sends its
            # stored terminal state and error.
            if existing.completed:
                state = existing.state or SimulationState.FAILED.value
                error: str | None = existing.error
                code: str | None = existing.error_code
            else:
                state = SimulationState.FAILED.value
                error = "already_submitted:outcome_unknown"
                code = SimulationErrorCode.OUTCOME_UNKNOWN
            ack = self._build_submit_ack(
                message,
                cmd_id=cmd_id,
                sim_id=existing.sim_id or sim_id,
                state=state,
                job_id=existing.job_id,
                error=error,
                error_code=code,
            )
        else:
            return None
        await self.transport.send(ack)
        return ack

    async def _prepare_or_reject(
        self,
        message: RcpMessage,
        *,
        header: dict,
        payload_hash: str,
        cmd_id: str,
        sim_id: str,
    ) -> PreparedSubmit | RcpMessage:
        """Validate a new submit command, or send and return a rejection ack.

        Returns:
            The prepared submission, or the rejection ack that was sent.

        """
        raw = message.payload.get(PAYLOAD_KEY)
        if not isinstance(raw, str):
            return await self._reject_submit(message, error="payload_missing", sim_id=sim_id, cmd_id=cmd_id)
        try:
            body = parse_payload(raw)
            return prepare_submit(
                body=body,
                header=header,
                params=message.payload.get("params"),
                config=self.submit_config,
                local_provenance=_local_provenance(),
                token=cmd_id or payload_hash,
            )
        except SimulationExecutionError as exc:
            ack = self._build_submit_ack(
                message,
                cmd_id=cmd_id,
                sim_id=sim_id,
                state=SimulationState.FAILED.value,
                job_id=None,
                error=str(exc),
                error_code=exc.code,
            )
            if cmd_id:
                self._persist_processed(
                    ProcessedCommand(
                        cmd_id=cmd_id,
                        completed=True,
                        payload_hash=payload_hash,
                        sim_id=sim_id,
                        state=SimulationState.FAILED.value,
                        error=str(exc),
                        error_code=exc.code,
                    )
                )
            await self.transport.send(ack)
            return ack

    async def _handle_submit(self, message: RcpMessage) -> RcpMessage | None:
        cmd_id = str(message.payload.get("cmd_id", ""))
        header = message.payload.get("header")
        header = header if isinstance(header, dict) else {}
        payload_hash = str(header.get("payload_hash", ""))
        sim_id = str(header.get("sim_id", ""))
        # Idempotency: same cmd_id + same payload hash is a replay (re-ack); a
        # different payload under the same cmd_id is rejected; a changed
        # simulation must use a fresh cmd_id.
        existing = self._processed.get(cmd_id) if cmd_id else None
        if existing is not None:
            replay_ack = await self._submit_replay_ack(
                message, existing, cmd_id=cmd_id, sim_id=sim_id, payload_hash=payload_hash
            )
            if replay_ack is not None:
                return replay_ack
        # A genuinely new command.  Persist *before* executing so a crash
        # mid-build cannot cause a re-run on restart.
        if cmd_id:
            self._persist_processed(ProcessedCommand(cmd_id=cmd_id, payload_hash=payload_hash, sim_id=sim_id))

        # Validate *before* accepting: a rejected command must report the reason
        # in its single ack, not as a lifecycle event for a sim that never
        # started (design section 2.2).
        prepared = await self._prepare_or_reject(
            message,
            header=header,
            payload_hash=payload_hash,
            cmd_id=cmd_id,
            sim_id=sim_id,
        )
        if isinstance(prepared, RcpMessage):
            return prepared

        sim_id = prepared.payload.sim_id
        # First ack: accepted (coarse; per-stage acks wait for upstream #55).
        # Emitted on the receive path so a multi-minute build never delays the
        # control plane; the build itself runs as a background task below.
        accepted = self._build_submit_ack(
            message,
            cmd_id=cmd_id,
            sim_id=sim_id,
            state=SimulationState.ACCEPTED.value,
            job_id=None,
        )
        await self.transport.send(accepted)
        # While serving, the build runs in a background task so the receive loop
        # keeps answering other commands.  A direct ``handle`` call (a unit
        # test, not owning an event loop) runs it inline, preserving the
        # pre-existing synchronous contract and leaving no task behind.
        if self._serving:
            self._start_submit_task(
                cmd_id=cmd_id,
                prepared=prepared,
                payload_hash=payload_hash,
                declared_sim_id=sim_id,
            )
        else:
            await self._run_submit(
                cmd_id=cmd_id,
                prepared=prepared,
                payload_hash=payload_hash,
                declared_sim_id=sim_id,
            )
        return accepted

    def _start_submit_task(
        self,
        *,
        cmd_id: str,
        prepared: PreparedSubmit,
        payload_hash: str,
        declared_sim_id: str,
    ) -> None:
        """Run one accepted submission's build/workflow/follow in the background.

        The task is tracked (``cmd_id`` for the submission slot, plus the shared
        background set) so :meth:`_cancel_submit_tasks` can reap it on shutdown.
        """
        task = asyncio.create_task(
            self._run_submit(
                cmd_id=cmd_id,
                prepared=prepared,
                payload_hash=payload_hash,
                declared_sim_id=declared_sim_id,
            ),
        )
        self._submit_tasks[cmd_id] = task
        self._background_tasks.add(task)

        def _reap(finished: asyncio.Task[None]) -> None:
            self._background_tasks.discard(finished)
            if self._submit_tasks.get(cmd_id) is finished:
                del self._submit_tasks[cmd_id]
            # ``execute_submit`` reports its own stage failures as events; any
            # other exception is a bug and must not vanish with the task.
            if not finished.cancelled() and (error := finished.exception()) is not None:
                log.error("error running accepted submit %s", cmd_id, exc_info=error)

        task.add_done_callback(_reap)

    async def _run_submit(
        self,
        *,
        cmd_id: str,
        prepared: PreparedSubmit,
        payload_hash: str,
        declared_sim_id: str,
    ) -> None:
        """Execute one accepted submission and report its outcome.

        Runs in a background task: acquires the explicit build gate, then drives
        ``execute_submit`` to completion and starts the follow watcher.  A
        failure is reported as a ``failed`` event and recorded; the accepted ack
        was already sent by :meth:`_handle_submit`.
        """
        async with self._build_semaphore:
            await self._execute_accepted_submit(
                cmd_id=cmd_id,
                prepared=prepared,
                payload_hash=payload_hash,
                declared_sim_id=declared_sim_id,
            )

    async def _execute_accepted_submit(
        self,
        *,
        cmd_id: str,
        prepared: PreparedSubmit,
        payload_hash: str,
        declared_sim_id: str,
    ) -> None:
        """Build, run and follow one already-accepted submission."""
        sim_id = declared_sim_id

        async def emit(state: SimulationState, *, job_id: int | None = None, **fields: object) -> None:
            event = self._build_submit_event(
                cmd_id=cmd_id,
                sim_id=sim_id,
                state=state,
                job_id=job_id,
                **fields,
            )
            await self.transport.send(event)

        try:
            result = await execute_submit(
                prepared=prepared,
                emit=emit,
                job_id_reader=self._read_submission_job_id,
            )
        except SimulationExecutionError as exc:
            await emit(
                SimulationState.FAILED,
                stage=exc.stage or SimulationStage.BUILD,
                error=str(exc),
                error_code=exc.code,
            )
            if cmd_id:
                self._persist_processed(
                    ProcessedCommand(
                        cmd_id=cmd_id,
                        completed=True,
                        payload_hash=payload_hash,
                        sim_id=sim_id,
                        state=SimulationState.FAILED.value,
                        error=str(exc),
                        error_code=exc.code,
                    )
                )
            return
        if cmd_id:
            self._persist_processed(
                ProcessedCommand(
                    cmd_id=cmd_id,
                    completed=True,
                    job_id=result.get("job_id"),
                    payload_hash=payload_hash,
                    sim_id=str(result.get("sim_id", sim_id)),
                    state=str(result.get("state", SimulationState.WORKFLOW_FINISHED.value)),
                )
            )
        await self._start_follower(
            cmd_id=cmd_id,
            sim_id=str(result.get("sim_id", sim_id)),
            job_id=result.get("job_id"),
            run_dir=str(result.get("run_dir", "")),
            stdout_path=result.get("stdout_path"),
            submit_system=prepared.params.submit_system,
        )

    async def _start_follower(
        self,
        *,
        cmd_id: str,
        sim_id: str,
        job_id: int | None,
        run_dir: str,
        stdout_path: str | None,
        submit_system: str,
    ) -> None:
        """Store follow-state and start the detached watcher for one simulation.

        The watcher is skipped for a submit system with neither a scheduler job
        id nor an ``sbatch`` contract (e.g. local ``bash`` execution): there is
        no job to follow and the workflow already finished synchronously.
        """
        if not self._serving or (job_id is None and submit_system != "sbatch"):
            return
        tracked = TrackedSim(
            sim_id=sim_id,
            cmd_id=cmd_id,
            job_id=job_id,
            run_dir=run_dir,
            stdout_path=stdout_path,
            submit_system=submit_system,
        )
        # A resubmission of an identical simulation yields the same ``sim_id``
        # (payload-hash prefix).  Cancel the previous watcher *before* replacing
        # it, so it cannot keep polling and emitting under its stale ``cmd_id``;
        # keep it in ``_follow_tasks_all`` until it has actually finished so
        # shutdown reaps it even after the dict slot is reused.
        #
        # The whole cancel/await/replace sequence is guarded by a per-``sim_id``
        # lock: the cancel ``await`` below suspends this coroutine, so without
        # the guard a second completion for the same ``sim_id`` (reachable with
        # build concurrency > 1) could pop the (already popped) slot, install
        # its watcher, and then be overwritten by this one -- leaving two live
        # followers emitting for one sim.
        async with self._follow_locks.setdefault(sim_id, asyncio.Lock()):
            previous = self._follow_tasks.pop(sim_id, None)
            if previous is not None:
                previous.cancel()
                self._follow_tasks_all.add(previous)
                previous.add_done_callback(self._follow_tasks_all.discard)
                # Await it so the cancelled watcher's in-flight ``scontrol``
                # subprocess is reaped before the replacement starts (the
                # follower shields its poll for exactly this reason).
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await previous
            self._tracked[sim_id] = tracked

            async def emit(
                state: SimulationState,
                *,
                job_id: int | None = None,
                **fields: object,
            ) -> None:
                event = self._build_submit_event(cmd_id=cmd_id, sim_id=sim_id, state=state, job_id=job_id, **fields)
                await self.transport.send(event)

            follower = JobFollower(
                sim=self.sim,
                emit=emit,
                tracked=tracked,
                job_info=self.slurm.job_info,
                initial_interval_s=self.poll_interval_s,
                max_interval_s=self.poll_max_interval_s,
                job_accounting=self.slurm.job_accounting,
            )
            task = asyncio.create_task(follower.run())
            self._follow_tasks[sim_id] = task
            self._follow_tasks_all.add(task)
            task.add_done_callback(self._follow_tasks_all.discard)

    async def _cancel_followers(self) -> None:
        """Stop and await every detached watcher task (idempotent).

        Cancels both the current per-``sim_id`` watchers and any superseded
        (orphaned) watcher still winding down, then awaits them all.
        """
        tasks = set(self._follow_tasks.values()) | self._follow_tasks_all
        for follower_task in tasks:
            follower_task.cancel()
        for follower_task in tasks:
            # A cancelled follower raises CancelledError; a follower that crashed
            # before cancellation may raise its stored exception.  Best-effort.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await follower_task
        self._follow_tasks.clear()
        self._follow_tasks_all.clear()

    async def _cancel_submit_tasks(self) -> None:
        """Drain/cancel background submission tasks on shutdown (idempotent).

        A build executes in ``asyncio.to_thread`` and cannot be cancelled
        mid-compile, so cancelling the awaiting coroutine would leave the
        worker thread (and the CWL subprocesses it owns) running while
        :meth:`serve` returned; the interpreter then blocks at exit joining
        the non-daemon thread.  Shutdown therefore *waits* for the in-flight
        build/run coroutines to finish, bounded by ``shutdown_grace_s``
        (:data:`DEFAULT_SHUTDOWN_GRACE_S`), and cancels them -- best-effort,
        the thread still runs -- only if the grace expires.  A partial build
        directory is left for the operator either way.

        Non-submission control-plane tasks (message dispatches) own no build
        thread and are cancelled immediately.
        """
        submits = {task for task in self._submit_tasks.values() if not task.done()}
        others = {task for task in self._background_tasks if task not in submits and not task.done()}
        for task in others:
            task.cancel()
        for task in others:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if submits:
            _done, pending = await asyncio.wait(submits, timeout=self.shutdown_grace_s)
            if pending:
                log.warning(
                    "shutdown grace (%.1fs) expired with %d build(s) still running; "
                    "cancelling the coroutines (the worker threads finish in the background)",
                    self.shutdown_grace_s,
                    len(pending),
                )
                for task in pending:
                    task.cancel()
                for task in pending:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
            # A submission normally reports its own failures as events; any
            # exception that reaches the task itself is a bug and must not
            # vanish with the task.
            for task in submits:
                if not task.cancelled() and (error := task.exception()) is not None:
                    log.error("error in an in-flight submit during shutdown", exc_info=error)
        self._submit_tasks.clear()
        self._background_tasks.clear()

    @staticmethod
    def _state_for_info(info: JobInfo, run_dir: str) -> str:
        """Map a live SLURM snapshot to a coarse lifecycle state.

        Returns:
            The :class:`SimulationState` value to report in a status ack.

        """
        if info.state is SlurmJobState.COMPLETED and info.exit_code in {None, 0}:
            if run_dir and (Path(run_dir) / "simOutput").exists():
                return SimulationState.RESULTS_READY.value
            return SimulationState.JOB_FINISHED.value
        mapping = {
            SlurmJobState.PENDING: SimulationState.SUBMITTED,
            SlurmJobState.RUNNING: SimulationState.JOB_RUNNING,
            SlurmJobState.COMPLETING: SimulationState.JOB_RUNNING,
            SlurmJobState.COMPLETED: SimulationState.JOB_FINISHED,
            SlurmJobState.FAILED: SimulationState.JOB_FAILED,
            SlurmJobState.CANCELLED: SimulationState.CANCELLED,
            SlurmJobState.TIMEOUT: SimulationState.JOB_FAILED,
            SlurmJobState.UNKNOWN: SimulationState.WORKFLOW_FINISHED,
        }
        return mapping[info.state].value

    @staticmethod
    def _probe_status_suspect(tracked: TrackedSim, state: str) -> str | None:
        """Probe a status-promoted terminal run for the empty-output flag (F4).

        A status pull does not go through :meth:`JobFollower._emit_terminal`, so
        without this the beta-4 "successful-but-empty" run would be promoted to
        ``results.ready`` through the status door with ``suspect`` unset.  The
        probe is lazy and guarded exactly like the follower's: a missing results
        engine or a probe failure yields ``None`` (not suspect) rather than
        breaking the status ack.

        Returns:
            The all-zero warning when the run is successfully empty, else None.

        """
        if state != SimulationState.RESULTS_READY.value:
            return None
        try:
            from pic_agentic.results import probe_vacuity  # ruff: ignore[import-outside-top-level] - lazy seam
        except ImportError:
            return None
        try:
            return probe_vacuity(tracked.sim_id, run_dir=Path(tracked.run_dir))
        except Exception:  # ruff: ignore[blind-except] - the health probe is best-effort ack data
            log.warning("vacuity probe failed for sim %s status pull", tracked.sim_id)
            return None

    async def _live_job_state(self, tracked: TrackedSim) -> SlurmJobState | None:
        """Query the tracked simulation's current SLURM state.

        Args:
            tracked: The follow-state to query.

        Returns:
            The SLURM job state, or None when there is no job id or the query
            failed (a transient failure is treated as "unknown", never raised).

        """
        if tracked.job_id is None:
            return None
        try:
            info = await self.slurm.job_info(tracked.job_id)
        except Exception as exc:  # ruff: ignore[blind-except] - a control failure is ack data
            log.warning("control state query failed for sim %s: %s", tracked.sim_id, exc)
            return None
        return info.state

    async def _parse_control_or_reject(self, message: RcpMessage) -> ControlParams | RcpMessage:
        """Parse a control request, or send and return a shaped rejection.

        An op outside this client's compiled set is a version drift and is
        answered with the actionable ``unsupported_by_client`` error; anything
        else that fails validation is an ordinary invalid-params rejection.

        Returns:
            The parsed params, or the rejection ack that was sent.

        """
        raw_op = str(message.payload.get("op", ""))
        if raw_op and not self.capabilities.supports_control_op(raw_op):
            error = client_capability_mismatch(self.capabilities, op=raw_op)
            return await self._send(
                self._build_control_rejection(
                    message,
                    error=error,
                    error_code=SimulationErrorCode.UNSUPPORTED_BY_CLIENT,
                )
            )
        try:
            return ControlParams.model_validate(
                {"sim_id": message.payload.get("sim_id"), "op": message.payload.get("op")},
            )
        except ValueError as exc:
            ack = self._build_control_rejection(message, error=f"invalid_control_params:{exc}")
            await self.transport.send(ack)
            return ack

    async def _handle_control(self, message: RcpMessage) -> RcpMessage:
        """Answer a ``control_request`` (M3).

        Guards run before the injected ``control_fn``: an unknown sim, a signal
        op on a non-``RUNNING`` job and a cancel of an already-terminal job are
        answered with a non-error-shaped ``ok=False`` ack.  Success emits a
        ``simulation.checkpoint`` event for the checkpoint op only; the stop and
        cancel transitions are observed by the watcher.

        Returns:
            The signed ``control_ack`` that was sent.

        """
        params = await self._parse_control_or_reject(message)
        if isinstance(params, RcpMessage):
            return params
        cmd_id = str(message.payload.get("cmd_id", ""))
        cache_key = (cmd_id, params.sim_id, params.op.value)
        cached = self._control_acks.get(cache_key) if cmd_id else None
        if cached is not None:
            # A redelivered control command: re-send the original ack instead of
            # signalling/cancelling twice.
            await self.transport.send(cached)
            return cached
        tracked = self._tracked.get(params.sim_id)
        gate = await self._control_gate(message, params, tracked)
        if gate is not None:
            return gate
        if self.control_fn is None:
            ack = self._build_control_rejection(message, error="control_disabled")
            await self.transport.send(ack)
            return ack
        assert tracked is not None  # narrowed by _control_gate
        try:
            slurm_reason = await self.control_fn(params.op, tracked)
        except Exception as exc:  # ruff: ignore[blind-except] - a control failure is ack data
            log.warning("control op %s failed for sim %s: %s", params.op.value, params.sim_id, exc)
            return await self._send_control_ack(
                message,
                params=params,
                tracked=tracked,
                ok=False,
                signal=CONTROL_SIGNAL[params.op],
                error=f"control_failed:{exc}",
                error_code=SimulationErrorCode.RUN_FAILED,
            )
        if params.op is SimulationOp.CHECKPOINT:
            await self.transport.send(
                self._build_submit_event(
                    cmd_id=tracked.cmd_id,
                    sim_id=tracked.sim_id,
                    state=SimulationState.CHECKPOINT,
                    job_id=tracked.job_id,
                ),
            )
        return await self._send_control_ack(
            message,
            params=params,
            tracked=tracked,
            ok=True,
            signal=CONTROL_SIGNAL[params.op],
            slurm_reason=slurm_reason,
            state=SimulationState.CHECKPOINT.value if params.op is SimulationOp.CHECKPOINT else None,
        )

    async def _control_gate(
        self,
        message: RcpMessage,
        params: ControlParams,
        tracked: TrackedSim | None,
    ) -> RcpMessage | None:
        """Apply the M3 control guards and answer a rejected request.

        Args:
            message: The inbound control command.
            params: The parsed request.
            tracked: The follow-state for ``params.sim_id``, or None.

        Returns:
            The ``ok=False`` ack that was sent, or None when the request passes
            every gate and should be translated to SLURM.

        """
        if tracked is None:
            return await self._send_control_ack(
                message,
                params=params,
                tracked=None,
                ok=False,
                error="unknown_sim",
                error_code=SimulationErrorCode.NO_RESULTS,
            )
        if tracked.job_id is None:
            # No scheduler job id (a local ``bash`` run, or an unparseable id):
            # there is nothing signalable, and ``scancel`` must never see a
            # placeholder like ``0`` (some SLURM versions treat it as "all my
            # jobs").  Applies to cancel too, not only the signal ops.
            return await self._send_control_ack(
                message,
                params=params,
                tracked=tracked,
                ok=False,
                error="no_job_id",
                error_code=SimulationErrorCode.NOT_SIGNALABLE,
            )
        state = await self._live_job_state(tracked)
        if params.op in CONTROL_REQUIRES_RUNNING and state is not SlurmJobState.RUNNING:
            return await self._send_control_ack(
                message,
                params=params,
                tracked=tracked,
                ok=False,
                state=state.value if state is not None else None,
                error_code=SimulationErrorCode.NOT_SIGNALABLE,
            )
        if params.op is SimulationOp.CANCEL and state is not None and state.terminal:
            return await self._send_control_ack(
                message,
                params=params,
                tracked=tracked,
                ok=False,
                state=state.value,
                error_code=SimulationErrorCode.NOT_TERMINAL,
            )
        return None

    async def _send_control_ack(
        self,
        message: RcpMessage,
        *,
        params: ControlParams,
        tracked: TrackedSim | None,
        ok: bool,
        signal: str | None = None,
        slurm_reason: str | None = None,
        state: str | None = None,
        error: str | None = None,
        error_code: SimulationErrorCode | None = None,
    ) -> RcpMessage:
        """Build, send and return one ``control_ack``.

        Returns:
            The signed acknowledgement that was sent.

        """
        ack = self._build_control_ack(
            message,
            cmd_id=str(message.payload.get("cmd_id", "")),
            sim_id=params.sim_id,
            op=params.op,
            ok=ok,
            job_id=tracked.job_id if tracked is not None else None,
            signal=signal,
            slurm_reason=slurm_reason,
            state=state,
            error=error,
            error_code=error_code,
        )
        await self.transport.send(ack)
        cmd_id = ack.payload.get("cmd_id", "")
        sim_id = ack.payload.get("sim_id", "")
        if cmd_id:
            self._control_acks[cmd_id, sim_id, params.op.value] = ack
            if len(self._control_acks) > _MAX_PROCESSED:
                for stale in list(self._control_acks)[:-_MAX_PROCESSED]:
                    self._control_acks.pop(stale, None)
        return ack

    async def _handle_result(self, message: RcpMessage) -> RcpMessage:
        """Answer a ``result_request`` (M3).

        The heavy lifting lives in :mod:`pic_agentic.results`, which is imported
        lazily so the control/reporting paths keep working when the results
        engine is absent.  A missing engine degrades to a
        ``reader_unavailable`` ack; the scan-only ``describe`` path does not
        move any bulk data and never touches openPMD.

        Returns:
            The signed ``result_ack`` that was sent.

        """
        raw_op = str(message.payload.get("op", ""))
        if raw_op and not self.capabilities.supports_result_op(raw_op):
            # An op this client's enum does not know: a version drift, not a
            # malformed request.  Say so instead of a generic param error.
            error = client_capability_mismatch(self.capabilities, op=raw_op)
            return await self._send(
                self._build_result_rejection(
                    message,
                    error=error,
                    error_code=SimulationErrorCode.UNSUPPORTED_BY_CLIENT,
                )
            )
        try:
            params = ResultParams.model_validate(
                {key: message.payload.get(key) for key in ResultParams.model_fields if key in message.payload},
            )
        except ValueError as exc:
            ack = self._build_result_rejection(message, error=f"invalid_result_params:{exc}")
            await self.transport.send(ack)
            return ack
        tracked = self._tracked.get(params.sim_id)
        if tracked is None:
            ack = self._build_result_ack(
                message,
                cmd_id=str(message.payload.get("cmd_id", "")),
                sim_id=params.sim_id,
                op=params.op,
                error="unknown_sim",
                error_code=SimulationErrorCode.NO_RESULTS,
            )
            await self.transport.send(ack)
            return ack
        try:
            if params.op is ResultOp.ANALYZE:
                payload = await asyncio.to_thread(self._analyze, tracked)
            else:
                from pic_agentic import results  # ruff: ignore[import-outside-top-level] - lazy optional engine

                payload = await asyncio.to_thread(
                    results.resolve_result,
                    params,
                    run_dir=Path(tracked.run_dir),
                    sim_id=params.sim_id,
                    local_root="",
                )
        except ImportError:
            payload = {"error": "reader_unavailable", "error_code": SimulationErrorCode.READER_UNAVAILABLE}
        except Exception as exc:  # ruff: ignore[blind-except] - a result failure is ack data
            log.warning("result op %s failed for sim %s: %s", params.op.value, params.sim_id, exc)
            payload = {"error": f"result_failed:{exc}", "error_code": SimulationErrorCode.RUN_FAILED}
        ack = self._build_result_ack(
            message,
            cmd_id=str(message.payload.get("cmd_id", "")),
            sim_id=params.sim_id,
            op=params.op,
            **{
                key: payload.get(key)
                for key in ("manifest", "result", "data", "data_encoding", "n_points", "stats", "error", "error_code")
            },
        )
        await self.transport.send(ack)
        return ack

    async def _handle_status(self, message: RcpMessage) -> RcpMessage:
        """Answer a ``status_request`` with a live or last-known snapshot.

        Errors are reported as data in the ack, never raised: a status pull
        must not tear down the serve loop.

        Returns:
            The signed ``status_ack`` that was sent.

        """
        cmd_id = str(message.payload.get("cmd_id", ""))
        sim_id = str(message.payload.get("sim_id", ""))
        tracked = self._tracked.get(sim_id)
        if tracked is None:
            ack = self._build_status_ack(
                message,
                cmd_id=cmd_id,
                sim_id=sim_id,
                state=SimulationState.FAILED.value,
                error="unknown_sim",
                error_code="unknown_sim",
            )
            await self.transport.send(ack)
            return ack
        state = SimulationState.WORKFLOW_FINISHED.value
        slurm_state: str | None = None
        exit_code: int | None = None
        error: str | None = None
        error_code: str | None = None
        if tracked.job_id is not None:
            try:
                info = await self.slurm.job_info(tracked.job_id)
                state = self._state_for_info(info, tracked.run_dir)
                slurm_state = info.state.value
                exit_code = info.exit_code
            except Exception as exc:  # ruff: ignore[blind-except] - a transient scontrol failure is ack data
                log.warning("status query failed for sim %s: %s", sim_id, exc)
                error = f"job_info_failed:{exc}"
                error_code = "job_info_failed"
        # A status pull can be the *only* thing that promotes a finished run to
        # ``results.ready`` (the follower missed the terminal transition).  Run
        # the same vacuity probe here so the F4 flag cannot be lost through the
        # status-pull door.
        suspect = await asyncio.to_thread(self._probe_status_suspect, tracked, state)
        ack = self._build_status_ack(
            message,
            cmd_id=cmd_id,
            sim_id=sim_id,
            state=state,
            slurm_state=slurm_state,
            job_id=tracked.job_id,
            step=tracked.last_step,
            percent=tracked.last_percent if tracked.last_percent >= 0 else None,
            walltime=tracked.last_walltime,
            avg_per_step=tracked.last_avg_per_step,
            eta_s=tracked.last_eta_s,
            exit_code=exit_code,
            error=error,
            error_code=error_code,
            suspect=suspect,
        )
        await self.transport.send(ack)
        return ack

    @staticmethod
    def _log_path(tracked: TrackedSim, stream: str) -> Path | None:
        """Resolve the on-disk file backing a log stream.

        Returns:
            The file path to read, or None when the stream has no known file.

        """
        run_dir = Path(tracked.run_dir)
        if stream == "stdout":
            if tracked.stdout_path and Path(tracked.stdout_path).is_file():
                return Path(tracked.stdout_path)
            return run_dir / "stdout"
        if stream == "stderr":
            return run_dir / "stderr"
        discovered = find_stdout_path(run_dir)
        return Path(discovered) if discovered else None

    @staticmethod
    def _read_tail(path: Path, tail: int) -> tuple[list[str], int]:
        """Read up to ``tail`` trailing lines without slurping the whole file.

        The read window is bounded by :data:`_MAX_LOG_READ_BYTES`, so a
        multi-hundred-MB PIConGPU ``stdout`` cannot OOM the simclient.  When
        the window does not reach the start of the file the first (partial)
        line is dropped, and ``total_lines`` then counts only the lines in the
        window (the true total is unknowable without reading everything).

        Args:
            path: The log file to read.
            tail: Maximum number of trailing lines to return.

        Returns:
            The ``(lines, total_lines)`` pair; ``([], 0)`` on an unreadable file.

        """
        try:
            size = path.stat().st_size
        except OSError:
            return [], 0
        window = min(_MAX_LOG_READ_BYTES, max(tail, 1) * _ASSUMED_MAX_LINE_BYTES + 1)
        start = max(0, size - window)
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                data = handle.read()
        except OSError:
            return [], 0
        if start > 0:
            newline = data.find(b"\n")
            # No newline in the window means a single line longer than the
            # whole window; report nothing rather than a bogus partial line.
            if newline < 0:
                return [], 0
            data = data[newline + 1 :]
        all_lines = data.decode("utf-8", errors="replace").splitlines()
        return (all_lines[-tail:] if tail else []), len(all_lines)

    async def _handle_logs(self, message: RcpMessage) -> RcpMessage:
        """Answer a ``logs_request`` with up to ``tail`` lines of a stream.

        A missing file is reported as an empty log (not an error).  Errors are
        ack data, never raised.

        Returns:
            The signed ``logs_ack`` that was sent.

        """
        cmd_id = str(message.payload.get("cmd_id", ""))
        sim_id = str(message.payload.get("sim_id", ""))
        stream = str(message.payload.get("stream", "stdout"))
        try:
            tail = int(message.payload.get("tail", 100))
        except (TypeError, ValueError):
            tail = 100
        tail = max(0, min(tail, 10_000))
        tracked = self._tracked.get(sim_id)
        if tracked is None:
            # The sim may be known but not yet tracked: a build in flight has
            # persisted its ``sim_id`` but starts the follower only once the job
            # is launched, so no log file exists yet.  A *rejected* submit also
            # persists a record (``state="failed"``) before validation and is
            # never tracked, so consult the record's state: reporting a
            # terminally failed sim as "not started yet" would hide the real
            # failure.  Keep ``unknown_sim`` for a sim that was never submitted.
            latest = next(
                (record for record in reversed(self._processed.values()) if record.sim_id == sim_id),
                None,
            )
            if latest is not None:
                failed = latest.state == SimulationState.FAILED.value or latest.error_code is not None
                if failed:
                    ack = self._build_logs_ack(
                        message,
                        cmd_id=cmd_id,
                        sim_id=sim_id,
                        stream=stream,
                        lines=[],
                        total_lines=0,
                        error=(
                            "the simulation failed before any log was written"
                            f"{f': {latest.error}' if latest.error else ''}"
                        ),
                        error_code=latest.error_code or SimulationState.FAILED.value,
                    )
                else:
                    ack = self._build_logs_ack(
                        message,
                        cmd_id=cmd_id,
                        sim_id=sim_id,
                        stream=stream,
                        lines=[],
                        total_lines=0,
                        error="the simulation has not started yet; logs appear after submission",
                        error_code="logs_not_available",
                    )
            else:
                ack = self._build_logs_ack(
                    message,
                    cmd_id=cmd_id,
                    sim_id=sim_id,
                    stream=stream,
                    lines=[],
                    total_lines=0,
                    error="unknown_sim",
                    error_code="unknown_sim",
                )
            await self.transport.send(ack)
            return ack
        lines: list[str] = []
        total = 0
        path = self._log_path(tracked, stream)
        if path is not None:
            lines, total = await asyncio.to_thread(self._read_tail, path, tail)
        ack = self._build_logs_ack(
            message,
            cmd_id=cmd_id,
            sim_id=sim_id,
            stream=stream,
            lines=lines,
            total_lines=total,
        )
        await self.transport.send(ack)
        return ack

    @staticmethod
    def _analyze(tracked: TrackedSim) -> dict[str, object]:
        """Compose the milestone-A analysis sections for a tracked simulation.

        The analysis engine is imported lazily (it is stdlib-only but kept off
        the control/reporting import path).  ``setup_dir`` is derived from the
        tracked ``run_dir``; a missing setup simply yields empty sections.  The
        sections ride in the ack's ``result`` field (the frozen result schema
        has no dedicated ``analysis`` field).

        Returns:
            ``{"result": <rocrate, metadata, openpmd, answer>}``.

        """
        from pic_agentic import analysis  # ruff: ignore[import-outside-top-level] - lazy optional engine

        run_dir = Path(tracked.run_dir)
        sections = analysis.analyze(
            run_dir=run_dir,
            setup_dir=derive_setup_dir(run_dir),
            output_dir=run_dir / "simOutput",
        )
        return {"result": sections}

    @staticmethod
    def _read_submission_job_id(run_dir: Path, _payload: object) -> int | None:
        """Parse the SLURM job id from ``run_dir/submission_information.txt``.

        Returns:
            The job id, or None when the file is absent or carries no id (the
            local ``bash`` submit system writes a PID instead of a job id).

        """
        try:
            text = (Path(run_dir) / "submission_information.txt").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        return SlurmClient.parse_job_id(text)

    def _build_submit_ack(
        self,
        message: RcpMessage,
        *,
        cmd_id: str,
        sim_id: str,
        state: str,
        job_id: int | None,
        error: str | None = None,
        error_code: str | None = None,
    ) -> RcpMessage:
        return build_submit_ack(
            sim=self.sim,
            seq=self.sequences.next_seq(self.sim, SenderRole.SIMCLIENT),
            cmd_id=cmd_id,
            sim_id=sim_id,
            state=SimulationState(state),
            in_reply_to=message.transport_event_id,
            job_id=job_id,
            error=error,
            error_code=error_code,
        ).sign(self.secret)

    def _build_status_ack(
        self,
        message: RcpMessage,
        *,
        cmd_id: str,
        sim_id: str,
        state: str,
        slurm_state: str | None = None,
        job_id: int | None = None,
        step: int | None = None,
        percent: int | None = None,
        walltime: str | None = None,
        avg_per_step: str | None = None,
        eta_s: int | None = None,
        exit_code: int | None = None,
        error: str | None = None,
        error_code: str | None = None,
        suspect: str | None = None,
    ) -> RcpMessage:
        return build_status_ack(
            sim=self.sim,
            seq=self.sequences.next_seq(self.sim, SenderRole.SIMCLIENT),
            cmd_id=cmd_id,
            sim_id=sim_id,
            in_reply_to=message.transport_event_id,
            state=state,
            slurm_state=slurm_state,
            job_id=job_id,
            step=step,
            percent=percent,
            walltime=walltime,
            avg_per_step=avg_per_step,
            eta_s=eta_s,
            exit_code=exit_code,
            error=error,
            error_code=error_code,
            suspect=suspect,
        ).sign(self.secret)

    def _build_logs_ack(
        self,
        message: RcpMessage,
        *,
        cmd_id: str,
        sim_id: str,
        stream: str,
        lines: list[str],
        total_lines: int,
        error: str | None = None,
        error_code: str | None = None,
    ) -> RcpMessage:
        return build_logs_ack(
            sim=self.sim,
            seq=self.sequences.next_seq(self.sim, SenderRole.SIMCLIENT),
            cmd_id=cmd_id,
            sim_id=sim_id,
            in_reply_to=message.transport_event_id,
            stream=stream,
            lines=lines,
            total_lines=total_lines,
            error=error,
            error_code=error_code,
        ).sign(self.secret)

    def _build_control_ack(
        self,
        message: RcpMessage,
        *,
        cmd_id: str,
        sim_id: str,
        op: SimulationOp,
        ok: bool,
        job_id: int | None = None,
        signal: str | None = None,
        slurm_reason: str | None = None,
        state: str | None = None,
        error: str | None = None,
        error_code: SimulationErrorCode | None = None,
    ) -> RcpMessage:
        return build_control_ack(
            sim=self.sim,
            seq=self.sequences.next_seq(self.sim, SenderRole.SIMCLIENT),
            cmd_id=cmd_id,
            sim_id=sim_id,
            op=op,
            ok=ok,
            in_reply_to=message.transport_event_id,
            job_id=job_id,
            signal=signal,
            slurm_reason=slurm_reason,
            state=state,
            error=error,
            error_code=str(error_code) if error_code is not None else None,
        ).sign(self.secret)

    def _build_result_ack(
        self,
        message: RcpMessage,
        *,
        cmd_id: str,
        sim_id: str,
        op: ResultOp,
        manifest: dict[str, object] | None = None,
        result: dict[str, object] | None = None,
        data: list[float] | str | None = None,
        data_encoding: str | None = None,
        n_points: int | None = None,
        stats: dict[str, float | int] | None = None,
        error: str | None = None,
        error_code: SimulationErrorCode | None = None,
    ) -> RcpMessage:
        """Build and sign one ``result_ack`` from a resolved payload.

        Returns:
            The signed ``result_ack``.

        """
        return build_result_ack(
            sim=self.sim,
            seq=self.sequences.next_seq(self.sim, SenderRole.SIMCLIENT),
            cmd_id=cmd_id,
            sim_id=sim_id,
            op=op,
            in_reply_to=message.transport_event_id,
            manifest=manifest,
            result=result,
            data=data,
            data_encoding=data_encoding,
            n_points=n_points,
            stats=stats,
            error=error,
            error_code=str(error_code) if error_code is not None else None,
        ).sign(self.secret)

    def _build_submit_event(
        self,
        *,
        cmd_id: str,
        sim_id: str,
        state: SimulationState,
        job_id: int | None,
        stage: SimulationStage | None = None,
        error: str | None = None,
        error_code: str | None = None,
        submit_system: str | None = None,
        results_linked: bool | None = None,
        step: int | None = None,
        percent: int | None = None,
        walltime: str | None = None,
        avg_per_step: str | None = None,
        eta_s: int | None = None,
        slurm_state: str | None = None,
        exit_code: int | None = None,
        core_hours: float | None = None,
        gpu_hours: float | None = None,
        manifest: dict[str, object] | None = None,
        suspect: str | None = None,
    ) -> RcpMessage:
        return build_submit_event(
            sim=self.sim,
            seq=self.sequences.next_seq(self.sim, SenderRole.SIMCLIENT),
            cmd_id=cmd_id,
            sim_id=sim_id,
            state=state,
            job_id=job_id,
            stage=stage,
            error=error,
            error_code=error_code,
            submit_system=submit_system,
            results_linked=results_linked,
            step=step,
            percent=percent,
            walltime=walltime,
            avg_per_step=avg_per_step,
            eta_s=eta_s,
            slurm_state=slurm_state,
            exit_code=exit_code,
            core_hours=core_hours,
            gpu_hours=gpu_hours,
            manifest=manifest,
            suspect=suspect,
        ).sign(self.secret)

    def _build_ack(self, message: RcpMessage, *, cmd_id: str, result: HelloResult) -> RcpMessage:
        return build_hello_ack(
            sim=self.sim,
            seq=self.sequences.next_seq(self.sim, SenderRole.SIMCLIENT),
            cmd_id=cmd_id,
            in_reply_to=message.transport_event_id,
            job_id=result.job_id,
            cluster_output=result.cluster_output,
            error=result.error,
            error_code=result.error_code,
            capabilities=self.capabilities,
        ).sign(self.secret)

    def _outfile_path(self, cmd_id: str) -> Path:
        outdir = self.message_dir / "out"
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir / f"hello-{cmd_id}.out"

    async def _dispatch_message(self, message: RcpMessage) -> None:
        """Validate, handle and reap one inbound message.

        Runs as a background task so a slow build cannot stall the receive loop;
        a handler failure is logged, never propagated (the loop lives on).

        Raises:
            asyncio.CancelledError: If the task is cancelled at shutdown.

        """
        try:
            await self.handle(message)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("error handling inbound RCP message")

    def _track_dispatched(self, task: asyncio.Task[None]) -> None:
        """Keep a dispatched handler task alive until it finishes."""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def serve(self) -> None:
        """Consume inbound messages until the transport closes.

        Each message is dispatched as a background task, so the receive loop
        never awaits a build: ``hello``/``status``/``logs``/``result``/control
        keep flowing while a submission is in flight.

        On exit, ``serve`` grants an in-flight build a bounded grace period to
        finish (``shutdown_grace_s``) before it logs and returns; a build runs
        in a thread and cannot be cancelled mid-compile, so draining it is the
        only way to guarantee the process does not return from ``serve`` with a
        build still writing.  Control dispatch tasks and follow watchers are
        cancelled and awaited immediately.  If the grace expires, the
        submission coroutines are cancelled and a warning is logged; the worker
        threads still running are a best-effort, non-blocking exit (the
        operator sees the warning).
        """
        self._serving = True
        try:
            async for message in self.transport.receive():
                self._track_dispatched(asyncio.create_task(self._dispatch_message(message)))
        finally:
            self._serving = False
            await self._cancel_submit_tasks()
            await self._cancel_followers()
