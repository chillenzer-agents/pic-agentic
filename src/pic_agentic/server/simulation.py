# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""MCP-side ``submit_simulation`` orchestration (design sections 4.1, 8.2).

The service turns an LLM-supplied PICMI script into a ``Runner`` dump in a
disposable subprocess (the server never imports the script), wraps it in a
:class:`~pic_agentic.protocol.simulation.SimulationPayload`, embeds it in the
signed command and sends that.  It waits for the simclient's immediate
``accepted`` ack, so the LLM learns the ``sim_id`` right away; the later
lifecycle events (``simulation.submitted``/``workflow.finished``/
``simulation.failed``) are recorded as they arrive for the M2 reporting tools.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from pic_agentic.protocol.simulation import (
    LOG_STREAMS,
    MAX_INLINE_PAYLOAD_BYTES,
    ClientCapabilities,
    ControlParams,
    ResultOp,
    ResultParams,
    SimulationOp,
    SimulationPayload,
    SimulationState,
    SimulationType,
    SubmitParams,
    build_control_command,
    build_logs_command,
    build_result_command,
    build_status_command,
    build_submit_command,
    client_capability_mismatch,
    payload_wire_size,
)
from pic_agentic.rcp import Kind, RcpMessage, SenderRole, SequenceState, new_cmd_id
from pic_agentic.server.hello import AckTimeoutError, SendFn
from pic_agentic.simulation_build import BuiltSimulation, SimulationBuildError, build_runner_dump
from pic_agentic.version import local_provenance

log = logging.getLogger(__name__)

#: Signature of the injectable runner-dump builder (test seam).
RunnerDumpBuilder = Callable[..., Awaitable[BuiltSimulation]]

#: Registry states after which no further lifecycle event is expected.
TERMINAL_STATES = frozenset(
    {
        SimulationState.RESULTS_READY.value,
        SimulationState.FAILED.value,
        SimulationState.JOB_FAILED.value,
        SimulationState.CANCELLED.value,
    },
)

#: Every state name a caller may name as a wait target, plus the special
#: ``"terminal"`` alias that expands to :data:`TERMINAL_STATES`.
_KNOWN_WAIT_STATES = frozenset(state.value for state in SimulationState)

#: Default bound for :meth:`SubmitService.wait_for_state`.  Deliberately large
#: enough to cover a 15-20 min PIConGPU compile plus the queue wait; a caller
#: that needs longer calls again (the tool returns ``timed_out``, never an
#: error).
DEFAULT_WAIT_TIMEOUT_S = 1800.0

#: Floor for ``timeout_s`` (rejects zero/negative); small enough for tests.
MIN_WAIT_TIMEOUT_S = 0.1

#: Ceiling for ``timeout_s``: a single MCP tool call must not hold the
#: connection indefinitely.  Callers wait in bounded slices.
MAX_WAIT_TIMEOUT_S = 3600.0

#: Default wakeup cadence.  The wait is event-driven (it wakes as soon as a
#: lifecycle event is projected); the poll is only the ceiling for a wakeup so a
#: lost event cannot wedge the call until the deadline.
DEFAULT_WAIT_POLL_S = 2.0

#: Floor for ``poll_interval_s``.
MIN_WAIT_POLL_S = 0.05

#: Cap on the retained event log (the registry is projected from it and, on a
#: fresh start, from the signed-room backfill).
DEFAULT_EVENT_LOG_MAX = 1000

#: Cap on rows returned by ``get_events``.
MAX_EVENT_PAGE = 200

#: The ack type each request/response pull expects, so ``on_message`` can match
#: a pending pull by kind and not only by the (possibly colliding) cmd_id.
_PULL_ACK_FOR_REQUEST: dict[SimulationType, SimulationType] = {
    SimulationType.STATUS_COMMAND: SimulationType.STATUS_ACK,
    SimulationType.LOGS_COMMAND: SimulationType.LOGS_ACK,
    SimulationType.CONTROL_COMMAND: SimulationType.CONTROL_ACK,
    SimulationType.RESULT_COMMAND: SimulationType.RESULT_ACK,
}

#: Fields projected from an event/ack payload into a :class:`SimRecord` when the
#: payload actually carries a non-None value (a later event omitting the field
#: must not erase an earlier known one, e.g. the job id).
_RECORD_FIELDS = (
    "job_id",
    "slurm_state",
    "step",
    "percent",
    "walltime",
    "avg_per_step",
    "eta_s",
    "exit_code",
    "core_hours",
    "gpu_hours",
    "run_dir",
    "error",
    "error_code",
    "stage",
    "suspect",
)

#: The failure-reason subset of :data:`_RECORD_FIELDS`, projected from a submit
#: ack as well (a rejection reports its reason on the ack, not as an event).
_FAILURE_FIELDS = ("error", "error_code", "stage")


class SimRecord(BaseModel):
    """The server's projection of one simulation's lifecycle.

    The state lives in the signed room: this record is rebuilt by replaying the
    ``rcp.simulation_event`` log (and the submit acks) on server start, so it
    survives MCP-server restarts without a separate database.
    """

    model_config = ConfigDict(extra="forbid")

    sim_id: str
    cmd_id: str
    job_id: int | None = None
    state: str = ""
    slurm_state: str | None = None
    step: int | None = None
    percent: int | None = None
    walltime: str | None = None
    avg_per_step: str | None = None
    eta_s: int | None = None
    exit_code: int | None = None
    #: Actual resource usage reported on the terminal lifecycle event (gap 4).
    core_hours: float | None = None
    gpu_hours: float | None = None
    run_dir: str | None = None
    #: Human-readable failure reason reported by the simclient (ack or event).
    error: str | None = None
    #: Stable machine-readable failure code (e.g. ``unsupported``).
    error_code: str | None = None
    #: Pipeline stage the failure occurred in, when the simclient reported one.
    stage: str | None = None
    #: The "successful-but-empty" health flag: the all-zero warning text when the
    #: run completed but its only numeric artifact reads zero (F4).  None when
    #: the run is not suspect or was never probed.
    suspect: str | None = None
    last_event_type: str | None = None
    last_event_ts: str | None = None
    active: bool = True


def condense_events(
    event_log: Iterable[RcpMessage],
    *,
    sim_id: str,
    since: str | None = None,
    types: Iterable[str] | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Condense the event log into a bounded, deduplicated event list.

    Consecutive events carrying the same ``state`` collapse to a single entry
    holding the most recent payload (the bounded progress cadence therefore
    yields one ``simulation.step_finished`` row, while the full stream stays
    available via ``get_logs``).

    Args:
        event_log: The retained event log (oldest first).
        sim_id: Only events for this simulation are considered.
        since: Optional ISO-8601 lower bound; events with ``ts < since`` are
            dropped (the canonical ``Z`` timestamps sort lexicographically).
        types: Optional set of ``state`` values to keep.
        limit: Maximum number of condensed rows (capped at
            :data:`MAX_EVENT_PAGE`).

    Returns:
        The condensed payload dicts, each augmented with its envelope ``ts``,
        oldest first.

    """
    wanted = set(types) if types else None
    capped = max(0, min(limit, MAX_EVENT_PAGE))
    if capped == 0:
        return []
    condensed: list[dict[str, Any]] = []
    for message in event_log:
        if message.kind is not Kind.EVENT or message.type != SimulationType.EVENT:
            continue
        payload = message.payload
        if str(payload.get("sim_id", "")) != sim_id:
            continue
        ts = message.ts
        if since is not None and ts < since:
            continue
        state = str(payload.get("state", ""))
        if wanted is not None and state not in wanted:
            continue
        entry: dict[str, Any] = {"ts": ts, **payload}
        if condensed and condensed[-1].get("state") == state:
            condensed[-1] = entry
        else:
            condensed.append(entry)
    return condensed[-capped:]


class SubmitOutcome(BaseModel):
    """The MCP-side result of one ``submit_simulation`` exchange."""

    model_config = ConfigDict(extra="forbid")

    sim: str
    cmd_id: str
    sim_id: str
    state: str
    job_id: int | None = None
    acked: bool = False
    error: str | None = None
    error_code: str | None = None
    #: Pipeline stage the rejection occurred in, when the ack carries one.
    stage: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ok(self) -> bool:
        """Whether the command was accepted without an error."""
        return not self.error


class WaitOutcome(BaseModel):
    """The result of one bounded wait for a simulation to reach a state.

    A timeout is data (``timed_out=True``) carrying the last-known record, not
    an error: the wait is a convenience over the event stream, so "not yet"
    must never be reported as a failure.  The caller decides whether to wait
    again.
    """

    model_config = ConfigDict(extra="forbid")

    sim_id: str
    #: The state the wait resolved on (the requested target, or the last-known
    #: state on a timeout).
    state: str
    #: Whether ``state`` is one the caller asked to wait for.
    matched: bool
    #: True iff the deadline elapsed before any target was reached.
    timed_out: bool
    #: Seconds actually waited (bounded by the deadline).
    waited_s: float
    #: The target states the wait was watching (the expanded set for
    #: ``"terminal"``), so a caller can see what it was waiting for.
    target_states: list[str] = Field(default_factory=list)
    #: The terminal/requested events observed for this simulation in the
    #: registry's event log, oldest first (bounded by ``condense_events``).
    events: list[dict[str, Any]] = Field(default_factory=list)
    #: The last-known registry projection at return time (progress and failure
    #: detail); never invented -- it is exactly what the event stream says.
    last_status: dict[str, Any] = Field(default_factory=dict)
    #: A build/queue-phase note when no scheduler job id is known yet, so a
    #: timeout during the 15-20 min compile/queue window is not read as a hang.
    note: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ok(self) -> bool:
        """Whether the wait executed (a timeout is data, not a failure).

        Always True for a constructed outcome: the only failure mode -- a bad
        ``sim_id`` or target -- is rejected before a :class:`WaitOutcome` is
        built (the tool then returns ``ok: false`` with an ``error``).  Use
        ``matched``/``timed_out`` to distinguish a reached target from a
        deadline, never ``ok``.
        """
        return True


class BuiltSpec(BaseModel):
    """A dry-run build result: a Runner spec plus the size it would occupy.

    Plain data (no runtime resources), so the pydantic model carries both the
    wire spec an agenda leaf needs and the provenance/size fields that make the
    48 KiB inline limit visible to the caller before any job is submitted.
    """

    model_config = ConfigDict(extra="forbid")

    #: The wire spec (``{"sim": <pypicongpu Simulation dump>}``) for a leaf.
    spec: dict[str, Any]
    picongpu_version: str = ""
    picongpu_revision: str = ""
    schema_hash: str = ""
    #: Escaped homeserver-facing size of the payload built from ``spec``.
    wire_bytes: int
    #: The inline submission budget ``wire_bytes`` is measured against.
    inline_limit_bytes: int = MAX_INLINE_PAYLOAD_BYTES
    #: Whether ``wire_bytes`` fits the inline cap and can be submitted as-is.
    within_inline_limit: bool

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ok(self) -> bool:
        """Whether the spec was built (the build itself never fails here)."""
        return True


class SubmitService:
    """Turn PICMI scripts into commands and await their acks."""

    def __init__(
        self,
        sim: str,
        secret: str,
        *,
        picongpu_python: str = "",
        picongpu_revision: str = "",
        ack_timeout_s: float = 90.0,
        runner_dump_builder: RunnerDumpBuilder = build_runner_dump,
        event_log_max: int = DEFAULT_EVENT_LOG_MAX,
        results_root: str = "",
    ) -> None:
        """Create a service for one simulation.

        Args:
            sim: Simulation id.
            secret: Shared per-simulation RCP secret.
            picongpu_python: Interpreter with the pinned PIConGPU install.
            picongpu_revision: Pinned revision carried in the payload header.
            ack_timeout_s: Maximum wait for an ack (submit and pull).
            runner_dump_builder: Subprocess runner-dump builder (test seam).
            event_log_max: Maximum retained lifecycle events.
            results_root: Optional local mirror of a run's ``simOutput`` used
                for the contract-4 ``readable`` check; never moved from.

        """
        self.sim = sim
        self.secret = secret
        self.picongpu_python = picongpu_python
        self.picongpu_revision = picongpu_revision
        self.ack_timeout_s = ack_timeout_s
        self.runner_dump_builder = runner_dump_builder
        self.event_log_max = event_log_max
        self.results_root = results_root
        self.sequences = SequenceState()
        self._pending: dict[str, asyncio.Future[RcpMessage]] = {}
        #: Pending status/logs pulls, keyed by cmd_id, and the ack type each
        #: one expects (so a mis-routed ack cannot resolve the wrong future).
        self._pending_pull: dict[str, asyncio.Future[RcpMessage]] = {}
        self._pending_pull_kind: dict[str, SimulationType] = {}
        #: Ordered retained event log (all sims), capped *per sim_id* so one
        #: busy simulation cannot evict another's history; its sim_id-keyed
        #: projection is :attr:`registry`.  This is the single store
        #: ``get_events``/``condense_events`` read from.
        self.event_log: list[RcpMessage] = []
        self.registry: dict[str, SimRecord] = {}
        #: Latest capability set the simclient advertised over the ``hello``
        #: handshake; None until a hello runs or the client predates the probe.
        self.client_capabilities: ClientCapabilities | None = None
        #: Wakeup queues for live :meth:`wait_for_state` callers, keyed by
        #: ``sim_id``.  Each waiter owns its own ``asyncio.Event`` so a
        #: projected event wakes *every* concurrent waiter for the sim; a
        #: returning waiter removes only its own event and cannot strand a
        #: sibling (M2).
        self._waiters: dict[str, set[asyncio.Event]] = {}

    def set_client_capabilities(self, capabilities: ClientCapabilities | None) -> None:
        """Record the client capabilities learned from the ``hello`` handshake.

        Args:
            capabilities: The advertised set, or None to clear it.

        """
        self.client_capabilities = capabilities

    def capability_mismatch(
        self,
        *,
        op: ResultOp | SimulationOp | None = None,
        request_type: SimulationType | None = None,
    ) -> str | None:
        """Return a version-drift message when the client cannot handle a request.

        A proactive guard: when the client advertised its capabilities in the
        ``hello`` handshake and the requested op/type is absent, name the drift
        rather than sending a command the client answers opaquely.  An unknown
        client (no advertisement yet) yields None, so the reactive per-ack guard
        remains the backstop for a server that never ran a hello.

        Args:
            op: The result/control op about to be sent.
            request_type: The request ``type`` value about to be sent.

        Returns:
            The actionable mismatch message, or None when the request is safe.

        """
        if self.client_capabilities is None:
            return None
        type_value = request_type.value if request_type is not None else ""
        missing = self.client_capabilities.unsupported(
            op=op,
            request_type=type_value,
        )
        if missing is None:
            return None
        if type_value and missing == type_value:
            return client_capability_mismatch(self.client_capabilities, request_type=type_value)
        return client_capability_mismatch(self.client_capabilities, op=op.value if op is not None else missing)

    async def build_payload(
        self,
        script_path: Path,
        *,
        params: SubmitParams | None = None,
        cmd_id: str | None = None,
    ) -> tuple[str, SimulationPayload, RcpMessage]:
        """Build the payload and embed it in the signed command.

        Args:
            script_path: Path to the PICMI script (already resolved).
            params: Optional build/run flags.
            cmd_id: Optional command id (generated when omitted).

        Returns:
            The ``(cmd_id, payload, command)`` triple, the command signed.

        """
        command_id = cmd_id or new_cmd_id()
        built = await self.runner_dump_builder(script_path=script_path, interpreter=self.picongpu_python)
        # Provenance comes from the *child* that produced the dump, not from the
        # server process: the server may run a different interpreter (and, with
        # PIC_AGENTIC_PICONGPU_PYTHON, may not have PIConGPU at all).
        payload = SimulationPayload.build(
            picongpu_version=built.picongpu_version,
            picongpu_revision=self.picongpu_revision or built.picongpu_revision,
            schema_hash=built.schema_hash,
            runner_dump=built.runner,
        )
        payload.check_allowlist()
        seq = self.sequences.next_seq(self.sim, SenderRole.MCP_SERVER)
        command = build_submit_command(
            sim=self.sim,
            seq=seq,
            payload=payload,
            params=params,
            cmd_id=command_id,
        ).sign(self.secret)
        return command_id, payload, command

    async def build_spec(self, script_path: Path) -> BuiltSpec:
        """Build the wire spec for a PICMI script **without** submitting it.

        Runs the same disposable-subprocess builder and payload validation as
        :meth:`build_payload`, then returns the inline wire spec (``{"sim":
        ...}``) an agenda leaf or a later ``submit_spec`` call needs.  The
        payload is fully validated here (allow-list and the 48 KiB inline cap)
        so the caller learns the failure mode up front rather than at submit
        time; ``payload_wire_size`` reports the escaped size the homeserver
        would see, and ``within_inline_limit`` is False when it would not fit.

        Args:
            script_path: Path to the PICMI script (already resolved).

        Returns:
            The built wire spec plus its provenance and wire size.

        """
        built = await self.runner_dump_builder(script_path=script_path, interpreter=self.picongpu_python)
        payload = SimulationPayload.build(
            picongpu_version=built.picongpu_version,
            picongpu_revision=self.picongpu_revision or built.picongpu_revision,
            schema_hash=built.schema_hash,
            runner_dump=built.runner,
        )
        payload.check_allowlist()
        size = payload_wire_size(payload)
        return BuiltSpec(
            spec=payload.simulation,
            picongpu_version=payload.picongpu_version,
            picongpu_revision=payload.picongpu_revision,
            schema_hash=payload.schema_hash,
            wire_bytes=size,
            within_inline_limit=size <= MAX_INLINE_PAYLOAD_BYTES,
        )

    def on_message(self, message: RcpMessage) -> None:
        """Feed an inbound message; resolve pending futures and project events.

        Args:
            message: An inbound RCP message.

        """
        if not message.verify(self.secret) or message.sim != self.sim:
            return
        if message.sender_role is not SenderRole.SIMCLIENT:
            return
        cmd_id = str(message.payload.get("cmd_id", ""))
        if message.kind is Kind.EVENT and message.type == SimulationType.EVENT:
            self._append_event_log(message)
            self._project_event(message)
            return
        if message.kind is Kind.ACK and message.type in {
            SimulationType.STATUS_ACK,
            SimulationType.LOGS_ACK,
            SimulationType.CONTROL_ACK,
            SimulationType.RESULT_ACK,
        }:
            # Match the ack's *kind* to the pending request, not just its
            # cmd_id: a mis-routed ``logs_ack`` carrying a status request's
            # cmd_id must not resolve the status future.
            expected = _PULL_ACK_FOR_REQUEST.get(self._pending_pull_kind.get(cmd_id))
            if expected is not None and message.type == expected:
                future = self._pending_pull.get(cmd_id)
                if future is not None and not future.done():
                    future.set_result(message)
            else:
                log.warning(
                    "ignoring %s for pending %s request %s",
                    message.type,
                    self._pending_pull_kind.get(cmd_id),
                    cmd_id,
                )
            return
        if message.kind is Kind.ACK and message.type == SimulationType.ACK:
            self._project_ack(message)
            future = self._pending.get(cmd_id)
            if future is not None and not future.done():
                future.set_result(message)

    def ingest_backfill(self, messages: Iterable[RcpMessage]) -> None:
        """Rebuild the registry from a replayed (signed-room) message stream.

        Replaying the same events must converge to the same registry:
        :meth:`_project_event` and :meth:`_project_ack` overwrite record fields
        only when the payload carries them, and the event log is append-only, so
        a replay is idempotent in effect.

        Args:
            messages: The messages to replay, oldest first.

        """
        for message in messages:
            self.on_message(message)

    def _append_event_log(self, message: RcpMessage) -> None:
        """Append one event to the bounded, ordered event log.

        The cap is applied *per sim_id*: at most :attr:`event_log_max` events
        per simulation are retained, so a busy simulation cannot evict another
        simulation's early lifecycle history.

        Args:
            message: The event to retain.

        """
        self.event_log.append(message)
        sim_id = str(message.payload.get("sim_id", ""))
        matching = [
            index for index, entry in enumerate(self.event_log) if str(entry.payload.get("sim_id", "")) == sim_id
        ]
        overflow = len(matching) - self.event_log_max
        if overflow > 0:
            for index in reversed(matching[:overflow]):
                del self.event_log[index]

    def _project_event(self, message: RcpMessage) -> None:
        """Project one lifecycle event into the sim_id-keyed registry.

        Args:
            message: A verified ``rcp.simulation_event`` from the simclient.

        """
        payload = message.payload
        sim_id = str(payload.get("sim_id", ""))
        if not sim_id:
            return
        state = str(payload.get("state", ""))
        cmd_id = str(payload.get("cmd_id", ""))
        record = self._record_for(sim_id, cmd_id=cmd_id, ts=message.ts)
        # A replayed/old-run event (its cmd_id predates the latest run) must not
        # touch the current record.
        if cmd_id and cmd_id != record.cmd_id:
            return
        # Terminal is monotonic within a run: a replayed or out-of-order
        # non-terminal event (e.g. a late ``step_finished``) must never flip a
        # finished record back to active.  A genuinely new run under the same
        # sim_id gets a fresh (non-terminal) record from :meth:`_record_for`.
        if record.state in TERMINAL_STATES:
            return
        for field in _RECORD_FIELDS:
            value = payload.get(field)
            if value is not None:
                setattr(record, field, value)
        # The F4 health flag is only meaningful on a completed run's
        # ``results.ready`` event.  An event allow-list keeps a buggy or
        # replayed client from stamping ``suspect`` onto a still-running record
        # (the fleet/agenda guards would ignore it, but the record should never
        # hold a contradictory value).
        if state != SimulationState.RESULTS_READY.value:
            record.suspect = None
        record.state = state
        record.last_event_type = state or record.last_event_type
        record.last_event_ts = message.ts
        record.active = state not in TERMINAL_STATES
        self.registry[sim_id] = record
        self._wake_waiters(sim_id)

    def _project_ack(self, message: RcpMessage) -> None:
        """Register the simulation named by a submit ack, before its first event.

        Args:
            message: A verified submit ack from the simclient.

        """
        payload = message.payload
        sim_id = str(payload.get("sim_id", ""))
        if not sim_id:
            return
        cmd_id = str(payload.get("cmd_id", ""))
        record = self._record_for(sim_id, cmd_id=cmd_id, ts=message.ts)
        # A replayed ack from an older run must not touch the latest record.
        if cmd_id and cmd_id != record.cmd_id:
            return
        # The ack seeds a fresh record only: a late or re-delivered ack must
        # never regress a state already projected from a later event.
        state = str(payload.get("state", ""))
        if state and not record.state:
            record.state = state
            record.last_event_type = state
            record.active = state not in TERMINAL_STATES
        if payload.get("job_id") is not None:
            record.job_id = payload["job_id"]
        # A rejected submission carries its reason on the ack (not an event), so
        # project the failure fields here too.
        for field in _FAILURE_FIELDS:
            value = payload.get(field)
            if value is not None:
                setattr(record, field, value)
        if record.last_event_ts is None:
            record.last_event_ts = message.ts
        self.registry[sim_id] = record

    def _record_for(self, sim_id: str, *, cmd_id: str, ts: str | None = None) -> SimRecord:
        """Return the record for the simulation's latest run.

        A resubmission of an identical simulation yields the same ``sim_id``
        but a fresh ``cmd_id``.  Such a message starts a *new run*: the record
        is reset to the new run, rather than reporting the first run's
        ``cmd_id`` alongside the second run's ``state``/``job_id``.  The
        registry therefore keeps one (latest-run) record per ``sim_id`` and the
        event log separates runs by ``cmd_id``.

        A message from an *older* run (a backfill replay of run 1 after run 2
        has started) must not create or switch records: a new ``cmd_id`` only
        starts a run when its timestamp is not older than the record's.

        Args:
            sim_id: The simulation id.
            cmd_id: The command id naming this run.
            ts: The message timestamp (for the old-run guard), if known.

        Returns:
            The mutable record (also stored in :attr:`registry`).

        """
        record = self.registry.get(sim_id)
        if record is None:
            record = SimRecord(sim_id=sim_id, cmd_id=cmd_id)
            self.registry[sim_id] = record
        elif (
            cmd_id
            and cmd_id != record.cmd_id
            and (ts is None or record.last_event_ts is None or ts >= record.last_event_ts)
        ):
            # New run under the same sim_id: reset the run-scoped projection
            # (keep the sim_id) so it reflects the latest run only.
            record = SimRecord(sim_id=sim_id, cmd_id=cmd_id)
            self.registry[sim_id] = record
        return record

    def get(self, sim_id: str) -> SimRecord | None:
        """Return the registry record for ``sim_id``, if known.

        Returns:
            The record, or None.

        """
        return self.registry.get(sim_id)

    def list(self, *, active_only: bool = False) -> list[SimRecord]:
        """Return the registry records in registration order.

        Args:
            active_only: When True, only simulations still in a non-terminal
                state are returned.

        Returns:
            The selected records.

        """
        records = list(self.registry.values())
        if active_only:
            return [record for record in records if record.active]
        return records

    def _wake_waiters(self, sim_id: str) -> None:
        """Wake every live :meth:`wait_for_state` caller for ``sim_id``.

        Called after an event is projected so a wait returns promptly on the
        transition it is watching instead of on its poll ceiling.

        Args:
            sim_id: The simulation whose record just advanced.

        """
        for event in self._waiters.get(sim_id, set()):
            event.set()

    async def wait_for_state(
        self,
        sim_id: str,
        *,
        target_states: Iterable[str] | None = None,
        timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
        poll_interval_s: float = DEFAULT_WAIT_POLL_S,
    ) -> WaitOutcome:
        """Block until ``sim_id`` reaches a target state, or the deadline.

        The wait is **event-driven**: :meth:`_project_event` sets a wakeup
        when a lifecycle event for this simulation is projected, so the call
        returns on the terminal transition rather than on the poll cadence.
        ``poll_interval_s`` is only the ceiling on a wakeup (and bounds how
        often the pending status is re-read), so a dropped event cannot wedge
        the call past the deadline.

        This does not itself pull from the cluster: it waits on the registry the
        simclient's pushed events feed.  For the build/queue phase in which the
        simclient has not yet emitted a scheduler job id, the outcome's
        ``last_status.phase`` reads ``"building"`` (never an invented lifecycle
        state); the wait then keeps watching until the deadline.

        Args:
            sim_id: The simulation to wait on.
            target_states: State names to wait for.  Defaults to the terminal
                set (:data:`TERMINAL_STATES`); the special name ``"terminal"``
                expands to that set.  Unknown names are rejected.
            timeout_s: Bounded wait, in seconds.  Must lie within
                ``[MIN_WAIT_TIMEOUT_S, MAX_WAIT_TIMEOUT_S]``, else a
                ``ValueError`` is raised.
            poll_interval_s: Wakeup ceiling, in seconds.  Clamped to at least
                :data:`MIN_WAIT_POLL_S`.

        Returns:
            The :class:`WaitOutcome`; a deadline elapse yields
            ``timed_out=True`` with the last-known status, never an error.

        Raises:
            KeyError: If ``sim_id`` is unknown to the registry.

        """
        targets, terminal_only = _resolve_wait_targets(target_states)
        timeout = _validate_timeout(timeout_s)
        if self.get(sim_id) is None:
            msg = f"unknown simulation {sim_id!r}"
            raise KeyError(msg)
        interval = max(poll_interval_s, MIN_WAIT_POLL_S)
        wakeup = asyncio.Event()
        self._waiters.setdefault(sim_id, set()).add(wakeup)
        started = asyncio.get_running_loop().time()
        deadline = started + timeout
        try:
            while True:
                record = self.get(sim_id)
                if record is not None and record.state in targets:
                    observed = record.state
                    break
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    observed = record.state if record is not None else ""
                    return self._wait_outcome(
                        sim_id,
                        state=observed,
                        matched=False,
                        timed_out=True,
                        waited_s=asyncio.get_running_loop().time() - started,
                        targets=sorted(targets),
                        terminal_only=terminal_only,
                    )
                wakeup.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(wakeup.wait(), timeout=min(interval, remaining))
        finally:
            waiters = self._waiters.get(sim_id)
            if waiters is not None:
                waiters.discard(wakeup)
                if not waiters:
                    self._waiters.pop(sim_id, None)
        return self._wait_outcome(
            sim_id,
            state=observed,
            matched=True,
            timed_out=False,
            waited_s=asyncio.get_running_loop().time() - started,
            targets=sorted(targets),
            terminal_only=terminal_only,
        )

    def _wait_outcome(
        self,
        sim_id: str,
        *,
        state: str,
        matched: bool,
        timed_out: bool,
        waited_s: float,
        targets: list[str],
        terminal_only: bool,
    ) -> WaitOutcome:
        """Assemble a :class:`WaitOutcome` from the current registry state.

        Returns:
            The outcome, its events/status taken verbatim from the event log
            and registry projection (never invented).

        """
        record = self.get(sim_id)
        # The whole condensed history (all states), so the caller sees the path
        # to the matched state, not only the target rows.
        events = condense_events(self.event_log, sim_id=sim_id, limit=MAX_EVENT_PAGE)
        return WaitOutcome(
            sim_id=sim_id,
            state=state,
            matched=matched,
            timed_out=timed_out,
            waited_s=round(waited_s, 3),
            target_states=targets,
            events=events,
            last_status=_record_status(record) if record is not None else {},
            note=_wait_note(record, timed_out=timed_out, terminal_only=terminal_only),
        )

    def _outcome_from_ack(self, cmd_id: str, ack: RcpMessage) -> SubmitOutcome:
        return SubmitOutcome(
            sim=self.sim,
            cmd_id=cmd_id,
            sim_id=str(ack.payload.get("sim_id", "")),
            state=str(ack.payload.get("state", "")),
            job_id=ack.payload.get("job_id"),
            acked=True,
            error=ack.payload.get("error"),
            error_code=ack.payload.get("error_code"),
            stage=ack.payload.get("stage"),
        )

    async def submit(
        self,
        send: SendFn,
        script_path: Path,
        *,
        params: SubmitParams | None = None,
    ) -> SubmitOutcome:
        """Build, send and await one ``submit_simulation`` command.

        Args:
            send: Async callable ``send(RcpMessage) -> event_id``.
            script_path: Path to the PICMI script.
            params: Optional build/run flags.

        Returns:
            The outcome; ``state`` is the simclient's first ack state (normally
            ``accepted``).  A missing ack surfaces as the
            :class:`~pic_agentic.server.hello.AckTimeoutError` raised by
            :meth:`_dispatch`.

        """
        cmd_id, _payload, command = await self.build_payload(script_path, params=params)
        return await self._dispatch(send, cmd_id, command)

    async def submit_spec(
        self,
        send: SendFn,
        runner_dump: dict[str, Any],
        *,
        params: SubmitParams | None = None,
        cmd_id: str | None = None,
    ) -> SubmitOutcome:
        """Build, send and await one ``submit_simulation`` from a Runner spec.

        Unlike :meth:`submit`, the payload is built directly from an
        already-produced ``Runner.model_dump(mode="json")`` (an agenda leaf
        holds such a spec), skipping the PICMI-to-Runner subprocess builder.
        The provenance tuple is taken from this install via
        :func:`~pic_agentic.version.local_provenance` (falling back to the
        configured ``picongpu_revision`` and then to a ``provenance`` block
        carried inside the spec, so a server without PIConGPU can still carry
        the authoring revision), and the payload is still validated by
        :meth:`~pic_agentic.protocol.simulation.SimulationPayload.check_allowlist`.

        Args:
            send: Async callable ``send(RcpMessage) -> event_id``.
            runner_dump: A full runner dump (or a wire spec carrying ``sim``).
            params: Optional build/run flags.
            cmd_id: Optional stable idempotency key; a retried submission of the
                same spec re-uses it so the simclient replays its recorded ack
                instead of running the job twice.

        Returns:
            The outcome; ``state`` is the simclient's first ack state.

        """
        cmd_id, _payload, command = self._build_spec_payload(runner_dump, params=params, cmd_id=cmd_id)
        return await self._dispatch(send, cmd_id, command)

    def prepare_spec(self, runner_dump: dict[str, Any]) -> SimulationPayload:
        """Build and allow-list a Runner spec exactly as a submission would.

        The shared validation front-end for every path that turns an existing
        spec into a submit command: :meth:`_build_spec_payload` (submission) and
        the agenda's ``create_campaign`` (campaign creation) both call this, so
        a leaf is validated by the same allow-list check it will meet at submit
        time -- and rejected at creation rather than at the next tick.  No
        command is signed and no sequence is consumed.

        Args:
            runner_dump: A full runner dump (or a wire spec carrying ``sim``).

        Returns:
            The validated payload; an allow-list failure propagates from
            ``SimulationPayload.check_allowlist``.

        """
        provenance = _spec_provenance(runner_dump, self.picongpu_revision)
        payload = SimulationPayload.build(
            picongpu_version=provenance["picongpu_version"],
            picongpu_revision=provenance["picongpu_revision"],
            schema_hash=provenance["schema_hash"],
            runner_dump=runner_dump,
        )
        payload.check_allowlist()
        return payload

    def _build_spec_payload(
        self,
        runner_dump: dict[str, Any],
        *,
        params: SubmitParams | None = None,
        cmd_id: str | None = None,
    ) -> tuple[str, SimulationPayload, RcpMessage]:
        """Wrap a Runner spec directly in a signed submit command.

        Mirrors :meth:`build_payload` but skips the ``runner_dump_builder`` (the
        spec already exists), so it is synchronous.  The provenance tuple is
        resolved with :func:`_spec_provenance`, which falls back from the local
        install to a ``provenance`` mapping carried by the spec.

        Returns:
            The ``(cmd_id, payload, command)`` triple, the command signed.

        """
        command_id = cmd_id or new_cmd_id()
        payload = self.prepare_spec(runner_dump)
        seq = self.sequences.next_seq(self.sim, SenderRole.MCP_SERVER)
        command = build_submit_command(
            sim=self.sim,
            seq=seq,
            payload=payload,
            params=params,
            cmd_id=command_id,
        ).sign(self.secret)
        return command_id, payload, command

    async def _dispatch(self, send: SendFn, cmd_id: str, command: RcpMessage) -> SubmitOutcome:
        """Send one signed submit command and await its ack.

        Shared by :meth:`submit` and :meth:`submit_spec` so the pending-future
        bookkeeping lives in one place.

        Returns:
            The outcome built from the ack.

        Raises:
            AckTimeoutError: If no ack arrives within the configured wait.

        """
        future: asyncio.Future[RcpMessage] = asyncio.get_running_loop().create_future()
        self._pending[cmd_id] = future
        try:
            await send(command)
            try:
                ack = await asyncio.wait_for(future, timeout=self.ack_timeout_s)
            except TimeoutError:
                msg = f"no ack for submit command {cmd_id} within {self.ack_timeout_s}s"
                raise AckTimeoutError(msg) from None
        finally:
            self._pending.pop(cmd_id, None)
        return self._outcome_from_ack(cmd_id, ack)

    async def fetch_status(self, send: SendFn, sim_id: str) -> dict[str, Any]:
        """Send a live-status request and await its ack.

        A timeout is returned as data (``{"error": "timeout"}``), never raised:
        the caller can fall back to its registry projection.

        Args:
            send: Async callable ``send(RcpMessage) -> event_id``.
            sim_id: The simulation to query.

        Returns:
            The ``status_ack`` payload, or ``{"sim_id", "error"}`` on timeout.

        """
        capability_error = self._capability_error(
            sim_id=sim_id,
            op_value="",
            op=None,
            request_type=SimulationType.STATUS_COMMAND,
        )
        if capability_error is not None:
            return capability_error
        return await self._fetch(
            send,
            sim_id,
            lambda cmd_id: build_status_command(
                sim=self.sim,
                seq=self.sequences.next_seq(self.sim, SenderRole.MCP_SERVER),
                sim_id=sim_id,
                cmd_id=cmd_id,
            ),
        )

    async def fetch_logs(
        self,
        send: SendFn,
        sim_id: str,
        *,
        stream: str = "stdout",
        tail: int = 100,
    ) -> dict[str, Any]:
        """Send a log request and await its ack.

        Args:
            send: Async callable ``send(RcpMessage) -> event_id``.
            sim_id: The simulation to query.
            stream: One of :data:`~pic_agentic.protocol.simulation.LOG_STREAMS`.
            tail: Maximum number of trailing lines.

        Returns:
            The ``logs_ack`` payload, or ``{"sim_id", "stream", "error"}`` on
            timeout.

        Raises:
            ValueError: If ``stream`` is not a known log stream.

        """
        if stream not in LOG_STREAMS:
            msg = f"unknown log stream {stream!r}; expected one of {LOG_STREAMS}"
            raise ValueError(msg)
        capability_error = self._capability_error(
            sim_id=sim_id,
            op_value="",
            op=None,
            request_type=SimulationType.LOGS_COMMAND,
        )
        if capability_error is not None:
            return capability_error
        return await self._fetch(
            send,
            sim_id,
            lambda cmd_id: build_logs_command(
                sim=self.sim,
                seq=self.sequences.next_seq(self.sim, SenderRole.MCP_SERVER),
                sim_id=sim_id,
                stream=stream,
                tail=tail,
                cmd_id=cmd_id,
            ),
        )

    async def control(self, send: SendFn, sim_id: str, op: SimulationOp) -> dict[str, Any]:
        """Send a control request and await its ack.

        Args:
            send: Async callable ``send(RcpMessage) -> event_id``.
            sim_id: The simulation to control.
            op: The control verb (checkpoint/stop/cancel).

        Returns:
            The ``control_ack`` payload, or ``{"sim_id", "error"}`` on timeout.

        """
        params = ControlParams(sim_id=sim_id, op=op)
        capability_error = self._capability_error(
            sim_id=sim_id,
            op_value=op.value,
            op=op,
            request_type=SimulationType.CONTROL_COMMAND,
        )
        if capability_error is not None:
            return capability_error
        return await self._fetch(
            send,
            sim_id,
            lambda cmd_id: build_control_command(
                sim=self.sim,
                seq=self.sequences.next_seq(self.sim, SenderRole.MCP_SERVER),
                params=params,
                cmd_id=cmd_id,
            ),
        )

    def _capability_error(
        self,
        *,
        sim_id: str,
        op_value: str,
        op: ResultOp | SimulationOp | None,
        request_type: SimulationType,
    ) -> dict[str, Any] | None:
        """Return a soft error dict when the client cannot handle the request.

        Returns:
            The ``unsupported_by_client`` payload, or None when the request is
            safe to send (or the client never advertised its capabilities).

        """
        mismatch = self.capability_mismatch(op=op, request_type=request_type)
        if mismatch is None:
            return None
        return {
            "sim_id": sim_id,
            "op": op_value,
            "error": mismatch,
            "error_code": "unsupported_by_client",
        }

    async def fetch_result(self, send: SendFn, params: ResultParams) -> dict[str, Any]:
        """Send a results request and await its ack.

        Args:
            send: Async callable ``send(RcpMessage) -> event_id``.
            params: The validated result request (op plus its knobs).

        Returns:
            The ``result_ack`` payload, or ``{"sim_id", "error"}`` on timeout.

        """
        sim_id = params.sim_id
        capability_error = self._capability_error(
            sim_id=sim_id,
            op_value=params.op.value,
            op=params.op,
            request_type=SimulationType.RESULT_COMMAND,
        )
        if capability_error is not None:
            return capability_error
        payload = await self._fetch(
            send,
            sim_id,
            lambda cmd_id: build_result_command(
                sim=self.sim,
                seq=self.sequences.next_seq(self.sim, SenderRole.MCP_SERVER),
                params=params,
                cmd_id=cmd_id,
            ),
        )
        self._mark_readable(sim_id, payload)
        return payload

    async def fetch_manifest(self, send: SendFn, sim_id: str) -> dict[str, Any]:
        """Send a ``describe`` result request and await its ack.

        Convenience wrapper over :meth:`fetch_result` for the common manifest
        pull.

        Args:
            send: Async callable ``send(RcpMessage) -> event_id``.
            sim_id: The simulation to describe.

        Returns:
            The ``result_ack`` payload, or ``{"sim_id", "error"}`` on timeout.

        """
        return await self.fetch_result(send, ResultParams(sim_id=sim_id, op=ResultOp.DESCRIBE))

    def _mark_readable(self, sim_id: str, payload: dict[str, Any]) -> None:
        """Annotate a result ack with the contract-4 local-mirror readability.

        When ``results_root`` is configured and
        ``results_root/<sim_id>/simOutput/<path>`` exists, the corresponding
        ``ResultRef.readable`` (and ``ResultManifest.readable_local``) are set.
        This is a pure existence check: no data is moved and nothing outside
        the mirror is touched.

        Args:
            sim_id: The simulation the ack belongs to.
            payload: The ``result_ack`` payload, updated in place.

        """
        if not self.results_root:
            return
        root = Path(self.results_root).expanduser().resolve()
        base = (root / sim_id / "simOutput").resolve()
        # ``sim_id`` is LLM-supplied: refuse anything that escapes the mirror
        # root, so the readability flag cannot become an existence oracle for
        # arbitrary paths.
        if base != root and root not in base.parents:
            return
        manifest = payload.get("manifest")
        if isinstance(manifest, dict):
            manifest["readable_local"] = base.is_dir()
            files = manifest.get("files")
            if isinstance(files, list):
                for ref in files:
                    if isinstance(ref, dict) and isinstance(ref.get("path"), str):
                        ref["readable"] = self._mirror_has(base, ref["path"])
        result = payload.get("result")
        if isinstance(result, dict):
            for ref in (result, result.get("ref")):
                if isinstance(ref, dict) and isinstance(ref.get("path"), str):
                    ref["readable"] = self._mirror_has(base, ref["path"])

    @staticmethod
    def _mirror_has(base: Path, relpath: str) -> bool:
        """Whether ``relpath`` exists inside ``base`` without escaping it.

        Returns:
            True when the contained path exists.

        """
        candidate = (base / relpath).resolve()
        if candidate != base and base not in candidate.parents:
            return False
        return candidate.exists()

    async def _fetch(self, send: SendFn, sim_id: str, build: Callable[[str], RcpMessage]) -> dict[str, Any]:
        """Sign, send and await one request/response pull.

        Args:
            send: Async callable ``send(RcpMessage) -> event_id``.
            sim_id: The simulation the request names (for the timeout payload).
            build: Builder taking a fresh ``cmd_id`` and returning the command.

        Returns:
            The ack payload, or an error dict on timeout.

        """
        cmd_id = new_cmd_id()
        command = build(cmd_id).sign(self.secret)
        future: asyncio.Future[RcpMessage] = asyncio.get_running_loop().create_future()
        self._pending_pull[cmd_id] = future
        self._pending_pull_kind[cmd_id] = command.type
        try:
            await send(command)
            try:
                ack = await asyncio.wait_for(future, timeout=self.ack_timeout_s)
            except TimeoutError:
                return {"sim_id": sim_id, "error": "timeout"}
        finally:
            self._pending_pull.pop(cmd_id, None)
            self._pending_pull_kind.pop(cmd_id, None)
        return dict(ack.payload)


def _resolve_wait_targets(target_states: Iterable[str] | None) -> tuple[frozenset[str], bool]:
    """Expand and validate the requested wait target states.

    Args:
        target_states: The caller's names, or None for the terminal default.

    Returns:
        ``(targets, terminal_only)`` where ``targets`` is the expanded set and
        ``terminal_only`` records that the caller asked only for the terminal
        set (used to phrase the build/queue note).

    Raises:
        ValueError: If any name is unknown.

    """
    if target_states is None:
        return TERMINAL_STATES, True
    requested = list(target_states)
    if not requested or requested == ["terminal"]:
        return TERMINAL_STATES, True
    targets: set[str] = set()
    for name in requested:
        if name == "terminal":
            targets |= TERMINAL_STATES
            continue
        if name not in _KNOWN_WAIT_STATES:
            msg = f"unknown wait target state {name!r}"
            raise ValueError(msg)
        targets.add(name)
    return frozenset(targets), targets <= TERMINAL_STATES


def _validate_timeout(timeout_s: float) -> float:
    """Validate a requested wait timeout against the supported bounds.

    A bounded wait is the whole point of the primitive, so an out-of-range
    request is refused explicitly rather than silently clamped: the caller
    learns the real limit instead of getting a shorter wait than it asked for.

    Args:
        timeout_s: The caller's requested bound.

    Returns:
        ``timeout_s`` as a float, unchanged.

    Raises:
        ValueError: If ``timeout_s`` is outside
            ``[MIN_WAIT_TIMEOUT_S, MAX_WAIT_TIMEOUT_S]``.

    """
    value = float(timeout_s)
    if not MIN_WAIT_TIMEOUT_S <= value <= MAX_WAIT_TIMEOUT_S:
        msg = f"timeout_s must be between {MIN_WAIT_TIMEOUT_S:g} and {MAX_WAIT_TIMEOUT_S:g} seconds, got {value:g}"
        raise ValueError(msg)
    return value


def _record_status(record: SimRecord) -> dict[str, Any]:
    """Project one record into the wait's ``last_status`` block.

    A non-terminal record with no scheduler ``job_id`` is reported with
    ``phase: "building"`` (the build/prepare/queue window) so a caller is told
    "still building", not "still running"; a record with a job id is
    ``phase: "queued"``/``"running"`` from its state.  The data is the
    registry's own projection, never invented.

    Args:
        record: The latest-run record for the simulation.

    Returns:
        A JSON-safe status dict.

    """
    status: dict[str, Any] = {
        "state": record.state,
        "job_id": record.job_id,
        "slurm_state": record.slurm_state,
        "step": record.step,
        "percent": record.percent,
        "eta_s": record.eta_s,
        "error": record.error,
        "error_code": record.error_code,
        "active": record.active,
    }
    phase = _record_phase(record)
    if phase is not None:
        status["phase"] = phase
    return status


def _record_phase(record: SimRecord) -> str | None:
    """Derive a build/queue/running phase hint from a record.

    ``None`` for a terminal record (no phase is meaningful).  The hint is
    purely a relabelling of the states the event stream already reported, so it
    never asserts more than the registry knows.

    Args:
        record: The latest-run record.

    Returns:
        ``"building"``, ``"queued"``, ``"running"``, ``"finalizing"`` or None.

    """
    # If a canonical ``phase`` ever lands on the record (H4/workstream C), use
    # it rather than deriving a second opinion: the registry stays the single
    # source of truth for the phase label.
    canonical = getattr(record, "phase", None)
    if isinstance(canonical, str) and canonical:
        return canonical
    if record.state in TERMINAL_STATES:
        return None
    if record.job_id is None:
        # accepted / workflow.finished with no scheduler id yet: the compile and
        # prepare stages, before sbatch returned a job id.
        return "building"
    if record.state == SimulationState.JOB_RUNNING.value or record.slurm_state == "RUNNING":
        return "running"
    if record.state == SimulationState.JOB_FINISHED.value:
        # The scheduler job is done; the simclient is linking/probing results.
        return "finalizing"
    return "queued"


def _wait_note(record: SimRecord | None, *, timed_out: bool, terminal_only: bool) -> str | None:
    """Phrase the human-readable note for a wait outcome.

    A timeout during the build/queue window is deliberately called out: a
    PIConGPU compile can take 15-20 min and the queue wait longer, so "no
    terminal state yet" is not a hang and the caller should wait again.

    Args:
        record: The latest-run record, if known.
        timed_out: Whether the wait hit its deadline.
        terminal_only: Whether the caller waited only for terminal states.

    Returns:
        The note, or None when the outcome needs no explanation.

    """
    if not timed_out:
        return None
    if record is None:
        return "unsupported: the simulation is not in the registry"
    if record.job_id is None:
        return (
            "still in the build/queue phase (no scheduler job id yet; a PIConGPU "
            "compile can take 15-20 min); call wait_for_simulation again to keep waiting"
        )
    if not terminal_only:
        return "the requested state was not reached before the deadline; call again to keep waiting"
    return (
        f"last-known state {record.state!r}; the run is not terminal yet, "
        "call wait_for_simulation again to keep waiting"
    )


def resolve_script(picmi_script: str, *, workdir: Path) -> Path:
    """Resolve the tool's ``picmi_script`` argument to a file path.

    Args:
        picmi_script: An existing file path, or inline PICMI code.
        workdir: Directory for inline code (a temp dir on the server).

    Returns:
        The path to the PICMI script.

    """
    candidate = Path(picmi_script).expanduser()
    if "\n" not in picmi_script and candidate.is_file():
        return candidate
    workdir.mkdir(parents=True, exist_ok=True)
    # A unique name so concurrent submissions never overwrite each other.
    fd, name = tempfile.mkstemp(prefix="picmi_script-", suffix=".py", dir=str(workdir))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(picmi_script)
    return Path(name)


def _spec_provenance(runner_dump: dict[str, Any], fallback_revision: str) -> dict[str, str]:
    """Resolve the provenance tuple for a direct spec submission.

    Prefers this install's:func:`~pic_agentic.version.local_provenance`; when
    that is unavailable (a server without the pinned PIConGPU) it falls back to
    a ``provenance`` mapping carried inside the spec, then to the configured
    revision.  A spec produced on a host with PIConGPU therefore still names its
    authoring revision when submitted from a PIConGPU-less server.

    Args:
        runner_dump: The Runner spec (or a wire spec carrying ``sim``).
        fallback_revision: The configured ``picongpu_revision``.

    Returns:
        ``{"picongpu_version", "picongpu_revision", "schema_hash"}``.

    """
    local = local_provenance()
    carried = runner_dump.get("provenance")
    carried = carried if isinstance(carried, dict) else {}
    return {
        "picongpu_version": local["picongpu_version"] or str(carried.get("picongpu_version", "")),
        "picongpu_revision": fallback_revision
        or local["picongpu_revision"]
        or str(carried.get("picongpu_revision", "")),
        "schema_hash": local["schema_hash"] or str(carried.get("schema_hash", "")),
    }


__all__ = [
    "DEFAULT_EVENT_LOG_MAX",
    "DEFAULT_WAIT_POLL_S",
    "DEFAULT_WAIT_TIMEOUT_S",
    "MAX_EVENT_PAGE",
    "MAX_WAIT_TIMEOUT_S",
    "MIN_WAIT_POLL_S",
    "MIN_WAIT_TIMEOUT_S",
    "TERMINAL_STATES",
    "AckTimeoutError",
    "BuiltSpec",
    "SimRecord",
    "SimulationBuildError",
    "SubmitOutcome",
    "SubmitService",
    "WaitOutcome",
    "condense_events",
    "resolve_script",
]
