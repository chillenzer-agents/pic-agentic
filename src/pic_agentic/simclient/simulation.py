# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Simulation-side execution of an M2 ``submit_simulation`` command.

The handler is deliberately paranoid (design sections 6.4, 8.2): it re-validates
the inline payload, checks its byte hash and the provenance tuple against the
local install, and only then imports PIConGPU.  All cluster locations come from
local configuration, never from the payload.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pic_agentic.protocol.simulation import (
    DEFAULT_SUBMIT_SYSTEM,
    SimulationPayload,
    SimulationStage,
    SimulationState,
    SubmitParams,
    UnsupportedPayloadError,
    provenance_mismatches,
)
from pic_agentic.rcp import canonical_bytes

log = logging.getLogger(__name__)

#: Per-command directory token: the command id (or, defensively, the payload
#: hash) -- lowercase hex only, so it can never traverse or be absolute.
_TOKEN_RE = re.compile(r"^[0-9a-f]{8,64}$")


class SimulationErrorCode(StrEnum):
    """Stable machine-readable error codes reported in acks and events."""

    PATH_UNSAFE = "path_unsafe"
    PAYLOAD_INVALID = "payload_invalid"
    UNSUPPORTED = "unsupported"
    HASH_MISMATCH = "hash_mismatch"
    VERSION_MISMATCH = "version_mismatch"
    SUBMIT_SYSTEM_MISMATCH = "submit_system_mismatch"
    REJECTED = "rejected_by_policy"
    #: A submit/hello replay found a *pending* idempotency record: the command
    #: was received (so a job may already exist or even be running) but this
    #: process died before recording an outcome.  Distinct from
    #: :data:`REJECTED` because it is not a policy rejection of the payload: the
    #: exactly-once ``cmd_id`` makes a retry safe, so a caller must classify it
    #: as transient (defer and retry) rather than terminal.
    OUTCOME_UNKNOWN = "outcome_unknown"
    PICONGPU_UNAVAILABLE = "picongpu_unavailable"
    GENERATE_FAILED = "generate_failed"
    #: The CWL workflow's *build* step (the PIConGPU ``pic-build`` compile)
    #: failed: a compile error such as a CFL ``static_assert``.  Distinct from
    #: :data:`RUN_FAILED` because the run never started -- the failure is in the
    #: build that precedes submission, and its remediation (fix the setup /
    #: params) differs from a runtime crash.
    BUILD_FAILED = "build_failed"
    RUN_FAILED = "run_failed"
    #: M3 control/results error codes.
    NOT_SIGNALABLE = "not_signalable"
    NOT_TERMINAL = "not_terminal"
    READER_UNAVAILABLE = "reader_unavailable"
    NO_RESULTS = "no_results"
    RESULT_TOO_LARGE = "result_too_large"
    #: The cluster client does not recognise the request's op/type (an unknown
    #: enum member or a missing handler), i.e. the server and the deployed
    #: client are on different versions.  Naming the unsupported capability
    #: turns a silent opaque rejection into an actionable version-drift error.
    UNSUPPORTED_BY_CLIENT = "unsupported_by_client"


class SimulationExecutionError(RuntimeError):
    """A stage failure carrying a machine-readable error code."""

    def __init__(
        self,
        code: SimulationErrorCode,
        message: str,
        stage: SimulationStage | None = None,
        summary: str | None = None,
    ) -> None:
        """Create the error.

        Args:
            code: Stable error code reported in the ack/event.
            message: Human-readable detail.
            stage: Pipeline stage, when the failure happened after acceptance.
            summary: Optional short, human-readable cause extracted from a long
                raw error (e.g. the compiler line from a cwltool dump).

        """
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.summary = summary


@dataclass
class SubmitConfig:
    """Cluster-local policy for executing a submitted simulation."""

    setup_root: Path
    template_dir: str = ""
    preset: int | None = None

    def __post_init__(self) -> None:
        """Normalise the path to absolute form."""
        self.setup_root = Path(self.setup_root)


def parse_payload(raw: str) -> dict[str, Any]:
    """Parse the inline payload JSON string into a mapping.

    The payload is transported as a JSON *string* (see
    :data:`~pic_agentic.protocol.simulation.PAYLOAD_KEY`): Matrix's canonical
    JSON rejects floats in event-content objects, and the simulation has many.

    Args:
        raw: The JSON string from the command's payload.

    Returns:
        The decoded payload mapping.

    Raises:
        SimulationExecutionError: If the string is not a JSON object.

    """
    try:
        body = json.loads(raw)
    except ValueError as exc:
        msg = f"inline payload is not JSON: {exc}"
        raise SimulationExecutionError(SimulationErrorCode.PAYLOAD_INVALID, msg) from exc
    if not isinstance(body, dict):
        msg = "inline payload is not a JSON object"
        raise SimulationExecutionError(SimulationErrorCode.PAYLOAD_INVALID, msg)
    return body


def check_payload_hash(body: dict[str, Any], header: dict[str, Any]) -> None:
    """Verify the transmitted hash against the embedded payload.

    Args:
        body: The embedded ``SimulationPayload`` mapping from the command.
        header: The command's ``header`` mapping.

    Raises:
        SimulationExecutionError: On a malformed payload or hash mismatch.

    """
    try:
        simulation = body["simulation"]
    except (KeyError, TypeError) as exc:
        msg = f"cannot read simulation from payload: {exc}"
        raise SimulationExecutionError(SimulationErrorCode.PAYLOAD_INVALID, msg) from exc
    actual = hashlib.sha256(canonical_bytes(simulation)).hexdigest()
    expected = str(header.get("payload_hash", ""))
    if not expected or actual != expected:
        msg = f"payload hash {actual[:12]} does not match header {expected[:12] or '<missing>'}"
        raise SimulationExecutionError(SimulationErrorCode.HASH_MISMATCH, msg)


def _detect_submit_system() -> str | None:
    """Return the local ``tbg_submit`` from the cluster rc params.

    Returns:
        The configured submit command, ``None`` when PIConGPU or the rc
        parameter is unavailable.

    """
    try:
        from picongpu import rc_params  # ruff: ignore[import-outside-top-level] - optional dependency
    except ImportError:
        return None
    value = rc_params.get("tbg_submit", "")
    return str(value) if value else None


def _check_submit_system(requested: str) -> None:
    """Reject a request that would not submit via SLURM ``sbatch``.

    The picongpu workflow default is ``"bash"`` (local execution on the
    submission node, no SLURM job), so the request must itself be ``sbatch``
    (the wire contract only supports SLURM), and a *configured* cluster-local
    ``tbg_submit`` must not contradict it.  An unset ``tbg_submit`` is not an
    error: the explicit ``submit="sbatch"`` flag still overrides the workflow
    default, so the job lands on SLURM either way.

    Args:
        requested: The submit system the command asked for.

    Raises:
        SimulationExecutionError: If the request is not ``sbatch`` or a
            configured cluster-local setting differs.

    """
    if requested != DEFAULT_SUBMIT_SYSTEM:
        msg = f"only {DEFAULT_SUBMIT_SYSTEM!r} submissions are supported, got {requested!r}"
        raise SimulationExecutionError(SimulationErrorCode.SUBMIT_SYSTEM_MISMATCH, msg)
    local = _detect_submit_system()
    if local is not None and local != requested:
        msg = f"cluster tbg_submit={local!r} but the command requests {requested!r}"
        raise SimulationExecutionError(SimulationErrorCode.SUBMIT_SYSTEM_MISMATCH, msg)


def runner_from_payload(payload: SimulationPayload, config: SubmitConfig, token: str) -> Any:
    """Rebuild a fresh ``Runner`` with cluster-local, per-command directories.

    The payload's own directories are never read; only ``sim`` is taken.  The
    ``token`` makes the directories unique per command: ``Runner.generate()``
    asserts the setup directory does not exist, so a legitimate *resubmission*
    of an identical simulation (a new ``cmd_id``) would otherwise collide with
    the previous run's directory.  ``token`` is a validated hex command id.

    Args:
        payload: The validated payload.
        config: The cluster-local submit policy.
        token: Per-command unique token (the ``cmd_id``).

    Returns:
        A ``pypicongpu.Runner`` instance.

    Raises:
        SimulationExecutionError: If PIConGPU is unavailable or the simulation
            does not validate against the local schema.

    """
    if not _TOKEN_RE.match(token):
        msg = f"unsafe per-command token: {token!r}"
        raise SimulationExecutionError(SimulationErrorCode.PATH_UNSAFE, msg)
    base = (config.setup_root / payload.sim_id / token).resolve()
    # Defence in depth: even with a validated token, never build outside the
    # configured root (pathlib discards earlier components on an absolute path).
    root = config.setup_root.resolve()
    if root != base and root not in base.parents:
        msg = f"generated setup dir escapes {config.setup_root}: {base}"
        raise SimulationExecutionError(SimulationErrorCode.PATH_UNSAFE, msg)
    try:
        from picongpu.pypicongpu.runner import Runner  # ruff: ignore[import-outside-top-level] - optional dependency
    except ImportError as exc:
        msg = "PIConGPU is not installed on the cluster"
        raise SimulationExecutionError(SimulationErrorCode.PICONGPU_UNAVAILABLE, msg) from exc
    setup_dir = (base / "input").absolute()
    run_dir = (base / "run").absolute()
    sim_dump = payload.simulation["sim"]
    dump: dict[str, Any] = {"sim": sim_dump, "setup_dir": str(setup_dir), "run_dir": str(run_dir)}
    if config.template_dir:
        dump["template_dir"] = [config.template_dir]
    try:
        runner = Runner.model_validate(dump)
    except Exception as exc:
        msg = f"simulation does not validate: {exc}"
        raise SimulationExecutionError(SimulationErrorCode.PAYLOAD_INVALID, msg) from exc
    # The nested pypicongpu models do not set ``extra="forbid"``, so an unknown
    # field inside ``sim`` would be silently dropped instead of rejected.  The
    # pin guarantees a lossless ``Runner`` round-trip, so a dump that does not
    # reproduce itself carried something outside the schema: reject it as
    # ``unsupported`` per the plan rather than running a silently altered sim.
    if runner.sim.model_dump(mode="json") != sim_dump:
        msg = "simulation carries fields outside the pinned pypicongpu schema"
        raise SimulationExecutionError(SimulationErrorCode.UNSUPPORTED, msg)
    return runner


@dataclass
class PreparedSubmit:
    """A validated, ready-to-execute submit command."""

    payload: SimulationPayload
    params: SubmitParams
    runner: Any
    config: SubmitConfig


def prepare_submit(
    *,
    body: dict[str, Any],
    header: dict[str, Any],
    params: dict[str, Any] | None,
    config: SubmitConfig,
    local_provenance: dict[str, str],
    token: str,
) -> PreparedSubmit:
    """Validate a submit command and rebuild its runner.

    Every check here happens *before* an ``accepted`` ack is sent: a failure means
    the command was never accepted, so the error belongs in the ack (design
    section 2.2).  Stage failures during execution are reported as events
    instead (see :func:`execute_submit`).

    Args:
        body: The embedded payload mapping from the command.
        header: The command's provenance header.
        params: The command's build/run flags.
        config: Cluster-local submit policy.
        local_provenance: This install's provenance tuple.
        token: Per-command unique token for the generated directories.

    Returns:
        The validated payload, flags and fresh runner.

    Raises:
        SimulationExecutionError: On any validation failure.

    """
    check_payload_hash(body, header)

    try:
        payload = SimulationPayload.model_validate(body)
    except ValueError as exc:
        raise SimulationExecutionError(SimulationErrorCode.PAYLOAD_INVALID, str(exc)) from exc
    if payload.payload_hash != str(header.get("payload_hash", "")):
        msg = "payload/header hash mismatch"
        raise SimulationExecutionError(SimulationErrorCode.HASH_MISMATCH, msg)
    if payload.sim_id != str(header.get("sim_id", "")):
        msg = f"payload sim_id {payload.sim_id} does not match header {header.get('sim_id')!r}"
        raise SimulationExecutionError(SimulationErrorCode.HASH_MISMATCH, msg)
    try:
        payload.check_allowlist()
    except UnsupportedPayloadError as exc:
        raise SimulationExecutionError(SimulationErrorCode.UNSUPPORTED, str(exc)) from exc
    mismatches = provenance_mismatches(payload, local_provenance)
    if mismatches:
        raise SimulationExecutionError(SimulationErrorCode.VERSION_MISMATCH, "; ".join(mismatches))

    try:
        submit_params = SubmitParams.model_validate(params or {})
    except ValueError as exc:
        msg = f"invalid submit params: {exc}"
        raise SimulationExecutionError(SimulationErrorCode.PAYLOAD_INVALID, msg) from exc
    _check_submit_system(submit_params.submit_system)

    return PreparedSubmit(
        payload=payload,
        params=submit_params,
        runner=runner_from_payload(payload, config, token),
        config=config,
    )


#: Cap on the captured error text sent in a failure event.
_MAX_ERROR_DETAIL = 4000
#: Files above this size are skipped by the cache scan (compiled binaries).
_MAX_SCAN_FILE_BYTES = 2_000_000
#: Stop the cache scan after this many matching lines.
_MAX_SCAN_HITS = 50

#: Lines worth keeping from the captured stderr for the failure event.
_ERROR_LINE_RE = re.compile(
    r"error|fatal|not found|No such file|command not found|exited with status|permanentFail|missing expected",
    re.IGNORECASE,
)

#: Cap on the short failure summary sent alongside the full error text.
_MAX_FAILURE_SUMMARY = 500

#: A compiler/CMake error line worth promoting into the summary.  Anchored on
#: the ``error:``/``error #``/``error N`` markers a C++ compiler (or nvcc) emits,
#: plus the ``gmake``/``make`` error tail and CMake's own ``CMake Error``.
_COMPILER_ERROR_RE = re.compile(
    r"(?:^|\W)error(?:\s*[:#]|\s+\d+)|gmake(?:\[\d+\])?: \*\*\*|make(?:\[\d+\])?: \*\*\*|CMake Error",
)

#: The CWL workflow's build (compile) step, as it appears in a cwltool failure.
#: A failure naming this step is a *build* failure.
_BUILD_STEP_RE = re.compile(r"\[\s*(?:job|step)\s+build_step(?:_\d+)?\s*\]", re.IGNORECASE)

#: Any named cwltool workflow step (the pinned workflow names every step
#: ``<name>_step``).  A failure naming a step other than the build step is not a
#: compile failure -- it is the submit machinery or, for a job that really ran and
#: crashed, reported by the SLURM follower as ``job_failed`` instead.
_ANY_STEP_RE = re.compile(r"\[\s*(?:job|step)\s+\w+_step(?:_\d+)?\s*\]", re.IGNORECASE)

#: Build-step markers, as a fallback when no step-named line survived (e.g. a
#: truncated capture holding only the C++ tail).  These require a genuine
#: compiler/make/CMake *error* marker, not a bare build artifact path: a
#: submit-machinery failure whose truncated tail merely echoes a
#: ``.../build/main.x.cpp`` path or the word ``cmake`` must not be relabelled a
#: compile failure.
_BUILD_MARKER_RE = re.compile(
    r"\bpic-build\b"
    r"|gmake(?:\[\d+\])?: \*\*\*"
    r"|make(?:\[\d+\])?: \*\*\*"
    r"|CMake Error"
    r"|\bnvcc\b[^\n]*\b(?:error|fatal)\b"
    r"|\.(?:hpp|cpp|cu|cuh)(?:\(\d+\)|:\d+)?:\s*(?:fatal\s+)?error\b",
    re.IGNORECASE,
)


#: Per-thread capture scratch for :class:`_CapturingStderr`; set for the
#: duration of one workflow run so concurrent runs never share a buffer.  Holds
#: the text ``buffer``, the write end of the run's pipe (``write_fd``) and a
#: ``lock`` serialising the two writers (the pipe reader and the cwltool logger
#: handler) onto the shared buffer.
_CAPTURE_STATE = threading.local()

#: Serialises the one-time ``RuntimeContext`` install.
_CAPTURE_INSTALL_LOCK = threading.Lock()

#: Serialises the one-time cwltool logger-handler install.
_CAPTURE_HANDLER_LOCK = threading.Lock()

#: The process-wide cwltool logger handler (installed once); it routes a record
#: to the emitting thread's buffer, so concurrent runs stay isolated.
_CAPTURE_LOGGER_HANDLER: _ThreadBufferHandler | None = None

#: The shared capture proxy injected process-wide; per-run isolation comes from
#: the thread-local buffer, not from the proxy identity.
_CAPTURE_PROXY: _CapturingStderr | None = None

#: Chunk size for the per-run pipe reader (matches the removed ``os.dup2``
#: reader).
_CAPTURE_READ_CHUNK = 4096

#: Number of trailing output lines retained while a build/run workflow runs, so
#: ``get_logs`` can show the compiler tail during the multi-minute build window
#: instead of a flat "not started yet".  Bounded so a long build cannot grow the
#: simclient's memory.
_BUILD_TAIL_LINES = 200

#: Cap on the bytes retained for the live build tail (a backstop against a few
#: very long lines), matching the failure-detail budget order of magnitude.
_MAX_BUILD_TAIL_BYTES = 64 * 1024


class _RingTail:
    """A bounded, thread-safe ring of the last captured lines.

    Bounded by both line count (:data:`_BUILD_TAIL_LINES`) and total bytes
    (:data:`_MAX_BUILD_TAIL_BYTES`) so a build emitting a few very long lines
    cannot grow the simclient's memory either.
    """

    def __init__(self, maxlen: int = _BUILD_TAIL_LINES) -> None:
        """Create an empty tail."""
        self._lines: deque[str] = deque(maxlen=maxlen)
        self._bytes = 0
        self._lock = threading.Lock()

    def add(self, text: str) -> None:
        """Append the lines in ``text``, normalising CRLF and dropping blanks.

        The byte accounting must survive the deque's own ``maxlen`` eviction:
        when the deque is full, ``append`` silently drops the oldest line, so
        the running total has to subtract that line's length too.  Recomputing
        ``sum`` after the insertion is the simplest way to stay exact and keeps
        the explicit byte-budget eviction from popping lines the deque already
        accounts for.

        Args:
            text: A chunk of captured output (may contain several lines).

        """
        with self._lock:
            for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
                if not line:
                    continue
                self._lines.append(line)
            while self._lines and sum(len(line) for line in self._lines) > _MAX_BUILD_TAIL_BYTES:
                self._lines.popleft()
            self._bytes = sum(len(line) for line in self._lines)

    def lines(self) -> list[str]:
        """Return the retained lines, oldest first.

        Returns:
            A copy of the current tail.

        """
        with self._lock:
            return list(self._lines)


#: Per-run live build tails, keyed by ``run_dir`` (a string), so ``get_logs``
#: can read the compiler output while :func:`_run_workflow` is still running.
_live_build_tails: dict[str, _RingTail] = {}
_live_build_tails_lock = threading.Lock()


def _register_live_build_tail(run_dir: Path | str) -> _RingTail:
    """Register (or reuse) the live build tail for ``run_dir``.

    Args:
        run_dir: The run directory (the tail key).

    Returns:
        The shared tail to feed and to expose via :func:`live_build_log`.

    """
    key = str(run_dir)
    with _live_build_tails_lock:
        tail = _live_build_tails.get(key)
        if tail is None:
            tail = _RingTail()
            _live_build_tails[key] = tail
        return tail


def _unregister_live_build_tail(run_dir: Path | str) -> None:
    """Drop the live build tail for ``run_dir`` (after the workflow ends).

    The failure ``error`` already carries the compiler tail, so the live tail
    only exists for the duration of the build.

    Args:
        run_dir: The run directory whose tail is no longer needed.

    """
    with _live_build_tails_lock:
        _live_build_tails.pop(str(run_dir), None)


def live_build_log(run_dir: Path | str) -> list[str]:
    """Return the retained build tail for a run, if a build was captured.

    Args:
        run_dir: The run directory (a resolve of the derived setup's sibling).

    Returns:
        The retained lines, oldest first; empty when nothing was captured.

    """
    with _live_build_tails_lock:
        tail = _live_build_tails.get(str(run_dir))
    return tail.lines() if tail is not None else []


class _CapturingStderr:
    """A per-run ``stderr`` proxy that tees into the invoking thread's buffer.

    cwltool passes ``RuntimeContext.default_stderr`` to every step subprocess
    and uses its file descriptor for the child's stderr, so the fd must be a
    real OS descriptor.  Each run therefore owns a pipe: :func:`_run_workflow`
    starts a daemon reader on the read end that tees every chunk to *both* the
    real stderr (operator logs stay visible, as the removed ``os.dup2`` reader
    did) and the run's text buffer, and points this proxy's ``fileno()`` at the
    write end.  Two workflows in different threads have their own pipes and
    buffers, so concurrent runs never observe each other's output.  cwltool's
    *own* diagnostics (``cwltool._logger``) do not go through the child fd, so
    they are captured separately by a thread-routed logging handler (see
    :func:`_install_capture_logger`).
    """

    def __init__(self, real: Any) -> None:
        """Wrap ``real`` (the process stderr) as the pass-through sink."""
        self._real = real

    @staticmethod
    def _buffer() -> Any:
        """Return the current thread's capture buffer, if one is active.

        Returns:
            The thread-local buffer, or ``None`` when this thread is not
            running a workflow.

        """
        return getattr(_CAPTURE_STATE, "buffer", None)

    def _target(self) -> Any:
        """Return the thread's capture buffer, or fall back to the real sink.

        Returns:
            The active buffer for this thread, else the wrapped real stream.

        """
        return self._buffer() or self._real

    def write(self, text: str) -> int:
        """Write ``text`` to the per-run buffer and the real stderr (tee).

        Returns:
            The number of characters written to the capture buffer, or to the
            real stderr when no run is active on this thread.

        """
        buffer = self._buffer()
        if buffer is None:
            return self._real.write(text)
        with _CAPTURE_STATE.lock:
            written = buffer.write(text)
        self._real.write(text)
        return written

    def flush(self) -> None:
        """Flush the buffer and the real stderr (both may hold pending text)."""
        buffer = self._buffer()
        if buffer is not None:
            with _CAPTURE_STATE.lock:
                buffer.flush()
        self._real.flush()

    def fileno(self) -> int:
        """Return the descriptor a step subprocess should inherit.

        While a run is active on this thread this is the write end of the run's
        capture pipe (read and teed by :func:`_run_workflow`); outside a run it
        falls back to the real stderr.

        Returns:
            The run's pipe write fd, or the real stderr's fd.

        """
        write_fd = getattr(_CAPTURE_STATE, "write_fd", None)
        if write_fd is not None:
            return write_fd
        return self._real.fileno()

    def isatty(self) -> bool:
        """Report whether the real stderr is a terminal.

        Returns:
            True when the real stderr is a terminal.

        """
        return self._real.isatty()

    @staticmethod
    def writable() -> bool:
        """Report that the sink accepts writes.

        Returns:
            Always True.

        """
        return True

    @property
    def encoding(self) -> str:
        """The real stderr's text encoding (used by subprocess)."""
        return getattr(self._real, "encoding", None) or "utf-8"

    def close(self) -> None:
        """No-op: the per-run buffer outlives the stream cwltool may close."""

    def __getattr__(self, name: str) -> Any:
        """Delegate any other attribute access to the real stderr.

        Returns:
            The named attribute of the wrapped real stream.

        """
        return getattr(self._real, name)


class _ThreadBufferHandler(logging.Handler):
    """Route the ``cwltool`` logger's output into the emitting thread's buffer.

    cwltool emits its own diagnostics (e.g. a missing ``baseCommand``
    executable) through ``cwltool._logger.error(...)``, which never touches
    ``RuntimeContext.default_stderr``; without this handler a ``Completed
    permanentFail`` carries no cause (M2).  The handler is installed once,
    process-wide, and routes by thread: a record emitted on a thread running a
    workflow lands in that run's buffer, while a record from any other thread
    is dropped here (it still reaches the real stderr via cwltool's own
    ``defaultStreamHandler``), so concurrent runs stay isolated.

    This routing (and :meth:`_CapturingStderr.fileno`) assumes cwltool's
    default ``SingleJobExecutor``, which runs each step in the calling thread.
    A ``MultithreadedJobExecutor`` (or cwltool spawning its own worker threads)
    would resolve the wrong thread's buffer/fd and cross-attribute stderr; the
    pinned ``pypicongpu.Runner`` does not configure one.
    """

    def __init__(self) -> None:
        """Create the handler with a message-only formatter."""
        super().__init__()
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        """Append the formatted record to the current thread's buffer and tail."""
        buffer = getattr(_CAPTURE_STATE, "buffer", None)
        if buffer is None:
            return
        try:
            text = self.format(record) + "\n"
            with _CAPTURE_STATE.lock:
                buffer.write(text)
        except Exception:  # ruff: ignore[blind-except] - logging must never raise into cwltool
            self.handleError(record)
            return
        tail = getattr(_CAPTURE_STATE, "tail", None)
        if tail is not None:
            tail.add(text)


def _install_capture_logger() -> None:
    """Install the process-wide ``cwltool`` logger capture handler once.

    Safe under concurrency: the handler is shared, but dispatch is thread-local
    (see :class:`_ThreadBufferHandler`).
    """
    global _CAPTURE_LOGGER_HANDLER  # ruff: ignore[global-statement] - one-time process-wide install
    with _CAPTURE_HANDLER_LOCK:
        if _CAPTURE_LOGGER_HANDLER is not None:
            return
        try:
            import cwltool.loghandler  # ruff: ignore[import-outside-top-level] - optional dependency, imported lazily
        except ImportError:
            return
        _CAPTURE_LOGGER_HANDLER = _ThreadBufferHandler()
        cwltool.loghandler._logger.addHandler(_CAPTURE_LOGGER_HANDLER)  # ruff: ignore[private-member-access] - cwltool's module logger


def _capture_runtime_context(base: Any, proxy: _CapturingStderr) -> type:
    """Build a ``RuntimeContext`` subclass that injects the capture proxy.

    Returns:
        A subclass forcing ``default_stderr`` to ``proxy``.

    """

    class _RuntimeContext(base):  # type: ignore[misc, valid-type]
        def __init__(self, kwargs: Any = None) -> None:
            merged = dict(kwargs or {})
            merged.setdefault("default_stderr", proxy)
            super().__init__(merged)

    return _RuntimeContext


def _run_workflow(runner: Any, capture: list[str] | None = None) -> str:
    """Run the CWL workflow, returning its captured error lines.

    Captures two streams for *this* run:

    - the step subprocesses' stderr, via a per-run pipe: their inherited fd is
      the pipe's write end and a daemon reader tees every chunk to the real
      stderr (operator visibility) and this run's buffer;
    - cwltool's own logger diagnostics (e.g. a missing ``baseCommand``), via a
      process-wide handler that routes records to the emitting thread's buffer.

    cwltool raises only ``Completed permanentFail``; both streams above carry
    the actual cause.  The capture is per run (see :class:`_CapturingStderr`),
    so concurrent submissions never observe each other's output.

    Args:
        runner: The ``pypicongpu.Runner`` to run.
        capture: Optional single-element list receiving the captured lines even
            when ``runner.run()`` raises (the failure path needs the detail).

    Returns:
        The matching stderr lines for *this* run, or ``""`` when the workflow
        does not use the pinned runner (e.g. a test double).

    """
    buffer = tempfile.TemporaryFile(mode="w+", encoding="utf-8")  # ruff: ignore[open-file-with-context-handler]
    _install_capture_context()
    # Tee the child-fd output to the same real stream the proxy passes writes
    # to, so both capture paths agree (and a test can substitute the sink).
    real = _CAPTURE_PROXY._real if _CAPTURE_PROXY is not None else sys.stderr  # ruff: ignore[private-member-access] - same class
    read_fd, write_fd = os.pipe()

    # The live build tail (B2): the same captured output, bounded and exposed by
    # ``get_logs`` while the workflow runs, so the multi-minute build window is
    # observable instead of a flat "not started yet".  A test double without a
    # ``run_dir`` gets no tail.
    run_dir = getattr(runner, "run_dir", None)
    live_tail = _register_live_build_tail(run_dir) if run_dir is not None else None

    lock = threading.Lock()

    def reader() -> None:
        # Tee each chunk to the real stderr (operator logs stay visible) and
        # the run's buffer/tail; a daemon thread so a stuck child cannot wedge
        # exit.  ``buffering=0`` is important: a *buffered* ``read(n)`` blocks
        # until ``n`` bytes or EOF, which would hold the whole build's output
        # until the step ended and defeat the live tail (B2).
        with os.fdopen(read_fd, "rb", buffering=0) as stream:
            for chunk in iter(lambda: stream.read(_CAPTURE_READ_CHUNK), b""):
                text = chunk.decode("utf-8", errors="replace")
                real.write(text)
                with lock:
                    buffer.write(text)
                if live_tail is not None:
                    live_tail.add(text)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    _install_capture_logger()
    _CAPTURE_STATE.buffer = buffer
    _CAPTURE_STATE.lock = lock
    _CAPTURE_STATE.write_fd = write_fd
    _CAPTURE_STATE.tail = live_tail
    try:
        runner.run()
    finally:
        _CAPTURE_STATE.buffer = None
        _CAPTURE_STATE.write_fd = None
        _CAPTURE_STATE.tail = None
        # Close our write end first so the reader sees EOF once every inheriting
        # child has also closed it; only then join.
        with contextlib.suppress(OSError):  # pragma: no cover - a child could not close our fd
            os.close(write_fd)
        thread.join(timeout=10)
        text = ""
        try:
            buffer.seek(0)
            text = buffer.read()
        finally:
            buffer.close()
        lines = "\n".join(line.rstrip() for line in text.splitlines() if _ERROR_LINE_RE.search(line))
        if capture is not None:
            capture.append(lines)
        if live_tail is not None:
            # The workflow (and its compiler output) has ended; the failure
            # ``error`` already folds the tail in, and the success path reads the
            # real log files.  Drop the live tail so it cannot be served stale.
            _unregister_live_build_tail(run_dir)
    return lines


def _install_capture_context() -> None:
    """Install the capture ``RuntimeContext`` once, process-wide.

    ``pypicongpu.Runner.run`` constructs its own ``RuntimeContext`` at call
    time, so the class it imports must be replaced to inject the capture proxy.
    Installing once (under a lock) is safe under concurrency: the process-wide
    class is shared, while :data:`_CAPTURE_STATE` keeps each run's buffer
    thread-local, so concurrent runs cannot observe one another's output.
    """
    global _CAPTURE_PROXY  # ruff: ignore[global-statement] - one-time process-wide install
    with _CAPTURE_INSTALL_LOCK:
        if _CAPTURE_PROXY is not None:
            return
        try:
            from picongpu.pypicongpu import runner as runner_module  # ruff: ignore[import-outside-top-level]
        except ImportError:
            return
        proxy = _CapturingStderr(sys.stderr)
        runner_module.RuntimeContext = _capture_runtime_context(runner_module.RuntimeContext, proxy)
        _CAPTURE_PROXY = proxy


def _workflow_failure_detail(run_dir: Path, captured: str = "") -> str:
    """Return the captured workflow stderr plus any retained step log.

    Args:
        run_dir: The run directory (for the .cwl_cache fallback scan).
        captured: This run's captured stderr lines (empty for a test double).

    Returns:
        A truncated, redaction-ready error string (possibly empty).

    """
    detail = captured.strip()
    if not detail:
        detail = _scan_retained_step_logs(run_dir)
    return detail[-_MAX_ERROR_DETAIL:]


def _classify_workflow_failure(detail: str) -> tuple[SimulationErrorCode, SimulationStage]:
    """Classify a cwltool workflow failure as a build or a run-stage failure.

    The captured cwltool detail names the failing step.  A failure of the
    ``build_step`` is the PIConGPU compile (``pic-build``): a cwltool job that
    exits with a compiler ``error:`` and/or did not produce the ``bin``
    directory.  A failure of any other workflow step (the
    ``prepare_submission``/``submit``/``organize_output`` machinery) is a
    run-stage failure; a genuine numerical/runtime failure surfaces through a
    SLURM ``job_failed`` (not a workflow exception) and keeps its own distinct
    reporting.

    Args:
        detail: The captured workflow failure text.

    Returns:
        The ``(error_code, stage)`` pair for a ``simulation.failed`` event.

    """
    if _BUILD_STEP_RE.search(detail):
        return SimulationErrorCode.BUILD_FAILED, SimulationStage.BUILD
    if _ANY_STEP_RE.search(detail):
        # A different workflow step failed (prepare/submit/organize machinery);
        # the compiler never ran, so it is not a build-stage compile failure.
        return SimulationErrorCode.RUN_FAILED, SimulationStage.RUN
    if _BUILD_MARKER_RE.search(detail):
        # No step line survived, but the detail carries compiler/CMake output.
        return SimulationErrorCode.BUILD_FAILED, SimulationStage.BUILD
    return SimulationErrorCode.RUN_FAILED, SimulationStage.RUN


def _failure_summary(detail: str) -> str | None:
    """Extract a bounded one-of-the-first compiler/CMake error lines.

    The raw ``error`` is a cwltool ``permanentFail`` dump followed by a
    truncated C++ compiler tail; the actual cause (e.g. a CFL ``static_assert``)
    is a single line buried in it.  Promote the first compiler/CMake error line
    (plus the ``make`` tail when present) into a short summary so the cause is
    readable without scrolling the dump; the full text stays in ``error``.

    Args:
        detail: The captured workflow failure text.

    Returns:
        A short summary string, or None when no recognizable error line exists.

    """
    summary: list[str] = []
    for line in detail.splitlines():
        stripped = line.strip()
        if not stripped or not _COMPILER_ERROR_RE.search(stripped):
            continue
        # Skip cwltool's own wrapper diagnostics (``... cwltool: [job
        # build_step] Job error:``): they name the failing step but not the
        # cause.  The compiler/CMake/make lines below them carry the cause.
        if "cwltool:" in stripped:
            continue
        summary.append(stripped)
        if len("\n".join(summary)) >= _MAX_FAILURE_SUMMARY:
            break
    if not summary:
        return None
    return "\n".join(summary)[:_MAX_FAILURE_SUMMARY]


def _scan_retained_step_logs(run_dir: Path) -> str:
    """Best-effort scan of the retained cwltool cache for an error line.

    Returns:
        The most relevant-looking error line, or ``""``.

    """
    cache = Path(run_dir) / ".cwl_cache"
    if not cache.is_dir():
        return ""
    pattern = re.compile(r"error|fatal|not found|No such file|command not found|exited with status", re.IGNORECASE)
    hits: list[str] = []
    for path in cache.rglob("*"):
        if not path.is_file() or path.stat().st_size > _MAX_SCAN_FILE_BYTES:
            continue
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                hits.extend(line.rstrip() for line in handle if pattern.search(line))
        except OSError:
            continue
        if len(hits) > _MAX_SCAN_HITS:
            break
    return "\n".join(hits[-20:])


def _normalise_workflow_vars(setup_dir: Path) -> None:
    """Patch the generated ``input.yaml`` so CWL accepts ``run_overwrite_vars``.

    The pinned ``Runner.generate()`` serialises its ``TBGFlags.overwrite_vars``
    list straight into the workflow input as a YAML/JSON list, but the pinned
    ``workflow.cwl`` declares ``run_overwrite_vars`` as ``type: string?`` (tbg
    takes a single ``-o`` argument it word-splits itself).  Left as a list, CWL
    validation fails with "value is a CommentedSeq, expected null or string",
    so every submission using the flag would die as ``RUN_FAILED`` after
    ``accepted``.  Join the (already validated, single-token) entries into the
    one space-separated string the tool expects.

    Args:
        setup_dir: The runner's setup directory (holds ``workflow/input.yaml``).

    """
    input_path = Path(setup_dir) / "workflow" / "input.yaml"
    if not input_path.is_file():
        return
    data = json.loads(input_path.read_text(encoding="utf-8"))
    value = data.get("run_overwrite_vars")
    if isinstance(value, list):
        data["run_overwrite_vars"] = " ".join(str(entry) for entry in value)
        input_path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def link_run_results(run_dir: Path) -> bool:
    """Link the simulation output into ``run_dir`` via the generated script.

    The CWL workflow writes PIConGPU output inside its per-step cache directory
    and generates ``link_results.sh`` to expose it as ``run_dir/simOutput``, but
    never runs that script in the run directory.  Running it here makes
    ``run_dir/simOutput`` present before the workflow-finished event, matching
    the design's expectation that results are organised when the run finishes.

    Args:
        run_dir: The runner's run directory.

    Returns:
        True if the script ran successfully (or the link already exists), False
        otherwise; a missing link is not fatal, the job may still be running.

    """
    run_dir = Path(run_dir)
    if (run_dir / "simOutput").exists():
        return True
    script = run_dir / "link_results.sh"
    if not script.is_file():
        return False
    try:
        result = subprocess.run(
            ["/bin/bash", str(script), str(run_dir)],
            cwd=str(run_dir),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and (run_dir / "simOutput").exists()


def find_stdout_path(run_dir: Path) -> str | None:
    """Locate the job's ``stdout`` inside the cwltool step cache.

    The CWL step is submitted with ``#SBATCH -o stdout`` and a ``--chdir`` into
    its per-step cache directory, so the SLURM output lands at
    ``run_dir/.cwl_cache/*/stdout``.  The cache layout is internal to cwltool
    (the plan's known risk), so discovery is a best-effort glob: when several
    step caches match, the newest file (by mtime) wins.

    Args:
        run_dir: The runner's run directory.

    Returns:
        The absolute path to the newest matching ``stdout``, or None when the
        cache carries none.

    """
    run_dir = Path(run_dir)
    candidates = [path for path in run_dir.glob(".cwl_cache/*/stdout") if path.is_file()]
    if not candidates:
        return None
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    return str(newest)


async def execute_submit(
    *,
    prepared: PreparedSubmit,
    emit: Any,
    job_id_reader: Any,
) -> dict[str, Any]:
    """Run a prepared submission (accepted already sent).

    Args:
        prepared: The validated submission.
        emit: Async callable ``emit(state, **fields)`` posting a lifecycle event.
        job_id_reader: Callable ``job_id_reader(run_dir, payload) -> int | None``.

    Returns:
        ``{"sim_id", "state", "job_id", "run_dir", "stdout_path"}``.

    Raises:
        SimulationExecutionError: On a build/run stage failure.

    """
    runner = prepared.runner
    flags = prepared.params.picongpu_flags()
    # The cluster-local preset is a default: an explicit command flag wins.
    if prepared.config.preset is not None and flags.get("preset") is None:
        flags["preset"] = prepared.config.preset
    # build stage: generate the setup (must not pre-exist).  generate() is
    # synchronous and can run for minutes, so keep it off the event loop.
    try:
        await asyncio.to_thread(lambda: runner.generate(**flags))
    except Exception as exc:
        msg = f"generate failed: {exc}"
        raise SimulationExecutionError(SimulationErrorCode.GENERATE_FAILED, msg, SimulationStage.BUILD) from exc
    # The pinned workflow.cwl types run_overwrite_vars as a single string while
    # Runner.generate() writes the list through; patch the input so a
    # submission using -o does not fail CWL validation (see the helper).
    await asyncio.to_thread(_normalise_workflow_vars, runner.setup_dir)

    # Submit stage: run the workflow; job id from submission_information.txt.
    # cwltool raises a generic ``Completed permanentFail`` and logs the actual
    # step error at ERROR level, so capture its log for the failure event;
    # otherwise the event is undiagnosable from the room.  ``capture`` receives
    # this run's lines even on the exception path.
    capture: list[str] = []
    try:
        await asyncio.to_thread(_run_workflow, runner, capture)
    except Exception as exc:
        detail = _workflow_failure_detail(runner.run_dir, capture[0] if capture else "")
        msg = f"workflow failed: {exc}"
        if detail:
            msg = f"{msg}\n{detail}"
        code, stage = _classify_workflow_failure(detail)
        summary = _failure_summary(detail)
        raise SimulationExecutionError(code, msg, stage, summary=summary) from exc

    job_id = job_id_reader(runner.run_dir, prepared.payload)
    if job_id is not None:
        await emit(SimulationState.SUBMITTED, job_id=job_id, submit_system=prepared.params.submit_system)
    # A submit system without a scheduler job id (e.g. local ``bash`` execution,
    # or a scheduler whose output has no parseable id) has nothing to report in
    # ``simulation.submitted``; the ``workflow.finished`` event below still fires
    # with ``job_id=None``, so the lifecycle is not silently truncated.
    link_ready = await asyncio.to_thread(link_run_results, runner.run_dir)
    await emit(SimulationState.WORKFLOW_FINISHED, job_id=job_id, results_linked=link_ready)
    stdout_path = await asyncio.to_thread(find_stdout_path, runner.run_dir)
    return {
        "sim_id": prepared.payload.sim_id,
        "state": SimulationState.WORKFLOW_FINISHED.value,
        "job_id": job_id,
        "run_dir": str(runner.run_dir),
        "stdout_path": stdout_path,
    }


__all__ = [
    "PreparedSubmit",
    "SimulationErrorCode",
    "SimulationExecutionError",
    "SubmitConfig",
    "check_payload_hash",
    "execute_submit",
    "find_stdout_path",
    "prepare_submit",
]
