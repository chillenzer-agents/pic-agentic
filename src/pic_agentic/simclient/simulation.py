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
import hashlib
import json
import logging
import re
import subprocess
import sys
import tempfile
import threading
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
    PICONGPU_UNAVAILABLE = "picongpu_unavailable"
    GENERATE_FAILED = "generate_failed"
    RUN_FAILED = "run_failed"
    #: M3 control/results error codes.
    NOT_SIGNALABLE = "not_signalable"
    NOT_TERMINAL = "not_terminal"
    READER_UNAVAILABLE = "reader_unavailable"
    NO_RESULTS = "no_results"
    RESULT_TOO_LARGE = "result_too_large"


class SimulationExecutionError(RuntimeError):
    """A stage failure carrying a machine-readable error code."""

    def __init__(self, code: SimulationErrorCode, message: str, stage: SimulationStage | None = None) -> None:
        """Create the error.

        Args:
            code: Stable error code reported in the ack/event.
            message: Human-readable detail.
            stage: Pipeline stage, when the failure happened after acceptance.

        """
        super().__init__(message)
        self.code = code
        self.stage = stage


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


#: Per-thread scratch buffer for :class:`_CapturingStderr`; set for the
#: duration of one workflow run so concurrent runs never share a buffer.
_CAPTURE_STATE = threading.local()

#: Serialises the one-time ``RuntimeContext`` install.
_CAPTURE_INSTALL_LOCK = threading.Lock()

#: The shared capture proxy injected process-wide; per-run isolation comes from
#: the thread-local buffer, not from the proxy identity.
_CAPTURE_PROXY: _CapturingStderr | None = None


class _CapturingStderr:
    """A per-run ``stderr`` proxy that tees into the invoking thread's buffer.

    cwltool passes ``RuntimeContext.default_stderr`` to every step subprocess
    and, for its own messages, hands it to the ``cwltool`` logger.  Redirecting
    *that* object (rather than process-wide file descriptor 2) makes the capture
    per run: two workflows in different threads write to their own buffers,
    since :class:`_CapturingStderr` resolves the current thread's buffer on
    every write.  The real stderr is still written through, so operator logs
    stay visible.
    """

    def __init__(self, real: Any) -> None:
        """Wrap ``real`` (the process stderr) as the pass-through sink."""
        self._real = real

    def _target(self) -> Any:
        """Return the thread's capture buffer, or fall back to the real sink.

        Returns:
            The active buffer for this thread, else the wrapped real stream.

        """
        return getattr(_CAPTURE_STATE, "buffer", None) or self._real

    def write(self, text: str) -> int:
        """Write ``text`` to the per-run buffer and the real stderr.

        Returns:
            The number of characters written (as reported by the sink).

        """
        return self._target().write(text)

    def flush(self) -> None:
        """Flush the current thread's sink."""
        self._target().flush()

    def fileno(self) -> int:
        """Return the underlying file descriptor.

        Returns:
            The file descriptor of the current thread's sink.

        """
        return self._target().fileno()

    def isatty(self) -> bool:
        """Report whether the current thread's sink is a terminal.

        Returns:
            True when the sink is a terminal.

        """
        return self._target().isatty()

    @staticmethod
    def writable() -> bool:
        """Report that the sink accepts writes.

        Returns:
            Always True.

        """
        return True

    @property
    def encoding(self) -> str:
        """The sink's text encoding (used by subprocess)."""
        return getattr(self._target(), "encoding", None) or "utf-8"

    def close(self) -> None:
        """No-op: the per-run buffer outlives the stream cwltool may close."""

    def __getattr__(self, name: str) -> Any:
        """Delegate any other attribute access to the real stderr.

        Returns:
            The named attribute of the wrapped real stream.

        """
        return getattr(self._real, name)


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

    cwltool raises only ``Completed permanentFail``; the actual step command
    error (e.g. ``cmake: command not found``) is written to stderr by cwltool
    and the step subprocess, bypassing the ``cwltool`` logger.  The capture is
    per run (see :class:`_CapturingStderr`), so concurrent submissions never
    observe each other's output.

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
    _CAPTURE_STATE.buffer = buffer
    try:
        runner.run()
    finally:
        _CAPTURE_STATE.buffer = None
        text = ""
        try:
            buffer.seek(0)
            text = buffer.read()
        finally:
            buffer.close()
        lines = "\n".join(line.rstrip() for line in text.splitlines() if _ERROR_LINE_RE.search(line))
        if capture is not None:
            capture.append(lines)
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
        raise SimulationExecutionError(SimulationErrorCode.RUN_FAILED, msg, SimulationStage.RUN) from exc

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
