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
import io
import json
import logging
import re
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


#: Per-process buffer of the last workflow's cwltool ERROR output.  A shared
#: buffer is acceptable because the simclient executes one submission at a time.
_LAST_CWLT_ERROR = io.StringIO()

#: Cap on the captured error text sent in a failure event.
_MAX_ERROR_DETAIL = 4000
#: Files above this size are skipped by the cache scan (compiled binaries).
_MAX_SCAN_FILE_BYTES = 2_000_000
#: Stop the cache scan after this many matching lines.
_MAX_SCAN_HITS = 50


def _run_workflow(runner: Any) -> None:
    """Run the CWL workflow, capturing cwltool's own ERROR log.

    cwltool's exception is only ``Completed permanentFail``; the step command
    error (e.g. ``cmake: command not found``) is emitted on the ``cwltool``
    logger at ERROR level.  Attach a temporary handler to retain it.

    Args:
        runner: The ``pypicongpu.Runner`` to run.

    """
    import logging  # ruff: ignore[import-outside-top-level] - only needed for this call

    handler = logging.StreamHandler(_LAST_CWLT_ERROR)
    handler.setLevel(logging.ERROR)
    logger = logging.getLogger("cwltool")
    logger.addHandler(handler)
    try:
        runner.run()
    finally:
        logger.removeHandler(handler)


def _workflow_failure_detail(run_dir: Path) -> str:
    """Return the captured cwltool error plus any retained step log.

    Args:
        run_dir: The run directory (for the .cwl_cache fallback scan).

    Returns:
        A truncated, redaction-ready error string (possibly empty).

    """
    captured = _LAST_CWLT_ERROR.getvalue()
    _LAST_CWLT_ERROR.seek(0)
    _LAST_CWLT_ERROR.truncate(0)
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
        ``{"sim_id", "state", "job_id"}``.

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

    # Submit stage: run the workflow; job id from submission_information.txt.
    # cwltool raises a generic ``Completed permanentFail`` and logs the actual
    # step error at ERROR level, so capture its log for the failure event;
    # otherwise the event is undiagnosable from the room.
    try:
        await asyncio.to_thread(_run_workflow, runner)
    except Exception as exc:
        detail = _workflow_failure_detail(runner.run_dir)
        msg = f"workflow failed: {exc}"
        if detail:
            msg = f"{msg}\n{detail}"
        raise SimulationExecutionError(SimulationErrorCode.RUN_FAILED, msg, SimulationStage.RUN) from exc

    job_id = job_id_reader(runner.run_dir, prepared.payload)
    if job_id is not None:
        await emit(SimulationState.SUBMITTED, job_id=job_id, submit_system=prepared.params.submit_system)
    await emit(SimulationState.RESULTS_READY, job_id=job_id)
    return {"sim_id": prepared.payload.sim_id, "state": SimulationState.RESULTS_READY.value, "job_id": job_id}


__all__ = [
    "PreparedSubmit",
    "SimulationErrorCode",
    "SimulationExecutionError",
    "SubmitConfig",
    "check_payload_hash",
    "execute_submit",
    "prepare_submit",
]
