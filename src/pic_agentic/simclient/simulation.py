# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Simulation-side execution of an M2 ``submit_simulation`` command.

The handler is deliberately paranoid (design sections 6.4, 8.2): it re-validates
the server-generated payload path, checks the payload hash and the provenance
tuple against the local install, and only then imports PIConGPU.  All cluster
locations come from local configuration, never from the payload.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pic_agentic.protocol.simulation import (
    SimulationPayload,
    SimulationStage,
    SimulationState,
    SubmitParams,
    UnsupportedPayloadError,
    provenance_mismatches,
)
from pic_agentic.rcp import canonical_bytes
from pic_agentic.simclient.safety import UnsafePathError, validate_message_path

log = logging.getLogger(__name__)


class SimulationErrorCode(StrEnum):
    """Stable machine-readable error codes reported in acks and events."""

    PATH_UNSAFE = "path_unsafe"
    PAYLOAD_UNREADABLE = "payload_unreadable"
    PAYLOAD_INVALID = "payload_invalid"
    UNSUPPORTED = "unsupported"
    HASH_MISMATCH = "hash_mismatch"
    VERSION_MISMATCH = "version_mismatch"
    SUBMIT_SYSTEM_MISMATCH = "submit_system_mismatch"
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

    message_dir: Path
    setup_root: Path
    template_dir: str = ""
    preset: int | None = None

    def __post_init__(self) -> None:
        """Normalise the paths to absolute form."""
        self.message_dir = Path(self.message_dir)
        self.setup_root = Path(self.setup_root)


def check_payload_hash(raw: bytes, header: dict[str, Any]) -> None:
    """Verify the transmitted hash against the raw payload bytes.

    Args:
        raw: The payload file contents.
        header: The command's ``header`` mapping.

    Raises:
        SimulationExecutionError: On a malformed payload or hash mismatch.

    """
    try:
        data = json.loads(raw)
        simulation = data["simulation"]
    except (ValueError, KeyError, TypeError) as exc:
        msg = f"cannot read simulation from payload: {exc}"
        raise SimulationExecutionError(SimulationErrorCode.PAYLOAD_INVALID, msg) from exc
    actual = hashlib.sha256(canonical_bytes(simulation)).hexdigest()
    expected = str(header.get("payload_hash", ""))
    if not expected or actual != expected:
        msg = f"payload hash {actual[:12]} does not match header {expected[:12] or '<missing>'}"
        raise SimulationExecutionError(SimulationErrorCode.HASH_MISMATCH, msg)


def _detect_submit_system() -> str:
    """Return the local ``tbg_submit`` from the cluster rc params.

    Returns:
        The configured submit command, or ``""`` when PIConGPU or the rc
        parameter is unavailable.

    """
    try:
        from picongpu import rc_params  # ruff: ignore[import-outside-top-level] - optional dependency
    except ImportError:
        return ""
    value = rc_params.get("tbg_submit", "")
    return str(value) if value else ""


def _runner_from_payload(payload: SimulationPayload, config: SubmitConfig) -> Any:
    """Rebuild a fresh ``Runner`` with cluster-local directories.

    The payload's own directories are never read; only ``sim`` is taken.

    Args:
        payload: The validated payload.
        config: The cluster-local submit policy.

    Returns:
        A ``pypicongpu.Runner`` instance.

    Raises:
        SimulationExecutionError: If PIConGPU is unavailable or the simulation
            does not validate against the local schema.

    """
    try:
        from picongpu.pypicongpu.runner import Runner  # ruff: ignore[import-outside-top-level] - optional dependency
    except ImportError as exc:
        msg = "PIConGPU is not installed on the cluster"
        raise SimulationExecutionError(SimulationErrorCode.PICONGPU_UNAVAILABLE, msg) from exc
    setup_dir = (config.setup_root / payload.sim_id / "input").absolute()
    run_dir = (config.setup_root / payload.sim_id / "run").absolute()
    dump: dict[str, Any] = {"sim": payload.simulation["sim"], "setup_dir": str(setup_dir), "run_dir": str(run_dir)}
    if config.template_dir:
        dump["template_dir"] = [config.template_dir]
    try:
        return Runner.model_validate(dump)
    except Exception as exc:
        msg = f"simulation does not validate: {exc}"
        raise SimulationExecutionError(SimulationErrorCode.PAYLOAD_INVALID, msg) from exc


@dataclass
class PreparedSubmit:
    """A validated, ready-to-execute submit command."""

    payload: SimulationPayload
    params: SubmitParams
    runner: Any
    config: SubmitConfig


def prepare_submit(
    *,
    payload_path: str,
    header: dict[str, Any],
    params: dict[str, Any] | None,
    config: SubmitConfig,
    local_provenance: dict[str, str],
) -> PreparedSubmit:
    """Validate a submit command and rebuild its runner.

    Every check here happens *before* an ``accepted`` ack is sent: a failure means
    the command was never accepted, so the error belongs in the ack (design
    section 2.2).  Stage failures during execution are reported as events
    instead (see :func:`execute_submit`).

    Args:
        payload_path: Server-generated path of the payload file.
        header: The command's provenance header.
        params: The command's build/run flags.
        config: Cluster-local submit policy.
        local_provenance: This install's provenance tuple.

    Returns:
        The validated payload, flags and fresh runner.

    Raises:
        SimulationExecutionError: On any validation failure.

    """
    try:
        resolved = validate_message_path(payload_path, str(config.message_dir))
    except UnsafePathError as exc:
        raise SimulationExecutionError(SimulationErrorCode.PATH_UNSAFE, str(exc)) from exc
    try:
        raw = Path(resolved).read_bytes()
    except OSError as exc:
        msg = f"cannot read payload: {exc}"
        raise SimulationExecutionError(SimulationErrorCode.PAYLOAD_UNREADABLE, msg) from exc
    check_payload_hash(raw, header)

    try:
        payload = SimulationPayload.model_validate_json(raw)
    except ValueError as exc:
        raise SimulationExecutionError(SimulationErrorCode.PAYLOAD_INVALID, str(exc)) from exc
    if payload.payload_hash != str(header.get("payload_hash", "")):
        msg = "payload/header hash mismatch"
        raise SimulationExecutionError(SimulationErrorCode.HASH_MISMATCH, msg)
    try:
        payload.check_allowlist()
    except UnsupportedPayloadError as exc:
        raise SimulationExecutionError(SimulationErrorCode.UNSUPPORTED, str(exc)) from exc
    mismatches = provenance_mismatches(payload, local_provenance)
    if mismatches:
        raise SimulationExecutionError(SimulationErrorCode.VERSION_MISMATCH, "; ".join(mismatches))

    submit_params = SubmitParams.model_validate(params or {})
    local_submit = _detect_submit_system()
    if local_submit and local_submit != submit_params.submit_system:
        msg = f"cluster tbg_submit={local_submit!r} but the command requests {submit_params.submit_system!r}"
        raise SimulationExecutionError(SimulationErrorCode.SUBMIT_SYSTEM_MISMATCH, msg)

    return PreparedSubmit(
        payload=payload,
        params=submit_params,
        runner=_runner_from_payload(payload, config),
        config=config,
    )


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
    flags = prepared.params.as_flags()
    # The cluster-local preset is a default: an explicit command flag wins.
    if prepared.config.preset is not None and flags.get("build_preset") is None:
        flags["build_preset"] = prepared.config.preset
    # build stage: generate the setup (must not pre-exist).
    try:
        runner.generate(**flags)
    except Exception as exc:
        msg = f"generate failed: {exc}"
        raise SimulationExecutionError(SimulationErrorCode.GENERATE_FAILED, msg, SimulationStage.BUILD) from exc

    # Submit stage: run the workflow; job id from submission_information.txt.
    try:
        await asyncio.to_thread(runner.run)
    except Exception as exc:
        msg = f"workflow failed: {exc}"
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
