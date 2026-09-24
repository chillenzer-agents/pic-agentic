# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""M2 ``submit_simulation`` RCP messages (design sections 4.1, 8.2).

Wire format (decided before implementation): the payload carries a
``pypicongpu.Runner`` *spec* -- only the ``sim`` field -- and never the runner's
cluster-local directories.  The MCP server writes the payload to a
server-generated path on the shared file system; the simclient re-validates the
path, the provenance tuple and the payload hash *before* it imports PIConGPU or
writes anything, then rebuilds a fresh ``Runner`` with cluster-local
``setup_dir``/``run_dir``/``template_dir`` (design section 6.4, gap 3).

The payload itself contains no ``rc_params``: those are cluster-local (the
``picongpurc.toml``) and some of their fields are shell code (design section
4.1, milestone note).

The module stays import-safe without PIConGPU; importing/validating the runner
is the simclient's job (the ``sim`` extra).
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, computed_field

from pic_agentic.rcp import Kind, RcpMessage, SenderRole, canonical_bytes, new_cmd_id
from pic_agentic.version import WIRE_FORMAT_VERSION

#: The only top-level key a transmitted runner spec may carry.
ALLOWED_SIMULATION_KEYS = frozenset({"sim"})

#: Default submit command; a payload asking for anything else is rejected.
DEFAULT_SUBMIT_SYSTEM = "sbatch"


class UnsupportedPayloadError(ValueError):
    """Raised when a payload carries fields outside the accepted wire schema."""


class SimulationType(StrEnum):
    """RCP ``type`` values of the M2 ``submit_simulation`` exchange."""

    COMMAND = "rcp.simulation_submit"
    ACK = "rcp.simulation_submit_ack"
    EVENT = "rcp.simulation_event"


class SimulationState(StrEnum):
    """Coarse lifecycle state reported in acks and events (gap 9)."""

    ACCEPTED = "accepted"
    SUBMITTED = "simulation.submitted"
    RESULTS_READY = "results.ready"
    FAILED = "simulation.failed"


class SimulationStage(StrEnum):
    """Which pipeline stage a failure occurred in."""

    BUILD = "build"
    PREPARE = "prepare"
    SUBMIT = "submit"
    RUN = "run"


class SubmitParams(BaseModel):
    """Build/run flags carried alongside the payload (design section 4.1).

    Only flags that are not cluster-local policy are accepted; unlike the
    ``rc_params`` they are validated JSON scalars, never shell code.
    """

    model_config = ConfigDict(extra="forbid")

    build_jobs: int | None = None
    build_cmake: str | None = None
    build_preset: int | None = None
    build_force: bool = False
    cfg_file: str | None = None
    #: The simclient enforces this; the workflow default is ``"bash"`` (local).
    submit_system: str = DEFAULT_SUBMIT_SYSTEM
    overwrite_vars: dict[str, str] | None = None

    def as_flags(self) -> dict[str, Any]:
        """Return only the explicitly set flags for ``Runner.generate(**flags)``.

        Returns:
            A mapping with ``None`` values dropped.

        """
        return {k: v for k, v in self.model_dump().items() if v is not None}


def simulation_spec_from_runner_dump(runner_dump: dict[str, Any]) -> dict[str, Any]:
    """Reduce a full ``Runner.model_dump(mode="json")`` to the wire spec.

    The cluster-local directories are deliberately dropped: the simclient always
    sets its own, and a payload trying to set them fails
    :func:`SimulationPayload.check_allowlist`.

    Args:
        runner_dump: A full runner dump as produced by the pinned PIConGPU.

    Returns:
        ``{"sim": <pypicongpu Simulation dump>}``.

    Raises:
        UnsupportedPayloadError: If the dump has no ``sim`` field.

    """
    if "sim" not in runner_dump:
        msg = "runner dump has no 'sim' field"
        raise UnsupportedPayloadError(msg)
    return {"sim": runner_dump["sim"]}


class SimulationPayload(BaseModel):
    """The serialised simulation plus its provenance tuple (design section 2.2).

    ``extra="forbid"`` rejects unknown top-level fields; the nested
    ``simulation`` mapping is separately allow-listed by
    :meth:`check_allowlist`.
    """

    model_config = ConfigDict(extra="forbid")

    wire_format_version: int = WIRE_FORMAT_VERSION
    picongpu_version: str
    picongpu_revision: str = ""
    schema_hash: str
    simulation: dict[str, Any]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def payload_hash(self) -> str:
        """SHA-256 of the canonical simulation bytes."""
        return hashlib.sha256(canonical_bytes(self.simulation)).hexdigest()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sim_id(self) -> str:
        """8-hex simulation id derived from :attr:`payload_hash`."""
        return self.payload_hash[:8]

    def check_allowlist(self) -> None:
        """Reject any simulation key outside :data:`ALLOWED_SIMULATION_KEYS`.

        Raises:
            UnsupportedPayloadError: If the simulation mapping carries an
                unsupported (or absent) top-level key.

        """
        keys = set(self.simulation)
        unsupported = keys - ALLOWED_SIMULATION_KEYS
        missing = ALLOWED_SIMULATION_KEYS - keys
        if unsupported or missing:
            parts = []
            if unsupported:
                parts.append(f"unsupported field(s): {', '.join(sorted(unsupported))}")
            if missing:
                parts.append(f"missing field(s): {', '.join(sorted(missing))}")
            raise UnsupportedPayloadError("; ".join(parts))

    def counts(self) -> dict[str, Any]:
        """Return a small redaction-safe summary for logs and acks.

        Returns:
            The provenance header fields plus the simulation id.

        """
        return {
            "wire_format_version": self.wire_format_version,
            "picongpu_version": self.picongpu_version,
            "picongpu_revision": self.picongpu_revision,
            "schema_hash": self.schema_hash,
            "sim_id": self.sim_id,
            "payload_hash": self.payload_hash,
        }

    @classmethod
    def build(
        cls,
        *,
        picongpu_version: str,
        picongpu_revision: str,
        schema_hash: str,
        runner_dump: dict[str, Any],
        wire_format_version: int = WIRE_FORMAT_VERSION,
    ) -> SimulationPayload:
        """Build a payload from a full runner dump and provenance values.

        Args:
            picongpu_version: The sender's PIConGPU version string.
            picongpu_revision: The sender's pinned revision.
            schema_hash: The sender's ``Runner`` schema hash.
            runner_dump: A full ``Runner.model_dump(mode="json")``.
            wire_format_version: The payload contract version.

        Returns:
            The validated payload.

        """
        return cls(
            wire_format_version=wire_format_version,
            picongpu_version=picongpu_version,
            picongpu_revision=picongpu_revision,
            schema_hash=schema_hash,
            simulation=simulation_spec_from_runner_dump(runner_dump),
        )


def provenance_mismatches(payload: SimulationPayload, local: dict[str, str]) -> list[str]:
    """Compare the payload provenance tuple against the local install.

    A blank revision on either side is treated as "unknown" and skipped, so an
    editable/local install without ``vcs_info`` does not spuriously reject a
    payload; the version and schema hash are always compared.

    Args:
        payload: The received payload.
        local: The local provenance tuple from ``version.local_provenance()``.

    Returns:
        Human-readable mismatch descriptions (empty when compatible).

    """
    mismatches: list[str] = []
    if payload.wire_format_version != WIRE_FORMAT_VERSION:
        mismatches.append(f"wire_format_version {payload.wire_format_version} != {WIRE_FORMAT_VERSION}")
    if local["picongpu_version"] and payload.picongpu_version != local["picongpu_version"]:
        mismatches.append(f"picongpu_version {payload.picongpu_version!r} != {local['picongpu_version']!r}")
    if local["schema_hash"] and payload.schema_hash != local["schema_hash"]:
        mismatches.append("schema_hash differs")
    payload_rev = payload.picongpu_revision
    local_rev = local["picongpu_revision"]
    if payload_rev and local_rev and payload_rev != local_rev:
        mismatches.append(f"picongpu_revision {payload_rev[:12]} != {local_rev[:12]}")
    return mismatches


def build_submit_command(
    *,
    sim: str,
    seq: int,
    payload_path: str,
    payload: SimulationPayload,
    params: SubmitParams | None = None,
    cmd_id: str | None = None,
    in_reply_to: str | None = None,
) -> RcpMessage:
    """Build the MCP-server-to-simclient ``submit_simulation`` command.

    Args:
        sim: Simulation id.
        seq: Per-sender sequence number.
        payload_path: Server-generated absolute path of the payload file.
        payload: The payload written to ``payload_path``.
        params: Optional build/run flags.
        cmd_id: Optional command id (generated when omitted).
        in_reply_to: Optional transport event id being replied to.

    Returns:
        The unsigned ``rcp.simulation_submit`` command.

    """
    return RcpMessage(
        sim=sim,
        kind=Kind.COMMAND,
        type=SimulationType.COMMAND,
        seq=seq,
        sender_role=SenderRole.MCP_SERVER,
        in_reply_to=in_reply_to,
        payload={
            "cmd_id": cmd_id or new_cmd_id(),
            "payload_path": payload_path,
            "header": payload.counts(),
            "params": (params or SubmitParams()).model_dump(mode="json"),
        },
    )


def build_submit_ack(
    *,
    sim: str,
    seq: int,
    cmd_id: str,
    sim_id: str,
    state: SimulationState,
    in_reply_to: str | None,
    job_id: int | None = None,
    error: str | None = None,
    error_code: str | None = None,
) -> RcpMessage:
    """Build the simclient's acknowledgement of a submit command.

    Returns:
        The unsigned ``rcp.simulation_submit_ack`` message.

    """
    payload: dict[str, Any] = {"cmd_id": cmd_id, "sim_id": sim_id, "state": state.value, "job_id": job_id}
    if error:
        payload["error"] = error
    if error_code:
        payload["error_code"] = error_code
    return RcpMessage(
        sim=sim,
        kind=Kind.ACK,
        type=SimulationType.ACK,
        seq=seq,
        sender_role=SenderRole.SIMCLIENT,
        in_reply_to=in_reply_to,
        payload=payload,
    )


def build_submit_event(
    *,
    sim: str,
    seq: int,
    cmd_id: str,
    sim_id: str,
    state: SimulationState,
    job_id: int | None = None,
    stage: SimulationStage | None = None,
    error: str | None = None,
    error_code: str | None = None,
    submit_system: str | None = None,
) -> RcpMessage:
    """Build one M2 lifecycle event.

    Returns:
        The unsigned ``rcp.simulation_event`` event.

    """
    payload: dict[str, Any] = {"cmd_id": cmd_id, "sim_id": sim_id, "state": state.value, "job_id": job_id}
    if stage is not None:
        payload["stage"] = stage.value
    if submit_system is not None:
        payload["submit_system"] = submit_system
    if error:
        payload["error"] = error
    if error_code:
        payload["error_code"] = error_code
    return RcpMessage(
        sim=sim,
        kind=Kind.EVENT,
        type=SimulationType.EVENT,
        seq=seq,
        sender_role=SenderRole.SIMCLIENT,
        payload=payload,
    )
