# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""M2 ``submit_simulation`` RCP messages (design sections 4.1, 8.2).

Wire format: the signed command carries a ``pypicongpu.Runner`` *spec* inline
-- only the ``sim`` field -- and never the runner's cluster-local directories.
The simclient validates the provenance tuple and the payload hash, then rebuilds
a fresh ``Runner`` with cluster-local ``setup_dir``/``run_dir``/``template_dir``
(design section 6.4, gap 3).

The payload travels *inside* the Matrix ``m.room.message`` (not as a shared-FS
file), so the MCP server and the simclient need no common file system: the
container/cluster split of the deployment is preserved.  The cost is that the
command is bounded by the homeserver's event-size limit, hence
:data:`MAX_INLINE_PAYLOAD_BYTES` (a realistic simulation is a few KiB; the
largest stress case measured ~53 KiB).

The payload itself contains no ``rc_params``: those are cluster-local (the
``picongpurc.toml``) and some of their fields are shell code (design section
4.1, milestone note).

The module stays import-safe without PIConGPU; importing/validating the runner
is the simclient's job (the ``sim`` extra).
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, FiniteFloat, computed_field, field_validator, model_validator

from pic_agentic.rcp import Kind, RcpMessage, SenderRole, canonical_bytes, encode_wire, new_cmd_id
from pic_agentic.version import WIRE_FORMAT_VERSION

#: The only top-level key a transmitted runner spec may carry.
ALLOWED_SIMULATION_KEYS = frozenset({"sim"})

#: Default submit command; a payload asking for anything else is rejected.
DEFAULT_SUBMIT_SYSTEM = "sbatch"

#: Cap on the *encoded* event content carried in one Matrix command.  Synapse's
#: default limit is 64 KiB for the whole event content, so 48 KiB leaves headroom
#: for the envelope, the human-readable body and the params.  The size check
#: counts the **escaped** payload (it is embedded as a JSON string, so its
#: quotes/backslashes are doubled) plus :data:`_ENVELOPE_ALLOWANCE_BYTES`, i.e.
#: what actually goes on the wire -- not the inner simulation object.
MAX_INLINE_PAYLOAD_BYTES = 48 * 1024

#: Reserved budget for the signed envelope, the room body line, the copy of the
#: provenance header and the params that travel alongside the payload in the
#: same event content.
_ENVELOPE_ALLOWANCE_BYTES = 4 * 1024

#: ``cfg_file``: a relative path to a ``.cfg`` inside the generated setup.
#: Absolute paths, ``..`` and shell metacharacters are rejected outright so the
#: value can never be interpreted as shell code by the cluster's ``tbg`` (which
#: ``eval``\\s the configuration file name).
_CFG_FILE_RE = re.compile(r"^[A-Za-z0-9._/-]+\.cfg$")

#: One ``NAME=value`` overwrite entry.  The strict charset excludes every shell
#: metacharacter (spaces, ``$``, backticks, ``;``, quotes, ``(``/``)``, ``<``,
#: ``>``, ``|``, ``&``, ``\\``, ``*``, ``?``, ``~``, ``%``); ``tbg`` applies
#: these with ``eval``/``for word in $extra_op``, so only inert ``name=value``
#: data may pass.
_OVERWRITE_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9._:/+-]*$")


#: Key of the embedded :class:`SimulationPayload` inside the command.
#:
#: The value is the payload's canonical JSON as a *string*, not a nested
#: object: Matrix's canonical JSON (Synapse) rejects any floating-point value
#: in an event content object, and the simulation is full of floats (every
#: length, density and timestep).  Floats inside a JSON string are fine, so the
#: payload is serialised once and parsed on receipt.
PAYLOAD_KEY = "payload"


class PayloadTooLargeError(ValueError):
    """Raised when a simulation is too large to send inline in one command."""


class UnsupportedPayloadError(ValueError):
    """Raised when a payload carries fields outside the accepted wire schema."""


class SimulationType(StrEnum):
    """RCP ``type`` values of the M2 ``submit_simulation`` exchange."""

    COMMAND = "rcp.simulation_submit"
    ACK = "rcp.simulation_submit_ack"
    EVENT = "rcp.simulation_event"
    #: M2b request/response: the MCP server asks the simclient (which sees the
    #: cluster) for live status / logs; the simclient answers with an ``*_ACK``.
    STATUS_COMMAND = "rcp.status_request"
    STATUS_ACK = "rcp.status_ack"
    LOGS_COMMAND = "rcp.logs_request"
    LOGS_ACK = "rcp.logs_ack"
    #: M3 request/response: control verbs (checkpoint/stop/cancel) and
    #: results access (describe/slice/read/export), both demand-driven pulls.
    CONTROL_COMMAND = "rcp.control_request"
    CONTROL_ACK = "rcp.control_ack"
    RESULT_COMMAND = "rcp.result_request"
    RESULT_ACK = "rcp.result_ack"


class SimulationState(StrEnum):
    """Coarse lifecycle state reported in acks and events (gap 9).

    ``workflow.finished`` marks the end of the CWL workflow the simclient
    drives (build -> prepare -> submit -> organize).  It deliberately does not
    claim the SLURM job finished, nor that results exist: the job may still be
    queued or running.

    The ``simulation.job_*`` events (M2b) follow the SLURM job itself:
    ``job_running`` once it starts, ``job_finished`` on a clean terminal state,
    ``job_failed`` otherwise.  ``results.ready`` is emitted only after
    ``job_finished`` and a present ``run_dir/simOutput``.
    """

    ACCEPTED = "accepted"
    SUBMITTED = "simulation.submitted"
    WORKFLOW_FINISHED = "workflow.finished"
    JOB_RUNNING = "simulation.job_running"
    JOB_FINISHED = "simulation.job_finished"
    JOB_FAILED = "simulation.job_failed"
    STEP_FINISHED = "simulation.step_finished"
    RESULTS_READY = "results.ready"
    FAILED = "simulation.failed"
    #: M3 control outcomes: a checkpoint was requested (non-terminal; the
    #: simulation keeps running) and a graceful stop/cancel ended the job.
    CHECKPOINT = "simulation.checkpoint"
    CANCELLED = "simulation.cancelled"


class SimulationOp(StrEnum):
    """M3 control verbs (``rcp.control_request``).

    Mapped 1:1 onto PIConGPU's signal handler (``pmacc/simulationControl/
    signal.cpp``) delivered to every MPI rank via ``scancel --signal=<SIG>``:
    ``checkpoint``->``SIGUSR1`` (dump a checkpoint at the next common step and
    keep running), ``stop``->``SIGTERM`` (clean stop at the next step),
    ``checkpoint_and_stop``->``SIGALRM`` (checkpoint *then* stop) and
    ``cancel``->plain ``scancel`` (immediate job death).  ``scontrol signal`` is
    NOT a Slurm command (a stale early-design assumption).
    """

    CHECKPOINT = "checkpoint"
    STOP = "stop"
    CHECKPOINT_AND_STOP = "checkpoint_and_stop"
    CANCEL = "cancel"


#: Control op -> the signal delivered to the job's tasks; ``None`` means the op
#: uses a plain ``scancel`` instead of ``scancel --signal``.
CONTROL_SIGNAL: dict[SimulationOp, str | None] = {
    SimulationOp.CHECKPOINT: "USR1",
    SimulationOp.STOP: "TERM",
    SimulationOp.CHECKPOINT_AND_STOP: "ALRM",
    SimulationOp.CANCEL: None,
}

#: Control ops that require the job to be ``RUNNING`` (a queued job has no
#: process with the handlers installed).  ``CANCEL`` works pre-launch too.
CONTROL_REQUIRES_RUNNING = frozenset(
    {SimulationOp.CHECKPOINT, SimulationOp.STOP, SimulationOp.CHECKPOINT_AND_STOP},
)


class ResultOp(StrEnum):
    """M3 results-access verbs (``rcp.result_request``).

    ``describe`` scans the linked output directory (no openPMD needed);
    ``slice``/``stats``/``image``/``read`` reduce on the cluster via the
    optional openPMD reader; ``export`` returns a transfer ticket and never
    moves the bulk data itself.
    """

    DESCRIBE = "describe"
    SLICE = "slice"
    STATS = "stats"
    IMAGE = "image"
    EXPORT = "export"
    READ = "read"
    #: Milestone A: compose the RO-Crate + pypicongpu metadata + a deterministic
    #: ``answer`` summary (no LLM call) for one run.
    ANALYZE = "analyze"
    #: Secure tailored analysis: evaluate a validated declarative program
    #: (selectors + expression AST + reductions) on the cluster.  No code is
    #: executed; see :mod:`pic_agentic.analysis_program`.
    COMPUTE = "compute"
    #: G1: run one of PIConGPU's shipped plugin readers on the cluster and
    #: return a bounded summary.  The reader is selected by name from
    #: :data:`PLUGIN_READER_NAMES`.
    PLUGIN = "plugin"


#: Registered PIConGPU plugin readers, by the name carried in
#: :attr:`ResultParams.reader`.  The first four are the text plugins (the field
#: energy monitor is parsed by the engine's own stdlib parser, not a shipped
#: ``picongpu.extra.plugins.data`` reader); the last four are the openPMD/image
#: readers (phase space, radiation, calorimeter, PNG).  The full registry
#: (filename pattern, reader class, allowed kwargs) lives in
#: :mod:`pic_agentic.results`; the names are frozen here so the wire model can
#: validate ``reader`` without importing the optional engine.
PLUGIN_READER_NAMES = (
    "energy_histogram",
    "energy_fields",
    "emittance",
    "transition_radiation",
    "phase_space",
    "radiation",
    "calorimeter",
    "png",
)


class ClientCapabilities(BaseModel):
    """What one deployed cluster client can handle.

    Advertised by the simclient in its ``hello`` ack so the server can tell a
    version drift *before* sending an op the older client cannot handle.  The
    sets are the enum member values compiled into the client, so a client that
    predates a new op simply does not list it.

    Each set is ``None`` when the advertisement did not carry it.  ``None``
    means *unknown*, not *empty*: a partial or forward-compatible advertisement
    that omits a set must never be read as "supports nothing" and must never
    block a request.  Only an explicitly advertised, non-``None`` set can
    reject.
    """

    model_config = ConfigDict(extra="forbid")

    #: The client's package version, for a drift message that names both sides.
    client_version: str = ""
    #: ``ResultOp`` values the client recognises, or None when not advertised.
    result_ops: frozenset[str] | None = None
    #: ``SimulationOp`` values the client recognises, or None when not advertised.
    control_ops: frozenset[str] | None = None
    #: Request ``type`` values (``rcp.*``) whose handler is compiled in, or None
    #: when not advertised.
    supported_types: frozenset[str] | None = None

    @classmethod
    def current(cls, *, client_version: str = "") -> ClientCapabilities:
        """Return the capabilities of this (running) client code.

        Returns:
            The capability set derived from the compiled enums and the enabled
            handlers.

        """
        return cls(
            client_version=client_version,
            result_ops=frozenset(op.value for op in ResultOp),
            control_ops=frozenset(op.value for op in SimulationOp),
            supported_types=frozenset(
                {
                    SimulationType.COMMAND.value,
                    SimulationType.STATUS_COMMAND.value,
                    SimulationType.LOGS_COMMAND.value,
                    SimulationType.CONTROL_COMMAND.value,
                    SimulationType.RESULT_COMMAND.value,
                }
            ),
        )

    def supports_result_op(self, op: str) -> bool:
        """Whether the client advertised support for a result op.

        Returns:
            True when the op is listed, or when ``result_ops`` was not
            advertised (unknown, so never a blocker).

        """
        return self.result_ops is None or op in self.result_ops

    def supports_control_op(self, op: str) -> bool:
        """Whether the client advertised support for a control op.

        Returns:
            True when the op is listed, or when ``control_ops`` was not
            advertised (unknown, so never a blocker).

        """
        return self.control_ops is None or op in self.control_ops

    def supports_type(self, request_type: str) -> bool:
        """Whether the client advertised a handler for a request type.

        Returns:
            True when the type is listed, or when ``supported_types`` was not
            advertised (unknown, so never a blocker).

        """
        return not request_type or self.supported_types is None or request_type in self.supported_types

    def unsupported(self, *, op: ResultOp | SimulationOp | None = None, request_type: str = "") -> str | None:
        """Return the capability label this client cannot handle, if any.

        Args:
            op: The requested result/control op, when the request carries one.
            request_type: The request ``type`` value.

        Returns:
            A human-readable label such as ``plugin``, ``control op 'stop'`` or
            the request type, or None when the client supports the request or
            did not advertise the relevant set (unknown is never a blocker).

        """
        if request_type and not self.supports_type(request_type):
            return request_type
        if isinstance(op, ResultOp) and not self.supports_result_op(op.value):
            return op.value
        if isinstance(op, SimulationOp) and not self.supports_control_op(op.value):
            return op.value
        return None


def client_capability_mismatch(capabilities: ClientCapabilities, *, op: str = "", request_type: str = "") -> str:
    """Compose the actionable version-drift message for an unsupported request.

    Args:
        capabilities: The client's advertised capabilities.
        op: The unknown op name (``ResultOp``/``SimulationOp`` value).
        request_type: The unknown request ``type`` value.

    Returns:
        A user-facing message naming the missing capability and the client's
        reported version, leaving it to the user to decide which side to update
        (the client could be older *or* the server ahead of it).

    """
    if op:
        missing = f"operation {op!r}"
    elif request_type:
        missing = f"request type {request_type!r}"
    else:
        missing = "this request"
    version = f" (client version {capabilities.client_version})" if capabilities.client_version else ""
    return (
        f"cluster client does not support {missing}{version}: "
        "the deployed simclient did not advertise this capability; "
        "check for a version drift between the client and the MCP server and update the side that is behind"
    )


#: Upper bound on the *escaped* wire size of one result ack (reduced arrays,
#: text tails or a thumbnail).  Deliberately the same 48 KiB budget as
#: :data:`MAX_INLINE_PAYLOAD_BYTES`: Synapse rejects event content above its
#: (64 KiB default) limit, and a rejected ack would surface as a silent pull
#: timeout rather than a clean error.  Results above this are answered with a
#: ``RESULT_TOO_LARGE`` error instead of being sent.
MAX_RESULT_BYTES = 48 * 1024

#: Cap on the number of reduced points a ``slice`` may return.
SLICE_MAX_POINTS = 4096

#: Cap on the bytes of a single ``read`` (text) response.
RESULT_TEXT_MAX_BYTES = 48 * 1024

#: ``path`` argument of a result request: a relative path under ``simOutput``.
#: The strict charset excludes absolute paths, ``..`` and shell
#: metacharacters; the client additionally refuses escapes from its base.
_RESULT_REL_RE = re.compile(r"^[A-Za-z0-9._/-]+$")

#: ``species`` filter argument: a PIConGPU species or particle-filter name, a
#: bare identifier.  Kept stricter than ``path`` (no ``/``: a species never
#: contains a directory separator) so a plugin request cannot name a file.
_RESULT_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


class SimulationStage(StrEnum):
    """Which pipeline stage a failure occurred in."""

    BUILD = "build"
    PREPARE = "prepare"
    SUBMIT = "submit"
    RUN = "run"


#: Progress events at least this far apart (percent) are emitted; the terminal
#: event is always emitted.  Matches the design's condensation (section 4.2).
PROGRESS_EVENT_STEP_PERCENT = 25

#: Log streams ``get_logs`` can request.
LOG_STREAMS = ("stdout", "stderr", "workflow")


class SubmitParams(BaseModel):
    """Build/run flags carried alongside the payload (design section 4.1).

    Only flags that are not cluster-local policy are accepted; unlike the
    ``rc_params`` they are validated JSON scalars, never shell code.

    The field names mirror the design's tool signature (``build_*``/``cfg_*``);
    :meth:`picongpu_flags` maps them to the aliases the pinned
    ``PicBuildFlags``/``TBGFlags`` models actually accept (``jobs``, ``cmake``,
    ``preset``, ``force``, ``cfg``, ``submit``).  Passing the field names
    straight through is silently ignored by pydantic (their validation aliases
    do not include the ``build_`` prefix; ``populate_by_name`` is off), which
    would drop ``submit_system`` and run the job locally via ``bash``.
    """

    model_config = ConfigDict(extra="forbid")

    build_jobs: int | None = None
    build_cmake: str | None = None
    build_preset: int | None = None
    build_force: bool = False
    cfg_file: str | None = None
    #: The submit command; the simclient enforces its local ``tbg_submit``
    #: matches this.  A NON-sbatch value cannot be requested over the wire:
    #: ``prepare_submit`` rejects anything but ``sbatch`` outright.
    submit_system: str = DEFAULT_SUBMIT_SYSTEM
    overwrite_vars: list[str] | None = None

    @field_validator("cfg_file")
    @classmethod
    def _validate_cfg_file(cls, value: str | None) -> str | None:
        r"""Reject a ``cfg_file`` that is not a relative, inert ``.cfg`` path.

        The cluster's ``tbg`` ``eval``\s the configuration file name, so an
        arbitrary path or any shell metacharacter would be wire-supplied shell
        code -- forbidden by the design (sections 5/9).

        Returns:
            The validated path, or ``None`` when unset.

        Raises:
            ValueError: If the path is absolute, escapes upward, or contains
                characters outside the safe set.

        """
        if value is None:
            return None
        if not _CFG_FILE_RE.fullmatch(value) or ".." in value.split("/") or value.startswith("/"):
            msg = f"cfg_file must be a relative path matching {_CFG_FILE_RE.pattern!r}, got {value!r}"
            raise ValueError(msg)
        return value

    @field_validator("overwrite_vars")
    @classmethod
    def _validate_overwrite_vars(cls, value: list[str] | None) -> list[str] | None:
        """Reject any ``overwrite_vars`` entry that is not inert ``NAME=value``.

        ``tbg`` expands each entry with ``eval``/``for word in $extra_op``, so a
        value such as ``PARAM=$(cmd)`` or one containing whitespace/backticks
        would be remote code execution on the submission node.

        Returns:
            The validated list, or ``None`` when unset.

        Raises:
            ValueError: If any entry contains shell metacharacters or does not
                match ``NAME=value``.

        """
        if value is None:
            return None
        for entry in value:
            if not _OVERWRITE_VAR_RE.fullmatch(entry):
                msg = f"overwrite_vars entries must match {_OVERWRITE_VAR_RE.pattern!r}, got {entry!r}"
                raise ValueError(msg)
        return value

    def picongpu_flags(self) -> dict[str, Any]:
        """Map to the aliases ``Runner.generate(**flags)`` forwards to picongpu.

        Returns:
            The flags with ``build_``/``cfg_`` names translated and unset
            options dropped (so picongpu keeps its own defaults).

        """
        mapping = {
            "build_jobs": "jobs",
            "build_cmake": "cmake",
            "build_preset": "preset",
            "cfg_file": "cfg",
            "submit_system": "submit",
            # The pinned TBGFlags accepts overwrite_vars only under the short
            # ``o`` alias (it has no populate_by_name), so the long name alone
            # would be silently ignored.
            "overwrite_vars": "o",
        }
        flags = {alias: getattr(self, field) for field, alias in mapping.items() if getattr(self, field) is not None}
        if self.build_force:
            flags["force"] = True
        return flags


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

    #: Required: a payload that omits the version is rejected rather than
    #: silently treated as the current version (design section 2.2).  The
    #: sender always supplies it via :meth:`build`.
    wire_format_version: int
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


def payload_wire_size(payload: SimulationPayload) -> int:
    """Return the homeserver-facing size of an inline payload, in bytes.

    This is the same measure :func:`payload_wire_bytes` enforces: the canonical
    payload body is carried as a JSON *string* inside the event content, so the
    escaped form (``json.dumps`` doubles every quote/backslash) plus
    :data:`_ENVELOPE_ALLOWANCE_BYTES` is what the homeserver's event limit sees.
    Exposed so the dry-run builder can report the size it would send without
    building the binary body.

    Args:
        payload: The payload to measure.

    Returns:
        The escaped body size plus the envelope allowance.

    """
    body = payload.model_dump_json(exclude_computed_fields=True)
    return len(json.dumps(body, ensure_ascii=True).encode("ascii")) + _ENVELOPE_ALLOWANCE_BYTES


def payload_wire_bytes(payload: SimulationPayload) -> bytes:
    """Serialise a payload for inline transport, enforcing the size cap.

    The cap is checked against the size the payload actually occupies on the
    wire: the payload body is a JSON *string* embedded in the event content, so
    its quotes/backslashes are escaped once more.  Measuring the inner
    simulation object (as an earlier version did) undercounted by up to 2x and
    let an "under-cap" payload produce an over-64-KiB Matrix event.

    The computed fields (``payload_hash``/``sim_id``) are excluded: they are
    recomputed on read and would otherwise be rejected by
    ``extra="forbid"``.

    Args:
        payload: The payload to serialise.

    Returns:
        The canonical JSON bytes to embed in the command.

    Raises:
        PayloadTooLargeError: If the encoded payload plus
            :data:`_ENVELOPE_ALLOWANCE_BYTES` exceeds
            :data:`MAX_INLINE_PAYLOAD_BYTES`.

    """
    body = payload.model_dump_json(exclude_computed_fields=True).encode("utf-8")
    # The body is carried as a JSON string inside the event content, so measure
    # the escaped form (json.dumps doubles every quote/backslash) plus the
    # envelope budget -- that is what the homeserver's 64 KiB event limit sees.
    size = payload_wire_size(payload)
    if size > MAX_INLINE_PAYLOAD_BYTES:
        msg = (
            f"encoded simulation payload is ~{size} bytes; the inline limit is "
            f"{MAX_INLINE_PAYLOAD_BYTES} (out-of-band payload transport is not implemented yet)"
        )
        raise PayloadTooLargeError(msg)
    return body


def build_submit_command(
    *,
    sim: str,
    seq: int,
    payload: SimulationPayload,
    params: SubmitParams | None = None,
    cmd_id: str | None = None,
    in_reply_to: str | None = None,
) -> RcpMessage:
    """Build the MCP-server-to-simclient ``submit_simulation`` command.

    The payload travels inline in the signed envelope, so the command is
    self-contained: the simclient needs no shared file system to read it.

    Args:
        sim: Simulation id.
        seq: Per-sender sequence number.
        payload: The simulation payload to embed.
        params: Optional build/run flags.
        cmd_id: Optional command id (generated when omitted).
        in_reply_to: Optional transport event id being replied to.

    Returns:
        The unsigned ``rcp.simulation_submit`` command.  A simulation larger
        than :data:`MAX_INLINE_PAYLOAD_BYTES` is rejected by
        :func:`payload_wire_bytes` with :class:`PayloadTooLargeError`.

    """
    # The payload is carried as a JSON string (see PAYLOAD_KEY): Synapse
    # rejects floats in event-content objects, and the simulation has many.
    body = payload_wire_bytes(payload).decode("utf-8")
    return RcpMessage(
        sim=sim,
        kind=Kind.COMMAND,
        type=SimulationType.COMMAND,
        seq=seq,
        sender_role=SenderRole.MCP_SERVER,
        in_reply_to=in_reply_to,
        payload={
            "cmd_id": cmd_id or new_cmd_id(),
            "header": payload.counts(),
            PAYLOAD_KEY: body,
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
    manifest: dict[str, Any] | None = None,
    suspect: str | None = None,
) -> RcpMessage:
    """Build one M2 lifecycle event.

    Only the fields relevant to the event's ``state`` are carried; the rest stay
    absent so the payloads remain small and the room body readable.

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
    if results_linked is not None:
        payload["results_linked"] = results_linked
    if manifest is not None:
        payload["manifest"] = manifest
    if suspect:
        payload["suspect"] = suspect
    payload.update(
        {
            key: value
            for key, value in (
                ("step", step),
                ("percent", percent),
                ("walltime", walltime),
                ("avg_per_step", avg_per_step),
                ("eta_s", eta_s),
                ("slurm_state", slurm_state),
                ("exit_code", exit_code),
                ("core_hours", core_hours),
                ("gpu_hours", gpu_hours),
            )
            if value is not None
        },
    )
    return RcpMessage(
        sim=sim,
        kind=Kind.EVENT,
        type=SimulationType.EVENT,
        seq=seq,
        sender_role=SenderRole.SIMCLIENT,
        payload=payload,
    )


def build_status_command(
    *,
    sim: str,
    seq: int,
    sim_id: str,
    cmd_id: str | None = None,
    in_reply_to: str | None = None,
) -> RcpMessage:
    """Build the MCP-server-to-simclient live-status request (M2b).

    Returns:
        The unsigned ``rcp.status_request`` command.

    """
    return RcpMessage(
        sim=sim,
        kind=Kind.COMMAND,
        type=SimulationType.STATUS_COMMAND,
        seq=seq,
        sender_role=SenderRole.MCP_SERVER,
        in_reply_to=in_reply_to,
        payload={"cmd_id": cmd_id or new_cmd_id(), "sim_id": sim_id},
    )


def build_logs_command(
    *,
    sim: str,
    seq: int,
    sim_id: str,
    stream: str = "stdout",
    tail: int = 100,
    cmd_id: str | None = None,
    in_reply_to: str | None = None,
) -> RcpMessage:
    """Build the MCP-server-to-simclient log request (M2b).

    Returns:
        The unsigned ``rcp.logs_request`` command.

    Raises:
        ValueError: If ``stream`` is not one of :data:`LOG_STREAMS`.

    """
    if stream not in LOG_STREAMS:
        msg = f"unknown log stream {stream!r}; expected one of {LOG_STREAMS}"
        raise ValueError(msg)
    return RcpMessage(
        sim=sim,
        kind=Kind.COMMAND,
        type=SimulationType.LOGS_COMMAND,
        seq=seq,
        sender_role=SenderRole.MCP_SERVER,
        in_reply_to=in_reply_to,
        payload={"cmd_id": cmd_id or new_cmd_id(), "sim_id": sim_id, "stream": stream, "tail": tail},
    )


def build_status_ack(
    *,
    sim: str,
    seq: int,
    cmd_id: str,
    sim_id: str,
    in_reply_to: str | None,
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
    """Build the simclient's live-status response (M2b).

    ``suspect`` carries the "successful-but-empty" health flag (F4) when the
    status pull promotes a completed run to ``results.ready`` without a prior
    terminal event, so the flag cannot be lost through the status-pull door.

    Returns:
        The unsigned ``rcp.status_ack`` message.

    """
    payload: dict[str, Any] = {"cmd_id": cmd_id, "sim_id": sim_id, "state": state}
    payload.update(
        {
            key: value
            for key, value in (
                ("slurm_state", slurm_state),
                ("job_id", job_id),
                ("step", step),
                ("percent", percent),
                ("walltime", walltime),
                ("avg_per_step", avg_per_step),
                ("eta_s", eta_s),
                ("exit_code", exit_code),
                ("error", error),
                ("error_code", error_code),
                ("suspect", suspect),
            )
            if value is not None
        },
    )
    return RcpMessage(
        sim=sim,
        kind=Kind.ACK,
        type=SimulationType.STATUS_ACK,
        seq=seq,
        sender_role=SenderRole.SIMCLIENT,
        in_reply_to=in_reply_to,
        payload=payload,
    )


def build_logs_ack(
    *,
    sim: str,
    seq: int,
    cmd_id: str,
    sim_id: str,
    in_reply_to: str | None,
    stream: str,
    lines: list[str],
    total_lines: int,
    error: str | None = None,
    error_code: str | None = None,
) -> RcpMessage:
    """Build the simclient's log response (M2b).

    Returns:
        The unsigned ``rcp.logs_ack`` message.

    """
    payload: dict[str, Any] = {
        "cmd_id": cmd_id,
        "sim_id": sim_id,
        "stream": stream,
        "lines": lines,
        "total_lines": total_lines,
    }
    if error:
        payload["error"] = error
    if error_code:
        payload["error_code"] = error_code
    return RcpMessage(
        sim=sim,
        kind=Kind.ACK,
        type=SimulationType.LOGS_ACK,
        seq=seq,
        sender_role=SenderRole.SIMCLIENT,
        in_reply_to=in_reply_to,
        payload=payload,
    )


class ControlParams(BaseModel):
    """One M3 control request (``rcp.control_request``)."""

    model_config = ConfigDict(extra="forbid")

    sim_id: str
    op: SimulationOp


class ResultParams(BaseModel):
    """One M3 results request (``rcp.result_request``).

    Only the knobs relevant to the selected :class:`ResultOp` are meaningful;
    the rest stay unset.  ``path`` is a relative path under the run's linked
    ``simOutput`` directory and is validated against a strict charset so it can
    never escape that directory or reach a shell.
    """

    model_config = ConfigDict(extra="forbid")

    sim_id: str
    op: ResultOp
    path: str | None = None
    record: str | None = None
    component: str | None = None
    iteration: int | str | None = None
    axis: int | None = None
    index: int | None = None
    downsample: int | None = None
    stream: str | None = None
    tail: int | None = None
    #: The declarative analysis program for ``COMPUTE`` (validated by
    #: :class:`~pic_agentic.analysis_program.AnalysisProgram` before evaluation).
    program: dict[str, Any] | None = None
    #: The registered PIConGPU plugin reader for ``PLUGIN`` (one of
    #: :data:`PLUGIN_READER_NAMES`).
    reader: str | None = None
    #: The particle species whose plugin output is read (``PLUGIN``).
    species: str | None = None
    #: The particle-filter name (``PLUGIN``).  ``None`` means "unspecified" and
    #: is normalized to PIConGPU's default ``"all"`` filter by the reader path;
    #: leaving it unset keeps it off the wire for non-plugin result ops.
    species_filter: str | None = None
    #: Energy-histogram window lower/upper edge [keV] for ``PLUGIN`` with the
    #: ``energy_histogram`` reader.  When unset the reader derives the window
    #: from the populated bins, so the default always covers the populated range
    #: rather than clipping it to the old fixed 100--1000 keV window (F2).
    #: Both edges must be given together, be non-negative and have the maximum
    #: exceed the minimum.  When set they are forwarded on the wire for any
    #: ``PLUGIN`` reader; readers other than ``energy_histogram`` ignore them.
    min_kev: FiniteFloat | None = None
    max_kev: FiniteFloat | None = None

    @model_validator(mode="after")
    def _validate_energy_window(self) -> ResultParams:
        """Require a coherent, paired energy window when one is given.

        Returns:
            The validated model.

        Raises:
            ValueError: If only one edge is set, an edge is negative, or
                ``max_kev <= min_kev``.

        """
        if (self.min_kev is None) != (self.max_kev is None):
            msg = "min_kev and max_kev must be set together"
            raise ValueError(msg)
        if self.min_kev is not None and self.min_kev < 0:
            msg = f"min_kev ({self.min_kev}) must be non-negative"
            raise ValueError(msg)
        if self.max_kev is not None and self.max_kev < 0:
            msg = f"max_kev ({self.max_kev}) must be non-negative"
            raise ValueError(msg)
        if self.min_kev is not None and self.max_kev is not None and self.max_kev <= self.min_kev:
            msg = f"max_kev ({self.max_kev}) must be greater than min_kev ({self.min_kev})"
            raise ValueError(msg)
        return self

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str | None) -> str | None:
        """Reject a ``path`` that is absolute, escaping, or unsafe.

        Returns:
            The validated relative path, or ``None`` when unset.

        Raises:
            ValueError: If the path is unsafe.

        """
        if value is None:
            return None
        if value.startswith("/") or not _RESULT_REL_RE.match(value) or ".." in value.split("/"):
            msg = f"unsafe result path: {value!r}"
            raise ValueError(msg)
        return value

    @field_validator("stream")
    @classmethod
    def _validate_stream(cls, value: str | None) -> str | None:
        """Restrict a text ``read`` to the two captured streams.

        Returns:
            The validated stream name, or ``None`` when unset.

        Raises:
            ValueError: If the stream is not ``stdout``/``stderr``.

        """
        if value is None:
            return None
        if value not in {"stdout", "stderr"}:
            msg = f"unknown result stream {value!r}; expected 'stdout' or 'stderr'"
            raise ValueError(msg)
        return value

    @field_validator("reader")
    @classmethod
    def _validate_reader(cls, value: str | None) -> str | None:
        """Restrict a ``PLUGIN`` request to a registered reader name.

        Returns:
            The validated reader name, or ``None`` when unset.

        Raises:
            ValueError: If the name is not in :data:`PLUGIN_READER_NAMES`.

        """
        if value is None:
            return None
        if value not in PLUGIN_READER_NAMES:
            msg = f"unknown plugin reader {value!r}; expected one of {PLUGIN_READER_NAMES}"
            raise ValueError(msg)
        return value

    @field_validator("species", "species_filter")
    @classmethod
    def _validate_species_name(cls, value: str | None) -> str | None:
        """Reject a species/filter name outside the safe identifier charset.

        The value is interpolated into the output filename, so it must never
        carry a path separator or shell metacharacter.

        Returns:
            The validated name, or ``None`` when unset.

        Raises:
            ValueError: If the name contains an unsafe character.

        """
        if value is None:
            return None
        if not _RESULT_NAME_RE.match(value):
            msg = f"unsafe species name: {value!r}"
            raise ValueError(msg)
        return value

    @field_validator("program")
    @classmethod
    def _validate_program(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """Validate a ``COMPUTE`` program (and its source-size cap) eagerly.

        Validating here means an invalid or oversized program is rejected by the
        model, before it can reach the evaluator and before any data is read.

        Returns:
            The validated program dict, or ``None`` when unset.

        Raises:
            ValueError: If the program is invalid, too large, or non-finite.

        """
        if value is None:
            return None
        from pic_agentic.analysis_program import (  # ruff: ignore[import-outside-top-level] - avoids a protocol/models cycle
            MAX_SOURCE_BYTES,
            AnalysisProgram,
        )

        encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        if len(encoded) > MAX_SOURCE_BYTES:
            msg = f"analysis program exceeds {MAX_SOURCE_BYTES} bytes"
            raise ValueError(msg)
        try:
            AnalysisProgram.model_validate(value)
        except Exception as exc:
            msg = f"invalid analysis program: {exc}"
            raise ValueError(msg) from exc
        return value


class ResultRef(BaseModel):
    """A reference to one output file or directory (never its contents)."""

    model_config = ConfigDict(extra="forbid")

    path: str
    uri: str
    #: The sniffed format: ``openpmd-adios2``/``openpmd-hdf5``, ``text``,
    #: ``dir``, ``binary``, or a registered plugin reader name
    #: (:data:`PLUGIN_READER_NAMES`) when the filename matches one.  The match
    #: is a filename-only heuristic; the reader re-validates the file.
    format: str
    size_bytes: int
    sha256: str | None = None
    records: list[str] = []
    iterations: list[int] = []
    #: Whether the file is resolvable in the *server's* optional local mirror.
    readable: bool = False


class ResultManifest(BaseModel):
    """The light, scandir-level description of a run's linked output.

    Deliberately does not open the files: it is cheap enough to attach to the
    ``results.ready`` event and fast enough to answer without the optional
    openPMD reader.
    """

    model_config = ConfigDict(extra="forbid")

    sim_id: str
    run_dir: str
    output_dir: str | None = None
    total_bytes: int = 0
    #: ``"openpmd"`` when the optional reader is importable, else ``None``.
    reader: str | None = None
    readable_local: bool = False
    files: list[ResultRef] = []
    #: Set when ``files`` was shortened to fit the ack wire budget; the summary
    #: fields remain exact.
    truncated: bool = False


def _outcome_payload(**fields: Any) -> dict[str, Any]:
    """Drop ``None`` fields from a payload dict.

    Returns:
        The payload with only non-``None`` values.

    """
    return {key: value for key, value in fields.items() if value is not None}


def build_control_command(
    *,
    sim: str,
    seq: int,
    params: ControlParams,
    cmd_id: str | None = None,
    in_reply_to: str | None = None,
) -> RcpMessage:
    """Build the MCP-server-to-simclient control request (M3).

    Returns:
        The unsigned ``rcp.control_request`` command.

    """
    return RcpMessage(
        sim=sim,
        kind=Kind.COMMAND,
        type=SimulationType.CONTROL_COMMAND,
        seq=seq,
        sender_role=SenderRole.MCP_SERVER,
        in_reply_to=in_reply_to,
        payload={"cmd_id": cmd_id or new_cmd_id(), "sim_id": params.sim_id, "op": params.op.value},
    )


def build_control_ack(
    *,
    sim: str,
    seq: int,
    cmd_id: str,
    sim_id: str,
    op: SimulationOp,
    ok: bool,
    in_reply_to: str | None,
    job_id: int | None = None,
    signal: str | None = None,
    slurm_reason: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_code: str | None = None,
) -> RcpMessage:
    """Build the simclient's control response (M3).

    Returns:
        The unsigned ``rcp.control_ack`` message.

    """
    payload: dict[str, Any] = {"cmd_id": cmd_id, "sim_id": sim_id, "op": op.value, "ok": ok}
    payload.update(
        _outcome_payload(
            job_id=job_id,
            signal=signal,
            slurm_reason=slurm_reason,
            state=state,
            error=error,
            error_code=error_code,
        ),
    )
    return RcpMessage(
        sim=sim,
        kind=Kind.ACK,
        type=SimulationType.CONTROL_ACK,
        seq=seq,
        sender_role=SenderRole.SIMCLIENT,
        in_reply_to=in_reply_to,
        payload=payload,
    )


def build_result_command(
    *,
    sim: str,
    seq: int,
    params: ResultParams,
    cmd_id: str | None = None,
    in_reply_to: str | None = None,
) -> RcpMessage:
    """Build the MCP-server-to-simclient results request (M3).

    Returns:
        The unsigned ``rcp.result_request`` command.

    """
    payload: dict[str, Any] = {"cmd_id": cmd_id or new_cmd_id(), "sim_id": params.sim_id, "op": params.op.value}
    payload.update(
        _outcome_payload(
            path=params.path,
            record=params.record,
            component=params.component,
            iteration=params.iteration,
            axis=params.axis,
            index=params.index,
            downsample=params.downsample,
            stream=params.stream,
            tail=params.tail,
            program=params.program,
            reader=params.reader,
            species=params.species,
            species_filter=params.species_filter,
            min_kev=params.min_kev,
            max_kev=params.max_kev,
        ),
    )
    return RcpMessage(
        sim=sim,
        kind=Kind.COMMAND,
        type=SimulationType.RESULT_COMMAND,
        seq=seq,
        sender_role=SenderRole.MCP_SERVER,
        in_reply_to=in_reply_to,
        payload=payload,
    )


def build_result_ack(
    *,
    sim: str,
    seq: int,
    cmd_id: str,
    sim_id: str,
    op: ResultOp,
    in_reply_to: str | None,
    manifest: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    data: list[float] | str | None = None,
    data_encoding: str | None = None,
    n_points: int | None = None,
    stats: dict[str, float | int] | None = None,
    error: str | None = None,
    error_code: str | None = None,
) -> RcpMessage:
    """Build the simclient's results response (M3).

    ``data`` is either a reduced numeric array (``data_encoding="float"``), a
    base64 thumbnail (``"png"``) or a list of text lines (``"text"``); all are
    bounded by :data:`MAX_RESULT_BYTES` at the client before building.

    Returns:
        The unsigned ``rcp.result_ack`` message.

    """
    payload: dict[str, Any] = {"cmd_id": cmd_id, "sim_id": sim_id, "op": op.value}
    payload.update(
        _outcome_payload(
            manifest=manifest,
            result=result,
            data=data,
            data_encoding=data_encoding,
            n_points=n_points,
            stats=stats,
            error=error,
            error_code=error_code,
        ),
    )
    # Single chokepoint: a result ack must fit the homeserver event budget, so
    # an over-budget encoding is replaced by a clean RESULT_TOO_LARGE error
    # rather than being sent and rejected (which would look like a pull timeout).
    # Measure the *encoded* payload: a float is carried as the larger tagged
    # object {"$rcp_float": "..."}, so the raw form would under-count.
    if len(json.dumps(encode_wire(payload), ensure_ascii=True, separators=(",", ":"))) > MAX_RESULT_BYTES:
        payload = {
            "cmd_id": cmd_id,
            "sim_id": sim_id,
            "op": op.value,
            "error": "result exceeds the ack wire budget",
            "error_code": "result_too_large",
        }
    return RcpMessage(
        sim=sim,
        kind=Kind.ACK,
        type=SimulationType.RESULT_ACK,
        seq=seq,
        sender_role=SenderRole.SIMCLIENT,
        in_reply_to=in_reply_to,
        payload=payload,
    )
