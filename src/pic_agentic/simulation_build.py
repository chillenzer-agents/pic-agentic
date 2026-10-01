# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Build a ``Runner`` JSON dump from a PICMI script in a disposable subprocess.

The MCP server must never import or execute the LLM-supplied PICMI script in
its own process (design section 4.1).  The script is written to a temporary
file and handed to a fresh interpreter (``picongpu_python`` when configured,
else the current one); the child imports ``picongpu`` and the script, converts
the resulting ``picmi.Simulation`` to ``pypicongpu`` and prints one JSON line.

Security scope (important): this child is **not a sandbox**.  It runs
same-uid, so the LLM-supplied script can still read the parent's environment
via ``/proc/<ppid>/environ``, and any redaction of its output is best-effort.
What this module *does* provide is a **disposable interpreter with a minimal
environment**: the RCP secret and MAS tokens are not passed in, ``HOME`` points
at a fresh scratch directory (so the 0600 ``~/.config/pic-agentic/config.toml``
is not reachable by ``~``), and the child's stderr is only echoed back as a
bounded, clearly-labelled tail.  Full isolation (separate uid or a namespace)
is a documented non-goal for this PoC; executing the PICMI script on the
submission node is the design's premise (``M2-SUBMIT-PLAN.md``, design section
6.5).

The provenance tuple the child reports is a **drift/consistency check, not a
trust boundary**: the script that builds the simulation is arbitrary code by
design, so it is inherently untrusted and could in principle forge the tuple.
The check catches accidental version/schema drift between the server and the
cluster install, not a hostile script (see :func:`build_runner_dump`).

Isolating this in a module of its own keeps the test seam small: the tests
monkeypatch :func:`build_runner_dump` rather than the subprocess.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

#: Child script: run the PICMI script, convert and print the runner dump plus
#: the provenance of *this* interpreter (not the server's), so the payload
#: header describes the exact tree that produced the dump.
_CHILD_SOURCE = r"""
import hashlib
import json
import os
import runpy
import sys

from picongpu import __version__
from picongpu import picmi
from picongpu.pypicongpu.runner import Runner


def _noop(*_args, **_kwargs):
    return None


# The tool only needs the ``picmi.Simulation`` object, but the documented PICMI
# examples end with ``sim.run(...)``.  Neutralise the build/submit entry points
# so a trailing run() (or write_input_file) does not compile or touch the
# cluster; the script's simulation object is still fully constructed by then.
picmi.Simulation.run = _noop
picmi.Simulation.picongpu_run = _noop
picmi.Simulation.write_input_file = _noop


def _canonical_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _normalise(value):
    if isinstance(value, str):
        return "<abs>" if value.startswith("/") else value
    if isinstance(value, dict):
        return {key: _normalise(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    return value


def _revision():
    try:
        from importlib.metadata import distribution
    except ImportError:
        return ""
    try:
        dist = distribution("picongpu")
    except Exception:
        return ""
    direct_url = dist.read_text("direct_url.json")
    if not direct_url:
        return ""
    try:
        data = json.loads(direct_url)
    except ValueError:
        return ""
    commit = data.get("vcs_info", {}).get("commit_id")
    return commit if isinstance(commit, str) else ""


schema_hash = hashlib.sha256(_canonical_bytes(_normalise(Runner.model_json_schema()))).hexdigest()

path = sys.argv[1]
# Marker injected by the server: the trusted harness emits exactly one line
# prefixed with it *after* ``runpy`` returns, so a stray/forged line from the
# script's ``atexit``/stdout games cannot be mistaken for the dump.  The nonce
# is visible to the script (same process) and so is NOT an authentication
# secret -- it is a line-selection aid, not a trust boundary (see module doc).
marker = sys.argv[2] if len(sys.argv) > 2 else ""
namespace = runpy.run_path(path)
sims = [v for v in namespace.values() if isinstance(v, picmi.Simulation)]
if not sims:
    print("no picmi.Simulation found in script", file=sys.stderr)
    raise SystemExit(2)
if len(sims) > 1:
    print("multiple picmi.Simulation objects found; expose exactly one", file=sys.stderr)
    raise SystemExit(2)
runner = Runner(sim=sims[0])
provenance = {
    "picongpu_version": str(__version__),
    "picongpu_revision": os.environ.get("PIC_AGENTIC_PICONGPU_REVISION", "") or _revision(),
    "schema_hash": schema_hash,
}
print(marker + json.dumps({"runner": runner.model_dump(mode="json"), "provenance": provenance}))
"""


#: Marker key proving the child produced the new {runner, provenance} envelope.
_CHILD_RUNNER_KEY = "runner"

#: Environment variables safe to pass into the untrusted PICMI child.  The
#: child executes arbitrary LLM-supplied code, so it must NOT inherit the RCP
#: secret or MAS tokens (which would otherwise be readable and printable back
#: into the tool result).  ``HOME`` is intentionally absent: it is set to a
#: fresh scratch directory by :func:`_safe_child_env` so ``~`` cannot reach the
#: 0600 ``~/.config/pic-agentic/config.toml``.  ``PYTHONPATH``/``PYTHONHOME``/
#: ``VIRTUAL_ENV`` are omitted too: the interpreter is invoked by absolute path
#: (its own venv/``sys.executable`` resolves ``picongpu``), so they are not
#: needed and only widen what the script can reach.
_SAFE_ENV_KEYS = frozenset(
    {
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TZ",
        "USER",
        "LOGNAME",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_ROOT",
        "PICSRC",
        "PIC_BACKEND",
        "PIC_CFG",
        # Not a secret; the operator's explicit revision pin (see version.py).
        "PIC_AGENTIC_PICONGPU_REVISION",
    },
)


def _safe_child_env(home: str) -> dict[str, str]:
    """Return a minimal environment for the untrusted PICMI child.

    Args:
        home: Scratch directory to use as the child's ``HOME`` (so the 0600
            config under the real ``~/.config`` is not reachable via ``~``).

    Returns:
        The allow-listed environment plus ``HOME`` and ``PYTHONUNBUFFERED``.

    """
    env = {key: value for key, value in os.environ.items() if key in _SAFE_ENV_KEYS}
    env["HOME"] = home
    env["PYTHONUNBUFFERED"] = "1"
    return env


class SimulationBuildError(RuntimeError):
    """Raised when the PICMI script cannot be turned into a runner dump."""


@dataclass(frozen=True)
class BuiltSimulation:
    """The child's output: a runner dump and the provenance of its interpreter."""

    runner: dict[str, object]
    picongpu_version: str
    picongpu_revision: str
    schema_hash: str


#: Curated ``pypicongpu`` ``@computed_field`` placements (field -> the ``sim``
#: sub-path it belongs on).  A caller copying a dump around may store such a
#: field one level too deep; that dump fails the simclient's round-trip gate as
#: ``unsupported``.  Naming the expected parent here lets a server without
#: PIConGPU still produce the actionable create-time error (the exact check runs
#: when the pin is importable).  Extend as further computed fields surface.
_KNOWN_COMPUTED_FIELDS = {"num_tmp_field_slots": "collisional_physics"}

#: Sentinel paths created by :func:`check_spec_round_trip` when it replays the
#: simclient gate.  They are deliberately outside any real spec (a leading
#: double underscore is not a pypicongpu field) and only serve to satisfy
#: ``Runner``'s required ``setup_dir``/``run_dir``.
_CHECK_SETUP_DIR = "/__pic_agentic_spec_check__/input"
_CHECK_RUN_DIR = "/__pic_agentic_spec_check__/run"

#: Bounds on the paraphrased pydantic-validation message.  A pathological spec
#: can produce hundreds of errors; the message stays a short, actionable list of
#: distinct offending fields plus a count of the rest.
_MAX_VALIDATION_ERRORS = 5
_MAX_VALUE_CHARS = 40

#: Substring of pydantic's internal union-branch location segments (e.g.
#: ``function-after[check(), Grid3D]``).  A union emits one alternation per
#: branch, so the same real field would otherwise be reported twice.
_INTERNAL_LOC_MARKER = "function-"

#: Distinguishes "no input recorded" from a recorded ``None``.
_MISSING = object()


def check_spec_round_trip(runner_dump: dict[str, object]) -> str | None:
    """Check that ``sim`` reproduces itself through the pinned ``Runner``.

    This mirrors the simclient's gate *exactly*: the simclient rebuilds a
    ``Runner`` from a **reduced** dump -- ``{"sim": <sim>, "setup_dir": ...,
    "run_dir": ...}`` plus ``template_dir`` only when it is configured -- and
    rejects the payload when either (a) ``Runner.model_validate`` fails, which
    the simclient reports as ``payload_invalid``, or (b) the re-validated
    ``sim`` differs from ``<sim>``, reported as ``unsupported``.  It never
    validates the caller's sibling keys, so this check must not either (doing
    so would reject a valid spec over an unrelated ``template_dir`` or
    ``setup_dir`` shape the simclient drops).  Running the same check at
    campaign-creation time turns a later catastrophic leaf failure into an
    immediate, precise error.

    A validation failure is *rejected* (never swallowed): the simclient would
    refuse the same leaf with ``payload_invalid`` at submit time, so treating it
    as a pass would let ``create_campaign`` persist a campaign whose every leaf
    fails -- the beta-3 defect.  A spec that validates but re-serialises
    differently is only rejected when a caller-supplied leaf path is genuinely
    dropped: if every leaf survives (the pin may add computed metadata such as
    ``precision_overrides``) the spec is accepted rather than over-rejected.

    When PIConGPU is not importable (a server without the pin) the check degrades
    to a curated, **best-effort** detector of known computed-field shapes rather
    than skipping entirely.  It cannot catch an arbitrary unknown field or a
    type-invalid value (those need the pin), so the create-time guarantee is
    exact only where the pin is importable -- and deliberately never
    over-rejects, since a pin-less server cannot reproduce the pin's schema.

    Args:
        runner_dump: A wire spec carrying ``sim`` (a ``Runner`` dump).

    Returns:
        An actionable message when the spec fails to validate or a requested
        field would be silently dropped, else ``None`` (including when there is
        no ``sim`` mapping to check).

    """
    sim_dump = runner_dump.get("sim")
    if not isinstance(sim_dump, dict):
        return None
    try:
        from picongpu.pypicongpu.runner import (  # ruff: ignore[import-outside-top-level] - optional dependency
            Runner,
        )
    except ImportError:
        return _detect_misplaced_computed_field(sim_dump)
    # Build the same reduced dump the simclient builds.  The caller's sibling
    # keys (run_dir/setup_dir/template_dir/provenance/...) are dropped, exactly
    # as the simclient drops them; the sentinel directories satisfy Runner's
    # required fields without claiming any caller value.
    reduced: dict[str, object] = {
        "sim": sim_dump,
        "setup_dir": _CHECK_SETUP_DIR,
        "run_dir": _CHECK_RUN_DIR,
    }
    try:
        # A validation warning (e.g. a laser pulse truncated by a too-short run)
        # is not a schema violation; the simclient validates without ``error``
        # filters, so suppress warnings here to match its semantics.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            runner = Runner.model_validate(reduced)
    except ValidationError as exc:
        # The simclient rejects the same leaf as ``payload_invalid``; accepting
        # it here would persist a campaign whose every leaf fails (beta-3).  The
        # bounded, paraphrased errors name the offending fields actionably.
        return _validation_error_message(exc)
    except Exception as exc:  # ruff: ignore[blind-except] - any other failure is an unusable spec
        return f"spec does not validate against the pinned pypicongpu schema: {exc}"
    dumped = runner.sim.model_dump(mode="json")
    # Accept when nothing the caller asked for was lost: either the dump is
    # identical, or it differs only by metadata the pin normalised/added (e.g. a
    # recomputed ``precision_overrides`` or computed defaults a minimal test spec
    # leaves out).  Only a silently dropped caller leaf is rejected.
    if dumped == sim_dump or not _dropped_leaf_paths(sim_dump, dumped):
        return None
    return _round_trip_diff_message(sim_dump, dumped)


def _dropped_leaf_paths(before: object, after: object) -> list[str]:
    """Return the caller-supplied leaf paths absent from the round-tripped dump.

    These are the fields the schema silently dropped.  A computed field the
    caller stored at the wrong level also shows up here, so
    :func:`_round_trip_diff_message` can name the move; re-serialisation that
    only changes or adds values (not paths) contributes nothing.

    Returns:
        The dropped paths, in the input's deterministic order.

    """
    to_paths = set(_leaf_paths(after))
    return [path for path in _leaf_paths(before) if path not in to_paths]


def _round_trip_diff_message(before: object, after: object) -> str:
    """Return an actionable message for a spec that changed under re-validation.

    A leaf path present in the input but absent from the round-tripped dump was
    dropped by the schema.  When the same final segment reappears elsewhere in
    the dump, the field was misplaced (a common copy-through-the-wrong-level
    mistake) and the message names the correct path.

    Returns:
        The message naming the offending path.

    """
    from_paths = _leaf_paths(before)
    to_paths = _leaf_paths(after)
    dropped = [path for path in from_paths if path not in to_paths]
    for path in dropped:
        segment = path.rsplit(".", 1)[-1]
        suggestion = next((other for other in to_paths if other.rsplit(".", 1)[-1] == segment), None)
        if suggestion is not None:
            return (
                f"{path} is not part of the pinned pypicongpu schema at this path; move it to {suggestion} and resubmit"
            )
        return f"{path} is not part of the pinned pypicongpu schema and would be silently dropped"
    return "spec does not reproduce itself under the pinned pypicongpu schema"


def _validation_error_message(exc: ValidationError) -> str:
    """Return a concise, actionable message for a ``Runner`` validation failure.

    ``str(ValidationError)`` dumps every error with its full input, which for a
    large spec is pages of noise.  This paraphrases the errors into one line per
    distinct offending field, naming the field and its expected shape so the
    caller can fix the spec, and bounds the output for a pathological spec.

    Args:
        exc: The pydantic error raised by the pinned ``Runner``.

    Returns:
        A message of the form ``sim.species Input should be a valid list (got
        str '')``.

    """
    specific: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for error in exc.errors():
        location = _validation_error_location(error.get("loc", ()))
        if not location or location in seen:
            continue
        seen.add(location)
        line = f"{location} {_validation_error_clause(error)}"
        # A type-invalid value is the informative error for its field; the same
        # field can additionally surface as ``missing`` because a failed union
        # branch discards the whole object.  Report the informative kind first.
        (missing if error.get("type") == "missing" else specific).append(line)
    lines = [*specific, *missing]
    if not lines:
        return "spec does not validate against the pinned pypicongpu schema"
    if len(lines) > _MAX_VALIDATION_ERRORS:
        hidden = len(lines) - _MAX_VALIDATION_ERRORS
        lines = [*lines[:_MAX_VALIDATION_ERRORS], f"... and {hidden} more error(s)"]
    return "spec does not validate against the pinned pypicongpu schema: " + "; ".join(lines)


def _validation_error_location(location: object) -> str:
    """Render a pydantic ``loc`` as a dotted field path, dropping union internals.

    A tagged union reports each branch as an extra ``function-after[...]``
    segment; those are pydantic implementation detail, not addressable fields,
    so they are dropped and the two branches collapse to one path (which also
    lets callers de-duplicate the field).

    Returns:
        The dotted location, or ``""`` when nothing meaningful remains.

    """
    if not isinstance(location, (list, tuple)):
        return ""
    parts = [str(part) for part in location]
    collapsed = [part for part in parts if _INTERNAL_LOC_MARKER not in part]
    return ".".join(collapsed)


def _validation_error_clause(error: dict[str, object]) -> str:
    """Return the expected-shape clause plus a short ``(got ...)`` note.

    Args:
        error: One entry from ``ValidationError.errors()``.

    Returns:
        ``Input should be a valid list (got str '')``.  A missing field records
        the whole enclosing object as its pydantic ``input``, which is not a
        useful value, so the ``(got ...)`` note is omitted for ``type=missing``.

    """
    clause = str(error.get("msg", "is invalid")).removeprefix("Value error, ")
    value = error.get("input", _MISSING)
    if error.get("type") == "missing" or value is _MISSING:
        return clause
    return f"{clause} (got {_describe_value(value)})"


def _describe_value(value: object) -> str:
    """Summarise a JSON value as ``type repr``, bounded in length.

    Returns:
        e.g. ``str ''`` or ``dict {...}``.

    """
    rendered = repr(value)
    if len(rendered) > _MAX_VALUE_CHARS:
        rendered = rendered[:_MAX_VALUE_CHARS] + "..."
    return f"{type(value).__name__} {rendered}"


def _detect_misplaced_computed_field(sim_dump: dict[str, object]) -> str | None:
    """Best-effort detection of a computed field stored one level too deep.

    Used when PIConGPU is not importable server-side.  A known ``@computed_field``
    under a sub-model is reported with its likely correct parent path.

    Returns:
        The actionable message, or ``None`` when nothing matches.

    """
    for path in _leaf_paths(sim_dump):
        segment = path.rsplit(".", 1)[-1]
        expected_parent = _KNOWN_COMPUTED_FIELDS.get(segment)
        if expected_parent is None:
            continue
        container, _, _ = path.rpartition(".")
        if container == expected_parent:
            # Already at its correct level.
            continue
        return (
            f"{path} is not part of the pinned pypicongpu schema at this path; "
            f"move the computed field {segment!r} to {expected_parent}.{segment} and resubmit"
        )
    return None


def _leaf_paths(value: object, prefix: str = "") -> list[str]:
    """Return the dotted paths of every scalar leaf in a nested JSON value.

    Returns:
        The leaf paths, in deterministic (insertion/ascending) order.

    """
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_leaf_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_leaf_paths(item, f"{prefix}[{index}]"))
    else:
        paths.append(prefix)
    return paths


#: Maximum number of stderr characters echoed back in a build error.  The
#: child is untrusted, so its stderr is treated as hostile input: only a
#: bounded tail is kept, and the message labels it as untrusted.
_MAX_STDERR_CHARS = 2000

#: Prefix marking the harness-produced JSON line.  See ``_CHILD_SOURCE``.
_OUTPUT_MARKER = "__pic_agentic_runner_dump__:"


def _extract_payload(stdout: bytes, marker: str) -> dict[str, object] | None:
    """Return the harness JSON object from the child's stdout, if present.

    The trusted harness prints exactly one line prefixed with ``marker`` after
    ``runpy`` returns.  Only the last such line is taken, so a script's own
    ``atexit``/stdout output (which cannot know the server-injected nonce in
    advance) cannot masquerade as the dump.  This is a *line-selection* aid,
    not authentication: the script runs in the same process and can read the
    nonce from ``sys.argv``.  Provenance is a drift check, not a trust boundary
    (see the module docstring).

    Args:
        stdout: The child's raw stdout.
        marker: The nonce prefix the server injected.

    Returns:
        The decoded JSON mapping, or ``None`` if no marked line parses.

    """
    prefix = marker.encode("utf-8") if marker else b""
    for line in reversed(stdout.decode("utf-8", "replace").splitlines()):
        candidate = line
        if prefix:
            if not candidate.startswith(marker):
                continue
            candidate = candidate[len(marker) :]
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict) and _CHILD_RUNNER_KEY in data:
            return data
    return None


def _bounded_stderr(stderr: bytes, stdout: bytes) -> str:
    """Return a labeled, bounded tail of the child's error output.

    Args:
        stderr: The child's stderr bytes (preferred).
        stdout: The child's stdout bytes (fallback when stderr is empty).

    Returns:
        The last :data:`_MAX_STDERR_CHARS` characters, prefixed to make clear
        the text is untrusted child output.

    """
    detail = (stderr or stdout).decode("utf-8", "replace").strip()
    if not detail:
        return ""
    tail = detail[-_MAX_STDERR_CHARS:]
    return f"<untrusted child stderr, tail> {tail}"


async def build_runner_dump(
    *,
    script_path: Path,
    interpreter: str = "",
    timeout_s: float = 300.0,
) -> BuiltSimulation:
    """Run ``script_path`` in a child interpreter and return its build result.

    The child executes untrusted, LLM-supplied code, so it gets a minimal
    allow-listed environment (no RCP secret/MAS tokens, a scratch ``HOME``) and
    its error output is only echoed back as a bounded, labeled tail.  This is a
    disposable interpreter, **not** a security sandbox: the child runs same-uid
    and can still read ``/proc/<ppid>/environ`` (see the module docstring).

    The provenance tuple is a drift/consistency check, not a trust boundary;
    the harness prints it on a nonce-marked line after ``runpy`` returns so
    stray script output is not mistaken for it.

    Args:
        script_path: Path to the (already written) PICMI script.
        interpreter: Interpreter with the pinned PIConGPU; empty uses
            :data:`sys.executable`.
        timeout_s: Maximum wall time for the child.

    Returns:
        The runner dump plus the child's own provenance tuple.

    Raises:
        SimulationBuildError: If the child fails, times out, or prints no JSON.

    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as child:
        child.write(_CHILD_SOURCE)
        child_path = child.name
    # A fresh scratch HOME per invocation: ``~`` in the child cannot reach the
    # real 0600 config.toml, and the directory is removed with the child.
    scratch_home = tempfile.mkdtemp(prefix="pic-agentic-home-")
    marker = f"{_OUTPUT_MARKER}{secrets.token_hex(16)}"
    argv = [interpreter or sys.executable, child_path, str(script_path), marker]
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_safe_child_env(scratch_home),
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except TimeoutError:
        process.kill()
        await process.wait()
        msg = f"PICMI script exceeded {timeout_s}s"
        raise SimulationBuildError(msg) from None
    finally:
        Path(child_path).unlink(missing_ok=True)
        shutil.rmtree(scratch_home, ignore_errors=True)
    if process.returncode != 0:
        detail = _bounded_stderr(stderr, stdout)
        msg = (
            f"PICMI script failed (rc={process.returncode}); the script must define exactly one "
            "picmi.Simulation object and does not need to run it (a trailing sim.run() is ignored)"
        )
        if detail:
            msg = f"{msg}: {detail}"
        raise SimulationBuildError(msg)
    data = _extract_payload(stdout, marker)
    if data is None:
        msg = "PICMI script produced no runner dump"
        raise SimulationBuildError(msg)
    runner = data[_CHILD_RUNNER_KEY]
    provenance = data.get("provenance", {})
    if not isinstance(runner, dict) or not isinstance(provenance, dict):
        msg = "PICMI script output is malformed"
        raise SimulationBuildError(msg)
    return BuiltSimulation(
        runner=runner,
        picongpu_version=str(provenance.get("picongpu_version", "")),
        picongpu_revision=str(provenance.get("picongpu_revision", "")),
        schema_hash=str(provenance.get("schema_hash", "")),
    )
