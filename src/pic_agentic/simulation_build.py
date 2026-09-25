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
from dataclasses import dataclass
from pathlib import Path

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
        msg = f"PICMI script failed (rc={process.returncode})"
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
