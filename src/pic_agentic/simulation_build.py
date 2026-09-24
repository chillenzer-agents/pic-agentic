# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Build a ``Runner`` JSON dump from a PICMI script in a disposable subprocess.

The MCP server must never import or execute the LLM-supplied PICMI script in
its own process (design section 4.1).  The script is written to a temporary
file and handed to a fresh interpreter (``picongpu_python`` when configured,
else the current one); the child imports ``picongpu`` and the script, converts
the resulting ``picmi.Simulation`` to ``pypicongpu`` and prints one JSON line.

Isolating this in a module of its own keeps the test seam small: the tests
monkeypatch :func:`build_runner_dump` rather than the subprocess.
"""

from __future__ import annotations

import asyncio
import json
import os
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
print(json.dumps({"runner": runner.model_dump(mode="json"), "provenance": provenance}))
"""


#: Marker key proving the child produced the new {runner, provenance} envelope.
_CHILD_RUNNER_KEY = "runner"

#: Environment variables safe to pass into the untrusted PICMI child.  The
#: child executes arbitrary LLM-supplied code, so it must NOT inherit the RCP
#: secret or MAS tokens (which would otherwise be readable and printable back
#: into the tool result).
_SAFE_ENV_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TZ",
        "USER",
        "LOGNAME",
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
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


def _safe_child_env() -> dict[str, str]:
    """Return a minimal environment for the untrusted PICMI child.

    Returns:
        The allow-listed environment plus ``PYTHONUNBUFFERED``.

    """
    env = {key: value for key, value in os.environ.items() if key in _SAFE_ENV_KEYS}
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


async def build_runner_dump(
    *,
    script_path: Path,
    interpreter: str = "",
    timeout_s: float = 300.0,
) -> BuiltSimulation:
    """Run ``script_path`` in a child interpreter and return its build result.

    The child executes untrusted, LLM-supplied code, so it gets a minimal
    allow-listed environment and never the RCP secret or MAS tokens.

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
    argv = [interpreter or sys.executable, child_path, str(script_path)]
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_safe_child_env(),
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
    if process.returncode != 0:
        detail = (stderr or stdout).decode("utf-8", "replace").strip()
        msg = f"PICMI script failed (rc={process.returncode}): {detail[-2000:]}"
        raise SimulationBuildError(msg)
    text = stdout.decode("utf-8", "replace").strip().splitlines()
    if not text:
        msg = "PICMI script produced no output"
        raise SimulationBuildError(msg)
    try:
        data = json.loads(text[-1])
    except ValueError as exc:
        msg = f"PICMI script output was not JSON: {exc}"
        raise SimulationBuildError(msg) from exc
    if not isinstance(data, dict) or _CHILD_RUNNER_KEY not in data:
        msg = "PICMI script output has no runner dump"
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
