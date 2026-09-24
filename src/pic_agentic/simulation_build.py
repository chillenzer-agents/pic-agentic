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
from pathlib import Path

#: Child script: run the PICMI script, convert and print the runner dump.
_CHILD_SOURCE = r"""
import json
import runpy
import sys

from picongpu import picmi
from picongpu.pypicongpu.runner import Runner

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
print(json.dumps(runner.model_dump(mode="json")))
"""


class SimulationBuildError(RuntimeError):
    """Raised when the PICMI script cannot be turned into a runner dump."""


async def build_runner_dump(
    *,
    script_path: Path,
    interpreter: str = "",
    timeout_s: float = 300.0,
) -> dict[str, object]:
    """Run ``script_path`` in a child interpreter and return the runner dump.

    Args:
        script_path: Path to the (already written) PICMI script.
        interpreter: Interpreter with the pinned PIConGPU; empty uses
            :data:`sys.executable`.
        timeout_s: Maximum wall time for the child.

    Returns:
        The parsed ``Runner.model_dump(mode="json")`` mapping.

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
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
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
    if not isinstance(data, dict):
        msg = "runner dump is not a JSON object"
        raise SimulationBuildError(msg)
    return data
