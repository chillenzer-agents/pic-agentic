# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""The build child must tolerate a documented trailing ``sim.run(...)``.

The PICMI docs' canonical examples end with ``sim.run(...)``.  The builder only
wants the ``picmi.Simulation`` object, so the child neutralises ``run`` /
``write_input_file`` before executing the script.  These tests run the *real*
``_CHILD_SOURCE`` offline against a stub ``picongpu`` package (injected via a
small interpreter shim), so they assert the neutralisation without a pinned
PIConGPU install or a compile.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from pic_agentic.simulation_build import SimulationBuildError, build_runner_dump

_STUB_PICMI = textwrap.dedent(
    """
    class Simulation:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, *args, **kwargs):
            import pathlib
            pathlib.Path(__file__).with_name("run_called").write_text("1")
            return None

        def write_input_file(self, *args, **kwargs):
            import pathlib
            pathlib.Path(__file__).with_name("write_input_file_called").write_text("1")
            return None
    """
)

_STUB_RUNNER = textwrap.dedent(
    """
    class Runner:
        def __init__(self, sim=None, **kwargs):
            self.sim = sim

        @classmethod
        def model_json_schema(cls):
            return {"title": "Runner", "type": "object"}

        def model_dump(self, mode="json"):
            return {"sim": self.sim.kwargs}
    """
)


def _stub_picongpu(root: Path) -> Path:
    """Create a minimal ``picongpu`` package tree and return its import root."""
    site = root / "site"
    pkg = site / "picongpu"
    (pkg / "pypicongpu").mkdir(parents=True)
    (pkg / "__init__.py").write_text('__version__ = "0.0.0-stub"\n', encoding="utf-8")
    (pkg / "picmi.py").write_text(_STUB_PICMI, encoding="utf-8")
    (pkg / "pypicongpu" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "pypicongpu" / "runner.py").write_text(_STUB_RUNNER, encoding="utf-8")
    return site


def _shim_interpreter(root: Path, fake_dir: Path) -> Path:
    """Return an interpreter shim that puts ``fake_dir`` on the child's path."""
    shim = root / "child_python.sh"
    shim.write_text(
        f'#!/bin/sh\nPYTHONPATH="{fake_dir}" exec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim


async def test_trailing_run_is_neutralised_and_spec_is_returned(tmp_path: Path) -> None:
    """A documented trailing ``sim.run(...)`` builds and does not compile."""
    fake_dir = _stub_picongpu(tmp_path)
    shim = _shim_interpreter(tmp_path, fake_dir)
    script = tmp_path / "sim.py"
    script.write_text(
        "from picongpu import picmi\n"
        "sim = picmi.Simulation(time_step_size=1.0, max_steps=2)\n"
        'sim.run(setup_dir="setup", run_dir="run")\n',
        encoding="utf-8",
    )

    built = await build_runner_dump(script_path=script, interpreter=str(shim))

    assert built.runner == {"sim": {"time_step_size": 1.0, "max_steps": 2}}
    assert built.picongpu_version == "0.0.0-stub"
    assert not (fake_dir / "picongpu" / "run_called").exists()
    assert not (fake_dir / "picongpu" / "write_input_file_called").exists()


async def test_no_simulation_still_errors_clearly(tmp_path: Path) -> None:
    fake_dir = _stub_picongpu(tmp_path)
    shim = _shim_interpreter(tmp_path, fake_dir)
    script = tmp_path / "sim.py"
    script.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(SimulationBuildError, match=r"exactly one picmi\.Simulation") as excinfo:
        await build_runner_dump(script_path=script, interpreter=str(shim))
    assert "no picmi.Simulation found" in str(excinfo.value)


async def test_multiple_simulations_still_error_clearly(tmp_path: Path) -> None:
    fake_dir = _stub_picongpu(tmp_path)
    shim = _shim_interpreter(tmp_path, fake_dir)
    script = tmp_path / "sim.py"
    script.write_text(
        "from picongpu import picmi\nfirst = picmi.Simulation(max_steps=1)\nsecond = picmi.Simulation(max_steps=2)\n",
        encoding="utf-8",
    )

    with pytest.raises(SimulationBuildError, match=r"exactly one picmi\.Simulation") as excinfo:
        await build_runner_dump(script_path=script, interpreter=str(shim))
    assert "multiple picmi.Simulation objects found" in str(excinfo.value)
