# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Optional integration test against a real pinned PIConGPU install.

Runs only when PIConGPU is importable (the ``sim`` extra).  The default
offline suite skips it, so CI without the pin stays fast; a developer or the
cluster venv gets a genuine ``Runner`` round-trip and a real
``SchemaHash``/payload-hash agreement check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pic_agentic.protocol.simulation import SimulationPayload, simulation_spec_from_runner_dump
from pic_agentic.version import local_provenance, picongpu_revision, picongpu_version, runner_schema_hash

picongpu = pytest.importorskip("picongpu")

FIXTURE = Path(__file__).parent / "fixtures" / "pypicongpu_runner.json"


def _cwltool_beside_interpreter() -> str | None:
    """Return a ``cwltool`` next to the running interpreter, if present.

    ``shutil.which`` only sees the current PATH; the pinned ``[sim]`` venv
    installs ``cwltool`` as its sibling, which is where the real validation
    runs.  Returns ``None`` when neither is available.
    """
    import shutil
    import sys

    candidate = Path(sys.executable).parent / "cwltool"
    if candidate.is_file():
        return str(candidate)
    return shutil.which("cwltool")


@pytest.mark.integration
def test_real_provenance_is_available() -> None:
    assert picongpu_version()
    assert runner_schema_hash()
    assert local_provenance()["schema_hash"] == runner_schema_hash()
    # The revision is best-effort (editable installs have no vcs_info), so only
    # assert it is a string.
    assert isinstance(picongpu_revision(), str)


@pytest.mark.integration
async def test_child_reports_the_same_provenance_as_version_py(tmp_path: Path) -> None:
    """The child's inline schema hash must match ``version.py``'s.

    The child cannot import ``pic_agentic`` (it may run under a different
    interpreter), so it re-implements the normalisation.  This guards the two
    implementations against drift, which would otherwise silently reject every
    payload.
    """
    from pic_agentic.simulation_build import build_runner_dump

    script = tmp_path / "sim.py"
    script.write_text(
        "from picongpu import picmi\n"
        "grid = picmi.Cartesian3DGrid(number_of_cells=[8, 8, 8], lower_bound=[0, 0, 0], "
        "upper_bound=[1e-6, 1e-6, 1e-6], lower_boundary_conditions=['periodic'] * 3, "
        "upper_boundary_conditions=['periodic'] * 3)\n"
        "solver = picmi.ElectromagneticSolver(method='Yee', grid=grid)\n"
        "sim = picmi.Simulation(time_step_size=1e-15, max_steps=2, solver=solver)\n",
        encoding="utf-8",
    )
    built = await build_runner_dump(script_path=script)
    assert built.schema_hash == runner_schema_hash()
    assert built.picongpu_version == picongpu_version()


@pytest.mark.integration
def test_schema_hash_is_path_independent() -> None:
    """The schema hash must not depend on the install prefix.

    The raw ``Runner`` schema embeds the absolute package template path, which
    differs between the server venv and the cluster venv even for the same
    commit; the normalised hash must drop it.
    """
    import hashlib

    from picongpu.pypicongpu.runner import Runner

    from pic_agentic.rcp.crypto import canonical_bytes
    from pic_agentic.version import normalise_schema_paths

    raw = Runner.model_json_schema()
    # Every absolute-path string in the raw schema (the template_dir default).
    assert any(
        isinstance(value, str) and value.startswith("/") for value in _iter_schema_values(raw) if isinstance(value, str)
    )
    # Normalising twice is idempotent and yields the same hash as the helper.
    normalised = normalise_schema_paths(raw)
    assert normalise_schema_paths(normalised) == normalised
    assert hashlib.sha256(canonical_bytes(normalised)).hexdigest() == runner_schema_hash()


def _iter_schema_values(node: object):
    if isinstance(node, dict):
        for value in node.values():
            yield from _iter_schema_values(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_schema_values(value)
    else:
        yield node


@pytest.mark.integration
async def test_child_env_has_no_secrets_and_scratch_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real child: the RCP secret/tokens are absent and HOME is a scratch dir."""
    from pic_agentic.simulation_build import SimulationBuildError, build_runner_dump

    monkeypatch.setenv("PIC_AGENTIC_RCP_SECRET", "deadbeefcafesecret")
    monkeypatch.setenv("PIC_AGENTIC_ACCESS_TOKEN", "syt_access_secret")
    monkeypatch.setenv("PIC_AGENTIC_REFRESH_TOKEN", "refresh_secret")
    # Print the child's secret-ish env and HOME, then fail so the stdout/stderr
    # is returned in the error message for inspection.
    script = tmp_path / "sim.py"
    script.write_text(
        "import os, sys\n"
        "print('CHILD_ENV', {k: v for k, v in os.environ.items() if 'SECRET' in k or 'TOKEN' in k})\n"
        "print('CHILD_HOME', os.environ.get('HOME'))\n"
        "sys.exit(7)\n",
        encoding="utf-8",
    )
    with pytest.raises(SimulationBuildError) as excinfo:
        await build_runner_dump(script_path=script)
    message = str(excinfo.value)
    assert "deadbeefcafesecret" not in message
    assert "syt_access_secret" not in message
    assert "refresh_secret" not in message
    # HOME is a scratch dir, not the parent's real home (which holds the 0600
    # config.toml).
    assert "pic-agentic-home-" in message


@pytest.mark.integration
def test_overwrite_vars_input_validates_against_real_cwl(tmp_path: Path) -> None:
    """``run_overwrite_vars`` must validate as the string the pinned CWL wants.

    Regression: ``Runner.generate`` writes the ``overwrite_vars`` list through
    to ``input.yaml`` while ``workflow.cwl`` types the input as ``string?``; the
    simclient's ``_normalise_workflow_vars`` joins it before CWL validation.
    Without the join, ``cwltool --validate`` rejects the list ("CommentedSeq,
    expected null or string") and any submission using ``-o`` dies as
    ``RUN_FAILED``.
    """
    import shutil
    import subprocess

    from picongpu.pypicongpu.runner import Runner

    from pic_agentic.simclient.simulation import _normalise_workflow_vars

    dump = json.loads(FIXTURE.read_text())
    runner = Runner(
        sim=dump["sim"],
        setup_dir=str(tmp_path / "input"),
        run_dir=str(tmp_path / "run"),
    )
    runner.generate(o=["A=1", "PATH_X=/a/b-c.d:e"], submit="sbatch")
    raw = runner.workflow_input_path.read_text(encoding="utf-8")
    assert isinstance(json.loads(raw)["run_overwrite_vars"], list)
    _normalise_workflow_vars(runner.setup_dir)
    assert json.loads(runner.workflow_input_path.read_text(encoding="utf-8"))["run_overwrite_vars"] == (
        "A=1 PATH_X=/a/b-c.d:e"
    )
    cwltool = shutil.which("cwltool") or _cwltool_beside_interpreter()
    if cwltool is None:  # pragma: no cover - only when the [sim] venv lacks cwltool
        pytest.skip("cwltool not available")
    result = subprocess.run(
        [cwltool, "--validate", str(runner.workflow_definition_path), str(runner.workflow_input_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "is valid CWL" in result.stdout


@pytest.mark.integration
def test_nested_unknown_field_is_rejected(tmp_path: Path) -> None:
    """A field outside the pinned schema must not be silently dropped."""
    from pic_agentic.simclient.simulation import (
        SimulationErrorCode,
        SimulationExecutionError,
        SubmitConfig,
        runner_from_payload,
    )

    dump = json.loads(FIXTURE.read_text())
    dump["sim"]["totally_unknown_key"] = 123
    payload = SimulationPayload.build(
        picongpu_version=picongpu_version(),
        picongpu_revision=picongpu_revision(),
        schema_hash=runner_schema_hash(),
        runner_dump=dump,
    )
    config = SubmitConfig(setup_root=tmp_path)
    with pytest.raises(SimulationExecutionError) as excinfo:
        runner_from_payload(payload, config, "deadbeef")
    assert excinfo.value.code is SimulationErrorCode.UNSUPPORTED


@pytest.mark.integration
def test_fixture_validates_against_real_runner() -> None:
    from picongpu.pypicongpu.runner import Runner

    dump = json.loads(FIXTURE.read_text())
    runner = Runner.model_validate(dump)
    assert runner.model_dump(mode="json") == dump
    payload = SimulationPayload.build(
        picongpu_version=picongpu_version(),
        picongpu_revision=picongpu_revision(),
        schema_hash=runner_schema_hash(),
        runner_dump=dump,
    )
    spec = simulation_spec_from_runner_dump(dump)
    assert payload.simulation == spec
    # The payload hash is stable across a validation round trip.
    reloaded = SimulationPayload.model_validate_json(payload.model_dump_json(exclude_computed_fields=True))
    assert reloaded.payload_hash == payload.payload_hash
