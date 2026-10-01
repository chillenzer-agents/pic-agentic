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
async def test_trailing_sim_run_builds_without_compiling(tmp_path: Path) -> None:
    """The documented trailing ``sim.run(...)`` must not trigger a build.

    Regression (live beta run): ``sim.run()`` fell through to a real
    ``cwltool`` compile in the child and surfaced as a raw ``permanentFail``.
    ``build_spec`` only wants the ``Simulation``, so the child neutralises
    ``run``/``write_input_file``; the spec must come back and no compile dirs
    may be created.
    """
    import asyncio

    from pic_agentic.simulation_build import build_runner_dump

    script = tmp_path / "sim.py"
    script.write_text(
        "from picongpu import picmi\n"
        "grid = picmi.Cartesian3DGrid(number_of_cells=[8, 8, 8], lower_bound=[0, 0, 0], "
        "upper_bound=[1e-6, 1e-6, 1e-6], lower_boundary_conditions=['periodic'] * 3, "
        "upper_boundary_conditions=['periodic'] * 3)\n"
        "solver = picmi.ElectromagneticSolver(method='Yee', grid=grid)\n"
        "sim = picmi.Simulation(time_step_size=1e-15, max_steps=2, solver=solver)\n"
        "sim.run(setup_dir=str(__import__('pathlib').Path(__file__).with_name('setup')), "
        "run_dir=str(__import__('pathlib').Path(__file__).with_name('run')))\n",
        encoding="utf-8",
    )
    built = await asyncio.wait_for(build_runner_dump(script_path=script), timeout=120)
    assert built.runner["sim"]
    assert not (tmp_path / "setup").exists()
    assert not (tmp_path / "run").exists()
    assert built.schema_hash == runner_schema_hash()


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
def test_check_spec_round_trip_matches_the_simclient_gate() -> None:
    """The create-time check agrees with the simclient's round-trip gate (P2b)."""
    from pic_agentic.simulation_build import check_spec_round_trip

    specs = json.loads((Path(__file__).parent / "fixtures" / "campaign_specs.json").read_text())
    for index in (0, 1):
        detail = check_spec_round_trip(specs[index])
        assert detail is not None
        assert "collisional_physics.numerics_config.num_tmp_field_slots" in detail
        assert "collisional_physics.num_tmp_field_slots" in detail
    # The correctly-shaped spec is accepted unchanged.
    assert check_spec_round_trip(specs[2]) is None


@pytest.mark.integration
def test_check_spec_round_trip_rejects_a_validation_failure() -> None:
    """A spec the pin refuses to validate is rejected, not swallowed (beta-3).

    The beta-3 agent fed ``create_campaign`` LLM-typed specs whose non-null
    fields held ``""`` where a list/dict/tuple is expected.  ``Runner`` rejects
    them (the simclient answers ``payload_invalid``), but the check used to
    return ``None`` on any validation failure, so the campaign persisted and
    every leaf then failed at submit time.  The check must name the offending
    fields instead.
    """
    from pic_agentic.simulation_build import check_spec_round_trip

    bad = json.loads((Path(__file__).parent / "fixtures" / "beta3_bad_spec.json").read_text())
    detail = check_spec_round_trip(bad)
    assert detail is not None
    assert "sim.customuserinput Input should be a valid list" in detail
    assert "sim.moving_window" in detail
    # A single type-invalid nested leaf is named precisely.
    dump = json.loads(FIXTURE.read_text())
    dump["sim"]["species"] = ""
    detail = check_spec_round_trip(dump)
    assert detail is not None
    assert "sim.species Input should be a valid list (got str '')" in detail


@pytest.mark.integration
async def test_create_campaign_rejects_the_beta_bad_spec_and_persists_nothing(tmp_path) -> None:
    """The exact beta-3 spec is refused at creation with nothing persisted."""
    from pic_agentic.config import Config
    from pic_agentic.rcp import new_secret_hex
    from pic_agentic.server.app import build_server
    from pic_agentic.transport.memory import MemoryTransport

    bad = json.loads((Path(__file__).parent / "fixtures" / "beta3_bad_spec.json").read_text())
    campaign_file = tmp_path / "campaign.json"
    config = Config(rcp_secret=new_secret_hex(), agenda_file=str(campaign_file))
    mcp_transport, sim_transport = MemoryTransport.create_pair()
    server, runtime = build_server(config, "7f3a2b1c")
    runtime._transport = mcp_transport
    try:
        # create_campaign only validates and persists; it starts no cluster work,
        # so no sim responder/pump is needed.
        result = (
            await server.call_tool(
                "create_campaign",
                {"name": "beta3", "base_spec": bad, "patch_path": "sim.time_steps", "values": [100, 200]},
            )
        ).structured_content
    finally:
        await mcp_transport.close()
        await sim_transport.close()
    assert result["ok"] is False
    assert result["error"] == "invalid_campaign_spec"
    assert str(result["detail"]).startswith("spec does not validate against the pinned pypicongpu schema")
    assert "sim.customuserinput Input should be a valid list" in result["detail"]
    assert not campaign_file.exists()


@pytest.mark.integration
def test_check_spec_round_trip_ignores_sibling_keys_like_the_simclient() -> None:
    """The check replays the simclient's *reduced* dump (B1).

    The simclient validates only ``{sim, setup_dir, run_dir}`` (+template_dir)
    and drops every other key, so a full ``build_spec`` dump -- or one whose
    sibling ``template_dir``/``setup_dir`` has a shape ``Runner`` would reject --
    must not be refused by the create-time check.
    """
    from pic_agentic.simulation_build import check_spec_round_trip

    dump = json.loads(FIXTURE.read_text())
    assert check_spec_round_trip(dump) is None
    assert check_spec_round_trip({"sim": dump["sim"], "template_dir": "not-a-list"}) is None
    assert check_spec_round_trip({"sim": dump["sim"], "setup_dir": 123}) is None


@pytest.mark.integration
async def test_check_spec_round_trip_accepts_a_fresh_build_spec_dump(tmp_path) -> None:
    """A dump built from the real pin passes the create-time check (B1)."""
    from pic_agentic.simulation_build import build_runner_dump, check_spec_round_trip

    script = tmp_path / "sim.py"
    script.write_text(
        "from picongpu import picmi\n"
        "grid = picmi.Cartesian3DGrid(number_of_cells=[8, 8, 8], lower_bound=[0, 0, 0], "
        "upper_bound=[1e-6, 1e-6, 1e-6], lower_boundary_conditions=['periodic'] * 3, "
        "upper_boundary_conditions=['periodic'] * 3)\n"
        "solver = picmi.ElectromagneticSolver(method='Yee', grid=grid)\n"
        "sim = picmi.Simulation(time_step_size=1e-15, max_steps=4, solver=solver)\n",
        encoding="utf-8",
    )
    import sys

    built = await build_runner_dump(script_path=script, interpreter=sys.executable)
    assert check_spec_round_trip(built.runner) is None
    assert check_spec_round_trip({"sim": built.runner["sim"]}) is None


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
