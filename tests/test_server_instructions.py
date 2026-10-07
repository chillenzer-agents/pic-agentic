# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""The seeded MCP instructions must point an unfamiliar agent at PIConGPU docs."""

from __future__ import annotations

import pytest

from pic_agentic.config import Config
from pic_agentic.rcp import new_secret_hex
from pic_agentic.server.app import SERVER_INSTRUCTIONS, build_server

SIM = "7f3a2b1c"


async def test_instructions_are_seeded_on_the_server() -> None:
    server, _runtime = build_server(Config(rcp_secret=new_secret_hex()), SIM)
    assert server.instructions == SERVER_INSTRUCTIONS


def test_instructions_point_at_the_picongpu_documentation_and_examples() -> None:
    # The onboarding contract: an agent with no PICMI knowledge can find how to
    # define and scan simulations, and where the worked examples are.
    assert "defining_simulation" in SERVER_INSTRUCTIONS
    assert "readthedocs" in SERVER_INSTRUCTIONS
    assert "lib/python/examples" in SERVER_INSTRUCTIONS
    # The campaign-entry tools the agent needs are named.
    assert "build_spec" in SERVER_INSTRUCTIONS
    assert "create_campaign" in SERVER_INSTRUCTIONS
    # The reset path for starting a fresh campaign is named too.
    assert "delete_campaign" in SERVER_INSTRUCTIONS
    # H5: the add_agenda_leaf escape hatch for multi-node/varying-node studies
    # and its by-reference spec_path form are named.
    assert "add_agenda_leaf" in SERVER_INSTRUCTIONS
    assert "spec_path=..." in SERVER_INSTRUCTIONS
    # L5: the "which tool when" front-door line is present.
    assert "Which tool" in SERVER_INSTRUCTIONS
    # H8: the documented focal example is empty for timing/geometry reasons, not
    # merely a missing plasma species; the corrected caveat names both and the
    # actionable fix. Assert the invariant semantics rather than exact prose, so
    # the wording can be tightened without breaking the test.
    assert "plasma species" in SERVER_INSTRUCTIONS
    assert "gas" in SERVER_INSTRUCTIONS
    assert "max_steps" in SERVER_INSTRUCTIONS
    assert "necessary but not sufficient" in SERVER_INSTRUCTIONS
    # H9: the pinned API is authoritative over readthedocs; the mismatched names
    # and the version-matched classes are named.
    assert "installed" in SERVER_INSTRUCTIONS
    assert "picongpu" in SERVER_INSTRUCTIONS
    assert "FieldDiagnostic" in SERVER_INSTRUCTIONS
    assert "picmi.Cartesian3DGrid" in SERVER_INSTRUCTIONS
    assert "picmi.UniformDistribution" in SERVER_INSTRUCTIONS
    # The list-indexed patch_path form is publicised, not only the top-level
    # sim.time_steps example.
    assert "sim.laser.0.focus_pos_si.1.component" in SERVER_INSTRUCTIONS


def test_instructions_warn_about_rendered_snapshot_and_denormalized_fields() -> None:
    # E1 (A1's trap): a Runner spec is a rendered snapshot with denormalized
    # fields, so a fixed-box resolution patch must co-vary the whole
    # grid/time_description set. Patching only cell_cnt (or only cell_size) is a
    # box-size change, not a resolution change -- and box-size sweeps are
    # allowed, so the text must not claim a lone cell_cnt patch is refused.
    for field in ("cell_cnt", "cell_size", "cell_depth", "delta_t_si", "time_steps"):
        assert field in SERVER_INSTRUCTIONS
    assert "denormalized" in SERVER_INSTRUCTIONS
    assert "co-vary" in SERVER_INSTRUCTIONS
    assert "box-size" in SERVER_INSTRUCTIONS
    assert "box-size sweeps are allowed" in SERVER_INSTRUCTIONS
    assert "lone cell_cnt patch" in SERVER_INSTRUCTIONS
    # The old refusal claim is gone: no consistency gate rejects the patch.
    assert "refused" not in SERVER_INSTRUCTIONS
    assert "consistency gate" not in SERVER_INSTRUCTIONS
    # The fixed-box route is spelled out: a whole sim.grid patch, or one whole
    # spec per leaf via add_agenda_leaf(spec_path=...).
    assert "sim.grid" in SERVER_INSTRUCTIONS
    assert "add_agenda_leaf(spec_path=...)" in SERVER_INSTRUCTIONS


def test_instructions_warn_about_the_cfl_condition() -> None:
    # E1 (2): finer dx needs a CFL-consistent delta_t_si/step count; changing
    # only the grid fails the compile with the Yee CFL static_assert.
    assert "CFL" in SERVER_INSTRUCTIONS
    assert "static_assert" in SERVER_INSTRUCTIONS
    assert "delta_t_si" in SERVER_INSTRUCTIONS


def test_instructions_warn_about_the_super_cell_stencil_constraint() -> None:
    # F8 / beta-7: a super cell smaller than the default is not a mere
    # performance knob -- the default particle shape's stencil needs the
    # default (8, 8, 4) [3D] / (16, 16) [2D] super cell, and (2, 2, 2) fails
    # the build with the Esirkepov static_assert. Assert the invariant, not the
    # exact prose.
    assert "picongpu_super_cell_size" in SERVER_INSTRUCTIONS
    assert "(8, 8, 4)" in SERVER_INSTRUCTIONS
    assert "Esirkepov" in SERVER_INSTRUCTIONS
    assert "too small for stencil" in SERVER_INSTRUCTIONS


def test_instructions_warn_that_the_huygens_surface_must_clear_the_pml() -> None:
    # E1 (3): an 8-cell Huygens placement sits inside the 12-cell PML absorber
    # and segfaults at step 0; the GaussianLaser 16-cell default is right.
    assert "Huygens" in SERVER_INSTRUCTIONS
    assert "PML" in SERVER_INSTRUCTIONS
    assert "12-cell" in SERVER_INSTRUCTIONS
    # 12 cells is the default PML; the exponential absorber uses 32.
    assert "default PML" in SERVER_INSTRUCTIONS
    assert "16 cells" in SERVER_INSTRUCTIONS
    assert "exit 139" in SERVER_INSTRUCTIONS


def test_instructions_carry_a_field_energy_monitor_snippet() -> None:
    # E1: the beta-6 agent reverse-engineered TimeStepSpec[::20, -1] from
    # source; the seeded snippet must name the pinned classes and the slice
    # syntax inline, in the same snippet mechanism as the base script.
    assert "picmi.diagnostics.FieldEnergyMonitor" in SERVER_INSTRUCTIONS
    assert "picmi.diagnostics.TimeStepSpec[::20, -1]" in SERVER_INSTRUCTIONS
    assert "diagnostics=[...]" in SERVER_INSTRUCTIONS


def test_instructions_report_time_step_spec_schema_introspection_gap() -> None:
    # E2: model_json_schema() fails for TimeStepSpec-backed diagnostics because
    # the pin's TimeStepSpec is a plain class, not a pydantic model; point the
    # agent at the pinned class list instead. Attribute it to the pin, not to
    # the PICMI standard, which does not ship the class.
    assert "model_json_schema" in SERVER_INSTRUCTIONS
    assert "not a pydantic model" in SERVER_INSTRUCTIONS
    assert "pin's TimeStepSpec" in SERVER_INSTRUCTIONS
    assert "upstream PICMI" not in SERVER_INSTRUCTIONS
    assert "picmi/diagnostics/" in SERVER_INSTRUCTIONS


def test_field_energy_monitor_snippet_is_valid_against_the_pin() -> None:
    # Exercise the snippet exactly as seeded: the same classes and slice syntax
    # must construct and attach to a pinned picmi.Simulation. Skipped where the
    # picongpu pin is absent (the offline suite).
    pytest.importorskip("picongpu")
    from picongpu import picmi

    period = picmi.diagnostics.TimeStepSpec[::20, -1]
    monitor = picmi.diagnostics.FieldEnergyMonitor(period=period)
    grid = picmi.Cartesian3DGrid(
        number_of_cells=[32, 32, 32],
        lower_bound=[0.0, 0.0, 0.0],
        upper_bound=[1e-6, 1e-6, 1e-6],
        lower_boundary_conditions=["open"] * 3,
        upper_boundary_conditions=["open"] * 3,
    )
    solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.95, grid=grid)
    electrons = picmi.Species(
        name="electrons",
        particle_type="electron",
        initial_distribution=picmi.UniformDistribution(density=1e24, rms_velocity=[1e6, 1e6, 1e6]),
    )
    sim = picmi.Simulation(
        max_steps=10,
        solver=solver,
        species=[electrons],
        layouts=[picmi.PseudoRandomLayout(n_macroparticles_per_cell=2)],
        diagnostics=[monitor],
    )
    assert sim is not None


async def test_create_campaign_description_publicises_the_list_indexed_path() -> None:
    # The beta-4 agent found the indexed patch_path form only by reading source;
    # the tool description must name it so an agent can scan a nested list field.
    server, _runtime = build_server(Config(rcp_secret=new_secret_hex()), SIM)
    tools = {tool.name: tool for tool in await server.list_tools()}
    description = tools["create_campaign"].description
    assert "sim.laser.0.focus_pos_si.1.component" in description
    # The rule must match the patcher: the *node* decides, so a numeric segment
    # is a list index on a list but a dict key on a dict (the ``sim.bc.0``
    # boundary-condition map).  A spelling-based "numeric means list" rule would
    # contradict ``_patch_spec`` and ``test_create_campaign_reaches_a_numeric_dict_key``.
    assert "a list index on a list, or a " in description
    assert "dict key on a dict" in description


async def test_add_agenda_leaf_description_publicises_the_escape_hatch() -> None:
    # H5: a study that must vary more than one spec node cannot use create_campaign's
    # single patch_path; add_agenda_leaf is the documented escape hatch and takes a
    # whole spec by reference.
    server, _runtime = build_server(Config(rcp_secret=new_secret_hex()), SIM)
    tools = {tool.name: tool for tool in await server.list_tools()}
    description = tools["add_agenda_leaf"].description
    assert "escape hatch" in description
    assert "spec_path" in description
    assert "grid cells and time_steps" in description
