# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Offline tests for :func:`pic_agentic.simulation_build.check_spec_round_trip`.

These run without PIConGPU: they pin the best-effort fallback and that the check
never over-rejects a valid spec over sibling keys the simclient drops.  The
exact-pin behaviour (including the exact-parity rejections) is covered by
``test_submit_integration.py``.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from pic_agentic.simulation_build import check_spec_consistency, check_spec_round_trip

#: Whether the pinned PIConGPU is importable in this interpreter.  The fallback
#: tests only describe the no-pin behaviour; the exact check is covered by the
#: integration suite.
_HAS_PICONGPU = importlib.util.find_spec("picongpu") is not None

CAMPAIGN_SPECS = Path(__file__).parent / "fixtures" / "campaign_specs.json"
RUNNER_FIXTURE = Path(__file__).parent / "fixtures" / "pypicongpu_runner.json"
BETA_BAD_SPEC = Path(__file__).parent / "fixtures" / "beta3_bad_spec.json"


def _campaign_spec(index: int) -> dict:
    return json.loads(CAMPAIGN_SPECS.read_text(encoding="utf-8"))[index]


def _runner_dump() -> dict:
    return json.loads(RUNNER_FIXTURE.read_text(encoding="utf-8"))


def test_fallback_reports_a_misplaced_computed_field() -> None:
    """The curated fallback still catches the known bad shape (P2b)."""
    for index in (0, 1):
        detail = check_spec_round_trip(_campaign_spec(index))
        assert detail is not None
        assert "collisional_physics.numerics_config.num_tmp_field_slots" in detail
        assert "collisional_physics.num_tmp_field_slots" in detail
    # The correctly-shaped spec is accepted unchanged.
    assert check_spec_round_trip(_campaign_spec(2)) is None


def test_valid_spec_with_sibling_keys_is_not_rejected() -> None:
    """A full runner dump or odd sibling keys never trip the check (B1).

    The simclient validates a *reduced* dump and drops every other key, so the
    create-time check must not reject over a ``template_dir``/``setup_dir``
    shape it ignores.
    """
    dump = _runner_dump()
    assert check_spec_round_trip(dump) is None
    assert check_spec_round_trip({"sim": dump["sim"]}) is None
    assert check_spec_round_trip({"sim": dump["sim"], "template_dir": "not-a-list"}) is None
    assert check_spec_round_trip({"sim": dump["sim"], "setup_dir": 123}) is None


@pytest.mark.skipif(_HAS_PICONGPU, reason="behaviour is about the no-pin fallback")
def test_fallback_is_documented_best_effort_for_unknown_fields() -> None:
    """Offline the fallback cannot see arbitrary unknown fields (documented)."""
    sim = {**_runner_dump()["sim"], "totally_unknown_key": 1}
    # No pin importable here, so the curated detector passes it; the exact check
    # (integration test) rejects it.  This pins the documented degradation.
    assert check_spec_round_trip({"sim": sim}) is None


@pytest.mark.skipif(not _HAS_PICONGPU, reason="needs the pinned pypicongpu schema")
def test_validation_failure_is_rejected_with_the_offending_field() -> None:
    """A spec the pin cannot validate is refused, naming the field (beta-3).

    The beta-3 defect was that ``check_spec_round_trip`` swallowed the
    ``Runner.model_validate`` failure and returned ``None``; the simclient then
    rejected every leaf as ``payload_invalid``.  The check must now reject and
    paraphrase the pydantic error concisely.
    """
    dump = _runner_dump()
    dump["sim"]["species"] = ""
    detail = check_spec_round_trip(dump)
    assert detail is not None
    assert detail.startswith("spec does not validate against the pinned pypicongpu schema")
    assert "sim.species Input should be a valid list (got str '')" in detail
    # A field the pin is happy with is still accepted.
    assert check_spec_round_trip({"sim": _runner_dump()["sim"]}) is None


@pytest.mark.skipif(not _HAS_PICONGPU, reason="needs the pinned pypicongpu schema")
def test_beta3_bad_spec_is_rejected_by_the_check() -> None:
    """The exact LLM-typed beta-3 spec is refused at the round-trip gate."""
    bad = json.loads(BETA_BAD_SPEC.read_text(encoding="utf-8"))
    detail = check_spec_round_trip(bad)
    assert detail is not None
    assert detail.startswith("spec does not validate against the pinned pypicongpu schema")
    assert "sim.customuserinput Input should be a valid list" in detail
    assert "sim.moving_window" in detail
    # The message is bounded even for a many-error spec.
    assert "more error(s)" in detail


def test_consistency_ignores_a_patch_that_does_not_touch_the_grid() -> None:
    """An unrelated patch never trips the derived-invariant check (A1, no false reject)."""
    dump = _campaign_spec(2)
    assert check_spec_consistency(dump, "sim.laser.0.focus_pos_si.1.component") is None
    assert check_spec_consistency(dump, "sim.time_steps") is None


def test_consistency_flags_a_stale_cell_depth() -> None:
    """A ``cell_size`` patch that leaves ``cell_depth`` stale is reported (A1)."""
    dump = _campaign_spec(2)
    dump["sim"]["grid"]["cell_size"] = {"x": 6.25e-8, "y": 6.25e-8, "z": 6.25e-8}
    detail = check_spec_consistency(dump, "sim.grid.cell_size")
    assert detail is not None
    assert "cell_depth" in detail
    assert "sim.grid.cell_size.z" in detail


def test_consistency_flags_a_cfl_violation_on_a_delta_t_patch() -> None:
    """A ``delta_t_si`` patch beyond the Yee CFL limit is reported (A1)."""
    dump = _campaign_spec(2)
    dump["sim"]["delta_t_si"] = 5.0e-15
    detail = check_spec_consistency(dump, "sim.delta_t_si")
    assert detail is not None
    assert "CFL" in detail
    assert "sim.delta_t_si" in detail


def test_consistency_skips_the_lehe_and_none_solvers() -> None:
    """Solvers whose CFL limit is undetermined from the wire are skipped (A1)."""
    for solver in ({"type_lehe": True, "name": "Lehe<>"}, {"type_none": True, "name": "None"}):
        dump = _campaign_spec(2)
        dump["sim"]["solver"] = solver
        dump["sim"]["delta_t_si"] = 5.0e-15
        assert check_spec_consistency(dump, "sim.delta_t_si") is None


def test_consistency_matches_the_pin_cfl_limit_for_ao() -> None:
    """The arbitrary-order FDTD limit follows the pin's CFLChecker arithmetic (A1)."""
    dump = _campaign_spec(2)
    dump["sim"]["solver"] = {"type_arbitraryorderfdtd": True, "name": "ArbitraryOrderFDTD<2>", "neighbors": 2}
    # dt chosen to sit exactly at the (tighter) AO limit is accepted, above it is not.
    cell = [1.772e-7, 4.43e-8, 1.772e-7]
    yee = 1.0 / math.sqrt(sum(1.0 / d**2 for d in cell))
    ao_limit = yee / (7.0 / 6.0)
    dump["sim"]["delta_t_si"] = ao_limit / 299792458.0
    assert check_spec_consistency(dump, "sim.delta_t_si") is None
    dump["sim"]["delta_t_si"] = ao_limit * 1.01 / 299792458.0
    assert check_spec_consistency(dump, "sim.delta_t_si") is not None


def test_consistency_cfl_slack_is_only_a_tiny_relative_tolerance() -> None:
    """The CFL check is the pin limit up to ``1e-6`` relative slack, no more (A1).

    The slack keeps a spec sitting exactly on the pin's assert from being a
    false rejection; it must not accept a materially over-limit ``delta_t_si``.
    """
    dump = _campaign_spec(2)
    cell = list(dump["sim"]["grid"]["cell_size"].values())
    limit = 1.0 / math.sqrt(sum(1.0 / d**2 for d in cell))
    # Exactly on the limit and within the 1e-6 slack: accepted.
    dump["sim"]["delta_t_si"] = limit * (1.0 + 1e-7) / 299792458.0
    assert check_spec_consistency(dump, "sim.delta_t_si") is None
    # Just beyond the slack: rejected.
    dump["sim"]["delta_t_si"] = limit * (1.0 + 1e-3) / 299792458.0
    assert check_spec_consistency(dump, "sim.delta_t_si") is not None


def test_consistency_flags_a_grid_dist_that_no_longer_sums() -> None:
    """A ``cell_cnt`` patch that leaves ``grid_dist`` stale is reported (A1).

    The explicit distribution uses the *real* pin wire shape,
    ``{"device_cells": N}`` per GPU (``pypicongpu/grid.py``
    ``serialise_grid_dist3``); the flat form is rejected by the pin's own
    round-trip, so only this shape can reach the consistency check.
    """
    dump = _campaign_spec(2)
    dump["sim"]["grid"]["grid_dist"] = {
        "x": [{"device_cells": 96}, {"device_cells": 96}],
        "y": [{"device_cells": 1024}, {"device_cells": 1024}],
        "z": [{"device_cells": 96}, {"device_cells": 96}],
    }
    dump["sim"]["grid"]["cell_cnt"] = {"x": 200, "y": 2048, "z": 192}
    detail = check_spec_consistency(dump, "sim.grid.cell_cnt")
    assert detail is not None
    assert "grid_dist" in detail
    assert "sums to 192" in detail


def test_consistency_accepts_a_consistent_real_grid_dist() -> None:
    """A real-shape ``grid_dist`` that does sum to ``cell_cnt`` is accepted (A1).

    Guards against over-rejecting once the ``{"device_cells": N}`` shape is
    understood: the base counts and the distribution agree here.
    """
    dump = _campaign_spec(2)
    dump["sim"]["grid"]["grid_dist"] = {
        "x": [{"device_cells": 192}],
        "y": [{"device_cells": 2048}],
        "z": [{"device_cells": 192}],
    }
    assert check_spec_consistency(dump, "sim.grid.cell_cnt") is None


@pytest.mark.skipif(_HAS_PICONGPU, reason="documents the no-pin behaviour")
def test_consistency_is_exact_without_the_pin() -> None:
    """The consistency check is pure arithmetic, so it never degrades to a false reject (A1).

    Unlike the round-trip gate (which is best-effort offline), the derived
    invariants depend only on the wire spec, so the same stale ``cell_depth`` is
    reported with or without PIConGPU importable.
    """
    dump = _campaign_spec(2)
    dump["sim"]["grid"]["cell_size"] = {"x": 6.25e-8, "y": 6.25e-8, "z": 6.25e-8}
    assert check_spec_consistency(dump, "sim.grid.cell_size") is not None
    # A valid spec is still accepted offline.
    assert check_spec_consistency(_campaign_spec(2), "sim.grid") is None
