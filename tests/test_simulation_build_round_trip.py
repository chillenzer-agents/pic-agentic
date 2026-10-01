# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Offline tests for :func:`pic_agentic.simulation_build.check_spec_round_trip`.

These run without PIConGPU: they pin the best-effort fallback and, crucially,
that the check never over-rejects a genuinely valid spec because of sibling
keys the simclient drops or metadata the pin would recompute.  The exact-pin
behaviour is covered by ``test_submit_integration.py`` (B1).
"""

from __future__ import annotations

import json
from pathlib import Path

from pic_agentic.simulation_build import check_spec_round_trip

CAMPAIGN_SPECS = Path(__file__).parent / "fixtures" / "campaign_specs.json"
RUNNER_FIXTURE = Path(__file__).parent / "fixtures" / "pypicongpu_runner.json"


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


def test_fallback_is_documented_best_effort_for_unknown_fields() -> None:
    """Offline the fallback cannot see arbitrary unknown fields (documented)."""
    sim = {**_runner_dump()["sim"], "totally_unknown_key": 1}
    # No pin importable here, so the curated detector passes it; the exact check
    # (integration test) rejects it.  This pins the documented degradation.
    assert check_spec_round_trip({"sim": sim}) is None
