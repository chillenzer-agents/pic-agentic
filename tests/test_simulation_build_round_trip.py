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
from pathlib import Path

import pytest

from pic_agentic.simulation_build import check_spec_round_trip

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
