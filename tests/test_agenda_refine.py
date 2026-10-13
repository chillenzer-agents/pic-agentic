# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for the deterministic sweep-refinement helpers (gap 2)."""

from __future__ import annotations

import pytest

from pic_agentic.agenda.refine import best_point, is_converged, refine_around, summary


def test_best_point_picks_the_max_and_ignores_none() -> None:
    assert best_point({"a": 1.0, "b": 3.0, "c": None}) == ("b", 3.0)
    assert best_point({"a": None, "b": None}) is None
    assert best_point({}) is None


def test_refine_around_brackets_the_best() -> None:
    points = {"a": 1.0, "b": 2.0}
    suggestions = refine_around(points, fraction=0.1, count=2)
    assert [s.label for s in suggestions] == ["b+0.1", "b-0.1"]
    assert suggestions[0].value == pytest.approx(2.1)
    assert suggestions[1].value == pytest.approx(1.9)


def test_refine_around_clamps_non_positive_and_caps_count() -> None:
    # Best at 0.05 with a single sample: step = 0.05 * 0.1 = 0.005; the negative
    # side would be 0.045 (positive, kept), the second negative 0.04 ...
    suggestions = refine_around({"a": 0.05}, fraction=0.1, count=1)
    assert len(suggestions) == 1
    assert suggestions[0].value > 0

    # A best near zero whose negative offsets would be non-positive are dropped.
    dropped = refine_around({"a": 1e-9}, fraction=1.0, count=2)
    assert all(s.value > 0 for s in dropped)


def test_refine_around_empty_and_zero_count() -> None:
    assert refine_around({}) == []
    assert refine_around({"a": 1.0}, count=0) == []


def test_is_converged_requires_two_samples_within_tolerance() -> None:
    assert is_converged({"a": 1.0, "b": 1.01}, rel_tol=0.05) is True
    assert is_converged({"a": 1.0, "b": 2.0}, rel_tol=0.05) is False
    assert is_converged({"a": 1.0}, rel_tol=0.05) is False  # one sample
    assert is_converged({}) is False


def test_summary_shape_and_no_suggestions_when_converged() -> None:
    converged = summary({"a": 1.0, "b": 1.0}, rel_tol=0.05)
    assert converged["best"] == {"label": "b", "value": 1.0} or converged["best"]["value"] == pytest.approx(1.0)
    assert converged["converged"] is True
    assert converged["suggestions"] == []
    assert converged["analysed"] == 2

    open_sweep = summary({"a": 1.0, "b": 5.0}, rel_tol=0.05)
    assert open_sweep["converged"] is False
    assert open_sweep["best"]["value"] == pytest.approx(5.0)
    assert open_sweep["suggestions"]


def test_summary_without_any_analysis_reports_no_best() -> None:
    """A sweep with no analysed leaf must not invent a best point.

    Regression for F1: before the fix ``summary`` ranked ``None``-free point
    values, so a sweep that had only been *simulated* looked converged.
    """
    result = summary({"leaf000": None, "leaf001": None, "leaf002": None})
    assert result["best"] is None
    assert result["analysed"] == 0
    assert result["converged"] is False
    assert result["suggestions"] == []

    assert summary({})["best"] is None
    assert summary({})["analysed"] == 0


def test_summary_ranks_only_recorded_analyses() -> None:
    """``best`` follows the analyses, ignoring leaves that are not analysed."""
    result = summary({"leaf000": 0.31, "leaf001": None, "leaf002": 0.29}, rel_tol=0.05)
    assert result["best"] == {"label": "leaf000", "value": pytest.approx(0.31)}
    assert result["analysed"] == 2


# --------------------------------------------------------------------------
# F2 (beta-7): an edge optimum must not be reported as converged.
# --------------------------------------------------------------------------


def test_edge_optimum_low_is_not_converged_and_extends_downward() -> None:
    """Best at the low edge with an improving trend -> extend below it."""
    points = {"leaf000": 100.0, "leaf001": 99.0, "leaf002": 98.0}
    coordinates = {"leaf000": 4.0, "leaf001": 4.6, "leaf002": 5.2}
    result = summary(points, rel_tol=0.05, coordinates=coordinates)
    assert result["best"] == {"label": "leaf000", "value": pytest.approx(100.0)}
    assert result["converged"] is False
    assert [s["label"] for s in result["suggestions"]] == ["leaf000-0.12", "leaf000-0.24"]
    assert all(s["value"] < 4.0 for s in result["suggestions"])


def test_edge_optimum_high_is_not_converged_and_extends_upward() -> None:
    """Best at the high edge with an improving trend -> extend above it."""
    points = {"leaf000": 98.0, "leaf001": 99.0, "leaf002": 100.0}
    coordinates = {"leaf000": 4.0, "leaf001": 4.6, "leaf002": 5.2}
    result = summary(points, rel_tol=0.05, coordinates=coordinates)
    assert result["converged"] is False
    assert [s["label"] for s in result["suggestions"]] == ["leaf002+0.12", "leaf002+0.24"]
    assert all(s["value"] > 5.2 for s in result["suggestions"])


def test_interior_flat_optimum_is_converged() -> None:
    """A flat optimum whose best sits inside the range stays converged."""
    result = summary(
        {"a": 0.30, "b": 0.31, "c": 0.30},
        rel_tol=0.05,
        coordinates={"a": 1.0, "b": 2.0, "c": 3.0},
    )
    assert result["best"] == {"label": "b", "value": pytest.approx(0.31)}
    assert result["converged"] is True
    assert result["suggestions"] == []


def test_boundary_plateau_without_outward_slope_is_converged() -> None:
    """A flat edge plateau without an outward slope stays converged.

    The edge alone is not enough: there must be an improving trend.
    """
    result = summary(
        {"a": 5.0, "b": 5.0, "c": 5.0},
        rel_tol=0.05,
        coordinates={"a": 1.0, "b": 2.0, "c": 3.0},
    )
    assert result["converged"] is True
    assert result["suggestions"] == []


def test_monotone_trend_is_not_converged() -> None:
    """A strictly monotone sweep is one-sided by construction: never converged."""
    points = {"a": 1.0, "b": 2.0, "c": 3.0}
    coordinates = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert is_converged(points, rel_tol=0.05, coordinates=coordinates) is False
    # Without coordinates the strictly increasing top-two gap already fails the
    # tolerance check; with coordinates it must also fail even if the gap is small.
    flat_monotone = {"a": 1.00, "b": 1.01, "c": 1.02}
    assert is_converged(flat_monotone, rel_tol=0.05) is True
    assert is_converged(flat_monotone, rel_tol=0.05, coordinates=coordinates) is False


def test_low_sample_behaviour_is_unchanged_with_coordinates() -> None:
    """A single sample is not converged; the edge logic needs a frame."""
    assert is_converged({"a": 1.0}, coordinates={"a": 1.0}) is False
    assert is_converged({}, coordinates={}) is False


def test_beta7_focal_scan_edge_optimum_regression() -> None:
    """Exact beta-7 numbers: the 3-point focal scan must not read converged.

    With scores for 4.0/4.6/5.2e-5 descending, the best (``leaf000``) sits on
    the range minimum and still improves toward it.  Before the fix the tool
    reported ``converged: true`` and ``suggestions: []``; it must now be open
    and propose points below 4.0e-5.
    """
    points = {"leaf000": 66715949903.0, "leaf001": 65595828126.0, "leaf002": 64137641778.0}
    coordinates = {"leaf000": 4.0e-5, "leaf001": 4.6e-5, "leaf002": 5.2e-5}
    result = summary(points, rel_tol=0.05, coordinates=coordinates)
    assert result["best"] == {"label": "leaf000", "value": pytest.approx(66715949903.0)}
    assert result["converged"] is False
    assert result["suggestions"]
    assert all(s["value"] < 4.0e-5 for s in result["suggestions"])
