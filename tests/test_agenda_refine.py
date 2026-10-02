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
