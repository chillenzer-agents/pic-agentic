# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Deterministic sweep-refinement helpers (research-loop gap 2).

A campaign sweeps a parameter, analyses each run, and refines around the
optimum until it converges.  The *decision* logic -- which point is best, which
points to add next, whether the sweep has converged -- is small, pure and
testable, so it lives here rather than being hand-computed by the agent each
time.  Extraction-ready: stdlib only.

A sample is a ``{label: value}`` mapping (the sweep point's label -> its
**analysed score**, e.g. peak energy).  ``None`` means "not (yet) analysed" and
is ignored by every function; a sample with no usable values is handled without
raising.  Callers must pass analysed scores only: the sweep *point* is an input,
not a measured outcome, so ranking on it would invent an optimum.  This layer
cannot tell the two apart, so the distinction is the caller's
:func:`~pic_agentic.agenda.refine.summary` input contract.

A flat top two scores are *not* sufficient for convergence: a monotone sweep
whose best point sits at the edge of the tested range still carries a trend
toward the untested region, so declaring it converged would ship a wrong
optimum (beta-7 F2).  Callers that know the *position* of each sample may pass
``coordinates`` (``label -> swept value``); then an edge optimum or a one-sided
monotone trend is reported as **not** converged and refinement extends the
range on the improving side.  Without coordinates only the score spacing is
available and the legacy behaviour is kept.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

#: A refinement step smaller than this is treated as zero (the local span is
#: flat), so no point is suggested.
_EPSILON = 1e-12

#: A sweep is "converged" from this many usable samples upward.
_MIN_CONVERGENCE_SAMPLES = 2


@dataclass(frozen=True)
class Refinement:
    """One refinement suggestion: a label and the value to simulate next."""

    label: str
    value: float


def best_point(points: Mapping[str, float | None]) -> tuple[str, float] | None:
    """Return the ``(label, value)`` of the highest usable sample.

    Args:
        points: ``label -> value`` (``None`` ignored).

    Returns:
        The best ``(label, value)``, or ``None`` when no sample is usable.

    """
    usable = [(label, float(value)) for label, value in points.items() if value is not None]
    if not usable:
        return None
    return max(usable, key=operator.itemgetter(1))


def refine_around(
    points: Mapping[str, float | None],
    *,
    fraction: float = 0.1,
    count: int = 2,
    coordinates: Mapping[str, float | None] | None = None,
) -> list[Refinement]:
    """Suggest ``count`` new points bracketing the best sample.

    The step is ``fraction`` of the span between the best and second-best point
    (or of the best value's own magnitude when there is only one usable point).
    Suggestions are placed on both sides of the best point and clamped to be
    positive (a physical parameter is assumed positive; a non-positive
    suggestion is dropped rather than emitted).

    When ``coordinates`` identify the sweep position of every usable sample and
    the optimum sits on the **edge** of the tested range with an outward
    (improving) trend, the suggestions are placed on that one side only, to
    extend the range beyond the tested edge.  The step there is ``fraction`` of
    the tested coordinate span.

    Args:
        points: ``label -> value`` (``None`` ignored).
        fraction: Step size as a fraction of the local span.
        count: Maximum number of suggestions to return.
        coordinates: ``label -> swept value`` for every usable ``label``.
            Optional; without it the legacy two-sided behaviour is kept.

    Returns:
        Up to ``count`` :class:`Refinement`s, nearest-to-best first; empty when
        there is nothing to refine around.

    """
    if count <= 0:
        return []
    best = best_point(points)
    if best is None:
        return []
    best_label, best_value = best
    ordered = _ordered_samples(points, coordinates)
    if coordinates is not None and all(coord is not None for _, _, coord in ordered):
        base = float(coordinates[best_label])
        direction = _improving_edge(points, coordinates)
        if direction != 0:
            step = _edge_step(ordered, fraction=fraction)
            offsets = _one_sided_offsets(step, count, sign=direction)
        else:
            step = _interior_step(ordered, best_label, fraction=fraction)
            offsets = _offsets(step, count)
        if math.isclose(step, 0.0, abs_tol=_EPSILON):
            return []
        candidates = [Refinement(label=f"{best_label}{offset:+g}", value=base + offset) for offset in offsets]
        return [candidate for candidate in candidates if candidate.value > 0.0]
    others = [float(value) for label, value in points.items() if value is not None and label != best_label]
    span = abs(best_value - max(others)) if others else abs(best_value)
    step = abs(span) * fraction
    if math.isclose(step, 0.0, abs_tol=_EPSILON):
        step = abs(best_value) * fraction
    if math.isclose(step, 0.0, abs_tol=_EPSILON):
        return []
    candidates = [
        Refinement(label=f"{best_label}{offset:+g}", value=best_value + offset) for offset in _offsets(step, count)
    ]
    return [candidate for candidate in candidates if candidate.value > 0.0]


def _offsets(step: float, count: int) -> list[float]:
    """Return alternating ``+step, -step, +2step, -2step, ...`` offsets.

    Returns:
        The first ``count`` offsets, positive side first.

    """
    offsets: list[float] = []
    magnitude = 1
    while len(offsets) < count:
        offsets.append(step * magnitude)
        if len(offsets) < count:
            offsets.append(-step * magnitude)
        magnitude += 1
    return offsets[:count]


def _one_sided_offsets(step: float, count: int, *, sign: int) -> list[float]:
    """Return ``count`` offsets stepping outward on one side only.

    Returns:
        ``sign * step * 1, sign * step * 2, ...``.

    """
    return [sign * step * (index + 1) for index in range(count)]


def _interior_step(
    ordered: list[tuple[str, float, float | None]],
    best_label: str,
    *,
    fraction: float,
) -> float:
    """Step around an interior optimum by ``fraction`` of the nearest gap.

    Uses the coordinate spacing (not the score span) because ``Refinement.value``
    is the parameter value to simulate next, and falls back to the best
    coordinate's own magnitude for a lone sample.

    Returns:
        The non-negative step (0.0 when it cannot be formed).

    """
    best_coord = next((coord for label, _, coord in ordered if label == best_label), None)
    if best_coord is None:
        return 0.0
    best_coord = float(best_coord)
    gaps = [abs(float(coord) - best_coord) for label, _, coord in ordered if label != best_label and coord is not None]
    span = min(gaps) if gaps else abs(best_coord)
    step = span * fraction
    if math.isclose(step, 0.0, abs_tol=_EPSILON):
        step = abs(best_coord) * fraction
    return step if not math.isclose(step, 0.0, abs_tol=_EPSILON) else 0.0


def _ordered_samples(
    points: Mapping[str, float | None],
    coordinates: Mapping[str, float | None] | None,
) -> list[tuple[str, float, float | None]]:
    """List usable samples as ``(label, score, coordinate)`` along the axis.

    The result is ordered by coordinate when ``coordinates`` covers every usable
    label (a complete positional frame); otherwise the mapping's insertion order
    is kept and the coordinate is ``None`` for every entry, so callers can only
    fall back to score spacing.

    Returns:
        The usable samples, coordinate-ordered when the frame is complete.

    """
    if coordinates is not None:
        frame = [
            (float(coord), label, float(value))
            for label, value in points.items()
            if value is not None and (coord := coordinates.get(label)) is not None
        ]
        usable = sum(1 for value in points.values() if value is not None)
        if frame and len(frame) == usable:
            frame.sort(key=operator.itemgetter(0))
            return [(label, value, coord) for coord, label, value in frame]
    return [(label, float(value), None) for label, value in points.items() if value is not None]


def _improving_edge(
    points: Mapping[str, float | None],
    coordinates: Mapping[str, float | None] | None,
) -> int:
    """Return the direction an edge optimum should extend, or 0 for interior.

    ``-1`` means the best sample is at the low edge and the scores improve
    toward it (extend the range downward); ``+1`` means the high edge.  ``0``
    means either no usable positional frame, or a genuine interior optimum.

    An edge counts only when the best is a **strict** maximum over its inward
    neighbour (there is an outward slope), so a perfectly flat plateau at the
    boundary stays converged.  A strictly monotone sequence is likewise
    one-sided and its optimum is at an edge by construction.

    Returns:
        ``-1``, ``+1``, or ``0``.

    """
    if coordinates is None:
        return 0
    ordered = _ordered_samples(points, coordinates)
    if len(ordered) < _MIN_CONVERGENCE_SAMPLES:
        return 0
    coords = [coord for _, _, coord in ordered]
    if any(coord is None for coord in coords) or len(set(coords)) != len(coords):
        return 0
    scores = [value for _, value, _ in ordered]
    best_index = max(range(len(scores)), key=scores.__getitem__)
    if best_index == 0 and scores[0] > scores[1]:
        return -1
    if best_index == len(scores) - 1 and scores[-1] > scores[-2]:
        return +1
    return 0


def _edge_step(ordered: list[tuple[str, float, float | None]], *, fraction: float) -> float:
    """Step outward from the tested range by ``fraction`` of its span.

    Returns:
        The non-negative step (0.0 when the span is flat).

    """
    coords = [float(coord) for _, _, coord in ordered if coord is not None]
    if len(coords) < _MIN_CONVERGENCE_SAMPLES:
        return 0.0
    span = coords[-1] - coords[0]
    step = abs(span) * fraction
    if math.isclose(step, 0.0, abs_tol=_EPSILON):
        step = abs(coords[-1]) * fraction
    return step if not math.isclose(step, 0.0, abs_tol=_EPSILON) else 0.0


def is_converged(
    points: Mapping[str, float | None],
    *,
    rel_tol: float = 0.05,
    coordinates: Mapping[str, float | None] | None = None,
) -> bool:
    """Whether the usable samples have converged to a flat interior optimum.

    Converged when there are at least two usable samples and the best two
    values differ by at most ``rel_tol`` (relative to the best).  A single
    sample is *not* converged (nothing to compare).

    When ``coordinates`` give every usable sample's sweep position, a flat top
    two is still **not** converged if the optimum sits on the edge of the
    tested range with an outward trend: the untested region on that side may be
    better, so the sweep must be extended (beta-7 F2).  A flat interior
    optimum, and a flat boundary plateau with no outward slope, stay converged.

    Args:
        points: ``label -> value`` (``None`` ignored).
        rel_tol: Relative tolerance between the top two values.
        coordinates: ``label -> swept value`` for every usable ``label``.
            Optional; without it only score spacing is assessed.

    Returns:
        True when the top two samples agree within ``rel_tol`` and the optimum
        is not an improving edge.

    """
    usable = sorted((float(value) for value in points.values() if value is not None), reverse=True)
    if len(usable) < _MIN_CONVERGENCE_SAMPLES:
        return False
    best, second = usable[0], usable[1]
    scale = abs(best) if not math.isclose(best, 0.0, abs_tol=_EPSILON) else 1.0
    if abs(best - second) / scale > rel_tol:
        return False
    return _improving_edge(points, coordinates) == 0


def summary(
    points: Mapping[str, float | None],
    *,
    rel_tol: float = 0.05,
    coordinates: Mapping[str, float | None] | None = None,
) -> dict[str, Any]:
    """Compose a serialisable refinement summary for a sweep.

    Args:
        points: ``label -> analysed score`` (``None`` ignored).  Pass only real
            analyses; a leaf that was merely simulated must be ``None``.
        rel_tol: Relative tolerance for :func:`is_converged`.
        coordinates: ``label -> swept value`` for every usable ``label``.  When
            given, an improving edge optimum keeps the sweep open and
            suggestions extend the range on that side, instead of reporting a
            spurious convergence (beta-7 F2).

    Returns:
        ``{"best": {"label", "value"} | None, "converged": bool,
        "suggestions": [{"label", "value"}], "analysed": int}``.  ``best`` is
        ``None`` and ``analysed`` is 0 when nothing has been analysed.

    """
    best = best_point(points)
    converged = is_converged(points, rel_tol=rel_tol, coordinates=coordinates)
    suggestions = [] if converged else refine_around(points, coordinates=coordinates)
    return {
        "best": {"label": best[0], "value": best[1]} if best is not None else None,
        "converged": converged,
        "suggestions": [{"label": item.label, "value": item.value} for item in suggestions],
        "analysed": sum(1 for value in points.values() if value is not None),
    }


__all__ = [
    "Refinement",
    "best_point",
    "is_converged",
    "refine_around",
    "summary",
]
