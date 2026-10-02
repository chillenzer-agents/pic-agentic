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
) -> list[Refinement]:
    """Suggest ``count`` new points bracketing the best sample.

    The step is ``fraction`` of the span between the best and second-best point
    (or of the best value's own magnitude when there is only one usable point).
    Suggestions are placed on both sides of the best point and clamped to be
    positive (a physical parameter is assumed positive; a non-positive
    suggestion is dropped rather than emitted).

    Args:
        points: ``label -> value`` (``None`` ignored).
        fraction: Step size as a fraction of the local span.
        count: Maximum number of suggestions to return.

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


def is_converged(points: Mapping[str, float | None], *, rel_tol: float = 0.05) -> bool:
    """Whether the usable samples have converged to a flat optimum.

    Converged when there are at least two usable samples and the best two
    values differ by at most ``rel_tol`` (relative to the best).  A single
    sample is *not* converged (nothing to compare).

    Args:
        points: ``label -> value`` (``None`` ignored).
        rel_tol: Relative tolerance between the top two values.

    Returns:
        True when the top two samples agree within ``rel_tol``.

    """
    usable = sorted((float(value) for value in points.values() if value is not None), reverse=True)
    if len(usable) < _MIN_CONVERGENCE_SAMPLES:
        return False
    best, second = usable[0], usable[1]
    scale = abs(best) if not math.isclose(best, 0.0, abs_tol=_EPSILON) else 1.0
    return abs(best - second) / scale <= rel_tol


def summary(points: Mapping[str, float | None], *, rel_tol: float = 0.05) -> dict[str, Any]:
    """Compose a serialisable refinement summary for a sweep.

    Args:
        points: ``label -> analysed score`` (``None`` ignored).  Pass only real
            analyses; a leaf that was merely simulated must be ``None``.
        rel_tol: Relative tolerance for :func:`is_converged`.

    Returns:
        ``{"best": {"label", "value"} | None, "converged": bool,
        "suggestions": [{"label", "value"}], "analysed": int}``.  ``best`` is
        ``None`` and ``analysed`` is 0 when nothing has been analysed.

    """
    best = best_point(points)
    suggestions = refine_around(points) if not is_converged(points, rel_tol=rel_tol) else []
    return {
        "best": {"label": best[0], "value": best[1]} if best is not None else None,
        "converged": is_converged(points, rel_tol=rel_tol),
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
