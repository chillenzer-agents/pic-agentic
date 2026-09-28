# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Evaluate a validated :class:`AnalysisProgram` with stdlib math.

No ``eval``, no ``exec``, no ``pickle``, no ``sympathy``/``sympify``: the walk
dispatches explicitly on the (already validated, closed) node vocabulary.  The
only inputs are the program and a mapping of selector name -> numeric sequence,
so evaluation has no ambient authority at all.

The evaluator is O(n) or O(n log n) in the input length, which the caller caps
at :data:`~pic_agentic.protocol.simulation.SLICE_MAX_POINTS` before it ever sees
the data.  Every value may be a scalar (``float``) or a sequence; reductions
collapse a sequence to a scalar.

Extraction-ready: stdlib only (``math``, ``statistics``, ``cmath``).
"""

from __future__ import annotations

import cmath
import math
import operator
import statistics
from typing import TYPE_CHECKING, Any

from pic_agentic.analysis_program import MAX_BINS, MAX_POINTS_OUT, AnalysisProgram, VarRef

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pic_agentic.analysis_program import BinOp, Reduce, UnOp

#: A resolved value is either a scalar or a sequence of values.
Value = float | list[float]

#: A value at or below this magnitude is treated as zero for the few places a
#: sign/zero test matters (avoids an exact float comparison).
_TINY = 1e-300


class ProgramError(RuntimeError):
    """Raised when a (validated) program cannot be evaluated on the data."""


def evaluate(program: AnalysisProgram, resolve: Any) -> dict[str, Any]:
    """Evaluate ``program``, resolving variable selectors through ``resolve``.

    Args:
        program: A validated :class:`AnalysisProgram`.
        resolve: ``resolve(selector) -> Sequence[float]`` returning the data for
            one :class:`~pic_agentic.analysis_program.VarRef`.

    Returns:
        ``{"result": <float|list[float]>, "n_points": int,
        "result_kind": "scalar"|"array"}`` plus ``"points"`` when the program
        sets a parallel expression.

    Raises:
        ProgramError: If a selector cannot be resolved, an operand shape is
            wrong, or the result is too large.

    """
    data = _typecheck(program, program.output, resolve)
    result = _eval(program.output, data)
    payload: dict[str, Any] = {
        "result": result,
        "n_points": _n_points(result),
        "result_kind": "array" if isinstance(result, list) else "scalar",
    }
    if program.points is not None:
        data = _typecheck(program, program.points, resolve)
        payload["points"] = _eval(program.points, data)
    if isinstance(result, list) and len(result) > MAX_POINTS_OUT:
        msg = f"result has {len(result)} points, above the {MAX_POINTS_OUT} cap"
        raise ProgramError(msg)
    return payload


def run_program(program: AnalysisProgram, data: Mapping[str, Value]) -> dict[str, Any]:
    """Evaluate ``program`` against a pre-resolved ``{selector: value}`` map.

    A convenience for tests and callers that already hold the data (the cluster
    path resolves selectors from openPMD first).

    Args:
        program: A validated :class:`AnalysisProgram`.
        data: ``selector name -> scalar or sequence``.

    Returns:
        The :func:`evaluate` payload.

    """

    def resolve(selector: VarRef) -> Sequence[float]:
        if selector.name not in data:
            msg = f"selector {selector.name!r} has no data"
            raise ProgramError(msg)
        value = data[selector.name]
        return value if isinstance(value, list) else [value]

    return evaluate(program, resolve)


def _typecheck(program: AnalysisProgram, node: Any, resolve: Any) -> dict[str, Value]:
    """Resolve and cache every selector the program actually references.

    Resolving only referenced selectors keeps the openPMD reads minimal and
    makes an unused declared selector harmless.

    Returns:
        A ``selector name -> value`` cache.

    Raises:
        ProgramError: If a selector cannot be resolved.

    """
    data: dict[str, Value] = {}
    for name in _selector_names(node):
        selector = next((item for item in program.selectors if item.name == name), None)
        if selector is None:
            # A bare ``var`` not declared in ``selectors`` is still resolvable by
            # name alone (a declared list is documentation, not a gate).
            selector = VarRef(name=name)
        try:
            values = resolve(selector)
        except ProgramError:
            raise
        except Exception as exc:
            msg = f"cannot resolve selector {name!r}: {exc}"
            raise ProgramError(msg) from exc
        data[name] = [float(value) for value in values]
    return data


def _selector_names(node: Any) -> list[str]:
    """Return every selector name referenced by ``node``, in first-seen order.

    Returns:
        The referenced selector names.

    """
    names: list[str] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if getattr(current, "kind", None) == "var" and current.name not in names:
            names.append(current.name)
        for field in ("left", "right", "operand"):
            child = getattr(current, field, None)
            if child is not None:
                stack.append(child)
    return names


def _eval(node: Any, data: Mapping[str, Value]) -> Value:
    """Evaluate one node against the resolved selector data.

    Returns:
        The scalar or sequence value.

    Raises:
        ProgramError: On an unknown node or an invalid operation.

    """
    kind = getattr(node, "kind", None)
    if kind == "const":
        return float(node.value)
    if kind == "var":
        if node.name not in data:
            msg = f"selector {node.name!r} has no data"
            raise ProgramError(msg)
        return data[node.name]
    if kind == "binop":
        return _eval_binop(node, data)
    if kind == "unop":
        return _eval_unop(node, data)
    if kind == "reduce":
        return _eval_reduce(node, data)
    msg = f"unknown node kind: {kind!r}"
    raise ProgramError(msg)


def _eval_binop(node: BinOp, data: Mapping[str, Value]) -> Value:
    """Evaluate a binary operation, broadcasting scalar/array operands.

    A ``pow`` is only allowed with a constant scalar exponent in [0,
    ``MAX_EXPONENT``] -- raising an array to an array power, or to a huge
    power, is refused rather than attempted.

    Returns:
        The result value.

    Raises:
        ProgramError: If the operand shapes are incompatible or a division by
            zero / non-positive power occurs.

    """
    left = _eval(node.left, data)
    right = _eval(node.right, data)
    op = node.op
    if op == "pow":
        if not isinstance(right, float):
            msg = "pow requires a constant scalar exponent"
            raise ProgramError(msg)
        if isinstance(left, float) and math.isclose(left, 0.0, abs_tol=_TINY) and right < 0:
            msg = "0 raised to a negative power is undefined"
            raise ProgramError(msg)
    try:
        return _combine(op, left, right)
    except (ZeroDivisionError, ValueError, OverflowError, TypeError) as exc:
        msg = f"{op} failed: {exc}"
        raise ProgramError(msg) from exc


#: The closed binary-operator set (mirrors ``BinaryOp``).
_BINARY: dict[str, Any] = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "pow": operator.pow,
}


def _combine(op: str, left: Value, right: Value) -> Value:
    """Apply ``op`` to two values, broadcasting as needed.

    Returns:
        The combined value.

    Raises:
        ValueError: If two arrays have different lengths.

    """
    operation = _BINARY[op]
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            msg = f"array length mismatch: {len(left)} vs {len(right)}"
            raise ValueError(msg)
        return [float(operation(a, b)) for a, b in zip(left, right, strict=True)]
    if isinstance(left, list):
        return [float(operation(value, right)) for value in left]
    if isinstance(right, list):
        return [float(operation(left, value)) for value in right]
    return float(operation(left, right))


def _eval_unop(node: UnOp, data: Mapping[str, Value]) -> Value:
    """Evaluate a unary function elementwise.

    Returns:
        The result value.

    Raises:
        ProgramError: If the function is undefined on a value.

    """
    operand = _eval(node.operand, data)
    function = _UNARY[node.op]
    try:
        if isinstance(operand, list):
            return [float(function(value)) for value in operand]
        return float(function(operand))
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        msg = f"{node.op} failed: {exc}"
        raise ProgramError(msg) from exc


#: The closed unary-function set (mirrors ``UnaryOp``).  ``sin``/``cos`` use
#: ``math``; ``tanh`` is ``math.tanh``.
_UNARY = {
    "neg": operator.neg,
    "abs": abs,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tanh": math.tanh,
    "sign": lambda value: (value > 0) - (value < 0),
}


def _eval_reduce(node: Reduce, data: Mapping[str, Value]) -> Value:
    """Evaluate a reduction.

    Returns:
        A scalar, or a list for ``histogram``/``fft_peak``/``fft_freq``.

    Raises:
        ProgramError: If the operand is empty or a reduction is undefined.

    """
    operand = _eval(node.operand, data)
    values = operand if isinstance(operand, list) else [operand]
    if not values:
        msg = f"{node.op} over an empty sequence is undefined"
        raise ProgramError(msg)
    op = node.op
    # Scalar reductions share one lookup; the shape-changing ones are explicit.
    scalar_reductions: dict[str, Any] = {
        "sum": math.fsum,
        "mean": statistics.fmean,
        "min": min,
        "max": max,
        "median": statistics.median,
        "argmax": lambda vs: max(range(len(vs)), key=vs.__getitem__),
        "argmin": lambda vs: min(range(len(vs)), key=vs.__getitem__),
    }
    if op in scalar_reductions:
        return float(scalar_reductions[op](values))
    if op == "std":
        return float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    if op == "quantile":
        return _quantile(values, node.q if node.q is not None else 0.5)
    if op == "histogram":
        counts, _edges = _histogram(values, node.bins or 32)
        return counts
    return _fft(values, frequencies=op == "fft_freq")


def _quantile(values: list[float], q: float) -> float:
    """Return the ``q`` quantile by linear interpolation (numpy-style).

    Returns:
        The interpolated quantile value.

    """
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _histogram(values: list[float], bins: int) -> tuple[list[float], list[float]]:
    """Compute a uniform histogram.

    Returns:
        ``(counts, edges)`` with ``len(counts) == bins``.

    """
    low, high = min(values), max(values)
    if high == low:
        high = low + 1.0
    width = (high - low) / bins
    counts = [0.0] * bins
    for value in values:
        index = int((value - low) / width)
        index = min(max(index, 0), bins - 1)
        counts[index] += 1.0
    edges = [low + index * width for index in range(bins + 1)]
    return counts, edges


def _fft(values: list[float], *, frequencies: bool) -> list[float]:
    """Return the FFT magnitudes (or bin frequencies) of a real sequence.

    A bounded, dependency-free DFT is used: the input is capped at
    ``MAX_POINTS_OUT`` by the caller, so the O(n^2) cost is bounded.  Bin 0 (DC)
    is included; only the first ``n // 2 + 1`` non-redundant bins are returned.

    Returns:
        The magnitudes (``fft_peak``) or the bin frequencies (``fft_freq``).

    """
    n = len(values)
    half = n // 2 + 1
    if frequencies:
        return [float(index) for index in range(half)]
    magnitudes: list[float] = []
    for index in range(half):
        acc = 0j
        for position, value in enumerate(values):
            angle = -2j * math.pi * index * position / n
            acc += value * cmath.exp(angle)
        magnitudes.append(abs(acc))
    return magnitudes


def _n_points(result: Value) -> int:
    """Return the number of points a result carries.

    Returns:
        The sequence length, or 1 for a scalar.

    """
    return len(result) if isinstance(result, list) else 1


__all__ = ["MAX_BINS", "ProgramError", "Value", "evaluate", "run_program"]
