# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Numeric tests for the analysis-program evaluator (stdlib, no code execution)."""

from __future__ import annotations

import math

import pytest

from pic_agentic.analysis_eval import ProgramError, run_program
from pic_agentic.analysis_program import AnalysisProgram


def _run(output: dict, data: dict, **kw: object) -> dict:
    program = AnalysisProgram.model_validate({"output": output, **kw})
    return run_program(program, data)


def _var(name: str) -> dict:
    return {"kind": "var", "name": name}


def _reduce(op: str, operand: dict, **kw: object) -> dict:
    return {"kind": "reduce", "op": op, "operand": operand, **kw}


def test_scalar_arithmetic_and_unary() -> None:
    expr = {
        "kind": "binop",
        "op": "add",
        "left": {"kind": "const", "value": 2.0},
        "right": {"kind": "const", "value": 3.0},
    }
    out = _run(expr, {})
    assert out["result"] == pytest.approx(5.0)
    assert out["result_kind"] == "scalar"

    sqrt = _run({"kind": "unop", "op": "sqrt", "operand": {"kind": "const", "value": 9.0}}, {})
    assert sqrt["result"] == pytest.approx(3.0)


def test_array_arithmetic_broadcasts_a_scalar() -> None:
    out = _run(
        {"kind": "binop", "op": "mul", "left": _var("x"), "right": {"kind": "const", "value": 2.0}},
        {"x": [1.0, 2.0, 3.0]},
    )
    assert out["result"] == [2.0, 4.0, 6.0]


def test_array_length_mismatch_is_an_error() -> None:
    with pytest.raises(ProgramError):
        _run(
            {"kind": "binop", "op": "add", "left": _var("x"), "right": _var("y")},
            {"x": [1.0], "y": [1.0, 2.0]},
        )


@pytest.mark.parametrize(
    ("op", "values", "expected"),
    [
        ("sum", [1.0, 2.0, 3.0], 6.0),
        ("mean", [1.0, 2.0, 3.0], 2.0),
        ("std", [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], 2.0),
        ("min", [3.0, 1.0, 2.0], 1.0),
        ("max", [3.0, 1.0, 2.0], 3.0),
        ("median", [4.0, 1.0, 3.0, 2.0], 2.5),
        ("argmax", [3.0, 7.0, 2.0], 1.0),
        ("argmin", [3.0, 7.0, 2.0], 2.0),
    ],
)
def test_reductions(op: str, values: list[float], expected: float) -> None:
    out = _run(_reduce(op, _var("x")), {"x": values})
    assert out["result"] == pytest.approx(expected)


def test_quantile_by_interpolation() -> None:
    out = _run(_reduce("quantile", _var("x"), q=0.5), {"x": [4.0, 1.0, 3.0, 2.0]})
    assert out["result"] == pytest.approx(2.5)
    q90 = _run(_reduce("quantile", _var("x"), q=0.9), {"x": [0.0, 10.0]})
    assert q90["result"] == pytest.approx(9.0)


def test_histogram_counts_and_bins() -> None:
    out = _run(_reduce("histogram", _var("x"), bins=4), {"x": [0.0, 0.1, 0.9, 1.0]})
    assert out["result_kind"] == "array"
    assert out["n_points"] == 4
    assert sum(out["result"]) == pytest.approx(4.0)


def test_fft_peak_finds_the_dominant_frequency() -> None:
    # A pure sinusoid at bin 2 (n=8) must peak at index 2.
    values = [math.sin(2 * math.pi * 2 * i / 8) for i in range(8)]
    peaks = _run(_reduce("fft_peak", _var("x")), {"x": values})
    assert peaks["result"].index(max(peaks["result"])) == 2
    freqs = _run(_reduce("fft_freq", _var("x")), {"x": values})
    assert freqs["result"] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_energy_spectrum_example() -> None:
    """histogram(sqrt(px^2+py^2+pz^2)) -- the acceptance-test shape."""
    expr = _reduce(
        "histogram",
        {
            "kind": "unop",
            "op": "sqrt",
            "operand": {
                "kind": "binop",
                "op": "add",
                "left": {"kind": "binop", "op": "mul", "left": _var("px"), "right": _var("px")},
                "right": {"kind": "binop", "op": "mul", "left": _var("py"), "right": _var("py")},
            },
        },
        bins=8,
    )
    out = _run(expr, {"px": [3.0, 0.0, 1.0, 0.0], "py": [4.0, 0.0, 0.0, 0.0]})
    assert sum(out["result"]) == pytest.approx(4.0)  # every particle counted once


def test_parallel_points_expression() -> None:
    out = _run(
        {"kind": "reduce", "op": "mean", "operand": _var("x")},
        {"x": [1.0, 2.0, 3.0]},
        points={"kind": "reduce", "op": "sum", "operand": _var("x")},
    )
    assert out["result"] == pytest.approx(2.0)
    assert out["points"] == pytest.approx(6.0)


def test_missing_selector_is_an_error() -> None:
    with pytest.raises(ProgramError):
        _run(_reduce("mean", _var("nope")), {"x": [1.0]})


def test_empty_reduction_is_an_error() -> None:
    with pytest.raises(ProgramError):
        _run(_reduce("mean", _var("x")), {"x": []})


def test_division_by_zero_is_a_program_error() -> None:
    with pytest.raises(ProgramError):
        _run(
            {
                "kind": "binop",
                "op": "div",
                "left": {"kind": "const", "value": 1.0},
                "right": {"kind": "const", "value": 0.0},
            },
            {},
        )


def test_pow_requires_a_constant_exponent() -> None:
    with pytest.raises(ProgramError):
        _run({"kind": "binop", "op": "pow", "left": _var("x"), "right": _var("y")}, {"x": [1.0], "y": [2.0]})


def test_resolve_error_is_wrapped() -> None:
    program = AnalysisProgram.model_validate({"output": _reduce("mean", _var("x"))})

    def boom(_selector: object) -> list[float]:
        msg = "no such record"
        raise RuntimeError(msg)

    from pic_agentic.analysis_eval import evaluate

    with pytest.raises(ProgramError):
        evaluate(program, boom)
