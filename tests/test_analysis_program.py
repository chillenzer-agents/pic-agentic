# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Validation tests for the declarative analysis program (no code execution)."""

from __future__ import annotations

import pytest

from pic_agentic.analysis_program import (
    MAX_DEPTH,
    MAX_EXPONENT,
    MAX_NODES,
    MAX_SELECTORS,
    MAX_SOURCE_BYTES,
    AnalysisProgram,
)


def _var(name: str) -> dict:
    return {"kind": "var", "name": name}


def _reduce(op: str, operand: dict, **kw: object) -> dict:
    return {"kind": "reduce", "op": op, "operand": operand, **kw}


def _program(output: dict, **kw: object) -> dict:
    return {"output": output, **kw}


def test_parses_a_full_energy_spectrum_program() -> None:
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
        bins=64,
    )
    program = AnalysisProgram.model_validate(
        _program(expr, selectors=[{"kind": "var", "name": "px"}, {"kind": "var", "name": "py"}]),
    )
    assert program.output.kind == "reduce"
    assert program.output.op == "histogram"


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "unop", "op": "exec", "operand": {"kind": "const", "value": 1.0}},
        {"kind": "unop", "op": "__import__", "operand": {"kind": "const", "value": 1.0}},
        {"kind": "reduce", "op": "system", "operand": _var("x")},
        {"kind": "call", "fn": "os.system", "args": []},
        {"kind": "var", "name": "os; rm -rf /"},
        {"kind": "var", "name": "../../etc/passwd"},
    ],
)
def test_unknown_or_unsafe_nodes_are_rejected(bad: dict) -> None:
    with pytest.raises(ValueError, match=r"validation error|unsafe|Extra inputs|Input should"):
        AnalysisProgram.model_validate(_program(bad))


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        AnalysisProgram.model_validate(_program({"kind": "const", "value": 1.0, "extra": 1}))


def test_non_finite_constant_is_rejected() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            AnalysisProgram.model_validate(_program({"kind": "const", "value": value}))


def test_pow_exponent_is_bounded() -> None:
    big = _program(
        {"kind": "binop", "op": "pow", "left": _var("x"), "right": {"kind": "const", "value": MAX_EXPONENT + 1}},
    )
    with pytest.raises(ValueError, match="exponent"):
        AnalysisProgram.model_validate(big)
    # The boundary is allowed.
    AnalysisProgram.model_validate(
        _program({"kind": "binop", "op": "pow", "left": _var("x"), "right": {"kind": "const", "value": MAX_EXPONENT}}),
    )


def test_too_many_selectors_are_rejected() -> None:
    selectors = [{"kind": "var", "name": f"v{i}"} for i in range(MAX_SELECTORS + 1)]
    with pytest.raises(ValueError, match="selectors"):
        AnalysisProgram.model_validate(_program(_var("v0"), selectors=selectors))


def test_quantile_and_bins_ranges_are_enforced() -> None:
    with pytest.raises(ValueError, match="quantile"):
        AnalysisProgram.model_validate(_program(_reduce("quantile", _var("x"), q=1.5)))
    with pytest.raises(ValueError, match="bins"):
        AnalysisProgram.model_validate(_program(_reduce("histogram", _var("x"), bins=0)))


def test_depth_and_node_caps_are_enforced() -> None:
    # A deep left-leaning add chain exceeds MAX_DEPTH.
    expr: dict = _var("x")
    for _ in range(MAX_DEPTH + 2):
        expr = {"kind": "binop", "op": "add", "left": expr, "right": {"kind": "const", "value": 1.0}}
    with pytest.raises(ValueError, match="depth"):
        AnalysisProgram.model_validate(_program(expr))

    # A wide/deep tree is rejected -- by the node cap or, if pydantic's own
    # recursion guard trips first, by that; either way it never validates.
    wide: dict = _var("x")
    for _ in range(MAX_NODES):
        wide = {"kind": "binop", "op": "add", "left": wide, "right": {"kind": "const", "value": 1.0}}
    with pytest.raises(ValueError, match=r"nodes|depth|recursion"):
        AnalysisProgram.model_validate(_program(wide))


def test_source_size_is_capped() -> None:
    # A pathological program whose JSON exceeds MAX_SOURCE_BYTES; built directly
    # (the node cap would reject it first through the model, but the protocol
    # validator checks size before parsing).
    from pic_agentic.protocol.simulation import ResultParams

    huge = {"kind": "var", "name": "x" * (MAX_SOURCE_BYTES + 10)}
    with pytest.raises(ValueError, match="exceeds"):
        ResultParams(sim_id="s", op="compute", program=_program(huge))


def test_program_round_trips_json() -> None:
    program = AnalysisProgram.model_validate(_program(_reduce("mean", _var("x"))))
    again = AnalysisProgram.model_validate_json(program.model_dump_json())
    assert again.model_dump() == program.model_dump()
