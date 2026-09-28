# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Integration tests for the secure ``COMPUTE`` result op and the tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from pic_agentic import results
from pic_agentic.protocol.simulation import ResultOp, ResultParams

SIM_ID = "abcd1234"


def _var(name: str) -> dict:
    return {"kind": "var", "name": name}


def _spectrum_program() -> dict:
    return {
        "output": {
            "kind": "reduce",
            "op": "histogram",
            "bins": 4,
            "operand": {
                "kind": "binop",
                "op": "add",
                "left": {"kind": "binop", "op": "mul", "left": _var("px"), "right": _var("px")},
                "right": {"kind": "binop", "op": "mul", "left": _var("py"), "right": _var("py")},
            },
        },
        "selectors": [
            {"kind": "var", "name": "px", "record": "E", "component": "x"},
            {"kind": "var", "name": "py", "record": "E", "component": "y"},
        ],
    }


@pytest.fixture
def fake_compute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a simOutput dir plus a monkeypatched reader returning canned arrays."""
    run = tmp_path / "run"
    out = run / "simOutput"
    out.mkdir(parents=True)
    (out / "fields.bp").write_bytes(b"x")
    monkeypatch.setattr(results, "_reader_name", lambda: "openpmd")

    def fake_load(_path: Path, record: str | None, component: str | None, _iteration: object) -> list[float]:
        table = {
            ("E", "x"): [1.0, 2.0, 3.0],
            ("E", "y"): [4.0, 5.0, 6.0],
            ("rho", None): [0.0, 1.0],
        }
        key = (record or "E", component)
        if key not in table:
            msg = f"record {record!r}/{component!r} not found"
            raise results.ResultsReaderError(msg)
        return table[key]

    monkeypatch.setattr(results, "_load_dataset", fake_load)
    return run


def test_compute_scalar_vector_and_histogram(fake_compute: Path) -> None:
    run = fake_compute
    # A scalar: sum of E.x.
    scalar = results.resolve_result(
        ResultParams(
            sim_id=SIM_ID,
            op=ResultOp.COMPUTE,
            record="E",
            component="x",
            program={"output": {"kind": "reduce", "op": "sum", "operand": _var("x")}},
        ),
        run_dir=run,
        sim_id=SIM_ID,
    )
    assert scalar["stats"]["value"] == pytest.approx(6.0)
    assert scalar["result"]["result_kind"] == "scalar"

    # The spectrum-shaped program over the two components.
    spec = results.resolve_result(
        ResultParams(sim_id=SIM_ID, op=ResultOp.COMPUTE, program=_spectrum_program()),
        run_dir=run,
        sim_id=SIM_ID,
    )
    assert spec["data_encoding"] == "float"
    assert spec["n_points"] == 4
    assert spec["result"]["result_kind"] == "array"


def test_compute_requires_a_program(fake_compute: Path) -> None:
    result = results.resolve_result(
        ResultParams(sim_id=SIM_ID, op=ResultOp.COMPUTE),
        run_dir=fake_compute,
        sim_id=SIM_ID,
    )
    assert result["error_code"] == "unsupported"


def test_compute_without_reader_is_unavailable(fake_compute: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(results, "_reader_name", lambda: None)
    result = results.resolve_result(
        ResultParams(sim_id=SIM_ID, op=ResultOp.COMPUTE, program={"output": {"kind": "const", "value": 1.0}}),
        run_dir=fake_compute,
        sim_id=SIM_ID,
    )
    assert result["error_code"] == "reader_unavailable"


def test_compute_unknown_record_is_a_clean_error(fake_compute: Path) -> None:
    result = results.resolve_result(
        ResultParams(
            sim_id=SIM_ID,
            op=ResultOp.COMPUTE,
            program={"output": {"kind": "reduce", "op": "sum", "operand": _var("x")}},
        ),
        run_dir=fake_compute,
        sim_id=SIM_ID,
    )
    # The raw selector names "x" but the fake table keys require a record;
    # whatever the failure, it is shaped as data, never raised.
    assert "error_code" in result


def test_compute_oversized_output_is_capped(fake_compute: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(results, "MAX_RESULT_BYTES", 50)
    result = results.resolve_result(
        ResultParams(
            sim_id=SIM_ID,
            op=ResultOp.COMPUTE,
            record="E",
            component="x",
            program={"output": {"kind": "var", "name": "x"}},
        ),
        run_dir=fake_compute,
        sim_id=SIM_ID,
    )
    assert result["error_code"] == "result_too_large"


def test_invalid_program_is_rejected_by_the_model() -> None:
    unsafe = {"output": {"kind": "unop", "op": "exec", "operand": {"kind": "const", "value": 1.0}}}
    with pytest.raises(ValueError, match=r"validation error|unsafe|Input should"):
        ResultParams(sim_id=SIM_ID, op=ResultOp.COMPUTE, program=unsafe)


def test_compute_program_round_trips_on_the_wire() -> None:
    """The program must survive the builder, not just the pydantic model.

    Regression for a live bug: ``build_result_command`` omitted ``program``, so
    every ``run_analysis``/``COMPUTE`` request reached the simclient with no
    program and failed with "compute requires a program".
    """
    from pic_agentic.protocol.simulation import build_result_command

    params = ResultParams(sim_id=SIM_ID, op=ResultOp.COMPUTE, program=_spectrum_program())
    command = build_result_command(sim=SIM_ID, seq=1, params=params, cmd_id="c")
    assert command.payload["program"] == _spectrum_program()
    # A non-compute op must not carry a program key at all.
    read = build_result_command(sim=SIM_ID, seq=2, params=ResultParams(sim_id=SIM_ID, op=ResultOp.READ), cmd_id="d")
    assert "program" not in read.payload


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("openpmd_api") is None,
    reason="openpmd_api not installed",
)
def test_compute_against_a_real_openpmd_series(tmp_path: Path) -> None:
    """Evaluate a program against a real openPMD series (reader-API guard)."""
    import numpy as np
    import openpmd_api as api

    from pic_agentic.analysis_eval import evaluate
    from pic_agentic.analysis_program import AnalysisProgram

    run = tmp_path / "run"
    out = run / "simOutput"
    out.mkdir(parents=True)
    series = api.Series(str(out / "fields.h5"), api.Access.create)
    mesh = series.iterations[0].meshes["E"]
    mesh["x"].reset_dataset(api.Dataset(api.Datatype.DOUBLE, [4]))
    mesh["x"].store_chunk(np.array([3.0, 0.0, 1.0, 0.0]))
    mesh["y"].reset_dataset(api.Dataset(api.Datatype.DOUBLE, [4]))
    mesh["y"].store_chunk(np.array([4.0, 0.0, 0.0, 0.0]))
    series.close()

    program = AnalysisProgram.model_validate(
        {
            "selectors": [
                {"kind": "var", "name": "px", "record": "E", "component": "x"},
                {"kind": "var", "name": "py", "record": "E", "component": "y"},
            ],
            "output": {
                "kind": "reduce",
                "op": "sum",
                "operand": {
                    "kind": "unop",
                    "op": "sqrt",
                    "operand": {
                        "kind": "binop",
                        "op": "add",
                        "left": {
                            "kind": "binop",
                            "op": "mul",
                            "left": {"kind": "var", "name": "px"},
                            "right": {"kind": "var", "name": "px"},
                        },
                        "right": {
                            "kind": "binop",
                            "op": "mul",
                            "left": {"kind": "var", "name": "py"},
                            "right": {"kind": "var", "name": "py"},
                        },
                    },
                },
            },
        },
    )

    def resolve(selector) -> list[float]:
        return results._load_dataset(out / "fields.h5", selector.record, selector.component, None)

    payload = evaluate(program, resolve)
    # |(3,4)| + |(0,0)| + |(1,0)| + |(0,0)| = 5 + 0 + 1 + 0 = 6.
    assert payload["result"] == pytest.approx(6.0)
