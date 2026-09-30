# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for the G1 PIConGPU text-plugin result readers.

The offline tests use a stub reader injected in place of the optional PIConGPU
package, so the default suite (no picongpu) still passes.  The final test runs
the *real* ``EnergyHistogramData`` reader and is skipped unless PIConGPU is
importable; run it with a PIConGPU venv, e.g.::

    PYTHONPATH=<repo>/src /tmp/opencode/pic-stack-venv/bin/python -m pytest tests/test_results_plugin.py -q

"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from plugin_fixtures import energy_histogram_dat, write_output_unit

from pic_agentic import results
from pic_agentic.protocol.simulation import ResultOp, ResultParams

SIM_ID = "abcd1234"


class _StubReader:
    """A minimal stand-in for ``EnergyHistogramData``."""

    def __init__(self, run_directory: str) -> None:
        self.run_directory = run_directory

    @staticmethod
    def get_iterations(species: str, species_filter: str = "all") -> list[int]:
        _ = (species, species_filter)
        return [0, 50, 100]

    @staticmethod
    def get(iteration: int, species: str, species_filter: str = "all", **kwargs: object) -> tuple:
        _ = (species, species_filter, kwargs)
        bins = [1000.0 * (i + 1) / 10 for i in range(10)]
        counts = [0.0] * 10
        counts[0] = 1.0
        counts[4] = 42.0
        counts[-1] = 1.0
        return counts, bins, [iteration], 1e-16


def _tree(tmp_path: Path) -> Path:
    """Build a run dir with one energy-histogram output and the unit file."""
    run = tmp_path / "run"
    write_output_unit(run)
    energy_histogram_dat(run)
    return run


def _params(**kwargs: object) -> ResultParams:
    return ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="energy_histogram", **kwargs)


def test_registry_matches_the_wire_names() -> None:
    from pic_agentic.protocol.simulation import PLUGIN_READER_NAMES

    assert set(results._PLUGIN_READERS) == set(PLUGIN_READER_NAMES)
    assert set(results._PLUGIN_BUILDERS) == set(PLUGIN_READER_NAMES)


def test_scandir_names_the_plugin_reader(tmp_path: Path) -> None:
    run = _tree(tmp_path)
    manifest = results.scan_output(run / "simOutput", sim_id=SIM_ID, run_dir=str(run))
    formats = {ref.path: ref.format for ref in manifest.files}
    assert formats["e_energyHistogram_all.dat"] == "energy_histogram"
    assert formats["output"] == "binary"


def test_plugin_missing_output_is_no_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(results, "_import_plugin_reader", lambda _name: _StubReader)
    run = tmp_path / "run"
    (run / "simOutput").mkdir(parents=True)
    payload = results.resolve_result(_params(species="e"), run_dir=run, sim_id=SIM_ID)
    assert payload["error_code"] == "no_results"


def test_plugin_unknown_reader_is_unsupported(tmp_path: Path) -> None:
    run = _tree(tmp_path)
    # Bypass the wire validator to exercise the engine's own guard.
    params = ResultParams.model_construct(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="bogus", species_filter="all")
    payload = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)
    assert payload["error_code"] == "unsupported"


def test_plugin_reader_unavailable_without_picongpu(tmp_path: Path) -> None:
    if importlib.util.find_spec("picongpu") is not None:
        pytest.skip("picongpu is installed in this environment")
    run = _tree(tmp_path)
    payload = results.resolve_result(_params(species="e"), run_dir=run, sim_id=SIM_ID)
    assert payload["error_code"] == "reader_unavailable"


def test_plugin_stub_summary_and_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(results, "_import_plugin_reader", lambda _name: _StubReader)
    run = _tree(tmp_path)
    payload = results.resolve_result(_params(species="e", iteration="last"), run_dir=run, sim_id=SIM_ID)
    summary = payload["result"]
    assert summary["iteration"] == 100
    assert summary["total"] == pytest.approx(44.0)
    assert summary["count_in_window"]["count"] == pytest.approx(44.0)
    assert len(summary["bins_kev"]) == len(summary["counts"]) == 10
    assert summary["downsampled"] is False


def test_plugin_emittance_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubEmittance:
        def __init__(self, run_directory: str) -> None:
            _ = run_directory

        @staticmethod
        def get_iterations(species: str, species_filter: str = "all") -> list[int]:
            _ = (species, species_filter)
            return [0]

        @staticmethod
        def get(iteration: int, species: str, species_filter: str = "all", **kwargs: object) -> tuple:
            _ = (species, species_filter, kwargs)
            # ``EmittanceData`` returns ``[emit_all, *slice_emit]``, one element
            # longer than ``y_slices``.
            return [6.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0], [iteration], 1e-16

    monkeypatch.setattr(results, "_import_plugin_reader", lambda _name: _StubEmittance)
    run = tmp_path / "run"
    write_output_unit(run)
    (run / "simOutput" / "e_emittance_all.dat").write_text("x\n", encoding="utf-8")
    params = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="emittance", species="e")
    payload = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)
    summary = payload["result"]
    assert summary["slice_emit_mrad"] == [1.0, 2.0, 3.0]
    assert summary["total_emit_mrad"] == pytest.approx(6.0)
    assert summary["max_emit_mrad"] == pytest.approx(3.0)
    assert summary["max_y_slice_m"] == pytest.approx(2.0)


def test_plugin_emittance_stub_last_slice_peak_does_not_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A peak in the last slice must not index past ``y_slices`` (B1)."""

    class _StubEmittance:
        def __init__(self, run_directory: str) -> None:
            _ = run_directory

        @staticmethod
        def get_iterations(species: str, species_filter: str = "all") -> list[int]:
            _ = (species, species_filter)
            return [0]

        @staticmethod
        def get(iteration: int, species: str, species_filter: str = "all", **kwargs: object) -> tuple:
            _ = (species, species_filter, kwargs)
            return [9.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0], [iteration], 1e-16

    monkeypatch.setattr(results, "_import_plugin_reader", lambda _name: _StubEmittance)
    run = tmp_path / "run"
    write_output_unit(run)
    (run / "simOutput" / "e_emittance_all.dat").write_text("x\n", encoding="utf-8")
    params = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="emittance", species="e")
    payload = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)
    assert "result" in payload, payload
    summary = payload["result"]
    assert summary["total_emit_mrad"] == pytest.approx(9.0)
    assert summary["max_emit_mrad"] == pytest.approx(4.0)
    assert summary["max_y_slice_m"] == pytest.approx(3.0)


def test_bound_plugin_strides_to_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(results, "MAX_RESULT_BYTES", 512)
    summary = {
        "bins_kev": [float(i) for i in range(1000)],
        "counts": [float(i) for i in range(1000)],
        "total": 1.0,
        "iteration": 0,
        "downsampled": False,
    }
    bounded = results._bound_plugin(summary)
    assert bounded is not None
    assert bounded["downsampled"] is True
    assert len(bounded["bins_kev"]) < 1000
    assert results._escaped_size(bounded) <= 512


def test_stride_keeps_the_last_element() -> None:
    """``_stride`` must not drop the array tail it claims to keep (M1)."""
    values = [float(i) for i in range(1024)]
    strided, downsampled = results._stride(values, 256)
    assert downsampled is True
    assert strided[0] == pytest.approx(0.0)
    assert strided[-1] == pytest.approx(1023.0)
    assert len(strided) <= 257  # 256 strided points plus the appended last


def test_bound_plugin_keeps_the_strided_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    """The over-budget striding path also keeps each array's last element (M1)."""
    monkeypatch.setattr(results, "MAX_RESULT_BYTES", 256)
    summary = {
        "bins_kev": [float(i) for i in range(50)],
        "counts": [1000.0 + i for i in range(50)],
        "downsampled": False,
    }
    bounded = results._bound_plugin(summary)
    assert bounded is not None
    assert bounded["downsampled"] is True
    assert bounded["bins_kev"][-1] == pytest.approx(49.0)
    assert bounded["counts"][-1] == pytest.approx(1049.0)


def test_result_params_rejects_extra_fields() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="bogus"):
        ResultParams.model_validate({"sim_id": SIM_ID, "op": "plugin", "reader": "energy_histogram", "bogus": 1})


def test_real_energy_histogram_reader_end_to_end(tmp_path: Path) -> None:
    """The real ``EnergyHistogramData`` runs through the tool path.

    Builds the exact ``rcp.result_request`` the ``read_plugin_result`` tool sends,
    parses it back into ``ResultParams`` (as the simclient does) and resolves it
    through the engine, so the wire + engine path is covered with a real reader.
    Run with a PIConGPU venv (see the module docstring); skipped in the default
    suite where picongpu is absent.
    """
    pytest.importorskip("picongpu")
    from pic_agentic.protocol.simulation import build_result_command

    run = _tree(tmp_path)
    params = _params(species="e", iteration=50)
    command = build_result_command(sim=SIM_ID, seq=1, params=params, cmd_id="cmd")
    assert command.payload["op"] == "plugin"
    assert command.payload["reader"] == "energy_histogram"
    parsed = ResultParams.model_validate(
        {key: command.payload[key] for key in ResultParams.model_fields if key in command.payload},
    )
    payload = results.resolve_result(parsed, run_dir=run, sim_id=SIM_ID)
    assert "result" in payload, payload
    summary = payload["result"]
    assert summary["iteration"] == 50
    assert summary["count_in_window"]["count"] == pytest.approx(44.0)
    assert summary["total"] == pytest.approx(44.0)
    assert summary["bins_kev"][-1] == pytest.approx(1000.0)
    # The highest populated bin is 1000 keV even though the modal bin is 500.
    assert summary["max_energy_kev"] == pytest.approx(1000.0)


def test_max_energy_is_the_highest_populated_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``max_energy_kev`` is not the modal (argmax-count) edge (M2)."""

    class _StubHistogram:
        def __init__(self, run_directory: str) -> None:
            _ = run_directory

        @staticmethod
        def get_iterations(species: str, species_filter: str = "all") -> list[int]:
            _ = (species, species_filter)
            return [0]

        @staticmethod
        def get(iteration: int, species: str, species_filter: str = "all", **kwargs: object) -> tuple:
            _ = (species, species_filter, kwargs)
            bins = [100.0, 500.0, 1000.0]
            counts = [0.0, 42.0, 2.0]
            return counts, bins, [iteration], 1e-16

    monkeypatch.setattr(results, "_import_plugin_reader", lambda _name: _StubHistogram)
    run = tmp_path / "run"
    write_output_unit(run)
    (run / "simOutput" / "e_energyHistogram_all.dat").write_text("x\n", encoding="utf-8")
    params = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="energy_histogram", species="e")
    payload = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)
    assert "result" in payload, payload
    assert payload["result"]["max_energy_kev"] == pytest.approx(1000.0)


def test_real_emittance_reader_end_to_end(tmp_path: Path) -> None:
    """The real ``EmittanceData`` returns total and slices aligned (B1).

    Drives the real reader through ``resolve_result``; skipped unless PIConGPU
    is importable.  Iteration 50's last slice is its maximum, which used to
    index past ``y_slices`` and degrade to ``no_results``.
    """
    pytest.importorskip("picongpu")
    from plugin_fixtures import emittance_dat

    run = tmp_path / "run"
    write_output_unit(run)
    emittance_dat(run)
    params = ResultParams(
        sim_id=SIM_ID,
        op=ResultOp.PLUGIN,
        reader="emittance",
        species="e",
        iteration=50,
    )
    payload = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)
    assert "result" in payload, payload
    summary = payload["result"]
    assert summary["y_slices_m"] == [0.0, 1.0, 2.0, 3.0]
    assert summary["slice_emit_mrad"] == [1.0, 2.0, 3.0, 9.0]
    assert summary["total_emit_mrad"] == pytest.approx(9.0)
    assert summary["max_emit_mrad"] == pytest.approx(9.0)
    assert summary["max_y_slice_m"] == pytest.approx(3.0)
    assert summary["iteration"] == 50


def test_real_transition_radiation_reader_ignores_other_dat_files(tmp_path: Path) -> None:
    """A coexisting histogram file must not break transRad (B2).

    The shipped ``TransitionRadiationData.get_iterations`` globs *every* ``*.dat``
    and int-parses the name, so ``e_energyHistogram_all.dat`` made it raise
    ``ValueError`` and the request degrade to ``no_results``.  Enumerating the
    matching ``_transRad_<int>.dat`` names fixes that.
    """
    pytest.importorskip("picongpu")
    from plugin_fixtures import energy_histogram_dat, transrad_dat

    run = tmp_path / "run"
    write_output_unit(run)
    for iteration in (0, 50, 100):
        transrad_dat(run, iteration=iteration)
    # The documented LWFA case: a histogram output coexists in simOutput.
    energy_histogram_dat(run)

    params = ResultParams(
        sim_id=SIM_ID,
        op=ResultOp.PLUGIN,
        reader="transition_radiation",
        species="e",
        iteration=100,
    )
    payload = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)
    assert "result" in payload, payload
    summary = payload["result"]
    assert summary["iteration"] == 100
    assert summary["omega_per_s"] == [1e15, 2e15, 3e15, 4e15]
    assert summary["peak_intensity"] == pytest.approx(7.0)
    assert summary["peak_omega_per_s"] == pytest.approx(4e15)


def test_real_transition_radiation_reader_defaults_to_latest(tmp_path: Path) -> None:
    """The real ``TransitionRadiationData`` picks the latest step by default (B3).

    The files are written ``0``, ``50`` and ``100``; selecting the iteration
    baked into the alphabetically first filename would choose ``0``.
    """
    pytest.importorskip("picongpu")
    from plugin_fixtures import transrad_dat

    run = tmp_path / "run"
    write_output_unit(run)
    for iteration in (0, 50, 100):
        transrad_dat(run, iteration=iteration)

    default = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="transition_radiation", species="e")
    payload = results.resolve_result(default, run_dir=run, sim_id=SIM_ID)
    assert "result" in payload, payload
    assert payload["result"]["iteration"] == 100

    explicit = default.model_copy(update={"iteration": "last"})
    last = results.resolve_result(explicit, run_dir=run, sim_id=SIM_ID)
    assert "result" in last, last
    assert last["result"]["iteration"] == 100

    first = default.model_copy(update={"iteration": 0})
    zero = results.resolve_result(first, run_dir=run, sim_id=SIM_ID)
    assert "result" in zero, zero
    assert zero["result"]["iteration"] == 0
