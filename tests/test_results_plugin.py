# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for the G1 PIConGPU plugin result readers.

The offline tests use a stub reader injected in place of the optional PIConGPU
package, so the default suite (no picongpu) still passes.  The final tests run
the *real* readers (``EnergyHistogramData``, ``EmittanceData``,
``TransitionRadiationData``, ``PhaseSpaceData``, ``particleCalorimeter``,
``RadiationData``, ``PNGData``) and are skipped unless PIConGPU is importable;
run them with a PIConGPU venv, e.g.::

    PYTHONPATH=<repo>/src /tmp/opencode/pic-stack-venv/bin/python -m pytest tests/test_results_plugin.py -q

"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import ClassVar

import pytest
from plugin_fixtures import (
    calorimeter_h5,
    energy_histogram_dat,
    phase_space_h5,
    png_file,
    radiation_h5,
    write_output_unit,
)

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


def test_sniff_format_is_a_documented_filename_heuristic(tmp_path: Path) -> None:
    """A plugin-looking name is labelled by shape; the reader stays authoritative (N2).

    The species/filter components are arbitrary identifiers, so the classifier
    cannot require a known species list; it labels ``notes_emittance_x.dat`` as
    ``emittance``.  That is harmless because a request against the file is still
    resolved (and rejected) by the reader, never crashed on.
    """
    assert results._sniff_format("e_emittance_all.dat") == "emittance"
    assert results._sniff_format("notes_emittance_x.dat") == "emittance"
    assert results._sniff_format("notes.txt") == "text"

    run = tmp_path / "run"
    write_output_unit(run)
    (run / "simOutput" / "notes_emittance_x.dat").write_text("x\n", encoding="utf-8")
    params = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="emittance", species="notes")
    payload = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)
    # Either the reader is absent (clean reader_unavailable) or it rejects the
    # mislabelled file (clean no_results); never a crash or generic failure.
    assert payload.get("error_code") in {"reader_unavailable", "no_results"}


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


def test_plugin_reader_soft_error_is_no_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed-but-parseable file that raises TypeError is a soft error (N4)."""

    class _BrokenReader:
        def __init__(self, run_directory: str) -> None:
            _ = run_directory

        @staticmethod
        def get_iterations(species: str, species_filter: str = "all") -> list[int]:
            _ = (species, species_filter)
            return [0]

        @staticmethod
        def get(iteration: int, species: str, species_filter: str = "all", **kwargs: object) -> tuple:
            _ = (iteration, species, species_filter, kwargs)
            msg = "None column labels"
            raise TypeError(msg)

    monkeypatch.setattr(results, "_import_plugin_reader", lambda _name: _BrokenReader)
    run = tmp_path / "run"
    write_output_unit(run)
    (run / "simOutput" / "e_energyHistogram_all.dat").write_text("x\n", encoding="utf-8")
    payload = results.resolve_result(_params(species="e"), run_dir=run, sim_id=SIM_ID)
    assert payload["error_code"] == "no_results"


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


# --- openPMD / image readers (slice 2) -------------------------------------


def _openpmd_tree(tmp_path: Path) -> Path:
    """Build a run with a phase-space series on disk (no openpmd needed)."""
    run = tmp_path / "run"
    write_output_unit(run)
    (run / "simOutput" / "phaseSpace").mkdir(parents=True, exist_ok=True)
    for iteration in (0, 50, 100):
        (run / "simOutput" / "phaseSpace" / f"PhaseSpace_e_all_ypy_{iteration}.h5").write_bytes(b"x")
    return run


def test_registry_matches_the_wire_names_including_openpmd() -> None:
    from pic_agentic.protocol.simulation import PLUGIN_READER_NAMES

    assert set(results._PLUGIN_READERS) == set(PLUGIN_READER_NAMES)
    assert set(results._PLUGIN_BUILDERS) == set(PLUGIN_READER_NAMES)
    assert {"phase_space", "radiation", "calorimeter", "png"} <= set(PLUGIN_READER_NAMES)


def test_sniff_format_names_the_openpmd_and_image_readers(tmp_path: Path) -> None:
    assert results._sniff_format("PhaseSpace_e_all_ypy_100.h5") == "phase_space"
    assert results._sniff_format("e_radAmplitudes_100_0_0_0.h5") == "radiation"
    assert results._sniff_format("e_calorimeter_all_100.h5") == "calorimeter"
    assert results._sniff_format("e_png_yx_0.5_000100.png") == "png"
    # A plain field series is still reported by its backend, not a plugin.
    assert results._sniff_format("fields_100.h5") == "openpmd-hdf5"
    assert results._sniff_format("fields.bp") == "openpmd-adios2"

    run = _openpmd_tree(tmp_path)
    manifest = results.scan_output(run / "simOutput", sim_id=SIM_ID, run_dir=str(run))
    formats = {ref.path: ref.format for ref in manifest.files}
    assert formats["phaseSpace/PhaseSpace_e_all_ypy_100.h5"] == "phase_space"


class _StubPhaseSpace:
    """Stand-in for ``PhaseSpaceData`` (returns a 2D plane + metadata)."""

    class _Meta:
        r_edges: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)
        p_edges: tuple[float, ...] = (-1.0, 0.0, 1.0)

    def __init__(self, run_directory: str) -> None:
        self.run_directory = run_directory

    @staticmethod
    def get(iteration: int, ps: str, species: str, species_filter: str = "all", **kwargs: object) -> tuple:
        _ = (ps, species, species_filter, kwargs)
        return [[float(iteration), 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]], _StubPhaseSpace._Meta


def test_phase_space_stub_summary_and_iteration_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(results, "_import_plugin_reader", lambda _name: _StubPhaseSpace)
    run = _openpmd_tree(tmp_path)
    params = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="phase_space", species="e")
    payload = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)
    assert "result" in payload, payload
    summary = payload["result"]
    # The files are 0/50/100; the default must be the newest.
    assert summary["iteration"] == 100
    assert summary["n_r"] == 3
    assert summary["n_p"] == 3
    assert summary["r_min_m"] == pytest.approx(0.0)
    assert summary["r_max_m"] == pytest.approx(3.0)
    assert summary["p_min"] == pytest.approx(-1.0)
    assert summary["p_max"] == pytest.approx(1.0)
    assert summary["total_count"] == pytest.approx(103.0)
    assert summary["downsampled"] is False


def test_phase_space_stub_explicit_and_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(results, "_import_plugin_reader", lambda _name: _StubPhaseSpace)
    run = _openpmd_tree(tmp_path)
    explicit = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="phase_space", species="e", iteration=50)
    payload = results.resolve_result(explicit, run_dir=run, sim_id=SIM_ID)
    assert payload["result"]["iteration"] == 50

    missing = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="phase_space", species="positron")
    none = results.resolve_result(missing, run_dir=run, sim_id=SIM_ID)
    assert none["error_code"] == "no_results"


class _StubCalorimeter:
    """Stand-in for ``particleCalorimeter`` (2D and 3D data shapes)."""

    detector_params: ClassVar[dict[str, int]] = {"N_yaw": 4, "N_pitch": 3, "N_energy": 2}

    def __init__(self, series_filename: str) -> None:
        self.series_filename = series_filename

    @staticmethod
    def getEnergy() -> list[float]:
        return [10.0, 1000.0]

    @staticmethod
    def getData(iteration: int) -> list:
        _ = iteration
        return [[float(p * 4 + y) for y in range(4)] for p in range(3)]


def test_calorimeter_stub_projects_both_shapes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(results, "_import_plugin_reader", lambda _name: _StubCalorimeter)
    run = tmp_path / "run"
    write_output_unit(run)
    (run / "simOutput" / "cal").mkdir(parents=True, exist_ok=True)
    (run / "simOutput" / "cal" / "e_calorimeter_all_100.h5").write_bytes(b"x")
    params = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="calorimeter", species="e")
    payload = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)
    assert "result" in payload, payload
    summary = payload["result"]
    assert summary["iteration"] == 100
    assert summary["n_pitch"] == 3
    assert summary["n_yaw"] == 4
    # The 2D shape has no energy axis; pitch is the outer axis.
    assert summary["energy_keV"] is None
    assert summary["per_pitch_J"] == [6.0, 22.0, 38.0]
    assert summary["per_yaw_J"] == [12.0, 15.0, 18.0, 21.0]
    assert summary["total_energy_J"] == pytest.approx(66.0)


def test_openpmd_reader_missing_dependency_is_reader_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _openpmd_tree(tmp_path)
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "openpmd_api":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(results.importlib.util, "find_spec", fake_find_spec)
    params = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="phase_space", species="e")
    payload = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)
    assert payload["error_code"] == "reader_unavailable"


def test_unknown_reader_names_are_rejected_by_the_wire() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="unknown plugin reader"):
        ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="openpmd")


# --- real readers (skipped without PIConGPU) --------------------------------


def test_real_phase_space_reader_end_to_end(tmp_path: Path) -> None:
    """The real ``PhaseSpaceData`` opens the generated openPMD series (slice 2).

    The fixture is written with the pinned ``openpmd_api`` in the layout the
    reader expects; this exercises the real class (and ``openpmd_api``) through
    the engine, not a stub.  Skipped unless PIConGPU + openpmd_api are present.
    """
    pytest.importorskip("picongpu")
    pytest.importorskip("openpmd_api")

    run = tmp_path / "run"
    write_output_unit(run)
    phase_space_h5(run, shape=(4, 3))

    params = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="phase_space", species="e")
    payload = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)
    assert "result" in payload, payload
    summary = payload["result"]
    # Default is the latest iteration (100), not the alphabetically first (0).
    assert summary["iteration"] == 100
    assert summary["ps"] == "ypy"
    assert summary["n_r"] == 4
    assert summary["n_p"] == 3
    assert summary["r_min_m"] == pytest.approx(0.0)
    assert summary["r_max_m"] == pytest.approx(4e-6)
    assert summary["p_min"] == pytest.approx(-1.0)
    assert summary["p_max"] == pytest.approx(1.0)
    assert summary["downsampled"] is False
    # The ramp is 0..11 plus the iteration; assert the exact projections and
    # peak so a scrambled/NaN on-disk series (B1) fails rather than passing by
    # write-order accident.
    assert summary["total_count"] == pytest.approx(66.0 + 100 * 12)
    assert summary["projection_r"] == pytest.approx([303.0, 312.0, 321.0, 330.0])
    assert summary["projection_p"] == pytest.approx([418.0, 422.0, 426.0])
    assert summary["max_count"] == pytest.approx(111.0)
    assert summary["max_r_m"] == pytest.approx(3e-6)
    assert summary["max_p"] == pytest.approx(1.0 / 3.0)


def test_real_calorimeter_reader_end_to_end(tmp_path: Path) -> None:
    """The real ``particleCalorimeter`` opens the generated series (slice 2)."""
    pytest.importorskip("picongpu")
    pytest.importorskip("openpmd_api")

    run = tmp_path / "run"
    write_output_unit(run)
    calorimeter_h5(run, shape=(2, 3, 4))

    params = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="calorimeter", species="e", iteration=50)
    payload = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)
    assert "result" in payload, payload
    summary = payload["result"]
    assert summary["iteration"] == 50
    assert summary["n_energy"] == 2
    assert summary["n_pitch"] == 3
    assert summary["n_yaw"] == 4
    assert summary["energy_keV"] == [10.0, 1000.0]
    # Sum of 0..23 plus 50 per cell over 24 cells.  Assert the marginals and
    # peak so a scrambled/NaN on-disk series (B1) fails rather than passing on
    # the one iteration whose garbage happens to match.
    assert summary["total_energy_J"] == pytest.approx(276.0 + 50 * 24)
    assert summary["per_pitch_J"] == pytest.approx([460.0, 492.0, 524.0])
    assert summary["per_yaw_J"] == pytest.approx([360.0, 366.0, 372.0, 378.0])
    assert summary["max_energy_J"] == pytest.approx(73.0)


def test_real_calorimeter_reader_end_to_end_bp(tmp_path: Path) -> None:
    """The ``calorimeter`` reader serves an ADIOS2 ``.bp`` directory series (B2/B3/M2).

    An ADIOS2 series is a *directory* per iteration; the registry used to reject
    the directory (wrong ``kind``) and iteration discovery skipped it, so a real
    ``.bp`` calorimeter returned ``no_results``.  Both the explicit and the
    default (latest) iteration are read to cover directory iteration discovery.
    """
    pytest.importorskip("picongpu")
    pytest.importorskip("openpmd_api")
    from plugin_fixtures import _adios2_available

    _adios2_available()
    run = tmp_path / "run"
    write_output_unit(run)
    calorimeter_h5(run, shape=(2, 3, 4), ext="bp")
    series_dir = run / "simOutput" / "e_calorimeter" / "e_calorimeter_all_50.bp"
    assert series_dir.is_dir(), series_dir

    explicit = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="calorimeter", species="e", iteration=50)
    payload = results.resolve_result(explicit, run_dir=run, sim_id=SIM_ID)
    assert "result" in payload, payload
    summary = payload["result"]
    assert summary["iteration"] == 50
    assert summary["n_energy"] == 2
    assert summary["n_pitch"] == 3
    assert summary["n_yaw"] == 4
    assert summary["total_energy_J"] == pytest.approx(276.0 + 50 * 24)
    assert summary["per_pitch_J"] == pytest.approx([460.0, 492.0, 524.0])

    # The default must discover the latest iteration from the directories.
    default = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="calorimeter", species="e")
    latest = results.resolve_result(default, run_dir=run, sim_id=SIM_ID)
    assert "result" in latest, latest
    assert latest["result"]["iteration"] == 100
    assert latest["result"]["total_energy_J"] == pytest.approx(276.0 + 100 * 24)


def test_real_radiation_reader_end_to_end(tmp_path: Path) -> None:
    """The real ``RadiationData`` opens the generated series (slice 2)."""
    pytest.importorskip("picongpu")
    pytest.importorskip("openpmd_api")

    run = tmp_path / "run"
    write_output_unit(run)
    radiation_h5(run, n_directions=2, n_frequencies=5)

    params = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="radiation", species="e")
    payload = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)
    assert "result" in payload, payload
    summary = payload["result"]
    assert summary["iteration"] == 100
    assert summary["n_directions"] == 2
    assert summary["n_frequencies"] == 5
    assert summary["omega_per_s"] == [1e14, 2e14, 3e14, 4e14, 5e14]
    # Re(A) = component + iteration (x/y/z => 101/102/103 at step 100), Im = 0,
    # so each cell's spectrum is 101^2 + 102^2 + 103^2 = 31214; two observation
    # directions share every frequency bin, giving 62428 per bin.
    per_frequency = 2 * (101.0**2 + 102.0**2 + 103.0**2)
    assert summary["spectrum"] == pytest.approx([per_frequency] * 5)
    assert summary["total_energy_J"] == pytest.approx(5 * per_frequency)
    assert summary["peak_spectrum_Js"] == pytest.approx(per_frequency)
    assert summary["peak_omega_per_s"] == pytest.approx(1e14)


def test_real_png_reader_returns_metadata(tmp_path: Path) -> None:
    """The real ``PNGData`` returns dimensions, not pixels (slice 2)."""
    pytest.importorskip("picongpu")
    pytest.importorskip("imageio")

    run = tmp_path / "run"
    write_output_unit(run)
    for iteration in (0, 50, 100):
        png_file(run, iteration=iteration, height=8, width=12)

    params = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="png", species="e")
    payload = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)
    assert "result" in payload, payload
    summary = payload["result"]
    assert summary["iteration"] == 100
    assert summary["height_px"] == 8
    assert summary["width_px"] == 12
    assert summary["channels"] == 3
    assert summary["image_via"] == "export"
    # No pixel data travels on the wire.
    assert "data" not in summary
