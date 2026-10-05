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
    fields_energy_dat,
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


def test_sniff_format_names_the_field_energy_monitor() -> None:
    """H2: ``fields_energy.dat`` is plain text, not ``binary``."""
    assert results._sniff_format("fields_energy.dat") == "energy_fields"
    # Only the fixed ``fields_energy.dat`` name is this plugin's output; the
    # ``EnergyParticles`` plugin's ``<species>_energy_<filter>.dat`` is a
    # different artifact and must not be mislabelled (m2).
    assert results._sniff_format("myrun_energy.dat") == "binary"
    # A name that matches no plugin pattern is still generic/binary.
    assert results._sniff_format("energy.dat") == "binary"
    assert results._sniff_format("mystery.dat") == "binary"


def test_field_energy_reader_summarizes_a_real_format_file(tmp_path: Path) -> None:
    """H2: the native ``energy_fields`` reader parses the real file with no PIConGPU."""
    run = tmp_path / "run"
    write_output_unit(run)
    fields_energy_dat(run, steps=(0, 50, 100), totals=(1.0e-5, 2.0e-5, 1.5e-5))
    params = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="energy_fields", iteration="last")
    payload = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)
    assert "result" in payload, payload
    summary = payload["result"]
    assert summary["step"] == [0, 50, 100]
    assert summary["total_J"] == pytest.approx([1.0e-5, 2.0e-5, 1.5e-5])
    assert summary["selected_step"] == 100
    assert summary["step_first"] == 0
    assert summary["step_last"] == 100
    assert summary["n_steps"] == 3
    assert summary["total_J_min"] == pytest.approx(1.0e-5)
    assert summary["total_J_max"] == pytest.approx(2.0e-5)
    assert summary["total_J_last"] == pytest.approx(1.5e-5)
    assert summary["units_J"] == "Joule"
    assert summary["component_names"] == ["Bx", "By", "Bz", "Ex", "Ey", "Ez"]
    assert summary["component_last_J"]["Bx"] == pytest.approx(1.5e-5 / 6)
    assert summary["source_path"] == "fields_energy.dat"
    assert summary["source_size_bytes"] > 0
    assert summary["truncated"] is False
    assert "warning" not in summary


def test_field_energy_reader_selects_an_explicit_step(tmp_path: Path) -> None:
    """The ``iteration`` selector picks a row of the file's own history."""
    run = tmp_path / "run"
    write_output_unit(run)
    fields_energy_dat(run, steps=(0, 50, 100), totals=(1.0e-5, 2.0e-5, 1.5e-5))
    params = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="energy_fields", iteration=50)
    payload = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)
    summary = payload["result"]
    assert summary["selected_step"] == 50
    assert summary["component_last_J"]["Ex"] == pytest.approx(2.0e-5 / 6)
    # The selected row's total is step 50's 2e-5, not the file's last 1.5e-5
    # (M1): the two must not be conflated under an explicit iteration.
    assert summary["total_J_selected"] == pytest.approx(2.0e-5)
    assert summary["total_J_last"] == pytest.approx(1.5e-5)

    missing = params.model_copy(update={"iteration": 7})
    gone = results.resolve_result(missing, run_dir=run, sim_id=SIM_ID)
    assert gone["error_code"] == "no_results"


def test_field_energy_physics_fact_names_the_selected_step(tmp_path: Path) -> None:
    """The H2 physics fact uses the selected row's total, not the latest (M1)."""
    from pic_agentic import analysis

    run = tmp_path / "run"
    write_output_unit(run)
    fields_energy_dat(run, steps=(0, 50, 100), totals=(1.0e-5, 2.0e-5, 3.0e-5))
    params = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="energy_fields", iteration=50)
    summary = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)["result"]
    facts = analysis._plugin_physics_facts({"energy_fields": summary})
    joined = "; ".join(facts)
    # 2e-5 is step 50's total; 3e-5 is step 100's and must not be tagged as 50.
    assert "at iteration 50 is 2e-05 J" in joined
    assert "at iteration 50 is 3e-05" not in joined


def test_field_energy_file_reads_as_a_bounded_text_tail(tmp_path: Path) -> None:
    """H2: ``read_result`` no longer rejects ``fields_energy.dat`` as binary."""
    run = tmp_path / "run"
    write_output_unit(run)
    fields_energy_dat(run)
    params = ResultParams(sim_id=SIM_ID, op=ResultOp.READ, path="fields_energy.dat", tail=10)
    payload = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)
    assert payload["data_encoding"] == "text"
    assert payload["data"][0].startswith("#step")


def test_field_energy_is_in_describe_and_analyze(tmp_path: Path) -> None:
    """H2: the artifact is labelled and summarized by ``describe``/``analyze``."""
    run = tmp_path / "run"
    write_output_unit(run)
    fields_energy_dat(run)
    manifest = results.scan_output(run / "simOutput", sim_id=SIM_ID, run_dir=str(run))
    formats = {ref.path: ref.format for ref in manifest.files}
    assert formats["fields_energy.dat"] == "energy_fields"

    from pic_agentic import analysis

    summaries = analysis.read_plugin_summaries(run / "simOutput")
    assert summaries["energy_fields"]["total_J_last"] == pytest.approx(1.5e-5)
    answer = analysis.synthesize_answer("what is the total field energy?", {}, {}, {}, summaries)
    assert "field energy" in answer


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
    assert summary["source_path"] == "e_energyHistogram_all.dat"
    assert summary["source_size_bytes"] > 0
    assert "warning" not in summary


def test_plugin_all_zero_histogram_carries_a_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An all-zero histogram must never be returned silently (beta run 3).

    The reviewer could not tell an empty (plasma-free) result from a reader bug
    because the summary was plain zeros; the warning names the ambiguity and the
    source fields let a caller sanity-check the file on disk.
    """

    class _ZeroReader:
        def __init__(self, run_directory: str) -> None:
            _ = run_directory

        @staticmethod
        def get_iterations(species: str, species_filter: str = "all") -> list[int]:
            _ = (species, species_filter)
            return [0]

        @staticmethod
        def get(iteration: int, species: str, species_filter: str = "all", **kwargs: object) -> tuple:
            _ = (species, species_filter, kwargs)
            return [0.0] * 10, [1000.0 * (i + 1) / 10 for i in range(10)], [iteration], 1e-16

    monkeypatch.setattr(results, "_import_plugin_reader", lambda _name: _ZeroReader)
    run = _tree(tmp_path)
    payload = results.resolve_result(_params(species="e"), run_dir=run, sim_id=SIM_ID)
    assert "result" in payload, payload
    summary = payload["result"]
    assert summary["total"] == pytest.approx(0.0)
    assert summary["max_energy_kev"] is None
    assert "warning" in summary
    assert "all zeros" in summary["warning"]
    assert summary["source_size_bytes"] > 0


def test_probe_vacuity_flags_an_all_zero_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A done run whose only numeric artifact is empty is suspect (F4).

    This is the beta-4 failure: three done leaves, no failures, zero electrons.
    The probe must reuse the all-zero warning path rather than invent a new
    detector.
    """

    class _ZeroReader:
        def __init__(self, run_directory: str) -> None:
            _ = run_directory

        @staticmethod
        def get_iterations(species: str, species_filter: str = "all") -> list[int]:
            _ = (species, species_filter)
            return [0]

        @staticmethod
        def get(iteration: int, species: str, species_filter: str = "all", **kwargs: object) -> tuple:
            _ = (species, species_filter, kwargs)
            return [0.0] * 10, [1000.0 * (i + 1) / 10 for i in range(10)], [iteration], 1e-16

    monkeypatch.setattr(results, "_import_plugin_reader", lambda _name: _ZeroReader)
    run = tmp_path / "run"
    write_output_unit(run)
    energy_histogram_dat(run, zero=True)
    warning = results.probe_vacuity(SIM_ID, run_dir=run)
    assert warning is not None
    assert "all zeros" in warning


def test_probe_vacuity_clears_a_populated_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run with a populated histogram is not suspect."""

    class _LiveReader:
        def __init__(self, run_directory: str) -> None:
            _ = run_directory

        @staticmethod
        def get_iterations(species: str, species_filter: str = "all") -> list[int]:
            _ = (species, species_filter)
            return [0]

        @staticmethod
        def get(iteration: int, species: str, species_filter: str = "all", **kwargs: object) -> tuple:
            _ = (species, species_filter, kwargs)
            counts = [0.0] * 10
            counts[4] = 42.0
            return counts, [1000.0 * (i + 1) / 10 for i in range(10)], [iteration], 1e-16

    monkeypatch.setattr(results, "_import_plugin_reader", lambda _name: _LiveReader)
    run = tmp_path / "run"
    write_output_unit(run)
    energy_histogram_dat(run)
    assert results.probe_vacuity(SIM_ID, run_dir=run) is None


def test_probe_vacuity_without_a_numeric_artifact_is_not_suspect(tmp_path: Path) -> None:
    """No numeric artifact cannot be judged, so it is not flagged as empty."""
    run = tmp_path / "run"
    write_output_unit(run)
    assert results.probe_vacuity(SIM_ID, run_dir=run) is None


def test_probe_vacuity_examines_every_matching_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every species' artifact participates in the verdict, not just the first.

    This is the B1 regression from the R10 review: with two species the probe
    read only ``a_energyHistogram_all.dat`` (alphabetically first), so a
    populated ``b`` was a false positive and a populated ``a`` with an empty
    ``b`` was a false negative.  The stub is species-aware so the test pins the
    aggregation without needing PIConGPU; the real readers are exercised by the
    plugin tests below when PIConGPU is importable.
    """

    class _SpeciesReader:
        def __init__(self, run_directory: str) -> None:
            self.run_directory = Path(run_directory)

        @staticmethod
        def get_iterations(species: str, species_filter: str = "all") -> list[int]:
            _ = (species, species_filter)
            return [0]

        def get(self, iteration: int, species: str, species_filter: str = "all", **kwargs: object) -> tuple:
            _ = kwargs
            path = self.run_directory / "simOutput" / f"{species}_energyHistogram_{species_filter}.dat"
            populated = "42" in path.read_text(encoding="utf-8")
            counts = [0.0] * 10
            if populated:
                counts[4] = 42.0
            return counts, [1000.0 * (i + 1) / 10 for i in range(10)], [iteration], 1e-16

    monkeypatch.setattr(results, "_import_plugin_reader", lambda _name: _SpeciesReader)
    run = tmp_path / "run"
    write_output_unit(run)

    energy_histogram_dat(run, species="a", zero=True)
    energy_histogram_dat(run, species="b")
    assert results.probe_vacuity(SIM_ID, run_dir=run) is None, "populated b must clear the run"

    # Both species empty: the run is now suspect.
    energy_histogram_dat(run, species="a", zero=True)
    energy_histogram_dat(run, species="b", zero=True)
    warning = results.probe_vacuity(SIM_ID, run_dir=run)
    assert warning is not None
    assert "all zeros" in warning


def test_annotate_vacuous_covers_the_numeric_readers() -> None:
    """Each numeric text reader's value array drives the warning independently."""
    cases = {
        "energy_histogram": {"counts": [0.0, 0.0], "bins_kev": [1.0, 2.0]},
        "emittance": {"slice_emit_mrad": [0.0, 0.0], "y_slices_m": [0.0, 1.0]},
        "transition_radiation": {"total_intensity": 0.0, "intensity": [0.0]},
    }
    for reader, summary in cases.items():
        annotated = results._annotate_vacuous(reader, summary)
        assert "all zeros" in annotated["warning"]
    nonzero = results._annotate_vacuous("energy_histogram", {"counts": [0.0, 3.0], "bins_kev": [1.0, 2.0]})
    assert "warning" not in nonzero


def test_transition_radiation_vacuity_uses_the_total_not_the_stride() -> None:
    """The all-zero warning keys on ``total_intensity``, not the strided view.

    ``intensity`` is subsampled with ``[::step]``; a stride that happens to
    select only zeros must not raise a false "all zeros", and a genuinely
    nonzero total must not be masked by an all-zero strided view.
    """
    # Strided view is all zero but the measured total is nonzero: no warning.
    live = {"total_intensity": 7.0, "intensity": [0.0, 0.0], "omega_per_s": [1e15, 2e15]}
    assert "warning" not in results._annotate_vacuous("transition_radiation", live)
    # A zero total warns even if the strided view is somehow nonzero.
    dead = {"total_intensity": 0.0, "intensity": [1.0], "omega_per_s": [1e15]}
    assert "all zeros" in results._annotate_vacuous("transition_radiation", dead)["warning"]


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
    assert summary["source_path"] == "e_energyHistogram_all.dat"
    assert summary["source_size_bytes"] > 0
    assert "warning" not in summary


def test_real_all_zero_histogram_is_flagged(tmp_path: Path) -> None:
    """The real reader on an empty spectrum still yields a warning, not silence.

    This is the beta-run-3 shape: the file parses and reports a valid axis, but
    every count is zero because the run had no particles in range.
    """
    pytest.importorskip("picongpu")

    run = tmp_path / "run"
    write_output_unit(run)
    energy_histogram_dat(run, species="e", zero=True)
    payload = results.resolve_result(_params(species="e"), run_dir=run, sim_id=SIM_ID)
    assert "result" in payload, payload
    summary = payload["result"]
    assert summary["total"] == pytest.approx(0.0)
    assert summary["max_energy_kev"] is None
    assert "all zeros" in summary["warning"]


def test_real_nonzero_histogram_survives_the_real_reader(tmp_path: Path) -> None:
    """The faithful writer layout yields nonzero counts (the reader is not the bug).

    The C++ writer terminates its header with a trailing space before ``>``; the
    old synthetic fixture omitted that space and hid a real-reader bug where a
    count could be paired with the lower-edge bin.  This drives the real reader
    over the faithful layout and asserts the window count.
    """
    pytest.importorskip("picongpu")

    run = tmp_path / "run"
    write_output_unit(run)
    energy_histogram_dat(run, species="e", peak_bin=4, peak_count=42)
    payload = results.resolve_result(_params(species="e", iteration=50), run_dir=run, sim_id=SIM_ID)
    assert "result" in payload, payload
    summary = payload["result"]
    assert summary["total"] == pytest.approx(44.0)
    assert summary["count_in_window"]["count"] == pytest.approx(44.0)
    assert "warning" not in summary


def test_real_nonzero_histogram_with_a_nonzero_minimum(tmp_path: Path) -> None:
    """A non-zero ``minEnergy`` histogram reports its real edges and window (B1).

    ``EnergyHistogramData`` returns only upper edges, so the first bin's lower
    edge is the header's ``minEnergy`` - not a hard-coded ``0``.  The reviewer's
    real-pin repro used a single populated low bin; here the header starts at
    1000 keV, so a hard-coded 0 would mis-place every edge and undercount.
    """
    pytest.importorskip("picongpu")

    run = tmp_path / "run"
    write_output_unit(run)
    # ``min_kev=1000`` over a 1000 keV span in 10 bins: the fixture populates
    # bin 0 [1000, 1100), bin 4 [1400, 1500) and bin 9 [1900, 2000).
    energy_histogram_dat(run, species="e", min_kev=1000.0, max_kev=2000.0, peak_bin=4, peak_count=5, iterations=(0,))
    payload = results.resolve_result(_params(species="e", iteration=0), run_dir=run, sim_id=SIM_ID)
    assert "result" in payload, payload
    summary = payload["result"]
    # The first bin's lower edge is the header's 1000 keV, not 0.
    assert summary["min_energy_kev"] == pytest.approx(1000.0)
    assert summary["max_energy_kev"] == pytest.approx(2000.0)
    assert summary["count_in_window"]["min_kev"] == pytest.approx(1000.0)
    assert summary["count_in_window"]["max_kev"] == pytest.approx(2000.0)
    assert summary["count_in_window"]["count"] == pytest.approx(7.0)
    assert summary["total"] == pytest.approx(7.0)

    # The exact physical first-window [1000, 1100) counts the first bin (1),
    # not a spurious 0 from a mis-placed 0 edge.
    first = results.resolve_result(
        _params(species="e", iteration=0, min_kev=1000.0, max_kev=1100.0),
        run_dir=run,
        sim_id=SIM_ID,
    )
    assert first["result"]["count_in_window"]["count"] == pytest.approx(1.0)


def test_real_probe_vacuity_aggregates_every_species(tmp_path: Path) -> None:
    """The real reader path probes *all* species, not the first (B1).

    The R10 review reproduced a false positive and a false negative on the real
    pin: with ``a_energyHistogram_all.dat`` empty and ``b_...`` populated the
    probe read only ``a`` and flagged the run; the reverse order reported it
    clean.  Every matching artifact must participate.
    """
    pytest.importorskip("picongpu")

    run = tmp_path / "run"
    write_output_unit(run)
    energy_histogram_dat(run, species="a", zero=True)
    energy_histogram_dat(run, species="b", zero=False)
    assert results.probe_vacuity(SIM_ID, run_dir=run) is None

    energy_histogram_dat(run, species="a", zero=True)
    energy_histogram_dat(run, species="b", zero=True)
    warning = results.probe_vacuity(SIM_ID, run_dir=run)
    assert warning is not None
    assert "all zeros" in warning


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


class _HighEnergyHistogram:
    """An LWFA-like spectrum whose populated bins start above 1000 keV.

    The beta-4 shape: the fixed 100--1000 keV default window held no particles
    at all while the spectrum is populated in the few-MeV range.
    """

    #: Upper edges [keV] of the four bins; only the last three hold particles.
    _EDGES: ClassVar[list[float]] = [2500.0, 5000.0, 10000.0, 20000.0]
    _COUNTS: ClassVar[list[float]] = [0.0, 1.0e8, 9.0e7, 1.0e7]

    def __init__(self, run_directory: str) -> None:
        _ = run_directory

    @staticmethod
    def get_iterations(species: str, species_filter: str = "all") -> list[int]:
        _ = (species, species_filter)
        return [100]

    @staticmethod
    def get(iteration: int, species: str, species_filter: str = "all", **kwargs: object) -> tuple:
        _ = (species, species_filter, kwargs)
        return list(_HighEnergyHistogram._COUNTS), list(_HighEnergyHistogram._EDGES), [iteration], 1e-16


def _high_energy_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(results, "_import_plugin_reader", lambda _name: _HighEnergyHistogram)
    return _high_energy_tree(tmp_path)


def _high_energy_tree(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    write_output_unit(run)
    (run / "simOutput" / "e_energyHistogram_all.dat").write_text("x\n", encoding="utf-8")
    return run


def test_energy_histogram_default_window_tracks_a_high_energy_spectrum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default window must never be structurally empty (F2).

    The preferred 100--1000 keV window has no populated bin here, so the default
    must widen to the populated range and report the real ~2e8 count instead of
    the misleading ``count: 0`` beta-4 produced.
    """
    run = _high_energy_run(tmp_path, monkeypatch)
    payload = results.resolve_result(_params(species="e"), run_dir=run, sim_id=SIM_ID)
    summary = payload["result"]
    # Half-open bins: the lowest populated bin is [2500, 5000) keV, so the
    # populated range starts at its lower edge (2500), and the derived window
    # [2500, 20000) counts all three populated bins.
    assert summary["count_in_window"]["min_kev"] == pytest.approx(2500.0)
    assert summary["count_in_window"]["max_kev"] == pytest.approx(20000.0)
    assert summary["count_in_window"]["count"] == pytest.approx(2.0e8)
    assert summary["n_nonzero_bins"] == 3
    assert summary["min_energy_kev"] == pytest.approx(2500.0)
    assert summary["max_energy_kev"] == pytest.approx(20000.0)
    # A populated spectrum with a matching default is not a mis-window.
    assert "warning" not in summary


def test_energy_histogram_prefers_the_standard_window_when_populated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The preferred window is kept when the populated span fits (F2 no-regression)."""

    class _WindowedHistogram:
        """A spectrum whose lowest populated bin is [100, 200) keV.

        Bin 0 ([0, 100)) is empty, so the populated half-open span is
        [100, 1000), which fits in the preferred 100--1000 keV window.
        """

        def __init__(self, run_directory: str) -> None:
            _ = run_directory

        @staticmethod
        def get_iterations(species: str, species_filter: str = "all") -> list[int]:
            _ = (species, species_filter)
            return [0]

        @staticmethod
        def get(iteration: int, species: str, species_filter: str = "all", **kwargs: object) -> tuple:
            _ = (species, species_filter, kwargs)
            bins = [100.0 * (i + 1) for i in range(10)]
            counts = [0.0] * 10
            counts[1] = 42.0
            counts[-1] = 1.0
            return counts, bins, [iteration], 1e-16

    monkeypatch.setattr(results, "_import_plugin_reader", lambda _name: _WindowedHistogram)
    run = _high_energy_tree(tmp_path)
    payload = results.resolve_result(_params(species="e"), run_dir=run, sim_id=SIM_ID)
    summary = payload["result"]
    assert summary["count_in_window"] == {"min_kev": 100.0, "max_kev": 1000.0, "count": pytest.approx(43.0)}
    assert summary["n_nonzero_bins"] == 2


class _SpanningHistogram:
    """A spectrum that spans the 100--1000 keV window to 20 MeV (F2 Major).

    Its lowest populated bin is [0, 900) keV and it reaches 20 MeV, so the
    populated half-open span [0, 20000) is not contained in the preferred
    100--1000 keV window; "any populated bin inside" would silently return a
    count of 1 and hide ~1e8 electrons.
    """

    _EDGES: ClassVar[list[float]] = [900.0, 20000.0]
    _COUNTS: ClassVar[list[float]] = [1.0, 1.0e8]

    def __init__(self, run_directory: str) -> None:
        _ = run_directory

    @staticmethod
    def get_iterations(species: str, species_filter: str = "all") -> list[int]:
        _ = (species, species_filter)
        return [100]

    @staticmethod
    def get(iteration: int, species: str, species_filter: str = "all", **kwargs: object) -> tuple:
        _ = (species, species_filter, kwargs)
        return list(_SpanningHistogram._COUNTS), list(_SpanningHistogram._EDGES), [iteration], 1e-16


def test_energy_histogram_default_window_covers_a_spanning_spectrum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A default that only partially covers the data must widen, not clip (F2).

    The spectrum's lowest populated bin is [0, 900) keV and the population
    reaches 20000 keV, so the populated span is not inside 100--1000 keV.  The
    derived window must capture the whole range (and thus the whole count), not
    return the sliver inside the preferred window.
    """
    monkeypatch.setattr(results, "_import_plugin_reader", lambda _name: _SpanningHistogram)
    run = _high_energy_tree(tmp_path)
    payload = results.resolve_result(_params(species="e"), run_dir=run, sim_id=SIM_ID)
    summary = payload["result"]
    assert summary["count_in_window"]["min_kev"] == pytest.approx(0.0)
    assert summary["count_in_window"]["max_kev"] == pytest.approx(20000.0)
    assert summary["count_in_window"]["count"] == pytest.approx(1.0e8 + 1.0)
    assert summary["total"] == pytest.approx(1.0e8 + 1.0)
    assert "warning" not in summary


def test_energy_histogram_explicit_partial_window_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A requested window that captures a sliver of the data warns (F2 Minor).

    The caller chose the band, so it is honoured literally, but a count that is
    a tiny fraction of the total must still be flagged rather than presented as
    the electron count.
    """
    run = _high_energy_run(tmp_path, monkeypatch)
    # Bins are [0,2500),[2500,5000),[5000,10000),[10000,20000) keV. The
    # half-open window [2500, 5000) counts only the [2500,5000) bin (1e8).
    payload = results.resolve_result(
        _params(species="e", min_kev=2500.0, max_kev=5000.0),
        run_dir=run,
        sim_id=SIM_ID,
    )
    summary = payload["result"]
    assert summary["count_in_window"] == {"min_kev": 2500.0, "max_kev": 5000.0, "count": pytest.approx(1.0e8)}
    assert summary["total"] == pytest.approx(2.0e8)
    assert "warning" in summary
    assert "count_in_window is 1e+08 of 2e+08" in summary["warning"]


def test_energy_histogram_single_bin_default_is_not_degenerate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A single populated bin must not yield the degenerate ``max == min`` (F2 Minor)."""

    class _OneBinHistogram:
        def __init__(self, run_directory: str) -> None:
            _ = run_directory

        @staticmethod
        def get_iterations(species: str, species_filter: str = "all") -> list[int]:
            _ = (species, species_filter)
            return [100]

        @staticmethod
        def get(iteration: int, species: str, species_filter: str = "all", **kwargs: object) -> tuple:
            _ = (species, species_filter, kwargs)
            return [7.0], [2500.0], [iteration], 1e-16

    monkeypatch.setattr(results, "_import_plugin_reader", lambda _name: _OneBinHistogram)
    run = _high_energy_tree(tmp_path)
    payload = results.resolve_result(_params(species="e"), run_dir=run, sim_id=SIM_ID)
    summary = payload["result"]
    window = summary["count_in_window"]
    # Half-open: the single bin is [0, 2500) keV, so the derived window is
    # [0, 2500) and counts the bin.
    assert window["min_kev"] == pytest.approx(0.0)
    assert window["max_kev"] == pytest.approx(2500.0)
    assert window["max_kev"] > window["min_kev"]
    assert window["count"] == pytest.approx(7.0)
    # The window is one the wire model would accept if requested.
    ResultParams(
        sim_id=SIM_ID,
        op=ResultOp.PLUGIN,
        reader="energy_histogram",
        min_kev=window["min_kev"],
        max_kev=window["max_kev"],
    )


def test_energy_histogram_requestable_window_below_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit window below the data yields 0 but flags it as a mis-window (F2).

    A requested window is honoured literally, so a caller can ask for any band;
    the ``n_nonzero_bins``/populated-range fields and the warning make the empty
    count self-explanatory rather than a silent contradiction.
    """
    run = _high_energy_run(tmp_path, monkeypatch)
    payload = results.resolve_result(
        _params(species="e", min_kev=100.0, max_kev=1000.0),
        run_dir=run,
        sim_id=SIM_ID,
    )
    summary = payload["result"]
    assert summary["count_in_window"] == {"min_kev": 100.0, "max_kev": 1000.0, "count": pytest.approx(0.0)}
    assert summary["n_nonzero_bins"] == 3
    assert summary["min_energy_kev"] == pytest.approx(2500.0)
    assert "warning" in summary
    assert "count_in_window is 0" in summary["warning"]


def test_energy_histogram_requestable_window_above_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit window above the data is an empty (but transparent) count (F2)."""
    run = _high_energy_run(tmp_path, monkeypatch)
    payload = results.resolve_result(
        _params(species="e", min_kev=50000.0, max_kev=100000.0),
        run_dir=run,
        sim_id=SIM_ID,
    )
    summary = payload["result"]
    assert summary["count_in_window"]["count"] == pytest.approx(0.0)
    assert summary["count_in_window"]["min_kev"] == pytest.approx(50000.0)
    assert summary["n_nonzero_bins"] == 3
    assert "warning" in summary


def test_energy_histogram_requestable_window_selects_a_subrange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A requested window that covers part of the range counts only that part (F2)."""
    run = _high_energy_run(tmp_path, monkeypatch)
    payload = results.resolve_result(
        _params(species="e", min_kev=2500.0, max_kev=10000.0),
        run_dir=run,
        sim_id=SIM_ID,
    )
    summary = payload["result"]
    # Bins [2500,5000), [5000,10000), [10000,20000) have lower edges 2500,
    # 5000, 10000. The half-open window [2500,10000) selects the first two
    # (1e8 + 9e7); the [10000,20000) bin's lower edge 10000 is excluded.
    assert summary["count_in_window"]["count"] == pytest.approx(1.9e8)
    assert "warning" not in summary


def test_energy_histogram_all_zero_still_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuinely all-zero histogram keeps the existing vacuous warning (F2)."""

    class _ZeroHistogram:
        def __init__(self, run_directory: str) -> None:
            _ = run_directory

        @staticmethod
        def get_iterations(species: str, species_filter: str = "all") -> list[int]:
            _ = (species, species_filter)
            return [0]

        @staticmethod
        def get(iteration: int, species: str, species_filter: str = "all", **kwargs: object) -> tuple:
            _ = (species, species_filter, kwargs)
            return [0.0] * 4, [2500.0, 5000.0, 10000.0, 20000.0], [iteration], 1e-16

    monkeypatch.setattr(results, "_import_plugin_reader", lambda _name: _ZeroHistogram)
    run = _high_energy_tree(tmp_path)
    payload = results.resolve_result(_params(species="e"), run_dir=run, sim_id=SIM_ID)
    summary = payload["result"]
    assert summary["total"] == pytest.approx(0.0)
    assert summary["n_nonzero_bins"] == 0
    assert summary["min_energy_kev"] is None
    # With no populated bins the default window stays the standard range.
    assert summary["count_in_window"] == {"min_kev": 100.0, "max_kev": 1000.0, "count": pytest.approx(0.0)}
    assert "warning" in summary
    assert "all zeros" in summary["warning"]


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


def test_radiation_rejects_a_species_filter(tmp_path: Path) -> None:
    """A non-default ``species_filter`` for radiation is rejected, not ignored (m3).

    ``*_radAmplitudes_*`` carries no filter component, so honouring the request
    is impossible; the engine must say so instead of silently returning the
    ``all`` series.
    """
    run = tmp_path / "run"
    write_output_unit(run)
    (run / "simOutput" / "radiationOpenPMD").mkdir(parents=True)
    params = ResultParams(
        sim_id=SIM_ID,
        op=ResultOp.PLUGIN,
        reader="radiation",
        species="e",
        species_filter="laser",
    )
    payload = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)
    assert payload["error_code"] == "unsupported"
    assert "species_filter" in payload["error"]
    # The default "all" filter is accepted (and proceeds to the missing-file path).
    default = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="radiation", species="e")
    payload = results.resolve_result(default, run_dir=run, sim_id=SIM_ID)
    assert payload["error_code"] == "no_results"


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


def test_real_phase_space_reader_end_to_end_bp(tmp_path: Path) -> None:
    """``PhaseSpaceData`` reads an ADIOS2 ``.bp`` series when given the suffix (M1).

    ``PhaseSpaceData.get_data_path`` defaults to ``h5``, so without threading the
    target's suffix through, a ``.bp`` series was unreadable and returned
    ``no_results`` even though the pattern advertised it.  The reader also only
    accepts the exact suffix, so this is the regression guard for that path.
    """
    pytest.importorskip("picongpu")
    pytest.importorskip("openpmd_api")
    from plugin_fixtures import _adios2_available

    _adios2_available()
    run = tmp_path / "run"
    write_output_unit(run)
    phase_space_h5(run, shape=(4, 3), ext="bp")
    assert (run / "simOutput" / "phaseSpace" / "PhaseSpace_e_all_ypy_100.bp").is_dir()

    params = ResultParams(sim_id=SIM_ID, op=ResultOp.PLUGIN, reader="phase_space", species="e", iteration=100)
    payload = results.resolve_result(params, run_dir=run, sim_id=SIM_ID)
    assert "result" in payload, payload
    summary = payload["result"]
    assert summary["iteration"] == 100
    assert summary["total_count"] == pytest.approx(66.0 + 100 * 12)
    assert summary["projection_r"] == pytest.approx([303.0, 312.0, 321.0, 330.0])


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


@pytest.mark.filterwarnings("ignore:Starting with ImageIO v3:DeprecationWarning")
def test_real_png_reader_returns_metadata(tmp_path: Path) -> None:
    """The real ``PNGData`` returns dimensions, not pixels (slice 2).

    PIConGPU's shipped ``PNGData`` calls the deprecated ``imageio.imread``
    (imageio v3 semantics), so this one test tolerates that third-party
    ``DeprecationWarning``; the ignore is scoped here rather than globally.
    """
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
