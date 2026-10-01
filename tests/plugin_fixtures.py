# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Reproducible fixtures for the PIConGPU plugin readers.

The text fixtures are chosen to round-trip through the shipped
``picongpu.extra.plugins.data`` readers: the energy-histogram header splits on
single spaces and needs the ``>`` marker and ``count`` word to be *two*
space-separated tokens, so the C++ writer's literal ``">"`` is emitted with a
space before it, and the counts line has ``num_bins + 2`` values (underflow, the
real bins, overflow) plus the sum.

The openPMD fixtures (:func:`phase_space_h5`, :func:`calorimeter_h5`,
:func:`radiation_h5`) are written with the pinned ``openpmd_api`` and reproduce
the *real* layout the shipped readers expect (record names, attributes and the
``%T`` series pattern).  They are the closest possible integration: the
``PhaseSpaceData``/``particleCalorimeter``/``RadiationData`` classes open the
generated series and return arrays through the engine, exactly as on a cluster -
only the producing C++ binary is replaced by the Python API.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

#: Default time step the synthetic ``simOutput/output`` records.
DEFAULT_DT_SI = 1.39e-16


def _openpmd() -> object:
    """Import the optional ``openpmd_api``, skipping the test when absent.

    Returns:
        The ``openpmd_api`` module.

    """
    import pytest

    if importlib.util.find_spec("openpmd_api") is None:
        pytest.skip("openpmd_api not installed")
    import openpmd_api

    return openpmd_api


def _warm_reader(submodule: str) -> None:
    """Import a reader submodule before writing a series it later reads.

    With the pinned ``openpmd_api`` 0.17.1 HDF5 backend, reading a
    ``%T`` series with the shipped reader returns garbage for some iterations
    unless the reader module was imported *before* the series was written in
    the same process (the backend caches HDF5 state on first reader import).
    Importing it here makes the fixture hermetic: the file read by the test is
    the file this function wrote.
    """
    importlib.import_module(f"picongpu.extra.plugins.data.{submodule}")


def write_output_unit(run_dir: Path, dt_si: float = DEFAULT_DT_SI) -> Path:
    """Write the sibling ``simOutput/output`` file the readers need for ``dt``.

    ``FindTime`` parses ``sim.unit.time() <float>`` from this file.

    Returns:
        The written file's path.

    """
    sim_output = Path(run_dir) / "simOutput"
    sim_output.mkdir(parents=True, exist_ok=True)
    path = sim_output / "output"
    path.write_text(f"\tsim.unit.time() {dt_si}\n", encoding="utf-8")
    return path


def energy_histogram_dat(
    run_dir: Path,
    *,
    species: str = "e",
    species_filter: str = "all",
    iterations: tuple[int, ...] = (0, 50, 100),
    num_bins: int = 10,
    min_kev: float = 0.0,
    max_kev: float = 1000.0,
    peak_bin: int = 4,
    peak_count: int = 42,
    zero: bool = False,
) -> Path:
    """Write a ``<species>_energyHistogram_<filter>.dat`` the real reader parses.

    Every iteration puts ``peak_count`` into ``peak_bin`` and one count into the
    first and last bin, so the default ``count_in_window`` is non-trivial.  Set
    ``zero=True`` to write the all-zero spectrum an empty/plasma-free run
    produces (the diagnostic path under test).

    Returns:
        The written data file's path.

    """
    sim_output = Path(run_dir) / "simOutput"
    sim_output.mkdir(parents=True, exist_ok=True)
    bin_energy = (max_kev - min_kev) / num_bins
    edges = [min_kev + (i + 1) * bin_energy for i in range(num_bins)]
    edge_tokens = " ".join(f"{edge:.10g}" for edge in edges)
    header = "#step <" + f"{min_kev:.10g}" + " " + edge_tokens + f" >{max_kev:.10g} count"
    rows = []
    for iteration in iterations:
        counts = [0.0] * num_bins
        if not zero:
            counts[0] = 1.0
            counts[peak_bin] = float(peak_count)
            counts[-1] = 1.0
        real = [str(int(count)) for count in counts]
        row = [str(iteration), "0", *real, "0", str(int(sum(counts)))]
        rows.append(" ".join(row))
    path = sim_output / f"{species}_energyHistogram_{species_filter}.dat"
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def emittance_dat(
    run_dir: Path,
    *,
    species: str = "e",
    species_filter: str = "all",
    iterations: tuple[int, ...] = (0, 50, 100),
    y_slices: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0),
    totals: tuple[float, ...] = (9.0, 9.0, 10.0),
    slices: tuple[tuple[float, ...], ...] = (
        (1.0, 2.0, 3.0, 4.0),
        (1.0, 2.0, 3.0, 9.0),
        (1.0, 2.0, 3.0, 5.0),
    ),
) -> Path:
    """Write a ``<species>_emittance_<filter>.dat`` the real reader parses.

    Each row is ``iteration sum <slice0> <slice1> ...``: the ``sum`` column is
    the total emittance and the remaining columns the per-slice values, matching
    ``EmittanceData``'s ``[emit_all, *slices]`` return shape.  ``iterations``,
    ``totals`` and ``slices`` are parallel.

    Returns:
        The written data file's path.

    """
    sim_output = Path(run_dir) / "simOutput"
    sim_output.mkdir(parents=True, exist_ok=True)
    header = "iteration sum " + " ".join(f"{y:.10g}" for y in y_slices)
    lines = [header]
    for iteration, total, row in zip(iterations, totals, slices, strict=True):
        lines.append(" ".join([str(iteration), str(int(total)), *[str(int(value)) for value in row]]))
    path = sim_output / f"{species}_emittance_{species_filter}.dat"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def transrad_dat(
    run_dir: Path,
    *,
    species: str = "e",
    iteration: int = 0,
    n_omega: int = 4,
    omega_min: float = 1e15,
    omega_max: float = 4e15,
    n_phi: int = 2,
    n_theta: int = 3,
    peak_row: int = 0,
    peak_column: int = 3,
    peak_intensity: float = 7.0,
) -> Path:
    r"""Write a ``<species>_transRad_<iteration>.dat`` the real reader parses.

    Mirrors the ``TransitionRadiation.x.cpp`` writer: a ``# \\t`` header line of
    ``mode n_omega omega_min omega_max n_phi phi_min phi_max n_theta theta_min
    theta_max`` followed by ``n_theta * n_phi`` tab-separated rows of
    ``n_omega`` intensities (row index ``theta * n_phi + phi``).

    Returns:
        The written data file's path.

    """
    sim_output = Path(run_dir) / "simOutput"
    sim_output.mkdir(parents=True, exist_ok=True)
    header = "# \t" + "\t".join(
        [
            "lin",
            str(n_omega),
            f"{omega_min:.10g}",
            f"{omega_max:.10g}",
            str(n_phi),
            "0",
            "6.28",
            str(n_theta),
            "0",
            "3.14",
        ],
    )
    lines = [header]
    for row_index in range(n_theta * n_phi):
        row = [0.0] * n_omega
        if row_index == peak_row:
            row[peak_column] = peak_intensity
        lines.append("\t".join(f"{value:.10g}" for value in row))
    path = sim_output / f"{species}_transRad_{iteration}.dat"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def phase_space_h5(
    run_dir: Path,
    *,
    species: str = "e",
    species_filter: str = "all",
    ps: str = "ypy",
    iterations: tuple[int, ...] = (0, 50, 100),
    shape: tuple[int, int] = (4, 3),
    ext: str = "h5",
) -> Path:
    """Write a ``PhaseSpace_<species>_<filter>_<ps>_<iteration>.<ext>`` series.

    Reproduces the real layout the shipped ``PhaseSpaceData`` reads: a scalar
    mesh named ``<species>_<filter>_<ps>`` carrying the ``dV``/``dr_unit``/
    ``sim_unit``/``p_unit``/``p_min``/``p_max``/``movingWindowOffset``/
    ``movingWindowSize``/``_global_start``/``dr`` attributes.  Each iteration
    adds ``iteration`` to a fixed ramp so a reader can tell steps apart.
    ``ext`` may be ``h5`` or ``bp``/``bp5`` (giving an ADIOS2 directory series).

    Returns:
        The openPMD pattern path (with ``%T``).

    """
    np = _numpy()
    opmd = _openpmd()
    _warm_reader("phase_space")
    out = Path(run_dir) / "simOutput" / "phaseSpace"
    out.mkdir(parents=True, exist_ok=True)
    pattern = out / f"PhaseSpace_{species}_{species_filter}_{ps}_%T.{ext}"
    series = opmd.Series(str(pattern), opmd.Access.create)
    # openPMD does not copy the buffer handed to ``store_chunk``; keeping the
    # arrays alive until close (and flushing per step) makes the on-disk data
    # deterministic rather than the contents of freed memory (B1).
    keep: list[object] = []
    for step in iterations:
        record = series.iterations[step].meshes[f"{species}_{species_filter}_{ps}"][opmd.Mesh_Record_Component.SCALAR]
        data = np.array(np.arange(shape[0] * shape[1], dtype=np.float64).reshape(shape) + step)
        keep.append(data)
        record.reset_dataset(opmd.Dataset(data.dtype, data.shape))
        record.store_chunk(data)
        record.unit_SI = 1.0
        for name, value in {
            "dV": 1e-18,
            "dr_unit": 1.0,
            "sim_unit": 1.0,
            "p_unit": 1.0,
            "p_min": -1.0,
            "p_max": 1.0,
            "movingWindowOffset": 0,
            "movingWindowSize": shape[0],
            "_global_start": np.array([0, 0, 0], dtype=np.uint64),
            "dr": 1e-6,
        }.items():
            record.set_attribute(name, value)
        series.flush()
    series.close()
    return pattern


def _adios2_available() -> bool:
    """Whether the pinned ``openpmd_api`` was built with the ADIOS2 backend.

    Returns:
        True when ``.bp``/``.bp5`` series can be written and read.

    """
    import pytest

    if importlib.util.find_spec("openpmd_api") is None:
        pytest.skip("openpmd_api not installed")
    import openpmd_api

    variants = getattr(openpmd_api, "variants", {})
    if not variants.get("adios2", False):
        pytest.skip("openpmd_api was built without the ADIOS2 backend")
    return True


def calorimeter_h5(
    run_dir: Path,
    *,
    species: str = "e",
    species_filter: str = "all",
    iterations: tuple[int, ...] = (0, 50, 100),
    shape: tuple[int, int, int] = (2, 3, 4),
    ext: str = "h5",
) -> Path:
    """Write a ``<species>_calorimeter_<filter>_<iteration>.<ext>`` series.

    Reproduces the shipped ``particleCalorimeter`` layout: a scalar ``calorimeter``
    mesh of shape ``(N_energy, N_pitch, N_yaw)`` with the yaw/pitch/energy
    metadata attributes.  ``ext`` may be ``h5`` (HDF5) or ``bp``/``bp5``
    (ADIOS2, giving a directory series) to exercise both backends.

    Returns:
        The openPMD pattern path (with ``%T``).

    """
    np = _numpy()
    opmd = _openpmd()
    _warm_reader("calorimeter")
    out = Path(run_dir) / "simOutput" / "e_calorimeter"
    out.mkdir(parents=True, exist_ok=True)
    pattern = out / f"{species}_calorimeter_{species_filter}_%T.{ext}"
    series = opmd.Series(str(pattern), opmd.Access.create)
    # See ``phase_space_h5``: pin the written buffers until close and flush per
    # step so the series carries the intended values rather than freed memory.
    keep: list[object] = []
    for step in iterations:
        record = series.iterations[step].meshes["calorimeter"][opmd.Mesh_Record_Component.SCALAR]
        data = np.array(np.arange(shape[0] * shape[1] * shape[2], dtype=np.float64).reshape(shape) + step)
        keep.append(data)
        record.reset_dataset(opmd.Dataset(data.dtype, data.shape))
        record.store_chunk(data)
        record.unit_SI = 1.0
        for name, value in {
            "maxYaw[deg]": 30.0,
            "maxPitch[deg]": 40.0,
            "posYaw[deg]": 0.0,
            "posPitch[deg]": 0.0,
            "minEnergy[keV]": 10.0,
            "maxEnergy[keV]": 1000.0,
            "logScale": False,
        }.items():
            record.set_attribute(name, value)
        series.flush()
    series.close()
    return pattern


def radiation_h5(
    run_dir: Path,
    *,
    species: str = "e",
    iterations: tuple[int, ...] = (0, 50, 100),
    n_directions: int = 2,
    n_frequencies: int = 5,
) -> Path:
    """Write a ``<species>_radAmplitudes_<iteration>_0_0_0.h5`` series.

    Reproduces the shipped ``RadiationData`` layout: an ``Amplitude`` record with
    six components ``{x,y,z}_{Re,Im}`` of shape ``(n_dir, n_freq, 1)`` and a
    ``DetectorFrequency/omega`` record.

    Returns:
        The openPMD pattern path (with ``%T``).

    """
    np = _numpy()
    opmd = _openpmd()
    _warm_reader("radiation")
    out = Path(run_dir) / "simOutput" / "radiationOpenPMD"
    out.mkdir(parents=True, exist_ok=True)
    pattern = out / f"{species}_radAmplitudes_%T_0_0_0.h5"
    series = opmd.Series(str(pattern), opmd.Access.create)
    # See ``phase_space_h5``: pin the written buffers until close and flush per
    # step.  The real part is ``component + step`` so every iteration carries a
    # distinct, assertable spectrum; the imaginary part is zero.
    keep: list[object] = []
    for step in iterations:
        iteration = series.iterations[step]
        amplitude = iteration.meshes["Amplitude"]
        for index, component in enumerate(("x", "y", "z")):
            for part in ("Re", "Im"):
                record = amplitude[f"{component}_{part}"]
                value = float(index + 1 + step) if part == "Re" else 0.0
                data = np.array(np.full((n_directions, n_frequencies, 1), value, dtype=np.float64))
                keep.append(data)
                record.reset_dataset(opmd.Dataset(data.dtype, data.shape))
                record.store_chunk(data)
                record.unit_SI = 1.0
        omega = iteration.meshes["DetectorFrequency"]["omega"]
        values = np.array(
            np.array([1e14 * (i + 1) for i in range(n_frequencies)], dtype=np.float64).reshape(1, n_frequencies, 1),
        )
        keep.append(values)
        omega.reset_dataset(opmd.Dataset(values.dtype, values.shape))
        omega.store_chunk(values)
        omega.unit_SI = 1.0
        series.flush()
    series.close()
    return pattern


def png_file(
    run_dir: Path,
    *,
    species: str = "e",
    axis: str = "yx",
    slice_point: float = 0.5,
    iteration: int = 0,
    height: int = 8,
    width: int = 12,
) -> Path:
    """Write one ``<species>_png_<axis>_<slice_point>_<iteration>.png`` image.

    The PNG plugin names its output directory ``png<SpeciesLongName><AXIS>``;
    only the ``e``→``Electrons`` mapping is shipped.

    Returns:
        The written image's path.

    """
    import imageio

    np = _numpy()
    long_name = {"e": "Electrons"}.get(species, species)
    out = Path(run_dir) / "simOutput" / f"png{long_name}{axis.upper()}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{species}_png_{axis}_{slice_point}_{iteration:06d}.png"
    imageio.imwrite(path, (np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3) + iteration))
    return path


def _numpy() -> object:
    """Import ``numpy`` (already required by the openPMD/PNG readers).

    Returns:
        The ``numpy`` module.

    """
    import pytest

    if importlib.util.find_spec("numpy") is None:
        pytest.skip("numpy not installed")
    import numpy as np

    return np
