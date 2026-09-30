# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Reproducible fixtures for the PIConGPU text-plugin readers.

The energy-histogram header format is chosen to round-trip through the shipped
``picongpu.extra.plugins.data.EnergyHistogramData`` reader: the reader splits
the header on single spaces and needs the ``>`` marker and ``count`` word to be
*two* space-separated tokens, so the C++ writer's literal ``">"`` is emitted
with a space before it.  The counts line therefore has ``num_bins + 2`` values
(underflow, the real bins, overflow) plus the sum, matching ``realNumBins``.
"""

from __future__ import annotations

from pathlib import Path

#: Default time step the synthetic ``simOutput/output`` records.
DEFAULT_DT_SI = 1.39e-16


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
) -> Path:
    """Write a ``<species>_energyHistogram_<filter>.dat`` the real reader parses.

    Every iteration puts ``peak_count`` into ``peak_bin`` and one count into the
    first and last bin, so the default ``count_in_window`` is non-trivial.

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
