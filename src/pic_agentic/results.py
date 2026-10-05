# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Transport-agnostic results engine (M3 contract sections 5 and 8b).

The engine has two deliberately separated layers:

* :func:`scan_output` is a cheap, openPMD-free ``os.scandir`` walk that turns a
  run's linked ``simOutput`` directory into a light :class:`ResultManifest`.  It
  never opens a file, so it can answer ``describe`` and ride along on the
  ``results.ready`` event even on a machine without the optional reader.
* :func:`resolve_result` turns one :class:`ResultParams` request into the exact
  keyword fields of :func:`~pic_agentic.protocol.simulation.build_result_ack`.

``openpmd_api`` (and Pillow for image thumbnails) are optional: the module
imports cleanly when they are absent, and the reader entry points raise
:class:`ResultsUnavailable` so the caller can answer
``error_code="reader_unavailable"`` instead of crashing.  The plugin reader
surface spans the text plugins (``energy_histogram``, ``emittance``,
``transition_radiation``) and the openPMD/image plugins (``phase_space``,
``radiation``, ``calorimeter``, ``png``); each imports its own PIConGPU
submodule lazily, so a reader whose optional dependency is missing degrades only
itself.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import importlib
import importlib.util
import io
import json
import logging
import math
import os
import re
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pic_agentic.protocol.simulation import (
    MAX_RESULT_BYTES,
    RESULT_TEXT_MAX_BYTES,
    SLICE_MAX_POINTS,
    ResultManifest,
    ResultOp,
    ResultParams,
    ResultRef,
)
from pic_agentic.rcp import encode_wire
from pic_agentic.simclient.simulation import SimulationErrorCode, find_stdout_path

if TYPE_CHECKING:
    from collections.abc import Callable

    from pic_agentic.analysis_program import AnalysisProgram

#: Maximum directory depth a scan descends.  PIConGPU output nests a handful of
#: levels at most; a deeper tree is either pathological or a symlink cycle.
_MAX_SCAN_DEPTH = 6

#: Default number of trailing lines a text ``read`` returns.
_DEFAULT_READ_TAIL = 200

#: Filename suffix -> manifest ``format`` for the text files readable without
#: the optional openPMD reader.
_TEXT_SUFFIXES = frozenset({".txt", ".csv", ".log"})

#: Filename suffix -> manifest ``format`` for the openPMD backends.  The pinned
#: ``openpmd_api`` 0.17.1 recognises ``.h5``/``.bp``/``.bp5`` by suffix but
#: rejects ``.hdf5`` outright ("Unknown file format"), so it is not advertised.
_OPENPMD_SUFFIXES = {".bp": "openpmd-adios2", ".bp5": "openpmd-adios2", ".h5": "openpmd-hdf5"}

#: Iteration selectors that mean "the newest available iteration".
_LAST_ITERATIONS = frozenset({None, "last"})

#: Maximal digit run in a series name: the iteration infix PIConGPU writes
#: (``fields_000050.h5``, radiation's ``spec_000050_0_0_0.h5``).  Only the
#: **first** run is the iteration, so the template is derived with ``count=1``.
_ITERATION_INFIX_RE = re.compile(r"\d+")

#: Substring marking a checkpoint series; deprioritised so a coincidental
#: record in a checkpoint is never silently preferred over the field output.
_CHECKPOINT_MARKER = "checkpoint"

#: The sentinel component name openPMD reports for a scalar (componentless)
#: mesh record; it must not be presented as a selectable component.
_SCALAR_COMPONENT = "\x0bScalar"

#: Directory (relative to ``run_dir``) the workflow links the results into.
_SIM_OUTPUT = "simOutput"

#: Preferred [keV] window for the ``count_in_window`` histogram reduction.
#: PIConGPU energy histograms are commonly configured over 0--1000 keV, so this
#: window is kept for a spectrum that lies entirely inside it.  Otherwise the
#: default is derived from the populated range so it always captures the whole
#: population (F2): a spectrum that starts at 2500 keV is covered end to end
#: rather than clipped at 1000 keV.  A caller can override either way with
#: ``min_kev``/``max_kev`` on the request.
_DEFAULT_WINDOW_KEV = (100.0, 1000.0)

#: Fraction of the total counts below which a window is reported as a
#: mis-window by a ``warning``.  The empty case is covered for any fraction;
#: this threshold additionally flags a window that only captures a sliver of a
#: spectrum (e.g. a fixed 100--1000 keV default on a spectrum spanning 900--
#: 20000 keV, F2).  A full-range window never trips it.
_WINDOW_COVERAGE_WARNING = 0.9

#: Target number of array points per plugin summary.  The summary is strided
#: down further if it still exceeds :data:`MAX_RESULT_BYTES`.
_PLUGIN_MAX_POINTS = 256

#: Reader -> the summary arrays that carry *measured values* (as opposed to the
#: coordinate axes such as ``bins_kev``/``y_slices_m``/``omega_per_s``, which are
#: nonzero by construction).  An all-zero value array is a likely vacuous result.
#: The openPMD and image readers have no such scalar (a phase-space plane or a
#: PNG legitimately has zero-valued cells), so they are not annotated.
_VACUOUS_VALUE_KEYS: dict[str, tuple[str, ...]] = {
    "energy_histogram": ("counts",),
    "emittance": ("slice_emit_mrad",),
    # ``intensity`` is the *strided* (brightest-angle subsampled) spectrum, so
    # keying vacuity on it could warn -- or stay silent -- on the subsample
    # while the measured total is nonzero.  The honest scalar is the total.
    "transition_radiation": ("total_intensity",),
}

#: Array-rank constants for the openPMD reader summaries.  The shipped readers
#: return 2D phase-space planes and 2D radiation spectra, and a calorimeter cube
#: that is 2D when no energy binning was configured; naming the ranks keeps the
#: degenerate-shape branches legible.
_NDIM_2D = 2
_NDIM_3D = 3

#: Element kinds the openPMD plugin reader classes live under.  ``"openpmd"``
#: selects the ``%T``-series constructor, ``"image"`` the run-directory PNG
#: reader, and ``"text"`` the delimiter-separated readers.
_KIND_TEXT = "text"
_KIND_OPENPMD = "openpmd"
_KIND_IMAGE = "image"


@dataclasses.dataclass(frozen=True)
class _PluginReader:
    """One registered PIConGPU plugin reader.

    Attributes:
        pattern: The ``simOutput`` filename glob the reader's output matches.
            Named groups (``species``, ``filter``, ``iteration``, ``ps``,
            ``axis``, ``slice_point``) carry the components a reader needs.
        module: The class name inside the reader's PIConGPU submodule.
        submodule: The ``picongpu.extra.plugins.data`` submodule to import.
        kind: ``"text"`` for the delimiter-separated plugins, ``"openpmd"`` for
            the openPMD series plugins and ``"image"`` for the PNG plugin.  The
            kind selects the constructor and summary builder.
        needs: The importable optional module this reader requires; a missing
            module degrades only this reader to ``reader_unavailable``.

    """

    pattern: re.Pattern[str]
    module: str
    submodule: str
    kind: str
    needs: str
    #: ``True`` for a reader the engine parses itself (stdlib) rather than a
    #: shipped ``picongpu.extra.plugins.data`` class.  A native reader has no
    #: ``module``/``submodule``/``needs`` and needs no PIConGPU install: only the
    #: run's plain-text output.  It is still a ``kind="text"`` reader for target
    #: discovery (a regular file, never an openPMD series directory).
    native: bool = False


#: Filename patterns for the shipped readers.  The ``species``/``species_filter``
#: components are arbitrary PIConGPU identifiers, so these patterns cannot be
#: tightened to a known species list.  They are therefore a *heuristic* for the
#: manifest ``format`` only (see :func:`_sniff_format`): a filename that merely
#: resembles plugin output (e.g. ``notes_emittance_x.dat``) may be labelled as
#: one.  The reader remains the authority - it re-resolves the file from the
#: species/filter and returns a clean ``no_results`` when the name does not
#: correspond.
_PLUGIN_READERS: dict[str, _PluginReader] = {
    "energy_histogram": _PluginReader(
        re.compile(r"^(?P<species>[A-Za-z0-9_]+)_energyHistogram_(?P<filter>[A-Za-z0-9_]+)\.dat$"),
        "EnergyHistogramData",
        "energy_histogram",
        _KIND_TEXT,
        "picongpu",
    ),
    # The EnergyFields plugin writers emit exactly ``fields_energy.dat``
    # (``EnergyFields.x.cpp`` hard-codes ``pluginPrefix = "fields_energy"``, so
    # there is no per-run prefix).  Unlike the ``*_energyHistogram_*.dat``
    # family this ships no reader in ``picongpu.extra.plugins.data``; the engine
    # parses it natively (stdlib), so the integrated field-energy artifact is no
    # longer labelled ``binary``/unreadable (H2).  (The ``EnergyParticles``
    # plugin's ``<species>_energy_<filter>.dat`` is a *different* artifact with
    # a species filter and is deliberately not matched.)
    "energy_fields": _PluginReader(
        re.compile(r"^fields_energy\.dat$"),
        "",
        "",
        _KIND_TEXT,
        "picongpu",
        native=True,
    ),
    "emittance": _PluginReader(
        re.compile(r"^(?P<species>[A-Za-z0-9_]+)_emittance_(?P<filter>[A-Za-z0-9_]+)\.dat$"),
        "EmittanceData",
        "emittance",
        _KIND_TEXT,
        "picongpu",
    ),
    "transition_radiation": _PluginReader(
        re.compile(r"^(?P<species>[A-Za-z0-9_]+)_transRad_(?P<iteration>[0-9]+)\.dat$"),
        "TransitionRadiationData",
        "transitionradiation",
        _KIND_TEXT,
        "picongpu",
    ),
    # ``PhaseSpace_<species>_<species_filter>_<ps>_<iteration>.h5`` (openPMD).
    # ``<ps>`` is the 3-char spatial+momentum selection (e.g. ``ypy``).
    "phase_space": _PluginReader(
        re.compile(
            r"^PhaseSpace_(?P<species>[A-Za-z0-9_]+)_(?P<filter>[A-Za-z0-9_]+)_(?P<ps>[A-Za-z0-9]+)_(?P<iteration>[0-9]+)"
            r"\.(?:h5|bp|bp5)$",
        ),
        "PhaseSpaceData",
        "phase_space",
        _KIND_OPENPMD,
        "openpmd_api",
    ),
    # ``<species>_radAmplitudes_<iteration>_0_0_0.h5`` (openPMD).  The trailing
    # ``_0_0_0`` is PIConGPU's radiation-plugin filename suffix.
    "radiation": _PluginReader(
        re.compile(r"^(?P<species>[A-Za-z0-9_]+)_radAmplitudes_(?P<iteration>[0-9]+)(?:_[0-9]+)*\.(?:h5|bp|bp5)$"),
        "RadiationData",
        "radiation",
        _KIND_OPENPMD,
        "openpmd_api",
    ),
    # ``<species>_calorimeter_<filter>_<iteration>.h5`` (openPMD).
    "calorimeter": _PluginReader(
        re.compile(
            r"^(?P<species>[A-Za-z0-9_]+)_calorimeter_(?P<filter>[A-Za-z0-9_]+)_(?P<iteration>[0-9]+)\.(?:h5|bp|bp5)$",
        ),
        "particleCalorimeter",
        "calorimeter",
        _KIND_OPENPMD,
        "openpmd_api",
    ),
    # ``<species>_png_<axis>_<slicePoint>_<iteration>.png``.  The PNG plugin has
    # no ``read`` surface (metadata only); images travel via ``export``.
    "png": _PluginReader(
        re.compile(
            r"^(?P<species>[A-Za-z0-9_]+)_png_(?P<axis>[A-Za-z0-9]+)_(?P<slice_point>[0-9.]+)_(?P<iteration>[0-9]+)\.png$",
        ),
        "PNGData",
        "png",
        _KIND_IMAGE,
        "imageio",
    ),
}


#: The sniffed ``format`` labels that name a *plain-text* plugin artifact.  A
#: ``read`` of such a file serves its text tail (bounded) rather than rejecting
#: it as "not a text result": ``fields_energy.dat`` is plain text with no shipped
#: reader, so before H2 it was unreachable through either the plugin or the text
#: path.
_NATIVE_TEXT_FORMATS = frozenset(reader for reader, spec in _PLUGIN_READERS.items() if spec.kind == _KIND_TEXT)


class ResultsUnavailable(RuntimeError):  # ruff: ignore[error-suffix-on-exception-name] - contract-frozen name
    """Raised when an optional reader is required but not installed."""


class ResultsReaderError(RuntimeError):
    """Raised when the optional reader is present but cannot serve a request."""


log = logging.getLogger(__name__)

#: Exceptions a plugin reader can raise on a malformed-but-parseable file that
#: should degrade to a clean ``no_results`` rather than escape as a generic
#: ``result_failed``.  ``TypeError``/``AttributeError`` cover e.g. ``None``
#: column labels from a corrupted file; the openPMD readers add an ``Exception``
#: catch-all at the call site because ``openpmd_api`` raises plain ``Exception``
#: for a bad series.
_PLUGIN_SOFT_ERRORS = (ResultsReaderError, KeyError, OSError, ValueError, IndexError, TypeError, AttributeError)


def _reader_name() -> str | None:
    """Probe for the optional openPMD reader without importing it.

    Returns:
        ``"openpmd"`` when ``openpmd_api`` is importable, else ``None``.

    """
    if importlib.util.find_spec("openpmd_api") is None:
        return None
    return "openpmd"


def _sniff_format(name: str, *, is_dir: bool = False) -> str:
    """Classify an entry by its filename alone (never opens it).

    The plugin formats are matched by a filename-shape *heuristic* only (the
    species/filter components are arbitrary identifiers); a name that merely
    resembles plugin output may be labelled as one.  This is safe precisely
    because the classification drives no destructive action - ``read`` only
    advertises the reader and the plugin path re-validates the file and returns
    a clean ``no_results`` on a mismatch.

    Args:
        name: The entry's basename.
        is_dir: Whether the entry is a directory.

    Returns:
        One of ``openpmd-adios2``, ``openpmd-hdf5``, a registered plugin reader
        name, ``text``, ``dir`` or ``binary``.

    """
    if is_dir:
        return "dir"
    # The plugin patterns are checked *before* the openPMD suffixes so a
    # plugin series (``PhaseSpace_*_100.h5``, ``*_radAmplitudes_*.h5``,
    # ``*_calorimeter_*_100.h5``) is named for its reader rather than merely as
    # ``openpmd-hdf5``; a plain field series matches no plugin pattern and is
    # still reported by its backend suffix below.
    for reader, spec in _PLUGIN_READERS.items():
        if spec.pattern.fullmatch(name):
            return reader
    suffix = Path(name).suffix.lower()
    if suffix in _OPENPMD_SUFFIXES:
        return _OPENPMD_SUFFIXES[suffix]
    if suffix in _TEXT_SUFFIXES:
        return "text"
    return "binary"


def _mirror_path(local_root: str, sim_id: str, relpath: str) -> Path:
    """Map a run-relative output path into the server's local mirror.

    Args:
        local_root: The configured results mirror root (empty when unset).
        sim_id: The simulation id (the mirror's per-sim directory).
        relpath: The path relative to ``simOutput``.

    Returns:
        The candidate mirror path (not necessarily existing).

    """
    return Path(local_root) / sim_id / _SIM_OUTPUT / relpath


def _collect_entries(root: Path) -> list[tuple[Path, bool]]:
    """Walk ``root`` depth-first within :data:`_MAX_SCAN_DEPTH`.

    The walk uses ``os.scandir`` and skips symlinks (which would otherwise let
    a crafted tree escape the scan root or cycle).  It never opens a file.

    Args:
        root: The scan root.

    Returns:
        ``(path, is_dir)`` pairs for every regular file and directory found.

    """
    found: list[tuple[Path, bool]] = []
    work: list[tuple[Path, int]] = [(root, 0)]
    while work:
        directory, depth = work.pop()
        try:
            with os.scandir(directory) as scan:
                entries = list(scan)
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                is_dir = entry.is_dir(follow_symlinks=False)
                if not is_dir and not entry.is_file(follow_symlinks=False):
                    continue
            except OSError:
                continue
            child = Path(entry.path)
            found.append((child, is_dir))
            if is_dir and depth < _MAX_SCAN_DEPTH:
                work.append((child, depth + 1))
    return found


def _entry_size(path: Path) -> int:
    """Return a file's size, or 0 when it vanished or is unreadable.

    Args:
        path: The file path.

    Returns:
        The size in bytes, or 0 on failure.

    """
    try:
        return path.stat(follow_symlinks=False).st_size
    except OSError:
        return 0


def scan_output(
    output_dir: Path | str,
    *,
    sim_id: str,
    run_dir: str,
    local_root: str = "",
) -> ResultManifest:
    """Build the light manifest of a run's linked output.

    The walk is a recursive, depth-bounded ``os.scandir``: it **never opens a
    file**, so it is safe to run on the submit node and cheap enough to attach
    to an event.  A missing ``output_dir`` yields an empty manifest rather than
    an error.

    Args:
        output_dir: The linked ``simOutput`` directory (may be absent).
        sim_id: The simulation id.
        run_dir: The run directory, recorded verbatim in the manifest.
        local_root: Optional server-side mirror root; when set, an entry is
            marked ``readable`` if ``local_root/<sim_id>/simOutput/<rel>``
            exists.

    Returns:
        The manifest, with per-entry sizes, sniffed formats and reader flag.

    """
    root = Path(output_dir)
    files: list[ResultRef] = []
    total_bytes = 0
    if root.is_dir():
        for path, is_dir in _collect_entries(root):
            try:
                relpath = str(path.relative_to(root))
            except ValueError:  # pragma: no cover - every entry lives under root
                relpath = path.name
            size = 0 if is_dir else _entry_size(path)
            total_bytes += size
            readable = bool(local_root) and _mirror_path(local_root, sim_id, relpath).exists()
            files.append(
                ResultRef(
                    path=relpath,
                    uri=path.resolve().as_uri(),
                    format=_sniff_format(path.name, is_dir=is_dir),
                    size_bytes=size,
                    readable=readable,
                ),
            )
        files.sort(key=lambda ref: ref.path)
    readable_local = bool(local_root) and _mirror_path(local_root, sim_id, "").is_dir()
    return ResultManifest(
        sim_id=sim_id,
        run_dir=run_dir,
        output_dir=str(root) if root.is_dir() else None,
        total_bytes=total_bytes,
        reader=_reader_name(),
        readable_local=readable_local,
        files=files,
    )


def _import_openpmd() -> Any:
    """Import the optional openPMD reader.

    Returns:
        The imported ``openpmd_api`` module.

    Raises:
        ResultsUnavailable: If the module is not installed.

    """
    try:
        import openpmd_api  # ruff: ignore[import-outside-top-level] - optional dependency
    except ImportError as exc:
        msg = "the optional openpmd_api reader is not installed"
        raise ResultsUnavailable(msg) from exc
    return openpmd_api


def _open_series(path: Path) -> Any:
    """Open an openPMD series for reading.

    Args:
        path: The series file or directory.

    Returns:
        An openPMD ``Series``.

    Raises:
        ResultsReaderError: If the series cannot be opened.

    """
    api = _import_openpmd()
    # openPMD >= 0.14 exposes the access modes as attributes on ``Access``
    # (``api.Access.read_only``); older builds used an ``Access_Type`` enum.
    access = getattr(api.Access, "read_only", None) or api.Access_Type.READ_ONLY
    try:
        return api.Series(str(path), access)
    except Exception as exc:
        msg = f"cannot open openPMD series at {path}: {exc}"
        raise ResultsReaderError(msg) from exc


def _series_patterns(path: Path) -> list[Path]:
    """Return openPMD ``%T`` pattern candidates for a concrete series name.

    PIConGPU's iteration infix is a digit run, but its position varies: standard
    field output is ``fields_000050.h5`` (trailing run) while the radiation
    plugin writes ``spec_000050_0_0_0.h5`` (leading run).  Rather than guess, the
    first and last digit runs are offered as candidates (first first, so the
    common trailing-only name is unchanged); :func:`_find_series` keeps whichever
    actually opens.

    Returns:
        Candidate pattern paths (first-run substitution first), deduplicated.

    """
    spans = [match.span() for match in _ITERATION_INFIX_RE.finditer(path.stem)]
    if not spans:
        return [path]
    # Trailing run first (PIConGPU's standard field output), then the leading run
    # (the radiation plugin's ``spec_000050_0_0_0.h5``).  At most two candidates.
    ordered = [(spans[-1])] if len(spans) == 1 else [(spans[-1]), (spans[0])]
    patterns: list[Path] = []
    seen: set[str] = set()
    for start, end in ordered:
        stem = f"{path.stem[:start]}%T{path.stem[end:]}"
        name = stem + path.suffix
        if name not in seen:
            seen.add(name)
            patterns.append(path.with_name(name))
    return patterns


def _is_series_name(path: Path) -> bool:
    """Whether a filename suffix marks an openPMD series.

    Returns:
        True for a ``.bp``/``.bp5``/``.h5`` name or directory.

    """
    return path.suffix.lower() in _OPENPMD_SUFFIXES


def _series_candidates(base: Path) -> list[Path]:
    """Return deterministic openPMD series pattern candidates at or under ``base``.

    Treats an openPMD-suffixed **directory** as a series leaf (ADIOS2 ``.bp``/
    ``.bp5`` series are directories) and does not descend into it.  The result is
    sorted lexically and checkpoint series are moved last, so discovery does not
    depend on ``os.scandir`` order and never silently prefers a checkpoint.

    Args:
        base: A file or directory to search.

    Returns:
        Candidate series patterns (possibly empty).

    """
    if base.is_file() or (base.is_dir() and _is_series_name(base)):
        # A file, or an explicit path naming the series itself (ADIOS2 directory):
        # map it directly instead of walking into it.
        return _series_patterns(base) if _is_series_name(base) else []
    if not base.is_dir():
        return []
    found = [pattern for path in _walk_series(base) for pattern in _series_patterns(path)]
    found.sort(key=lambda path: (_CHECKPOINT_MARKER in path.name.lower(), str(path)))
    return _dedupe(found)


def _walk_series(root: Path) -> list[Path]:
    """Return every openPMD series leaf at or under ``root``.

    Returns:
        The concrete series files/directories (not patterns), unsorted.

    """
    found: list[Path] = []
    work: list[tuple[Path, int]] = [(root, 0)]
    while work:
        directory, depth = work.pop()
        try:
            with os.scandir(directory) as scan:
                entries = sorted(scan, key=lambda entry: entry.name)
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                is_dir = entry.is_dir(follow_symlinks=False)
                if not is_dir and not entry.is_file(follow_symlinks=False):
                    continue
            except OSError:
                continue
            child = Path(entry.path)
            if _is_series_name(child):
                found.append(child)  # a series leaf; never descend into it
            elif is_dir and depth < _MAX_SCAN_DEPTH:
                work.append((child, depth + 1))
    return found


def _dedupe(paths: list[Path]) -> list[Path]:
    """Return ``paths`` without repeats, preserving order.

    Returns:
        The deduplicated list.

    """
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _find_series(output: Path, relpath: str | None, record: str | None = None) -> Path | None:
    """Locate an *openable* openPMD series for a run's ``simOutput``.

    A PICMI diagnostic's own ``result_path(prefix)`` returns an openPMD *pattern*
    with a ``%T`` iteration wildcard (e.g. ``.../openPMD/simOutput/fields_%06T.h5``),
    which is exactly what :meth:`openpmd_api.Series` accepts.  That pattern is not
    persisted to the cluster, so it is reconstructed from the entries on disk
    (:func:`_series_candidates`, :func:`_series_patterns`).  Candidates that do
    not open are skipped, so a mis-derived ``%T`` position cannot mask a good
    series, and tools that merely *list* candidates never open a file.  When
    several series open and ``record`` is given, the one that contains that
    record is preferred, so a request cannot bind to the wrong family (e.g.
    ``particles_*`` for an ``E`` slice).

    Args:
        output: The run's linked ``simOutput`` directory.
        relpath: The request's relative path (a file, a directory, or None); an
            unsafe value makes ``_safe_join`` raise, which the caller maps to
            ``path_unsafe``.
        record: Optional record name used to disambiguate several series.

    Returns:
        The discovered series path, or None when the path is absent or contains
        no openable series.

    """
    base = output
    if relpath:
        base = _safe_join(output, relpath)  # raises ValueError when unsafe
    candidates = _series_candidates(base)
    if not candidates:
        return None
    if len(candidates) == 1:
        # Unambiguous: open it lazily in the reader, so callers that stub
        # ``_load_dataset`` (the evaluator tests) are not forced to build a
        # real series and a listing caller never opens a file.
        return candidates[0]
    openable = [path for path in candidates if _can_open_series(path)]
    if not openable:
        return candidates[0]
    if record:
        return next((path for path in openable if _series_has_record(path, record)), openable[0])
    return openable[0]


def _can_open_series(path: Path) -> bool:
    """Whether ``path`` opens as an openPMD series.

    Returns:
        True when the series opens; a failure is simply "not a candidate".

    """
    try:
        series = _open_series(path)
    except Exception:  # ruff: ignore[blind-except] - a bad candidate is not an error
        return False
    closer = getattr(series, "close", None)
    if callable(closer):
        closer()
    return True


def _series_has_record(path: Path, record: str) -> bool:
    """Whether an openable series exposes ``record`` in any iteration.

    Best-effort and bounded: a probe that cannot be opened simply does not match,
    so a broken candidate never masks a good one.

    Returns:
        True when the record is present.

    """
    try:
        series = _open_series(path)
        for step in series.iterations:
            if record in series.iterations[step].meshes:
                return True
    except Exception:  # ruff: ignore[blind-except] - a probe is best-effort
        return False
    return False


def _select_iteration(series: Any, iteration: int | str | None) -> int:
    """Pick the iteration to read from a series.

    Args:
        series: An open series.
        iteration: ``None``/``"last"`` for the newest, ``"first"`` for the
            oldest, or an explicit step number.

    Returns:
        The selected iteration number.

    Raises:
        ResultsReaderError: If the series has no iterations.

    """
    available = list(series.iterations)
    if not available:
        msg = "openPMD series has no iterations"
        raise ResultsReaderError(msg)
    if iteration in _LAST_ITERATIONS:
        return max(available)
    if iteration == "first":
        return min(available)
    return int(iteration)


def _flatten(data: Any) -> list[float]:
    """Flatten an openPMD chunk into a list of floats.

    Args:
        data: A numpy-like array returned by ``load_chunk``.

    Returns:
        The row-major flattened values.

    """
    flat = data.reshape(-1) if hasattr(data, "reshape") else data
    return [float(value) for value in flat]


def _record_components(mesh: Any) -> list[str]:
    r"""List a mesh record's component names (openPMD-API version safe).

    ``openpmd_api`` >= 0.15 has no ``mesh.components`` attribute: component
    names are the record's iteration keys.  A scalar mesh exposes a single
    sentinel component ``"\x0bScalar"`` which callers should treat as "no
    explicit component".

    Args:
        mesh: An openPMD mesh record.

    Returns:
        The component names (possibly empty).

    """
    try:
        return [str(name) for name in mesh]
    except TypeError:  # pragma: no cover - very old readers expose .components instead
        return [str(name) for name in getattr(mesh, "components", [])]


def _load_dataset(path: Path, record: str | None, component: str | None, iteration: int | str | None) -> list[float]:
    """Load the requested mesh component's raw chunk.

    Args:
        path: The openPMD series path.
        record: Mesh name, or ``None`` for the first available.
        component: Component name, or ``None`` for the first/scalar.
        iteration: Iteration selector.

    Returns:
        The flattened chunk values.

    Raises:
        ResultsReaderError: If the record/component cannot be resolved.

    """
    series = _open_series(path)
    try:
        step = series.iterations[_select_iteration(series, iteration)]
    except (KeyError, IndexError) as exc:
        # openpmd_api raises IndexError for an unknown iteration number and
        # KeyError for some backends; both mean "no such iteration".
        msg = f"iteration {iteration!r} not found in the series"
        raise ResultsReaderError(msg) from exc
    names = list(step.meshes)
    if not names:
        msg = "openPMD iteration has no meshes"
        raise ResultsReaderError(msg)
    name = record or names[0]
    if name not in names:
        msg = f"record {name!r} not found; available: {', '.join(names)}"
        raise ResultsReaderError(msg)
    mesh = step.meshes[name]
    components = [c for c in _record_components(mesh) if c != _SCALAR_COMPONENT]
    selected = component
    if selected is None:
        selected = components[0] if components else None
    if selected is not None and components and selected not in components:
        msg = f"component {selected!r} not found; available: {', '.join(components)}"
        raise ResultsReaderError(msg)
    # ``mesh[name]`` yields a Record_Component; a componentless scalar mesh
    # loads straight from the record.
    dataset = mesh[selected] if selected is not None and components else mesh
    try:
        chunk = dataset.load_chunk()
        series.flush()
    except Exception as exc:
        msg = f"cannot read record {name!r}: {exc}"
        raise ResultsReaderError(msg) from exc
    return _flatten(chunk)


def _dataset_unit_info(
    path: Path,
    record: str | None,
    component: str | None,
    iteration: int | str | None,
) -> dict[str, Any]:
    """Read the resolved mesh component's openPMD unit metadata, best-effort.

    openPMD records expose ``Record_Component.unit_SI`` (a scale factor to SI)
    and ``Mesh.unit_dimension`` (the seven SI base exponents); PIConGPU sets both
    for E/B fields (L4/M3).  The same allow-listed record/component resolution as
    :func:`_load_dataset` is used, so the reported units describe exactly the
    component the program read.

    Returns:
        ``{"unit_SI": float, "unit_dimension": [float, ...]}`` with whichever
        keys the backend exposes; ``{}`` when neither is present.

    Raises:
        ResultsReaderError: If the series/record/component cannot be resolved.

    """
    series = _open_series(path)
    try:
        step = series.iterations[_select_iteration(series, iteration)]
    except (KeyError, IndexError) as exc:
        msg = f"iteration {iteration!r} not found in the series"
        raise ResultsReaderError(msg) from exc
    names = list(step.meshes)
    if not names:
        msg = "openPMD iteration has no meshes"
        raise ResultsReaderError(msg)
    name = record or names[0]
    if name not in names:
        msg = f"record {name!r} not found; available: {', '.join(names)}"
        raise ResultsReaderError(msg)
    mesh = step.meshes[name]
    components = [c for c in _record_components(mesh) if c != _SCALAR_COMPONENT]
    selected = component or (components[0] if components else None)
    if selected is not None and components and selected not in components:
        msg = f"component {selected!r} not found; available: {', '.join(components)}"
        raise ResultsReaderError(msg)
    dataset = mesh[selected] if selected is not None and components else mesh
    info: dict[str, Any] = {}
    unit_si = getattr(dataset, "unit_SI", None)
    if unit_si is not None:
        info["unit_SI"] = float(unit_si)
    dimension = getattr(mesh, "unit_dimension", None)
    if dimension is not None:
        with contextlib.suppress(TypeError, ValueError):  # an odd backend value is simply skipped
            info["unit_dimension"] = [float(exponent) for exponent in dimension]
    return info


def read_slice(
    path: Path | str,
    *,
    record: str | None = None,
    component: str | None = None,
    iteration: int | str | None = None,
    axis: int | None = None,  # ruff: ignore[unused-function-argument] - reserved for slicing
    index: int | None = None,  # ruff: ignore[unused-function-argument] - see axis
    downsample: int | None = None,  # ruff: ignore[unused-function-argument] - applied by the caller
) -> dict[str, Any]:
    """Reduce one openPMD record to a flat numeric slice.

    Args:
        path: The openPMD series path.
        record: Mesh name (defaults to the first).
        component: Component name (defaults to the first).
        iteration: Iteration selector.
        axis: Reserved for axis-aligned slicing.
        index: Reserved for axis-aligned slicing.
        downsample: Reserved; the caller strides the returned data.

    Returns:
        ``{"data": list[float], "n_points": int}``.

    """
    values = _load_dataset(Path(path), record, component, iteration)
    return {"data": values, "n_points": len(values)}


def read_stats(
    path: Path | str,
    *,
    record: str | None = None,
    component: str | None = None,
    iteration: int | str | None = None,
) -> dict[str, Any]:
    """Reduce one openPMD record to scalar statistics.

    Args:
        path: The openPMD series path.
        record: Mesh name (defaults to the first).
        component: Component name (defaults to the first).
        iteration: Iteration selector.

    Returns:
        ``{"stats": {"min","max","mean","n"}, "n_points": int}``.

    """
    values = _load_dataset(Path(path), record, component, iteration)
    if not values:
        return {"stats": {"min": 0.0, "max": 0.0, "mean": 0.0, "n": 0}, "n_points": 0}
    total = sum(values)
    return {
        "stats": {"min": min(values), "max": max(values), "mean": total / len(values), "n": len(values)},
        "n_points": len(values),
    }


def read_image(
    path: Path | str,
    *,
    record: str | None = None,
    component: str | None = None,
    iteration: int | str | None = None,
    max_pixels: int = 512,
) -> dict[str, Any]:
    """Render one openPMD record as a base64 PNG thumbnail.

    Pillow is optional; when it is absent this raises
    :class:`ResultsUnavailable`.

    Args:
        path: The openPMD series path.
        record: Mesh name (defaults to the first).
        component: Component name (defaults to the first).
        iteration: Iteration selector.
        max_pixels: Longest edge of the thumbnail.

    Returns:
        ``{"data": <base64 png>, "data_encoding": "png"}``.

    Raises:
        ResultsUnavailable: If Pillow is absent.

    """
    if importlib.util.find_spec("PIL") is None:
        msg = "Pillow is required for image thumbnails"
        raise ResultsUnavailable(msg)
    values = _load_dataset(Path(path), record, component, iteration)
    return _thumbnail(values, max_pixels)


def _thumbnail(values: list[float], max_pixels: int) -> dict[str, Any]:
    """Encode a flat value list as a square-ish grayscale PNG.

    Args:
        values: The flattened values.
        max_pixels: Longest edge of the thumbnail.

    Returns:
        ``{"data": <base64 png>, "data_encoding": "png"}``.

    """
    from PIL import Image  # ruff: ignore[import-outside-top-level] - optional dependency

    count = len(values)
    size = max(1, math.ceil(math.sqrt(count))) if count else 1
    low = min(values) if values else 0.0
    high = max(values) if values else 1.0
    span = (high - low) or 1.0
    pixels = bytes(int(255 * (value - low) / span) for value in values) or b"\x00"
    padded = pixels[: size * size].ljust(size * size, b"\x00")
    image = Image.frombytes("L", (size, size), padded)
    image.thumbnail((max_pixels, max_pixels))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {"data": base64.b64encode(buffer.getvalue()).decode("ascii"), "data_encoding": "png"}


def read_text_tail(path: Path | str, tail: int = _DEFAULT_READ_TAIL) -> list[str]:
    """Read up to ``tail`` trailing lines of a text file.

    The read window is bounded by :data:`RESULT_TEXT_MAX_BYTES`, so a huge log
    cannot exhaust memory.  This path needs no openPMD reader.

    Args:
        path: The text file.
        tail: Maximum number of trailing lines.

    Returns:
        The trailing lines (possibly empty).

    """
    lines, _total = _tail_text(Path(path), tail)
    return lines


def _tail_text(path: Path, tail: int) -> tuple[list[str], int]:
    """Read a bounded trailing window of a file.

    Args:
        path: The file to read.
        tail: Maximum number of trailing lines.

    Returns:
        ``(lines, total_lines_in_window)``; ``([], 0)`` on failure.

    """
    try:
        size = path.stat().st_size
    except OSError:
        return [], 0
    window = min(RESULT_TEXT_MAX_BYTES, max(tail, 1) * 512 + 1)
    start = max(0, size - window)
    try:
        with path.open("rb") as handle:
            handle.seek(start)
            data = handle.read()
    except OSError:
        return [], 0
    if start > 0:
        newline = data.find(b"\n")
        if newline < 0:
            return [], 0
        data = data[newline + 1 :]
    all_lines = data.decode("utf-8", errors="replace").splitlines()
    return (all_lines[-tail:] if tail else []), len(all_lines)


def _error(code: SimulationErrorCode, message: str) -> dict[str, Any]:
    """Build an ack error field pair.

    Args:
        code: The stable error code.
        message: The human-readable detail.

    Returns:
        ``{"error": message, "error_code": code.value}``.

    """
    return {"error": message, "error_code": code.value}


def _safe_join(base: Path, relpath: str) -> Path:
    """Resolve ``relpath`` under ``base``, refusing escapes and absolute paths.

    Args:
        base: The directory the path must stay inside.
        relpath: A relative path.

    Returns:
        The resolved candidate path.

    Raises:
        ValueError: If the path is absolute or escapes ``base``.

    """
    if not relpath or Path(relpath).is_absolute():
        msg = f"unsafe result path: {relpath!r}"
        raise ValueError(msg)
    base_resolved = base.resolve()
    candidate = (base_resolved / relpath).resolve()
    if candidate != base_resolved and base_resolved not in candidate.parents:
        msg = f"result path escapes {base}: {relpath!r}"
        raise ValueError(msg)
    return candidate


def _escaped_size(value: Any) -> int:
    """Return the JSON-escaped *wire* size of a value.

    The value is passed through :func:`~pic_agentic.rcp.encode_wire` first, so
    the size reflects what actually travels: a float is carried as the larger
    tagged object ``{"$rcp_float": "..."}``, and measuring the pre-encoding form
    would under-count a float-heavy slice and let an over-budget ack reach the
    homeserver (where it is dropped).  Non-JSON scalars are normalised too.

    Args:
        value: The value that will travel inside an ack payload.

    Returns:
        The length in bytes of its ASCII-escaped JSON encoding.

    """
    return len(json.dumps(encode_wire(value), ensure_ascii=True).encode("ascii"))


def _describe(output: Path, *, sim_id: str, run_dir: Path, local_root: str) -> dict[str, Any]:
    """Answer ``describe``: the scandir-only manifest.

    The file list is truncated (``truncated=True``) and the escaped payload is
    size-checked so a directory with many files cannot overflow the homeserver's
    event-size limit.

    Returns:
        ``{"manifest": <manifest dump>}`` or a ``RESULT_TOO_LARGE`` error.

    """
    manifest = scan_output(output, sim_id=sim_id, run_dir=str(run_dir), local_root=local_root)
    dump = manifest.model_dump()
    # Keep the manifest under the wire budget by dropping the (sorted) tail of
    # the file list; the summary fields stay exact.
    while dump["files"] and _escaped_size(dump) > MAX_RESULT_BYTES:
        dump["files"] = dump["files"][: len(dump["files"]) // 2]
        dump["truncated"] = True
    if _escaped_size(dump) > MAX_RESULT_BYTES:
        return _error(SimulationErrorCode.RESULT_TOO_LARGE, "manifest exceeds the wire budget")
    return {"manifest": dump}


def _read_target(params: ResultParams, run_dir: Path, output: Path) -> Path | None:
    """Resolve the file a text ``read`` refers to.

    Prefers the explicit ``path`` under ``simOutput``; falls back to the
    captured ``stdout``/``stderr`` streams when only ``stream`` is set.

    Returns:
        The resolved path, or None when neither a safe path nor a known stream
        applies.

    """
    if params.path is not None:
        try:
            return _safe_join(output, params.path)
        except ValueError:
            return None
    if params.stream == "stderr":
        return run_dir / "stderr"
    if params.stream == "stdout":
        discovered = find_stdout_path(run_dir)
        return Path(discovered) if discovered else run_dir / "stdout"
    return None


def _read(params: ResultParams, *, run_dir: Path, output: Path) -> dict[str, Any]:
    """Answer a text ``read`` request.

    Returns:
        The ack fields (``data``/``data_encoding``/``n_points``) or an error.

    """
    target = _read_target(params, run_dir, output)
    if params.path is not None and target is None:
        return _error(SimulationErrorCode.PATH_UNSAFE, "unsafe or missing result path")
    if target is None or not target.is_file():
        return _error(SimulationErrorCode.NO_RESULTS, "no such result file")
    # The captured stdout/stderr streams have no filename suffix; treat the
    # known capture paths as text so the advertised stream read works.
    from_stream = params.path is None and params.stream in {"stdout", "stderr"}
    if not from_stream and _sniff_format(target.name) not in {"text", *_NATIVE_TEXT_FORMATS}:
        return _error(SimulationErrorCode.READER_UNAVAILABLE, "not a text result; use an openPMD operation")
    if _entry_size(target) > RESULT_TEXT_MAX_BYTES:
        return _error(SimulationErrorCode.RESULT_TOO_LARGE, f"text result exceeds {RESULT_TEXT_MAX_BYTES} bytes")
    tail = params.tail if params.tail is not None else _DEFAULT_READ_TAIL
    lines = read_text_tail(target, max(0, tail))
    if _escaped_size(lines) > MAX_RESULT_BYTES:
        return _error(SimulationErrorCode.RESULT_TOO_LARGE, f"text result exceeds {MAX_RESULT_BYTES} wire bytes")
    return {"data": lines, "data_encoding": "text", "n_points": len(lines)}


def _import_plugin_reader(name: str) -> type:
    """Import one shipped PIConGPU plugin reader class lazily.

    The reader is imported from its own submodule rather than the package
    ``__init__``, so a reader whose optional dependency is missing degrades only
    itself: ``picongpu.extra.plugins.data.__init__`` eagerly imports every
    reader, so importing it would make e.g. ``png`` fail when ``openpmd_api`` is
    absent.  The submodule is imported without executing that package
    ``__init__`` (a stub package is registered under the same name).

    Args:
        name: A registered reader name.

    Returns:
        The reader class from ``picongpu.extra.plugins.data``.

    Raises:
        ResultsUnavailable: If PIConGPU, the reader's optional dependency, or
            the reader class itself is not available.

    """
    spec = _PLUGIN_READERS[name]
    if spec.native:
        # A native reader is parsed by the engine; there is no shipped class to
        # import and no PIConGPU install is required.
        msg = f"{name} is parsed natively and has no shipped reader class"
        raise ResultsUnavailable(msg)
    try:
        importlib.import_module("picongpu.extra.plugins")
    except ImportError as exc:
        msg = "the optional picongpu plugin readers are not installed"
        raise ResultsUnavailable(msg) from exc
    if spec.needs != "picongpu" and importlib.util.find_spec(spec.needs) is None:
        msg = f"the optional {spec.needs} reader is not installed"
        raise ResultsUnavailable(msg)
    module = _import_reader_module(spec)
    try:
        return getattr(module, spec.module)
    except AttributeError as exc:  # pragma: no cover - a PIConGPU version drift
        msg = f"picongpu does not ship the {spec.module} reader"
        raise ResultsUnavailable(msg) from exc


def _import_reader_module(spec: _PluginReader) -> types.ModuleType:
    """Import the module that carries one plugin reader class.

    The normal path imports the ``picongpu.extra.plugins.data`` package, so the
    whole process sees PIConGPU's real package.  When that package (or the
    cached stub left by an earlier degraded import) does not expose the class -
    a *sibling* reader's optional dependency is missing, because the package
    ``__init__`` imports every reader - it falls back to importing the bare
    submodule, which bypasses the package ``__init__`` and lets readers degrade
    independently.

    Returns:
        The package module, or the reader's submodule.

    Raises:
        ResultsUnavailable: If the submodule itself cannot be imported.

    """
    try:
        module = importlib.import_module("picongpu.extra.plugins.data")
    except ImportError:
        module = None
    if module is not None and hasattr(module, spec.module):
        return module
    plugins = sys.modules.get("picongpu.extra.plugins")
    plugins_file = getattr(plugins, "__file__", None)
    data_dir = Path(plugins_file).parent / "data" if plugins_file else Path()
    _load_data_package(str(data_dir))
    try:
        return importlib.import_module(f"picongpu.extra.plugins.data.{spec.submodule}")
    except ImportError as exc:
        msg = f"the optional picongpu {spec.submodule} reader is not installed"
        raise ResultsUnavailable(msg) from exc


def _load_data_package(data_dir: str) -> types.ModuleType:
    """Register (or reuse) a stub ``picongpu.extra.plugins.data`` package.

    A package needs a module object with a ``__path__`` so its submodules import;
    this one deliberately does not run PIConGPU's ``__init__``.

    Returns:
        The package module whose ``__path__`` points at ``data_dir``.

    """
    pkg_name = "picongpu.extra.plugins.data"
    existing = sys.modules.get(pkg_name)
    if isinstance(existing, types.ModuleType):
        return existing
    package = types.ModuleType(pkg_name)
    package.__path__ = [data_dir]
    package.__package__ = pkg_name
    sys.modules[pkg_name] = package
    return package


def _plugin_targets(output: Path, params: ResultParams, spec: _PluginReader) -> list[Path]:
    """Resolve every plugin output file a request refers to.

    A ``path`` narrows the search to exactly that file; otherwise every filename
    matching the reader's pattern (optionally narrowed by ``species``) is
    returned in deterministic name order.  For the openPMD-backed plugins the
    matching entry may be an ADIOS2 series *directory* (``*.bp``/``*.bp5``), so
    directories are accepted there too.

    Returning *all* matches is what lets :func:`probe_vacuity` honour its
    "every present artifact" contract on a multi-species run: a single-species
    read only needs the first entry (:func:`_plugin_target`).

    Returns:
        The resolved files or series directories; empty when none is found or
        the path is unsafe.

    """
    if params.path is not None:
        try:
            return [_safe_join(output, params.path)]
        except ValueError:
            return []
    candidates = [
        path
        for path, is_dir in _collect_entries(output)
        if (not is_dir or spec.kind == _KIND_OPENPMD)
        and spec.pattern.fullmatch(path.name)
        and _plugin_species_matches(path.name, params, spec)
    ]
    candidates.sort(key=lambda path: path.name)
    return candidates


def _plugin_target(output: Path, params: ResultParams, spec: _PluginReader) -> Path | None:
    """Resolve the single plugin output file a request refers to.

    A ``path`` narrows the search when given; otherwise the first filename
    matching the reader's pattern (optionally narrowed by ``species``) is used.
    For the openPMD-backed plugins the matching entry may be an ADIOS2 series
    *directory* (``*.bp``/``*.bp5``), so directories are accepted there too.

    Returns:
        The resolved file or series directory, or None when none is found or the
        path is unsafe.

    """
    targets = _plugin_targets(output, params, spec)
    return targets[0] if targets else None


def _plugin_filename_groups(spec: _PluginReader, name: str) -> dict[str, str] | None:
    """Parse a plugin filename into its named components.

    Returns:
        The named groups (``species``, ``filter``, ``iteration`` and the
        reader-specific ``ps``/``axis``/``slice_point``), or None when the name
        does not match the reader's pattern.

    """
    match = spec.pattern.fullmatch(name)
    return match.groupdict() if match is not None else None


def _plugin_species_matches(name: str, params: ResultParams, spec: _PluginReader) -> bool:
    """Whether a plugin filename belongs to the requested species/filter.

    Returns:
        True when the request names no species/filter, or the filename's
        components match them.

    """
    groups = _plugin_filename_groups(spec, name)
    if groups is None:
        return False
    # A reader whose filename carries no species component (the field-energy
    # monitor) cannot be narrowed by one: a requested species is honour-neutral
    # there rather than turning every match into a miss.
    if "species" in spec.pattern.groupindex and params.species is not None and groups.get("species") != params.species:
        return False
    # An unset filter means PIConGPU's default "all"; only an explicit
    # non-default filter narrows the match.  A reader without a filter component
    # (radiation) matches any requested filter.
    requested = params.species_filter
    filter_component = groups.get("filter")
    return requested in {None, "all"} or filter_component in {None, requested}


def _resolve_plugin_iteration(available: list[int], iteration: int | str | None) -> int:
    """Pick a concrete iteration from a plugin's available steps.

    Returns:
        The selected iteration.

    Raises:
        ResultsReaderError: If no iterations exist or the request is missing.

    """
    if not available:
        msg = "plugin output has no iterations"
        raise ResultsReaderError(msg)
    if iteration in _LAST_ITERATIONS:
        return max(available)
    if iteration == "first":
        return min(available)
    selected = int(iteration)
    if selected not in available:
        msg = f"iteration {selected} is not available; have {sorted(available)}"
        raise ResultsReaderError(msg)
    return selected


def _stride(values: list[float], max_points: int = _PLUGIN_MAX_POINTS) -> tuple[list[float], bool]:
    """Stride a numeric array down to ``max_points``.

    The first and last elements are always kept: a plain ``values[::step]``
    only guarantees the first, and the dropped tail is exactly the high-energy
    overflow bin (histograms) or the last slice, which the summary must not
    silently lose.  Appending the last element at most adds one point.

    Returns:
        ``(values, downsampled)``; striding keeps the endpoints.

    """
    if len(values) <= max_points:
        return values, False
    step = math.ceil(len(values) / max_points)
    strided = values[::step]
    if strided[-1] != values[-1]:
        strided = [*strided, values[-1]]
    return strided, True


def _plugin_iterations(output: Path, spec: _PluginReader, species: str, species_filter: str) -> list[int]:
    """List the iterations a plugin's matching files on disk provide.

    The iteration is read from the reader's filename pattern, so this works for
    every per-iteration plugin that names its step (transition radiation,
    radiation, phase space, calorimeter and PNG).  It is used instead of the
    shipped readers' ``get_iterations`` where that would be unsafe: the
    ``TransitionRadiationData`` reader globs *every* ``*.dat`` and raises on a
    coexisting histogram/emittance file, and ``RadiationData`` has no iteration
    listing at all (its constructor needs a concrete step).

    Returns:
        The sorted, de-duplicated iterations of the matching files.

    """
    iterations: set[int] = set()
    for path, is_dir in _collect_entries(output):
        # An openPMD ADIOS2 series (``*.bp``/``*.bp5``) is itself a directory
        # whose name still carries the iteration, so directories are valid
        # sources for the openPMD readers; other readers only ever write files.
        if is_dir and spec.kind != _KIND_OPENPMD:
            continue
        if "iteration" not in spec.pattern.groupindex:
            continue
        groups = _plugin_filename_groups(spec, path.name)
        if groups is None:
            continue
        if species and groups.get("species") != species:
            continue
        # An unset/default filter means "all"; only an explicit non-default
        # filter narrows the match.
        if species_filter != "all" and groups.get("filter") not in {None, species_filter}:
            continue
        iterations.add(int(groups["iteration"]))
    return sorted(iterations)


def _plugin_available_iterations(
    reader: str,
    spec: _PluginReader,
    output: Path,
    groups: dict[str, str],
) -> list[int]:
    """Determine the iterations a plugin reader can serve.

    Prefers disk enumeration of the reader's matching filenames (robust against
    a reader that cannot list its own steps); falls back to the shipped reader's
    own listing for the text plugins, whose filenames carry no iteration and
    therefore must be asked.

    Returns:
        The sorted available iterations.

    """
    species = groups.get("species", "")
    species_filter = groups.get("filter", "all")
    from_disk = _plugin_iterations(output, spec, species, species_filter)
    if from_disk:
        return from_disk
    if reader in {"energy_histogram", "emittance", "transition_radiation"}:
        instance = _import_plugin_reader(reader)(str(output.parent))
        return [int(step) for step in instance.get_iterations(species, species_filter)]
    return []


def _plugin_summary(
    reader: str,
    instance: Any,
    groups: dict[str, str],
    iteration: int | str | None,
    available: list[int],
    *,
    target: Path,
    window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Call one plugin reader and shape a bounded numeric summary.

    Returns:
        The reader-specific summary dict.

    """
    selected = _resolve_plugin_iteration(available, iteration)
    return _PLUGIN_BUILDERS[reader](instance, groups, selected, target, window=window)


def _openpmd_pattern(spec: _PluginReader, name: str) -> str | None:
    """Replace a concrete filename's iteration with openPMD's ``%T`` wildcard.

    The radiation and calorimeter readers open the *whole* series (so a single
    handle can select any step), unlike ``PhaseSpaceData``/``PNGData`` which
    build the ``%T`` pattern internally.  The wildcard is placed exactly where
    the reader's regex matched the iteration, so a filename with other digit
    runs (radiation's ``_0_0_0`` suffix) is left intact.

    Returns:
        The pattern filename, or None when the name has no iteration group.

    """
    match = spec.pattern.fullmatch(name)
    if match is None or "iteration" not in spec.pattern.groupindex:
        return None
    start, end = match.span("iteration")
    return f"{name[:start]}%T{name[end:]}"


def _build_plugin_instance(
    reader: str,
    spec: _PluginReader,
    output: Path,
    target: Path,
    selected: int,
) -> Any:
    """Construct one plugin reader for a discovered target and selected step.

    Every reader is instantiated against the run directory or an openPMD
    ``%T`` pattern, never a single concrete file: the radiation and calorimeter
    readers open a whole series, so a step other than the one baked into the
    alphabetically first filename can still be read.

    Returns:
        The reader instance.

    """
    reader_class = _import_plugin_reader(reader)
    if spec.kind == _KIND_TEXT:
        return reader_class(str(output.parent))
    if reader in {"phase_space", "png"}:
        # These readers build the ``%T`` pattern (and, for PNG, the per-axis
        # directory) from the run directory themselves.
        return reader_class(str(output.parent))
    # Radiation and calorimeter open the series directly; use the ``%T`` pattern
    # so any selected step can be read, not only the one in the found filename.
    pattern = _openpmd_pattern(spec, target.name)
    series_path = str(target.parent / pattern) if pattern is not None else str(target)
    if reader == "radiation":
        return reader_class(series_path, selected)
    return reader_class(series_path)


def _plugin_result(
    reader: str,
    spec: _PluginReader,
    output: Path,
    params: ResultParams,
    target: Path,
) -> dict[str, Any]:
    """Resolve and run one plugin reader for a discovered target file.

    Returns:
        The reader-specific summary dict.

    Raises:
        ResultsReaderError: If the target filename stops matching its reader.

    """
    groups = _plugin_filename_groups(spec, target.name)
    if groups is None:  # pragma: no cover - guarded by the registry pattern
        msg = f"filename {target.name!r} does not match the {reader} reader"
        raise ResultsReaderError(msg)
    # ``iteration=None`` must mean "latest" for every reader, so never fall back
    # to the iteration baked into the (alphabetically first) filename:
    # ``_resolve_plugin_iteration`` maps ``None``/``"last"`` to the maximum.
    available = _plugin_available_iterations(reader, spec, output, groups)
    selected = _resolve_plugin_iteration(available, params.iteration)
    instance = _build_plugin_instance(reader, spec, output, target, selected)
    requested = None if params.min_kev is None or params.max_kev is None else (params.min_kev, params.max_kev)
    summary = _plugin_summary(reader, instance, groups, selected, available, target=target, window=requested)
    return _annotate_vacuous(reader, summary)


def _as_nested(data: Any) -> Any:
    """Return ``data`` as plain Python lists (numpy arrays become nested lists).

    The openPMD/PNG readers return numpy arrays, but the default server venv has
    no numpy (it is optional with openpmd_api).  Reading the arrays through
    ``tolist`` keeps the summaries pure-Python and lets the offline stub tests
    drive them with lists.

    Returns:
        A nested ``list`` when ``data`` exposes ``tolist``; ``data`` unchanged
        otherwise.

    """
    if hasattr(data, "tolist"):
        return data.tolist()
    return data


def _nested_shape(value: Any) -> tuple[int, ...]:
    """Return the shape of a (possibly ragged) nested list.

    Returns:
        The leading dimensions, stopping at the first non-sequence.

    """
    shape: list[int] = []
    current = value
    while isinstance(current, (list, tuple)):
        shape.append(len(current))
        current = current[0] if current else None
    return tuple(shape)


def _nested_sum(value: Any) -> float:
    """Sum every leaf of a nested list.

    Returns:
        The total.

    """
    if isinstance(value, (list, tuple)):
        return sum(_nested_sum(item) for item in value)
    return float(value)


def _nested_max(value: Any) -> float:
    """Return the maximum leaf of a nested list.

    Returns:
        The maximum.

    """
    if isinstance(value, (list, tuple)):
        return max((_nested_max(item) for item in value), default=0.0)
    return float(value)


def _phase_space_reduce(plane: list[list[float]]) -> tuple[int, int, list[float], list[float], int, int, float]:
    """Project a 2D plane onto its axes and locate the peak bin.

    Returns:
        ``(n_r, n_p, projection_r, projection_p, peak_r, peak_p, max_count)``.

    """
    n_r, n_p = _nested_shape(plane)
    projected_r = [sum(float(value) for value in row) for row in plane]
    projected_p = [sum(float(plane[row][column]) for row in range(n_r)) for column in range(n_p)]
    max_count = 0.0
    peak_r = peak_p = 0
    for row in range(n_r):
        for column in range(n_p):
            value = float(plane[row][column])
            if value > max_count:
                max_count, peak_r, peak_p = value, row, column
    return n_r, n_p, projected_r, projected_p, peak_r, peak_p, max_count


def _build_phase_space(
    instance: Any,
    groups: dict[str, str],
    iteration: int,
    target: Path,
    *,
    window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Reduce a ``PhaseSpaceData`` histogram to a bounded summary.

    The full 2D phase-space plane is far too large for the wire, so the summary
    carries the histogram's axis ranges, its projections onto each axis (the
    marginal distributions, strided to ``_PLUGIN_MAX_POINTS``) and the peak
    location.  That answers "how many particles, over what ranges, peaked
    where" without shipping the plane.

    ``PhaseSpaceData.get`` defaults to the HDF5 suffix (``h5``) and would miss
    an ADIOS2 series, so the concrete suffix of the resolved target is passed
    through explicitly (M1).

    Returns:
        Axis ranges, strided projections and scalars.

    """
    _ = window
    plane, meta = instance.get(
        iteration=iteration,
        species=groups["species"],
        species_filter=groups["filter"],
        ps=groups["ps"],
        file_ext=target.suffix.lstrip("."),
    )
    plane = _as_nested(plane)
    n_r, n_p, projected_r, projected_p, peak_r, peak_p, max_count = _phase_space_reduce(plane)
    strided_r, downsampled = _stride(projected_r)
    strided_p = _stride(projected_p)[0]
    r_edges = [float(value) for value in meta.r_edges]
    p_edges = [float(value) for value in meta.p_edges]
    return {
        "species": groups["species"],
        "species_filter": groups["filter"],
        "ps": groups["ps"],
        "n_r": n_r,
        "n_p": n_p,
        "r_min_m": r_edges[0] if r_edges else None,
        "r_max_m": r_edges[-1] if r_edges else None,
        "p_min": p_edges[0] if p_edges else None,
        "p_max": p_edges[-1] if p_edges else None,
        "projection_r": strided_r,
        "projection_p": strided_p,
        "total_count": _nested_sum(plane),
        "max_count": max_count,
        "max_r_m": r_edges[peak_r] if peak_r < len(r_edges) else None,
        "max_p": p_edges[peak_p] if peak_p < len(p_edges) else None,
        **_plugin_source(target, iteration),
        "downsampled": downsampled,
    }


def _build_radiation(
    instance: Any,
    groups: dict[str, str],
    iteration: int,
    target: Path,
    *,
    window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Reduce a ``RadiationData`` series to a bounded summary.

    Returns:
        Frequency (SI 1/s), the direction-summed spectrum and scalars.

    """
    _ = groups, window
    spectra = _as_nested(instance.get_Spectra())
    omegas = [float(value) for value in instance.get_omega()]
    n_directions, n_frequencies = _nested_shape(spectra)
    # Sum the (n_directions, n_frequencies) spectra over the observation
    # directions so the summary is the total spectrum.
    total_spectrum = [
        sum(float(spectra[direction][frequency]) for direction in range(n_directions))
        for frequency in range(n_frequencies)
    ]
    peak = max(range(len(total_spectrum)), key=total_spectrum.__getitem__) if total_spectrum else 0
    strided_omega, downsampled = _stride(omegas)
    strided_spectrum, _ = _stride(total_spectrum)
    return {
        "n_directions": n_directions,
        "n_frequencies": n_frequencies,
        "omega_per_s": strided_omega,
        "spectrum": strided_spectrum,
        "total_energy_J": _nested_sum(spectra),
        "peak_spectrum_Js": max(total_spectrum) if total_spectrum else None,
        "peak_omega_per_s": omegas[peak] if peak < len(omegas) else None,
        **_plugin_source(target, iteration),
        "downsampled": downsampled,
    }


def _calorimeter_projections(
    instance: Any,
    iteration: int,
) -> tuple[list[float] | None, list[float], list[float], float, float | None]:
    """Project a calorimeter cube onto its energy, pitch and yaw axes.

    The reader returns ``(n_energy, n_pitch, n_yaw)`` with energy binning or
    ``(n_pitch, n_yaw)`` without; both are handled by one projection.

    Returns:
        ``(energy_keV or None, per_pitch_J, per_yaw_J, total_J, max_cell_J)``.

    """
    energy = instance.getEnergy()
    data = _as_nested(instance.getData(iteration))
    shape = _nested_shape(data)
    if len(shape) == _NDIM_2D:
        n_pitch, n_yaw = shape
        per_pitch = [sum(float(value) for value in data[pitch]) for pitch in range(n_pitch)]
        per_yaw = [sum(float(data[pitch][yaw]) for pitch in range(n_pitch)) for yaw in range(n_yaw)]
        energy_kev = None
    else:
        n_energy, n_pitch, n_yaw = shape
        per_pitch = [
            sum(float(data[energy_index][pitch][yaw]) for energy_index in range(n_energy) for yaw in range(n_yaw))
            for pitch in range(n_pitch)
        ]
        per_yaw = [
            sum(float(data[energy_index][pitch][yaw]) for energy_index in range(n_energy) for pitch in range(n_pitch))
            for yaw in range(n_yaw)
        ]
        energy_kev = [float(value) for value in energy] if energy is not None else None
    total = _nested_sum(data)
    peak = _nested_max(data) if total else None
    return energy_kev, per_pitch, per_yaw, total, peak


def _build_calorimeter(
    instance: Any,
    groups: dict[str, str],
    iteration: int,
    target: Path,
    *,
    window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Reduce a ``particleCalorimeter`` result to a bounded summary.

    Returns:
        Bin counts, energy edges, the yaw/pitch marginals and scalars.

    """
    _ = groups, window
    energy_kev, per_pitch, per_yaw, total, peak = _calorimeter_projections(instance, iteration)
    strided_pitch, downsampled = _stride(per_pitch)
    strided_yaw, _ = _stride(per_yaw)
    return {
        "n_pitch": int(instance.detector_params["N_pitch"]),
        "n_yaw": int(instance.detector_params["N_yaw"]),
        "n_energy": instance.detector_params["N_energy"],
        "energy_keV": energy_kev,
        "per_pitch_J": strided_pitch,
        "per_yaw_J": strided_yaw,
        "total_energy_J": total,
        "max_energy_J": peak,
        **_plugin_source(target, iteration),
        "downsampled": downsampled,
    }


def _build_png(
    instance: Any,
    groups: dict[str, str],
    iteration: int,
    target: Path,
    *,
    window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Describe a ``PNGData`` image without shipping pixels.

    The wire budget cannot carry a full image, and the openPMD thumbnail path
    already exists for mesh data, so the PNG reader returns metadata only
    (dimensions, iteration, path).  The image itself is fetched through
    ``export``; the summary says so via ``image_via="export"``.

    Returns:
        Image dimensions, selector and the relative path.

    """
    _ = window
    species, axis, slice_point = groups["species"], groups["axis"], float(groups["slice_point"])
    image = _as_nested(
        instance.get(
            iteration=iteration,
            species=species,
            species_filter=groups.get("filter", "all"),
            axis=axis,
            slice_point=slice_point,
        ),
    )
    shape = _nested_shape(image)
    height, width = (*shape, 0, 0)[:_NDIM_2D]
    channels = shape[_NDIM_2D] if len(shape) == _NDIM_3D else 1
    return {
        "species": species,
        "axis": axis,
        "slice_point": slice_point,
        "width_px": int(width),
        "height_px": int(height),
        "channels": int(channels),
        "path": target.name,
        "source_size_bytes": _entry_size(target),
        "image_via": "export",
        "iteration": iteration,
        "downsampled": False,
    }


def _plugin_source(target: Path, iteration: int) -> dict[str, Any]:
    """Describe the file a plugin summary was read from.

    A caller that sees suspicious numbers (e.g. all-zero counts) can sanity
    check the named file's size and iteration instead of guessing which output
    was read.

    Returns:
        ``{"source_path", "source_size_bytes", "iteration"}``.

    """
    return {
        "source_path": target.name,
        "source_size_bytes": _entry_size(target),
        "iteration": iteration,
    }


def _has_nonzero_leaf(value: Any) -> bool:
    """Whether a nested numeric structure holds any nonzero value.

    Returns:
        True when at least one leaf is not zero.

    """
    if isinstance(value, (list, tuple)):
        return any(_has_nonzero_leaf(item) for item in value)
    try:
        return math.fabs(float(value)) > 0.0
    except (TypeError, ValueError):
        return False


def _annotate_vacuous(reader: str, summary: dict[str, Any]) -> dict[str, Any]:
    """Add a ``warning`` when a numeric summary is entirely zero.

    The reviewer of beta run 3 could not tell an empty (physically real) result
    from a reader/units bug because the summary was silently all zeros.  An
    all-zero ``energy_histogram`` is a legitimate outcome - e.g. the documented
    focal example configures no plasma species, so no electrons are ionised -
    so the fix is transparency, not an error: the summary carries an explicit
    warning that the diagnostic may be empty or misconfigured.

    Returns:
        ``summary`` with a ``warning`` added when it is all zeros.

    """
    keys = _VACUOUS_VALUE_KEYS.get(reader)
    if not keys:
        return summary
    values = [summary[key] for key in keys if key in summary]
    if not values or any(_has_nonzero_leaf(value) for value in values):
        return summary
    return {
        **summary,
        "warning": (
            f"{reader} is all zeros; the run may have no particles in range or the diagnostic may be misconfigured"
        ),
    }


def _histogram_min_energy_kev(target: Path) -> float:
    """Read the histogram's first lower edge (``minEnergy``) from the file header.

    The shipped ``EnergyHistogramData`` returns only the **upper** edges, so the
    first bin's lower bound is unrecoverable from the reader alone.  PIConGPU's
    ``BinEnergyParticles`` writes it as the first number inside the header's
    ``#step <minEnergy ... >maxEnergy count`` bracket
    (``BinEnergyParticles.x.cpp``).  ``minEnergy`` is a supported non-zero
    configuration, so hard-coding ``0.0`` misplaces every edge and undercounts
    the exact ``[minEnergy, first_upper)`` window; the header is the authority.

    Returns:
        The parsed lower bound [keV], or ``0.0`` when the header is absent or
        unparsable (the historical default).

    """
    try:
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            first = handle.readline()
    except OSError:  # pragma: no cover - target existence is checked earlier
        return 0.0
    match = re.match(r"\s*#step\s*<\s*(\S+)", first)
    if match is None:
        return 0.0
    try:
        return float(match.group(1))
    except ValueError:
        return 0.0


def _histogram_lower_edges(upper_edges: list[float], *, first_lower: float = 0.0) -> list[float]:
    """Reconstruct a histogram's per-bin lower edges from its reported upper edges.

    The shipped ``EnergyHistogramData`` returns one **upper** edge per count
    (``bins[i]`` bounds the bin whose count is ``counts[i]``); the lower edge of
    bin ``i`` is the previous reported edge.  The first bin's lower bound is the
    histogram's configured ``minEnergy``, read from the file header (B1); it is
    not necessarily 0 keV.  Bins are therefore half-open ``[lower_i, upper_i)``,
    which is what makes the ``count_in_window`` comparison exact at the window's
    upper edge (L1: a bin ending at 5 MeV must not be counted in a ">= 5 MeV"
    window).

    Args:
        upper_edges: The reader's reported upper edges [keV].
        first_lower: The first bin's lower edge [keV] (``minEnergy``); defaults
            to ``0.0`` for a header-less source.

    Returns:
        The lower edge per bin, same length as ``upper_edges``.

    """
    return [first_lower, *upper_edges[:-1]]


def _default_window(
    populated_lower: list[float],
    populated_upper: list[float],
) -> tuple[float, float]:
    """Pick the ``count_in_window`` window when the request leaves it unset.

    The preferred 100--1000 keV window is kept only when the whole populated
    range ``[min(populated_lower), max(populated_upper))`` lies inside it, so
    the reported number never changes for the common case.  A spectrum that
    reaches outside it (an LWFA spectrum starting at 2500 keV, or one spanning
    0--20000 keV) is instead covered end to end, so the default captures the
    whole population rather than clipping it to a sliver (F2).  The window is
    always at least as wide as one bin (max > min), so it is never the
    degenerate ``max == min`` that ``ResultParams`` would reject if requested.

    Returns:
        ``(min_kev, max_kev)`` for the summary's ``count_in_window``.

    """
    low, high = _DEFAULT_WINDOW_KEV
    if not populated_lower or (min(populated_lower) >= low and max(populated_upper) <= high):
        return low, high
    span_low, span_high = min(populated_lower), max(populated_upper)
    if span_low >= span_high:
        return span_low, span_low + 1.0
    return span_low, span_high


def _energy_window(
    requested: tuple[float, float] | None,
    populated_lower: list[float],
    populated_upper: list[float],
) -> tuple[float, float]:
    """Return the explicit request window or a derived, non-empty default.

    Returns:
        The ``(min_kev, max_kev)`` window to count in.

    """
    if requested is not None:
        return requested
    return _default_window(populated_lower, populated_upper)


def probe_vacuity(sim_id: str, *, run_dir: Path | str) -> str | None:
    """Return the all-zero warning when a run's only numeric artifact is empty.

    A beta-4 campaign reached three ``done`` leaves, no failures and no alerts,
    yet produced *zero* electrons; the only thing that caught it was a reviewer
    reading an all-zero ``energy_histogram`` by hand.  The autonomy loop needs
    to be suspicious of "successful runs with nothing in them", so this probe
    reuses :func:`_annotate_vacuous` - the exact all-zero warning path - to
    classify a completed run's linked output.

    Each numeric text reader in :data:`_VACUOUS_VALUE_KEYS` is read against the
    run's ``simOutput`` (best-effort; a missing reader or absent artifact is
    skipped) and **every** matching artifact of that reader is examined, so the
    verdict holds on multi-species runs (e.g. ``a_energyHistogram_all.dat`` and
    ``b_energyHistogram_all.dat``).  The run is *suspect* only when at least one
    numeric artifact is present and every present one carries the all-zero
    warning: a single non-empty diagnostic clears it, and a run with no numeric
    artifact at all cannot be judged (e.g. the optional reader is not installed)
    and is not flagged.

    Known limitation (deliberately narrow): only the numeric *text* readers are
    probed.  A run whose energy histogram is all-zero while its (non-numeric)
    phase space or radiation output is populated is still flagged, because the
    vacuity signal only ever looked at scalar value arrays; treat the flag as
    "no particles in the numeric diagnostics", not "no particles at all".

    Args:
        sim_id: The simulation id.
        run_dir: The run directory (``simOutput`` lives under it).

    Returns:
        The all-zero warning text when the run is suspect, else None.

    """
    warning: str | None = None
    saw_artifact = False
    output = Path(run_dir) / _SIM_OUTPUT
    if not output.is_dir():
        return None
    for reader in _VACUOUS_VALUE_KEYS:
        spec = _PLUGIN_READERS[reader]
        params = ResultParams(sim_id=sim_id, op=ResultOp.PLUGIN, reader=reader, iteration="last")
        try:
            targets = _plugin_targets(output, params, spec)
        except Exception:  # ruff: ignore[blind-except] - a probe must never break the caller
            log.debug("vacuity probe enumeration for %r failed", reader)
            continue
        for target in targets:
            try:
                payload = _read_plugin_target(reader, spec, output, params, target)
            except Exception:  # ruff: ignore[blind-except] - a probe must never break the caller
                log.debug("vacuity probe for %r on %r failed", reader, target.name)
                continue
            result = payload.get("result")
            if not isinstance(result, dict):
                # Reader unavailable or artifact unreadable: this artifact
                # cannot be judged, so it neither clears nor flags the run.
                continue
            saw_artifact = True
            if "warning" not in result:
                return None
            warning = str(result["warning"])
    return warning if saw_artifact else None


def _build_energy_histogram(  # ruff: ignore[too-many-locals] - one linear reduction
    instance: Any,
    groups: dict[str, str],
    iteration: int,
    target: Path,
    *,
    window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Reduce an ``EnergyHistogramData`` result to a bounded summary.

    ``window`` is the caller's requested ``(min_kev, max_kev)`` or ``None`` to
    derive a non-empty default from the populated bins (F2).

    The window is **half-open** ``[min_kev, max_kev)`` and a bin is counted when
    its *lower* edge lies in that interval: bins are ``[lower_i, upper_i)``, so
    a bin whose upper edge is exactly ``max_kev`` is not counted (L1: a bin
    ending at 5 MeV must not satisfy a ">= 5 MeV" window, which instead starts
    at the 5 MeV bin's lower edge).  The first bin's lower edge is the file's
    configured ``minEnergy`` (read from the header, B1), not necessarily 0.
    ``min_energy_kev``/``max_energy_kev`` are the populated lower/upper edges,
    i.e. the half-open span ``[min_energy_kev, max_energy_kev)`` covers exactly
    the populated bins.

    Returns:
        Bins (keV), counts, the count in the window and scalars.

    """
    species, species_filter = groups["species"], groups["filter"]
    counts, bins, _iteration, _dt = instance.get(
        iteration=iteration,
        species=species,
        species_filter=species_filter,
    )
    counts = [float(value) for value in counts]
    bins = [float(value) for value in bins]
    # ``EnergyHistogramData`` returns one upper-edge per count; a header/data
    # column mismatch can still leave the arrays ragged, so pair them
    # positionally up to the common length.  This avoids the old
    # ``zip(strict=True)`` crash on such a parse.
    paired = min(len(bins), len(counts))
    upper_edges = bins[:paired]
    lower_edges = _histogram_lower_edges(upper_edges, first_lower=_histogram_min_energy_kev(target))
    counts = counts[:paired]
    # ``max_energy_kev`` is the upper edge of the highest populated bin, not the
    # modal (argmax-count) edge: the high-energy tail is the number the caller
    # is after.  The matching lower edges make the half-open span explicit.
    populated_bins = [index for index, count in enumerate(counts) if count > 0]
    populated_lower = [lower_edges[index] for index in populated_bins]
    populated_upper = [upper_edges[index] for index in populated_bins]
    low, high = _energy_window(window, populated_lower, populated_upper)
    in_window = sum(count for lower, count in zip(lower_edges, counts, strict=True) if low <= lower < high)
    strided_bins, downsampled = _stride(upper_edges)
    strided_counts, _ = _stride(counts)
    total = sum(counts)
    summary: dict[str, Any] = {
        "bins_kev": strided_bins,
        "counts": strided_counts,
        "count_in_window": {"min_kev": low, "max_kev": high, "count": in_window},
        "total": total,
        # The number of populated bins and their edges make a partial window
        # self-evidently a mis-window: ``n_nonzero_bins``/``min``/``max`` next
        # to a ``count_in_window`` far below ``total`` shows the window missed
        # the data (F2).
        "n_nonzero_bins": len(populated_bins),
        "min_energy_kev": min(populated_lower) if populated_lower else None,
        "max_energy_kev": max(populated_upper) if populated_upper else None,
        **_plugin_source(target, iteration),
        "downsampled": downsampled,
    }
    if populated_bins and total > 0 and in_window < _WINDOW_COVERAGE_WARNING * total:
        summary["warning"] = (
            f"energy_histogram has {len(populated_bins)} populated bins "
            f"({min(populated_lower):g}-{max(populated_upper):g} keV, half-open) but count_in_window "
            f"is {in_window:g} of {total:g} for the [{low:g}, {high:g}) keV window; the window does "
            f"not cover the populated range"
        )
    return summary


def _build_emittance(
    instance: Any,
    groups: dict[str, str],
    iteration: int,
    target: Path,
    *,
    window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Reduce an ``EmittanceData`` result to a bounded summary.

    Returns:
        Slice positions (m), slice emittances (m rad) and scalars.

    """
    _ = window
    species, species_filter = groups["species"], groups["filter"]
    raw, y_slices, _iteration, _dt = instance.get(
        iteration=iteration,
        species=species,
        species_filter=species_filter,
    )
    # ``EmittanceData`` returns ``[emit_all, *slice_emit]``: the first value is
    # the total (unsliced) emittance and the rest are the per-slice values, so
    # the total must be split off before the slices are aligned with
    # ``y_slices`` (which has exactly one fewer entry).
    raw = [float(value) for value in raw]
    y_slices = [float(value) for value in y_slices]
    total_emit = raw[0] if raw else None
    slice_emit = raw[1:]
    peak = max(range(len(slice_emit)), key=slice_emit.__getitem__) if slice_emit else 0
    strided_y, downsampled = _stride(y_slices)
    strided_emit, _ = _stride(slice_emit)
    return {
        "y_slices_m": strided_y,
        "slice_emit_mrad": strided_emit,
        "total_emit_mrad": total_emit,
        "max_emit_mrad": max(slice_emit) if slice_emit else None,
        "max_y_slice_m": y_slices[peak] if peak < len(y_slices) else None,
        **_plugin_source(target, iteration),
        "downsampled": downsampled,
    }


def _build_transition_radiation(
    instance: Any,
    groups: dict[str, str],
    iteration: int,
    target: Path,
    *,
    window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Reduce a ``TransitionRadiationData`` result to a bounded summary.

    The reader's ``spectrum`` view (the brightest angles) is a bounded 1D
    spectrum, which suits a wire summary better than the full 3D cube.

    Returns:
        Frequency (SI 1/s), intensity and scalars.

    """
    _ = window
    species = groups["species"]
    omegas, spectrum = instance.get(
        iteration=iteration,
        species=species,
        type="spectrum",
        theta=None,
        phi=None,
        omega=None,
    )
    omegas = [float(value) for value in omegas]
    spectrum = [float(value) for value in spectrum]
    peak = max(range(len(spectrum)), key=spectrum.__getitem__) if spectrum else 0
    strided_omega, downsampled = _stride(omegas)
    strided_spectrum, _ = _stride(spectrum)
    return {
        "omega_per_s": strided_omega,
        "intensity": strided_spectrum,
        "total_intensity": sum(spectrum),
        "peak_intensity": max(spectrum) if spectrum else None,
        "peak_omega_per_s": omegas[peak] if omegas else None,
        **_plugin_source(target, iteration),
        "downsampled": downsampled,
    }


#: Canonical column names of the ``EnergyFields`` plugin output, used when the
#: ``#step total[Joule] Bx[Joule] ...`` header is missing or unrecognised.  The
#: C++ writer emits exactly these (``EnergyFields.x.cpp``).
_FIELD_ENERGY_COLUMNS = ("total", "Bx", "By", "Bz", "Ex", "Ey", "Ez")

#: Byte cap on a native text plugin file the engine parses itself.  The
#: ``fields_energy.dat`` history is one short row per output step; 8 MiB bounds a
#: pathological file without truncating any realistic run.
_NATIVE_TEXT_MAX_BYTES = 8 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class _FieldEnergyRow:
    """One ``fields_energy.dat`` row: a step and its field energies [Joule]."""

    step: int
    total: float
    components: list[float]


def _field_energy_header(line: str) -> list[str]:
    """Parse the ``#step total[Joule] Bx[Joule] ...`` header into column names.

    The bracket unit suffix is stripped, so ``Bx[Joule]`` becomes ``Bx``.  The
    leading ``#step`` token is dropped.  A header whose token count does not
    match :data:`_FIELD_ENERGY_COLUMNS` is rejected (the caller falls back to
    the canonical names), so a corrupted header cannot mislabel the columns.

    Returns:
        The six component column names, or ``[]`` when the header is unusable.

    """
    tokens = line.lstrip("#").split()
    if not tokens or tokens[0] != "step":
        return []
    names = [re.sub(r"\[[^\]]*\]$", "", token) for token in tokens[1:]]
    if len(names) != len(_FIELD_ENERGY_COLUMNS) or names[0] != "total":
        return []
    return names[1:]


def _parse_fields_energy(text: str) -> tuple[list[_FieldEnergyRow], list[str], bool]:
    """Parse an ``EnergyFields`` plain-text artifact.

    The format (pinned ``EnergyFields.x.cpp``) is a ``#``-prefixed header
    ``#step total[Joule] Bx[Joule] By[Joule] Bz[Joule] Ex[Joule] Ey[Joule]
    Ez[Joule]`` followed by one whitespace-separated row per output step:
    ``<step> <total[J]> <Bx> <By> <Bz> <Ex> <Ey> <Ez>``, each ``std::scientific``
    with ~17 significant digits.  The parser is tolerant of a missing header
    (falls back to :data:`_FIELD_ENERGY_COLUMNS`) and skips malformed rows rather
    than failing the whole read.

    Args:
        text: The decoded file text.

    Returns:
        ``(rows, component_names, truncated)``.  ``truncated`` is True when a
        row was dropped as malformed, so the summary can say so.

    """
    rows: list[_FieldEnergyRow] = []
    components = list(_FIELD_ENERGY_COLUMNS[1:])
    seen_header = False
    truncated = False
    # A data row is ``<step> <total> <one value per component>``.
    expected = len(components) + 2
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if not seen_header:
                parsed = _field_energy_header(line)
                if parsed:
                    components = parsed
                    expected = len(parsed) + 2
                seen_header = True
            continue
        if not seen_header:
            # A missing/corrupt header still parses; the space after the comment
            # is optional, so fall back to the canonical names and full width.
            seen_header = True
        tokens = line.split()
        if len(tokens) < expected:
            truncated = True
            continue
        try:
            step = int(float(tokens[0]))
            values = [float(token) for token in tokens[1:expected]]
        except ValueError:
            truncated = True
            continue
        rows.append(_FieldEnergyRow(step=step, total=values[0], components=values[1:]))
    return rows, components, truncated


def _build_energy_fields(
    instance: Any,
    groups: dict[str, str],
    iteration: int,
    target: Path,
    *,
    window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Reduce an ``EnergyFields`` (``fields_energy.dat``) history to a summary.

    The file is the integrated electromagnetic field energy versus step - the
    natural convergence artifact for a field-energy study - but it ships no
    ``picongpu.extra.plugins.data`` reader, so it is parsed natively.  The
    summary carries the total-energy trajectory (strided) plus scalar min/max/last
    values for the total and for each field component, so the trend is visible
    without shipping the whole history.

    Returns:
        The bounded summary dict.

    """
    _ = instance, groups, window
    rows, components, truncated = _read_fields_energy(target)
    return _summarize_energy_fields(rows, components, iteration, target, truncated=truncated)


def _read_fields_energy(target: Path) -> tuple[list[_FieldEnergyRow], list[str], bool]:
    """Read a native text plugin file under a byte cap and parse it once.

    The read is bounded by :data:`_NATIVE_TEXT_MAX_BYTES`, so a pathologically
    large ``fields_energy.dat`` cannot be pulled wholly into memory despite the
    cap the comment documents (m1); a chunky read stops just past the cap and
    the parser then sees only the prefix (excess rows are dropped as malformed,
    and ``truncated`` says so).

    Returns:
        ``(rows, component_names, truncated)`` from :func:`_parse_fields_energy`.

    Raises:
        ResultsReaderError: If the file cannot be read.

    """
    try:
        with target.open("rb") as handle:
            raw = handle.read(_NATIVE_TEXT_MAX_BYTES + 1)
    except OSError as exc:  # pragma: no cover - target existence is checked earlier
        msg = f"cannot read {target.name}: {exc}"
        raise ResultsReaderError(msg) from exc
    text = raw.decode("utf-8", errors="replace")
    return _parse_fields_energy(text)


def _summarize_energy_fields(
    rows: list[_FieldEnergyRow],
    components: list[str],
    iteration: int,
    target: Path,
    *,
    truncated: bool,
) -> dict[str, Any]:
    """Shape already-parsed ``EnergyFields`` rows into the bounded summary.

    Split from :func:`_build_energy_fields` so the native plugin path can parse
    the file once and reuse the rows for both the step resolution and the
    summary (m3).

    Returns:
        The bounded summary dict.

    Raises:
        ResultsReaderError: If there are no parsable rows.

    """
    if not rows:
        msg = f"{target.name} has no parsable field-energy rows"
        raise ResultsReaderError(msg)
    steps = [row.step for row in rows]
    totals = [row.total for row in rows]
    index = steps.index(iteration) if iteration in steps else len(rows) - 1
    component_values = [[row.components[position] for row in rows] for position in range(len(components))]
    strided_steps, downsampled = _stride([float(step) for step in steps])
    strided_totals, _ = _stride(totals)
    return {
        "step": [int(step) for step in strided_steps],
        "total_J": strided_totals,
        "component_names": list(components),
        "component_last_J": {name: values[index] for name, values in zip(components, component_values, strict=True)},
        "component_min_J": {name: min(values) for name, values in zip(components, component_values, strict=True)},
        "component_max_J": {name: max(values) for name, values in zip(components, component_values, strict=True)},
        "selected_step": steps[index],
        "step_first": steps[0],
        "step_last": steps[-1],
        "n_steps": len(rows),
        "total_J_min": min(totals),
        "total_J_max": max(totals),
        "total_J_last": totals[-1],
        # The ``iteration`` selector picks one row; ``total_J_last`` is always
        # the file's *latest* step, so a selected row also carries its own total
        # (M1).
        "total_J_selected": totals[index],
        "units_J": "Joule",
        **_plugin_source(target, steps[index]),
        "truncated": truncated,
        "downsampled": downsampled,
    }


def _native_plugin_result(
    reader: str,
    output: Path,
    params: ResultParams,
    target: Path,
) -> dict[str, Any]:
    """Run a native (engine-parsed) text plugin reader against one file.

    Unlike the shipped-reader path this needs no PIConGPU install and no
    per-iteration filenames: the step lives *inside* the file.  The iteration
    selector therefore picks a row of the file's own history (``last`` by
    default), and an unknown selector is a clean ``no_results``.

    Returns:
        The bounded summary dict.

    Raises:
        ResultsReaderError: If the file cannot be parsed or the iteration is
            not present.

    """
    _ = output
    rows, components, truncated = _read_fields_energy(target)
    available = [row.step for row in rows]
    if not available:
        msg = f"{target.name} has no parsable field-energy rows"
        raise ResultsReaderError(msg)
    selected = _resolve_plugin_iteration(available, params.iteration)
    summary = _summarize_energy_fields(rows, components, selected, target, truncated=truncated)
    return _annotate_vacuous(reader, summary)


#: Reader name -> the summary builder that calls the reader instance.  Every
#: builder shares the signature ``(instance, groups, iteration, target, *,
#: window)`` so the openPMD/image readers can reach their extra selector
#: components; only the energy-histogram builder uses ``window``.
_PLUGIN_BUILDERS: dict[str, Callable[..., dict[str, Any]]] = {
    "energy_histogram": _build_energy_histogram,
    "energy_fields": _build_energy_fields,
    "emittance": _build_emittance,
    "transition_radiation": _build_transition_radiation,
    "phase_space": _build_phase_space,
    "radiation": _build_radiation,
    "calorimeter": _build_calorimeter,
    "png": _build_png,
}


def _bound_plugin(summary: dict[str, Any]) -> dict[str, Any] | None:
    """Stride a plugin summary's arrays until it fits the wire budget.

    Args:
        summary: The reader-specific summary (arrays + scalars).

    Returns:
        The bounded summary, or None when even the scalars overflow the budget.

    """
    if _escaped_size(summary) <= MAX_RESULT_BYTES:
        return summary
    bounded = dict(summary)
    bounded["downsampled"] = True
    while _escaped_size(bounded) > MAX_RESULT_BYTES:
        arrays = [key for key, value in bounded.items() if isinstance(value, list) and len(value) > 1]
        if not arrays:
            return None
        for key in arrays:
            values = bounded[key][::2]
            if values[-1] != bounded[key][-1]:
                values = [*values, bounded[key][-1]]
            bounded[key] = values
    return bounded


def _read_plugin_target(
    reader: str,
    spec: _PluginReader,
    output: Path,
    params: ResultParams,
    target: Path,
) -> dict[str, Any]:
    """Run one reader against one resolved target file or series.

    This is the single I/O site shared by the ``PLUGIN`` request path and the
    vacuity probe, so a multi-artifact probe reads every present artifact with
    exactly the semantics a direct read would use.

    Returns:
        ``{"result": <summary>}`` or a clean error pair.

    """
    if not (target.is_file() or (spec.kind == _KIND_OPENPMD and target.is_dir())):
        return _error(SimulationErrorCode.NO_RESULTS, "no such plugin result file")
    try:
        if spec.native:
            summary = _native_plugin_result(reader, output, params, target)
        else:
            summary = _plugin_result(reader, spec, output, params, target)
    except ResultsUnavailable as exc:
        return _error(SimulationErrorCode.READER_UNAVAILABLE, str(exc))
    except _PLUGIN_SOFT_ERRORS as exc:
        log.debug("plugin reader %r failed: %s", reader, exc)
        return _error(SimulationErrorCode.NO_RESULTS, str(exc))
    except Exception as exc:  # ruff: ignore[blind-except] - a reader crash is ack data, never a 500
        log.debug("plugin reader %r crashed: %s", reader, exc)
        return _error(SimulationErrorCode.NO_RESULTS, str(exc))
    bounded = _bound_plugin(summary)
    if bounded is None:
        return _error(SimulationErrorCode.RESULT_TOO_LARGE, f"plugin summary exceeds {MAX_RESULT_BYTES} wire bytes")
    return {"result": bounded}


def _plugin(
    params: ResultParams,
    *,
    output: Path,
) -> dict[str, Any]:
    """Answer a ``PLUGIN`` request by running a shipped PIConGPU reader.

    The reader runs in-process on the cluster (the caller wraps this in
    :func:`asyncio.to_thread`); its raw output is reduced to a bounded numeric
    summary that fits :data:`MAX_RESULT_BYTES`.

    Returns:
        ``{"result": <summary>}`` or a clean error pair.

    """
    reader = params.reader
    if reader is None or reader not in _PLUGIN_READERS:
        return _error(SimulationErrorCode.UNSUPPORTED, f"unknown plugin reader {reader!r}")
    spec = _PLUGIN_READERS[reader]
    # The radiation reader's filename carries no species-filter component, so a
    # requested non-default filter cannot be honoured; reject it cleanly rather
    # than silently returning the "all" series (m3).
    if reader == "radiation" and params.species_filter not in {None, "all"}:
        return _error(
            SimulationErrorCode.UNSUPPORTED,
            "the radiation reader does not support species_filter; its output has no filter component",
        )
    if not output.is_dir():
        return _error(SimulationErrorCode.NO_RESULTS, "run has no linked simOutput directory")
    target = _plugin_target(output, params, spec)
    if target is None:
        return _error(SimulationErrorCode.NO_RESULTS, "no such plugin result file")
    return _read_plugin_target(reader, spec, output, params, target)


def _dispatch_reader(params: ResultParams, target: Path) -> dict[str, Any]:
    """Call the reader entry point for a ``slice``/``stats``/``image`` op.

    Args:
        params: The result request.
        target: The resolved series path.

    Returns:
        The raw reader result.

    """
    if params.op is ResultOp.SLICE:
        return read_slice(
            target,
            record=params.record,
            component=params.component,
            iteration=params.iteration,
            axis=params.axis,
            index=params.index,
            downsample=params.downsample,
        )
    if params.op is ResultOp.STATS:
        return read_stats(
            target,
            record=params.record,
            component=params.component,
            iteration=params.iteration,
        )
    return read_image(
        target,
        record=params.record,
        component=params.component,
        iteration=params.iteration,
    )


def _compute(params: ResultParams, *, output: Path) -> dict[str, Any]:  # ruff: ignore[too-many-return-statements,complex-structure] - one return per clean error
    """Answer a ``COMPUTE`` request by evaluating a validated analysis program.

    Each ``var`` selector in the program is resolved to one openPMD mesh
    component through the same allow-listed record/component/iteration logic as
    the existing reads (no paths, no code), then the program is evaluated with
    stdlib math.  The result is capped to the same 48 KiB wire budget.

    Returns:
        The ack fields (``result``/``n_points``/``result_kind`` or ``data``), or
        a clean error when the reader, output or program is unusable.

    """
    from pic_agentic.analysis_eval import (  # ruff: ignore[import-outside-top-level] - optional seam
        ProgramError,
        evaluate,
    )
    from pic_agentic.analysis_program import AnalysisProgram  # ruff: ignore[import-outside-top-level] - optional seam

    if params.program is None:
        return _error(SimulationErrorCode.UNSUPPORTED, "compute requires a program")
    try:
        program = AnalysisProgram.model_validate(params.program)
    except Exception as exc:  # ruff: ignore[blind-except] - an invalid program is ack data
        return _error(SimulationErrorCode.UNSUPPORTED, f"invalid analysis program: {exc}")

    preflight = _compute_preflight(params, program, output)
    if preflight is not None:
        return preflight

    units_by_selector: dict[str, dict[str, Any]] = {}

    def resolve(selector: Any) -> list[float]:
        # A selector resolves to one mesh component; the reader applies the
        # same validation the slice path uses.  A selector that omits
        # record/component falls back to the request's top-level ones, and an
        # empty record means the first available mesh.  Series discovery is
        # lazy: a constant/pure program that reads no data must not require the
        # run to have openPMD output.
        record = getattr(selector, "record", None) or params.record
        component = getattr(selector, "component", None) or params.component
        iteration = getattr(selector, "iteration", None)
        if iteration is None:
            iteration = params.iteration
        series = _find_series(output, params.path, record)
        if series is None:
            msg = "no openPMD output found for this run"
            raise ResultsReaderError(msg)
        values = _load_dataset(series, record, component, iteration)
        # The same resolution is reused to read the component's unit metadata.
        # This is best-effort: a backend that exposes none, or a stubbed reader
        # used in tests, simply yields no keys and the note falls back to the
        # internal/normalized convention (M3).
        try:
            info = _dataset_unit_info(series, record, component, iteration)
        except (ResultsUnavailable, ResultsReaderError, KeyError, OSError, ValueError):
            info = {}
        if info:
            units_by_selector[selector.name] = info
        return values

    try:
        payload = evaluate(program, resolve)
    except ResultsUnavailable as exc:
        return _error(SimulationErrorCode.READER_UNAVAILABLE, str(exc))
    except ProgramError as exc:
        return _error(SimulationErrorCode.UNSUPPORTED, str(exc))
    except (ResultsReaderError, KeyError, OSError, ValueError) as exc:
        return _error(SimulationErrorCode.NO_RESULTS, str(exc))
    return _shape_compute(payload, program=program, params=params, units_by_selector=units_by_selector)


def _compute_preflight(params: ResultParams, program: AnalysisProgram, output: Path) -> dict[str, Any] | None:
    """Return a compute error for a request that cannot proceed, else None.

    The output directory is only required when the program actually reads data
    (it declares ``selectors``); a constant/pure program evaluates on any run.

    Args:
        params: The validated result request.
        program: The validated analysis program.
        output: The run's linked ``simOutput`` directory.

    Returns:
        An error pair, or None when the request is ready to evaluate.

    """
    if _reader_name() is None:
        return _error(SimulationErrorCode.READER_UNAVAILABLE, "the optional openpmd_api reader is not installed")
    if params.program is None:
        return _error(SimulationErrorCode.UNSUPPORTED, "compute requires a program")
    if program.selectors and not output.is_dir():
        return _error(SimulationErrorCode.NO_RESULTS, "run has no linked simOutput directory")
    return None


def _referenced_selector_names(program: Any) -> list[str]:
    """Return every selector name a program references, in first-seen order.

    Mirrors :func:`pic_agentic.analysis_eval._selector_names` for the validated
    node objects (``output`` plus any ``points``), so a *bare* ``var`` that is
    not listed in ``selectors`` is still echoed with its unit note rather than
    being mistaken for a program that reads no data.

    Returns:
        The referenced selector names (possibly empty).

    """
    names: list[str] = []
    roots = [program.output, *([program.points] if program.points is not None else [])]
    stack = list(roots)
    while stack:
        current = stack.pop()
        if getattr(current, "kind", None) == "var" and current.name not in names:
            names.append(current.name)
        for field in ("left", "right", "operand"):
            child = getattr(current, field, None)
            if child is not None:
                stack.append(child)
    return names


def _compute_units(
    program: Any,
    params: ResultParams,
    units_by_selector: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Describe the units of a compute result truthfully (L4/M3).

    openPMD records *do* expose unit metadata - ``Record_Component.unit_SI`` (a
    scale factor to SI) and ``Mesh.unit_dimension`` - which the reader can reach,
    so the note must not claim otherwise.  The selector echo therefore carries
    each resolved component's ``unit_SI``/``unit_dimension`` where the file
    records them, and the note points at those fields rather than asserting a
    physical unit for the *derived* result (which is the program's arithmetic,
    not any single component's raw unit).  Only when no selector has any unit
    metadata does the note fall back to the internal/normalized convention.

    Returns:
        A ``{"unit_note", "selectors"}`` fragment.

    """
    units_by_selector = units_by_selector or {}
    referenced = _referenced_selector_names(program)
    declared = {selector.name: selector for selector in program.selectors}
    # Echo the declared selectors in declaration order first (the caller's own
    # listing), then any bare ``var`` the program references without declaring.
    ordered: list[tuple[str, Any]] = [
        (selector.name, selector) for selector in program.selectors if selector.name in referenced
    ]
    ordered += [(name, None) for name in referenced if name not in declared]
    selectors = [
        {
            "name": name,
            "record": (selector.record if selector is not None else None) or params.record,
            "component": (selector.component if selector is not None else None) or params.component,
            "iteration": (
                selector.iteration if selector is not None and selector.iteration is not None else params.iteration
            ),
            **units_by_selector.get(name, {}),
        }
        for name, selector in ordered
    ]
    if not selectors:
        # A constant/pure program reads no mesh data.
        note = "unitless: the program reads no mesh data (a constant or pure expression)"
    elif any("unit_SI" in selector or "unit_dimension" in selector for selector in selectors):
        note = (
            "result is computed from the raw openPMD mesh components named in `selectors`: the values are "
            "in each component's internal units, and `unit_SI` (the scale factor to SI) and `unit_dimension` "
            "(the seven SI base exponents) are echoed per selector where the file records them; the derived "
            "result's unit follows the program's arithmetic, so it is not asserted here"
        )
    else:
        note = (
            "result is in PIConGPU internal (normalized) code units: the file records no unit metadata "
            "for the mesh components named in `selectors`, so no physical unit is asserted"
        )
    return {"unit_note": note, "selectors": selectors}


def _shape_compute(
    payload: dict[str, Any],
    *,
    program: Any = None,
    params: ResultParams | None = None,
    units_by_selector: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Shape an evaluator payload into the frozen result-ack fields.

    The ack carries a fixed key set, so the numeric array travels as ``data``
    (``data_encoding="float"``), a scalar as ``stats["value"]``, and the
    evaluator's metadata as ``result``.  When the program is supplied the
    result metadata is augmented with a truthful ``unit_note`` and the selector
    echo (L4/M3).

    Returns:
        The shaped ack fields, or a ``RESULT_TOO_LARGE`` error.

    """
    result = payload.get("result")
    metadata: dict[str, Any] = {"result_kind": payload.get("result_kind", "scalar")}
    if "points" in payload:
        metadata["points"] = payload["points"]
    if program is not None and params is not None:
        metadata.update(_compute_units(program, params, units_by_selector))
    if isinstance(result, list):
        if len(result) > SLICE_MAX_POINTS:
            result = result[:SLICE_MAX_POINTS]
        shaped: dict[str, Any] = {
            "result": metadata,
            "data": result,
            "data_encoding": "float",
            "n_points": len(result),
        }
    else:
        shaped = {
            "result": metadata,
            "stats": {"value": float(result) if result is not None else 0.0},
            "n_points": 1,
        }
    if _escaped_size(shaped) > MAX_RESULT_BYTES:
        return _error(SimulationErrorCode.RESULT_TOO_LARGE, f"result exceeds {MAX_RESULT_BYTES} wire bytes")
    return shaped


def _reader_op(  # ruff: ignore[too-many-return-statements]
    params: ResultParams,
    *,
    output: Path,
) -> dict[str, Any]:
    """Answer a ``slice``/``stats``/``image`` request via the optional reader.

    Returns:
        The ack fields, or a clean error when the reader or output is missing.

    """
    if _reader_name() is None:
        return _error(SimulationErrorCode.READER_UNAVAILABLE, "the optional openpmd_api reader is not installed")
    if params.op is ResultOp.SLICE and (params.index is not None or (params.axis not in {None, 0})):
        # The reader currently returns a flattened chunk; refuse a spatial
        # selection it cannot honour rather than silently returning all data.
        return _error(
            SimulationErrorCode.UNSUPPORTED,
            "axis/index selection is not supported yet; omit them or use downsample",
        )
    if not output.is_dir():
        return _error(SimulationErrorCode.NO_RESULTS, "run has no linked simOutput directory")
    try:
        target = _find_series(output, params.path, params.record)
    except ValueError:
        return _error(SimulationErrorCode.PATH_UNSAFE, "unsafe result path")
    if target is None:
        return _error(SimulationErrorCode.NO_RESULTS, "no openPMD output found for this run")
    try:
        result = _dispatch_reader(params, target)
    except ResultsUnavailable as exc:
        return _error(SimulationErrorCode.READER_UNAVAILABLE, str(exc))
    except (ResultsReaderError, KeyError, OSError, ValueError) as exc:
        return _error(SimulationErrorCode.NO_RESULTS, str(exc))
    if params.op is ResultOp.SLICE:
        return _cap_slice(result, params.downsample)
    if _escaped_size(result) > MAX_RESULT_BYTES:
        return _error(SimulationErrorCode.RESULT_TOO_LARGE, f"result exceeds {MAX_RESULT_BYTES} wire bytes")
    return result


def _cap_slice(result: dict[str, Any], downsample: int | None) -> dict[str, Any]:
    """Stride, hard-cap and size-check a slice result.

    The slice must fit :data:`MAX_RESULT_BYTES` *on the wire*, where each float
    costs the larger tagged encoding (see
    :func:`~pic_agentic.rcp.encode_wire`), so a ``SLICE_MAX_POINTS``-point slice
    of floats does not fit and is strided down until it does.  Striding (rather
    than truncating) keeps the returned points spread across the whole range.

    Args:
        result: ``{"data": list[float], "n_points": int}`` from the reader.
        downsample: Optional stride (values ``<= 1`` are ignored).

    Returns:
        The bounded slice, or a ``RESULT_TOO_LARGE`` error only when even a
        single point overflows.

    """
    data = list(result.get("data", []))
    if downsample is not None and downsample > 1:
        data = data[::downsample]
    if len(data) > SLICE_MAX_POINTS:
        data = data[:SLICE_MAX_POINTS]
    while len(data) > 1 and _escaped_size({"data": data, "n_points": len(data)}) > MAX_RESULT_BYTES:
        data = data[::2]
    capped = {"data": data, "n_points": len(data)}
    if _escaped_size(capped) > MAX_RESULT_BYTES:
        return _error(SimulationErrorCode.RESULT_TOO_LARGE, f"slice exceeds {MAX_RESULT_BYTES} wire bytes")
    return capped


def _export(output: Path, *, sim_id: str, run_dir: Path, local_root: str) -> dict[str, Any]:
    """Answer ``export`` with a ticket (no data is ever moved).

    Returns:
        ``{"result": {"ref", "transfer", "resolved", "local_path"}}``.

    """
    manifest = scan_output(output, sim_id=sim_id, run_dir=str(run_dir), local_root=local_root)
    mirror = _mirror_path(local_root, sim_id, "")
    resolved = bool(local_root) and mirror.is_dir()
    destination = mirror if local_root else Path("<destination>") / sim_id / _SIM_OUTPUT
    transfer = f"rsync -a {output.resolve()}/ {destination}/"
    ticket = {
        "ref": manifest.files[0].model_dump() if manifest.files else None,
        "transfer": transfer,
        "resolved": resolved,
        "local_path": str(mirror) if resolved else None,
    }
    if _escaped_size(ticket) > MAX_RESULT_BYTES:
        return _error(SimulationErrorCode.RESULT_TOO_LARGE, "export ticket exceeds the wire budget")
    return {"result": ticket}


def resolve_result(  # ruff: ignore[too-many-return-statements] - one dispatch per op
    params: ResultParams,
    *,
    run_dir: Path | str,
    sim_id: str,
    local_root: str = "",
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Return the ack payload fields for one result request.

    The returned keys are exactly the optional
    :func:`~pic_agentic.protocol.simulation.build_result_ack` kwargs:
    ``manifest``, ``result``, ``data``, ``data_encoding``, ``n_points``,
    ``stats``, ``error``, ``error_code``.  A bad-but-valid request yields an
    ``error``/``error_code`` pair rather than an exception; only truly
    unexpected I/O may raise.

    Args:
        params: The validated result request.
        run_dir: The run directory (``simOutput`` lives under it).
        sim_id: The simulation id.
        local_root: Optional server-side results mirror root.
        output_dir: Optional explicit output directory, bypassing the
            ``run_dir/simOutput`` convention.  A caller holding the linked
            output directory directly (rather than a run that names it
            ``simOutput``) passes it here.

    Returns:
        The ack fields for this request.

    """
    run = Path(run_dir)
    output = Path(output_dir) if output_dir is not None else run / _SIM_OUTPUT
    if params.op is ResultOp.DESCRIBE:
        return _describe(output, sim_id=sim_id, run_dir=run, local_root=local_root)
    if params.op is ResultOp.READ:
        return _read(params, run_dir=run, output=output)
    if params.op is ResultOp.EXPORT:
        return _export(output, sim_id=sim_id, run_dir=run, local_root=local_root)
    if params.op is ResultOp.ANALYZE:
        # ANALYZE is handled by the simclient's analysis engine, not here; a
        # direct caller must not silently fall through to the openPMD reader.
        return _error(SimulationErrorCode.UNSUPPORTED, "analyze is served by the analysis engine, not the reader")
    if params.op is ResultOp.COMPUTE:
        return _compute(params, output=output)
    if params.op is ResultOp.PLUGIN:
        return _plugin(params, output=output)
    return _reader_op(params, output=output)


__all__ = [
    "ResultsReaderError",
    "ResultsUnavailable",
    "probe_vacuity",
    "read_image",
    "read_slice",
    "read_stats",
    "read_text_tail",
    "resolve_result",
    "scan_output",
]
