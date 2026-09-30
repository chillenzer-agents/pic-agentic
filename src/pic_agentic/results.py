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
``error_code="reader_unavailable"`` instead of crashing.
"""

from __future__ import annotations

import base64
import dataclasses
import importlib.util
import io
import json
import math
import os
import re
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

#: Filename suffix -> manifest ``format`` for the openPMD backends.
_OPENPMD_SUFFIXES = {".bp": "openpmd-adios2", ".bp5": "openpmd-adios2", ".h5": "openpmd-hdf5", ".hdf5": "openpmd-hdf5"}

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

#: Default [keV] window for the ``count_in_window`` histogram reduction.  PIConGPU
#: energy histograms are commonly configured over 0--1000 keV; the window is
#: always reported in the summary, so the caller need not know the default.
_DEFAULT_WINDOW_KEV = (100.0, 1000.0)

#: Target number of array points per plugin summary.  The summary is strided
#: down further if it still exceeds :data:`MAX_RESULT_BYTES`.
_PLUGIN_MAX_POINTS = 256


@dataclasses.dataclass(frozen=True)
class _PluginReader:
    """One registered PIConGPU text-plugin reader.

    Attributes:
        pattern: The ``simOutput`` filename glob the reader's output matches.
        module: The ``picongpu.extra.plugins.data`` class name to import.

    """

    pattern: re.Pattern[str]
    module: str


#: Registry of shipped text-plugin readers, keyed by the frozen wire name.  The
#: filename patterns mirror the PIConGPU readers: energy histogram and emittance
#: are per-species/file-name text files; transition radiation is per iteration.
_PLUGIN_READERS: dict[str, _PluginReader] = {
    "energy_histogram": _PluginReader(
        re.compile(r"^[A-Za-z0-9_]+_energyHistogram_[A-Za-z0-9_]+\.dat$"),
        "EnergyHistogramData",
    ),
    "emittance": _PluginReader(
        re.compile(r"^[A-Za-z0-9_]+_emittance_[A-Za-z0-9_]+\.dat$"),
        "EmittanceData",
    ),
    "transition_radiation": _PluginReader(
        re.compile(r"^[A-Za-z0-9_]+_transRad_[0-9]+\.dat$"),
        "TransitionRadiationData",
    ),
}


class ResultsUnavailable(RuntimeError):  # ruff: ignore[error-suffix-on-exception-name] - contract-frozen name
    """Raised when an optional reader is required but not installed."""


class ResultsReaderError(RuntimeError):
    """Raised when the optional reader is present but cannot serve a request."""


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

    Args:
        name: The entry's basename.
        is_dir: Whether the entry is a directory.

    Returns:
        One of ``openpmd-adios2``, ``openpmd-hdf5``, a registered plugin reader
        name, ``text``, ``dir`` or ``binary``.

    """
    if is_dir:
        return "dir"
    suffix = Path(name).suffix.lower()
    if suffix in _OPENPMD_SUFFIXES:
        return _OPENPMD_SUFFIXES[suffix]
    for reader, spec in _PLUGIN_READERS.items():
        if spec.pattern.fullmatch(name):
            return reader
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
        True for a ``.bp``/``.bp5``/``.h5``/``.hdf5`` name or directory.

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
    if not from_stream and _sniff_format(target.name) != "text":
        return _error(SimulationErrorCode.READER_UNAVAILABLE, "not a text result; use an openPMD operation")
    if _entry_size(target) > RESULT_TEXT_MAX_BYTES:
        return _error(SimulationErrorCode.RESULT_TOO_LARGE, f"text result exceeds {RESULT_TEXT_MAX_BYTES} bytes")
    tail = params.tail if params.tail is not None else _DEFAULT_READ_TAIL
    lines = read_text_tail(target, max(0, tail))
    if _escaped_size(lines) > MAX_RESULT_BYTES:
        return _error(SimulationErrorCode.RESULT_TOO_LARGE, f"text result exceeds {MAX_RESULT_BYTES} wire bytes")
    return {"data": lines, "data_encoding": "text", "n_points": len(lines)}


def _import_plugin_reader(name: str) -> type:
    """Import one shipped PIConGPU text-plugin reader class.

    Args:
        name: A registered reader name.

    Returns:
        The reader class from ``picongpu.extra.plugins.data``.

    Raises:
        ResultsUnavailable: If PIConGPU (with its readers) is not installed.

    """
    import importlib  # ruff: ignore[import-outside-top-level] - optional dependency

    spec = _PLUGIN_READERS[name]
    try:
        module = importlib.import_module("picongpu.extra.plugins.data")
    except ImportError as exc:
        msg = "the optional picongpu plugin readers are not installed"
        raise ResultsUnavailable(msg) from exc
    try:
        return getattr(module, spec.module)
    except AttributeError as exc:  # pragma: no cover - a PIConGPU version drift
        msg = f"picongpu does not ship the {spec.module} reader"
        raise ResultsUnavailable(msg) from exc


def _plugin_target(output: Path, params: ResultParams, spec: _PluginReader) -> Path | None:
    """Resolve the plugin output file a request refers to.

    A ``path`` narrows the search when given; otherwise the first filename
    matching the reader's pattern (optionally narrowed by ``species``) is used.

    Returns:
        The resolved file, or None when none is found or the path is unsafe.

    """
    if params.path is not None:
        try:
            return _safe_join(output, params.path)
        except ValueError:
            return None
    candidates = [
        path
        for path, is_dir in _collect_entries(output)
        if not is_dir and spec.pattern.fullmatch(path.name) and _plugin_species_matches(path.name, params)
    ]
    candidates.sort(key=lambda path: path.name)
    return candidates[0] if candidates else None


def _plugin_species_matches(name: str, params: ResultParams) -> bool:
    """Whether a plugin filename belongs to the requested species/filter.

    Returns:
        True when the request names no species, or the name matches it.

    """
    if params.species is not None and not name.startswith(f"{params.species}_"):
        return False
    return not (params.species_filter != "all" and not name.endswith(f"_{params.species_filter}.dat"))


def _derive_plugin_names(reader: str, name: str) -> tuple[str, str, int | None]:
    """Derive (species, filter, iteration) from a plugin output filename.

    The shipped readers re-resolve their file from these values, so the target
    found by discovery must agree with them.

    Returns:
        The species, species filter and (transition-radiation only) iteration.

    Raises:
        ResultsReaderError: If the filename does not match the reader pattern.

    """
    if reader == "energy_histogram":
        match = re.fullmatch(r"(?P<species>.+)_energyHistogram_(?P<filter>.+)\.dat", name)
    elif reader == "emittance":
        match = re.fullmatch(r"(?P<species>.+)_emittance_(?P<filter>.+)\.dat", name)
    else:
        match = re.fullmatch(r"(?P<species>.+)_transRad_(?P<iteration>\d+)\.dat", name)
    if match is None:  # pragma: no cover - guarded by the registry pattern
        msg = f"filename {name!r} does not match the {reader} reader"
        raise ResultsReaderError(msg)
    groups = match.groupdict()
    return groups["species"], groups.get("filter", "all"), int(groups["iteration"]) if "iteration" in groups else None


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

    Used for the transition-radiation reader, whose per-iteration filenames
    (``<species>_transRad_<iteration>.dat``) carry the step.  The shipped
    ``TransitionRadiationData.get_iterations`` ignores ``species`` and globs
    *every* ``*.dat`` in ``simOutput``, so a coexisting ``_energyHistogram_`` /
    ``_emittance_`` file makes it raise ``ValueError``; enumerating the matching
    filenames here avoids that entirely.

    Returns:
        The sorted, de-duplicated iterations of the matching files.

    """
    iterations: set[int] = set()
    for path, is_dir in _collect_entries(output):
        if is_dir or not spec.pattern.fullmatch(path.name):
            continue
        if species and not path.name.startswith(f"{species}_"):
            continue
        if species_filter != "all" and not path.name.endswith(f"_{species_filter}.dat"):
            continue
        match = re.fullmatch(r".+_transRad_(?P<iteration>\d+)\.dat", path.name)
        if match is not None:
            iterations.add(int(match.group("iteration")))
    return sorted(iterations)


def _plugin_summary(  # ruff: ignore[too-many-positional-arguments] - reader inputs stay explicit
    reader: str,
    instance: Any,
    species: str,
    species_filter: str,
    iteration: int | str | None,
    available: list[int],
) -> dict[str, Any]:
    """Call one plugin reader and shape a bounded numeric summary.

    Returns:
        The reader-specific summary dict.

    """
    selected = _resolve_plugin_iteration(available, iteration)
    return _PLUGIN_BUILDERS[reader](instance, species, species_filter, selected)


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

    """
    species, species_filter, _derived_iteration = _derive_plugin_names(reader, target.name)
    # ``iteration=None`` must mean "latest" for every reader, so never fall back
    # to the iteration baked into the (alphabetically first) filename:
    # ``_resolve_plugin_iteration`` maps ``None``/``"last"`` to the maximum.
    iteration = params.iteration
    instance = _import_plugin_reader(reader)(str(output.parent))
    if reader == "transition_radiation":
        # The shipped ``get_iterations`` globs every ``*.dat`` and chokes on a
        # coexisting histogram/emittance file, so enumerate the matching
        # ``_transRad_<int>.dat`` names ourselves.
        available = _plugin_iterations(output, spec, species, species_filter)
    else:
        available = [int(step) for step in instance.get_iterations(species, species_filter)]
    return _plugin_summary(reader, instance, species, species_filter, iteration, available)


def _build_energy_histogram(instance: Any, species: str, species_filter: str, iteration: int) -> dict[str, Any]:
    """Reduce an ``EnergyHistogramData`` result to a bounded summary.

    Returns:
        Bins (keV), counts, the count in the default window and scalars.

    """
    counts, bins, _iteration, _dt = instance.get(
        iteration=iteration,
        species=species,
        species_filter=species_filter,
    )
    counts = [float(value) for value in counts]
    bins = [float(value) for value in bins]
    low, high = _DEFAULT_WINDOW_KEV
    in_window = sum(count for bin_kev, count in zip(bins, counts, strict=True) if low <= bin_kev <= high)
    peak = max(range(len(counts)), key=counts.__getitem__) if counts else 0
    strided_bins, downsampled = _stride(bins)
    strided_counts, _ = _stride(counts)
    return {
        "bins_kev": strided_bins,
        "counts": strided_counts,
        "count_in_window": {"min_kev": low, "max_kev": high, "count": in_window},
        "total": sum(counts),
        "max_energy_kev": bins[peak] if bins else None,
        "iteration": iteration,
        "downsampled": downsampled,
    }


def _build_emittance(instance: Any, species: str, species_filter: str, iteration: int) -> dict[str, Any]:
    """Reduce an ``EmittanceData`` result to a bounded summary.

    Returns:
        Slice positions (m), slice emittances (m rad) and scalars.

    """
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
        "iteration": iteration,
        "downsampled": downsampled,
    }


def _build_transition_radiation(instance: Any, species: str, species_filter: str, iteration: int) -> dict[str, Any]:
    """Reduce a ``TransitionRadiationData`` result to a bounded summary.

    The reader's ``spectrum`` view (the brightest angles) is a bounded 1D
    spectrum, which suits a wire summary better than the full 3D cube.

    Returns:
        Frequency (SI 1/s), intensity and scalars.

    """
    _ = species_filter
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
        "iteration": iteration,
        "downsampled": downsampled,
    }


#: Reader name -> the summary builder that calls the reader instance.
_PLUGIN_BUILDERS: dict[str, Callable[[Any, str, str, int], dict[str, Any]]] = {
    "energy_histogram": _build_energy_histogram,
    "emittance": _build_emittance,
    "transition_radiation": _build_transition_radiation,
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


def _plugin(  # ruff: ignore[too-many-return-statements] - one return per clean error
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
    if not output.is_dir():
        return _error(SimulationErrorCode.NO_RESULTS, "run has no linked simOutput directory")
    target = _plugin_target(output, params, spec)
    if target is None or not target.is_file():
        return _error(SimulationErrorCode.NO_RESULTS, "no such plugin result file")
    try:
        summary = _plugin_result(reader, spec, output, params, target)
    except ResultsUnavailable as exc:
        return _error(SimulationErrorCode.READER_UNAVAILABLE, str(exc))
    except (ResultsReaderError, KeyError, OSError, ValueError, IndexError) as exc:
        return _error(SimulationErrorCode.NO_RESULTS, str(exc))
    bounded = _bound_plugin(summary)
    if bounded is None:
        return _error(SimulationErrorCode.RESULT_TOO_LARGE, f"plugin summary exceeds {MAX_RESULT_BYTES} wire bytes")
    return {"result": bounded}


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


def _compute(params: ResultParams, *, output: Path) -> dict[str, Any]:  # ruff: ignore[too-many-return-statements] - one return per clean error
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
        return _load_dataset(series, record, component, iteration)

    try:
        payload = evaluate(program, resolve)
    except ResultsUnavailable as exc:
        return _error(SimulationErrorCode.READER_UNAVAILABLE, str(exc))
    except ProgramError as exc:
        return _error(SimulationErrorCode.UNSUPPORTED, str(exc))
    except (ResultsReaderError, KeyError, OSError, ValueError) as exc:
        return _error(SimulationErrorCode.NO_RESULTS, str(exc))
    return _shape_compute(payload)


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


def _shape_compute(payload: dict[str, Any]) -> dict[str, Any]:
    """Shape an evaluator payload into the frozen result-ack fields.

    The ack carries a fixed key set, so the numeric array travels as ``data``
    (``data_encoding="float"``), a scalar as ``stats["value"]``, and the
    evaluator's metadata as ``result``.

    Returns:
        The shaped ack fields, or a ``RESULT_TOO_LARGE`` error.

    """
    result = payload.get("result")
    metadata: dict[str, Any] = {"result_kind": payload.get("result_kind", "scalar")}
    if "points" in payload:
        metadata["points"] = payload["points"]
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

    Returns:
        The ack fields for this request.

    """
    run = Path(run_dir)
    output = run / _SIM_OUTPUT
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
    "read_image",
    "read_slice",
    "read_stats",
    "read_text_tail",
    "resolve_result",
    "scan_output",
]
