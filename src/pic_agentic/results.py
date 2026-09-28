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
import importlib.util
import io
import json
import math
import os
import re
from pathlib import Path
from typing import Any

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

#: Trailing integer run in a series file's stem: the iteration infix PIConGPU
#: writes (``fields_000050.h5`` -> ``000050``).  Rewritten to openPMD's ``%T``
#: wildcard so a series is opened across every iteration, not just one file.
_ITERATION_INFIX_RE = re.compile(r"\d+$")

#: The sentinel component name openPMD reports for a scalar (componentless)
#: mesh record; it must not be presented as a selectable component.
_SCALAR_COMPONENT = "\x0bScalar"

#: Directory (relative to ``run_dir``) the workflow links the results into.
_SIM_OUTPUT = "simOutput"


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
        One of ``openpmd-adios2``, ``openpmd-hdf5``, ``text``, ``dir`` or
        ``binary``.

    """
    if is_dir:
        return "dir"
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


def _find_series(output: Path, relpath: str | None) -> Path | None:
    """Locate the openPMD series file/pattern for a run's ``simOutput``.

    A PICMI diagnostic's own ``result_path(prefix)`` returns an openPMD *pattern*
    with a ``%T`` iteration wildcard (e.g. ``.../openPMD/simOutput/fields_%06T.h5``),
    which is exactly what :meth:`openpmd_api.Series` accepts.  That pattern is not
    persisted to the cluster, so it is reconstructed here from the files on disk:
    the caller's ``path`` selects a file or tree, or the whole ``simOutput`` is
    searched, and the first openPMD-backend file found is rewritten to a ``%T``
    pattern by replacing its trailing iteration number.  This is picongpu-free and
    makes no assumption about the nesting depth PIConGPU uses.

    Args:
        output: The run's linked ``simOutput`` directory.
        relpath: The request's relative path (a file, a directory, or None).

    Returns:
        The discovered series path (possibly a ``%T`` pattern), or None when the
        path is unsafe, absent, or contains no openPMD file.

    """
    base = output
    if relpath:
        try:
            base = _safe_join(output, relpath)
        except ValueError:
            return None
    if base.is_file():
        return base
    if not base.is_dir():
        return None
    for path, entry_is_dir in _collect_entries(base):
        if entry_is_dir or path.suffix.lower() not in _OPENPMD_SUFFIXES:
            continue
        # Substitute on the *stem* only: ``\d+$`` on the whole name would match
        # the trailing digit of the extension (``.h5``), not the iteration.
        pattern = _ITERATION_INFIX_RE.sub("%T", path.stem) + path.suffix
        return path.with_name(pattern)
    return None


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

    preflight = _compute_preflight(params, output)
    if preflight is not None:
        return preflight
    try:
        program = AnalysisProgram.model_validate(params.program)
    except Exception as exc:  # ruff: ignore[blind-except] - an invalid program is ack data
        return _error(SimulationErrorCode.UNSUPPORTED, f"invalid analysis program: {exc}")

    series = _find_series(output, params.path)
    if series is None:
        return _error(SimulationErrorCode.NO_RESULTS, "no openPMD output found for this run")

    def resolve(selector: Any) -> list[float]:
        # A selector resolves to one mesh component; the reader applies the
        # same validation the slice path uses.  A selector that omits
        # record/component falls back to the request's top-level ones, and an
        # empty record means the first available mesh.
        record = getattr(selector, "record", None) or params.record
        component = getattr(selector, "component", None) or params.component
        iteration = getattr(selector, "iteration", None)
        if iteration is None:
            iteration = params.iteration
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


def _compute_preflight(params: ResultParams, output: Path) -> dict[str, Any] | None:
    """Return a compute error for a request that cannot proceed, else None.

    Args:
        params: The validated result request.
        output: The run's linked ``simOutput`` directory.

    Returns:
        An error pair, or None when the request is ready to evaluate.

    """
    if _reader_name() is None:
        return _error(SimulationErrorCode.READER_UNAVAILABLE, "the optional openpmd_api reader is not installed")
    if params.program is None:
        return _error(SimulationErrorCode.UNSUPPORTED, "compute requires a program")
    if not output.is_dir():
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
    target = _find_series(output, params.path)
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


def resolve_result(
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
