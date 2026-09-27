# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Execute an agenda group as a real CWL workflow (research-loop gap 1).

The agenda engine is the *orchestration* layer (budget, approval gates,
callbacks, agent mutation); CWL is the *execution substrate* for a fan-out.
This module makes that substrate real: it materialises an
:class:`~pic_agentic.agenda.model.AgendaGroup` as a cwltool workflow tree and
runs it, so the emitted CWL is no longer dead code.

Each :class:`~pic_agentic.agenda.model.AgendaSim` becomes a ``CommandLineTool``
whose command is supplied by the caller (``leaf_command(name, sim) -> argv``),
and each :class:`~pic_agentic.agenda.model.AgendaGroup` becomes a nested
``Workflow``.  Dependencies become CWL step-input bindings, exactly as
:func:`~pic_agentic.agenda.cwl.to_cwl_workflow` emits them.

``cwltool`` is an optional dependency (the ``[sim]`` extra): this module imports
it lazily and reports a clean, soft failure when it is absent, so the rest of
the package stays importable without it.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pic_agentic.agenda.cwl import DEPENDENCY_INPUT_PREFIX, dump_cwl_workflow, to_cwl_workflow
from pic_agentic.agenda.model import AgendaGroup

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pic_agentic.agenda.model import AgendaSim

#: The command that runs CWL when invoked as a module.
_CWLTool_MODULE = "cwltool"

#: Cap on the retained log tail (the full log is not needed; a failing step is).
_MAX_LOG_TAIL = 4000


class CwlUnavailableError(RuntimeError):
    """Raised when cwltool is not importable."""


def cwltool_available() -> bool:
    """Whether the optional cwltool runner is importable.

    Returns:
        True when cwltool can be imported.

    """
    return importlib.util.find_spec(_CWLTool_MODULE) is not None


def run_agenda_cwl(
    group: AgendaGroup,
    *,
    root: Path | str,
    leaf_command: Callable[[str, AgendaSim], Sequence[str]],
    exec_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Materialise and run ``group`` as a CWL workflow via cwltool.

    Args:
        group: The agenda group to execute.
        root: Directory the workflow tree is written under.
        leaf_command: Maps ``(name, sim)`` to the argv a leaf runs.
        exec_dir: Where cwltool runs (defaults to a temp dir); kept separate so
            a caller's ``root`` tree stays pristine.

    Returns:
        ``{"ok": bool, "available": bool, "root": str, "outputs": {...} |
        None, "error": str | None, "log_tail": str}``; never raises for a
        missing cwltool or a failing command (a failing step is reported as
        ``ok=False`` with the captured log).

    """
    root_path = Path(root)
    if not cwltool_available():
        return {
            "ok": False,
            "available": False,
            "root": str(root_path),
            "outputs": None,
            "error": "cwltool is not installed (install the [sim] extra)",
            "log_tail": "",
        }
    _write_tree(group, root_path, leaf_command)
    workflow = root_path / f"{group.name}.cwl"
    workdir = Path(exec_dir) if exec_dir is not None else Path(tempfile.mkdtemp(prefix="agenda-cwl-"))
    workdir.mkdir(parents=True, exist_ok=True)
    outdir = workdir / "outputs"
    provenance = workdir / "cwl-outputs.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            _CWLTool_MODULE,
            "--no-container",
            "--provenance",
            str(provenance),
            "--outdir",
            str(outdir),
            str(workflow),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=workdir,
    )
    combined = proc.stdout + proc.stderr
    # cwltool does not always propagate a step failure to its exit code (a
    # ``permanentFail`` can still exit 0), so the run status is read from the
    # log as well.
    failed = proc.returncode != 0 or "Final process status is permanentFail" in combined
    return {
        "ok": not failed,
        "available": True,
        "root": str(root_path),
        "outputs": _list_outputs(provenance) if not failed else None,
        "error": None if not failed else f"cwltool reported failure (exit {proc.returncode})",
        "log_tail": combined[-_MAX_LOG_TAIL:],
    }


def _write_tree(
    group: AgendaGroup,
    root: Path,
    leaf_command: Callable[[str, AgendaSim], Sequence[str]],
    *,
    prefix: str = "",
) -> str:
    """Write ``group`` and its descendants as a CWL workflow tree under ``root``.

    Every workflow file is named after the entry's *path* (``<prefix><name>.cwl``)
    and all files live in one directory, so no two share a basename -- cwltool's
    provenance scanner merges dependencies by basename and rejects a collision.

    Args:
        group: The group to materialise.
        root: The tree root.
        leaf_command: Maps a leaf to its argv.
        prefix: The accumulated path prefix (``sub__`` for nested entries).

    Returns:
        The file name of ``group``'s own workflow (relative to ``root``).

    """
    group_file = f"{prefix}{group.name}.cwl"
    root.mkdir(parents=True, exist_ok=True)
    # All workflows sit in ``root``, so a reference is just the file name.
    references = {name: f"{prefix}{name}.cwl" for name in group.entries}
    (root / group_file).write_text(
        dump_cwl_workflow(
            to_cwl_workflow(
                group,
                leaf_workflow=lambda name, _sim: references[name],
                group_workflow=lambda name: references[name],
            ),
        ),
        encoding="utf-8",
    )
    for name, entry in group.entries.items():
        if isinstance(entry, AgendaGroup):
            _write_tree(entry, root, leaf_command, prefix=f"{prefix}{name}__")
        else:
            (root / f"{prefix}{name}.cwl").write_text(
                dump_cwl_workflow(_leaf_tool(f"{prefix}{name}", entry, leaf_command)),
                encoding="utf-8",
            )
    return group_file


def _leaf_tool(name: str, sim: AgendaSim, leaf_command: Callable[[str, AgendaSim], Sequence[str]]) -> dict[str, Any]:
    """Build the ``CommandLineTool`` for one leaf.

    Returns:
        The CWL command-line tool dict.

    """
    inputs: dict[str, Any] = {
        f"{DEPENDENCY_INPUT_PREFIX}{dep}": {"type": "File?", "default": None} for dep in sim.depends_on
    }
    return {
        "cwlVersion": "v1.2",
        "class": "CommandLineTool",
        "label": f"agenda leaf {name}",
        "baseCommand": list(leaf_command(name, sim)),
        "inputs": inputs,
        "outputs": {
            "input_directory": {"type": "Directory", "outputBinding": {"glob": "."}},
            "submission_information": {
                "type": "File",
                "outputBinding": {"glob": f"{name}.submission"},
            },
        },
    }


def _list_outputs(path: Path) -> dict[str, Any] | None:
    """Return the ``--outdir`` file listing if the runner wrote provenance JSON.

    Args:
        path: The expected cwltool provenance path.

    Returns:
        The decoded JSON object, or None when absent/unreadable.

    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def manifest_for(group: AgendaGroup) -> dict[str, Any]:
    """Return the workflow YAML a caller can inspect without running it.

    Returns:
        ``{"workflow": <yaml text>, "leaves": [names]}``.

    """
    return {
        "workflow": dump_cwl_workflow(to_cwl_workflow(group)),
        "leaves": [path for path, _ in group.simulations()],
    }


__all__ = [
    "CwlUnavailableError",
    "cwltool_available",
    "manifest_for",
    "run_agenda_cwl",
]
