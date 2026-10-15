# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Milestone A analysis closure: RO-Crate + pypicongpu metadata + a summary.

The module is deliberately extraction-friendly: stdlib + ``json`` only, plus the
openPMD-free :func:`pic_agentic.results.scan_output` walk.  ``openpmd_api`` and
``rocrate`` are *not* imported (they may not be installed on the submission
node); the RO-Crate metadata is parsed from plain JSON.

Every reader degrades to an empty section on a missing or malformed input and
:func:`analyze` never raises: the result travels back to the LLM as data, never
as a tool exception.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pic_agentic.results import scan_output

#: Keys whose *name* marks a secret value that must not leave the cluster.  The
#: match is a case-insensitive substring, so ``api_key``, ``access_token`` and
#: ``client_secret`` are all dropped.
_SECRET_KEY_RE = re.compile(r"secret|token|password|key", re.IGNORECASE)

#: openPMD filename suffix -> backend name (the suffix survives even when the
#: ``.bp``/``.bp5`` series is itself a directory in ADIOS2).
_OPENPMD_SUFFIXES = {".bp": "adios2", ".bp5": "adios2", ".h5": "hdf5", ".hdf5": "hdf5"}

#: Numeric run of an openPMD file stem, e.g. ``sim_00000100`` -> ``100``.
_ITERATION_RE = re.compile(r"\d+")

#: Maximum number of metadata field names quoted in the synthesized answer.
_MAX_FACT_FIELDS = 12


def _as_path(value: Path | str | None) -> Path | None:
    """Coerce a value to a :class:`Path`, returning None when it cannot be.

    Args:
        value: A path, a string path, or None.

    Returns:
        The path, or None for a missing/uncoercible value.

    """
    if value is None:
        return None
    try:
        return Path(value)
    except (TypeError, ValueError):
        return None


def _load_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object from ``path`` without ever raising.

    Args:
        path: The JSON file to read.

    Returns:
        The decoded mapping, or ``{}`` when the file is missing, unreadable,
        malformed or not a JSON object.

    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        # ValueError covers a null byte in the path / a UnicodeDecodeError.
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _derive_setup_dir(run_dir: Path) -> Path | None:
    """Derive ``<base>/input`` from ``<base>/run`` with a containment check.

    Mirrors :func:`pic_agentic.simclient.client.derive_setup_dir`: the derived
    directory is accepted only when it resolves to a direct child of the run's
    parent, so a symlinked ``input`` cannot escape the run base.

    Args:
        run_dir: The run directory (``<base>/run``).

    Returns:
        The resolved ``<base>/input`` path, or None when it is not a direct
        sibling of the run directory.

    """
    try:
        base = run_dir.parent.resolve()
        setup = (run_dir.parent / "input").resolve()
    except (OSError, ValueError):
        return None
    if setup.parent != base:
        return None
    return setup


def _redact_secrets(value: Any) -> Any:
    """Recursively drop mapping entries whose key names a secret.

    Args:
        value: The decoded JSON value.

    Returns:
        The value with every ``secret|token|password|key`` entry removed.

    """
    if isinstance(value, dict):
        return {
            key: _redact_secrets(item)
            for key, item in value.items()
            if not (isinstance(key, str) and _SECRET_KEY_RE.search(key))
        }
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


def read_rocrate(setup_dir: Path | str) -> dict[str, Any]:
    """Parse the RO-Crate metadata descriptor of a generated setup.

    The root data entity (``@id == "./"``) supplies the ``name``,
    ``datePublished`` and ``mainEntity`` fields, and its ``instrument`` (a
    ``SoftwareApplication``) is reduced to ``{"name", "id"}``.

    Args:
        setup_dir: The runner's setup directory (holds
            ``ro-crate-metadata.json``).

    Returns:
        ``{"name", "datePublished", "mainEntity", "software"}``, or ``{}``
        when the descriptor is missing, malformed or carries no root entity.

    """
    root_dir = _as_path(setup_dir)
    if root_dir is None:
        return {}
    data = _load_json_object(root_dir / "ro-crate-metadata.json")
    if not data:
        return {}
    graph = data.get("@graph")
    entities = graph if isinstance(graph, list) else [data]
    root = next((e for e in entities if isinstance(e, dict) and e.get("@id") == "./"), None)
    if root is None:
        root = next(
            (e for e in entities if isinstance(e, dict) and ("mainEntity" in e or "instrument" in e)),
            None,
        )
    if root is None:
        return {}
    return {
        "name": root.get("name"),
        "datePublished": root.get("datePublished"),
        "mainEntity": root.get("mainEntity"),
        "software": _instrument_summary(root.get("instrument")),
    }


def _instrument_summary(instrument: Any) -> dict[str, Any]:
    """Reduce a RO-Crate ``instrument`` to its name and identifier.

    Args:
        instrument: The root entity's ``instrument`` value (a mapping or a
            list of mappings).

    Returns:
        ``{"name", "id"}`` (values may be None), or ``{}`` when no instrument
        mapping is present.

    """
    candidate = instrument
    if isinstance(candidate, list):
        candidate = next((item for item in candidate if isinstance(item, dict)), None)
    if not isinstance(candidate, dict):
        return {}
    return {"name": candidate.get("name"), "id": candidate.get("@id")}


def read_pypicongpu_metadata(setup_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Read the three pypicongpu metadata files under ``metadata/``.

    All three sections are redacted (reuse :func:`_redact_secrets`): the
    contract requires it for ``rc_params``, and applying it uniformly is
    defence in depth so no metadata file can leak a credential into tool
    output.

    Args:
        setup_dir: The runner's setup directory.

    Returns:
        ``{"runner", "rc_params", "rendering_context"}``; a missing or
        malformed file yields ``{}`` for its section.

    """
    root = _as_path(setup_dir)
    if root is None:
        return {"runner": {}, "rc_params": {}, "rendering_context": {}}
    files = {
        "runner": "pypicongpu_runner.json",
        "rc_params": "rc_params.json",
        "rendering_context": "pypicongpu_rendering_context.json",
    }
    sections: dict[str, dict[str, Any]] = {}
    for field, filename in files.items():
        sections[field] = _redact_secrets(_load_json_object(root / "metadata" / filename))
    return sections


def read_openpmd_summary(output_dir: Path | str) -> dict[str, Any]:
    """Summarize the linked openPMD output using the scandir-only walk.

    Reuses :func:`pic_agentic.results.scan_output` (which never opens a file and
    never imports ``openpmd_api``), then derives the available iteration
    numbers, the latest step and the backend names from the file names.

    Args:
        output_dir: The run's linked ``simOutput`` directory.

    Returns:
        ``{"iterations", "latest_step", "backends"}``, or ``{}`` when no
        openPMD output (or no directory) is present.

    """
    root = _as_path(output_dir)
    if root is None:
        return {}
    try:
        manifest = scan_output(root, sim_id="", run_dir=str(root))
    except Exception:  # ruff: ignore[blind-except] - a scan failure is an empty section
        return {}
    backends: set[str] = set()
    iterations: set[int] = set()
    for ref in manifest.files:
        suffix = Path(ref.path).suffix.lower()
        backend = _OPENPMD_SUFFIXES.get(suffix)
        if backend is None:
            continue
        backends.add(backend)
        numbers = _ITERATION_RE.findall(Path(ref.path).stem)
        if numbers:
            iterations.add(int(numbers[-1]))
    if not backends:
        return {}
    return {
        "iterations": sorted(iterations),
        "latest_step": max(iterations) if iterations else None,
        "backends": sorted(backends),
    }


def _facts(rocrate: Any, metadata: Any, openpmd: Any) -> list[str]:
    """Build the deterministic fact lines from the three sections.

    Args:
        rocrate: The :func:`read_rocrate` result.
        metadata: The :func:`read_pypicongpu_metadata` result.
        openpmd: The :func:`read_openpmd_summary` result.

    Returns:
        Human-readable fact strings (possibly empty).

    """
    return _rocrate_facts(rocrate) + _metadata_facts(metadata) + _openpmd_facts(openpmd)


def _rocrate_facts(rocrate: Any) -> list[str]:
    """Fact lines describing the RO-Crate section.

    Returns:
        The RO-Crate fact strings.

    """
    if not isinstance(rocrate, dict) or not rocrate:
        return []
    facts: list[str] = []
    if rocrate.get("name"):
        facts.append(f"experiment name is {rocrate['name']}")
    if rocrate.get("datePublished"):
        facts.append(f"published on {rocrate['datePublished']}")
    software = rocrate.get("software")
    if isinstance(software, dict) and software.get("name"):
        ident = f" ({software['id']})" if software.get("id") else ""
        facts.append(f"generated with {software['name']}{ident}")
    main = rocrate.get("mainEntity")
    main_id = main.get("@id") if isinstance(main, dict) else main if isinstance(main, str) else None
    if main_id:
        facts.append(f"main entity is {main_id}")
    return facts


def _metadata_facts(metadata: Any) -> list[str]:
    """Fact lines describing the pypicongpu metadata sections.

    Returns:
        The metadata fact strings.

    """
    if not isinstance(metadata, dict):
        return []
    labels = {"runner": "runner", "rc_params": "rc parameters", "rendering_context": "rendering context"}
    facts: list[str] = []
    for key, label in labels.items():
        section = metadata.get(key)
        if isinstance(section, dict) and section:
            names = ", ".join(sorted(str(field) for field in section)[:_MAX_FACT_FIELDS])
            facts.append(f"{label} metadata carries {len(section)} field(s): {names}")
    return facts


def _openpmd_facts(openpmd: Any) -> list[str]:
    """Fact lines describing the openPMD summary section.

    Returns:
        The openPMD fact strings.

    """
    if not isinstance(openpmd, dict) or not openpmd:
        return []
    facts: list[str] = []
    if openpmd.get("iterations"):
        facts.append(f"openPMD iterations {openpmd['iterations']}")
    if openpmd.get("latest_step") is not None:
        facts.append(f"latest openPMD step is {openpmd['latest_step']}")
    if openpmd.get("backends"):
        facts.append("openPMD backends: " + ", ".join(openpmd["backends"]))
    return facts


def synthesize_answer(
    query: str | None,
    rocrate: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    openpmd: dict[str, Any] | None,
) -> str:
    """Compose a deterministic, template-based natural-language summary.

    No LLM is called.  Without a ``query`` the answer lists every extracted
    fact; with one, the facts are filtered by case-insensitive keyword match and
    the answer says what matched (or that nothing did).

    Args:
        query: Optional free-text question used to select relevant facts.
        rocrate: The :func:`read_rocrate` section.
        metadata: The :func:`read_pypicongpu_metadata` section.
        openpmd: The :func:`read_openpmd_summary` section.

    Returns:
        The synthesized answer string.

    """
    facts = _facts(rocrate, metadata, openpmd)
    if query:
        tokens = [token.lower() for token in re.split(r"\W+", query) if token]
        matched = [fact for fact in facts if any(token in fact.lower() for token in tokens)]
        if matched:
            return f"Matched the query {query!r}: " + "; ".join(matched) + "."
        if not facts:
            return f"No analysis metadata is available for this simulation; nothing matched {query!r}."
        return f"No analysis fields matched the query {query!r}. " + _default_answer(facts)
    return _default_answer(facts)


def _default_answer(facts: list[str]) -> str:
    """Format the full fact list as the default summary.

    Args:
        facts: The fact strings from :func:`_facts`.

    Returns:
        The default summary sentence.

    """
    if not facts:
        return "No analysis metadata is available for this simulation."
    return "Analysis summary: " + "; ".join(facts) + "."


def analyze(
    run_dir: Path | str | None,
    setup_dir: Path | str | None,
    output_dir: Path | str | None,
    query: str | None = None,
) -> dict[str, Any]:
    """Compose the four analysis sections for one simulation.

    Missing or malformed inputs degrade to empty sections; this function never
    raises.  When ``setup_dir`` is omitted it is derived from ``run_dir`` as the
    sibling ``input`` directory, but only when that path really is a direct
    child of the run's parent: a symlinked ``input`` escaping the run base is
    rejected, so analysis can never read outside the run directory.

    Args:
        run_dir: The run directory (``<base>/run``); used only as a fallback
            source for ``setup_dir`` when ``setup_dir`` is None.
        setup_dir: The setup directory (``<base>/input``), or None to derive it.
        output_dir: The linked ``simOutput`` directory.
        query: Optional free-text question for the synthesized answer.

    Returns:
        ``{"rocrate", "metadata", "openpmd", "answer"}``.

    """
    run = _as_path(run_dir)
    setup = _as_path(setup_dir)
    if setup is None and run is not None:
        setup = _derive_setup_dir(run)
    output = _as_path(output_dir)

    try:
        rocrate = read_rocrate(setup) if setup is not None else {}
    except Exception:  # ruff: ignore[blind-except] - analyze must never raise
        rocrate = {}
    try:
        metadata = read_pypicongpu_metadata(setup) if setup is not None else {}
    except Exception:  # ruff: ignore[blind-except] - analyze must never raise
        metadata = {}
    try:
        openpmd = read_openpmd_summary(output) if output is not None else {}
    except Exception:  # ruff: ignore[blind-except] - analyze must never raise
        openpmd = {}
    try:
        answer = synthesize_answer(query, rocrate, metadata, openpmd)
    except Exception:  # ruff: ignore[blind-except] - analyze must never raise
        answer = "No analysis metadata is available for this simulation."
    return {"rocrate": rocrate, "metadata": metadata, "openpmd": openpmd, "answer": answer}


__all__ = [
    "analyze",
    "read_openpmd_summary",
    "read_pypicongpu_metadata",
    "read_rocrate",
    "synthesize_answer",
]
