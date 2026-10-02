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
import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pic_agentic.results import scan_output

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)

#: Keys whose *name* marks a secret value that must not leave the cluster.  The
#: match is a case-insensitive substring, so ``api_key``, ``access_token`` and
#: ``client_secret`` are all dropped.
_SECRET_KEY_RE = re.compile(r"secret|token|password|key", re.IGNORECASE)

#: Plugin readers sampled (in this order) when the run has no openPMD output but
#: may have a text/openPMD plugin artifact.  The order is the physical priority
#: of an LWFA-style run: the energy histogram first, then the momentum-space and
#: radiation diagnostics.  The list mirrors the shipped reader registry; a reader
#: that is not installed or has no matching file is simply skipped.
_PLUGIN_SUMMARY_READERS = (
    "energy_histogram",
    "phase_space",
    "emittance",
    "transition_radiation",
    "radiation",
    "calorimeter",
    "png",
)

#: Maximum number of array points kept per plugin summary in the analysis
#: section.  The analysis answer only needs the scalars; the full strided arrays
#: already travel through ``read_plugin_result``.  Sampling a few points keeps
#: the analyze ack (which carries every reader's section) inside the 48 KiB wire
#: budget while retaining a hint of the distribution's shape.
_PLUGIN_SAMPLE_POINTS = 8

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


def read_plugin_summaries(output_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Summarize the run's plugin artifacts by reusing the results engine.

    The beta-4 physics question (``which point maximizes the high-energy
    tail?``) could not be answered because :func:`analyze` read only RO-Crate
    bookkeeping; a run whose sole artifact is a text plugin histogram (no
    openPMD field output) therefore produced pure metadata.  This function
    reuses :func:`pic_agentic.results.resolve_result` - the exact bounded
    summaries ``read_plugin_result`` returns - so :func:`synthesize_answer` can
    name the physics (populated spectrum range, maximum energy, counts) rather
    than duplicating any reader or reduction logic.

    Each reader is attempted independently against the linked ``simOutput``:
    the first file matching the reader's pattern is read at iteration ``last``,
    and a reader that is not installed (``reader_unavailable``) or has no
    matching output (``no_results``) is skipped.  The first error for a reader
    whose optional dependency is missing is remembered under the reserved
    ``__unavailable__`` key so the answer can be explicit about *why* there is
    no physics; a plain absence of artifacts is not an error.

    Args:
        output_dir: The run's linked ``simOutput`` directory.

    Returns:
        ``{reader: summary}`` for every reader that produced a summary, plus an
        optional ``{"__unavailable__": {"reason": ...}}`` when a reader could
        not run because its optional dependency is absent (e.g. no
        ``picongpu``/``openpmd_api``).  ``{}`` when there is no output
        directory.

    """
    root = _as_path(output_dir)
    if root is None or not root.is_dir():
        return {}
    from pic_agentic import results  # ruff: ignore[import-outside-top-level] - lazy optional engine
    from pic_agentic.protocol.simulation import ResultOp, ResultParams  # ruff: ignore[import-outside-top-level]

    summaries: dict[str, dict[str, Any]] = {}
    unavailable: str | None = None
    for reader in _PLUGIN_SUMMARY_READERS:
        try:
            params = ResultParams(sim_id="", op=ResultOp.PLUGIN, reader=reader, iteration="last")
            payload = results.resolve_result(params, run_dir=root.parent, sim_id="")
        except Exception as exc:  # ruff: ignore[blind-except] - a summary read is best-effort
            log.debug("plugin summary %r failed: %s", reader, exc)
            continue
        summary = payload.get("result")
        if isinstance(summary, dict):
            summaries[reader] = _sample_plugin_arrays(summary)
            continue
        if payload.get("error_code") == "reader_unavailable" and unavailable is None:
            unavailable = str(payload.get("error") or "the plugin readers are not installed")
    if unavailable is not None:
        summaries["__unavailable__"] = {"reason": unavailable}
    return summaries


def _sample_plugin_arrays(summary: dict[str, Any]) -> dict[str, Any]:
    """Stride a plugin summary's arrays so the analysis ack stays bounded.

    ``analyze`` carries every reader's section in one ack, unlike
    ``read_plugin_result`` which sends a single summary.  The answer is built
    from the scalars, so the arrays only need a shape hint; sampling
    :data:`_PLUGIN_SAMPLE_POINTS` points (endpoints kept) keeps the whole ack
    inside :data:`~pic_agentic.protocol.simulation.MAX_RESULT_BYTES`.

    Returns:
        The summary with each list array sampled down, or unchanged when small.

    """
    sampled: dict[str, Any] = dict(summary)
    for key, value in summary.items():
        if not isinstance(value, list) or len(value) <= _PLUGIN_SAMPLE_POINTS:
            continue
        step = math.ceil(len(value) / _PLUGIN_SAMPLE_POINTS)
        points = value[::step]
        if points[-1] != value[-1]:
            points = [*points, value[-1]]
        sampled[key] = points
    return sampled


def _facts(rocrate: Any, metadata: Any, openpmd: Any) -> list[str]:
    """Build the deterministic *bookkeeping* fact lines.

    The physics facts are built separately by :func:`_plugin_physics_facts`: a
    caller must be able to tell "no physics values" from "readers missing", and
    the reader-unavailable note is not a fact.

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


def _plugin_physics_facts(plugins: Any) -> list[str]:
    """Physics facts from the readers that actually produced a summary.

    The reserved ``__unavailable__`` entry is not a physics fact and is excluded,
    so the query path can tell "no physics values" from "readers missing".

    Returns:
        The physics fact strings (possibly empty).

    """
    if not isinstance(plugins, dict):
        return []
    facts: list[str] = []
    for reader, summary in plugins.items():
        if reader.startswith("__") or not isinstance(summary, dict):
            continue
        facts.extend(_one_plugin_facts(reader, summary))
    return facts


def _plugin_unavailable_reason(plugins: Any) -> str | None:
    """Return the first reader-unavailable reason recorded, if any.

    Returns:
        The reason string, or None when no reader reported unavailability.

    """
    if not isinstance(plugins, dict):
        return None
    unavailable = plugins.get("__unavailable__")
    if isinstance(unavailable, dict) and unavailable.get("reason"):
        return str(unavailable["reason"])
    return None


def _one_plugin_facts(reader: str, summary: dict[str, Any]) -> list[str]:
    """Fact lines for one plugin reader's bounded summary.

    Only the physics-bearing scalars are turned into prose; the full strided
    arrays already travel in the section.  Readers without a meaningful scalar
    (e.g. ``png``, which carries image dimensions only) contribute nothing.

    Returns:
        The fact strings for this reader (possibly empty).

    """
    source = summary.get("source_path")
    where = f" from {source}" if source else ""
    iteration = summary.get("iteration")
    step = f" at iteration {iteration}" if iteration is not None else ""
    builder = _PLUGIN_FACT_BUILDERS.get(reader)
    facts = builder(summary, where, step) if builder is not None else []
    warning = summary.get("warning")
    if warning:
        facts.append(str(warning))
    return facts


def _energy_histogram_physics(summary: dict[str, Any], where: str, step: str) -> list[str]:
    """Physics facts for the ``energy_histogram`` summary.

    Returns:
        The fact strings for this reader (possibly empty).

    """
    facts: list[str] = []
    if (total := summary.get("total")) is not None:
        facts.append(f"energy histogram{where}{step} holds {total:.6g} particles in total")
    if (maximum := summary.get("max_energy_kev")) is not None:
        facts.append(f"the highest populated energy bin is {maximum:.6g} keV")
    window = summary.get("count_in_window")
    if isinstance(window, dict) and window.get("count") is not None:
        low, high = window.get("min_kev"), window.get("max_kev")
        facts.append(f"{window['count']:.6g} particles lie in the {low}-{high} keV window")
    return facts


def _emittance_physics(summary: dict[str, Any], where: str, step: str) -> list[str]:
    """Physics facts for the ``emittance`` summary.

    Returns:
        The fact strings for this reader (possibly empty).

    """
    facts: list[str] = []
    if (total := summary.get("total_emit_mrad")) is not None:
        facts.append(f"total slice emittance{where}{step} is {total:.6g} mrad")
    if (maximum := summary.get("max_emit_mrad")) is not None:
        facts.append(f"the peak slice emittance is {maximum:.6g} mrad")
    return facts


def _transition_radiation_physics(summary: dict[str, Any], where: str, step: str) -> list[str]:
    """Physics facts for the ``transition_radiation`` summary.

    Returns:
        The fact strings for this reader (possibly empty).

    """
    facts: list[str] = []
    if (total := summary.get("total_intensity")) is not None:
        facts.append(f"transition-radiation total intensity{where}{step} is {total:.6g}")
    if (peak := summary.get("peak_omega_per_s")) is not None:
        facts.append(f"the intensity peaks at {peak:.6g} rad/s")
    return facts


def _phase_space_physics(summary: dict[str, Any], where: str, step: str) -> list[str]:
    """Physics facts for the ``phase_space`` summary.

    Returns:
        The fact strings for this reader (possibly empty).

    """
    facts: list[str] = []
    if (total := summary.get("total_count")) is not None:
        facts.append(f"phase-space plane{where}{step} holds {total:.6g} particles")
    r_max, p_max = summary.get("r_max_m"), summary.get("p_max")
    if r_max is not None or p_max is not None:
        facts.append(f"phase-space ranges reach r={r_max} m and p={p_max}")
    return facts


def _radiation_physics(summary: dict[str, Any], where: str, step: str) -> list[str]:
    """Physics facts for the ``radiation`` summary.

    Returns:
        The fact strings for this reader (possibly empty).

    """
    facts: list[str] = []
    if (total := summary.get("total_energy_J")) is not None:
        facts.append(f"radiation spectrum{where}{step} carries {total:.6g} J")
    if (peak := summary.get("peak_omega_per_s")) is not None:
        facts.append(f"the spectrum peaks at {peak:.6g} rad/s")
    return facts


def _calorimeter_physics(summary: dict[str, Any], where: str, step: str) -> list[str]:
    """Physics facts for the ``calorimeter`` summary.

    Returns:
        The fact strings for this reader (possibly empty).

    """
    facts: list[str] = []
    if (total := summary.get("total_energy_J")) is not None:
        facts.append(f"calorimeter{where}{step} deposits {total:.6g} J")
    if (maximum := summary.get("max_energy_J")) is not None:
        facts.append(f"the calorimeter's largest cell holds {maximum:.6g} J")
    return facts


#: Reader name -> the scalar-to-prose builder.  Kept in lock-step with
#: :data:`_PLUGIN_SUMMARY_READERS`; a reader without an entry (``png``)
#: contributes its warning (if any) but no physics prose.
_PLUGIN_FACT_BUILDERS: dict[str, Callable[[dict[str, Any], str, str], list[str]]] = {
    "energy_histogram": _energy_histogram_physics,
    "emittance": _emittance_physics,
    "transition_radiation": _transition_radiation_physics,
    "phase_space": _phase_space_physics,
    "radiation": _radiation_physics,
    "calorimeter": _calorimeter_physics,
}


def synthesize_answer(
    query: str | None,
    rocrate: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    openpmd: dict[str, Any] | None,
    plugins: dict[str, Any] | None = None,
) -> str:
    """Compose a deterministic, template-based natural-language summary.

    No LLM is called.  Without a ``query`` the answer lists the physics facts
    (from the plugin summaries, when present) before the bookkeeping metadata.
    With a ``query``, facts are filtered by case-insensitive keyword match; when
    nothing matches literally but physics facts exist, those physics facts are
    reported anyway, so a physics question is answered with physics rather than
    with RO-Crate metadata.  When no physics could be read at all, the answer
    says so explicitly instead of returning metadata only.

    Args:
        query: Optional free-text question used to select relevant facts.
        rocrate: The :func:`read_rocrate` section.
        metadata: The :func:`read_pypicongpu_metadata` section.
        openpmd: The :func:`read_openpmd_summary` section.
        plugins: The :func:`read_plugin_summaries` section (physics facts).

    Returns:
        The synthesized answer string.

    """
    physics = _plugin_physics_facts(plugins)
    unavailable = _plugin_unavailable_reason(plugins)
    notes = [f"no plugin reader could summarize the output: {unavailable}"] if unavailable else []
    bookkeeping = _facts(rocrate, metadata, openpmd)
    facts = physics + notes + bookkeeping
    if query:
        tokens = [token.lower() for token in re.split(r"\W+", query) if token]
        matched = [fact for fact in facts if any(token in fact.lower() for token in tokens)]
        if matched:
            return f"Matched the query {query!r}: " + "; ".join(matched) + "."
        if physics:
            return (
                f"No fact matched the query {query!r} literally; the available physics is: " + "; ".join(physics) + "."
            )
        return _no_physics_answer(query, openpmd, unavailable, bookkeeping)
    return _default_answer(facts)


def _no_physics_answer(
    query: str,
    openpmd: dict[str, Any] | None,
    unavailable: str | None,
    bookkeeping: list[str],
) -> str:
    """Explain, explicitly, that no physics could be read for this run.

    The beta-4 run's only artifact was a text plugin histogram; with no openPMD
    field output the old answer silently fell back to RO-Crate metadata.  This
    message names the missing artifact(s) and still carries the bookkeeping
    facts, so a caller can tell "no physics was produced" from "physics was not
    summarized".

    Args:
        query: The caller's question.
        openpmd: The openPMD summary section.
        unavailable: The reader-unavailable reason, when one was recorded.
        bookkeeping: The non-physics fact strings.

    Returns:
        The synthesized no-physics answer.

    """
    tail = f" Bookkeeping metadata: {'; '.join(bookkeeping)}." if bookkeeping else ""
    if unavailable:
        return (
            f"Cannot answer the physics question {query!r}: no plugin artifact could be summarized "
            f"({unavailable}).{tail}"
        )
    if not openpmd:
        return (
            f"Cannot answer the physics question {query!r}: the run has no openPMD output and no "
            f"plugin histogram, so no spectrum or energy data exists to report. Only metadata is "
            f"available.{tail}"
        )
    return (
        f"Cannot answer the physics question {query!r}: openPMD output is present but no plugin "
        f"histogram (energy_histogram/phase_space/...) was summarized, so no physics values are "
        f"available.{tail}"
    )


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
    plugins: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose the four analysis sections for one simulation.

    Missing or malformed inputs degrade to empty sections; this function never
    raises.  When ``setup_dir`` is omitted it is derived from ``run_dir`` as the
    sibling ``input`` directory, but only when that path really is a direct
    child of the run's parent: a symlinked ``input`` escaping the run base is
    rejected, so analysis can never read outside the run directory.  When
    ``plugins`` is omitted but an ``output_dir`` is given, the run's plugin
    artifacts are summarized by reusing the results engine, so a run whose only
    artifact is a text plugin histogram still yields physics in its answer.

    Args:
        run_dir: The run directory (``<base>/run``); used only as a fallback
            source for ``setup_dir`` when ``setup_dir`` is None.
        setup_dir: The setup directory (``<base>/input``), or None to derive it.
        output_dir: The linked ``simOutput`` directory.
        query: Optional free-text question for the synthesized answer.
        plugins: Optional pre-read plugin summaries (server-side re-synthesis);
            when omitted they are read from ``output_dir``.

    Returns:
        ``{"rocrate", "metadata", "openpmd", "plugins", "answer"}``.

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
    if plugins is None:
        try:
            plugins = read_plugin_summaries(output) if output is not None else {}
        except Exception:  # ruff: ignore[blind-except] - analyze must never raise
            plugins = {}
    try:
        answer = synthesize_answer(query, rocrate, metadata, openpmd, plugins)
    except Exception:  # ruff: ignore[blind-except] - analyze must never raise
        answer = "No analysis metadata is available for this simulation."
    return {"rocrate": rocrate, "metadata": metadata, "openpmd": openpmd, "plugins": plugins, "answer": answer}


__all__ = [
    "analyze",
    "read_openpmd_summary",
    "read_plugin_summaries",
    "read_pypicongpu_metadata",
    "read_rocrate",
    "synthesize_answer",
]
