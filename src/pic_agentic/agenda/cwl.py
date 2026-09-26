# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Emit a CWL ``Workflow`` for an agenda group.

This is the one agenda module that needs :mod:`yaml`; the model and budget
layers stay dependency-light.  The emitted shape deliberately mirrors the
proven structure of the PIConGPU ``SimulationGroup`` prototype (one nested
sub-workflow per child, ``SubworkflowFeatureRequirement``), but it is built as
a plain ``dict`` and dumped with ``yaml.safe_dump`` -- never string-formatted.

CWL is already a dependency-carrying workflow language, so agenda dependencies
become CWL step dependencies and we build on that rather than inventing a
scheduler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

from pic_agentic.agenda.model import AgendaGroup, AgendaSim

if TYPE_CHECKING:
    from collections.abc import Callable

#: The per-leaf output suffixes re-exposed by the group workflow, mapped to
#: their CWL type and human label (ordered: input dir before submission info).
_GROUP_OUTPUT_DEFS: dict[str, tuple[str, str]] = {
    "input_directory": ("Directory", "input directory"),
    "submission_information": ("File", "submission information"),
}

#: Relative path, from a group directory, of a leaf simulation's workflow.
_LEAF_WORKFLOW = "workflow/workflow.cwl"

#: Relative path, from a group directory, of a nested group's workflow.
_GROUP_WORKFLOW = "workflow/group_workflow.cwl"

#: Input name a *successor* workflow must declare to receive the "predecessor
#: finished" signal.  CWL expresses ordering only through data flow, so an
#: agenda dependency becomes a step-input binding from the predecessor's
#: ``submission_information`` output to this input on the successor.  The
#: per-simulation workflow generator (the execution milestone) must therefore
#: emit an optional input of this name for any leaf that can be a dependency
#: target; until then cwltool reports a checker *warning* (not an error) and
#: the emitted workflow remains structurally valid CWL.
DEPENDENCY_INPUT_PREFIX = "depends_on_"


def to_cwl_workflow(
    group: AgendaGroup,
    *,
    leaf_workflow: Callable[[str, AgendaSim], str] | None = None,
) -> dict[str, Any]:
    """Build the CWL ``Workflow`` dict for ``group``.

    Args:
        group: The agenda group to emit.
        leaf_workflow: Maps ``(name, sim)`` to the leaf's workflow reference
            (relative to the group directory).  Defaults to the per-leaf
            ``../<name>/workflow/workflow.cwl``.

    Returns:
        The CWL workflow as a plain dictionary.

    """
    reference = leaf_workflow or (lambda name, _sim: f"../{name}/{_LEAF_WORKFLOW}")
    steps: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    for name, entry in group.entries.items():
        if isinstance(entry, AgendaGroup):
            step = {
                "run": f"../{name}/{_GROUP_WORKFLOW}",
                "in": {},
                "out": list(_GROUP_OUTPUT_DEFS),
            }
        else:
            step = {
                "run": reference(name, entry),
                "in": {},
                "out": list(_GROUP_OUTPUT_DEFS),
            }
        for dep in _sibling_dependencies(group, name):
            # CWL expresses ordering only via data flow, so a dependency is a
            # step-input binding from the predecessor's completion output to a
            # well-known successor input (see DEPENDENCY_INPUT_PREFIX).
            step["in"][f"{DEPENDENCY_INPUT_PREFIX}{dep}"] = f"{dep}_step/submission_information"
            step.setdefault("doc", "Runs after its declared dependencies.")
        steps[f"{name}_step"] = step
        for suffix, (cwl_type, label) in _GROUP_OUTPUT_DEFS.items():
            outputs[f"{name}_{suffix}"] = {
                "type": cwl_type,
                "outputSource": f"{name}_step/{suffix}",
                "label": f"{name} {label}",
            }
    return {
        "cwlVersion": "v1.2",
        "class": "Workflow",
        "label": f"PIConGPU agenda group workflow: {group.name}",
        "doc": (
            "Overarching agenda workflow. Each step nests one self-contained "
            "per-simulation (or sub-group) workflow; declared agenda "
            "dependencies become CWL step dependencies."
        ),
        "requirements": {"SubworkflowFeatureRequirement": {}},
        "inputs": {},
        "outputs": outputs,
        "steps": steps,
    }


def _sibling_dependencies(group: AgendaGroup, name: str) -> list[str]:
    """Return the dependency names of ``name`` that are actual siblings.

    ``depends_on`` may name a path (``scan/run``); only entries that live
    directly in ``group`` become local step dependencies -- dependencies on
    deeper paths are the responsibility of the sub-group that contains them.

    Returns:
        The sibling dependency names, in declaration order.

    """
    entry = group.entries[name]
    siblings = set(group.entries)
    return [
        dep.split("/")[-1] for dep in entry.depends_on if dep.split("/")[-1] in siblings and dep.split("/")[-1] != name
    ]


def dump_cwl_workflow(workflow: dict[str, Any]) -> str:
    """Serialise a CWL workflow dict to YAML.

    Returns:
        The YAML text, with stable (insertion) key order.

    """
    return yaml.safe_dump(workflow, sort_keys=False, width=120)


__all__ = ["dump_cwl_workflow", "to_cwl_workflow"]
