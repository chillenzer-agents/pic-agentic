# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for the agenda -> CWL emitter."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest
import yaml

from pic_agentic.agenda.cwl import dump_cwl_workflow, to_cwl_workflow
from pic_agentic.agenda.model import AgendaGroup, AgendaSim


def _leaf(name: str, **kw: object) -> AgendaSim:
    return AgendaSim(name=name, spec={"sim": {}}, **kw)


def _group() -> AgendaGroup:
    group = AgendaGroup(name="study")
    group = group.add(alpha=_leaf("alpha"))
    group = group.add(beta=_leaf("beta", depends_on=["alpha"]))
    return group.add(sub=AgendaGroup(name="sub").add(gamma=_leaf("gamma")))


def test_workflow_shape_and_nesting() -> None:
    wf = to_cwl_workflow(_group())
    assert wf["cwlVersion"] == "v1.2"
    assert wf["class"] == "Workflow"
    assert wf["requirements"] == {"SubworkflowFeatureRequirement": {}}
    assert set(wf["steps"]) == {"alpha_step", "beta_step", "sub_step"}
    assert wf["steps"]["alpha_step"]["run"] == "../alpha/workflow/workflow.cwl"
    assert wf["steps"]["sub_step"]["run"] == "../sub/workflow/group_workflow.cwl"


def test_dependency_becomes_cwl_step_input() -> None:
    wf = to_cwl_workflow(_group())
    assert wf["steps"]["beta_step"]["in"] == {"depends_on_alpha": "alpha_step/submission_information"}
    # alpha has no declared dependency.
    assert wf["steps"]["alpha_step"]["in"] == {}


def test_outputs_are_reexposed_per_child() -> None:
    wf = to_cwl_workflow(_group())
    assert wf["outputs"]["alpha_input_directory"]["outputSource"] == "alpha_step/input_directory"
    assert wf["outputs"]["sub_submission_information"]["outputSource"] == "sub_step/submission_information"


def test_dependency_on_non_sibling_is_ignored() -> None:
    group = AgendaGroup(name="g").add(leaf=_leaf("leaf", depends_on=["nonexistent"]))
    wf = to_cwl_workflow(group)
    assert wf["steps"]["leaf_step"]["in"] == {}


def test_dump_is_valid_yaml_and_round_trips() -> None:
    wf = to_cwl_workflow(_group())
    text = dump_cwl_workflow(wf)
    assert yaml.safe_load(text) == wf


def test_leaf_workflow_reference_is_overridable() -> None:
    group = AgendaGroup(name="g").add(a=_leaf("a"))
    wf = to_cwl_workflow(group, leaf_workflow=lambda name, _sim: f"${{inputs.{name}}}")
    assert wf["steps"]["a_step"]["run"] == "${inputs.a}"


@pytest.mark.skipif(importlib.util.find_spec("cwltool") is None, reason="cwltool not installed")
def test_cwl_validates_with_cwltool(tmp_path) -> None:
    """The emitted group workflow is accepted by ``cwltool --validate``.

    Stub leaf workflows are written so the nested references resolve and the
    dependency-input contract (:data:`DEPENDENCY_INPUT_PREFIX`) is satisfied,
    exercising a clean validation rather than only a structural parse.
    """
    group = AgendaGroup(name="g").add(
        a=_leaf("a"),
        b=_leaf("b", depends_on=["a"]),
    )
    leaf_stub = {
        "cwlVersion": "v1.2",
        "class": "CommandLineTool",
        "baseCommand": ["/bin/echo"],
        "inputs": {
            "depends_on_a": {"type": "File?", "default": None},
            "msg": {"type": "string", "default": "x"},
        },
        "outputs": {
            "input_directory": {"type": "Directory", "outputBinding": {"glob": "."}},
            "submission_information": {"type": "File", "outputBinding": {"glob": "*.submission"}},
        },
    }
    (tmp_path / "workflow").mkdir()
    wf_path = tmp_path / "workflow" / "group_workflow.cwl"
    wf_path.write_text(dump_cwl_workflow(to_cwl_workflow(group)))
    for name in ("a", "b"):
        (tmp_path / name / "workflow").mkdir(parents=True)
        (tmp_path / name / "workflow" / "workflow.cwl").write_text(yaml.safe_dump(leaf_stub))

    proc = subprocess.run(
        [sys.executable, "-m", "cwltool", "--validate", str(wf_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    assert "is valid CWL" in combined
    assert "is not a valid CWL" not in combined
