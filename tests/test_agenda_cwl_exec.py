# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for the real CWL execution of an agenda (gap 1)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from pic_agentic.agenda.cwl_exec import manifest_for, run_agenda_cwl
from pic_agentic.agenda.model import AgendaGroup

_HAS_CWLTool = importlib.util.find_spec("cwltool") is not None


def _two_leaf_group() -> AgendaGroup:
    group = AgendaGroup(name="study")
    group = group.add_sim(name="alpha", spec={"sim": {"replica": 0}})
    return group.add_sim(name="beta", spec={"sim": {"replica": 1}})


def test_manifest_lists_leaves() -> None:
    manifest = manifest_for(_two_leaf_group())
    assert manifest["leaves"] == ["alpha", "beta"]
    assert "class: Workflow" in manifest["workflow"]


def test_run_reports_unavailable_without_cwltool(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("pic_agentic.agenda.cwl_exec.cwltool_available", lambda: False)
    result = run_agenda_cwl(_two_leaf_group(), root=tmp_path / "tree", leaf_command=lambda *_: ["/bin/true"])
    assert result["ok"] is False
    assert result["available"] is False
    assert "cwltool" in result["error"]


@pytest.mark.skipif(not _HAS_CWLTool, reason="cwltool not installed")
def test_run_executes_each_leaf(tmp_path: Path) -> None:
    """A real cwltool run materialises and executes every leaf."""
    ran: list[str] = []

    def leaf_command(name: str, _sim: object) -> list[str]:
        return ["/bin/sh", "-c", f"echo {name} > {name}.submission; echo done"]

    result = run_agenda_cwl(
        _two_leaf_group(),
        root=tmp_path / "tree",
        leaf_command=leaf_command,
        exec_dir=tmp_path / "exec",
    )
    assert result["available"] is True
    assert result["ok"] is True, result["log_tail"]
    assert (tmp_path / "tree" / "study.cwl").is_file()
    assert (tmp_path / "tree" / "alpha.cwl").is_file()
    assert (tmp_path / "tree" / "beta.cwl").is_file()
    _ = ran


@pytest.mark.skipif(not _HAS_CWLTool, reason="cwltool not installed")
def test_run_reports_a_failing_step(tmp_path: Path) -> None:
    def leaf_command(_name: str, _sim: object) -> list[str]:
        return ["/bin/false"]

    result = run_agenda_cwl(
        _two_leaf_group(),
        root=tmp_path / "tree",
        leaf_command=leaf_command,
        exec_dir=tmp_path / "exec",
    )
    assert result["ok"] is False
    assert result["error"]
