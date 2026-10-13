# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for the durable agenda store (atomic save/load, expansion workflow)."""

from __future__ import annotations

import json

import pytest

from pic_agentic.agenda.model import AgendaGroup, AgendaSim, AgendaSweep
from pic_agentic.agenda.store import AgendaStore


def _sim(name: str) -> AgendaSim:
    return AgendaSim(name=name, spec={"sim": {"time_steps": 5}})


def test_save_load_round_trip(tmp_path) -> None:
    store = AgendaStore(tmp_path)
    group = AgendaGroup(name="g").add(a=_sim("a"))
    store.save(group)
    assert store.exists()
    assert store.load().model_dump_json() == group.model_dump_json()


def test_default_path_and_missing_file(tmp_path) -> None:
    store = AgendaStore(tmp_path)
    assert store.path == tmp_path / "agenda.json"
    with pytest.raises(FileNotFoundError):
        store.load()


def test_save_is_atomic_no_temp_left_behind(tmp_path) -> None:
    store = AgendaStore(tmp_path)
    store.save(AgendaGroup(name="g").add(a=_sim("a")))
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "agenda.json"]
    assert leftovers == []


def test_expand_then_resave_workflow(tmp_path) -> None:
    """The documented agenda lifecycle: save, load, expand, re-save."""
    store = AgendaStore(tmp_path)
    store.save(AgendaGroup(name="scan").add(seed=_sim("seed")))

    loaded = store.load()
    expanded = loaded.expand(AgendaSweep(parameter="i", values=[1, 2]), lambda p: {"sim": {"i": p["i"]}})
    store.save(expanded)

    final = store.load()
    paths = [path for path, _ in final.simulations()]
    assert paths == ["seed", "scan__i=1", "scan__i=2"]
    # The persisted JSON is valid and self-describing.
    raw = json.loads((tmp_path / "agenda.json").read_text())
    assert raw["entries"]["scan__i=2"]["point"] == {"i": 2}


def test_store_survives_restart_by_reloading(tmp_path) -> None:
    """A fresh store object over the same path sees the persisted state."""
    AgendaStore(tmp_path).save(AgendaGroup(name="g").add(a=_sim("a")))
    reopened = AgendaStore(tmp_path)
    assert [p for p, _ in reopened.load().simulations()] == ["a"]
