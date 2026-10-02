# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Unit tests for the agenda model (round-trip, expansion, validation)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from pic_agentic.agenda.model import (
    AgendaGroup,
    AgendaSim,
    AgendaSweep,
    readable_label,
    validate_entry_name,
)


def _leaf(name: str, **kw: object) -> AgendaSim:
    return AgendaSim(name=name, spec={"sim": {"time_steps": 10}}, **kw)


def test_recursive_round_trip_is_byte_stable() -> None:
    group = AgendaGroup(name="study")
    group = group.add(single=_leaf("single"))
    group = group.add(scan=AgendaGroup(name="scan", sweep=AgendaSweep(parameter="intensity", values=[1e18, 2e18])))
    group = group.add(diag=_leaf("diag", depends_on=["scan"], point={"intensity": 1e18}))

    dumped = group.model_dump_json()
    reloaded = AgendaGroup.model_validate_json(dumped)
    assert reloaded.model_dump_json() == dumped
    # A second generation is stable too.
    assert AgendaGroup.model_validate_json(reloaded.model_dump_json()).model_dump_json() == dumped


def test_discriminated_union_rehydrates_nested_types() -> None:
    group = AgendaGroup(name="g")
    group = group.add(sub=AgendaGroup(name="sub"))
    group = group.add(leaf=_leaf("leaf"))
    reloaded = AgendaGroup.model_validate_json(group.model_dump_json())
    assert isinstance(reloaded.entries["sub"], AgendaGroup)
    assert isinstance(reloaded.entries["leaf"], AgendaSim)


def test_add_returns_new_group_and_does_not_mutate() -> None:
    base = AgendaGroup(name="g")
    expanded = base.add(a=_leaf("a"))
    assert "a" not in base.entries
    assert "a" in expanded.entries


def test_add_rejects_duplicate_and_bad_names() -> None:
    group = AgendaGroup(name="g").add(a=_leaf("a"))
    with pytest.raises(ValueError, match="already exists"):
        group.add(a=_leaf("a"))
    with pytest.raises(ValueError, match="non-empty"):
        group.add(**{"": _leaf("x")})
    with pytest.raises(ValueError, match="must not start with"):
        group.add(**{".hidden": _leaf("x")})
    with pytest.raises(ValueError, match="must not contain"):
        group.add(**{"has/slash": _leaf("x")})


def test_validate_entry_name_rules() -> None:
    assert validate_entry_name("ok-name_1") == "ok-name_1"
    for bad, match in (("", "non-empty"), (".x", "must not start with"), ("a/b", "must not contain")):
        with pytest.raises(ValueError, match=match):
            validate_entry_name(bad)


def test_depends_on_rejects_duplicates() -> None:
    with pytest.raises(ValidationError):
        _leaf("x", depends_on=["a", "a"])


def test_simulations_are_flattened_depth_first_in_order() -> None:
    group = AgendaGroup(name="g")
    group = group.add(z=_leaf("z"))
    group = group.add(inner=AgendaGroup(name="inner").add(m=_leaf("m"), n=_leaf("n")))
    group = group.add(a=_leaf("a"))
    assert [path for path, _ in group.simulations()] == ["z", "inner/m", "inner/n", "a"]


def test_expand_is_pure_and_assigns_points() -> None:
    base = AgendaGroup(name="scan").add(seed=_leaf("seed"))
    sweep = AgendaSweep(parameter="intensity", values=[1e18, 2e18, 3e18])
    expanded = base.expand(sweep, lambda point: {"sim": {"intensity": point["intensity"]}})

    leaves = expanded.simulations()
    generated = [(p, s) for p, s in leaves if p != "seed"]
    assert len(generated) == 3
    for _path, sim in generated:
        assert sim.spec["sim"]["intensity"] in {1e18, 2e18, 3e18}
        assert sim.point == {"intensity": sim.spec["sim"]["intensity"]}
        # The sweep records the readable parameter name alongside the point.
        assert sim.sweep_parameter == "intensity"
    # Source untouched.
    assert [p for p, _ in base.simulations()] == ["seed"]


def test_readable_label_preserves_unicode_and_drops_control_chars() -> None:
    """The label sanitiser keeps printable Unicode but strips control runs."""
    assert readable_label("focus y [μm]") == "focus y [μm]"
    assert readable_label("焦距") == "焦距"
    assert readable_label("a\x00b") == "a b"
    assert readable_label("  padded  ") == "padded"
    # An empty or all-whitespace label means "absent", not "".
    assert readable_label("") is None
    assert readable_label("\n\t") is None


def test_expand_sanitises_the_sweep_parameter() -> None:
    """``expand`` runs the caller's parameter through the display sanitiser."""
    base = AgendaGroup(name="scan")
    sweep = AgendaSweep(parameter="a\x00b", values=[1])
    expanded = base.expand(sweep, lambda point: {"sim": point})
    _path, sim = expanded.simulations()[0]
    assert sim.sweep_parameter == "a b"


def test_expand_names_are_distinct_for_int_vs_float() -> None:
    sweep = AgendaSweep(parameter="n", values=[1, 1.0])
    # repr() distinguishes the values, so the child names never collide.
    assert sweep.label(1) != sweep.label(1.0)


def test_expand_rejects_empty_sweep() -> None:
    with pytest.raises(ValidationError):
        AgendaSweep(parameter="x", values=[])


def test_add_sim_convenience() -> None:
    group = AgendaGroup(name="g").add_sim(name="only", spec={"sim": {}})
    assert isinstance(group.entries["only"], AgendaSim)


def test_plain_dict_coercion_uses_kind_discriminator() -> None:
    group = AgendaGroup(name="g").add(
        leaf={"kind": "sim", "spec": {"sim": {}}},
        sub={"kind": "group"},
    )
    assert isinstance(group.entries["leaf"], AgendaSim)
    assert isinstance(group.entries["sub"], AgendaGroup)


def test_spec_is_opaque_and_json_serialisable() -> None:
    group = AgendaGroup(name="g").add(a=_leaf("a"))
    payload = json.loads(group.model_dump_json())
    assert payload["entries"]["a"]["spec"] == {"sim": {"time_steps": 10}}


def test_non_finite_point_is_rejected() -> None:
    """nan/inf cannot round-trip through JSON, so they are refused up front."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError, match="finite"):
            AgendaSim(name="s", spec={}, point={"p": bad})


def test_non_finite_sweep_value_is_rejected() -> None:
    with pytest.raises(ValidationError, match="finite"):
        AgendaSweep(parameter="p", values=[float("inf")])


def test_illegal_sweep_parameter_is_rejected() -> None:
    for bad in ("", ".p", "a/b"):
        with pytest.raises(ValidationError, match="legal name component"):
            AgendaSweep(parameter=bad, values=[1])


def test_add_deep_copies_the_caller_entry() -> None:
    """Mutating a caller's entry after add must not corrupt the agenda."""
    entry = AgendaSim(name="x", spec={"nested": {"v": 1}})
    group = AgendaGroup(name="r").add(x=entry)
    entry.spec["nested"]["v"] = 999
    assert group.entries["x"].spec == {"nested": {"v": 1}}


def test_expand_rejects_colliding_leaf_name() -> None:
    base = AgendaGroup(name="scan").add(**{"scan__i=1": _leaf("scan__i=1")})
    with pytest.raises(ValueError, match="overwrite"):
        base.expand(AgendaSweep(parameter="i", values=[1]), lambda point: {"sim": point})


def test_expand_rejects_illegal_leaf_name() -> None:
    base = AgendaGroup(name="scan")
    # A base name that yields an illegal leaf segment is rejected.
    with pytest.raises(ValueError, match="must not start with"):
        base.expand(AgendaSweep(parameter="i", values=[1]), lambda point: {"sim": point}, base_name=".bad")
