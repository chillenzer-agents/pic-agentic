# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""The agenda declaration object: a serialisable, expanding simulation group.

This module is deliberately **extraction-ready**: it imports only the standard
library and ``pydantic``.  It must not import :mod:`pic_agentic`,
:mod:`picongpu`, ``cwltool`` or any transport, so the model can later move into
``pypicongpu`` (or a shared package) unchanged.

An *agenda* is a recursive tree of :class:`AgendaGroup` and :class:`AgendaSim`
nodes.  A group may carry a :class:`AgendaSweep` and can be expanded into
a larger tree, one leaf per sweep value.  Expansion is pure: it returns a new
group and never mutates the source, so an agenda can be serialised, expanded
incrementally and re-serialised at every step.

The leaf payload (``AgendaSim.spec``) is an opaque ``dict``: the agenda model
treats it as data and does not depend on whatever produced it (in this project
a ``pypicongpu.Runner`` spec).  That keeps the declaration layer independent of
the execution layer.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

#: Characters forbidden in agenda entry names; validated by
#: :func:`validate_entry_name` (a single source of truth for the rules).
_ILLEGAL_NAME_CHARS = ("/",)


def _validate_dependencies(deps: list[str]) -> list[str]:
    """Validate a ``depends_on`` list (names legal, no duplicates).

    Returns:
        The validated dependency list.

    Raises:
        ValueError: On a duplicate or illegal name.

    """
    if len(set(deps)) != len(deps):
        msg = f"duplicate names in depends_on: {deps}"
        raise ValueError(msg)
    for dep in deps:
        validate_entry_name(dep)
    return deps


def validate_entry_name(name: str) -> str:
    """Validate one group entry name.

    Args:
        name: The candidate name.

    Returns:
        The name, unchanged, when valid.

    Raises:
        ValueError: If the name is empty, starts with ``.``, or contains ``/``.

    """
    if not name:
        msg = "agenda entry names must be non-empty"
        raise ValueError(msg)
    if name.startswith("."):
        msg = f"agenda entry names must not start with '.': {name!r}"
        raise ValueError(msg)
    for char in _ILLEGAL_NAME_CHARS:
        if char in name:
            msg = f"agenda entry names must not contain {char!r}: {name!r}"
            raise ValueError(msg)
    return name


class AgendaSweep(BaseModel):
    """A one-dimensional parameter grid that an :class:`AgendaGroup` expands.

    A sweep is intentionally the *smallest useful* grid (one parameter, a list
    of values).  Multi-dimensional studies compose by nesting groups, which
    keeps the serialised form and the expansion logic trivial.
    """

    model_config = ConfigDict(extra="forbid")

    parameter: str
    values: list[float | int | str]
    unit: str | None = None

    @field_validator("parameter")
    @classmethod
    def _validate_parameter(cls, parameter: str) -> str:
        r"""Reject a sweep parameter that cannot form a legal entry name.

        The generated leaf name is ``<base>__<parameter>=<value>``, so a
        parameter containing ``/`` or a leading ``.`` would produce an illegal
        name.  Rejecting it here yields a clear sweep-level error.

        Returns:
            The validated parameter.

        Raises:
            ValueError: If the parameter cannot form a legal entry name.

        """
        if not parameter or parameter.startswith(".") or "/" in parameter:
            msg = f"sweep parameter must be a legal name component: {parameter!r}"
            raise ValueError(msg)
        return parameter

    @field_validator("values")
    @classmethod
    def _non_empty(cls, values: list[float | int | str]) -> list[float | int | str]:
        """Reject an empty sweep and non-finite numeric values.

        Non-finite values (``nan``/``inf``) cannot round-trip through JSON
        (pydantic serialises them as ``null``, which fails to rehydrate), so
        they are rejected at construction rather than corrupting a persisted
        agenda.

        Returns:
            The non-empty value list.

        Raises:
            ValueError: If no values are given or one is non-finite.

        """
        if not values:
            msg = "a sweep requires at least one value"
            raise ValueError(msg)
        for value in values:
            if isinstance(value, float) and not math.isfinite(value):
                msg = f"sweep values must be finite, got {value!r}"
                raise ValueError(msg)
        return values

    def assignment(self, value: float | str) -> dict[str, float | int | str]:
        """Return the ``{parameter: value}`` point assignment for one value.

        Returns:
            The single-entry parameter assignment.

        """
        return {self.parameter: value}

    def label(self, value: float | str) -> str:
        """Return the deterministic child-name suffix for one value.

        Uses ``repr`` so ``1``, ``1.0`` and ``1e0`` cannot collide, then strips
        characters that are illegal in an entry name.

        Returns:
            A filesystem/name-safe suffix such as ``laser.intensity=1e+18``.

        """
        text = repr(value)
        for char in _ILLEGAL_NAME_CHARS:
            text = text.replace(char, "_")
        return f"{self.parameter}={text}"


class AgendaSim(BaseModel):
    """One leaf of the agenda: a simulation specification plus its state.

    ``spec`` is the opaque execution payload (a ``pypicongpu.Runner`` spec in
    this project).  ``point`` records the sweep assignment that produced this
    leaf, so expanded agendas remain self-describing.  ``depends_on`` names
    sibling entries that must complete before this one (carried into CWL).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["sim"] = "sim"
    name: str
    spec: dict[str, Any]
    point: dict[str, float | int | str] | None = None
    status: Literal["planned", "submitted", "running", "done", "failed"] = "planned"
    depends_on: list[str] = Field(default_factory=list)
    #: The RCP simulation id once this leaf has been submitted (the engine's
    #: idempotency key: a leaf with a sim_id is never submitted twice).
    sim_id: str | None = None
    #: Whether this leaf must be approved before the engine submits it.
    requires_approval: bool = False
    #: Whether a human has pre-approved this leaf (set by the approval tool).
    approved: bool = False
    #: Core-hours the engine reserved for this leaf at submission time.
    estimated_core_hours: float = 0.0
    #: GPU-hours the engine reserved for this leaf at submission time.
    estimated_gpu_hours: float = 0.0
    #: Actual core-hours reported by the cluster on completion (gap 4).
    actual_core_hours: float | None = None
    #: Actual GPU-hours reported by the cluster on completion (gap 4).
    actual_gpu_hours: float | None = None
    #: Whether the leaf ran on a GPU (so actual GPU-hours are counted).
    is_gpu: bool = False
    #: Whether this leaf was satisfied by content-addressed reuse of an earlier
    #: identical run instead of being submitted.  A reused leaf accrues no
    #: estimated usage and is exempt from actual-cost reconciliation (the cost
    #: was accounted by the campaign that first ran it).
    reused: bool = False
    #: Human-readable reason this leaf failed, when known (e.g. the simclient's
    #: rejection message).  None for a failure whose cause was not observed.
    error: str | None = None
    #: Machine-readable failure code from the simclient (e.g. ``unsupported``).
    error_code: str | None = None
    #: Pipeline stage the failure occurred in (``build``/``prepare``/``submit``/
    #: ``run``), when the simclient reported one.
    stage: str | None = None
    #: The "successful-but-empty" health flag (F4): the all-zero warning text
    #: when the leaf completed but its only numeric artifact reads zero.  None
    #: when the leaf is not suspect or its health was never probed.  A leaf can
    #: be ``done`` *and* ``suspect`` at once - that is the whole point.
    suspect: str | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, name: str) -> str:
        """Validate the leaf name.

        Returns:
            The validated name.

        """
        return validate_entry_name(name)

    @field_validator("point")
    @classmethod
    def _finite_point(
        cls,
        point: dict[str, float | int | str] | None,
    ) -> dict[str, float | int | str] | None:
        """Reject non-finite values in a sweep point (JSON round-trip safety).

        Non-finite floats serialise to ``null`` and would make the persisted
        agenda unloadable.

        Returns:
            The validated point, or None.

        Raises:
            ValueError: If any numeric value is non-finite.

        """
        for key, value in (point or {}).items():
            if isinstance(value, float) and not math.isfinite(value):
                msg = f"point value for {key!r} must be finite, got {value!r}"
                raise ValueError(msg)
        return point

    @field_validator("depends_on")
    @classmethod
    def _validate_depends_on(cls, deps: list[str]) -> list[str]:
        """Validate dependency names and reject duplicates.

        Returns:
            The validated dependency list.

        """
        return _validate_dependencies(deps)


class AgendaGroup(BaseModel):
    """A recursive group of agenda entries.

    ``entries`` maps each child name to either an :class:`AgendaSim` or a
    nested :class:`AgendaGroup`.  ``sweep`` describes an expansion this group
    can undergo (see :meth:`expand`); ``depends_on`` names sibling entries that
    must complete first (carried into CWL as step dependencies).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["group"] = "group"
    name: str
    entries: dict[str, AgendaEntry] = Field(default_factory=dict)
    sweep: AgendaSweep | None = None
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, name: str) -> str:
        """Validate the group name.

        Returns:
            The validated name.

        """
        return validate_entry_name(name)

    @field_validator("depends_on")
    @classmethod
    def _validate_depends_on(cls, deps: list[str]) -> list[str]:
        """Validate dependency names and reject duplicates.

        Returns:
            The validated dependency list.

        """
        return _validate_dependencies(deps)

    def add(self, **entries: AgendaSim | AgendaGroup | dict[str, Any]) -> AgendaGroup:
        """Add children by name and return the expanded group.

        This is the documented expansion primitive::

            group = group.add(scan_a=AgendaGroup(name="scan_a", ...))

        Args:
            **entries: ``name -> AgendaSim | AgendaGroup | dict`` children.

        Returns:
            A new group with the children inserted (the receiver is unchanged).

        Raises:
            ValueError: On an empty/duplicate/illegal name.

        """
        updated = self.model_copy(deep=True)
        for name, entry in entries.items():
            validate_entry_name(name)
            if name in updated.entries:
                msg = f"agenda entry already exists: {name!r}"
                raise ValueError(msg)
            updated.entries[name] = _coerce_entry(name, entry)
        return updated

    def add_sim(
        self,
        *,
        name: str,
        spec: dict[str, Any],
        point: dict[str, float | int | str] | None = None,
    ) -> AgendaGroup:
        """Add one leaf simulation and return the expanded group.

        Returns:
            A new group with the leaf inserted.

        """
        return self.add(**{name: AgendaSim(name=name, spec=spec, point=point)})

    def simulations(self) -> list[tuple[str, AgendaSim]]:
        """Flatten the tree to ``(path, sim)`` pairs, depth-first.

        The order is deterministic (insertion order per level), so serialised
        agendas and their expansions compare stably.

        Returns:
            The flattened leaves.

        """
        return list(_walk_leaves(self, ""))

    def expand(
        self,
        sweep: AgendaSweep,
        synthesized: Callable[[dict[str, float | int | str]], dict[str, Any]],
        *,
        base_name: str | None = None,
    ) -> AgendaGroup:
        """Return a new group with one leaf per sweep value.

        Pure: the receiver is not mutated.  Each generated leaf is named
        ``<base>__<parameter>=<value>`` and holds ``synthesized(point)`` as its
        spec, with the point recorded on the leaf.

        Args:
            sweep: The parameter grid to expand.
            synthesized: Maps a point assignment to a leaf spec.
            base_name: Optional base for the generated names (defaults to the
                group's own name).

        Returns:
            The expanded group (leaves merged into a copy of this group's
            entries).

        Raises:
            ValueError: If a generated leaf name is illegal or would overwrite
                an existing entry.

        """
        base = base_name or self.name
        validate_entry_name(base)
        expanded = self.model_copy(deep=True)
        for value in sweep.values:
            point = sweep.assignment(value)
            leaf_name = f"{base}__{sweep.label(value)}"
            validate_entry_name(leaf_name)
            if leaf_name in expanded.entries:
                msg = f"expand would overwrite an existing entry: {leaf_name!r}"
                raise ValueError(msg)
            expanded.entries[leaf_name] = AgendaSim(name=leaf_name, spec=synthesized(point), point=point)
        return expanded


#: Discriminated union so nested groups and leaves round-trip unambiguously.
AgendaEntry = Annotated[AgendaSim | AgendaGroup, Field(discriminator="kind")]

AgendaGroup.model_rebuild()


def _coerce_entry(name: str, entry: AgendaSim | AgendaGroup | dict[str, Any]) -> AgendaSim | AgendaGroup:
    """Coerce an ``add`` argument into an agenda entry with the right name.

    Args:
        name: The entry's key (used when the entry does not carry its own name).
        entry: An ``AgendaSim``/``AgendaGroup``/plain mapping.

    Returns:
        The validated entry.

    """
    if isinstance(entry, (AgendaSim, AgendaGroup)):
        # Deep-copy so the returned agenda shares no mutable state with the
        # caller's entry (e.g. its ``spec``): mutating the argument afterwards
        # must not corrupt the agenda.
        copy = entry.model_copy(deep=True)
        if copy.name != name:
            copy = copy.model_copy(update={"name": name})
        return copy
    # A plain dict: the discriminant decides the concrete model.
    kind = entry.get("kind", "sim")
    payload = {**entry, "name": name}
    model = AgendaGroup if kind == "group" else AgendaSim
    return model.model_validate(payload)


def _walk_leaves(group: AgendaGroup, prefix: str) -> Iterator[tuple[str, AgendaSim]]:
    """Depth-first walk yielding ``(path, sim)`` for every leaf.

    Yields:
        The ``(path, sim)`` pairs in deterministic insertion order.

    """
    for name, entry in group.entries.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(entry, AgendaGroup):
            yield from _walk_leaves(entry, path)
        else:
            yield path, entry


__all__ = [
    "AgendaEntry",
    "AgendaGroup",
    "AgendaSim",
    "AgendaSweep",
    "validate_entry_name",
]
