# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Content-addressed reuse of identical simulations.

This is the "same spec, reuse the result" feature.  A simulation's ``sim_id`` is
only the first 8 hex of its payload hash (32 bits), so two *different* specs can
share a ``sim_id``; it is a label, not a cache key.  This module keys reuse on
the **full** wire hash instead and records only runs that finished successfully,
so a leaf whose spec is byte-identical to a run that already completed can be
*linked* to that run instead of being submitted again.

The registry is plain pydantic data persisted atomically by
:class:`~pic_agentic.agenda.store.AgendaStore`, so it survives restarts.  The
engine consults it through two optional callables (lookup/record); the wiring
lives in the server so the offline engine tests stay free of any I/O.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

#: Terminal states that make a recorded run reusable.
REUSABLE_STATES = frozenset({"done"})

#: Default file name for a serialised reuse registry.
DEFAULT_REUSE_FILE = "reuse-registry.json"


class ReuseRecord(BaseModel):
    """One completed run eligible for reuse, keyed by its full wire hash."""

    model_config = ConfigDict(extra="forbid")

    wire_hash: str
    sim_id: str
    state: str
    run_dir: str | None = None
    ts: str | None = None


class ReuseRegistry(BaseModel):
    """A ``wire_hash -> ReuseRecord`` map persisted atomically."""

    model_config = ConfigDict(extra="forbid")

    records: dict[str, ReuseRecord] = Field(default_factory=dict)

    def lookup(self, wire_hash: str) -> ReuseRecord | None:
        """Return the reusable record for ``wire_hash``, if any.

        Returns:
            The record when it exists and its state is reusable, else None.

        """
        record = self.records.get(wire_hash)
        if record is None or record.state not in REUSABLE_STATES:
            return None
        return record

    def remember(self, record: ReuseRecord) -> ReuseRegistry:
        """Return a copy with ``record`` associated with its wire hash.

        Re-recording a hash overwrites the previous entry (last-write-wins),
        which is what makes replay/backfill idempotent.

        Returns:
            The updated registry (a new object; the input is not mutated).

        """
        return self.model_copy(update={"records": {**self.records, record.wire_hash: record}})


__all__ = ["DEFAULT_REUSE_FILE", "REUSABLE_STATES", "ReuseRecord", "ReuseRegistry"]
