# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Content-addressed reuse of identical simulations.

This is the "same spec, reuse the result" feature.  A simulation's ``sim_id`` is
only the first 8 hex of its payload hash (32 bits), so two *different* specs can
share a ``sim_id``; it is a label, not a cache key.  This module keys reuse on
the **full** content key instead and records only runs that finished
successfully, so a leaf whose content key matches a run that already completed
can be *linked* to that run instead of being submitted again.

The content key requires identical **content, provenance and sweep point**
(option A): two byte-identical specs at *different* sweep points (e.g.
``point={"x": 1.0}`` and ``point={"x": 2.0}``) are different simulations and
must not reuse one another, otherwise a leaf's provenance could attribute its
result to a point that never ran.  A direct ``submit_simulation`` has no sweep
point and therefore keys under the point-less key; it never matches a
point-carrying campaign leaf.  Finer-grained reuse (e.g. point-insensitive reuse
of the same physics) is deliberately out of scope here and left to future work.

The registry is plain pydantic data persisted atomically by
:class:`~pic_agentic.agenda.store.AgendaStore`, so it survives restarts.  The
engine consults it through two optional callables (lookup/record); the wiring
lives in the server so the offline engine tests stay free of any I/O.

A record is written in **two phases**.  A direct ``submit_simulation`` knows the
spec's content key and the run's batch identity the moment the simclient accepts
it, but its result is not ready until the later ``results.ready`` event -- and
that event may arrive *after* the server restarted, so it cannot be correlated
from an in-memory table.  The pending entry therefore stores the run-batch
identity, and :meth:`ReuseRegistry.promote` flips it to the reusable ``done``
state when the matching ``results.ready`` is observed (in the same pass that
handles the event).  Only ``done`` is reusable, so a run whose result never
arrives is never reused.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

#: Terminal states that make a recorded run reusable.
REUSABLE_STATES = frozenset({"done"})

#: A run that has been accepted but whose result is not ready yet.  The entry
#: records the run-batch identity so a later ``results.ready`` can promote it to
#: :data:`REUSABLE_STATES`; it is never returned by :meth:`ReuseRegistry.lookup`.
PENDING_STATE = "pending"

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
    #: The run-batch identity that accepted this spec (the submission's stable
    #: command id).  A ``pending`` entry keeps it so the matching
    #: ``results.ready`` event can be attributed to this record when the server
    #: re-projects the signed room after a restart; the ``done`` entry keeps it
    #: for provenance.
    run_id: str | None = None
    #: Run-batch identities still awaiting their ``results.ready``.  A second
    #: identical submission must not repoint :attr:`run_id` at the new, unready
    #: run: the completed run would be orphaned (its event could no longer
    #: promote the record) *and* the surfaced identity would name a run that may
    #: never finish.  The pending ids are accumulated so **any** of the
    #: outstanding runs can promote the record once it completes; ``run_id``
    #: itself always names the run that actually produced the reusable result.
    pending_run_ids: list[str] = Field(default_factory=list)


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

    def record_direct(self, record: ReuseRecord) -> ReuseRegistry:
        """Record an accepted direct submission without orphaning a result.

        Direct ``submit_simulation`` calls can arrive back to back for the *same
        spec*: a ``submit -> inspect -> re-submit`` iteration, or two identical
        study points.  A naive last-write-wins would repoint the entry's
        :attr:`ReuseRecord.run_id` at the newest, still-unfinished run, so the
        already-completed run's ``results.ready`` could no longer promote the
        record (its id no longer matches) and the surfaced run identity would
        name a run that may never finish.  Instead:

        - no existing record: store the new ``pending`` record;
        - existing ``pending``: accumulate the run id so **any** of the
          outstanding runs can promote the record, whichever completes first;
        - existing reusable (``done``): keep it untouched -- a later run of the
          same content must not demote a completed result.

        Returns:
            The updated registry (a new object; the input is not mutated).

        """
        existing = self.records.get(record.wire_hash)
        if existing is None:
            return self.remember(record)
        if existing.state in REUSABLE_STATES:
            return self
        if record.state in REUSABLE_STATES:
            # The ack and the run's ``results.ready`` arrived back to back, so
            # the incoming record is already promotable: promote the existing
            # entry with the run that actually produced the result.
            promoted = existing.model_copy(
                update={"state": record.state, "run_id": record.run_id, "pending_run_ids": []},
            )
            return self.remember(promoted)
        pending_ids = list(existing.pending_run_ids)
        for run_id in (existing.run_id, record.run_id):
            if run_id and run_id not in pending_ids:
                pending_ids.append(run_id)
        if pending_ids == existing.pending_run_ids:
            return self
        merged = existing.model_copy(update={"pending_run_ids": pending_ids})
        return self.remember(merged)

    def promote(self, run_id: str, *, state: str = "done") -> ReuseRegistry:
        """Mark the pending record accepted by ``run_id`` as reusable.

        Called when a ``results.ready`` event arrives: the event carries the
        submission's stable command id (the run-batch identity), not the spec,
        so the matching record is found by identity rather than by re-deriving
        the content key.  A record may have accumulated several pending run ids
        (identical repeated submissions); the promoting run becomes the
        record's :attr:`ReuseRecord.run_id` -- the run that actually produced
        the reusable result -- and the remaining pending ids are cleared.  An
        already-``done`` record is left untouched, so a replayed event is
        idempotent.

        Args:
            run_id: The run-batch identity (the submission's command id).
            state: The terminal state to promote the record to.

        Returns:
            The updated registry (a new object; the input is not mutated), or
            ``self`` unchanged when no pending record matches.

        """
        if not run_id:
            return self
        for wire_hash, record in self.records.items():
            if record.state != PENDING_STATE:
                continue
            candidates = {record.run_id, *record.pending_run_ids}
            if run_id not in candidates:
                continue
            promoted = record.model_copy(
                update={"state": state, "run_id": run_id, "pending_run_ids": []},
            )
            return self.model_copy(update={"records": {**self.records, wire_hash: promoted}})
        return self


__all__ = [
    "DEFAULT_REUSE_FILE",
    "PENDING_STATE",
    "REUSABLE_STATES",
    "ReuseRecord",
    "ReuseRegistry",
]
