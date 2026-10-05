# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Milestone D: aggregate fleet observability over the sim registry.

A pure, deterministic projection of the server's simulation registry into a
compact summary plus a list of alerts the agent can act on.  The module takes
opaque records (any object exposing the :class:`SimRecord` attributes) rather
than importing the server layer, so it stays offline-testable and free of a
server→fleet import cycle.

Time is always injected (``now``/``stall_after_s``), never read from the clock
inside: the functions are pure and their tests are not time-dependent.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pic_agentic.protocol.simulation import SimulationPhase, simulation_phase

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pic_agentic.server.simulation import SimRecord

#: Alert kinds the fleet view can raise.
AlertKind = Literal["stalled", "failed", "cancelled", "nonzero_exit", "suspect"]


class FleetSummary(BaseModel):
    """A compact, serialisable aggregate over the fleet."""

    model_config = ConfigDict(extra="forbid")

    total: int = 0
    active: int = 0
    terminal: int = 0
    running: int = 0
    done: int = 0
    failed: int = 0
    #: Done runs whose only numeric artifact reads all-zero (F4): a
    #: "successful-but-empty" health signal, distinct from a failure.
    suspect: int = 0
    by_state: dict[str, int] = Field(default_factory=dict)
    #: How many records are in each coarse run phase (H4: ``building``/``queued``
    #: make a long pre-``job_id`` wait visible instead of looking wedged).
    by_phase: dict[str, int] = Field(default_factory=dict)
    #: Mean progress of the records that report a ``percent``, else None.
    aggregate_percent: float | None = None


class FleetAlert(BaseModel):
    """One actionable condition in the fleet."""

    model_config = ConfigDict(extra="forbid")

    sim_id: str
    kind: AlertKind
    detail: str
    last_event_ts: str | None = None


#: States that mean a run finished successfully.
_DONE_STATES = frozenset({"results.ready", "simulation.job_finished"})
#: States that mean a run failed.
_FAILED_STATES = frozenset({"simulation.failed", "simulation.job_failed"})

#: The valid phase values; a duck-typed record's ``phase`` is trusted only when
#: it is one of these, so a wrong string cannot leak into ``by_phase``.
_KNOWN_PHASES = frozenset(member.value for member in SimulationPhase)


def fleet_summary(records: Sequence[SimRecord] | Iterable[SimRecord]) -> FleetSummary:
    """Aggregate the fleet into counts and a mean progress percentage.

    Args:
        records: The registry records.

    Returns:
        The summary; ``aggregate_percent`` is None when no record reports one.

    """
    records = list(records)
    by_state: dict[str, int] = {}
    by_phase: dict[str, int] = {}
    percents: list[int] = []
    summary = FleetSummary()
    for record in records:
        state = str(getattr(record, "state", "") or "")
        by_state[state] = by_state.get(state, 0) + 1
        phase = _phase_of(record)
        by_phase[phase] = by_phase.get(phase, 0) + 1
        active = bool(getattr(record, "active", False))
        summary.total += 1
        if active:
            summary.active += 1
        else:
            summary.terminal += 1
        if state in _DONE_STATES:
            summary.done += 1
            if getattr(record, "suspect", None):
                summary.suspect += 1
        elif state in _FAILED_STATES or state == "simulation.cancelled":
            summary.failed += 1
        if active and state not in _DONE_STATES and state not in _FAILED_STATES:
            summary.running += 1
        percent = getattr(record, "percent", None)
        if isinstance(percent, int):
            percents.append(percent)
    summary.by_state = dict(sorted(by_state.items()))
    summary.by_phase = dict(sorted(by_phase.items()))
    summary.aggregate_percent = sum(percents) / len(percents) if percents else None
    return summary


def detect_alerts(
    records: Sequence[SimRecord] | Iterable[SimRecord],
    *,
    now: datetime,
    stall_after_s: float,
) -> list[FleetAlert]:
    """Return the fleet's actionable conditions in deterministic sim_id order.

    Args:
        records: The registry records.
        now: The reference time (injected, so the function is pure).
        stall_after_s: How long a **running** record may go without a lifecycle
            event before it is reported as ``stalled``.  Build/queue records
            (no running event yet) are exempt: they legitimately go many
            minutes without an event (H3).

    Returns:
        One alert per condition; a record can raise at most one (failure kinds
        take precedence over ``stalled``).

    """
    alerts: list[FleetAlert] = []
    for record in sorted(records, key=lambda item: str(getattr(item, "sim_id", ""))):
        sim_id = str(getattr(record, "sim_id", ""))
        state = str(getattr(record, "state", "") or "")
        last_event_ts = getattr(record, "last_event_ts", None)
        exit_code = getattr(record, "exit_code", None)
        if state in _FAILED_STATES:
            alerts.append(FleetAlert(sim_id=sim_id, kind="failed", detail=state, last_event_ts=last_event_ts))
            continue
        if state == "simulation.cancelled":
            alerts.append(FleetAlert(sim_id=sim_id, kind="cancelled", detail=state, last_event_ts=last_event_ts))
            continue
        if not bool(getattr(record, "active", False)) and isinstance(exit_code, int) and exit_code != 0:
            alerts.append(
                FleetAlert(
                    sim_id=sim_id,
                    kind="nonzero_exit",
                    detail=f"exit_code={exit_code}",
                    last_event_ts=last_event_ts,
                ),
            )
            continue
        # A "successful-but-empty" run is done (not failed), so it is reported
        # after the hard-failure kinds: the run must be neither failed, cancelled
        # nor a non-zero exit to reach here.
        suspect = getattr(record, "suspect", None)
        if state in _DONE_STATES and suspect:
            alerts.append(FleetAlert(sim_id=sim_id, kind="suspect", detail=str(suspect), last_event_ts=last_event_ts))
            continue
        stalled = _is_stalled(record, now=now, stall_after_s=stall_after_s)
        if stalled is not None:
            alerts.append(
                FleetAlert(
                    sim_id=sim_id,
                    kind="stalled",
                    detail=f"no event for {stalled:g}s (limit {stall_after_s:g}s)",
                    last_event_ts=last_event_ts,
                ),
            )
    return alerts


def fleet_view(
    records: Sequence[SimRecord] | Iterable[SimRecord],
    *,
    now: datetime,
    stall_after_s: float,
) -> dict[str, Any]:
    """Compose the summary and alerts into the tool's serialisable shape.

    Returns:
        ``{"summary": {...}, "alerts": [...]}``.

    """
    records = list(records)
    return {
        "summary": fleet_summary(records).model_dump(),
        "alerts": [alert.model_dump() for alert in detect_alerts(records, now=now, stall_after_s=stall_after_s)],
    }


def _phase_of(record: SimRecord) -> str:
    """Return a record's coarse run phase, deriving it if the object lacks one.

    Records are duck-typed here, so a test double or an older projection may not
    carry the computed :attr:`~pic_agentic.server.simulation.SimRecord.phase`;
    in that case derive the phase from ``state``/``job_id`` with the same rule
    the record uses.

    Returns:
        One of the :class:`~pic_agentic.protocol.simulation.SimulationPhase`
        values.

    """
    phase = getattr(record, "phase", None)
    if isinstance(phase, str) and phase in _KNOWN_PHASES:
        return phase
    return simulation_phase(
        str(getattr(record, "state", "") or ""),
        getattr(record, "job_id", None),
    )


def _is_stalled(record: SimRecord, *, now: datetime, stall_after_s: float) -> float | None:
    """Return the idle seconds of a stalled active record, else None.

    Only a record in the ``running`` phase can be stalled (H3): the window
    between ``accepted`` and the SLURM job legitimately runs for many minutes
    with no lifecycle event (the local CWL build, then the queue wait), so a
    ``building``/``queued`` record is never reported as wedged.  A record
    without a parseable ``last_event_ts`` cannot be judged and is not reported.

    Returns:
        The idle seconds when the record is stalled, else None.

    """
    if not bool(getattr(record, "active", False)):
        return None
    if _phase_of(record) != SimulationPhase.RUNNING.value:
        return None
    last_event_ts = getattr(record, "last_event_ts", None)
    if not last_event_ts:
        return None
    try:
        seen = datetime.fromisoformat(str(last_event_ts))
    except ValueError:
        return None
    if seen.tzinfo is None:
        seen = seen.replace(tzinfo=now.tzinfo)
    idle = (now - seen).total_seconds()
    return idle if idle > stall_after_s else None


__all__ = [
    "AlertKind",
    "FleetAlert",
    "FleetSummary",
    "detect_alerts",
    "fleet_summary",
    "fleet_view",
]
