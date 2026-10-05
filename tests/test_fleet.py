# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for the aggregate fleet view (milestone D)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from pic_agentic.fleet import detect_alerts, fleet_summary, fleet_view
from pic_agentic.server.simulation import SimRecord

NOW = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)


def _record(sim_id: str, state: str, *, active: bool, **extra: Any) -> SimRecord:
    return SimRecord(sim_id=sim_id, cmd_id="c", state=state, active=active, **extra)


def test_summary_counts_and_aggregate_percent() -> None:
    records = [
        _record("a", "simulation.job_running", active=True, percent=10),
        _record("b", "simulation.job_running", active=True, percent=30),
        _record("c", "results.ready", active=False, percent=100),
        _record("d", "simulation.job_failed", active=False),
    ]
    summary = fleet_summary(records)
    assert summary.total == 4
    assert summary.active == 2
    assert summary.terminal == 2
    assert summary.running == 2
    assert summary.done == 1
    assert summary.failed == 1
    assert summary.aggregate_percent == pytest.approx(140 / 3)
    assert summary.by_state == {
        "results.ready": 1,
        "simulation.job_failed": 1,
        "simulation.job_running": 2,
    }


def test_summary_percent_is_none_without_progress() -> None:
    assert fleet_summary([_record("a", "simulation.submitted", active=True)]).aggregate_percent is None


def test_detect_failed_and_cancelled() -> None:
    records = [
        _record("z", "simulation.job_failed", active=False),
        _record("a", "simulation.cancelled", active=False),
    ]
    alerts = detect_alerts(records, now=NOW, stall_after_s=60)
    assert [a.kind for a in alerts] == ["cancelled", "failed"]  # deterministic sim_id order
    assert {a.sim_id for a in alerts} == {"a", "z"}


def test_detect_nonzero_exit() -> None:
    records = [_record("a", "results.ready", active=False, exit_code=137)]
    alerts = detect_alerts(records, now=NOW, stall_after_s=60)
    assert [a.kind for a in alerts] == ["nonzero_exit"]
    assert "137" in alerts[0].detail


def test_zero_exit_is_not_an_alert() -> None:
    assert detect_alerts([_record("a", "results.ready", active=False, exit_code=0)], now=NOW, stall_after_s=60) == []


def test_detect_stalled_boundary() -> None:
    stale = (NOW - timedelta(seconds=120)).strftime("%Y-%m-%dT%H:%M:%SZ")
    fresh = (NOW - timedelta(seconds=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    records = [
        _record("old", "simulation.job_running", active=True, last_event_ts=stale),
        _record("new", "simulation.job_running", active=True, last_event_ts=fresh),
    ]
    alerts = detect_alerts(records, now=NOW, stall_after_s=60)
    assert [a.sim_id for a in alerts] == ["old"]
    assert alerts[0].kind == "stalled"


def test_stale_but_terminal_is_not_stalled() -> None:
    stale = (NOW - timedelta(seconds=9999)).strftime("%Y-%m-%dT%H:%M:%SZ")
    records = [_record("a", "results.ready", active=False, last_event_ts=stale)]
    assert detect_alerts(records, now=NOW, stall_after_s=60) == []


def test_building_record_is_not_stalled() -> None:
    """H3: a record still building/queued (no job_id) is exempt from ``stalled``.

    The 15-20 min window between ``accepted`` and the SLURM ``job_id`` reports
    no lifecycle event, so a stale ``last_event_ts`` there is normal, not a
    wedge.  Both the pre-submit (``accepted``) and the workflow-returned cases
    must be exempt.
    """
    stale = (NOW - timedelta(seconds=1581)).strftime("%Y-%m-%dT%H:%M:%SZ")
    records = [
        _record("accepted", "accepted", active=True, job_id=None, last_event_ts=stale),
        _record("workflow", "workflow.finished", active=True, job_id=None, last_event_ts=stale),
    ]
    assert detect_alerts(records, now=NOW, stall_after_s=900) == []


def test_queued_record_with_job_id_is_not_stalled() -> None:
    """H3: a queued record (job_id known, not yet running) is exempt too."""
    stale = (NOW - timedelta(seconds=5000)).strftime("%Y-%m-%dT%H:%M:%SZ")
    records = [_record("a", "workflow.finished", active=True, job_id=99, last_event_ts=stale)]
    assert detect_alerts(records, now=NOW, stall_after_s=60) == []


def test_idle_running_record_is_still_stalled() -> None:
    """H3: the exemption must not disarm a genuinely wedged running job."""
    stale = (NOW - timedelta(seconds=9999)).strftime("%Y-%m-%dT%H:%M:%SZ")
    records = [_record("a", "simulation.job_running", active=True, job_id=99, last_event_ts=stale)]
    alerts = detect_alerts(records, now=NOW, stall_after_s=60)
    assert [a.kind for a in alerts] == ["stalled"]


def test_summary_counts_by_phase() -> None:
    """H4: the summary exposes building/queued/running counts."""
    records = [
        _record("a", "accepted", active=True, job_id=None),
        _record("b", "workflow.finished", active=True, job_id=1),
        _record("c", "simulation.job_running", active=True, job_id=2),
        _record("d", "results.ready", active=False, job_id=3),
    ]
    summary = fleet_summary(records)
    assert summary.by_phase == {"building": 1, "done": 1, "queued": 1, "running": 1}


def test_summary_counts_failed_and_cancelled_phases() -> None:
    """The phase histogram also covers the terminal failure outcomes."""
    records = [
        _record("a", "simulation.job_failed", active=False),
        _record("b", "simulation.cancelled", active=False),
        _record("c", "simulation.checkpoint", active=True, job_id=9),
    ]
    summary = fleet_summary(records)
    assert summary.by_phase == {"cancelled": 1, "failed": 1, "running": 1}


def test_checkpoint_record_is_still_stalled() -> None:
    """A stale checkpointed run is mid-run, so it must stay ``stalled``-eligible.

    ``simulation.checkpoint`` is non-terminal (the simulation keeps running);
    misclassifying it as ``queued`` would silently disarm the stall alert.
    """
    stale = (NOW - timedelta(seconds=9999)).strftime("%Y-%m-%dT%H:%M:%SZ")
    records = [_record("a", "simulation.checkpoint", active=True, job_id=99, last_event_ts=stale)]
    alerts = detect_alerts(records, now=NOW, stall_after_s=60)
    assert [a.kind for a in alerts] == ["stalled"]


def test_invalid_duck_typed_phase_falls_back_to_derivation() -> None:
    """A bogus ``phase`` string must not leak into ``by_phase`` (Nit)."""
    from types import SimpleNamespace

    record = SimpleNamespace(state="simulation.job_running", active=True, job_id=2, phase="bogus")
    assert fleet_summary([record]).by_phase == {"running": 1}


def test_active_without_timestamp_is_not_stalled() -> None:
    records = [_record("a", "simulation.submitted", active=True)]
    assert detect_alerts(records, now=NOW, stall_after_s=60) == []


def test_unparseable_timestamp_is_not_stalled() -> None:
    records = [_record("a", "simulation.job_running", active=True, last_event_ts="not-a-time")]
    assert detect_alerts(records, now=NOW, stall_after_s=60) == []


def test_failure_precedes_stalled() -> None:
    stale = (NOW - timedelta(seconds=9999)).strftime("%Y-%m-%dT%H:%M:%SZ")
    records = [_record("a", "simulation.job_failed", active=False, last_event_ts=stale)]
    alerts = detect_alerts(records, now=NOW, stall_after_s=60)
    assert [a.kind for a in alerts] == ["failed"]


def test_done_empty_run_is_suspect_and_alerted() -> None:
    records = [_record("a", "results.ready", active=False, suspect="energy_histogram is all zeros")]
    summary = fleet_summary(records)
    assert summary.done == 1
    assert summary.suspect == 1
    alerts = detect_alerts(records, now=NOW, stall_after_s=60)
    assert [a.kind for a in alerts] == ["suspect"]
    assert "all zeros" in alerts[0].detail


def test_done_populated_run_is_not_suspect() -> None:
    records = [_record("a", "results.ready", active=False)]
    assert fleet_summary(records).suspect == 0
    assert detect_alerts(records, now=NOW, stall_after_s=60) == []


def test_suspect_only_counts_done_runs() -> None:
    """A running record carrying a stale suspect flag must not be counted."""
    records = [_record("a", "simulation.job_running", active=True, suspect="stale")]
    assert fleet_summary(records).suspect == 0
    assert detect_alerts(records, now=NOW, stall_after_s=60) == []


def test_failure_takes_precedence_over_suspect() -> None:
    records = [_record("a", "simulation.job_failed", active=False, suspect="stale")]
    alerts = detect_alerts(records, now=NOW, stall_after_s=60)
    assert [a.kind for a in alerts] == ["failed"]


def test_fleet_view_shape() -> None:
    records = [_record("a", "simulation.job_running", active=True, percent=50)]
    view = fleet_view(records, now=NOW, stall_after_s=60)
    assert view["summary"]["total"] == 1
    assert view["alerts"] == []
