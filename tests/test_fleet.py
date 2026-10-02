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
