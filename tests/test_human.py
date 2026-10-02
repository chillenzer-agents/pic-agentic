# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for the human chat layer (command parsing, formatting, dispatch)."""

from __future__ import annotations

from pic_agentic.human import (
    format_fleet,
    format_leaves,
    format_png_caption,
    format_status,
    help_text,
    notification_text,
    parse_command,
)


def test_parse_known_commands() -> None:
    assert parse_command("!status").verb == "status"
    assert parse_command("  !FLEET  ").verb == "fleet"
    assert parse_command("!leaves").verb == "leaves"
    assert parse_command("!stop").verb == "stop"
    assert parse_command("!pause").verb == "pause"
    assert parse_command("!resume").verb == "resume"
    assert parse_command("!help").verb == "help"


def test_parse_png_with_args() -> None:
    command = parse_command("!png abc123 field density")
    assert command.verb == "png"
    assert command.arg == "abc123"
    assert command.extras == ["field", "density"]


def test_parse_unknown_or_plain_text_is_help() -> None:
    assert parse_command("hello there").verb == "help"
    assert parse_command("!wat").verb == "help"
    assert parse_command("").verb == "help"
    assert parse_command("!").verb == "help"


def test_help_text_lists_commands() -> None:
    text = help_text()
    for command in ("!status", "!fleet", "!leaves", "!png", "!pause", "!resume", "!stop"):
        assert command in text


def test_format_status_with_no_campaign() -> None:
    assert "No campaign" in format_status({"ok": False, "error": "no_campaign"}, {})


def test_format_status_and_leaves() -> None:
    status = {
        "name": "study",
        "state": "running",
        "complete": False,
        "counts": {"planned": 1, "done": 2},
        "usage": {"core_hours": 3.5, "gpu_hours": 0.0, "jobs_submitted": 3},
        "leaves": [{"path": "a", "status": "done", "sim_id": "sim1", "point": {"i": 1}}],
    }
    fleet = {"summary": {"total": 3, "running": 0, "done": 2, "failed": 1}}
    text = format_status(status, fleet)
    assert "study" in text
    assert "running" in text
    assert "core-h" in text

    leaves = format_leaves(status)
    assert "a: done [sim1]" in leaves


def test_format_fleet_with_alerts() -> None:
    fleet = {
        "summary": {"total": 2, "active": 1, "terminal": 1, "by_state": {"running": 1}, "aggregate_percent": 50.0},
        "alerts": [{"sim_id": "s", "kind": "stalled", "detail": "no events", "last_event_ts": None}],
    }
    text = format_fleet(fleet)
    assert "Fleet" in text
    assert "mean progress: 50%" in text
    assert "stalled" in text


def test_format_fleet_reports_suspect_count() -> None:
    fleet = {"summary": {"total": 1, "active": 0, "terminal": 1, "by_state": {}, "suspect": 1}, "alerts": []}
    assert "suspect (all-zero output): 1" in format_fleet(fleet)


def test_format_leaves_marks_suspect() -> None:
    status = {
        "name": "study",
        "leaves": [{"path": "a", "status": "done", "sim_id": "sim1", "suspect": "all zeros"}],
    }
    assert "a: done [sim1] SUSPECT" in format_leaves(status)


def test_notification_text_mentions_suspicious_runs() -> None:
    text = notification_text([{"kind": "done", "suspect": "all zeros"}], [])
    assert text is not None
    assert "suspicious" in text


def test_format_png_caption() -> None:
    assert format_png_caption("s", None, None) == "sim s: field"
    assert format_png_caption("s", "E", "x") == "sim s: E/x"


def test_notification_text() -> None:
    assert notification_text([], []) is None
    text = notification_text([{"kind": "done"}, {"kind": "failed"}], [{"sim_id": "s"}])
    assert text is not None
    assert "finished" in text
    assert "failed" in text
    assert "alert" in text
