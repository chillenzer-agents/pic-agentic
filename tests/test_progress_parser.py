"""Progress-line parser tests.

The `emu_line` helper reproduces the exact C++ `std::setw` stream from
`SimulationHelper.cpp` + `TimeInterval.hpp`, so the tests are generated from
the code that prints the line rather than hand-copied constants.
"""

from __future__ import annotations

import pytest

from pic_agentic.parsing.progress import parse_progress_line, parse_time_ms


def _setw(width: int, value: object) -> str:
    s = str(value)
    return " " * max(0, width - len(s)) + s


def _print_time(h: int, m: int, s: int, ms: int) -> str:
    # Percent formatting mirrors the C++ std::setw stream exactly; do not
    # rewrite to f-strings.
    if h > 0:
        return "%2dh %2dmin %2dsec %3dmsec" % (h, m, s, ms)
    if m > 0:
        return "%2dmin %2dsec %3dmsec" % (m, s, ms)
    if s > 0:
        return "%2dsec %3dmsec" % (s, ms)
    return "%3dmsec" % ms


def emu_line(pct: int, step: int, elapsed: tuple[int, int, int, int], avg: tuple[int, int, int, int]) -> str:
    return (
        _setw(3, pct)
        + " % = "
        + _setw(8, step)
        + " | time elapsed:"
        + _setw(25, _print_time(*elapsed))
        + " | avg time per step: "
        + _print_time(*avg)
    )


# The design document's two worked examples, kept verbatim as regression anchors.
DOC_EXAMPLE_1 = "  5 % =      500 | time elapsed:       1min  2sec 345msec | avg time per step:  1sec 234msec"
DOC_EXAMPLE_2 = " 25 % = 12345678 | time elapsed:  25h  1min  1sec   0msec | avg time per step:  1min 30sec  61msec"


@pytest.mark.parametrize(
    ("pct", "step", "elapsed", "avg"),
    [
        (5, 500, (0, 1, 2, 345), (0, 0, 1, 234)),
        (25, 12345678, (25, 1, 1, 0), (0, 1, 30, 61)),
        (50, 5000, (0, 13, 0, 0), (0, 0, 2, 345)),
        (100, 10000, (1, 2, 0, 0), (0, 0, 0, 345)),
        (1, 9, (0, 0, 0, 345), (0, 0, 1, 0)),
        (1, 1, (0, 0, 0, 0), (0, 0, 0, 0)),
        (75, 7500000, (0, 59, 59, 999), (0, 0, 1, 1)),
    ],
)
def test_parses_code_generated_progress_lines(pct, step, elapsed, avg):
    parsed = parse_progress_line(emu_line(pct, step, elapsed, avg))
    assert parsed is not None
    assert parsed.percent == pct
    assert parsed.step == step
    # The regex consumes surrounding whitespace, including printTime's internal
    # setw(2) leading space; the meaningful token is the stripped form.
    assert parsed.elapsed == _print_time(*elapsed).strip()
    assert parsed.avg_per_step == _print_time(*avg).strip()


def test_doc_worked_examples_are_code_accurate():
    assert emu_line(5, 500, (0, 1, 2, 345), (0, 0, 1, 234)) == DOC_EXAMPLE_1
    assert emu_line(25, 12345678, (25, 1, 1, 0), (0, 1, 30, 61)) == DOC_EXAMPLE_2
    for line in (DOC_EXAMPLE_1, DOC_EXAMPLE_2):
        assert parse_progress_line(line) is not None


def test_captures_both_time_fields():
    parsed = parse_progress_line(DOC_EXAMPLE_1)
    assert parsed is not None
    assert parsed.elapsed == "1min  2sec 345msec"
    assert parsed.avg_per_step == "1sec 234msec"
    assert parsed.elapsed_ms == 62_345
    assert parsed.avg_per_step_ms == 1_234


def test_eta_is_derived_from_avg():
    parsed = parse_progress_line(emu_line(25, 2500, (0, 1, 0, 0), (0, 0, 2, 0)))
    assert parsed is not None
    assert parsed.eta_seconds(10000) == pytest.approx(2.0 * 7500)


@pytest.mark.parametrize(
    "line",
    [
        "",
        "not a progress line",
        "  5 % =      500 | time elapsed:",
        "someother output\n",
        "  5%=500",
    ],
)
def test_rejects_non_progress_lines(line):
    assert parse_progress_line(line) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("345msec", 345),
        ("2sec 345msec", 2_345),
        ("1min 2sec 345msec", 62_345),
        ("25h 1min 1sec 0msec", 90_061_000),
        ("0msec", 0),
    ],
)
def test_parse_time_ms(text, expected):
    assert parse_time_ms(text) == expected
