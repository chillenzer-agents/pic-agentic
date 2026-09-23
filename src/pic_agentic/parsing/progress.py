"""Best-effort parser for the PIConGPU per-step progress line.

The exact format is emitted by ``SimulationHelper.cpp`` (rank 0 only)::

    std::setw(3) << percent << " % = " << std::setw(8) << currentStep
      << " | time elapsed:" << std::setw(25) << tSimCalculation.printInterval()
      << " | avg time per step: " << TimeInterval::printTime(roundAvg / progressInterval)

Two nuances that the design document gets slightly wrong (spec amendment):

* Only the **elapsed** time field is ``setw(25)`` right-aligned.  The
  **avg per step** field has no outer ``setw``; its spacing comes from
  ``TimeInterval::printTime``'s internal ``setw(2)``/``setw(3)``.
* Consequently the number of spaces after either ``:`` is not fixed.  The
  regex below anchors on the literals and consumes any run of spaces, so it
  is robust to both.

This format is an internal implementation detail and may change; treat this
parser as best-effort and fall back to openPMD iteration metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Capture groups: 1=percent, 2=step, 3=elapsed, 4=avg_per_step.
PROGRESS_RE = re.compile(
    r"^\s*(\d{1,3}) % = \s*(\d+) \| time elapsed:\s*(\S(?:.*\S)?) \| avg time per step:\s*(\S(?:.*\S)?)\s*$"
)

_UNIT_MS = {"h": 3_600_000, "min": 60_000, "sec": 1_000, "msec": 1}
_TOKEN_RE = re.compile(r"(\d+)\s*(msec|sec|min|h)")


@dataclass(frozen=True)
class ProgressLine:
    percent: int
    step: int
    elapsed: str
    avg_per_step: str
    elapsed_ms: int | None = None
    avg_per_step_ms: int | None = None

    def eta_seconds(self, total_steps: int) -> float | None:
        """Derived ETA: ``avg_per_step * steps_remaining`` (best-effort)."""
        if self.avg_per_step_ms is None or total_steps <= 0:
            return None
        remaining = max(0, total_steps - self.step)
        return self.avg_per_step_ms * remaining / 1000.0


def parse_time_ms(text: str) -> int | None:
    """Parse ``Hh Mmin Ssec mmm msec`` (zero components omitted) into ms."""
    total = 0
    found = False
    for value, unit in _TOKEN_RE.findall(text):
        total += int(value) * _UNIT_MS[unit]
        found = True
    return total if found else None


def parse_progress_line(line: str) -> ProgressLine | None:
    """Return a :class:`ProgressLine` or ``None`` if ``line`` is not progress."""
    match = PROGRESS_RE.match(line)
    if not match:
        return None
    percent, step, elapsed, avg = match.groups()
    return ProgressLine(
        percent=int(percent),
        step=int(step),
        elapsed=elapsed,
        avg_per_step=avg,
        elapsed_ms=parse_time_ms(elapsed),
        avg_per_step_ms=parse_time_ms(avg),
    )
