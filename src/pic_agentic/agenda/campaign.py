# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""The persisted campaign: an agenda plus its budget and consumed usage.

Extraction-ready (stdlib + pydantic + the agenda model/budget only).  A
campaign is the durable unit an :class:`~pic_agentic.agenda.engine.AgendaEngine`
loads, advances and saves; keeping the agenda, its budget and its accounting in
one serialised object is what makes an engine restart resumable -- the engine
holds no authoritative state of its own.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from pic_agentic.agenda.budget import Budget, BudgetUsage
from pic_agentic.agenda.model import AgendaGroup  # ruff: ignore[typing-only-first-party-import] - runtime


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 ``Z`` string.

    Returns:
        The timestamp, e.g. ``2026-09-27T12:00:00Z``.

    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Campaign(BaseModel):
    """An agenda with its resource budget and consumed usage."""

    model_config = ConfigDict(extra="forbid")

    name: str
    agenda: AgendaGroup
    budget: Budget = Budget()
    usage: BudgetUsage = BudgetUsage()
    #: ISO-8601 creation timestamp, set on the first save.
    created_ts: str | None = None

    def with_created_ts(self) -> Campaign:
        """Return a copy with ``created_ts`` set when it is not already.

        Returns:
            The campaign, stamped on first use.

        """
        if self.created_ts:
            return self
        return self.model_copy(update={"created_ts": _now_iso()})


__all__ = ["Campaign"]
