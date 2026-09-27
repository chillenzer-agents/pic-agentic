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
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pic_agentic.agenda.budget import Budget, BudgetUsage
from pic_agentic.agenda.model import AgendaGroup  # ruff: ignore[typing-only-first-party-import] - runtime

#: The campaign lifecycle state.  ``running`` executes; ``paused`` folds
#: observations and emits callbacks but submits nothing (resumable);
#: ``stopped`` is the kill-switch terminal state.
CampaignState = Literal["running", "paused", "stopped"]


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 ``Z`` string.

    Returns:
        The timestamp, e.g. ``2026-09-27T12:00:00Z``.

    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Callback(BaseModel):
    """A durable decision-point record the agent drains (poll-based callback).

    An MCP server cannot call the LLM, so a callback is a pollable record: the
    engine appends one when a leaf reaches a terminal status, the agent drains
    them with ``take_agenda_callbacks`` and decides what to do (analyse, refine
    the agenda, stop).  Keeping them in the persisted campaign means a restart
    between the transition and the poll does not lose a decision point.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    kind: Literal["done", "failed"]
    sim_id: str | None = None
    ts: str | None = None


class Campaign(BaseModel):
    """An agenda with its resource budget and consumed usage."""

    model_config = ConfigDict(extra="forbid")

    name: str
    agenda: AgendaGroup
    budget: Budget = Budget()
    usage: BudgetUsage = BudgetUsage()
    #: ISO-8601 creation timestamp, set on the first save.
    created_ts: str | None = None
    #: Pending (undrained) decision-point callbacks, in emission order.
    callbacks: list[Callback] = Field(default_factory=list)
    #: Lifecycle state: ``running`` executes, ``paused`` holds submissions
    #: (resumable), ``stopped`` is the kill-switch terminal state.
    state: CampaignState = "running"
    #: Recorded analyses, keyed by leaf path (the ``analyze_output`` sections).
    analyses: dict[str, dict[str, Any]] = Field(default_factory=dict)
    #: The agent's declared conclusion for the campaign, if any.
    conclusion: str | None = None

    def with_created_ts(self) -> Campaign:
        """Return a copy with ``created_ts`` set when it is not already.

        Returns:
            The campaign, stamped on first use.

        """
        if self.created_ts:
            return self
        return self.model_copy(update={"created_ts": utc_now_iso()})


__all__ = ["Callback", "Campaign", "CampaignState", "utc_now_iso"]
