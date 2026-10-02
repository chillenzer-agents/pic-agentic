# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Human interaction over Matrix: command parsing and message formatting.

A human talks to the research agent in a Matrix room.  This module owns the
*deterministic* half -- parsing a plain room message into a command and
formatting agent state into human-readable text -- and imports only stdlib +
pydantic, so it is testable without a homeserver or a cluster.  The transport
and dispatch live in :mod:`pic_agentic.transport.matrix` and
:mod:`pic_agentic.server.app` respectively.

Command grammar (a leading ``!``; case-insensitive)::

    !help                     show the command list
    !status                   the current campaign status (if any)
    !fleet                    the whole fleet summary + alerts
    !leaves                   per-leaf status of the campaign
    !png <sim_id> [record] [component]
                              a small PNG of one openPMD record/component
    !stop                     stop the campaign and cancel in-flight jobs
    !pause / !resume          pause or resume the campaign

Anything else (or an unknown ``!`` command) is answered with help, never an
error, so the bot is forgiving to a human typist.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

#: The verb of a parsed human command.
HumanVerb = Literal["help", "status", "fleet", "leaves", "png", "stop", "pause", "resume"]


class HumanCommand(BaseModel):
    """One parsed human room message."""

    model_config = ConfigDict(extra="forbid")

    verb: HumanVerb
    #: The first positional argument (the ``sim_id`` for ``png``), if any.
    arg: str | None = None
    #: The remaining positional arguments (openPMD record / component for png).
    extras: list[str] = []


class HumanAction(BaseModel):
    """The reply the agent should send for a human message."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["text", "image"] = "text"
    #: The text to send (for ``kind="text"``; the caption for an image).
    text: str = ""
    #: The base64 PNG (for ``kind="image"``).
    png_base64: str | None = None


#: Verbs and whether they take positional arguments (for the help text).
_COMMANDS: list[tuple[str, str]] = [
    ("!help", "show this help"),
    ("!status", "the campaign status (name, state, counts, usage)"),
    ("!fleet", "the whole fleet: summary and alerts"),
    ("!leaves", "per-leaf status of the campaign"),
    ("!png <sim_id> [record] [component]", "a small PNG of an openPMD field"),
    ("!pause / !resume", "pause or resume the campaign"),
    ("!stop", "stop the campaign and cancel its running jobs"),
]


def parse_command(text: str) -> HumanCommand:
    """Parse a plain room message into a :class:`HumanCommand`.

    A message that is not a recognised command (including ordinary chat) parses
    to ``help``, so an unknown input yields the help text rather than silence.

    Args:
        text: The raw message body.

    Returns:
        The parsed command (never raises).

    """
    tokens = text.strip().split()
    if not tokens or not tokens[0].startswith("!"):
        return HumanCommand(verb="help")
    verb = tokens[0][1:].lower()
    if verb == "help" or not verb:
        return HumanCommand(verb="help")
    if verb in {"status", "fleet", "leaves", "stop", "pause", "resume"}:
        return HumanCommand(verb=verb)  # type: ignore[arg-type]
    if verb == "png":
        arg = tokens[1] if len(tokens) > 1 else None
        return HumanCommand(verb="png", arg=arg, extras=tokens[2:])
    return HumanCommand(verb="help")


def help_text() -> str:
    """Return the command reference.

    Returns:
        The help message.

    """
    lines = ["PIConGPU research agent commands:"]
    lines.extend(f"  {command:<38} {description}" for command, description in _COMMANDS)
    return "\n".join(lines)


def format_status(campaign: dict, fleet: dict) -> str:
    """Format a campaign status and fleet summary as a human message.

    Args:
        campaign: The ``agenda_status`` dict (or ``{"ok": False}``).
        fleet: The ``fleet_status`` dict.

    Returns:
        A short, human-readable message.

    """
    if not campaign.get("ok", "error" not in campaign):
        return f"No campaign is being tracked ({campaign.get('error', 'unknown')})."

    name = campaign.get("name", "campaign")
    state = campaign.get("state", "?")
    complete = "complete" if campaign.get("complete") else "in progress"
    counts = campaign.get("counts", {})
    usage = campaign.get("usage", {})
    summary = fleet.get("summary", {})
    lines = [
        f"Campaign {name!r}: {state}, {complete}.",
        "  leaves: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())),
        (
            f"  usage: {usage.get('core_hours', 0):g} core-h, "
            f"{usage.get('gpu_hours', 0):g} GPU-h, "
            f"{usage.get('jobs_submitted', 0)} submitted"
        ),
        (
            f"  fleet: {summary.get('total', 0)} sims "
            f"({summary.get('running', 0)} running, {summary.get('done', 0)} done, "
            f"{summary.get('failed', 0)} failed)"
        ),
    ]
    return "\n".join(lines)


def format_fleet(fleet: dict) -> str:
    """Format a fleet summary and its alerts as a human message.

    Args:
        fleet: The ``fleet_status`` dict.

    Returns:
        A short, human-readable message.

    """
    summary = fleet.get("summary", {})
    by_state = summary.get("by_state", {})
    alerts = fleet.get("alerts", [])
    lines = [
        (
            f"Fleet: {summary.get('total', 0)} sims — "
            f"{summary.get('active', 0)} active, {summary.get('terminal', 0)} terminal."
        ),
        "  states: " + (", ".join(f"{key}={value}" for key, value in sorted(by_state.items())) or "none"),
    ]
    percent = summary.get("aggregate_percent")
    if percent is not None:
        lines.append(f"  mean progress: {percent:.0f}%")
    if alerts:
        lines.append(f"  alerts ({len(alerts)}):")
        lines.extend(f"    {alert.get('sim_id')}: {alert.get('kind')} ({alert.get('detail')})" for alert in alerts[:10])
    else:
        lines.append("  no alerts.")
    return "\n".join(lines)


def format_leaves(status: dict) -> str:
    """Format the per-leaf campaign view as a human message.

    Args:
        status: The ``agenda_status`` dict.

    Returns:
        A short, human-readable message.

    """
    if not status.get("ok", "error" not in status):
        return f"No campaign is being tracked ({status.get('error', 'unknown')})."
    leaves = status.get("leaves", [])
    if not leaves:
        return "The campaign has no leaves."
    lines = [f"Leaves ({len(leaves)}):"]
    for leaf in leaves:
        sim_id = leaf.get("sim_id") or "-"
        point = leaf.get("point")
        parameter = leaf.get("sweep_parameter")
        point_text = ""
        if isinstance(point, dict) and len(point) == 1:
            key = next(iter(point))
            point_text = f" {parameter or key}={point[key]}"
        elif point:
            point_text = f" {point}"
        lines.append(f"  {leaf.get('path')}: {leaf.get('status')} [{sim_id}]{point_text}")
    return "\n".join(lines)


def format_png_caption(sim_id: str, record: str | None, component: str | None) -> str:
    """Return the caption for a PNG message.

    Args:
        sim_id: The simulation id.
        record: The openPMD record name, if given.
        component: The openPMD component, if given.

    Returns:
        A one-line caption.

    """
    fields = record or "field"
    if component:
        fields = f"{fields}/{component}"
    return f"sim {sim_id}: {fields}"


def notification_text(callbacks: list[dict], alerts: list[dict]) -> str | None:
    """Compose a one-line push notification for callbacks/alerts, if any.

    Args:
        callbacks: The callbacks emitted by a tick.
        alerts: The fleet alerts at that point.

    Returns:
        The notification text, or None when there is nothing to report.

    """
    if not callbacks and not alerts:
        return None
    parts: list[str] = []
    done = [callback for callback in callbacks if callback.get("kind") == "done"]
    failed = [callback for callback in callbacks if callback.get("kind") == "failed"]
    if done:
        parts.append(f"{len(done)} run(s) finished")
    if failed:
        parts.append(f"{len(failed)} run(s) failed")
    if alerts:
        parts.append(f"{len(alerts)} alert(s)")
    return "Research agent: " + ", ".join(parts) + "."


__all__ = [
    "HumanAction",
    "HumanCommand",
    "HumanVerb",
    "format_fleet",
    "format_leaves",
    "format_png_caption",
    "format_status",
    "help_text",
    "notification_text",
    "parse_command",
]
