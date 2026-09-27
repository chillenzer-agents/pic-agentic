# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Coverage-gap tests for the agenda/fleet/server modules.

Targets the error paths and boundary branches the coverage audit flagged as
untested (store write-failure cleanup, leaf/path guards, dependency resolution
forms, naive stall timestamps, the server soft-error paths).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pytest

from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.engine import AgendaEngine, leaf_at
from pic_agentic.agenda.model import AgendaGroup, AgendaSim
from pic_agentic.agenda.planner import next_actions
from pic_agentic.agenda.store import DEFAULT_CAMPAIGN_FILE, AgendaStore
from pic_agentic.fleet import detect_alerts
from pic_agentic.server.simulation import SimRecord

# --------------------------------------------------------------------------
# store: atomic-write failure cleanup
# --------------------------------------------------------------------------


def test_store_write_failure_removes_the_temp_file(tmp_path, monkeypatch) -> None:
    store = AgendaStore(tmp_path, filename=DEFAULT_CAMPAIGN_FILE)

    def boom(*_args: object, **_kwargs: object) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr("pic_agentic.agenda.store._write_and_replace", boom)
    campaign = Campaign(name="c", agenda=AgendaGroup(name="g"))
    with pytest.raises(OSError, match="disk full"):
        store.save(campaign)
    # No half-written temp file survives a failed save.
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".")]
    assert leftovers == []
    assert not store.exists()


# --------------------------------------------------------------------------
# engine: leaf_at guards
# --------------------------------------------------------------------------


def test_leaf_at_returns_none_for_a_leaf_intermediate_or_missing_group() -> None:
    agenda = AgendaGroup(name="g").add_sim(name="a", spec={"replica": 0})
    # ``a`` is a leaf, so ``a/b`` cannot resolve.
    assert leaf_at(agenda, "a/b") is None
    # A missing group.
    assert leaf_at(agenda, "x/y") is None
    # A group path (not a leaf) resolves to None.
    nested = agenda.add(sub=AgendaGroup(name="sub"))
    assert leaf_at(nested, "sub") is None


# --------------------------------------------------------------------------
# planner: dependency resolution forms
# --------------------------------------------------------------------------


def test_dependency_resolves_by_last_segment() -> None:
    # A bare name resolves sibling-first; the last-segment fallback matters for
    # a name that matches a nested leaf's segment.
    agenda = AgendaGroup(name="study")
    agenda = agenda.add_sim(name="a", spec={"replica": 0})
    agenda = agenda.add_sim(name="b", spec={"replica": 1})
    agenda.entries["b"].depends_on = ["a"]
    steps = {s.path: s.action for s in next_actions(agenda, {}, budget=_budget(), usage=_usage())}
    assert steps["b"] == "wait"
    steps = {s.path: s.action for s in next_actions(agenda, {"a": "results.ready"}, budget=_budget(), usage=_usage())}
    assert steps["b"] == "submit"

    # A nested sibling is resolved by its bare last segment too.
    scan = AgendaGroup(name="scan").add_sim(name="scan__i=1", spec={"replica": 0})
    study = AgendaGroup(name="study").add(scan=scan)
    study = study.add(summary=AgendaSim(name="summary", spec={"replica": 9}, depends_on=["scan__i=1"]))
    resolved = next_actions(study, {}, budget=_budget(), usage=_usage())
    assert next(s for s in resolved if s.path == "summary").action == "wait"


# --------------------------------------------------------------------------
# fleet: a naive (tz-less) timestamp is still judged
# --------------------------------------------------------------------------


def test_naive_last_event_ts_is_judged_for_stalls() -> None:
    now = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    record = SimRecord(sim_id="s", cmd_id="c", state="simulation.job_running", active=True)
    record.last_event_ts = "2026-09-27T11:00:00"  # naive, an hour stale
    alerts = detect_alerts([record], now=now, stall_after_s=60)
    assert [a.kind for a in alerts] == ["stalled"]


# --------------------------------------------------------------------------
# model: _coerce_entry rename branch
# --------------------------------------------------------------------------


def test_add_renames_an_entry_whose_name_differs_from_its_key() -> None:
    group = AgendaGroup(name="g").add(alias=AgendaSim(name="other", spec={"replica": 0}))
    assert group.entries["alias"].name == "alias"


# --------------------------------------------------------------------------
# server: soft-error paths
# --------------------------------------------------------------------------


def _budget():
    from pic_agentic.agenda.budget import Budget

    return Budget()


def _usage():
    from pic_agentic.agenda.budget import BudgetUsage

    return BudgetUsage()


async def test_server_soft_errors(tmp_path: Path) -> None:
    from pic_agentic.config import Config
    from pic_agentic.server.agenda import AgendaService

    missing = Config(rcp_secret="x", agenda_file=str(tmp_path / "missing.json"))
    service = AgendaService(missing, _FakeSubmit())
    assert await service.set_state("paused") == {"ok": False, "error": "no_campaign"}
    assert await service.take_callbacks() == {"ok": False, "error": "no_campaign"}
    assert await service.approve("leaf0") == {"ok": False, "error": "no_campaign"}
    assert await service.add_leaf("leaf0", {"sim": {}}) == {"ok": False, "error": "no_campaign"}

    # A corrupt campaign file degrades every mutator to a redacted soft error.
    path = tmp_path / "bad.json"
    path.write_text("{ not json")
    corrupt = Config(rcp_secret="x", agenda_file=str(path))
    service2 = AgendaService(corrupt, _FakeSubmit())
    for result in (
        await service2.set_state("paused"),
        await service2.take_callbacks(),
        await service2.approve("leaf0"),
        await service2.add_leaf("leaf0", {"sim": {}}),
    ):
        assert result["ok"] is False


async def test_set_state_rejects_an_invalid_state(tmp_path: Path) -> None:
    from pic_agentic.config import Config
    from pic_agentic.server.agenda import AgendaService

    campaign = tmp_path / "campaign.json"
    AgendaStore(tmp_path, filename="campaign.json").save(Campaign(name="c", agenda=AgendaGroup(name="g")))
    service = AgendaService(Config(rcp_secret="x", agenda_file=str(campaign)), _FakeSubmit())
    result = await service.set_state("bogus")  # type: ignore[arg-type]
    assert result["ok"] is False
    # The on-disk state is unchanged (still running).
    assert AgendaStore(tmp_path, filename="campaign.json").load(Campaign).state == "running"


class _FakeSubmit:
    """A submit service stub exposing only what the service touches."""

    registry: ClassVar[dict] = {}


def test_engine_submit_for_a_vanished_leaf_is_a_no_op(tmp_path) -> None:
    """A leaf removed between plan and submit is skipped, not a crash."""
    import asyncio

    store = AgendaStore(tmp_path, filename=DEFAULT_CAMPAIGN_FILE)
    store.save(Campaign(name="c", agenda=AgendaGroup(name="g").add_sim(name="a", spec={"replica": 0})))

    async def submit(_spec: dict, _key: str) -> str:
        return "s"

    engine = AgendaEngine(store=store, submit=submit, observe=dict)

    async def run() -> None:
        # ``leaf_at`` on a leaf path that is actually a group prefix is None.
        original = engine._do_submit

        async def patched(campaign, path, step):
            campaign.agenda.entries.pop("a", None)
            return await original(campaign, path, step)

        engine._do_submit = patched  # type: ignore[method-assign]
        await engine.tick()

    asyncio.run(run())
