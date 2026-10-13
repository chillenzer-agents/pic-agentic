# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for the research-loop closure: analysis, conclusion, accounting."""

from __future__ import annotations

from pathlib import Path

import pytest

from pic_agentic.agenda.budget import BudgetUsage
from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.engine import ActualUsage, AgendaEngine
from pic_agentic.agenda.model import AgendaGroup
from pic_agentic.agenda.planner import reconcile
from pic_agentic.agenda.provenance import campaign_rocrate
from pic_agentic.agenda.store import DEFAULT_CAMPAIGN_FILE, AgendaStore
from pic_agentic.slurm.client import parse_accounting


def _by_id(crate: dict, entity_id: str) -> dict:
    return next(entity for entity in crate["@graph"] if entity["@id"] == entity_id)


# --------------------------------------------------------------------------
# gap 2: analysis + conclusion in the provenance
# --------------------------------------------------------------------------


def test_campaign_round_trips_analyses_and_conclusion() -> None:
    campaign = Campaign(name="c", agenda=AgendaGroup(name="g"))
    campaign = campaign.model_copy(
        update={"analyses": {"leaf0": {"answer": "peak", "score": 3.0}}, "conclusion": "optimum at 3"},
    )
    again = Campaign.model_validate_json(campaign.model_dump_json())
    assert again.analyses == {"leaf0": {"answer": "peak", "score": 3.0}}
    assert again.conclusion == "optimum at 3"


def test_provenance_links_analyses_and_conclusion() -> None:
    group = AgendaGroup(name="g").add_sim(name="leaf0", spec={"sim": {"replica": 0}})
    campaign = Campaign(name="c", agenda=group, analyses={"leaf0": {"answer": "peak"}}).with_created_ts()
    crate = campaign_rocrate(campaign, analyses=campaign.analyses)
    # An analysis entity and its linking action exist for the analysed leaf.
    assert _by_id(crate, "#leaf0/analysis")["@type"] == "Dataset"
    assert _by_id(crate, "#leaf0/analysis_action")["result"] == {"@id": "#leaf0/analysis"}


# --------------------------------------------------------------------------
# gap 4: accounting parse + reconcile
# --------------------------------------------------------------------------


def test_parse_accounting_computes_core_and_gpu_hours() -> None:
    # 3600 s * 8 cpus = 8 core-h; 3600 s * 2 gpus = 2 gpu-h.
    output = "JobID|ElapsedRaw|AllocCPUS|TRESUsageInTot\n4711|3600|8|gres/gpu=2\n"
    rows = parse_accounting(output)
    assert rows[4711].core_hours == pytest.approx(8.0)
    assert rows[4711].gpu_hours == pytest.approx(2.0)
    assert rows[4711].is_gpu is True


def test_parse_accounting_skips_malformed_rows() -> None:
    output = "JobID|ElapsedRaw|AllocCPUS|TRESUsageInTot\nbogus|x|y\n4711|1800|4|\n"
    rows = parse_accounting(output)
    assert set(rows) == {4711}
    assert rows[4711].core_hours == pytest.approx(2.0)
    assert rows[4711].is_gpu is False


def test_parse_accounting_falls_back_to_alloctres_for_gpu_count() -> None:
    """F5: an empty ``TRESUsageInTot`` must not zero out ``gpu_hours``.

    On the beta-7 cluster ``TRESUsageInTot`` was empty for completed GPU jobs,
    so ``gpu_hours`` read 0.0 for every run while ``core_hours`` accrued; the
    GPU count has to fall back to the ``AllocTRES`` reservation.  The column
    shape mirrors ``sacct --parsable2
    --format=JobID,ElapsedRaw,AllocCPUS,TRESUsageInTot,AllocTRES``.
    """
    output = "JobID|ElapsedRaw|AllocCPUS|TRESUsageInTot|AllocTRES\n606037|2118|8||gres/gpu=4\n"
    rows = parse_accounting(output)
    assert rows[606037].core_hours == pytest.approx(2118 * 8 / 3600)
    assert rows[606037].gpu_hours == pytest.approx(2118 * 4 / 3600)
    assert rows[606037].is_gpu is True


def test_parse_accounting_prefers_tres_usage_over_alloc() -> None:
    """``TRESUsageInTot`` wins when present; the fallback is only a fallback."""
    output = "JobID|ElapsedRaw|AllocCPUS|TRESUsageInTot|AllocTRES\n4711|3600|8|gres/gpu=2|gres/gpu=4\n"
    rows = parse_accounting(output)
    assert rows[4711].gpu_hours == pytest.approx(2.0)


def test_parse_accounting_gpu_less_job_stays_zero() -> None:
    """A genuinely CPU-only job keeps ``gpu_hours == 0.0`` (no false positive)."""
    output = "JobID|ElapsedRaw|AllocCPUS|TRESUsageInTot|AllocTRES\n4700|3600|8||cpu=8,mem=32G\n"
    rows = parse_accounting(output)
    assert rows[4700].gpu_hours == pytest.approx(0.0)
    assert rows[4700].is_gpu is False


def test_reconcile_replaces_the_estimate_with_the_actual() -> None:
    # Reserved 10 core-h; actually used 3 -> usage drops by 7.
    usage = BudgetUsage(core_hours=10.0, jobs_submitted=1)
    out = reconcile(usage, estimated_core_hours=10.0, actual_core_hours=3.0)
    assert out.core_hours == pytest.approx(3.0)
    assert usage.core_hours == pytest.approx(10.0)  # pure


def test_reconcile_none_actual_keeps_the_estimate() -> None:
    usage = BudgetUsage(core_hours=10.0)
    assert reconcile(usage, estimated_core_hours=10.0, actual_core_hours=None).core_hours == pytest.approx(10.0)


def test_reconcile_gpu_only_when_is_gpu() -> None:
    usage = BudgetUsage(core_hours=1.0, gpu_hours=4.0)
    out = reconcile(
        usage,
        estimated_core_hours=1.0,
        actual_core_hours=1.0,
        estimated_gpu_hours=4.0,
        actual_gpu_hours=1.0,
        is_gpu=True,
    )
    assert out.gpu_hours == pytest.approx(1.0)
    # A non-GPU leaf ignores gpu actuals.
    untouched = reconcile(
        BudgetUsage(gpu_hours=0.0),
        estimated_core_hours=1.0,
        actual_core_hours=1.0,
        estimated_gpu_hours=4.0,
        actual_gpu_hours=1.0,
        is_gpu=False,
    )
    assert untouched.gpu_hours == pytest.approx(0.0)


async def test_engine_reconciles_actual_usage_on_completion(tmp_path: Path) -> None:
    store = AgendaStore(tmp_path, filename=DEFAULT_CAMPAIGN_FILE)
    group = AgendaGroup(name="g").add_sim(name="a", spec={"sim": {"replica": 0}, "resources": {"est_core_hours": 10.0}})
    store.save(Campaign(name="c", agenda=group))
    state: dict[str, str] = {}
    actuals: dict[str, ActualUsage] = {}

    async def submit(_spec: dict, _key: str) -> str:
        state["sim1"] = "simulation.job_running"
        return "sim1"

    engine = AgendaEngine(store=store, submit=submit, observe=lambda: dict(state), actuals=lambda: actuals)
    first = await engine.tick()
    assert first.usage.core_hours == pytest.approx(10.0)  # estimate reserved

    state["sim1"] = "results.ready"
    actuals["sim1"] = ActualUsage(core_hours=2.5)
    second = await engine.tick()
    assert second.usage.core_hours == pytest.approx(2.5)  # corrected to actual
    reloaded = store.load(Campaign)
    assert reloaded.agenda.entries["a"].actual_core_hours == pytest.approx(2.5)
    # A further tick does not double-apply (the actual is stamped).
    third = await engine.tick()
    assert third.usage.core_hours == pytest.approx(2.5)
