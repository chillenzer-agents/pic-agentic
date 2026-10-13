# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Server-tool tests for the research-loop closure (analysis/conclusion/CWL)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.model import AgendaGroup, AgendaSim
from pic_agentic.agenda.store import AgendaStore
from pic_agentic.config import Config
from pic_agentic.protocol.simulation import SimulationState, SimulationType, build_submit_ack
from pic_agentic.rcp import new_secret_hex
from pic_agentic.server.app import build_server
from pic_agentic.transport.memory import MemoryTransport

SECRET = new_secret_hex()
SIM = "7f3a2b1c"


def _campaign_file(tmp_path: Path) -> str:
    agenda = AgendaGroup(name="group")
    agenda = agenda.add(leaf0=AgendaSim(name="leaf0", spec={"sim": {"replica": 0}}, point={"i": 2.0}))
    AgendaStore(tmp_path, filename="campaign.json").save(Campaign(name="study", agenda=agenda))
    return str(tmp_path / "campaign.json")


def _sweep_file(tmp_path: Path, values: list[float]) -> str:
    """Build a campaign whose leaves record the given sweep ``point`` values."""
    agenda = AgendaGroup(name="group")
    for index, value in enumerate(values):
        name = f"leaf{index:03d}"
        agenda = agenda.add(
            **{name: AgendaSim(name=name, spec={"sim": {"value": value}}, point={"component": value})},
        )
    AgendaStore(tmp_path, filename="campaign.json").save(Campaign(name="study", agenda=agenda))
    return str(tmp_path / "campaign.json")


async def _serve(sim_transport: MemoryTransport) -> asyncio.Task:
    async def responder() -> None:
        counter = 0
        async for command in sim_transport.receive():
            if command.type != SimulationType.COMMAND:
                continue
            counter += 1
            ack = build_submit_ack(
                sim=SIM,
                seq=counter,
                cmd_id=str(command.payload.get("cmd_id", "")),
                sim_id=f"sim{counter:04d}",
                state=SimulationState.ACCEPTED,
                in_reply_to=command.transport_event_id,
            ).sign(SECRET)
            await sim_transport.send(ack)

    return asyncio.create_task(responder())


async def _call(config: Config, name: str, arguments: dict):
    mcp_t, sim_t = MemoryTransport.create_pair()
    server, runtime = build_server(config, SIM)
    runtime._transport = mcp_t
    tasks = [await _serve(sim_t)]
    try:
        return (await server.call_tool(name, arguments)).structured_content
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()


async def test_record_analysis_and_conclusion(tmp_path: Path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    assert await _call(config, "record_agenda_analysis", {"path": "leaf0", "analysis": {"score": 9.0}}) == {
        "ok": True,
        "path": "leaf0",
    }
    assert await _call(config, "conclude_agenda", {"conclusion": "optimum at 9"}) == {
        "ok": True,
        "conclusion": "optimum at 9",
    }
    # Persisted.
    campaign = AgendaStore(tmp_path, filename="campaign.json").load(Campaign)
    assert campaign.analyses["leaf0"]["score"] == pytest.approx(9.0)
    assert campaign.conclusion == "optimum at 9"


async def test_record_analysis_unknown_leaf(tmp_path: Path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    result = await _call(config, "record_agenda_analysis", {"path": "nope", "analysis": {}})
    assert result["ok"] is False
    assert result["error"] == "no_such_leaf"


async def test_conclude_rejects_empty(tmp_path: Path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    result = await _call(config, "conclude_agenda", {"conclusion": "   "})
    assert result == {"ok": False, "error": "empty_conclusion"}


async def test_suggest_refinement_uses_recorded_scores(tmp_path: Path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    await _call(config, "record_agenda_analysis", {"path": "leaf0", "analysis": {"score": 9.0}})
    result = await _call(config, "suggest_agenda_refinement", {"rel_tol": 0.05})
    assert result["ok"] is True
    assert result["best"]["value"] == pytest.approx(9.0)
    assert result["converged"] is False  # a single sample


async def test_suggest_refinement_without_analyses_never_invents_a_best(tmp_path: Path) -> None:
    """F1 regression: the beta-4 focal sweep must not rank the sweep value.

    With ``values=[4.4e-5, 4.6e-5, 4.8e-5]`` and no analyses recorded, the old
    code reported ``leaf002`` (4.8e-5, the physically worst point) as ``best``
    and ``analysed: 3``.  It must instead report no best point and count zero
    analyses, with an actionable message.
    """
    config = Config(rcp_secret=SECRET, agenda_file=_sweep_file(tmp_path, [4.4e-5, 4.6e-5, 4.8e-5]))
    result = await _call(config, "suggest_agenda_refinement", {"rel_tol": 0.05})
    assert result["ok"] is True
    assert result["best"] is None
    assert result["analysed"] == 0
    assert result["converged"] is False
    assert result["suggestions"] == []
    assert "record_agenda_analysis" in result["message"]


async def test_suggest_refinement_ranks_recorded_analyses_not_sweep_values(tmp_path: Path) -> None:
    """The best point is the highest analysed score, regardless of its point."""
    # Sweep points ascend, but the analyses rank the *smallest* point highest.
    config = Config(rcp_secret=SECRET, agenda_file=_sweep_file(tmp_path, [4.4e-5, 4.6e-5, 4.8e-5]))
    await _call(config, "record_agenda_analysis", {"path": "leaf000", "analysis": {"score": 0.31}})
    await _call(config, "record_agenda_analysis", {"path": "leaf001", "analysis": {"score": 0.30}})
    await _call(config, "record_agenda_analysis", {"path": "leaf002", "analysis": {"score": 0.29}})
    result = await _call(config, "suggest_agenda_refinement", {"rel_tol": 0.05})
    assert result["analysed"] == 3
    assert result["best"] == {"label": "leaf000", "value": pytest.approx(0.31)}


async def test_suggest_refinement_does_not_converge_on_a_boundary_optimum(tmp_path: Path) -> None:
    """F2 (beta-7) regression: the focal scan's best sits on the range minimum.

    ``values=[4.0e-5, 4.6e-5, 5.2e-5]`` with the descending beta-7 scores make
    ``leaf000`` (the range minimum) the best while still improving toward it.
    The old code reported ``converged: true`` and ``suggestions: []``; the
    server must thread the leaf coordinates through so the edge optimum keeps
    the sweep open and proposes points *below* 4.0e-5.
    """
    config = Config(rcp_secret=SECRET, agenda_file=_sweep_file(tmp_path, [4.0e-5, 4.6e-5, 5.2e-5]))
    scores = {"leaf000": 66715949903.0, "leaf001": 65595828126.0, "leaf002": 64137641778.0}
    for path, score in scores.items():
        assert await _call(config, "record_agenda_analysis", {"path": path, "analysis": {"score": score}}) == {
            "ok": True,
            "path": path,
        }
    result = await _call(config, "suggest_agenda_refinement", {"rel_tol": 0.05})
    assert result["best"] == {"label": "leaf000", "value": pytest.approx(66715949903.0)}
    assert result["converged"] is False
    assert result["suggestions"]
    assert all(s["value"] < 4.0e-5 for s in result["suggestions"])


async def test_suggest_refinement_distinguishes_unscored_recorded_analyses(tmp_path: Path) -> None:
    """Beta-4 end state: analyses recorded, but none carries a ranked score.

    The invariant build's ``analyze_output`` sections (``focal_position_m`` /
    ``total_electrons`` / ``high_energy_tail``) have no top-level
    ``score``/``value``/``peak``.  Recording them must not be reported as "no
    analyses are recorded yet": the summary keeps ``analysed: 0`` (nothing is
    ranked) but the message must distinguish recorded-but-unscored analyses and
    point at the score contract, rather than inviting a pointless re-record loop.
    """
    config = Config(rcp_secret=SECRET, agenda_file=_sweep_file(tmp_path, [4.4e-5, 4.6e-5, 4.8e-5]))
    beta4_payloads = {
        "leaf000": {
            "focal_position_m": 4.4e-05,
            "total_electrons": 444440000000,
            "high_energy_tail": {"gt_7.5MeV": 733180000, "gt_10MeV": 207690000, "gt_15MeV": 276200},
            "max_energy_MeV": 17.5,
        },
        "leaf001": {
            "focal_position_m": 4.6e-05,
            "total_electrons": 443490000000,
            "high_energy_tail": {"gt_7.5MeV": 718040000, "gt_10MeV": 202880000, "gt_15MeV": 87650},
            "max_energy_MeV": 15.0,
        },
        "leaf002": {
            "focal_position_m": 4.8e-05,
            "total_electrons": 442030000000,
            "high_energy_tail": {"gt_7.5MeV": 703420000, "gt_10MeV": 198230000, "gt_15MeV": 20280},
            "max_energy_MeV": 15.0,
        },
    }
    for path, payload in beta4_payloads.items():
        assert await _call(config, "record_agenda_analysis", {"path": path, "analysis": payload}) == {
            "ok": True,
            "path": path,
        }
    result = await _call(config, "suggest_agenda_refinement", {"rel_tol": 0.05})
    assert result["ok"] is True
    assert result["best"] is None
    assert result["analysed"] == 0
    assert result["suggestions"] == []
    assert "No analyses are recorded yet" not in result["message"]
    assert "score" in result["message"]
    assert "record_agenda_analysis" in result["message"]


async def test_export_agenda_cwl(tmp_path: Path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    result = await _call(config, "export_agenda_cwl", {})
    assert result["ok"] is True
    assert "class: Workflow" in result["workflow"]
    assert "leaf0" in result["workflow"]
    assert result["leaves"] == ["leaf0"]


async def test_campaign_provenance_includes_recorded_analysis(tmp_path: Path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    await _call(config, "record_agenda_analysis", {"path": "leaf0", "analysis": {"score": 1.0}})
    crate = await _call(config, "campaign_provenance", {})
    ids = {entity["@id"] for entity in crate["@graph"]}
    assert "#leaf0/analysis" in ids
