# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Offline tests for the D+E server tools (fleet_status, campaign_provenance)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.model import AgendaGroup, AgendaSim
from pic_agentic.agenda.provenance import RO_CRATE_METADATA_FILE
from pic_agentic.agenda.store import AgendaStore
from pic_agentic.config import Config
from pic_agentic.rcp import new_secret_hex
from pic_agentic.server.app import build_server

SECRET = new_secret_hex()
SIM = "7f3a2b1c"


def _campaign_file(tmp_path: Path) -> str:
    agenda = AgendaGroup(name="group")
    agenda = agenda.add(leaf0=AgendaSim(name="leaf0", spec={"sim": {"replica": 0}}))
    AgendaStore(tmp_path, filename="campaign.json").save(Campaign(name="study", agenda=agenda).with_created_ts())
    return str(tmp_path / "campaign.json")


async def test_tools_are_registered() -> None:
    server, _runtime = build_server(Config(rcp_secret=SECRET), SIM)
    tools = {tool.name: tool for tool in await server.list_tools()}
    assert {"fleet_status", "campaign_provenance"} <= set(tools)
    assert tools["fleet_status"].annotations.read_only_hint is True
    assert tools["campaign_provenance"].annotations.read_only_hint is True


async def test_fleet_status_reports_registry_records() -> None:
    server, runtime = build_server(Config(rcp_secret=SECRET), SIM)
    runtime.submit_service.registry["a"] = runtime.submit_service._record_for("a", cmd_id="c1")
    runtime.submit_service.registry["a"].state = "simulation.job_running"
    runtime.submit_service.registry["a"].active = True
    runtime.submit_service.registry["a"].percent = 40

    payload = (await server.call_tool("fleet_status", {})).structured_content
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["running"] == 1
    assert payload["summary"]["aggregate_percent"] == 40


async def test_fleet_status_reports_stalled_with_configured_threshold() -> None:
    server, runtime = build_server(Config(rcp_secret=SECRET, fleet_stall_after_s=0.0), SIM)
    runtime.submit_service.registry["a"] = runtime.submit_service._record_for("a", cmd_id="c1")
    runtime.submit_service.registry["a"].state = "simulation.job_running"
    runtime.submit_service.registry["a"].active = True
    runtime.submit_service.registry["a"].last_event_ts = (datetime.now(UTC) - timedelta(seconds=9999)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    payload = (await server.call_tool("fleet_status", {})).structured_content
    assert [alert["kind"] for alert in payload["alerts"]] == ["stalled"]


async def test_fleet_status_redacts_secrets() -> None:
    secret = "syt_super_secret"
    server, runtime = build_server(Config(rcp_secret=SECRET, access_token=secret), SIM)
    runtime.submit_service.registry["evil"] = runtime.submit_service._record_for("evil", cmd_id="c")
    runtime.submit_service.registry["evil"].state = f"leak:{secret}"
    payload = (await server.call_tool("fleet_status", {})).structured_content
    assert secret not in str(payload)


async def test_campaign_provenance_returns_a_crate(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path))
    server, _runtime = build_server(config, SIM)
    payload = (await server.call_tool("campaign_provenance", {})).structured_content
    assert payload["ok"] is True
    ids = {entity["@id"] for entity in payload["@graph"]}
    assert RO_CRATE_METADATA_FILE in ids
    assert "./" in ids
    assert "#leaf0" in ids
    assert "#picongpu" in ids


async def test_campaign_provenance_without_campaign_is_soft_error(tmp_path) -> None:
    config = Config(rcp_secret=SECRET, agenda_file=str(tmp_path / "missing.json"))
    server, _runtime = build_server(config, SIM)
    payload = (await server.call_tool("campaign_provenance", {})).structured_content
    assert payload == {"ok": False, "error": "no_campaign"}
