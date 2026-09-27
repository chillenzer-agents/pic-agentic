# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Dispatch tests for the human chat handler and the new research-loop tools."""

from __future__ import annotations

import base64
from pathlib import Path

from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.model import AgendaGroup, AgendaSim
from pic_agentic.agenda.store import AgendaStore
from pic_agentic.config import Config
from pic_agentic.server.app import build_server
from pic_agentic.transport.memory import MemoryTransport

SECRET = "test-secret"


def _campaign_file(tmp_path: Path) -> str:
    agenda = AgendaGroup(name="group")
    agenda = agenda.add(leaf0=AgendaSim(name="leaf0", spec={"sim": {"replica": 0}}))
    AgendaStore(tmp_path, filename="campaign.json").save(Campaign(name="study", agenda=agenda).with_created_ts())
    return str(tmp_path / "campaign.json")


def _runtime(tmp_path: Path, **kw: object):
    config = Config(rcp_secret=SECRET, agenda_file=_campaign_file(tmp_path), human_room_id="!human:hs", **kw)
    _server, runtime = build_server(config, "7f3a2b1c")
    runtime._transport = MemoryTransport()
    return runtime


async def test_dispatch_help_and_status(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    help_action = await runtime.dispatch_human("hello")
    assert "!status" in help_action.text

    status = await runtime.dispatch_human("!status")
    assert "study" in status.text


async def test_dispatch_fleet_and_leaves(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    assert "Fleet" in (await runtime.dispatch_human("!fleet")).text
    assert "leaf0" in (await runtime.dispatch_human("!leaves")).text


async def test_dispatch_pause_resume_stop(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    assert "paused" in (await runtime.dispatch_human("!pause")).text
    assert "running" in (await runtime.dispatch_human("!resume")).text
    assert "Stopped" in (await runtime.dispatch_human("!stop")).text


async def test_dispatch_png_returns_an_image_action(tmp_path: Path, monkeypatch) -> None:
    runtime = _runtime(tmp_path)

    async def fake_fetch(_send, _params):
        return {"data": base64.b64encode(b"PNG").decode("ascii")}

    monkeypatch.setattr(runtime.submit_service, "fetch_result", fake_fetch)
    action = await runtime.dispatch_human("!png sim1 E x")
    assert action.kind == "image"
    assert action.png_base64 is not None
    assert "sim1" in action.text


async def test_dispatch_png_without_data_is_text_error(tmp_path: Path, monkeypatch) -> None:
    runtime = _runtime(tmp_path)

    async def fake_fetch(_send, _params):
        return {"error": "unavailable"}

    monkeypatch.setattr(runtime.submit_service, "fetch_result", fake_fetch)
    action = await runtime.dispatch_human("!png sim1")
    assert action.kind == "text"
    assert "No image" in action.text


async def test_handle_human_filters_room_and_sender(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    transport = runtime._transport
    # Wrong room -> ignored.
    await runtime._handle_human("!other:hs", "@me:hs", "!status")
    # Our own message -> ignored.
    await runtime._handle_human("!human:hs", runtime.config.user_id, "!status")
    assert transport.sent_text == []
    # The right room and a human -> answered.
    await runtime._handle_human("!human:hs", "@me:hs", "!status")
    assert transport.sent_text
    assert "study" in transport.sent_text[0][1]


async def test_notify_human_respects_the_flag(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)  # notify defaults to False
    await runtime.notify_human([{"kind": "done"}], [])
    assert runtime._transport.sent_text == []

    runtime2 = _runtime(tmp_path, notify=True)
    await runtime2.notify_human([{"kind": "done"}], [])
    assert runtime2._transport.sent_text
