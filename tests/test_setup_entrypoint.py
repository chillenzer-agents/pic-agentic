# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Offline tests for the shipped room-minting entry point.

No homeserver is contacted: the HTTP seam (:func:`pic_agentic.server.setup.api`)
and the MAS token provider are stubbed.
"""

from __future__ import annotations

import json
import stat
import tomllib
from pathlib import Path

import pytest

from pic_agentic.config import Config
from pic_agentic.server import setup as setup_mod

REPO = Path(__file__).resolve().parent.parent


class _ApiStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []

    def __call__(self, _homeserver: str, method: str, path: str, _token: str, data: dict | None = None) -> dict:
        self.calls.append((method, path, data))
        if path.endswith("/account/whoami"):
            return {"user_id": "@bot:hs"}
        if path.endswith("/createRoom"):
            return {"room_id": "!fresh:hs"}
        raise AssertionError(path)


async def _token() -> str:
    return "tok"


def test_create_room_and_secret_writes_0600_state(tmp_path, monkeypatch) -> None:
    stub = _ApiStub()
    monkeypatch.setattr(setup_mod, "api", stub)
    state_path = tmp_path / "nested" / "cluster-check.json"

    state = setup_mod.create_room_and_secret(
        homeserver="https://hs",
        access_token_provider=_token,
        sim="cluster",
        message_dir="/scratch/user/pic-agentic/shared",
        state_path=state_path,
    )

    assert state["homeserver"] == "https://hs"
    assert state["room_id"] == "!fresh:hs"
    assert state["user_id"] == "@bot:hs"
    assert state["sim"] == "cluster"
    assert state["message_dir"] == "/scratch/user/pic-agentic/shared"
    assert len(state["rcp_secret"]) == 64
    int(state["rcp_secret"], 16)

    on_disk = json.loads(state_path.read_text())
    assert on_disk == state
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    create = [c for c in stub.calls if c[1].endswith("/createRoom")]
    expected_body = {"name": setup_mod.ROOM_NAME, "preset": "private_chat"}
    assert create == [("POST", "/_matrix/client/v3/createRoom", expected_body)]


def test_create_room_and_secret_rejects_relative_message_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(setup_mod, "api", _ApiStub())
    with pytest.raises(SystemExit):
        setup_mod.create_room_and_secret(
            homeserver="https://hs",
            access_token_provider=_token,
            sim="cluster",
            message_dir="relative/dir",
            state_path=tmp_path / "state.json",
        )
    assert not (tmp_path / "state.json").exists()


def test_main_mints_room_with_stubbed_http(tmp_path, monkeypatch, capsys) -> None:
    stub = _ApiStub()
    monkeypatch.setattr(setup_mod, "api", stub)
    config = Config(
        homeserver="https://hs",
        token_endpoint="https://auth/oauth2/token",
        client_id="cid",
        refresh_token="refresh",
    )
    monkeypatch.setattr(Config, "load", classmethod(lambda _cls, _path=None: config))
    monkeypatch.setattr(setup_mod, "_config_token_provider", lambda _cfg: _token)
    state_path = tmp_path / "state.json"

    rc = setup_mod.main(
        ["--state", str(state_path), "--sim", "cluster", "--message-dir", "/scratch/user/shared"],
    )

    assert rc == 0
    state = json.loads(state_path.read_text())
    assert state["room_id"] == "!fresh:hs"
    out = capsys.readouterr().out
    assert "export PIC_AGENTIC_ROOM_ID='!fresh:hs'" in out
    assert "export PIC_AGENTIC_RCP_SECRET='" in out
    assert str(state_path) in out


def test_main_rejects_missing_message_dir(tmp_path, monkeypatch) -> None:
    config = Config(
        homeserver="https://hs",
        token_endpoint="https://auth/oauth2/token",
        client_id="cid",
        refresh_token="refresh",
    )
    monkeypatch.setattr(Config, "load", classmethod(lambda _cls, _path=None: config))
    with pytest.raises(SystemExit):
        setup_mod.main(["--state", str(tmp_path / "state.json")])


def test_entry_point_is_registered_in_pyproject() -> None:
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["pic-agentic-setup"] == "pic_agentic.server.setup:main"


def test_entry_point_target_resolves_to_main() -> None:
    module_name, _, attr = "pic_agentic.server.setup:main".partition(":")
    module = __import__(module_name, fromlist=["_"])
    assert getattr(module, attr) is setup_mod.main
