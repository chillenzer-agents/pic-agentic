# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Offline tests for the shipped room-minting entry point.

No homeserver is contacted: the HTTP seam (:func:`pic_agentic.server.setup.api`)
and the MAS token provider are stubbed.
"""

from __future__ import annotations

import asyncio
import json
import stat
import tomllib
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from pic_agentic.config import Config
from pic_agentic.server import setup as setup_mod

REPO = Path(__file__).resolve().parent.parent


class _ApiStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []

    async def __call__(self, _homeserver: str, method: str, path: str, _token: str, data: dict | None = None) -> dict:
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

    state = asyncio.run(
        setup_mod.create_room_and_secret(
            homeserver="https://hs",
            access_token_provider=_token,
            sim="cluster",
            message_dir="/scratch/user/pic-agentic/shared",
            state_path=state_path,
        ),
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
        asyncio.run(
            setup_mod.create_room_and_secret(
                homeserver="https://hs",
                access_token_provider=_token,
                sim="cluster",
                message_dir="relative/dir",
                state_path=tmp_path / "state.json",
            ),
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
    assert "export PIC_AGENTIC_RCP_SECRET=" in out
    assert str(state_path) in out


def test_build_export_quotes_hostile_values(tmp_path: Path) -> None:
    # A server-provided room_id must not break out of the export line when the
    # block is sourced in a shell: the value survives verbatim but stays data.
    import subprocess

    hostile = "!x'; touch /tmp/PWNED; echo '"
    export = setup_mod.build_export(hostile, hostile, "deadbeef", "cluster", "/scratch/shared")
    marker = tmp_path / "pwned"
    script = export + f"\ntest -e {marker} && echo LEAKED || echo SAFE\n"
    result = subprocess.run(["/bin/sh", "-c", script], capture_output=True, text=True, check=False)
    assert "SAFE" in result.stdout
    assert "LEAKED" not in result.stdout
    # and the value is preserved verbatim
    check = subprocess.run(
        ["/bin/sh", "-c", export + '\nprintf "%s" "$PIC_AGENTIC_ROOM_ID"'],
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.stdout == hostile


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


async def _echo_whoami(request: web.Request) -> web.Response:
    return web.json_response({"user_id": "@bot:hs"})


async def test_api_uses_aiohttp_not_urllib(monkeypatch) -> None:
    # Regression guard for the IPv6 blackhole: setup.api must speak HTTP through
    # aiohttp (Happy Eyeballs connector), never urllib.request.urlopen.  The
    # request is served by a real local aiohttp server; urllib is forbidden.
    def _forbid(*_args: object, **_kwargs: object) -> None:
        msg = "urllib.request.urlopen must not be used"
        raise AssertionError(msg)

    monkeypatch.setattr(setup_mod.urllib.request, "urlopen", _forbid)

    app = web.Application()
    app.router.add_get("/_matrix/client/v3/account/whoami", _echo_whoami)
    server = TestServer(app)
    await server.start_server()
    try:
        who = await setup_mod.api(str(server.make_url("")), "GET", "/_matrix/client/v3/account/whoami", "tok")
    finally:
        await server.close()

    assert who == {"user_id": "@bot:hs"}


async def test_api_surfaces_http_error(monkeypatch) -> None:
    async def _boom(_request: web.Request) -> web.Response:
        return web.Response(status=403, text="forbidden detail")

    app = web.Application()
    app.router.add_get("/_matrix/client/v3/account/whoami", _boom)
    server = TestServer(app)
    await server.start_server()
    try:
        with pytest.raises(SystemExit, match=r"failed \(403\): forbidden detail"):
            await setup_mod.api(str(server.make_url("")), "GET", "/_matrix/client/v3/account/whoami", "tok")
    finally:
        await server.close()


def test_api_rejects_non_http_scheme() -> None:
    with pytest.raises(SystemExit, match="must be an http"):
        asyncio.run(setup_mod.api("ftp://hs", "GET", "/x", "tok"))


def test_setup_module_no_longer_imports_urllib_request() -> None:
    # AST-based guard: setup.py may still mention urllib in prose, but must not
    # import or call urllib.request (the IPv6-blackhole path we replaced).
    import ast

    tree = ast.parse((REPO / "src" / "pic_agentic" / "server" / "setup.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "urllib.request" not in imported
    assert not any(name == "urllib.request" or name.startswith("urllib.request.") for name in imported)


async def test_api_timeout_is_a_system_exit() -> None:
    # A stalled request must surface as the documented SystemExit, not a raw
    # TimeoutError escaping to the caller.
    async def _stall(_request: web.Request) -> web.Response:
        await asyncio.sleep(3600)
        raise AssertionError  # pragma: no cover - never reached

    app = web.Application()
    app.router.add_get("/x", _stall)
    server = TestServer(app)
    await server.start_server()
    try:
        with pytest.raises(SystemExit, match="failed:"):
            await setup_mod.api(str(server.make_url("")), "GET", "/x", "tok", timeout_s=0.2)
    finally:
        await server.close()


def test_entry_point_is_registered_in_pyproject() -> None:
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["pic-agentic-setup"] == "pic_agentic.server.setup:main"


def test_entry_point_target_resolves_to_main() -> None:
    module_name, _, attr = "pic_agentic.server.setup:main".partition(":")
    module = __import__(module_name, fromlist=["_"])
    assert getattr(module, attr) is setup_mod.main
