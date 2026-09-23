# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""MAS token-store behaviour: expiry, rotation, cache sharing, locking."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from pic_agentic.auth import mas
from pic_agentic.auth.mas import MasTokens, MasTokenStore, TokenRefreshError


class FakeServer:
    """Minimal in-process MAS token endpoint double."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.valid_refresh: str | None = None
        self.counter = 0
        self.fail = False

    def seed(self, refresh: str) -> None:
        self.valid_refresh = refresh

    async def __call__(self, _store: MasTokenStore, data: dict[str, str]) -> dict:
        self.calls.append(data)
        if self.fail:
            msg = "endpoint down"
            raise TokenRefreshError(msg)
        if data["grant_type"] != "refresh_token":
            msg = f"unexpected grant {data['grant_type']}"
            raise TokenRefreshError(msg)
        if data["refresh_token"] != self.valid_refresh:
            # Real MAS answers invalid_grant when a consumed token is reused.
            msg = "token endpoint rejected the refresh (400): invalid_grant"
            raise TokenRefreshError(msg)
        self.counter += 1
        self.valid_refresh = f"refresh-{self.counter + 1}"
        return {
            "access_token": f"access-{self.counter}",
            "refresh_token": self.valid_refresh,
            "expires_in": 300,
            "token_type": "Bearer",
        }


@pytest.fixture
def store(tmp_path, monkeypatch):
    server = FakeServer()
    server.seed("refresh-1")
    s = MasTokenStore(
        token_endpoint="https://auth.test/oauth2/token",
        client_id="client-x",
        cache_path=tmp_path / "mas-tokens.json",
        refresh_token="refresh-1",
        expires_in=0,  # force a refresh on first use
    )
    monkeypatch.setattr(s, "_post_token", lambda data: server(s, data))
    return s, server


def test_parses_token_response() -> None:
    tokens = mas._tokens_from_response(
        {"access_token": "a", "refresh_token": "r", "expires_in": 60}, fallback_refresh="x"
    )
    assert tokens.access_token == "a"
    assert tokens.refresh_token == "r"
    assert tokens.valid(skew_s=0, now=tokens.expires_at - 10)
    assert not tokens.valid(skew_s=0, now=tokens.expires_at + 1)


def test_keeps_fallback_refresh_when_not_rotated() -> None:
    tokens = mas._tokens_from_response({"access_token": "a", "expires_in": 60}, fallback_refresh="keep-me")
    assert tokens.refresh_token == "keep-me"


async def test_refresh_writes_cache_0600(store) -> None:
    s, server = store
    token = await s.access_token()
    assert token == "access-1"
    assert len(server.calls) == 1
    cache = s.cache_path
    assert cache.exists()
    assert (cache.stat().st_mode & 0o777) == 0o600
    assert json.loads(cache.read_text())["refresh_token"] == "refresh-2"
    assert "refresh-1" not in cache.read_text()


async def test_valid_cached_token_is_not_refreshed(store) -> None:
    s, server = store
    await s.access_token()
    calls = len(server.calls)
    # Reuse within the lifetime must not touch the endpoint again.
    s._seed = None
    assert await s.access_token() == "access-1"
    assert len(server.calls) == calls


async def test_second_process_reuses_rotated_refresh_token(tmp_path, monkeypatch) -> None:
    """The cache is the source of truth; a stale in-memory token must not be reused."""
    server = FakeServer()
    server.seed("refresh-1")
    cache = tmp_path / "tokens.json"

    def make():
        s = MasTokenStore(
            token_endpoint="https://auth.test/oauth2/token",
            client_id="c",
            cache_path=cache,
            refresh_token="refresh-1",
            expires_in=0,
        )
        monkeypatch.setattr(s, "_post_token", lambda data: server(s, data))
        return s

    first = make()
    assert await first.access_token() == "access-1"

    # Expire the cached access token so the second process must refresh, while
    # the cache still holds the live refresh-2 from the rotation chain.
    cached = MasTokens.model_validate_json(cache.read_text())
    cache.write_text(
        MasTokens(
            access_token=cached.access_token,
            refresh_token=cached.refresh_token,
            expires_at=time.time() - 5,
        ).model_dump_json()
    )

    # A second process starts with the now-consumed refresh-1 in its seed but
    # must pick up refresh-2 from the shared cache instead of failing.
    second = MasTokenStore(
        token_endpoint="https://auth.test/oauth2/token",
        client_id="c",
        cache_path=cache,
        refresh_token="refresh-1",
        expires_in=0,
    )
    monkeypatch.setattr(second, "_post_token", lambda data: server(second, data))
    assert await second.access_token() == "access-2"
    assert server.calls[-1]["refresh_token"] == "refresh-2"


async def test_concurrent_callers_refresh_once(tmp_path, monkeypatch) -> None:
    server = FakeServer()
    server.seed("refresh-1")
    s = MasTokenStore(
        token_endpoint="https://auth.test/oauth2/token",
        client_id="c",
        cache_path=tmp_path / "tokens.json",
        refresh_token="refresh-1",
        expires_in=0,
    )
    monkeypatch.setattr(s, "_post_token", lambda data: server(s, data))
    results = await asyncio.gather(*[s.access_token() for _ in range(5)])
    assert results == ["access-1"] * 5
    assert len(server.calls) == 1


async def test_expired_cached_token_triggers_refresh(tmp_path, monkeypatch) -> None:
    server = FakeServer()
    server.seed("refresh-1")
    cache = tmp_path / "tokens.json"
    stale = MasTokens(access_token="old", refresh_token="refresh-1", expires_at=time.time() - 5)
    cache.write_text(stale.model_dump_json())
    s = MasTokenStore(
        token_endpoint="https://auth.test/oauth2/token",
        client_id="c",
        cache_path=cache,
        refresh_token="refresh-1",
        expires_in=0,
    )
    monkeypatch.setattr(s, "_post_token", lambda data: server(s, data))
    assert await s.access_token() == "access-1"


async def test_refresh_failure_is_surfaced(store) -> None:
    s, server = store
    server.fail = True
    with pytest.raises(TokenRefreshError):
        await s.access_token()


def test_enabled_without_refresh_token(tmp_path) -> None:
    s = MasTokenStore(token_endpoint="https://auth.test/oauth2/token", client_id="c", cache_path=tmp_path / "t.json")
    assert s.enabled() is False
    with pytest.raises(TokenRefreshError):
        asyncio.run(s.access_token())
