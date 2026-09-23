# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""MAS access/refresh token store with rotation-safe cross-process locking.

Matrix Authentication Service (e.g. ``chat.academiccloud.de``) issues short
lived access tokens (5 minutes by default) plus a rotating refresh token:
using a refresh token consumes it and returns a new one.  ``matrix-nio`` only
speaks the bearer-token API, so this module refreshes on demand and keeps the
current pair in a 0600 cache file.

Because the MCP server and the simclient may share one account (and therefore
one refresh-token chain), the refresh is serialised with a file lock and the
cache is treated as the source of truth: a process always re-reads the cache
under the lock and only refreshes the currently stored refresh token, so the
two processes cannot invalidate each other's grant.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import time
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pic_agentic.config import Config

log = logging.getLogger(__name__)

#: Default access-token lifetime when the server omits ``expires_in``.
DEFAULT_EXPIRES_S = 300.0
#: Refresh this many seconds before the access token actually expires.
DEFAULT_SKEW_S = 30.0

_MATRIX_API_SCOPE = "urn:matrix:org.matrix.msc2967.client:api:*"


class TokenRefreshError(RuntimeError):
    """Raised when a MAS access token cannot be refreshed."""


class MasTokens(BaseModel):
    """One access/refresh token pair with an absolute expiry."""

    access_token: str
    refresh_token: str
    expires_at: float
    # "token_type" is the OAuth response field name, not a credential.
    token_type: str = "Bearer"  # ruff: ignore[hardcoded-password-string]
    scope: str = ""

    def valid(self, *, skew_s: float = DEFAULT_SKEW_S, now: float | None = None) -> bool:
        """Return whether the access token is usable for at least ``skew_s`` more.

        Args:
            skew_s: Safety margin subtracted from the expiry.
            now: Override for the current epoch time (tests).

        Returns:
            True if the token has more than ``skew_s`` seconds of life left.

        """
        return (now if now is not None else time.time()) + skew_s < self.expires_at


def _tokens_from_response(body: dict[str, Any], *, fallback_refresh: str) -> MasTokens:
    """Build a :class:`MasTokens` from an OAuth token response.

    Args:
        body: The decoded token-endpoint JSON.
        fallback_refresh: Refresh token to keep when the server does not rotate.

    Returns:
        The parsed token pair with an absolute expiry.

    """
    expires_in = float(body.get("expires_in", DEFAULT_EXPIRES_S))
    return MasTokens(
        access_token=str(body["access_token"]),
        refresh_token=str(body.get("refresh_token") or fallback_refresh),
        expires_at=time.time() + expires_in,
        token_type=str(body.get("token_type", "Bearer")),
        scope=str(body.get("scope", "")),
    )


class MasTokenStore:
    """Own the refresh chain for one MAS account and hand out bearer tokens."""

    def __init__(
        self,
        *,
        token_endpoint: str,
        client_id: str,
        cache_path: Path,
        refresh_token: str = "",
        access_token: str = "",
        expires_in: float = DEFAULT_EXPIRES_S,
        skew_s: float = DEFAULT_SKEW_S,
    ) -> None:
        """Create a token store.

        Args:
            token_endpoint: MAS ``/oauth2/token`` URL.
            client_id: Public OAuth client id used for the refresh grant.
            cache_path: 0600 JSON cache holding the live token pair.
            refresh_token: Seed refresh token (used only before the first cache).
            access_token: Seed access token (optional).
            expires_in: Lifetime to assume for the seed access token.
            skew_s: Refresh margin before expiry.

        """
        self.token_endpoint = token_endpoint
        self.client_id = client_id
        self.cache_path = Path(cache_path)
        self._skew_s = skew_s
        self._seed = (
            MasTokens(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_at=time.time() + expires_in,
            )
            if refresh_token
            else None
        )
        self._tokens: MasTokens | None = None
        self._local_lock = asyncio.Lock()

    @classmethod
    def from_config(cls, config: Config) -> MasTokenStore | None:
        """Build a store from configuration, or None for static-token servers.

        A local Synapse (or any server without a refresh token) has no
        ``token_endpoint``/``refresh_token``, so the caller keeps using the
        plain access token.

        Args:
            config: The resolved configuration.

        Returns:
            The store, or None when the configuration has no refresh chain.

        Raises:
            TokenRefreshError: If the configuration lacks a refresh chain.

        """
        if not (config.token_endpoint and config.refresh_token):
            msg = "cannot build a MAS token store without token_endpoint/refresh_token"
            raise TokenRefreshError(msg)
        cache = (
            Path(config.token_cache_path).expanduser() if config.token_cache_path else cls.default_cache_path(config)
        )
        return cls(
            token_endpoint=config.token_endpoint,
            client_id=config.client_id,
            cache_path=cache,
            refresh_token=config.refresh_token,
            access_token=config.access_token,
            expires_in=DEFAULT_EXPIRES_S,
        )

    @staticmethod
    def default_cache_path(config: Config) -> Path:
        """Return the default cache path for a configuration.

        Args:
            config: The resolved configuration (uses ``client_id`` when present).

        Returns:
            A per-client 0600 cache path below ``~/.config/pic-agentic``.

        """
        suffix = config.client_id or "default"
        return Path("~/.config/pic-agentic").expanduser() / f"mas-tokens-{suffix}.json"

    def enabled(self) -> bool:
        """Return whether the store can refresh (has an endpoint and refresh token).

        Returns:
            True if a refresh chain could be formed.

        """
        return bool(self.token_endpoint and (self._seed or self.cache_path.exists()))

    async def access_token(self) -> str:
        """Return a currently valid access token, refreshing when necessary.

        Returns:
            A bearer token that is valid for at least the configured skew.

        """
        async with self._local_lock:
            tokens = self._read_cache() or self._tokens or self._seed
            if tokens is not None and tokens.valid(skew_s=self._skew_s):
                self._tokens = tokens
                return tokens.access_token
            refreshed = await self._refresh_locked(tokens)
            return refreshed.access_token

    async def _refresh_locked(self, stale: MasTokens | None) -> MasTokens:
        """Refresh under the cross-process lock, re-checking the shared cache.

        Args:
            stale: The caller's last known token pair (may be out of date).

        Returns:
            The valid token pair after the critical section.

        Raises:
            TokenRefreshError: If no refresh token is available or the grant fails.

        """
        async with self._file_lock():
            # Another process may have refreshed while we waited; the cache is
            # the source of truth and its refresh token is the live link in the
            # rotation chain (reusing a consumed one fails with invalid_grant).
            current = self._read_cache()
            if current is not None and current.valid(skew_s=self._skew_s):
                self._tokens = current
                return current
            seed = current or stale or self._tokens or self._seed
            refresh_token = seed.refresh_token if seed is not None else ""
            if not refresh_token:
                msg = "no refresh token available"
                raise TokenRefreshError(msg)
            body = await self._post_token({"grant_type": "refresh_token", "refresh_token": refresh_token})
            tokens = _tokens_from_response(body, fallback_refresh=refresh_token)
            self._write_cache(tokens)
            self._tokens = tokens
            return tokens

    async def _post_token(self, data: dict[str, str]) -> dict[str, Any]:
        """POST to the MAS token endpoint.

        Args:
            data: The form fields (grant type and code/token).

        Returns:
            The decoded JSON body.

        Raises:
            TokenRefreshError: On transport failure or a non-200 response.

        """
        payload = {**data, "client_id": self.client_id}
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(self.token_endpoint, data=payload) as response,
            ):
                status = response.status
                body = await response.json(content_type=None)
        except aiohttp.ClientError as exc:
            msg = f"token endpoint request failed: {exc}"
            raise TokenRefreshError(msg) from exc
        if status != HTTPStatus.OK:
            detail = body.get("error_description", body.get("error", "")) if isinstance(body, dict) else str(body)
            msg = f"token endpoint rejected the refresh ({status}): {detail}"
            raise TokenRefreshError(msg)
        if not isinstance(body, dict) or "access_token" not in body:
            msg = f"token endpoint returned no access_token: {body!r}"
            raise TokenRefreshError(msg)
        return body

    @asynccontextmanager
    async def _file_lock(self) -> AsyncIterator[None]:
        """Hold an exclusive advisory lock on ``<cache>.lock`` for the block.

        Yields:
            None once the cross-process lock is held.

        """
        lock_path = self.cache_path.with_suffix(self.cache_path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            await asyncio.to_thread(fcntl.flock, fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _read_cache(self) -> MasTokens | None:
        """Read the token cache, returning None when absent or unreadable.

        Returns:
            The cached tokens, or None.

        """
        try:
            raw = self.cache_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            log.warning("cannot read token cache %s: %s", self.cache_path, exc)
            return None
        try:
            return MasTokens.model_validate_json(raw)
        except ValueError as exc:
            log.warning("ignoring malformed token cache %s: %s", self.cache_path, exc)
            return None

    def _write_cache(self, tokens: MasTokens) -> None:
        """Atomically write the token cache with 0600 permissions.

        Args:
            tokens: The token pair to persist.

        """
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        tmp.write_text(tokens.model_dump_json(), encoding="utf-8")
        tmp.chmod(0o600)
        tmp.replace(self.cache_path)


__all__ = ["_MATRIX_API_SCOPE", "MasTokenStore", "MasTokens", "TokenRefreshError", "_tokens_from_response"]
