# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Matrix transport built on matrix-nio (pinned 0.26.0).

Matrix carries small RCP metadata messages only; heavy data never travels
through it (design section 1.1/5).  The room timeline is the audit trail, so
both clients backfill on (re)connect via the ``since`` token rather than
relying on live delivery only.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nio import AsyncClient, AsyncClientConfig, RoomMessageText, SyncResponse

from pic_agentic.rcp.envelope import RCP_NAMESPACE, RcpMessage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = logging.getLogger(__name__)

#: Retry delay after a failed Matrix sync.
_SYNC_RETRY_S = 1.0


class MatrixTransport:
    """Send/receive RCP messages in a single Matrix room."""

    def __init__(
        self,
        homeserver: str,
        user_id: str,
        access_token: str,
        room_id: str,
        *,
        sync_timeout_ms: int = 10_000,
        store_path: str | None = None,
    ) -> None:
        """Create a Matrix transport for one room.

        Args:
            homeserver: Homeserver base URL.
            user_id: Bot user id.
            access_token: Bot access token.
            room_id: The RCP room id.
            sync_timeout_ms: Long-poll timeout for ``/sync``.
            store_path: Optional matrix-nio store directory.

        """
        config = AsyncClientConfig(store_sync_tokens=True, encryption_enabled=False)
        default_store = Path(tempfile.gettempdir()) / f"pic-agentic-nio-{user_id.replace('@', '').replace(':', '-')}"
        store = store_path or str(default_store)
        self._client = AsyncClient(homeserver, user_id, config=config, store_path=store)
        self._client.access_token = access_token
        self._room_id = room_id
        self._sync_timeout_ms = sync_timeout_ms
        self._since: str | None = None
        self._closed = False

    async def send(self, message: RcpMessage) -> str:
        """Send one signed RCP message to the room.

        Args:
            message: The signed RCP message.

        Returns:
            The Matrix event id.

        Raises:
            ValueError: If the message is not signed.
            RuntimeError: If ``room_send`` reports a failure.

        """
        if message.sig is None:
            msg = "refusing to send an unsigned RCP message"
            raise ValueError(msg)
        # to_content re-signs only when sig is None, which we have excluded.
        content: dict[str, Any] = message.to_content()
        response = await self._client.room_send(self._room_id, "m.room.message", content)
        event_id = getattr(response, "event_id", None)
        if event_id is None:
            msg = f"room_send failed: {getattr(response, 'message', response)}"
            raise RuntimeError(msg)
        return str(event_id)

    async def _sync(self) -> SyncResponse:
        response = await self._client.sync(timeout=self._sync_timeout_ms, since=self._since, full_state=False)
        if isinstance(response, SyncResponse):
            self._since = response.next_batch
        return response

    async def backfill(self) -> list[RcpMessage]:
        """Fetch currently known RCP messages.

        Returns:
            The RCP messages in the room timeline at this point.

        """
        response = await self._sync()
        return self._extract(response)

    async def receive(self) -> AsyncIterator[RcpMessage]:
        """Yield RCP messages as they arrive.

        Yields:
            Each inbound :class:`RcpMessage`.

        Raises:
            asyncio.CancelledError: If the consuming task is cancelled.

        """
        while not self._closed:
            try:
                response = await self._sync()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # ruff: ignore[blind-except] - sync may raise many network errors  # pragma: no cover
                log.warning("matrix sync failed: %s", exc)
                await asyncio.sleep(_SYNC_RETRY_S)
                continue
            for message in self._extract(response):
                yield message

    def _extract(self, response: SyncResponse) -> list[RcpMessage]:
        messages: list[RcpMessage] = []
        joined = getattr(response.rooms, "join", {}) or {}
        room = joined.get(self._room_id)
        if room is None:
            return messages
        for event in room.timeline.events:
            if not isinstance(event, RoomMessageText):
                continue
            content = getattr(event, "source", {}).get("content")
            if not isinstance(content, dict):
                continue
            raw = content.get(RCP_NAMESPACE)
            if not isinstance(raw, dict):
                continue
            try:
                message = RcpMessage.from_dict(raw)
            except (KeyError, ValueError) as exc:
                log.warning("dropping malformed RCP message: %s", exc)
                continue
            message.transport_sender = event.sender
            message.transport_event_id = event.event_id
            messages.append(message)
        return messages

    @property
    def client(self) -> AsyncClient:
        """The underlying matrix-nio client."""
        return self._client

    async def close(self) -> None:
        """Close the sync loop and the underlying client."""
        self._closed = True
        await self._client.close()
