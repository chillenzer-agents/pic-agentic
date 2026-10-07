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

from nio import AsyncClient, AsyncClientConfig, MessageDirection, RoomMessagesResponse, RoomMessageText, SyncResponse

from pic_agentic.rcp.envelope import RCP_NAMESPACE, RcpMessage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    #: Returns a currently valid bearer token (refreshing when necessary).
    TokenProvider = Callable[[], Awaitable[str]]

log = logging.getLogger(__name__)

#: Retry delay after a failed Matrix sync.
_SYNC_RETRY_S = 1.0

#: Page size for back-paginating the room timeline via ``/messages``.  The
#: ``/sync`` window is small (10 events by default), so a long campaign can have
#: more RCP events than one sync returns; backfill walks older pages to
#: reconstruct the full signed-room history the registry is projected from.
_BACKFILL_PAGE_LIMIT = 100

#: Safety cap on the total number of events backfill will accumulate.  Bounds
#: memory for a very long room; when the cap is hit the history is marked
#: *incomplete* (see :attr:`MatrixTransport.history_complete`) so an
#: unreconstructed run is not silently mistaken for one that never existed.
_MAX_BACKFILL_EVENTS = 10_000


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
        token_provider: TokenProvider | None = None,
    ) -> None:
        """Create a Matrix transport for one room.

        Args:
            homeserver: Homeserver base URL.
            user_id: Bot user id.
            access_token: Bot access token (used until ``token_provider`` refreshes).
            room_id: The RCP room id.
            sync_timeout_ms: Long-poll timeout for ``/sync``.
            store_path: Optional matrix-nio store directory.
            token_provider: Optional async callable returning a valid bearer
                token; used for MAS servers whose access tokens expire.

        """
        config = AsyncClientConfig(store_sync_tokens=True, encryption_enabled=False)
        default_store = Path(tempfile.gettempdir()) / f"pic-agentic-nio-{user_id.replace('@', '').replace(':', '-')}"
        store = store_path or str(default_store)
        self._client = AsyncClient(homeserver, user_id, config=config, store_path=store)
        self._client.access_token = access_token
        self._token_provider = token_provider
        self._room_id = room_id
        self._sync_timeout_ms = sync_timeout_ms
        self._since: str | None = None
        self._closed = False
        #: Whether the last :meth:`backfill` reconstructed the *whole* signed
        #: timeline (True) or stopped early once its safety cap was reached
        #: (False).  When False the caller must not treat an absent run as one
        #: that never existed -- see the registry's ``unknown_after_restart``.
        self.history_complete = True

    async def _refresh_token(self) -> None:
        """Swap in a fresh access token before a request, when configured."""
        if self._token_provider is not None:
            self._client.access_token = await self._token_provider()

    async def send(self, message: RcpMessage) -> str:
        """Send one signed RCP message to the room.

        Args:
            message: The signed RCP message.

        Returns:
            The Matrix event id.

        Raises:
            ValueError: If the message is not signed.

        """
        if message.sig is None:
            msg = "refusing to send an unsigned RCP message"
            raise ValueError(msg)
        # to_content re-signs only when sig is None, which we have excluded.
        content: dict[str, Any] = message.to_content()
        return await self._send_content(self._room_id, content)

    async def send_text(self, room_id: str, text: str) -> str:
        """Send a plain human-readable text message to a room.

        Args:
            room_id: The target room (the configured RCP room, or the human
                room when different).
            text: The message body.

        Returns:
            The Matrix event id.

        """
        content = {"msgtype": "m.text", "body": text}
        return await self._send_content(room_id, content)

    async def send_image(
        self,
        room_id: str,
        png: bytes,
        *,
        body: str = "image",
        filename: str = "plot.png",
        mimetype: str = "image/png",
    ) -> str:
        """Upload a PNG and send it as an ``m.image`` message.

        Args:
            room_id: The target room.
            png: The PNG bytes.
            body: The message body / alt text.
            filename: The uploaded file name.
            mimetype: The content type.

        Returns:
            The Matrix event id.

        Raises:
            RuntimeError: If the upload or the send fails.

        """
        await self._refresh_token()
        response, _ = await self._client.upload(png, content_type=mimetype, filename=filename)
        content_uri = getattr(response, "content_uri", None)
        if not content_uri:
            msg = f"media upload failed: {getattr(response, 'message', response)}"
            raise RuntimeError(msg)
        content = {
            "msgtype": "m.image",
            "body": body,
            "url": content_uri,
            "info": {"mimetype": mimetype, "size": len(png)},
        }
        return await self._send_content(room_id, content)

    async def _send_content(self, room_id: str, content: dict[str, Any]) -> str:
        """Send one content object and return its event id.

        Returns:
            The Matrix event id.

        Raises:
            RuntimeError: If ``room_send`` reports a failure.

        """
        await self._refresh_token()
        response = await self._client.room_send(room_id, "m.room.message", content)
        event_id = getattr(response, "event_id", None)
        if event_id is None:
            msg = f"room_send failed: {getattr(response, 'message', response)}"
            raise RuntimeError(msg)
        return str(event_id)

    async def _sync(self) -> SyncResponse:
        await self._refresh_token()
        response = await self._client.sync(timeout=self._sync_timeout_ms, since=self._since, full_state=False)
        if isinstance(response, SyncResponse):
            self._since = response.next_batch
        return response

    async def backfill(self) -> list[RcpMessage]:
        """Fetch the *whole* currently-known RCP history of the room.

        The first ``/sync`` window is small (10 events by default) and stays
        clamped at 10 even when a longer ``limit`` is requested without a
        filter, so a single sync returns only the tail of a busy campaign.  The
        registry is projected from this replay, so stopping at the tail would
        silently drop every run older than the last ten events -- exactly the
        "a live job reads as gone after a restart" failure.  This walks the
        ``prev_batch`` token back through ``/messages`` until the room's start
        (or the safety cap) and returns the events oldest-first.

        Returns:
            The RCP messages in the room timeline at this point, oldest first.

        """
        response = await self._sync()
        messages: list[RcpMessage] = []
        joined = getattr(response.rooms, "join", {}) or {}
        room = joined.get(self._room_id)
        prev_batch = getattr(getattr(room, "timeline", None), "prev_batch", None) if room is not None else None
        self.history_complete = True
        while prev_batch and len(messages) < _MAX_BACKFILL_EVENTS:
            page = await self._backfill_page(prev_batch)
            if page is None:
                # A pagination error leaves the history partial: do not claim
                # completeness (an absent run may simply be off the page).
                self.history_complete = False
                break
            older, prev_batch = page
            if not older:
                break
            messages = older + messages
        if len(messages) >= _MAX_BACKFILL_EVENTS:
            self.history_complete = False
        # The sync window is the newest slice; the paged events are older, so
        # they precede it to keep the whole replay oldest-first.
        return messages + self._extract(response)

    async def _backfill_page(self, from_token: str) -> tuple[list[RcpMessage], str | None] | None:
        """Fetch one older page of the room timeline via ``/messages``.

        Args:
            from_token: The ``prev_batch`` token to page backwards from.

        Returns:
            ``(messages_oldest_first, next_prev_token)``; ``next_prev_token`` is
            None at the room start.  Returns None on a request failure so the
            caller degrades to a partial (incomplete) history rather than
            aborting the whole backfill.

        """
        await self._refresh_token()
        try:
            response = await self._client.room_messages(
                self._room_id,
                start=from_token,
                direction=MessageDirection.back,
                limit=_BACKFILL_PAGE_LIMIT,
            )
        except Exception as exc:  # ruff: ignore[blind-except] - a pagination error must not kill startup  # pragma: no cover
            log.warning("matrix backfill pagination failed: %s", exc)
            return None
        if not isinstance(response, RoomMessagesResponse):
            log.warning("matrix backfill pagination error: %s", getattr(response, "message", response))
            return None
        # ``/messages`` returns events newest-first for a backward page; reverse
        # to keep the accumulated history oldest-first.
        page = self._extract_events(list(reversed(response.chunk)))
        return page, response.end

    async def human_messages(self) -> list[tuple[str, str, str]]:
        """Return plain (non-RCP) text messages since the last call.

        A human room carries ordinary ``m.room.message`` events rather than
        signed RCP envelopes.  Each returned tuple is
        ``(room_id, sender, body)``; the caller applies the bot/room filters.
        This shares the sync position with :meth:`receive`, so a caller must not
        run both loops against the same transport instance -- use
        :meth:`drain` to consume both in one loop.

        Returns:
            The ``(room_id, sender, body)`` tuples from joined rooms.

        """
        _rcp, human = await self.drain()
        return human

    async def drain(self) -> tuple[list[RcpMessage], list[tuple[str, str, str]]]:
        """Run one sync and return both the RCP messages and the human chat.

        Keeps a single sync position for both streams, so a server pump can
        consume signed RCP envelopes and ordinary human messages in one loop.

        Returns:
            A ``(rcp_messages, human_messages)`` pair; human messages are
            ``(room_id, sender, body)`` tuples from joined rooms.

        """
        response = await self._sync()
        rcp = self._extract(response)
        human: list[tuple[str, str, str]] = []
        joined = getattr(response.rooms, "join", {}) or {}
        for room_id, room in joined.items():
            for event in room.timeline.events:
                if not isinstance(event, RoomMessageText):
                    continue
                content = getattr(event, "source", {}).get("content")
                if isinstance(content, dict) and RCP_NAMESPACE in content:
                    continue  # an RCP message, not human chat
                human.append((room_id, event.sender, event.body))
        return rcp, human

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
        return self._extract_events(list(room.timeline.events))

    @staticmethod
    def _extract_events(events: list[Any]) -> list[RcpMessage]:
        """Extract the RCP messages from a list of Matrix timeline events.

        Returns:
            The verified-shaped RCP messages, in the events' own order; a
            non-RCP or malformed event is skipped.

        """
        messages: list[RcpMessage] = []
        for event in events:
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
