# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""In-process transport for tests and the offline direct-echo M1 variant."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pic_agentic.rcp.envelope import RcpMessage

#: Poll interval for the receive iterator so ``close`` is noticed promptly.
_POLL_INTERVAL_S = 0.05


class _ChannelState:
    """Closure flag shared by both ends of one channel."""

    def __init__(self) -> None:
        """Start an open channel with no messages sent yet."""
        self.closed = False
        self.counter = 0


class MemoryTransport:
    """One end of a two-party in-process channel.

    Use :meth:`create_pair` to obtain two wired ends.  Messages sent on one end
    appear on the other end's :meth:`receive` iterator.

    This is deliberately a plain class, not a pydantic model: it holds live
    :class:`asyncio.Queue` objects that are not data to validate or serialise.
    """

    def __init__(
        self,
        inbox: asyncio.Queue[RcpMessage] | None = None,
        peer_inbox: asyncio.Queue[RcpMessage] | None = None,
        state: _ChannelState | None = None,
    ) -> None:
        """Create an unpaired end; prefer :meth:`create_pair`."""
        self.inbox: asyncio.Queue[RcpMessage] = inbox if inbox is not None else asyncio.Queue()
        self.peer_inbox = peer_inbox
        self.state = state if state is not None else _ChannelState()
        #: Human-facing messages (text/image) sent by this end, in order.
        self.sent_text: list[tuple[str, str]] = []
        self.sent_images: list[tuple[str, bytes, str]] = []
        #: Inbound human room messages for :meth:`drain` (test-only).
        self.human_inbox: asyncio.Queue[tuple[str, str, str]] = asyncio.Queue()

    @classmethod
    def create_pair(cls) -> tuple[MemoryTransport, MemoryTransport]:
        """Create two wired ends of one channel.

        Returns:
            The ``(left, right)`` transport pair.

        """
        left, right = cls(), cls()
        left.peer_inbox = right.inbox
        right.peer_inbox = left.inbox
        right.state = left.state
        return left, right

    @property
    def closed(self) -> bool:
        """Whether the channel has been closed."""
        return self.state.closed

    async def send(self, message: RcpMessage) -> str:
        """Queue ``message`` for the peer end.

        Args:
            message: The RCP message to deliver.

        Returns:
            A synthetic transport event id.

        Raises:
            RuntimeError: If the transport is closed or unpaired.

        """
        if self.closed:
            msg = "transport is closed"
            raise RuntimeError(msg)
        if self.peer_inbox is None:
            msg = "transport is not paired"
            raise RuntimeError(msg)
        self.state.counter += 1
        event_id = f"$memory{self.state.counter}"
        # Mirror MatrixTransport: stamp the transport event id on receive so
        # DedupStore attributes each delivery uniquely (the seq counter resets
        # per process and must not be the primary dedup key).
        message.transport_event_id = event_id
        await self.peer_inbox.put(message)
        return event_id

    async def receive(self) -> AsyncIterator[RcpMessage]:
        """Yield messages delivered by the peer until the channel closes.

        Yields:
            Each inbound :class:`RcpMessage`.

        """
        while not self.closed:
            try:
                message = await asyncio.wait_for(self.inbox.get(), timeout=_POLL_INTERVAL_S)
            except TimeoutError:
                continue
            yield message

    async def send_text(self, room_id: str, text: str) -> str:
        """Record a human-facing text message (test double).

        Returns:
            A synthetic transport event id.

        """
        self.state.counter += 1
        self.sent_text.append((room_id, text))
        return f"$memory-text{self.state.counter}"

    async def send_image(
        self,
        room_id: str,
        png: bytes,
        *,
        body: str = "image",
        filename: str = "plot.png",
        mimetype: str = "image/png",
    ) -> str:
        """Record a human-facing image message (test double).

        Returns:
            A synthetic transport event id.

        """
        _ = filename, mimetype
        self.state.counter += 1
        self.sent_images.append((room_id, png, body))
        return f"$memory-image{self.state.counter}"

    async def drain(self) -> tuple[list[RcpMessage], list[tuple[str, str, str]]]:
        """Non-blocking drain of the inbox (test double for the human pump).

        Returns:
            A ``(rcp_messages, human_messages)`` pair; ``human_messages`` are
            read from a companion :attr:`human_inbox` if one is set.

        """
        rcp: list[RcpMessage] = []
        while not self.inbox.empty():
            rcp.append(self.inbox.get_nowait())
        human: list[tuple[str, str, str]] = []
        while not self.human_inbox.empty():
            human.append(self.human_inbox.get_nowait())
        await asyncio.sleep(0)  # yield so a pump loop can be driven in tests
        return rcp, human

    async def close(self) -> None:
        """Close the channel for both ends."""
        self.state.closed = True
