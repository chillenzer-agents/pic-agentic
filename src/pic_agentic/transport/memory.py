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
        await self.peer_inbox.put(message)
        return f"$memory{self.state.counter}"

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

    async def close(self) -> None:
        """Close the channel for both ends."""
        self.state.closed = True
