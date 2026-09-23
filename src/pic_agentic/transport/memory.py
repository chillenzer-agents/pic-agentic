"""In-process transport for tests and the offline direct-echo M1 variant."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from pic_agentic.rcp.envelope import RcpMessage


class MemoryTransport:
    """One end of a two-party in-process channel.

    Use :meth:`create_pair` to obtain two wired ends.  Messages sent on one end
    appear on the other end's ``receive()`` iterator.
    """

    def __init__(self) -> None:
        self._inbox: asyncio.Queue[RcpMessage] = asyncio.Queue()
        self._peer: MemoryTransport | None = None
        self._counter = 0
        self.closed = False

    @classmethod
    def create_pair(cls) -> tuple[MemoryTransport, MemoryTransport]:
        left, right = cls(), cls()
        left._peer = right
        right._peer = left
        return left, right

    async def send(self, message: RcpMessage) -> str:
        if self.closed:
            raise RuntimeError("transport is closed")
        if self._peer is None:
            raise RuntimeError("transport is not paired")
        self._counter += 1
        await self._peer._inbox.put(message)
        return f"$memory{self._counter}"

    async def receive(self) -> AsyncIterator[RcpMessage]:
        while not self.closed:
            try:
                message = await asyncio.wait_for(self._inbox.get(), timeout=0.05)
            except (asyncio.TimeoutError, TimeoutError):
                continue
            yield message

    async def close(self) -> None:
        self.closed = True
        if self._peer is not None:
            self._peer.closed = True
