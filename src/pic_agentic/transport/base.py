"""Transport interface."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from pic_agentic.rcp.envelope import RcpMessage


class Transport(Protocol):
    """Minimal async send/receive contract shared by all transports."""

    async def send(self, message: RcpMessage) -> str:
        """Send ``message``; return a transport-level event id."""

    def receive(self) -> AsyncIterator[RcpMessage]:
        """Yield inbound :class:`RcpMessage` objects (valid or not signed)."""

    async def close(self) -> None:
        """Release transport resources."""
