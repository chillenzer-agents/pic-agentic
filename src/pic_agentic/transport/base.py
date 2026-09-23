# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Transport interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pic_agentic.rcp.envelope import RcpMessage


class Transport(Protocol):
    """Minimal async send/receive contract shared by all transports."""

    async def send(self, message: RcpMessage) -> str:
        """Send ``message``; return a transport-level event id."""

    def receive(self) -> AsyncIterator[RcpMessage]:
        """Yield inbound :class:`RcpMessage` objects (valid or not signed)."""

    async def close(self) -> None:
        """Release transport resources."""
