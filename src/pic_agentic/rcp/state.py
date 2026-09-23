# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Per-sender sequencing and deduplication state."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pic_agentic.rcp.envelope import RcpMessage, SenderRole


class SequenceState:
    """Monotonic ``seq`` counter, one per ``(sim, sender_role)``.

    Two senders share a simulation (simclient + MCP server); a single merged
    counter would create spurious gaps, so the counter is per sender.
    """

    def __init__(self) -> None:
        """Start with no observed sequences."""
        self._counters: dict[tuple[str, str], int] = {}

    def next_seq(self, sim: str, role: SenderRole) -> int:
        """Return and advance the sender's next sequence number.

        Args:
            sim: Simulation id.
            role: Sender role whose counter to advance.

        Returns:
            The next sequence number (1-based).

        """
        key = (sim, role.value)
        value = self._counters.get(key, 0) + 1
        self._counters[key] = value
        return value

    def observe(self, sim: str, role: SenderRole, seq: int) -> None:
        """Track the highest sequence number seen for a sender.

        Args:
            sim: Simulation id.
            role: Sender role the sequence belongs to.
            seq: Observed sequence number.

        """
        key = (sim, role.value)
        if seq > self._counters.get(key, 0):
            self._counters[key] = seq

    def highest(self, sim: str, role: SenderRole) -> int:
        """Return the highest sequence number seen for a sender.

        Args:
            sim: Simulation id.
            role: Sender role to query.

        Returns:
            The highest observed sequence number, or 0 if none was seen.

        """
        return self._counters.get((sim, role.value), 0)


class DedupStore:
    """Bounded set of ``(sim, sender_role, seq, type)`` keys already processed.

    Matrix delivery is at-least-once, so receivers must drop duplicates.
    """

    def __init__(self, maxlen: int = 4096) -> None:
        """Create a store retaining at most ``maxlen`` recent keys."""
        self._seen: set[tuple[str, str, int, str]] = set()
        self._order: deque[tuple[str, str, int, str]] = deque()
        self._maxlen = maxlen

    def seen(self, message: RcpMessage) -> bool:
        """Record the message key and report whether it is new.

        Args:
            message: The received message.

        Returns:
            True if the key is new; False if it is a duplicate.

        """
        key = message.dedup_key()
        if key in self._seen:
            return False
        self._seen.add(key)
        self._order.append(key)
        if len(self._order) > self._maxlen:
            self._seen.discard(self._order.popleft())
        return True
