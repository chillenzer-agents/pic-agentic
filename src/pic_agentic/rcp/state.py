"""Per-sender sequencing and deduplication state."""

from __future__ import annotations

from collections import deque

from pic_agentic.rcp.envelope import RcpMessage, SenderRole


class SequenceState:
    """Monotonic ``seq`` counter, one per ``(sim, sender_role)``.

    Two senders share a simulation (simclient + MCP server); a single merged
    counter would create spurious gaps, so the counter is per sender.
    """

    def __init__(self) -> None:
        self._counters: dict[tuple[str, str], int] = {}

    def next_seq(self, sim: str, role: SenderRole) -> int:
        key = (sim, role.value)
        value = self._counters.get(key, 0) + 1
        self._counters[key] = value
        return value

    def observe(self, sim: str, role: SenderRole, seq: int) -> None:
        """Track the highest seen seq (used by receivers)."""
        key = (sim, role.value)
        if seq > self._counters.get(key, 0):
            self._counters[key] = seq

    def highest(self, sim: str, role: SenderRole) -> int:
        return self._counters.get((sim, role.value), 0)


class DedupStore:
    """Bounded set of ``(sim, sender_role, seq, type)`` keys already processed.

    Matrix delivery is at-least-once, so receivers must drop duplicates.
    """

    def __init__(self, maxlen: int = 4096) -> None:
        self._seen: set[tuple[str, str, int, str]] = set()
        self._order: deque[tuple[str, str, int, str]] = deque()
        self._maxlen = maxlen

    def seen(self, message: RcpMessage) -> bool:
        """Return True and record the key if it is new; False if a duplicate."""
        key = message.dedup_key()
        if key in self._seen:
            return False
        self._seen.add(key)
        self._order.append(key)
        if len(self._order) > self._maxlen:
            self._seen.discard(self._order.popleft())
        return True
