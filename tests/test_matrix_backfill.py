# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for the Matrix transport's full-history backfill (F1).

The registry the server rebuilds on startup is projected from the signed-room
replay.  ``/sync`` returns only a small window (10 events by default), so a
single sync drops every run older than the last ten events -- observed live as
"a still-RUNNING job reads as ``unknown_sim`` after a restart".  These tests
drive :meth:`MatrixTransport.backfill` against a fake matrix-nio client whose
sync window is truncated and whose ``/messages`` pages carry the older events,
and assert the whole history is reconstructed (and marked complete).
"""

from __future__ import annotations

from typing import Any

from nio import MessageDirection, RoomMessagesResponse, SyncResponse

from pic_agentic.protocol.simulation import (
    SimulationState,
    build_submit_event,
)
from pic_agentic.rcp import RcpMessage, new_secret_hex
from pic_agentic.server.simulation import SubmitService
from pic_agentic.transport.matrix import MatrixTransport

SECRET = new_secret_hex()
SIM = "7f3a2b1c"
ROOM = "!room:hs"
SIM_ID = "699b6e01"  # the transcript's live-at-restart run
RUN_CMD = "run-4"


def _event(state: SimulationState, *, seq: int, ts: str, job_id: int | None = None, **fields: Any) -> RcpMessage:
    message = build_submit_event(sim=SIM, seq=seq, cmd_id=RUN_CMD, sim_id=SIM_ID, state=state, job_id=job_id, **fields)
    message.ts = ts
    return message.sign(SECRET)


def _matrix_event(message: RcpMessage, event_id: str) -> dict[str, Any]:
    content = message.to_content()
    content["body"] = message.body()
    return {
        "type": "m.room.message",
        "event_id": event_id,
        "sender": "@simclient:hs",
        "origin_server_ts": 1,
        "content": content,
    }


class _FakeClient:
    """A matrix-nio client double with a truncated sync and a ``/messages`` page."""

    def __init__(self, recent: list[dict[str, Any]], older: list[dict[str, Any]], prev_batch: str | None) -> None:
        """Create the fake with a recent sync window and one older page."""
        self.access_token = "tok"
        self._recent = recent
        self._older = older
        self._prev_batch = prev_batch
        self.messages_calls: list[dict[str, Any]] = []

    async def sync(self, **kwargs: Any) -> SyncResponse:
        """Return a sync whose timeline holds only the recent window."""
        _ = kwargs
        timeline: dict[str, Any] = {"events": self._recent, "limited": True}
        if self._prev_batch is not None:
            timeline["prev_batch"] = self._prev_batch
        return SyncResponse.from_dict(
            {
                "next_batch": "n1",
                "rooms": {
                    "join": {
                        ROOM: {
                            "timeline": timeline,
                            "state": {"events": []},
                            "ephemeral": {"events": []},
                            "account_data": {"events": []},
                            "unread_notifications": {},
                            "summary": {},
                        }
                    }
                },
                "device_lists": {},
                "device_one_time_keys_count": {},
                "to_device": {"events": []},
                "presence": {"events": []},
                "account_data": {"events": []},
            }
        )

    async def room_messages(
        self,
        room_id: str,
        start: str | None = None,
        *,
        direction: MessageDirection = MessageDirection.back,
        limit: int = 10,
    ) -> RoomMessagesResponse:
        """Return one older page of events (newest-first, as the API does)."""
        self.messages_calls.append({"start": start, "direction": direction, "limit": limit})
        return RoomMessagesResponse.from_dict(
            {"start": start or "", "chunk": list(reversed(self._older))},
            room_id,
        )

    @staticmethod
    async def close() -> None:
        """No-op close."""


def _transport(client: _FakeClient) -> MatrixTransport:
    transport = MatrixTransport("https://hs", "@mcp:hs", "tok", ROOM, store_path="/tmp/nio-backfill-test")
    transport._client = client  # type: ignore[assignment]
    return transport


async def test_backfill_paginates_to_reconstruct_older_runs() -> None:
    """A run older than the sync window is restored by back-pagination (F1)."""
    older = [
        _matrix_event(_event(SimulationState.SUBMITTED, seq=1, ts="2026-10-07T12:00:00Z", job_id=4242), "$old1"),
        _matrix_event(
            _event(SimulationState.JOB_RUNNING, seq=2, ts="2026-10-07T12:00:30Z", job_id=4242, slurm_state="RUNNING"),
            "$old2",
        ),
    ]
    # The sync window holds only a later, unrelated event (a different run).
    recent = [_matrix_event(_event(SimulationState.RESULTS_READY, seq=9, ts="2026-10-07T13:00:00Z", job_id=9), "$new1")]
    client = _FakeClient(recent, older, prev_batch="p1")
    transport = _transport(client)

    backfilled = await transport.backfill()
    assert transport.history_complete is True
    # The older run's events precede the newer window, oldest first.
    states = [message.payload.get("state") for message in backfilled]
    assert states == [
        SimulationState.SUBMITTED.value,
        SimulationState.JOB_RUNNING.value,
        SimulationState.RESULTS_READY.value,
    ]
    assert client.messages_calls == [{"start": "p1", "direction": MessageDirection.back, "limit": 100}]

    # The reconstructed replay restores the live run in the registry.
    service = SubmitService(sim=SIM, secret=SECRET, ack_timeout_s=0.05)
    service.ingest_backfill(backfilled)
    record = service.get(SIM_ID)
    assert record is not None
    assert record.state in {SimulationState.JOB_RUNNING.value, SimulationState.RESULTS_READY.value}


async def test_backfill_marks_incomplete_when_pagination_fails() -> None:
    """A pagination error yields a partial history marked incomplete (F1)."""

    class _FailingClient(_FakeClient):
        @staticmethod
        async def room_messages(*_args: Any, **_kwargs: Any) -> Any:
            msg = "boom"
            raise RuntimeError(msg)

    recent = [_matrix_event(_event(SimulationState.JOB_RUNNING, seq=2, ts="2026-10-07T13:00:00Z", job_id=9), "$r1")]
    client = _FailingClient(recent, [], prev_batch="p1")
    transport = _transport(client)

    backfilled = await transport.backfill()
    assert transport.history_complete is False
    assert len(backfilled) == 1


async def test_backfill_without_prev_batch_stays_complete() -> None:
    """An empty/complete room reports a complete history and no paging."""
    recent = [_matrix_event(_event(SimulationState.JOB_RUNNING, seq=1, ts="2026-10-07T13:00:00Z", job_id=1), "$r1")]
    client = _FakeClient(recent, [], prev_batch=None)
    transport = _transport(client)
    backfilled = await transport.backfill()
    assert transport.history_complete is True
    assert len(backfilled) == 1
    assert client.messages_calls == []
