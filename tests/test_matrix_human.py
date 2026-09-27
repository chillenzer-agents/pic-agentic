# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Unit tests for the Matrix transport's human text/image send."""

from __future__ import annotations

from dataclasses import dataclass

from pic_agentic.transport.matrix import MatrixTransport


@dataclass
class _FakeResponse:
    event_id: str | None = None
    content_uri: str | None = None
    message: str = ""


class _FakeClient:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, dict]] = []
        self.uploads: list[tuple[bytes, str, str]] = []
        self.fail_send = False
        self.fail_upload = False
        self.access_token = ""

    async def room_send(self, room_id, event_type, content):
        if self.fail_send:
            return _FakeResponse(message="boom")
        self.sent.append((room_id, event_type, content))
        return _FakeResponse(event_id=f"$ev{len(self.sent)}")

    async def upload(self, data, content_type="", filename=None):
        if self.fail_upload:
            return _FakeResponse(message="upload boom"), None
        self.uploads.append((data, content_type, filename))
        return _FakeResponse(content_uri="mxc://hs/abc"), None

    @staticmethod
    async def close() -> None:
        return None


def _transport() -> tuple[MatrixTransport, _FakeClient]:
    transport = MatrixTransport("https://hs", "@bot:hs", "tok", "!room:hs", store_path="/tmp/nio-test")
    client = _FakeClient()
    transport._client = client  # type: ignore[assignment]
    return transport, client


async def test_send_text() -> None:
    transport, client = _transport()
    event_id = await transport.send_text("!human:hs", "hello")
    assert event_id == "$ev1"
    room_id, event_type, content = client.sent[0]
    assert room_id == "!human:hs"
    assert event_type == "m.room.message"
    assert content == {"msgtype": "m.text", "body": "hello"}


async def test_send_image_uploads_then_sends() -> None:
    transport, client = _transport()
    event_id = await transport.send_image("!human:hs", b"PNGDATA", body="plot", filename="p.png")
    assert event_id == "$ev1"
    assert client.uploads == [(b"PNGDATA", "image/png", "p.png")]
    _room, _type, content = client.sent[0]
    assert content["msgtype"] == "m.image"
    assert content["url"] == "mxc://hs/abc"
    assert content["body"] == "plot"
    assert content["info"]["size"] == len(b"PNGDATA")


async def test_send_image_upload_failure_raises() -> None:
    transport, client = _transport()
    client.fail_upload = True
    try:
        await transport.send_image("!human:hs", b"x")
    except RuntimeError as exc:
        assert "media upload failed" in str(exc)


async def test_send_failure_raises() -> None:
    transport, client = _transport()
    client.fail_send = True
    try:
        await transport.send_text("!human:hs", "x")
    except RuntimeError as exc:
        assert "room_send failed" in str(exc)
