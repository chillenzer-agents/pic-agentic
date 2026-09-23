"""RCP message envelope (version 0)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pic_agentic.rcp.crypto import sign, verify

RCP_NAMESPACE = "io.picongpu.rcp"
VERSION = 0

#: Envelope fields that participate in the HMAC (everything except ``sig``).
SIGNED_FIELDS = ("version", "sim", "kind", "type", "seq", "ts", "sender_role", "in_reply_to", "payload")

#: Payload keys echoed into the human-readable room body, in this order.
_BODY_KEYS = ("cmd_id", "job_id", "submit_system", "state", "step", "percent", "message")


class Kind(str, Enum):
    EVENT = "event"
    COMMAND = "command"
    ACK = "ack"


class SenderRole(str, Enum):
    SIMCLIENT = "simclient"
    MCP_SERVER = "mcpserver"


def now_ts() -> str:
    """UTC timestamp in the RCP canonical form (seconds resolution, ``Z``)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class RcpMessage:
    """A single RCP message carried in one ``m.room.message`` event."""

    sim: str
    kind: Kind
    type: str
    seq: int
    sender_role: SenderRole
    payload: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=now_ts)
    in_reply_to: str | None = None
    version: int = VERSION
    sig: str | None = None
    #: Matrix user id of the transport-level sender.  Set by the transport on
    #: receive; NOT part of the signed envelope (it is transport metadata used
    #: for the design's defence-in-depth identity check, section 6.4).
    transport_sender: str | None = None
    #: Transport-level event id of the message that carried this envelope.
    #: Set by the transport on receive; used to fill ``in_reply_to`` on acks.
    transport_event_id: str | None = None

    def signed_dict(self) -> dict[str, Any]:
        """Envelope fields covered by the signature, in canonical form."""
        data: dict[str, Any] = {
            "version": self.version,
            "sim": self.sim,
            "kind": self.kind.value,
            "type": self.type,
            "seq": self.seq,
            "ts": self.ts,
            "sender_role": self.sender_role.value,
            "in_reply_to": self.in_reply_to,
            "payload": self.payload,
        }
        return data

    def to_dict(self) -> dict[str, Any]:
        data = self.signed_dict()
        data["sig"] = self.sig
        return data

    def sign(self, secret: str) -> RcpMessage:
        self.sig = sign(secret, self.signed_dict())
        return self

    def verify(self, secret: str) -> bool:
        if self.sig is None:
            return False
        return verify(secret, self.signed_dict(), self.sig)

    def dedup_key(self) -> tuple[str, str, int, str]:
        """Per the design: ``(sim, sender_role, seq, type)``."""
        return (self.sim, self.sender_role.value, self.seq, self.type)

    def body(self) -> str:
        """Short human-readable line so rooms stay readable in Element."""
        parts = [f"[rcp] sim={self.sim}", self.type]
        for key in _BODY_KEYS:
            if key in self.payload and isinstance(self.payload[key], (str, int, float, bool)):
                value = str(self.payload[key])
                if len(value) > 60:
                    value = value[:57] + "..."
                parts.append(f"{key}={value}")
        return " ".join(parts)

    def to_content(self, secret: str | None = None) -> dict[str, Any]:
        """Matrix ``content`` dict for a single ``m.room.message``.

        Signs with ``secret`` if the message is not signed yet; raises if it is
        neither signed nor given a secret.
        """
        if self.sig is None:
            if secret is None:
                raise ValueError("message is unsigned and no secret was provided")
            self.sign(secret)
        return {"msgtype": "m.text", "body": self.body(), RCP_NAMESPACE: self.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RcpMessage:
        return cls(
            version=int(data.get("version", VERSION)),
            sim=str(data["sim"]),
            kind=Kind(data["kind"]),
            type=str(data["type"]),
            seq=int(data["seq"]),
            ts=str(data["ts"]),
            sender_role=SenderRole(data["sender_role"]),
            in_reply_to=data.get("in_reply_to"),
            payload=dict(data.get("payload") or {}),
            sig=data.get("sig"),
        )

    @classmethod
    def from_content(cls, content: dict[str, Any]) -> RcpMessage:
        """Extract the RCP envelope from a Matrix ``content`` dict."""
        raw = content.get(RCP_NAMESPACE)
        if not isinstance(raw, dict):
            raise ValueError(f"content has no {RCP_NAMESPACE} envelope")
        return cls.from_dict(raw)
