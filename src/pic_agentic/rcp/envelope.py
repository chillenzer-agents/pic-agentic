# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""RCP message envelope (version 0).

The envelope is a pydantic model: validation and (de)serialisation come from
the model itself, and the signed payload is a filtered JSON dump so the wire
format and the signed bytes cannot drift apart.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pic_agentic.rcp.crypto import sign, verify

RCP_NAMESPACE = "io.picongpu.rcp"
VERSION = 0

#: Envelope fields that participate in the HMAC (everything except ``sig`` and
#: the transport-only metadata fields).
SIGNED_FIELDS = frozenset({"version", "sim", "kind", "type", "seq", "ts", "sender_role", "in_reply_to", "payload"})

#: Payload keys echoed into the human-readable room body, in this order.
_BODY_KEYS = ("cmd_id", "job_id", "submit_system", "state", "results_linked", "step", "percent", "message")

#: Maximum length of a payload value echoed into the room body.
_BODY_VALUE_MAX = 60
#: Suffix appended to a truncated body value.
_BODY_ELLIPSIS = "..."


class Kind(StrEnum):
    """What an RCP message is: an event, a command, or an acknowledgement."""

    EVENT = "event"
    COMMAND = "command"
    ACK = "ack"


class SenderRole(StrEnum):
    """Which RCP party a message originates from."""

    SIMCLIENT = "simclient"
    MCP_SERVER = "mcpserver"


def now_ts() -> str:
    """Return the current UTC timestamp in the RCP canonical form.

    Returns:
        A ``YYYY-MM-DDTHH:MM:SSZ`` (seconds resolution) timestamp.

    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class RcpMessage(BaseModel):
    """A single RCP message carried in one ``m.room.message`` event."""

    # ``type`` intentionally shadows the builtin: it is the protocol field name.
    model_config = ConfigDict(extra="forbid")

    sim: str
    kind: Kind
    type: str
    seq: int
    sender_role: SenderRole
    payload: dict[str, Any] = Field(default_factory=dict)
    ts: str = Field(default_factory=now_ts)
    in_reply_to: str | None = None
    version: int = VERSION
    sig: str | None = None
    #: Matrix user id of the transport-level sender.  Set by the transport on
    #: receive; NOT part of the signed envelope (it is transport metadata used
    #: for the design's defence-in-depth identity check, section 6.4).
    transport_sender: str | None = Field(default=None, exclude=True)
    #: Transport-level event id of the message that carried this envelope.
    #: Set by the transport on receive; used to fill ``in_reply_to`` on acks.
    transport_event_id: str | None = Field(default=None, exclude=True)

    def signed_payload(self) -> dict[str, Any]:
        """Return the envelope fields covered by the signature.

        Returns:
            The signed fields as a JSON-mode mapping (enums as strings), which
            is exactly the subset hashed by :func:`~pic_agentic.rcp.crypto.sign`.

        """
        return self.model_dump(mode="json", include=set(SIGNED_FIELDS))

    def to_dict(self) -> dict[str, Any]:
        """Return the full wire envelope including the signature.

        Returns:
            The signed fields plus ``sig``; the transport-only metadata fields
            are excluded by the model.

        """
        return self.model_dump(mode="json")

    def sign(self, secret: str) -> RcpMessage:
        """Sign this message in place.

        Returns:
            ``self``, signed, for call chaining.

        """
        self.sig = sign(secret, self.signed_payload())
        return self

    def verify(self, secret: str) -> bool:
        """Check this message's signature.

        Returns:
            True if the envelope is signed and the signature verifies.

        """
        if self.sig is None:
            return False
        return verify(secret, self.signed_payload(), self.sig)

    def dedup_key(self) -> tuple[str, ...]:
        """Return the receiver's deduplication key.

        Deduplication guards against at-least-once *delivery*, so the primary
        key is the transport event id, which is unique per delivered event and
        is set by the transport on receive.

        The design's ``(sim, sender_role, seq, type)`` fallback is only safe for
        a single process lifetime: ``seq`` is an in-memory per-sender counter
        that restarts at 1 whenever the sending process restarts, so two
        *distinct* commands from restarted MCP servers would otherwise collide
        and the second would be silently dropped (observed live). Idempotency
        of re-sent commands is handled separately by the ``cmd_id`` guard.

        Returns:
            The transport event id when present, else
            ``(sim, sender_role, seq, type)``.

        """
        if self.transport_event_id:
            return (self.transport_event_id,)
        return (self.sim, self.sender_role.value, str(self.seq), self.type)

    def body(self) -> str:
        """Return the short human-readable room line.

        Returns:
            A one-line summary so the room stays readable in Element.

        """
        parts = [f"[rcp] sim={self.sim}", self.type]
        for key in _BODY_KEYS:
            if key in self.payload and isinstance(self.payload[key], (str, int, float, bool)):
                value = str(self.payload[key])
                if len(value) > _BODY_VALUE_MAX:
                    keep = _BODY_VALUE_MAX - len(_BODY_ELLIPSIS)
                    value = value[:keep] + _BODY_ELLIPSIS
                parts.append(f"{key}={value}")
        return " ".join(parts)

    def to_content(self, secret: str | None = None) -> dict[str, Any]:
        """Return the Matrix ``content`` dict for a single ``m.room.message``.

        Signs with ``secret`` if the message is not signed yet.

        Args:
            secret: Per-simulation RCP secret, required for unsigned messages.

        Returns:
            The Matrix event content carrying the RCP envelope.

        Raises:
            ValueError: If the message is unsigned and no secret was provided.

        """
        if self.sig is None:
            if secret is None:
                msg = "message is unsigned and no secret was provided"
                raise ValueError(msg)
            self.sign(secret)
        return {"msgtype": "m.text", "body": self.body(), RCP_NAMESPACE: self.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RcpMessage:
        """Build a message from a raw wire envelope mapping.

        Args:
            data: The envelope mapping, typically from a Matrix event.

        Returns:
            The validated message.

        """
        return cls.model_validate(data)

    @classmethod
    def from_content(cls, content: dict[str, Any]) -> RcpMessage:
        """Extract the RCP envelope from a Matrix ``content`` dict.

        Args:
            content: A Matrix event content mapping.

        Returns:
            The validated message.

        Raises:
            TypeError: If ``content`` carries no RCP envelope mapping.

        """
        raw = content.get(RCP_NAMESPACE)
        if not isinstance(raw, dict):
            msg = f"content has no {RCP_NAMESPACE} envelope"
            raise TypeError(msg)
        return cls.from_dict(raw)
