# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Unit tests for the transport-agnostic RCP core."""

from __future__ import annotations

import pytest

from pic_agentic.rcp import (
    RCP_NAMESPACE,
    SIGNED_FIELDS,
    VERSION,
    DedupStore,
    Kind,
    RcpMessage,
    SenderRole,
    SequenceState,
    canonical_bytes,
    new_cmd_id,
    new_secret_hex,
    sign,
    verify,
)

SECRET = "0123456789abcdef" * 2


def make(kind=Kind.COMMAND, type_="hello", seq=1, role=SenderRole.MCP_SERVER, payload=None) -> RcpMessage:
    return RcpMessage(
        sim="7f3a2b1c",
        kind=kind,
        type=type_,
        seq=seq,
        sender_role=role,
        payload=payload if payload is not None else {"cmd_id": new_cmd_id(), "message": "Hello World"},
    )


def test_canonical_bytes_is_deterministic_and_sorted() -> None:
    assert canonical_bytes({"b": 1, "a": {"d": 2, "c": 3}}) == b'{"a":{"c":3,"d":2},"b":1}'


def test_sign_verify_roundtrip() -> None:
    msg = make()
    msg.sign(SECRET)
    assert msg.verify(SECRET)


def test_verify_rejects_tampered_payload() -> None:
    msg = make()
    msg.sign(SECRET)
    msg.payload["message"] = "tampered"
    assert not msg.verify(SECRET)


def test_verify_binds_sender_role_and_in_reply_to() -> None:
    msg = make()
    msg.sign(SECRET)
    msg.sender_role = SenderRole.SIMCLIENT
    assert not msg.verify(SECRET)
    msg2 = make(kind=Kind.ACK, type_="hello_ack")
    msg2.sign(SECRET)
    msg2.in_reply_to = "$someEvent"
    assert not msg2.verify(SECRET)


def test_verify_rejects_missing_or_foreign_signature() -> None:
    assert not make().verify(SECRET)
    assert not verify(SECRET, {"a": 1}, "not-a-sig")
    assert not verify(SECRET, {"a": 1}, "hmac-sha256:deadbeef")
    good = sign(SECRET, {"a": 1})
    assert not verify("another-secret", {"a": 1}, good)


def test_envelope_dict_roundtrip_through_matrix_content() -> None:
    msg = make().sign(SECRET)
    content = msg.to_content(SECRET)
    assert content["msgtype"] == "m.text"
    assert content["body"].startswith("[rcp] sim=7f3a2b1c hello")
    assert RCP_NAMESPACE in content
    restored = RcpMessage.from_content(content)
    assert restored.to_dict() == msg.to_dict()
    assert restored.verify(SECRET)


def test_body_stays_human_readable() -> None:
    msg = make(payload={"cmd_id": "abc123", "message": "Hello World"})
    assert "Hello World" in msg.body()
    long = make(payload={"cmd_id": "abc", "message": "x" * 200})
    assert "..." in long.body()
    assert len(long.body()) < 160


def test_per_sender_sequence_counts_independently() -> None:
    state = SequenceState()
    assert state.next_seq("simA", SenderRole.MCP_SERVER) == 1
    assert state.next_seq("simA", SenderRole.MCP_SERVER) == 2
    assert state.next_seq("simA", SenderRole.SIMCLIENT) == 1
    assert state.next_seq("simB", SenderRole.MCP_SERVER) == 1
    state.observe("simA", SenderRole.SIMCLIENT, 5)
    assert state.highest("simA", SenderRole.SIMCLIENT) == 5


def test_dedup_keys_on_sim_role_seq_type() -> None:
    store = DedupStore()
    first = make(seq=1)
    duplicate = make(seq=1)
    assert store.seen(first) is True
    assert store.seen(duplicate) is False
    other_role = make(seq=1, role=SenderRole.SIMCLIENT)
    assert store.seen(other_role) is True
    other_type = make(seq=1, type_="ping")
    assert store.seen(other_type) is True
    assert store.seen(make(seq=2)) is True


def test_dedup_store_is_bounded() -> None:
    store = DedupStore(maxlen=2)
    msgs = [make(seq=i) for i in range(3)]
    for m in msgs:
        store.seen(m)
    assert store.seen(make(seq=0)) is True


def test_new_secret_and_cmd_id_shapes() -> None:
    s = new_secret_hex()
    assert len(s) == 64
    assert int(s, 16) >= 0
    c = new_cmd_id()
    assert len(c) == 32
    assert int(c, 16) >= 0


@pytest.mark.parametrize("kind", list(Kind))
def test_all_kinds_roundtrip(kind) -> None:
    msg = make(kind=kind).sign(SECRET)
    assert RcpMessage.from_dict(msg.to_dict()).kind is kind


def test_model_validates_and_coerces_types() -> None:
    # A raw wire dict with string numbers and a foreign payload key validates,
    # because pydantic parses the declared field types.
    msg = RcpMessage.model_validate(
        {"sim": "s", "kind": "command", "type": "hello", "seq": "7", "sender_role": "mcpserver"},
    )
    assert msg.seq == 7
    assert msg.version == VERSION
    assert msg.payload == {}


def test_model_rejects_unknown_kind() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RcpMessage.model_validate({"sim": "s", "kind": "bogus", "type": "t", "seq": 1, "sender_role": "mcpserver"})


def test_wire_dict_excludes_transport_metadata() -> None:
    msg = make().sign(SECRET)
    msg.transport_sender = "@someone:hs"
    msg.transport_event_id = "$evt"
    wire = msg.to_dict()
    assert "transport_sender" not in wire
    assert "transport_event_id" not in wire
    # The signed subset is exactly the documented signed fields.
    assert set(msg.signed_payload()) == SIGNED_FIELDS


def test_sig_not_covered_by_signature() -> None:
    msg = make().sign(SECRET)
    msg.sig = "hmac-sha256:" + "0" * 64
    # A forged sig does not verify, but the signed payload is unaffected by it.
    assert not msg.verify(SECRET)
