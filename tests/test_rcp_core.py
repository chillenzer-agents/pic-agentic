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


def test_dedup_prefers_transport_event_id() -> None:
    store = DedupStore()
    first = make(seq=1)
    first.transport_event_id = "$ev1"
    # A redelivery of the same event is dropped ...
    redelivery = make(seq=1)
    redelivery.transport_event_id = "$ev1"
    assert store.seen(first) is True
    assert store.seen(redelivery) is False
    # ... but a different event is not, even with the same seq/role/type.
    other = make(seq=1)
    other.transport_event_id = "$ev2"
    assert store.seen(other) is True


def test_dedup_distinct_commands_from_restarted_sender() -> None:
    """Two distinct commands after a sender restart share seq=1.

    Regression from the live cluster run: the MCP server's seq counter is
    in-memory and restarts at 1 per process, so dedup on (sim, role, seq, type)
    silently dropped every command after the first. Distinct transport events
    must both be processed.
    """
    store = DedupStore()
    a = make(seq=1, payload={"cmd_id": "a" * 32})
    a.transport_event_id = "$restart-1"
    b = make(seq=1, payload={"cmd_id": "b" * 32})
    b.transport_event_id = "$restart-2"
    assert store.seen(a) is True
    assert store.seen(b) is True


def test_dedup_falls_back_to_stable_key_without_event_id() -> None:
    store = DedupStore()
    first = make(seq=1)
    duplicate = make(seq=1)
    assert store.seen(first) is True
    assert store.seen(duplicate) is False
    assert store.seen(make(seq=1, role=SenderRole.SIMCLIENT)) is True
    assert store.seen(make(seq=1, type_="ping")) is True
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


def _synapse_canonical_json(value: object) -> object:
    """Mimic Synapse's ``canonicaljson``: raise on ANY float in event content.

    The reference homeserver refuses an event whose content carries a float
    (``Bad JSON value: float``), whether finite or not.  This encoder is the
    offline stand-in for that rule: if it accepts the content, Synapse will too.

    Returns:
        ``value`` unchanged, so it can be chained into a JSON dump.

    Raises:
        TypeError: If a float is found anywhere in ``value``.

    """
    if isinstance(value, float):
        msg = f"Synapse would reject the float {value!r}"
        raise TypeError(msg)
    if isinstance(value, dict):
        for item in value.values():
            _synapse_canonical_json(item)
    elif isinstance(value, list):
        for item in value:
            _synapse_canonical_json(item)
    return value


def test_no_float_survives_into_matrix_event_content() -> None:
    """A payload full of floats must be wire-safe (regression: live terminal event)."""
    msg = make(
        kind=Kind.EVENT,
        payload={
            "cmd_id": "c",
            "job_id": 571830,
            "core_hours": 0.0027777,
            "gpu_hours": 0.0,
            "data": [1.0, 2.5, 3.0],
            "stats": {"min": 1.0, "mean": 2.1666, "n": 3},
            "nested": [{"x": 0.5}],
        },
    ).sign(SECRET)
    content = msg.to_content()
    # The whole Matrix content object, not just the RCP envelope, is float-free.
    _synapse_canonical_json(content)
    import json

    json.dumps(content, allow_nan=False)  # also proves no NaN/Inf slipped through


def test_float_wire_encoding_roundtrips_and_verifies() -> None:
    msg = make(kind=Kind.EVENT, payload={"core_hours": 0.0027777, "gpu_hours": 0.0, "data": [1.0, 2.5]}).sign(SECRET)
    raw = msg.to_content()["io.picongpu.rcp"]
    # The wire form carries no float, and the tagged form is what got signed.
    _synapse_canonical_json(raw)
    restored = RcpMessage.from_dict(raw)
    assert restored.verify(SECRET)
    # ``repr`` round-trips a float bit-exactly; assert on it rather than ``==``.
    assert repr(restored.payload["core_hours"]) == repr(0.0027777)
    assert repr(restored.payload["gpu_hours"]) == repr(0.0)
    assert [repr(item) for item in restored.payload["data"]] == [repr(1.0), repr(2.5)]


def test_non_finite_floats_are_made_wire_safe() -> None:
    for value in (float("inf"), float("-inf"), float("nan")):
        msg = make(kind=Kind.EVENT, payload={"usage": value}).sign(SECRET)
        content = msg.to_content()
        _synapse_canonical_json(content)
        restored = RcpMessage.from_dict(content["io.picongpu.rcp"])
        assert restored.verify(SECRET)
        assert str(restored.payload["usage"]) == str(value)


def test_booleans_are_not_treated_as_floats() -> None:
    msg = make(payload={"ok": True, "n": 2, "name": "x"}).sign(SECRET)
    restored = RcpMessage.from_dict(msg.to_content()["io.picongpu.rcp"])
    assert restored.payload == {"ok": True, "n": 2, "name": "x"}
    assert restored.verify(SECRET)


def test_signature_binds_the_float_free_wire_form() -> None:
    """Tampering the float value on the wire must invalidate the signature."""
    msg = make(kind=Kind.EVENT, payload={"core_hours": 1.0}).sign(SECRET)
    raw = msg.to_content()["io.picongpu.rcp"]
    raw["payload"]["core_hours"] = {"$rcp_float": "999.0"}
    assert not RcpMessage.from_dict(raw).verify(SECRET)
