# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Unit tests for the M2 ``submit_simulation`` wire contract (offline)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pic_agentic.protocol.simulation import (
    DEFAULT_SUBMIT_SYSTEM,
    MAX_INLINE_PAYLOAD_BYTES,
    PAYLOAD_KEY,
    PayloadTooLargeError,
    SimulationPayload,
    SimulationState,
    SubmitParams,
    UnsupportedPayloadError,
    build_submit_ack,
    build_submit_command,
    build_submit_event,
    payload_wire_bytes,
    provenance_mismatches,
    simulation_spec_from_runner_dump,
)
from pic_agentic.rcp import Kind, SenderRole, canonical_bytes
from pic_agentic.version import WIRE_FORMAT_VERSION

FIXTURE = Path(__file__).parent / "fixtures" / "pypicongpu_runner.json"
#: Schema hash of the pinned tree (see pyproject [sim] pin); regenerated when
#: the pin moves.  Guards against silent picongpu schema drift in CI.
PINNED_SCHEMA_HASH = "f6471fe1244f6a9d819951e090a281862b58b5b7170b5ae209b8841c3d94e4d9"


def _runner_dump() -> dict:
    return json.loads(FIXTURE.read_text())


def _payload() -> SimulationPayload:
    return SimulationPayload.build(
        picongpu_version="0.9.0-dev",
        picongpu_revision="04855606583209a09659a0c81553bddf2ce7bdac",
        schema_hash=PINNED_SCHEMA_HASH,
        runner_dump=_runner_dump(),
    )


def test_spec_drops_cluster_local_dirs() -> None:
    spec = simulation_spec_from_runner_dump(_runner_dump())
    assert set(spec) == {"sim"}
    assert "setup_dir" not in spec
    assert "run_dir" not in spec
    assert "template_dir" not in spec


def test_spec_requires_sim() -> None:
    with pytest.raises(UnsupportedPayloadError):
        simulation_spec_from_runner_dump({"setup_dir": "/x"})


def test_payload_hash_and_sim_id_are_deterministic() -> None:
    first = _payload()
    second = _payload()
    assert first.payload_hash == second.payload_hash
    assert first.sim_id == first.payload_hash[:8]
    # Stable across a model (de)serialisation round trip.
    reloaded = SimulationPayload.model_validate_json(first.model_dump_json(exclude_computed_fields=True))
    assert reloaded.payload_hash == first.payload_hash


def test_payload_file_excludes_computed_fields_but_revalidates() -> None:
    payload = _payload()
    body = payload.model_dump_json(exclude_computed_fields=True)
    assert "payload_hash" not in json.loads(body)
    restored = SimulationPayload.model_validate_json(body)
    assert restored.simulation == payload.simulation
    assert restored.payload_hash == payload.payload_hash


def test_extra_top_level_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        SimulationPayload.model_validate(
            {
                "picongpu_version": "0.9.0-dev",
                "schema_hash": "x",
                "simulation": {"sim": {}},
                "rc_params": {"tbg_submit": "sbatch"},
            }
        )


def test_allowlist_rejects_extra_simulation_key() -> None:
    payload = SimulationPayload(
        picongpu_version="0.9.0-dev",
        schema_hash="x",
        simulation={"sim": {}, "setup_dir": "/etc"},
    )
    with pytest.raises(UnsupportedPayloadError, match="unsupported field"):
        payload.check_allowlist()


def test_allowlist_reports_missing_sim() -> None:
    payload = SimulationPayload(picongpu_version="0.9.0-dev", schema_hash="x", simulation={})
    with pytest.raises(UnsupportedPayloadError, match="missing field"):
        payload.check_allowlist()


def test_provenance_mismatches() -> None:
    payload = _payload()
    local = {
        "picongpu_version": "0.9.0-dev",
        "picongpu_revision": "04855606583209a09659a0c81553bddf2ce7bdac",
        "schema_hash": PINNED_SCHEMA_HASH,
    }
    assert provenance_mismatches(payload, local) == []

    assert provenance_mismatches(payload, {**local, "schema_hash": "other"})
    assert provenance_mismatches(payload, {**local, "picongpu_version": "1.0"})
    assert provenance_mismatches(payload, {**local, "picongpu_revision": "deadbeef"})


def test_provenance_skips_unknown_local_values() -> None:
    payload = _payload()
    # An editable install without vcs_info and with no schema hash must not
    # spuriously reject the payload.
    assert provenance_mismatches(payload, {"picongpu_version": "", "picongpu_revision": "", "schema_hash": ""}) == []


def test_command_header_carries_provenance_and_hash() -> None:
    payload = _payload()
    command = build_submit_command(sim="s", seq=1, payload=payload)
    header = command.payload["header"]
    assert header["payload_hash"] == hashlib.sha256(canonical_bytes(payload.simulation)).hexdigest()
    assert header["sim_id"] == payload.sim_id
    assert header["wire_format_version"] == WIRE_FORMAT_VERSION
    assert command.kind is Kind.COMMAND
    assert command.sender_role is SenderRole.MCP_SERVER
    # The payload travels inline as a JSON string; no shared-FS path is
    # involved.  A string (not a nested object) is required because Matrix's
    # canonical JSON rejects floats in event content.
    assert "payload_path" not in command.payload
    raw = command.payload[PAYLOAD_KEY]
    assert isinstance(raw, str)
    assert set(json.loads(raw)["simulation"]) == {"sim"}


def test_params_round_trip_and_default_submit_system() -> None:
    params = SubmitParams(build_jobs=4)
    command = build_submit_command(sim="s", seq=1, payload=_payload(), params=params)
    assert command.payload["params"]["submit_system"] == DEFAULT_SUBMIT_SYSTEM
    # picongpu_flags maps the design's build_* names to the aliases the pinned
    # PicBuildFlags/TBGFlags actually accept, and drops unset options (a False
    # build_force is dropped so picongpu keeps its own default).
    assert params.picongpu_flags() == {"jobs": 4, "submit": DEFAULT_SUBMIT_SYSTEM}


def test_params_maps_every_field_to_its_picongpu_alias() -> None:
    params = SubmitParams(
        build_jobs=8,
        build_cmake="-DX=1",
        build_preset=3,
        build_force=True,
        cfg_file="my.cfg",
        overwrite_vars=["a=1"],
    )
    assert params.picongpu_flags() == {
        "jobs": 8,
        "cmake": "-DX=1",
        "preset": 3,
        "force": True,
        "cfg": "my.cfg",
        "submit": "sbatch",
        "o": ["a=1"],
    }


def test_payload_wire_bytes_excludes_computed_fields_and_round_trips() -> None:
    payload = _payload()
    raw = payload_wire_bytes(payload)
    body = json.loads(raw)
    assert "payload_hash" not in body
    assert "sim_id" not in body
    restored = SimulationPayload.model_validate(body)
    assert restored.payload_hash == payload.payload_hash


def test_payload_wire_bytes_rejects_oversized_simulation() -> None:
    payload = SimulationPayload(
        picongpu_version="0.9.0-dev",
        schema_hash="x",
        simulation={"sim": {"blob": "a" * (MAX_INLINE_PAYLOAD_BYTES + 1)}},
    )
    with pytest.raises(PayloadTooLargeError, match="inline limit"):
        payload_wire_bytes(payload)


def test_ack_and_event_shape() -> None:
    ack = build_submit_ack(
        sim="s", seq=2, cmd_id="c", sim_id="abcd1234", state=SimulationState.ACCEPTED, in_reply_to=None
    )
    assert ack.kind is Kind.ACK
    assert ack.payload["state"] == "accepted"
    assert ack.sender_role is SenderRole.SIMCLIENT

    event = build_submit_event(sim="s", seq=3, cmd_id="c", sim_id="abcd1234", state=SimulationState.SUBMITTED, job_id=7)
    assert event.kind is Kind.EVENT
    assert event.payload["job_id"] == 7
