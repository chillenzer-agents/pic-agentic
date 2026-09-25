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
        picongpu_revision="91c3ee5fb4c9425b00d4673d9608f4370593cacf",
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
                "wire_format_version": WIRE_FORMAT_VERSION,
                "picongpu_version": "0.9.0-dev",
                "schema_hash": "x",
                "simulation": {"sim": {}},
                "rc_params": {"tbg_submit": "sbatch"},
            }
        )


def test_omitted_wire_format_version_is_rejected() -> None:
    """An omitted version must not silently masquerade as the current one."""
    with pytest.raises(ValueError, match="wire_format_version"):
        SimulationPayload.model_validate(
            {
                "picongpu_version": "0.9.0-dev",
                "schema_hash": "x",
                "simulation": {"sim": {}},
            }
        )


def test_allowlist_rejects_extra_simulation_key() -> None:
    payload = SimulationPayload(
        wire_format_version=WIRE_FORMAT_VERSION,
        picongpu_version="0.9.0-dev",
        schema_hash="x",
        simulation={"sim": {}, "setup_dir": "/etc"},
    )
    with pytest.raises(UnsupportedPayloadError, match="unsupported field"):
        payload.check_allowlist()


def test_allowlist_reports_missing_sim() -> None:
    payload = SimulationPayload(
        wire_format_version=WIRE_FORMAT_VERSION,
        picongpu_version="0.9.0-dev",
        schema_hash="x",
        simulation={},
    )
    with pytest.raises(UnsupportedPayloadError, match="missing field"):
        payload.check_allowlist()


def test_provenance_mismatches() -> None:
    payload = _payload()
    local = {
        "picongpu_version": "0.9.0-dev",
        "picongpu_revision": "91c3ee5fb4c9425b00d4673d9608f4370593cacf",
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


def test_overwrite_vars_flags_are_joined_for_the_workflow(tmp_path: Path) -> None:
    """input.yaml gets the string CWL's ``run_overwrite_vars`` requires.

    The pinned workflow.cwl declares ``run_overwrite_vars`` as ``type: string?``
    while ``Runner.generate`` writes the list through; the simclient joins it
    before CWL validation (see ``_normalise_workflow_vars``).
    """
    from pic_agentic.simclient.simulation import _normalise_workflow_vars

    workflow_dir = tmp_path / "workflow"
    workflow_dir.mkdir()
    (workflow_dir / "input.yaml").write_text(
        json.dumps({"run_overwrite_vars": ["A=1", "B=2"], "run_cfg_file": "etc/N.cfg"}), encoding="utf-8"
    )
    _normalise_workflow_vars(tmp_path)
    patched = json.loads((workflow_dir / "input.yaml").read_text(encoding="utf-8"))
    assert patched["run_overwrite_vars"] == "A=1 B=2"
    # Absent/None and string values are left untouched.
    (workflow_dir / "input.yaml").write_text(json.dumps({"run_overwrite_vars": None}), encoding="utf-8")
    _normalise_workflow_vars(tmp_path)
    assert json.loads((workflow_dir / "input.yaml").read_text())["run_overwrite_vars"] is None


def test_cfg_file_rejects_unsafe_values() -> None:
    for bad in ("/etc/passwd", "../../x.cfg", "a/../../x.cfg", "x;rm -rf.cfg", "x$(id).cfg", "x.cfg\n"):
        with pytest.raises(ValueError, match="cfg_file"):
            SubmitParams(cfg_file=bad)
    assert SubmitParams(cfg_file="etc/picongpu/N.cfg").cfg_file == "etc/picongpu/N.cfg"
    assert SubmitParams(cfg_file="my.cfg").cfg_file == "my.cfg"


def test_overwrite_vars_rejects_shell_metacharacters() -> None:
    for bad in ("PARAM=$(touch /tmp/pwned)", "A=1 B=2", "A=1;id", "A=`id`", "1BAD=1", "A=a|b", ""):
        with pytest.raises(ValueError, match="overwrite_vars"):
            SubmitParams(overwrite_vars=[bad])
    assert SubmitParams(overwrite_vars=["A=1", "PATH_X=/a/b-c.d:e"]).overwrite_vars == ["A=1", "PATH_X=/a/b-c.d:e"]


def test_payload_wire_bytes_excludes_computed_fields_and_round_trips() -> None:
    payload = _payload()
    raw = payload_wire_bytes(payload)
    body = json.loads(raw)
    assert "payload_hash" not in body
    assert "sim_id" not in body
    restored = SimulationPayload.model_validate(body)
    assert restored.payload_hash == payload.payload_hash


def test_payload_wire_bytes_rejects_oversized_wire_encoding() -> None:
    # A quote-dense blob doubles in size when embedded as a JSON string; the cap
    # must measure the *escaped* payload, not the inner simulation object.
    payload = SimulationPayload(
        wire_format_version=WIRE_FORMAT_VERSION,
        picongpu_version="0.9.0-dev",
        schema_hash="x",
        simulation={"sim": {"blob": '"' * (MAX_INLINE_PAYLOAD_BYTES // 2)}},
    )
    with pytest.raises(PayloadTooLargeError, match="inline limit"):
        payload_wire_bytes(payload)


def test_emitted_event_stays_under_synapse_limit() -> None:
    """An accepted payload must not exceed Synapse's 64 KiB event content."""
    from pic_agentic.rcp.crypto import new_secret_hex

    # A payload comfortably under the cap, encoded: the whole event must fit
    # 64 KiB.
    payload = SimulationPayload(
        wire_format_version=WIRE_FORMAT_VERSION,
        picongpu_version="0.9.0-dev",
        schema_hash="x",
        simulation={"sim": {"blob": "a" * (MAX_INLINE_PAYLOAD_BYTES - 12 * 1024)}},
    )
    command = build_submit_command(sim="s", seq=1, payload=payload)
    content = command.to_content(new_secret_hex())
    encoded = json.dumps(content, separators=(",", ":")).encode("utf-8")
    assert len(encoded) < 64 * 1024, f"event content is {len(encoded)} bytes"


def test_child_env_drops_secrets_and_points_home_at_scratch() -> None:
    """The PICMI child must not inherit secrets or the real 0600 config HOME."""
    import os

    from pic_agentic.simulation_build import _safe_child_env

    sentinel = "syt_access_secret"
    monkey_env = {
        "PIC_AGENTIC_RCP_SECRET": "deadbeefcafesecret",
        "PIC_AGENTIC_ACCESS_TOKEN": sentinel,
        "PIC_AGENTIC_REFRESH_TOKEN": "refresh_secret",
        "HOME": "/home/real",
        "PATH": "/usr/bin",
        "PYTHONPATH": "/workspace/src",
        "VIRTUAL_ENV": "/venv",
        "PYTHONHOME": "/py",
    }
    old = {key: os.environ.get(key) for key in monkey_env}
    os.environ.update(monkey_env)
    try:
        env = _safe_child_env("/scratch/home")
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    assert env["HOME"] == "/scratch/home"
    assert env["PATH"] == "/usr/bin"
    assert env["PYTHONUNBUFFERED"] == "1"
    for key in ("PIC_AGENTIC_RCP_SECRET", "PIC_AGENTIC_ACCESS_TOKEN", "PIC_AGENTIC_REFRESH_TOKEN"):
        assert key not in env
    for key in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME"):
        assert key not in env


def test_build_error_stderr_is_labelled_and_bounded() -> None:
    from pic_agentic.simulation_build import _bounded_stderr

    detail = _bounded_stderr(b"x" * 10_000, b"")
    assert detail.startswith("<untrusted child stderr, tail>")
    assert len(detail) < 10_000
    assert not _bounded_stderr(b"", b"")


def test_extract_payload_ignores_unmarked_and_forged_lines() -> None:
    from pic_agentic.simulation_build import _OUTPUT_MARKER, _extract_payload

    marker = _OUTPUT_MARKER + "nonce123"
    forged = b'{"runner": {"forged": true}, "provenance": {}}'
    good = (marker + '{"runner": {"ok": 1}, "provenance": {}}').encode()
    stdout = forged + b"\n" + good + b"\n" + forged + b"\n"
    assert _extract_payload(stdout, marker) == {"runner": {"ok": 1}, "provenance": {}}
    assert _extract_payload(forged, marker) is None


def test_submit_tool_errors_cover_protocol_and_build_failures() -> None:
    """The tool's soft-error tuple must include the payload protocol errors.

    Regression: ``PayloadTooLargeError`` escaped ``submit_simulation`` as an
    unhandled exception because ``app.py`` only caught ack/build/path/OSError.
    """
    from pic_agentic.server.app import _SUBMIT_TOOL_ERRORS

    assert issubclass(PayloadTooLargeError, _SUBMIT_TOOL_ERRORS)
    assert issubclass(UnsupportedPayloadError, _SUBMIT_TOOL_ERRORS)
    # ValueError also covers pydantic ValidationError (bad injection params).
    assert issubclass(ValueError, _SUBMIT_TOOL_ERRORS)


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
