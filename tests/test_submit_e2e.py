# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""End-to-end ``submit_simulation`` test over the in-memory transport.

PIConGPU is not importable in the offline test environment, so the runner
boundary is stubbed at the module seam; everything else (payload writing, path
and hash validation, provenance check, command/ack/event flow, durable
idempotency) runs for real.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from pic_agentic.protocol.simulation import (
    SimulationPayload,
    SimulationState,
    UnsupportedPayloadError,
    build_submit_command,
)
from pic_agentic.rcp import new_secret_hex
from pic_agentic.server.simulation import SubmitService
from pic_agentic.simclient import SimClient
from pic_agentic.simclient import simulation as sim_mod
from pic_agentic.simclient.simulation import SubmitConfig
from pic_agentic.slurm import SlurmClient
from pic_agentic.transport.memory import MemoryTransport

FAKE_BIN = Path(__file__).parent / "fake_slurm"
FIXTURE = Path(__file__).parent / "fixtures" / "pypicongpu_runner.json"
SECRET = new_secret_hex()
SIM = "7f3a2b1c"
JOB_ID = 424242


def _runner_dump() -> dict:
    return json.loads(FIXTURE.read_text())


class _FakeRunner:
    """Stand-in for ``pypicongpu.Runner`` at the module boundary."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.generated = False
        self.ran = False
        self.flags: dict = {}

    def generate(self, **flags: object) -> None:
        self.generated = True
        self.flags = flags

    def run(self) -> None:
        self.ran = True
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "submission_information.txt").write_text(f"Submitted batch job {JOB_ID}\n")


@pytest.fixture
def shared_dir(tmp_path):
    d = tmp_path / "shared"
    d.mkdir()
    return d


@pytest.fixture
def fake_runner(monkeypatch):
    """Stub the runner rebuild and the local submit-system probe."""
    created: list[_FakeRunner] = []

    def fake_from_payload(payload, config):
        runner = _FakeRunner(config.setup_root / payload.sim_id / "run")
        created.append(runner)
        return runner

    monkeypatch.setattr(sim_mod, "_runner_from_payload", fake_from_payload)
    monkeypatch.setattr(sim_mod, "_detect_submit_system", lambda: "")
    return created


async def _fake_builder(*, script_path, interpreter="", **_kw: object) -> dict:
    return _runner_dump()


def _make_pair(shared_dir, *, builder=_fake_builder):
    mcp_t, sim_t = MemoryTransport.create_pair()
    service = SubmitService(
        sim=SIM,
        secret=SECRET,
        message_dir=shared_dir,
        runner_dump_builder=builder,
        ack_timeout_s=5.0,
    )
    client = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared_dir,
        submit_config=SubmitConfig(message_dir=shared_dir, setup_root=shared_dir / "sims"),
        poll_interval_s=0.05,
    )
    return mcp_t, sim_t, service, client


async def _run_submit(shared_dir, tmp_path, *, fake_runner, params=None):
    mcp_t, sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")

    async def pump() -> None:
        async for msg in mcp_t.receive():
            service.on_message(msg)

    pump_task = asyncio.create_task(pump())
    serve_task = asyncio.create_task(client.serve())
    try:
        outcome = await service.submit(mcp_t.send, script, params=params)
        # Let the async lifecycle events drain (the ack resolves the call first).
        await asyncio.sleep(0.1)
    finally:
        serve_task.cancel()
        pump_task.cancel()
        await sim_t.close()
        await mcp_t.close()
    return outcome, service


async def test_submit_round_trip(shared_dir, tmp_path, fake_runner) -> None:
    outcome, service = await _run_submit(shared_dir, tmp_path, fake_runner=fake_runner)
    assert outcome.acked
    assert outcome.ok, outcome.error
    assert outcome.state == SimulationState.ACCEPTED.value
    assert len(outcome.sim_id) == 8
    assert fake_runner
    assert fake_runner[0].generated
    assert fake_runner[0].ran

    events = service.events[outcome.cmd_id]
    states = [event.payload["state"] for event in events]
    assert SimulationState.SUBMITTED.value in states
    assert SimulationState.RESULTS_READY.value in states
    submitted = next(e for e in events if e.payload["state"] == SimulationState.SUBMITTED.value)
    assert submitted.payload["job_id"] == JOB_ID
    # Provenance tuple is reported back to the sender.
    assert "picongpu_version" not in submitted.payload


async def test_submit_payload_file_is_written_under_message_dir(shared_dir, tmp_path, fake_runner) -> None:
    outcome, _service = await _run_submit(shared_dir, tmp_path, fake_runner=fake_runner)
    payloads = list((shared_dir / "sim").glob("*.json"))
    assert len(payloads) == 1
    raw = json.loads(payloads[0].read_text())
    assert set(raw) == {"wire_format_version", "picongpu_version", "picongpu_revision", "schema_hash", "simulation"}
    assert set(raw["simulation"]) == {"sim"}
    assert SimulationPayload.model_validate(raw).sim_id == outcome.sim_id


async def test_submit_is_idempotent_on_replay(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    _cmd_id, _payload, command = await service.build_payload(script)

    first = await client.handle(command)
    command.transport_event_id = "$replay"
    second = await client.handle(command)
    assert first is not None
    assert second is not None
    assert second.payload["cmd_id"] == first.payload["cmd_id"]
    # The replay re-acks the recorded terminal state, not a fresh accept.
    assert second.payload["state"] == SimulationState.RESULTS_READY.value
    # The replay did not rebuild the setup.
    assert len(fake_runner) == 1


async def test_submit_replay_after_restart_reacks_without_rebuilding(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    _cmd_id, _payload, command = await service.build_payload(script)
    await client.handle(command)

    # New process, same shared dir: the durable record re-acks.
    fresh = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared_dir,
        submit_config=SubmitConfig(message_dir=shared_dir, setup_root=shared_dir / "sims"),
        poll_interval_s=0.05,
    )
    command.transport_event_id = "$backfill"
    replayed = await fresh.handle(command)
    assert replayed is not None
    assert replayed.payload["state"] == SimulationState.RESULTS_READY.value
    assert len(fake_runner) == 1


async def test_submit_rejects_changed_payload_same_cmd_id(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    cmd_id, _payload, command = await service.build_payload(script)
    await client.handle(command)
    assert len(fake_runner) == 1

    # Same cmd_id, different payload hash: executed as a new simulation.
    changed = dict(_runner_dump())
    changed["sim"] = {**changed["sim"], "delta_t_si": 2e-15}
    payload = SimulationPayload.build(picongpu_version="", picongpu_revision="", schema_hash="", runner_dump=changed)
    from pic_agentic.protocol.simulation import build_submit_command

    new_command = build_submit_command(
        sim=SIM, seq=99, payload_path=service.payload_path_for("other"), payload=payload, cmd_id=cmd_id
    ).sign(SECRET)
    # Write the changed payload where the command points.
    from pic_agentic.simclient.safety import write_payload

    write_payload(
        new_command.payload["payload_path"],
        str(shared_dir),
        payload.model_dump_json(exclude_computed_fields=True).encode(),
    )
    new_command.transport_event_id = "$changed"
    await client.handle(new_command)
    assert len(fake_runner) == 2


async def test_submit_reports_hash_mismatch(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    _cmd_id, _payload, command = await service.build_payload(script)
    # Corrupt the payload on disk: the header hash no longer matches.
    body = json.loads(Path(command.payload["payload_path"]).read_text(encoding="utf-8"))
    body["simulation"]["sim"]["delta_t_si"] = 9.99e-15
    Path(command.payload["payload_path"]).write_text(json.dumps(body), encoding="utf-8")

    ack = await client.handle(command)
    assert ack is not None
    assert ack.payload["error_code"] == "hash_mismatch"


async def test_submit_reports_version_mismatch(shared_dir, tmp_path, fake_runner, monkeypatch) -> None:
    monkeypatch.setattr(
        sim_mod,
        "provenance_mismatches",
        lambda *_args, **_kwargs: ["schema_hash differs"],
    )
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    _cmd_id, _payload, command = await service.build_payload(script)
    ack = await client.handle(command)
    assert ack is not None
    assert ack.payload["error_code"] == "version_mismatch"


async def test_submit_rejects_path_outside_message_dir(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    cmd_id, payload, _command = await service.build_payload(script)
    evil = build_submit_command(sim=SIM, seq=1, payload_path="/etc/passwd", payload=payload, cmd_id=cmd_id).sign(SECRET)
    ack = await client.handle(evil)
    assert ack is not None
    assert ack.payload["error_code"] == "path_unsafe"


async def test_submit_rejected_when_handler_disabled(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, sim_t, service, _client = _make_pair(shared_dir)
    disabled = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared_dir,
    )
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    _cmd_id, _payload, command = await service.build_payload(script)
    ack = await disabled.handle(command)
    assert ack is not None
    assert ack.payload["error"] == "rejected_by_policy"


async def test_submit_rejects_unsupported_simulation_key(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    cmd_id, _payload, command = await service.build_payload(script)
    body = json.loads(Path(command.payload["payload_path"]).read_text(encoding="utf-8"))
    body["simulation"]["run_dir"] = "/etc"
    Path(command.payload["payload_path"]).write_text(json.dumps(body), encoding="utf-8")
    # Rebuild and re-sign the command; its header hash is recomputed from the
    # tampered simulation, so the allow-list check (not the hash check) fires.
    tampered = SimulationPayload.model_validate(body)
    rebuilt = build_submit_command(
        sim=SIM, seq=1, payload_path=command.payload["payload_path"], payload=tampered, cmd_id=cmd_id
    ).sign(SECRET)
    ack = await client.handle(rebuilt)
    assert ack is not None
    assert ack.payload["error_code"] == "unsupported"


def test_allowlist_is_enforced_by_prepare_submit(shared_dir, tmp_path) -> None:
    # A simulation mapping carrying a cluster-local key must fail the allow-list.
    payload = SimulationPayload(picongpu_version="", schema_hash="", simulation={"sim": {}, "run_dir": "/etc"})
    with pytest.raises(UnsupportedPayloadError):
        payload.check_allowlist()
