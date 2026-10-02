# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""End-to-end M3 control test over the in-memory transport and fake SLURM.

A real ``submit_simulation`` drives the follow watcher to ``RUNNING``; the
``CONTROL_COMMAND`` path is then exercised against a test-injected ``control_fn``
that wraps the real :class:`~pic_agentic.slurm.SlurmClient`, i.e. the fake
``scancel`` double (``scontrol`` is only used for ``show job``; real Slurm has no
``scontrol signal``/``cancel``).  State-gate rejections plus a successful
hard cancel are covered too.  No cluster or homeserver is involved.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from pic_agentic.protocol.simulation import (
    CONTROL_SIGNAL,
    ControlParams,
    SimulationOp,
    SimulationState,
    SimulationType,
    build_control_command,
)
from pic_agentic.rcp import RcpMessage, new_secret_hex
from pic_agentic.server.simulation import SubmitService
from pic_agentic.simclient import SimClient
from pic_agentic.simclient import simulation as sim_mod
from pic_agentic.simclient.follow import TrackedSim
from pic_agentic.simclient.simulation import SubmitConfig
from pic_agentic.simulation_build import BuiltSimulation
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
    """Stand-in for ``pypicongpu.Runner`` that mimics the follow-time artifacts."""

    def __init__(self, run_dir: Path, state_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.setup_dir = self.run_dir.parent / "input"
        self.state_dir = Path(state_dir)
        self.generated = False
        self.ran = False

    def generate(self, **_flags: object) -> None:
        self.generated = True

    def run(self) -> None:
        self.ran = True
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "submission_information.txt").write_text(f"Submitted batch job {JOB_ID}\n")
        # A persistent RUNNING state: the control handler and the watcher may
        # query concurrently, so the test must not depend on a pop sequence.
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / f"{JOB_ID}.state").write_text("RUNNING\n", encoding="utf-8")
        cache = self.run_dir / ".cwl_cache" / "steps"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "stdout").write_text("PIConGPU: startup\n", encoding="utf-8")
        output = cache / "simOutput"
        output.mkdir(exist_ok=True)
        (self.run_dir / "link_results.sh").write_text(f'#!/bin/bash\nln -s "{output}" "$1"\n')


class _RecordingControl:
    """A test ``control_fn`` wrapping the real SLURM client and recording ops."""

    def __init__(self, slurm: SlurmClient) -> None:
        self.slurm = slurm
        self.calls: list[tuple[SimulationOp, int | None]] = []

    async def __call__(self, op: SimulationOp, tracked: TrackedSim) -> str:
        self.calls.append((op, tracked.job_id))
        if op is SimulationOp.CANCEL:
            return await self.slurm.cancel_job(tracked.job_id or 0)
        return await self.slurm.signal_job(tracked.job_id or 0, CONTROL_SIGNAL[op] or "")


@pytest.fixture
def shared_dir(tmp_path, monkeypatch):
    state_dir = tmp_path / "fake-slurm"
    monkeypatch.setenv("FAKE_SLURM_STATE", str(state_dir))
    d = tmp_path / "shared"
    d.mkdir()
    return d, state_dir


@pytest.fixture
def fake_runner(monkeypatch):
    created: list[_FakeRunner] = []

    def fake_from_payload(payload, config, token):
        runner = _FakeRunner(
            config.setup_root / payload.sim_id / token / "run",
            state_dir=Path(os.environ["FAKE_SLURM_STATE"]),
        )
        created.append(runner)
        return runner

    monkeypatch.setattr(sim_mod, "runner_from_payload", fake_from_payload)
    monkeypatch.setattr(sim_mod, "_detect_submit_system", lambda: "sbatch")
    return created


async def _fake_builder(*, script_path, interpreter="", **_kw: object) -> BuiltSimulation:
    return BuiltSimulation(
        runner=_runner_dump(),
        picongpu_version="0.9.0-dev",
        picongpu_revision="122160f6125eccd0606bb8ceb11c815c88ffaf2d",
        schema_hash="f6471fe1244f6a9d819951e090a281862b58b5b7170b5ae209b8841c3d94e4d9",
    )


def _make_pair(shared: Path):
    mcp_t, sim_t = MemoryTransport.create_pair()
    service = SubmitService(sim=SIM, secret=SECRET, runner_dump_builder=_fake_builder, ack_timeout_s=5.0)
    control = _RecordingControl(SlurmClient(bin_dir=str(FAKE_BIN)))
    client = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared,
        submit_config=SubmitConfig(setup_root=shared / "sims"),
        poll_interval_s=0.01,
        poll_max_interval_s=0.02,
        control_fn=control,
    )
    return mcp_t, sim_t, service, client, control


async def _wait_for(predicate, *, limit_s: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limit_s
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met before timeout")


def _control(sim_id: str, op: SimulationOp, *, seq: int) -> RcpMessage:
    params = ControlParams(sim_id=sim_id, op=op)
    return build_control_command(sim=SIM, seq=seq, params=params, cmd_id=f"ctl-{op.value}").sign(SECRET)


async def test_submit_checkpoint_stop_e2e(shared_dir, tmp_path, fake_runner) -> None:
    shared, state_dir = shared_dir
    mcp_t, sim_t, service, client, control = _make_pair(shared)
    received: list[RcpMessage] = []

    async def pump() -> None:
        async for msg in mcp_t.receive():
            received.append(msg)
            service.on_message(msg)

    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    pump_task = asyncio.create_task(pump())
    serve_task = asyncio.create_task(client.serve())
    try:
        outcome = await service.submit(mcp_t.send, script)
        assert outcome.acked, outcome.error
        assert outcome.ok, outcome.error
        sim_id = outcome.sim_id
        await _wait_for(lambda: sim_id in client._tracked)

        # checkpoint: ok ack, USR1 recorded by the fake scancel, event emitted.
        cp_ack = await client.handle(_control(sim_id, SimulationOp.CHECKPOINT, seq=200))
        assert cp_ack is not None
        assert cp_ack.type == SimulationType.CONTROL_ACK
        assert cp_ack.payload["ok"] is True
        assert cp_ack.payload["op"] == "checkpoint"
        assert cp_ack.payload["signal"] == "USR1"
        assert cp_ack.payload["state"] == SimulationState.CHECKPOINT.value
        assert cp_ack.payload["job_id"] == JOB_ID
        assert cp_ack.payload["slurm_reason"] == f"scancel: sent USR1 to job {JOB_ID}"
        assert (state_dir / f"{JOB_ID}.signals").read_text(encoding="utf-8").splitlines() == ["USR1"]
        await _wait_for(
            lambda: any(
                message.type == SimulationType.EVENT
                and message.payload.get("state") == SimulationState.CHECKPOINT.value
                for message in received
            ),
        )
        assert control.calls == [(SimulationOp.CHECKPOINT, JOB_ID)]

        # stop: ok ack with TERM; then the job reaches terminal and the watcher
        # reports it (the stop itself does not emit a state event).
        stop_ack = await client.handle(_control(sim_id, SimulationOp.STOP, seq=201))
        assert stop_ack is not None
        assert stop_ack.payload["ok"] is True
        assert stop_ack.payload["signal"] == "TERM"
        assert control.calls[-1] == (SimulationOp.STOP, JOB_ID)
        (state_dir / f"{JOB_ID}.state").write_text("COMPLETED 0\n", encoding="utf-8")
        await _wait_for(
            lambda: any(
                message.type == SimulationType.EVENT
                and message.payload.get("state") == SimulationState.JOB_FINISHED.value
                for message in received
            ),
        )
    finally:
        serve_task.cancel()
        pump_task.cancel()
        await asyncio.gather(serve_task, pump_task, return_exceptions=True)
        await sim_t.close()
        await mcp_t.close()


async def test_cancel_e2e_flips_state(shared_dir, tmp_path) -> None:
    shared, state_dir = shared_dir
    mcp_t, sim_t, _service, client, control = _make_pair(shared)
    state_dir.mkdir(parents=True, exist_ok=True)
    job_id = 555001
    (state_dir / f"{job_id}.state").write_text("RUNNING\n", encoding="utf-8")
    client._serving = True
    received: list[RcpMessage] = []

    async def pump() -> None:
        async for msg in mcp_t.receive():
            received.extend((msg,))

    pump_task = asyncio.create_task(pump())
    try:
        await client._start_follower(
            cmd_id="cmd-cancel",
            sim_id="canc1234",
            job_id=job_id,
            run_dir=str(tmp_path),
            stdout_path=None,
            submit_system="sbatch",
        )
        cancel_ack = await client.handle(
            build_control_command(
                sim=SIM,
                seq=300,
                params=ControlParams(sim_id="canc1234", op=SimulationOp.CANCEL),
                cmd_id="ctl-cancel",
            ).sign(SECRET),
        )
        assert cancel_ack is not None
        assert cancel_ack.payload["ok"] is True
        assert cancel_ack.payload["op"] == "cancel"
        assert "signal" not in cancel_ack.payload
        assert cancel_ack.payload["slurm_reason"] == f"scancel: cancelled job {job_id}"
        assert (state_dir / f"{job_id}.state").read_text(encoding="utf-8").splitlines()[0] == "CANCELLED"
        assert control.calls == [(SimulationOp.CANCEL, job_id)]
        # The watcher observes CANCELLED and reports its own terminal state.
        await _wait_for(
            lambda: any(
                message.type == SimulationType.EVENT and message.payload.get("state") == SimulationState.CANCELLED.value
                for message in received
            ),
        )
    finally:
        pump_task.cancel()
        await asyncio.gather(pump_task, return_exceptions=True)
        client._serving = False
        await client._cancel_followers()
        await sim_t.close()
        await mcp_t.close()


async def test_control_state_gates(shared_dir, tmp_path) -> None:
    shared, _state_dir = shared_dir
    _mcp_t, sim_t, _service, client, control = _make_pair(shared)
    state_dir = Path(os.environ["FAKE_SLURM_STATE"])
    state_dir.mkdir(parents=True, exist_ok=True)
    run = tmp_path / "run"
    run.mkdir()

    def track(sim_id: str, job_id: int, state: str) -> None:
        (state_dir / f"{job_id}.state").write_text(f"{state}\n", encoding="utf-8")
        client._tracked[sim_id] = TrackedSim(
            sim_id=sim_id,
            cmd_id="c",
            job_id=job_id,
            run_dir=str(run),
            stdout_path=None,
            submit_system="sbatch",
        )

    track("pend1234", 600001, "PENDING")
    track("done1234", 600002, "COMPLETED 0")

    try:
        # unknown sim -> no_results / unknown_sim, never a hello ack.
        unknown = await client.handle(_control("deadbeef", SimulationOp.CHECKPOINT, seq=400))
        assert unknown is not None
        assert unknown.type == SimulationType.CONTROL_ACK
        assert unknown.payload["ok"] is False
        assert unknown.payload["error"] == "unknown_sim"
        assert unknown.payload["error_code"] == "no_results"

        # signal op on a non-RUNNING job -> not_signalable.
        not_signalable = await client.handle(_control("pend1234", SimulationOp.CHECKPOINT, seq=401))
        assert not_signalable is not None
        assert not_signalable.payload["ok"] is False
        assert not_signalable.payload["error_code"] == "not_signalable"
        assert not_signalable.payload["state"] == "PENDING"

        # cancel of a terminal job -> not_terminal.
        not_terminal = await client.handle(_control("done1234", SimulationOp.CANCEL, seq=402))
        assert not_terminal is not None
        assert not_terminal.payload["ok"] is False
        assert not_terminal.payload["error_code"] == "not_terminal"

        # No control_fn call reached SLURM for any rejected request.
        assert control.calls == []
    finally:
        await sim_t.close()


async def test_control_rejected_when_submit_disabled(shared_dir) -> None:
    shared, _state_dir = shared_dir
    _mcp_t, sim_t = MemoryTransport.create_pair()
    disabled = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared,
    )
    ack = await disabled.handle(_control("deadbeef", SimulationOp.CHECKPOINT, seq=1))
    assert ack is not None
    assert ack.type == SimulationType.CONTROL_ACK
    assert ack.payload["ok"] is False
    assert ack.payload["error"] == "rejected_by_policy"
    assert ack.payload["error_code"] == "rejected_by_policy"


async def test_cancel_without_job_id_never_reaches_slurm(shared_dir, tmp_path) -> None:
    """A tracked sim with no job id must not yield ``scancel 0``."""
    shared, _state_dir = shared_dir
    _mcp_t, sim_t, _service, client, control = _make_pair(shared)
    run = tmp_path / "run"
    run.mkdir()
    client._tracked["nojob123"] = TrackedSim(
        sim_id="nojob123",
        cmd_id="c",
        job_id=None,
        run_dir=str(run),
        stdout_path=None,
        submit_system="bash",
    )
    try:
        ack = await client.handle(_control("nojob123", SimulationOp.CANCEL, seq=900))
        assert ack is not None
        assert ack.payload["ok"] is False
        assert ack.payload["error"] == "no_job_id"
        assert ack.payload["error_code"] == "not_signalable"
        assert control.calls == []
    finally:
        await sim_t.close()


async def test_control_redelivery_is_idempotent(shared_dir, tmp_path) -> None:
    """A redelivered control command re-acks instead of signalling twice."""
    shared, state_dir = shared_dir
    _mcp_t, sim_t, _service, client, control = _make_pair(shared)
    state_dir.mkdir(parents=True, exist_ok=True)
    job_id = 910001
    (state_dir / f"{job_id}.state").write_text("RUNNING\n", encoding="utf-8")
    client._tracked["redel123"] = TrackedSim(
        sim_id="redel123",
        cmd_id="c",
        job_id=job_id,
        run_dir=str(tmp_path),
        stdout_path=None,
        submit_system="sbatch",
    )
    try:
        first = await client.handle(_control("redel123", SimulationOp.CHECKPOINT, seq=901))
        # A redelivery arrives with a fresh envelope (new seq) but the same
        # cmd_id; it must re-ack, not signal again.
        second = await client.handle(_control("redel123", SimulationOp.CHECKPOINT, seq=902))
        assert first is not None
        assert second is not None
        assert first.payload["slurm_reason"] == second.payload["slurm_reason"]
        assert len(control.calls) == 1
        assert (state_dir / f"{job_id}.signals").read_text(encoding="utf-8").splitlines() == ["USR1"]
    finally:
        await sim_t.close()
