# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Regression tests for the simclient's non-blocking control plane.

A live beta run showed every MCP tool call timing out (``-32001``) while a build
was in flight: ``serve`` awaited each handler, so a multi-minute build stalled
``hello``/``status``/``logs``/``result`` and control acks.  These tests pin the
new contract: inbound messages are dispatched concurrently, a submit acks
``accepted`` before its build, and the workflow stderr capture is per run.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from pic_agentic.protocol.hello import HelloType, build_hello_command
from pic_agentic.protocol.simulation import (
    ResultOp,
    ResultParams,
    SimulationStage,
    SimulationState,
    SimulationType,
    SubmitParams,
    build_result_command,
    build_status_command,
)
from pic_agentic.rcp import new_secret_hex
from pic_agentic.server.simulation import SubmitService
from pic_agentic.simclient import SimClient
from pic_agentic.simclient import client as client_mod
from pic_agentic.simclient import simulation as sim_mod
from pic_agentic.simclient.follow import TrackedSim
from pic_agentic.simclient.simulation import (
    PreparedSubmit,
    SimulationErrorCode,
    SimulationExecutionError,
    SubmitConfig,
    execute_submit,
)
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
    """Stand-in for ``pypicongpu.Runner`` at the module boundary."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.setup_dir = self.run_dir.parent / "input"
        self.generated = False

    def generate(self, **flags: object) -> None:
        self.generated = True
        self.flags = flags

    def run(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "submission_information.txt").write_text(f"Submitted batch job {JOB_ID}\n")


@pytest.fixture
def fake_runner(monkeypatch):
    created: list[_FakeRunner] = []

    def fake_from_payload(payload, config, token):
        runner = _FakeRunner(config.setup_root / payload.sim_id / token / "run")
        created.append(runner)
        return runner

    monkeypatch.setattr(sim_mod, "runner_from_payload", fake_from_payload)
    monkeypatch.setattr(sim_mod, "_detect_submit_system", lambda: "sbatch")
    return created


async def _fake_builder(*, script_path, interpreter="", **_kw: object) -> BuiltSimulation:
    return BuiltSimulation(
        runner=_runner_dump(),
        picongpu_version="0.9.0-dev",
        picongpu_revision="667c537620e685486aceeaa77deb6550ac9972cf",
        schema_hash="f6471fe1244f6a9d819951e090a281862b58b5b7170b5ae209b8841c3d94e4d9",
    )


def _client(tmp_path: Path, sim_transport: MemoryTransport) -> SimClient:
    return SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_transport,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=tmp_path,
        submit_config=SubmitConfig(setup_root=tmp_path / "sims"),
        poll_interval_s=0.01,
    )


async def _build_command(tmp_path: Path, *, cmd_id: str):
    service = SubmitService(sim=SIM, secret=SECRET, runner_dump_builder=_fake_builder, ack_timeout_s=5.0)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    return await service.build_payload(script, cmd_id=cmd_id)


async def _wait_for(predicate, *, timeout_s: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met before timeout")


async def _wait_for_accepted(received: list[object], count: int) -> None:
    await _wait_for(
        lambda: (
            sum(
                1
                for message in received
                if message.type == SimulationType.ACK and message.payload.get("state") == SimulationState.ACCEPTED.value
            )
            >= count
        ),
        timeout_s=3.0,
    )


def _collect(mcp_t: MemoryTransport, sink: list[object]) -> asyncio.Task[None]:
    async def pump() -> None:
        async for message in mcp_t.receive():
            sink.append(message)  # ruff: ignore[manual-list-comprehension] - must append as they arrive

    return asyncio.create_task(pump())


def _seed_tracked(client: SimClient, run_dir: Path, state_dir: Path, sim_id: str) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / f"{JOB_ID}.state").write_text("RUNNING\n", encoding="utf-8")
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "simOutput"
    output.mkdir(parents=True, exist_ok=True)
    (output / "energy.txt").write_text("step\ten\n0\t2.5\n", encoding="utf-8")
    client._tracked[sim_id] = TrackedSim(
        sim_id=sim_id,
        cmd_id="c",
        job_id=JOB_ID,
        run_dir=str(run_dir),
        stdout_path=None,
        submit_system="sbatch",
    )


async def test_control_plane_answers_while_a_build_is_in_flight(tmp_path, monkeypatch, fake_runner) -> None:
    """The beta regression: hello/status/result must not wait for a build."""
    state_dir = tmp_path / "fake-slurm"
    monkeypatch.setenv("FAKE_SLURM_STATE", str(state_dir))
    mcp_t, sim_t = MemoryTransport.create_pair()
    client = _client(tmp_path, sim_t)

    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_submit(*, prepared, emit, job_id_reader) -> dict:
        started.set()
        await release.wait()
        return {
            "sim_id": prepared.payload.sim_id,
            "state": SimulationState.WORKFLOW_FINISHED.value,
            "job_id": None,
            "run_dir": str(tmp_path / "slow" / "run"),
            "stdout_path": None,
        }

    monkeypatch.setattr(client_mod, "execute_submit", slow_submit)
    received: list = []
    pump_task = _collect(mcp_t, received)
    serve_task = asyncio.create_task(client.serve())
    try:
        _cmd_id, _payload, command = await _build_command(tmp_path, cmd_id="a" * 32)
        await mcp_t.send(command)
        await _wait_for(lambda: any(m.type == SimulationType.ACK for m in received))
        sim_id = next(m for m in received if m.type == SimulationType.ACK).payload["sim_id"]
        await asyncio.wait_for(started.wait(), timeout=3)
        assert not release.is_set()

        run_dir = tmp_path / "run"
        _seed_tracked(client, run_dir, state_dir, sim_id)

        hello_path = tmp_path / "hello.txt"
        await mcp_t.send(
            build_hello_command(sim=SIM, seq=1, message_path=str(hello_path), message="hi", cmd_id="h" * 32).sign(
                SECRET
            )
        )
        await mcp_t.send(build_status_command(sim=SIM, seq=2, sim_id=sim_id, cmd_id="s" * 32).sign(SECRET))
        await mcp_t.send(
            build_result_command(
                sim=SIM,
                seq=3,
                params=ResultParams(sim_id=sim_id, op=ResultOp.READ, path="energy.txt"),
                cmd_id="r" * 32,
            ).sign(SECRET)
        )

        # All three acks arrive while the build is still blocked.
        await _wait_for(lambda: any(m.type == HelloType.ACK for m in received), timeout_s=3.0)
        await _wait_for(lambda: any(m.type == SimulationType.STATUS_ACK for m in received), timeout_s=3.0)
        await _wait_for(lambda: any(m.type == SimulationType.RESULT_ACK for m in received), timeout_s=3.0)
        assert not release.is_set()
        assert next(m for m in received if m.type == SimulationType.STATUS_ACK).payload["job_id"] == JOB_ID
        assert next(m for m in received if m.type == SimulationType.RESULT_ACK).payload["data_encoding"] == "text"
    finally:
        release.set()
        serve_task.cancel()
        pump_task.cancel()
        await asyncio.gather(serve_task, pump_task, return_exceptions=True)
        await sim_t.close()
        await mcp_t.close()


async def test_two_submits_are_accepted_promptly(tmp_path, monkeypatch, fake_runner) -> None:
    """Distinct cmd_ids each get accepted without waiting for the other's build."""
    mcp_t, sim_t = MemoryTransport.create_pair()
    client = _client(tmp_path, sim_t)
    entered: list[int] = []
    release = asyncio.Event()

    async def slow_submit(*, prepared, emit, job_id_reader) -> dict:
        entered.append(1)
        await release.wait()
        return {
            "sim_id": prepared.payload.sim_id,
            "state": SimulationState.WORKFLOW_FINISHED.value,
            "job_id": None,
            "run_dir": str(tmp_path / "slow" / "run"),
            "stdout_path": None,
        }

    monkeypatch.setattr(client_mod, "execute_submit", slow_submit)
    received: list = []
    pump_task = _collect(mcp_t, received)
    serve_task = asyncio.create_task(client.serve())
    try:
        commands = [await _build_command(tmp_path, cmd_id=c) for c in ("b" * 32, "c" * 32)]
        for _cmd_id, _payload, command in commands:
            await mcp_t.send(command)

        # Both accepted acks arrive while the (serialised) build is blocked.
        await _wait_for_accepted(received, 2)
        assert not release.is_set()
        assert len(entered) <= 1
    finally:
        release.set()
        serve_task.cancel()
        pump_task.cancel()
        await asyncio.gather(serve_task, pump_task, return_exceptions=True)
        await sim_t.close()
        await mcp_t.close()


class _StderrRunner:
    """Fake runner whose ``run`` writes through the injected stderr capture."""

    def __init__(self, marker: str, barrier: threading.Barrier | None = None) -> None:
        self.marker = marker
        self.barrier = barrier
        self.setup_dir = Path("/nonexistent/input")
        self.run_dir = Path("/nonexistent/run")
        self.proxy = None

    @staticmethod
    def generate(**_flags: object) -> None:
        return

    def run(self) -> None:
        runtime_context = sys.modules["picongpu.pypicongpu.runner"].RuntimeContext
        context = runtime_context(kwargs={})
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        context.default_stderr.write(f"MARKER-{self.marker}: cmake: command not found\n")
        raise SimulationExecutionError(SimulationErrorCode.RUN_FAILED, "boom", SimulationStage.RUN)


@pytest.fixture
def fake_picongpu_runner(monkeypatch):
    """Install a fake pinned-runner module and reset the one-time capture install."""
    module = types.ModuleType("picongpu.pypicongpu.runner")

    class _RuntimeContext:
        def __init__(self, kwargs=None) -> None:
            for key, value in (kwargs or {}).items():
                setattr(self, key, value)

    module.RuntimeContext = _RuntimeContext
    picongpu = types.ModuleType("picongpu")
    pypicongpu = types.ModuleType("picongpu.pypicongpu")
    pypicongpu.runner = module
    picongpu.pypicongpu = pypicongpu
    monkeypatch.setitem(sys.modules, "picongpu", picongpu)
    monkeypatch.setitem(sys.modules, "picongpu.pypicongpu", pypicongpu)
    monkeypatch.setitem(sys.modules, "picongpu.pypicongpu.runner", module)
    monkeypatch.setattr(sim_mod, "_CAPTURE_PROXY", None)
    return module


def test_per_run_stderr_capture_is_isolated(fake_picongpu_runner) -> None:
    """Two concurrent failing builds keep their own stderr detail."""
    barrier = threading.Barrier(2)
    captures: dict[str, list[str]] = {}
    errors: dict[str, Exception] = {}

    def worker(marker: str, capture: list[str]) -> None:
        runner = _StderrRunner(marker, barrier)
        try:
            sim_mod._run_workflow(runner, capture)
        except Exception as exc:
            errors[marker] = exc

    captures = {"a": [], "b": []}
    threads = [threading.Thread(target=worker, args=(marker, captures[marker])) for marker in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert captures["a"]
    assert "MARKER-a" in captures["a"][0]
    assert "MARKER-b" not in captures["a"][0]
    assert captures["b"]
    assert "MARKER-b" in captures["b"][0]
    assert "MARKER-a" not in captures["b"][0]
    # The helper consumes the per-run capture, not a process-global slot.
    assert "MARKER-a" in sim_mod._workflow_failure_detail(Path("/nonexistent/run"), captures["a"][0])
    assert "MARKER-b" in sim_mod._workflow_failure_detail(Path("/nonexistent/run"), captures["b"][0])


async def test_execute_submit_reports_this_runs_stderr(fake_picongpu_runner, tmp_path) -> None:
    """The failed-run detail in the raised error is this run's stderr."""
    runner = _StderrRunner("solo")
    prepared = PreparedSubmit(
        payload=SimpleNamespace(sim_id="solo1234"),
        params=SubmitParams(),
        runner=runner,
        config=SubmitConfig(setup_root=tmp_path),
    )

    async def emit(*_args: object, **_kwargs: object) -> None:
        return

    with pytest.raises(SimulationExecutionError) as excinfo:
        await execute_submit(prepared=prepared, emit=emit, job_id_reader=lambda _run, _payload: None)
    assert "MARKER-solo" in str(excinfo.value)


def test_build_concurrency_resolver(monkeypatch) -> None:
    assert client_mod.resolve_build_concurrency(None) == 1
    monkeypatch.setenv(client_mod.BUILD_CONCURRENCY_ENV, "3")
    assert client_mod.resolve_build_concurrency(None) == 3
    assert client_mod.resolve_build_concurrency("0") == 1
    assert client_mod.resolve_build_concurrency("nonsense") == 1


async def test_build_gate_serialises_by_default(tmp_path, monkeypatch, fake_runner) -> None:
    """Default concurrency 1 admits one build at a time (the explicit gate)."""
    mcp_t, sim_t = MemoryTransport.create_pair()
    client = _client(tmp_path, sim_t)
    assert client.build_concurrency == 1
    active = 0
    peak = 0
    both_started = asyncio.Semaphore(0)
    release = asyncio.Event()

    async def counting_submit(*, prepared, emit, job_id_reader) -> dict:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        both_started.release()
        await release.wait()
        active -= 1
        return {
            "sim_id": prepared.payload.sim_id,
            "state": SimulationState.WORKFLOW_FINISHED.value,
            "job_id": None,
            "run_dir": str(tmp_path / "slow" / "run"),
            "stdout_path": None,
        }

    monkeypatch.setattr(client_mod, "execute_submit", counting_submit)
    received: list = []
    pump_task = _collect(mcp_t, received)
    serve_task = asyncio.create_task(client.serve())
    try:
        commands = [await _build_command(tmp_path, cmd_id=c) for c in ("d" * 32, "e" * 32)]
        for _cmd_id, _payload, command in commands:
            await mcp_t.send(command)
        await _wait_for_accepted(received, 2)
        # Only the first build is admitted; the second waits on the gate.
        await asyncio.wait_for(both_started.acquire(), timeout=3)
        assert peak == 1
    finally:
        release.set()
        serve_task.cancel()
        pump_task.cancel()
        await asyncio.gather(serve_task, pump_task, return_exceptions=True)
        await sim_t.close()
        await mcp_t.close()


async def test_build_concurrency_above_one_admits_both(tmp_path, monkeypatch, fake_runner) -> None:
    """An explicit concurrency of 2 admits two builds at once."""
    mcp_t, sim_t = MemoryTransport.create_pair()
    client = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=tmp_path,
        submit_config=SubmitConfig(setup_root=tmp_path / "sims"),
        poll_interval_s=0.01,
        build_concurrency=2,
    )
    assert client.build_concurrency == 2
    active = 0
    peak = 0
    both_started = asyncio.Event()
    release = asyncio.Event()

    async def counting_submit(*, prepared, emit, job_id_reader) -> dict:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active >= 2:
            both_started.set()
        await release.wait()
        active -= 1
        return {
            "sim_id": prepared.payload.sim_id,
            "state": SimulationState.WORKFLOW_FINISHED.value,
            "job_id": None,
            "run_dir": str(tmp_path / "slow" / "run"),
            "stdout_path": None,
        }

    monkeypatch.setattr(client_mod, "execute_submit", counting_submit)
    received: list = []
    pump_task = _collect(mcp_t, received)
    serve_task = asyncio.create_task(client.serve())
    try:
        commands = [await _build_command(tmp_path, cmd_id=c) for c in ("f" * 32, "0" * 32)]
        for _cmd_id, _payload, command in commands:
            await mcp_t.send(command)
        await asyncio.wait_for(both_started.wait(), timeout=3)
        assert peak == 2
    finally:
        release.set()
        serve_task.cancel()
        pump_task.cancel()
        await asyncio.gather(serve_task, pump_task, return_exceptions=True)
        await sim_t.close()
        await mcp_t.close()
