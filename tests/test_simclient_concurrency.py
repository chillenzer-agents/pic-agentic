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
import importlib.util
import io
import json
import logging
import os
import sys
import threading
import time
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
    monkeypatch.setattr(sim_mod, "_CAPTURE_LOGGER_HANDLER", None)
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


class _RealStub:
    """A minimal real-stream stub for the proxy tee test."""

    def __init__(self) -> None:
        self.data = io.StringIO()

    def write(self, text: str) -> int:
        return self.data.write(text)

    def flush(self) -> None:
        self.data.flush()

    def isatty(self) -> bool:
        return self.data.isatty()


def test_capturing_stderr_tees_to_real_stderr() -> None:
    """M1: a proxy write reaches both the run buffer and the real stderr."""
    real = _RealStub()
    buffer = io.StringIO()
    proxy = sim_mod._CapturingStderr(real)
    old_buffer = getattr(sim_mod._CAPTURE_STATE, "buffer", None)
    sim_mod._CAPTURE_STATE.buffer = buffer
    sim_mod._CAPTURE_STATE.lock = threading.Lock()
    try:
        proxy.write("HELLO-STEP-ERROR\n")
    finally:
        sim_mod._CAPTURE_STATE.buffer = old_buffer
    assert buffer.getvalue() == "HELLO-STEP-ERROR\n"
    assert real.data.getvalue() == "HELLO-STEP-ERROR\n", "the operator's stderr must still receive the text"


def test_capturing_stderr_without_a_run_uses_real_stderr() -> None:
    """Outside a run the proxy is a transparent pass-through."""
    real = _RealStub()
    proxy = sim_mod._CapturingStderr(real)
    sim_mod._CAPTURE_STATE.buffer = None
    proxy.write("plain\n")
    assert real.data.getvalue() == "plain\n"


class _LoggerOnlyRunner:
    """Fake runner that emits a cwltool log record and raises permanentFail."""

    setup_dir = Path("/nonexistent/input")
    run_dir = Path("/nonexistent/run")

    @staticmethod
    def generate(**_flags: object) -> None:
        return

    @staticmethod
    def run() -> None:
        # cwltool's own diagnostics go through its logger, not the child fd.
        import cwltool.loghandler

        cwltool.loghandler._logger.error("'definitely-not-a-real-command-xyz' not found")
        raise SimulationExecutionError(SimulationErrorCode.RUN_FAILED, "Completed permanentFail", SimulationStage.RUN)


@pytest.mark.skipif(importlib.util.find_spec("cwltool") is None, reason="cwltool not installed")
def test_capture_includes_cwltool_logger_errors(tmp_path) -> None:
    """M2: a permanentFail carries the cause cwltool logged, not just its raise."""
    capture: list[str] = []
    with pytest.raises(SimulationExecutionError):
        sim_mod._run_workflow(_LoggerOnlyRunner(), capture)
    detail = sim_mod._workflow_failure_detail(Path("/nonexistent/run"), capture[0] if capture else "")
    assert "definitely-not-a-real-command-xyz" in detail


class _FdWriterRunner:
    """Fake runner that writes to the inherited stderr fd (like a child process)."""

    setup_dir = Path("/nonexistent/input")
    run_dir = Path("/nonexistent/run")

    @staticmethod
    def generate(**_flags: object) -> None:
        return

    @staticmethod
    def run() -> None:
        runtime_context = sys.modules["picongpu.pypicongpu.runner"].RuntimeContext
        context = runtime_context(kwargs={})
        os.write(context.default_stderr.fileno(), b"child: command not found\n")
        raise SimulationExecutionError(SimulationErrorCode.RUN_FAILED, "boom", SimulationStage.RUN)


def test_child_stderr_fd_is_teed_and_captured(fake_picongpu_runner, tmp_path) -> None:
    """M1: child-fd stderr reaches both the run buffer and the real stderr."""
    sim_mod._install_capture_context()
    real_path = tmp_path / "real-stderr.log"
    saved_target = sim_mod._CAPTURE_PROXY._real
    with real_path.open("w", encoding="utf-8") as real_fh:
        sim_mod._CAPTURE_PROXY._real = real_fh
        capture: list[str] = []
        try:
            with pytest.raises(SimulationExecutionError):
                sim_mod._run_workflow(_FdWriterRunner(), capture)
        finally:
            sim_mod._CAPTURE_PROXY._real = saved_target
    assert "child: command not found" in (capture[0] if capture else ""), "child fd output must be captured per run"
    assert "child: command not found" in real_path.read_text(encoding="utf-8"), "child output must stay visible"


def test_thread_buffer_handler_routes_to_emitting_thread() -> None:
    """M2: the shared logger handler captures only the emitting run's buffer."""
    handler = sim_mod._ThreadBufferHandler()
    record = logging.LogRecord("cwltool", logging.ERROR, __file__, 1, "boom %s", ("cause",), None)
    run_buffer = io.StringIO()
    old_buffer = getattr(sim_mod._CAPTURE_STATE, "buffer", None)
    old_lock = getattr(sim_mod._CAPTURE_STATE, "lock", None)
    sim_mod._CAPTURE_STATE.lock = threading.Lock()
    try:
        # A thread with no active run drops the record (cwltool's own stream
        # handler still prints it).
        sim_mod._CAPTURE_STATE.buffer = None
        handler.emit(record)
        sim_mod._CAPTURE_STATE.buffer = run_buffer
        handler.emit(record)
    finally:
        sim_mod._CAPTURE_STATE.buffer = old_buffer
        sim_mod._CAPTURE_STATE.lock = old_lock
        handler.close()
    assert "boom cause" in run_buffer.getvalue()


def test_build_concurrency_resolver(monkeypatch) -> None:
    assert client_mod.resolve_build_concurrency(None) == 1
    monkeypatch.setenv(client_mod.BUILD_CONCURRENCY_ENV, "3")
    assert client_mod.resolve_build_concurrency(None) == 3
    assert client_mod.resolve_build_concurrency("0") == 1
    assert client_mod.resolve_build_concurrency("nonsense") == 1


def test_shutdown_grace_resolver(monkeypatch) -> None:
    assert client_mod.resolve_shutdown_grace(None) == pytest.approx(client_mod.DEFAULT_SHUTDOWN_GRACE_S)
    monkeypatch.setenv(client_mod.SHUTDOWN_GRACE_ENV, "7.5")
    assert client_mod.resolve_shutdown_grace(None) == pytest.approx(7.5)
    assert client_mod.resolve_shutdown_grace("0") == pytest.approx(0.0)
    assert client_mod.resolve_shutdown_grace("-3") == pytest.approx(0.0)
    assert client_mod.resolve_shutdown_grace("nonsense") == pytest.approx(client_mod.DEFAULT_SHUTDOWN_GRACE_S)


async def test_shutdown_drains_an_in_flight_build(tmp_path, monkeypatch) -> None:
    """B2: serve() waits for a build (to_thread) to finish before returning."""
    mcp_t, sim_t = MemoryTransport.create_pair()
    client = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=tmp_path,
        submit_config=SubmitConfig(setup_root=tmp_path / "sims"),
        poll_interval_s=0.01,
        shutdown_grace_s=5.0,
    )
    started = asyncio.Event()
    finished = False

    async def slow_submit(*, prepared, emit, job_id_reader) -> dict:
        nonlocal finished
        started.set()
        # The real build runs uncancellably in a worker thread; model that so
        # cancelling the coroutine cannot stop it.
        await asyncio.to_thread(time.sleep, 0.5)
        finished = True
        return {
            "sim_id": prepared.payload.sim_id,
            "state": SimulationState.WORKFLOW_FINISHED.value,
            "job_id": None,
            "run_dir": "/tmp/x",
            "stdout_path": None,
        }

    monkeypatch.setattr(client_mod, "execute_submit", slow_submit)
    prepared = PreparedSubmit(
        payload=SimpleNamespace(sim_id=SIM),
        params=SubmitParams(),
        runner=None,
        config=SubmitConfig(setup_root=tmp_path / "sims"),
    )
    serve_task = asyncio.create_task(client.serve())
    try:
        await asyncio.sleep(0.02)
        client._serving = True
        client._start_submit_task(cmd_id="a" * 32, prepared=prepared, payload_hash="h", declared_sim_id=SIM)
        await asyncio.wait_for(started.wait(), timeout=3)
        serve_task.cancel()
        await asyncio.gather(serve_task, return_exceptions=True)
    finally:
        await sim_t.close()
        await mcp_t.close()
    assert finished, "serve() returned while the in-flight build was still running"


async def test_shutdown_grace_expiry_cancels_without_hanging(tmp_path, monkeypatch, caplog) -> None:
    """A build past the grace is abandoned (logged), not awaited forever."""
    mcp_t, sim_t = MemoryTransport.create_pair()
    client = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=tmp_path,
        submit_config=SubmitConfig(setup_root=tmp_path / "sims"),
        poll_interval_s=0.01,
        shutdown_grace_s=0.2,
    )
    started = asyncio.Event()

    async def slow_submit(*, prepared, emit, job_id_reader) -> dict:
        started.set()
        await asyncio.sleep(5)
        return {
            "sim_id": prepared.payload.sim_id,
            "state": SimulationState.WORKFLOW_FINISHED.value,
            "job_id": None,
            "run_dir": "/tmp/x",
            "stdout_path": None,
        }

    monkeypatch.setattr(client_mod, "execute_submit", slow_submit)
    prepared = PreparedSubmit(
        payload=SimpleNamespace(sim_id=SIM),
        params=SubmitParams(),
        runner=None,
        config=SubmitConfig(setup_root=tmp_path / "sims"),
    )
    serve_task = asyncio.create_task(client.serve())
    try:
        await asyncio.sleep(0.02)
        client._serving = True
        client._start_submit_task(cmd_id="b" * 32, prepared=prepared, payload_hash="h", declared_sim_id=SIM)
        await asyncio.wait_for(started.wait(), timeout=3)
        loop = asyncio.get_running_loop()
        with caplog.at_level("WARNING", logger="pic_agentic.simclient.client"):
            deadline = loop.time() + 2.0
            serve_task.cancel()
            await asyncio.gather(serve_task, return_exceptions=True)
            assert loop.time() < deadline
    finally:
        await sim_t.close()
        await mcp_t.close()
    assert any("shutdown grace" in record.message for record in caplog.records)


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
