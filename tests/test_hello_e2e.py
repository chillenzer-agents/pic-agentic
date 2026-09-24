# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""End-to-end ``hello`` test over the in-memory transport and fake SLURM.

This exercises the M1 acceptance path without a cluster or homeserver:
MCP-side service -> RCP command -> simclient -> fake ``sbatch``/``scontrol``
-> ack -> MCP-side result.  The Matrix transport is covered separately by a
live smoke test.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pic_agentic.rcp import new_secret_hex
from pic_agentic.server.hello import AckTimeoutError, HelloService
from pic_agentic.simclient import SimClient
from pic_agentic.simclient.client import ProcessedCommand
from pic_agentic.slurm import SlurmClient
from pic_agentic.transport.memory import MemoryTransport

FAKE_BIN = Path(__file__).parent / "fake_slurm"
SECRET = new_secret_hex()
SIM = "7f3a2b1c"

#: Plain message used by the round-trip and file-content assertions.
MESSAGE = "Hello World"
#: Written to the shared file system by the round-trip test; also asserted.
SHORT_MESSAGE = "hi"
#: Shell metacharacters only; carries no destructive command so a broken
#: assertion can never execute anything harmful.
SHELL_METACHARACTERS = "$(touch /tmp/pwned); `id`; '; echo pwned"


@pytest.fixture
def shared_dir(tmp_path, monkeypatch):
    state = tmp_path / "fake-slurm"
    monkeypatch.setenv("FAKE_SLURM_STATE", str(state))
    d = tmp_path / "shared"
    d.mkdir()
    return d


async def _run_pair(shared_dir: Path, message: str, *, wait_timeout: float = 60.0):
    mcp_t, sim_t = MemoryTransport.create_pair()
    service = HelloService(sim=SIM, secret=SECRET, message_dir=shared_dir, ack_timeout_s=10.0)
    simclient = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared_dir,
        job_wait_timeout_s=wait_timeout,
        poll_interval_s=0.05,
    )

    async def pump() -> None:
        async for msg in mcp_t.receive():
            service.on_message(msg)

    pump_task = asyncio.create_task(pump())
    serve_task = asyncio.create_task(simclient.serve())
    try:
        outcome = await service.hello(mcp_t.send, message)
    finally:
        serve_task.cancel()
        pump_task.cancel()
        await sim_t.close()
        await mcp_t.close()
    return outcome


async def test_hello_round_trip(shared_dir) -> None:
    outcome = await _run_pair(shared_dir, MESSAGE)
    assert outcome.acked
    assert outcome.ok, outcome.error
    assert isinstance(outcome.job_id, int)
    assert outcome.job_id > 0
    assert outcome.cluster_output is not None
    assert MESSAGE in outcome.cluster_output


async def test_hello_message_is_not_shell_interpolated(shared_dir) -> None:
    outcome = await _run_pair(shared_dir, SHELL_METACHARACTERS)
    assert outcome.ok, outcome.error
    assert outcome.cluster_output is not None
    # The literal string comes back verbatim; nothing was executed.
    assert SHELL_METACHARACTERS.strip() in outcome.cluster_output
    assert not Path("/tmp/pwned").exists()


async def test_message_file_is_written_under_base_dir(shared_dir) -> None:
    outcome = await _run_pair(shared_dir, SHORT_MESSAGE)
    msgs = list((shared_dir / "msg").glob("*.txt"))
    assert len(msgs) == 1
    assert msgs[0].read_text() == SHORT_MESSAGE
    assert outcome.job_id is not None


async def test_rejects_unsigned_commands(shared_dir) -> None:
    from pic_agentic.protocol.hello import build_hello_command

    mcp_t, sim_t = MemoryTransport.create_pair()
    simclient = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared_dir,
    )
    forged = build_hello_command(sim=SIM, seq=1, message_path=str(shared_dir / "msg" / "x.txt"))
    # not signed
    await mcp_t.send(forged)
    result = await asyncio.wait_for(simclient.handle(forged), timeout=2)
    assert result is None


async def test_rejects_path_outside_base_dir(shared_dir) -> None:
    from pic_agentic.protocol.hello import build_hello_command

    _mcp_t, sim_t = MemoryTransport.create_pair()
    simclient = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared_dir,
    )
    evil = build_hello_command(sim=SIM, seq=1, message_path="/etc/passwd", message="x").sign(SECRET)
    ack = await simclient.handle(evil)
    assert ack is not None
    assert ack.payload["error"].startswith("unexpected:path escapes")


async def test_duplicate_command_is_idempotent(shared_dir) -> None:
    _mcp_t, sim_t = MemoryTransport.create_pair()
    service = HelloService(sim=SIM, secret=SECRET, message_dir=shared_dir)
    simclient = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared_dir,
        poll_interval_s=0.05,
    )
    command = service.build_command("dup")
    first = await simclient.handle(command)
    # A replay is a distinct delivery event (new transport event id) carrying the
    # same cmd_id; dedup must not swallow it before the idempotency guard.
    command.transport_event_id = "$memory-replay"
    second = await simclient.handle(command)
    assert first is not None
    assert second is not None  # re-sent command is re-acked, not silently dropped
    assert second.payload["cmd_id"] == first.payload["cmd_id"]
    assert second.payload["job_id"] == first.payload["job_id"]
    jobs = list((shared_dir / "out").glob("*.out"))
    assert len(jobs) == 1


async def test_replay_resends_ack_without_resubmitting(shared_dir) -> None:
    """A lost ack replayed after a restart must be re-acked, not left to time out.

    Regression from the live run: the processed-id guard returned ``None`` on a
    replay, so ``HelloService`` blocked for the full ack timeout although the
    job had already completed.  The stored result must be re-sent.
    """
    mcp_t, sim_t = MemoryTransport.create_pair()
    service = HelloService(sim=SIM, secret=SECRET, message_dir=shared_dir, ack_timeout_s=2.0)
    command = service.build_command("replay")
    cmd_id = str(command.payload["cmd_id"])

    first = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared_dir,
        poll_interval_s=0.05,
    )
    original = await first.handle(command)
    assert original is not None
    assert len(list((shared_dir / "out").glob("*.out"))) == 1

    # Simulate a restart whose original ack was lost: a fresh client loads the
    # durable record and the sender replays the command through the pump.
    second = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared_dir,
        poll_interval_s=0.05,
    )
    pump_task = asyncio.create_task(_pump(mcp_t, service))
    serve_task = asyncio.create_task(second.serve())
    try:
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        service._pending[cmd_id] = future
        await mcp_t.send(command)
        ack = await asyncio.wait_for(future, timeout=2.0)
    finally:
        serve_task.cancel()
        pump_task.cancel()
        await sim_t.close()
        await mcp_t.close()

    assert ack.payload["cmd_id"] == cmd_id
    assert ack.payload["job_id"] == original.payload["job_id"]
    assert ack.payload["cluster_output"] == original.payload["cluster_output"]
    # The replay did not submit a second job.
    assert len(list((shared_dir / "out").glob("*.out"))) == 1


async def _pump(mcp_t, service) -> None:
    async for msg in mcp_t.receive():
        service.on_message(msg)


async def test_processed_command_survives_restart(shared_dir) -> None:
    """A restart that backfills the room must not re-submit an executed command.

    Regression from the live cluster run: the transport replays the timeline on
    reconnect and the in-memory cmd_id set was lost on restart, so the simclient
    re-executed a backfilled command and submitted a duplicate cluster job.
    """
    _mcp_t, sim_t = MemoryTransport.create_pair()
    service = HelloService(sim=SIM, secret=SECRET, message_dir=shared_dir)
    command = service.build_command("restart")

    first = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared_dir,
        poll_interval_s=0.05,
    )
    assert await first.handle(command) is not None
    assert len(list((shared_dir / "out").glob("*.out"))) == 1

    # New process, same shared dir: the backfilled command is re-acked, not
    # re-executed.
    second = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared_dir,
        poll_interval_s=0.05,
    )
    replayed = await second.handle(command)
    assert replayed is not None
    assert replayed.payload["job_id"] is not None
    assert len(list((shared_dir / "out").glob("*.out"))) == 1


async def test_crash_before_result_is_not_resubmitted(shared_dir) -> None:
    """A record persisted before execution (no outcome) must still block a resubmit."""
    _mcp_t, sim_t = MemoryTransport.create_pair()
    service = HelloService(sim=SIM, secret=SECRET, message_dir=shared_dir)
    command = service.build_command("crash")
    cmd_id = str(command.payload["cmd_id"])

    stalled = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared_dir,
        poll_interval_s=0.05,
    )
    # Mimic the pre-execution record a crash would leave behind.
    stalled._persist_processed(ProcessedCommand(cmd_id=cmd_id))

    second = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared_dir,
        poll_interval_s=0.05,
    )
    ack = await second.handle(command)
    assert ack is not None
    assert ack.payload["error"] == "already_submitted:outcome_unknown"
    assert not list((shared_dir / "out").glob("*.out"))


async def test_rejects_command_from_unexpected_transport_sender(shared_dir) -> None:
    from pic_agentic.protocol.hello import build_hello_command

    _mcp_t, sim_t = MemoryTransport.create_pair()
    simclient = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared_dir,
        allowed_sender_user_id="@mcpserver:localhost",
    )
    command = build_hello_command(sim=SIM, seq=1, message_path=str(shared_dir / "msg" / "s.txt"), message="x").sign(
        SECRET,
    )
    command.transport_sender = "@intruder:localhost"
    assert await simclient.handle(command) is None

    command.transport_sender = "@mcpserver:localhost"
    ack = await simclient.handle(command)
    assert ack is not None
    assert ack.payload.get("error") is None


async def test_ack_timeout_when_no_simclient(shared_dir) -> None:
    mcp_t, _sim_t = MemoryTransport.create_pair()
    service = HelloService(sim=SIM, secret=SECRET, message_dir=shared_dir, ack_timeout_s=0.2)
    with pytest.raises(AckTimeoutError):
        await service.hello(mcp_t.send, "nobody home")
