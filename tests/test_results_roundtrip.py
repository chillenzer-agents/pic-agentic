# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Server<->client M3 round-trip test (the seam between the two wave-1 halves).

Unlike ``test_results_tools`` (a fake responder) and ``test_results_client_e2e``
(the client alone), this wires the real :class:`SubmitService` to the real
:class:`SimClient` over ``MemoryTransport`` and asserts that the server's
pending result pull is resolved by the client's ``result_ack`` -- i.e. the
``_PULL_ACK_FOR_REQUEST`` kind-matching actually matches the new M3 ack types.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pic_agentic.protocol.simulation import ResultOp, ResultParams, SimulationOp
from pic_agentic.rcp import new_secret_hex
from pic_agentic.server.simulation import SubmitService
from pic_agentic.simclient import SimClient
from pic_agentic.simclient.follow import TrackedSim
from pic_agentic.simclient.simulation import SubmitConfig
from pic_agentic.slurm import SlurmClient
from pic_agentic.transport.memory import MemoryTransport

FAKE_BIN = Path(__file__).parent / "fake_slurm"
SECRET = new_secret_hex()
SIM = "7f3a2b1c"


async def test_server_result_pull_resolves_against_real_client(tmp_path) -> None:
    run = tmp_path / "run"
    output = run / "simOutput"
    output.mkdir(parents=True)
    (output / "energy.txt").write_text("step\ten\n0\t2.5\n", encoding="utf-8")

    mcp_t, sim_t = MemoryTransport.create_pair()
    service = SubmitService(sim=SIM, secret=SECRET, ack_timeout_s=5.0)
    client = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=tmp_path,
        submit_config=SubmitConfig(setup_root=tmp_path / "sims"),
    )
    client._tracked["res12345"] = TrackedSim(
        sim_id="res12345",
        cmd_id="c",
        job_id=1,
        run_dir=str(run),
        stdout_path=None,
        submit_system="sbatch",
    )

    async def pump_client() -> None:
        async for msg in sim_t.receive():
            await client.handle(msg)

    async def pump_server() -> None:
        async for msg in mcp_t.receive():
            service.on_message(msg)

    tasks = [asyncio.create_task(pump_client()), asyncio.create_task(pump_server())]
    try:
        payload = await service.fetch_result(
            mcp_t.send,
            ResultParams(sim_id="res12345", op=ResultOp.READ, path="energy.txt"),
        )
        assert payload["data_encoding"] == "text"
        assert "2.5" in "\n".join(payload["data"])
        assert payload["op"] == "read"
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await sim_t.close()
        await mcp_t.close()


async def test_server_control_pull_resolves_against_real_client(tmp_path) -> None:
    """The server's control pull is resolved by the real client's CONTROL_ACK."""
    state_dir = tmp_path / "fake-slurm"
    state_dir.mkdir()
    job_id = 700777
    (state_dir / f"{job_id}.state").write_text("RUNNING\n", encoding="utf-8")
    monkey = __import__("os").environ
    monkey["FAKE_SLURM_STATE"] = str(state_dir)
    try:
        mcp_t, sim_t = MemoryTransport.create_pair()
        service = SubmitService(sim=SIM, secret=SECRET, ack_timeout_s=5.0)
        client = SimClient(
            sim=SIM,
            secret=SECRET,
            transport=sim_t,
            slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
            message_dir=tmp_path,
            submit_config=SubmitConfig(setup_root=tmp_path / "sims"),
            control_fn=_real_control_fn,
        )
        client._tracked["ctl12345"] = TrackedSim(
            sim_id="ctl12345",
            cmd_id="c",
            job_id=job_id,
            run_dir=str(tmp_path),
            stdout_path=None,
            submit_system="sbatch",
        )

        async def pump_client() -> None:
            async for msg in sim_t.receive():
                await client.handle(msg)

        async def pump_server() -> None:
            async for msg in mcp_t.receive():
                service.on_message(msg)

        tasks = [asyncio.create_task(pump_client()), asyncio.create_task(pump_server())]
        try:
            payload = await service.control(mcp_t.send, "ctl12345", SimulationOp.CHECKPOINT)
            assert payload["ok"] is True
            assert payload["signal"] == "USR1"
            assert "Signal USR1 sent" in payload["slurm_reason"]
            assert (state_dir / f"{job_id}.signals").read_text(encoding="utf-8").splitlines() == ["USR1"]
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await sim_t.close()
            await mcp_t.close()
    finally:
        monkey.pop("FAKE_SLURM_STATE", None)


async def _real_control_fn(op: SimulationOp, tracked: TrackedSim) -> str:
    slurm = SlurmClient(bin_dir=str(FAKE_BIN))
    if op is SimulationOp.CANCEL:
        return await slurm.cancel_job(tracked.job_id or 0)
    from pic_agentic.protocol.simulation import CONTROL_SIGNAL

    return await slurm.signal_job(tracked.job_id or 0, CONTROL_SIGNAL[op] or "")
