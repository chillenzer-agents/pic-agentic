# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Client-side M3 result-request test (the ``_handle_result`` seam).

Exercises the simclient's ``RESULT_COMMAND`` dispatch against a real on-disk
``simOutput`` tree: ``describe`` and ``read`` must work with no openPMD reader
installed (the scan/text paths never import it), while a mesh op on a ``.bp``
file degrades to ``reader_unavailable``.  No cluster or homeserver involved.
"""

from __future__ import annotations

from pathlib import Path

from pic_agentic.protocol.simulation import (
    ResultOp,
    ResultParams,
    SimulationType,
    build_result_command,
)
from pic_agentic.rcp import new_secret_hex
from pic_agentic.simclient import SimClient
from pic_agentic.simclient.follow import TrackedSim
from pic_agentic.simclient.simulation import SubmitConfig
from pic_agentic.slurm import SlurmClient
from pic_agentic.transport.memory import MemoryTransport

FAKE_BIN = Path(__file__).parent / "fake_slurm"
SECRET = new_secret_hex()
SIM = "7f3a2b1c"


def _client(tmp_path: Path) -> tuple[MemoryTransport, SimClient]:
    _mcp_t, sim_t = MemoryTransport.create_pair()
    client = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=tmp_path,
        submit_config=SubmitConfig(setup_root=tmp_path / "sims"),
    )
    return sim_t, client


def _track_sim(client: SimClient, run_dir: Path, sim_id: str = "res12345") -> None:
    client._tracked[sim_id] = TrackedSim(
        sim_id=sim_id,
        cmd_id="c",
        job_id=1,
        run_dir=str(run_dir),
        stdout_path=None,
        submit_system="sbatch",
    )


def _request(sim_id: str, op: ResultOp, *, seq: int = 1, **knobs: object) -> object:
    params = ResultParams(sim_id=sim_id, op=op, **knobs)
    return build_result_command(sim=SIM, seq=seq, params=params, cmd_id=f"res-{op.value}-{seq}").sign(SECRET)


async def test_describe_and_read_without_openpmd(tmp_path) -> None:
    run = tmp_path / "run"
    output = run / "simOutput"
    (output / "openPMD").mkdir(parents=True)
    (output / "openPMD" / "sim_0000.bp").write_bytes(b"\x00\x01\x02\x03")
    (output / "energy.txt").write_text("step\tenergy\n0\t1.5\n", encoding="utf-8")
    sim_t, client = _client(tmp_path)
    try:
        _track_sim(client, run)

        describe = await client.handle(_request("res12345", ResultOp.DESCRIBE))
        assert describe is not None
        assert describe.type == SimulationType.RESULT_ACK
        manifest = describe.payload["manifest"]
        assert manifest["sim_id"] == "res12345"
        assert manifest["total_bytes"] == 4 + len((output / "energy.txt").read_text(encoding="utf-8"))
        assert {ref["path"] for ref in manifest["files"]} >= {"openPMD/sim_0000.bp", "energy.txt"}

        read = await client.handle(_request("res12345", ResultOp.READ, seq=2, path="energy.txt"))
        assert read is not None
        assert read.payload["data_encoding"] == "text"
        assert "energy" in "\n".join(read.payload["data"])

        mesh = await client.handle(
            _request("res12345", ResultOp.SLICE, seq=3, path="openPMD/sim_0000.bp", record="E"),
        )
        assert mesh is not None
        # The 4-byte fake is not a valid openPMD series: without the reader this
        # is reader_unavailable, with it a clean no_results/reader error.  Never
        # a crash.
        assert mesh.payload.get("error_code") in {
            "reader_unavailable",
            "no_results",
            "result_failed:ResultsUnavailable",
        }
    finally:
        await sim_t.close()


async def test_result_unknown_sim_and_bad_path(tmp_path) -> None:
    sim_t, client = _client(tmp_path)
    try:
        unknown = await client.handle(_request("deadbeef", ResultOp.DESCRIBE))
        assert unknown is not None
        assert unknown.type == SimulationType.RESULT_ACK
        assert unknown.payload["error"] == "unknown_sim"
        assert unknown.payload["error_code"] == "no_results"

        # A path escape never parses into ResultParams, so it is rejected
        # before reaching the engine.
        bad = build_result_command(
            sim=SIM,
            seq=2,
            params=ResultParams(sim_id="deadbeef", op=ResultOp.READ, path="ok.txt"),
            cmd_id="res-bad",
        )
        # Simulate a forged payload that bypassed model validation.
        forged = bad.model_copy(
            update={"payload": {**bad.payload, "path": "../../etc/passwd"}},
        ).sign(SECRET)
        _track_sim(client, tmp_path)
        rejected = await client.handle(forged)
        assert rejected is not None
        assert rejected.type == SimulationType.RESULT_ACK
    finally:
        await sim_t.close()
