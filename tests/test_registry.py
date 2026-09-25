# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Offline tests for the sim_id-keyed registry projected from room events.

The registry is a projection of the signed room, so these tests build real,
signed ``RcpMessage`` envelopes (via the frozen protocol builders) and drive
them through :meth:`SubmitService.on_message` / ``ingest_backfill``.  No
cluster, transport or homeserver is involved.
"""

from __future__ import annotations

from pic_agentic.protocol.simulation import (
    SimulationPayload,
    SimulationState,
    SubmitParams,
    build_submit_ack,
    build_submit_command,
    build_submit_event,
)
from pic_agentic.rcp import RcpMessage, new_secret_hex
from pic_agentic.server.simulation import SimRecord, SubmitService

SECRET = new_secret_hex()
SIM = "7f3a2b1c"
SIM_ID = "abcd1234"
CMD_ID = "cmd-1"


def _service() -> SubmitService:
    return SubmitService(sim=SIM, secret=SECRET, ack_timeout_s=0.05)


def _event(
    state: SimulationState,
    *,
    sim_id: str = SIM_ID,
    cmd_id: str = CMD_ID,
    seq: int = 1,
    ts: str | None = None,
    **fields: object,
) -> RcpMessage:
    message = build_submit_event(sim=SIM, seq=seq, cmd_id=cmd_id, sim_id=sim_id, state=state, **fields)
    if ts is not None:
        message.ts = ts
    return message.sign(SECRET)


def _ack(
    *,
    state: SimulationState = SimulationState.ACCEPTED,
    sim_id: str = SIM_ID,
    cmd_id: str = CMD_ID,
    seq: int = 1,
    job_id: int | None = None,
) -> RcpMessage:
    return build_submit_ack(
        sim=SIM,
        seq=seq,
        cmd_id=cmd_id,
        sim_id=sim_id,
        state=state,
        in_reply_to=None,
        job_id=job_id,
    ).sign(SECRET)


def _submit_command(cmd_id: str = CMD_ID, seq: int = 1) -> RcpMessage:
    payload = SimulationPayload.build(
        picongpu_version="0.9.0-dev",
        picongpu_revision="",
        schema_hash="",
        runner_dump={"sim": {"delta_t_si": 1e-15}},
    )
    return build_submit_command(
        sim=SIM,
        seq=seq,
        payload=payload,
        params=SubmitParams(),
        cmd_id=cmd_id,
    ).sign(SECRET)


def test_submit_ack_registers_the_sim_before_any_event() -> None:
    service = _service()
    service.on_message(_ack(state=SimulationState.ACCEPTED, job_id=None))

    record = service.get(SIM_ID)
    assert record is not None
    assert record.sim_id == SIM_ID
    assert record.cmd_id == CMD_ID
    assert record.state == SimulationState.ACCEPTED.value
    assert record.active is True
    assert record.last_event_ts is not None


def test_events_accumulate_into_one_record() -> None:
    service = _service()
    service.on_message(_ack())
    submitted = _event(SimulationState.SUBMITTED, seq=2, job_id=4242)
    # ``run_dir`` is cluster-local and only present when the simclient includes
    # it; the registry projects it when it is.
    submitted.payload["run_dir"] = "/cluster/run"
    submitted.sign(SECRET)
    service.on_message(submitted)
    service.on_message(_event(SimulationState.JOB_RUNNING, seq=3, job_id=4242, slurm_state="RUNNING"))
    service.on_message(
        _event(
            SimulationState.STEP_FINISHED,
            seq=4,
            job_id=4242,
            step=250,
            percent=25,
            walltime="1sec",
            avg_per_step="4msec",
            eta_s=750,
        )
    )

    assert len(service.list()) == 1
    record = service.get(SIM_ID)
    assert record is not None
    assert record.state == SimulationState.STEP_FINISHED.value
    assert record.last_event_type == SimulationState.STEP_FINISHED.value
    assert record.job_id == 4242
    assert record.run_dir == "/cluster/run"
    assert record.slurm_state == "RUNNING"
    assert record.step == 250
    assert record.percent == 25
    assert record.walltime == "1sec"
    assert record.avg_per_step == "4msec"
    assert record.eta_s == 750
    assert record.active is True


def test_terminal_state_marks_inactive() -> None:
    service = _service()
    service.on_message(_ack())
    service.on_message(_event(SimulationState.JOB_FINISHED, seq=2, job_id=1, slurm_state="COMPLETED", exit_code=0))
    assert service.get(SIM_ID).active is True  # type: ignore[union-attr]

    service.on_message(_event(SimulationState.RESULTS_READY, seq=3, job_id=1, results_linked=True))
    record = service.get(SIM_ID)
    assert record is not None
    assert record.active is False
    assert record.state == SimulationState.RESULTS_READY.value


def test_job_failed_and_simulation_failed_are_terminal() -> None:
    for terminal in (SimulationState.JOB_FAILED, SimulationState.FAILED):
        service = _service()
        service.on_message(_ack())
        service.on_message(_event(terminal, seq=2, job_id=1, slurm_state="FAILED", exit_code=3))
        record = service.get(SIM_ID)
        assert record is not None
        assert record.active is False
        assert record.exit_code == 3


def test_an_event_omitting_a_field_does_not_erase_it() -> None:
    service = _service()
    service.on_message(_event(SimulationState.SUBMITTED, seq=1, job_id=4242))
    # A later progress event carries no job_id; the known one must survive.
    service.on_message(_event(SimulationState.STEP_FINISHED, seq=2, step=10, percent=25))
    record = service.get(SIM_ID)
    assert record is not None
    assert record.job_id == 4242
    assert record.step == 10


def test_backfill_replay_is_idempotent_and_converges() -> None:
    events = [
        _ack(),
        _event(SimulationState.SUBMITTED, seq=2, job_id=4242, ts="2026-09-25T10:00:00Z"),
        _event(SimulationState.JOB_RUNNING, seq=3, job_id=4242, slurm_state="RUNNING", ts="2026-09-25T10:00:30Z"),
        _event(SimulationState.RESULTS_READY, seq=4, job_id=4242, ts="2026-09-25T10:05:00Z"),
    ]

    incremental = _service()
    for message in events:
        incremental.on_message(message)

    rebuilt = _service()
    rebuilt.ingest_backfill(events)
    rebuilt.ingest_backfill(events)  # replay twice

    assert rebuilt.get(SIM_ID) == incremental.get(SIM_ID)
    # A fresh service built from a replayed room converges to the same record.
    assert rebuilt.get(SIM_ID).active is False  # type: ignore[union-attr]


def test_backfill_ignores_unsigned_and_foreign_messages() -> None:
    service = _service()
    foreign = build_submit_event(
        sim="other",
        seq=1,
        cmd_id=CMD_ID,
        sim_id=SIM_ID,
        state=SimulationState.SUBMITTED,
    ).sign(SECRET)
    unsigned = build_submit_event(sim=SIM, seq=2, cmd_id=CMD_ID, sim_id=SIM_ID, state=SimulationState.SUBMITTED)
    tampered = build_submit_event(sim=SIM, seq=3, cmd_id=CMD_ID, sim_id=SIM_ID, state=SimulationState.SUBMITTED).sign(
        SECRET
    )
    tampered.payload["state"] = SimulationState.RESULTS_READY.value

    service.ingest_backfill([foreign, unsigned, tampered])
    assert service.registry == {}


def test_get_and_list_active_only() -> None:
    service = _service()
    service.on_message(_event(SimulationState.JOB_RUNNING, sim_id="aaaa1111", cmd_id="c1", seq=1, job_id=1))
    service.on_message(_event(SimulationState.JOB_RUNNING, sim_id="bbbb2222", cmd_id="c2", seq=2, job_id=2))
    service.on_message(_event(SimulationState.RESULTS_READY, sim_id="aaaa1111", cmd_id="c1", seq=3, job_id=1))

    assert service.get("missing") is None
    assert [record.sim_id for record in service.list()] == ["aaaa1111", "bbbb2222"]
    active = service.list(active_only=True)
    assert [record.sim_id for record in active] == ["bbbb2222"]
    assert all(isinstance(record, SimRecord) for record in active)


def test_event_log_is_bounded(monkeypatch) -> None:
    service = SubmitService(sim=SIM, secret=SECRET, ack_timeout_s=0.05, event_log_max=3)
    for index in range(5):
        service.on_message(_event(SimulationState.STEP_FINISHED, seq=index + 1, step=index, percent=index))
    assert len(service.event_log) == 3
    # The most recent events are retained.
    assert [message.payload["step"] for message in service.event_log] == [2, 3, 4]


def test_submit_ack_still_resolves_the_pending_future() -> None:
    import asyncio

    async def run() -> None:
        service = _service()
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        service._pending[CMD_ID] = future
        service.on_message(_ack(job_id=4242))
        ack = await asyncio.wait_for(future, timeout=1.0)
        assert ack.payload["job_id"] == 4242

    asyncio.run(run())


def test_submit_command_from_server_is_not_projected() -> None:
    # The room echoes our own outbound commands; they must not seed records.
    service = _service()
    service.on_message(_submit_command())
    assert service.registry == {}
    assert service.event_log == []
