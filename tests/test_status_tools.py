# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Offline tests for the server-side status/log pulls and event condensation.

A tiny fake responder answers ``status_request`` / ``logs_request`` over
``MemoryTransport``; it deliberately does not depend on the real simclient.
The event-condensation helper is exercised directly with a synthetic log.
"""

from __future__ import annotations

import asyncio

import pytest

from pic_agentic.protocol.simulation import (
    SimulationState,
    SimulationType,
    build_logs_ack,
    build_status_ack,
    build_submit_event,
)
from pic_agentic.rcp import new_secret_hex
from pic_agentic.server.simulation import MAX_EVENT_PAGE, SubmitService, condense_events
from pic_agentic.transport.memory import MemoryTransport

SECRET = new_secret_hex()
SIM = "7f3a2b1c"
SIM_ID = "abcd1234"
JOB_ID = 4242


def _service(*, ack_timeout_s: float = 2.0) -> SubmitService:
    return SubmitService(sim=SIM, secret=SECRET, ack_timeout_s=ack_timeout_s)


async def _serve_acks(sim_transport: MemoryTransport, service: SubmitService) -> None:
    """Answer status/logs requests on the sim side; resolve acks on the server.

    Runs both directions until cancelled: the sim-side responder and the
    server-side pump that feeds ``service.on_message``.
    """

    async def responder() -> None:
        async for command in sim_transport.receive():
            payload = command.payload
            if command.type == SimulationType.STATUS_COMMAND:
                ack = build_status_ack(
                    sim=SIM,
                    seq=1,
                    cmd_id=str(payload.get("cmd_id", "")),
                    sim_id=str(payload.get("sim_id", "")),
                    in_reply_to=command.transport_event_id,
                    state=SimulationState.JOB_RUNNING.value,
                    slurm_state="RUNNING",
                    job_id=JOB_ID,
                    step=250,
                    percent=25,
                    walltime="1min 0sec 0msec",
                    avg_per_step="4msec",
                    eta_s=750,
                ).sign(SECRET)
            elif command.type == SimulationType.LOGS_COMMAND:
                ack = build_logs_ack(
                    sim=SIM,
                    seq=2,
                    cmd_id=str(payload.get("cmd_id", "")),
                    sim_id=str(payload.get("sim_id", "")),
                    in_reply_to=command.transport_event_id,
                    stream=str(payload.get("stream", "stdout")),
                    lines=["line-1", "line-2"],
                    total_lines=42,
                ).sign(SECRET)
            else:  # pragma: no cover - the server never sends anything else here
                continue
            await sim_transport.send(ack)

    return asyncio.create_task(responder())


async def _pump(mcp_transport: MemoryTransport, service: SubmitService) -> asyncio.Task:
    async def pump() -> None:
        async for message in mcp_transport.receive():
            service.on_message(message)

    return asyncio.create_task(pump())


async def test_fetch_status_round_trip() -> None:
    mcp_t, sim_t = MemoryTransport.create_pair()
    service = _service()
    tasks = [await _serve_acks(sim_t, service), await _pump(mcp_t, service)]
    try:
        payload = await service.fetch_status(mcp_t.send, SIM_ID)
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()

    assert payload["sim_id"] == SIM_ID
    assert payload["state"] == SimulationState.JOB_RUNNING.value
    assert payload["slurm_state"] == "RUNNING"
    assert payload["job_id"] == JOB_ID
    assert payload["step"] == 250
    assert payload["percent"] == 25
    assert payload["eta_s"] == 750


async def test_fetch_logs_round_trip() -> None:
    mcp_t, sim_t = MemoryTransport.create_pair()
    service = _service()
    tasks = [await _serve_acks(sim_t, service), await _pump(mcp_t, service)]
    try:
        payload = await service.fetch_logs(mcp_t.send, SIM_ID, stream="stderr", tail=7)
    finally:
        for task in tasks:
            task.cancel()
        await mcp_t.close()
        await sim_t.close()

    assert payload["stream"] == "stderr"
    assert payload["lines"] == ["line-1", "line-2"]
    assert payload["total_lines"] == 42


async def test_fetch_logs_rejects_unknown_stream() -> None:
    service = _service()
    with pytest.raises(ValueError, match="unknown log stream"):
        await service.fetch_logs(lambda _m: None, SIM_ID, stream="nope")  # type: ignore[arg-type]


async def test_wrong_ack_type_does_not_resolve_a_pending_pull() -> None:
    """A logs_ack carrying a status request's cmd_id must not resolve it (#6a)."""
    service = _service(ack_timeout_s=0.05)
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    service._pending_pull["shared-cmd"] = future
    service._pending_pull_kind["shared-cmd"] = SimulationType.STATUS_COMMAND
    misrouted = build_logs_ack(
        sim=SIM,
        seq=1,
        cmd_id="shared-cmd",
        sim_id=SIM_ID,
        in_reply_to=None,
        stream="stdout",
        lines=["nope"],
        total_lines=1,
    ).sign(SECRET)
    service.on_message(misrouted)
    assert not future.done()
    assert not service._pending_pull["shared-cmd"].done()


async def test_fetch_status_times_out_without_raising() -> None:
    mcp_t, _sim_t = MemoryTransport.create_pair()
    service = _service(ack_timeout_s=0.05)
    try:
        payload = await service.fetch_status(mcp_t.send, SIM_ID)
    finally:
        await mcp_t.close()
    assert payload == {"sim_id": SIM_ID, "error": "timeout"}


def _event(
    state: SimulationState,
    *,
    ts: str,
    sim_id: str = SIM_ID,
    seq: int = 1,
    **fields: object,
):
    message = build_submit_event(sim=SIM, seq=seq, cmd_id="c1", sim_id=sim_id, state=state, **fields)
    message.ts = ts
    return message.sign(SECRET)


def test_condense_dedups_consecutive_states() -> None:
    log = [
        _event(SimulationState.ACCEPTED, ts="2026-09-25T10:00:00Z", seq=1),
        _event(SimulationState.SUBMITTED, ts="2026-09-25T10:00:01Z", seq=2, job_id=JOB_ID),
        _event(SimulationState.STEP_FINISHED, ts="2026-09-25T10:00:02Z", seq=3, step=250, percent=25),
        _event(SimulationState.STEP_FINISHED, ts="2026-09-25T10:00:03Z", seq=4, step=500, percent=50),
        _event(SimulationState.STEP_FINISHED, ts="2026-09-25T10:00:04Z", seq=5, step=750, percent=75),
        _event(SimulationState.RESULTS_READY, ts="2026-09-25T10:00:05Z", seq=6),
    ]
    condensed = condense_events(log, sim_id=SIM_ID)
    assert [entry["state"] for entry in condensed] == [
        SimulationState.ACCEPTED.value,
        SimulationState.SUBMITTED.value,
        SimulationState.STEP_FINISHED.value,
        SimulationState.RESULTS_READY.value,
    ]
    # The surviving progress row keeps the most recent payload and its ts.
    step = next(entry for entry in condensed if entry["state"] == SimulationState.STEP_FINISHED.value)
    assert step["percent"] == 75
    assert step["ts"] == "2026-09-25T10:00:04Z"


def test_condense_filters_by_since_and_types() -> None:
    log = [
        _event(SimulationState.SUBMITTED, ts="2026-09-25T10:00:00Z", seq=1, job_id=JOB_ID),
        _event(SimulationState.RESULTS_READY, ts="2026-09-25T10:10:00Z", seq=2),
    ]
    assert [entry["state"] for entry in condense_events(log, sim_id=SIM_ID, since="2026-09-25T10:05:00Z")] == [
        SimulationState.RESULTS_READY.value
    ]
    assert [
        entry["state"] for entry in condense_events(log, sim_id=SIM_ID, types=[SimulationState.SUBMITTED.value])
    ] == [SimulationState.SUBMITTED.value]


def test_condense_caps_limit_and_ignores_other_sims() -> None:
    log = [
        _event(SimulationState.STEP_FINISHED, ts="2026-09-25T10:00:00Z", seq=1, percent=25),
        _event(SimulationState.ACCEPTED, ts="2026-09-25T10:00:01Z", sim_id="other", seq=2),
        _event(SimulationState.STEP_FINISHED, ts="2026-09-25T10:00:02Z", seq=3, percent=50),
        _event(SimulationState.RESULTS_READY, ts="2026-09-25T10:00:03Z", seq=4),
    ]
    condensed = condense_events(log, sim_id=SIM_ID, limit=2)
    assert len(condensed) == 2
    # The tail is retained, and the other sim's event is never included.
    assert [entry["state"] for entry in condensed] == [
        SimulationState.STEP_FINISHED.value,
        SimulationState.RESULTS_READY.value,
    ]
    assert condense_events(log, sim_id=SIM_ID, limit=0) == []
    assert condense_events(log, sim_id=SIM_ID, limit=10_000_000) == condense_events(
        log, sim_id=SIM_ID, limit=MAX_EVENT_PAGE
    )


async def test_get_status_tool_merges_the_live_view() -> None:
    from pic_agentic.config import Config
    from pic_agentic.server.app import build_server

    mcp_t, sim_t = MemoryTransport.create_pair()
    config = Config(rcp_secret=SECRET, access_token="super-secret-token")
    server, runtime = build_server(config, SIM)
    service = runtime.submit_service
    # Seed the registry with an active, non-terminal record and wire the
    # transport so the tool performs a live pull.
    service.on_message(_event(SimulationState.SUBMITTED, ts="2026-09-25T10:00:00Z", seq=1, job_id=JOB_ID))
    runtime._transport = mcp_t
    responder = await _serve_acks(sim_t, service)
    pump = await _pump(mcp_t, service)
    try:
        result = await server.call_tool("get_status", {"sim_id": SIM_ID})
    finally:
        responder.cancel()
        pump.cancel()
        await mcp_t.close()
        await sim_t.close()

    payload = result.structured_content
    # The live scontrol view wins over the stale registry projection.
    assert payload["state"] == SimulationState.JOB_RUNNING.value
    assert payload["slurm_state"] == "RUNNING"
    assert payload["step"] == 250
    assert payload["percent"] == 25
    assert payload["since_last_event_s"] is not None


async def test_get_status_reports_progress_from_step_finished_events() -> None:
    """A running sim's projection carries step/percent/eta_s (mid-run progress).

    The bounded ``step_finished`` events carry the progress, so the registry
    projection (used without a live pull, and as the fallback when a live ack
    omits a field) must expose it while the job runs, not only at completion.
    """
    from pic_agentic.config import Config
    from pic_agentic.server.app import build_server

    config = Config(rcp_secret=SECRET)
    server, runtime = build_server(config, SIM)
    service = runtime.submit_service
    service.on_message(_event(SimulationState.JOB_RUNNING, ts="2026-09-25T10:00:00Z", seq=1, job_id=JOB_ID))
    service.on_message(
        _event(
            SimulationState.STEP_FINISHED,
            ts="2026-09-25T10:00:01Z",
            seq=2,
            step=250,
            percent=25,
            walltime="1min 0sec 0msec",
            avg_per_step="4msec",
            eta_s=750,
        )
    )
    result = await server.call_tool("get_status", {"sim_id": SIM_ID})
    payload = result.structured_content
    assert payload["state"] == SimulationState.STEP_FINISHED.value
    assert payload["step"] == 250
    assert payload["percent"] == 25
    assert payload["eta_s"] == 750
    assert payload["walltime"] == "1min 0sec 0msec"


async def test_get_status_surfaces_the_suspect_flag() -> None:
    """get_status and list_simulations must carry the F4 health flag (M1).

    The PR documents the flag reaching get_status/fleet_status; fleet_status was
    wired but the two registry-backed tools omitted it, so an operator asking
    about a single terminal run saw a clean one.
    """
    from pic_agentic.config import Config
    from pic_agentic.server.app import build_server

    config = Config(rcp_secret=SECRET)
    server, runtime = build_server(config, SIM)
    record = runtime.submit_service._record_for(SIM_ID, cmd_id="c1")
    record.state = SimulationState.RESULTS_READY.value
    record.active = False
    record.suspect = "energy_histogram is all zeros"
    runtime.submit_service.registry[SIM_ID] = record

    status = (await server.call_tool("get_status", {"sim_id": SIM_ID})).structured_content
    assert status["suspect"] == "energy_histogram is all zeros"
    rows = (await server.call_tool("list_simulations", {})).structured_content["simulations"]
    assert rows[0]["suspect"] == "energy_histogram is all zeros"


def test_status_dict_surfaces_failure_fields() -> None:
    """B1: a terminal build failure is legible from the get_status projection.

    The event payload and ``last_status`` carry these fields, but the primary
    ``get_status`` projection omitted them, so ``stage=build`` was invisible on
    the surface an operator actually polls.
    """
    from pic_agentic.server.app import _status_dict
    from pic_agentic.server.simulation import SimRecord

    record = SimRecord(
        sim_id=SIM_ID,
        cmd_id="c1",
        state=SimulationState.FAILED.value,
        error_code="build_failed",
        stage="build",
        failure_summary="Yee.hpp(56): error: CFL static_assert",
        exit_code=139,
    )
    status = _status_dict(record)
    assert status["error_code"] == "build_failed"
    assert status["stage"] == "build"
    assert status["failure_summary"] == "Yee.hpp(56): error: CFL static_assert"
    assert status["exit_code"] == 139
    assert status["exit_signal"] == "SIGSEGV"


def test_status_dict_surfaces_the_pending_reason() -> None:
    """F3: a blocked pending job reports its scheduler reason and a flag.

    ``slurm_state: PENDING`` alone cannot tell a doomed job from one waiting its
    turn, so ``get_status``/``list_simulations`` also expose ``slurm_reason``
    and the derived ``slurm_blocked``.
    """
    from pic_agentic.server.app import _status_dict
    from pic_agentic.server.simulation import SimRecord

    record = SimRecord(
        sim_id=SIM_ID,
        cmd_id="c1",
        state=SimulationState.SUBMITTED.value,
        job_id=JOB_ID,
        slurm_state="PENDING",
        slurm_reason="PartitionNodeLimit",
    )
    status = _status_dict(record)
    assert status["slurm_state"] == "PENDING"
    assert status["slurm_reason"] == "PartitionNodeLimit"
    assert status["slurm_blocked"] is True


def test_status_dict_does_not_flag_a_transient_reason() -> None:
    """A ``Resources`` wait is not a permanent block (F3 must not cry wolf)."""
    from pic_agentic.server.app import _status_dict
    from pic_agentic.server.simulation import SimRecord

    record = SimRecord(
        sim_id=SIM_ID,
        cmd_id="c1",
        state=SimulationState.SUBMITTED.value,
        job_id=JOB_ID,
        slurm_state="PENDING",
        slurm_reason="Resources",
    )
    status = _status_dict(record)
    assert status["slurm_reason"] == "Resources"
    assert status["slurm_blocked"] is False


def test_merge_status_recomputes_blocked_from_live_reason() -> None:
    """A live ack changing the reason must update the derived ``slurm_blocked``."""
    from pic_agentic.server.app import _merge_status

    projection = {
        "state": SimulationState.SUBMITTED.value,
        "job_id": JOB_ID,
        "slurm_state": "PENDING",
        "slurm_reason": "Resources",
        "slurm_blocked": False,
    }
    _merge_status(projection, {"slurm_state": "PENDING", "slurm_reason": "PartitionNodeLimit"})
    assert projection["slurm_blocked"] is True


def test_running_event_clears_a_stale_pending_reason() -> None:
    """A transient reason must not linger once the job is running (F3)."""
    service = _service()
    service.on_message(
        _event(
            SimulationState.SUBMITTED,
            ts="2026-09-25T10:00:01Z",
            seq=1,
            job_id=JOB_ID,
            slurm_state="PENDING",
            slurm_reason="Resources",
        ),
    )
    assert service.get(SIM_ID).slurm_reason == "Resources"
    service.on_message(_event(SimulationState.JOB_RUNNING, ts="2026-09-25T10:00:02Z", seq=2, job_id=JOB_ID))
    record = service.get(SIM_ID)
    assert record.slurm_reason is None
    assert record.slurm_blocked is False


async def test_get_status_projects_the_pending_reason() -> None:
    """F3 end-to-end through the tool: a blocked job is legible from get_status."""
    from pic_agentic.config import Config
    from pic_agentic.server.app import build_server

    config = Config(rcp_secret=SECRET)
    server, runtime = build_server(config, SIM)
    service = runtime.submit_service
    service.on_message(
        _event(
            SimulationState.SUBMITTED,
            ts="2026-09-25T10:00:01Z",
            seq=2,
            job_id=JOB_ID,
            slurm_state="PENDING",
            slurm_reason="PartitionNodeLimit",
        ),
    )
    status = (await server.call_tool("get_status", {"sim_id": SIM_ID})).structured_content
    row = (await server.call_tool("list_simulations", {})).structured_content["simulations"][0]
    assert status["slurm_reason"] == "PartitionNodeLimit"
    assert status["slurm_blocked"] is True
    assert row["slurm_reason"] == "PartitionNodeLimit"
    assert row["slurm_blocked"] is True


def test_merge_status_keeps_projection_progress_the_ack_omits() -> None:
    """A live ack that omits a progress field must not erase the projection."""
    from pic_agentic.server.app import _merge_status

    projection = {"step": 250, "percent": 25, "eta_s": 750, "walltime": "1min 0sec 0msec"}
    _merge_status(projection, {"state": "simulation.job_running", "step": None, "percent": None, "eta_s": None})
    assert projection["state"] == "simulation.job_running"
    assert projection["step"] == 250
    assert projection["percent"] == 25
    assert projection["eta_s"] == 750
    # A non-null ack value does override the projection.
    _merge_status(projection, {"percent": 40})
    assert projection["percent"] == 40


async def test_get_status_reports_phase_transitions() -> None:
    """H4: get_status/list_simulations expose the build-vs-queue substate.

    The phase is derived from the record's state and job id: ``accepted`` with
    no job id is ``building``; a job id without a running event is ``queued``;
    a running/job-progress event is ``running``; a completed run is ``done``.
    """
    from pic_agentic.config import Config
    from pic_agentic.server.app import build_server

    config = Config(rcp_secret=SECRET)
    server, runtime = build_server(config, SIM)
    service = runtime.submit_service

    async def phase() -> str:
        status = (await server.call_tool("get_status", {"sim_id": SIM_ID})).structured_content
        rows = (await server.call_tool("list_simulations", {})).structured_content["simulations"]
        assert status["phase"] == rows[0]["phase"]
        return status["phase"]

    service.on_message(_event(SimulationState.ACCEPTED, ts="2026-09-25T10:00:00Z", seq=1))
    assert await phase() == "building"

    service.on_message(_event(SimulationState.WORKFLOW_FINISHED, ts="2026-09-25T10:00:01Z", seq=2, job_id=JOB_ID))
    assert await phase() == "queued"

    service.on_message(_event(SimulationState.JOB_RUNNING, ts="2026-09-25T10:00:02Z", seq=3, job_id=JOB_ID))
    assert await phase() == "running"

    service.on_message(_event(SimulationState.RESULTS_READY, ts="2026-09-25T10:10:00Z", seq=4, job_id=JOB_ID))
    assert await phase() == "done"


async def test_get_events_explains_an_empty_building_phase() -> None:
    """H4: an empty get_events result names the build phase instead of silence."""
    from pic_agentic.config import Config
    from pic_agentic.server.app import build_server

    config = Config(rcp_secret=SECRET)
    server, runtime = build_server(config, SIM)
    record = runtime.submit_service._record_for(SIM_ID, cmd_id="c1")
    record.state = SimulationState.ACCEPTED.value
    record.active = True

    payload = (await server.call_tool("get_events", {"sim_id": SIM_ID})).structured_content
    assert payload["events"] == []
    assert payload["phase"] == "building"
    assert "building" in payload["note"]
    assert "SLURM job" in payload["note"]


async def test_get_events_terminal_note_does_not_blame_filters() -> None:
    """A finished, unfiltered empty result must not point at `since`/`types`."""
    from pic_agentic.config import Config
    from pic_agentic.server.app import build_server

    config = Config(rcp_secret=SECRET)
    server, runtime = build_server(config, SIM)
    record = runtime.submit_service._record_for(SIM_ID, cmd_id="c1")
    record.state = SimulationState.RESULTS_READY.value
    record.active = False

    payload = (await server.call_tool("get_events", {"sim_id": SIM_ID})).structured_content
    assert payload["events"] == []
    assert payload["phase"] == "done"
    assert "finished" in payload["note"]
    assert "filters" not in payload["note"]


async def test_get_events_terminal_note_with_filter_mentions_filters() -> None:
    """A finished run whose filter excluded everything says so honestly."""
    from pic_agentic.config import Config
    from pic_agentic.server.app import build_server

    config = Config(rcp_secret=SECRET)
    server, runtime = build_server(config, SIM)
    record = runtime.submit_service._record_for(SIM_ID, cmd_id="c1")
    record.state = SimulationState.RESULTS_READY.value
    record.active = False

    payload = (
        await server.call_tool("get_events", {"sim_id": SIM_ID, "types": ["simulation.job_running"]})
    ).structured_content
    assert payload["events"] == []
    assert "filters" in payload["note"]


async def test_get_status_redacts_secrets() -> None:
    from pic_agentic.config import Config
    from pic_agentic.server.app import build_server

    secret_token = "syt_super_secret_access_token"
    config = Config(rcp_secret=SECRET, access_token=secret_token)
    server, runtime = build_server(config, SIM)
    # Force a registry state that embeds the secret, then read it back.
    runtime.submit_service.registry["evil"] = runtime.submit_service._record_for("evil", cmd_id="c")
    record = runtime.submit_service.registry["evil"]
    record.state = f"leak:{secret_token}"

    result = await server.call_tool("get_status", {"sim_id": "evil"})
    assert secret_token not in str(result.structured_content)
    assert "[REDACTED]" in str(result.structured_content)
