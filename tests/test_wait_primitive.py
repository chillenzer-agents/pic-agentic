# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Offline tests for the bounded ``wait_for_simulation`` primitive (H10).

The wait is event-driven over the registry projection, so these tests build
real signed events and drive them through :meth:`SubmitService.on_message`
while a ``wait_for_state`` call is in flight -- no cluster, transport, sleep
loop in the code under test, or homeserver is involved.
"""

from __future__ import annotations

import asyncio

import pytest

from pic_agentic.config import Config
from pic_agentic.protocol.simulation import (
    SimulationState,
    build_submit_event,
)
from pic_agentic.rcp import RcpMessage, new_secret_hex
from pic_agentic.server.app import build_server
from pic_agentic.server.simulation import (
    DEFAULT_WAIT_TIMEOUT_S,
    MAX_WAIT_TIMEOUT_S,
    MIN_WAIT_TIMEOUT_S,
    WAIT_CLIENT_TIMEOUT_SKEW_S,
    SubmitService,
    WaitExceedsClientTimeoutError,
)

SECRET = new_secret_hex()
SIM = "7f3a2b1c"
SIM_ID = "abcd1234"
CMD_ID = "cmd-1"
JOB_ID = 4242


def _service(*, ack_timeout_s: float = 0.05) -> SubmitService:
    return SubmitService(sim=SIM, secret=SECRET, ack_timeout_s=ack_timeout_s)


def _event(
    state: SimulationState,
    *,
    sim_id: str = SIM_ID,
    cmd_id: str = CMD_ID,
    seq: int = 1,
    **fields: object,
) -> RcpMessage:
    return build_submit_event(sim=SIM, seq=seq, cmd_id=cmd_id, sim_id=sim_id, state=state, **fields).sign(SECRET)


def _seed_running(service: SubmitService) -> None:
    service.on_message(_event(SimulationState.SUBMITTED, seq=1, job_id=JOB_ID))
    service.on_message(_event(SimulationState.JOB_RUNNING, seq=2, job_id=JOB_ID, slurm_state="RUNNING"))


async def test_wait_returns_on_a_terminal_event() -> None:
    service = _service()
    _seed_running(service)

    async def projector() -> None:
        await asyncio.sleep(0.02)
        service.on_message(_event(SimulationState.RESULTS_READY, seq=3, job_id=JOB_ID, results_linked=True))

    task = asyncio.create_task(projector())
    outcome = await service.wait_for_state(SIM_ID, timeout_s=5.0, poll_interval_s=10.0)
    await task

    assert outcome.matched is True
    assert outcome.timed_out is False
    assert outcome.state == SimulationState.RESULTS_READY.value
    assert outcome.ok is True
    assert SimulationState.RESULTS_READY.value in outcome.target_states
    assert outcome.target_states[0] == SimulationState.RESULTS_READY.value  # sorted
    assert outcome.last_status["state"] == SimulationState.RESULTS_READY.value
    assert outcome.note is None
    # The event-driven wait woke on the push, not on its 10 s poll ceiling.
    assert outcome.waited_s < 1.0
    # The condensed history is returned for the caller.
    assert [entry["state"] for entry in outcome.events][-1] == SimulationState.RESULTS_READY.value


async def test_wait_times_out_with_the_last_status_not_an_error() -> None:
    service = _service()
    _seed_running(service)

    outcome = await service.wait_for_state(SIM_ID, timeout_s=0.2, poll_interval_s=0.05)

    assert outcome.timed_out is True
    assert outcome.ok is True  # a timeout is data, not an error
    assert outcome.matched is False
    assert outcome.state == SimulationState.JOB_RUNNING.value
    assert outcome.last_status["state"] == SimulationState.JOB_RUNNING.value
    assert outcome.last_status["job_id"] == JOB_ID
    assert outcome.last_status["phase"] == "running"
    assert outcome.note is not None
    assert "not terminal" in outcome.note
    assert outcome.waited_s >= 0.1


async def test_wait_for_a_non_terminal_target_state() -> None:
    """A caller may ask for an early state (e.g. job_running) explicitly."""
    service = _service()
    service.on_message(_event(SimulationState.SUBMITTED, seq=1, job_id=JOB_ID))

    async def projector() -> None:
        await asyncio.sleep(0.02)
        service.on_message(_event(SimulationState.JOB_RUNNING, seq=2, job_id=JOB_ID, slurm_state="RUNNING"))

    task = asyncio.create_task(projector())
    outcome = await service.wait_for_state(
        SIM_ID,
        target_states=[SimulationState.JOB_RUNNING.value],
        timeout_s=5.0,
        poll_interval_s=10.0,
    )
    await task

    assert outcome.matched is True
    assert outcome.state == SimulationState.JOB_RUNNING.value
    assert outcome.target_states == [SimulationState.JOB_RUNNING.value]


async def test_wait_wakes_every_concurrent_waiter_for_a_sim() -> None:
    """Two overlapping waits on one sim are both woken by the terminal push.

    Guards M2: a single shared ``asyncio.Event`` popped by the first returning
    waiter left the second to observe the terminal state only on its poll
    ceiling (or the deadline) instead of on the push.
    """
    service = _service()
    service.on_message(_event(SimulationState.SUBMITTED, seq=1, job_id=JOB_ID))

    # Waiter A resolves on job_running; waiter B watches the terminal set with a
    # poll ceiling far larger than the test, so only a real wake returns it.
    # Both must be parked before the transitions are projected.
    first = asyncio.create_task(
        service.wait_for_state(
            SIM_ID,
            target_states=[SimulationState.JOB_RUNNING.value],
            timeout_s=5.0,
            poll_interval_s=100.0,
        )
    )
    await asyncio.sleep(0)  # let A register its wakeup
    second = asyncio.create_task(service.wait_for_state(SIM_ID, timeout_s=5.0, poll_interval_s=100.0))
    await asyncio.sleep(0)  # let B register its wakeup

    # A returns on this push and (before the fix) popped the shared event,
    # stranding B; B must still be woken by the later terminal push.
    service.on_message(_event(SimulationState.JOB_RUNNING, seq=2, job_id=JOB_ID, slurm_state="RUNNING"))
    outcome_a = await asyncio.wait_for(first, timeout=1.0)
    service.on_message(_event(SimulationState.RESULTS_READY, seq=3, job_id=JOB_ID, results_linked=True))
    outcome_b = await asyncio.wait_for(second, timeout=1.0)

    assert outcome_a.matched is True
    assert outcome_a.state == SimulationState.JOB_RUNNING.value
    assert outcome_b.matched is True
    assert outcome_b.state == SimulationState.RESULTS_READY.value
    assert outcome_b.waited_s < 1.0
    # Both waiters are deregistered once they return (no stranded entry).
    assert SIM_ID not in service._waiters


async def test_wait_on_a_terminal_run_with_a_non_terminal_target_returns_at_once() -> None:
    """M3: a terminal run can never reach another target -- do not burn the deadline."""
    service = _service()
    service.on_message(
        _event(SimulationState.FAILED, seq=1, job_id=None, error="boom", error_code="failed"),
    )

    outcome = await service.wait_for_state(
        SIM_ID,
        target_states=[SimulationState.RESULTS_READY.value],
        timeout_s=5.0,
        poll_interval_s=100.0,
    )

    assert outcome.matched is False
    assert outcome.timed_out is False
    assert outcome.state == SimulationState.FAILED.value
    assert outcome.waited_s < 1.0
    assert outcome.note is not None
    assert "terminal state" in outcome.note
    assert "not one of the requested targets" in outcome.note
    # The misleading "not terminal yet / call again" wording must not appear.
    assert "not terminal yet" not in outcome.note


async def test_wait_catches_a_terminal_event_projected_before_it_registers() -> None:
    """N1: the wake-before-await race -- an event landing at the first await.

    A projector task that yields exactly once projects the terminal event at the
    first suspension point of the wait (after its pre-loop state check and
    ``wakeup.clear()``, before it has parked).  The loop re-reads the registry
    after every wake, so the outcome must match even though the event did not
    set a parked waiter's event.
    """
    service = _service()
    service.on_message(_event(SimulationState.ACCEPTED, seq=1, job_id=None))

    async def projector() -> None:
        await asyncio.sleep(0)
        service.on_message(_event(SimulationState.RESULTS_READY, seq=2, job_id=JOB_ID, results_linked=True))

    task = asyncio.create_task(projector())
    outcome = await service.wait_for_state(SIM_ID, timeout_s=5.0, poll_interval_s=100.0)
    await task

    assert outcome.matched is True
    assert outcome.state == SimulationState.RESULTS_READY.value
    assert outcome.waited_s < 1.0


async def test_wait_terminal_alias_expands_to_the_terminal_set() -> None:
    service = _service()
    service.on_message(_event(SimulationState.FAILED, seq=1, job_id=None, error="boom", error_code="failed"))

    outcome = await service.wait_for_state(SIM_ID, target_states=["terminal"], timeout_s=1.0)

    assert outcome.matched is True
    assert outcome.state == SimulationState.FAILED.value
    assert SimulationState.FAILED.value in outcome.target_states
    assert SimulationState.RESULTS_READY.value in outcome.target_states


async def test_wait_reports_the_build_phase_without_inventing_a_state() -> None:
    """No ``job_id`` yet: the note says building, and no state is invented."""
    service = _service()
    # ``accepted`` with no scheduler job id -- the compile/prepare window.
    service.on_message(_event(SimulationState.ACCEPTED, seq=1, job_id=None))

    outcome = await service.wait_for_state(SIM_ID, timeout_s=0.15, poll_interval_s=0.05)

    assert outcome.timed_out is True
    # The state stays exactly what the event stream reported; only the phase is
    # a derived label.
    assert outcome.state == SimulationState.ACCEPTED.value
    assert outcome.last_status["state"] == SimulationState.ACCEPTED.value
    assert outcome.last_status["job_id"] is None
    assert outcome.last_status["phase"] == "building"
    assert outcome.note is not None
    assert "build/queue phase" in outcome.note


async def test_wait_in_budget_blocks_and_returns_on_terminal() -> None:
    """A wait that fits the known client budget still blocks event-driven."""
    service = SubmitService(sim=SIM, secret=SECRET, ack_timeout_s=0.05, client_timeout_s=300.0)
    _seed_running(service)

    async def projector() -> None:
        await asyncio.sleep(0.02)
        service.on_message(_event(SimulationState.RESULTS_READY, seq=3, job_id=JOB_ID, results_linked=True))

    task = asyncio.create_task(projector())
    outcome = await service.wait_for_state(SIM_ID, timeout_s=200.0, poll_interval_s=100.0)
    await task

    assert outcome.matched is True
    assert outcome.waited_s < 1.0


async def test_wait_over_budget_is_refused_before_blocking() -> None:
    """D1: timeout_s beyond the client budget+skew raises a clear error, no block."""
    service = SubmitService(sim=SIM, secret=SECRET, ack_timeout_s=0.05, client_timeout_s=120.0)
    _seed_running(service)

    with pytest.raises(WaitExceedsClientTimeoutError) as excinfo:
        await asyncio.wait_for(service.wait_for_state(SIM_ID, timeout_s=600.0), timeout=1.0)

    err = excinfo.value
    assert err.requested_s == pytest.approx(600.0)
    assert err.client_timeout_s == pytest.approx(120.0)
    assert err.limit_s == pytest.approx(120.0 - WAIT_CLIENT_TIMEOUT_SKEW_S)
    assert "wait_exceeds_client_timeout" in str(err)
    assert "PIC_AGENTIC_MCP_TIMEOUT_MS" in str(err)


async def test_wait_unknown_budget_validates_only() -> None:
    """No budget -> H10 behaviour: bounded only by [MIN, MAX], never refused."""
    service = SubmitService(sim=SIM, secret=SECRET, ack_timeout_s=0.05)  # client_timeout_s=None
    # Terminal already, so the 600 s request is accepted and returns at once;
    # with a known 120 s budget the same request would be refused up front.
    service.on_message(_event(SimulationState.FAILED, seq=1, job_id=None, error="boom", error_code="failed"))

    outcome = await service.wait_for_state(SIM_ID, timeout_s=600.0)
    assert outcome.matched is True
    assert outcome.ok is True


async def test_wait_exactly_at_budget_minus_skew_is_allowed() -> None:
    """The skew is a strict margin: budget - skew is still accepted."""
    service = SubmitService(sim=SIM, secret=SECRET, ack_timeout_s=0.05, client_timeout_s=120.0)
    service.on_message(_event(SimulationState.FAILED, seq=1, job_id=None, error="boom", error_code="failed"))

    outcome = await service.wait_for_state(SIM_ID, target_states=["terminal"], timeout_s=115.0)
    assert outcome.matched is True


async def test_wait_unknown_simulation_raises_keyerror() -> None:
    service = _service()
    with pytest.raises(KeyError, match="unknown simulation"):
        await service.wait_for_state("nope", timeout_s=1.0)


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, MIN_WAIT_TIMEOUT_S / 2, MAX_WAIT_TIMEOUT_S + 1])
async def test_wait_validates_timeout_bounds(timeout_s: float) -> None:
    service = _service()
    service.on_message(_event(SimulationState.ACCEPTED, seq=1, job_id=None))
    with pytest.raises(ValueError, match="timeout_s must be between"):
        await service.wait_for_state(SIM_ID, timeout_s=timeout_s)


async def test_wait_rejects_an_unknown_target_state() -> None:
    service = _service()
    service.on_message(_event(SimulationState.ACCEPTED, seq=1, job_id=None))
    with pytest.raises(ValueError, match="unknown wait target state"):
        await service.wait_for_state(SIM_ID, target_states=["bogus"], timeout_s=1.0)


async def test_wait_default_timeout_accommodates_a_long_build() -> None:
    """The default bound must cover the documented 15-20 min compile."""
    assert DEFAULT_WAIT_TIMEOUT_S >= 20 * 60
    assert MAX_WAIT_TIMEOUT_S >= DEFAULT_WAIT_TIMEOUT_S


async def test_wait_for_simulation_tool_returns_on_terminal() -> None:
    config = Config(rcp_secret=SECRET)
    server, runtime = build_server(config, SIM)
    service = runtime.submit_service
    _seed_running(service)

    async def projector() -> None:
        await asyncio.sleep(0.02)
        service.on_message(_event(SimulationState.RESULTS_READY, seq=3, job_id=JOB_ID, results_linked=True))

    task = asyncio.create_task(projector())
    result = await server.call_tool("wait_for_simulation", {"sim_id": SIM_ID, "timeout_s": 5.0})
    await task

    payload = result.structured_content
    assert payload["matched"] is True
    assert payload["timed_out"] is False
    assert payload["state"] == SimulationState.RESULTS_READY.value
    assert payload["last_status"]["state"] == SimulationState.RESULTS_READY.value


async def test_wait_for_simulation_tool_timeout_is_data() -> None:
    config = Config(rcp_secret=SECRET)
    server, runtime = build_server(config, SIM)
    runtime.submit_service.on_message(_event(SimulationState.JOB_RUNNING, seq=1, job_id=JOB_ID, slurm_state="RUNNING"))

    result = await server.call_tool(
        "wait_for_simulation",
        {"sim_id": SIM_ID, "timeout_s": 0.2, "poll_interval_s": 0.05},
    )

    payload = result.structured_content
    assert payload["timed_out"] is True
    assert payload["last_status"]["state"] == SimulationState.JOB_RUNNING.value


async def test_wait_for_simulation_tool_bad_arguments_are_soft_errors() -> None:
    config = Config(rcp_secret=SECRET)
    server, runtime = build_server(config, SIM)
    runtime.submit_service.on_message(_event(SimulationState.ACCEPTED, seq=1, job_id=None))

    unknown = (await server.call_tool("wait_for_simulation", {"sim_id": "ghost", "timeout_s": 1.0})).structured_content
    assert unknown["ok"] is False
    assert "unknown simulation" in unknown["error"]

    bad_timeout = (
        await server.call_tool("wait_for_simulation", {"sim_id": SIM_ID, "timeout_s": 10_000.0})
    ).structured_content
    assert bad_timeout["ok"] is False
    assert "timeout_s must be between" in bad_timeout["error"]

    bad_target = (
        await server.call_tool("wait_for_simulation", {"sim_id": SIM_ID, "target_states": ["bogus"], "timeout_s": 1.0})
    ).structured_content
    assert bad_target["ok"] is False
    assert "unknown wait target state" in bad_target["error"]


async def test_wait_for_simulation_tool_over_budget_is_a_clear_soft_error() -> None:
    """D1: the tool returns wait_exceeds_client_timeout, not a -32001/clamp."""
    config = Config(rcp_secret=SECRET, mcp_timeout_ms=120_000)
    server, runtime = build_server(config, SIM)
    runtime.submit_service.on_message(_event(SimulationState.JOB_RUNNING, seq=1, job_id=JOB_ID, slurm_state="RUNNING"))

    payload = (await server.call_tool("wait_for_simulation", {"sim_id": SIM_ID, "timeout_s": 600.0})).structured_content

    assert payload["ok"] is False
    assert payload["error"] == "wait_exceeds_client_timeout"
    assert payload["client_timeout_s"] == pytest.approx(120.0)
    assert payload["requested_timeout_s"] == pytest.approx(600.0)
    assert payload["max_timeout_s"] == pytest.approx(115.0)
    assert "PIC_AGENTIC_MCP_TIMEOUT_MS" in payload["detail"]
    # The wait did not block and did not clamp: it returned the error up front.
    assert "timed_out" not in payload


async def test_wait_for_simulation_tool_default_call_is_not_refused_at_shipped_budget() -> None:
    """D1: the no-``timeout_s`` default call must succeed under the shipped budget.

    The shipped install stamps a client budget above the server's
    ``MAX_WAIT_TIMEOUT_S``, so the documented default (1800 s) is inside it and
    the plain "wait for my sim" call is not pre-empted with
    ``wait_exceeds_client_timeout``.  The simulation is already terminal, so the
    default call returns at once if it is accepted at all.
    """
    config = Config(rcp_secret=SECRET, mcp_timeout_ms=3_900_000)
    server, runtime = build_server(config, SIM)
    runtime.submit_service.on_message(
        _event(SimulationState.FAILED, seq=1, job_id=None, error="boom", error_code="failed"),
    )

    payload = (await server.call_tool("wait_for_simulation", {"sim_id": SIM_ID})).structured_content

    assert payload["ok"] is True, payload
    assert payload["matched"] is True
    assert "error" not in payload
    assert config.mcp_client_timeout_s() - WAIT_CLIENT_TIMEOUT_SKEW_S >= DEFAULT_WAIT_TIMEOUT_S


async def test_wait_for_simulation_tool_in_budget_still_returns_terminal() -> None:
    """A long wait within a generous budget remains event-driven data."""
    config = Config(rcp_secret=SECRET, mcp_timeout_ms=1_800_000)
    server, runtime = build_server(config, SIM)
    service = runtime.submit_service
    _seed_running(service)

    async def projector() -> None:
        await asyncio.sleep(0.02)
        service.on_message(_event(SimulationState.RESULTS_READY, seq=3, job_id=JOB_ID, results_linked=True))

    task = asyncio.create_task(projector())
    payload = (await server.call_tool("wait_for_simulation", {"sim_id": SIM_ID, "timeout_s": 600.0})).structured_content
    await task

    assert payload["matched"] is True
    assert payload["timed_out"] is False


async def test_wait_for_simulation_tool_unknown_budget_preserves_old_behaviour() -> None:
    """No configured budget -> the 10000 s bound still validates as before."""
    config = Config(rcp_secret=SECRET)  # mcp_timeout_ms unset
    server, runtime = build_server(config, SIM)
    # Already terminal: the 600 s request is accepted and returns at once,
    # proving it was not refused for exceeding a (nonexistent) budget.
    runtime.submit_service.on_message(
        _event(SimulationState.FAILED, seq=1, job_id=None, error="boom", error_code="failed"),
    )

    payload_no_budget = (
        await server.call_tool("wait_for_simulation", {"sim_id": SIM_ID, "timeout_s": 600.0})
    ).structured_content
    assert payload_no_budget["ok"] is True
    assert payload_no_budget["matched"] is True

    # The [0.1, 3600] bound still applies with no budget.
    payload_over_max = (
        await server.call_tool("wait_for_simulation", {"sim_id": SIM_ID, "timeout_s": 10_000.0})
    ).structured_content
    assert payload_over_max["ok"] is False
    assert "timeout_s must be between" in payload_over_max["error"]
