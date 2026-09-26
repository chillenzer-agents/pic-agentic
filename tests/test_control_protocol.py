# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Protocol-level tests for the M3 control surface (contract section 8)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pic_agentic.protocol.simulation import (
    CONTROL_REQUIRES_RUNNING,
    CONTROL_SIGNAL,
    ControlParams,
    SimulationOp,
    SimulationState,
    SimulationType,
    build_control_ack,
    build_control_command,
)
from pic_agentic.rcp import Kind, SenderRole, new_secret_hex
from pic_agentic.simclient.simulation import SimulationErrorCode

SIM = "7f3a2b1c"
SECRET = new_secret_hex()


def test_control_type_values() -> None:
    assert SimulationType.CONTROL_COMMAND == "rcp.control_request"
    assert SimulationType.CONTROL_ACK == "rcp.control_ack"
    assert SimulationState.CHECKPOINT == "simulation.checkpoint"
    assert SimulationState.CANCELLED == "simulation.cancelled"


def test_op_signal_map() -> None:
    assert CONTROL_SIGNAL == {
        SimulationOp.CHECKPOINT: "USR1",
        SimulationOp.STOP: "TERM",
        SimulationOp.CHECKPOINT_AND_STOP: "ALRM",
        SimulationOp.CANCEL: None,
    }


def test_control_requires_running() -> None:
    assert set(CONTROL_REQUIRES_RUNNING) == {
        SimulationOp.CHECKPOINT,
        SimulationOp.STOP,
        SimulationOp.CHECKPOINT_AND_STOP,
    }
    assert SimulationOp.CANCEL not in CONTROL_REQUIRES_RUNNING


def test_error_codes() -> None:
    assert SimulationErrorCode.NOT_SIGNALABLE == "not_signalable"
    assert SimulationErrorCode.NOT_TERMINAL == "not_terminal"
    assert SimulationErrorCode.NO_RESULTS == "no_results"


@pytest.mark.parametrize("op", list(SimulationOp))
def test_control_params_round_trip(op: SimulationOp) -> None:
    params = ControlParams(sim_id=SIM, op=op)
    assert params.model_dump(mode="json") == {"sim_id": SIM, "op": op.value}


def test_control_params_rejects_extra() -> None:
    with pytest.raises(ValidationError):
        ControlParams(sim_id=SIM, op=SimulationOp.STOP, bogus=1)


def test_control_params_rejects_unknown_op() -> None:
    with pytest.raises(ValidationError):
        ControlParams(sim_id=SIM, op="reboot")


def test_build_control_command_shape_and_signature() -> None:
    params = ControlParams(sim_id=SIM, op=SimulationOp.STOP)
    command = build_control_command(sim=SIM, seq=7, params=params, cmd_id="ctl-1", in_reply_to="$evt")
    assert command.kind is Kind.COMMAND
    assert command.type == SimulationType.CONTROL_COMMAND
    assert command.sender_role is SenderRole.MCP_SERVER
    assert command.seq == 7
    assert command.in_reply_to == "$evt"
    assert command.payload == {"cmd_id": "ctl-1", "sim_id": SIM, "op": "stop"}
    signed = command.sign(SECRET)
    assert signed.verify(SECRET)


def test_build_control_ack_shape_and_signature() -> None:
    ack = build_control_ack(
        sim=SIM,
        seq=3,
        cmd_id="ctl-1",
        sim_id=SIM,
        op=SimulationOp.CHECKPOINT,
        ok=True,
        in_reply_to="$evt",
        job_id=4711,
        signal="USR1",
        slurm_reason="Signal USR1 sent to JobId=4711",
    )
    assert ack.kind is Kind.ACK
    assert ack.type == SimulationType.CONTROL_ACK
    assert ack.sender_role is SenderRole.SIMCLIENT
    assert ack.payload == {
        "cmd_id": "ctl-1",
        "sim_id": SIM,
        "op": "checkpoint",
        "ok": True,
        "job_id": 4711,
        "signal": "USR1",
        "slurm_reason": "Signal USR1 sent to JobId=4711",
    }
    assert ack.sign(SECRET).verify(SECRET)


def test_build_control_ack_drops_none_fields() -> None:
    ack = build_control_ack(
        sim=SIM,
        seq=1,
        cmd_id="c",
        sim_id=SIM,
        op=SimulationOp.CANCEL,
        ok=False,
        in_reply_to=None,
        error="unknown_sim",
        error_code=SimulationErrorCode.NO_RESULTS.value,
    )
    assert ack.payload == {
        "cmd_id": "c",
        "sim_id": SIM,
        "op": "cancel",
        "ok": False,
        "error": "unknown_sim",
        "error_code": "no_results",
    }
