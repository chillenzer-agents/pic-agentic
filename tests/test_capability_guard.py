# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Server/client capability-drift guards.

The live beta run had a cluster simclient older than the MCP server: a request
for a newly added op (``plugin``) was answered as an opaque rejection and the
agent had to read the source to discover a version drift.  These tests pin the
two guards that make the drift explicit:

* the reactive per-ack error naming the unsupported capability
  (``unsupported_by_client``), and
* the proactive ``hello`` handshake, where the client advertises what it can
  handle and the server refuses to send an op the client does not know.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pic_agentic.protocol.hello import build_hello_command
from pic_agentic.protocol.simulation import (
    ClientCapabilities,
    ResultOp,
    ResultParams,
    SimulationOp,
    SimulationType,
    build_result_command,
    client_capability_mismatch,
)
from pic_agentic.rcp import Kind, RcpMessage, SenderRole, new_secret_hex
from pic_agentic.server.hello import HelloService
from pic_agentic.server.simulation import SubmitService
from pic_agentic.simclient import SimClient
from pic_agentic.simclient.simulation import SimulationErrorCode, SubmitConfig
from pic_agentic.slurm import SlurmClient
from pic_agentic.transport.memory import MemoryTransport

FAKE_BIN = Path(__file__).parent / "fake_slurm"
SECRET = new_secret_hex()
SIM = "7f3a2b1c"

OLD_VERSION = "0.0.1+old"


def _old_client_capabilities() -> ClientCapabilities:
    """Capabilities of a client that predates the ``plugin`` result op.

    Returns:
        A capability set mirroring the current client minus ``plugin``.

    """
    current = ClientCapabilities.current(client_version=OLD_VERSION)
    return current.model_copy(update={"result_ops": current.result_ops - {ResultOp.PLUGIN.value}})


def _client(shared: Path, transport: MemoryTransport, **kwargs: object) -> SimClient:
    return SimClient(
        sim=SIM,
        secret=SECRET,
        transport=transport,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared,
        **kwargs,
    )


def _plugin_command() -> RcpMessage:
    params = ResultParams(sim_id=SIM, op=ResultOp.PLUGIN, reader="energy_histogram")
    return build_result_command(sim=SIM, seq=1, params=params, cmd_id="res-plugin").sign(SECRET)


def test_client_capabilities_current_lists_the_plugin_op() -> None:
    caps = ClientCapabilities.current(client_version="x")
    assert ResultOp.PLUGIN.value in caps.result_ops
    assert SimulationOp.STOP.value in caps.control_ops
    assert caps.unsupported(op=ResultOp.PLUGIN) is None


def test_capability_mismatch_message_names_the_capability_and_version() -> None:
    message = client_capability_mismatch(_old_client_capabilities(), op=ResultOp.PLUGIN.value)
    assert ResultOp.PLUGIN.value in message
    assert OLD_VERSION in message
    assert "older" in message


async def test_unknown_result_op_returns_unsupported_by_client(tmp_path) -> None:
    _mcp_t, sim_t = MemoryTransport.create_pair()
    client = _client(tmp_path, sim_t, submit_config=SubmitConfig(setup_root=tmp_path / "sims"))
    # Simulate the deployed client being older: drop the newly added op.
    client.capabilities = _old_client_capabilities()

    ack = await client.handle(_plugin_command())

    assert ack is not None
    assert ack.type == SimulationType.RESULT_ACK.value
    assert ack.payload["error_code"] == SimulationErrorCode.UNSUPPORTED_BY_CLIENT.value
    assert ResultOp.PLUGIN.value in ack.payload["error"]
    assert OLD_VERSION in ack.payload["error"]


async def test_unknown_op_string_returns_unsupported_by_client(tmp_path) -> None:
    """A current client also names an op its enum has never heard of."""
    _mcp_t, sim_t = MemoryTransport.create_pair()
    client = _client(tmp_path, sim_t, submit_config=SubmitConfig(setup_root=tmp_path / "sims"))
    command = RcpMessage(
        sim=SIM,
        kind=Kind.COMMAND,
        type=SimulationType.RESULT_COMMAND,
        seq=1,
        sender_role=SenderRole.MCP_SERVER,
        payload={"cmd_id": "res-x", "sim_id": SIM, "op": "some_future_op"},
    ).sign(SECRET)

    ack = await client.handle(command)

    assert ack is not None
    assert ack.payload["error_code"] == SimulationErrorCode.UNSUPPORTED_BY_CLIENT.value
    assert "some_future_op" in ack.payload["error"]


async def test_unknown_control_op_returns_unsupported_by_client(tmp_path) -> None:
    _mcp_t, sim_t = MemoryTransport.create_pair()
    client = _client(tmp_path, sim_t, submit_config=SubmitConfig(setup_root=tmp_path / "sims"))
    client.capabilities = ClientCapabilities.current(client_version=OLD_VERSION).model_copy(
        update={"control_ops": frozenset()},
    )
    command = RcpMessage(
        sim=SIM,
        kind=Kind.COMMAND,
        type=SimulationType.CONTROL_COMMAND,
        seq=1,
        sender_role=SenderRole.MCP_SERVER,
        payload={"cmd_id": "ctl-1", "sim_id": SIM, "op": SimulationOp.STOP.value},
    ).sign(SECRET)

    ack = await client.handle(command)

    assert ack is not None
    assert ack.type == SimulationType.CONTROL_ACK.value
    assert ack.payload["error_code"] == SimulationErrorCode.UNSUPPORTED_BY_CLIENT.value
    assert SimulationOp.STOP.value in ack.payload["error"]


async def test_unknown_request_type_returns_unsupported_by_client(tmp_path) -> None:
    _mcp_t, sim_t = MemoryTransport.create_pair()
    client = _client(tmp_path, sim_t, submit_config=None)
    unknown = RcpMessage(
        sim=SIM,
        kind=Kind.COMMAND,
        type="rcp.analysis_request",
        seq=1,
        sender_role=SenderRole.MCP_SERVER,
        payload={"cmd_id": "x-1", "sim_id": SIM},
    ).sign(SECRET)

    ack = await client.handle(unknown)

    assert ack is not None
    assert ack.payload["error_code"] == SimulationErrorCode.UNSUPPORTED_BY_CLIENT.value
    assert "rcp.analysis_request" in ack.payload["error"]


async def test_hello_ack_advertises_capabilities(tmp_path) -> None:
    mcp_t, sim_t = MemoryTransport.create_pair()
    service = HelloService(sim=SIM, secret=SECRET, message_dir=tmp_path, ack_timeout_s=5.0)
    client = _client(tmp_path, sim_t, poll_interval_s=0.02)

    async def pump() -> None:
        async for msg in mcp_t.receive():
            service.on_message(msg)

    pump_task = asyncio.create_task(pump())
    serve_task = asyncio.create_task(client.serve())
    try:
        outcome = await service.hello(mcp_t.send, "hi")
    finally:
        serve_task.cancel()
        pump_task.cancel()
        await sim_t.close()
        await mcp_t.close()

    assert outcome.ok, outcome.error
    assert outcome.capabilities is not None
    assert ResultOp.PLUGIN.value in outcome.capabilities.result_ops
    # The service caches the handshake for the proactive guard.
    assert service.capabilities is not None
    assert ResultOp.PLUGIN.value in service.capabilities.result_ops


async def test_proactive_guard_blocks_unsupported_result_op(tmp_path) -> None:
    mcp_t, sim_t = MemoryTransport.create_pair()
    service = SubmitService(sim=SIM, secret=SECRET, ack_timeout_s=1.0)
    service.set_client_capabilities(_old_client_capabilities())

    payload = await service.fetch_result(
        mcp_t.send,
        ResultParams(sim_id=SIM, op=ResultOp.PLUGIN, reader="energy_histogram"),
    )

    assert payload["error_code"] == "unsupported_by_client"
    assert ResultOp.PLUGIN.value in payload["error"]
    assert OLD_VERSION in payload["error"]
    # The command never left the server: the client's inbox is empty.
    assert sim_t.inbox.empty()


async def test_unknown_client_does_not_block_request(tmp_path) -> None:
    """A client that never advertised capabilities is sent the request as before."""
    mcp_t, _sim_t = MemoryTransport.create_pair()
    service = SubmitService(sim=SIM, secret=SECRET, ack_timeout_s=0.1)
    assert service.client_capabilities is None

    payload = await service.fetch_result(
        mcp_t.send,
        ResultParams(sim_id=SIM, op=ResultOp.PLUGIN, reader="energy_histogram"),
    )

    # No proactive rejection; the pull simply times out (client is not serving).
    assert payload.get("error_code") != "unsupported_by_client"
    assert payload.get("error") == "timeout"


def test_hello_command_signature_is_unaffected_by_capabilities(tmp_path) -> None:
    """The handshake rides only in the ack; the command shape is unchanged."""
    command = build_hello_command(sim=SIM, seq=1, message_path=str(tmp_path / "m.txt"))
    assert "capabilities" not in command.payload


@pytest.mark.parametrize("op", [ResultOp.DESCRIBE, ResultOp.PLUGIN])
def test_supported_ops_are_not_flagged(op: ResultOp) -> None:
    caps = ClientCapabilities.current()
    assert caps.unsupported(op=op) is None
