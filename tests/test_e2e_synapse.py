# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Full M1 acceptance E2E: MCP stdio client -> Synapse -> simclient -> fake SLURM.

Requires a local Synapse with the two bot accounts and room from
``scripts/dev_synapse.py``.  Skipped automatically when unavailable so the
offline suite stays green.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from pic_agentic.rcp import new_secret_hex

BOTS_FILE = Path("/run/terok/work/bots.json")
FAKE_BIN = Path(__file__).parent / "fake_slurm"
PROVISION_SCRIPT = Path(__file__).parent.parent / "scripts" / "dev_synapse.py"

pytestmark = pytest.mark.skipif(not BOTS_FILE.exists(), reason="local Synapse not provisioned")


@pytest.fixture
def bots(tmp_path):
    """Provision a fresh room with the two bot tokens for this test run."""
    out = tmp_path / "bots.json"
    proc = subprocess.run(
        [sys.executable, str(PROVISION_SCRIPT), "--new-room", "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not out.exists():
        pytest.skip(f"could not provision local Synapse: {proc.stderr or proc.stdout}")
    return json.loads(out.read_text())


async def _reachable(homeserver: str) -> bool:
    """Return whether the local homeserver answers ``/_matrix/client/versions``."""
    url = homeserver + "/_matrix/client/versions"
    try:
        # The scheme is a test-fixture localhost URL, never user input.
        with urllib.request.urlopen(url, timeout=5) as response:  # ruff: ignore[suspicious-url-open-usage]
            return response.status == 200
    except Exception:
        return False


def test_hello_e2e_through_stdio_and_synapse(bots, tmp_path, monkeypatch) -> None:
    if not asyncio.run(_reachable(bots["hs"])):
        pytest.skip("Synapse not reachable")

    shared = tmp_path / "shared"
    shared.mkdir()
    state = tmp_path / "fake-slurm"
    secret = new_secret_hex()

    env = dict(os.environ)
    env.update(
        {
            "PIC_AGENTIC_HOMESERVER": bots["hs"],
            "PIC_AGENTIC_ROOM_ID": bots["room_id"],
            # Keep the run hermetic: never read a developer's real 0600 config
            # (which may carry a live MAS refresh chain).
            "PIC_AGENTIC_CONFIG": str(tmp_path / "no-such-config.toml"),
            "PIC_AGENTIC_RCP_SECRET": secret,
            "PIC_AGENTIC_MESSAGE_DIR": str(shared),
            "PIC_AGENTIC_SIM": "e2esim",
            "FAKE_SLURM_STATE": str(state),
            "PIC_AGENTIC_POLL_INTERVAL_S": "0.2",
            "PIC_AGENTIC_JOB_WAIT_TIMEOUT_S": "30",
            "PIC_AGENTIC_ACK_TIMEOUT_S": "45",
            "PIC_AGENTIC_NIO_STORE_DIR": str(tmp_path / "nio-mcp"),
        },
    )

    async def scenario():
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        from pic_agentic.simclient import SimClient
        from pic_agentic.slurm import SlurmClient
        from pic_agentic.transport.matrix import MatrixTransport

        # Simulation-side client (as a separate Matrix client) in-process.
        sim_transport = MatrixTransport(
            bots["hs"],
            "@simclient:localhost",
            bots["tokens"]["simclient"],
            bots["room_id"],
            sync_timeout_ms=500,
            store_path=str(tmp_path / "nio-sim"),
        )
        # Drain any pre-existing timeline before serving.
        await sim_transport.backfill()
        simclient = SimClient(
            sim="e2esim",
            secret=secret,
            transport=sim_transport,
            slurm=SlurmClient(bin_dir=str(FAKE_BIN), timeout_s=40),
            message_dir=shared,
            job_wait_timeout_s=30,
            poll_interval_s=0.2,
        )
        serve = asyncio.create_task(simclient.serve())

        # MCP server process, driven over stdio like a real LLM host.
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "pic_agentic.server"],
            env={
                **env,
                "PIC_AGENTIC_USER_ID": "@mcpserver:localhost",
                "PIC_AGENTIC_ACCESS_TOKEN": bots["tokens"]["mcpserver"],
            },
        )
        try:
            async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert any(tool.name == "hello" for tool in tools.tools)
                return await session.call_tool("hello", {"message": "Hello World from E2E"})
        finally:
            serve.cancel()
            await sim_transport.close()

    result = asyncio.run(scenario())
    assert result.is_error is False
    assert result.structured_content is not None
    sc = result.structured_content
    assert sc["ok"] is True, sc
    assert isinstance(sc["job_id"], int)
    assert sc["job_id"] > 0
    assert "Hello World from E2E" in (sc["cluster_output"] or "")
