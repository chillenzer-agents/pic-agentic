# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Run the simulation-side client: ``python -m pic_agentic.simclient``.

This is the M1 standalone client (design section 8.1): it joins the room,
waits for ``rcp.hello`` commands, executes the trivial job and acks.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from pic_agentic.auth import MasTokenStore
from pic_agentic.config import Config
from pic_agentic.protocol.simulation import SimulationOp
from pic_agentic.simclient import SimClient
from pic_agentic.simclient.client import DEFAULT_POLL_INTERVAL_S, DEFAULT_POLL_MAX_INTERVAL_S
from pic_agentic.simclient.simulation import SubmitConfig
from pic_agentic.slurm import SlurmClient
from pic_agentic.transport.matrix import MatrixTransport

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pic_agentic.simclient.follow import TrackedSim


def _make_control_fn(
    slurm: SlurmClient,
) -> Callable[[SimulationOp, TrackedSim], Awaitable[str]]:
    """Build the default M3 control translation backed by ``SlurmClient``.

    Returns:
        An async ``(op, tracked) -> str`` that signals or cancels the tracked
        job and returns SLURM's output.

    """

    async def control_fn(op: SimulationOp, tracked: TrackedSim) -> str:
        if op is SimulationOp.CANCEL:
            return await slurm.cancel_job(tracked.job_id or 0)
        signal = {
            SimulationOp.CHECKPOINT: "USR1",
            SimulationOp.STOP: "TERM",
            SimulationOp.CHECKPOINT_AND_STOP: "ALRM",
        }[op]
        return await slurm.signal_job(tracked.job_id or 0, signal)

    return control_fn


async def run() -> None:
    """Run the simulation-side client until interrupted."""
    config = Config.load()
    config.require("homeserver", "user_id", "access_token", "room_id", "rcp_secret", "message_dir")
    token_provider = None
    if config.has_refresh_chain():
        token_provider = MasTokenStore.from_config(config).access_token
    transport = MatrixTransport(
        config.homeserver,
        config.user_id,
        config.access_token,
        config.room_id,
        store_path=config.nio_store_dir or None,
        token_provider=token_provider,
    )
    sim = os.environ.get("PIC_AGENTIC_SIM", "poc")
    message_dir = Path(config.message_dir).resolve()
    # The M2 submit handler is enabled only when a setup root is configured;
    # without it the client stays M1-only (no cluster-local run dir).
    submit_config = None
    if config.sim_setup_root:
        submit_config = SubmitConfig(
            setup_root=Path(config.sim_setup_root).resolve(),
            template_dir=config.cluster_template_dir,
            preset=config.cluster_preset,
        )
    slurm = SlurmClient(bin_dir=config.slurm_bin_dir, timeout_s=config.job_wait_timeout_s + 10)
    client = SimClient(
        sim=sim,
        secret=config.rcp_secret,
        transport=transport,
        slurm=slurm,
        message_dir=message_dir,
        job_wait_timeout_s=config.job_wait_timeout_s,
        poll_interval_s=float(os.environ.get("PIC_AGENTIC_POLL_INTERVAL_S", str(DEFAULT_POLL_INTERVAL_S))),
        poll_max_interval_s=float(os.environ.get("PIC_AGENTIC_POLL_MAX_INTERVAL_S", str(DEFAULT_POLL_MAX_INTERVAL_S))),
        allowed_sender_user_id=os.environ.get("PIC_AGENTIC_ALLOWED_SENDER"),
        submit_config=submit_config,
        control_fn=_make_control_fn(slurm),
    )
    try:
        await client.serve()
    finally:
        await transport.close()


def main() -> None:
    """Console-script entry point for ``pic-agentic-simclient``."""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())


if __name__ == "__main__":
    main()
