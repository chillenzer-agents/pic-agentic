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

from pic_agentic.auth import MasTokenStore
from pic_agentic.config import Config
from pic_agentic.simclient import SimClient
from pic_agentic.slurm import SlurmClient
from pic_agentic.transport.matrix import MatrixTransport


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
    client = SimClient(
        sim=sim,
        secret=config.rcp_secret,
        transport=transport,
        slurm=SlurmClient(bin_dir=config.slurm_bin_dir, timeout_s=config.job_wait_timeout_s + 10),
        message_dir=Path(config.message_dir).resolve(),
        job_wait_timeout_s=config.job_wait_timeout_s,
        poll_interval_s=float(os.environ.get("PIC_AGENTIC_POLL_INTERVAL_S", "5")),
        allowed_sender_user_id=os.environ.get("PIC_AGENTIC_ALLOWED_SENDER"),
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
