"""Run the simulation-side client: ``python -m pic_agentic.simclient``.

This is the M1 standalone client (design section 8.1): it joins the room,
waits for ``rcp.hello`` commands, executes the trivial job and acks.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from pic_agentic.config import Config
from pic_agentic.simclient import SimClient
from pic_agentic.slurm import SlurmClient
from pic_agentic.transport.matrix import MatrixTransport


async def run() -> None:
    config = Config.load()
    config.require("homeserver", "user_id", "access_token", "room_id", "rcp_secret", "message_dir")
    transport = MatrixTransport(
        config.homeserver,
        config.user_id,
        config.access_token,
        config.room_id,
        store_path=config.nio_store_dir or None,
    )
    sim = os.environ.get("PIC_AGENTIC_SIM", "poc")
    client = SimClient(
        sim=sim,
        secret=config.rcp_secret,
        transport=transport,
        slurm=SlurmClient(bin_dir=config.slurm_bin_dir, timeout_s=config.job_wait_timeout_s + 10),
        message_dir=str(Path(config.message_dir).resolve()),
        job_wait_timeout_s=config.job_wait_timeout_s,
        poll_interval_s=float(os.environ.get("PIC_AGENTIC_POLL_INTERVAL_S", "5")),
        allowed_sender_user_id=os.environ.get("PIC_AGENTIC_ALLOWED_SENDER"),
    )
    print(f"simclient up: sim={sim} room={config.room_id}")
    try:
        await client.serve()
    finally:
        await transport.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())


if __name__ == "__main__":
    main()
