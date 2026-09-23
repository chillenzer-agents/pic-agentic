# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Run the MCP server over stdio: ``python -m pic_agentic.server``."""

from __future__ import annotations

import logging
import os

from pic_agentic.config import Config
from pic_agentic.server.app import build_server


def main() -> None:
    """Console-script entry point for ``pic-agentic-mcp``."""
    logging.basicConfig(level=logging.INFO)
    config = Config.load()
    sim = os.environ.get("PIC_AGENTIC_SIM", "poc")
    server, _runtime = build_server(config, sim)
    server.run("stdio")


if __name__ == "__main__":
    main()
