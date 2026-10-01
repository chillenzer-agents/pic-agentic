# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""The seeded MCP instructions must point an unfamiliar agent at PIConGPU docs."""

from __future__ import annotations

from pic_agentic.config import Config
from pic_agentic.rcp import new_secret_hex
from pic_agentic.server.app import SERVER_INSTRUCTIONS, build_server

SIM = "7f3a2b1c"


async def test_instructions_are_seeded_on_the_server() -> None:
    server, _runtime = build_server(Config(rcp_secret=new_secret_hex()), SIM)
    assert server.instructions == SERVER_INSTRUCTIONS


def test_instructions_point_at_the_picongpu_documentation_and_examples() -> None:
    # The onboarding contract: an agent with no PICMI knowledge can find how to
    # define and scan simulations, and where the worked examples are.
    assert "defining_simulation" in SERVER_INSTRUCTIONS
    assert "readthedocs" in SERVER_INSTRUCTIONS
    assert "lib/python/examples" in SERVER_INSTRUCTIONS
    # The campaign-entry tools the agent needs are named.
    assert "build_spec" in SERVER_INSTRUCTIONS
    assert "create_campaign" in SERVER_INSTRUCTIONS
    # The reset path for starting a fresh campaign is named too.
    assert "delete_campaign" in SERVER_INSTRUCTIONS
