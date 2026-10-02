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
    # The documented focal example is not runnable as written; the pointer warns
    # that it needs the LWFA tutorial's plasma species for a non-empty spectrum.
    assert "plasma species" in SERVER_INSTRUCTIONS
    assert "empty spectrum" in SERVER_INSTRUCTIONS
    # The list-indexed patch_path form is publicised, not only the top-level
    # sim.time_steps example.
    assert "sim.laser.0.focus_pos_si.1.component" in SERVER_INSTRUCTIONS


async def test_create_campaign_description_publicises_the_list_indexed_path() -> None:
    # The beta-4 agent found the indexed patch_path form only by reading source;
    # the tool description must name it so an agent can scan a nested list field.
    server, _runtime = build_server(Config(rcp_secret=new_secret_hex()), SIM)
    tools = {tool.name: tool for tool in await server.list_tools()}
    description = tools["create_campaign"].description
    assert "sim.laser.0.focus_pos_si.1.component" in description
    # The rule must match the patcher: the *node* decides, so a numeric segment
    # is a list index on a list but a dict key on a dict (the ``sim.bc.0``
    # boundary-condition map).  A spelling-based "numeric means list" rule would
    # contradict ``_patch_spec`` and ``test_create_campaign_reaches_a_numeric_dict_key``.
    assert "a list index on a list, or a " in description
    assert "dict key on a dict" in description
