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
    # H5: the add_agenda_leaf escape hatch for multi-node/varying-node studies
    # and its by-reference spec_path form are named.
    assert "add_agenda_leaf" in SERVER_INSTRUCTIONS
    assert "spec_path=..." in SERVER_INSTRUCTIONS
    # L5: the "which tool when" front-door line is present.
    assert "Which tool" in SERVER_INSTRUCTIONS
    # H8: the documented focal example is empty for timing/geometry reasons, not
    # merely a missing plasma species; the corrected caveat says both.
    assert "plasma species" in SERVER_INSTRUCTIONS
    assert "never reaches the gas" in SERVER_INSTRUCTIONS
    assert "PULSE_INIT=15" in SERVER_INSTRUCTIONS
    # H9: the pinned API is authoritative over readthedocs; the mismatched names
    # and the version-matched classes are named.
    assert "installed" in SERVER_INSTRUCTIONS
    assert "picongpu" in SERVER_INSTRUCTIONS
    assert "FieldDiagnostic" in SERVER_INSTRUCTIONS
    assert "picmi.Cartesian3DGrid" in SERVER_INSTRUCTIONS
    assert "picmi.UniformDistribution" in SERVER_INSTRUCTIONS
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


async def test_add_agenda_leaf_description_publicises_the_escape_hatch() -> None:
    # H5: a study that must vary more than one spec node cannot use create_campaign's
    # single patch_path; add_agenda_leaf is the documented escape hatch and takes a
    # whole spec by reference.
    server, _runtime = build_server(Config(rcp_secret=new_secret_hex()), SIM)
    tools = {tool.name: tool for tool in await server.list_tools()}
    description = tools["add_agenda_leaf"].description
    assert "escape hatch" in description
    assert "spec_path" in description
    assert "grid cells and time_steps" in description
