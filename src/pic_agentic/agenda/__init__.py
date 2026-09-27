# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Agenda declaration: serialisable, expanding simulation groups.

The model layer (:mod:`pic_agentic.agenda.model`) and the budget layer
(:mod:`pic_agentic.agenda.budget`) are extraction-ready (stdlib + pydantic
only); :mod:`pic_agentic.agenda.cwl` emits CWL and needs PyYAML.
"""

from pic_agentic.agenda.budget import (
    Budget,
    BudgetExceededError,
    BudgetUsage,
    ResourceRequest,
    check_admission,
)
from pic_agentic.agenda.campaign import Callback, Campaign, CampaignState
from pic_agentic.agenda.cwl import dump_cwl_workflow, to_cwl_workflow
from pic_agentic.agenda.engine import AgendaEngine, EnginePolicy, TickResult
from pic_agentic.agenda.model import AgendaGroup, AgendaSim, AgendaSweep, validate_entry_name
from pic_agentic.agenda.planner import PlanStep, account, apply_states, next_actions
from pic_agentic.agenda.provenance import RO_CRATE_METADATA_FILE, campaign_rocrate
from pic_agentic.agenda.store import DEFAULT_AGENDA_FILE, DEFAULT_CAMPAIGN_FILE, AgendaStore

__all__ = [
    "DEFAULT_AGENDA_FILE",
    "DEFAULT_CAMPAIGN_FILE",
    "RO_CRATE_METADATA_FILE",
    "AgendaEngine",
    "AgendaGroup",
    "AgendaSim",
    "AgendaStore",
    "AgendaSweep",
    "Budget",
    "BudgetExceededError",
    "BudgetUsage",
    "Callback",
    "Campaign",
    "CampaignState",
    "EnginePolicy",
    "PlanStep",
    "ResourceRequest",
    "TickResult",
    "account",
    "apply_states",
    "campaign_rocrate",
    "check_admission",
    "dump_cwl_workflow",
    "next_actions",
    "to_cwl_workflow",
    "validate_entry_name",
]
