# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Remote control protocol (RCP) carried in Matrix room messages.

The protocol shape (envelope, simulation tag, event taxonomy) is deliberately
independent of the transport so MCP server and simulation-side client can
evolve separately.
"""

from pic_agentic.rcp.crypto import canonical_bytes, new_cmd_id, new_secret_hex, sign, verify
from pic_agentic.rcp.envelope import (
    RCP_NAMESPACE,
    SIGNED_FIELDS,
    VERSION,
    Kind,
    RcpMessage,
    SenderRole,
    now_ts,
)
from pic_agentic.rcp.state import DedupStore, SequenceState

__all__ = [
    "RCP_NAMESPACE",
    "SIGNED_FIELDS",
    "VERSION",
    "DedupStore",
    "Kind",
    "RcpMessage",
    "SenderRole",
    "SequenceState",
    "canonical_bytes",
    "new_cmd_id",
    "new_secret_hex",
    "now_ts",
    "sign",
    "verify",
]
