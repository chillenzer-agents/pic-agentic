# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""RCP transports.

The RCP envelope is transport-independent (design section 2); these classes
move :class:`~pic_agentic.rcp.envelope.RcpMessage` objects between the two
parties.  ``MatrixTransport`` is the production path; ``MemoryTransport`` is a
deterministic direct-echo variant used for offline unit tests and for the
design's optional M1.0 de-risking variant.
"""

from pic_agentic.transport.base import Transport
from pic_agentic.transport.matrix import MatrixTransport
from pic_agentic.transport.memory import MemoryTransport

__all__ = ["MatrixTransport", "MemoryTransport", "Transport"]
