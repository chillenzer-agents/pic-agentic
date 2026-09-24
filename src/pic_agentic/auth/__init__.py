# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Matrix Authentication Service (MAS) token handling.

``chat.academiccloud.de`` (and other MAS-fronted homeservers) issue short
lived access tokens and rotating refresh tokens.  ``matrix-nio`` has no
refresh support, so :class:`MasTokenStore` owns the token lifecycle and the
transports ask it for a currently valid bearer token.
"""

from pic_agentic.auth.mas import MasTokens, MasTokenStore, TokenRefreshError

__all__ = ["MasTokenStore", "MasTokens", "TokenRefreshError"]
