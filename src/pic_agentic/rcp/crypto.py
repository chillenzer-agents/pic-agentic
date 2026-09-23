"""Signing primitives for the remote control protocol.

Messages are signed with HMAC-SHA256 over a canonical JSON encoding of the
envelope (every field except ``sig`` itself).  The per-simulation secret is
shared between the MCP server and the simulation-side client out of band.

Design note (spec amendment vs. TASK-14-MCP-DESIGN.md section 2.1): the design
lists the signed set as ``{version, sim, kind, type, seq, ts, payload}``.  We
additionally bind ``sender_role`` and ``in_reply_to`` into the signature so a
compromised homeserver cannot silently flip a message's origin or reply
linkage.  The wider set is a superset of the documented one; since no peer
implementation exists yet, this is the canonical definition for version 0.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import uuid
from typing import Any

SIG_PREFIX = "hmac-sha256"
SIG_ALGORITHM = "sha256"


def canonical_bytes(obj: Any) -> bytes:
    """Deterministic JSON encoding used as the signed byte string."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sign(secret: str, obj: Any) -> str:
    """Return ``hmac-sha256:<hex>`` over the canonical encoding of ``obj``."""
    digest = hmac.new(secret.encode("utf-8"), canonical_bytes(obj), hashlib.sha256).hexdigest()
    return f"{SIG_PREFIX}:{digest}"


def verify(secret: str, obj: Any, signature: str) -> bool:
    """Constant-time verification of ``signature`` against ``obj``."""
    if not isinstance(signature, str) or not signature.startswith(f"{SIG_PREFIX}:"):
        return False
    expected = sign(secret, obj)
    return hmac.compare_digest(expected, signature)


def new_secret_hex(nbytes: int = 32) -> str:
    """Generate a fresh per-simulation RCP secret."""
    return secrets.token_hex(nbytes)


def new_cmd_id() -> str:
    """Generate a stable command id used for idempotency across re-sends."""
    return uuid.uuid4().hex
