# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Shell-safety helpers for inbound RCP payloads (design section 6.4)."""

from __future__ import annotations

import re
from pathlib import Path

SAFE_CHARSET = re.compile(r"^[A-Za-z0-9._/-]+$")
#: Hard cap on the LLM-supplied message written to the shared file system.
MAX_MESSAGE_BYTES = 4096
#: Hard cap on a serialised simulation payload.  It is written to the shared
#: file system (the Matrix command only carries its path), so the Matrix
#: ``m.room.message`` size limit does not apply; this is a sanity bound on a
#: hostile sender.
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


class UnsafePathError(ValueError):
    """Raised when an RCP payload path is not safe to write to."""


def validate_message_path(path: str, base_dir: str) -> Path:
    """Allow only absolute paths with a safe charset under ``base_dir``.

    The MCP server generates the path; the simclient re-checks it as
    defence in depth so a buggy or compromised server cannot cause a write
    outside the configured directory.

    Args:
        path: The server-generated path to validate.
        base_dir: Directory the path must stay inside.

    Returns:
        The normalised path.

    Raises:
        UnsafePathError: If the path is empty, padded, relative, unsafe, or
            escapes ``base_dir``.

    """
    if path != path.strip() or not path:
        msg = "path must be a non-empty, unpadded string"
        raise UnsafePathError(msg)
    if not path.startswith("/"):
        msg = f"path must be absolute: {path!r}"
        raise UnsafePathError(msg)
    if not SAFE_CHARSET.match(path):
        msg = f"path has characters outside the safe charset: {path!r}"
        raise UnsafePathError(msg)
    # Resolve BOTH sides symmetrically: a cluster home or scratch directory is
    # routinely a symlink (e.g. /home/<user> -> /data/home2/<user>), and if only
    # the base were resolved the candidate would look like it escaped its own
    # directory.  This matches validate_shared_path in slurm/client.py.
    base = Path(base_dir).resolve()
    resolved = Path(path).resolve()
    if resolved != base and base not in resolved.parents:
        msg = f"path escapes base directory {base_dir!r}: {path!r}"
        raise UnsafePathError(msg)
    return resolved


def safe_write_message(path: str, base_dir: str, *, default: str = "") -> Path:
    """Validate ``path`` then write the message file atomically.

    Args:
        path: The server-generated target path.
        base_dir: Directory the path must stay inside.
        default: The message text to write.

    Returns:
        The path that was written.

    Raises:
        UnsafePathError: If the path is unsafe or the message is too large.

    """
    target = validate_message_path(path, base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = default or ""
    data = text.encode("utf-8")
    if len(data) > MAX_MESSAGE_BYTES:
        msg = f"message exceeds {MAX_MESSAGE_BYTES} bytes"
        raise UnsafePathError(msg)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(target)
    return target


def write_payload(path: str, base_dir: str, data: bytes) -> Path:
    """Validate ``path`` then write a serialised simulation payload atomically.

    Used on the MCP-server side to place a payload on the shared file system;
    the simclient only ever *reads* it (after re-validating the path).

    Args:
        path: The server-generated target path.
        base_dir: Directory the path must stay inside.
        data: The payload bytes to write.

    Returns:
        The path that was written.

    Raises:
        UnsafePathError: If the path is unsafe or the payload is too large.

    """
    target = validate_message_path(path, base_dir)
    if len(data) > MAX_PAYLOAD_BYTES:
        msg = f"payload exceeds {MAX_PAYLOAD_BYTES} bytes"
        raise UnsafePathError(msg)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(target)
    return target
