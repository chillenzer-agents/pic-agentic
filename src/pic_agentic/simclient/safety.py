# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Shell-safety helpers for inbound RCP payloads (design section 6.4)."""

from __future__ import annotations

import os
import re
from pathlib import Path

SAFE_CHARSET = re.compile(r"^[A-Za-z0-9._/-]+$")
#: Hard cap on the LLM-supplied message written to the shared file system.
MAX_MESSAGE_BYTES = 4096


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
    base = Path(base_dir).resolve()
    resolved = Path(os.path.normpath(path))
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
