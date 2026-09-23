"""Shell-safety helpers for inbound RCP payloads (design section 6.4)."""

from __future__ import annotations

import os
import re
from pathlib import Path

SAFE_CHARSET = re.compile(r"^[A-Za-z0-9._/-]+$")
#: Hard cap on the LLM-supplied message written to the shared file system.
MAX_MESSAGE_BYTES = 4096


class UnsafePathError(ValueError):
    pass


def validate_message_path(path: str, base_dir: str) -> Path:
    """Allow only absolute paths with a safe charset under ``base_dir``.

    The MCP server generates the path; the simclient re-checks it as
    defence in depth so a buggy or compromised server cannot cause a write
    outside the configured directory.
    """
    if path != path.strip() or not path:
        raise UnsafePathError("path must be a non-empty, unpadded string")
    if not path.startswith("/"):
        raise UnsafePathError(f"path must be absolute: {path!r}")
    if not SAFE_CHARSET.match(path):
        raise UnsafePathError(f"path has characters outside the safe charset: {path!r}")
    base = Path(base_dir).resolve()
    resolved = Path(os.path.normpath(path))
    if resolved != base and base not in resolved.parents:
        raise UnsafePathError(f"path escapes base directory {base_dir!r}: {path!r}")
    return resolved


def safe_write_message(path: str, base_dir: str, *, default: str = "") -> Path:
    """Validate ``path`` then write the message file atomically."""
    target = validate_message_path(path, base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = default if default else ""
    data = text.encode("utf-8")
    if len(data) > MAX_MESSAGE_BYTES:
        raise UnsafePathError(f"message exceeds {MAX_MESSAGE_BYTES} bytes")
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(target)
    return target
