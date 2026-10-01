# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Safe staging of JSON wire-spec files between MCP tools.

``build_spec`` returns a tens-of-KiB Runner spec inline, and an agent that then
had to re-type it into ``create_campaign`` mangled it.  This module lets one
tool **write** a spec to a server-controlled staging directory and the next
**read** it back by path, so the spec never travels through the LLM.

The staging directory is the configured ``spec_dir`` (or, when that is unset, a
``spec`` subdirectory of the shared ``message_dir``).  Only paths that resolve
strictly inside that root are accepted: an LLM-supplied ``/etc/passwd`` or a
``../../`` escape is refused before any file is touched, matching the
:mod:`pic_agentic.simclient.safety` and :mod:`pic_agentic.results` discipline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pic_agentic.simclient.safety import SAFE_CHARSET, UnsafePathError

if TYPE_CHECKING:
    from pic_agentic.config import Config

#: Root-relative subdirectory used when ``config.spec_dir`` is unset: a
#: dedicated staging area below the shared message directory.
DEFAULT_SPEC_SUBDIR = "spec"

#: Hard cap on a staged spec file.  Specs are tens of KiB; the inline submission
#: cap is 48 KiB, but this path is server-local and never crosses the homeserver,
#: so the bound is deliberately larger and explicit rather than the M2 inline cap.
MAX_SPEC_FILE_BYTES = 4 * 1024 * 1024


def spec_root(config: Config) -> Path:
    """Return the sole directory a ``base_spec_path`` may resolve under.

    The configured ``spec_dir`` wins; an unset value falls back to a ``spec``
    subdirectory of the shared message directory, so the feature is available
    (and testable) without extra configuration.

    Returns:
        The absolute staging root (not necessarily existing yet).

    """
    if config.spec_dir:
        return Path(config.spec_dir).expanduser().absolute()
    message_dir = config.message_dir or "."
    return (Path(message_dir).expanduser().absolute() / DEFAULT_SPEC_SUBDIR)


def validate_spec_path(config: Config, path: str) -> Path:
    """Resolve an LLM-supplied spec path, refusing anything outside the root.

    The path may be absolute or relative; both are resolved against the staging
    root and must land strictly inside it.  Symlinks are resolved on **both**
    sides (a cluster scratch directory is routinely a symlink) exactly as
    :func:`~pic_agentic.simclient.safety.validate_message_path` does, so neither
    an escaping link nor a legitimate symlinked root is misjudged.

    Args:
        config: The resolved server configuration (supplies the root).
        path: The caller-supplied path (absolute or root-relative).

    Returns:
        The resolved, in-root target path.

    Raises:
        UnsafePathError: If the path is empty/padded, carries characters outside
            the safe charset, or resolves outside the staging root.

    """
    if path != path.strip() or not path:
        msg = "base_spec_path must be a non-empty, unpadded string"
        raise UnsafePathError(msg)
    if not SAFE_CHARSET.match(path):
        msg = f"base_spec_path has characters outside the safe charset: {path!r}"
        raise UnsafePathError(msg)
    root = spec_root(config).resolve()
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if resolved == root or root not in resolved.parents:
        msg = f"base_spec_path escapes the spec staging directory {str(root)!r}: {path!r}"
        raise UnsafePathError(msg)
    return resolved


def write_spec_file(config: Config, path: str, spec: dict) -> Path:
    """Validate ``path`` then atomically write ``spec`` as JSON.

    Args:
        config: The resolved server configuration (supplies the root).
        path: The caller-supplied target (absolute or root-relative).
        spec: The ``{"sim": ...}`` wire spec to serialise.

    Returns:
        The path that was written.

    Raises:
        UnsafePathError: If the path is unsafe or the payload exceeds
            :data:`MAX_SPEC_FILE_BYTES`.

    """
    target = validate_spec_path(config, path)
    data = (json.dumps(spec, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
    if len(data) > MAX_SPEC_FILE_BYTES:
        msg = f"spec file exceeds {MAX_SPEC_FILE_BYTES} bytes"
        raise UnsafePathError(msg)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(target)
    return target


def read_spec_file(config: Config, path: str) -> dict:
    """Validate ``path`` then read and JSON-decode a staged wire spec.

    Args:
        config: The resolved server configuration (supplies the root).
        path: The caller-supplied path (absolute or root-relative).

    Returns:
        The decoded ``{"sim": ...}`` wire spec.

    Raises:
        UnsafePathError: If the path is unsafe, the file is missing/oversized,
            or its contents are not a JSON object.

    """
    target = validate_spec_path(config, path)
    try:
        size = target.stat().st_size
    except OSError as exc:
        msg = f"base_spec_path not readable: {path!r} ({exc})"
        raise UnsafePathError(msg) from exc
    if size > MAX_SPEC_FILE_BYTES:
        msg = f"base_spec_path exceeds {MAX_SPEC_FILE_BYTES} bytes: {path!r}"
        raise UnsafePathError(msg)
    try:
        loaded = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        msg = f"base_spec_path is not valid JSON: {path!r} ({exc})"
        raise UnsafePathError(msg) from exc
    if not isinstance(loaded, dict):
        msg = f"base_spec_path must hold a JSON object, got {type(loaded).__name__}: {path!r}"
        raise UnsafePathError(msg)
    return loaded
