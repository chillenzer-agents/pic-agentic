# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Wire-format and PIConGPU provenance used to gate a ``submit_simulation``.

The wire-format version is hand-bumped when the payload contract changes.  The
PIConGPU version string alone is insufficient for drift detection (the pinned
tree reports only ``0.9.0-dev``), so the payload also carries the pinned
revision and a hash of the ``Runner`` JSON schema; the simclient rejects a
payload whose provenance does not match its own install *before* it writes
anything to the shared file system (design section 2.2).

PIConGPU is an optional dependency (the ``sim`` extra), so this module imports
it lazily and degrades to empty/None values when it is absent.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from typing import Any

from pic_agentic.rcp.crypto import canonical_bytes

#: Hand-bumped version of the ``submit_simulation`` payload contract.
WIRE_FORMAT_VERSION = 1

#: Environment variable pinning the PIConGPU revision the payload is built for.
REVISION_ENV = "PIC_AGENTIC_PICONGPU_REVISION"


class PicongpuUnavailableError(RuntimeError):
    """Raised when a PIConGPU-dependent value is requested without an install."""


@lru_cache(maxsize=1)
def picongpu_version() -> str:
    """Return the installed PIConGPU version string.

    Returns:
        ``picongpu.__version__`` (e.g. ``0.9.0-dev``).

    Raises:
        PicongpuUnavailableError: If PIConGPU is not installed.

    """
    try:
        import picongpu  # ruff: ignore[import-outside-top-level] - optional dependency, imported lazily
    except ImportError as exc:
        msg = "PIConGPU is not installed (install the 'sim' extra)"
        raise PicongpuUnavailableError(msg) from exc
    return str(picongpu.__version__)


@lru_cache(maxsize=1)
def _installed_commit_id() -> str:
    """Read the pip-recorded VCS commit id, if any.

    Returns:
        The ``vcs_info.commit_id`` from the installed picongpu distribution's
        ``direct_url.json``, or ``""`` for a non-VCS (editable/local) install.

    """
    try:
        from importlib.metadata import (  # ruff: ignore[import-outside-top-level] - cheap, but keep import local
            PackageNotFoundError,
            distribution,
        )
    except ImportError:  # pragma: no cover - stdlib on 3.11+
        return ""
    try:
        dist = distribution("picongpu")
    except PackageNotFoundError:
        return ""
    direct_url = dist.read_text("direct_url.json")
    if not direct_url:
        return ""
    try:
        data: dict[str, Any] = json.loads(direct_url)
    except ValueError:
        return ""
    commit = data.get("vcs_info", {}).get("commit_id")
    # Editable/local installs record a file URL with no vcs_info; leave blank
    # rather than inventing a revision (the env pin is the operator's job).
    return commit if isinstance(commit, str) else ""


@lru_cache(maxsize=1)
def picongpu_revision() -> str:
    """Return the pinned PIConGPU revision.

    Precedence: the ``PIC_AGENTIC_PICONGPU_REVISION`` environment variable,
    then the ``vcs_info.commit_id`` recorded by pip in the installed
    distribution's ``direct_url.json``, then an empty string.

    Returns:
        A commit id when known, else ``""``.

    """
    pinned = os.environ.get(REVISION_ENV, "").strip()
    if pinned:
        return pinned
    return _installed_commit_id() or ""


@lru_cache(maxsize=1)
def runner_schema_hash() -> str:
    """Return the SHA-256 of the canonical ``Runner`` JSON schema.

    Returns:
        The hex digest used as the robust drift check.

    Raises:
        PicongpuUnavailableError: If PIConGPU is not installed.

    """
    try:
        from picongpu.pypicongpu.runner import Runner  # ruff: ignore[import-outside-top-level] - optional dependency
    except ImportError as exc:
        msg = "PIConGPU is not installed (install the 'sim' extra)"
        raise PicongpuUnavailableError(msg) from exc
    canonical = canonical_bytes(Runner.model_json_schema()).decode("ascii")
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def local_provenance() -> dict[str, str]:
    """Return this install's provenance tuple.

    Returns:
        ``{"picongpu_version", "picongpu_revision", "schema_hash"}``; the
        version/schema entries are empty strings when PIConGPU is unavailable,
        so a mismatching payload is always rejected rather than crashing the
        comparison.

    """
    try:
        return {
            "picongpu_version": picongpu_version(),
            "picongpu_revision": picongpu_revision(),
            "schema_hash": runner_schema_hash(),
        }
    except PicongpuUnavailableError:
        return {"picongpu_version": "", "picongpu_revision": "", "schema_hash": ""}
