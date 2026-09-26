# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Durable, atomic persistence for an agenda or campaign on a filesystem.

The agenda is the resumable state of a campaign, so it is written with a
write-to-temp-then-:func:`Path.replace` pattern: a crash mid-write can never
leave a truncated file behind.  This is what lets an agenda survive MCP
server restarts and be re-loaded, expanded and re-saved at each step.

The store is generic over pydantic models: it defaults to
:class:`~pic_agentic.agenda.model.AgendaGroup` (the original use) but accepts
any model class via :class:`~pic_agentic.agenda.typevars.ModelT`, so the same
atomic-write logic persists a whole :class:`~pic_agentic.agenda.campaign.Campaign`.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from pic_agentic.agenda.model import AgendaGroup

#: Any pydantic model the store can round-trip.
ModelT = TypeVar("ModelT", bound=BaseModel)

#: Default file name for a serialised agenda.
DEFAULT_AGENDA_FILE = "agenda.json"

#: Default file name for a serialised campaign.
DEFAULT_CAMPAIGN_FILE = "campaign.json"


class AgendaStore:
    """Read and write one pydantic model atomically.

    The store is a plain class (it holds a filesystem path, a runtime
    resource), not a pydantic model.
    """

    def __init__(self, root: str | os.PathLike[str], *, filename: str = DEFAULT_AGENDA_FILE) -> None:
        """Create a store rooted at ``root``.

        Args:
            root: Directory holding the file.
            filename: File name within ``root``.

        """
        self.root = Path(root)
        self.path = self.root / filename

    def exists(self) -> bool:
        """Whether the file is present.

        Returns:
            True when the file exists.

        """
        return self.path.is_file()

    def save(self, model: BaseModel) -> Path:
        """Atomically write ``model`` to the store.

        Args:
            model: The pydantic model to persist.

        Returns:
            The path written.

        """
        self.root.mkdir(parents=True, exist_ok=True)
        data = model.model_dump_json(indent=2)
        fd, tmp = tempfile.mkstemp(dir=str(self.root), prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            _write_and_replace(fd, tmp, data, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp).unlink()
            raise
        return self.path

    def load(self, model: type[ModelT] = AgendaGroup) -> ModelT:  # type: ignore[assignment]
        """Load the stored model.

        Args:
            model: The pydantic model class to validate against; defaults to
                :class:`AgendaGroup` (the original agenda-only behaviour).

        Returns:
            The deserialised model.  A missing or malformed file surfaces as
            the natural :class:`FileNotFoundError` / validation error.

        """
        return model.model_validate_json(self.path.read_text(encoding="utf-8"))


def _write_and_replace(fd: int, tmp: str, data: str, target: Path) -> None:
    """Write ``data`` to ``fd``/``tmp``, fsync it and atomically replace ``target``.

    Args:
        fd: The open file descriptor from :func:`tempfile.mkstemp`.
        tmp: The temp file path.
        data: The serialised payload.
        target: The final file path.

    """
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    Path(tmp).replace(target)
    _fsync_dir(target.parent)


def _fsync_dir(directory: Path) -> None:
    """Best-effort fsync of a directory so a rename survives a crash.

    Args:
        directory: The directory to fsync.

    """
    with contextlib.suppress(OSError):
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


__all__ = ["DEFAULT_AGENDA_FILE", "DEFAULT_CAMPAIGN_FILE", "AgendaStore"]
