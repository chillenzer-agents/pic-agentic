# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Durable, atomic persistence for an agenda on a (shared) filesystem.

The agenda is the resumable state of a campaign, so it is written with a
write-to-temp-then-:func:`os.replace` pattern: a crash mid-write can never
leave a truncated agenda behind.  This is what lets an agenda survive MCP
server restarts and be re-loaded, expanded and re-saved at each step.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

from pic_agentic.agenda.model import AgendaGroup

#: Default file name for a serialised agenda.
DEFAULT_AGENDA_FILE = "agenda.json"


class AgendaStore:
    """Read and write one agenda file atomically.

    The store is a plain class (it holds a filesystem path, a runtime
    resource), not a pydantic model.
    """

    def __init__(self, root: str | os.PathLike[str], *, filename: str = DEFAULT_AGENDA_FILE) -> None:
        """Create a store rooted at ``root``.

        Args:
            root: Directory holding the agenda file.
            filename: File name within ``root``.

        """
        self.root = Path(root)
        self.path = self.root / filename

    def exists(self) -> bool:
        """Whether an agenda file is present.

        Returns:
            True when the agenda file exists.

        """
        return self.path.is_file()

    def save(self, agenda: AgendaGroup) -> Path:
        """Atomically write ``agenda`` to the store.

        Args:
            agenda: The agenda to persist.

        Returns:
            The path written.

        """
        self.root.mkdir(parents=True, exist_ok=True)
        data = agenda.model_dump_json(indent=2)
        fd, tmp = tempfile.mkstemp(dir=str(self.root), prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            Path(tmp).replace(self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp).unlink()
            raise
        return self.path

    def load(self) -> AgendaGroup:
        """Load the agenda from the store.

        Returns:
            The deserialised agenda.  A missing or malformed file surfaces as
            the natural :class:`FileNotFoundError` / validation error.

        """
        return AgendaGroup.model_validate_json(self.path.read_text(encoding="utf-8"))


__all__ = ["DEFAULT_AGENDA_FILE", "AgendaStore"]
