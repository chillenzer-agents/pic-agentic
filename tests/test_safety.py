# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Path-safety tests for inbound RCP payloads (simclient/safety.py)."""

from __future__ import annotations

import pytest

from pic_agentic.simclient.safety import UnsafePathError, safe_write_message, validate_message_path


def test_accepts_path_inside_base(tmp_path) -> None:
    target = tmp_path / "msg" / "m.txt"
    assert validate_message_path(str(target), str(tmp_path)) == target.resolve()


def test_symlinked_base_accepts_server_path(tmp_path) -> None:
    """A cluster home/scratch is often a symlink; both sides must resolve.

    Regression: the server sends the unresolved prefix (e.g. /home/<user>/...)
    while the simclient's message_dir resolves through a symlink
    (/home/<user> -> /data/home2/<user>); resolving only the base rejected a
    legitimate path as an escape. Observed live on a cluster login node.
    """
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    server_path = str(link / "msg" / "m.txt")
    assert validate_message_path(server_path, str(link)) == (real / "msg" / "m.txt").resolve()
    assert validate_message_path(server_path, str(real)) == (real / "msg" / "m.txt").resolve()


def test_rejects_escape(tmp_path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    with pytest.raises(UnsafePathError):
        validate_message_path(str(base / ".." / "outside.txt"), str(base))
    with pytest.raises(UnsafePathError):
        validate_message_path("/etc/passwd", str(base))


def test_rejects_relative_and_unsafe(tmp_path) -> None:
    with pytest.raises(UnsafePathError):
        validate_message_path("relative/path", str(tmp_path))
    with pytest.raises(UnsafePathError):
        validate_message_path("/abs/with space", str(tmp_path))


def test_safe_write_creates_and_writes(tmp_path) -> None:
    target = tmp_path / "msg" / "out.txt"
    written = safe_write_message(str(target), str(tmp_path), default="hello")
    assert written.read_text() == "hello"


def test_safe_write_rejects_oversize(tmp_path) -> None:
    target = tmp_path / "msg" / "big.txt"
    with pytest.raises(UnsafePathError):
        safe_write_message(str(target), str(tmp_path), default="x" * 5000)
