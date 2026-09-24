# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""``scripts/mas_login.py`` config-rewrite behaviour."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "mas_login",
    Path(__file__).resolve().parents[1] / "scripts" / "mas_login.py",
)
assert _SPEC is not None
assert _SPEC.loader is not None
mas_login = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mas_login)


def _write(path: Path, *, tokens: dict | None = None) -> None:
    mas_login.write_config(
        path,
        homeserver="https://hs",
        auth_base="https://auth",
        client_id="cid",
        tokens=tokens or {"access_token": "AT", "refresh_token": "RT", "user_id": "@u:hs"},
    )


def test_write_config_preserves_unknown_keys(tmp_path) -> None:
    """Keys the login script does not manage (e.g. rcp_secret) must survive."""
    path = tmp_path / "config.toml"
    path.write_text(
        '[pic_agentic]\nrcp_secret = "THIS-SHOULD-SURVIVE"\n'
        'slurm_bin_dir = "/opt/slurm/bin"\nroom_id = "!r:hs"\nack_timeout_s = 123.0\n'
    )
    _write(path)
    data = tomllib.loads(path.read_text())["pic_agentic"]
    assert data["rcp_secret"] == "THIS-SHOULD-SURVIVE"
    assert data["slurm_bin_dir"] == "/opt/slurm/bin"
    assert data["room_id"] == "!r:hs"
    # Existing scalar values are re-emitted as TOML strings (the writer's
    # format); pydantic parses them back into the declared types.
    assert data["ack_timeout_s"] == "123.0"

    from pic_agentic.config import Config

    cfg = Config.load(path)
    assert cfg.rcp_secret == "THIS-SHOULD-SURVIVE"
    assert cfg.ack_timeout_s == pytest.approx(123.0)
    assert cfg.access_token == "AT"
    # Login-owned fields are refreshed.
    assert data["access_token"] == "AT"
    assert data["refresh_token"] == "RT"
    assert data["token_endpoint"] == "https://auth/oauth2/token"
    assert data["client_id"] == "cid"


def test_write_config_creates_0600(tmp_path) -> None:
    path = tmp_path / "config.toml"
    _write(path)
    assert (path.stat().st_mode & 0o777) == 0o600
