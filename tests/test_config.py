# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Config loading, precedence and redaction tests."""

from __future__ import annotations

import pytest

from pic_agentic.config import Config, ConfigError


def test_env_overrides_are_read(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PIC_AGENTIC_HOMESERVER", "http://env-hs")
    monkeypatch.setenv("PIC_AGENTIC_RCP_SECRET", "env-secret")
    cfg = Config.load(tmp_path / "missing.toml")
    assert cfg.homeserver == "http://env-hs"
    assert cfg.rcp_secret == "env-secret"


def test_toml_is_read_and_env_wins(tmp_path, monkeypatch) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[pic_agentic]\nhomeserver = "http://toml-hs"\nroom_id = "!toml:hs"\nack_timeout_s = 12.5\n')
    cfg = Config.load(path)
    assert cfg.homeserver == "http://toml-hs"
    assert cfg.room_id == "!toml:hs"
    assert cfg.ack_timeout_s == pytest.approx(12.5)

    monkeypatch.setenv("PIC_AGENTIC_HOMESERVER", "http://env-hs")
    assert Config.load(path).homeserver == "http://env-hs"


def test_require_lists_all_missing(tmp_path) -> None:
    cfg = Config.load(tmp_path / "missing.toml")
    with pytest.raises(ConfigError) as excinfo:
        cfg.require("homeserver", "room_id")
    assert "homeserver" in str(excinfo.value)
    assert "room_id" in str(excinfo.value)


def test_redact_removes_secrets() -> None:
    cfg = Config(access_token="tok-123", rcp_secret="sec-456")
    text = cfg.redact("prefix tok-123 middle sec-456 suffix")
    assert "tok-123" not in text
    assert "sec-456" not in text
    assert text.count("[REDACTED]") == 2
    assert not cfg.redact("")
