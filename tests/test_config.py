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


def test_config_path_env_override(tmp_path, monkeypatch) -> None:
    path = tmp_path / "from-env.toml"
    path.write_text('[pic_agentic]\nhomeserver = "http://env-path-hs"\n')
    monkeypatch.setenv("PIC_AGENTIC_CONFIG", str(path))
    assert Config.load().homeserver == "http://env-path-hs"


def test_refresh_chain_and_redaction(tmp_path) -> None:
    cfg = Config(
        access_token="tok-1",
        refresh_token="refresh-1",
        rcp_secret="secret-1",
        client_id="cid",
        token_endpoint="https://auth.example/oauth2/token",
    )
    assert cfg.has_refresh_chain() is True
    text = cfg.redact("tok-1 refresh-1 secret-1")
    assert "tok-1" not in text
    assert "refresh-1" not in text
    assert "secret-1" not in text


def test_no_refresh_chain_without_credentials() -> None:
    assert Config(client_id="cid").has_refresh_chain() is False
    assert Config(token_endpoint="u", client_id="c").has_refresh_chain() is False


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


def test_m2_submit_fields_from_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PIC_AGENTIC_PICONGPU_REVISION", "abc123")
    monkeypatch.setenv("PIC_AGENTIC_PICONGPU_PYTHON", "/opt/pic/bin/python")
    monkeypatch.setenv("PIC_AGENTIC_CLUSTER_TEMPLATE_DIR", "/opt/pic/templates")
    monkeypatch.setenv("PIC_AGENTIC_CLUSTER_PRESET", "3")
    monkeypatch.setenv("PIC_AGENTIC_SIM_SETUP_ROOT", "/shared/sims")
    cfg = Config.load(tmp_path / "missing.toml")
    assert cfg.picongpu_revision == "abc123"
    assert cfg.picongpu_python == "/opt/pic/bin/python"
    assert cfg.cluster_template_dir == "/opt/pic/templates"
    assert cfg.cluster_preset == 3
    assert cfg.sim_setup_root == "/shared/sims"


def test_m2_submit_fields_from_toml_and_defaults(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[pic_agentic]\nsim_setup_root = "/srv/sims"\n')
    cfg = Config.load(path)
    assert cfg.sim_setup_root == "/srv/sims"
    # Unset M2 fields default to empty (M2 submit handler stays disabled).
    assert not cfg.picongpu_revision
    assert not cfg.picongpu_python
    assert not cfg.cluster_template_dir
    assert not cfg.cluster_preset
