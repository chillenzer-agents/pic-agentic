"""Config loading, precedence and redaction tests."""

from __future__ import annotations

from pic_agentic.config import Config


def test_env_overrides_are_read(tmp_path, monkeypatch):
    monkeypatch.setenv("PIC_AGENTIC_HOMESERVER", "http://env-hs")
    monkeypatch.setenv("PIC_AGENTIC_RCP_SECRET", "env-secret")
    cfg = Config.load(tmp_path / "missing.toml")
    assert cfg.homeserver == "http://env-hs"
    assert cfg.rcp_secret == "env-secret"


def test_toml_is_read_and_env_wins(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('[pic_agentic]\nhomeserver = "http://toml-hs"\nroom_id = "!toml:hs"\nack_timeout_s = 12.5\n')
    cfg = Config.load(path)
    assert cfg.homeserver == "http://toml-hs"
    assert cfg.room_id == "!toml:hs"
    assert cfg.ack_timeout_s == 12.5

    monkeypatch.setenv("PIC_AGENTIC_HOMESERVER", "http://env-hs")
    assert Config.load(path).homeserver == "http://env-hs"


def test_require_lists_all_missing(tmp_path):
    cfg = Config.load(tmp_path / "missing.toml")
    try:
        cfg.require("homeserver", "room_id")
    except Exception as exc:  # noqa: BLE001
        assert "homeserver" in str(exc)
        assert "room_id" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError")


def test_redact_removes_secrets():
    cfg = Config(access_token="tok-123", rcp_secret="sec-456")
    text = cfg.redact("prefix tok-123 middle sec-456 suffix")
    assert "tok-123" not in text
    assert "sec-456" not in text
    assert text.count("[REDACTED]") == 2
    assert cfg.redact("") == ""
