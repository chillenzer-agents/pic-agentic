# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Configuration loading for the MCP server and the simulation-side client.

Credentials come from the environment first, then a 0600 TOML file at
``~/.config/pic-agentic/config.toml`` (design section 6.1).  Nothing is ever
stored in the repository or echoed into the LLM context.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("~/.config/pic-agentic/config.toml").expanduser()

ENV_MAP = {
    "homeserver": "PIC_AGENTIC_HOMESERVER",
    "room_id": "PIC_AGENTIC_ROOM_ID",
    "access_token": "PIC_AGENTIC_ACCESS_TOKEN",
    "user_id": "PIC_AGENTIC_USER_ID",
    "rcp_secret": "PIC_AGENTIC_RCP_SECRET",
    "message_dir": "PIC_AGENTIC_MESSAGE_DIR",
    "slurm_bin_dir": "PIC_AGENTIC_SLURM_BIN_DIR",
    "submit_system": "PIC_AGENTIC_SUBMIT_SYSTEM",
    "job_wait_timeout_s": "PIC_AGENTIC_JOB_WAIT_TIMEOUT_S",
    "ack_timeout_s": "PIC_AGENTIC_ACK_TIMEOUT_S",
    "nio_store_dir": "PIC_AGENTIC_NIO_STORE_DIR",
}

#: Fields parsed as floats when read from the environment or the TOML file.
_FLOAT_FIELDS = ("job_wait_timeout_s", "ack_timeout_s")

REDACTED = "[REDACTED]"


@dataclass
class Config:
    """The merged environment/TOML configuration of one RCP party."""

    homeserver: str = ""
    room_id: str = ""
    access_token: str = ""
    user_id: str = ""
    rcp_secret: str = ""
    #: Absolute directory for RCP payload files on the shared file system.
    message_dir: str = ""
    #: Directory holding the ``sbatch``/``scontrol``/``scancel`` executables.
    slurm_bin_dir: str = ""
    submit_system: str = "sbatch"
    job_wait_timeout_s: float = 60.0
    #: Ack wait; must exceed ``job_wait_timeout_s`` for the M1 hello round trip.
    ack_timeout_s: float = 90.0
    #: Optional matrix-nio store directory (isolation between dev runs/tests).
    nio_store_dir: str = ""

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        """Build a configuration from the TOML file and the environment.

        Environment values take precedence over the ``[pic_agentic]`` table.

        Args:
            path: Config path override; defaults to ``DEFAULT_CONFIG_PATH``.

        Returns:
            The merged configuration.

        """
        data: dict[str, object] = {}
        cfg_path = path or DEFAULT_CONFIG_PATH
        if cfg_path.exists():
            with cfg_path.open("rb") as handle:
                data = tomllib.load(handle).get("pic_agentic", {})
        for key, env in ENV_MAP.items():
            if env in os.environ:
                data[key] = os.environ[env]
        known = set(cls.__dataclass_fields__)
        filtered = {k: v for k, v in data.items() if k in known}
        for field_name in _FLOAT_FIELDS:
            if field_name in filtered:
                filtered[field_name] = float(filtered[field_name])
        return cls(**filtered)

    def require(self, *names: str) -> None:
        """Assert that the named fields are set.

        Args:
            *names: Field names that must be non-empty.

        Raises:
            ConfigError: If any named field is empty.

        """
        missing = [n for n in names if not getattr(self, n, "")]
        if missing:
            msg = f"missing required configuration: {', '.join(missing)}"
            raise ConfigError(msg)

    def redact(self, text: str) -> str:
        """Replace known secrets in ``text``.

        Args:
            text: The string about to leave toward the LLM.

        Returns:
            ``text`` with every configured secret replaced by ``[REDACTED]``.

        """
        if not text:
            return text
        for secret in (self.access_token, self.rcp_secret):
            if secret:
                text = text.replace(secret, REDACTED)
        return text


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""
