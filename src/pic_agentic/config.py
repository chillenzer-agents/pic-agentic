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
from pathlib import Path

from pydantic import BaseModel

DEFAULT_CONFIG_PATH = Path("~/.config/pic-agentic/config.toml").expanduser()

#: Environment variable overriding the config-file path.  Tests set it to a
#: temp path so a developer's real 0600 config (which may carry a live MAS
#: refresh chain) never leaks into a hermetic run.
CONFIG_PATH_ENV = "PIC_AGENTIC_CONFIG"

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
    "client_id": "PIC_AGENTIC_CLIENT_ID",
    "token_endpoint": "PIC_AGENTIC_TOKEN_ENDPOINT",
    "refresh_token": "PIC_AGENTIC_REFRESH_TOKEN",
    "token_cache_path": "PIC_AGENTIC_TOKEN_CACHE_PATH",
    "picongpu_revision": "PIC_AGENTIC_PICONGPU_REVISION",
    "picongpu_python": "PIC_AGENTIC_PICONGPU_PYTHON",
    "cluster_template_dir": "PIC_AGENTIC_CLUSTER_TEMPLATE_DIR",
    "cluster_preset": "PIC_AGENTIC_CLUSTER_PRESET",
    "sim_setup_root": "PIC_AGENTIC_SIM_SETUP_ROOT",
}

REDACTED = "[REDACTED]"


class Config(BaseModel):
    """The merged environment/TOML configuration of one RCP party.

    Pydantic parses the TOML/environment strings into the declared types
    (e.g. ``PIC_AGENTIC_ACK_TIMEOUT_S`` into ``float``), so ``load`` only has
    to merge the sources and filter unknown keys.
    """

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
    #: MAS OAuth client id (public).  Set together with ``token_endpoint`` and
    #: ``refresh_token`` to enable automatic access-token refresh.
    client_id: str = ""
    #: MAS token endpoint, e.g. ``https://auth.example.org/oauth2/token``.
    token_endpoint: str = ""
    #: Rotating refresh token for the MAS account.
    refresh_token: str = ""
    #: Optional override for the shared 0600 token cache path.
    token_cache_path: str = ""
    #: Pinned PIConGPU revision carried in the ``simulation.submit`` payload and
    #: checked by the simclient (design section 2.2; the wire-format drift
    #: check).  Empty means "use the revision of the installed tree".
    picongpu_revision: str = ""
    #: Optional interpreter that has the pinned PIConGPU installed.  The MCP
    #: server builds the simulation payload in a disposable subprocess with it;
    #: empty means "the interpreter running the server".
    picongpu_python: str = ""
    #: Cluster-local template directory the simclient renders the setup from.
    cluster_template_dir: str = ""
    #: Cluster-local CMake configure preset number (``pic-build -p``).  Unset
    #: uses the picongpu default preset.
    cluster_preset: int | None = None
    #: Base directory on the shared file system under which the simclient
    #: creates per-simulation ``input``/``run`` directories.
    sim_setup_root: str = ""

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        """Build a configuration from the TOML file and the environment.

        Environment values take precedence over the ``[pic_agentic]`` table.

        Args:
            path: Config path override; defaults to ``PIC_AGENTIC_CONFIG`` when
                set, else ``DEFAULT_CONFIG_PATH``.

        Returns:
            The merged configuration.

        """
        data: dict[str, object] = {}
        env_path = os.environ.get(CONFIG_PATH_ENV)
        cfg_path = path or (Path(env_path).expanduser() if env_path else DEFAULT_CONFIG_PATH)
        if cfg_path.exists():
            with cfg_path.open("rb") as handle:
                data = tomllib.load(handle).get("pic_agentic", {})
        for key, env in ENV_MAP.items():
            if env in os.environ:
                data[key] = os.environ[env]
        # Unknown keys are ignored: the config file may carry settings for
        # other tools, and pydantic would otherwise reject them.
        filtered = {k: v for k, v in data.items() if k in cls.model_fields}
        return cls.model_validate(filtered)

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
        for secret in (self.access_token, self.refresh_token, self.rcp_secret):
            if secret:
                text = text.replace(secret, REDACTED)
        return text

    def has_refresh_chain(self) -> bool:
        """Return whether MAS token refresh is configured.

        Returns:
            True if a token endpoint, a client id and a refresh token are set.

        """
        return bool(self.token_endpoint and self.client_id and self.refresh_token)


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""
