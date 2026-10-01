# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Mint a fresh Matrix room + RCP secret for one beta run.

This is the shipped counterpart of the ``--setup`` mode in the developer driver
``scripts/local_mcp_check.py``: a beta host that installs only the wheel (and
therefore has no ``scripts/`` checkout) can still mint a room via the
``pic-agentic-setup`` console script.  Both callers share
:func:`create_room_and_secret`, which uses the MAS refresh chain from the
installed configuration to call the homeserver directly.

The state file records ``homeserver``, ``room_id``, ``rcp_secret``, ``user_id``,
``sim`` and ``message_dir`` and is written ``0600`` atomically so a partially
written secret can never be observed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pic_agentic.auth import MasTokenStore
from pic_agentic.config import Config
from pic_agentic.rcp.crypto import new_secret_hex

if TYPE_CHECKING:
    from collections.abc import Awaitable

#: Default state file, matching the developer driver's ``--state`` default.
DEFAULT_STATE = Path("~/.config/pic-agentic/cluster-check.json").expanduser()

#: Room name used for every minted run.
ROOM_NAME = "pic-agentic cluster check"


class TokenProvider(Protocol):
    """Supply a live bearer token for the homeserver."""

    def __call__(self) -> Awaitable[str]:
        """Return a valid access token, refreshing it if needed."""
        ...


def api(homeserver: str, method: str, path: str, token: str, data: dict | None = None) -> dict:
    """Call one Matrix client-server endpoint and return the JSON body.

    Returns:
        The decoded response body.

    Raises:
        SystemExit: On an HTTP error, with the server's message.

    """
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(homeserver + path, data=body, method=method)  # ruff: ignore[suspicious-url-open-usage]
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=40) as response:  # ruff: ignore[suspicious-url-open-usage]
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        msg = f"{method} {path} failed ({exc.code}): {detail}"
        raise SystemExit(msg) from exc


def resolve_room_user(
    *,
    homeserver: str,
    access_token_provider: TokenProvider,
) -> tuple[str, str]:
    """Resolve the access token and the account's Matrix user id.

    Returns:
        The ``(access_token, user_id)`` pair.

    """
    access = asyncio.run(access_token_provider())
    who = api(homeserver, "GET", "/_matrix/client/v3/account/whoami", access)
    user = who["user_id"]
    return access, user


def create_room_and_secret(
    *,
    homeserver: str,
    access_token_provider: TokenProvider,
    sim: str,
    message_dir: str,
    state_path: Path,
) -> dict:
    """Create a room, mint an RCP secret and write the 0600 state file.

    Args:
        homeserver: Matrix homeserver base URL.
        access_token_provider: Zero-arg async callable returning a valid token.
        sim: Simulation label carried in the state (``PIC_AGENTIC_SIM``).
        message_dir: Absolute cluster-side directory for message files.
        state_path: Target path for the 0600 state JSON.

    Returns:
        The state dict that was written.

    Raises:
        SystemExit: If ``message_dir`` is not absolute or a homeserver call
            fails.

    """
    if not message_dir.startswith("/"):
        msg = f"message_dir must be an absolute path: {message_dir!r}"
        raise SystemExit(msg)

    access, user = resolve_room_user(homeserver=homeserver, access_token_provider=access_token_provider)
    room = api(
        homeserver,
        "POST",
        "/_matrix/client/v3/createRoom",
        access,
        {"name": ROOM_NAME, "preset": "private_chat"},
    )
    state = {
        "homeserver": homeserver,
        "room_id": room["room_id"],
        "rcp_secret": new_secret_hex(32),
        "user_id": user,
        "sim": sim,
        "message_dir": message_dir,
    }
    _write_state(state_path, state)
    return state


def _write_state(state_path: Path, state: dict) -> None:
    """Write the state JSON to ``state_path`` atomically with mode 0600.

    The file is created via ``os.open`` with ``O_CREAT|O_WRONLY|O_TRUNC`` and
    ``0o600`` so the secret is never observable with looser permissions, then
    the whole payload is written in one call and ``fsync``ed before the
    directory entry is relied upon.

    Args:
        state_path: Target path.
        state: The state to serialise.

    """
    payload = (json.dumps(state, indent=2) + "\n").encode("utf-8")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(state_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    state_path.chmod(0o600)


def build_export(homeserver: str, room_id: str, rcp_secret: str, sim: str, message_dir: str) -> str:
    """Return the cluster-side ``export`` shell lines for the minted room.

    Returns:
        The newline-terminated export block, without a trailing blank line.

    """
    return (
        f"export PIC_AGENTIC_HOMESERVER='{homeserver}'\n"
        f"export PIC_AGENTIC_ROOM_ID='{room_id}'\n"
        f"export PIC_AGENTIC_RCP_SECRET='{rcp_secret}'\n"
        f"export PIC_AGENTIC_SIM='{sim}'\n"
        f"export PIC_AGENTIC_MESSAGE_DIR='{message_dir}'\n"
    )


def _config_token_provider(config: Config) -> TokenProvider:
    """Return the async token callable for a config's MAS refresh chain.

    Returns:
        An async callable returning a fresh access token.

    """

    async def token() -> str:
        return await MasTokenStore.from_config(config).access_token()

    return token


def _parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        The configured parser.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE, help="state file to write (0600)")
    parser.add_argument("--homeserver", default="", help="homeserver base URL (defaults to the config)")
    parser.add_argument("--sim", default="cluster", help="sim label carried in the state")
    parser.add_argument(
        "--message-dir",
        default="",
        help="CLUSTER-side absolute directory for message files (must match PIC_AGENTIC_MESSAGE_DIR)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point for ``pic-agentic-setup``.

    Args:
        argv: Argument list override (defaults to ``sys.argv[1:]``).

    Returns:
        The process exit code.

    Raises:
        SystemExit: If no refresh chain is configured or ``--message-dir`` is
            missing.

    """
    args = _parser().parse_args(argv)
    config = Config.load()
    if not config.has_refresh_chain():
        msg = "no MAS refresh chain in the config; run mas_login.py first"
        raise SystemExit(msg)
    if not args.message_dir:
        msg = (
            "set --message-dir to the CLUSTER absolute path for message files "
            "(e.g. /scratch/<user>/pic-agentic/shared); it must match "
            "PIC_AGENTIC_MESSAGE_DIR on the login node"
        )
        raise SystemExit(msg)
    homeserver = args.homeserver or config.homeserver
    state = create_room_and_secret(
        homeserver=homeserver,
        access_token_provider=_config_token_provider(config),
        sim=args.sim,
        message_dir=args.message_dir,
        state_path=args.state,
    )
    export = build_export(
        state["homeserver"],
        state["room_id"],
        state["rcp_secret"],
        state["sim"],
        state["message_dir"],
    )
    print(f"Room created on {state['homeserver']} as {state['user_id']}:")
    print(f"  {state['room_id']}\n")
    print("Copy the repo to the cluster (or let the script clone it), then run:\n")
    print(export)
    print(f"  PIC_AGENTIC_MESSAGE_DIR='{state['message_dir']}' bash cluster_simclient.sh")
    print(f"\nState saved to {args.state} (0600).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
