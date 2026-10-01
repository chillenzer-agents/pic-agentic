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
import shlex
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import aiohttp

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


async def api(homeserver: str, method: str, path: str, token: str, data: dict | None = None) -> dict:
    """Call one Matrix client-server endpoint and return the JSON body.

    Uses ``aiohttp`` (as the MAS token store already does) so the underlying
    connector performs Happy Eyeballs: on a host whose DNS returns an AAAA
    record first but has no IPv6 route, it races the IPv4 address instead of
    blackholing on IPv6 as ``urllib`` would.

    Returns:
        The decoded response body.

    Raises:
        SystemExit: On an HTTP error, with the server's message.

    """
    if urllib.parse.urlparse(homeserver).scheme not in {"http", "https"}:
        msg = f"homeserver must be an http(s) URL: {homeserver!r}"
        raise SystemExit(msg)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=40)
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.request(method, homeserver + path, json=data, headers=headers, timeout=timeout) as response,
        ):
            if not response.ok:
                detail = (await response.text(errors="replace"))[:300]
                msg = f"{method} {path} failed ({response.status}): {detail}"
                raise SystemExit(msg)
            return await response.json(content_type=None)
    except aiohttp.ClientError as exc:
        msg = f"{method} {path} failed: {exc}"
        raise SystemExit(msg) from exc


async def resolve_room_user(
    *,
    homeserver: str,
    access_token_provider: TokenProvider,
) -> tuple[str, str]:
    """Resolve the access token and the account's Matrix user id.

    Returns:
        The ``(access_token, user_id)`` pair.

    """
    access = await access_token_provider()
    who = await api(homeserver, "GET", "/_matrix/client/v3/account/whoami", access)
    user = who["user_id"]
    return access, user


async def create_room_and_secret(
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

    access, user = await resolve_room_user(homeserver=homeserver, access_token_provider=access_token_provider)
    room = await api(
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

    The payload is written to a same-directory temporary file created with mode
    ``0600``, ``fsync``ed, then ``os.replace``d over the target, so a crash or a
    failed write never leaves a truncated or world-readable secret behind.

    Args:
        state_path: Target path.
        state: The state to serialise.

    """
    payload = (json.dumps(state, indent=2) + "\n").encode("utf-8")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(state_path.parent), prefix=f".{state_path.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    Path(tmp).replace(state_path)


def build_export(homeserver: str, room_id: str, rcp_secret: str, sim: str, message_dir: str) -> str:
    """Return the cluster-side ``export`` shell lines for the minted room.

    Returns:
        The newline-terminated export block, without a trailing blank line.

    """
    return (
        f"export PIC_AGENTIC_HOMESERVER={shlex.quote(homeserver)}\n"
        f"export PIC_AGENTIC_ROOM_ID={shlex.quote(room_id)}\n"
        f"export PIC_AGENTIC_RCP_SECRET={shlex.quote(rcp_secret)}\n"
        f"export PIC_AGENTIC_SIM={shlex.quote(sim)}\n"
        f"export PIC_AGENTIC_MESSAGE_DIR={shlex.quote(message_dir)}\n"
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
    state = asyncio.run(
        create_room_and_secret(
            homeserver=homeserver,
            access_token_provider=_config_token_provider(config),
            sim=args.sim,
            message_dir=args.message_dir,
            state_path=args.state,
        ),
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
