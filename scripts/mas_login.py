# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Log in to a MAS-fronted homeserver with the OAuth device-code flow.

``chat.academiccloud.de`` (and other Matrix Authentication Service deployments)
allow no password login and issue short lived access tokens, so the interactive
bootstrap runs once and stores a rotating refresh token for the agent.

Usage::

    python scripts/mas_login.py                 # uses the bundled client id
    python scripts/mas_login.py --client-id <id>
    python scripts/mas_login.py --out ~/.config/pic-agentic/config.toml

It prints a short user code and a verification URL, waits for approval, then
writes the client id, token endpoint and refresh token to a 0600 config file.
Secrets are never echoed back to the terminal or the log.
"""

from __future__ import annotations

import argparse
import json
import secrets
import stat
import string
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from pathlib import Path

DEFAULT_AUTH_BASE = "https://auth.chat.academiccloud.de"
DEFAULT_HOMESERVER = "https://chat.academiccloud.de"
#: Public client registered for this tool (dynamic client registration, no secret).
DEFAULT_CLIENT_ID = "01M36ZY2E6MZGTASWSFNV2ARX4"
API_SCOPE = "urn:matrix:org.matrix.msc2967.client:api:*"
DEVICE_SCOPE_PREFIX = "urn:matrix:org.matrix.msc2967.client:device:"
DEFAULT_OUT = Path("~/.config/pic-agentic/config.toml").expanduser()


def build_scope(device_id: str) -> str:
    """Return the requested scope, including the device scope MAS needs.

    MAS only provisions a device on the homeserver when the granted scope
    carries a ``...client:device:<id>`` token (``oauth2/token.rs``); without a
    device, Synapse rejects ``m.room.message`` sends.

    Args:
        device_id: Matrix device id to bind the token to.

    Returns:
        The space-separated scope string.

    """
    return f"openid {API_SCOPE} {DEVICE_SCOPE_PREFIX}{device_id}"


def new_device_id() -> str:
    """Return a fresh alphanumeric device id (matches MAS's own format).

    Returns:
        A 10-character alphanumeric device id.

    """
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(10))


def post_form(url: str, data: dict[str, str], *, timeout: float = 30.0) -> tuple[int, dict]:
    """POST an application/x-www-form-urlencoded body and decode the JSON reply.

    Returns:
        The ``(status, body)`` pair; error bodies are returned, not raised.

    """
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=body)
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode())
        except ValueError:
            return exc.code, {"error": "http_error"}


def request_device(auth_base: str, client_id: str, scope: str) -> dict:
    """Start a device authorization and return the device/user codes.

    Args:
        auth_base: MAS base URL.
        client_id: OAuth client id.
        scope: Requested scope (must include the device scope).

    Returns:
        The decoded device-authorization response.

    Raises:
        SystemExit: If the authorization endpoint rejects the request.

    """
    status, body = post_form(f"{auth_base}/oauth2/device", {"client_id": client_id, "scope": scope})
    if status != HTTPStatus.OK or "device_code" not in body:
        msg = f"device authorization failed ({status}): {body}"
        raise SystemExit(msg)
    return body


def poll_token(auth_base: str, client_id: str, device_code: str, *, interval: int, expires_in: int) -> dict:
    """Poll the token endpoint until the user approves or the code expires.

    Returns:
        The decoded token response.

    Raises:
        SystemExit: If the code expires or the endpoint reports a hard error.

    """
    deadline = time.monotonic() + expires_in
    while time.monotonic() < deadline:
        time.sleep(interval)
        status, body = post_form(
            f"{auth_base}/oauth2/token",
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": device_code,
                "client_id": client_id,
            },
        )
        if status == HTTPStatus.OK and "access_token" in body:
            return body
        error = body.get("error", "")
        if error in {"authorization_pending", "slow_down"}:
            if error == "slow_down":
                interval += 1
            continue
        msg = f"device authorization failed ({status}): {body}"
        raise SystemExit(msg)
    msg = "device code expired before approval; run the script again"
    raise SystemExit(msg)


def write_config(path: Path, *, homeserver: str, auth_base: str, client_id: str, tokens: dict) -> None:
    """Write the 0600 config file, preserving an existing room/message dir.

    Args:
        path: Target config path.
        homeserver: Homeserver base URL.
        auth_base: MAS base URL (token endpoint is derived).
        client_id: OAuth client id.
        tokens: The token response (access + refresh).

    """
    existing: dict[str, str] = {}
    if path.exists():
        import tomllib  # ruff: ignore[import-outside-top-level] - only needed for an existing config

        with path.open("rb") as handle:
            existing = tomllib.load(handle).get("pic_agentic", {})
    # Start from the existing settings so keys this script does not manage
    # (e.g. ``rcp_secret``, ``slurm_bin_dir``, ``ack_timeout_s``) survive the
    # rewrite; the token fields below override only what login owns.
    values = {**existing}
    values.update(
        {
            "homeserver": homeserver,
            "user_id": tokens.get("user_id", existing.get("user_id", "")),
            "client_id": client_id,
            "token_endpoint": f"{auth_base}/oauth2/token",
            "access_token": tokens["access_token"],
            "refresh_token": tokens.get("refresh_token", ""),
        }
    )
    lines = ["# pic-agentic configuration (0600). Secrets; never commit.", "[pic_agentic]"]
    lines += [f'{key} = "{value}"' for key, value in values.items() if value]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def main() -> None:
    """Run the interactive device-code login and persist the refresh token."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth-base", default=DEFAULT_AUTH_BASE)
    parser.add_argument("--homeserver", default=DEFAULT_HOMESERVER)
    parser.add_argument("--client-id", default=DEFAULT_CLIENT_ID)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    device_id = new_device_id()
    device = request_device(args.auth_base, args.client_id, build_scope(device_id))
    print(f"1. Open {device['verification_uri']}")
    print(f"2. Enter code: {device['user_code']}")
    print(f"   (valid for {device['expires_in'] // 60} minutes)")
    print("3. Approve the 'pic-agentic CLI' client, then wait here ...")
    sys.stdout.flush()

    tokens = poll_token(
        args.auth_base,
        args.client_id,
        device["device_code"],
        interval=int(device.get("interval", 5)),
        expires_in=int(device.get("expires_in", 600)),
    )
    # Fill in the user id so the config is self-contained (never printed).
    status, whoami = 0, {}
    try:
        req = urllib.request.Request(f"{args.homeserver}/_matrix/client/v3/account/whoami")
        req.add_header("Authorization", f"Bearer {tokens['access_token']}")
        with urllib.request.urlopen(req, timeout=30) as response:
            status, whoami = response.status, json.loads(response.read().decode())
    except Exception:  # ruff: ignore[blind-except] - best-effort convenience only
        status = 0
    if status == HTTPStatus.OK:
        tokens["user_id"] = whoami.get("user_id", "")

    write_config(
        args.out,
        homeserver=args.homeserver,
        auth_base=args.auth_base,
        client_id=args.client_id,
        tokens=tokens,
    )
    print(f"Wrote {args.out} (mode 0600). Refresh support is now configured.")


if __name__ == "__main__":
    main()
