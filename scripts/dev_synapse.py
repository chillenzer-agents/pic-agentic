"""Provision a throwaway local Synapse for development and tests.

Registers the two bot accounts (``mcpserver``, ``simclient``) and creates a
private room.  Prints a JSON blob with the homeserver URL, room id and access
tokens.  Usage::

    python scripts/dev_synapse.py [--new-room]

This is a development helper only; it assumes a local homeserver with open
registration on ``PIC_AGENTIC_HOMESERVER`` (default ``http://127.0.0.1:8008``).
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

DEFAULT_HS = os.environ.get("PIC_AGENTIC_HOMESERVER", "http://127.0.0.1:8008")


def call(method: str, path: str, data=None, token: str | None = None, hs: str = DEFAULT_HS):
    req = urllib.request.Request(
        hs + path,
        data=json.dumps(data).encode() if data is not None else None,
        method=method,
    )
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read().decode())


def login_or_register(user: str, password: str, hs: str) -> str:
    response = call(
        "POST",
        "/_matrix/client/v3/register",
        {"username": user, "password": password, "device_name": "bot", "auth": {"type": "m.login.dummy"}},
        hs=hs,
    )
    if "access_token" not in response:
        response = call(
            "POST",
            "/_matrix/client/v3/login",
            {
                "type": "m.login.password",
                "identifier": {"type": "m.id.user", "user": user},
                "password": password,
            },
            hs=hs,
        )
    if "access_token" not in response:
        raise SystemExit(f"could not authenticate {user}: {response}")
    return response["access_token"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hs", default=DEFAULT_HS)
    parser.add_argument("--server-name", default="localhost")
    parser.add_argument("--new-room", action="store_true", help="always create a fresh room")
    parser.add_argument("--out", default=None, help="write JSON blob to this path")
    args = parser.parse_args()

    tokens = {
        "mcpserver": login_or_register("mcpserver", "pw-mcp-123", args.hs),
        "simclient": login_or_register("simclient", "pw-sim-123", args.hs),
    }
    room = call(
        "POST",
        "/_matrix/client/v3/createRoom",
        {
            "name": "PIConGPU pic-agentic PoC",
            "preset": "private_chat",
            "invite": [f"@simclient:{args.server_name}"],
            "topic": "PIConGPU RCP PoC room",
        },
        token=tokens["mcpserver"],
        hs=args.hs,
    )
    call("POST", f"/_matrix/client/v3/join/{room['room_id']}", {}, token=tokens["simclient"], hs=args.hs)
    blob = {"hs": args.hs, "room_id": room["room_id"], "tokens": tokens}
    text = json.dumps(blob, indent=2)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
        print(f"wrote {args.out} (room {room['room_id']})")
    else:
        print(text)


if __name__ == "__main__":
    main()
