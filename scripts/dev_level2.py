# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Run a full level-2 local smoke test of the M1 ``hello`` path.

Level 2 means a real Matrix homeserver with a real simulation-side client and
the fake SLURM doubles, driven exactly like an MCP host would drive the server:

    MCP stdio client -> pic_agentic.server -> Matrix -> pic_agentic.simclient
                     -> tests/fake_slurm -> ack -> MCP client

This script performs the level-2 setup and acts as the fake client:

1. provision a fresh room + bot tokens (via ``scripts/dev_synapse.py``),
2. start ``pic_agentic.simclient`` in a subprocess against the fake SLURM,
3. drive ``pic_agentic.server`` over stdio and call the ``hello`` tool,
4. print the result and the artifacts written under the shared directory,
5. tear the simclient down again.

It assumes a local homeserver is already running and reachable (it is the one
external service, like the cluster).  See the README, "Run the M1 PoC", for how
to start one.  Usage::

    python scripts/dev_level2.py
    python scripts/dev_level2.py --shared-dir /tmp/pic-agentic-shared --keep

Exits non-zero if the round trip does not produce a job id.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from time import monotonic

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from pic_agentic.rcp import new_secret_hex

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEV_SYNAPSE = HERE / "dev_synapse.py"
DEFAULT_SLURM_BIN = REPO / "tests" / "fake_slurm"
DEFAULT_HS = os.environ.get("PIC_AGENTIC_HOMESERVER", "http://127.0.0.1:8008")
DEFAULT_MESSAGE = "Hello World from level-2"
HTTP_OK = 200


def reachable(hs: str) -> bool:
    """Return whether the homeserver answers its versions endpoint.

    Returns:
        True if the local homeserver responds with HTTP 200.

    """
    try:
        # The scheme is a localhost dev URL, never user input.
        with urllib.request.urlopen(hs + "/_matrix/client/versions", timeout=5) as response:
            return response.status == HTTP_OK
    except Exception:  # ruff: ignore[blind-except] - any failure means "unreachable"
        return False


def provision(hs: str, server_name: str, out_path: Path) -> dict:
    """Run dev_synapse.py to register the bots and create a fresh room.

    Returns:
        The ``{hs, room_id, tokens}`` blob written by ``dev_synapse.py``.

    Raises:
        SystemExit: If provisioning fails.

    """
    proc = subprocess.run(
        [
            sys.executable,
            str(DEV_SYNAPSE),
            "--hs",
            hs,
            "--server-name",
            server_name,
            "--new-room",
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not out_path.exists():
        msg = f"could not provision the room: {proc.stderr or proc.stdout}"
        raise SystemExit(msg)
    return json.loads(out_path.read_text(encoding="utf-8"))


def base_env(blob: dict, workspace: Path, shared_dir: Path, secret: str, sim: str) -> dict:
    """Build the environment shared by both RCP parties.

    Returns:
        The environment variable mapping common to both processes.

    """
    return {
        "PIC_AGENTIC_HOMESERVER": blob["hs"],
        "PIC_AGENTIC_ROOM_ID": blob["room_id"],
        "PIC_AGENTIC_RCP_SECRET": secret,
        "PIC_AGENTIC_MESSAGE_DIR": str(shared_dir),
        "PIC_AGENTIC_SIM": sim,
        "PIC_AGENTIC_NIO_STORE_DIR": str(workspace / "nio"),
    }


def simclient_env(blob: dict, args, workspace: Path, shared_dir: Path, secret: str) -> dict:
    """Build the environment for the simulation-side client subprocess.

    Returns:
        The environment mapping for ``pic_agentic.simclient``.

    """
    env = os.environ | base_env(blob, workspace, shared_dir, secret, args.sim)
    env |= {
        "PIC_AGENTIC_USER_ID": f"@simclient:{args.server_name}",
        "PIC_AGENTIC_ACCESS_TOKEN": blob["tokens"]["simclient"],
        "PIC_AGENTIC_SLURM_BIN_DIR": str(args.slurm_bin_dir),
        "PIC_AGENTIC_POLL_INTERVAL_S": "0.2",
        "FAKE_SLURM_STATE": str(workspace / "slurm-state"),
    }
    return env


def mcp_env(blob: dict, args, workspace: Path, shared_dir: Path, secret: str) -> dict:
    """Build the environment for the MCP server subprocess.

    Returns:
        The environment mapping for ``pic_agentic.server``.

    """
    env = os.environ | base_env(blob, workspace, shared_dir, secret, args.sim)
    env |= {
        "PIC_AGENTIC_USER_ID": f"@mcpserver:{args.server_name}",
        "PIC_AGENTIC_ACCESS_TOKEN": blob["tokens"]["mcpserver"],
        "PIC_AGENTIC_NIO_STORE_DIR": str(workspace / "nio-mcp"),
        "PIC_AGENTIC_ACK_TIMEOUT_S": str(args.ack_timeout),
    }
    return env


async def drive_mcp(env: dict, message: str, attempts: int, delay: float) -> dict:
    """Start the MCP server over stdio and call the ``hello`` tool.

    It does not raise on a failed round trip: raising inside the client task
    group produces a noisy exception group, so the caller reports the failure.

    Returns:
        The last structured ``hello`` result (empty if it never answered).

    """
    params = StdioServerParameters(command=sys.executable, args=["-m", "pic_agentic.server"], env=env)
    last: dict = {}
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = [tool.name for tool in (await session.list_tools()).tools]
        print(f"tools: {tools}")
        for _ in range(attempts):
            result = await session.call_tool("hello", {"message": message})
            last = result.structured_content or {}
            if last.get("ok") and last.get("job_id"):
                return last
            await asyncio.sleep(delay)
    return last


def run(args) -> int:
    """Execute the level-2 flow.

    Returns:
        The process exit code (0 on a successful round trip).

    Raises:
        SystemExit: If no homeserver is reachable, or provisioning fails.

    """
    if not reachable(args.hs):
        msg = (
            f"no homeserver at {args.hs}.\n"
            "Start one first (see README, 'Run the M1 PoC'); this script only needs "
            "it reachable, it does not start it."
        )
        raise SystemExit(msg)

    keep = args.keep or args.shared_dir is not None
    workspace = Path(args.shared_dir) if args.shared_dir else Path(tempfile.mkdtemp(prefix="pic-agentic-level2-"))
    shared_dir = workspace / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    secret = args.secret or new_secret_hex()

    blob = provision(args.hs, args.server_name, workspace / "bots.json")
    print(f"room: {blob['room_id']}")
    print(f"workspace: {workspace}")

    sim_log = workspace / "simclient.log"
    # The simclient logs verbosely; never give it a PIPE we do not drain, or it
    # blocks on a full pipe buffer before it can serve the command.
    with sim_log.open("w", encoding="utf-8") as log_handle:
        sim_proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "pic_agentic.simclient"],
            env=simclient_env(blob, args, workspace, shared_dir, secret),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            code = _run_round_trip(args, workspace, shared_dir, blob, secret)
            if code:
                print(f"--- simclient log tail ({sim_log}) ---", file=sys.stderr)
                print("\n".join(sim_log.read_text(encoding="utf-8").splitlines()[-15:]), file=sys.stderr)
        finally:
            sim_proc.terminate()
            try:
                sim_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sim_proc.kill()

    if keep:
        print(f"kept artifacts under {workspace}")
    else:
        shutil.rmtree(workspace, ignore_errors=True)
    return code


def _run_round_trip(args, workspace: Path, shared_dir: Path, blob: dict, secret: str) -> int:
    """Drive the MCP client against the running simclient and report.

    Returns:
        0 if the round trip produced a job id and the verbatim message, else 1.

    """
    outcome = asyncio.run(
        drive_mcp(mcp_env(blob, args, workspace, shared_dir, secret), args.message, args.attempts, 1.0)
    )
    print("hello result:")
    print(json.dumps(outcome, indent=2))

    msg_files = sorted((shared_dir / "msg").glob("*.txt"))
    out_files = sorted((shared_dir / "out").glob("*.out"))
    print(f"message file: {msg_files[0] if msg_files else '(none)'}")
    print(f"job output:   {out_files[0] if out_files else '(none)'}")

    ok = bool(outcome.get("ok") and outcome.get("job_id"))
    verbatim = args.message in (outcome.get("cluster_output") or "")
    if not ok:
        print("FAILED: no job id in the acknowledgement", file=sys.stderr)
    elif not verbatim:
        print("FAILED: the message did not come back verbatim", file=sys.stderr)
    else:
        print("OK: level-2 round trip succeeded")
    return 0 if ok and verbatim else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hs", default=DEFAULT_HS, help="homeserver base URL")
    parser.add_argument("--server-name", default="localhost")
    parser.add_argument("--shared-dir", default=None, help="keep artifacts under this directory")
    parser.add_argument("--slurm-bin-dir", default=str(DEFAULT_SLURM_BIN), help="dir with sbatch/scontrol/scancel")
    parser.add_argument("--sim", default="level2", help="simulation id for this run")
    parser.add_argument("--secret", default=None, help="RCP secret (random per run if omitted)")
    parser.add_argument("--message", default=DEFAULT_MESSAGE)
    parser.add_argument("--attempts", type=int, default=10, help="hello tool attempts before giving up")
    parser.add_argument("--ack-timeout", type=float, default=5.0, help="per-attempt ack wait, seconds")
    parser.add_argument("--keep", action="store_true", help="keep the temp workspace")
    args = parser.parse_args()
    args.shared_dir = Path(args.shared_dir) if args.shared_dir else None
    args.slurm_bin_dir = Path(args.slurm_bin_dir)

    started = monotonic()
    code = run(args)
    print(f"elapsed: {monotonic() - started:.1f}s")
    sys.exit(code)


if __name__ == "__main__":
    main()
