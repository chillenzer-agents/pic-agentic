# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Local MCP-side driver for the real-cluster connectivity check.

Modes:

* ``--setup`` creates a private room on the homeserver using the local account
  (see ``scripts/mas_login.py``), generates the shared RCP secret, and prints
  the two values to export on the cluster before running
  ``scripts/cluster_simclient.sh``.  State is saved to ``--state`` so ``--run``
  can pick it up.

* ``--run`` drives the real MCP stdio server (``python -m pic_agentic.server``)
  and calls the ``hello`` tool, retrying until the cluster simclient acks or
  ``--wait-s`` elapses.

* ``--submit --picmi-script <path>`` calls the M2 ``submit_simulation`` tool
  with a PICMI script instead, and reports the ``sim_id`` and coarse state.
  The server needs the ``[sim]`` extra on the machine running this driver
  (``PIC_AGENTIC_PICONGPU_PYTHON`` may point at another interpreter).

* ``--watch`` reads the RCP room and prints the lifecycle events
  (``simulation.submitted``/``workflow.finished``/``simulation.job_*``/
  ``results.ready``/``simulation.failed``) as they
  arrive, until ``--wait-s`` elapses.  Use it in a second terminal after
  ``--submit`` to follow the run.

* ``--control {checkpoint,stop,checkpoint_and_stop,cancel} --sim-id <id>`` calls
  the matching M3 control tool and prints the ack.  ``checkpoint``/``stop`` map
  to ``checkpoint_simulation``/``stop_simulation``; ``checkpoint_and_stop`` has
  no own server tool, so it is sent as a checkpoint followed by a stop.

* ``--describe --sim-id <id>`` prints the M3 results manifest summary.

* ``--analyze --sim-id <id> [--query <text>]`` prints the milestone-A analysis
  sections (RO-Crate, redacted metadata, openPMD summary, deterministic answer).

* ``--read --sim-id <id> [--result-path <rel>] [--stream {stdout,stderr}]
  [--tail N]`` prints the returned text lines.

* ``--slice --sim-id <id> --record <name> [--component C] [--iteration N|last]
  [--downsample N]`` prints ``n_points`` and a short data prefix.

* ``--export --sim-id <id>`` prints the transfer ticket.

* ``--wait-results --sim-id <id>`` polls ``describe_results`` until the manifest
  appears or ``--wait-s`` elapses; use it to wait for a run before
  ``--describe``.

* ``--agenda-init --agenda-file F [--agenda-script S] [--agenda-replicas N]
  [--agenda-patch sim.time_steps --agenda-values 4,8,16]`` builds a campaign
  file for the agenda tools: N replicated leaves of the Runner spec (defaulting
  to the ``tests/fixtures/pypicongpu_runner.json`` fixture) or one leaf per
  patched value.  This is pure local state; it does not talk to the server.

* ``--agenda-advance --agenda-file F`` calls the ``advance_agenda`` tool (one
  engine tick) and prints the tick result.

* ``--agenda-status --agenda-file F`` calls ``agenda_status`` and prints the
  aggregate campaign view.

The MCP server and the cluster simclient use the same Matrix account but log in
separately, so each gets its own MAS session/device and its own refresh chain
(no rotation conflict).  RCP messages are role-tagged, so the self-echo works.

Usage::

    python scripts/local_mcp_check.py --setup
    # ... start scripts/cluster_simclient.sh on the login node ...
    python scripts/local_mcp_check.py --run
    python scripts/local_mcp_check.py --submit --picmi-script ./my_sim.py
    python scripts/local_mcp_check.py --watch
    python scripts/local_mcp_check.py --control checkpoint --sim-id <id>
    python scripts/local_mcp_check.py --wait-results --sim-id <id>
    python scripts/local_mcp_check.py --describe --sim-id <id>
    python scripts/local_mcp_check.py --analyze --sim-id <id> --query spect
    python scripts/local_mcp_check.py --read --sim-id <id> --stream stdout --tail 50
    python scripts/local_mcp_check.py --slice --sim-id <id> --record E --component z
    python scripts/local_mcp_check.py --export --sim-id <id>
    python scripts/local_mcp_check.py --agenda-init --agenda-file ./campaign.json --agenda-replicas 3
    python scripts/local_mcp_check.py --agenda-init --agenda-patch sim.time_steps --agenda-values 4,8
    python scripts/local_mcp_check.py --agenda-advance --agenda-file ./campaign.json
    python scripts/local_mcp_check.py --agenda-status --agenda-file ./campaign.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.model import AgendaGroup, AgendaSim
from pic_agentic.agenda.store import AgendaStore
from pic_agentic.auth import MasTokenStore
from pic_agentic.config import Config
from pic_agentic.server.setup import build_export, create_room_and_secret

DEFAULT_STATE = Path("~/.config/pic-agentic/cluster-check.json").expanduser()
DEFAULT_MESSAGE = "Hello from the MCP server to the cluster"
SLICE_PREFIX = 8


def cmd_setup(args: argparse.Namespace) -> int:
    """Create the room + secret and print the cluster-side exports.

    Thin wrapper over :func:`pic_agentic.server.setup.create_room_and_secret`,
    so the driver and the shipped ``pic-agentic-setup`` entry point share one
    implementation.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If no MAS refresh chain is configured or no cluster
            message directory was given.

    """
    config = Config.load()
    if not config.has_refresh_chain():
        msg = "no MAS refresh chain in the config; run scripts/mas_login.py first"
        raise SystemExit(msg)
    if not args.message_dir:
        msg = (
            "set --message-dir to the CLUSTER absolute path for message files "
            "(e.g. /scratch/<user>/pic-agentic/shared); it must match "
            "PIC_AGENTIC_MESSAGE_DIR on the login node"
        )
        raise SystemExit(msg)

    async def token() -> str:
        return await MasTokenStore.from_config(config).access_token()

    state = asyncio.run(
        create_room_and_secret(
            homeserver=config.homeserver,
            access_token_provider=token,
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
    print(f"\nState saved to {args.state} (0600). Then run: python scripts/local_mcp_check.py --run")
    return 0


def _server_env(state: dict) -> dict:
    """Build the environment for the MCP stdio server subprocess.

    Returns:
        ``os.environ`` plus the per-run RCP/MCP settings.

    """
    env = dict(os.environ)
    env.update(
        {
            "PIC_AGENTIC_HOMESERVER": state["homeserver"],
            "PIC_AGENTIC_ROOM_ID": state["room_id"],
            "PIC_AGENTIC_RCP_SECRET": state["rcp_secret"],
            "PIC_AGENTIC_SIM": state["sim"],
            # Used by the M1 hello path; the M2 payload travels inline, so this
            # is only the simclient-side bookkeeping directory.
            "PIC_AGENTIC_MESSAGE_DIR": state["message_dir"],
            "PIC_AGENTIC_POLL_INTERVAL_S": "2",
            # The cluster side may wait in the SLURM queue; keep the ack wait
            # generous.  The `accepted` ack is immediate, so this only bounds a
            # missing/unreachable simclient.
            "PIC_AGENTIC_ACK_TIMEOUT_S": str(state.get("ack_timeout_s", 900)),
        }
    )
    # picongpu_python/revision may come from the state file or the environment.
    for key in ("PIC_AGENTIC_PICONGPU_PYTHON", "PIC_AGENTIC_PICONGPU_REVISION"):
        if state.get(key):
            env[key] = state[key]
    # The server-side agenda store must be the same file the driver wrote.
    if state.get("agenda_file"):
        env["PIC_AGENTIC_AGENDA_FILE"] = state["agenda_file"]
    return env


async def _call_tool(state: dict, tool: str, arguments: dict) -> dict:
    """Drive the MCP stdio server and call one tool once.

    Returns:
        The tool's structured content.

    """
    params = StdioServerParameters(command=sys.executable, args=["-m", "pic_agentic.server"], env=_server_env(state))
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool(tool, arguments)
        return result.structured_content or {"ok": result.is_error is False}


async def _call_hello(state: dict, message: str) -> dict:
    """Call the ``hello`` tool once.

    Returns:
        The tool's structured content.

    """
    return await _call_tool(state, "hello", {"message": message})


async def _call_submit(state: dict, script_path: str) -> dict:
    """Call the ``submit_simulation`` tool once with a PICMI script path.

    Returns:
        The tool's structured content.

    """
    return await _call_tool(state, "submit_simulation", {"picmi_script": script_path})


def cmd_run(args: argparse.Namespace) -> int:
    """Wait for the cluster simclient, then exercise the hello round trip.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If no setup state exists.

    """
    if not args.state.exists():
        msg = f"no state at {args.state}; run --setup first"
        raise SystemExit(msg)
    state = json.loads(args.state.read_text())
    state["ack_timeout_s"] = args.ack_timeout_s

    deadline = time.monotonic() + args.wait_s
    attempt = 0
    while True:
        attempt += 1
        print(f"[attempt {attempt}] sending hello (per-attempt ack wait {args.ack_timeout_s:.0f}s) ...", flush=True)
        try:
            result = asyncio.run(_call_hello(state, args.message))
        except Exception as exc:  # ruff: ignore[blind-except] - the server reports failure as data normally
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result, indent=2), flush=True)
        if result.get("ok") and args.message in (result.get("cluster_output") or ""):
            print("\nCOMMUNICATION OK: the cluster simclient received the command and acked.")
            return 0
        if time.monotonic() >= deadline:
            print("\nFAILED: no successful ack before the wait expired.", file=sys.stderr)
            return 1
        print("no ack yet; is the cluster simclient running? retrying in 15s ...", flush=True)
        time.sleep(15)


def cmd_submit(args: argparse.Namespace) -> int:
    """Drive the MCP server and call the M2 submit_simulation tool once.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If the state or the PICMI script is missing.

    """
    if not args.state.exists():
        msg = f"no state at {args.state}; run --setup first"
        raise SystemExit(msg)
    if not args.picmi_script:
        msg = "--submit requires --picmi-script <path>"
        raise SystemExit(msg)
    script = Path(args.picmi_script).expanduser()
    if not script.is_file():
        msg = f"PICMI script not found: {script}"
        raise SystemExit(msg)
    state = json.loads(args.state.read_text())
    state["ack_timeout_s"] = args.ack_timeout_s
    if args.picongpu_python:
        state["PIC_AGENTIC_PICONGPU_PYTHON"] = args.picongpu_python
    try:
        result = asyncio.run(_call_submit(state, str(script.resolve())))
    except Exception as exc:  # ruff: ignore[blind-except] - the server reports failure as data normally
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, indent=2), flush=True)
    if not result.get("ok"):
        print("\nFAILED: the submit command was not accepted.", file=sys.stderr)
        return 1
    print(f"\nSUBMIT ACCEPTED: sim_id={result.get('sim_id')} state={result.get('state')}")
    print("Watch the room / run --watch to follow the run to results.ready.")
    return 0


def _call_report_tool(state: dict, tool: str, arguments: dict) -> int:
    """Drive the MCP server and call one M2b reporting tool once.

    A fresh server process backfills the room into its registry on startup, so
    ``get_status``/``list_simulations``/``get_events`` see prior history.

    Returns:
        The process exit code.

    """
    try:
        result = asyncio.run(_call_tool(state, tool, arguments))
    except Exception as exc:  # ruff: ignore[blind-except] - surfaced as data normally
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if "error" not in result else 1


def cmd_status(args: argparse.Namespace) -> int:
    """Call ``get_status`` / ``list_simulations`` / ``get_events`` on the room.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If no setup state exists.

    """
    if not args.state.exists():
        msg = f"no state at {args.state}; run --setup first"
        raise SystemExit(msg)
    state = json.loads(args.state.read_text())
    if args.events:
        return _call_report_tool(state, "get_events", {"sim_id": args.sim_id, "limit": 50})
    if args.sim_id:
        return _call_report_tool(state, "get_status", {"sim_id": args.sim_id})
    return _call_report_tool(state, "list_simulations", {"active_only": False})


def cmd_logs(args: argparse.Namespace) -> int:
    """Call ``get_logs`` for one sim on the room.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If no setup state exists.

    """
    if not args.state.exists():
        msg = f"no state at {args.state}; run --setup first"
        raise SystemExit(msg)
    state = json.loads(args.state.read_text())
    return _call_report_tool(state, "get_logs", {"sim_id": args.sim_id, "stream": args.stream, "tail": args.tail})


def _load_state(args: argparse.Namespace) -> dict:
    """Load the setup state and validate a required ``--sim-id``.

    Returns:
        The parsed state dict.

    Raises:
        SystemExit: If no state exists or ``--sim-id`` is missing.

    """
    if not args.state.exists():
        msg = f"no state at {args.state}; run --setup first"
        raise SystemExit(msg)
    if not args.sim_id:
        mode = getattr(args, "control", None) or "results"
        msg = f"the {mode} mode requires --sim-id <sim_id>"
        raise SystemExit(msg)
    return json.loads(args.state.read_text())


def _result_tool_call(state: dict, tool: str, arguments: dict) -> dict:
    """Drive the MCP server and call one M3 tool, returning the payload.

    Returns:
        The tool's structured content (or a soft-error dict).

    """
    try:
        return asyncio.run(_call_tool(state, tool, arguments))
    except Exception as exc:  # ruff: ignore[blind-except] - surfaced as data normally
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def cmd_control(args: argparse.Namespace) -> int:
    """Call the M3 control tool matching ``--control``.

    Each op maps to its own server tool, including the atomic
    ``checkpoint_and_stop_simulation`` (a single SIGALRM).

    Returns:
        The process exit code (non-zero if ``ok`` is false).

    """
    state = _load_state(args)
    tool_for = {
        "checkpoint": "checkpoint_simulation",
        "stop": "stop_simulation",
        "cancel": "cancel_simulation",
        "checkpoint_and_stop": "checkpoint_and_stop_simulation",
    }
    result = _result_tool_call(state, tool_for[args.control], {"sim_id": args.sim_id})
    print(json.dumps(result, indent=2, default=str), flush=True)
    if not result.get("ok"):
        print("\nFAILED: the control command was not accepted.", file=sys.stderr)
        return 1
    print(f"\nCONTROL OK: op={args.control} sim_id={args.sim_id}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Call ``analyze_output`` and print the composed analysis sections.

    Milestone A live test: exercises the RO-Crate / pypicongpu-metadata /
    openPMD-summary readers on the cluster side and the deterministic answer.

    Returns:
        The process exit code (non-zero if ``ok`` is false).

    """
    state = _load_state(args)
    arguments: dict = {"sim_id": args.sim_id}
    if args.query:
        arguments["query"] = args.query
    result = _result_tool_call(state, "analyze_output", arguments)
    if not result.get("ok", True):
        print(json.dumps(result, indent=2, default=str), flush=True)
        print("\nFAILED: analyze_output returned an error.", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0


def cmd_compute(args: argparse.Namespace) -> int:
    """Call ``run_analysis`` with a program read from a JSON file or inline.

    The program is the safe declarative analysis AST (see
    :mod:`pic_agentic.analysis_program`): no code is executed.  Supply it with
    ``--program-file`` or ``--program``.

    Returns:
        The process exit code (non-zero on an error).

    Raises:
        SystemExit: If no program was supplied or it is not valid JSON.

    """
    state = _load_state(args)
    if args.program_file:
        try:
            program = json.loads(Path(args.program_file).expanduser().read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            msg = f"cannot read analysis program {args.program_file!r}: {exc}"
            raise SystemExit(msg) from exc
        if not isinstance(program, dict):
            bad = "analysis program must be a JSON object"
            raise SystemExit(bad)
    elif args.program:
        try:
            program = json.loads(args.program)
        except ValueError as exc:
            msg = f"--program is not valid JSON: {exc}"
            raise SystemExit(msg) from exc
    else:
        program = _default_energy_spectrum_program()
    result = _result_tool_call(state, "run_analysis", {"sim_id": args.sim_id, "program": program})
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result.get("ok", "error" not in result) else 1


def _default_energy_spectrum_program() -> dict:
    """Return a small default analysis program: an energy spectrum histogram.

    Returns:
        A declarative program computing ``histogram(sqrt(px^2+py^2))`` over the
        ``E`` record's ``x``/``y`` components.

    """

    def var(comp: str) -> dict:
        return {"kind": "var", "name": comp, "record": "E", "component": comp}

    return {
        "selectors": [var("x"), var("y")],
        "output": {
            "kind": "reduce",
            "op": "histogram",
            "bins": 64,
            "operand": {
                "kind": "unop",
                "op": "sqrt",
                "operand": {
                    "kind": "binop",
                    "op": "add",
                    "left": {"kind": "binop", "op": "mul", "left": var("x"), "right": var("x")},
                    "right": {"kind": "binop", "op": "mul", "left": var("y"), "right": var("y")},
                },
            },
        },
    }


def cmd_describe(args: argparse.Namespace) -> int:
    """Call ``describe_results`` and print the manifest summary.

    Returns:
        The process exit code (non-zero if ``ok`` is false).

    """
    state = _load_state(args)
    result = _result_tool_call(state, "describe_results", {"sim_id": args.sim_id})
    if not result.get("ok", True):
        print(json.dumps(result, indent=2, default=str), flush=True)
        print("\nFAILED: describe_results returned an error.", file=sys.stderr)
        return 1
    manifest = result.get("manifest") or {}
    files = manifest.get("files") or []
    print(f"sim_id        : {manifest.get('sim_id')}")
    print(f"output_dir    : {manifest.get('output_dir')}")
    print(f"total_bytes   : {manifest.get('total_bytes')}")
    print(f"reader        : {manifest.get('reader')}")
    print(f"readable_local: {manifest.get('readable_local')}")
    print(f"files         : {len(files)}")
    for ref in files:
        print(f"  - {ref.get('path')}  format={ref.get('format')}  size={ref.get('size_bytes')}")
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    """Call ``read_result`` and print the returned lines.

    Returns:
        The process exit code (non-zero if ``ok`` is false).

    """
    state = _load_state(args)
    if args.result_path:
        arguments: dict = {"sim_id": args.sim_id, "path": args.result_path}
    else:
        arguments = {"sim_id": args.sim_id, "stream": args.stream or None}
    if args.tail is not None:
        arguments["tail"] = args.tail
    result = _result_tool_call(state, "read_result", arguments)
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result.get("ok", True) else 1


def cmd_slice(args: argparse.Namespace) -> int:
    """Call ``get_result_slice`` and print ``n_points`` plus a data prefix.

    Returns:
        The process exit code (non-zero if ``ok`` is false).

    Raises:
        SystemExit: If no setup state exists, ``--sim-id`` is missing, or
            ``--record`` is missing.

    """
    state = _load_state(args)
    if not args.record:
        msg = "--slice requires --record <name>"
        raise SystemExit(msg)
    arguments: dict = {"sim_id": args.sim_id, "record": args.record, "iteration": args.iteration}
    if args.result_path:
        arguments["path"] = args.result_path
    if args.component:
        arguments["component"] = args.component
    if args.downsample is not None:
        arguments["downsample"] = args.downsample
    result = _result_tool_call(state, "get_result_slice", arguments)
    if not result.get("ok", True):
        print(json.dumps(result, indent=2, default=str), flush=True)
        print("\nFAILED: get_result_slice returned an error.", file=sys.stderr)
        return 1
    data = result.get("data") or []
    prefix = ", ".join(str(value) for value in data[:SLICE_PREFIX])
    more = ", ..." if len(data) > SLICE_PREFIX else ""
    print(f"n_points: {result.get('n_points', len(data))}")
    print(f"data[:{SLICE_PREFIX}]: [{prefix}{more}]")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Call ``export_results`` and print the transfer ticket.

    Returns:
        The process exit code (non-zero if ``ok`` is false).

    """
    state = _load_state(args)
    result = _result_tool_call(state, "export_results", {"sim_id": args.sim_id})
    if not result.get("ok", True):
        print(json.dumps(result, indent=2, default=str), flush=True)
        print("\nFAILED: export_results returned an error.", file=sys.stderr)
        return 1
    ticket = result.get("result") or {}
    print(f"ref       : {(ticket.get('ref') or {}).get('path')}")
    print(f"transfer  : {ticket.get('transfer')}")
    print(f"resolved  : {ticket.get('resolved')}")
    print(f"local_path: {ticket.get('local_path')}")
    return 0


def cmd_wait_results(args: argparse.Namespace) -> int:
    """Poll ``describe_results`` until a manifest appears or ``--wait-s`` runs out.

    Returns:
        The process exit code (0 once a manifest appears, 1 on timeout/error).

    """
    state = _load_state(args)
    deadline = time.monotonic() + args.wait_s
    interval = 15.0
    attempt = 0
    while True:
        attempt += 1
        result = _result_tool_call(state, "describe_results", {"sim_id": args.sim_id})
        manifest = result.get("manifest") if isinstance(result, dict) else None
        if manifest and manifest.get("output_dir"):
            print(json.dumps(manifest, indent=2, default=str), flush=True)
            print(f"\nRESULTS READY: sim_id={args.sim_id} files={len(manifest.get('files') or [])}")
            return 0
        if not result.get("ok", True):
            print(json.dumps(result, indent=2, default=str), flush=True)
        print(f"[attempt {attempt}] no result manifest yet; retrying in {interval:.0f}s ...", flush=True)
        if time.monotonic() >= deadline:
            print("\nFAILED: no result manifest before the wait expired.", file=sys.stderr)
            return 1
        time.sleep(interval)


#: Default Runner-spec fixture used by ``--agenda-init`` when no script is given.
_AGENDA_SPEC_FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "pypicongpu_runner.json"

#: A dotted patch-path segment that indexes a list rather than a dict key.
_LIST_INDEX_RE = re.compile(r"-?\d+")


def _agenda_spec(path: str) -> dict:
    """Load a Runner spec (a full runner dump or a bare ``{"sim": ...}``).

    Returns:
        The parsed spec dict.

    Raises:
        SystemExit: If the file does not exist or is not a JSON object.

    """
    spec_path = Path(path).expanduser()
    if not spec_path.is_file():
        msg = f"agenda spec not found: {spec_path}"
        raise SystemExit(msg)
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        msg = f"agenda spec {spec_path} is not valid JSON: {exc}"
        raise SystemExit(msg) from exc
    if not isinstance(spec, dict):
        msg = f"agenda spec {spec_path} must be a JSON object"
        raise SystemExit(msg)
    return spec


def _patch_spec(spec: dict, dotted: str, value: object) -> dict:
    r"""Return a copy of ``spec`` with the dotted JSON path set to ``value``.

    Behaviourally equivalent to the server's ``_patch_spec``
    (``pic_agentic.server.agenda``): the two agree on every valid path.  They
    deliberately diverge only in *error convention* -- this CLI raises
    ``SystemExit`` while the server raises ``TypeError``/``IndexError`` (turned
    into the ``invalid_campaign`` soft error) -- so they are not byte-identical
    and need not be.

    The rule for a segment is decided by the *current node*: a dict node is
    indexed by its key (so a numeric-looking dict key such as a
    boundary-condition map ``{"0": "periodic"}`` is reachable via
    ``sim.bc.0``), and a list node is indexed by the integer the segment spells
    (negative indices count from the end).

    Args:
        spec: The base Runner spec (mutated in a deep copy).
        dotted: A dotted path such as ``sim.time_steps``.
        value: The JSON value to set.

    Returns:
        The patched deep copy.

    Raises:
        SystemExit: If a dict segment is missing, a list node is addressed by a
            non-numeric segment, or a list index is out of range.

    """
    patched = json.loads(json.dumps(spec))
    node: object = patched
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = _patch_child(node, dotted, part)
    last = parts[-1]
    if isinstance(node, list):
        if not _LIST_INDEX_RE.fullmatch(last):
            msg = f"agenda patch path {dotted!r} has no numeric index at {last!r}"
            raise SystemExit(msg)
        try:
            node[int(last)] = value
        except IndexError as exc:
            msg = f"agenda patch path {dotted!r} has no list element at {last!r}"
            raise SystemExit(msg) from exc
    elif isinstance(node, dict):
        node[last] = value
    else:
        msg = f"agenda patch path {dotted!r} has no object at {last!r}"
        raise SystemExit(msg)
    return patched


def _patch_child(node: object, dotted: str, part: str) -> object:
    """Return the child of ``node`` addressed by one intermediate path segment.

    A dict node is indexed by ``part`` as a key; a list node is indexed by the
    integer ``part`` spells (Python indexing, so ``-1`` is the last element).
    Mirrors the server's ``_patch_child`` behaviourally for valid paths.

    Args:
        node: The current dict or list node.
        dotted: The whole dotted path, used in the error message.
        part: The segment to descend through.

    Returns:
        The addressed child node (dict or list).

    Raises:
        SystemExit: If a dict segment is missing, a list node is addressed by a
            non-numeric segment, or an intermediate node is neither.

    """
    if isinstance(node, dict):
        child = node.get(part)
        if not isinstance(child, (dict, list)):
            msg = f"agenda patch path {dotted!r} has no object at {part!r}"
            raise SystemExit(msg)
        return child
    if isinstance(node, list):
        if not _LIST_INDEX_RE.fullmatch(part):
            msg = f"agenda patch path {dotted!r} has no numeric index at {part!r}"
            raise SystemExit(msg)
        try:
            return node[int(part)]
        except IndexError as exc:
            msg = f"agenda patch path {dotted!r} has no list element at {part!r}"
            raise SystemExit(msg) from exc
    msg = f"agenda patch path {dotted!r} has no object at {part!r}"
    raise SystemExit(msg)


def _tag_replica(spec: dict, index: int) -> dict:
    """Return a copy of ``spec`` tagged with a distinct per-replica marker.

    A simulation's ``sim_id`` is the prefix of its payload hash, so identical
    replica specs would collapse into one cluster job and one observation.  The
    marker rides in the pypicongpu ``customuserinput`` rendering context (a
    legitimate, round-trip-safe field), making each replica's payload distinct
    without changing the physics.

    Args:
        spec: The base Runner spec.
        index: The replica index (also the number of extra time steps).

    Returns:
        The tagged deep copy.

    """
    tagged = json.loads(json.dumps(spec))
    sim = tagged.get("sim")
    if not isinstance(sim, dict):
        return tagged
    existing = sim.get("customuserinput")
    context = dict(existing) if isinstance(existing, dict) else {}
    tags = list(context.pop("tags", [])) if context.get("tags") else []
    if "pic_agentic_replica" not in tags:
        tags.append("pic_agentic_replica")
    context["tags"] = tags
    context["pic_agentic_replica"] = index
    sim["customuserinput"] = context
    return tagged


def _parse_agenda_values(values: str) -> list[object]:
    """Parse a comma-separated value list, JSON-decoding each entry.

    Bare tokens are decoded with :func:`json.loads` so ``4`` becomes an int and
    ``1e18`` a float; an undecodable token is kept as a string.

    Returns:
        The parsed values.

    Raises:
        SystemExit: If the list is empty.

    """
    raw = [token.strip() for token in values.split(",") if token.strip()]
    if not raw:
        msg = "--agenda-values must list at least one value"
        raise SystemExit(msg)
    parsed: list[object] = []
    for token in raw:
        try:
            parsed.append(json.loads(token))
        except ValueError:
            parsed.append(token)
    return parsed


def _point_for(parameter: str, value: object) -> dict[str, float | int | str] | None:
    """Return the leaf ``point`` for one sweep value, or None when unusable.

    ``AgendaSim.point`` accepts only ``float | int | str`` (and rejects bools via
    pydantic).  A value that is not one of those (a list/dict/null, or a bool)
    cannot be a valid point, so it is omitted rather than raising a validation
    error and aborting the whole ``--agenda-init``.

    Returns:
        ``{parameter: value}`` when the value is a valid point, else None.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    return {parameter: value}


def cmd_agenda_init(args: argparse.Namespace) -> int:
    """Build and save a campaign file from a Runner-spec fixture or script.

    With ``--agenda-patch``/``--agenda-values`` one leaf per value is created
    (the dotted JSON path in the spec is set to each value); otherwise
    ``--agenda-replicas`` identical leaves are created.  The campaign is written
    with :class:`~pic_agentic.agenda.store.AgendaStore` so the server-side
    agenda tools load it from ``PIC_AGENTIC_AGENDA_FILE``.

    Returns:
        The process exit code.

    """
    spec_source = args.agenda_script or str(_AGENDA_SPEC_FIXTURE)
    base_spec = _agenda_spec(spec_source)
    if args.agenda_patch:
        # Parse the sweep parameter name from the dotted patch path (the leaf's
        # ``point`` is what the refinement engine scores: see
        # ``server.agenda._leaf_score``) and reuse the server's public
        # derivation of the human-readable ``sweep_parameter`` so the driver
        # records the same self-describing label.
        from pic_agentic.agenda.model import readable_label  # ruff: ignore[import-outside-top-level] - test driver
        from pic_agentic.server.agenda import (  # ruff: ignore[import-outside-top-level] - test driver
            sweep_parameter_for,
        )

        parameter = args.agenda_patch.rsplit(".", 1)[-1]
        sweep_parameter = readable_label(sweep_parameter_for(args.agenda_patch, base_spec))
        leaves = [
            (
                f"leaf{index:03d}",
                _patch_spec(base_spec, args.agenda_patch, value),
                _point_for(parameter, value),
                sweep_parameter,
            )
            for index, value in enumerate(_parse_agenda_values(args.agenda_values))
        ]
    else:
        leaves = [
            (f"leaf{index:03d}", _tag_replica(base_spec, index), None, None)
            for index in range(max(1, args.agenda_replicas))
        ]

    agenda = AgendaGroup(name="campaign")
    for name, spec, point, sweep_parameter in leaves:
        agenda = agenda.add(**{name: AgendaSim(name=name, spec=spec, point=point, sweep_parameter=sweep_parameter)})
    campaign = Campaign(name=args.agenda_name, agenda=agenda).with_created_ts()

    file_path = Path(args.agenda_file).expanduser()
    store = AgendaStore(file_path.parent, filename=file_path.name)
    store.save(campaign)
    print(json.dumps({"ok": True, "agenda_file": str(file_path), "leaves": len(leaves)}, indent=2), flush=True)
    return 0


def cmd_agenda_advance(args: argparse.Namespace) -> int:
    """Call the ``advance_agenda`` server tool for the configured campaign.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If no setup state exists.

    """
    if not args.state.exists():
        msg = f"no state at {args.state}; run --setup first"
        raise SystemExit(msg)
    state = json.loads(args.state.read_text())
    state["agenda_file"] = str(Path(args.agenda_file).expanduser())
    result = _result_tool_call(state, "advance_agenda", {})
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result.get("ok", "error" not in result) else 1


def cmd_agenda_status(args: argparse.Namespace) -> int:
    """Call the ``agenda_status`` server tool and print the campaign view.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If no setup state exists.

    """
    if not args.state.exists():
        msg = f"no state at {args.state}; run --setup first"
        raise SystemExit(msg)
    state = json.loads(args.state.read_text())
    state["agenda_file"] = str(Path(args.agenda_file).expanduser())
    result = _result_tool_call(state, "agenda_status", {})
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result.get("ok", "error" not in result) else 1


def cmd_agenda_approve(args: argparse.Namespace) -> int:
    """Call ``approve_agenda_leaf`` for one gated campaign leaf.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If no setup state exists or no leaf path was given.

    """
    if not args.state.exists():
        msg = f"no state at {args.state}; run --setup first"
        raise SystemExit(msg)
    if not args.agenda_leaf:
        msg = "--agenda-approve requires --agenda-leaf PATH"
        raise SystemExit(msg)
    state = json.loads(args.state.read_text())
    state["agenda_file"] = str(Path(args.agenda_file).expanduser())
    result = _result_tool_call(state, "approve_agenda_leaf", {"path": args.agenda_leaf})
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result.get("ok", "error" not in result) else 1


def cmd_fleet_status(args: argparse.Namespace) -> int:
    """Call the ``fleet_status`` server tool and print the aggregate view.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If no setup state exists.

    """
    if not args.state.exists():
        msg = f"no state at {args.state}; run --setup first"
        raise SystemExit(msg)
    state = json.loads(args.state.read_text())
    result = _result_tool_call(state, "fleet_status", {})
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result.get("ok", "error" not in result) else 1


def cmd_campaign_provenance(args: argparse.Namespace) -> int:
    """Call the ``campaign_provenance`` server tool and print the RO-Crate.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If no setup state exists.

    """
    if not args.state.exists():
        msg = f"no state at {args.state}; run --setup first"
        raise SystemExit(msg)
    state = json.loads(args.state.read_text())
    state["agenda_file"] = str(Path(args.agenda_file).expanduser())
    result = _result_tool_call(state, "campaign_provenance", {})
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result.get("ok", "error" not in result) else 1


def cmd_agenda_callbacks(args: argparse.Namespace) -> int:
    """Call ``take_agenda_callbacks`` and print (and drain) the pending list.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If no setup state exists.

    """
    if not args.state.exists():
        msg = f"no state at {args.state}; run --setup first"
        raise SystemExit(msg)
    state = json.loads(args.state.read_text())
    state["agenda_file"] = str(Path(args.agenda_file).expanduser())
    result = _result_tool_call(state, "take_agenda_callbacks", {})
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result.get("ok", "error" not in result) else 1


def cmd_agenda_add_leaf(args: argparse.Namespace) -> int:
    """Add one refinement leaf to the campaign's root group.

    The spec is the fixture (or ``--agenda-script``) patched at ``--agenda-patch``
    with the first ``--agenda-values`` entry, so a refinement point is one
    command.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If no setup state exists or no leaf name was given.

    """
    if not args.state.exists():
        msg = f"no state at {args.state}; run --setup first"
        raise SystemExit(msg)
    if not args.agenda_leaf:
        msg = "--agenda-add-leaf requires --agenda-leaf NAME"
        raise SystemExit(msg)
    spec_source = args.agenda_script or str(_AGENDA_SPEC_FIXTURE)
    spec = _agenda_spec(spec_source)
    if args.agenda_patch and args.agenda_values:
        value = _parse_agenda_values(args.agenda_values)[0]
        spec = _patch_spec(spec, args.agenda_patch, value)
    state = json.loads(args.state.read_text())
    state["agenda_file"] = str(Path(args.agenda_file).expanduser())
    result = _result_tool_call(state, "add_agenda_leaf", {"name": args.agenda_leaf, "spec": spec})
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result.get("ok", "error" not in result) else 1


def _agenda_campaign_tool(args: argparse.Namespace, tool: str, arguments: dict) -> int:
    """Call one campaign tool with the agenda file exported.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If no setup state exists.

    """
    if not args.state.exists():
        msg = f"no state at {args.state}; run --setup first"
        raise SystemExit(msg)
    state = json.loads(args.state.read_text())
    state["agenda_file"] = str(Path(args.agenda_file).expanduser())
    result = _result_tool_call(state, tool, arguments)
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result.get("ok", "error" not in result) else 1


def cmd_agenda_refinement(args: argparse.Namespace) -> int:
    """Call ``suggest_agenda_refinement`` and print the summary.

    Returns:
        The process exit code.

    """
    return _agenda_campaign_tool(args, "suggest_agenda_refinement", {"rel_tol": args.rel_tol})


def cmd_agenda_conclude(args: argparse.Namespace) -> int:
    """Call ``conclude_agenda`` with the given text.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If no setup state exists or no conclusion was given.

    """
    if not args.conclusion:
        msg = "--agenda-conclude requires --conclusion TEXT"
        raise SystemExit(msg)
    return _agenda_campaign_tool(args, "conclude_agenda", {"conclusion": args.conclusion})


def cmd_agenda_cwl(args: argparse.Namespace) -> int:
    """Call ``export_agenda_cwl`` and print the workflow document.

    Returns:
        The process exit code.

    """
    return _agenda_campaign_tool(args, "export_agenda_cwl", {})


def cmd_agenda_record_analysis(args: argparse.Namespace) -> int:
    """Analyse one sim and record its analysis on a campaign leaf.

    Calls ``analyze_output`` for ``--sim-id`` and attaches the returned sections
    to ``--agenda-leaf`` via ``record_agenda_analysis``, so the campaign
    provenance links the run to its finding.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If no setup state exists, or no sim/leaf was given.

    """
    if not args.sim_id or not args.agenda_leaf:
        msg = "--agenda-record-analysis requires --sim-id and --agenda-leaf"
        raise SystemExit(msg)
    if not args.state.exists():
        msg = f"no state at {args.state}; run --setup first"
        raise SystemExit(msg)
    state = json.loads(args.state.read_text())
    state["agenda_file"] = str(Path(args.agenda_file).expanduser())
    analysis = _result_tool_call(state, "analyze_output", {"sim_id": args.sim_id})
    if not analysis.get("ok", "error" not in analysis):
        print(json.dumps(analysis, indent=2, default=str), flush=True)
        return 1
    result = _result_tool_call(state, "record_agenda_analysis", {"path": args.agenda_leaf, "analysis": analysis})
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result.get("ok", "error" not in result) else 1


def cmd_agenda_pause(args: argparse.Namespace) -> int:
    """Call ``pause_agenda`` and print the new lifecycle state.

    Returns:
        The process exit code.

    """
    return _agenda_lifecycle(args, "pause_agenda")


def cmd_agenda_resume(args: argparse.Namespace) -> int:
    """Call ``resume_agenda`` and print the new lifecycle state.

    Returns:
        The process exit code.

    """
    return _agenda_lifecycle(args, "resume_agenda")


def cmd_agenda_stop(args: argparse.Namespace) -> int:
    """Call the ``stop_agenda`` kill-switch and print the result.

    Returns:
        The process exit code.

    """
    return _agenda_lifecycle(args, "stop_agenda")


def _agenda_lifecycle(args: argparse.Namespace, tool: str) -> int:
    """Call one lifecycle tool with the campaign file exported.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If no setup state exists.

    """
    if not args.state.exists():
        msg = f"no state at {args.state}; run --setup first"
        raise SystemExit(msg)
    state = json.loads(args.state.read_text())
    state["agenda_file"] = str(Path(args.agenda_file).expanduser())
    result = _result_tool_call(state, tool, {})
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result.get("ok", "error" not in result) else 1


def cmd_watch(args: argparse.Namespace) -> int:
    """Read the RCP room and print lifecycle events until the wait expires.

    Returns:
        The process exit code (0 on ``results.ready``, 1 on failure/timeout).

    Raises:
        SystemExit: If no setup state exists.

    """
    if not args.state.exists():
        msg = f"no state at {args.state}; run --setup first"
        raise SystemExit(msg)
    state = json.loads(args.state.read_text())
    return asyncio.run(_watch(state, args.wait_s))


async def _watch(state: dict, wait_s: float) -> int:
    from pic_agentic.transport.matrix import (  # ruff: ignore[import-outside-top-level] - lazy transport import
        MatrixTransport,
    )

    config = Config.load()
    token_provider = MasTokenStore.from_config(config).access_token if config.has_refresh_chain() else None
    transport = MatrixTransport(
        state["homeserver"],
        config.user_id or state["user_id"],
        config.access_token,
        state["room_id"],
        store_path=os.environ.get("PIC_AGENTIC_NIO_STORE_DIR") or None,
        token_provider=token_provider,
    )
    seen: set[str] = set()
    deadline = time.monotonic() + wait_s
    try:
        for message in await transport.backfill():
            seen.add(message.transport_event_id or message.dedup_key()[0])
            rc = _print_event(message)
            if rc is not None:
                return rc
        iterator = aiter(transport.receive())
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print("\nNo terminal event before the wait expired.", file=sys.stderr)
                return 1
            try:
                message = await asyncio.wait_for(anext(iterator), timeout=remaining)
            except (TimeoutError, StopAsyncIteration):
                print("\nNo terminal event before the wait expired.", file=sys.stderr)
                return 1
            if message.transport_event_id in seen:
                continue
            seen.add(message.transport_event_id or message.dedup_key()[0])
            rc = _print_event(message)
            if rc is not None:
                return rc
    finally:
        await transport.close()


def _print_event(message) -> int | None:
    if not message.type.startswith("rcp.simulation"):
        return None
    payload = message.payload
    state = payload.get("state")
    job_id = payload.get("job_id")
    detail = ""
    if state == "simulation.step_finished":
        detail = f" step={payload.get('step')} percent={payload.get('percent')} eta_s={payload.get('eta_s')}"
    elif state in {"simulation.job_finished", "simulation.job_failed"}:
        detail = f" slurm_state={payload.get('slurm_state')} exit_code={payload.get('exit_code')}"
    print(f"[{message.type}] state={state} job_id={job_id}{detail}", flush=True)
    error = payload.get("error")
    if error:
        print(f"    error: {error}", flush=True)
    # M2b: `workflow.finished` is no longer terminal (the job may still run).
    # Only `results.ready` (results linked) is success; job/sim failures are not.
    if state == "results.ready":
        linked = payload.get("results_linked")
        print(f"\nRESULTS READY (results linked: {linked}).", flush=True)
        return 0
    if state in {"simulation.failed", "simulation.job_failed"}:
        print("\nSIMULATION FAILED.", file=sys.stderr, flush=True)
        return 1
    return None


def _add_analysis_mode_args(mode: argparse._MutuallyExclusiveGroup) -> None:
    """Add the read-only analysis mode flags to ``mode``.

    Args:
        mode: The mutually exclusive mode group.

    """
    mode.add_argument("--analyze", action="store_true", help="call analyze_output and print the analysis sections")
    mode.add_argument(
        "--compute",
        action="store_true",
        help="call run_analysis with a declarative program (--program FILE or --program JSON)",
    )


def _add_agenda_callback_args(mode: argparse._MutuallyExclusiveGroup) -> None:
    """Add the callback/refinement/lifecycle agenda mode flags to ``mode``.

    Args:
        mode: The mutually exclusive mode group.

    """
    mode.add_argument("--agenda-callbacks", action="store_true", help="drain and print pending agenda callbacks")
    mode.add_argument(
        "--agenda-add-leaf",
        action="store_true",
        help="add one refinement leaf (--agenda-leaf NAME; spec patched by --agenda-patch/--agenda-values)",
    )
    mode.add_argument("--agenda-pause", action="store_true", help="pause the campaign (hold submissions)")
    mode.add_argument("--agenda-resume", action="store_true", help="resume a paused campaign")
    mode.add_argument("--agenda-stop", action="store_true", help="kill-switch: stop and cancel in-flight jobs")
    mode.add_argument(
        "--agenda-refinement",
        action="store_true",
        help="call suggest_agenda_refinement (best point / convergence / next points)",
    )
    mode.add_argument("--agenda-conclude", action="store_true", help="call conclude_agenda (--conclusion TEXT)")
    mode.add_argument("--agenda-cwl", action="store_true", help="call export_agenda_cwl and print the workflow")
    mode.add_argument(
        "--agenda-record-analysis",
        action="store_true",
        help="analyze --sim-id and record it on --agenda-leaf",
    )


def _add_result_mode_args(mode: argparse._MutuallyExclusiveGroup) -> None:
    """Add the results-access mode flags to ``mode``.

    Args:
        mode: The mutually exclusive mode group.

    """
    mode.add_argument("--describe", action="store_true", help="call describe_results and print the manifest summary")
    _add_analysis_mode_args(mode)
    mode.add_argument("--read", action="store_true", help="call read_result for one sim")
    mode.add_argument("--slice", action="store_true", help="call get_result_slice for one sim")
    mode.add_argument("--export", action="store_true", help="call export_results for one sim")
    mode.add_argument("--wait-results", action="store_true", help="poll describe_results until results appear")


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all mutually exclusive modes.

    Returns:
        The configured parser.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--setup", action="store_true", help="create a room + secret for the cluster run")
    mode.add_argument("--run", action="store_true", help="drive the MCP server and send hello")
    mode.add_argument("--submit", action="store_true", help="drive the MCP server and call submit_simulation")
    mode.add_argument("--watch", action="store_true", help="follow lifecycle events in the RCP room")
    mode.add_argument("--status", action="store_true", help="call get_status / list_simulations / get_events")
    mode.add_argument("--logs", action="store_true", help="call get_logs for one sim")
    mode.add_argument(
        "--control",
        choices=["checkpoint", "stop", "checkpoint_and_stop", "cancel"],
        default="",
        help="call an M3 control tool for --sim-id",
    )
    _add_result_mode_args(mode)
    mode.add_argument("--agenda-init", action="store_true", help="build and save a campaign file for the agenda tools")
    mode.add_argument("--agenda-advance", action="store_true", help="call advance_agenda (one engine tick)")
    mode.add_argument("--agenda-status", action="store_true", help="call agenda_status and print the campaign view")
    mode.add_argument("--agenda-approve", action="store_true", help="approve one gated leaf (--agenda-leaf PATH)")
    mode.add_argument("--fleet-status", action="store_true", help="call fleet_status and print summary + alerts")
    mode.add_argument(
        "--campaign-provenance",
        action="store_true",
        help="call campaign_provenance and print the RO-Crate",
    )
    _add_agenda_callback_args(mode)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--sim", default="cluster")
    parser.add_argument("--message", default=DEFAULT_MESSAGE)
    parser.add_argument("--sim-id", default="", help="sim_id for the per-sim reporting/control/result modes")
    parser.add_argument("--events", action="store_true", help="with --status: return get_events instead")
    parser.add_argument("--stream", default="stdout", help="log stream for --logs/--read")
    parser.add_argument("--query", default="", help="free-text query for --analyze")
    parser.add_argument("--program", default="", help="inline JSON analysis program for --compute")
    parser.add_argument("--program-file", default="", help="JSON file analysis program for --compute")
    parser.add_argument("--tail", type=int, default=100, help="log lines for --logs")
    parser.add_argument("--result-path", default="", help="relative result path for --read")
    parser.add_argument("--record", default="", help="openPMD record name for --slice")
    parser.add_argument("--component", default="", help="openPMD component for --slice")
    parser.add_argument("--iteration", default="last", help="iteration (int or 'last') for --slice")
    parser.add_argument("--downsample", type=int, default=None, help="stride for --slice")
    parser.add_argument("--picmi-script", default="", help="PICMI script path for --submit")
    parser.add_argument(
        "--picongpu-python",
        default="",
        help=(
            "Interpreter with the [sim] extra that builds the payload subprocess "
            "(PIC_AGENTIC_PICONGPU_PYTHON); defaults to the server's interpreter."
        ),
    )
    parser.add_argument("--wait-s", type=float, default=1800.0, help="overall wait for an ack")
    parser.add_argument("--ack-timeout-s", type=float, default=900.0, help="per-attempt ack wait")
    parser.add_argument(
        "--message-dir",
        default="",
        help=(
            "CLUSTER-side absolute directory for message files (must match "
            "PIC_AGENTIC_MESSAGE_DIR / the default on the login node; e.g. "
            "/scratch/<user>/pic-agentic/shared). Required for --setup."
        ),
    )
    parser.add_argument(
        "--agenda-file",
        default="campaign.json",
        help=(
            "campaign file for the --agenda-* modes; for the server tools it is "
            "exported as PIC_AGENTIC_AGENDA_FILE and must match the server's."
        ),
    )
    parser.add_argument(
        "--agenda-script",
        default="",
        help="Runner-spec JSON for --agenda-init (defaults to the tests/fixtures runner).",
    )
    parser.add_argument("--agenda-name", default="campaign", help="campaign name written by --agenda-init")
    parser.add_argument("--agenda-replicas", type=int, default=1, help="number of identical leaves for --agenda-init")
    parser.add_argument("--agenda-patch", default="", help="dotted Runner-spec path to sweep, e.g. sim.time_steps")
    parser.add_argument("--agenda-values", default="", help="comma-separated values for --agenda-patch")
    parser.add_argument("--agenda-leaf", default="", help="leaf path for --agenda-approve, e.g. leaf000")
    parser.add_argument("--conclusion", default="", help="conclusion text for --agenda-conclude")
    parser.add_argument("--rel-tol", type=float, default=0.05, help="relative tolerance for --agenda-refinement")
    return parser


def main() -> None:
    """Parse arguments and dispatch to the selected mode.

    Raises:
        SystemExit: With the chosen command's exit code.

    """
    args = _build_parser().parse_args()
    dispatch = (
        (args.setup, cmd_setup),
        (args.submit, cmd_submit),
        (args.watch, cmd_watch),
        (args.status, cmd_status),
        (args.logs, cmd_logs),
        (bool(args.control), cmd_control),
        (args.describe, cmd_describe),
        (args.analyze, cmd_analyze),
        (args.compute, cmd_compute),
        (args.read, cmd_read),
        (args.slice, cmd_slice),
        (args.export, cmd_export),
        (args.wait_results, cmd_wait_results),
        (args.agenda_init, cmd_agenda_init),
        (args.agenda_advance, cmd_agenda_advance),
        (args.agenda_status, cmd_agenda_status),
        (args.agenda_approve, cmd_agenda_approve),
        (args.fleet_status, cmd_fleet_status),
        (args.campaign_provenance, cmd_campaign_provenance),
        (args.agenda_callbacks, cmd_agenda_callbacks),
        (args.agenda_add_leaf, cmd_agenda_add_leaf),
        (args.agenda_pause, cmd_agenda_pause),
        (args.agenda_resume, cmd_agenda_resume),
        (args.agenda_stop, cmd_agenda_stop),
        (args.agenda_refinement, cmd_agenda_refinement),
        (args.agenda_conclude, cmd_agenda_conclude),
        (args.agenda_cwl, cmd_agenda_cwl),
        (args.agenda_record_analysis, cmd_agenda_record_analysis),
    )
    for selected, command in dispatch:
        if selected:
            raise SystemExit(command(args))
    raise SystemExit(cmd_run(args))


if __name__ == "__main__":
    main()
