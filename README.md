<!--
SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf

SPDX-License-Identifier: CC-BY-4.0
-->

# PIC-Agentic

Agentic research infrastructure for PIConGPU: an MCP server and a
simulation-side client that let an LLM agent submit and follow PIConGPU
simulations on a remote SLURM cluster.

The control/reporting path is a small remote control protocol (RCP) carried in
Matrix rooms. Matrix is transport and human-readable audit trail only; heavy
data stays on the shared file system.

This implements milestone **M1** ("Hello World on the cluster") of the design
in `chillenzer-agents/picongpu` PR #50 (branch `task-14-mcp-server-design`):

```
LLM agent --MCP stdio--> MCP server --Matrix--> simclient --sbatch--> SLURM
                             ^                                            |
                             +---------------- ack (+ job id) -------------+
```

## Layout

| Path | Purpose |
|------|---------|
| `src/pic_agentic/rcp/` | RCP envelope (pydantic model), HMAC signing, sequencing, dedup |
| `src/pic_agentic/parsing/` | PIConGPU stdout progress-line parser |
| `src/pic_agentic/transport/` | `MatrixTransport` (matrix-nio) and `MemoryTransport` |
| `src/pic_agentic/slurm/` | Injection-safe `sbatch`/`scontrol`/`scancel` wrappers |
| `src/pic_agentic/simclient/` | Simulation-side client (`hello` + `submit_simulation` handlers) |
| `src/pic_agentic/server/` | MCP stdio server exposing `hello` and `submit_simulation` |
| `src/pic_agentic/protocol/` | Typed RCP message constructors (M1 `hello`, M2 submit) |
| `src/pic_agentic/simulation_build.py` | PICMI-script → `Runner` dump subprocess builder |
| `src/pic_agentic/version.py` | Wire-format/PIConGPU provenance (version, revision, schema hash) |
| `scripts/setup_synapse.sh` | Create a throwaway local Synapse venv + config |
| `scripts/start_synapse.sh` | Start that Synapse (logs errors instead of swallowing them) |
| `scripts/dev_synapse.py` | Provision bots + a fresh RCP room on a running homeserver |
| `scripts/dev_level2.py` | One-shot full-stack smoke test (room + simclient + fake MCP client) |
| `tests/fake_slurm/` | Local `sbatch`/`scontrol`/`scancel` doubles |

## Install

Requires Python 3.11+ (`uv` recommended):

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -e '.[dev]'
```

To build, serialise and run PyPIConGPU simulations, install the pinned
PIConGPU + cwltool as well (`[sim]`). This is needed on whichever side converts
a PICMI script into a `pypicongpu.Runner`: the MCP server (to build the
payload) and the cluster simclient (to rebuild and run it):

```bash
uv pip install --python .venv/bin/python -e '.[dev,sim]'
```

The pin points at the `chillenzer-agents/picongpu` fork stack head that
provides lossless `Runner` round-tripping (`pyproject.toml` `[sim]`). Installing
it requires `git`; the offline test suite does not need it.

## Authentication (MAS-fronted homeservers)

Production homeservers such as `chat.academiccloud.de` are fronted by Matrix
Authentication Service (MAS): there is no password login, access tokens expire
after ~5 minutes, and refresh tokens **rotate** (each refresh consumes the old
one). `matrix-nio` has no refresh support, so the token lifecycle lives in
`src/pic_agentic/auth/`.

Bootstrap once with the OAuth device-code flow:

```bash
python scripts/mas_login.py          # prints a code + URL; writes the 0600 config
```

The token store refreshes on demand and keeps the live pair in a shared 0600
cache guarded by a file lock, so the MCP server and the simclient (which may
share one account) never invalidate each other's rotating refresh token.

Two non-obvious requirements, both verified against a live MAS:

- The grant **must** include a device scope
  (`urn:matrix:org.matrix.msc2967.client:device:<id>`); MAS only provisions a
  homeserver device when that scope is present, and Synapse rejects
  `m.room.message` sends from a device-less session (`mas_login.py` adds it).
- Access tokens are short lived, so refresh is mandatory for any run longer
  than a few minutes.

Local Synapse needs none of this: it uses static tokens.

## Run the M1 PoC

0. Start a local homeserver (development only; production points at Helmholtz
   Matrix, see config below). The two helpers create and run a throwaway
   Synapse; both take `SYNAPSE_HOME` (default `~/synapse`):

   ```bash
   scripts/setup_synapse.sh     # once: venv + config + dev settings
   scripts/start_synapse.sh     # foreground; returns when the endpoint answers
   ```

   `start_synapse.sh` logs to `$SYNAPSE_HOME/synapse.log` and prints its tail
   when startup fails, so errors are never swallowed. It assumes the default
   homeserver URL `http://127.0.0.1:8008` (override with `SYNAPSE_URL`).

1. Provision a room (development shortcut; production uses Helmholtz Matrix):

   ```bash
   python scripts/dev_synapse.py --new-room --out bots.json
   ```

2. Start the simulation-side client (submission node), pointing at the SLURM
   CLIs (the test doubles work for an offline run):

   ```bash
   export PIC_AGENTIC_HOMESERVER=http://127.0.0.1:8008
   export PIC_AGENTIC_ROOM_ID='!room:localhost'
   export PIC_AGENTIC_USER_ID='@simclient:localhost'
   export PIC_AGENTIC_ACCESS_TOKEN='...'
   export PIC_AGENTIC_RCP_SECRET='...'
   export PIC_AGENTIC_MESSAGE_DIR=/shared/pic-agentic
   export PIC_AGENTIC_SLURM_BIN_DIR=$(pwd)/tests/fake_slurm   # or a real bin dir
   python -m pic_agentic.simclient
   ```

3. Point an MCP host (Claude Desktop, CLI agent) at the server, or drive it
   over stdio directly:

   ```bash
   PIC_AGENTIC_USER_ID='@mcpserver:localhost' \
   PIC_AGENTIC_ACCESS_TOKEN='...' python -m pic_agentic.server
   ```

   Call the `hello` tool; the result carries the SLURM `job_id` and the output
   the job printed.

### One-shot smoke test

Steps 1-3 are wrapped by `scripts/dev_level2.py`, which also acts as the fake
MCP client: it provisions a fresh room, starts the simclient against the fake
SLURM doubles, drives `pic_agentic.server` over stdio, prints the `hello`
result and the written artifacts, and tears everything down. It only needs the
homeserver to be reachable (it does not start Synapse):

```bash
python scripts/dev_level2.py                    # temp workspace, cleaned up
python scripts/dev_level2.py --shared-dir /tmp/level2 --keep
```

It exits non-zero if the round trip does not produce a job id.

## M2 `submit_simulation`

M2 adds a single MCP tool that turns a PICMI script into a running remote
simulation. The wire format deliberately carries the **`pypicongpu.Runner`
spec, not the picmi `Simulation`** (the picmi object does not serialise with
raw callables; see `M2-SUBMIT-PLAN.md`).

1. The MCP server writes the PICMI script to a temp file and runs it in a
   **disposable subprocess** (never imports it in-process), then serialises the
   `Runner` dump into a `SimulationPayload`. The payload carries a provenance
   tuple — `wire_format_version`, `picongpu_version`, `picongpu_revision`,
   `schema_hash` (SHA-256 of the `Runner` JSON schema), plus a `payload_hash`
   and 8-hex `sim_id` — and travels **inline inside the signed command**, so no
   shared file system between the two sides is needed. The payload is carried as
   a JSON **string**, because Matrix's canonical JSON (Synapse) rejects floats
   in event-content objects and a simulation is full of floats. Simulations
   larger than `MAX_INLINE_PAYLOAD_BYTES` (48 KiB) are rejected; realistic
   simulations are a few KiB.
2. The simclient re-validates the embedded payload's byte hash and provenance
   tuple against its own install **before** importing PIConGPU; all cluster
   locations (`setup_dir`, `run_dir`, `template_dir`) come from local config,
   never the payload. It rebuilds a fresh `Runner`, `generate()`s the setup and
   `run()`s the workflow (which invokes `sbatch`).
3. Acknowledgements are deliberately coarse: the simclient acks `accepted`
   immediately, then emits `simulation.submitted` (with the SLURM `job_id`
   parsed from `submission_information.txt`) and `results.ready`; a stage
   failure emits `simulation.failed{stage}`. Rejected commands report the
   reason in the single ack instead of an event. Per-stage acks wait for
   upstream PR #55.

Enable the handler by pointing the cluster simclient at a writable shared
directory (the `PIC_AGENTIC_SIM_SETUP_ROOT` environment variable; see
`scripts/cluster_simclient.sh`). Cluster-local `rc_params` are never
transmitted; the simclient asserts its local `tbg_submit` is `"sbatch"` so the
workflow's local-`bash` default cannot silently submit a non-SLURM job.

To exercise it against the cluster, after the `--setup`/`--run` connectivity
check:

```bash
python scripts/local_mcp_check.py --submit --picmi-script ./my_simulation.py
```

The MCP server needs the `[sim]` extra to build the payload (run the check in
that venv, or point `--picongpu-python` at it).

## Security model (M1)

- The LLM-supplied `message` is written to a server-generated absolute path
  and the job runs `cat '<path>'`; the message is **never** interpolated into
  a shell command. The simclient re-validates the path against a configured
  base directory and a safe charset.
- Every RCP message is HMAC-SHA256 signed with a shared per-simulation secret;
  receivers drop duplicates on the transport event id (falling back to
  `(sim, sender_role, seq, type)` when the transport did not stamp one), and
  re-sent commands are idempotent per `cmd_id`.
- Credentials come from the environment or a 0600 config file and are redacted
  from all tool output.

## Configuration

Environment variables (a `~/.config/pic-agentic/config.toml` `[pic_agentic]`
table is also read, with the environment taking precedence):

| Variable | Meaning |
|----------|---------|
| `PIC_AGENTIC_HOMESERVER` | Matrix homeserver base URL |
| `PIC_AGENTIC_ROOM_ID` | RCP room id |
| `PIC_AGENTIC_USER_ID` / `PIC_AGENTIC_ACCESS_TOKEN` | Bot identity and token |
| `PIC_AGENTIC_RCP_SECRET` | Shared per-simulation HMAC secret |
| `PIC_AGENTIC_MESSAGE_DIR` | Shared-FS directory for message files |
| `PIC_AGENTIC_SLURM_BIN_DIR` | Directory holding `sbatch`/`scontrol`/`scancel` |
| `PIC_AGENTIC_JOB_WAIT_TIMEOUT_S` | Simclient job wait (default 60) |
| `PIC_AGENTIC_ACK_TIMEOUT_S` | MCP-server ack wait (default 90) |
| `PIC_AGENTIC_NIO_STORE_DIR` | Optional matrix-nio store directory |
| `PIC_AGENTIC_CLIENT_ID` | MAS OAuth client id (public; from `mas_login.py`) |
| `PIC_AGENTIC_TOKEN_ENDPOINT` | MAS token endpoint for refresh |
| `PIC_AGENTIC_REFRESH_TOKEN` | Rotating refresh token (see `mas_login.py`) |
| `PIC_AGENTIC_TOKEN_CACHE_PATH` | Optional shared 0600 token-cache override |
| `PIC_AGENTIC_PICONGPU_REVISION` | Pinned PIConGPU revision carried in the payload |
| `PIC_AGENTIC_PICONGPU_PYTHON` | Interpreter (with the `[sim]` extra) used to build the payload subprocess |
| `PIC_AGENTIC_SIM_SETUP_ROOT` | Shared-FS root for generated setups (enables M2 submit) |
| `PIC_AGENTIC_CLUSTER_TEMPLATE_DIR` | Cluster-local picongpu template directory |
| `PIC_AGENTIC_CLUSTER_PRESET` | Cluster-local CMake preset name |

## Tests and tooling

```bash
.venv/bin/python -m pytest          # offline suite (MemoryTransport + fake SLURM)
uvx pre-commit run --all-files      # ruff, reuse, hygiene hooks
```

- **Ruff is the only Python linter and formatter**, configured with
  `select = ["ALL"]` plus `preview` in `pyproject.toml`. The `ignore` list is a
  deliberate exception ledger: every entry carries the reason it is off, so a
  new rule family is on by default and must be argued off. The one
  `per-file-ignore` set covers tests (not shipped API) and the standalone dev
  scripts.
- **REUSE** is enforced by the `reuse` pre-commit hook and by CI. Every
  committed file carries an SPDX header; files that cannot (`.python-version`,
  generated artifacts) are attributed in `REUSE.toml`.
- `pyproject.toml` is metadata-only for packaging; the license is declared as
  the SPDX expression `MIT` with `COPYING` as the license file.

`tests/test_e2e_synapse.py` runs the full path over a live local Synapse and
skips automatically when one is not reachable. Tests marked `integration`
require a homeserver, SLURM, or a PIConGPU install and are excluded from the
offline run; `tests/test_submit_integration.py` skips unless PIConGPU is
importable (run it in the `[sim]` venv to exercise the real `Runner`).

## License

Split license, per REUSE:

- **Code** (Python, tests, config, scripts): **MIT** — see `COPYING`.
- **Prose** (documentation such as this README and `AGENTS.md`): **CC-BY-4.0**.

Each file carries the matching SPDX header; `reuse lint` checks compliance and
`reuse spdx` emits the full bill of materials. See `LICENSE.md` for details.

## Known deviations from the design document

These are recorded here because they are deliberate amendments to M0; they
should be folded back into the design document.

- **`cmd_id`**: command payloads carry a UUID `cmd_id`; the simclient ignores a
  re-sent command with the same id, re-sending the stored ack so a sender whose
  original ack was lost does not time out.
- **HMAC scope**: `sender_role` and `in_reply_to` are signed in addition to the
  fields listed in the design's section 2.1.
- **Ack timeout**: the MCP-server ack wait must exceed the simclient job-wait
  timeout (default 90 s vs 60 s), otherwise the `hello` ack arrives after a
  "resend once" and a second job is submitted.
- **Progress format**: only the elapsed-time field is `setw(25)`; the
  avg-per-step field has no outer `setw`. The regex is robust to both.
- **M2 wire format**: the payload carries a `pypicongpu.Runner` *spec*, not the
  picmi `Simulation`, and `rc_params` are never transmitted. The payload travels
  **inline** in the signed command as a JSON string (bounded by
  `MAX_INLINE_PAYLOAD_BYTES`), alongside a provenance header and the JSON
  build/run flags. This keeps the agent host and the cluster split without a
  shared file system; the earlier shared-FS-path variant is available in the git
  history if larger payloads ever need out-of-band transport. The string
  encoding is required because Matrix's canonical JSON rejects floats in event
  content.
- **M2 acks**: coarse (`accepted`, then `simulation.submitted` /
  `results.ready` / `simulation.failed{stage}` events). Per-stage acks are
  deferred to upstream PR #55.
