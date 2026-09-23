<!--
SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf

SPDX-License-Identifier: MIT
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
| `src/pic_agentic/rcp/` | RCP envelope, HMAC signing, sequencing, dedup |
| `src/pic_agentic/parsing/` | PIConGPU stdout progress-line parser |
| `src/pic_agentic/transport/` | `MatrixTransport` (matrix-nio) and `MemoryTransport` |
| `src/pic_agentic/slurm/` | Injection-safe `sbatch`/`scontrol`/`scancel` wrappers |
| `src/pic_agentic/simclient/` | Simulation-side client (`hello` handler) |
| `src/pic_agentic/server/` | MCP stdio server exposing the `hello` tool |
| `scripts/dev_synapse.py` | Provision a throwaway local Synapse (bots + room) |
| `tests/fake_slurm/` | Local `sbatch`/`scontrol`/`scancel` doubles |

## Install

Requires Python 3.10+ (`uv` recommended):

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -e '.[dev]'
```

## Run the M1 PoC

1. Provision a local homeserver and room (development shortcut; production
   uses Helmholtz Matrix, see config below):

   ```bash
   # with a local Synapse listening on http://127.0.0.1:8008
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

## Security model (M1)

- The LLM-supplied `message` is written to a server-generated absolute path
  and the job runs `cat '<path>'`; the message is **never** interpolated into
  a shell command. The simclient re-validates the path against a configured
  base directory and a safe charset.
- Every RCP message is HMAC-SHA256 signed with a shared per-simulation secret;
  receivers drop duplicates on `(sim, sender_role, seq, type)`.
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
require a homeserver or SLURM and are excluded from the offline run.

## License

MIT (see `COPYING`). Each file additionally carries an SPDX header; the REUSE
compliance status can be checked with `reuse lint`.

## Known deviations from the design document

These are recorded here because they are deliberate amendments to M0; they
should be folded back into the design document.

- **`cmd_id`**: command payloads carry a UUID `cmd_id`; the simclient ignores a
  re-sent command with the same id.
- **HMAC scope**: `sender_role` and `in_reply_to` are signed in addition to the
  fields listed in the design's section 2.1.
- **Ack timeout**: the MCP-server ack wait must exceed the simclient job-wait
  timeout (default 90 s vs 60 s), otherwise the `hello` ack arrives after a
  "resend once" and a second job is submitted.
- **Progress format**: only the elapsed-time field is `setw(25)`; the
  avg-per-step field has no outer `setw`. The regex is robust to both.
