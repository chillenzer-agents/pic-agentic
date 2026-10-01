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

### Clean install for an agent host

A host that exposes this MCP server to an LLM agent must install it as a
**package**, never run it from a checkout. `scripts/install-mcp.sh` is the
supported production path:

```bash
bash scripts/install-mcp.sh          # install + register + preflight
bash scripts/install-mcp.sh --check  # read-only acceptance checks
bash scripts/install-mcp.sh reset    # clear campaign/reuse state only
```

It installs `pic-agentic[sim]` non-editably from a pinned 40-hex commit into a
dedicated venv, writes the 0600 config, registers the server under the
`pic-agentic` key in `~/.config/opencode/opencode.json`, and runs preflights
(tool surface, installed-revision/capability, plus a room/secret rendezvous
check).

A fresh beta run needs a new room + shared RCP secret. The shipped
`pic-agentic-setup` console script mints both from the installed package (no
repo checkout): it uses the config's MAS refresh chain to create the room,
generates a 32-byte hex secret, writes the 0600 state file, and prints the
cluster-side exports:

```bash
pic-agentic-setup --state ~/.config/pic-agentic/cluster-check.json \
  --sim cluster --message-dir /scratch/<user>/pic-agentic/shared
```

This is the same implementation the developer driver
`scripts/local_mcp_check.py --setup` calls, so the two cannot drift. The
environment topology it assumes:

- **The agent sees only the MCP tool surface.** The pic-agentic source, this
  README, `AGENTS.md`, `docs/beta-test-prompts.md`, the tests and the git
  history must not be reachable from the agent's workspace or home. The server
  is a wheel in a venv; the installer fetches the one helper script it needs
  from a pinned raw URL and deletes it, leaving no checkout behind.
- **Secrets stay in the 0600 config** (`~/.config/pic-agentic/config.toml`) or
  the MCP `environment` block — never in the workspace, logs, or tool output.
- **The cluster-side simclient uses the same room, secret and pinned stack.**
  The preflight fails loudly when the configured secret verifies none of the
  room's RCP messages, which is the signature of a stale/mismatched secret. On
  a fresh room with no traffic yet it only warns (there is nothing to verify
  against), and it verifies either role's messages since both sign with the
  same shared secret.
- **A stale pin fails loudly.** The installer asserts the venv's installed
  `pic-agentic` revision equals the pin and that its simclient carries the M3
  result handling and non-blocking dispatch, both at install time and in
  `--check`, so the old blocking control plane cannot be shipped silently.

The end-to-end invariants and the beta acceptance checks are recorded in the
beta-container handover; the installer's `--check` mode implements the
machine-checkable ones.

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
   **disposable interpreter with a minimal environment** (never imports it
   in-process), then serialises the `Runner` dump into a `SimulationPayload`.
   The payload carries a provenance tuple — `wire_format_version`,
   `picongpu_version`, `picongpu_revision`, `schema_hash` (SHA-256 of the
   `Runner` JSON schema), plus a `payload_hash` and 8-hex `sim_id` — and
   travels **inline inside the signed command**, so no shared file system
   between the two sides is needed. The payload is carried as a JSON **string**,
   because Matrix's canonical JSON (Synapse) rejects floats in event-content
   objects and a simulation is full of floats. The inline limit
   (`MAX_INLINE_PAYLOAD_BYTES`, 48 KiB) is checked against the **escaped** wire
   size plus an envelope allowance, so an accepted payload cannot produce an
   over-64-KiB Matrix event; realistic simulations are a few KiB.

   The child is **not a sandbox**: it runs same-uid and the PICMI script is
   arbitrary LLM-supplied code executed on the submission node by design, so it
   can still read the parent environment via `/proc/<ppid>/environ`. What the
   builder does provide is a *disposable interpreter*: it drops the RCP secret
   and MAS tokens from the child environment, points `HOME` at a fresh scratch
   directory (so the 0600 `~/.config/pic-agentic/config.toml` is not reachable
   via `~`), and only echoes a bounded, clearly-labelled tail of the child's
   stderr. Redaction of tool output is best-effort (it strips verbatim secrets,
   not encodings), and the provenance tuple is a drift/consistency check, not a
   trust boundary. Full isolation (separate uid/namespace) is a documented
   non-goal for this PoC.
2. The simclient re-validates the embedded payload's byte hash and provenance
   tuple against its own install **before** importing PIConGPU; all cluster
   locations (`setup_dir`, `run_dir`, `template_dir`) come from local config,
   never the payload. `cfg_file` must be a relative `*.cfg` path and every
   `overwrite_vars` entry a strict `NAME=value` token (no shell metacharacters),
   because the cluster's `tbg` `eval`s them; anything else is rejected with
   `payload_invalid` before `generate()`. It rebuilds a fresh `Runner`,
   `generate()`s the setup and `run()`s the workflow (which invokes `sbatch`).
   The simclient also normalises the generated `input.yaml` so
   `run_overwrite_vars` is the single string the pinned CWL workflow expects.
3. Acknowledgements are deliberately coarse: the simclient acks `accepted`
   immediately, then emits `simulation.submitted` (with the SLURM `job_id`
   parsed from `submission_information.txt`) and `workflow.finished` once the
   CWL workflow (build/prepare/submit/organize) has returned; a stage failure
   emits `simulation.failed{stage}`. `workflow.finished` deliberately does not
   claim the SLURM job finished — it may still be queued or running. The
   simclient also runs the generated `link_results.sh` first, so
   `run_dir/simOutput` is present when the event fires (reported as
   `results_linked`). The design's `results.ready` is reserved for a later
   job-state confirmation, deferred with per-stage events (upstream PR #55).
   Rejected commands report the reason in the single ack instead of an event.

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

### Campaign specs by reference

A Runner spec is tens of KiB, so an agent should never hand-copy one into the
LLM. `build_spec(picmi_script, write_to="base.json")` builds and returns the
spec inline **and** stages a JSON copy under the server's spec directory,
returning its absolute `spec_path`; `create_campaign` then takes
`base_spec_path=<spec_path>` instead of an inline `base_spec` (provide exactly
one of the two). The staged file is ordinary JSON (`{"sim": ...}`) and the
staging root is the configured `PIC_AGENTIC_SPEC_DIR`, or a `spec/`
subdirectory of `PIC_AGENTIC_MESSAGE_DIR` when unset. `base_spec_path` is
LLM-controlled, so it is resolved **strictly inside that root** (safe charset,
symlinks resolved on both sides) and refused otherwise; a path such as
`/etc/passwd` never reaches an open. Staged files are capped at 4 MiB
(`MAX_SPEC_FILE_BYTES`) — larger than the 48 KiB inline cap because the bytes
are server-local and never cross the homeserver.

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
  from all tool output. Redaction is best-effort: it strips verbatim secret
  strings, not base64/hex encodings, and the M2 PICMI child runs same-uid by
  design (see the M2 section) so it is not treated as a trust boundary.

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
| `PIC_AGENTIC_SPEC_DIR` | Staging dir for path-referenced campaign specs (default: `spec/` under `PIC_AGENTIC_MESSAGE_DIR`) |

## Results tools

The MCP server exposes a read-only results surface over a run's linked
`simOutput`:

| Tool | Purpose |
|------|---------|
| `describe_results` | scandir-only manifest (path, format, size) |
| `get_result_slice` | bounded 1D slice of one openPMD record/component |
| `get_result_image` | bounded base64 PNG thumbnail of one openPMD record |
| `read_result` | a small text tail (file, or captured stdout/stderr) |
| `read_plugin_result` | a bounded summary from a shipped PIConGPU plugin reader |
| `export_results` | a transfer ticket (the bulk data never moves itself) |

`describe_results` labels each file with a `format`: `openpmd-adios2` /
`openpmd-hdf5` for openPMD series, `text` / `dir` / `binary`, or a **plugin
reader name** when the filename matches one. The plugin match is a filename-only
heuristic; the reader re-validates the file and returns a clean `no_results` on
a mismatch.

The openPMD readers accept the `h5` (HDF5) and `bp`/`bp5` (ADIOS2) suffixes the
pinned `openpmd_api` 0.17.1 can actually open; `.hdf5` is **not** advertised
because that backend rejects it ("Unknown file format"). ADIOS2 series are
directories (`fields_000050.bp/`), which iteration discovery and the plugin
target resolver both accept.

`read_plugin_result` runs one registered reader on the cluster (in-process, over
the same 48 KiB wire budget as the other results) and returns a bounded summary:

| Reader | Output | Summary |
|--------|--------|---------|
| `energy_histogram` | `*_energyHistogram_*.dat` | bins/counts (keV), window count, `max_energy_kev` |
| `emittance` | `*_emittance_*.dat` | slice positions/emittances, total, peak |
| `transition_radiation` | `*_transRad_<iter>.dat` | omega/intensity spectrum, peak |
| `phase_space` | `PhaseSpace_<sp>_<filter>_<ps>_<iter>.h5` | axis ranges, projected marginals, peak bin |
| `radiation` | `*_radAmplitudes_<iter>_0_0_0.h5` | direction-summed spectrum, peak |
| `calorimeter` | `*_calorimeter_<filter>_<iter>.h5` | yaw/pitch marginals, energy edges |
| `png` | `*_png_<axis>_<slice>_<iter>.png` | metadata only (dimensions, path); image via `export` |

All openPMD readers accept the `h5` and `bp`/`bp5` suffixes. `species_filter`
is honoured where the reader's filename carries a filter component; the
`radiation` reader has none, so a non-default `species_filter` is rejected with
`unsupported` rather than silently ignored.

`iteration` defaults to `last`. A missing PIConGPU (or, for the openPMD/image
readers, `openpmd_api`/`imageio`) degrades that reader to `reader_unavailable`;
readers degrade independently, so `png` still works without `openpmd_api`.

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
  `workflow.finished` / `simulation.failed{stage}` events). `workflow.finished`
  replaces the design's premature `results.ready`: it marks the CWL workflow
  returning, not the SLURM job finishing. Per-stage/job-state events are
  deferred to upstream PR #55.
