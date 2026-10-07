<!--
SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf

SPDX-License-Identifier: CC-BY-4.0
-->

# Beta-test prompts

Two prompts for beta-testing the pic-agentic MCP server with an agent that has
**not** seen this repository. They are deliberately framed the way a domain
scientist would phrase a request: the agent gets a goal and the tool surface,
not a recipe. Setup (cluster, transport, configuration, the pinned PIConGPU
build) is assumed to be done by a human beforehand; the prompts only exercise
what the agent can do from the tool surface plus the PIConGPU documentation.

The server's seeded `instructions` point at the PyPIConGPU documentation
(`python_package/foundations/defining_simulation`, published on readthedocs)
and at `lib/python/examples/` in the PIConGPU source tree. Prompt 1 relies on
that pointer; prompt 2 is meant to test whether the agent can compose a new
study from the documentation without a verbatim snippet to copy.

Caveat for the pointer (picongpu upstream docs issue, not fixed here): the page's
focal example is empty as written, and not only because it defines no plasma
species. It reuses the LWFA tutorial's pulse timing (`PULSE_INIT=15`, so the
pulse starts ~11 µm in front of the box) in a 100-step run, and the pulse never
reaches the gas (whose plateau sits ~80 µm downstream; bridging that needs
~2000 steps). Folding in the plasma
species is necessary but not sufficient; the `instructions` now say both. A
run that finishes but whose only numeric artifact reads all-zero is the
`suspect` health signal below.

The `instructions` also warn that the readthedocs pages may describe a
**newer release than the installed pin**, and carry a minimal version-matched
snippet built from the pinned classes. The beta-5 agent copied
`FieldDiagnostic`/`PhaseSpaceDiagnostic`/`write_input_file` from the online pages
and then had to reverse-engineer `site-packages`; the pin uses
`picmi.diagnostics.NativeFieldDump`/`DerivedFieldDump`/`PhaseSpace`/
`EnergyHistogram`/`FieldEnergyMonitor`. The installed package is authoritative.
The seeded instructions also carry an inline
`FieldEnergyMonitor(period=TimeStepSpec[::20, -1])` snippet (the beta-6 agent
reverse-engineered that inclusive slice syntax from source), and the
constraints a Runner-spec / multi-GPU study needs: the spec is a *rendered snapshot* with
denormalized fields, so patching only `sim.grid.cell_cnt` (or only `cell_size`)
is a box-size, not a resolution, change — and box-size sweeps are allowed; a
fixed-box resolution sweep must instead co-vary `cell_size`, `cell_cnt` and
`cell_depth` together with a CFL-consistent `delta_t_si` and matching
`time_steps`; finer `dx` needs a CFL-consistent `delta_t_si`/step count or the
compile fails on the Yee CFL `static_assert`; a super cell smaller than the pin's
default `picongpu_super_cell_size` (`(8, 8, 4)` in 3D, `(16, 16)` in 2D) fails
the build with the Esirkepov "supercell or number of guard supercells is too
small for stencil" `static_assert`, so it is not a free performance knob; and a
laser's Huygens surface must sit outside the 12-cell default PML absorber (the
`GaussianLaser` 16-cell default is right; 8 cells segfaults at step 0). To spread
one run over several GPUs (e.g. to beat a single-GPU memory limit) the grid
carries `picongpu_n_gpus` (a bare int `N` means `(1, N, 1)`, or a per-axis
tuple); each GPU is one MPI rank and the pinned `gpu-v100` template hosts 4 per
node, so a request over the partition's node ceiling stays `PENDING` with Slurm
reason `PartitionNodeLimit` rather than failing — the ceiling is not knowable
from the spec. Because the pin's `TimeStepSpec` is a plain class, not a
pydantic model, `model_json_schema()` is
unavailable for `TimeStepSpec`-backed diagnostics; the instructions point at the
pinned classes under `lib/python/picongpu/picmi/diagnostics/` instead.

Independently of *why* a run is empty, a completed run whose only numeric plugin
artifact reads all-zero is now a first-class **health signal**, not silence:
`advance_agenda` reports it under `suspects`, `agenda_status` under
`suspects`/`suspect_count` and a `suspect` field per leaf, `fleet_status` as a
`suspect` summary count plus a `suspect` alert, and the leaf's `done` callback
carries the same warning under its `suspect` key. A run that finished but
produced no particles is
therefore never presented as a clean success — a "successful-but-empty" run
should be treated as inconclusive physics and re-checked, not reported as a
result.

## Prompt 1 — parameter scan from the documented LWFA example

> I want to know how the laser focal position affects electron acceleration in
> a laser wakefield accelerator. Set up and run a small scan over three focal
> positions around the middle of the simulation box and tell me which one
> produces the most electrons in the high-energy tail of the energy spectrum.
> I don't care about the intermediate bookkeeping — submit it, follow it, and
> come back with a clear recommendation and the numbers behind it.

What a successful run looks like:

- The agent finds a documented LWFA setup (the docs' multi-simulation example
  reuses the LWFA tutorial grid and Gaussian laser) and turns it into a PICMI
  script.
- It obtains a base Runner spec from that script (`build_spec`), then creates a
  campaign that scans the **nested, list-indexed** laser focal-position
  component across the three values (`create_campaign` with a `patch_path` such
  as `sim.laser.0.focus_pos_si.1.component`).
- It advances the campaign, follows the leaves to completion, takes and records
  the callback analyses, and concludes with a ranked recommendation.
- The usable answer is a per-position electron count and which focal position
  maximises it.

Verification notes for the maintainer: the focal-position sweep is only
reachable because `create_campaign`'s dotted `patch_path` accepts list indices
(`sim.laser.0....`); without that, the agent has to replace the whole laser list,
which is a finding rather than a failure.

## Prompt 2 — open-ended study with no copy-paste source

Prompt 2 is intentionally a study that appears in the documentation only in
pieces, so the agent must compose it:

> Run a small numerical convergence check on a warm-plasma-like setup: hold the
> physical size of the box fixed and run a few resolutions, then tell me at
> which resolution an integrated field quantity of your choice stops changing
> appreciably. Give me the trend across resolutions and the point where it
> flattens out.

The expected usable answer is a short trend (the quantity against resolution)
plus the resolution at which it converges within a stated tolerance. There is
no single documentation page to copy: the agent must pick a concrete setup from
the docs, choose (and justify) an integrated quantity, express the resolution
sweep, and interpret the result.

This resolution sweep cannot be one `create_campaign` `patch_path`: holding the
physical box size fixed means the grid cell count **and** `time_steps` must
co-vary, i.e. two spec nodes. The intended route is the `add_agenda_leaf` escape
hatch — create the campaign **empty** (`create_campaign(name=...)` with no
`patch_path`, so no identity patch is invented), then add one leaf per
resolution with its own whole spec (staged by `spec_path` if large). A
transcript in which the agent abandons the campaign machinery entirely and
hand-runs submissions is the H5 finding this documents against.

## Checks common to both prompts

- The agent never needs to edit pic-agentic or the PIConGPU sources; it works
  through the MCP tools only.
- The campaign survives the agent reconnecting (the server persists it), so the
  agent may advance the scan across more than one session.
- At the end the agent can produce the campaign provenance (`campaign_provenance`)
  and, if asked, a CWL export (`export_agenda_cwl`) — these are how "reproducible
  provenance" is checked.
