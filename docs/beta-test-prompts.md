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
focal example defines no plasma species, so it produces an empty spectrum as
written. The `instructions` now say the example needs the LWFA tutorial's plasma
species folded in for a non-empty result.

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

## Checks common to both prompts

- The agent never needs to edit pic-agentic or the PIConGPU sources; it works
  through the MCP tools only.
- The campaign survives the agent reconnecting (the server persists it), so the
  agent may advance the scan across more than one session.
- At the end the agent can produce the campaign provenance (`campaign_provenance`)
  and, if asked, a CWL export (`export_agenda_cwl`) — these are how "reproducible
  provenance" is checked.
