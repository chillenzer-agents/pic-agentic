<!--
SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf

SPDX-License-Identifier: CC-BY-4.0
-->

# AGENTS.md

## Commands

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest        # test suite (offline subset)
uvx ruff check src tests scripts  # lint (select = ALL)
uvx ruff format src tests scripts # format
uvx pre-commit run --all-files    # all hooks (ruff, reuse, hygiene)
```

## Conventions

- Python 3.11+ (see `.python-version`); pinned runtime deps in `pyproject.toml`
  (`mcp==2.1.1`, `matrix-nio==0.26.0`). Do not float these without checking the
  API.
- **Ruff is the only Python linter/formatter** and runs with `select = ["ALL"]`
  plus `preview`. Add to the `ignore` ledger in `pyproject.toml` only with a
  written reason; never silence a rule inline without a comment.
- **REUSE compliance is mandatory**: every new file gets an SPDX header (the
  copyright tag plus a license tag); use `reuse annotate` or add an
  `[[annotations]]` block in `REUSE.toml` when the file cannot carry a comment.
  Run `uvx reuse lint` before pushing.
- License split: **code is MIT** (see `COPYING`); **prose/docs are CC-BY-4.0**
  (see `LICENSE.md`). Use the matching SPDX identifier in new files.
- Never interpolate LLM/tool input into a shell command. The only values that
  may reach a shell are server-generated identifiers with the safe charset
  `[A-Za-z0-9._/-]` or absolute paths to server-controlled files.
- Secrets never appear in repository files, logs, or tool output.
- Tests must not require a cluster or a homeserver: use `MemoryTransport` and
  `tests/fake_slurm/`. The live-Synapse E2E test skips when unavailable.

## Context

- The authoritative design is `TASK-14-MCP-DESIGN.md` on the
  `task-14-mcp-server-design` branch of `chillenzer-agents/picongpu`
  (PR #50). Deliberate deviations are listed in `README.md`.
- `TASK-14-REVIEW.md` / `TASK-14-RESPONSE.md` on the same branch record the
  QA cycle for the design.
