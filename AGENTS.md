# AGENTS.md

## Commands

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest        # test suite
uvx ruff check src tests          # lint
uvx ruff format src tests         # format
uvx pre-commit run --all-files    # all hooks
```

## Conventions

- Python 3.10+; pinned runtime deps in `pyproject.toml` (`mcp==2.1.1`,
  `matrix-nio==0.26.0`). Do not float these without checking the API.
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
