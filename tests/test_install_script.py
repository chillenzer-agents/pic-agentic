# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Static guards for ``scripts/install-mcp.sh``.

The installer is shell, so these checks read it as text and exercise the parts
that must stay in lockstep with the code: the pinned revision and the simclient
capability assertion (B1), the fresh-room preflight (M3), and the `--help`
header trim (nit).  They do not run pip or contact a homeserver.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "install-mcp.sh"
BETA_SCRIPT = REPO / "scripts" / "beta-container-setup.sh"

#: The pre-fix base head the installer must never pin again.
STALE_PIN = "96d86a4611d239159e160092125ffe9cce33e6ac"

#: Capability markers the simulated preflight asserts; keep in sync with the
#: ``required`` mapping inside ``preflight_installed``.
REQUIRED_MARKERS = {
    "M3 result handling": "SimulationType.RESULT_COMMAND",
    "non-blocking dispatch": "_dispatch_message",
    "build gate": "_build_semaphore",
    "per-run stderr capture": "_CapturingStderr",
}


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_default_pin_is_valid_and_not_the_stale_base() -> None:
    match = re.search(r'PIC_AGENTIC_PIN="\$\{PIC_AGENTIC_PIN:-([0-9a-f]+)\}"', _script_text())
    assert match, "PIC_AGENTIC_PIN default not found"
    pin = match.group(1)
    assert len(pin) == 40, f"pin must be 40 hex chars, got {len(pin)}"
    assert pin != STALE_PIN, "installer pins the pre-fix base head"


def test_installer_capability_markers_exist_in_head_source() -> None:
    """Every marker the installer requires is present in this revision."""
    client = (REPO / "src/pic_agentic/simclient/client.py").read_text(encoding="utf-8")
    simulation = (REPO / "src/pic_agentic/simclient/simulation.py").read_text(encoding="utf-8")
    source = client + simulation
    for label, marker in REQUIRED_MARKERS.items():
        assert marker in source, f"installer requires {label!r} ({marker}) but HEAD lacks it"


def test_stale_base_source_would_be_rejected() -> None:
    """The base head lacks three markers, so the assertion catches it."""
    git = shutil.which("git")
    if git is None:
        return
    base_client = subprocess.run(
        [git, "show", f"{STALE_PIN}:src/pic_agentic/simclient/client.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    base_sim = subprocess.run(
        [git, "show", f"{STALE_PIN}:src/pic_agentic/simclient/simulation.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if base_client.returncode != 0 or base_sim.returncode != 0:
        # A shallow clone may not carry the base commit; the positive test above
        # still pins the contract.
        return
    source = base_client.stdout + base_sim.stdout
    missing = [label for label, marker in REQUIRED_MARKERS.items() if marker not in source]
    assert missing, "the stale base unexpectedly satisfies the capability assertion"


def test_preflight_installed_is_wired_into_install_and_check() -> None:
    text = _script_text()
    assert "preflight_installed || die" in text, "install path does not assert the installed revision"
    assert "if preflight_installed; then" in text, "`--check` does not report the installed revision"


def test_room_preflight_accepts_either_role_and_warns_on_empty_room() -> None:
    text = _script_text()
    # No server-role-only filter remains.
    assert 'm.sender_role == "mcpserver"' not in text
    assert "no RCP messages in the room yet" in text, "fresh-room warn path missing"
    assert "m.verify(c.rcp_secret)" in text, "verification over all roles missing"


def test_scripts_agree_on_the_mcp_client_budget() -> None:
    """D1: both setup scripts must default MCP_TIMEOUT_MS to the same value."""
    install = _script_text()
    beta = BETA_SCRIPT.read_text(encoding="utf-8")
    pattern = r'MCP_TIMEOUT_MS="\$\{PIC_AGENTIC_MCP_TIMEOUT_MS:-(\d+)\}"'
    install_match = re.search(pattern, install)
    beta_match = re.search(pattern, beta)
    assert install_match, "install-mcp.sh does not define the overridable budget default"
    assert beta_match, "beta-container-setup.sh does not define the overridable budget default"
    assert install_match.group(1) == beta_match.group(1), (
        f"budget mismatch: install-mcp.sh={install_match.group(1)} beta-container-setup.sh={beta_match.group(1)}"
    )


def test_scripts_stamp_the_budget_into_the_env_var() -> None:
    """D1: the same budget the entry timeout uses feeds the server env var."""
    for script in (_script_text(), BETA_SCRIPT.read_text(encoding="utf-8")):
        assert '"PIC_AGENTIC_MCP_TIMEOUT_MS": str(timeout_ms)' in script, (
            "the registration does not pass the budget to the server env"
        )
        assert '"timeout": int(timeout_ms)' in script, "the entry timeout does not use the shared budget"


def test_shipped_budget_exceeds_the_server_wait_ceiling() -> None:
    """The shipped budget must fit every accepted wait, so no default is refused.

    The server refuses a wait whose ``timeout_s`` plus the safety skew exceeds
    the budget (``WAIT_CLIENT_TIMEOUT_SKEW_S``); the same server caps
    ``timeout_s`` at ``MAX_WAIT_TIMEOUT_S``.  A shipped budget at or below that
    ceiling plus the skew would make the documented default (1800 s) -- or even
    an explicit in-ceiling wait -- fail 100% of the time under the shipped
    install (D1 follow-up).
    """
    from pic_agentic.server.simulation import MAX_WAIT_TIMEOUT_S, WAIT_CLIENT_TIMEOUT_SKEW_S

    pattern = r'MCP_TIMEOUT_MS="\$\{PIC_AGENTIC_MCP_TIMEOUT_MS:-(\d+)\}"'
    for text in (_script_text(), BETA_SCRIPT.read_text(encoding="utf-8")):
        match = re.search(pattern, text)
        assert match, "a setup script does not define the overridable budget default"
        budget_s = int(match.group(1)) / 1000.0
        assert budget_s > MAX_WAIT_TIMEOUT_S + WAIT_CLIENT_TIMEOUT_SKEW_S, (
            f"shipped budget {budget_s:g} s does not fit the server's "
            f"MAX_WAIT_TIMEOUT_S {MAX_WAIT_TIMEOUT_S:g} s + skew "
            f"{WAIT_CLIENT_TIMEOUT_SKEW_S:g} s"
        )


def test_help_does_not_print_the_set_euo_pipefail_line() -> None:
    bash = shutil.which("bash")
    if bash is None:
        return
    result = subprocess.run([bash, str(SCRIPT), "--help"], capture_output=True, text=True, check=True)
    assert "set -euo pipefail" not in result.stdout
