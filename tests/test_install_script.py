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


def test_help_does_not_print_the_set_euo_pipefail_line() -> None:
    bash = shutil.which("bash")
    if bash is None:
        return
    result = subprocess.run([bash, str(SCRIPT), "--help"], capture_output=True, text=True, check=True)
    assert "set -euo pipefail" not in result.stdout
