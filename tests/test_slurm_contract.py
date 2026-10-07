# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Contract tests for the SLURM seam against the fake CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from pic_agentic.slurm import SlurmClient, SlurmError
from pic_agentic.slurm.client import validate_shared_path

FAKE_BIN = Path(__file__).parent / "fake_slurm"


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SLURM_STATE", str(tmp_path / "state"))
    return tmp_path


async def test_parsable_job_id(fake_env) -> None:
    msg = fake_env / "m.txt"
    msg.write_text("hello")
    client = SlurmClient(bin_dir=str(FAKE_BIN))
    job_id = await client.submit_wrap_cat(str(msg), str(fake_env / "o.txt"))
    assert job_id > 0
    assert (fake_env / "o.txt").read_text() == "hello"


async def test_wait_for_job_reaches_completed(fake_env) -> None:
    msg = fake_env / "m.txt"
    msg.write_text("done")
    client = SlurmClient(bin_dir=str(FAKE_BIN))
    job_id = await client.submit_wrap_cat(str(msg), str(fake_env / "o.txt"))
    info = await client.wait_for_job(job_id, timeout_s=10, interval_s=0.05)
    assert info.state.value == "COMPLETED"
    assert info.state.terminal


async def test_job_info_surfaces_the_pending_reason(fake_env) -> None:
    """F3: the scheduler reason rides on :class:`JobInfo`, not just the state.

    A job stuck at ``PENDING (PartitionNodeLimit)`` is indistinguishable from
    one merely waiting its turn if only the state is reported; the fake
    ``scontrol`` walks the ``Reason=`` field the way the real one does.
    """
    state_dir = fake_env / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "4701.state").write_text("PENDING 0 PartitionNodeLimit\n", encoding="utf-8")
    client = SlurmClient(bin_dir=str(FAKE_BIN))
    info = await client.job_info(4701)
    assert info.state.value == "PENDING"
    assert info.reason == "PartitionNodeLimit"
    assert info.permanent is True


async def test_job_info_reason_none_is_not_a_reason(fake_env) -> None:
    """Slurm's ``Reason=None`` sentinel maps to ``None``, not the string."""
    state_dir = fake_env / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "4702.state").write_text("RUNNING\n", encoding="utf-8")
    client = SlurmClient(bin_dir=str(FAKE_BIN))
    info = await client.job_info(4702)
    assert info.reason is None
    assert info.permanent is False


@pytest.mark.parametrize("reason", ["Resources", "Priority", "QOSMaxJobsPerUserLimit", None, "None"])
def test_transient_reasons_are_not_permanent(reason: str | None) -> None:
    """Only the curated permanent set is flagged; a wait reason must not be."""
    from pic_agentic.slurm.client import is_permanent_reason

    assert is_permanent_reason(reason) is False


@pytest.mark.parametrize("reason", ["PartitionNodeLimit", "PartitionTimeLimit", "MaxNodes", "PartitionConfig"])
def test_partition_limit_reasons_are_permanent(reason: str) -> None:
    from pic_agentic.slurm.client import is_permanent_reason

    assert is_permanent_reason(reason) is True


async def test_parse_human_output_fallback() -> None:
    assert SlurmClient.parse_job_id("Submitted batch job 12345") == 12345
    assert SlurmClient.parse_job_id("12345\n") == 12345
    assert SlurmClient.parse_job_id("nonsense") is None


async def test_signal_requires_fixed_set(fake_env) -> None:
    client = SlurmClient(bin_dir=str(FAKE_BIN))
    with pytest.raises(SlurmError):
        await client.signal(4701, "INT")
    with pytest.raises(SlurmError):
        await client.signal(4701, "9; rm -rf /")


async def test_cancel_modes(fake_env) -> None:
    client = SlurmClient(bin_dir=str(FAKE_BIN))
    await client.cancel(4701, "graceful")
    await client.cancel(4701, "hard")
    with pytest.raises(SlurmError):
        await client.cancel(4701, "nonsense")


async def test_control_uses_scancel_not_scontrol(monkeypatch) -> None:
    """The M3 delivery commands must be real Slurm (``scancel``, not ``scontrol``).

    Regression for the live finding: the M3 contract assumed ``scontrol signal``
    and ``scontrol cancel``, neither of which exists in Slurm; the cluster
    rejected ``checkpoint`` with "invalid keyword: signal".
    """
    recorded: list[list[str]] = []

    async def fake_run(self: SlurmClient, argv: list[str], *, timeout_s: float | None = None):
        recorded.append(argv)
        return 0, "", ""

    monkeypatch.setattr(SlurmClient, "_run", fake_run)
    client = SlurmClient()  # empty bin_dir: _exe returns the bare name
    await client.signal_job(4701, "USR1")
    await client.cancel_job(4701)
    # No --batch on the step signals: the M3 path must reach the srun ranks,
    # not just the (untrapped) batch shell.
    assert recorded[0] == ["scancel", "--signal=USR1", "4701"]
    assert recorded[1] == ["scancel", "4701"]
    recorded.clear()
    await client.cancel(4701, "graceful")
    await client.cancel(4701, "hard")
    assert recorded[0] == ["scancel", "--signal=USR2", "4701"]
    assert recorded[1] == ["scancel", "--signal=KILL", "4701"]


def test_validate_shared_path_accepts_inside_base(tmp_path) -> None:
    base = tmp_path / "shared"
    base.mkdir()
    target = base / "msg" / "a.txt"
    assert validate_shared_path(str(target), str(base)) == str(target.resolve())


@pytest.mark.parametrize(
    "path",
    [
        "relative/path.txt",
        "/etc/passwd",
        "/tmp; rm -rf /",
        "/tmp/$(whoami)",
    ],
)
def test_validate_shared_path_rejects_unsafe(tmp_path, path) -> None:
    base = tmp_path / "shared"
    base.mkdir()
    with pytest.raises(SlurmError):
        validate_shared_path(path, str(base))


def test_validate_shared_path_rejects_escape_via_dotdot(tmp_path) -> None:
    base = tmp_path / "shared"
    base.mkdir()
    with pytest.raises(SlurmError):
        validate_shared_path(str(base / ".." / "outside.txt"), str(base))
