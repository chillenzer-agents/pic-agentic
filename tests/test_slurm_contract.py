"""Contract tests for the SLURM seam against the fake CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from pic_agentic.slurm import SlurmClient, SlurmError
from pic_agentic.slurm.client import validate_shared_path

FAKE_BIN = Path(__file__).parent / "fake_slurm"


@pytest.fixture()
def fake_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SLURM_STATE", str(tmp_path / "state"))
    return tmp_path


async def test_parsable_job_id(fake_env):
    msg = fake_env / "m.txt"
    msg.write_text("hello")
    client = SlurmClient(bin_dir=str(FAKE_BIN))
    job_id = await client.submit_wrap_cat(str(msg), str(fake_env / "o.txt"))
    assert job_id > 0
    assert (fake_env / "o.txt").read_text() == "hello"


async def test_wait_for_job_reaches_completed(fake_env):
    msg = fake_env / "m.txt"
    msg.write_text("done")
    client = SlurmClient(bin_dir=str(FAKE_BIN))
    job_id = await client.submit_wrap_cat(str(msg), str(fake_env / "o.txt"))
    info = await client.wait_for_job(job_id, timeout_s=10, interval_s=0.05)
    assert info.state.value == "COMPLETED"
    assert info.state.terminal


async def test_parse_human_output_fallback():
    assert SlurmClient.parse_job_id("Submitted batch job 12345") == 12345
    assert SlurmClient.parse_job_id("12345\n") == 12345
    assert SlurmClient.parse_job_id("nonsense") is None


async def test_signal_requires_fixed_set(fake_env):
    client = SlurmClient(bin_dir=str(FAKE_BIN))
    with pytest.raises(SlurmError):
        await client.signal(4701, "TERM")
    with pytest.raises(SlurmError):
        await client.signal(4701, "9; rm -rf /")


async def test_cancel_modes(fake_env):
    client = SlurmClient(bin_dir=str(FAKE_BIN))
    await client.cancel(4701, "graceful")
    await client.cancel(4701, "hard")
    with pytest.raises(SlurmError):
        await client.cancel(4701, "nonsense")


def test_validate_shared_path_accepts_inside_base(tmp_path):
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
def test_validate_shared_path_rejects_unsafe(tmp_path, path):
    base = tmp_path / "shared"
    base.mkdir()
    with pytest.raises(SlurmError):
        validate_shared_path(path, str(base))


def test_validate_shared_path_rejects_escape_via_dotdot(tmp_path):
    base = tmp_path / "shared"
    base.mkdir()
    with pytest.raises(SlurmError):
        validate_shared_path(str(base / ".." / "outside.txt"), str(base))
