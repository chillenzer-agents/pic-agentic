# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Thin, injection-safe wrappers around ``sbatch``/``scontrol``/``scancel``.

The ``hello`` path never interpolates a payload into a shell string.  The one
value passed to ``sbatch`` is an absolute path to a file the simclient wrote;
the job body is ``cat '<path>'``, and the path is validated against a safe
charset and a configured base directory before use.
"""

from __future__ import annotations

import asyncio
import re
import shlex
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

SAFE_CHARSET = re.compile(r"^[A-Za-z0-9._/-]+$")
SUBMITTED_RE = re.compile(r"Submitted batch job (\d+)")
STATE_RE = re.compile(r"JobState=(\w+)")


class SlurmError(RuntimeError):
    """Raised when a SLURM command fails or its output cannot be parsed."""


class SlurmJobState(StrEnum):
    """Subset of SLURM job states the RCP layer reacts to."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"

    @property
    def terminal(self) -> bool:
        """Whether the job will not change state again."""
        return self in {
            SlurmJobState.COMPLETED,
            SlurmJobState.FAILED,
            SlurmJobState.CANCELLED,
            SlurmJobState.TIMEOUT,
        }


@dataclass
class JobInfo:
    """A snapshot of one SLURM job."""

    job_id: int
    state: SlurmJobState
    exit_code: int | None = None


def validate_shared_path(path: str, base_dir: str) -> str:
    """Validate that ``path`` is an absolute path under ``base_dir``.

    Args:
        path: Candidate path (must match the safe charset).
        base_dir: Directory the path must stay inside.

    Returns:
        The resolved absolute path.

    Raises:
        SlurmError: If the path is relative, unsafe, or escapes ``base_dir``.

    """
    if not SAFE_CHARSET.match(path):
        msg = f"unsafe path: {path!r}"
        raise SlurmError(msg)
    if not path.startswith("/"):
        msg = f"path must be absolute: {path!r}"
        raise SlurmError(msg)
    base = Path(base_dir).resolve()
    resolved = Path(path).resolve()
    if resolved != base and base not in resolved.parents:
        msg = f"path escapes base directory {base_dir!r}: {path!r}"
        raise SlurmError(msg)
    return str(resolved)


class SlurmClient:
    """Invoke SLURM CLIs with argument arrays (never ``shell=True``)."""

    def __init__(self, bin_dir: str = "", *, timeout_s: float = 30.0) -> None:
        """Create a client.

        Args:
            bin_dir: Directory holding the SLURM executables; empty uses PATH.
            timeout_s: Default timeout for one CLI invocation, in seconds.

        """
        self._bin = Path(bin_dir) if bin_dir else None
        self._timeout = timeout_s

    def _exe(self, name: str) -> str:
        return str(self._bin / name) if self._bin else name

    async def _run(self, argv: list[str], *, timeout_s: float | None = None) -> tuple[int, str, str]:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s or self._timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            msg = f"command timed out: {argv[0]}"
            raise SlurmError(msg) from None
        return (
            process.returncode or 0,
            stdout.decode("utf-8", "replace"),
            stderr.decode("utf-8", "replace"),
        )

    async def submit_wrap_cat(self, path: str, outfile: str) -> int:
        """Submit a job whose body is ``cat '<path>'``, redirecting to ``outfile``.

        ``path`` and ``outfile`` must already have been validated by
        :func:`validate_shared_path` (or be otherwise server-generated).

        Args:
            path: Absolute shared-filesystem path whose contents to print.
            outfile: Absolute shared-filesystem path for the job's stdout.

        Returns:
            The parsed SLURM job id.

        Raises:
            SlurmError: If ``sbatch`` fails or its output has no job id.

        """
        # The wrap body is quoted as a single argv element; the path is not
        # re-parsed by a shell on the submission side.  SLURM runs the wrap via
        # /bin/sh on the compute node, where the validated charset has no
        # metacharacters.
        wrap = f"cat {shlex.quote(path)}"
        argv = [self._exe("sbatch"), "--parsable", f"--wrap={wrap}", f"--output={outfile}"]
        rc, stdout, stderr = await self._run(argv)
        if rc != 0:
            msg = f"sbatch failed ({rc}): {stderr.strip() or stdout.strip()}"
            raise SlurmError(msg)
        job_id = self.parse_job_id(stdout)
        if job_id is None:
            msg = f"could not parse job id from sbatch output: {stdout!r}"
            raise SlurmError(msg)
        return job_id

    @staticmethod
    def parse_job_id(output: str) -> int | None:
        """Parse ``--parsable`` output, falling back to the human line.

        Args:
            output: Raw ``sbatch`` stdout.

        Returns:
            The job id, or None if the output carries none.

        """
        text = output.strip()
        if text.isdigit():
            return int(text)
        match = SUBMITTED_RE.search(output)
        return int(match.group(1)) if match else None

    async def job_info(self, job_id: int) -> JobInfo:
        """Query ``scontrol show job`` for one job.

        Args:
            job_id: The SLURM job id.

        Returns:
            The parsed job snapshot.

        Raises:
            SlurmError: If ``scontrol`` fails or reports no state.

        """
        rc, stdout, stderr = await self._run([self._exe("scontrol"), "show", "job", str(job_id)])
        if rc != 0:
            msg = f"scontrol failed ({rc}): {stderr.strip()}"
            raise SlurmError(msg)
        match = STATE_RE.search(stdout)
        if not match:
            msg = f"no JobState in scontrol output for job {job_id}"
            raise SlurmError(msg)
        state = SlurmJobState(match.group(1))
        exit_code = None
        exit_match = re.search(r"ExitCode=(\d+):(\d+)", stdout)
        if exit_match:
            exit_code = int(exit_match.group(1))
        return JobInfo(job_id=job_id, state=state, exit_code=exit_code)

    async def wait_for_job(self, job_id: int, *, timeout_s: float, interval_s: float = 5.0) -> JobInfo:
        """Poll a job until it reaches a terminal state or ``timeout_s`` passes.

        Args:
            job_id: The SLURM job id.
            timeout_s: Maximum total wait, in seconds.
            interval_s: Delay between polls, in seconds.

        Returns:
            The last observed job snapshot.

        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            info = await self.job_info(job_id)
            if info.state.terminal:
                return info
            if loop.time() >= deadline:
                return info
            await asyncio.sleep(interval_s)

    async def signal(self, job_id: int, signal: str, *, batch: bool = True) -> None:
        """Send a signal via ``scancel``; ``signal`` comes from a fixed set.

        Args:
            job_id: The SLURM job id.
            signal: One of ``USR1``, ``USR2``, ``KILL``.
            batch: Whether to target the batch step (``--batch``).

        Raises:
            SlurmError: If the signal is not allowed or ``scancel`` fails.

        """
        if signal not in {"USR1", "USR2", "KILL"}:
            msg = f"unsupported signal: {signal}"
            raise SlurmError(msg)
        argv = [self._exe("scancel"), f"--signal={signal}"]
        if batch:
            argv.append("--batch")
        argv.append(str(job_id))
        rc, _stdout, stderr = await self._run(argv)
        if rc != 0:
            msg = f"scancel failed ({rc}): {stderr.strip()}"
            raise SlurmError(msg)

    async def cancel(self, job_id: int, mode: str = "graceful") -> None:
        """Cancel a job per design section 2.4.

        Args:
            job_id: The SLURM job id.
            mode: ``graceful`` (USR2, step-boundary stop) or ``hard`` (KILL).

        Raises:
            SlurmError: If ``mode`` is unknown or the signal fails.

        """
        if mode == "graceful":
            await self.signal(job_id, "USR2")
        elif mode == "hard":
            await self.signal(job_id, "KILL", batch=False)
        else:
            msg = f"unknown cancel mode: {mode}"
            raise SlurmError(msg)
