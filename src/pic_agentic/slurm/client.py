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
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

SAFE_CHARSET = re.compile(r"^[A-Za-z0-9._/-]+$")
SUBMITTED_RE = re.compile(r"Submitted batch job (\d+)")
STATE_RE = re.compile(r"JobState=(\w+)")

#: ``sacct --parsable2`` accounting columns (fixed ``--format`` order):
#: ``JobID|ElapsedRaw|AllocCPUS|TRESUsageInTot``.
_MIN_ACCOUNTING_FIELDS = 3
_COL_ELAPSED_RAW = 1
_COL_ALLOC_CPUS = 2
_COL_TRES = 3

#: Signals the client will deliver.  A fixed set, so a wire string can never be
#: interpolated into ``scontrol``/``scancel``.  ``TERM``/``ALRM`` are the M3
#: stop paths (PIConGPU's ``SIGTERM``/``SIGALRM`` handlers).
ALLOWED_SIGNALS = frozenset({"USR1", "USR2", "KILL", "TERM", "ALRM"})


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


class JobInfo(BaseModel):
    """A snapshot of one SLURM job."""

    job_id: int
    state: SlurmJobState
    exit_code: int | None = None


class JobAccounting(BaseModel):
    """Actual resource usage reported by ``sacct`` for one finished job."""

    job_id: int
    core_hours: float = 0.0
    gpu_hours: float = 0.0
    is_gpu: bool = False


def parse_accounting(output: str) -> dict[int, JobAccounting]:
    """Parse ``sacct --parsable2`` output into per-job actual usage.

    The expected columns are ``JobID|ElapsedRaw|AllocCPUS|TRESUsageInTot`` (or a
    ``--format`` subset in the same order); ``ElapsedRaw`` is seconds and
    ``AllocCPUS`` is the allocated CPU count, so
    ``core_hours = ElapsedRaw * AllocCPUS / 3600``.  A ``gres/gpu=N`` in the TRES
    usage contributes ``gpu_hours = elapsed_hours * N``.  A malformed row is
    skipped, never raised.

    Args:
        output: The raw ``sacct --parsable2`` stdout.

    Returns:
        ``{job_id: JobAccounting}`` for the base jobs that parsed.

    """
    accounting: dict[int, JobAccounting] = {}
    lines = output.splitlines()
    if not lines:
        return accounting
    # Column positions in the fixed ``--format=JobID,ElapsedRaw,AllocCPUS,
    # TRESUsageInTot`` order; ``|`` is the ``--parsable2`` delimiter.
    for line in lines[1:]:  # the first line is the header
        fields = [field.strip() for field in line.split("|")]
        if len(fields) < _MIN_ACCOUNTING_FIELDS:
            continue
        try:
            job_id = int(fields[0].split(".")[0])
        except ValueError:
            continue
        elapsed_raw = _int_or_none(fields[_COL_ELAPSED_RAW])
        alloc_cpus = _int_or_none(fields[_COL_ALLOC_CPUS])
        elapsed_hours = (elapsed_raw or 0) / 3600.0
        tres = fields[_COL_TRES] if len(fields) > _COL_TRES else ""
        gpu_count = _gpu_count(tres)
        accounting[job_id] = JobAccounting(
            job_id=job_id,
            core_hours=elapsed_hours * (alloc_cpus or 0),
            gpu_hours=elapsed_hours * gpu_count,
            is_gpu=gpu_count > 0,
        )
    return accounting


def _int_or_none(text: str) -> int | None:
    """Parse an integer field, returning None when it is not one.

    Returns:
        The integer, or None.

    """
    try:
        return int(text)
    except ValueError:
        return None


def _gpu_count(tres: str) -> int:
    """Extract ``N`` from a ``gres/gpu=N`` TRES usage field.

    Returns:
        The GPU count, or 0 when absent/malformed.

    """
    match = re.search(r"gres/gpu=(\d+)", tres)
    return int(match.group(1)) if match else 0


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

    async def job_accounting(self, job_id: int) -> JobAccounting | None:
        """Query ``sacct`` for one finished job's actual resource usage.

        Uses ``--parsable2`` with a fixed, argument-array command (never a
        shell), so the job id is an integer and cannot be interpolated.  A
        failure (``sacct`` missing, no accounting row, malformed output) is
        returned as ``None`` rather than raised: actual-cost reconciliation is
        best-effort and must never block a tick.

        Args:
            job_id: The SLURM job id.

        Returns:
            The parsed accounting, or None when unavailable.

        """
        argv = [
            self._exe("sacct"),
            "-j",
            str(job_id),
            "--parsable2",
            "--noheader",
            "--format=JobID,ElapsedRaw,AllocCPUS,TRESUsageInTot",
        ]
        try:
            rc, stdout, _stderr = await self._run(argv)
        except SlurmError:
            return None
        if rc != 0:
            return None
        # --noheader is requested, but accept a header line too (older sacct).
        rows = parse_accounting("JobID|ElapsedRaw|AllocCPUS|TRESUsageInTot\n" + stdout)
        return rows.get(job_id)

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
            signal: One of :data:`ALLOWED_SIGNALS`.
            batch: Whether to target the batch step (``--batch``).

        Raises:
            SlurmError: If the signal is not allowed or ``scancel`` fails.

        """
        if signal not in ALLOWED_SIGNALS:
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

    async def signal_job(self, job_id: int, signal: str) -> str:
        """Deliver a signal to *all* of a job's tasks via ``scontrol signal``.

        This is the M3 control path: PIConGPU installs its handlers in every MPI
        rank and each rank must receive the signal, which ``scontrol signal``
        (unlike ``scancel --signal``) does.  ``signal`` is checked against the
        fixed allow-list, so a wire value never reaches the command.

        Args:
            job_id: The SLURM job id.
            signal: One of :data:`ALLOWED_SIGNALS`.

        Returns:
            The combined stdout and stderr of ``scontrol``.

        Raises:
            SlurmError: If the signal is not allowed or ``scontrol`` fails.

        """
        if signal not in ALLOWED_SIGNALS:
            msg = f"unsupported signal: {signal}"
            raise SlurmError(msg)
        argv = [self._exe("scontrol"), "signal", signal, str(job_id)]
        rc, stdout, stderr = await self._run(argv)
        if rc != 0:
            msg = f"scontrol signal failed ({rc}): {stderr.strip() or stdout.strip()}"
            raise SlurmError(msg)
        return (stdout + stderr).strip()

    async def cancel_job(self, job_id: int) -> str:
        """Hard-cancel a job via ``scontrol cancel``.

        Args:
            job_id: The SLURM job id.

        Returns:
            The combined stdout and stderr of ``scontrol``.

        Raises:
            SlurmError: If ``scontrol`` fails.

        """
        argv = [self._exe("scontrol"), "cancel", str(job_id)]
        rc, stdout, stderr = await self._run(argv)
        if rc != 0:
            msg = f"scontrol cancel failed ({rc}): {stderr.strip() or stdout.strip()}"
            raise SlurmError(msg)
        return (stdout + stderr).strip()
