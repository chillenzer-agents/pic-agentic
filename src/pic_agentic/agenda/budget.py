# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Declarative resource governance for agendas.

Extraction-ready: stdlib + ``pydantic`` only.

A :class:`Budget` states the hard caps for one agenda; a :class:`BudgetUsage`
records what has actually been consumed (it serialises alongside the agenda so
a resumed agenda keeps its accounting).  :func:`check_admission` is the single
gate that every submission must pass *before* it reaches SLURM: it raises
:class:`BudgetExceededError` when a request would breach a cap.  It has no side
effects, so it is trivially testable and cannot itself corrupt the accounting.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BudgetExceededError(RuntimeError):
    """Raised when a submission would breach an agenda's resource budget."""


class Budget(BaseModel):
    """Hard caps for one agenda.  ``None`` means "no limit"."""

    model_config = ConfigDict(extra="forbid")

    max_core_hours: float | None = None
    max_gpu_hours: float | None = None
    max_concurrent_jobs: int | None = None
    max_total_jobs: int | None = None
    #: When set, a request may only target one of these SLURM partitions.
    allowed_partitions: list[str] | None = None


class BudgetUsage(BaseModel):
    """Consumed resources, persisted with the agenda across restarts."""

    model_config = ConfigDict(extra="forbid")

    core_hours: float = 0.0
    gpu_hours: float = 0.0
    jobs_submitted: int = 0
    jobs_running: int = 0


class ResourceRequest(BaseModel):
    """The resources one prospective submission would consume."""

    model_config = ConfigDict(extra="forbid")

    est_core_hours: float = 0.0
    est_gpu_hours: float = 0.0
    partition: str | None = None
    is_gpu: bool = False


def check_admission(budget: Budget, usage: BudgetUsage, request: ResourceRequest) -> None:
    """Raise if admitting ``request`` would breach ``budget``.

    Checks are strict: a request that would reach a cap exactly is allowed (the
    cap is inclusive), the one after it is not.

    Args:
        budget: The agenda's hard caps.
        usage: Resources already consumed.
        request: The prospective submission.

    Raises:
        BudgetExceededError: If any cap would be exceeded.

    """
    if budget.max_total_jobs is not None and usage.jobs_submitted + 1 > budget.max_total_jobs:
        msg = (
            f"job budget exhausted: {usage.jobs_submitted} submitted, "
            f"cap {budget.max_total_jobs} (this request would exceed it)"
        )
        raise BudgetExceededError(msg)
    if budget.max_concurrent_jobs is not None and usage.jobs_running + 1 > budget.max_concurrent_jobs:
        msg = (
            f"concurrency cap reached: {usage.jobs_running} running, "
            f"cap {budget.max_concurrent_jobs} (this request would exceed it)"
        )
        raise BudgetExceededError(msg)
    if budget.max_core_hours is not None and usage.core_hours + request.est_core_hours > budget.max_core_hours:
        remaining = budget.max_core_hours - usage.core_hours
        msg = (
            f"core-hour budget exhausted: {remaining:g} h remaining, "
            f"request needs {request.est_core_hours:g} h (cap {budget.max_core_hours:g} h)"
        )
        raise BudgetExceededError(msg)
    if (
        budget.max_gpu_hours is not None
        and request.is_gpu
        and usage.gpu_hours + request.est_gpu_hours > budget.max_gpu_hours
    ):
        remaining = budget.max_gpu_hours - usage.gpu_hours
        msg = (
            f"gpu-hour budget exhausted: {remaining:g} h remaining, "
            f"request needs {request.est_gpu_hours:g} h (cap {budget.max_gpu_hours:g} h)"
        )
        raise BudgetExceededError(msg)
    if budget.allowed_partitions is not None and request.partition not in budget.allowed_partitions:
        msg = (
            f"partition {request.partition!r} is not allowed; "
            f"permitted: {', '.join(budget.allowed_partitions) or '(none)'}"
        )
        raise BudgetExceededError(msg)


__all__ = [
    "Budget",
    "BudgetExceededError",
    "BudgetUsage",
    "ResourceRequest",
    "check_admission",
]
