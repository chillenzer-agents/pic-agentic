# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Unit tests for the agenda resource budget."""

from __future__ import annotations

import pytest

from pic_agentic.agenda.budget import (
    Budget,
    BudgetExceededError,
    BudgetUsage,
    ResourceRequest,
    check_admission,
)


def _request(**kw: object) -> ResourceRequest:
    return ResourceRequest(**kw)


def test_no_caps_admits_everything() -> None:
    check_admission(Budget(), BudgetUsage(), _request(est_core_hours=1e9, is_gpu=True))


def test_total_jobs_cap_boundary() -> None:
    budget = Budget(max_total_jobs=2)
    check_admission(budget, BudgetUsage(jobs_submitted=1), _request())
    # Second submission reaches the cap exactly: still allowed.
    with pytest.raises(BudgetExceededError, match="job budget exhausted"):
        check_admission(budget, BudgetUsage(jobs_submitted=2), _request())


def test_concurrency_cap() -> None:
    budget = Budget(max_concurrent_jobs=4)
    check_admission(budget, BudgetUsage(jobs_running=3), _request())
    with pytest.raises(BudgetExceededError, match="concurrency cap"):
        check_admission(budget, BudgetUsage(jobs_running=4), _request())


def test_core_hours_cap_inclusive_boundary() -> None:
    budget = Budget(max_core_hours=100.0)
    # Exactly at the cap is allowed.
    check_admission(budget, BudgetUsage(core_hours=90.0), _request(est_core_hours=10.0))
    with pytest.raises(BudgetExceededError, match="core-hour budget"):
        check_admission(budget, BudgetUsage(core_hours=90.0), _request(est_core_hours=10.1))


def test_gpu_hours_cap_only_applies_to_gpu_requests() -> None:
    budget = Budget(max_gpu_hours=50.0)
    usage = BudgetUsage(gpu_hours=50.0)
    # A non-GPU request is unaffected by the GPU cap.
    check_admission(budget, usage, _request(est_gpu_hours=10.0, is_gpu=False))
    with pytest.raises(BudgetExceededError, match="gpu-hour budget"):
        check_admission(budget, usage, _request(est_gpu_hours=1.0, is_gpu=True))


def test_partition_allowlist() -> None:
    budget = Budget(allowed_partitions=["cpu-genoa"])
    check_admission(budget, BudgetUsage(), _request(partition="cpu-genoa"))
    with pytest.raises(BudgetExceededError, match="not allowed"):
        check_admission(budget, BudgetUsage(), _request(partition="gpu-v100"))


def test_budget_and_usage_round_trip() -> None:
    budget = Budget(max_core_hours=1.5, allowed_partitions=["gpu-v100"])
    usage = BudgetUsage(core_hours=0.5, jobs_submitted=3, jobs_running=1)
    assert Budget.model_validate_json(budget.model_dump_json()) == budget
    assert BudgetUsage.model_validate_json(usage.model_dump_json()) == usage
