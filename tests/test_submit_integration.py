# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Optional integration test against a real pinned PIConGPU install.

Runs only when PIConGPU is importable (the ``sim`` extra).  The default
offline suite skips it, so CI without the pin stays fast; a developer or the
cluster venv gets a genuine ``Runner`` round-trip and a real
``SchemaHash``/payload-hash agreement check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pic_agentic.protocol.simulation import SimulationPayload, simulation_spec_from_runner_dump
from pic_agentic.version import local_provenance, picongpu_revision, picongpu_version, runner_schema_hash

picongpu = pytest.importorskip("picongpu")

FIXTURE = Path(__file__).parent / "fixtures" / "pypicongpu_runner.json"


@pytest.mark.integration
def test_real_provenance_is_available() -> None:
    assert picongpu_version()
    assert runner_schema_hash()
    assert local_provenance()["schema_hash"] == runner_schema_hash()
    # The revision is best-effort (editable installs have no vcs_info), so only
    # assert it is a string.
    assert isinstance(picongpu_revision(), str)


@pytest.mark.integration
def test_fixture_validates_against_real_runner() -> None:
    from picongpu.pypicongpu.runner import Runner

    dump = json.loads(FIXTURE.read_text())
    runner = Runner.model_validate(dump)
    assert runner.model_dump(mode="json") == dump
    payload = SimulationPayload.build(
        picongpu_version=picongpu_version(),
        picongpu_revision=picongpu_revision(),
        schema_hash=runner_schema_hash(),
        runner_dump=dump,
    )
    spec = simulation_spec_from_runner_dump(dump)
    assert payload.simulation == spec
    # The payload hash is stable across a validation round trip.
    reloaded = SimulationPayload.model_validate_json(payload.model_dump_json(exclude_computed_fields=True))
    assert reloaded.payload_hash == payload.payload_hash
