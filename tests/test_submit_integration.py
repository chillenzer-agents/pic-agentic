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
def test_schema_hash_is_path_independent() -> None:
    """The schema hash must not depend on the install prefix.

    The raw ``Runner`` schema embeds the absolute package template path, which
    differs between the server venv and the cluster venv even for the same
    commit; the normalised hash must drop it.
    """
    import hashlib

    from picongpu.pypicongpu.runner import Runner

    from pic_agentic.rcp.crypto import canonical_bytes
    from pic_agentic.version import normalise_schema_paths

    raw = Runner.model_json_schema()
    # Every absolute-path string in the raw schema (the template_dir default).
    assert any(
        isinstance(value, str) and value.startswith("/") for value in _iter_schema_values(raw) if isinstance(value, str)
    )
    # Normalising twice is idempotent and yields the same hash as the helper.
    normalised = normalise_schema_paths(raw)
    assert normalise_schema_paths(normalised) == normalised
    assert hashlib.sha256(canonical_bytes(normalised)).hexdigest() == runner_schema_hash()


def _iter_schema_values(node: object):
    if isinstance(node, dict):
        for value in node.values():
            yield from _iter_schema_values(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_schema_values(value)
    else:
        yield node


@pytest.mark.integration
def test_nested_unknown_field_is_rejected(tmp_path: Path) -> None:
    """A field outside the pinned schema must not be silently dropped."""
    from pic_agentic.simclient.simulation import (
        SimulationErrorCode,
        SimulationExecutionError,
        SubmitConfig,
        runner_from_payload,
    )

    dump = json.loads(FIXTURE.read_text())
    dump["sim"]["totally_unknown_key"] = 123
    payload = SimulationPayload.build(
        picongpu_version=picongpu_version(),
        picongpu_revision=picongpu_revision(),
        schema_hash=runner_schema_hash(),
        runner_dump=dump,
    )
    config = SubmitConfig(message_dir=tmp_path, setup_root=tmp_path)
    with pytest.raises(SimulationExecutionError) as excinfo:
        runner_from_payload(payload, config, "deadbeef")
    assert excinfo.value.code is SimulationErrorCode.UNSUPPORTED


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
