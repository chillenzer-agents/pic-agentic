# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Protocol-level tests for the M3 results surface.

Covers the frozen models/builders only; the engine behaviour lives in
``test_results_unit.py``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pic_agentic.protocol.simulation import (
    MAX_RESULT_BYTES,
    PLUGIN_READER_NAMES,
    RESULT_TEXT_MAX_BYTES,
    SLICE_MAX_POINTS,
    ResultManifest,
    ResultOp,
    ResultParams,
    ResultRef,
    SimulationType,
    build_result_ack,
    build_result_command,
)

SIM = "7f3a2b1c"
CMD_ID = "cmd-1"


def test_result_params_minimal_and_knobs_round_trip() -> None:
    params = ResultParams.model_validate(
        {
            "sim_id": SIM,
            "op": "slice",
            "path": "openPMD/fields.bp",
            "record": "E",
            "component": "y",
            "iteration": "last",
            "axis": 0,
            "index": 3,
            "downsample": 2,
            "tail": 10,
        },
    )
    assert params.op is ResultOp.SLICE
    assert params.iteration == "last"
    assert params.path == "openPMD/fields.bp"


def test_result_params_accepts_int_iteration() -> None:
    params = ResultParams(sim_id=SIM, op=ResultOp.STATS, iteration=100)
    assert params.iteration == 100


@pytest.mark.parametrize("path", ["/etc/passwd", "../secret.txt", "a/../../b", "a b.txt", "a;rm.txt", "ü.txt"])
def test_result_params_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        ResultParams(sim_id=SIM, op=ResultOp.READ, path=path)


@pytest.mark.parametrize("stream", ["workflow", "stdin", "other"])
def test_result_params_rejects_unknown_stream(stream: str) -> None:
    with pytest.raises(ValidationError):
        ResultParams(sim_id=SIM, op=ResultOp.READ, stream=stream)


def test_result_params_allows_stdout_stderr_streams() -> None:
    assert ResultParams(sim_id=SIM, op=ResultOp.READ, stream="stdout").stream == "stdout"
    assert ResultParams(sim_id=SIM, op=ResultOp.READ, stream="stderr").stream == "stderr"


@pytest.mark.parametrize("reader", ["bogus", "EnergyHistogram", "openpmd", ""])
def test_result_params_rejects_unknown_reader(reader: str) -> None:
    with pytest.raises(ValidationError):
        ResultParams(sim_id=SIM, op=ResultOp.PLUGIN, reader=reader)


def test_result_params_accepts_registered_readers() -> None:
    for reader in PLUGIN_READER_NAMES:
        assert ResultParams(sim_id=SIM, op=ResultOp.PLUGIN, reader=reader).reader == reader


@pytest.mark.parametrize("name", ["../e", "a/b", "e;rm", "e f", "ü"])
def test_result_params_rejects_unsafe_species(name: str) -> None:
    with pytest.raises(ValidationError):
        ResultParams(sim_id=SIM, op=ResultOp.PLUGIN, species=name)


def test_result_params_plugin_defaults_and_round_trip() -> None:
    params = ResultParams(sim_id=SIM, op=ResultOp.PLUGIN, reader="energy_histogram", species="e")
    # Unset means "unspecified" and is normalized by the reader path; it must
    # not be serialized for every result op (N1).
    assert params.species_filter is None
    assert params.iteration is None
    command = build_result_command(sim=SIM, seq=1, params=params, cmd_id=CMD_ID)
    assert command.payload["reader"] == "energy_histogram"
    assert command.payload["species"] == "e"
    assert "species_filter" not in command.payload


def test_result_params_explicit_species_filter_is_forwarded() -> None:
    params = ResultParams(sim_id=SIM, op=ResultOp.PLUGIN, reader="emittance", species="e", species_filter="openPMD")
    command = build_result_command(sim=SIM, seq=1, params=params, cmd_id=CMD_ID)
    assert command.payload["species_filter"] == "openPMD"


def test_result_params_energy_window_is_forwarded() -> None:
    """A requested ``min_kev``/``max_kev`` window travels on the wire (F2)."""
    params = ResultParams(
        sim_id=SIM,
        op=ResultOp.PLUGIN,
        reader="energy_histogram",
        species="e",
        min_kev=2500.0,
        max_kev=20000.0,
    )
    command = build_result_command(sim=SIM, seq=1, params=params, cmd_id=CMD_ID)
    assert command.payload["min_kev"] == pytest.approx(2500.0)
    assert command.payload["max_kev"] == pytest.approx(20000.0)


def test_result_params_unset_energy_window_is_omitted() -> None:
    params = ResultParams(sim_id=SIM, op=ResultOp.PLUGIN, reader="energy_histogram", species="e")
    command = build_result_command(sim=SIM, seq=1, params=params, cmd_id=CMD_ID)
    assert params.min_kev is None
    assert params.max_kev is None
    assert "min_kev" not in command.payload
    assert "max_kev" not in command.payload


@pytest.mark.parametrize("kwargs", [{"min_kev": 100.0}, {"max_kev": 1000.0}])
def test_result_params_window_requires_both_edges(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValidationError, match="must be set together"):
        ResultParams(sim_id=SIM, op=ResultOp.PLUGIN, reader="energy_histogram", **kwargs)


def test_result_params_window_requires_increasing_edges() -> None:
    with pytest.raises(ValidationError, match="must be greater"):
        ResultParams(
            sim_id=SIM,
            op=ResultOp.PLUGIN,
            reader="energy_histogram",
            min_kev=1000.0,
            max_kev=1000.0,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_result_params_window_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValidationError):
        ResultParams(
            sim_id=SIM,
            op=ResultOp.PLUGIN,
            reader="energy_histogram",
            min_kev=100.0,
            max_kev=value,
        )


def test_result_params_non_plugin_ops_omit_species_filter() -> None:
    for op in (ResultOp.DESCRIBE, ResultOp.READ, ResultOp.EXPORT):
        command = build_result_command(sim=SIM, seq=1, params=ResultParams(sim_id=SIM, op=op), cmd_id=CMD_ID)
        assert "species_filter" not in command.payload


def test_result_params_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ResultParams.model_validate({"sim_id": SIM, "op": "describe", "bogus": 1})


def test_result_params_all_ops() -> None:
    for op in ResultOp:
        assert ResultParams(sim_id=SIM, op=op).op is op


def test_result_command_carries_knobs_and_omits_unset() -> None:
    params = ResultParams(sim_id=SIM, op=ResultOp.READ, path="foo.txt", tail=5)
    command = build_result_command(sim=SIM, seq=1, params=params, cmd_id=CMD_ID)
    assert command.type == SimulationType.RESULT_COMMAND
    assert command.payload["op"] == "read"
    assert command.payload["path"] == "foo.txt"
    assert command.payload["tail"] == 5
    assert "record" not in command.payload


def test_build_result_ack_shape() -> None:
    ack = build_result_ack(
        sim=SIM,
        seq=2,
        cmd_id=CMD_ID,
        sim_id=SIM,
        op=ResultOp.DESCRIBE,
        in_reply_to="evt-1",
        manifest={"sim_id": SIM, "run_dir": "/run", "total_bytes": 1, "files": []},
    )
    assert ack.type == SimulationType.RESULT_ACK
    assert ack.payload["cmd_id"] == CMD_ID
    assert ack.payload["sim_id"] == SIM
    assert ack.payload["op"] == "describe"
    assert ack.payload["manifest"]["total_bytes"] == 1
    assert "data" not in ack.payload


def test_build_result_ack_data_and_stats() -> None:
    ack = build_result_ack(
        sim=SIM,
        seq=2,
        cmd_id=CMD_ID,
        sim_id=SIM,
        op=ResultOp.STATS,
        in_reply_to=None,
        data=[1.0, 2.0],
        data_encoding="float",
        n_points=2,
        stats={"min": 1.0, "max": 2.0},
    )
    assert ack.payload["data"] == [1.0, 2.0]
    assert ack.payload["n_points"] == 2
    assert ack.payload["stats"]["max"] == pytest.approx(2.0)


def test_build_result_ack_error_pair() -> None:
    ack = build_result_ack(
        sim=SIM,
        seq=3,
        cmd_id=CMD_ID,
        sim_id=SIM,
        op=ResultOp.SLICE,
        in_reply_to=None,
        error="reader_unavailable",
        error_code="reader_unavailable",
    )
    assert ack.payload["error_code"] == "reader_unavailable"
    assert "data" not in ack.payload


def test_build_result_ack_accepts_png_string_data() -> None:
    ack = build_result_ack(
        sim=SIM,
        seq=4,
        cmd_id=CMD_ID,
        sim_id=SIM,
        op=ResultOp.IMAGE,
        in_reply_to=None,
        data="aGVsbG8=",
        data_encoding="png",
    )
    assert ack.payload["data"] == "aGVsbG8="
    assert ack.payload["data_encoding"] == "png"


def test_result_ref_defaults_and_forbid() -> None:
    ref = ResultRef(path="foo.txt", uri="file:///run/foo.txt", format="text", size_bytes=3)
    assert ref.readable is False
    assert ref.records == []
    assert ref.sha256 is None
    with pytest.raises(ValidationError):
        ResultRef(path="x", uri="u", format="text", size_bytes=1, bogus=True)


def test_result_manifest_defaults() -> None:
    manifest = ResultManifest(sim_id=SIM, run_dir="/run")
    assert manifest.output_dir is None
    assert manifest.reader is None
    assert manifest.files == []
    assert manifest.readable_local is False
    with pytest.raises(ValidationError):
        ResultManifest(sim_id=SIM, run_dir="/run", bogus=1)


def test_result_caps_are_frozen() -> None:
    # The ack cap matches the M2a inline budget (48 KiB) so a result ack can
    # never exceed the homeserver's event-size limit and look like a timeout.
    assert MAX_RESULT_BYTES == 48 * 1024
    assert RESULT_TEXT_MAX_BYTES == 48 * 1024
    assert SLICE_MAX_POINTS == 4096
