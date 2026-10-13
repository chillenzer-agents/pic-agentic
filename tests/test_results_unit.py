# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Unit tests for the results engine (:mod:`pic_agentic.results`).

A fake ``simOutput`` tree is scanned directly; no cluster, transport or reader
is required.  The optional openPMD dependency is monkeypatched where a code
path must be exercised, so the suite passes with ``openpmd_api`` ABSENT.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from pic_agentic import results
from pic_agentic.protocol.simulation import SLICE_MAX_POINTS, ResultOp, ResultParams
from pic_agentic.results import (
    ResultsReaderError,
    read_stats,
    scan_output,
)

SIM_ID = "abcd1234"


def _tree(tmp_path):
    """Build a small fake output tree and return (simOutput, sizes)."""
    out = tmp_path / "simOutput"
    (out / "openPMD").mkdir(parents=True)
    (out / "openPMD" / "fields.bp").write_bytes(b"b" * 100)
    (out / "foo.txt").write_text("first\nsecond\nthird\nfourth\n")
    (out / "bar.csv").write_text("a,b\n1,2\n")
    (out / "adir").mkdir()
    (out / "noext").write_bytes(b"z" * 7)
    sizes = {
        "openPMD/fields.bp": 100,
        "foo.txt": (out / "foo.txt").stat().st_size,
        "bar.csv": (out / "bar.csv").stat().st_size,
        "noext": 7,
    }
    return out, sizes


def _scan(out, local_root=""):
    return scan_output(out, sim_id=SIM_ID, run_dir=str(out.parent), local_root=local_root)


def test_scan_output_never_needs_reader_but_reports_it(tmp_path) -> None:
    out, _ = _tree(tmp_path)
    manifest = _scan(out)
    assert manifest.reader == results._reader_name()
    if importlib.util.find_spec("openpmd_api") is None:
        assert manifest.reader is None


def test_scan_output_is_sorted_and_total_bytes_matches(tmp_path) -> None:
    out, sizes = _tree(tmp_path)
    manifest = _scan(out)
    by_path = {ref.path: ref for ref in manifest.files}
    assert set(by_path) == {*sizes, "adir", "openPMD"}
    assert [ref.path for ref in manifest.files] == sorted(by_path)
    assert manifest.total_bytes == sum(sizes.values())
    assert by_path["adir"].size_bytes == 0
    assert by_path["adir"].format == "dir"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("fields.bp", "openpmd-adios2"),
        ("fields.h5", "openpmd-hdf5"),
        # The pinned openpmd_api 0.17.1 rejects ``.hdf5`` ("Unknown file
        # format"); it is not advertised as an openPMD series suffix (M1).
        ("fields.hdf5", "binary"),
        ("foo.txt", "text"),
        ("bar.csv", "text"),
        ("run.log", "text"),
        ("noext", "binary"),
        ("lib.so", "binary"),
    ],
)
def test_scan_output_format_sniffing(tmp_path, name: str, expected: str) -> None:
    out = tmp_path / "simOutput"
    out.mkdir()
    (out / name).write_bytes(b"x")
    manifest = _scan(out)
    assert manifest.files[0].format == expected


def test_scan_output_uri_is_real_absolute(tmp_path) -> None:
    out, _ = _tree(tmp_path)
    manifest = _scan(out)
    ref = next(ref for ref in manifest.files if ref.path == "foo.txt")
    assert ref.uri == (out / "foo.txt").resolve().as_uri()
    assert ref.uri.startswith("file://")


def test_scan_output_missing_dir_is_empty_manifest(tmp_path) -> None:
    missing = tmp_path / "nope" / "simOutput"
    manifest = scan_output(missing, sim_id=SIM_ID, run_dir=str(tmp_path))
    assert manifest.output_dir is None
    assert manifest.total_bytes == 0
    assert manifest.files == []
    assert manifest.readable_local is False


def test_scan_output_readable_without_local_root(tmp_path) -> None:
    out, _ = _tree(tmp_path)
    manifest = _scan(out)
    assert manifest.readable_local is False
    assert all(ref.readable is False for ref in manifest.files)


def test_scan_output_readable_with_local_root(tmp_path) -> None:
    out, _ = _tree(tmp_path)
    mirror = tmp_path / "mirror"
    (mirror / SIM_ID / "simOutput").mkdir(parents=True)
    (mirror / SIM_ID / "simOutput" / "foo.txt").write_text("mirrored\n")
    manifest = _scan(out, local_root=str(mirror))
    by_path = {ref.path: ref for ref in manifest.files}
    assert manifest.readable_local is True
    assert by_path["foo.txt"].readable is True
    assert by_path["bar.csv"].readable is False
    assert by_path["adir"].readable is False


def _params(**kwargs: object) -> ResultParams:
    return ResultParams(sim_id=SIM_ID, **kwargs)


def test_resolve_result_explicit_output_dir_bypasses_the_convention(tmp_path) -> None:
    """A caller holding the output dir need not name it ``simOutput``.

    ``resolve_result`` normally looks under ``run_dir/simOutput``; an explicit
    ``output_dir`` lets a caller pass the linked directory directly.
    """
    run = tmp_path / "run"
    (run / "simOutput").mkdir(parents=True)
    (run / "simOutput" / "foo.txt").write_text("ignored\n")
    out = run / "outputs"
    out.mkdir()
    (out / "foo.txt").write_text("explicit\n")
    payload = results.resolve_result(
        _params(op=ResultOp.READ, path="foo.txt"),
        run_dir=run,
        sim_id=SIM_ID,
        output_dir=out,
    )
    assert payload["data"] == ["explicit"]


def test_resolve_read_tail_without_openpmd(tmp_path) -> None:
    out, _ = _tree(tmp_path)
    payload = results.resolve_result(
        _params(op=ResultOp.READ, path="foo.txt", tail=2),
        run_dir=out.parent,
        sim_id=SIM_ID,
    )
    assert payload["data"] == ["third", "fourth"]
    assert payload["data_encoding"] == "text"
    assert payload["n_points"] == 2
    assert "error_code" not in payload


def test_resolve_read_default_tail(tmp_path) -> None:
    out, _ = _tree(tmp_path)
    payload = results.resolve_result(_params(op=ResultOp.READ, path="foo.txt"), run_dir=out.parent, sim_id=SIM_ID)
    assert payload["data"] == ["first", "second", "third", "fourth"]


def test_resolve_read_csv_is_text(tmp_path) -> None:
    out, _ = _tree(tmp_path)
    payload = results.resolve_result(_params(op=ResultOp.READ, path="bar.csv"), run_dir=out.parent, sim_id=SIM_ID)
    assert payload["data"] == ["a,b", "1,2"]


def test_resolve_read_bp_is_reader_unavailable(tmp_path) -> None:
    out, _ = _tree(tmp_path)
    payload = results.resolve_result(
        _params(op=ResultOp.READ, path="openPMD/fields.bp"),
        run_dir=out.parent,
        sim_id=SIM_ID,
    )
    assert payload["error_code"] == "reader_unavailable"
    assert "data" not in payload


def test_resolve_read_missing_file_is_no_results(tmp_path) -> None:
    out, _ = _tree(tmp_path)
    payload = results.resolve_result(_params(op=ResultOp.READ, path="absent.txt"), run_dir=out.parent, sim_id=SIM_ID)
    assert payload["error_code"] == "no_results"


def test_resolve_describe_returns_manifest(tmp_path) -> None:
    out, sizes = _tree(tmp_path)
    payload = results.resolve_result(_params(op=ResultOp.DESCRIBE), run_dir=out.parent, sim_id=SIM_ID)
    assert payload["manifest"]["sim_id"] == SIM_ID
    assert payload["manifest"]["total_bytes"] == sum(sizes.values())


def test_resolve_slice_reader_unavailable_without_openpmd(tmp_path, monkeypatch) -> None:
    out, _ = _tree(tmp_path)
    monkeypatch.setattr(results, "_reader_name", lambda: None)
    payload = results.resolve_result(
        _params(op=ResultOp.SLICE, path="openPMD/fields.bp", record="E"),
        run_dir=out.parent,
        sim_id=SIM_ID,
    )
    assert payload["error_code"] == "reader_unavailable"


def test_resolve_slice_prereader_reports_unavailable(tmp_path) -> None:
    """On the reader-absent host even a missing output is reader_unavailable."""
    if results._reader_name() is not None:
        pytest.skip("openpmd_api is installed in this environment")
    payload = results.resolve_result(_params(op=ResultOp.SLICE), run_dir=tmp_path, sim_id=SIM_ID)
    assert payload["error_code"] == "reader_unavailable"


def test_resolve_slice_no_output_is_no_results(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(results, "_reader_name", lambda: "openpmd")
    payload = results.resolve_result(_params(op=ResultOp.SLICE), run_dir=tmp_path, sim_id=SIM_ID)
    assert payload["error_code"] == "no_results"


class _FakeValues:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def reshape(self, *_shape: object) -> list[float]:
        return self._values


class _FakeMesh:
    """Models the openPMD >= 0.15 record API: components via iteration."""

    def __init__(self, comps: dict[str, object]) -> None:
        self._comps = comps

    def __iter__(self) -> object:
        return iter(self._comps)

    def __getitem__(self, key: str) -> object:
        return self._comps[key]


class _FakeStep:
    def __init__(self, meshes: dict[str, object]) -> None:
        self.meshes = meshes

    def __getitem__(self, key: str) -> object:
        return self.meshes[key]


class _FakeIterations:
    def __init__(self, steps: dict[int, object]) -> None:
        self._steps = steps

    def __iter__(self) -> object:
        return iter(self._steps)

    def __getitem__(self, key: int) -> object:
        return self._steps[key]


class _FakeSeries:
    def __init__(self, steps: dict[int, object]) -> None:
        self.iterations = _FakeIterations(steps)

    def flush(self) -> None:
        pass


class _FakeDataset:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def load_chunk(self) -> _FakeValues:
        return _FakeValues(self._values)


class _FakeApi:
    Access = type("Access", (), {"read_only": "read_only"})

    def __init__(self, values: list[float]) -> None:
        self._dataset = _FakeDataset(values)

    def Series(self, path: object, access: object) -> _FakeSeries:
        _ = (path, access)
        return _FakeSeries({0: _FakeStep({}), 10: _FakeStep({"E": _FakeMesh({"x": self._dataset})})})


def _fake_api(values: list[float]) -> _FakeApi:
    return _FakeApi(values)


def test_resolve_slice_hard_caps_points(tmp_path, monkeypatch) -> None:
    out, _ = _tree(tmp_path)
    monkeypatch.setattr(results, "_reader_name", lambda: "openpmd")
    monkeypatch.setattr(results, "_import_openpmd", lambda: _fake_api(list(range(SLICE_MAX_POINTS * 2 + 5))))
    payload = results.resolve_result(
        _params(op=ResultOp.SLICE, path="openPMD/fields.bp", record="E", component="x"),
        run_dir=out.parent,
        sim_id=SIM_ID,
    )
    # The point cap is respected, and the float-tagged wire form fits the budget
    # (a full SLICE_MAX_POINTS float slice does not fit once floats are tagged,
    # so _cap_slice strides it down).
    assert 1 <= payload["n_points"] <= SLICE_MAX_POINTS
    assert len(payload["data"]) == payload["n_points"]
    assert results._escaped_size({"data": payload["data"], "n_points": payload["n_points"]}) <= results.MAX_RESULT_BYTES


def test_cap_slice_strides_a_float_heavy_slice_to_fit() -> None:
    """A SLICE_MAX_POINTS float slice exceeds the wire budget and is strided."""
    full = {"data": [0.0] * SLICE_MAX_POINTS, "n_points": SLICE_MAX_POINTS}
    assert results._escaped_size(full) > results.MAX_RESULT_BYTES
    capped = results._cap_slice(
        {"data": [float(i) for i in range(SLICE_MAX_POINTS)], "n_points": SLICE_MAX_POINTS},
        None,
    )
    assert "error_code" not in capped
    assert 1 < capped["n_points"] < SLICE_MAX_POINTS
    assert results._escaped_size(capped) <= results.MAX_RESULT_BYTES
    # Striding keeps the endpoints, spreading the returned points across the range.
    assert repr(capped["data"][0]) == repr(0.0)
    assert capped["data"][-1] > capped["data"][1]


def test_resolve_slice_downsample_applied(tmp_path, monkeypatch) -> None:
    out, _ = _tree(tmp_path)
    monkeypatch.setattr(results, "_reader_name", lambda: "openpmd")
    monkeypatch.setattr(results, "_import_openpmd", lambda: _fake_api(list(range(20))))
    payload = results.resolve_result(
        _params(op=ResultOp.SLICE, path="openPMD/fields.bp", record="E", component="x", downsample=5),
        run_dir=out.parent,
        sim_id=SIM_ID,
    )
    assert payload["data"] == [0.0, 5.0, 10.0, 15.0]
    assert payload["n_points"] == 4


def test_resolve_slice_too_large_budget(tmp_path, monkeypatch) -> None:
    out, _ = _tree(tmp_path)
    monkeypatch.setattr(results, "_reader_name", lambda: "openpmd")
    monkeypatch.setattr(results, "_import_openpmd", lambda: _fake_api(list(range(1000))))
    monkeypatch.setattr(results, "MAX_RESULT_BYTES", 32)
    payload = results.resolve_result(
        _params(op=ResultOp.SLICE, path="openPMD/fields.bp", record="E", component="x"),
        run_dir=out.parent,
        sim_id=SIM_ID,
    )
    assert payload["error_code"] == "result_too_large"
    assert "data" not in payload


def test_resolve_stats_with_fake_reader(tmp_path, monkeypatch) -> None:
    out, _ = _tree(tmp_path)
    monkeypatch.setattr(results, "_reader_name", lambda: "openpmd")
    monkeypatch.setattr(results, "_import_openpmd", lambda: _fake_api([1.0, 2.0, 3.0]))
    payload = results.resolve_result(
        _params(op=ResultOp.STATS, path="openPMD/fields.bp", record="E", component="x"),
        run_dir=out.parent,
        sim_id=SIM_ID,
    )
    assert payload["stats"]["min"] == pytest.approx(1.0)
    assert payload["stats"]["max"] == pytest.approx(3.0)
    assert payload["stats"]["mean"] == pytest.approx(2.0)


def test_read_stats_unknown_record_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(results, "_reader_name", lambda: "openpmd")
    monkeypatch.setattr(results, "_import_openpmd", lambda: _fake_api([1.0]))
    with pytest.raises(ResultsReaderError):
        read_stats(tmp_path / "fields.bp", record="missing")


@pytest.mark.skipif(importlib.util.find_spec("PIL") is not None, reason="Pillow installed")
def test_resolve_image_without_pillow_is_clean_error(tmp_path, monkeypatch) -> None:
    out, _ = _tree(tmp_path)
    monkeypatch.setattr(results, "_reader_name", lambda: "openpmd")
    monkeypatch.setattr(results, "_import_openpmd", lambda: _fake_api([1.0, 2.0]))
    payload = results.resolve_result(
        _params(op=ResultOp.IMAGE, path="openPMD/fields.bp", record="E", component="x"),
        run_dir=out.parent,
        sim_id=SIM_ID,
    )
    assert payload["error_code"] == "reader_unavailable"


def test_resolve_export_unresolved_is_ticket_with_rsync(tmp_path) -> None:
    out, _ = _tree(tmp_path)
    payload = results.resolve_result(_params(op=ResultOp.EXPORT), run_dir=out.parent, sim_id=SIM_ID)
    ticket = payload["result"]
    assert ticket["resolved"] is False
    assert ticket["local_path"] is None
    assert ticket["transfer"].startswith("rsync -a ")
    assert str(out.resolve()) in ticket["transfer"]
    assert ticket["ref"]["path"] == "adir"  # dirs sort before files, first entry


def test_resolve_export_resolved_with_local_root(tmp_path) -> None:
    out, _ = _tree(tmp_path)
    mirror = tmp_path / "mirror"
    (mirror / SIM_ID / "simOutput").mkdir(parents=True)
    payload = results.resolve_result(
        _params(op=ResultOp.EXPORT),
        run_dir=out.parent,
        sim_id=SIM_ID,
        local_root=str(mirror),
    )
    ticket = payload["result"]
    assert ticket["resolved"] is True
    assert ticket["local_path"] == str(mirror / SIM_ID / "simOutput")


def test_resolve_export_empty_output_has_no_ref(tmp_path) -> None:
    payload = results.resolve_result(_params(op=ResultOp.EXPORT), run_dir=tmp_path, sim_id=SIM_ID)
    assert payload["result"]["ref"] is None


async def test_follower_results_ready_carries_manifest(tmp_path) -> None:
    """The follow hook attaches the light manifest to ``results.ready``."""
    from pic_agentic.simclient.follow import JobFollower, TrackedSim
    from pic_agentic.slurm import JobInfo, SlurmJobState

    run_dir = tmp_path / "run"
    (run_dir / "simOutput").mkdir(parents=True)
    (run_dir / "simOutput" / "foo.txt").write_text("hello\n")
    events: list[tuple[object, dict]] = []

    async def emit(state: object, *, job_id: int | None = None, **fields: object) -> None:
        events.append((state, {"job_id": job_id, **fields}))

    follower = JobFollower(
        sim=SIM_ID,
        emit=emit,
        tracked=TrackedSim(
            sim_id=SIM_ID, cmd_id="c", job_id=1, run_dir=str(run_dir), stdout_path=None, submit_system="sbatch"
        ),
        job_info=None,  # type: ignore[arg-type]
    )
    info = JobInfo(job_id=1, state=SlurmJobState.COMPLETED, exit_code=0)
    await follower._emit_terminal(info)

    ready = next(fields for state, fields in events if fields.get("manifest") is not None)
    manifest = ready["manifest"]
    assert manifest["sim_id"] == SIM_ID
    assert any(ref["path"] == "foo.txt" for ref in manifest["files"])


async def test_follower_hook_is_guarded_when_engine_missing(monkeypatch, tmp_path) -> None:
    """A scan failure is swallowed: following still emits the plain event."""
    import builtins

    from pic_agentic.simclient.follow import JobFollower, TrackedSim
    from pic_agentic.slurm import JobInfo, SlurmJobState

    run_dir = tmp_path / "run"
    (run_dir / "simOutput").mkdir(parents=True)
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pic_agentic.results":
            msg = "engine not merged"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    events: list[tuple[object, dict]] = []

    async def emit(state: object, *, job_id: int | None = None, **fields: object) -> None:
        events.append((state, {"job_id": job_id, **fields}))

    follower = JobFollower(
        sim=SIM_ID,
        emit=emit,
        tracked=TrackedSim(
            sim_id=SIM_ID, cmd_id="c", job_id=1, run_dir=str(run_dir), stdout_path=None, submit_system="sbatch"
        ),
        job_info=None,  # type: ignore[arg-type]
    )
    info = JobInfo(job_id=1, state=SlurmJobState.COMPLETED, exit_code=0)
    await follower._emit_terminal(info)
    assert "manifest" not in events[-1][1]


def test_read_stream_stdout_without_openpmd(tmp_path) -> None:
    """The captured stdout stream has no suffix but must still read as text."""
    out, _ = _tree(tmp_path)
    run = out.parent
    (run / "stdout").write_text("SIGNAL: received.\nSIGNAL: Activate checkpointing for step 42\n", encoding="utf-8")
    payload = results.resolve_result(
        _params(op=ResultOp.READ, stream="stdout"),
        run_dir=run,
        sim_id=SIM_ID,
    )
    assert payload.get("data_encoding") == "text"
    assert "Activate checkpointing" in "\n".join(payload["data"])


def test_describe_truncates_huge_manifests(tmp_path) -> None:
    """A directory with many files must not overflow the ack wire budget."""
    out = tmp_path / "simOutput"
    out.mkdir(parents=True)
    for i in range(4000):
        (out / f"f{i:05d}.txt").write_text("x", encoding="utf-8")
    payload = results.resolve_result(
        _params(op=ResultOp.DESCRIBE),
        run_dir=out.parent,
        sim_id=SIM_ID,
    )
    assert "manifest" in payload
    dump = payload["manifest"]
    assert dump["truncated"] is True
    assert len(dump["files"]) < 4000
    assert results._escaped_size(dump) <= results.MAX_RESULT_BYTES


def test_result_ack_builder_enforces_wire_budget() -> None:
    """The ack builder is the last line of defence against a Synapse rejection."""
    from pic_agentic.protocol.simulation import ResultOp as _Op
    from pic_agentic.protocol.simulation import build_result_ack

    ack = build_result_ack(
        sim="s",
        seq=1,
        cmd_id="c",
        sim_id=SIM_ID,
        op=_Op.SLICE,
        in_reply_to=None,
        data=[1.0] * 1_000_000,
        data_encoding="float",
    )
    assert ack.payload.get("error_code") == "result_too_large"
    assert "data" not in ack.payload


@pytest.mark.skipif(importlib.util.find_spec("openpmd_api") is None, reason="openpmd_api not installed")
def test_real_openpmd_series_roundtrip(tmp_path) -> None:
    """Read a real openPMD series (guards against reader-API drift).

    The optional reader is only importable in the ``[sim]`` environment, so
    this is skipped offline.  It pins the component/iteration access that the
    mocked unit tests cannot: 0.17 exposes components via record iteration
    (not ``mesh.components``), a scalar mesh via a sentinel component, and an
    unknown iteration as ``IndexError``.
    """
    import numpy as np
    import openpmd_api as api

    out = tmp_path / "simOutput"
    out.mkdir()
    series = api.Series(str(out / "fields.h5"), api.Access.create)
    mesh = series.iterations[0].meshes["E"]
    mesh["x"].reset_dataset(api.Dataset(api.Datatype.DOUBLE, [4]))
    mesh["x"].store_chunk(np.array([1.0, 2.0, 3.0, 4.0]))
    rho = series.iterations[0].meshes["rho"]
    rho.reset_dataset(api.Dataset(api.Datatype.DOUBLE, [2]))
    rho.store_chunk(np.array([5.0, 7.0]))
    series.flush()
    series.close()
    del series

    def show(op, **kw: object):
        return results.resolve_result(ResultParams(sim_id=SIM_ID, op=op, **kw), run_dir=tmp_path, sim_id=SIM_ID)

    assert show(ResultOp.SLICE, path="fields.h5", record="E", component="x")["data"] == [1.0, 2.0, 3.0, 4.0]
    # Component-less selection resolves the first component.
    assert show(ResultOp.SLICE, path="fields.h5", record="E")["n_points"] == 4
    # A scalar (componentless) mesh loads straight from the record.
    assert show(ResultOp.STATS, path="fields.h5", record="rho")["stats"]["mean"] == pytest.approx(6.0)
    # Unknown record/component/iteration are clean errors, not exceptions.
    assert show(ResultOp.STATS, path="fields.h5", record="nope")["error_code"] == "no_results"
    assert show(ResultOp.STATS, path="fields.h5", record="E", component="nope")["error_code"] == "no_results"
    assert show(ResultOp.STATS, path="fields.h5", record="E", iteration=999)["error_code"] == "no_results"


@pytest.mark.skipif(importlib.util.find_spec("openpmd_api") is None, reason="openpmd_api not installed")
def test_real_openpmd_nested_picongpu_series_is_discovered(tmp_path) -> None:
    """A real PIConGPU layout (nested, ``%06T`` infix) is found without a path.

    Regression for the live gap: ``resolve_result`` opened ``<run>/simOutput``
    directly, but PIConGPU nests the series at
    ``simOutput/openPMD/simOutput/fields_%06T.h5`` (see the diagnostics'
    ``result_path``), so ``slice``/``stats``/``compute`` all failed with
    ``no_results``.  Discovery must reconstruct the ``%T`` pattern so *every*
    iteration is visible, not just one file.
    """
    import numpy as np
    import openpmd_api as api

    # Exactly the nesting NativeFieldDump.result_path(run_dir) yields.
    series_dir = tmp_path / "simOutput" / "openPMD" / "simOutput"
    series_dir.mkdir(parents=True)
    for step, value in ((0, 1.0), (50, 2.0), (200, 3.0)):
        series = api.Series(str(series_dir / "fields_%06T.h5"), api.Access.create)
        mesh = series.iterations[step].meshes["E"]
        mesh["x"].reset_dataset(api.Dataset(api.Datatype.DOUBLE, [4]))
        mesh["x"].store_chunk(np.full(4, value))
        series.flush()
        series.close()
        del series

    def show(op, **kw: object):
        return results.resolve_result(ResultParams(sim_id=SIM_ID, op=op, **kw), run_dir=tmp_path, sim_id=SIM_ID)

    # No path: the series is discovered under simOutput.
    assert show(ResultOp.SLICE, record="E", component="x")["n_points"] == 4
    # A path naming the series tree narrows the search the same way.
    assert show(ResultOp.SLICE, path="openPMD/simOutput", record="E", component="x")["n_points"] == 4
    # The pattern must span all iterations, so 'last' resolves to step 200.
    last = show(ResultOp.STATS, record="E", component="x", iteration="last")
    assert last["stats"]["mean"] == pytest.approx(3.0)
    assert show(ResultOp.STATS, record="E", component="x", iteration=50)["stats"]["mean"] == pytest.approx(2.0)
    # A missing series is still a clean error.
    empty = tmp_path / "empty"
    (empty / "simOutput").mkdir(parents=True)
    assert (
        results.resolve_result(
            ResultParams(sim_id=SIM_ID, op=ResultOp.SLICE, record="E"),
            run_dir=empty,
            sim_id=SIM_ID,
        )["error_code"]
        == "no_results"
    )


def _write_hdf5_series(directory: Path, prefix: str, record: str, component: str, values: dict[int, float]) -> None:
    """Write a small ``<prefix>_%06T.h5`` series with one component record."""
    import numpy as np
    import openpmd_api as api

    for step, value in values.items():
        series = api.Series(str(directory / f"{prefix}_%06T.h5"), api.Access.create)
        mesh = series.iterations[step].meshes[record]
        mesh[component].reset_dataset(api.Dataset(api.Datatype.DOUBLE, [2]))
        mesh[component].store_chunk(np.full(2, value))
        series.flush()
        series.close()
        del series


@pytest.mark.skipif(importlib.util.find_spec("openpmd_api") is None, reason="openpmd_api not installed")
def test_real_openpmd_adios2_directory_series_is_discovered(tmp_path) -> None:
    """A ``.bp5`` series is a *directory* and must still be discovered.

    PIConGPU's default backend is ``bp5``; each iteration is a directory
    (``fields_000000.bp5/``), so a discovery that only accepts files misses
    every default run.
    """
    import numpy as np
    import openpmd_api as api

    series_dir = tmp_path / "simOutput" / "openPMD" / "simOutput"
    series_dir.mkdir(parents=True)
    for step, value in ((0, 1.0), (50, 2.0)):
        series = api.Series(str(series_dir / "fields_%06T.bp5"), api.Access.create)
        mesh = series.iterations[step].meshes["E"]
        mesh["x"].reset_dataset(api.Dataset(api.Datatype.DOUBLE, [2]))
        mesh["x"].store_chunk(np.full(2, value))
        series.flush()
        series.close()
        del series
    assert (series_dir / "fields_000000.bp5").is_dir()  # ADIOS2 series is a directory

    slice_ = results.resolve_result(
        ResultParams(sim_id=SIM_ID, op=ResultOp.SLICE, record="E", component="x"),
        run_dir=tmp_path,
        sim_id=SIM_ID,
    )
    assert slice_["n_points"] == 2
    # An explicit concrete-directory path resolves to that series too.
    stats = results.resolve_result(
        ResultParams(
            sim_id=SIM_ID,
            op=ResultOp.STATS,
            path="openPMD/simOutput/fields_000050.bp5",
            record="E",
            component="x",
        ),
        run_dir=tmp_path,
        sim_id=SIM_ID,
    )
    assert stats["stats"]["mean"] == pytest.approx(2.0)


@pytest.mark.skipif(importlib.util.find_spec("openpmd_api") is None, reason="openpmd_api not installed")
def test_discovery_prefers_the_requested_record_family(tmp_path) -> None:
    """With several series, the one containing the requested record wins.

    Guards against silently binding an ``E`` slice to a ``particles_*`` series
    (which would return the wrong data or a spurious "record not found").
    """
    series_dir = tmp_path / "simOutput" / "openPMD" / "simOutput"
    series_dir.mkdir(parents=True)
    _write_hdf5_series(series_dir, "particles", "electrons", "x", {0: 7.0})
    _write_hdf5_series(series_dir, "fields", "E", "x", {0: 1.0})

    slice_ = results.resolve_result(
        ResultParams(sim_id=SIM_ID, op=ResultOp.SLICE, record="E", component="x"),
        run_dir=tmp_path,
        sim_id=SIM_ID,
    )
    assert slice_["data"] == [1.0, 1.0]
    particles = results.resolve_result(
        ResultParams(sim_id=SIM_ID, op=ResultOp.SLICE, record="electrons", component="x"),
        run_dir=tmp_path,
        sim_id=SIM_ID,
    )
    assert particles["data"] == [7.0, 7.0]


@pytest.mark.skipif(importlib.util.find_spec("openpmd_api") is None, reason="openpmd_api not installed")
def test_discovery_deprioritises_checkpoint_series(tmp_path) -> None:
    """A checkpoint series must never be preferred over the field output."""
    out = tmp_path / "simOutput"
    (out / "openPMD" / "simOutput").mkdir(parents=True)
    (out / "zeta_checkpoints").mkdir()
    _write_hdf5_series(out / "openPMD" / "simOutput", "fields", "E", "x", {0: 1.0})
    _write_hdf5_series(out / "zeta_checkpoints", "checkpoint", "E", "x", {0: 9.0})

    slice_ = results.resolve_result(
        ResultParams(sim_id=SIM_ID, op=ResultOp.SLICE, record="E", component="x"),
        run_dir=tmp_path,
        sim_id=SIM_ID,
    )
    assert slice_["data"] == [1.0, 1.0]


@pytest.mark.skipif(importlib.util.find_spec("openpmd_api") is None, reason="openpmd_api not installed")
def test_compute_const_program_needs_no_series(tmp_path) -> None:
    """A constant program evaluates even when the run has no openPMD output."""
    run = tmp_path / "run"
    (run / "simOutput").mkdir(parents=True)
    payload = results.resolve_result(
        ResultParams(
            sim_id=SIM_ID,
            op=ResultOp.COMPUTE,
            program={"output": {"kind": "const", "value": 42}},
        ),
        run_dir=run,
        sim_id=SIM_ID,
    )
    assert payload["stats"]["value"] == pytest.approx(42.0)
