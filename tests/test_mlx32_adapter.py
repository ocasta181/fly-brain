"""Tiny file-to-Metal adapter checks with literal discrete spike expectations."""

import builtins
import csv
import math
import os
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


SPIKE_SCHEMA = pa.schema([
    ("t", pa.float64()), ("time_ms", pa.float64()), ("trial", pa.int64()),
    ("neuron_index", pa.int64()), ("flywire_id", pa.int64()),
    ("exp_name", pa.string()),
])


@pytest.fixture
def adapter_case(tmp_path, monkeypatch, native):
    import benchmark
    import brian2 as brian
    import run_mlx32 as adapter

    ids = [2**53 + 101, 2**53 + 3]
    completeness = tmp_path / "completeness.csv"
    connectivity = tmp_path / "connectivity.parquet"
    pd.DataFrame({"complete": [True, True]}, index=ids).to_csv(completeness)
    pd.DataFrame({
        "Presynaptic_Index": pd.Series(dtype="int64"),
        "Postsynaptic_Index": pd.Series(dtype="int64"),
        "Excitatory x Connectivity": pd.Series(dtype="int64"),
    }).to_parquet(connectivity, index=False)
    monkeypatch.setattr(adapter, "path_comp", completeness)
    monkeypatch.setattr(adapter, "path_con", connectivity)
    monkeypatch.setattr(benchmark, "path_res", tmp_path / "results")
    monkeypatch.setattr(benchmark, "csv_path", tmp_path / "benchmark.csv")
    monkeypatch.delenv(benchmark.SPIKE_IO_ENV_VAR, raising=False)

    def reject_cpu_steps(*args, **kwargs):
        raise AssertionError("The adapter attempted Brian CPU timestepping")

    monkeypatch.setattr(brian.Network, "run", reject_cpu_steps)
    monkeypatch.setattr(brian, "run", reject_cpu_steps)
    experiment = {
        "name": "Deterministic adapter case", "key": "adapter_case",
        "neu_exc": [ids[1]], "neu_exc2": [ids[1], ids[0], ids[0]],
        "neu_slnc": [], "stim_rate": 10_000.0, "seed": 19,
    }
    return adapter, benchmark, experiment, ids


def assert_native_dispatches(
    native_launches, before, *, record, state_ticks=10, state_blocks=2, recorder_calls=2,
):
    launches = native_launches[before:]
    lengths = [[ticks for name, ticks in launches if name == kernel]
               for kernel in ("eon32_state", "eon32_propagate")]
    assert lengths[0] == lengths[1]
    for blocks in lengths:
        assert sum(blocks) == state_ticks
        assert len(blocks) == state_blocks
    assert [ticks for name, ticks in launches if name == "eon32_spike_events"] == (
        [1] * recorder_calls if record else []
    )


@pytest.mark.parametrize("row_cap, row_groups", [(None, [4]), (3, [3, 1]), (4, [4])])
def test_recorded_adapter_schema_events_results_and_csv(
    adapter_case, monkeypatch, native, native_launches, row_cap, row_groups,
):
    adapter, benchmark, experiment, ids = adapter_case
    monkeypatch.setattr(adapter, "_CHUNK_SIZE", 2)
    if row_cap is not None:
        monkeypatch.setattr(adapter, "_ARROW_ROWS", row_cap)
    before = len(native_launches)
    results = adapter.run_all_benchmarks(
        [0.0005], [2], experiment, benchmark.BenchmarkLogger(),
        run_label="adapter_case", round_idx=7,
    )
    assert_native_dispatches(
        native_launches, before, record=True, state_ticks=8, state_blocks=5, recorder_calls=5,
    )
    assert len(results) == 1
    result = results[0]
    target = benchmark.get_spike_output_path("mlx32_t0.0005s_n2", "adapter_case", 7)
    assert {key: value for key, value in result.items() if key != "timings"} == {
        "t_run_sec": 0.0005, "n_run": 2, "n_active_neurons": 1, "n_spikes": 4,
        "backend_key": "mlx32", "experiment_name": experiment["name"],
        "experiment_key": "adapter_case", "run_label": "adapter_case", "round": 7,
        "spike_path": str(target), "status": "success",
    }
    with pq.ParquetFile(target) as stored:
        assert stored.metadata.num_rows == 4
        assert [stored.metadata.row_group(i).num_rows
                for i in range(stored.metadata.num_row_groups)] == row_groups
        assert all(stored.metadata.row_group(i).column(j).compression == "BROTLI"
                   for i in range(stored.metadata.num_row_groups)
                   for j in range(len(SPIKE_SCHEMA)))
        table = stored.read()
    assert table.schema == SPIKE_SCHEMA
    expected_seconds = np.array([1, 1, 3, 3]) * 0.0001
    np.testing.assert_array_equal(table["t"].to_numpy(), expected_seconds)
    np.testing.assert_array_equal(table["time_ms"].to_numpy(), expected_seconds * 1000)
    assert table["trial"].to_pylist() == [0, 1, 0, 1]
    assert table["neuron_index"].to_pylist() == [1, 1, 1, 1]
    assert table["flywire_id"].to_pylist() == [ids[1]] * 4
    assert table["exp_name"].to_pylist() == ["mlx32_t0.0005s_n2"] * 4
    assert list(target.parent.iterdir()) == [target]

    timings = result["timings"]
    assert all(math.isfinite(value) and value >= 0 for value in timings.values())
    assert timings["simulation_total"] == timings["simulation_s"]
    assert timings["device_build"] == timings["compile_warmup_s"]
    assert timings["network_creation_total"] + timings["device_build"] == pytest.approx(
        timings["adapter_setup"] + timings["setup_compile_s"]
    )
    assert timings["spike_extraction_total"] == max(
        0, timings["collection_s"] - timings["sink_in_native"]
    )
    total_sink = timings["sink_collection"] + timings["sink_export"]
    if row_cap is not None and 4 % row_cap == 0:
        assert timings["sink_in_native"] == total_sink
    else:
        assert timings["sink_in_native"] < total_sink
    assert timings["result_collection"] >= timings["sink_collection"] + timings["snapshot_s"]
    assert timings["result_save"] >= timings["output_setup"] + timings["sink_export"]
    assert timings["runtime_s"] >= sum(timings[key] for key in (
        "input_generation_staging_s", "simulation_s", "collection_s",
    ))
    assert timings["native_call"] >= sum(timings[key] for key in (
        "setup_compile_s", "runtime_s", "snapshot_s",
    ))
    assert timings["phase_total"] == sum(timings[key] for key in (
        "network_creation_total", "device_build", "input_generation_staging_s",
        "simulation_total", "spike_extraction_total", "result_collection", "result_save",
    ))
    assert timings["phase_total"] <= timings["total_elapsed"]
    assert timings["realtime_ratio"] == 0.001 / timings["simulation_total"]
    assert timings["realtime_ratio_total"] == 0.001 / timings["total_elapsed"]

    with benchmark.csv_path.open(newline="") as saved:
        rows = list(csv.DictReader(saved))
    assert len(rows) == 1
    row = rows[0]
    assert list(row) == benchmark.CSV_COLUMNS
    for column, phase in (
        ("setup_time", "network_creation_total"), ("build_time", "device_build"),
        ("sim_time", "simulation_total"), ("total_time", "total_elapsed"),
    ):
        assert float(row[column]) == round(timings[phase], 3)
    assert float(row["realtime_ratio"]) == round(timings["realtime_ratio"], 4)
    for column, expected in {
        "framework": "MLX float32 (Metal)", "n_run": "2", "t_run": "0.0005",
        "spikes": "4", "active_neurons": "1", "status": "success",
        "backend_key": "mlx32", "experiment": experiment["name"],
        "experiment_key": "adapter_case", "run_label": "adapter_case", "round": "7",
        "spike_path": str(target),
    }.items():
        assert row[column] == expected
    with (benchmark.path_res / "adapter_case" / "manifest.csv").open(newline="") as saved:
        manifest, = csv.DictReader(saved)
    assert manifest["spike_path"] == str(target)
    assert manifest["spike_time_column"] == "time_ms"
    assert manifest["spike_time_unit"] == "ms"


def test_record_off_runs_metal_without_spike_output(
    adapter_case, monkeypatch, native, native_launches,
):
    adapter, benchmark, experiment, _ = adapter_case
    monkeypatch.setenv(benchmark.SPIKE_IO_ENV_VAR, "true")
    before = len(native_launches)
    result = adapter.run_single_benchmark(
        0.0005, 2, experiment, benchmark.BenchmarkLogger(),
    )
    assert_native_dispatches(native_launches, before, record=False)
    assert result["status"] == "success"
    assert result["n_spikes"] == result["n_active_neurons"] == 0
    assert result["spike_path"] == ""
    assert result["timings"]["sink_collection"] == result["timings"]["sink_export"] == 0
    assert not benchmark.path_res.exists()


def test_recorded_zero_events_publishes_empty_parquet(
    adapter_case, native, native_launches,
):
    adapter, benchmark, experiment, _ = adapter_case
    before = len(native_launches)
    result = adapter.run_single_benchmark(
        0.0005, 2, dict(experiment, stim_rate=0.0), benchmark.BenchmarkLogger(),
    )
    assert_native_dispatches(native_launches, before, record=True)
    assert result["status"] == "success"
    assert result["n_spikes"] == result["n_active_neurons"] == 0
    target = benchmark.get_spike_output_path("mlx32_t0.0005s_n2")
    assert result["spike_path"] == str(target)
    with pq.ParquetFile(target) as stored:
        assert stored.schema_arrow == SPIKE_SCHEMA
        assert stored.metadata.num_rows == stored.metadata.num_row_groups == 0
    assert result["timings"]["sink_collection"] == result["timings"]["sink_export"] == 0
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize("failure", ["stream", "tail", "close"])
def test_export_failure_preserves_target_and_removes_own_temporary_file(
    adapter_case, monkeypatch, native, native_launches, failure,
):
    adapter, benchmark, experiment, _ = adapter_case
    target = benchmark.get_spike_output_path("mlx32_t0.0005s_n2")
    target.parent.mkdir(parents=True)
    pq.write_table(pa.table({"existing": [29]}), target)
    original = target.read_bytes()
    real_writer = pq.ParquetWriter
    writers = []

    class FailingWriter:
        def __init__(self, *args, **kwargs):
            self.writer = real_writer(*args, **kwargs)
            self.writes = self.closes = 0
            writers.append(self)

        def write_table(self, table, **kwargs):
            if failure == "tail" or (failure == "stream" and self.writes):
                raise OSError("injected parquet export failure")
            self.writer.write_table(table, **kwargs)
            self.writes += 1

        def close(self):
            self.closes += 1
            self.writer.close()
            if failure == "close":
                raise OSError("injected parquet close failure")

    if failure == "stream":
        monkeypatch.setattr(adapter, "_ARROW_ROWS", 1)
    monkeypatch.setattr(pq, "ParquetWriter", FailingWriter)
    before = len(native_launches)
    result = adapter.run_single_benchmark(
        0.0005, 2, experiment, benchmark.BenchmarkLogger(),
    )
    assert_native_dispatches(native_launches, before, record=True)
    assert result["status"] == (
        "error: parquet close failed: injected parquet close failure"
        if failure == "close" else "error: injected parquet export failure"
    )
    assert result["n_spikes"] == result["n_active_neurons"] == 0
    assert result["spike_path"] == ""
    assert len(writers) == 1
    assert (writers[0].writes, writers[0].closes) == {
        "stream": (1, 1), "tail": (0, 1), "close": (1, 2),
    }[failure]
    assert target.read_bytes() == original
    assert list(target.parent.iterdir()) == [target]


@pytest.fixture
def graph_calls(monkeypatch, native):
    from mlx32 import engine, model

    builds, runs, parses = [], [], []
    real_make_graph, real_run = model.make_graph, engine.run_batch

    def make_graph(*args):
        graph = real_make_graph(*args)
        builds.append(graph)
        return graph

    def run_batch(graph, *args, **kwargs):
        runs.append(graph)
        return real_run(graph, *args, **kwargs)

    for name in ("read_csv", "read_parquet"):
        def read(source, *args, _name=name, _read=getattr(pd, name), **kwargs):
            parses.append((_name, isinstance(source, BytesIO)))
            return _read(source, *args, **kwargs)
        monkeypatch.setattr(pd, name, read)
    monkeypatch.setattr(model, "make_graph", make_graph)
    monkeypatch.setattr(engine, "run_batch", run_batch)
    return builds, runs, parses


@pytest.mark.parametrize("source, change", [
    ("comp", "same"), ("comp", "rewrite"), ("con", "rewrite"),
    ("comp", "missing"), ("con", "missing"),
    ("comp", "corrupt"), ("con", "corrupt"), ("comp", "csv_first"),
])
def test_grid_cache_checks_current_files_and_continues_after_errors(
    adapter_case, monkeypatch, graph_calls, native_launches, capsys, source, change,
):
    adapter, benchmark, experiment, ids = adapter_case
    builds, runs, parses = graph_calls
    path = adapter.path_comp if source == "comp" else adapter.path_con
    if source == "con":
        pd.DataFrame({"Presynaptic_Index": [0], "Postsynaptic_Index": [1],
                      "Excitatory x Connectivity": [1]}).to_parquet(
            path, index=False, compression=None,
        )
    original, stat = path.read_bytes(), path.stat()
    connectivity = adapter.path_con.read_bytes()
    if change == "rewrite":
        if source == "comp":
            lines = original.splitlines(keepends=True)
            replacement = b"".join([lines[0], lines[2], lines[1]])
        else:
            pd.DataFrame({"Presynaptic_Index": [0], "Postsynaptic_Index": [1],
                          "Excitatory x Connectivity": [2]}).to_parquet(
                path, index=False, compression=None,
            )
            replacement = path.read_bytes()
            path.write_bytes(original)
        assert len(replacement) == len(original)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    failure = change in {"missing", "corrupt", "csv_first"}
    saved, tables = [], []
    real_save = adapter.save_result_csv

    def save(backend, result):
        real_save(backend, result)
        saved.append(result)
        if result["spike_path"]:
            tables.append(pq.read_table(result["spike_path"]))
        if len(saved) == 1:
            if change in {"missing", "csv_first"}:
                (adapter.path_con if change == "csv_first" else path).unlink()
            if change in {"corrupt", "csv_first", "rewrite"}:
                path.write_bytes(replacement if change == "rewrite" else b"")
            if change == "rewrite":
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                assert path.stat().st_mtime_ns == stat.st_mtime_ns
        elif len(saved) == 2 and failure:
            path.write_bytes(original)
            adapter.path_con.write_bytes(connectivity)

    monkeypatch.setattr(adapter, "save_result_csv", save)
    before = len(native_launches)
    results = adapter.run_all_benchmarks(
        [0.0005] * (3 if failure else 2), [2], experiment, benchmark.BenchmarkLogger(),
    )
    assert results == saved
    assert results[0]["status"] == results[-1]["status"] == "success"
    if failure:
        assert results[1]["status"].startswith("error:")
    assert len(builds) == (2 if change == "rewrite" else 1)
    assert len(runs) == len(tables) == 2
    assert (runs[0] is runs[1]) == (change != "rewrite")
    assert all(snapshot for _, snapshot in parses)
    csv_reads = 2 if change == "missing" and source == "comp" else len(results)
    con_reads = 2 if change == "rewrite" or (change == "corrupt" and source == "con") else 1
    assert parses.count(("read_csv", True)) == csv_reads
    assert parses.count(("read_parquet", True)) == con_reads
    assert_native_dispatches(
        native_launches, before, record=True, state_ticks=20, state_blocks=4, recorder_calls=4,
    )
    for table in tables:
        assert table.schema == SPIKE_SCHEMA
        assert table["flywire_id"].to_pylist() == [ids[1]] * 4
        np.testing.assert_array_equal(
            table["t"].to_numpy(), np.array([1, 1, 3, 3]) * 0.0001,
        )
    if change == "rewrite" and source == "comp":
        assert builds[1].neuron_ids.tolist() == ids[::-1]
        assert tables[1]["neuron_index"].to_pylist() == [0] * 4
    else:
        assert tables[0].equals(tables[1])
    if change == "rewrite" and source == "con":
        assert builds[1].contacts.tolist() == [2]
    if failure:
        assert "ERROR:" in capsys.readouterr().out
        if change == "csv_first":
            assert results[1]["status"] == "error: No columns to parse from file"
    with benchmark.csv_path.open(newline="") as stored:
        assert [row["status"] for row in csv.DictReader(stored)] == [results[-1]["status"]]
    assert results[0]["timings"]["graph_creation"] > 0
    assert (results[-1]["timings"]["graph_creation"] == 0) == (change != "rewrite")


@pytest.mark.parametrize("mode", ["direct", "one_case", "separate_grids"])
def test_graph_cache_lifetime_and_path_loading(adapter_case, graph_calls, mode):
    adapter, benchmark, experiment, _ = adapter_case
    builds, runs, parses = graph_calls
    for _ in range(2):
        if mode == "direct":
            results = [adapter.run_single_benchmark(
                0.0005, 2, experiment, benchmark.BenchmarkLogger(),
            )]
        else:
            results = adapter.run_all_benchmarks(
                [0.0005] * (2 if mode == "separate_grids" else 1), [2],
                experiment, benchmark.BenchmarkLogger(),
            )
        assert all(result["status"] == "success" for result in results)
    assert len(builds) == 2
    assert len(runs) == (4 if mode == "separate_grids" else 2)
    assert builds[0] is not builds[1]
    assert all(snapshot == (mode == "separate_grids") for _, snapshot in parses)


@pytest.mark.parametrize("invalid", [False, True])
def test_empty_grid_and_invalid_scalar_do_not_read_files(
    adapter_case, monkeypatch, graph_calls, invalid,
):
    adapter, benchmark, experiment, _ = adapter_case
    builds, runs, parses = graph_calls
    real_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        assert invalid or name not in {"numpy", "pandas", "mlx32.engine", "mlx32.model"}
        return real_import(name, *args, **kwargs)

    def reject_read(*args, **kwargs):
        raise AssertionError("Unexpected input file access")

    monkeypatch.setattr(builtins, "__import__", checked_import)
    monkeypatch.setattr(type(adapter.path_comp), "read_bytes", reject_read)
    results = adapter.run_all_benchmarks(
        [0.0, 0.0] if invalid else [], [2], experiment, benchmark.BenchmarkLogger(),
    )
    assert [result["status"] for result in results] == [
        "error: t_run_sec must be finite and positive",
    ] * (2 if invalid else 0)
    assert not builds and not runs and not parses
