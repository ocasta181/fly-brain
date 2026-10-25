"""Fixed adapter schemas, source identity, and transactional output tests."""

from __future__ import annotations

import csv
import os
import platform
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


SPIKE_SCHEMA = pa.schema([
    ("t", pa.float64()), ("time_ms", pa.float64()), ("trial", pa.int64()),
    ("neuron_index", pa.int64()), ("flywire_id", pa.int64()), ("exp_name", pa.string()),
])
EVENTS = np.array([[1, 0, 1], [1, 1, 1], [3, 0, 1], [3, 1, 1]], dtype=np.int64)


@pytest.fixture
def adapter_case(tmp_path, monkeypatch):
    import benchmark
    import run_mlx as adapter

    ids = [2**53 + 101, 2**53 + 3]
    completeness = tmp_path / "completeness.csv"
    connectivity = tmp_path / "connectivity.parquet"
    pd.DataFrame({"complete": [True, True]}, index=ids).to_csv(completeness)
    pd.DataFrame({
        "Presynaptic_Index": pd.Series(dtype="int64"),
        "Postsynaptic_Index": pd.Series(dtype="int64"),
        "Excitatory x Connectivity": pd.Series(dtype="int64"),
        "unused_metadata": pd.Series(dtype="string"),
    }).to_parquet(connectivity, index=False)
    monkeypatch.setattr(adapter, "path_comp", completeness)
    monkeypatch.setattr(adapter, "path_con", connectivity)
    monkeypatch.setattr(benchmark, "path_res", tmp_path / "results")
    monkeypatch.setattr(benchmark, "csv_path", tmp_path / "benchmark.csv")
    monkeypatch.delenv(benchmark.SPIKE_IO_ENV_VAR, raising=False)
    experiment = {
        "name": "Fixed adapter case", "key": "fixed_adapter",
        "neu_exc": [ids[1]], "neu_exc2": [ids[1], ids[0], ids[0]],
        "neu_slnc": [ids[0]], "stim_rate": 10_000.0, "seed": 19,
    }
    return adapter, benchmark, experiment, ids


@pytest.fixture
def fake_engine(monkeypatch):
    """Exercise adapter I/O independently of Metal or the numerical oracle."""
    from mlx_backend import engine, model

    calls, builds = [], []
    real_make_graph = model.make_graph

    def make_graph(*args):
        graph = real_make_graph(*args)
        builds.append(graph)
        return graph

    def run_batch(graph, steps, drive, **options):
        calls.append((graph, steps, drive, options))
        events = EVENTS if drive.rates_hz.any() else np.empty((0, 3), dtype=np.int64)
        counts = np.zeros((2, graph.n_neurons), dtype=np.int64)
        if len(events):
            np.add.at(counts, (events[:, 1], events[:, 2]), 1)
        if options["record"] and options.get("spike_sink") is not None:
            for part in (events[:2], events[2:]):
                if len(part):
                    options["spike_sink"](part.copy())
        timings = dict.fromkeys((
            "setup_compile_s", "compile_warmup_s", "runtime_s", "simulation_s",
            "input_generation_staging_s", "collection_s", "snapshots_s",
        ), 0.0)
        timings["simulation_s"] = timings["runtime_s"] = 0.000001
        return SimpleNamespace(counts=counts, timings=timings)

    monkeypatch.setattr(model, "make_graph", make_graph)
    monkeypatch.setattr(engine, "run_batch", run_batch)
    return calls, builds


def assert_table(table, ids):
    assert table.schema == SPIKE_SCHEMA
    seconds = np.array([1, 1, 3, 3], dtype=np.float64) * 0.0001
    np.testing.assert_array_equal(table["t"].to_numpy(), seconds)
    np.testing.assert_array_equal(table["time_ms"].to_numpy(), seconds * 1000)
    assert table["trial"].to_pylist() == [0, 1, 0, 1]
    assert table["neuron_index"].to_pylist() == [1, 1, 1, 1]
    assert table["flywire_id"].to_pylist() == [ids[1]] * 4
    assert table["exp_name"].to_pylist() == ["mlx_t0.0005s_n2"] * 4


def test_adapter_exports_typed_ordered_rows_labels_rounds_csv_and_manifest(
    adapter_case, fake_engine, monkeypatch,
):
    adapter, benchmark, experiment, ids = adapter_case
    monkeypatch.setattr(adapter, "_ARROW_ROWS", 3)
    results = adapter.run_benchmarks(
        [0.0005], [2], experiment, benchmark.BenchmarkLogger(),
        run_label="fixed_case", round_idx=7,
    )
    result, = results
    target = benchmark.get_spike_output_path("mlx_t0.0005s_n2", "fixed_case", 7)
    assert {key: value for key, value in result.items() if key != "timings"} == {
        "t_run_sec": 0.0005, "n_run": 2, "n_active_neurons": 1, "n_spikes": 4,
        "backend_key": "mlx", "experiment_name": experiment["name"],
        "experiment_key": experiment["key"], "run_label": "fixed_case", "round": 7,
        "spike_path": str(target), "status": "success",
    }
    with pq.ParquetFile(target) as stored:
        assert [stored.metadata.row_group(i).num_rows
                for i in range(stored.metadata.num_row_groups)] == [3, 1]
        assert all(stored.metadata.row_group(i).column(j).compression == "BROTLI"
                   for i in range(stored.metadata.num_row_groups)
                   for j in range(len(SPIKE_SCHEMA)))
        assert_table(stored.read(), ids)
    assert list(target.parent.iterdir()) == [target]
    calls, builds = fake_engine
    (graph, steps, drive, options), = calls
    assert builds == [graph] and steps == 5
    assert graph.neuron_ids.tolist() == ids
    assert drive.targets.tolist() == [1, 1, 0, 0]
    assert drive.rates_hz.tolist() == [10000, 0, 0, 0]
    assert drive.seed == 19
    assert options["silenced"] == [0]
    assert options["trials"] == (0, 1) and options["round_idx"] == 7
    timings = result["timings"]
    assert all(np.isfinite(value) and value >= 0 for value in timings.values())
    assert timings["simulation_total"] == timings["simulation_s"]
    assert timings["device_build"] == timings["compile_warmup_s"]
    with benchmark.csv_path.open(newline="") as saved:
        row, = csv.DictReader(saved)
    assert list(row) == benchmark.CSV_COLUMNS
    assert row["backend_key"] == "mlx" and row["round"] == "7"
    assert row["spikes"] == "4" and row["spike_path"] == str(target)
    with (benchmark.path_res / "fixed_case" / "manifest.csv").open(newline="") as saved:
        manifest, = csv.DictReader(saved)
    assert manifest["spike_path"] == str(target)
    assert manifest["spike_time_column"] == "time_ms"
    assert manifest["spike_time_unit"] == "ms"


@pytest.mark.parametrize("record", [True, False])
def test_empty_typed_output_or_recording_off(adapter_case, fake_engine, monkeypatch, record):
    adapter, benchmark, experiment, _ = adapter_case
    if not record:
        monkeypatch.setenv(benchmark.SPIKE_IO_ENV_VAR, "true")
    result = adapter.run_single_benchmark(
        0.0005, 2, dict(experiment, stim_rate=0.0), benchmark.BenchmarkLogger(),
    )
    assert result["status"] == "success"
    assert result["n_spikes"] == result["n_active_neurons"] == 0
    if record:
        target = benchmark.get_spike_output_path("mlx_t0.0005s_n2")
        with pq.ParquetFile(target) as stored:
            assert stored.schema_arrow == SPIKE_SCHEMA
            assert stored.metadata.num_rows == stored.metadata.num_row_groups == 0
        assert list(target.parent.iterdir()) == [target]
    else:
        assert not result["spike_path"] and not benchmark.path_res.exists()
    calls, _ = fake_engine
    assert calls[0][3]["record"] is record


@pytest.mark.parametrize("failure", ["stream", "tail", "close"])
def test_export_failure_closes_writer_preserves_existing_target_and_cleans_temporary(
    adapter_case, fake_engine, monkeypatch, failure,
):
    adapter, benchmark, experiment, _ = adapter_case
    target = benchmark.get_spike_output_path("mlx_t0.0005s_n2")
    real_writer = pq.ParquetWriter
    writers = []

    class FailingWriter:
        def __init__(self, *args, **kwargs):
            self.writer = real_writer(*args, **kwargs)
            self.writes = self.closes = 0
            writers.append(self)

        def write_table(self, table, **kwargs):
            if failure == "tail" or (failure == "stream" and self.writes):
                raise OSError("injected export failure")
            self.writer.write_table(table, **kwargs)
            self.writes += 1

        def close(self):
            self.closes += 1
            self.writer.close()
            if failure == "close":
                raise OSError("injected close failure")

    if failure == "stream":
        monkeypatch.setattr(adapter, "_ARROW_ROWS", 1)
    monkeypatch.setattr(pq, "ParquetWriter", FailingWriter)
    result = adapter.run_single_benchmark(0.0005, 2, experiment, benchmark.BenchmarkLogger())
    assert result["status"].startswith("error:")
    assert "injected" in result["status"]
    assert result["n_spikes"] == result["n_active_neurons"] == 0
    assert not result["spike_path"]
    assert len(writers) == 1 and writers[0].closes >= 1
    assert not target.exists() and not list(target.parent.iterdir())


@pytest.mark.parametrize("source", ["completeness", "connectivity", "missing"])
def test_grid_cache_checks_current_content_even_when_size_and_mtime_match(
    adapter_case, fake_engine, monkeypatch, source,
):
    adapter, benchmark, experiment, ids = adapter_case
    path = adapter.path_comp if source == "completeness" else adapter.path_con
    if source == "connectivity":
        pd.DataFrame({"Presynaptic_Index": [0], "Postsynaptic_Index": [1],
                      "Excitatory x Connectivity": [1]}).to_parquet(path, index=False)
    original, stat = path.read_bytes(), path.stat()
    if source == "completeness":
        lines = original.splitlines(keepends=True)
        replacement = b"".join([lines[0], lines[2], lines[1]])
    elif source == "connectivity":
        pd.DataFrame({"Presynaptic_Index": [0], "Postsynaptic_Index": [1],
                      "Excitatory x Connectivity": [2]}).to_parquet(path, index=False)
        replacement = path.read_bytes()
        path.write_bytes(original)
    else:
        replacement = original
    assert len(replacement) == len(original)
    saved = []
    real_save = adapter.save_result_csv

    def save(backend, result):
        real_save(backend, result)
        saved.append(result)
        if len(saved) == 1:
            if source == "missing":
                path.unlink()
            else:
                path.write_bytes(replacement)
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        elif len(saved) == 2 and source == "missing":
            path.write_bytes(original)

    monkeypatch.setattr(adapter, "save_result_csv", save)
    results = adapter.run_benchmarks(
        [0.0005, 0.0006, 0.0007] if source == "missing" else [0.0005, 0.0006], [2], experiment,
        benchmark.BenchmarkLogger(),
    )
    assert results == saved and results[0]["status"] == results[-1]["status"] == "success"
    calls, builds = fake_engine
    assert len(calls) == 2
    assert len(builds) == (1 if source == "missing" else 2)
    assert (calls[0][0] is calls[1][0]) == (source == "missing")
    if source == "completeness":
        assert builds[1].neuron_ids.tolist() == ids[::-1]
        assert calls[1][2].targets.tolist() == [0, 0, 1, 1]
    elif source == "connectivity":
        assert builds[0].contacts.tolist() == [1] and builds[1].contacts.tolist() == [2]
    else:
        assert results[1]["status"].startswith("error:")


@pytest.mark.parametrize("duration,trials", [(0.0, 2), (0.00015, 2), (0.0005, True), (0.0005, 0)])
def test_invalid_adapter_dimensions_do_not_read_sources(
    adapter_case, fake_engine, monkeypatch, duration, trials,
):
    adapter, benchmark, experiment, _ = adapter_case

    def reject_read(*args, **kwargs):
        raise AssertionError("invalid dimensions reached source loading")

    monkeypatch.setattr(pd, "read_csv", reject_read)
    result = adapter.run_single_benchmark(duration, trials, experiment, benchmark.BenchmarkLogger())
    assert result["status"].startswith("error:") and "source loading" not in result["status"]
    assert fake_engine == ([], [])


def test_adapter_runs_actual_metal_without_brian_cpu_steps(adapter_case, monkeypatch):
    if sys.platform != "darwin" or platform.machine() != "arm64":
        pytest.skip("Native adapter test requires Apple Silicon Metal")
    import brian2 as brian
    import mlx.core as mx

    assert mx.metal.is_available(), "This Apple Silicon qualification requires actual Metal"

    def reject_cpu(*args, **kwargs):
        raise AssertionError("The adapter attempted Brian CPU timestepping")

    monkeypatch.setattr(brian.Network, "run", reject_cpu)
    monkeypatch.setattr(brian, "run", reject_cpu)
    adapter, benchmark, experiment, ids = adapter_case
    result = adapter.run_single_benchmark(
        0.0005, 2, experiment, benchmark.BenchmarkLogger(), run_label="native", round_idx=9,
    )
    assert result["status"] == "success" and result["n_spikes"] == 4
    assert result["timings"]["simulation_s"] > 0
    assert_table(pq.read_table(result["spike_path"]), ids)


def test_saved_fixed_output_feeds_existing_shared_comparisons(adapter_case, fake_engine):
    from compare_spike_outputs import compare_pair, load_spikes, prepare_metrics

    adapter, benchmark, experiment, ids = adapter_case
    result = adapter.run_single_benchmark(
        0.0005, 2, experiment, benchmark.BenchmarkLogger(), run_label="compare", round_idx=2,
    )
    assert result["status"] == "success"
    reference = benchmark.get_spike_output_path("brian2cpp_t0.0005s_n2", "compare", 2)
    pd.DataFrame({
        "t": EVENTS[:, 0].astype(np.float64) * 0.0001,
        "trial": EVENTS[:, 1], "flywire_id": [ids[1]] * len(EVENTS),
        "exp_name": "brian2cpp_t0.0005s_n2",
    }).to_parquet(reference, index=False)
    cpu, _ = load_spikes("brian2cpp", 0.0005, 2, "compare", 2)
    native, _ = load_spikes("mlx", 0.0005, 2, "compare", 2)
    assert native.flywire_id.tolist() == cpu.flywire_id.tolist() == [ids[1]] * 4
    pair = compare_pair("brian2cpp", prepare_metrics(cpu, 0.0005, 2),
                        "mlx", prepare_metrics(native, 0.0005, 2), 0.0005, 2, 0.000001)
    assert pair["timing_matches"] == 4 and pair["timing_f1"] == 1
    assert pair["active_jaccard"] == 1 and pair["rate_rmse_hz_shared"] == 0
