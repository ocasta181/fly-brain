"""Tiny file-to-Metal adapter checks with literal discrete spike expectations."""

import csv
import math

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


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


def assert_native_dispatches(native_launches, before, *, record):
    launches = native_launches[before:]
    # Five disposable warmup ticks plus five requested ticks, independent of B=2.
    assert launches.count("eon32_state") == 10
    assert launches.count("eon32_propagate") == 10
    assert launches.count("eon32_spike_events") == (2 if record else 0)


def test_recorded_adapter_schema_events_results_and_csv(
    adapter_case, native, native_launches,
):
    adapter, benchmark, experiment, ids = adapter_case
    before = len(native_launches)
    results = adapter.run_all_benchmarks(
        [0.0005], [2], experiment, benchmark.BenchmarkLogger(),
        run_label="adapter_case", round_idx=7,
    )
    assert_native_dispatches(native_launches, before, record=True)
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
        table = stored.read()
    assert table.schema == pa.schema([
        ("t", pa.float64()), ("time_ms", pa.float64()), ("trial", pa.int64()),
        ("neuron_index", pa.int64()), ("flywire_id", pa.int64()),
        ("exp_name", pa.string()),
    ])
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
        0, timings["collection_s"] - timings["sink_collection"] - timings["sink_export"]
    )
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


def test_export_failure_preserves_target_and_removes_own_temporary_file(
    adapter_case, monkeypatch, native, native_launches,
):
    adapter, benchmark, experiment, _ = adapter_case
    target = benchmark.get_spike_output_path("mlx32_t0.0005s_n2")
    target.parent.mkdir(parents=True)
    pq.write_table(pa.table({"existing": [29]}), target)
    original = target.read_bytes()
    real_writer = pq.ParquetWriter
    writers = []

    class FailAfterWrite:
        def __init__(self, *args, **kwargs):
            self.writer = real_writer(*args, **kwargs)
            self.writes = self.closes = 0
            writers.append(self)

        def write_table(self, table, **kwargs):
            if self.writes:
                raise OSError("injected parquet export failure")
            self.writer.write_table(table, **kwargs)
            self.writes += 1

        def close(self):
            self.closes += 1
            self.writer.close()

    monkeypatch.setattr(adapter, "_ARROW_ROWS", 1)
    monkeypatch.setattr(pq, "ParquetWriter", FailAfterWrite)
    before = len(native_launches)
    result = adapter.run_single_benchmark(
        0.0005, 2, experiment, benchmark.BenchmarkLogger(),
    )
    assert_native_dispatches(native_launches, before, record=True)
    assert result["status"] == "error: injected parquet export failure"
    assert result["n_spikes"] == result["n_active_neurons"] == 0
    assert result["spike_path"] == ""
    assert len(writers) == 1
    assert writers[0].writes == writers[0].closes == 1
    assert target.read_bytes() == original
    assert list(target.parent.iterdir()) == [target]
