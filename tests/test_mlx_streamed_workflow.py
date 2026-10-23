"""Actual native execution must produce identical bounded and retained exports."""

import json
from pathlib import Path

import benchmark
import mlx_engine
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from native_execution import require_metal as _require_metal
import run_mlx
from mlx_backend.inputs import file_sha256

pytestmark = pytest.mark.metal


def require_metal():
    mx = _require_metal('Metal is required')


@pytest.mark.parametrize("engine,strategy,duration", [
    ("window", "batch", 0.05),
    ("window", "sequential", 0.05),
    ("bucketed", "sequential", 0.005),
    ("window", "batch", 0.0),
])
def test_streamed_native_workflow_matches_retained_tables(
    tmp_path, monkeypatch, engine, strategy, duration
):
    require_metal()
    if engine == "bucketed":
        from mlx_backend import runner
        from mlx_reference.runner import run as run_reference

        monkeypatch.setattr(runner, "run", run_reference)
    completeness, connectivity = tmp_path / "neurons.csv", tmp_path / "edges.parquet"
    pd.DataFrame(index=["alpha", "middle", "omega"]).to_csv(completeness)
    pd.DataFrame({
        "Presynaptic_Index": [0, 0, 1], "Postsynaptic_Index": [1, 2, 2],
        "Excitatory x Connectivity": [400, -100, 200],
    }).to_parquet(connectivity)
    monkeypatch.setattr(run_mlx, "path_comp", completeness)
    monkeypatch.setattr(run_mlx, "path_con", connectivity)
    monkeypatch.setattr(benchmark, "path_res", tmp_path / "results")
    monkeypatch.delenv(benchmark.SPIKE_IO_ENV_VAR, raising=False)
    experiment = dict(key="custom", name="custom", neu_exc=["alpha"],
                      neu_exc2=[], neu_slnc=[], stim_rate=10000.0)
    retained = run_mlx.run_single_benchmark(
        duration, 2, experiment, benchmark.BenchmarkLogger(),
        engine=engine, strategy=strategy, run_label="retained", stream_output=False,
    )
    expected = pq.read_table(retained["spike_path"])
    monkeypatch.setattr(mlx_engine, "MAX_EAGER_EVENT_BYTES", 0)
    executed = []
    execute = mlx_engine.execute

    def capture(*args, **kwargs):
        assert kwargs.get("spike_sink") is not None
        result = execute(*args, **kwargs)
        assert result.spikes.steps.size == 0
        executed.append(result)
        return result

    monkeypatch.setattr(mlx_engine, "execute", capture)
    streamed = run_mlx.run_single_benchmark(
        duration, 2, experiment, benchmark.BenchmarkLogger(),
        engine=engine, strategy=strategy, run_label="streamed", stream_output=True,
    )
    actual = pq.read_table(streamed["spike_path"])
    assert actual.equals(expected)
    assert streamed["n_spikes"] == retained["n_spikes"] == actual.num_rows
    assert streamed["n_active_neurons"] == retained["n_active_neurons"]
    if duration:
        assert actual.num_rows > 0
    metadata = json.loads(Path(streamed["spike_path"]).with_suffix(".mlx.json").read_text())
    earlier = json.loads(Path(retained["spike_path"]).with_suffix(".mlx.json").read_text())
    assert metadata["output_mode"] == "bounded-spool"
    assert metadata["canonical_event_sha256"] == earlier["canonical_event_sha256"]
    assert metadata["spike_sha256"] == file_sha256(Path(streamed["spike_path"]))
    assert streamed["timings"]["result_save"] >= sum(
        result.timings.get("spike_spool_io_s", 0.0) for result in executed
    )


@pytest.mark.parametrize("chunk_steps", [113, 18432])
def test_streamed_full_native_sugar_reproduces_independent_cpu_fixture(tmp_path, monkeypatch, chunk_steps):
    require_metal()
    from mlx_backend import configured

    prepared_run = configured.run_prepared
    monkeypatch.setattr(configured, "run_prepared", lambda *args, **kwargs: prepared_run(
        *args, **kwargs, chunk_steps=chunk_steps
    ))
    monkeypatch.setattr(benchmark, "path_res", tmp_path)
    monkeypatch.setattr(mlx_engine, "MAX_EAGER_EVENT_BYTES", 0)
    monkeypatch.delenv(benchmark.SPIKE_IO_ENV_VAR, raising=False)
    result = run_mlx.run_single_benchmark(
        1.0, 1, benchmark.get_experiment("sugar"), benchmark.BenchmarkLogger(),
        run_label="streamed-sugar", stream_output=True,
    )
    actual = pq.read_table(result["spike_path"])
    fixture = Path(__file__).parent / "data/mlx-replay/sugar"
    with np.load(fixture / "spikes.npz", allow_pickle=False) as expected:
        np.testing.assert_array_equal(actual["neuron_index"].to_numpy(), expected["neuron_rows"])
        np.testing.assert_array_equal(
            np.rint(actual["time_ms"].to_numpy() / 0.1).astype(np.int64), expected["steps"]
        )
        np.testing.assert_array_equal(actual["flywire_id"].to_numpy(), expected["flywire_ids"])
    metadata = json.loads(Path(result["spike_path"]).with_suffix(".mlx.json").read_text())
    stimulus = json.loads((fixture / "stimulus.json").read_text())
    assert metadata["canonical_event_sha256"] == stimulus["per_trial_event_sha256"]
    assert metadata["output_mode"] == "bounded-spool"
