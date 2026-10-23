"""Recording changes must leave dynamics intact and timing workflows useful."""

import csv
import hashlib
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import benchmark
import mlx_engine
import numpy as np
import pytest
from native_execution import require_metal
import run_mlx
from mlx_backend.experiments import EXPERIMENTS
from mlx_backend.models import Connectome, SimulationRun, SpikeEvents, Stimulus

ROOT = Path(__file__).resolve().parents[1]


def empty_spikes():
    return SpikeEvents(*(np.array([], np.int64) for _ in range(3)))


def sugar_connectome():
    return Connectome(
        np.array(EXPERIMENTS["sugar"].activated_ids, dtype=np.int64),
        np.array([], np.int32),
        np.array([], np.int32),
        np.array([], np.int32),
        np.array([], np.float64),
    )


def test_timing_only_results_skip_export_and_preserve_round_streams(tmp_path, monkeypatch):
    monkeypatch.setenv(benchmark.SPIKE_IO_ENV_VAR, "1")
    monkeypatch.setattr(benchmark, "path_res", tmp_path)
    monkeypatch.setattr(benchmark, "csv_path", tmp_path / "benchmark.csv")
    monkeypatch.setattr(mlx_engine, "configure", lambda: None)
    monkeypatch.setattr(mlx_engine, "load_inputs", lambda *args: sugar_connectome())
    captured = []

    def execute(connectome, stimulus, silenced, *, engine, record):
        assert record is False
        captured.append(stimulus)
        return SimulationRun(
            empty_spikes(), {"setup_s": 0.0, "simulation_wall_s": 0.001}, 0, "test"
        )

    def unexpected_export(*args, **kwargs):
        pytest.fail("Timing-only mode must not collect/export a spike table")

    monkeypatch.setattr(mlx_engine, "execute", execute)
    monkeypatch.setattr(run_mlx, "spike_table", unexpected_export)
    monkeypatch.setattr(run_mlx, "get_spike_output_path", unexpected_export)
    experiment = benchmark.get_experiment("sugar")
    logger = benchmark.BenchmarkLogger()
    results = []
    for round_idx in (None, 1, 2, 2, 3):
        results.append(
            run_mlx.run_single_benchmark(
                0.1, 1, experiment, logger, run_label="rounds", round_idx=round_idx
            )
        )
    assert captured[0].seed == captured[1].seed == 20261004
    assert captured[0].sha256 == captured[1].sha256
    assert captured[2].seed == captured[3].seed != captured[4].seed
    assert captured[2].sha256 == captured[3].sha256
    assert len({captured[i].sha256 for i in (0, 2, 4)}) == 3
    for result in results:
        assert result["status"] == "success"
        assert result["n_spikes"] == result["n_active_neurons"] == 0
        assert result["spike_path"] == ""
        assert result["timings"]["result_collection"] == 0
        assert result["timings"]["result_save"] == 0
        assert result["timings"]["simulation_total"] > 0
        benchmark.save_result_csv(benchmark.BACKEND_NAMES["mlx"], result)
    assert not list(tmp_path.rglob("*.parquet"))
    assert not list(tmp_path.rglob("*.mlx.json"))
    assert not list(tmp_path.rglob("manifest.csv"))
    rows = list(csv.DictReader((tmp_path / "benchmark.csv").open()))
    assert len(rows) == 4
    assert {row["round"] for row in rows} == {"", "1", "2", "3"}
    assert all(row["spike_path"] == "" and row["spikes"] == "0" for row in rows)


def test_explicit_replay_is_unchanged_across_rounds(monkeypatch):
    monkeypatch.setenv(benchmark.SPIKE_IO_ENV_VAR, "1")
    monkeypatch.setattr(mlx_engine, "configure", lambda: None)
    connectome = sugar_connectome()
    # The retained archive uses rows from the native network; arrange those rows
    # while keeping this host-only test independent of the large dataset loader.
    replay = ROOT / "tests/data/mlx-replay/sugar"
    import json

    metadata = json.loads((replay / "stimulus.json").read_text())
    identifiers = np.arange(max(metadata["targets"]) + 1, dtype=np.int64)
    identifiers[metadata["targets"]] = EXPERIMENTS["sugar"].activated_ids
    connectome = replace(connectome, neuron_ids=identifiers)
    monkeypatch.setattr(mlx_engine, "load_inputs", lambda *args: connectome)
    captured = []

    def execute(connectome, stimulus, silenced, **kwargs):
        captured.append(stimulus)
        return SimulationRun(
            empty_spikes(), {"setup_s": 0.0, "simulation_wall_s": 0.001}, 0, "test"
        )

    monkeypatch.setattr(mlx_engine, "execute", execute)
    for round_idx in (1, 7):
        run_mlx.run_single_benchmark(
            0.1, 1, benchmark.get_experiment(), benchmark.BenchmarkLogger(),
            round_idx=round_idx, stimulus_path=replay,
        )
    assert captured[0].seed == captured[1].seed == metadata["seed"]
    assert captured[0].sha256 == captured[1].sha256


def metal():
    mx = require_metal('Metal is required')
    return mx


@pytest.mark.metal
@pytest.mark.parametrize("engine", ["bucketed", "window"])
def test_disabling_recording_preserves_native_final_state(monkeypatch, engine):
    metal()
    from mlx_backend import configured, runner
    if engine == "bucketed":
        from mlx_reference import runner

    connectome = Connectome(
        np.arange(3, dtype=np.int64),
        np.array([0, 0, 1], np.int32),
        np.array([1, 2, 2], np.int32),
        np.array([400, -100, 200], np.int32),
        np.array([110, -27.5, 55], np.float64),
    )
    events = np.zeros((2, 77, 1), np.uint8)
    events[:, [0, 3, 16, 17, 18, 34, 35], :] = 1
    events[1, 3] = 0
    stimulus = Stimulus(
        events, (0,), (200.0,), (4, 9), 0, 0,
        hashlib.sha256(events.tobytes()).hexdigest(),
    )
    final = []
    if engine == "window":
        run_prepared = configured.run_prepared

        def capture(*args, **kwargs):
            result, state = run_prepared(*args, **kwargs, return_state=True)
            final[:] = [state]
            return result

        monkeypatch.setattr(configured, "run_prepared", capture)
        def snapshot():
            values = {name: np.array(value) for name, value in final[0].values.items()}
            values["release"] = np.array(final[0].release)
            for kind in ("history", "due"):
                values.update({f"{kind}:{key}": np.array(value)
                               for key,value in getattr(final[0],kind).items()})
            return values
    else:
        prepare = runner.prepare

        def capture_prepare(*args, **kwargs):
            execution = prepare(*args, **kwargs)

            def capture(state, inputs):
                state, trace = execution.advance(state, inputs)
                final[:] = [state]
                return state, trace

            return replace(execution, advance=capture)

        monkeypatch.setattr(runner, "prepare", capture_prepare)
        names = ("voltage_offset_mv", "synaptic_mv", "last_spike_step", "queue")
        def snapshot():
            return {name: np.array(getattr(final[0], name)) for name in names}
    recorded = runner.run(connectome, stimulus, (), precision="0", engine=engine)
    assert recorded.spikes.steps.size > 0
    assert final[0].step == 77
    expected = snapshot()

    def unexpected_collection(*args, **kwargs):
        pytest.fail("Recording-disabled simulation invoked host spike collection")

    if engine == "window":
        monkeypatch.setattr(configured, "_collect_unit_windows", unexpected_collection)
    else:
        class WithoutRasterCollection:
            def __getattr__(self, name):
                if name in {"asarray", "nonzero"}:
                    return unexpected_collection
                return getattr(np, name)

        monkeypatch.setattr(runner, "np", WithoutRasterCollection())
    unrecorded = runner.run(
        connectome, stimulus, (), precision="0", engine=engine, record=False
    )
    assert final[0].step == 77
    actual_state = snapshot()
    assert actual_state.keys() == expected.keys()
    for name,actual in actual_state.items():
        if actual.dtype == np.float32:
            actual = actual.view(np.uint32)
            expected[name] = expected[name].view(np.uint32)
        np.testing.assert_array_equal(actual, expected[name])
    for name in ("trials", "neurons", "steps"):
        values = getattr(unrecorded.spikes, name)
        assert values.size == 0 and values.dtype == np.int64
    assert unrecorded.timings["simulation_wall_s"] > 0


@pytest.mark.metal
def test_native_timing_only_cli_keeps_round_csv_without_spike_outputs(tmp_path):
    metal()
    script = f"""
import runpy, sys
from pathlib import Path
sys.path.insert(0, {str(ROOT / 'code')!r})
import benchmark
benchmark.path_res = Path({str(tmp_path)!r})
benchmark.csv_path = benchmark.path_res / 'benchmark.csv'
sys.argv = ['main.py', '--mlx', '--disable-spike-io', '--no_log_file',
            '--t_run', '0.1', '--n_run', '1', '--rounds', '2',
            '--round-start', '3', '--run-label', 'timing-only']
runpy.run_path({str(ROOT / 'main.py')!r}, run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT,
        env={**os.environ, "MLX_ENABLE_TF32": "0"},
        text=True, capture_output=True, timeout=90, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = list(csv.DictReader((tmp_path / "benchmark.csv").open()))
    assert len(rows) == 2 and {row["round"] for row in rows} == {"3", "4"}
    assert all(
        row["status"] == "success" and row["spikes"] == "0"
        and row["active_neurons"] == "0" and row["spike_path"] == ""
        and float(row["sim_time"]) > 0 for row in rows
    )
    assert not list(tmp_path.rglob("*.parquet"))
    assert not list(tmp_path.rglob("*.mlx.json"))
    assert not list(tmp_path.rglob("manifest.csv"))
