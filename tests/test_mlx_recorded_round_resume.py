"""Recorded benchmark rounds resume without changing previous native outputs."""
from native_execution import forbidden_call, require_metal
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

import benchmark
import mlx_engine
import run_mlx


@pytest.mark.metal
def test_recorded_round_resume_preserves_prior_files_and_cpu_spikes(tmp_path, monkeypatch):
    mx = require_metal('Metal is required')
    from brian2.codegen.runtime.cython_rt.cython_rt import CythonCodeObject
    from brian2.codegen.runtime.numpy_rt.numpy_rt import NumpyCodeObject
    from configured_reference import reference
    from mlx_backend.model import resolve_model

    neurons, edges = tmp_path / 'neurons.csv', tmp_path / 'edges.parquet'
    pd.DataFrame(index=['left', 'middle', 'right']).to_csv(neurons)
    pd.DataFrame({
        'Presynaptic_Index': [0, 0, 1], 'Postsynaptic_Index': [1, 2, 2],
        'Excitatory x Connectivity': np.array([400, -100, 200], np.int32),
    }).to_parquet(edges)
    monkeypatch.setattr(run_mlx, 'path_comp', neurons)
    monkeypatch.setattr(run_mlx, 'path_con', edges)
    monkeypatch.setattr(benchmark, 'path_res', tmp_path / 'results')
    monkeypatch.setattr(benchmark, 'csv_path', benchmark.path_res / 'benchmark.csv')
    monkeypatch.delenv(benchmark.SPIKE_IO_ENV_VAR, raising=False)
    experiment = dict(key='custom', name='custom', neu_exc=['left'], neu_exc2=['right'],
                      neu_slnc=[], stim_rate=7300., stim_rate2=4100.)
    duration, trials, label = .005, 2, 'recorded-resume'

    forbid_cpu = forbidden_call('Benchmark workflow executed a CPU model timestep')

    def run_rounds(rounds, start):
        with monkeypatch.context() as trap:
            trap.setattr(CythonCodeObject, 'run', forbid_cpu)
            trap.setattr(NumpyCodeObject, 'run', forbid_cpu)
            return benchmark.run_benchmarks(
                ['mlx'], [duration], [trials], experiment, benchmark.BenchmarkLogger(),
                rounds=rounds, run_label=label, round_start=start)['mlx']

    first = run_rounds(2, 1)
    old_files = {path: path.read_bytes() for path in (benchmark.path_res / label).rglob('*')
                 if path.is_file() and path.suffix in ('.parquet', '.json')}
    old_manifest = list(csv.DictReader((benchmark.path_res / label / 'manifest.csv').open()))
    old_csv = list(csv.DictReader(benchmark.csv_path.open()))
    assert len(old_manifest) == len(old_csv) == 2
    # Resumption also crosses the retained-to-streamed output boundary.
    monkeypatch.setattr(mlx_engine, 'MAX_EAGER_EVENT_BYTES', 0)
    last = run_rounds(1, 3)
    assert all(path.read_bytes() == data for path, data in old_files.items())
    manifest = list(csv.DictReader((benchmark.path_res / label / 'manifest.csv').open()))
    rows = list(csv.DictReader(benchmark.csv_path.open()))
    assert manifest[:2] == old_manifest and rows[:2] == old_csv
    assert [row['round'] for row in manifest] == ['1', '2', '3']
    assert [row['round'] for row in rows] == ['1', '2', '3']

    graph = mlx_engine.load_inputs(neurons, edges)
    spec = mlx_engine.experiment_spec(experiment)
    model = resolve_model(dt_ms=.1)
    event_hashes = []
    for number, result in enumerate(first + last, 1):
        path = Path(result['spike_path'])
        metadata = json.loads(path.with_suffix('.mlx.json').read_text())
        actual = pq.read_table(path)
        assert result['status'] == 'success'
        assert metadata['spike_sha256'] == sha256(path.read_bytes()).hexdigest()
        assert metadata['round'] == number and metadata['record_spikes'] is True
        assert metadata['output_mode'] == ('bounded-spool' if number == 3 else 'retained')
        assert metadata['input_sha256'] == {
            'completeness': sha256(neurons.read_bytes()).hexdigest(),
            'connectivity': sha256(edges.read_bytes()).hexdigest(),
        }
        inputs, silenced = mlx_engine.stimuli(
            graph, spec, model.computed_steps(duration), trials, metadata['effective_seed'],
            stream=False)
        expected = [reference(graph, stimulus, model, silenced) for stimulus in inputs]
        table = run_mlx.spike_table(graph, expected, path.stem)
        assert actual.equals(table)
        assert result['n_spikes'] == actual.num_rows > 0
        assert set(actual['trial'].to_pylist()) == {0, 1}
        assert manifest[number-1]['spikes'] == str(actual.num_rows)
        assert manifest[number-1]['spike_time_column'] == 'time_ms'
        assert manifest[number-1]['spike_time_unit'] == 'ms'
        event_hashes.append(tuple(metadata['canonical_event_sha256']))
    assert len(set(event_hashes)) == 3
