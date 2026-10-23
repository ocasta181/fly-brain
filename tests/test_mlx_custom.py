"""Accepted upstream data and selections must survive the MLX adapter."""

import importlib.util
from pathlib import Path

import mlx_engine
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from native_execution import require_metal
import run_mlx
from mlx_backend.inputs import load_connectome
from mlx_backend.models import SimulationRun, SpikeEvents
from mlx_backend.stimuli import neuron_indices


@pytest.fixture
def reference():
    brian = pytest.importorskip("brian2")
    brian.start_scope()
    path = Path(__file__).resolve().parents[1] / "code/paper-phil-drosophila/model.py"
    spec = importlib.util.spec_from_file_location("eon_reference_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return brian, module


def graph_files(tmp_path, identifiers, coefficients):
    completeness, connectivity = tmp_path / "neurons.csv", tmp_path / "edges.parquet"
    pd.DataFrame({"arbitrary_metadata": ["a", "b", "c"]}, index=identifiers).to_csv(completeness)
    pd.DataFrame({
        "Presynaptic_Index": np.array([0, 1, 0], np.int32),
        "Postsynaptic_Index": np.array([1, 2, 2], np.int64),
        "Excitatory x Connectivity": coefficients,
    }).to_parquet(connectivity)
    return completeness, connectivity


@pytest.mark.parametrize("identifiers", [
    ["alpha", "middle", "omega"],
    [99, 2, 17],
    [2**63 + 5, 2**63 + 3, 2**63 + 9],
])
def test_reference_graph_weights_and_external_output_identity(tmp_path, reference, identifiers):
    brian, original = reference
    paths = graph_files(tmp_path, identifiers, [1.25, -2.5, 0.0])
    native = load_connectome(*paths)
    neurons, synapses, monitor = original.create_model(*paths, original.default_params)
    np.testing.assert_array_equal(native.sources, np.asarray(synapses.i[:]))
    np.testing.assert_array_equal(native.destinations, np.asarray(synapses.j[:]))
    # Conversion through Brian's SI-valued quantities can round twice.
    np.testing.assert_allclose(
        native.weights_mv, np.asarray(synapses.w[:] / brian.mV),
        rtol=2 * np.finfo(np.float64).eps, atol=0,
    )
    assert neuron_indices(native, (identifiers[2], identifiers[0], identifiers[2])) == (2, 0, 2)
    events = SpikeEvents(np.array([0, 0]), np.array([2, 0]), np.array([10, 0]))
    actual = run_mlx.spike_table(native, [SimulationRun(events, {}, 0, "test")], "custom")
    output = tmp_path / "spikes.parquet"
    pq.write_table(actual, output)
    exported = pd.read_parquet(output)
    expected = original.construct_dataframe(
        [{0: np.array([0.0]) * brian.second, 2: np.array([0.001]) * brian.second}],
        "custom", dict(enumerate(identifiers)),
    )
    assert exported["flywire_id"].tolist() == expected["flywire_id"].tolist()
    np.testing.assert_array_equal(exported["t"], expected["t"])
    assert actual.schema.field("flywire_id").type == pq.read_schema(output).field("flywire_id").type
    for array in (native.neuron_ids, native.sources, native.destinations, native.counts, native.weights_mv):
        assert not array.flags.writeable


def test_custom_targets_keep_duplicates_classes_zero_rates_and_silencing(tmp_path):
    paths = graph_files(tmp_path, ["a", "b", "c"], [400, -100, 200])
    native = mlx_engine.load_inputs(*paths)
    spec = mlx_engine.experiment_spec({
        "key": "custom", "neu_exc": ["c", "a", "c"], "neu_exc2": ["a"],
        "neu_slnc": ["b", "b"], "stim_rate": 10000.0, "stim_rate2": 0.0,
    })
    inputs, silenced = mlx_engine.stimuli(native, spec, 20, 2, 7)
    assert inputs[0].targets == (2, 0, 2, 0)
    assert spec.activation_classes == (1, 1, 1, 2)
    assert silenced == (1, 1)
    assert inputs[0].events.shape == (1, 20, 4)
    np.testing.assert_array_equal(inputs[0].events[0, :, :3], 1)
    np.testing.assert_array_equal(inputs[0].events[0, :, 3], 0)
    assert inputs[0].trial_indices == (0,) and inputs[1].trial_indices == (1,)


def test_alternate_input_identity_changes_with_graph_weights(tmp_path):
    paths = graph_files(tmp_path, [99, 2, 17], [400, -100, 200])
    first = load_connectome(*paths)
    expected = load_connectome(*paths, first.pin)
    assert first.pin == expected.pin
    graph_files(tmp_path, [99, 2, 17], [401, -100, 200])
    changed = load_connectome(*paths)
    assert first.pin != changed.pin
    with pytest.raises(ValueError, match="hash"):
        load_connectome(*paths, first.pin)


@pytest.mark.parametrize("rate", [-1, float("nan"), float("inf")])
def test_invalid_custom_rates_fail_before_execution(rate):
    with pytest.raises(ValueError, match="rates"):
        mlx_engine.experiment_spec({"neu_exc": [1], "stim_rate": rate})


@pytest.mark.metal
@pytest.mark.parametrize("silenced", [(), ("alpha",)])
def test_custom_dataset_and_overlapping_inputs_match_brian_cpu(
    tmp_path, reference, monkeypatch, silenced
):
    mx = require_metal('Metal is required')
    brian, original = reference
    monkeypatch.setitem(brian.prefs, "codegen.target", "numpy")
    paths = graph_files(tmp_path, ["alpha", "middle", "omega"], [400, -100, 200])
    parameters = dict(original.default_params)
    parameters.update(r_poi=10000 * brian.Hz, r_poi2=0 * brian.Hz)
    neurons, synapses, monitor = original.create_model(*paths, parameters)
    poisson, neurons = original.poi(neurons, [0, 0], [2], parameters)
    original.silence([0] if silenced else [], synapses)
    network = brian.Network(neurons, synapses, monitor, *poisson)
    network.run(8 * brian.ms)
    expected = sorted(zip(
        np.rint(np.asarray(monitor.t / (0.1 * brian.ms))).astype(np.int64),
        np.asarray(monitor.i),
    ))
    connectome = mlx_engine.load_inputs(*paths)
    spec = mlx_engine.experiment_spec({
        "key": "custom", "neu_exc": ["alpha", "alpha"],
        "neu_exc2": ["omega"], "neu_slnc": silenced,
        "stim_rate": 10000.0, "stim_rate2": 0.0,
    })
    inputs, rows = mlx_engine.stimuli(connectome, spec, 80, 1, 42)
    actual = mlx_engine.execute(connectome, inputs[0], rows)
    assert sorted(zip(actual.spikes.steps, actual.spikes.neurons)) == expected
    assert actual.spikes.steps.size > 0
    output = run_mlx.spike_table(connectome, [actual], "custom")
    assert set(output["flywire_id"].to_pylist()) <= {"alpha", "middle", "omega"}
