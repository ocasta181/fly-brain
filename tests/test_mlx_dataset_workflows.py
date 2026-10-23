"""Real alternate files, frozen Eon/Brian CPU, native Metal, and Parquet.

The CPU oracle loads the files itself through the original create_model;
neither native graph arrays nor native synaptic weights define its network.
"""

import importlib.util
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import mlx_engine
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from native_execution import require_metal
import run_mlx

from mlx_backend.export import SpoolSpikes
from mlx_backend.inputs import load_connectome
from mlx_backend.model import resolve_model
from mlx_backend.stimuli import neuron_indices


ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]
IDS = [
    [99, -7, 2, 17],
    ["zebra", "fly-α", "001", "middle"],
    [2**64 - 1, 2**63 + 5, 2**63 + 1, 2**63 + 17],
]


@pytest.fixture
def original(monkeypatch):
    b = pytest.importorskip("brian2")
    b.start_scope()
    monkeypatch.setitem(b.prefs, "codegen.target", "numpy")
    monkeypatch.setitem(b.prefs, "codegen.runtime.cython.cache_dir",
                        str(ROOT / "executions/brian-cache/dataset-numpy"))
    monkeypatch.setattr(b.defaultclock, "dt", .1 * b.ms)
    spec = importlib.util.spec_from_file_location(
        "dataset_frozen_eon", ROOT / "code/paper-phil-drosophila/model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return b, module


@pytest.fixture
def metal():
    mx = require_metal('Metal is required')
    mlx_engine.configure()
    return mx


def files(directory, identifiers=IDS[0], *, fractional=True, permutation=None,
          zero=False, metadata=True):
    directory.mkdir(parents=True, exist_ok=True)
    comp, con = directory / "neurons.csv", directory / "edges.parquet"
    sources = np.array([2, 0, 1, 0, 3, 0, 2, 0], np.int32)
    destinations = np.array([1, 1, 2, 1, 1, 3, 1, 2], np.int64)
    coefficients = np.array([-15, 280, 380, -4, 50, 150, -8, 0], np.float64)
    if fractional:
        coefficients[:7] += np.array([-.125, .25, .125, -.5, .25, .5, -.25])
    if zero:
        coefficients[:] = 0
    # Integral coefficients retain an integer Parquet schema as upstream does.
    if not fractional:
        coefficients = coefficients.astype(np.int64)
    if permutation is not None:
        inverse = np.argsort(permutation)
        identifiers = np.asarray(identifiers)[permutation]
        sources, destinations = inverse[sources], inverse[destinations]
    neurons = pd.DataFrame(index=identifiers)
    edges = pd.DataFrame(dict(zip(COLUMNS, [sources, destinations, coefficients])))
    if metadata:
        neurons["ignored description"] = ["a", "b", "c", "d"]
        # Plausible alternative names must not override the consumed coefficient.
        edges["weight"] = 999999.
        edges["Connectivity"] = -999999
        edges["signed_count"] = "unused"
    neurons.to_csv(comp)
    edges.to_parquet(con)
    return comp, con


def inputs(connectome, identifiers, silenced=(), *, steps=97, trials=2, empty=False):
    spec = mlx_engine.experiment_spec({
        "key": "alternate-dataset", "neu_exc": [] if empty else [identifiers[0]] * 2,
        "neu_exc2": [] if empty else [identifiers[3]], "neu_slnc": silenced,
        "stim_rate": 10000., "stim_rate2": 0.,
    })
    singletons, rows = mlx_engine.stimuli(connectome, spec, steps, trials, 82)
    return mlx_engine.combine_trials(singletons), rows


def cpu(original, paths, stimulus, silenced, *, identifiers=IDS[0], silence_ids=()):
    b, eon = original
    parameters = dict(eon.default_params)
    parameters.update(r_poi=10000*b.Hz, r_poi2=0*b.Hz)
    lookup = {identifier: row for row, identifier in
              enumerate(pd.read_csv(paths[0], index_col=0).index)}
    targets = () if not stimulus.targets else tuple(lookup[identifier] for identifier in
                                                   [identifiers[0], identifiers[0], identifiers[3]])
    silenced_rows = tuple(lookup[identifier] for identifier in silence_ids)
    assert stimulus.targets == targets
    assert silenced == silenced_rows
    expected, states = [], []
    for trial in stimulus.trial_indices:
        neurons, synapses, monitor = eon.create_model(*paths, parameters)
        # Channels use deterministic Poisson probabilities 1 and 0; this is
        # actual original PoissonInput, with no shared native input/queue code.
        if stimulus.targets:
            pois, neurons = eon.poi(neurons, list(targets[:2]), list(targets[2:]), parameters)
        else:
            pois = []
        eon.silence(silenced_rows, synapses)
        network = b.Network(neurons, synapses, monitor, *pois)
        network.run(stimulus.events.shape[1] * .1*b.ms, namespace={})
        steps = np.rint(np.asarray(monitor.t / (.1*b.ms))).astype(np.int64)
        expected.extend(zip(np.full(steps.size, trial, np.int64), steps,
                            np.asarray(monitor.i[:], dtype=np.int64)))
        states.append({name: np.asarray(getattr(neurons, name + "_")[:]).copy()
                       for name in ("v", "g", "rfc", "lastspike", "not_refractory")})
    return expected, {name: np.stack([state[name] for state in states]) for name in states[0]}


def assert_spikes(actual, expected):
    assert list(zip(actual.spikes.trials, actual.spikes.steps, actual.spikes.neurons)) == expected
    assert "Apple" in actual.device_name


def assert_export(path, connectome, expected, *, exp_name="alternate-dataset"):
    table = pq.read_table(path)
    assert table.schema == pa.schema([
        ("t", pa.float64()), ("time_ms", pa.float64()), ("trial", pa.int64()),
        ("neuron_index", pa.int64()), ("flywire_id", pa.array(connectome.neuron_ids).type),
        ("exp_name", pa.string()),
    ])
    assert list(zip(table["trial"].to_pylist(), table["neuron_index"].to_pylist())) == [
        (trial, row) for trial, step, row in expected]
    assert table["flywire_id"].to_pylist() == [connectome.neuron_ids[row] for _, _, row in expected]
    times = np.array([step for _, step, _ in expected], np.float64) * .0001
    np.testing.assert_array_equal(table["t"].to_numpy(), times)
    np.testing.assert_array_equal(table["time_ms"].to_numpy(), times * 1000)
    assert table["exp_name"].to_pylist() == [exp_name] * len(expected)
    return table


@pytest.mark.metal
@pytest.mark.parametrize("identifiers", IDS, ids=["int64", "string", "uint64"])
@pytest.mark.parametrize("fractional", [False, True], ids=["integral", "fractional"])
@pytest.mark.parametrize("silence_row", [None, 0, 1], ids=["active", "silence-source", "silence-middle"])
def test_files_through_public_adapter_cpu_and_both_exports(
    tmp_path, original, metal, identifiers, fractional, silence_row,
):
    paths = files(tmp_path, identifiers, fractional=fractional)
    connectome = mlx_engine.load_inputs(*paths)
    external_silence = () if silence_row is None else (identifiers[silence_row],)
    stimulus, silenced = inputs(connectome, identifiers, external_silence)
    expected, _ = cpu(original, paths, stimulus, silenced, identifiers=identifiers,
                      silence_ids=external_silence)
    actual = mlx_engine.execute(connectome, stimulus, silenced)
    assert_spikes(actual, expected)
    assert expected
    fired = {row for _, _, row in expected}
    if silence_row == 0:
        assert fired == {0}
    elif silence_row == 1:
        assert 1 in fired and 2 not in fired  # incoming survives; outgoing is zero
    else:
        assert fired == {0, 1, 2, 3}
    retained = tmp_path / "retained.parquet"
    pq.write_table(run_mlx.spike_table(connectome, [actual], "alternate-dataset"), retained)
    whole = assert_export(retained, connectome, expected)
    with SpoolSpikes(connectome, stimulus.trial_indices, "alternate-dataset",
                     directory=tmp_path, batch_rows=3) as spool:
        streamed = mlx_engine.execute(connectome, stimulus, silenced, spike_sink=spool)
        assert streamed.spikes.steps.size == 0
        assert spool.n_spikes == len(expected)
        assert spool.n_active_neurons == len(fired)
        path = tmp_path / "streamed.parquet"
        spool.finish(path)
    assert assert_export(path, connectome, expected).equals(whole)
    silent_recording = mlx_engine.execute(connectome, stimulus, silenced, record=False)
    assert silent_recording.spikes.steps.size == 0
    assert silent_recording.input_sha256 == actual.input_sha256 == streamed.input_sha256


@pytest.mark.metal
@pytest.mark.parametrize("fractional", [False, True])
def test_row_remapping_and_mutated_file_content_cannot_reuse_stale_layouts(
    tmp_path, original, metal, fractional,
):
    from mlx_backend import configured

    identifiers = IDS[1]
    paths = files(tmp_path, identifiers, fractional=fractional)
    first = load_connectome(*paths)
    stimulus, silenced = inputs(first, identifiers)
    prepare = lambda c, s: configured.prepare(c, s, resolve_model())
    layout = prepare(first, stimulus)
    assert prepare(load_connectome(*paths), stimulus) is layout
    baseline = mlx_engine.execute(first, stimulus, silenced)

    # A physically equivalent reordering remaps both endpoints and selections.
    files(tmp_path, identifiers, fractional=fractional, permutation=[2, 0, 3, 1])
    permuted = load_connectome(*paths)
    remapped, rows = inputs(permuted, identifiers)
    assert permuted.pin != first.pin
    assert prepare(permuted, remapped) is not layout
    expected, _ = cpu(original, paths, remapped, rows, identifiers=identifiers)
    actual = mlx_engine.execute(permuted, remapped, rows)
    assert_spikes(actual, expected)
    assert sorted((int(t), int(s), first.neuron_ids[n]) for t, s, n in
                  zip(baseline.spikes.trials, baseline.spikes.steps, baseline.spikes.neurons)) == sorted(
        (int(t), int(s), permuted.neuron_ids[n]) for t, s, n in expected)

    # Mutate the same paths to a different physical graph, then change only
    # harmless metadata. Both identities refresh, but only the former changes spikes.
    files(tmp_path, identifiers, fractional=fractional, zero=True)
    changed = load_connectome(*paths)
    changed_stimulus, rows = inputs(changed, identifiers)
    expected, _ = cpu(original, paths, changed_stimulus, rows, identifiers=identifiers)
    changed_run = mlx_engine.execute(changed, changed_stimulus, rows)
    assert_spikes(changed_run, expected)
    assert set(changed_run.spikes.neurons) == {0}
    assert changed.pin != first.pin
    assert prepare(changed, changed_stimulus) is not layout
    assert len(expected) < baseline.spikes.steps.size
    files(tmp_path, identifiers, fractional=fractional, zero=True, metadata=False)
    metadata_only = load_connectome(*paths)
    assert metadata_only.pin != changed.pin
    np.testing.assert_array_equal(metadata_only.counts, changed.counts)
    np.testing.assert_array_equal(metadata_only.weights_mv, changed.weights_mv)
    assert_spikes(mlx_engine.execute(metadata_only, changed_stimulus, rows), expected)
    with pytest.raises(ValueError, match="hash"):
        load_connectome(*paths, first.pin)


@pytest.mark.metal
@pytest.mark.parametrize("empty_inputs", [False, True])
def test_zero_coefficients_and_empty_selection_keep_typed_empty_output(
    tmp_path, original, metal, empty_inputs,
):
    paths = files(tmp_path, IDS[2], zero=True)
    connectome = load_connectome(*paths)
    stimulus, rows = inputs(connectome, IDS[2], steps=31, trials=1, empty=empty_inputs)
    expected, _ = cpu(original, paths, stimulus, rows, identifiers=IDS[2])
    assert_spikes(mlx_engine.execute(connectome, stimulus, rows), expected)
    with SpoolSpikes(connectome, stimulus.trial_indices, "alternate-dataset", directory=tmp_path) as spool:
        mlx_engine.execute(connectome, stimulus, rows, spike_sink=spool)
        output = tmp_path / "zero.parquet"
        spool.finish(output)
    assert_export(output, connectome, expected)
    if empty_inputs:
        assert expected == []


@pytest.mark.metal
@pytest.mark.parametrize("coefficients", [
    [np.nan, 280.25, 380.125, -4.5, 50.25, 150.5, -8.25, 0.],
    [-15.125, np.inf, 380.125, -4.5, 50.25, 150.5, -8.25, 0.],
    [-15.125, -np.inf, 380.125, -4.5, 50.25, 150.5, -8.25, 0.],
    [-15.125, np.inf, 380.125, -np.inf, 50.25, 150.5, -8.25, 0.],
], ids=["nullable-NaN", "positive-infinity", "negative-infinity", "opposite-infinities"])
def test_upstream_nonfinite_coefficients_propagate_on_metal(
    tmp_path, original, metal, coefficients,
):
    from mlx_backend import configured

    paths = files(tmp_path)
    edges = pd.read_parquet(paths[1])
    edges[COLUMNS[2]] = coefficients
    edges.to_parquet(paths[1])
    connectome = load_connectome(*paths)
    np.testing.assert_array_equal(connectome.counts, coefficients)
    stimulus, rows = inputs(connectome, IDS[0], steps=73, trials=1)
    expected, states = cpu(original, paths, stimulus, rows)
    assert_spikes(mlx_engine.execute(connectome, stimulus, rows), expected)
    prepared = configured.prepare(connectome, stimulus, resolve_model())
    actual, state = configured.run_prepared(prepared, stimulus, return_state=True, chunk_steps=23)
    assert_spikes(actual, expected)
    host = configured.host_state(state, prepared)
    def assert_state_bits(actual, expected, name):
        assert actual.dtype == expected.dtype, name
        if expected.dtype.kind == 'f':
            assert expected.dtype.itemsize in (4, 8), name
            unsigned = np.dtype(f'u{expected.dtype.itemsize}')
            actual, expected = actual.view(unsigned), expected.view(unsigned)
        np.testing.assert_array_equal(actual, expected, err_msg=name)

    for name, values in states.items():
        assert_state_bits(host[name], values, name)
    _, unrecorded = configured.run_prepared(prepared, stimulus, record=False, return_state=True,
                                           chunk_steps=17)
    for name, values in configured.host_state(unrecorded, prepared).items():
        assert_state_bits(values, host[name], f"record=False {name}")


@pytest.mark.metal
@pytest.mark.parametrize("coefficients", [[1e16, 1., -1e16], [1e16, -1e16, 1.]])
def test_repeated_file_edges_keep_binary64_addition_order(tmp_path, original, metal, coefficients):
    from mlx_backend import configured

    paths = files(tmp_path)
    pd.DataFrame(dict(zip(COLUMNS, [[0, 0, 0], [1, 1, 1], coefficients]))).to_parquet(paths[1])
    connectome = load_connectome(*paths)
    stimulus, rows = inputs(connectome, IDS[0], steps=20, trials=1)
    expected, states = cpu(original, paths, stimulus, rows)
    assert states["g"][0, 1] != 0
    assert_spikes(mlx_engine.execute(connectome, stimulus, rows), expected)
    prepared = configured.prepare(connectome, stimulus, resolve_model())
    _, state = configured.run_prepared(prepared, stimulus, return_state=True, chunk_steps=7)
    host = configured.host_state(state, prepared)
    for name, values in states.items():
        np.testing.assert_array_equal(host[name], values, err_msg=name)


@pytest.mark.metal
@pytest.mark.parametrize("coefficient", [2**24 + 1, 2**31 + 1])
def test_large_integral_file_weights_use_native_execution(tmp_path, original, metal, coefficient):
    paths = files(tmp_path, fractional=False)
    edges = pd.read_parquet(paths[1])
    edges.loc[1, COLUMNS[2]] = coefficient
    edges.to_parquet(paths[1])
    connectome = load_connectome(*paths)
    stimulus, rows = inputs(connectome, IDS[0], steps=73, trials=1)
    expected, _ = cpu(original, paths, stimulus, rows)
    actual = mlx_engine.execute(connectome, stimulus, rows)
    assert_spikes(actual, expected)


@pytest.mark.metal
def test_fractional_file_state_is_exact_across_chunks_and_recording_modes(tmp_path, original, metal):
    from mlx_backend import configured

    paths = files(tmp_path)
    connectome = load_connectome(*paths)
    stimulus, rows = inputs(connectome, IDS[0], (IDS[0][1],), steps=101)
    expected, states = cpu(original, paths, stimulus, rows, silence_ids=(IDS[0][1],))
    prepared = configured.prepare(connectome, stimulus, resolve_model(), rows)
    for record, chunk in [(True, 23), (False, 17)]:
        actual, state = configured.run_prepared(prepared, stimulus, record=record,
            return_state=True, chunk_steps=chunk, block_windows=2)
        if record:
            assert_spikes(actual, expected)
        else:
            assert actual.spikes.steps.size == 0
        for name, values in configured.host_state(state, prepared).items():
            np.testing.assert_array_equal(values, states[name], err_msg=name)


@pytest.mark.metal
@pytest.mark.parametrize("strategy,stream_output,identifiers", [
    ("batch", False, IDS[2]), ("sequential", True, IDS[1]),
])
def test_real_benchmark_adapter_exports_alternate_files_and_preserves_outputs(
    tmp_path, original, metal, monkeypatch, strategy, stream_output, identifiers,
):
    paths = files(tmp_path / "input", identifiers)
    connectome = load_connectome(*paths)
    stimulus, rows = inputs(connectome, identifiers)
    expected, _ = cpu(original, paths, stimulus, rows, identifiers=identifiers)
    monkeypatch.setattr(run_mlx, "path_comp", paths[0])
    monkeypatch.setattr(run_mlx, "path_con", paths[1])
    output = tmp_path / "output" / "benchmark.parquet"
    monkeypatch.setattr(run_mlx, "get_spike_output_path", lambda *args: output)
    monkeypatch.delenv("FLY_BRAIN_DISABLE_SPIKE_IO", raising=False)
    experiment = {"name": "Alternate dataset", "key": "alternate-dataset",
        "neu_exc": [identifiers[0]] * 2, "neu_exc2": [identifiers[3]],
        "stim_rate": 10000., "stim_rate2": 0.}

    class Logger:
        def log(self, message):
            pass

    result = run_mlx.run_single_benchmark(.0097, 2, experiment, Logger(),
        strategy=strategy, stream_output=stream_output)
    assert result["status"] == "success"
    assert result["n_spikes"] == len(expected)
    assert_export(output, connectome, expected, exp_name="mlx_t0.0097s_n2")
    sidecar = output.with_suffix(".mlx.json")
    metadata = json.loads(sidecar.read_text())
    assert metadata["input_sha256"] == {
        "completeness": connectome.pin.completeness_sha256,
        "connectivity": connectome.pin.connectivity_sha256,
    }
    assert metadata["activated_ids"] == [identifiers[0], identifiers[0], identifiers[3]]
    assert metadata["output_mode"] == ("bounded-spool" if stream_output else "retained")
    assert "Apple" in metadata["device"]
    original_bytes = (output.read_bytes(), sidecar.read_bytes())
    with pytest.raises(FileExistsError, match="already exists"):
        run_mlx.run_single_benchmark(.0097, 2, experiment, Logger(), strategy=strategy)
    monkeypatch.setenv("FLY_BRAIN_DISABLE_SPIKE_IO", "1")
    timing_only = run_mlx.run_single_benchmark(.0097, 2, experiment, Logger(), strategy=strategy)
    assert timing_only["status"] == "success" and timing_only["spike_path"] == ""
    assert original_bytes == (output.read_bytes(), sidecar.read_bytes())


@pytest.mark.parametrize("empty", ["neurons", "edges"])
def test_empty_files_are_rejected_by_the_original_model_and_loader(tmp_path, original, empty):
    _, eon = original
    paths = files(tmp_path)
    if empty == "neurons":
        pd.DataFrame(index=np.array([], np.int64)).to_csv(paths[0])
    else:
        pd.read_parquet(paths[1]).iloc[:0].to_parquet(paths[1])
    with pytest.raises(ValueError):
        eon.create_model(*paths, eon.default_params)
    with pytest.raises(ValueError):
        load_connectome(*paths)


@pytest.mark.parametrize("column", COLUMNS)
def test_only_the_original_consumed_columns_are_required(tmp_path, original, column):
    _, eon = original
    paths = files(tmp_path, metadata=False)
    accepted = load_connectome(*paths)
    assert accepted.neuron_ids.size == 4
    eon.create_model(*paths, eon.default_params)
    pd.read_parquet(paths[1]).drop(columns=[column]).to_parquet(paths[1])
    with pytest.raises(KeyError):
        eon.create_model(*paths, eon.default_params)
    with pytest.raises(ValueError, match="columns"):
        load_connectome(*paths)


@pytest.mark.parametrize("selection", ["neu_exc", "neu_exc2", "neu_slnc"])
def test_unknown_external_ids_fail_in_the_public_adapter(tmp_path, selection):
    connectome = load_connectome(*files(tmp_path, IDS[1]))
    with pytest.raises(ValueError, match="absent"):
        neuron_indices(connectome, ("missing",))
    spec = mlx_engine.experiment_spec({selection: ["missing"]})
    with pytest.raises(ValueError, match="absent"):
        mlx_engine.stimuli(connectome, spec, 10, 1, 82)


@pytest.mark.metal
@pytest.mark.parametrize('storage', ['immutable-bytes', 'owned-mutable', 'readonly-mutable-alias'])
@pytest.mark.parametrize('fractional', [False, True])
def test_input_hash_cache_preserves_actual_bytes_shape_and_cpu_results(
    tmp_path, original, metal, monkeypatch, storage, fractional,
):
    from configured_reference import reference
    from mlx_backend import configured, window

    connectome = load_connectome(*files(tmp_path, fractional=fractional))
    template, silenced = inputs(connectome, IDS[0], steps=73)
    model = resolve_model()
    owner = np.zeros(template.events.shape, np.uint8)
    owner[0, :65:4, 0] = 1
    owner[1, 2:65:5, 0] = 1
    if storage == 'immutable-bytes':
        events = np.frombuffer(owner.tobytes(), np.uint8).reshape(owner.shape)
        with pytest.raises(ValueError):
            events.setflags(write=True)
    elif storage == 'readonly-mutable-alias':
        events = owner.view()
        events.setflags(write=False)
    else:
        events = owner
    stimulus = replace(template, events=events, sha256=None)

    def check(stimulus):
        expected = reference(connectome, stimulus, model, silenced)
        result = mlx_engine.execute(connectome, stimulus, silenced)
        assert_spikes(result, list(zip(expected.spikes.trials, expected.spikes.steps,
                                      expected.spikes.neurons)))
        hashes = tuple(sha256(row.tobytes()).hexdigest() for row in stimulus.events)
        assert result.input_sha256 == hashes
        return hashes

    first_hashes = check(stimulus)
    # Exact replay reuses fingerprints only after pinning the same input bytes.
    with monkeypatch.context() as patch:
        def no_rehash(*args, **kwargs):
            raise AssertionError('Unchanged consumed bytes were rehashed')
        patch.setattr(configured, 'sha256', no_rehash)
        patch.setattr(window, 'sha256', no_rehash)
        prepared = configured.prepare(connectome, stimulus, model, silenced)
        result, state = configured.run_prepared(prepared, stimulus, chunk_steps=19,
                                                return_state=True)
        assert result.input_sha256 == first_hashes
        unrecorded = mlx_engine.execute(connectome, stimulus, silenced, record=False)
        assert unrecorded.spikes.steps.size == 0
        assert unrecorded.input_sha256 == first_hashes
    expected = reference(connectome, stimulus, model, silenced)
    for name, value in configured.host_state(state, prepared).items():
        np.testing.assert_array_equal(value, expected.state[name], err_msg=name)
    if storage == 'immutable-bytes':
        reshaped = replace(stimulus, events=events.reshape(1, 146, 3), trial_indices=(0,))
        assert len(check(reshaped)) == 1
        changed = owner.copy()
        changed[:, 0, 0] ^= 1
        stimulus = replace(stimulus, events=np.frombuffer(changed.tobytes(), np.uint8).reshape(changed.shape))
    else:
        # Readonly flags on the borrowed view do not prevent the owner changing.
        owner[:, 0, 0] ^= 1
    assert check(stimulus) != first_hashes


@pytest.mark.metal
@pytest.mark.parametrize('fractional', [False, True])
def test_input_hash_cache_pins_consumed_bytes_before_mutation(
    tmp_path, original, metal, monkeypatch, fractional,
):
    from configured_reference import reference
    from mlx_backend import configured, window

    connectome = load_connectome(*files(tmp_path, fractional=fractional))
    template, silenced = inputs(connectome, IDS[0], steps=73)
    owner = np.zeros(template.events.shape, np.uint8)
    owner[:, :65:3, 0] = 1
    stimulus = replace(template, events=owner, sha256=None)
    pinned = replace(stimulus, events=owner.copy())
    expected = reference(connectome, pinned, resolve_model(), silenced)
    engine = configured
    read_chunk = engine._event_chunk
    changed = False

    def mutate_then_read(raw, *args):
        nonlocal changed
        if not changed:
            owner[:, :, 0] ^= 1
            changed = True
        return read_chunk(raw, *args)

    with monkeypatch.context() as patch:
        patch.setattr(engine, '_event_chunk', mutate_then_read)
        result = mlx_engine.execute(connectome, stimulus, silenced)
    assert changed
    assert_spikes(result, list(zip(expected.spikes.trials, expected.spikes.steps,
                                  expected.spikes.neurons)))
    before = tuple(sha256(row.tobytes()).hexdigest() for row in pinned.events)
    after = tuple(sha256(row.tobytes()).hexdigest() for row in owner)
    assert before != after and result.input_sha256 == before
    changed_reference = reference(connectome, stimulus, resolve_model(), silenced)
    changed_result = mlx_engine.execute(connectome, stimulus, silenced)
    assert_spikes(changed_result, list(zip(changed_reference.spikes.trials,
        changed_reference.spikes.steps, changed_reference.spikes.neurons)))
    assert changed_result.input_sha256 == after


@pytest.mark.parametrize('dtype', [np.bool_, np.int16, np.int32, np.int64,
                                  np.uint64, np.float16, np.float32, np.float64])
@pytest.mark.parametrize('scalar32', [False, True])
def test_column_precision_matches_original_brian_weight_assignment(
    tmp_path, original, dtype, scalar32,
):
    from mlx_backend.inputs import coefficient_weights_si

    b, eon = original
    paths = files(tmp_path)
    edges = pd.read_parquet(paths[1])
    if np.dtype(dtype).kind == 'b':
        values = np.array([True, False, True, True, False, True, False, True])
    elif np.dtype(dtype).kind == 'u':
        values = np.array([2**64-1, 2**63+1, 1, 4, 0, 17, 8, 0], dtype)
    elif np.dtype(dtype) == np.dtype(np.int64):
        values = np.array([-(2**63), 2**63-1, 1, 4, -7, 17, -8, 0], dtype)
    elif np.dtype(dtype).kind == 'i':
        values = np.array([-15, 280, 380, -4, 50, 150, -8, 0], dtype)
    else:
        values = np.array([-15.125, 280.25, 380.125, -4.5, 50.25, 150.5, -8.25, -0.], dtype)
    edges[COLUMNS[2]] = values
    edges.to_parquet(paths[1])
    scalar = (b.Quantity(np.float32(.000275), dim=b.volt.dim)
              if scalar32 else eon.default_params['w_syn'])
    assert np.asarray(scalar).dtype == np.dtype(np.float32 if scalar32 else np.float64)
    params = dict(eon.default_params, w_syn=scalar)
    _, synapses, _ = eon.create_model(*paths, params)
    expected = np.asarray(synapses.w_[:])
    graph = load_connectome(*paths)
    actual = coefficient_weights_si(graph, np.asarray(scalar)[()])
    assert actual.dtype == expected.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert graph.coefficient_dtype == np.dtype(dtype).str
    narrow_float = np.dtype(dtype).kind == 'f' and np.dtype(dtype).itemsize < 8
    if narrow_float:
        assert graph.counts.dtype == np.dtype(dtype)
    if not scalar32 and narrow_float:
        np.testing.assert_array_equal(graph.weights_mv.view(np.uint64),
                                      (expected*1000).view(np.uint64))


@pytest.mark.metal
@pytest.mark.parametrize('integral', [False, True])
def test_float32_file_weights_keep_original_rounding_through_native_exports(
    tmp_path, original, metal, integral,
):
    _check_narrow_float_file_weights(tmp_path, original, integral, np.float32)


@pytest.mark.metal
@pytest.mark.parametrize('integral', [False, True])
def test_float16_file_weights_keep_original_rounding_through_native_exports(
    tmp_path, original, metal, integral,
):
    _check_narrow_float_file_weights(tmp_path, original, integral, np.float16)


def _check_narrow_float_file_weights(tmp_path, original, integral, storage):
    from mlx_backend import configured

    paths = files(tmp_path, fractional=not integral)
    edges = pd.read_parquet(paths[1])
    edges[COLUMNS[2]] = edges[COLUMNS[2]].astype(storage)
    edges.to_parquet(paths[1])
    graph = load_connectome(*paths)
    assert graph.counts.dtype == np.dtype(storage)
    stimulus, silenced = inputs(graph, IDS[0], steps=73, trials=1)
    expected, states = cpu(original, paths, stimulus, silenced)
    result = mlx_engine.execute(graph, stimulus, silenced)
    assert_spikes(result, expected)
    label = np.dtype(storage).name+'-source'
    output = tmp_path/(np.dtype(storage).name+'-output.parquet')
    pq.write_table(run_mlx.spike_table(graph, [result], label), output)
    assert_export(output, graph, expected, exp_name=label)
    prepared = configured.prepare(graph, stimulus, resolve_model(), silenced)
    _, state = configured.run_prepared(prepared, stimulus, return_state=True, chunk_steps=23)
    for name, actual in configured.host_state(state, prepared).items():
        wanted = states[name]
        assert actual.dtype == wanted.dtype
        if actual.dtype.kind == 'f':
            np.testing.assert_array_equal(actual.view(np.uint64), wanted.view(np.uint64), err_msg=name)
        else:
            np.testing.assert_array_equal(actual, wanted, err_msg=name)
    _, quiet = configured.run_prepared(prepared, stimulus, record=False, return_state=True, chunk_steps=17)
    for name, actual in configured.host_state(quiet, prepared).items():
        wanted = states[name]
        assert actual.dtype == wanted.dtype
        if actual.dtype.kind == 'f':
            np.testing.assert_array_equal(actual.view(np.uint64), wanted.view(np.uint64), err_msg=name)
        else:
            np.testing.assert_array_equal(actual, wanted, err_msg=name)


@pytest.mark.metal
@pytest.mark.parametrize('shadow_fires', [False, True], ids=['accepted-quiet-shadow', 'unmapped-active-shadow'])
def test_duplicate_labels_match_actual_original_run_exp_and_both_native_exports(
    tmp_path, original, metal, monkeypatch, shadow_fires,
):
    b, eon = original
    source = tmp_path/'input'
    source.mkdir()
    paths = source/'neurons.csv', source/'edges.parquet'
    pd.DataFrame(index=[42, 42, 9]).to_csv(paths[0])
    sources = [1, 1] if shadow_fires else [1]
    destinations = [2, 0] if shadow_fires else [2]
    pd.DataFrame({COLUMNS[0]: np.array(sources, np.int32),
                  COLUMNS[1]: np.array(destinations, np.int64),
                  COLUMNS[2]: np.full(len(sources), 400.25)}).to_parquet(paths[1])
    cpu_out = tmp_path/'cpu'
    cpu_out.mkdir()
    name = 'mlx_t0.0073s_n1'
    params = dict(eon.default_params, n_run=1, t_run=73*.1*b.ms,
                  r_poi=10000*b.Hz, r_poi2=0*b.Hz)
    if shadow_fires:
        with pytest.raises(KeyError) as failure:
            eon.run_exp(name, [42], cpu_out, *paths, params=params, n_proc=1)
        assert failure.value.args == (0,)
        assert not (cpu_out/(name+'.parquet')).exists()
        expected = None
    else:
        eon.run_exp(name, [42], cpu_out, *paths, params=params, n_proc=1)
        expected = pd.read_parquet(cpu_out/(name+'.parquet'))
        assert set(expected.flywire_id) == {42, 9}
    graph = mlx_engine.load_inputs(*paths)
    assert graph.neuron_ids.size == 3
    assert neuron_indices(graph, (42,)) == (1,)
    monkeypatch.setattr(run_mlx, 'path_comp', paths[0])
    monkeypatch.setattr(run_mlx, 'path_con', paths[1])
    monkeypatch.delenv('FLY_BRAIN_DISABLE_SPIKE_IO', raising=False)
    experiment = dict(name='Duplicate labels', key='duplicate-labels',
                      neu_exc=[42], stim_rate=10000.)

    class Logger:
        def log(self, message):
            pass

    for streamed in (False, True):
        output = tmp_path/('streamed.parquet' if streamed else 'retained.parquet')
        monkeypatch.setattr(run_mlx, 'get_spike_output_path', lambda *args: output)
        if shadow_fires:
            with pytest.raises(KeyError) as failure:
                run_mlx.run_single_benchmark(.0073, 1, experiment, Logger(),
                                              stream_output=streamed)
            assert failure.value.args == (0,)
            assert not output.exists() and not output.with_suffix('.mlx.json').exists()
        else:
            result = run_mlx.run_single_benchmark(.0073, 1, experiment, Logger(),
                                                  stream_output=streamed)
            assert result['status'] == 'success'
            table = pq.read_table(output)
            assert set(table['neuron_index'].to_pylist()) == {1, 2}
            observed = table.to_pandas()[expected.columns]
            keys = ['trial', 't', 'flywire_id']
            pd.testing.assert_frame_equal(expected.sort_values(keys).reset_index(drop=True),
                                          observed.sort_values(keys).reset_index(drop=True))


@pytest.mark.metal
def test_preparation_reuse_preserves_graph_targets_and_model_identity(
    tmp_path, original, metal, monkeypatch,
):
    from collections import OrderedDict
    from configured_reference import reference
    from mlx_backend import compiler, configured

    monkeypatch.setattr(configured, '_PROGRAMS', OrderedDict())
    monkeypatch.setattr(configured, '_LAYOUTS', OrderedDict())
    compile_original = compiler.compile_model
    compilations = []

    def count_compile(model):
        compilations.append(model.signature)
        return compile_original(model)

    monkeypatch.setattr(compiler, 'compile_model', count_compile)
    paths = files(tmp_path)
    connectome = load_connectome(*paths)
    stimulus, silenced = inputs(connectome, IDS[0], steps=73)
    model = resolve_model()

    def check(graph, stimulus, model, record=True):
        expected = reference(graph, stimulus, model, silenced)
        prepared = configured.prepare(graph, stimulus, model, silenced)
        result, state = configured.run_prepared(prepared, stimulus, record=record,
                                                return_state=True, chunk_steps=19)
        if record:
            assert_spikes(result, list(zip(expected.spikes.trials,
                expected.spikes.steps, expected.spikes.neurons)))
        else:
            assert result.spikes.steps.size == 0
        for name, values in configured.host_state(state, prepared).items():
            np.testing.assert_array_equal(values, expected.state[name], err_msg=name)
        return prepared

    first = check(connectome, stimulus, model)
    assert configured.prepare(connectome, stimulus, resolve_model(), silenced) is first
    changed_targets = replace(stimulus, targets=(1, 1, 3))
    assert check(connectome, changed_targets, model, record=False) is not first
    edges = pd.read_parquet(paths[1])
    edges.loc[1, COLUMNS[2]] = 20.25
    edges.to_parquet(paths[1])
    changed_graph = load_connectome(*paths)
    assert changed_graph.pin != connectome.pin
    assert check(changed_graph, stimulus, model) is not first
    analysed = resolve_model({})
    assert analysed.signature == model.signature and analysed.analysis is not model.analysis
    assert check(changed_graph, stimulus, analysed).program is not first.program
    changed_model = resolve_model({'tau': 7 * original[0].ms})
    assert check(changed_graph, stimulus, changed_model) is not first
    assert compilations == [model.signature, analysed.signature, changed_model.signature]
    assert len(configured._PROGRAMS) <= 2 and len(configured._LAYOUTS) <= 2


@pytest.mark.metal
def test_cached_integer_bound_keeps_silencing_and_file_identity(
    tmp_path, original, metal, monkeypatch,
):
    from collections import OrderedDict
    from mlx_backend import configured

    monkeypatch.setattr(configured, '_LAYOUTS', OrderedDict())
    paths = files(tmp_path, fractional=False)
    edges = pd.read_parquet(paths[1])
    edges.loc[1, COLUMNS[2]] = 2**24 + 1
    edges.to_parquet(paths[1])
    connectome = load_connectome(*paths)
    stimulus, silenced = inputs(connectome, IDS[0], steps=73)
    expected, _ = cpu(original, paths, stimulus, silenced)
    assert_spikes(mlx_engine.execute(connectome, stimulus, silenced), expected)

    model = resolve_model()
    prepared = configured.prepare(connectome, stimulus, model, silenced)

    def no_rescan(*args, **kwargs):
        raise AssertionError('Verified unchanged integer bounds were rescanned')

    with monkeypatch.context() as patch:
        patch.setattr(configured.np, 'bincount', no_rescan)
        assert_spikes(mlx_engine.execute(connectome, stimulus, silenced), expected)
        assert configured.prepare(connectome, stimulus, model, silenced) is prepared

    # Silencing and changed file contents must create distinct live layouts.
    muted = (0,)
    expected, _ = cpu(original, paths, stimulus, muted, silence_ids=(IDS[0][0],))
    assert configured.prepare(connectome, stimulus, model, muted) is not prepared
    assert_spikes(mlx_engine.execute(connectome, stimulus, muted), expected)
    edges.loc[1, COLUMNS[2]] = 1
    edges.to_parquet(paths[1])
    changed = load_connectome(*paths)
    assert changed.pin != connectome.pin
    assert configured.prepare(changed, stimulus, model, silenced) is not prepared
    expected, _ = cpu(original, paths, stimulus, silenced)
    assert_spikes(mlx_engine.execute(changed, stimulus, silenced), expected)
    assert len(configured._LAYOUTS) <= 2


@pytest.mark.metal
def test_uncacheable_models_and_unverified_bounds_remain_fresh(
    tmp_path, original, metal, monkeypatch,
):
    from collections import OrderedDict
    from mlx_backend import configured

    monkeypatch.setattr(configured, '_PROGRAMS', OrderedDict())
    monkeypatch.setattr(configured, '_LAYOUTS', OrderedDict())
    connectome = load_connectome(*files(tmp_path, fractional=False))
    stimulus, silenced = inputs(connectome, IDS[0], steps=19)
    uncacheable = replace(resolve_model(), signature=None)
    first = configured.prepare(connectome, stimulus, uncacheable, silenced)
    second = configured.prepare(connectome, stimulus, uncacheable, silenced)
    assert first.program is not second.program and not configured._PROGRAMS

    mutable = replace(connectome, pin=None, counts=connectome.counts.copy(),
                      weights_mv=connectome.weights_mv.copy())
    mutable.counts[1] = 2**24 + 1
    mutable.weights_mv[:] = mutable.counts.astype(np.float64)*.275
    model = resolve_model()
    large = configured.prepare(mutable, stimulus, model, silenced)
    mutable.counts[1] = 1
    mutable.weights_mv[:] = mutable.counts.astype(np.float64)*.275
    changed = configured.prepare(mutable, stimulus, model, silenced)
    assert changed is not large
    assert not np.array_equal(np.asarray(large.weights), np.asarray(changed.weights))
    assert not configured._LAYOUTS


@pytest.mark.parametrize('fractional', [False, True])
def test_loaded_graph_storage_cannot_invalidate_its_verified_bytes(tmp_path, fractional):
    graph = load_connectome(*files(tmp_path, fractional=fractional))
    for array in (graph.sources, graph.destinations, graph.counts, graph.weights_mv):
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.metal
def test_analysis_free_preparation_tracks_ambient_optimizer_and_flags(
    tmp_path, original, metal, monkeypatch,
):
    from collections import OrderedDict
    from mlx_backend import compiler, configured

    b = original[0]
    monkeypatch.setattr(configured, '_PROGRAMS', OrderedDict())
    monkeypatch.setattr(configured, '_LAYOUTS', OrderedDict())
    graph = load_connectome(*files(tmp_path))
    stimulus, silenced = inputs(graph, IDS[0], steps=19)
    model = resolve_model()
    assert model.analysis is None
    prepared = []
    for optimise, extra_flags in [(True, None), (False, None),
                                  (False, ['-O3', '-fno-associative-math'])]:
        with monkeypatch.context() as patch:
            patch.setitem(b.prefs, 'codegen.loop_invariant_optimisations', optimise)
            patch.setitem(b.prefs, 'codegen.cpp.extra_compile_args', extra_flags)
            # No CPU code is compiled in this source-preparation check. The
            # actual CPU/native physics checks use unchanged default flags.
            fresh = compiler.compile_model(model)
            candidate = configured.prepare(graph, stimulus, model, silenced)
            for name in ('scalar_code', 'state_code', 'threshold_code', 'reset_code',
                         'refractory_code', 'dynamics_code', 'scalar_values'):
                assert getattr(candidate.program, name) == getattr(fresh, name), name
            assert configured.prepare(graph, stimulus, model, silenced) is candidate
            prepared.append(candidate)
    assert len({id(p.program) for p in prepared}) == 3
    assert len({id(p) for p in prepared}) == 3
