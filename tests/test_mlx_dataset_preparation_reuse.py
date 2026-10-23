"""Preparation reuse must preserve fresh physical state and reject new inputs."""
from collections import OrderedDict
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from test_mlx_dataset_workflows import (
    COLUMNS, IDS, assert_spikes, cpu, files, inputs, metal, original,
)
from mlx_backend import configured
from mlx_backend.inputs import load_connectome
from mlx_backend.model import resolve_model


def fixture_files(path):
    paths = files(path)
    edges = pd.read_parquet(paths[1])
    edges[COLUMNS[2]] = edges[COLUMNS[2]].astype(np.float32)
    edges.to_parquet(paths[1])
    return paths, load_connectome(*paths)


def assert_state(actual, expected):
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        wanted = expected[name]
        assert value.dtype == wanted.dtype, name
        if value.dtype.kind == 'f':
            unsigned = np.dtype(f'u{value.dtype.itemsize}')
            value, wanted = value.view(unsigned), wanted.view(unsigned)
        np.testing.assert_array_equal(value, wanted, err_msg=name)


@pytest.mark.metal
def test_cached_layout_still_checks_actual_neuron_count_and_new_rows(
    tmp_path, original, metal, monkeypatch,
):
    monkeypatch.setattr(configured, '_LAYOUTS', OrderedDict())
    paths, graph = fixture_files(tmp_path)
    stimulus, silenced = inputs(graph, IDS[0], steps=19, trials=1)
    model = resolve_model()
    first = configured.prepare(graph, stimulus, model, silenced)
    assert configured.prepare(graph, stimulus, model, silenced) is first
    for targets in ((-1, 0, 3), (0, 0, 4), (0, 0, 2**40)):
        with pytest.raises(ValueError, match='neuron count or input target'):
            configured.prepare(graph, replace(stimulus, targets=targets), model)
    with pytest.raises(TypeError):
        configured.prepare(graph, replace(stimulus, targets=(0., 0, 3)), model)
    with pytest.raises(ValueError, match='neuron count or input target'):
        configured.prepare(graph, stimulus, model, (4,))
    # Even a caller retaining a pin while changing actual dimensions cannot
    # inherit the previous row-bound proof.
    for size in (0, 3):
        changed = replace(graph, neuron_ids=graph.neuron_ids[:size])
        with pytest.raises(ValueError, match='neuron count or input target'):
            configured.prepare(changed, stimulus, model)
    larger = replace(graph, neuron_ids=np.array([99, -7, 2, 17, 123]))
    rebuilt = configured.prepare(larger, stimulus, model)
    assert rebuilt is not first and rebuilt.neurons == 5
    class MutableIndex:
        row = 0

        def __index__(self):
            return self.row

    mutable = MutableIndex()
    borrowed = replace(stimulus, targets=(mutable, 0, 3))
    assert configured.prepare(graph, borrowed, model).targets == (0, 0, 3)
    mutable.row = 1
    assert configured.prepare(graph, borrowed, model).targets == (1, 0, 3)


@pytest.mark.metal
def test_reused_probe_keeps_fresh_states_and_validates_changed_event_bytes(
    tmp_path, original, metal, monkeypatch,
):
    monkeypatch.setattr(configured, '_WARMUPS', OrderedDict())
    paths, graph = fixture_files(tmp_path)
    stimulus, silenced = inputs(graph, IDS[0], steps=73, trials=1)
    model = resolve_model()
    prepared = configured.prepare(graph, stimulus, model, silenced)
    expected, states = cpu(original, paths, stimulus, silenced)
    for _ in range(2):
        result, state = configured.run_prepared(prepared, stimulus, return_state=True, chunk_steps=23)
        assert_spikes(result, expected)
        assert_state(configured.host_state(state, prepared), states)
    b, eon = original
    params = dict(eon.default_params, r_poi=0*b.Hz, r_poi2=0*b.Hz)
    neurons, synapses, monitor = eon.create_model(*paths, params)
    pois, neurons = eon.poi(neurons, [0, 0], [3], params)
    b.Network(neurons, synapses, monitor, *pois).run(73*.1*b.ms, namespace={})
    quiet_states = {name: np.asarray(getattr(neurons, name+'_')[:])[None].copy()
                    for name in states}
    quiet = replace(stimulus, events=np.zeros_like(stimulus.events), rates_hz=(0., 0., 0.))
    result, state = configured.run_prepared(prepared, quiet, return_state=True, chunk_steps=23)
    assert result.spikes.steps.size == len(monitor.i) == 0
    assert_state(configured.host_state(state, prepared), quiet_states)
    corrupt = quiet.events.copy()
    corrupt[0, 0, 0] = 2
    with pytest.raises(ValueError, match='canonical stimulus chunk'):
        configured.run_prepared(prepared, replace(quiet, events=corrupt), chunk_steps=23)


@pytest.mark.metal
def test_probe_identity_covers_trials_recording_chunks_and_kernel_variant(
    tmp_path, original, metal, monkeypatch,
):
    monkeypatch.setattr(configured, '_WARMUPS', OrderedDict())
    paths, graph = fixture_files(tmp_path)
    stimulus, silenced = inputs(graph, IDS[0], steps=73, trials=1)
    model = resolve_model()
    prepared = configured.prepare(graph, stimulus, model, silenced)
    for trials, record, chunk in ((1, True, 23), (2, True, 41), (2, False, 41), (1, False, 23)):
        current, _ = inputs(graph, IDS[0], steps=73, trials=trials)
        expected, states = cpu(original, paths, current, silenced)
        for candidate in (prepared, replace(prepared, compact_idle=False)):
            result, state = configured.run_prepared(candidate, current, record=record,
                                                    return_state=True, chunk_steps=chunk)
            if record:
                assert_spikes(result, expected)
            else:
                assert result.spikes.steps.size == 0
            assert_state(configured.host_state(state, candidate), states)
    assert len(configured._WARMUPS) <= 32


@pytest.mark.metal
def test_staged_inputs_keep_bytes_shape_clock_and_bounded_ownership(metal, monkeypatch):
    from mlx_backend.stimuli import event_input_snapshot
    from mlx_backend.window import _event_chunk

    monkeypatch.setattr(configured, '_STAGED_INPUTS', OrderedDict())
    monkeypatch.setattr(configured, '_MAX_STAGED_INPUT_BYTES', 2048)
    owner = np.resize(np.array([0, 1, 1], np.uint8), (1, 73, 3))
    original_bytes = owner.tobytes()
    for shape in ((1, 73, 3), (1, 219, 1)):
        raw, _, token = event_input_snapshot(np.frombuffer(original_bytes, np.uint8).reshape(shape))
        for begin in (0, 18, 36, 54):
            end = begin + 18
            chunk = _event_chunk(raw, begin, end, shape[0], shape[2])
            for dt_s in (.0001, np.nextafter(.0001, np.inf)):
                events, clock = configured._stage_inputs(chunk, token, begin, end, dt_s)
                repeated = configured._stage_inputs(chunk, token, begin, end, dt_s)
                assert repeated[0] is events and repeated[1] is clock
                np.testing.assert_array_equal(np.asarray(events), chunk.reshape(-1))
                expected = (np.arange(begin, end, dtype=np.float64)*dt_s).view(np.uint64)
                np.testing.assert_array_equal(np.asarray(clock), expected)
                assert len(configured._STAGED_INPUTS) <= 4
                assert sum(item[3] for item in configured._STAGED_INPUTS.values()) <= 2048
    # Mutable arrays are snapshotted again and cannot reuse stale device data.
    frozen, _, before = event_input_snapshot(owner)
    chunk = _event_chunk(frozen, 0, 18, 1, 3)
    first = configured._stage_inputs(chunk, before, 0, 18, .0001)
    owner[:, :18] ^= 1
    frozen, _, after = event_input_snapshot(owner)
    second = configured._stage_inputs(_event_chunk(frozen, 0, 18, 1, 3), after, 0, 18, .0001)
    assert first[0] is not second[0]
    np.testing.assert_array_equal(np.asarray(second[0]), owner[:, :18].reshape(-1))
    # A source without an immutable snapshot must stage independently.
    assert configured._stage_inputs(chunk, None, 0, 18, .0001)[0] is not first[0]


@pytest.mark.metal
@pytest.mark.parametrize('family', ['float32', 'large-int', 'float32-dense', 'float32-nonfinite'])
def test_all_targeted_rows_skip_compaction_and_match_original_cpu(
    tmp_path, original, metal, family,
):
    paths, graph = fixture_files(tmp_path)
    if family in ('float32-dense', 'float32-nonfinite'):
        edges = pd.read_parquet(paths[1])
        if family == 'float32-dense':
            edges = pd.concat([edges, edges.iloc[[1, 1, 1, 1]]], ignore_index=True)
        else:
            edges.loc[[0, 1, 3], COLUMNS[2]] = [np.nan, np.inf, -np.inf]
        edges.to_parquet(paths[1])
        graph = load_connectome(*paths)
    if family == 'large-int':
        edges = pd.read_parquet(paths[1])
        edges[COLUMNS[2]] = edges[COLUMNS[2]].astype(np.int64)
        edges.loc[1, COLUMNS[2]] = 2**24+1
        edges.to_parquet(paths[1])
        graph = load_connectome(*paths)
    import mlx_engine

    spec = mlx_engine.experiment_spec(dict(key='all-selected', neu_exc=[IDS[0][0]]*2,
        neu_exc2=IDS[0][1:], neu_slnc=[IDS[0][1]], stim_rate=10000., stim_rate2=0.))
    singletons, silenced = mlx_engine.stimuli(graph, spec, 97, 2, 82)
    stimulus = mlx_engine.combine_trials(singletons)
    b, eon = original
    params = dict(eon.default_params, r_poi=10000*b.Hz, r_poi2=0*b.Hz)
    expected, states = [], []
    for trial in stimulus.trial_indices:
        neurons, synapses, monitor = eon.create_model(*paths, params)
        pois, neurons = eon.poi(neurons, [0, 0], [1, 2, 3], params)
        eon.silence(silenced, synapses)
        b.Network(neurons, synapses, monitor, *pois).run(97*.1*b.ms, namespace={})
        steps = np.rint(np.asarray(monitor.t_[:])/.0001).astype(np.int64)
        expected.extend(zip(np.full(steps.size, trial), steps, np.asarray(monitor.i[:])))
        states.append({name: np.asarray(getattr(neurons, name+'_')[:]).copy()
                       for name in ('v', 'g', 'rfc', 'lastspike', 'not_refractory')})
    expected_state = {name: np.stack([state[name] for state in states]) for name in states[0]}
    prepared = configured.prepare(graph, stimulus, resolve_model(), silenced, component=True)
    assert not prepared.compact_idle
    assert prepared.scan_arrivals == (family != 'float32-dense')
    for candidate in (prepared, replace(prepared, compact_idle=True, components=None),
                      replace(prepared, scan_arrivals=not prepared.scan_arrivals, components=None)):
        for record in (True, False):
            actual, state = configured.run_prepared(candidate, stimulus, record=record,
                chunk_steps=23, return_state=True)
            if record:
                assert_spikes(actual, expected)
            else:
                assert actual.spikes.steps.size == 0
            assert_state(configured.host_state(state, candidate), expected_state)
    assert_spikes(mlx_engine.execute(graph, stimulus, silenced), expected)


@pytest.mark.metal
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('negative_zero', [False, True])
def test_fresh_packed_initial_state_matches_original_typed_bits(
    tmp_path, original, metal, monkeypatch, dtype, negative_zero,
):
    b, eon = original
    monkeypatch.setitem(b.prefs, 'core.default_float_dtype', dtype)
    paths, graph = fixture_files(tmp_path)
    stimulus, silenced = inputs(graph, IDS[0], steps=0, trials=2)
    params = dict(eon.default_params)
    if negative_zero:
        params['v_0'] = -0.*b.mV
    prepared = configured.prepare(graph, stimulus, resolve_model(params), silenced)
    neurons, synapses, monitor = eon.create_model(*paths, params)
    pois, neurons = eon.poi(neurons, [0, 0], [3], params)
    b.Network(neurons, synapses, monitor, *pois).run(0*b.ms, namespace={})
    wanted = {name: np.repeat(np.asarray(getattr(neurons, name+'_')[:])[None], 2, axis=0)
              for name in ('v', 'g', 'rfc', 'lastspike', 'not_refractory')}
    first = configured.initial_state(prepared, 2)
    second = configured.initial_state(prepared, 2)
    for state in (first, second):
        assert_state(configured.host_state(state, prepared), wanted)
        np.testing.assert_array_equal(np.asarray(state.release), np.zeros(8, np.int64))
    assert all(first.values[name] is not second.values[name] for name in first.values)


@pytest.mark.metal
def test_edge_free_window_growth_reuses_prepared_probe_without_reusing_state(tmp_path,original,metal):
    from configured_reference import reference
    paths,graph=fixture_files(tmp_path)
    graph=replace(graph,pin=None,sources=np.array([],np.int32),destinations=np.array([],np.int32),
                  counts=np.array([],np.float64),weights_mv=np.array([],np.float64))
    stimulus,silenced=inputs(graph,IDS[0],steps=73,trials=1)
    model=resolve_model()
    expected=reference(graph,stimulus,model)
    prepared=configured.prepare(graph,stimulus,model)
    assert prepared.independent_neurons
    cold,first=configured.run_prepared(prepared,stimulus,record=False,return_state=True)
    warm,second=configured.run_prepared(prepared,stimulus,record=False,return_state=True)
    assert cold.timings['jit_and_warmup_s']>0 and warm.timings['jit_and_warmup_s']==0
    assert first.values.buffers is not second.values.buffers
    assert_state(configured.host_state(first,prepared),expected.state)
    assert_state(configured.host_state(second,prepared),expected.state)


def test_default_model_keeps_explicit_selected_cpu_target():
    numpy=resolve_model(cpu_target='numpy')
    cpp=resolve_model(cpu_target='cpp')
    assert numpy.reference_codegen_target=='numpy'
    assert cpp.reference_codegen_target=='cpp'
    assert cpp.reference_optimise is not None
    assert not numpy.state_specialization and not cpp.state_specialization
    assert numpy.signature!=cpp.signature


def test_generated_and_combined_events_keep_exact_immutable_hash_owner(tmp_path,original):
    from hashlib import sha256
    from mlx_backend.stimuli import event_input_snapshot
    import mlx_engine
    graph=load_connectome(*files(tmp_path))
    spec=mlx_engine.experiment_spec(dict(key='immutable-input-owner',
        neu_exc=[IDS[0][0],IDS[0][0]],neu_exc2=[IDS[0][3]],
        stim_rate=3700.,stim_rate2=7200.))
    trials,_=mlx_engine.stimuli(graph,spec,73,2,813)
    combined=mlx_engine.combine_trials(trials)
    np.testing.assert_array_equal(combined.events,np.concatenate([trial.events for trial in trials]))
    for stimulus in (*trials,combined):
        owner=stimulus.events
        while type(owner) is np.ndarray:owner=owner.base
        assert type(owner) is bytes
        assert sha256(owner).hexdigest()==stimulus.sha256
        before=event_input_snapshot(stimulus.events)
        after=event_input_snapshot(stimulus.events)
        assert before[2][2] is after[2][2] is owner
        with pytest.raises(ValueError):stimulus.events.setflags(write=True)
