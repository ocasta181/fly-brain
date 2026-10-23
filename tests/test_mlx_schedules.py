"""Schedule-sensitive independent Brian CPU versus actual Metal execution."""

import importlib.util
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from native_execution import require_metal
mx = require_metal('Metal is required', allow_module_level=True)
b = pytest.importorskip("brian2")

from configured_reference import reference
from mlx_backend import configured
from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
from mlx_backend.models import Connectome
from test_mlx_configured import graph, inputs, assert_matches_cpu
from test_model_schedules import PHASES, EXTRA_SCHEDULES

pytestmark = pytest.mark.metal


def scheduled(monkeypatch, phases, delay=0, **params):
    monkeypatch.setitem(b.prefs, "core.network.default_schedule", list(phases))
    return resolve_model({"t_dly": delay*.1*b.ms, **params})


@pytest.mark.parametrize("neurons", [1, 17, 31, 32, 33])
@pytest.mark.parametrize("density", [0, .2, 1])
def test_unit_recording_keeps_trial_and_tail_word_boundaries(neurons, density):
    flags = (np.random.default_rng(104).random((3, 9, neurons)) < density).astype(np.uint32)
    ids = np.array([19, 2, 7], np.int64)
    starts = np.arange(9, dtype=np.int64)*3+4
    expected = sorted((int(ids[b]), int(n), int(starts[t]))
                      for b, t, n in zip(*np.nonzero(flags)))
    expected.sort(key=lambda row: (row[0], row[2], row[1]))
    with mx.stream(mx.gpu):
        parts = configured._collect_unit_windows(
            [mx.array(flags[:, t].reshape(-1)) for t in range(9)], starts, neurons, ids)
    actual = list(zip(*(np.concatenate([part[i] for part in parts]) for i in range(3)))) if parts else []
    assert actual == expected


@pytest.mark.parametrize("phases", list(permutations(PHASES)))
@pytest.mark.parametrize("delay", [0, 1])
def test_all_phase_orders_match_cpu_with_short_delays(monkeypatch, phases, delay):
    model = scheduled(monkeypatch, ("start", *phases, "end"), delay,
        eqs=DEFAULT_EQUATIONS + "count : integer\n",
        eq_rst="v=v_rst; g=0*mV; count+=1", t_rfc=.35*b.ms)
    result, _, p = assert_matches_cpu(graph(), inputs(model, steps=43), model, chunk_steps=7)
    assert result.spikes.steps.size
    assert p.window == max(1, model.arrival_delay_steps)


@pytest.mark.parametrize("schedule", EXTRA_SCHEDULES)
def test_extra_duplicate_automatic_and_omitted_slots_execute(monkeypatch, schedule):
    model = scheduled(monkeypatch, schedule, 7)
    # Model preparation/execution must use its snapshot after ambient changes.
    monkeypatch.setitem(b.prefs, "core.network.default_schedule", ["start", *PHASES, "end"])
    c = replace(graph(), pin="schedule-layout-identity")
    _, _, p = assert_matches_cpu(c, inputs(model, steps=47), model)
    assert p.model.schedule == schedule


def test_schedule_materially_changes_spikes_and_continuous_state(monkeypatch):
    monkeypatch.setitem(b.prefs, "codegen.target", "numpy")
    ordinary = scheduled(monkeypatch, ("start", *PHASES, "end"), 1)
    changed = scheduled(monkeypatch, ("start", "synapses", "thresholds", "resets", "groups", "end"), 1)
    c, s = graph(), inputs(ordinary, steps=53)
    before, state_before, p_before = assert_matches_cpu(c, s, ordinary)
    after, state_after, p_after = assert_matches_cpu(c, s, changed)
    assert not np.array_equal(before.spikes.steps, after.spikes.steps)
    assert not np.array_equal(configured.host_state(state_before, p_before)["v"],
                              configured.host_state(state_after, p_after)["v"])
    assert p_before.model.signature != p_after.model.signature


@pytest.mark.parametrize("delay", [0, 1])
@pytest.mark.parametrize("block_windows,chunk_steps", [(3, 7), (17, 19)])
def test_short_delay_export_orders_trials_across_recording_blocks(monkeypatch, delay, block_windows, chunk_steps):
    model = scheduled(monkeypatch, ("start", "resets", "groups", "thresholds", "synapses", "end"), delay)
    s = replace(inputs(model, steps=43), trial_indices=(19, 2))
    c = graph()
    expected = reference(c, s, model)
    p = configured.prepare(c, s, model)
    assert p.window == 1
    order = np.lexsort((expected.spikes.neurons, expected.spikes.steps, expected.spikes.trials))
    for stream in (mx.gpu, mx.cpu):
        with mx.stream(stream):
            actual, state = configured.run_prepared(p, s, block_windows=block_windows,
                                                   chunk_steps=chunk_steps, return_state=True)
        for name in ("trials", "neurons", "steps"):
            np.testing.assert_array_equal(getattr(actual.spikes, name), getattr(expected.spikes, name)[order])
        for name, values in expected.state.items():
            _assert_state_equal(configured.host_state(state, p)[name], values, name)


@pytest.mark.parametrize("phases,delay", [
    (("synapses", "resets", "thresholds", "groups"), 0),
    (("synapses", "resets", "thresholds", "groups"), 31),
    (("synapses", "groups", "thresholds", "resets"), 32),
    (("resets", "synapses", "groups", "thresholds"), 1),
    (("thresholds", "resets", "synapses", "groups"), 7),
    (("thresholds", "synapses", "resets", "groups"), 33),
    (("resets", "groups", "thresholds", "synapses"), 65),
])
def test_partial_continuation_matches_cpu_states_spikes_and_pending_arrivals(monkeypatch, phases, delay):
    model = scheduled(monkeypatch, ("start", *phases, "end"), delay, t_rfc=.35*b.ms)
    c, s = graph(), inputs(model, steps=103)
    expected = reference(c, s, model, record_state=True)
    actual, whole, p = assert_matches_cpu(c, s, model, chunk_steps=11)
    state = configured.initial_state(p, 2)
    recorded = []
    call = 0
    # Extend the horizon and replace staged inputs at a partial-window boundary.
    for event_begin, event_end in ((0, 17), (17, s.events.shape[1])):
        events = mx.array(np.ascontiguousarray(s.events[:, event_begin:event_end]).reshape(-1))
        while state.step < event_end:
            length = min((1, 3, 11, 7)[call % 4], p.window-state.step % p.window,
                         event_end-state.step)
            begin = state.step
            state, bits = configured.advance(p, state, events, event_end, length,
                event_begin=event_begin, event_steps=event_end-event_begin)
            configured.evaluate_state(state)
            for name, values in configured.host_state(state, p).items():
                _assert_state_equal(values, expected.traces[name][:, state.step-1],
                                    f"{phases} {name} after step {state.step-1}")
            words = np.asarray(bits)
            for bit in range(length):
                for row in np.flatnonzero(words & (1 << bit)):
                    recorded.append((s.trial_indices[row//p.neurons], row%p.neurons, begin+bit))
            _assert_pending(state, p, s, c, expected.spikes)
            call += 1
    assert sorted(recorded, key=lambda row: (row[0], row[2], row[1])) == list(zip(
        expected.spikes.trials, expected.spikes.neurons, expected.spikes.steps))
    for name in state.values:
        np.testing.assert_array_equal(np.asarray(state.values[name]), np.asarray(whole.values[name]))
    if state.spikes is not None:
        np.testing.assert_array_equal(np.asarray(state.spikes), np.asarray(whole.spikes))
    with monkeypatch.context() as patch:
        patch.setattr(configured, "collect_words", lambda *a, **k: pytest.fail("recording disabled collected spikes"))
        unrecorded, quiet = configured.run_prepared(p, s, record=False, return_state=True, chunk_steps=17)
    assert unrecorded.spikes.steps.size == 0
    assert unrecorded.timings["collection_s"] == 0
    assert unrecorded.input_sha256 == actual.input_sha256
    for name, values in configured.host_state(quiet, p).items():
        _assert_state_equal(values, expected.state[name], name)
    _assert_pending(quiet, p, s, c, expected.spikes)


def _assert_state_equal(actual, expected, message):
    if expected.dtype == np.float64:
        actual, expected = actual.view(np.uint64), expected.view(np.uint64)
    np.testing.assert_array_equal(actual, expected, err_msg=message)


def _assert_pending(state, p, s, c, spikes):
    """Derive queue markers and retained source words from CPU spikes alone."""
    delay, w, n = p.model.arrival_delay_steps, p.window, p.neurons
    history, due = {}, {}
    flags = np.zeros(n*len(s.trial_indices), np.uint32)
    trial_rows = {trial: row for row, trial in enumerate(s.trial_indices)}
    for trial, neuron, step in zip(spikes.trials, spikes.neurons, spikes.steps):
        if step >= state.step:
            continue
        row = trial_rows[trial]
        if step == state.step-1:
            flags[row*n+neuron] = 1
        if delay == 0:
            continue
        source_window, source_bit = divmod(step, w)
        if source_window >= (state.step-delay)//w:
            history.setdefault(source_window, np.zeros_like(flags))[row*n+neuron] |= 1 << source_bit
        arrival = step+delay
        if arrival < state.step:
            continue
        window, bit = divmod(arrival, w)
        for destination in c.destinations[c.sources == neuron]:
            due.setdefault(window, np.zeros_like(flags))[row*n+destination] |= 1 << bit
    if p.model.reset_uses_previous_spikes:
        np.testing.assert_array_equal(np.asarray(state.spikes), flags)
    else:
        assert state.spikes is None
    for actual, expected in ((state.history, history), (state.due, due)):
        for key in actual.keys() | expected.keys():
            np.testing.assert_array_equal(np.asarray(actual.get(key, np.zeros_like(flags))),
                                          expected.get(key, np.zeros_like(flags)))


@pytest.mark.parametrize("phases", [
    ("thresholds", "groups", "synapses", "resets"),
    ("resets", "synapses", "thresholds", "groups"),
    ("synapses", "groups", "resets", "thresholds"),
])
def test_reset_mutated_refractory_and_unconditional_writes(monkeypatch, phases):
    model = scheduled(monkeypatch, ("start", *phases, "end"), 0,
        eqs=DEFAULT_EQUATIONS.replace(" (unless refractory)", "") + "count : integer\n",
        eq_rst="v=v_rst; g=2*mV; rfc=.25*ms + .1*ms*int(i==1); count+=1",
        t_rfc=.35*b.ms)
    assert_matches_cpu(graph(), inputs(model, steps=49), model)


@pytest.mark.parametrize("phases,delay", [
    (("synapses", "resets", "groups", "thresholds"), 32),
    (("resets", "thresholds", "synapses", "groups"), 0),
])
def test_arrival_cache_overflow_keeps_source_and_edge_order(monkeypatch, phases, delay):
    weights = np.array([1e16, 1., -1e16, 3., -7., 11., .125])
    order = np.arange(len(weights))[::-1]
    c = Connectome(np.arange(8), order.astype(np.int32), np.full(7, 7, np.int32),
                   weights[order], weights[order])
    model = scheduled(monkeypatch, ("start", *phases, "end"), delay,
        eqs=DEFAULT_EQUATIONS.replace(" (unless refractory)", ""),
        eq_th="(i < 7) and (t < dt)", w_syn=1*b.volt)
    s = inputs(model, targets=(), trials=1, steps=model.arrival_delay_steps+1)
    assert_matches_cpu(c, s, model)


def test_zero_horizon_with_early_reset_preserves_initial_eventspace(monkeypatch):
    model = scheduled(monkeypatch, ("start", "resets", "thresholds", "synapses", "groups"))
    _, state, p = assert_matches_cpu(graph(), inputs(model, steps=0), model)
    np.testing.assert_array_equal(np.asarray(state.spikes), np.zeros(2*p.neurons, np.uint32))


@pytest.mark.parametrize("phases", [PHASES,
    ("resets", "thresholds", "synapses", "groups"),
    ("synapses", "groups", "resets", "thresholds"),
])
def test_poisson_channel_additions_remain_in_creation_order(monkeypatch, phases):
    model = scheduled(monkeypatch, ("start", *phases, "end"), 1, eq_th="False")
    c = Connectome(np.arange(1), np.array([], np.int32), np.array([], np.int32),
                   np.array([], np.float64), np.array([], np.float64))
    s = inputs(model, steps=1, trials=1, amplitudes=(1e19, -1e19, 1000.), targets=(0, 0, 0))
    _, state, p = assert_matches_cpu(c, s, model)
    if phases.index("synapses") > phases.index("groups"):
        assert configured.host_state(state, p)["v"][0, 0] == 1.


@pytest.mark.parametrize("phases", [
    ("synapses", "resets", "thresholds", "groups"),
    ("thresholds", "resets", "synapses", "groups"),
])
@pytest.mark.parametrize("target", ["numpy", "cython"])
def test_frozen_application_actual_poisson_inputs_match_metal(monkeypatch, tmp_path, phases, target):
    import pandas as pd
    monkeypatch.setitem(b.prefs, "codegen.target", target)
    monkeypatch.setattr(b.defaultclock, "dt", .1*b.ms)
    path = Path(__file__).resolve().parents[1]/"code/paper-phil-drosophila/model.py"
    spec = importlib.util.spec_from_file_location("schedule_frozen_application", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    model = scheduled(monkeypatch, ("start", *phases), 0,
                      r_poi=10000*b.Hz, r_poi2=0*b.Hz, t_rfc=.35*b.ms)
    c = graph()
    completeness, connectivity = tmp_path/"neurons.csv", tmp_path/"edges.parquet"
    pd.DataFrame(index=c.neuron_ids).to_csv(completeness)
    pd.DataFrame({"Presynaptic_Index": c.sources, "Postsynaptic_Index": c.destinations,
                  "Excitatory x Connectivity": c.counts}).to_parquet(connectivity)
    neurons, synapses, spikes = original.create_model(completeness, connectivity, model.brian_namespace())
    poisson, _ = original.poi(neurons, [0, 0], [2], model.brian_namespace())
    network = b.Network(neurons, synapses, spikes, *poisson)
    network.run(39*b.defaultclock.dt, namespace={})
    s = inputs(model, targets=(0, 0, 2), steps=39, trials=1)
    events = np.ones_like(s.events)
    events[:, :, 2] = 0
    s = replace(s, events=events)
    result, state, p = assert_matches_cpu(c, s, model)
    np.testing.assert_array_equal(result.spikes.steps, np.rint(spikes.t_[:]/model.dt_s))
    np.testing.assert_array_equal(result.spikes.neurons, spikes.i[:])
    for name, values in configured.host_state(state, p).items():
        _assert_state_equal(values[0], np.asarray(getattr(neurons, name+"_")[:]), name)


@pytest.mark.parametrize("record", [False, True])
def test_public_adapter_uses_native_schedule_after_independent_reference(monkeypatch, record):
    import mlx_engine
    model = scheduled(monkeypatch, ("start", "synapses", "resets", "thresholds", "groups"), 1)
    c, s = graph(), inputs(model, steps=41)
    expected = reference(c, s, model)
    monkeypatch.setattr(b.Network, "run", lambda *a, **k: pytest.fail("native execution ran Brian"))
    actual = mlx_engine.execute(c, s, (), model=model, record=record)
    assert "Apple" in actual.device_name
    for name in ("trials", "neurons", "steps"):
        values = getattr(actual.spikes, name)
        if record:
            np.testing.assert_array_equal(values, getattr(expected.spikes, name))
        else:
            assert values.size == 0
