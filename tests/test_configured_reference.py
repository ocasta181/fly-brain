"""CPU-only checks of the independent configurable Brian replay helper."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from configured_reference import reference
from mlx_backend.model import resolve_model
from mlx_backend.models import Connectome, Stimulus
from mlx_backend.stimuli import SeededEvents


@pytest.fixture
def brian(monkeypatch):
    b = pytest.importorskip("brian2")
    monkeypatch.setitem(b.prefs, "codegen.target", "numpy")
    monkeypatch.setitem(b.prefs, "core.default_float_dtype", np.float64)
    monkeypatch.setitem(b.prefs, "core.default_integer_dtype", np.int32)
    monkeypatch.setattr(b.defaultclock, "dt", .1 * b.ms)
    return b


def graph(neurons=3, sources=(1, 0, 0, 2, 1), destinations=(2, 1, 2, 1, 0),
          counts=(180.5, 300.25, -23.75, 7.25, -31.5)):
    # Deliberately inconsistent pre-scaled weights detect accidental use of the
    # legacy fixed .275 mV representation in the oracle.
    return Connectome(
        np.arange(neurons), np.asarray(sources, np.int32),
        np.asarray(destinations, np.int32), np.asarray(counts, np.float64),
        np.full(len(counts), 12345.0),
    )


def stimulus(events, targets, *, dt_ms=.1, amplitudes_mv=None, trial_indices=None):
    return Stimulus(
        events, tuple(targets), (0.0,) * len(targets),
        tuple(range(events.shape[0])) if trial_indices is None else tuple(trial_indices),
        7, 1, None, dt_ms, amplitudes_mv,
    )


@pytest.mark.parametrize("dt_ms,delay_ms,refractory_ms,changed", [
    (.1, 1.8, 2.2, False),
    (.05, 0., .35, False),
    (.2, .3, .29995, False),
    (.23, .46, .35, True),
    (.3, 0., 0., True),
])
@pytest.mark.parametrize("silenced", [(), (0,)])
def test_matches_frozen_application_with_deterministic_poisson_channels(
    brian, tmp_path, dt_ms, delay_ms, refractory_ms, changed, silenced,
):
    b = brian
    path = Path(__file__).resolve().parents[1] / "code/paper-phil-drosophila/model.py"
    spec = importlib.util.spec_from_file_location("configured_replay_frozen_model", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    b.defaultclock.dt = dt_ms * b.ms
    parameters = dict(original.default_params)
    parameters.update(t_dly=delay_ms * b.ms, t_rfc=refractory_ms * b.ms,
                      r_poi=1 / b.defaultclock.dt, r_poi2=0 * b.Hz)
    if changed:
        parameters.update(v_0=-61*b.mV, v_rst=-57*b.mV, v_th=-43*b.mV,
                          t_mbr=17*b.ms, tau=3*b.ms, w_syn=.4*b.mV, f_poi=123)
    connectome = graph()
    completeness, connectivity = tmp_path / "neurons.csv", tmp_path / "edges.parquet"
    pd.DataFrame(index=connectome.neuron_ids).to_csv(completeness)
    pd.DataFrame({
        "Presynaptic_Index": connectome.sources,
        "Postsynaptic_Index": connectome.destinations,
        "Excitatory x Connectivity": connectome.counts,
    }).to_parquet(connectivity)
    neurons, synapses, spikes = original.create_model(completeness, connectivity, parameters)
    inputs, neurons = original.poi(neurons, [0, 0], [2], parameters)
    original.silence(silenced, synapses)
    trace = b.StateMonitor(neurons, ["v", "g", "rfc", "lastspike", "not_refractory"],
                           record=True, when="end")
    network = b.Network(neurons, synapses, spikes, trace, *inputs)
    steps = 120
    network.run(steps * b.defaultclock.dt, namespace={})
    events = np.ones((1, steps, 3), np.uint8)
    events[:, :, 2] = 0
    actual = reference(connectome, stimulus(events, (0, 0, 2), dt_ms=dt_ms),
                       resolve_model(parameters), silenced, record_state=True)
    np.testing.assert_array_equal(actual.spikes.steps,
                                  np.rint(np.asarray(spikes.t_) / float(b.defaultclock.dt)))
    np.testing.assert_array_equal(actual.spikes.neurons, np.asarray(spikes.i))
    assert actual.spikes.steps.size > 0
    for name in ("v", "g", "rfc", "lastspike", "not_refractory"):
        np.testing.assert_array_equal(actual.state[name][0], getattr(neurons, name + "_")[:])
        np.testing.assert_array_equal(actual.traces[name][0], getattr(trace, name + "_").T)


def test_duplicate_channels_use_their_own_amplitudes_in_creation_order(brian):
    model = resolve_model({"eq_th": "False"})
    events = np.ones((1, 1, 3), np.uint8)
    actual = reference(graph(1, (), (), ()), stimulus(
        events, (0, 0, 0), amplitudes_mv=(1e19, -1e19, 1000.0),
    ), model)
    # Sequential binary64 additions lose the resting voltage in the first
    # channel.  Grouping channels into a single summed impulse gives .948 V.
    assert actual.state["v"][0, 0] == 1.0


def test_simultaneous_arrivals_follow_source_then_original_edge_order(brian):
    b = brian
    model = resolve_model({"eq_th": "i < 3", "t_dly": 0*b.ms, "w_syn": 1*b.volt})
    large = float(2**54)
    connectome = graph(4, (2, 0, 2, 1), (3, 3, 3, 3), (large, -large, 1., 1.))
    actual = reference(connectome, stimulus(np.zeros((1, 1, 0), np.uint8), ()), model)
    # Actual queue edge order is [1, 3, 0, 2]: [-large, 1, large, 1].
    # Iterating the whole original edge table would instead produce 2 V.
    assert actual.state["g"][0, 3] == 1.0


def test_input_on_a_threshold_step_is_discarded_before_reset(brian):
    model = resolve_model()
    events = np.array([[[1], [1], [0]]], np.uint8)
    actual = reference(graph(1, (), (), ()), stimulus(events, (0,)), model,
                       record_state=True)
    np.testing.assert_array_equal(actual.spikes.steps, [1])
    np.testing.assert_array_equal(actual.traces["not_refractory"][0, :, 0], [True, False, True])
    assert actual.traces["v"][0, 1, 0] == model.v_rst
    assert actual.state["v"][0, 0] == model.v_0
    assert actual.state["rfc"][0, 0] == 0


@pytest.mark.parametrize("unless_refractory", [True, False])
def test_zero_delay_self_arrival_is_overwritten_by_same_step_reset(brian, unless_refractory):
    b = brian
    flags = " (unless refractory)" if unless_refractory else ""
    model = resolve_model({
        "eqs": (f"dv/dt = (v_0-v)/t_mbr : volt{flags}\n"
                f"dg/dt = -g/tau : volt{flags}\nrfc : second\n"),
        "t_dly": 0*b.ms, "eq_rst": "v = v_rst; g = 3*mV",
    })
    events = np.array([[[1], [0]]], np.uint8)
    actual = reference(graph(1, (0,), (0,), (4.5,)), stimulus(events, (0,)),
                       model, record_state=True)
    np.testing.assert_array_equal(actual.spikes.steps, [1])
    assert actual.state["g"][0, 0] == .003


def test_arrivals_to_refractory_targets_are_discarded(brian):
    b = brian
    model = resolve_model({
        "eqs": ("dv/dt = (v_0-v)/t_mbr : volt (unless refractory)\n"
                "dg/dt = -g/tau : volt (unless refractory)\nrfc : second\n"),
        "eq_th": "True", "eq_rst": "v = v_rst", "t_rfc": .3*b.ms,
        "t_dly": .1*b.ms,
    })
    events = np.zeros((1, 5, 0), np.uint8)
    actual = reference(graph(1, (0,), (0,), (7.5,)), stimulus(events, ()),
                       model, record_state=True)
    np.testing.assert_array_equal(actual.spikes.steps, [0, 3])
    np.testing.assert_array_equal(actual.traces["g"], 0)


def test_model_schedule_controls_arrival_and_reset_order(brian, monkeypatch):
    b = brian
    parameters = {
        "eqs": ("dv/dt = (v_0-v)/t_mbr : volt\n"
                "dg/dt = -g/tau : volt\nrfc : second\n"),
        "eq_th": "True", "eq_rst": "v = v_rst; g = 3*mV", "t_dly": 0*b.ms,
    }
    ordinary = resolve_model(parameters)
    monkeypatch.setitem(b.prefs, "core.network.default_schedule",
                        ["start", "groups", "thresholds", "resets", "synapses", "end"])
    reordered = resolve_model(parameters)
    connectome = graph(1, (0,), (0,), (4.5,))
    inputs = stimulus(np.zeros((1, 1, 0), np.uint8), ())
    before = reference(connectome, inputs, ordinary)
    after = reference(connectome, inputs, reordered)
    assert before.state["g"][0, 0] == .003
    assert after.state["g"][0, 0] == .003 + 4.5 * ordinary.w_syn


def test_outgoing_only_silencing_preserves_inputs_and_incoming_synapses(brian):
    b = brian
    model = resolve_model({"t_dly": 0*b.ms, "eq_th": "(i == 0) and (v > v_th)"})
    events = np.zeros((1, 3, 1), np.uint8)
    events[0, 0, 0] = 1
    connectome = graph(2, (0,), (1,), (2.25,))
    incoming = reference(connectome, stimulus(events, (0,)), model, (1,))
    outgoing = reference(connectome, stimulus(events, (0,)), model, (0,))
    np.testing.assert_array_equal(outgoing.spikes.steps, incoming.spikes.steps)
    assert incoming.state["g"][0, 1] > 0
    assert outgoing.state["g"][0, 1] == 0


def test_streamed_events_trials_custom_state_and_preferences_are_preserved(brian):
    b = brian
    model = resolve_model({
        "eqs": resolve_model().eqs + "count : integer\nflag : boolean\n",
        "eq_rst": "v = v_rst; g = 0*mV; count += 1; flag = not flag",
    })
    source = SeededEvents(15, (4000.0,), (2**53 + 1, 2**53 + 3), 89, 7)
    inputs = stimulus(source, (0,), trial_indices=source.trial_indices)
    b.prefs.codegen.target = "auto"
    actual = reference(graph(1, (), (), ()), inputs, model, record_state=True)
    assert b.prefs.codegen.target == "auto"
    assert set(actual.spikes.trials) == set(source.trial_indices)
    assert actual.state["count"].dtype == np.dtype(np.int32)
    assert actual.state["flag"].dtype == np.dtype(bool)
    for row, trial_id in enumerate(source.trial_indices):
        single = replace(inputs, events=source.chunk(0, 15)[row:row + 1],
                         trial_indices=(trial_id,))
        expected = reference(graph(1, (), (), ()), single, model)
        selected = actual.spikes.trials == trial_id
        np.testing.assert_array_equal(actual.spikes.steps[selected], expected.spikes.steps)
        np.testing.assert_array_equal(actual.state["count"][row], expected.state["count"][0])
        assert actual.state["count"][row, 0] == np.count_nonzero(selected)


def test_zero_duration_retains_initial_state(brian):
    model = resolve_model()
    actual = reference(graph(1, (), (), ()), stimulus(np.empty((1, 0, 1), np.uint8), (0,)),
                       model, record_state=True)
    assert actual.spikes.steps.size == 0
    assert actual.traces["v"].shape == (1, 0, 1)
    assert actual.state["v"][0, 0] == model.v_0
    assert actual.state["lastspike"][0, 0] == -10000
