"""Native TimedArray execution, checked against independent Brian NumPy."""
from native_execution import forbid_call, require_metal
from state_assertions import assert_raw_state, assert_spikes_equal

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

mx = require_metal('Metal is required', allow_module_level=True)
b = pytest.importorskip("brian2")

from configured_reference import reference
from mlx_backend import configured
from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
from mlx_backend.models import Connectome, Stimulus

pytestmark = pytest.mark.metal


@pytest.fixture(autouse=True)
def numpy_reference_policy(monkeypatch):
    monkeypatch.setitem(b.prefs, "codegen.target", "numpy")


def _graph():
    return Connectome(np.arange(3), np.array([0, 1, 2], np.int32),
                      np.array([1, 2, 0], np.int32), np.array([.125, -.25, .5]),
                      np.array([.125, -.25, .5])*.275)


def _stimulus(model, steps=67, trials=2):
    events = np.zeros((trials, steps, 1), np.uint8)
    events[:, [step for step in (0, 1, 17, 31, 33) if step < steps], :] = 1
    return Stimulus(events, (0,), (100.,), tuple(range(trials)), 0, 0,
                    sha256(events.tobytes()).hexdigest(), model.dt_ms)


def _assert_state(state, prepared, expected):
    assert_raw_state(configured.host_state(state, prepared), expected)


@pytest.mark.parametrize("target", ["numpy", "cython"])
def test_clock_only_gpu_cache_matches_uncached_calls_and_state_dependent_time(target, monkeypatch):
    monkeypatch.setitem(b.prefs, "codegen.target", target)
    monkeypatch.setitem(b.prefs, "codegen.runtime.cython.cache_dir",
                        str(Path(__file__).resolve().parents[1]/"executions/cython-timedarray-clock-cache"))
    model = resolve_model({
        "drive": b.TimedArray([0., 1.+1e-9, -.125, .75, 0., 1.25, .5, 1.], dt=.15*b.ms),
        "eqs": DEFAULT_EQUATIONS+"sample : 1\ncounter : integer\n",
        "eq_th": "drive(t-2*dt)>.5",
        "eq_rst": "v=v_rst; sample=drive(t+dt/16)+drive(t+counter*dt); counter+=1",
        "t_rfc": 0*b.ms,
    })
    graph, stimulus = _graph(), _stimulus(model)
    expected = reference(graph, stimulus, model)
    prepared = configured.prepare(graph, stimulus, model)
    assert len(prepared.program.shared_timed_calls) == 2
    uncached = replace(prepared, program=replace(prepared.program, shared_timed_calls=(), timed_cache_name=None))
    forbid_call(b.Network, 'run', 'Native clock cache ran a CPU timestep', monkeypatch)
    for current in (prepared, uncached):
        result, state = configured.run_prepared(current, stimulus, return_state=True,
                                               chunk_steps=19, block_windows=2)
        assert_spikes_equal(result.spikes, expected.spikes)
        _assert_state(state, current, expected.state)


@pytest.mark.parametrize("target", ["numpy", "cython"])
def test_timedarray_cache_and_stable_math_preserve_exact_states(target, monkeypatch):
    monkeypatch.setitem(b.prefs, "codegen.target", target)
    model = resolve_model({
        "drive": b.TimedArray([0., 1.+1e-9, -.125, .75, 0., 1.25, .5, 1.], dt=.15*b.ms),
        "eqs": DEFAULT_EQUATIONS+"phase : integer\nsample : 1\na : 1\nc : 1\n",
        "eq_th": "drive(t+dt/16)>.25",
        "eq_rst": "v=v_rst; g=0*mV; sample=expm1(drive(t+dt/16)/16); "
                  "a=exprel(drive(t+dt/16)/16); c=log1p(drive(t+phase*dt)/16); phase+=1",
        "t_rfc": 0*b.ms,
    })
    graph, stimulus = _graph(), _stimulus(model)
    expected = reference(graph, stimulus, model)
    prepared = configured.prepare(graph, stimulus, model)
    forbid_call(b.Network, 'run', 'Native math/TimedArray execution ran a CPU timestep', monkeypatch)
    actual, state = configured.run_prepared(prepared, stimulus, return_state=True,
                                           chunk_steps=19, block_windows=2)
    assert_spikes_equal(actual.spikes, expected.spikes)
    _assert_state(state, prepared, expected.state)
    assert np.any(expected.state["phase"] > 1)


@pytest.mark.parametrize("dimensions,delay", [(1, 0), (1, 18), (2, 0), (2, 33)])
def test_timedarray_all_stages_match_cpu_across_chunks_and_continuation(dimensions, delay, monkeypatch):
    values = np.array([0., 1.+1e-9, -.125, .75, 0., 1.25, .5, 1.])
    if dimensions == 2:
        values = np.stack([values, values[::-1], values*.75], axis=1)
    drive = b.TimedArray(values, dt=.2*b.ms)
    call = "drive(t)" if dimensions == 1 else "drive(t, i-3)"
    threshold = "drive(t-2*dt)" if dimensions == 1 else "drive(t-2*dt, i-3)"
    reset = "drive(t+dt)" if dimensions == 1 else "drive(t+dt, i)"
    model = resolve_model({
        "drive": drive,
        "eqs": DEFAULT_EQUATIONS.replace("v_0 - v + g", f"v_0 - v + g + {call}*mV")
               + "sample : 1\ncounter : integer\n",
        "eq_th": threshold+" > .5",
        "eq_rst": f"v=v_rst; g=0*mV; sample={reset}; counter+=1",
        "t_rfc": 0*b.ms, "t_dly": delay*.1*b.ms,
    })
    c, stimulus = _graph(), _stimulus(model)
    expected = reference(c, stimulus, model)
    forbid_call(b.Network, 'run', 'Native TimedArray execution ran a CPU timestep', monkeypatch)
    prepared = configured.prepare(c, stimulus, model)
    drive.values[:] = 100.  # execution uses the immutable resolved snapshot
    actual, whole = configured.run_prepared(prepared, stimulus, return_state=True,
                                           chunk_steps=35, block_windows=2)
    assert_spikes_equal(actual.spikes, expected.spikes)
    _assert_state(whole, prepared, expected.state)
    state = configured.initial_state(prepared, 2)
    events = mx.array(stimulus.events.reshape(-1))
    continued_spikes = []
    while state.step < 67:
        begin = state.step
        length = min((1, 7, 3, 11)[state.step % 4], prepared.window-state.step % prepared.window, 67-state.step)
        state, bits = configured.advance(prepared, state, events, 67, length)
        words = np.asarray(bits)
        for bit in range(length):
            continued_spikes.extend((row//3, row%3, begin+bit)
                                    for row in np.flatnonzero(words & (1 << bit)))
    configured.evaluate_state(state)
    _assert_state(state, prepared, expected.state)
    assert sorted(continued_spikes, key=lambda item: (item[0], item[2], item[1])) == list(
        zip(expected.spikes.trials, expected.spikes.neurons, expected.spikes.steps))
    np.testing.assert_array_equal(np.asarray(state.release), np.asarray(whole.release))
    for field in ("history", "due"):
        left, right = getattr(state, field), getattr(whole, field)
        for key in left.keys() | right.keys():
            zeros = np.zeros(6, np.uint32)
            np.testing.assert_array_equal(np.asarray(left.get(key, zeros)), np.asarray(right.get(key, zeros)))


@pytest.mark.parametrize("threshold", ["drive(t, i+3) > -1", "drive(t, i-4) > -1",
                                      "drive(t, counter) > -1",
                                      "clip(v/mV, 0, drive(t, i+3)) > 0",
                                      "(v > v_th) and (drive(t, i+3) > 0)"])
def test_invalid_timedarray_columns_raise_without_device_out_of_bounds(threshold):
    drive = b.TimedArray(np.arange(9.).reshape(3, 3), dt=.1*b.ms)
    model = resolve_model({"drive": drive, "eqs": DEFAULT_EQUATIONS+"counter : integer\n",
                           "eq_th": threshold,
                           "eq_rst": "counter+=1", "t_rfc": 0*b.ms})
    c, stimulus = _graph(), _stimulus(model, steps=7, trials=1)
    with pytest.raises(Exception) as cpu_error:
        reference(c, stimulus, model)
    error = cpu_error.value
    while error.__cause__ is not None:
        error = error.__cause__
    assert isinstance(error, IndexError)
    prepared = configured.prepare(c, stimulus, model)
    with pytest.raises(IndexError, match="TimedArray column"):
        configured.run_prepared(prepared, stimulus)
    # Error state survives continuation and both public materialization paths.
    state = configured.initial_state(prepared)
    events = mx.array(stimulus.events.reshape(-1))
    while state.step < 7:
        state, _ = configured.advance(prepared, state, events, 7, 1)
    with pytest.raises(IndexError, match="TimedArray column"):
        configured.evaluate_state(state)
    with pytest.raises(IndexError, match="TimedArray column"):
        configured.host_state(state, prepared)


def test_two_timedarrays_and_no_recording_keep_distinct_values():
    a = b.TimedArray([1.+1e-9, 0., 2., 0.], dt=.15*b.ms)
    z = b.TimedArray([[0., 1., 2.], [3., 2., 1.]], dt=.1*b.ms)
    model = resolve_model({"a": a, "z": z, "eqs": DEFAULT_EQUATIONS+"sample : 1\n",
                           "eq_th": "a(t-2*dt)+z(t,i) > 2",
                           "eq_rst": "v=v_rst; sample=a(t)+z(t,i)", "t_rfc": 0*b.ms})
    c, stimulus = _graph(), _stimulus(model, steps=39)
    expected = reference(c, stimulus, model)
    prepared = configured.prepare(c, stimulus, model)
    assert len(prepared.program.input_arrays) == 2
    assert len(prepared.input_arrays) == 1  # one packed Metal binding
    result, state = configured.run_prepared(prepared, stimulus, record=False, return_state=True, chunk_steps=19)
    assert result.spikes.steps.size == 0
    _assert_state(state, prepared, expected.state)
    # Changing only TimedArray data must not reuse the preceding prepared data.
    changed = resolve_model({"a": b.TimedArray([5., 6., 7., 8.], dt=.15*b.ms),
                             "z": z, "eqs": model.eqs, "eq_th": model.eq_th,
                             "eq_rst": model.eq_rst, "t_rfc": 0*b.ms})
    changed_input = replace(stimulus, dt_ms=changed.dt_ms)
    changed_expected = reference(c, changed_input, changed)
    changed_prepared = configured.prepare(c, changed_input, changed)
    _, changed_state = configured.run_prepared(changed_prepared, changed_input, return_state=True)
    _assert_state(changed_state, changed_prepared, changed_expected.state)


def test_invariant_invalid_column_does_not_execute_for_zero_steps():
    model = resolve_model({"drive": b.TimedArray([[1., 2.]], dt=.1*b.ms),
                           "eqs": DEFAULT_EQUATIONS+"sample : 1\n", "eq_th": "t>=0*ms",
                           "eq_rst": "sample=drive(0*ms, 2)", "t_rfc": 0*b.ms})
    c, stimulus = _graph(), _stimulus(model, steps=0)
    expected = reference(c, stimulus, model)
    prepared = configured.prepare(c, stimulus, model)
    _, state = configured.run_prepared(prepared, stimulus, return_state=True)
    _assert_state(state, prepared, expected.state)


@pytest.mark.parametrize("delay", [0, 18])
def test_scalar_reset_timedarray_is_evaluated_even_without_spikes(delay):
    model = resolve_model({"drive": b.TimedArray([[1., 2.]], dt=.1*b.ms),
                           "eqs": DEFAULT_EQUATIONS+"sample : 1\n", "eq_th": "v>v_th",
                           "eq_rst": "sample=drive(0*ms, 2)", "t_dly": delay*.1*b.ms})
    c = _graph()
    stimulus = _stimulus(model, steps=3)
    stimulus.events[:] = 0
    with pytest.raises(Exception) as cpu_error:
        reference(c, stimulus, model)
    error = cpu_error.value
    while error.__cause__ is not None:
        error = error.__cause__
    assert isinstance(error, IndexError)
    prepared = configured.prepare(c, stimulus, model)
    with pytest.raises(IndexError, match="TimedArray column"):
        configured.run_prepared(prepared, stimulus)


@pytest.mark.parametrize("interval", [.0125, .1, 1.])
def test_half_step_rounding_executes_exactly_on_metal(interval):
    # A dynamic neuron column/time expression exposes all half-integer ties;
    # smaller-than-clock intervals exercise K=1 as well as K>1.
    drive = b.TimedArray(np.arange(64.)+1e-9, dt=interval*b.ms)
    from brian2.input.timedarray import _find_K
    epsilon = float(drive.dt) / _find_K(.0001, float(drive.dt))
    model = resolve_model({"drive": drive, "quantum": epsilon*b.second,
                           "eqs": DEFAULT_EQUATIONS+"sample : 1\n",
                           "eq_th": "t >= 0*ms", "eq_rst": "sample=drive((i+.5)*quantum)",
                           "t_rfc": 0*b.ms})
    c = Connectome(np.arange(257), np.array([], np.int32), np.array([], np.int32),
                   np.array([]), np.array([]))
    stimulus = _stimulus(model, steps=3)
    expected = reference(c, stimulus, model)
    prepared = configured.prepare(c, stimulus, model)
    _, state = configured.run_prepared(prepared, stimulus, return_state=True)
    _assert_state(state, prepared, expected.state)
