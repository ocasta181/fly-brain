"""Native configurable execution against an independent Brian CPU network."""
from state_assertions import assert_spikes_equal

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from native_execution import require_metal
mx = require_metal('Metal is required', allow_module_level=True)
b = pytest.importorskip("brian2")

from configured_reference import reference
from mlx_backend import configured
from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
from mlx_backend.models import Connectome, Stimulus

pytestmark = pytest.mark.metal


def graph():
    counts = np.array([400.125, -99.875, 200.25, 10.125, -3.125, 4.25])
    return Connectome(np.array(["left", "middle", "right"]),
        np.array([2, 0, 0, 1, 2, 0], np.int32),
        np.array([1, 1, 2, 2, 1, 1], np.int32), counts, counts * .275)


def inputs(model, *, targets=(0, 0, 1), steps=129, trials=2, amplitudes=None):
    events = np.zeros((trials, steps, len(targets)), np.uint8)
    if targets and steps:
        selected = [step for step in (0, 1, 6, 16, 17, 18, 31, 32, 33, 63, 64, 65, 97) if step < steps]
        events[:, selected, :] = 1
        if trials > 1:
            events[1, selected[::2], 0] = 0
    return Stimulus(events, targets, (100.,)*len(targets), tuple(5+2*i for i in range(trials)),
                    0, 0, sha256(events.tobytes()).hexdigest(), model.dt_ms, amplitudes)


def assert_matches_cpu(c, s, model, silenced=(), *, chunk_steps=79, block_windows=3):
    expected = reference(c, s, model, silenced)
    prepared = configured.prepare(c, s, model, silenced)
    actual, state = configured.run_prepared(prepared, s, return_state=True,
        chunk_steps=chunk_steps, block_windows=block_windows)
    assert_spikes_equal(actual.spikes, expected.spikes)
    # Software binary64 and preserved expression/edge order must agree exactly;
    # these requirements are fixed before observing native results.
    states = configured.host_state(state, prepared)
    for name, values in expected.state.items():
        np.testing.assert_array_equal(states[name], values, err_msg=f"final {name}")
    assert actual.input_sha256 == tuple(sha256(row.tobytes()).hexdigest() for row in s.events)
    return actual, state, prepared


@pytest.mark.parametrize("delay", [0, 1, 7, 18, 32, 33, 65])
def test_delay_fractional_weights_and_duplicate_inputs_match_cpu(delay):
    model = resolve_model({"t_dly": delay * .1 * b.ms})
    assert_matches_cpu(graph(), inputs(model), model)


@pytest.mark.parametrize("overrides,dt", [
    ({"v_0": -50*b.mV, "v_rst": -54*b.mV, "v_th": -44*b.mV}, .2),
    ({"tau": 7*b.ms, "t_mbr": 24*b.ms, "t_rfc": 3.15*b.ms}, .13),
    ({"w_syn": .387*b.mV, "f_poi": 150, "t_dly": 5.35*b.ms}, .2),
])
def test_changed_model_constants_timestep_and_amplitudes(overrides, dt):
    model = resolve_model(overrides, dt_ms=dt)
    assert_matches_cpu(graph(), inputs(model, amplitudes=(60.125, -2.75, 72.5)),
                       model, silenced=(2,), chunk_steps=17)


def test_extra_state_binary64_sensitive_threshold_and_reset():
    model = resolve_model({"eqs": DEFAULT_EQUATIONS + "a : 1\n",
        "eq_th": "(t < dt) or (a > 1)",
        "eq_rst": "v=v_rst; g=0*mV; a=1+epsilon", "epsilon": 1e-9,
        "t_rfc": 0*b.ms})
    s = inputs(model, targets=(), steps=37, trials=1)
    actual, _, _ = assert_matches_cpu(graph(), s, model)
    assert actual.spikes.steps.size == 37*3


def test_no_recording_and_staging_preserve_state_and_input_identity():
    model = resolve_model({"tau": 7*b.ms, "t_dly": 3.3*b.ms})
    c, s = graph(), inputs(model)
    _, expected_state, prepared = assert_matches_cpu(c, s, model, chunk_steps=70)
    actual, state = configured.run_prepared(prepared, s, record=False, return_state=True, chunk_steps=33)
    assert actual.spikes.steps.size == 0
    assert actual.timings["collection_s"] == 0
    for name in state.values:
        np.testing.assert_array_equal(np.asarray(state.values[name]), np.asarray(expected_state.values[name]))
    np.testing.assert_array_equal(np.asarray(state.release), np.asarray(expected_state.release))


def test_zero_horizon_preserves_initial_state_without_events():
    model = resolve_model({})
    assert_matches_cpu(graph(), inputs(model, steps=0), model)


def test_public_adapter_executes_resolved_model_without_ignoring_constants():
    import mlx_engine

    model = resolve_model({"t_dly": 3.35*b.ms, "w_syn": .387*b.mV}, dt_ms=.2)
    c, s = graph(), inputs(model)
    expected = reference(c, s, model)
    result = mlx_engine.execute(c, s, (), model=model)
    assert_spikes_equal(result.spikes, expected.spikes)


def test_typed_extra_states_and_reset_side_effects_execute_on_metal():
    model = resolve_model({"eqs": DEFAULT_EQUATIONS + "counter : integer\nflag : boolean\n",
        "eq_rst": "v=v_rst; g=0*mV; counter+=1; flag=not flag",
        "eq_th": "(v > v_th) or (flag and (counter > 2))"})
    assert_matches_cpu(graph(), inputs(model, steps=81), model)


def test_unconditional_voltage_and_current_writes_preserve_refractory_semantics():
    model = resolve_model({"eqs": DEFAULT_EQUATIONS.replace(" (unless refractory)", ""),
                           "t_dly": 0*b.ms, "t_rfc": 3.5*b.ms})
    assert_matches_cpu(graph(), inputs(model, steps=81), model)


def test_prepared_execution_rejects_a_changed_input_amplitude():
    model = resolve_model({})
    s = inputs(model)
    prepared = configured.prepare(graph(), s, model)
    changed = replace(s, amplitudes_mv=(1., 2., 3.))
    with pytest.raises(ValueError, match="canonical stimulus"):
        configured.run_prepared(prepared, changed)


@pytest.mark.parametrize("targets", [(), (0,)])
def test_compacted_idle_and_active_neurons_match_dense_execution_and_cpu(targets):
    model = resolve_model({})
    c = Connectome(np.arange(67), np.array([0, 31], np.int32),
        np.array([31, 66], np.int32), np.array([400.125, -2.25]),
        np.array([400.125, -2.25])*.275)
    s = inputs(model, targets=targets, steps=83)
    result, state, p = assert_matches_cpu(c, s, model, chunk_steps=41)
    assert p.compact_idle
    dense, dense_state = configured.run_prepared(replace(p, compact_idle=False), s,
        return_state=True, chunk_steps=41)
    assert_spikes_equal(result.spikes, dense.spikes)
    for name, value in state.values.items():
        np.testing.assert_array_equal(np.asarray(value), np.asarray(dense_state.values[name]))
    np.testing.assert_array_equal(np.asarray(state.release), np.asarray(dense_state.release))
    for name in ("history", "due"):
        actual, expected = getattr(state, name), getattr(dense_state, name)
        assert actual.keys() == expected.keys()
        for key in actual:
            np.testing.assert_array_equal(np.asarray(actual[key]), np.asarray(expected[key]))


@pytest.mark.parametrize("delay", [18, 33])
@pytest.mark.parametrize("arrivals", [3, 4, 5, 7])
def test_cached_arrivals_and_overflow_preserve_binary64_addition_order(delay, arrivals):
    counts = np.array([1e16, 1., -1e16, 3., -7., 11., .125])[:arrivals]
    order = np.arange(arrivals)[::-1]
    c = Connectome(np.arange(10), order.astype(np.int32),
        np.full(arrivals, 9, np.int32), counts[order], counts[order]*.275)
    model = resolve_model({"w_syn": 1*b.volt, "t_dly": delay*.1*b.ms})
    s = inputs(model, targets=tuple(range(arrivals)), steps=delay+2, trials=1)
    _, state, p = assert_matches_cpu(c, s, model)
    expected_sum = np.float64(0)
    for weight in counts:
        expected_sum += weight
    assert configured.host_state(state, p)["g"][0, 9] == expected_sum


def test_suprathreshold_rest_is_never_classified_as_idle():
    model = resolve_model({"v_th": -60*b.mV})
    _, _, p = assert_matches_cpu(graph(), inputs(model, targets=(), steps=83), model)
    assert not p.compact_idle


@pytest.mark.parametrize("delay", [0, 1, 18, 33, 65])
def test_partial_window_continuation_preserves_spikes_state_and_pending_arrivals(delay):
    model = resolve_model({"t_dly": delay*.1*b.ms})
    c, s = graph(), inputs(model, steps=131)
    expected, whole, p = assert_matches_cpu(c, s, model)
    state = configured.initial_state(p, 2)
    device_events = mx.array(s.events.reshape(-1))
    recorded = []
    widths = (7, 1, 11, 3, 13)
    call = 0
    while state.step < 131:
        length = min(widths[call % len(widths)], p.window-state.step % p.window, 131-state.step)
        begin = state.step
        state, bits = configured.advance(p, state, device_events, 131, length)
        configured.evaluate_state(state)
        words = np.asarray(bits)
        for bit in range(length):
            rows = np.flatnonzero(words & (1 << bit))
            recorded.extend((s.trial_indices[row//p.neurons], row%p.neurons, begin+bit) for row in rows)
        call += 1
    assert sorted(recorded, key=lambda row:(row[0],row[2],row[1])) == list(
        zip(expected.spikes.trials, expected.spikes.neurons, expected.spikes.steps))
    for name in state.values:
        np.testing.assert_array_equal(np.asarray(state.values[name]), np.asarray(whole.values[name]))
    np.testing.assert_array_equal(np.asarray(state.release), np.asarray(whole.release))
    for name in ("history", "due"):
        left, right = getattr(state, name), getattr(whole, name)
        for key in left.keys() | right.keys():
            zeros = np.zeros(p.neurons*2, np.uint32)
            np.testing.assert_array_equal(np.asarray(left.get(key, zeros)), np.asarray(right.get(key, zeros)))
