"""Legacy elapsed-time comparisons against independent CPU Brian and Metal."""
from state_assertions import assert_spikes_equal

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

b = pytest.importorskip("brian2")
from native_execution import require_metal
mx = require_metal('Metal is required', allow_module_level=True)

from mlx_backend import configured
from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
from mlx_backend.models import Connectome, InputPin
from configured_reference import reference
from test_mlx_configured import graph, inputs

pytestmark = pytest.mark.metal


@pytest.fixture(autouse=True)
def legacy(monkeypatch):
    monkeypatch.setitem(b.prefs, "legacy.refractory_timing", True)
    monkeypatch.setitem(b.prefs, "codegen.target", "numpy")
    cache = Path(__file__).resolve().parents[1]/"executions/caches/brian-cython/default"
    cache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setitem(b.prefs, "codegen.runtime.cython.cache_dir", str(cache))


def empty_graph(n=1):
    return Connectome(np.arange(n), np.array([], np.int32), np.array([], np.int32),
                      np.array([], np.float64), np.array([], np.float64))


def assert_raw(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if expected.dtype == np.float64:
        actual, expected = actual.view(np.uint64), expected.view(np.uint64)
    np.testing.assert_array_equal(actual, expected, err_msg=label)


@pytest.mark.parametrize("dt_ms,rfc_ms,cpu_target", [
    (.1, .2, "numpy"), (.13, .26, "numpy"), (.13, .315, "numpy"), (.1, 0., "numpy"),
    (.13, .26, "cython"),
])
def test_unmodified_brian_quantity_duration_matches_every_metal_step(dt_ms, rfc_ms, cpu_target, monkeypatch):
    """Quantity refractory avoids Brian 2.8's broken string-duration branch.

    This oracle uses unmodified Brian code, including its legacy predicate,
    conditional state writes and threshold side effects. Equality does not
    release: the next strictly greater rounded elapsed time does.
    """
    monkeypatch.setitem(b.prefs, "codegen.target", cpu_target)
    if cpu_target == "cython":
        cache = Path(__file__).resolve().parents[1]/"executions/caches/brian-cython/clang-exact"
        cache.mkdir(parents=True, exist_ok=True)
        monkeypatch.setitem(b.prefs, "codegen.runtime.cython.cache_dir", str(cache))
        monkeypatch.setitem(b.prefs, "codegen.cpp.extra_compile_args",
                            ["-w", "-O3", "-fno-fast-math", "-ffp-contract=off", "-std=c++11"])
    model = resolve_model({"t_rfc": rfc_ms*b.ms, "eq_th": "True",
                           "eq_rst": "v=v_rst; g=0*mV", "v_rst": -49*b.mV}, dt_ms=dt_ms)
    clock = b.Clock(dt=model.dt_s*b.second)
    neurons = b.NeuronGroup(1, model.eqs, method="linear", threshold=model.eq_th,
        reset=model.eq_rst, refractory=model.t_rfc*b.second,
        namespace=model.brian_namespace(), clock=clock)
    neurons.v = model.v_0*b.volt
    neurons.g = .03*b.volt
    neurons.rfc = model.t_rfc*b.second
    names = ("v", "g", "rfc", "lastspike", "not_refractory")
    monitor = b.StateMonitor(neurons, names, record=True, when="end")
    spikes = b.SpikeMonitor(neurons)
    steps = 19
    b.Network(neurons, monitor, spikes).run(steps*clock.dt, namespace={})
    s = inputs(model, targets=(), steps=steps, trials=1)
    p = configured.prepare(empty_graph(), s, model)
    assert p.program.refractory_elapsed_threshold is None and not p.compact_idle
    state = configured.initial_state(p)
    values = dict(state.values)
    values["g"] = mx.array(np.array([.03], np.float64).view(np.uint64))
    state = replace(state, values=values)
    fired = []
    for step in range(steps):
        state, bits = configured.advance(p, state, mx.zeros((1,), mx.uint8), steps, 1)
        configured.evaluate_state(state)
        if np.asarray(bits)[0] & 1:
            fired.append(step)
        actual = configured.host_state(state, p)
        for name in names:
            assert_raw(actual[name][0], np.asarray(getattr(monitor, name+"_"))[:, step],
                       f"{name} at step {step}")
    np.testing.assert_array_equal(fired, np.rint(np.asarray(spikes.t_)/model.dt_s).astype(np.int64))


def assert_state(actual, expected, label):
    assert actual.values.keys() == expected.values.keys()
    for name in actual.values:
        assert_raw(actual.values[name], expected.values[name], label+name)
    assert_raw(actual.release, expected.release, label+"release")
    for name in ("history", "due"):
        left, right = getattr(actual, name), getattr(expected, name)
        for key in left.keys() | right.keys():
            zeros = np.zeros(actual.values["v"].size, np.uint32)
            assert_raw(left.get(key, zeros), right.get(key, zeros), f"{label}{name}[{key}]")


@pytest.mark.parametrize("dimensions", [1, 2])
def test_compiled_timedarray_keeps_legacy_release_comparison(dimensions, monkeypatch):
    """Default-optimized Cython and cached TA calls retain strict release."""
    monkeypatch.setitem(b.prefs, "codegen.target", "cython")
    values = np.ones((4, 3)) if dimensions == 2 else np.ones(4)
    arguments = "t, i" if dimensions == 2 else "t"
    model = resolve_model({"drive": b.TimedArray(values, dt=.15*b.ms),
        "eq_th": f"drive({arguments}) > 0", "eq_rst": "v=-49*mV; g=9*mV",
        "t_rfc": .2*b.ms})
    clock = b.Clock(dt=model.dt_s*b.second)
    neurons = b.NeuronGroup(3, model.eqs, method="linear", threshold=model.eq_th,
        reset=model.eq_rst, refractory=model.t_rfc*b.second,
        namespace=model.brian_namespace(), clock=clock)
    neurons.v = model.v_0*b.volt
    neurons.g = .03*b.volt
    neurons.rfc = model.t_rfc*b.second
    names = ("v", "g", "rfc", "lastspike", "not_refractory")
    monitor = b.StateMonitor(neurons, names, record=True, when="end")
    spikes = b.SpikeMonitor(neurons)
    steps = 19
    b.Network(neurons, monitor, spikes).run(steps*clock.dt, namespace={})
    stimulus = inputs(model, targets=(), steps=steps, trials=1)
    prepared = configured.prepare(empty_graph(3), stimulus, model)
    state = configured.initial_state(prepared)
    values = dict(state.values)
    values["g"] = mx.array(np.full(3, .03, np.float64).view(np.uint64))
    state = replace(state, values=values)
    fired = []
    for step in range(steps):
        state, bits = configured.advance(prepared, state, mx.zeros((1,), mx.uint8), steps, 1)
        configured.evaluate_state(state)
        fired.extend((step, int(row)) for row in np.flatnonzero(np.asarray(bits) & 1))
        for name, actual in configured.host_state(state, prepared).items():
            assert_raw(actual[0], np.asarray(getattr(monitor, name+"_"))[:, step],
                       f"compiled legacy TA {name} at {step}")
    expected_steps = np.rint(np.asarray(spikes.t_)/model.dt_s).astype(np.int64)
    assert fired == list(zip(expected_steps, np.asarray(spikes.i[:])))


@pytest.mark.parametrize("delay,guard_v,guard_g", [
    (0, True, True), (1, False, True), (18, True, False), (33, False, False),
    (18, True, True), (33, True, True),
])
def test_delays_guards_duplicate_inputs_reset_and_partial_continuation(delay, guard_v, guard_g):
    lines = DEFAULT_EQUATIONS.splitlines()
    if not guard_v:
        lines[0] = lines[0].replace(" (unless refractory)", "")
    if not guard_g:
        lines[1] = lines[1].replace(" (unless refractory)", "")
    model = resolve_model({"eqs": "\n".join(lines)+"\ncounter : integer\n",
        "t_dly": delay*.13*b.ms, "t_rfc": .315*b.ms,
        "eq_rst": "v=-48*mV; g=9*mV; rfc=3*dt; counter+=1"}, dt_ms=.13)
    s = inputs(model, steps=113, amplitudes=(60.125, -2.75, 72.5))
    c, silenced = graph(), (2,)
    expected = reference(c, s, model, silenced, record_state=True)
    p = configured.prepare(c, s, model, silenced)
    assert not p.compact_idle and p.program.refractory_elapsed_threshold is None
    result, whole = configured.run_prepared(p, s, return_state=True, chunk_steps=47, block_windows=3)
    assert_spikes_equal(result.spikes, expected.spikes)
    for name, value in configured.host_state(whole, p).items():
        assert_raw(value, expected.state[name], "final "+name)
    assert result.input_sha256 == tuple(sha256(row.tobytes()).hexdigest() for row in s.events)
    assert np.any(expected.state["counter"] > 1), "case must exercise repeated spikes"

    state = configured.initial_state(p, 2)
    events = mx.array(s.events.reshape(-1))
    fired, call = [], 0
    widths = (7, 1, 11, 3, 13)
    while state.step < 113:
        begin = state.step
        length = min(widths[call % len(widths)], p.window-begin % p.window, 113-begin)
        state, bits = configured.advance(p, state, events, 113, length)
        configured.evaluate_state(state)
        for bit in range(length):
            for row in np.flatnonzero(np.asarray(bits) & (1 << bit)):
                fired.append((s.trial_indices[row//p.neurons], row%p.neurons, begin+bit))
        for name, value in configured.host_state(state, p).items():
            assert_raw(value, expected.traces[name][:, state.step-1, :], f"step {state.step-1} {name}")
        call += 1
    assert sorted(fired, key=lambda row:(row[0], row[2], row[1])) == list(
        zip(expected.spikes.trials, expected.spikes.neurons, expected.spikes.steps))
    assert_state(state, whole, "partial ")

    quiet, quiet_state = configured.run_prepared(p, s, record=False, return_state=True, chunk_steps=65)
    assert quiet.spikes.steps.size == 0 and quiet.timings["collection_s"] == 0
    assert_state(quiet_state, whole, "no recording ")


@pytest.mark.parametrize("steps,rfc_s", [(0, .0022), (5, 10001.), (37, 0.)])
def test_empty_inputs_initial_sentinel_zero_duration_and_zero_refractory(steps, rfc_s):
    model = resolve_model({"t_rfc": rfc_s*b.second, "eq_th": "True"}, dt_ms=.13)
    c, s = empty_graph(), inputs(model, targets=(), steps=steps, trials=1)
    expected = reference(c, s, model)
    p = configured.prepare(c, s, model)
    result, state = configured.run_prepared(p, s, return_state=True)
    np.testing.assert_array_equal(result.spikes.steps, expected.spikes.steps)
    for name, value in configured.host_state(state, p).items():
        assert_raw(value, expected.state[name], name)
    if rfc_s > 10000:
        assert expected.spikes.steps.size == 0
        assert expected.state["lastspike"][0, 0] == -10000
        assert not expected.state["not_refractory"][0, 0]


@pytest.mark.parametrize("reset", [
    "v=v_rst; g=9*mV; rfc=0*ms",
    "v=v_rst; g=9*mV; lastspike=t+dt",
    "v=v_rst; g=9*mV; not_refractory=True",
])
def test_reset_can_override_threshold_refractory_side_effects(reset):
    model = resolve_model({"eq_th": "True", "eq_rst": reset, "t_rfc": .26*b.ms}, dt_ms=.13)
    s = inputs(model, targets=(), steps=27, trials=1)
    c = empty_graph()
    expected = reference(c, s, model, record_state=True)
    p = configured.prepare(c, s, model)
    state = configured.initial_state(p)
    fired = []
    for step in range(27):
        state, bits = configured.advance(p, state, mx.zeros((1,), mx.uint8), 27, 1)
        configured.evaluate_state(state)
        if np.asarray(bits)[0] & 1:
            fired.append(step)
        for name, value in configured.host_state(state, p).items():
            assert_raw(value, expected.traces[name][:, step, :], f"reset {name} step {step}")
    np.testing.assert_array_equal(fired, expected.spikes.steps)


def test_preference_snapshot_layout_cache_and_public_adapter(monkeypatch):
    import mlx_engine

    legacy = resolve_model()
    monkeypatch.setitem(b.prefs, "legacy.refractory_timing", False)
    modern = resolve_model()
    assert modern.signature != legacy.signature
    assert not legacy.state_specialization
    c = replace(graph(), pin=InputPin("legacy-test-neurons", "legacy-test-edges", 3, 6))
    s = inputs(legacy, steps=81)
    p = configured.prepare(c, s, legacy)
    modern_p = configured.prepare(c, s, modern)
    assert configured.prepare(c, s, legacy) is p
    assert p is not modern_p
    assert p.program.refractory_elapsed_threshold is None
    assert modern_p.program.refractory_elapsed_threshold is not None
    expected = reference(c, s, legacy)
    actual = mlx_engine.execute(c, s, (), model=legacy)
    assert_spikes_equal(actual.spikes, expected.spikes)
    assert not b.prefs.legacy.refractory_timing


@pytest.mark.parametrize("begin", [2**40, 2**50, 2**53, 2**55])
@pytest.mark.parametrize("rfc_s", [0., .00026])
def test_large_absolute_clock_and_partial_continuation_against_unmodified_brian(begin, rfc_s):
    model = resolve_model({"eqs": DEFAULT_EQUATIONS+"counter : integer\n",
        "eq_th": "True", "eq_rst": "v=-49*mV; g=0*mV; counter+=1",
        "t_rfc": rfc_s*b.second}, dt_ms=.13)
    clock = b.Clock(dt=model.dt_s*b.second)
    neurons = b.NeuronGroup(1, model.eqs, method="linear", threshold=model.eq_th,
        reset=model.eq_rst, refractory=model.t_rfc*b.second,
        namespace=model.brian_namespace(), clock=clock)
    neurons.v = model.v_0*b.volt
    neurons.g = .03*b.volt
    neurons.rfc = model.t_rfc*b.second
    neurons.lastspike = np.float64(begin)*model.dt_s*b.second
    neurons.not_refractory = False
    names = ("v", "g", "rfc", "counter", "lastspike", "not_refractory")
    monitor = b.StateMonitor(neurons, names, record=True, when="end")
    network = b.Network(neurons, monitor)
    network.t_ = np.float64(begin)*model.dt_s
    # The CPU clock independently rounds its start/end. Read its actual range
    # so the test covers that accepted interval without assuming a run length.
    network.run(32*clock.dt, namespace={})
    count = len(monitor.t)
    end = int(clock.variables["timestep"].get_value()[0])
    start = end-count
    assert count >= 24 and abs(start-begin) < 16
    expected_times = np.asarray(monitor.t_)
    s = inputs(model, targets=(), steps=count, trials=1)
    p = configured.prepare(empty_graph(), s, model)
    state = configured.initial_state(p)
    values = dict(state.values)
    values["g"] = mx.array(np.array([.03], np.float64).view(np.uint64))
    values["lastspike"] = mx.array(np.array([np.float64(begin)*model.dt_s]).view(np.uint64))
    values["not_refractory"] = mx.array([False])
    state = replace(state, step=start, values=values)
    native_fired = []
    for offset in range(count):
        state, bits = configured.advance(p, state, mx.zeros((1,), mx.uint8), end, 1,
            event_begin=start, event_steps=count)
        configured.evaluate_state(state)
        if np.asarray(bits)[0] & 1:
            native_fired.append(offset)
        for name, value in configured.host_state(state, p).items():
            assert_raw(value[0], np.asarray(getattr(monitor, name+"_"))[:, offset],
                       f"large clock {start+offset} {name}")
    # Repeated floating clock times can label different integer steps equally;
    # an independent reset counter identifies the actual CPU firing steps.
    counts = np.asarray(monitor.counter_)[0]
    cpu_fired = np.flatnonzero(np.diff(np.r_[0, counts]))
    np.testing.assert_array_equal(native_fired, cpu_fired)
    assert_raw(configured._clock_times(start, end, model.dt_s), expected_times.view(np.uint64), "clock bits")


def test_active_guarded_state_stage_matches_independent_brian():
    """Observe guarded dynamics before a forced threshold/reset overwrites it."""
    from mlx_backend.binary64 import MSL_HEADER, literal

    model = resolve_model({"t_rfc": .26*b.ms, "eq_th": "True",
                           "eq_rst": "v=-49*mV; g=9*mV"}, dt_ms=.13)
    clock = b.Clock(dt=model.dt_s*b.second)
    neurons = b.NeuronGroup(3, model.eqs, method="linear", threshold=model.eq_th,
        reset=model.eq_rst, refractory=model.t_rfc*b.second,
        namespace=model.brian_namespace(), clock=clock)
    neurons.v = model.v_0*b.volt
    neurons.g = 0
    neurons.rfc = model.t_rfc*b.second
    names = ("v", "g", "rfc", "lastspike", "not_refractory")
    groups = b.StateMonitor(neurons, names, record=True, when="groups", order=1)
    ends = b.StateMonitor(neurons, names, record=True, when="end")
    spikes = b.SpikeMonitor(neurons)
    steps = 64
    b.Network(neurons, groups, ends, spikes).run(steps*clock.dt, namespace={})
    s = inputs(model, targets=(), steps=steps, trials=1)
    p = configured.prepare(empty_graph(3), s, model)
    variables, loads, stores = configured._variables_source(p.program)
    layout = configured._state_layout(p.program)
    stage = mx.fast.metal_kernel(name="legacy_active_guard_stage",
        input_names=["_eon_p_"+group.buffer_name for group in layout.groups]+["clock_time"],
        output_names=["_eon_q_"+group.buffer_name for group in layout.groups],
        header=MSL_HEADER, compile_options={"math_mode": "safe"}, source=f"""
        uint gid = thread_position_in_grid.x;
        if(gid >= 3) return;
        const int i = int(gid);
        const long NEURONS = 3, B = 1, N = 3, t_in_timesteps = 0;
        const ulong dt = {literal(model.dt_s)};
        const ulong t = clock_time[0];
        {loads}
        {p.program.scalar_code}
        {p.program.state_code}
        {stores}
        """)
    state = configured.initial_state(p)
    fired, updated, frozen = [], 0, 0
    for step in range(steps):
        before = configured.host_state(state, p)
        packed = configured._pack_values(p.program, state.values)
        outputs = stage(inputs=[*packed.buffers,
            mx.array(np.array([np.float64(step)*model.dt_s]).view(np.uint64))],
            grid=(3, 1, 1), threadgroup=(3, 1, 1),
            output_shapes=layout.shapes(3),
            output_dtypes=[configured._storage_dtype(group.dtype) for group in layout.groups])
        mx.eval(*outputs)
        stage_values = configured._PackedValues(layout, outputs, 3)
        for variable in variables:
            assert_raw(np.asarray(stage_values[variable.name])[None, :],
                np.asarray(getattr(groups, variable.name+"_"))[:, step][None, :],
                f"active groups {variable.name} step {step}")
        if step:
            g = np.asarray(stage_values["g"]).view(np.float64)
            if bool(np.asarray(groups.not_refractory_)[:, step][0]):
                assert np.all(g < before["g"][0])
                updated += 1
            else:
                assert_raw(g, before["g"][0], f"frozen g step {step}")
                frozen += 1
        state, bits = configured.advance(p, state, mx.zeros((1,), mx.uint8), steps, 1)
        configured.evaluate_state(state)
        if np.asarray(bits)[0] & 1:
            fired.append(step)
        for name, value in configured.host_state(state, p).items():
            assert_raw(value, np.asarray(getattr(ends, name+"_"))[:, step][None, :],
                       f"active end {name} step {step}")
    np.testing.assert_array_equal(fired, np.rint(np.asarray(spikes.t_)[np.asarray(spikes.i)==0]/model.dt_s).astype(np.int64))
    assert len(fired) >= 10 and updated >= 10 and frozen >= 10
