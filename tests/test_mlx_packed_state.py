"""Many-state models retain exact Brian behavior with a bounded Metal ABI."""
from state_assertions import assert_spikes_equal

from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
from native_execution import require_metal

from mlx_backend.compiler import CompiledVariable, compile_model
from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
from mlx_backend.models import Connectome, Stimulus
from mlx_backend.state_layout import StateLayout


def test_layout_groups_interleaved_variables_without_narrowing_or_extra_bindings():
    types = [("v", "float64", "ulong"), ("count", "int64", "long"),
             ("flag", "bool", "bool"), ("g", "float64", "ulong"),
             ("small_count", "int32", "int")]
    variables = {name: CompiledVariable(name, dtype, metal, False, False, False, None, True)
                 for name, dtype, metal in types}
    variables["dt"] = CompiledVariable("dt", "float64", "ulong", True, True, True, None, False)
    program = SimpleNamespace(variables=variables)
    layout = StateLayout.from_program(program)
    assert layout.names == ("v", "count", "flag", "g", "small_count")
    assert [(g.dtype, g.names) for g in layout.groups] == [
        ("float64", ("v", "g")), ("int64", ("count",)), ("bool", ("flag",)),
        ("int32", ("small_count",))]
    assert layout.shapes(7) == ((14,), (7,), (7,), (7,))
    loads, stores = layout.source(program, "work_index")
    assert "ulong g = _eon_p_packed_float64[1ul * (NEURONS * B) + gid];" in loads
    assert "long _eon_var_636f756e74 = _eon_p_packed_int64[0ul * (NEURONS * B) + gid];" in loads
    assert "_eon_q_packed_bool[0ul * (NEURONS * B) + work_index] = _eon_var_666c6167;" in stores
    assert "float(" not in loads + stores
    with pytest.raises(OverflowError, match="Packed state"):
        layout.shapes(2**30)


def many_state_model(integer_dtype="int32", delay=18, float_count=12, *, observational=False):
    b = pytest.importorskip("brian2")
    extra = "".join(f"extra{j}:1\ncount{j}:integer\nflag{j}:boolean\n" for j in range(12))
    extra += "".join(f"extra{j}:1\n" for j in range(12, float_count))
    integer_step = 2**40 if integer_dtype == "int64" else 0
    reset = "v=v_rst;g=0*mV"
    reset += ";" + ";".join(
        f"extra{j}+={j+1}/1024 + int(flag{j})/2;count{j}+={integer_step+j+1};flag{j}=not flag{j}"
        for j in range(12))
    reset += "".join(f";extra{j}+={j+1}/1024" for j in range(12, float_count))
    # This valid Brian name would collide with an unreserved packed argument.
    extra = extra.replace("extra0:", "p_packed_float64:")
    reset = reset.replace("extra0+=", "p_packed_float64+=")
    previous = b.prefs.core.default_integer_dtype
    try:
        b.prefs.core.default_integer_dtype = np.dtype(integer_dtype).type
        return resolve_model({"eqs": DEFAULT_EQUATIONS + extra, "eq_rst": reset,
            "eq_th": "v>v_th" if observational else "(v>v_th) or ((t>=.2*ms) and (flag0 or count0>=0))",
            "t_rfc": (2.2 if observational else 0)*b.ms, "t_dly": delay*.1*b.ms})
    finally:
        b.prefs.core.default_integer_dtype = previous


def test_valid_brian_model_requires_more_than_31_bindings_in_old_layout():
    program = compile_model(many_state_model())
    variables = [v for v in program.variables.values() if v.persistent]
    assert len(variables) == 41
    # Original neuron input/output lists bound every variable separately.
    assert 2*len(variables) + 15 > 31
    layout = StateLayout.from_program(program)
    assert len(layout.groups) == 3
    assert 2*len(layout.groups) + 15 <= 31


def test_passive_spike_observers_preserve_proven_autonomous_fixed_point():
    program = compile_model(many_state_model(observational=True))
    assert set(program.equilibrium) == {"v", "g"}
    assert program.refractory_elapsed_threshold is not None
    assert program.variables["count0"].code_name in program.reset_code
    # Observer thresholds still execute when canonical state work is elided.
    dependent_model = many_state_model()
    dependent = compile_model(dependent_model)
    assert "_eon_var_666c616730" in dependent.threshold_code
    assert "_eon_var_636f756e7430" in dependent.threshold_code
    if dependent_model.reference_codegen_target == "cython":
        assert dependent.canonical_custom_lif
        assert set(dependent.equilibrium) == {"v", "g"}
        np.testing.assert_array_equal(
            np.asarray([dependent.equilibrium[name] for name in ("v", "g")], np.float64).view(np.uint64),
            np.asarray([-52*.001, 0.], np.float64).view(np.uint64))
        assert dependent.refractory_elapsed_threshold is not None
    else:
        assert dependent.equilibrium is None
        assert dependent.refractory_elapsed_threshold is None


@pytest.mark.parametrize("extra,threshold,reset", [
    ("da/dt=1/second:1\n", "v>v_th", ()),
    ("a=t/second:1\n", "v>v_th", ()),
    ("a=rand():1 (constant over dt)\n", "v>v_th", ()),
    ("a:1 (shared)\n", "v>v_th", ()),
    ("a:1 (linked)\n", "v>v_th", ()),
    ("a:1\n", "v>v_th+a*mV", ()),
    ("v_th:volt\n", "v>v_th", ()),
    ("a:1\n", "v>v_th", (("rfc", "a*second"),)),
    ("a:1\n", "v>v_th", (("a", "t/second"),)),
    ("a:1\n", "v>v_th", (("a", "rand()"),)),
])
def test_extra_dynamics_or_live_dependencies_cannot_bypass_idle_work(extra, threshold, reset):
    pytest.importorskip("brian2")
    from mlx_backend.compiler import _static_lif
    model = SimpleNamespace(eqs=DEFAULT_EQUATIONS+extra, eq_th=threshold)
    statements = tuple(SimpleNamespace(var=name, expr=expr) for name, expr in reset)
    assert not _static_lif(model, statements)


def test_extra_state_cannot_influence_canonical_voltage_dynamics():
    pytest.importorskip("brian2")
    from mlx_backend.compiler import _static_lif
    eqs = DEFAULT_EQUATIONS.replace("v_0 - v + g", "v_0 - v + g + a*mV") + "a:1\n"
    assert not _static_lif(SimpleNamespace(eqs=eqs, eq_th="v>v_th"), ())


def metal_runtime():
    mx = require_metal('Metal is required')
    from mlx_backend import configured
    return mx, configured


def graph(neurons=3):
    if neurons == 3:
        sources, targets = np.array([0, 0, 1], np.int32), np.array([1, 2, 2], np.int32)
        counts = np.array([400.125, -2.25, 1.125])
    else:
        sources, targets, counts = np.array([0], np.int32), np.array([neurons-1], np.int32), np.array([2.25])
    return Connectome(np.arange(neurons), sources, targets, counts, counts*.275)


def stimulus(model, steps=39, trials=2, targets=(0,)):
    events = np.zeros((trials, steps, len(targets)), np.uint8)
    if targets:
        events[:, [x for x in (0, 1, 7, 18, 19, 32) if x < steps], :] = 1
    return Stimulus(events, targets, (100.,)*len(targets), tuple(7+3*i for i in range(trials)),
                    0, 0, sha256(events.tobytes()).hexdigest(), model.dt_ms)


def assert_state_bits(actual, expected):
    assert actual.keys() == expected.keys()
    for name, right in expected.items():
        left = actual[name]
        assert left.dtype == right.dtype, name
        if right.dtype == np.float64:
            left, right = left.view(np.uint64), right.view(np.uint64)
        np.testing.assert_array_equal(left, right, err_msg=name)


def assert_packed_footprint(state, program, size):
    layout = StateLayout.from_program(program)
    assert len(state.values.buffers) == len(layout.groups) == 3
    assert tuple(buffer.shape for buffer in state.values.buffers) == layout.shapes(size)
    expected_bytes = sum(np.dtype(v.dtype).itemsize * size for v in program.variables.values() if v.persistent)
    assert sum(buffer.nbytes for buffer in state.values.buffers) == expected_bytes


@pytest.mark.metal
@pytest.mark.parametrize("integer_dtype,delay,float_count", [("int32", 0, 12), ("int64", 18, 12),
                                                          ("int32", 18, 48)])
def test_many_mixed_states_match_independent_brian_and_recording_off(integer_dtype, delay, float_count):
    _, configured = metal_runtime()
    from configured_reference import reference
    model = many_state_model(integer_dtype, delay, float_count)
    c, s = graph(), stimulus(model)
    expected = reference(c, s, model)
    p = configured.prepare(c, s, model)
    actual, state = configured.run_prepared(p, s, return_state=True, chunk_steps=19, block_windows=2)
    assert_spikes_equal(actual.spikes, expected.spikes)
    assert actual.spikes.steps.size > 0
    assert_state_bits(configured.host_state(state, p), expected.state)
    assert_packed_footprint(state, p.program, 2*p.neurons)
    silent, silent_state = configured.run_prepared(p, s, record=False, return_state=True, chunk_steps=23)
    assert silent.spikes.steps.size == 0
    assert silent.timings["collection_s"] == 0
    assert_state_bits(configured.host_state(silent_state, p), expected.state)
    assert actual.input_sha256 == silent.input_sha256


@pytest.mark.metal
def test_many_state_partial_continuation_keeps_buffers_bounded_and_accepts_named_mapping(monkeypatch):
    mx, configured = metal_runtime()
    from configured_reference import reference
    model = many_state_model("int64", 18)
    c, s = graph(), stimulus(model, steps=53)
    expected = reference(c, s, model)
    p = configured.prepare(c, s, model)
    state = configured.initial_state(p, 2)
    # Existing callers can continue from an ordinary mapping of named arrays.
    state = replace(state, values=dict(state.values))
    events = mx.array(s.events.reshape(-1))
    state, first = configured.advance(p, state, events, 53, 1)
    configured.evaluate_state(state)
    rows = np.flatnonzero(np.asarray(first) & 1)
    recorded = [(s.trial_indices[row//p.neurons], row%p.neurons, 0) for row in rows]

    def no_repacking(*args, **kwargs):
        raise AssertionError("Continuing packed state must reuse its existing buffers")
    monkeypatch.setattr(mx, "concatenate", no_repacking)
    widths = (7, 1, 11, 3)
    call = 0
    while state.step < 53:
        begin = state.step
        length = min(widths[call % len(widths)], p.window-begin % p.window, 53-begin)
        state, words = configured.advance(p, state, events, 53, length)
        configured.evaluate_state(state)
        raw = np.asarray(words)
        for bit in range(length):
            for row in np.flatnonzero(raw & (1 << bit)):
                recorded.append((s.trial_indices[row//p.neurons], row%p.neurons, begin+bit))
        assert_packed_footprint(state, p.program, 2*p.neurons)
        assert len(state.history) <= model.delay_steps//p.window + 2
        assert len(state.due) <= model.delay_steps//p.window + 2
        call += 1
    assert sorted(recorded, key=lambda row: (row[0], row[2], row[1])) == list(zip(
        expected.spikes.trials, expected.spikes.neurons, expected.spikes.steps))
    assert_state_bits(configured.host_state(state, p), expected.state)


@pytest.mark.metal
def test_many_state_merge_kernel_preserves_mixed_slots_and_payload_bits():
    mx, configured = metal_runtime()
    program = compile_model(many_state_model("int64", 18))
    size = 67
    rng = np.random.default_rng(20261017)
    base, updated = {}, {}
    for variable in program.variables.values():
        if not variable.persistent:
            continue
        dtype = variable.dtype
        for target in (base, updated):
            if dtype == "float64":
                array = rng.integers(0, 2**64, size, dtype=np.uint64)
            elif dtype == "int64":
                array = rng.integers(-(2**63), 2**63-1, size, dtype=np.int64)
            else:
                array = rng.integers(0, 2, size).astype(bool)
            target[variable.name] = array
    active = np.arange(0, size, 3)
    positions = np.full(size, 0xffffffff, np.uint32)
    positions[active] = np.arange(active.size)[::-1]
    base_release = np.arange(size, dtype=np.int64)
    updated_release = np.arange(size, dtype=np.int64) + 1024
    updated_bits = np.arange(size, dtype=np.uint32) * 13
    base_values = configured._pack_values(program, {k: mx.array(v) for k, v in base.items()})
    updated_values = configured._pack_values(program, {k: mx.array(v) for k, v in updated.items()})
    layout = configured._state_layout(program)
    result = configured._merge_idle_kernel(program)(
        inputs=[*base_values.buffers, mx.array(base_release), mx.array(positions),
                *updated_values.buffers, mx.array(updated_release), mx.array(updated_bits),
                mx.array([31], dtype=mx.int64)],
        template=[("SIZE", size)], grid=(size, 1, 1), threadgroup=(64, 1, 1),
        output_shapes=[*layout.shapes(size), (size,), (size,)],
        output_dtypes=[configured._storage_dtype(g.dtype) for g in layout.groups]+[mx.int64, mx.uint32])
    mx.eval(*result)
    values = configured._PackedValues(layout, result[:-2], size)
    expected = {name: raw.copy() for name, raw in base.items()}
    for name in expected:
        expected[name][active] = updated[name][positions[active]]
    inactive = np.setdiff1d(np.arange(size), active)
    expected["not_refractory"][inactive] = 31 >= base_release[inactive]
    assert_state_bits({name: np.asarray(raw) for name, raw in values.items()}, expected)
    release = base_release.copy()
    release[active] = updated_release[positions[active]]
    spikes = np.zeros(size, np.uint32)
    spikes[active] = updated_bits[positions[active]]
    np.testing.assert_array_equal(np.asarray(result[-2]), release)
    np.testing.assert_array_equal(np.asarray(result[-1]), spikes)


@pytest.mark.metal
@pytest.mark.parametrize("delay", [0, 18])
def test_packed_states_timedarray_and_rounding_share_the_native_kernel(delay):
    metal_runtime()
    b = pytest.importorskip("brian2")
    from configured_reference import reference
    from mlx_backend import configured

    base = many_state_model("int64", delay)
    drive = b.TimedArray(np.arange(12).reshape(4, 3)/4 - 1.5, dt=.15*b.ms)

    def resolve(params):
        previous = b.prefs.core.default_integer_dtype
        try:
            b.prefs.core.default_integer_dtype = np.dtype(base.integer_dtype).type
            return resolve_model(params)
        finally:
            b.prefs.core.default_integer_dtype = previous

    model = resolve({"eqs": base.eqs, "eq_th": base.eq_th,
        "eq_rst": base.eq_rst + ";extra1=floor(drive(t,i-3));extra2=ceil(drive(t,i))",
        "drive": drive, "t_rfc": 0*b.ms, "t_dly": delay*.1*b.ms})
    c, s = graph(), stimulus(model)
    expected = reference(c, s, model)
    p = configured.prepare(c, s, model)
    assert p.program.input_arrays and p.program.index_error_name is not None
    actual, state = configured.run_prepared(p, s, return_state=True, chunk_steps=19)
    assert_spikes_equal(actual.spikes, expected.spikes)
    assert_state_bits(configured.host_state(state, p), expected.state)
    assert_packed_footprint(state, p.program, 2*p.neurons)

    broken = resolve({"eqs": base.eqs, "eq_th": base.eq_th,
        "eq_rst": base.eq_rst + ";extra1=drive(t,i+1)", "drive": drive,
        "t_rfc": 0*b.ms, "t_dly": delay*.1*b.ms})
    with pytest.raises(IndexError, match="TimedArray column"):
        configured.run_prepared(configured.prepare(c, s, broken), s, record=False)


@pytest.mark.metal
@pytest.mark.parametrize("integer_dtype", ["int32", "int64"])
def test_passive_counters_survive_spikes_long_idle_continuation_and_no_record(integer_dtype):
    mx, configured = metal_runtime()
    from configured_reference import reference
    model = many_state_model(integer_dtype, observational=True)
    c = graph(67)
    counts = np.array([400.125])
    c = replace(c, counts=counts, weights_mv=counts*.275)
    s = stimulus(model, steps=239, trials=1)
    events_host = np.zeros_like(s.events)
    events_host[0, 0, 0] = 1
    s = replace(s, events=events_host, sha256=sha256(events_host.tobytes()).hexdigest())
    expected = reference(c, s, model)
    p = configured.prepare(c, s, model)
    assert p.compact_idle
    assert expected.spikes.steps.size > 0
    counts = np.bincount(expected.spikes.neurons, minlength=p.neurons)
    assert counts[-1] > 0
    assert expected.spikes.steps.max() < 100
    assert expected.state["v"][0, -1].tobytes() == np.float64(model.v_0).tobytes()
    assert expected.state["g"][0, -1].tobytes() == np.float64(0.).tobytes()
    integer_step = 2**40 if integer_dtype == "int64" else 0
    np.testing.assert_array_equal(expected.state["count0"][0], counts*(integer_step+1))
    whole, whole_state = configured.run_prepared(p, s, return_state=True, chunk_steps=37, block_windows=2)
    assert_spikes_equal(whole.spikes, expected.spikes)
    assert_state_bits(configured.host_state(whole_state, p), expected.state)
    silent, silent_state = configured.run_prepared(p, s, record=False, return_state=True, chunk_steps=41)
    assert not len(silent.spikes.steps)
    assert_state_bits(configured.host_state(silent_state, p), expected.state)
    state = configured.initial_state(p)
    events = mx.array(s.events.reshape(-1))
    recorded, call = [], 0
    widths = (7, 1, 11, 3)
    while state.step < 239:
        begin = state.step
        length = min(widths[call % len(widths)], p.window-begin % p.window, 239-begin)
        state, words = configured.advance(p, state, events, 239, length)
        configured.evaluate_state(state)
        raw = np.asarray(words)
        for bit in range(length):
            for row in np.flatnonzero(raw & (1 << bit)):
                recorded.append((s.trial_indices[0], row, begin+bit))
        assert_packed_footprint(state, p.program, p.neurons)
        call += 1
    assert sorted(recorded, key=lambda row: (row[0], row[2], row[1])) == list(zip(
        expected.spikes.trials, expected.spikes.neurons, expected.spikes.steps))
    assert_state_bits(configured.host_state(state, p), expected.state)
