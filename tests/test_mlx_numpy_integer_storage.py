"""Independent NumPy/Brian trajectories through packed native integer state."""
from native_execution import forbid_call, require_metal
from dataclasses import replace

import numpy as np
import pytest


@pytest.mark.metal
@pytest.mark.parametrize('dtype', ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'])
def test_numpy_packed_integer_boundaries_and_continuation(monkeypatch, dtype, reset_kind='arithmetic'):
    import brian2 as b
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome, Stimulus
    mx = require_metal('Metal required')
    monkeypatch.setitem(b.prefs, 'codegen.target', 'numpy')
    monkeypatch.setitem(b.prefs, 'core.default_float_dtype', np.float64)
    monkeypatch.setitem(b.prefs, 'core.default_integer_dtype', np.dtype(dtype).type)
    monkeypatch.setitem(b.prefs, 'codegen.loop_invariant_optimisations', False)
    n, trials, steps = 12, 2, 4
    equations = 'v:volt\ng:volt\nrfc:second\nx:integer\ny:integer\nz:integer\nproduct:integer\nnegative:boolean\nflag:boolean\nf:1\n'
    reset = 'y = x+x; z = y-x; x = x+1; product=x*x; negative=(x+x)<0; f = x+i'
    if reset_kind == 'divmod':
        reset = 'y=x//product;z=x%product;x=x+1;negative=y<0;f=y'
    elif reset_kind == 'negative-scalar-divmod':
        reset = 'y=x//-3;z=x%-3;x=x+1;negative=y<0;f=y'
    elif reset_kind == 'power':
        reset = 'y=x**product;z=y-x;x=x+1;negative=y<0;f=y'
    model = resolve_model({'eqs': equations, 'eq_th': 'flag', 'eq_rst': reset,
                           't_rfc': 0*b.ms})
    limits = np.iinfo(dtype)
    integers = np.resize(np.array([limits.min, limits.max, 0, 1, limits.max-1], dtype), (trials, n))
    divisors = np.resize(np.array([-1,0,-3,3,limits.min] if np.dtype(dtype).kind=='i'
        else [limits.max,0,3,1,limits.max-1],dtype), (trials,n))
    if reset_kind == 'power':
        divisors = np.resize(np.array([0,1,2,3,7],dtype),(trials,n))
    flags = np.arange(trials*n).reshape(trials, n) % 3 != 1
    names = ('v', 'g', 'rfc', 'lastspike', 'not_refractory', 'x', 'y', 'z', 'product', 'negative', 'flag', 'f')
    expected = {name: [] for name in names}
    expected_spikes = []
    for trial in range(trials):
        clock = b.Clock(dt=model.dt_s*b.second)
        group = b.NeuronGroup(n, equations, threshold=model.eq_th, reset=reset,
            refractory='rfc', method='linear', clock=clock, namespace=model.brian_namespace())
        group.v = model.v_0*b.volt
        group.g = 0*b.volt
        group.rfc = 0*b.second
        group.x = integers[trial]
        if reset_kind in ('divmod','power'):
            group.product = divisors[trial]
        group.flag = flags[trial]
        states = b.StateMonitor(group, names, record=True, when='end')
        spikes = b.SpikeMonitor(group)
        b.Network(group, states, spikes).run(steps*clock.dt)
        for name in names:
            expected[name].append(np.asarray(getattr(states, name)).T.copy())
        expected_spikes.append((np.asarray(spikes.i[:]), np.rint(np.asarray(spikes.t[:])/model.dt_s).astype(int)))
    expected = {name: np.stack(values) for name, values in expected.items()}
    connectome = Connectome(np.arange(n), np.array([], np.int32), np.array([], np.int32),
                            np.array([], float), np.array([], float))
    stimulus = Stimulus(np.zeros((trials, steps, 0), np.uint8), (), (),
                        (2**40+3, 2**40+9), 0, 0, 'integer-storage', model.dt_ms)
    forbid_call(b.Network, 'run', 'Native integer execution used a CPU timestep', monkeypatch)
    # Later global changes must not change the saved preparation contract.
    monkeypatch.setitem(b.prefs, 'core.default_integer_dtype', np.int32)
    monkeypatch.setitem(b.prefs, 'codegen.loop_invariant_optimisations', True)
    prepared = configured.prepare(connectome, stimulus, model)
    assert prepared.model.integer_policy.target == 'numpy'
    assert prepared.model.integer_policy.vector_length == n
    assert prepared.model.integer_policy.trials == trials
    initial = configured.initial_state(prepared, trials)
    initial = replace(initial, values={**initial.values,
        'x': mx.array(integers.reshape(-1)), 'flag': mx.array(flags.reshape(-1))})
    if reset_kind in ('divmod','power'):
        initial = replace(initial, values={**initial.values,'product':mx.array(divisors.reshape(-1))})
    def assert_state(state):
        host = configured.host_state(state, prepared)
        for name in names:
            actual, wanted = host[name], expected[name][:, state.step-1]
            assert actual.dtype == wanted.dtype, (dtype, name)
            if actual.dtype.kind == 'f':
                actual, wanted = actual.view(np.uint64), wanted.view(np.uint64)
            np.testing.assert_array_equal(actual, wanted, err_msg=f'{dtype}/{name}/step{state.step}')
    state = initial
    for length in (1, 2, 1):
        state, words = configured.advance(prepared, state, mx.array(stimulus.events.reshape(-1)), steps, length)
        configured.evaluate_state(state)
        assert_state(state)
        np.testing.assert_array_equal(np.asarray(words).reshape(trials, n), flags*((1 << length)-1))
    uninterrupted, _ = configured.advance(prepared, initial, mx.array(stimulus.events.reshape(-1)), steps, steps)
    configured.evaluate_state(uninterrupted)
    assert_state(uninterrupted)
    monkeypatch.setattr(configured, 'initial_state', lambda *args, **kwargs: initial)
    for recording in (False, True):
        result, final = configured.run_prepared(prepared, stimulus, return_state=True, record=recording)
        assert_state(final)
        if not recording:
            assert not result.spikes.steps.size
        else:
            for trial, identity in enumerate(stimulus.trial_indices):
                selected = result.spikes.trials == identity
                np.testing.assert_array_equal(result.spikes.neurons[selected], expected_spikes[trial][0])
                np.testing.assert_array_equal(result.spikes.steps[selected], expected_spikes[trial][1])


@pytest.mark.metal
@pytest.mark.parametrize('dtype', ['int8','uint8','int16','uint16','int32','uint32','int64','uint64'])
def test_numpy_packed_integer_divmod_boundaries_and_continuation(monkeypatch,dtype):
    test_numpy_packed_integer_boundaries_and_continuation(monkeypatch,dtype,reset_kind='divmod')


@pytest.mark.metal
@pytest.mark.parametrize('dtype',['uint32','uint64'])
def test_numpy_unsigned_divmod_promotes_negative_scalar_before_store(monkeypatch,dtype):
    test_numpy_packed_integer_boundaries_and_continuation(monkeypatch,dtype,
        reset_kind='negative-scalar-divmod')


@pytest.mark.metal
@pytest.mark.parametrize('dtype',['int8','uint8','int16','uint16','int32','uint32','int64','uint64'])
def test_numpy_packed_integer_power_boundaries_and_continuation(monkeypatch,dtype):
    test_numpy_packed_integer_boundaries_and_continuation(monkeypatch,dtype,reset_kind='power')


def test_integer_target_identity_and_frozen_typed_optimization(monkeypatch):
    import brian2 as b
    from mlx_backend.compiler import compile_model
    from mlx_backend.model import resolve_model
    monkeypatch.setitem(b.prefs, 'codegen.target', 'numpy')
    monkeypatch.setitem(b.prefs, 'core.default_integer_dtype', np.int64)
    monkeypatch.setitem(b.prefs, 'codegen.loop_invariant_optimisations', False)
    params = {'eqs': 'v:volt\ng:volt\nrfc:second\nx:integer\ny:integer',
              'eq_th': 'True', 'eq_rst': 'y = x+x; x = x+1'}
    model = resolve_model(params)
    numpy_source = compile_model(model)
    cython_model = resolve_model(params, cpu_target='cython')
    assert cython_model.signature != model.signature
    monkeypatch.setitem(b.prefs, 'codegen.loop_invariant_optimisations', True)
    monkeypatch.setitem(b.prefs, 'core.default_float_dtype', np.float32)
    monkeypatch.setitem(b.prefs, 'core.default_integer_dtype', np.int32)
    other = compile_model(model)
    assert other.reset_code == numpy_source.reset_code
    assert other.state_code == numpy_source.state_code
    assert model.integer_policy.loop_invariant_optimisations is False


def test_integer_compiler_flags_and_requested_target_are_frozen(monkeypatch):
    import brian2 as b
    from mlx_backend.model import resolve_model
    monkeypatch.setitem(b.prefs, 'codegen.target', 'cython')
    monkeypatch.setitem(b.prefs, 'codegen.cpp.extra_compile_args', ['-O0', '-fwrapv'])
    params = {'eqs': 'v:volt\ng:volt\nrfc:second\nx:integer', 'eq_rst': 'x += 1'}
    model = resolve_model(params)
    assert model.integer_policy.requested_target == 'cython'
    assert model.integer_policy.flags == model.reference_compile_args == ('-O0', '-fwrapv')
    assert model.integer_policy.wraps_signed
    monkeypatch.setitem(b.prefs, 'codegen.cpp.extra_compile_args', ['-O3', '-fno-wrapv'])
    other = resolve_model(params)
    assert other.signature != model.signature
    assert not other.integer_policy.wraps_signed
    assert model.integer_policy.flags == ('-O0', '-fwrapv')
