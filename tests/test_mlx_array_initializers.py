"""Original Eon initializer assignments, with exact independent CPU state."""
from native_execution import forbid_cpu_steps as guard_cpu_steps

import numpy as np
import pytest

from original_api_reference import (
    assert_rng_state_equal, assert_snapshot_equal, build_original_api_fixture,
    capture_rng_state, cpu_runtime, original_runtime,
)

pytestmark = pytest.mark.metal


def _parameters(case):
    import brian2 as b
    if case == 'refractory':
        return {'t_rfc': np.array([.1, .2, .3, .4])*b.ms, 'eq_th': 't>=0*ms'}
    if case == 'voltage':
        return {'v_0': np.array([-44., -42., -48., -41.])*b.mV,
                'eqs': 'v:volt\ng:volt\nrfc:second'}
    if case in ('weights', 'self_local'):
        return {'w_syn': np.arange(1, 8)*.1*b.mV, 't_dly': 0*b.ms,
                'eq_th': 't>=0*ms', 't_rfc': np.array([.1,.2,.3,.4])*b.ms}
    if case == 'broadcast':
        return {'v_0': np.array([-44.])*b.mV, 't_rfc': np.array([.2])*b.ms,
                'w_syn': np.array([.3])*b.mV, 't_dly': np.array([.1])*b.ms,
                'eqs': 'v:volt\ng:volt\nrfc:second'}
    if case in ('dynamics', 'dynamics_scalar'):
        vector = case == 'dynamics'
        return {'v_0': (np.array([-44.,-45.,-46.,-47.]) if vector else -44.)*b.mV,
                't_rfc': (np.array([.1,.2,.3,.4]) if vector else .1)*b.ms,
                'eqs': 'dv/dt=(-52*mV-v+g)/t_mbr:volt\ndg/dt=-g/tau:volt\nrfc:second',
                'eq_th': 'v>v_th or t<dt', 'eq_rst': 'v=v_rst;g=30*mV'}
    if case == 'infinite':
        return {'v_0': np.array([-np.inf, -42., np.inf, -41.])*b.mV,
                't_rfc': np.array([np.inf, .2, -np.inf, .4])*b.ms,
                'eqs': 'v:volt\ng:volt\nrfc:second', 'eq_th': 't>=0*ms'}
    if case == 'random':
        return {'v_0': np.array([-44., -42., -48., -41.])*b.mV,
                't_rfc': np.array([.1,.2,.3,.4])*b.ms,
                'eqs': 'dv/dt=(-v+u*mV)/ms:volt\ng:volt\nrfc:second\nu=rand():1(shared,constant over dt)',
                'eq_th': 'rand()<.4', 'eq_rst': 'v=v_rst+randn()*mV;g=rand()*mV'}
    if case == 'initial_random':
        return {'v_0': '(-45+randn())*mV', 't_rfc': 'rand()*.4*ms',
                'eqs': 'v:volt\ng:volt\nrfc:second',
                'eq_th': 'rand()<.4', 'eq_rst': 'v=v_rst+randn()*mV;g=rand()*mV'}
    if case == 'string':
        return {'v_0': '(-44-i)*mV', 't_rfc': '(i+1)*.1*ms',
                'eqs': 'v:volt\ng:volt\nrfc:second'}
    raise AssertionError(case)


def _forbid_cpu_steps(fixture):
    guard_cpu_steps(fixture.network, None, 'CPU timestep executed on native path')


@pytest.mark.parametrize('target,case,dtype', [
    ('numpy', 'refractory', np.float64), ('cython', 'refractory', np.float64),
    ('numpy', 'voltage', np.float64), ('cython', 'voltage', np.float64),
    ('numpy', 'weights', np.float64), ('cython', 'weights', np.float64),
    ('numpy', 'self_local', np.float64), ('cython', 'self_local', np.float64),
    ('numpy', 'broadcast', np.float32), ('cython', 'broadcast', np.float32),
    ('numpy', 'string', np.float64), ('cython', 'string', np.float64),
    ('numpy', 'random', np.float64), ('cython', 'random', np.float64),
    ('numpy', 'random', np.float32), ('cython', 'random', np.float32),
    ('numpy', 'infinite', np.float64), ('cython', 'infinite', np.float64),
    ('numpy', 'dynamics', np.float64), ('cython', 'dynamics', np.float64),
    ('cython', 'dynamics_scalar', np.float64),
    ('numpy', 'initial_random', np.float64), ('cython', 'initial_random', np.float64),
])
def test_original_array_initializers(tmp_path, monkeypatch, target, case, dtype):
    import brian2 as b
    from mlx_backend.brian_device import original_api
    local_dispatch = []
    if case == 'self_local':
        import original_api_reference
        from mlx_backend import configured

        prepare = configured.prepare

        def observed_prepare(*args, **kwargs):
            prepared = prepare(*args, **kwargs)
            local_dispatch.append(prepared.local_synapses)
            return prepared

        monkeypatch.setattr(original_api_reference, "write_tiny_dataset", original_api_reference.synthetic_dataset(4, labelled=False, neuron_file="neurons.csv",
            edges=((2, 2, 1.), (0, 0, 2.), (0, 0, -1.), (3, 3, .5),
                   (1, 1, 0.), (2, 2, -.25), (3, 3, -0.))))
        monkeypatch.setattr(configured, 'prepare', observed_prepare)
    expected = []
    for native in (False, True):
        with original_runtime(original_api, seed=733, target=target, float_dtype=dtype, native=native):
            fixture = build_original_api_fixture(tmp_path/str(native), parameters=_parameters(case),
                                                 exc=(), exc2=(), silenced=())
            if native:
                _forbid_cpu_steps(fixture)
            durations = (.15, .05, 0., 3.4 if case == 'self_local' else .4)
            for i, duration in enumerate(durations):
                if i == 2:
                    fixture.neurons.rfc = np.array([.3,.1,.4,.2])*b.ms
                    fixture.neurons.v = np.array([-43.,-44.,-47.,-42.])*b.mV
                    fixture.synapses.w = np.arange(1,8)*.2*b.mV
                    # Initializer mutations change model/cache identity but
                    # do not reinitialize a continuing original network.
                    if case != 'string':
                        for name in ('v_0','t_rfc','w_syn'):
                            value = fixture.parameters[name]
                            if np.ndim(value):
                                value[...] *= 2
                fixture.run_ms(duration)
                snapshot, rng = fixture.capture(), capture_rng_state()
                if native:
                    assert_snapshot_equal(snapshot, expected[i][0])
                    assert_rng_state_equal(rng, expected[i][1])
                else:
                    expected.append((snapshot,rng))
    if case == 'self_local':
        assert local_dispatch and all(local_dispatch)


@pytest.mark.parametrize('case', ['external', 'shape', 'column', 'units'])
def test_original_invalid_array_assignments_and_repair(tmp_path, case):
    import brian2 as b
    from mlx_backend.brian_device import original_api
    failures=[];expected=None
    for native in (False, True):
        with original_runtime(original_api, native=native):
            parameters = {'v_0': np.array([-44.,-42.,-48.,-41.])*b.mV} if case == 'external' else {
                't_rfc': {'shape': np.ones(3)*b.ms, 'column': np.ones((4,1))*b.ms,
                          'units': np.ones(4)*b.mV}[case]}
            with pytest.raises(Exception) as caught:
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=parameters,
                                                   exc=(),exc2=(),silenced=())
                fixture.run_ms(.1)
            error=caught.value
            while error.__cause__:
                error=error.__cause__
            failures.append((type(error),str(error)))
            if case == 'external':
                fixture.neurons.namespace['v_0']=-52*b.mV
                if native:_forbid_cpu_steps(fixture)
                fixture.run_ms(.3)
                if native:assert_snapshot_equal(fixture.capture(),expected)
                else:expected=fixture.capture()
    assert failures[0] == failures[1]


@pytest.mark.parametrize('case', ['voltage', 'string'])
def test_array_snapshot_identity_and_trial_broadcast(case):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.model import resolve_model
    from mlx_backend.configured import prepare, initial_state
    from mlx_backend.models import Connectome, Stimulus
    with cpu_runtime():
        parameters=_parameters(case)
        parameters['t_rfc']=np.array([.1,.2,.3,.4])*b.ms
        model=resolve_model(parameters)
        if case == 'voltage':
            source=parameters['v_0']
            source[0]=-40*b.mV
            assert resolve_model(parameters).signature != model.signature
            assert model.v_0[0] == -.044
            with pytest.raises(ValueError):model.v_0.setflags(write=True)
        empty=np.empty(0,np.int32)
        graph=Connectome(np.arange(4),empty,empty,np.empty(0),np.empty(0))
        stimulus=Stimulus(np.empty((2,0,0),np.uint8),(),(),(0,1),0,0,None,model.dt_ms)
        prepared=prepare(graph,stimulus,model)
        state=initial_state(prepared,2)
        values=state.values
        mx.eval(*values.values())
        voltage=model.v_0 if case=='voltage' else (-44.-np.arange(4))*.001
        for name,value in [('v',voltage),('rfc',model.t_rfc)]:
            actual=np.asarray(values[name]).view(np.float64).reshape(2,4)
            assert actual.tobytes() == np.tile(value,(2,1)).tobytes()
