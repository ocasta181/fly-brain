"""Selected integer frontend across shared scalar/vector execution boundaries."""
from dataclasses import replace
import numpy as np
import pytest
import brian2 as b
import mlx.core as mx

from mlx_backend import configured
from mlx_backend.brian_device import original_api
from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
from original_api_reference import (original_runtime,build_original_api_fixture,
    assert_snapshot_equal,capture_rng_state,assert_rng_state_equal)
from configured_reference import reference
from original_api_protocol import check_original_api_runs, observe_run
from test_mlx_shared_union import case,bits_equal
from test_mlx_compiled_integer_adapter import _forbid_cpu_steps,_typed_fixture

pytestmark=pytest.mark.metal


def test_actual_cpp_shared_integer_carry_and_narrow_store(tmp_path):
    from test_mlx_compiled_integer_adapter import _actual_cpp_component
    _actual_cpp_component(tmp_path,'shared')


def test_shared_integer_disabled_threshold_preserves_cpu_eventspace(monkeypatch):
    equations='x:integer\ny:integer\ntick=timestep(t,dt):integer(shared,constant over dt)\n'
    expected=[]
    for native in (False,True):
        with original_runtime(original_api, target='cython', integer_dtype=np.int64, native=native) as device:
            fixture=_typed_fixture(dict(x=np.dtype('int32'),y=np.dtype('int64')),
                'y+=tick+1',dict(x=np.arange(9,dtype=np.int32)),
                threshold='int(x)>0',equations=equations)
            eventspace=fixture.neurons.variables['_spikespace'].get_value()
            eventspace[:3]=[0,2,4];eventspace[-1]=3
            if native:_forbid_cpu_steps(fixture,monkeypatch)
            for index,duration in enumerate((.3,.2)):
                current=fixture.run_ms(duration)
                if native:assert_snapshot_equal(current,expected[index])
                else:expected.append(current)
            if native:assert 'threshold' in device._sessions[fixture.network].prepared.program.disabled_stages
            else:
                assert fixture.neurons.thresholder['spike'].codeobj.compiled_code['run'] is None
                np.testing.assert_array_equal(expected[0].spike_i,np.tile([0,2,4],3))
                # Brian's next before_run clears the stale eventspace.
                np.testing.assert_array_equal(expected[1].spike_i,expected[0].spike_i)
                assert expected[1].clock_timestep==5
                np.testing.assert_array_equal(expected[1].state['tick'],[4])


def _check_shared_integer_sections(monkeypatch,integer_dtype,offset):
    monkeypatch.setitem(b.prefs,'codegen.target','cython')
    monkeypatch.setitem(b.prefs,'core.default_integer_dtype',integer_dtype)
    graph,stimulus,_=case('cython',True,steps=7)
    equations=DEFAULT_EQUATIONS.replace(
        'dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)','v:volt(shared)')+'''
tick=timestep(t,dt)+N:integer(shared,constant over dt)
held=tick+int(v/mV):integer(shared,constant over dt)
sampled=held+i*(int(3+t/ms)+OFFSET):integer(constant over dt)
seen:integer
'''.replace('OFFSET',str(offset))
    model=resolve_model(dict(eqs=equations,eq_th='True',eq_rst='g=0*mV;seen=sampled+2*held',
        t_dly=0*b.ms,t_rfc=0*b.ms),cpu_target='cython')
    graph=replace(graph,sources=np.array([0,2],np.int32),destinations=np.array([1,3],np.int32),
        counts=np.array([.25,-.125]),weights_mv=np.zeros(2))
    expected=reference(graph,stimulus,model,record_state=True)
    prepared=configured.prepare(graph,stimulus,model)
    assert 'subexpression' in prepared.program.compiled_integer_stages
    assert prepared.program.scalar_stages['subexpression'].scalar_code
    assert prepared.program.variables['sampled'].dtype==np.dtype(integer_dtype).name
    state=configured.initial_state(prepared,2);events=mx.array(stimulus.events.reshape(-1))
    for step in range(7):
        state,_=configured.advance(prepared,state,events,7,1)
        configured.evaluate_state(state)
        for name,value in configured.host_state(state,prepared).items():
            bits_equal(value,expected.traces[name][:,step],name)


def test_shared_integer_sections_and_zero_delay_vector_carry(monkeypatch):
    _check_shared_integer_sections(monkeypatch,np.int32,0)


def test_shared_integer_int64_auxiliary_carry_preserves_high_bits(monkeypatch):
    _check_shared_integer_sections(monkeypatch,np.int64,2**32)


def test_shared_integer_state_phase_carries_binary64_auxiliary(monkeypatch):
    monkeypatch.setitem(b.prefs,'codegen.target','cython')
    graph,stimulus,_=case('cython',True,steps=5)
    equations=DEFAULT_EQUATIONS.replace('(v_0 - v + g)',
        '(v_0 - v + g + tick*mV/1024)')+'\ntick=timestep(t,dt)+N:integer(shared,constant over dt)\n'
    model=resolve_model(dict(eqs=equations,eq_th='v+tick*mV/1024>v_th',
        eq_rst='v=v_rst;g=0*mV',t_rfc=0*b.ms),cpu_target='cython')
    expected=reference(graph,stimulus,model,record_state=True)
    prepared=configured.prepare(graph,stimulus,model)
    assert {'state','threshold'}<=set(prepared.program.compiled_integer_stages)
    assert any(value.dtype=='float64' for value in prepared.program.scalar_stages['state'].variables)
    state=configured.initial_state(prepared,2);events=mx.array(stimulus.events.reshape(-1))
    for step in range(5):
        state,_=configured.advance(prepared,state,events,5,1)
        configured.evaluate_state(state)
        for name,value in configured.host_state(state,prepared).items():
            bits_equal(value,expected.traces[name][:,step],name)


@pytest.mark.parametrize('mode',['scalar_error','disabled_reset'])
def test_shared_integer_scalar_error_or_disabled_stage(tmp_path,monkeypatch,mode):
    equations=DEFAULT_EQUATIONS+'\ntick=timestep(t,dt):integer(shared,constant over dt)\nx:integer\ny:integer\nf:1\n'
    reset='v=v_rst;g=0*mV;y=x+int(1/(t/dt-1))' if mode=='scalar_error' else 'v=v_rst;g=0*mV;y=f'
    expected=[]
    for native in (False,True):
        with original_runtime(
                original_api, seed=912, target='cython', integer_dtype=np.int64, native=native) as device:
            fixture=build_original_api_fixture(tmp_path/str(native),parameters=dict(
                eqs=equations,eq_th='False' if mode=='scalar_error' else 'True',
                eq_rst=reset,t_rfc=0*b.ms),exc=(),exc2=(),silenced=(),
                initial_state=dict(v=-40*b.mV,x=np.array([1,2,3,4],np.int64),y=17,f=.125))
            if native:_forbid_cpu_steps(fixture,monkeypatch)
            current=fixture.run_ms(.1)
            if native:assert_snapshot_equal(current,expected[0])
            else:expected.append(current)
            if mode=='scalar_error':
                with pytest.raises(b.BrianObjectException) as error:fixture.run_ms(.1)
                cause=error.value.__cause__
                assert isinstance(cause,ZeroDivisionError)
                if native:assert str(cause)==message
                else:message=str(cause)
                current=fixture.capture()
            else:
                current=fixture.run_ms(.2)
                if native:assert 'reset' in device._sessions[fixture.network].prepared.program.disabled_stages
                else:assert fixture.neurons.resetter['spike'].codeobj.compiled_code['run'] is None
            if native:assert_snapshot_equal(current,expected[1])
            else:expected.append(current)
            if native:
                assert_rng_state_equal(capture_rng_state(),rng)
                np.testing.assert_array_equal(np.random.random(10),draw)
            else:rng=capture_rng_state();draw=np.random.random(10)


def test_shared_integer_vector_failure_keeps_scalar_commit_and_repairs(tmp_path,monkeypatch):
    equations=DEFAULT_EQUATIONS+'''
tick=timestep(t,dt)+N:integer(shared,constant over dt)
den:integer
held=tick+11//den:integer(constant over dt)
seen:integer
'''
    def exercise(fixture, mode):
        failed=observe_run(fixture,.1,raises=b.BrianObjectException)
        assert isinstance(failed.error.__cause__,ZeroDivisionError)
        yield failed
        fixture.neurons.den[:]=2
        yield observe_run(fixture,.2)
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/str(mode == 'native'),
        parameters=dict(eqs=equations,eq_th='True',eq_rst='v=v_rst;g=0*mV;seen=held',t_rfc=0*b.ms),
        initial_state=dict(den=np.array([1,2,0,3],np.int64)),exc=(),exc2=(),silenced=()),
        exercise,monkeypatch,guard=_forbid_cpu_steps,seed=919,target='cython',integer_dtype=np.int64)


def test_shared_integer_fused_scalar_error_preserves_poisson_rng(tmp_path,monkeypatch):
    equations=DEFAULT_EQUATIONS+'\ntick=timestep(t,dt):integer(shared,constant over dt)\nseen:integer\n'
    expected=[]
    for native in (False,True):
        with original_runtime(
                original_api, seed=920, target='cython', integer_dtype=np.int64, native=native) as device:
            fixture=build_original_api_fixture(tmp_path/str(native),parameters=dict(
                eqs=equations,eq_th='False',eq_rst='v=v_rst;g=0*mV;seen=tick+int(1/(t/dt-1))',
                t_rfc=0*b.ms,r_poi=3700*b.Hz,r_poi2=7200*b.Hz,t_dly=.3*b.ms))
            if native:_forbid_cpu_steps(fixture,monkeypatch)
            with pytest.raises(b.BrianObjectException) as error:fixture.run_ms(.2)
            assert isinstance(error.value.__cause__,ZeroDivisionError)
            current=fixture.capture()
            if native:
                assert str(error.value.__cause__)==message
                assert_snapshot_equal(current,expected[0])
                assert device.last_timings['prepass_and_dependencies_s']==0
                assert_rng_state_equal(capture_rng_state(),rng)
                np.testing.assert_array_equal(np.random.random(10),draw)
            else:
                message=str(error.value.__cause__);expected.append(current)
                rng=capture_rng_state();draw=np.random.random(10)


@pytest.mark.parametrize('neurons',[4,5,257])
def test_shared_vector_before_start_alias_versioning(monkeypatch,neurons):
    equations='''
x:1
y:1
seen:1
scale=1+t/ms:1(shared,constant over dt)
held=scale*(x+y):1(constant over dt)
'''
    def build(mode):
        types={name:np.dtype('float32') for name in ('x','y','seen')}
        initial=dict(x=np.resize(np.array([.3,-.3,3.,-3.],np.float32),neurons),
            y=np.resize(np.array([.1,-.1,.5,2.1],np.float32),neurons))
        return _typed_fixture(types,'seen=held',initial,equations=equations)
    def exercise(fixture, mode):
        for duration in (0.,.15,.15):
            yield observe_run(fixture,duration)
    check_original_api_runs(build,exercise,monkeypatch,guard=_forbid_cpu_steps,
        target='cython',float_dtype=np.float32)
