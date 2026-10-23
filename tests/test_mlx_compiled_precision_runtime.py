"""Independent original Cython model behavior for native dynamic precision."""
import brian2 as b
import numpy as np
import pytest
from mlx_backend.brian_device import original_api
from original_api_reference import cpu_runtime, original_runtime,assert_snapshot_equal
from test_mlx_compiled_integer_adapter import (_typed_fixture,_forbid_cpu_steps,
                                               private_compiler_cache)

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('phase',['reset','threshold'])
@pytest.mark.parametrize('record',[True,False])
def test_actual_cython_forced_precision_retry_preserves_states_spikes_and_continuation(phase,record,monkeypatch):
    from mlx_backend import configured
    types={name:np.dtype('int64') for name in ('x','y','p','q')};types['seen']=np.dtype('float64')
    initial=dict(x=np.full(4,2,np.int64),y=np.full(4,3,np.int64),
                 p=np.array([3,120,-7,0],np.int64),q=np.array([2,70,-3,-2],np.int64),seen=np.zeros(4))
    expression='abs(x)**p-abs(y)**q'
    threshold='True' if phase=='reset' else 'abs(x)**p>abs(y)**q'
    reset=('seen='+expression if phase=='reset' else 'seen+=1')+';p+=1'
    with cpu_runtime(target='cython', integer_dtype=np.int64):
        cpu=_typed_fixture(types,reset,initial,threshold=threshold,record=record)
        expected=cpu.run_ms(.2)
        continued=cpu.run_ms(.1)
    compile_=configured._compiled_program;programs=[]
    def explicit_precision(model,*args,**kwargs):
        program=compile_(model,*args,native_precision_limbs=kwargs.get('native_precision_limbs',2))
        programs.append((model.signature,program.native_precision_limbs))
        return program
    monkeypatch.setattr(configured,'_compiled_program',explicit_precision)
    with original_runtime(original_api, target='cython', integer_dtype=np.int64):
        native=_typed_fixture(types,reset,initial,threshold=threshold,record=record)
        _forbid_cpu_steps(native,monkeypatch)
        assert_snapshot_equal(native.run_ms(.2),expected)
        assert_snapshot_equal(native.run_ms(.1),continued)
    assert programs and any(limbs==2 for _,limbs in programs)
    assert any(limbs>=4 for _,limbs in programs),programs
    assert any(signature==programs[0][0] and limbs>=4 for signature,limbs in programs),programs


@pytest.mark.parametrize('phase',['reset','threshold'])
def test_actual_cython_user_error_precedes_later_precision_request(phase,monkeypatch):
    from mlx_backend import configured
    types={name:np.dtype('int64') for name in ('x','y','p','q','r','den')};types['seen']=np.dtype('float64')
    initial=dict(x=np.full(4,2,np.int64),y=np.full(4,3,np.int64),q=np.full(4,2,np.int64),
        p=np.array([3,4096,3,3] if phase=='reset' else [4096,3,3,3],np.int64),
        r=np.ones(4,np.int64),den=np.array([0,1,1,1] if phase=='reset' else [1,0,1,1],np.int64),seen=np.zeros(4))
    reset='seen=abs(x)**p-abs(y)**q'+('+int(r/den)' if phase=='reset' else '')
    threshold='True' if phase=='reset' else 'r/den>0'
    with cpu_runtime(target='cython', integer_dtype=np.int64):
        cpu=_typed_fixture(types,reset,initial,threshold=threshold)
        with pytest.raises(b.BrianObjectException) as failure:cpu.run_ms(.1)
        cause=failure.value.__cause__;expected=cpu.capture()
        assert isinstance(cause,ZeroDivisionError)
        cpu.neurons.den[:]=1;cpu.neurons.p[:]=3;continued=cpu.run_ms(.2)
    compile_=configured._compiled_program;capacities=[]
    def explicit_precision(model,*args,**kwargs):
        capacity=kwargs.get('native_precision_limbs',2);capacities.append(capacity)
        return compile_(model,*args,native_precision_limbs=capacity)
    monkeypatch.setattr(configured,'_compiled_program',explicit_precision)
    with original_runtime(original_api, target='cython', integer_dtype=np.int64):
        native=_typed_fixture(types,reset,initial,threshold=threshold)
        _forbid_cpu_steps(native,monkeypatch)
        with pytest.raises(b.BrianObjectException) as failure:native.run_ms(.1)
        assert type(failure.value.__cause__) is type(cause)
        assert str(failure.value.__cause__)==str(cause)
        assert_snapshot_equal(native.capture(),expected)
        native.neurons.den[:]=1;native.neurons.p[:]=3
        assert_snapshot_equal(native.run_ms(.2),continued)
    assert capacities and max(capacities)==2,capacities


@pytest.mark.parametrize('boundary',['last_frontier_step','large_absolute_clock'])
def test_actual_cython_precision_frontier_clock_boundaries(boundary,monkeypatch):
    from dataclasses import replace
    from mlx_backend import configured
    types={name:np.dtype('int64') for name in ('x','y','p','q')};types['seen']=np.dtype('float64')
    initial=dict(x=np.full(4,2,np.int64),y=np.full(4,3,np.int64),
        p=np.array([3,63 if boundary=='last_frontier_step' else 120,-7,0],np.int64),
        q=np.array([2,2,-3,-2],np.int64),seen=np.zeros(4))
    reset='seen=abs(x)**p-abs(y)**q;p+=int(t>=6.2*ms)' if boundary=='last_frontier_step' else 'seen=abs(x)**p-abs(y)**q'
    duration=6.4 if boundary=='last_frontier_step' else .1
    def move_clock(fixture):
        if boundary!='large_absolute_clock':return
        step=(1<<46)+123;clock=fixture.neurons.clock
        clock.variables['timestep'].get_value()[0]=step
        clock.variables['t'].get_value()[0]=step*float(clock.dt)
        fixture.network.t_=step*float(clock.dt)
    with cpu_runtime(target='cython', integer_dtype=np.int64):
        cpu=_typed_fixture(types,reset,initial);move_clock(cpu)
        expected=cpu.run_ms(duration);continued=cpu.run_ms(.1)
    compile_=configured._compiled_program;capacities=[]
    def explicit_precision(model,*args,**kwargs):
        capacity=kwargs.get('native_precision_limbs',2);capacities.append(capacity)
        return compile_(model,*args,native_precision_limbs=capacity)
    monkeypatch.setattr(configured,'_compiled_program',explicit_precision)
    prepare_=configured.prepare
    def unit_windows(*args,**kwargs):return replace(prepare_(*args,**kwargs),window=1)
    monkeypatch.setattr(configured,'prepare',unit_windows)
    with original_runtime(original_api, target='cython', integer_dtype=np.int64):
        native=_typed_fixture(types,reset,initial);move_clock(native)
        _forbid_cpu_steps(native,monkeypatch)
        assert_snapshot_equal(native.run_ms(duration),expected)
        assert_snapshot_equal(native.run_ms(.1),continued)
    assert max(capacities)>=4,capacities
