"""Independent original RNG/state oracle across failed native precision attempts."""
import brian2 as b
import mlx.core as mx
import numpy as np
import pytest
from mlx_backend.brian_device import original_api
from mlx_backend.model import DEFAULT_EQUATIONS,DEFAULT_RESET
from original_api_reference import (cpu_runtime, original_runtime,assert_snapshot_equal,build_original_api_fixture,
                                    capture_rng_state,assert_rng_state_equal)
from test_mlx_compiled_integer_adapter import _forbid_cpu_steps,private_compiler_cache


@pytest.mark.metal
@pytest.mark.parametrize('path',['fused','preview'])
@pytest.mark.parametrize('dirty',[False,True])
def test_actual_cython_precision_retry_restores_poisson_cursor_overflow_and_rng(path,dirty,tmp_path,monkeypatch):
    from mlx_backend import configured
    from mlx_backend.brian_poisson import NativePoisson
    extra='x:integer\n' if path=='fused' else 'x:integer\ny:integer\np:integer\nq:integer\nseen:1\n'
    reset=('x=(abs(x)**power+1)%13' if path=='fused' else 'seen=abs(x)**p-abs(y)**q;p+=1')
    parameters=dict(eqs=DEFAULT_EQUATIONS+extra,eq_th='i%2==0',eq_rst=DEFAULT_RESET+';'+reset,
        t_rfc=0*b.ms,t_dly=.1*b.ms,r_poi=4900*b.Hz,r_poi2=3200*b.Hz,power=np.int64(67))
    initial=dict(x=np.full(4,2,np.int64),g=np.array([.01,.02,.03,.04])*b.volt)
    if path=='preview':initial.update(y=np.full(4,3,np.int64),p=np.array([120,0,120,0],np.int64),q=np.array([70,0,70,0],np.int64))
    options=dict(parameters=parameters,exc=(0,1,1),exc2=(1,),silenced=(),initial_state=initial)
    with cpu_runtime(target='cython', integer_dtype=np.int64):
        cpu=build_original_api_fixture(tmp_path/'cpu',**options)
        expected=cpu.run_ms(.4);rng_expected=capture_rng_state()
        continued=cpu.run_ms(.1);rng_continued=capture_rng_state();next_random=np.random.random(10)
    compile_=configured._compiled_program;capacities=[]
    def explicit_precision(model,*args,**kwargs):
        capacity=kwargs.get('native_precision_limbs',2);capacities.append(capacity)
        return compile_(model,*args,native_precision_limbs=capacity)
    monkeypatch.setattr(configured,'_compiled_program',explicit_precision)
    selected=[];can_fuse=NativePoisson.can_fuse
    def capture_fuse(self,p):
        result=can_fuse(self,p);selected.append(result);return result
    monkeypatch.setattr(NativePoisson,'can_fuse',capture_fuse)
    attempts=[]
    def dirty_failed_attempt(sampler,capacity,step,before):
        attempts.append((capacity,step,before[0],before[1]))
        if dirty and capacity==2 and not getattr(sampler,'_precision_test_dirty',False):
            # Adversarial temporary state belongs to the failed attempt. The
            # accepted retry must restore BOTH mutable tape outputs, not only
            # its counter. Values are inspected after the run, never per step.
            sampler.overflow=mx.ones((1,),mx.int32);sampler._precision_test_dirty=True
    if path=='preview':
        events_=NativePoisson.events
        def events(self,p,state,length,times):
            before=self.used,self.overflow;result=events_(self,p,state,length,times)
            dirty_failed_attempt(self,p.program.native_precision_limbs,state.step,before)
            return result
        monkeypatch.setattr(NativePoisson,'events',events)
    else:
        advance_=configured.advance
        def advance(p,state,*args,**kwargs):
            sampler=kwargs.get('native_poisson');before=(sampler.used,sampler.overflow) if sampler is not None else None
            result=advance_(p,state,*args,**kwargs)
            if sampler is not None:dirty_failed_attempt(sampler,p.program.native_precision_limbs,state.step,before)
            return result
        monkeypatch.setattr(configured,'advance',advance)
    with original_runtime(original_api, target='cython', integer_dtype=np.int64):
        native=build_original_api_fixture(tmp_path/'native',**options);_forbid_cpu_steps(native,monkeypatch)
        assert_snapshot_equal(native.run_ms(.4),expected);assert_rng_state_equal(capture_rng_state(),rng_expected)
        assert_snapshot_equal(native.run_ms(.1),continued);assert_rng_state_equal(capture_rng_state(),rng_continued)
        np.testing.assert_array_equal(np.random.random(10),next_random)
    assert selected and all(value==(path=='fused') for value in selected),selected
    assert max(capacities)>=4 and attempts
    if dirty:assert any(capacity==2 and int(np.asarray(overflow)[0])==1 for capacity,_,_,overflow in attempts)
    repeated=[(capacity,int(np.asarray(used)[0]),int(np.asarray(overflow)[0]))
              for capacity,step,used,overflow in attempts if step==0]
    assert repeated[0]==(2,0,0),repeated
    assert any(capacity>=4 and used==0 and overflow==0 for capacity,used,overflow in repeated),repeated


@pytest.mark.metal
def test_actual_shared_error_before_precision_preserves_poisson_rng_and_repairs(tmp_path,monkeypatch):
    from mlx_backend import configured
    parameters=dict(eqs=DEFAULT_EQUATIONS+'''
tick=timestep(t,dt):integer(shared,constant over dt)
x:integer
y:integer
p:integer
q:integer
den:integer
seen:1
''',eq_th='1/den>0',eq_rst=DEFAULT_RESET+';seen=abs(x)**p-abs(y)**q+tick',
        t_rfc=0*b.ms,t_dly=.1*b.ms,r_poi=4900*b.Hz,r_poi2=3200*b.Hz)
    initial=dict(x=np.full(4,2,np.int64),y=np.full(4,3,np.int64),
        p=np.array([3,4096,3,3],np.int64),q=np.full(4,2,np.int64),den=np.array([0,1,1,1],np.int64))
    options=dict(parameters=parameters,exc=(0,1,1),exc2=(1,),silenced=(),initial_state=initial)
    compile_=configured._compiled_program;capacities=[]
    def checked(model,*args,**kwargs):
        kwargs.setdefault('native_precision_limbs',2)
        result=compile_(model,*args,**kwargs);capacities.append(result.native_precision_limbs)
        return result
    from original_api_protocol import check_original_api_runs, observe_run
    following=[]
    def build(mode):
        if mode=='native':monkeypatch.setattr(configured,'_compiled_program',checked)
        return build_original_api_fixture(tmp_path/str(mode=='native'),**options)
    def exercise(fixture, mode):
        failed=observe_run(fixture,.2,raises=b.BrianObjectException,rng=True)
        assert isinstance(failed.error.__cause__,ZeroDivisionError)
        yield failed
        fixture.neurons.den[:]=1;fixture.neurons.p[:]=3
        yield observe_run(fixture,.2,rng=True)
        values=np.random.random(10)
        if mode=='cpu':following.append(values)
        else:np.testing.assert_array_equal(values,following[0])
    check_original_api_runs(build,exercise,monkeypatch,guard=_forbid_cpu_steps,
        target='cython',integer_dtype=np.int64)
    assert capacities and max(capacities)==2,capacities
