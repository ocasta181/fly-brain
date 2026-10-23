"""Actual selected Cython scalar coercion and native retry across phase traces."""
import brian2 as b
import numpy as np
import pytest
from mlx_backend import configured
from mlx_backend.brian_device import original_api
from original_api_reference import original_runtime,assert_snapshot_equal
from original_api_protocol import check_original_api_runs, observe_run
from test_mlx_compiled_integer_adapter import _typed_fixture,_forbid_cpu_steps,private_compiler_cache

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('case',['real_retry','integer_high_bits','scalar_overflow'])
def test_actual_cython_scalar_auxiliary_coercion_before_vector_execution(case,tmp_path,monkeypatch):
    real=case=='real_retry'
    equations='sx:integer(shared)\nsy:integer(shared)\np:integer(shared)\nq:integer(shared)\nseen:1\nseen_i:integer\n'
    if real:
        # A scalar checked boxed expression produces a real shared field;
        # vector work consumes it without narrowing its raw binary64 bits.
        equations+='held=abs(sx)**p-abs(sy)**q:1(shared,constant over dt)\n'
        reset='seen=held+i/8'
    else:
        # Brian hoists this Python-object power into a declared int64 scalar
        # auxiliary before the vector assignment. That actual coercion is an
        # observable boundary: high finite bits survive and overflow raises.
        reset='seen_i=abs(sx)**p+i'
    initial=dict(seen=np.zeros(4),seen_i=np.zeros(4,np.int64),sx=np.array([2 if real else 3],np.int64),
        sy=np.array([3],np.int64),p=np.array([120 if real else 40 if case=='scalar_overflow' else 39],np.int64),
        q=np.array([70],np.int64))
    types={name:np.dtype('float64' if name=='seen' else 'int64') for name in initial}
    expected=[];messages=[];capacities=[]
    compile_=configured._compiled_program
    def compile_checked(model,*args,**kwargs):
        kwargs.setdefault('native_precision_limbs',2)
        result=compile_(model,*args,**kwargs);capacities.append(result.native_precision_limbs)
        return result
    for native in (False,True):
        with original_runtime(original_api, target='cython', integer_dtype=np.int64, native=native) as device:
            if native:monkeypatch.setattr(configured,'_compiled_program',compile_checked)
            fixture=_typed_fixture(types,reset,initial,equations=equations,
                threshold='False' if case=='scalar_overflow' else 'i%2==0')
            if native:_forbid_cpu_steps(fixture,monkeypatch)
            for index,duration in enumerate((.2,.1)):
                if case=='scalar_overflow' and index==0:
                    with pytest.raises(b.BrianObjectException) as failure:fixture.run_ms(duration)
                    cause=failure.value.__cause__;assert isinstance(cause,OverflowError)
                    if native:assert str(cause)==messages[0]
                    else:messages.append(str(cause))
                    current=fixture.capture();fixture.neurons.p[:]=3
                else:current=fixture.run_ms(duration)
                if native:assert_snapshot_equal(current,expected[index])
                else:expected.append(current)
            if not native:
                source=str(fixture.neurons.resetter['spike'].codeobj.code)
                (tmp_path/(case+'-actual-cython-reset.txt')).write_text(source)
                if not real:assert 'cdef int64_t _lio_' in source
            else:
                program=device._sessions[fixture.network].prepared.program
                assert program.scalar_stages
                if not real:
                    assert any(v.dtype=='int64' for v in program.scalar_stages['reset'].variables)
    if real:assert max(capacities)>=4 and 2 in capacities,capacities


@pytest.mark.parametrize('error',['none','zero','overflow'])
def test_actual_cython_positive_boxed_fractional_power_and_repair(error,monkeypatch):
    types=dict(x=np.dtype('int64'),exponent=np.dtype('float64'),seen=np.dtype('float64'))
    initial=dict(x=np.array([0 if error=='zero' else 2,3,5,1],np.int64),
        exponent=np.array([-1 if error=='zero' else 1024 if error=='overflow' else .5,2.5,-2.5,1024.]),seen=np.zeros(4))
    def exercise(fixture, mode):
        if error!='none':
            failed=observe_run(fixture,.1,raises=b.BrianObjectException)
            assert isinstance(failed.error.__cause__,ZeroDivisionError if error=='zero' else OverflowError)
            yield failed
            fixture.neurons.x[0]=2 if error=='zero' else 1
        for duration in (.2,.1):
            yield observe_run(fixture,duration)
    check_original_api_runs(lambda mode:_typed_fixture(types,'seen=abs(x)**exponent',initial),
        exercise,monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)


def test_actual_cython_checked_to_ordinary_model_continuation(monkeypatch):
    from mlx_backend.model import DEFAULT_RESET
    initial=dict(x=np.full(4,2,np.int64),y=np.full(4,3,np.int64),
        p=np.full(4,120,np.int64),q=np.full(4,70,np.int64),seen=np.zeros(4))
    types={name:np.dtype('float64' if name=='seen' else 'int64') for name in initial}
    compile_=configured._compiled_program;expected=[]
    def checked(model,*args,**kwargs):
        kwargs.setdefault('native_precision_limbs',2)
        return compile_(model,*args,**kwargs)
    for native in (False,True):
        with original_runtime(original_api, target='cython', integer_dtype=np.int64, native=native) as device:
            if native:monkeypatch.setattr(configured,'_compiled_program',checked)
            fixture=_typed_fixture(types,'seen=abs(x)**p-abs(y)**q',initial)
            if native:_forbid_cpu_steps(fixture,monkeypatch)
            for index in range(2):
                if index:fixture.neurons.event_codes['spike']=DEFAULT_RESET+';seen+=1'
                current=fixture.run_ms(.1)
                if native:
                    assert_snapshot_equal(current,expected[index])
                    capacity=device._sessions[fixture.network].prepared.program.native_precision_limbs
                    assert capacity==0 if index else capacity>=4
                else:expected.append(current)
