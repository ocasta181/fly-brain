"""Accepted Python-integer compositions retain checked conversion boundaries."""
import brian2 as b
import numpy as np
import pytest
from original_api_protocol import check_original_api_runs, observe_run
from test_mlx_compiled_integer_adapter import (_typed_fixture,_forbid_cpu_steps,
                                               private_compiler_cache)

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('expression',['abs(x)*2','abs(x)+abs(z)','(x+4294967297)%7',
    'abs(x)*abs(z)','((x+1208925819614629174706176)*(z-1180591620717411303424))%4294967311',
    '(x+1208925819614629174706176)//(z+1208925819614629174706176)',
    '-(abs(x)+abs(z))%(-4294967311)',
    '(x+4294967297)%(z-1)','(x+4294967297)//(z-1)',
    'abs(x)**0','abs(x)**3','(abs(x)+2)**3','(abs(x)**5)%4294967311','abs(x)**(-1)',
    'abs(x)/z','abs(x)/(z-1)',
    '(abs(x)+1208925819614629174706176)/(z+1180591620717411303424)'])
def test_actual_cython_composed_python_integer_helpers(expression,monkeypatch):
    types={name:np.dtype('int64') for name in ('x','z','y')}
    initial=dict(x=np.array([-9,-(1<<63),7,2],np.int64),z=np.array([3,1,11,-7],np.int64),
                 y=np.full(4,17,np.int64))
    def exercise(fixture, mode):
        yield observe_run(fixture,.1,raises=b.BrianObjectException,optional=True)
        fixture.neurons.x[:]=-5;fixture.neurons.z[:]=3
        yield observe_run(fixture,.2)
    check_original_api_runs(lambda mode:_typed_fixture(types,'y='+expression,initial),exercise,
        monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)


@pytest.mark.parametrize('dtype',['int32','uint8','uint64'])
def test_actual_cython_python_composition_destination_width(dtype,monkeypatch):
    types=dict(x=np.dtype(dtype),z=np.dtype('int64'),y=np.dtype(dtype))
    limits=np.iinfo(dtype)
    initial=dict(x=np.array([0,127,128,255] if dtype=='uint8' else [0,limits.max,limits.min,3],dtype),
                 z=np.array([3,1,-7,5],np.int64))
    def exercise(fixture, mode):
        yield observe_run(fixture,.1,raises=b.BrianObjectException)
        fixture.neurons.x[:]=1;fixture.neurons.z[:]=2
        yield observe_run(fixture,.2)
    check_original_api_runs(lambda mode:_typed_fixture(types,'y=abs(x)+abs(z)',initial),exercise,
        monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)


@pytest.mark.parametrize('threshold',['abs(x)>9223372036854775807','abs(x)<abs(z)',
    '(x>0) and (abs(x)%(z-1)>0)','not(abs(x)>0)'])
def test_actual_cython_python_integer_threshold_preserves_spikes(threshold,monkeypatch):
    types={name:np.dtype('int64') for name in ('x','z','y')}
    initial=dict(x=np.array([-(1<<63),-5,0 if 'not' in threshold else 7,2],np.int64),
                 z=np.array([1,1,11,-7] if 'z-1' in threshold else [3,11,11,-7],np.int64))
    def exercise(fixture, mode):
        for duration in (.1,.2):
            yield observe_run(fixture,duration)
    check_original_api_runs(lambda mode:_typed_fixture(types,'y+=1',initial,threshold=threshold),
        exercise,monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)


def test_actual_cython_boxed_division_rounds_ratio_before_integer_conversion(monkeypatch):
    numerators=[(1<<63)-2,(1<<63)-1,(1<<53)+3,(1<<53)+1,(1<<53)+5,(1<<53)+7,
                (1<<52)-1,(1<<63)-1,1]
    denominators=[(1<<63)-1,3,1,1,2,2,2,7,2]
    types={name:np.dtype('int64') for name in ('x','z','y')}
    initial=dict(x=np.array(numerators*2,np.int64),
                 z=np.array(denominators+[-value for value in denominators],np.int64))
    def exercise(fixture, mode):
        for duration in (.1,.2):
            yield observe_run(fixture,duration)
    check_original_api_runs(lambda mode:_typed_fixture(types,'y=abs(x)/z',initial),exercise,
        monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('expression',['sqrt(abs(x))','abs(x)'])
def test_actual_cython_boxed_integer_composes_with_existing_native_math(dtype,expression,monkeypatch):
    types=dict(x=np.dtype('int64'),seen=np.dtype(dtype))
    initial=dict(x=np.array([-(1<<63),(1<<63)-1,(1<<53)+1,0],np.int64))
    def exercise(fixture, mode):
        for duration in (.1,.2):
            yield observe_run(fixture,duration)
    check_original_api_runs(lambda mode:_typed_fixture(types,'seen='+expression,initial),exercise,
        monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('negative',[False,True])
def test_actual_cython_boxed_float_conversion_overflow_prefix_and_repair(dtype,negative,monkeypatch):
    types=dict(x=np.dtype('int64'),seen=np.dtype(dtype))
    initial=dict(x=np.array([0,1,3,4],np.int64))
    constant=(1<<1024)-(1<<970)-4
    expression='seen='+('-abs(x)-' if negative else 'abs(x)+')+str(constant)
    def exercise(fixture, mode):
        failed=observe_run(fixture,.1,raises=b.BrianObjectException)
        if mode == 'cpu':
            assert isinstance(failed.error.__cause__,OverflowError)
            assert str(failed.error.__cause__)=='int too large to convert to float'
        yield failed
        fixture.neurons.x[:]=0
        yield observe_run(fixture,.2)
    check_original_api_runs(lambda mode:_typed_fixture(types,expression,initial),exercise,
        monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)
