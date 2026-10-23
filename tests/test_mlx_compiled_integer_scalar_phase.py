"""Actual compiled integer scalar sections run even with no selected rows."""
import brian2 as b
import numpy as np
import pytest
from original_api_protocol import check_original_api_runs, observe_run
from test_mlx_compiled_integer_adapter import (_typed_fixture,_forbid_cpu_steps,
                                               private_compiler_cache)

pytestmark = pytest.mark.metal


def test_actual_cython_integer_scalar_error_without_spikes(monkeypatch):
    types=dict(x=np.dtype('int64'),y=np.dtype('int64'))
    initial=dict(x=np.arange(4,dtype=np.int64),y=np.full(4,17,np.int64))
    reset='y=x+int(1/(t/dt-1))'
    def exercise(fixture, mode):
        yield observe_run(fixture,.1)
        failed=observe_run(fixture,.1,raises=b.BrianObjectException)
        if mode == 'cpu':
            assert isinstance(failed.error.__cause__,ZeroDivisionError)
        yield failed
    check_original_api_runs(lambda mode:_typed_fixture(types,reset,initial,threshold='False'),
        exercise,monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)


def test_actual_cython_mixed_narrow_and_boxed_abs_keeps_each_policy(monkeypatch):
    types=dict(narrow=np.dtype('int32'),wide=np.dtype('int64'),a=np.dtype('int32'),z=np.dtype('int64'))
    initial=dict(narrow=np.array([np.iinfo(np.int32).min,-5,0,6],np.int32),
                 wide=np.array([-9,-10,11,12],np.int64))
    reset='a=abs(narrow);z=abs(wide)'
    def exercise(fixture, mode):
        failed=observe_run(fixture,.1,raises=b.BrianObjectException)
        if mode == 'cpu':
            assert isinstance(failed.error.__cause__,OverflowError)
        yield failed
        fixture.neurons.narrow[0]=3
        yield observe_run(fixture,.2)
    check_original_api_runs(lambda mode:_typed_fixture(types,reset,initial),exercise,
        monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_actual_cython_integer_block_retains_paired_math_and_power(dtype,monkeypatch):
    types={name:np.dtype(dtype) for name in ('x','s','p')};types['counter']=np.dtype('int64')
    initial=dict(x=np.array([2.**26,.1,-3.125,1.000001],dtype))
    reset='counter+=1;s=sin(x)+cos(x);p=.125*x**3+.375'
    def exercise(fixture, mode):
        yield observe_run(fixture,.3)
    check_original_api_runs(lambda mode:_typed_fixture(types,reset,initial),exercise,
        monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64,
        float_dtype=np.dtype(dtype).type)


def test_actual_cython_integer_state_drives_ode_without_losing_compiler_order(monkeypatch):
    types=dict(x=np.dtype('int64'),f=np.dtype('float64'))
    x=np.array([2**53+1,2**53+3,-2**53-1,-2**53-3],np.int64)
    initial=dict(x=x,f=-x.astype(np.float64)*.0001)
    equations='x:integer (constant)\ndf/dt=(x-f)/second:1\n'
    def exercise(fixture, mode):
        for duration in (.1,.2):
            yield observe_run(fixture,duration)
    check_original_api_runs(lambda mode:_typed_fixture(types,'f=f',initial,threshold='False',equations=equations),
        exercise,monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)
