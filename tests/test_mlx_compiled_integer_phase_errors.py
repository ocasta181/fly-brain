"""Actual compiled phase errors preserve clock, committed rows and repair."""
import brian2 as b
import numpy as np
import pytest
from original_api_protocol import check_original_api_runs, observe_run
from test_mlx_compiled_integer_adapter import (_typed_fixture,_forbid_cpu_steps,
                                               private_compiler_cache)


@pytest.mark.metal
@pytest.mark.parametrize('phase',['threshold','state'])
def test_actual_cython_integer_phase_error_prefix_and_repaired_continuation(phase,monkeypatch):
    types={name:np.dtype('int64') for name in ('x','den','counter')}
    initial=dict(x=np.array([9,1,-7,2],np.int64),den=np.array([2,0,3,1],np.int64),
                 counter=np.zeros(4,np.int64))
    keywords=dict(threshold='x//den>0',equations=None)
    if phase=='state':
        types['f']=np.dtype('float64');initial['f']=np.array([1.,2.,3.,4.])
        keywords=dict(threshold='True',equations='x:integer\nden:integer\ncounter:integer\ndf/dt=(x/den-f)/second:1\n')
    def exercise(fixture, mode):
        failed=observe_run(fixture,.1,raises=b.BrianObjectException)
        if mode == 'cpu':
            assert isinstance(failed.error.__cause__,ZeroDivisionError)
        yield failed
        fixture.neurons.den[:]=3
        yield observe_run(fixture,.2)
    check_original_api_runs(lambda mode:_typed_fixture(types,'counter+=1',initial,**keywords),
        exercise,monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)
