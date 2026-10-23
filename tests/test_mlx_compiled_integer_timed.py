"""Existing native TimedArrays compose with complete integer-writing blocks."""
import brian2 as b
import numpy as np
import pytest
from original_api_protocol import check_original_api_runs, observe_run
from test_mlx_compiled_integer_adapter import (_typed_fixture,_forbid_cpu_steps,
                                               private_compiler_cache)


@pytest.mark.metal
@pytest.mark.parametrize('dimensions',[1,2])
def test_actual_cython_timedarray_in_integer_block_mask_and_continuation(dimensions,monkeypatch):
    types=dict(counter=np.dtype('int64'),column=np.dtype('int64'),seen=np.dtype('float64'))
    initial=dict(counter=np.zeros(4,np.int64),column=np.array([-1,0,1,3],np.int64))
    values=[.25,1.25,-1.5] if dimensions==1 else [[.25,1.25],[2.25,3.25],[4.25,5.25]]
    expression='drive(t-dt'+(',column)' if dimensions==2 else ')')
    reset='counter+=int('+expression+');seen='+expression
    def build(mode):
        drive=b.TimedArray(values,dt=.1*b.ms)
        return _typed_fixture(types,reset,initial,threshold='i%3!=1',namespace=dict(drive=drive))
    def exercise(fixture, mode):
        for duration in (.1,.3):
            yield observe_run(fixture,duration)
    check_original_api_runs(build,exercise,monkeypatch,guard=_forbid_cpu_steps,
        target='cython',integer_dtype=np.int64)


@pytest.mark.metal
@pytest.mark.parametrize('dimensions',[1,2])
def test_actual_cpp_timedarray_in_integer_block_mask_and_continuation(dimensions,tmp_path):
    from test_mlx_compiled_integer_adapter import _actual_cpp_component
    _actual_cpp_component(tmp_path,'timed'+str(dimensions))
