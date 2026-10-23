"""Preparation ABI names cannot shadow original accepted data symbols."""
import brian2 as b
import numpy as np
import pytest
from original_api_protocol import check_original_api_runs, observe_run
from test_mlx_compiled_integer_adapter import (_typed_fixture,_forbid_cpu_steps,
                                               private_compiler_cache)


@pytest.mark.metal
def test_actual_cython_integer_output_prefix_is_valid_data(monkeypatch):
    types={name:np.dtype('int64') for name in ('x','out_x')}
    initial=dict(x=np.array([-7,0,5,11],np.int64),out_x=np.array([3,11,-9,17],np.int64))
    expression='x+=out_x;out_x+=1'
    def exercise(fixture, mode):
        for duration in (.1,.2):
            yield observe_run(fixture,duration)
    check_original_api_runs(lambda mode:_typed_fixture(types,expression,initial),exercise,
        monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)


@pytest.mark.metal
def test_actual_cpp_integer_output_prefix_is_valid_data(tmp_path):
    from test_mlx_compiled_integer_adapter import _actual_cpp_component
    _actual_cpp_component(tmp_path,'names')
