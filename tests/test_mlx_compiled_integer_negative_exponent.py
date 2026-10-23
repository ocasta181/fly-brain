"""Independent Cython boxed truth/comparison and negative-power boundaries."""
import brian2 as b
import numpy as np
import pytest
from original_api_protocol import check_original_api_runs, observe_run
from test_mlx_compiled_integer_adapter import (_typed_fixture,_forbid_cpu_steps,
                                               private_compiler_cache)


@pytest.mark.metal
def test_actual_cython_negative_unit_boxed_power_uses_float_exponent_policy(monkeypatch):
    types={name:np.dtype('int64') for name in ('x','z','y')}
    initial=dict(x=np.ones(9,np.int64),z=np.array([np.iinfo(np.int64).min,-9007199254740995,-9007199254740993,-9007199254740992,-9007199254740991,-4,-3,-2,-1],np.int64))
    def exercise(fixture, mode):
        for duration in (.1,.2):
            yield observe_run(fixture,duration)
    check_original_api_runs(lambda mode:_typed_fixture(types,'y=(-abs(x))**z',initial),exercise,
        monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)
