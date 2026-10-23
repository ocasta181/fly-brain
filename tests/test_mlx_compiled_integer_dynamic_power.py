"""Actual Cython integer-power loops and backend-name composition."""
import brian2 as b
import numpy as np
import pytest
from original_api_protocol import check_original_api_runs, observe_run
from test_mlx_compiled_integer_adapter import (_typed_fixture,_forbid_cpu_steps,
                                               private_compiler_cache)

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('case',['positive','negative'])
@pytest.mark.parametrize('base_name',['base_value','k'])
@pytest.mark.parametrize('state_dtype',[np.float32,np.float64])
def test_actual_cython_dynamic_integer_power_and_repaired_continuation(case,base_name,state_dtype,monkeypatch,boxed=False):
    types={name:np.dtype('int64') for name in (base_name,'negative','power_value')}
    initial={base_name:np.array([2,3,5,7] if case=='positive' else [2,0,-1,7],np.int64),
             'negative':np.array([1,2,5,63] if case=='positive' else [-1,-1,-5,0],np.int64)}
    base=f'abs({base_name})' if boxed else base_name
    expression=f'power_value={base}**negative;{base_name}+=1'
    def exercise(fixture, mode):
        assert np.dtype(fixture.neurons.variables['v'].dtype)==np.dtype(state_dtype)
        yield observe_run(fixture,.1,raises=b.BrianObjectException,optional=True)
        getattr(fixture.neurons,base_name)[:]=3;fixture.neurons.negative[:]=2
        yield observe_run(fixture,.2)
    check_original_api_runs(lambda mode:_typed_fixture(types,expression,initial),exercise,
        monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64,
        float_dtype=state_dtype)


@pytest.mark.parametrize('dtype,power',[('int32',31),('int64',63),('uint8',7),('uint64',63)])
def test_actual_cython_boxed_dynamic_power_conversion_boundaries(dtype,power,monkeypatch):
    types=dict(base_value=np.dtype('int64'),negative=np.dtype('int64'),power_value=np.dtype(dtype))
    # Every earlier row is valid. The last row fails at the actual destination
    # boundary, exposing exact negative minima and committed-prefix stores.
    bases=[0,0,1,2,2,2,2]
    exponents=[0,1,-(1<<63),-1,2,power if dtype!='uint64' else power-1,power+1]
    initial=dict(base_value=np.array(bases,np.int64),negative=np.array(exponents,np.int64),
                 power_value=np.full(len(bases),17,dtype))
    expression='power_value=(-abs(base_value))**negative'
    def exercise(fixture, mode):
        yield observe_run(fixture,.1,raises=b.BrianObjectException)
        fixture.neurons.base_value[:]=2;fixture.neurons.negative[:]=2
        yield observe_run(fixture,.2)
    check_original_api_runs(lambda mode:_typed_fixture(types,expression,initial),exercise,
        monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)


@pytest.mark.parametrize('case',['positive','negative'])
@pytest.mark.parametrize('state_dtype',[np.float32,np.float64])
def test_actual_cython_boxed_dynamic_power_with_backend_names(case,state_dtype,monkeypatch):
    test_actual_cython_dynamic_integer_power_and_repaired_continuation(
        case,'k',state_dtype,monkeypatch,boxed=True)
