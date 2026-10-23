"""Prevent silent misuse while the accepted floating boxed-power gap is open."""
import brian2 as b
import numpy as np
import pytest
from mlx_backend.brian_device import original_api
from mlx_backend.compiler import UnsupportedModelError
from original_api_reference import cpu_runtime, original_runtime
from test_mlx_compiled_integer_adapter import (_typed_fixture,_forbid_cpu_steps,
                                               private_compiler_cache)


@pytest.mark.metal
def test_dynamic_boxed_power_floating_store_cannot_use_integer_saturation(monkeypatch):
    types=dict(x=np.dtype('int64'),z=np.dtype('int64'),seen=np.dtype('float64'))
    initial=dict(x=np.array([-(1<<63),(1<<63)-1,(1<<53)+1,0],np.int64),
                 z=np.array([3,7,2,1],np.int64))
    reset='seen=abs(x)**z;x+=0'
    with cpu_runtime(target='cython', integer_dtype=np.int64):
        cpu=_typed_fixture(types,reset,initial);expected=cpu.run_ms(.1)
        assert expected.state['seen'][0]==float((1<<63)**3)
    with original_runtime(original_api, target='cython', integer_dtype=np.int64):
        native=_typed_fixture(types,reset,initial);_forbid_cpu_steps(native,monkeypatch)
        with pytest.raises(UnsupportedModelError,match='contextual magnitude policy'):
            native.run_ms(.1)


@pytest.mark.metal
def test_boxed_floating_operand_cannot_be_reinterpreted_as_integer_bits(monkeypatch):
    types=dict(x=np.dtype('int64'),seen=np.dtype('float64'))
    initial=dict(x=np.array([2,-3,5,0],np.int64),seen=np.array([.25,.5,1.,2.]))
    with cpu_runtime(target='cython', integer_dtype=np.int64):
        cpu=_typed_fixture(types,'seen=abs(x)+seen',initial)
        assert cpu.run_ms(.1).state['seen'][0]==2.25
    with original_runtime(original_api, target='cython', integer_dtype=np.int64):
        native=_typed_fixture(types,'seen=abs(x)+seen',initial);_forbid_cpu_steps(native,monkeypatch)
        with pytest.raises(UnsupportedModelError,match='exact mixed numeric lowering'):
            native.run_ms(.1)
