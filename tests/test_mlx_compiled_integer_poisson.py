"""Checked integer errors compose with native fused Poisson and RNG recovery."""
import brian2 as b
import numpy as np
import pytest
from original_api_reference import build_original_api_fixture
from test_mlx_compiled_integer_adapter import _forbid_cpu_steps,private_compiler_cache


@pytest.mark.metal
def test_actual_cython_fused_integer_error_rng_and_repaired_continuation(tmp_path,monkeypatch):
    from mlx_backend.model import DEFAULT_EQUATIONS,DEFAULT_RESET
    from mlx_backend.brian_poisson import NativePoisson
    parameters=dict(eqs=DEFAULT_EQUATIONS+'x:integer\n',eq_th='i<2',
        eq_rst=DEFAULT_RESET+';x=abs(x)',t_rfc=0*b.ms,t_dly=.1*b.ms,
        r_poi=4900*b.Hz,r_poi2=3200*b.Hz)
    options=dict(parameters=parameters,exc=(0,1,1),exc2=(1,),silenced=(),
        initial_state=dict(x=np.array([9,-(1<<63),7,6],np.int64),g=np.array([.01,.02,.03,.04])*b.volt))

    from original_api_protocol import check_original_api_runs, observe_run
    following=[]
    fused=[];original_can_fuse=NativePoisson.can_fuse
    def can_fuse(self,prepared):
        selected=original_can_fuse(self,prepared);fused.append(selected);return selected

    def build(mode):
        if mode=='native':monkeypatch.setattr(NativePoisson,'can_fuse',can_fuse)
        return build_original_api_fixture(tmp_path/mode,**options)
    def exercise(fixture, mode):
        failed=observe_run(fixture,.3,raises=b.BrianObjectException,rng=True)
        if mode=='cpu':assert isinstance(failed.error.__cause__,OverflowError)
        yield failed
        fixture.neurons.x[1]=3
        yield observe_run(fixture,.3,rng=True)
        values=np.random.random(10)
        if mode=='cpu':following.append(values)
        else:np.testing.assert_array_equal(values,following[0])
    check_original_api_runs(build,exercise,monkeypatch,guard=_forbid_cpu_steps,
        target='cython',integer_dtype=np.int64)
    assert fused and all(fused)
