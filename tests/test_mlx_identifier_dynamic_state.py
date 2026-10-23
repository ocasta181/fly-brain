"""Accepted type-like state names also work in differential state updates."""
from native_execution import forbid_cpu_steps as guard_cpu_steps, require_metal

import numpy as np
import pytest


@pytest.mark.metal
@pytest.mark.parametrize('target,dtype',[('numpy',np.float64),('cython',np.float32)])
def test_actual_keyword_state_dynamics_with_namespace_constant_and_poisson(tmp_path,monkeypatch,target,dtype):
    import brian2 as b
    from original_api_protocol import check_original_api_runs, observe_run
    from original_api_reference import build_original_api_fixture
    mx = require_metal('Metal required')
    def build(mode):
        fixture=build_original_api_fixture(tmp_path/str(mode == 'native'),
            parameters=dict(eqs='v:volt\ng:volt\nrfc:second\n'
                'dfloat2/dt=-thread*float2/second:1\nswitch:boolean\n',
                thread=.25,eq_th='(i<2) and (float2>0)',
                eq_rst='v=v_rst;switch=not switch',t_rfc=0*b.ms,
                r_poi=4300*b.Hz,r_poi2=2800*b.Hz,t_dly=.3*b.ms),
            initial_state=dict(float2=np.array([.1,.2,-.3,.4],dtype),
                switch=[False,True,True,False]),exc=(0,0),exc2=(2,),silenced=())
        assert fixture.neurons.variables['float2'].dtype == np.dtype(dtype)
        return fixture
    def guard(fixture, patch):
        guard_cpu_steps(fixture.network, patch, 'Native identifier dynamics ran CPU timesteps', patch_preparation=True)
    def exercise(fixture, mode):
        for duration in (.1,.2,.15):
            yield observe_run(fixture,duration,rng=True)
        (tmp_path/str(mode == 'native')/'state-source.txt').write_text(
            str(fixture.neurons.state_updater.codeobj.code))
    check_original_api_runs(build,exercise,monkeypatch,guard=guard,seed=3189,
        target=target,float_dtype=dtype)
