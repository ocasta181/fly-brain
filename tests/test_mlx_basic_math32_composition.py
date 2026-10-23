"""Selected-target basic function types remain visible to enclosing power."""
from native_execution import forbid_cpu_steps as guard_cpu_steps
import pytest
import test_mlx_floating_graph as graph


pytestmark=pytest.mark.metal


@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('expression',[
    'exp(x)','log1p(abs(x))','expm1(abs(x))','log(abs(x)+1)','exprel(x)'])
def test_basic_function_power_composition(tmp_path,monkeypatch,target,expression):
    graph.test_actual_all_float32_power_graph_and_continuation(
        tmp_path,monkeypatch,'y',target=target,base_expression=expression)


@pytest.mark.parametrize('base,exponent',[
    ('exp(x)+y','y'), ('abs(x)','exp(y)'),
    ('exp(x)','expm1(y)'), ('exp(x)','exp(y)')])
def test_cython_math_power_cast_context(tmp_path,monkeypatch,base,exponent):
    graph.test_actual_all_float32_power_graph_and_continuation(
        tmp_path,monkeypatch,exponent,target='cython',base_expression=base)


@pytest.mark.parametrize('base',[
    'clip(exp(x),1.34567891,2.34567891)',
    'clip(log1p(abs(x)),.123456789,.987654321)'])
def test_cython_math_fused_clip_power_context(tmp_path,monkeypatch,base):
    graph.test_actual_all_float32_power_graph_and_continuation(
        tmp_path,monkeypatch,'y',target='cython',base_expression=base)


@pytest.mark.parametrize('target',['numpy','cython'])
def test_power_composition_shared_and_vector_before_start(tmp_path,monkeypatch,target):
    from pathlib import Path
    import brian2 as b
    import numpy as np
    from mlx_backend.model import DEFAULT_EQUATIONS
    from original_api_protocol import check_original_api_runs, observe_run
    from original_api_reference import build_original_api_fixture

    monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(
        Path(__file__).resolve().parents[1]/'executions/cython-floating-operators'))
    equations=DEFAULT_EQUATIONS+'''x:1
y:1
s:1
z:1
p:1
scale=exp(t/ms)**0.3:1(shared,constant over dt)
held=scale*(exp(x)**y):1(constant over dt)
'''
    parameters=dict(eqs=equations,eq_th='True',
        eq_rst='v=v_rst;g=0*mV;p=s*held+z',t_rfc=0*b.ms)
    initial=dict(x=[.3,-.3,3.,-3.],y=[.1,-.1,.5,2.1],
        s=[1.0000001,.03,3.000001,-2.1],z=[-.7,.2,-4.,5.])
    def exercise(fixture, mode):
        for index,duration in enumerate((0.,.15,.15)):
            if index==2:fixture.neurons.x_[:]=[-.75,.75,1.125,2.25]
            yield observe_run(fixture,duration)
    def guard(fixture, patch):
        guard_cpu_steps(fixture.network, patch, 'A CPU timestep executed', patch_preparation=True)
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/str(mode == 'native'),
        parameters=parameters,initial_state=initial,exc=(),exc2=(),silenced=()),
        exercise,monkeypatch,guard=guard,target=target,float_dtype=np.float32)
