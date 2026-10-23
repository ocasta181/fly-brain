"""Model expressions and original PoissonInput share one native stream."""
from native_execution import forbid_cpu_steps
import numpy as np
import pytest

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('delay_ms',[0.,.3])
def test_original_inputs_interleave_with_expression_rng_and_continuation(tmp_path,monkeypatch,target,delay_ms):
    import brian2 as b
    from original_api_reference import build_original_api_fixture
    from original_api_protocol import check_original_api_runs, observe_run
    parameters=dict(eq_th='rand()<.4',eq_rst='v=v_rst+randn()*mV;g=rand()*mV',
        r_poi=3700*b.Hz,r_poi2=7200*b.Hz,t_dly=delay_ms*b.ms)
    durations=(0.,.15,.05,.3)

    future=[]
    def build(mode):
        return build_original_api_fixture(tmp_path/('cpu' if mode=='cpu' else 'metal'),parameters=parameters)
    def guard(fixture, patches):
        forbid_cpu_steps(fixture.network, patches, 'CPU model/input timestep executed', patch_preparation=True)
    def exercise(fixture, mode):
        for duration in durations:
            yield observe_run(fixture,duration,rng=True)
        values=np.random.randint(0,2**32,624,dtype=np.uint32)
        if mode=='cpu':future.append(values)
        else:
            from brian2.devices.device import get_device
            np.testing.assert_array_equal(values,future[0])
            assert get_device().native_runs==len(durations)
    check_original_api_runs(build,exercise,monkeypatch,guard=guard,seed=875,dt_ms=.1,target=target)


@pytest.mark.parametrize('delay_ms,first',[(.2,False),(0.,False),(0.,True)])
def test_invalid_input_after_expression_draws_preserves_repair_continuation(tmp_path,monkeypatch,delay_ms,first):
    import brian2 as b
    from original_api_reference import build_original_api_fixture
    from original_api_protocol import check_original_api_runs, observe_run

    future=[]
    def build(mode):
        fixture=build_original_api_fixture(tmp_path/str(mode=='native'),parameters=dict(
            eq_th='rand()<.7',eq_rst='v=v_rst+randn()*mV;g=rand()*mV',
            r_poi=(12000 if first else 3700)*b.Hz,r_poi2=(3700 if first else 12000)*b.Hz,t_dly=delay_ms*b.ms),exc=(0,),silenced=())
        return fixture

    def guard(fixture, patches):
        forbid_cpu_steps(fixture.network, patches, 'CPU model/input timestep executed', patch_preparation=True)
    def exercise(fixture, mode):
        invalid=fixture.inputs[0 if first else -1]
        yield observe_run(fixture,0,rng=True)
        invalid.active=False
        yield observe_run(fixture,.3,rng=True)
        invalid.active=True
        failed=observe_run(fixture,.4,raises=b.BrianObjectException,rng=True)
        assert type(failed.error.__cause__)is ValueError
        assert str(failed.error.__cause__)=='p < 0, p > 1 or p is NaN'
        assert failed.error._brian_objname==invalid.codeobj.owner.name
        yield failed
        invalid.active=False
        for duration in(.1,.3,.1):yield observe_run(fixture,duration,rng=True)
        values=np.random.randint(0,2**32,624,dtype=np.uint32)
        if mode=='cpu':future.append(values)
        else:np.testing.assert_array_equal(values,future[0])
    check_original_api_runs(build,exercise,monkeypatch,guard=guard,seed=731)


def test_checked_state_failure_precedes_threshold_and_original_input_draws(tmp_path,monkeypatch):
    import brian2 as b
    from original_api_reference import build_original_api_fixture
    from original_api_protocol import check_original_api_runs, observe_run

    import mlx_backend.expression_general as general
    prefix=general._advance_prefix
    def guarded_prefix(p,*args,**kwargs):
        assert general._prefix(p.program) is None
        return prefix(p,*args,**kwargs)
    monkeypatch.setattr(general,'_advance_prefix',guarded_prefix)

    def build(mode):
        fixture=build_original_api_fixture(tmp_path/str(mode=='native'),parameters=dict(
            eqs='dv/dt=(g/den-v)/ms:volt\ng:volt\nrfc:second\nden:integer',
            eq_th='rand()<.7',eq_rst='v=v_rst+randn()*mV;g=0*mV',t_rfc=0*b.ms,
            r_poi=3700*b.Hz,r_poi2=7200*b.Hz,t_dly=.2*b.ms),silenced=(),initial_state={'den':[1,1,1,1],'g':b.mV})
        return fixture

    def guard(fixture, patches):
        forbid_cpu_steps(fixture.network, patches, 'CPU model/input timestep executed')
    def exercise(fixture, mode):
        yield observe_run(fixture,.2,rng=True)
        fixture.neurons.den[1]=0
        failed=observe_run(fixture,.3,raises=b.BrianObjectException,rng=True)
        assert type(failed.error.__cause__)is ZeroDivisionError
        yield failed
        fixture.neurons.den[1]=1
        yield observe_run(fixture,.3,rng=True)
    check_original_api_runs(build,exercise,monkeypatch,guard=guard,seed=731,target='cython')
