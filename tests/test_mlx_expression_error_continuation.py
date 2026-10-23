"""Actual expression exceptions preserve clock, draw prefix and repair."""
from native_execution import forbid_cpu_steps
from contextlib import nullcontext
import numpy as np
import pytest

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('stage,shared,invalid',[
    ('threshold',False,[-1.]),('threshold',False,[np.nan]),('threshold',False,[np.inf]),
    ('threshold',False,[-1.,np.inf]),('state',False,[-1.]),
    ('threshold',True,[np.nan]),('reset',True,[-1.]),
],ids=['original-negative','vector-nan','vector-inf','vector-precedence','state-before-input','scalar-nan','empty-reset-scalar'])
def test_numpy_poisson_expression_error_and_repair(tmp_path,monkeypatch,stage,shared,invalid):
    import brian2 as b
    from original_api_protocol import check_original_api_runs, observe_run
    from original_api_reference import build_original_api_fixture
    def build(mode):
        eqs=('dv/dt=(y*mV-v)/ms:volt'if stage=='state'else'v:volt')+'\ng:volt\nrfc:second\nlam:1'+(' (shared)'if shared else'')+'\nx=rand():1 (constant over dt)'
        if stage=='state':eqs+='\ny=poisson(lam):1 (constant over dt)'
        return build_original_api_fixture(tmp_path/str(mode == 'native'),parameters=dict(
            eqs=eqs,eq_th='poisson(lam)>0'if stage=='threshold'else'rand()<0'if stage=='reset'else'rand()<.7',
            eq_rst='v=v_rst+'+('poisson(lam)*mV+'if stage=='reset'else'')+'randn()*mV;g=0*mV',
            r_poi=3700*b.Hz,r_poi2=7200*b.Hz,t_dly=.2*b.ms),
            exc=(0,0)if stage=='state'else(),exc2=(2,)if stage=='state'else(),silenced=(),
            initial_state={'lam':1. if shared else[1.,2.,3.,4.]})
    def exercise(fixture, mode):
        yield observe_run(fixture,.3,rng=True)
        if shared:fixture.neurons.lam=invalid[0]
        else:fixture.neurons.lam[1:1+len(invalid)]=invalid
        failed=observe_run(fixture,.4,raises=b.BrianObjectException,rng=True)
        assert type(failed.error.__cause__)is ValueError
        yield failed
        if shared:fixture.neurons.lam=1.
        else:fixture.neurons.lam=[1.,2.,3.,4.]
        yield observe_run(fixture,.4,rng=True)
    check_original_api_runs(build,exercise,monkeypatch,seed=731)


@pytest.mark.parametrize('phase',['reset','threshold'])
def test_original_cython_expression_fault_retains_completed_rows_and_pending_queue(tmp_path,monkeypatch,phase,shared=False,scalar_den=False,dtype=np.float64,time_varying=False):
    import brian2 as b
    from original_api_protocol import check_original_api_runs, observe_run
    from original_api_reference import build_original_api_fixture
    def build(mode):
        parameters=dict(
            eqs=('dv/dt=(s*mV/den-v)/ms:volt'if phase=='state'else'v:volt')+'\ng:volt\nrfc:second\na:1\nden:1'+(' (shared)'if scalar_den else'')+('\ns=rand():1 (shared, constant over dt)'if shared else''),
            eq_th=('(s+rand())/den>.2'if shared else'rand()/den>.2')if phase=='threshold'else'rand()<1',
            eq_rst=('a=rand()/den;'if phase=='reset'else'a=rand();')+'v=v_rst+randn()*mV;g=0*mV',
            t_rfc=0*b.ms,r_poi=3700*b.Hz,r_poi2=7200*b.Hz,t_dly=.2*b.ms)
        if time_varying:
            parameters['drive']=b.TimedArray([0.,.5,-.25,.75,1.,-.125],dt=.2*b.ms)
            parameters['eqs']=parameters['eqs'].replace('s*mV','(s+drive(t))*mV')
            parameters['eq_th']=parameters['eq_th'].replace('(s+rand())','(s+rand()+drive(t))')
        return build_original_api_fixture(tmp_path/str(mode == 'native'),parameters=parameters,
            silenced=(),initial_state={'den':1. if scalar_den else[1.,1.,1.,1.]})
    def exercise(fixture, mode):
        yield observe_run(fixture,.2,rng=True)
        if scalar_den:fixture.neurons.den=0
        else:fixture.neurons.den[1]=0
        failed=observe_run(fixture,.3,raises=b.BrianObjectException,rng=True)
        assert type(failed.error.__cause__)is ZeroDivisionError
        yield failed
        if scalar_den:fixture.neurons.den=1
        else:fixture.neurons.den[1]=1
        yield observe_run(fixture,.3,rng=True)
    check_original_api_runs(build,exercise,monkeypatch,seed=731,target='cython',float_dtype=dtype)


@pytest.mark.parametrize('target',['numpy','cython'])
def test_nested_poisson_liveness_and_original_shared_stream_continuation(tmp_path,monkeypatch,target,dtype=np.float64,time_varying=False):
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from original_api_reference import build_original_api_fixture,cpu_runtime,capture_rng_state,assert_snapshot_equal,assert_rng_state_equal
    for native in(False,True):
        with cpu_runtime(seed=941):
            b.prefs.codegen.target=target
            b.prefs.core.default_float_dtype=dtype
            with original_api(workers=False)if native else nullcontext():
                parameters=dict(
                    eqs='v:volt\ng:volt\nrfc:second\na:1\nb:1\nlam:1\ns=rand():1 (shared, constant over dt)\ny=poisson(1+s):1 (constant over dt)\nx=poisson(lam+y+rand()):1 (constant over dt)',
                    eq_th='poisson(x+rand())>1',
                    eq_rst='a=poisson(2+rand());b=poisson(a+x)+randn();v=v_rst+b*mV;g=0*mV',
                    t_rfc=0*b.ms,r_poi=3700*b.Hz,r_poi2=7200*b.Hz,t_dly=.2*b.ms)
                if time_varying:
                    parameters.update(drive=b.TimedArray([0.,.5,.25,.75],dt=.2*b.ms),
                        offset=b.TimedArray([[0.,.25],[.5,.125],[.75,0.]],dt=.1*b.ms))
                    parameters['eqs']=parameters['eqs'].replace('s=rand()', 's=rand()+drive(t)').replace('poisson(1+s)', 'poisson(1+s+offset(t,i%2))')
                    parameters['eq_rst']=parameters['eq_rst'].replace('poisson(2+rand())','poisson(2+rand()+drive(t))')
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=parameters,
                    exc=(0,0),exc2=(2,),silenced=(),initial_state={'lam':[0.,1.,3.,10.]})
                if native:
                    forbid_cpu_steps(fixture.network, monkeypatch)
                states=[]
                for duration in(.2,0.,.3):
                    fixture.run_ms(duration);states.append((fixture.capture(),capture_rng_state()))
                    fixture.neurons.lam=[2.,5.,0.,1.]
                if native:
                    for index,((actual,rng),(expected,reference_rng))in enumerate(zip(states,references)):
                        try:
                            assert_rng_state_equal(rng,reference_rng)
                            assert_snapshot_equal(actual,expected)
                        except AssertionError as error:
                            raise AssertionError(f'boundary {index}, scalar sequence '+str([(float(a.state['s'][0]),float(e.state['s'][0]))for (a,_),(e,_)in zip(states,references)]))from error
                else:references=states


@pytest.mark.parametrize('target',['numpy','cython'])
def test_float32_nested_draw_temporaries_and_original_inputs(tmp_path,monkeypatch,target):
    test_nested_poisson_liveness_and_original_shared_stream_continuation(tmp_path,monkeypatch,target,dtype=np.float32)


@pytest.mark.parametrize('dtype',[np.float64,np.float32],ids=['float64','float32'])
def test_original_shared_draw_checked_cython_threshold(tmp_path,monkeypatch,dtype):
    test_original_cython_expression_fault_retains_completed_rows_and_pending_queue(
        tmp_path,monkeypatch,'threshold',shared=True,dtype=dtype)


@pytest.mark.parametrize('dtype',[np.float64,np.float32],ids=['float64','float32'])
def test_original_shared_draw_checked_cython_state(tmp_path,monkeypatch,dtype):
    test_original_cython_expression_fault_retains_completed_rows_and_pending_queue(
        tmp_path,monkeypatch,'state',shared=True,dtype=dtype)


@pytest.mark.parametrize('dtype',[np.float64,np.float32],ids=['float64','float32'])
def test_original_shared_draw_checked_cython_scalar_preloop(tmp_path,monkeypatch,dtype):
    test_original_cython_expression_fault_retains_completed_rows_and_pending_queue(
        tmp_path,monkeypatch,'state',shared=True,scalar_den=True,dtype=dtype)


@pytest.mark.parametrize('phase',['state','threshold','scalar_preloop'])
def test_original_timed_shared_draw_fault_and_repair(tmp_path,monkeypatch,phase):
    test_original_cython_expression_fault_retains_completed_rows_and_pending_queue(
        tmp_path,monkeypatch,'state'if phase=='scalar_preloop'else phase,
        shared=True,scalar_den=phase=='scalar_preloop',time_varying=True)


@pytest.mark.parametrize('target',['numpy','cython'])
def test_two_timedarrays_nested_shared_draws_and_original_inputs(tmp_path,monkeypatch,target):
    test_nested_poisson_liveness_and_original_shared_stream_continuation(
        tmp_path,monkeypatch,target,time_varying=True)


@pytest.mark.parametrize('target,conditional',[('numpy',False),('cython',False),('cython',True)],
    ids=['numpy-selected','cython-selected','cython-conditional'])
def test_two_timedarrays_selected_and_conditional_draws(tmp_path,monkeypatch,target,conditional):
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from mlx_backend import expression_numpy_packed,expression_single_call
    from original_api_reference import build_original_api_fixture,original_runtime,capture_rng_state,assert_snapshot_equal,assert_rng_state_equal
    calls=[]
    module=expression_numpy_packed if target=='numpy'else expression_single_call
    name='stage'if conditional else'advance'if target=='numpy'else'advance_packed'
    original=getattr(module,name)
    def counted(*args,**kwargs):
        calls.append(True)
        return original(*args,**kwargs)
    monkeypatch.setattr(module,name,counted)
    for native in(False,True):
        with original_runtime(original_api, seed=941, target=target, native=native):
            fixture=build_original_api_fixture(tmp_path/str(native),parameters=dict(
                eqs='v:volt\ng:volt\nrfc:second\na:1',
                eq_th='(i<2) and (rand()<.7)'if conditional else'rand()<.7',
                eq_rst='a=randn()+drive(t+i*dt)+offset(t-i*dt);v=v_rst+a*mV;g=0*mV',
                drive=b.TimedArray([0.,.5,-.25,.75],dt=.2*b.ms),
                offset=b.TimedArray([.125,-.5,1.],dt=.1*b.ms),
                t_rfc=0*b.ms,r_poi=3700*b.Hz,r_poi2=7200*b.Hz,t_dly=.2*b.ms),
                silenced=())
            if native:
                forbid_cpu_steps(fixture.network, monkeypatch)
            states=[]
            for duration in(.2,0.,.3):
                fixture.run_ms(duration);states.append((fixture.capture(),capture_rng_state()))
            if native:
                for(actual,rng),(expected,reference_rng)in zip(states,references):
                    assert_snapshot_equal(actual,expected);assert_rng_state_equal(rng,reference_rng)
            else:references=states
    assert calls, 'The affected native consumer must execute'
