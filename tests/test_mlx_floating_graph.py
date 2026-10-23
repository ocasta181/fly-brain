"""Actual default Cython/CPP typed power graph versus native original objects."""
from native_execution import forbid_model_cpu_steps
from contextlib import nullcontext
from pathlib import Path

import brian2 as b
import numpy as np
import pytest

from mlx_backend.brian_device import original_api
from mlx_backend.model import DEFAULT_EQUATIONS
from original_api_reference import build_original_api_fixture,cpu_runtime,assert_snapshot_equal
from original_api_protocol import RunObservation, check_original_api_runs, observe_run, set_floating_operator_cache


pytestmark=pytest.mark.metal


@pytest.mark.parametrize('exponent',['y','k','0.5','3','4','-3','31'])
def test_actual_all_float32_power_graph_and_continuation(tmp_path,monkeypatch,exponent,target='cython',base_expression='x'):
    set_floating_operator_cache(monkeypatch, b.prefs)
    parameters=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\ns:1\nz:1\np:1\nk:integer\n',
        eq_th='True',eq_rst=f'v=v_rst;p=s*(({base_expression})**({exponent}))+z',t_rfc=0*b.ms)
    state=dict(x=np.array([.3,-.3,3.,-3.],np.float32),y=[.1,-.1,.5,2.1],
        s=np.array([1.0000001,.03,3.000001,-2.1],np.float32),z=[-.7,.2,-4.,5.],k=[3,-3,4,-2])
    def exercise(fixture, mode):
        for index,duration in enumerate((0.,.15,.15)):
            if index==2:fixture.neurons.x_[:]=[-.75,.75,1.125,2.25]
            yield observe_run(fixture,duration)
            if mode == 'native':
                assert b.get_device()._sessions[fixture.network].prepared.model.state_dtype=='float32'
    def guard(fixture, patch):
        forbid_model_cpu_steps(fixture, patch, 'A CPU timestep executed', patch_preparation=True)
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/str(mode == 'native'),
        parameters=parameters,initial_state=state,exc=(),exc2=(),silenced=()),
        exercise,monkeypatch,guard=guard,target=target,float_dtype=np.float32)


def test_actual_all_float32_numpy_dynamic_power_and_continuation(tmp_path,monkeypatch):
    test_actual_all_float32_power_graph_and_continuation(tmp_path,monkeypatch,'y',target='numpy')


def test_actual_all_float32_cython_literal_base_power_and_continuation(tmp_path,monkeypatch):
    test_actual_all_float32_power_graph_and_continuation(
        tmp_path,monkeypatch,'y',base_expression='2.0')


@pytest.mark.parametrize('exponent',['k','0.5','3'])
def test_actual_all_float32_numpy_scalar_power_and_continuation(tmp_path,monkeypatch,exponent):
    test_actual_all_float32_power_graph_and_continuation(tmp_path,monkeypatch,exponent,target='numpy')


@pytest.mark.parametrize('expression',['x%y','x//y'])
def test_actual_all_float32_remainder_graph_context(tmp_path,monkeypatch,expression,target='cython'):
    set_floating_operator_cache(monkeypatch, b.prefs)
    parameters=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\ns:1\nz:1\np:1\n',
        eq_th='True',eq_rst=f'v=v_rst;p=s*({expression})+z',t_rfc=0*b.ms)
    initial=dict(x=np.array([.3,-.3,3.,-3.],np.float32),y=[.1,-.1,.7,2.1],
        s=[1.1,1.3,.1,1.7],z=[-.3,-1.,-.1,-.123])
    def exercise(fixture,mode):
        for index,duration in enumerate((.15,.15)):
            if index:fixture.neurons.x_[:]=[-.75,.75,1.125,2.25]
            yield observe_run(fixture,duration)
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/str(mode=='native'),
        parameters=parameters,initial_state=initial,exc=(),exc2=(),silenced=()),
        exercise,monkeypatch,guard=lambda fixture,patch:None,target=target,float_dtype=np.float32)


@pytest.mark.parametrize('expression',['x%y','x//y'])
def test_actual_all_float32_numpy_remainder_context(tmp_path,monkeypatch,expression):
    test_actual_all_float32_remainder_graph_context(tmp_path,monkeypatch,expression,target='numpy')


@pytest.mark.parametrize('target',['numpy','cython'])
def test_actual_all_float32_augmented_modulo_and_continuation(tmp_path,monkeypatch,target,default_dtype=np.float32):
    set_floating_operator_cache(monkeypatch, b.prefs)
    parameters=dict(eqs=DEFAULT_EQUATIONS+'p:1\nr:1\nq:1\ny:1\n',
        eq_th='True',eq_rst='v=v_rst;p=p**y;r%=y;q=q//y',t_rfc=0*b.ms)
    initial=dict(p=[.3,.75,1.125,2.25],r=[.3,-.3,3.,-3.],q=[.3,-.3,3.,-3.],
        y=[.1,-.1,.7,2.1])
    def exercise(fixture,mode):
        for index,duration in enumerate((.15,.15)):
            if index:fixture.neurons.y_[:]=[.2,-.2,1.3,-2.3]
            yield observe_run(fixture,duration)
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/str(mode=='native'),
        parameters=parameters,initial_state=initial,exc=(),exc2=(),silenced=()),
        exercise,monkeypatch,guard=lambda fixture,patch:None,target=target,float_dtype=default_dtype)


def test_actual_binary64_augmented_modulo_and_continuation(tmp_path,monkeypatch):
    test_actual_all_float32_augmented_modulo_and_continuation(
        tmp_path,monkeypatch,'cython',default_dtype=np.float64)


@pytest.mark.parametrize('exponent',[3,-3])
def test_actual_all_float32_cython_immutable_power_preparation(tmp_path,monkeypatch,exponent):
    set_floating_operator_cache(monkeypatch, b.prefs)
    parameters=dict(eqs=DEFAULT_EQUATIONS+'x:1\np:1\n',eq_th='True',
        eq_rst=f'v=v_rst;p=x*(constant**({exponent}))',constant=np.float32(.3),t_rfc=0*b.ms)
    def exercise(fixture,mode):
        for index,duration in enumerate((0.,.15,.15)):
            if index==2:fixture.neurons.x_[:]=[.75,1.125,2.25,3.3]
            yield observe_run(fixture,duration)
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/str(mode=='native'),
        parameters=parameters,initial_state=dict(x=[.3,-.3,3.,-3.]),exc=(),exc2=(),silenced=()),
        exercise,monkeypatch,guard=lambda fixture,patch:None,target='cython',float_dtype=np.float32)


def test_actual_binary64_cython_two_product_operator_context(tmp_path,monkeypatch,exponent='y',base_expression='x'):
    import os,json
    audit=os.environ.get('MLX_FLOATING_GRAPH_AUDIT')
    if audit:
        from mlx_backend import cpu_arithmetic
        audit=Path(audit);audit.mkdir(parents=True,exist_ok=True)
        optimize=cpu_arithmetic._optimize
        def capture_source(source,identity):
            result=optimize(source,identity)
            (audit/'native-source.cpp').write_text(source)
            (audit/'native.ll').write_text(result)
            return result
        monkeypatch.setattr(cpu_arithmetic,'_optimize',capture_source)
    set_floating_operator_cache(monkeypatch, b.prefs)
    parameters=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\ns:1\nz:1\nh:1\np:1\nr:1\nq:1\n',
        eq_th='True',eq_rst=f'v=v_rst;p=s*(({base_expression})**({exponent}))+z*h;r=s*(x%y)+z*h;q=s*(x//y)+z*h',t_rfc=0*b.ms)
    initial=dict(x=[.3,3.,1.3,.1],y=[.1,.3,.7,2.1],s=[1.1,1.3,.1,1.7],
        z=[-.3,-1.,-.1,-.123],h=[-.77,.33,1.2,-1.9])
    expected=[]
    for native in (False,True):
        with cpu_runtime():
            b.prefs.codegen.target='cython'
            with original_api(workers=False) if native else nullcontext() as device:
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=parameters,
                    initial_state=initial,exc=(),exc2=(),silenced=())
                if native:
                    forbid_model_cpu_steps(fixture, monkeypatch, 'Native binary64 graph used a CPU timestep', patch_preparation=True)
                for index,duration in enumerate((0.,.15,.15)):
                    if index==2:fixture.neurons.x_[:]=[.75,1.125,2.25,3.3]
                    observed=fixture.run_ms(duration)
                    if audit and index==1:
                        runner=fixture.neurons.resetter['spike'].codeobj
                        if native:
                            program=device._sessions[fixture.network].prepared.program
                            (audit/'native.metal').write_text(program.header_code+'\n'+program.reset_vector_code)
                            for number,policy in enumerate(program.operator_policies):
                                (audit/f'policy-{number}.cpp').write_text(policy.source)
                                (audit/f'policy-{number}.s').write_text(policy.assembly)
                        else:
                            (audit/'actual.pyx').write_text(str(runner.code.run))
                            (audit/'actual.json').write_text(json.dumps(dict(
                                module=runner.compiled_code['run'].__file__,
                                flags=runner.extra_compile_args),indent=2))
                    if native:assert_snapshot_equal(observed,expected[index])
                    else:expected.append(observed)


def test_actual_binary64_cython_cube_two_product_context(tmp_path,monkeypatch):
    test_actual_binary64_cython_two_product_operator_context(tmp_path,monkeypatch,exponent='3')


@pytest.mark.parametrize('base_expression',['sin(x)','exp(x)','log1p(abs(x))','2.0'])
def test_actual_binary64_power_function_composition(tmp_path,monkeypatch,base_expression):
    test_actual_binary64_cython_two_product_operator_context(
        tmp_path,monkeypatch,base_expression=base_expression)


@pytest.mark.parametrize('single_names',[('x','s'),('y','z','h')],ids=['single-base','single-addend'])
def test_actual_mixed_storage_two_product_graph(monkeypatch,single_names):
    from original_api_reference import load_original_model,OriginalAPIFixture
    set_floating_operator_cache(monkeypatch, b.prefs)
    def build():
        original=load_original_model();params=dict(original.default_params)
        params.update(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\ns:1\nz:1\nh:1\np:1\nr:1\nq:1\n',
            eq_th='True',eq_rst='v=v_rst;p=s*(x**y)+z*h;r=s*(x%y)+z*h;q=s*(x//y)+z*h',t_rfc=0*b.ms)
        group=b.NeuronGroup(4,params['eqs'],threshold=params['eq_th'],reset=params['eq_rst'],
            refractory='rfc',method='linear',namespace=params,
            dtype={name:np.float32 for name in single_names})
        group.v=params['v_0'];group.g=0*b.volt;group.rfc=0*b.ms
        for name,values in dict(x=[.3,3.,1.3,.1],y=[.1,.3,.7,2.1],s=[1.1,1.3,.1,1.7],
                               z=[-.3,-1.,-.1,-.123],h=[-.77,.33,1.2,-1.9]).items():
            getattr(group,name+'_')[:]=values
        syn=b.Synapses(group,group,'w:volt',on_pre='g_post+=w',delay=params['t_dly'])
        syn.connect(i=[0],j=[0]);syn.w=0*b.volt
        monitor=b.SpikeMonitor(group)
        return OriginalAPIFixture(original,params,(),group,syn,monitor,[],b.Network(group,syn,monitor))
    def exercise(fixture, mode):
        for index,duration in enumerate((0.,.15,.15)):
            if index==2:fixture.neurons.x_[:]=[.75,1.125,2.25,3.3]
            yield observe_run(fixture,duration)
    def guard(fixture, patch):
        forbid_model_cpu_steps(fixture, patch, 'A CPU timestep executed', patch_preparation=True)
    check_original_api_runs(lambda mode:build(),exercise,monkeypatch,guard=guard,target='cython')


@pytest.mark.parametrize('dimensions',[1,2])
def test_actual_timed_array_operator_graph_and_continuation(tmp_path,monkeypatch,dimensions,default_dtype=np.float64,poisson=False):
    from original_api_reference import capture_rng_state,assert_rng_state_equal
    set_floating_operator_cache(monkeypatch, b.prefs)
    expected=[];expected_rng=[];next_random=None
    for native in (False,True):
        with cpu_runtime():
            b.prefs.codegen.target='cython';b.prefs.core.default_float_dtype=default_dtype
            data=(np.array([.3,3.,1.3,.1]) if dimensions==1 else
                  np.arange(16,dtype=np.float64).reshape(4,4)/10+.3)
            drive=b.TimedArray(data,dt=.1*b.ms)
            call='drive(t)' if dimensions==1 else 'drive(t,i)'
            params=dict(eqs=DEFAULT_EQUATIONS+'y:1\np:1\nr:1\nq:1\n',
                eq_th=f'{call}**y>.5' if poisson else 'True',
                eq_rst=f'v=v_rst;p={call}**y;r={call}%y;q={call}//y',
                drive=drive,t_rfc=0*b.ms,r_poi=4900*b.Hz,r_poi2=3200*b.Hz)
            with original_api(workers=False) if native else nullcontext():
                initial=dict(y=[.1,.3,.7,2.1])
                if poisson:initial['rfc']=np.full(4,.15)*b.ms
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=params,
                    initial_state=initial,exc=(0,0) if poisson else (),
                    exc2=(2,) if poisson else (),silenced=())
                if native:
                    forbid_model_cpu_steps(fixture, monkeypatch, 'A CPU timestep executed', patch_preparation=True)
                for index,duration in enumerate((0.,.15,.15)):
                    if index==2:drive.values[2:]+=.25
                    observed=fixture.run_ms(duration)
                    if native:
                        assert_snapshot_equal(observed,expected[index])
                        if poisson:assert_rng_state_equal(capture_rng_state(),expected_rng[index])
                    else:
                        expected.append(observed)
                        if poisson:expected_rng.append(capture_rng_state())
                if poisson:
                    if native:np.testing.assert_array_equal(np.random.random(10),next_random)
                    else:next_random=np.random.random(10)


@pytest.mark.parametrize('dimensions',[1,2])
def test_actual_all_float32_timed_array_operator_continuation(tmp_path,monkeypatch,dimensions):
    test_actual_timed_array_operator_graph_and_continuation(
        tmp_path,monkeypatch,dimensions,default_dtype=np.float32)


@pytest.mark.parametrize('default_dtype',[np.float64,np.float32])
def test_timed_power_threshold_and_reset_preserve_poisson_state(tmp_path,monkeypatch,default_dtype):
    test_actual_timed_array_operator_graph_and_continuation(
        tmp_path,monkeypatch,2,default_dtype=default_dtype,poisson=True)


@pytest.mark.parametrize('expression',['x%y+x//z','x//z+x%y'])
def test_binary64_first_operator_error_follows_actual_cython_order(tmp_path,monkeypatch,expression):
    set_floating_operator_cache(monkeypatch, b.prefs)
    parameters=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\nz:1\np:1\n',
        eq_th='True',eq_rst='v=v_rst;p='+expression,t_rfc=0*b.ms)
    cause=None
    def exercise(fixture,mode):
        nonlocal cause
        with pytest.raises(b.BrianObjectException) as error:fixture.run_ms(.1)
        if mode=='native':
            assert str(error.value.__cause__)==cause
        else:cause=str(error.value.__cause__)
        yield RunObservation(fixture.capture())
        fixture.neurons.y_[:]=[.1,.2,.3,.4];fixture.neurons.z_[:]=[.2,.3,.4,.5]
        yield observe_run(fixture,.15)
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/str(mode=='native'),
        parameters=parameters,initial_state=dict(x=[.3,.7,1.1,1.7],y=np.zeros(4),z=np.zeros(4)),
        exc=(),exc2=(),silenced=()),exercise,monkeypatch,guard=lambda fixture,patch:None,target='cython')


def test_binary64_scalar_reset_error_occurs_without_spikes(tmp_path,monkeypatch,parameter_dtype=np.float64,default_dtype=np.float64):
    set_floating_operator_cache(monkeypatch, b.prefs)
    parameters=dict(eqs=DEFAULT_EQUATIONS+'p:1\n',eq_th='False',
        eq_rst='v=v_rst;p=numerator//denominator',numerator=parameter_dtype(.3),
        denominator=parameter_dtype(0),t_rfc=0*b.ms)
    cause=None
    def exercise(fixture,mode):
        nonlocal cause
        zero=fixture.run_ms(0.)
        with pytest.raises(b.BrianObjectException) as error:fixture.run_ms(.1)
        if mode=='native':
            assert str(error.value.__cause__)==cause
        else:cause=str(error.value.__cause__)
        yield RunObservation(zero)
        yield RunObservation(fixture.capture())
        fixture.neurons.namespace['denominator']=parameter_dtype(.1)
        yield observe_run(fixture,.15)
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/str(mode=='native'),
        parameters=parameters,exc=(),exc2=(),silenced=()),exercise,monkeypatch,
        guard=lambda fixture,patch:None,target='cython',float_dtype=default_dtype)


@pytest.mark.parametrize('default_dtype',[np.float64,np.float32])
def test_typed_scalar_reset_error_occurs_without_spikes(tmp_path,monkeypatch,default_dtype):
    test_binary64_scalar_reset_error_occurs_without_spikes(
        tmp_path,monkeypatch,parameter_dtype=np.float32,default_dtype=default_dtype)


def test_actual_all_float32_operator_error_and_repaired_continuation(tmp_path,monkeypatch):
    from test_mlx_floating_schedule_errors import test_cython_operator_failure_and_resume_follow_actual_schedule
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    test_cython_operator_failure_and_resume_follow_actual_schedule(tmp_path,monkeypatch,
        phases=('groups','thresholds','resets','synapses'),delay_ms=.1)


def test_all_float32_hoisted_state_operator_error_precedes_conditional_write(tmp_path,monkeypatch):
    set_floating_operator_cache(monkeypatch, b.prefs)
    parameters=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\ndp/dt=(-p+x%y)/second :1 (unless refractory)\n',
        eq_th='False',eq_rst='v=v_rst',t_rfc=1*b.ms)
    cause=None
    def exercise(fixture,mode):
        nonlocal cause
        # Brian hoists the modulo into an auxiliary assignment before
        # the conditional persistent write, so it raises even though
        # every neuron is refractory. Follow its actual stage order.
        with pytest.raises(b.BrianObjectException) as error:fixture.run_ms(.1)
        if mode=='native':
            assert str(error.value.__cause__)==cause
        else:cause=str(error.value.__cause__)
        yield RunObservation(fixture.capture())
        fixture.neurons.rfc_=0.
        fixture.neurons.y_[:]=[.1,.2,.3,.4]
        yield observe_run(fixture,.1)
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/str(mode=='native'),
        parameters=parameters,initial_state=dict(x=[.3,.7,1.1,1.7],y=[0.,0.,0.,0.],p=[1.,2.,3.,4.],
            lastspike=np.zeros(4)*b.second),exc=(),exc2=(),silenced=()),
        exercise,monkeypatch,guard=lambda fixture,patch:None,target='cython',float_dtype=np.float32)


def test_all_float32_threshold_error_retains_runtime_short_circuit(tmp_path,monkeypatch):
    set_floating_operator_cache(monkeypatch, b.prefs)
    parameters=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\n',
        eq_th='(t>=dt) and (x%y>=0)',eq_rst='v=v_rst',t_rfc=0*b.ms)
    cause=None
    def exercise(fixture,mode):
        nonlocal cause
        first=fixture.run_ms(.1)
        with pytest.raises(b.BrianObjectException) as error:fixture.run_ms(.1)
        failed=fixture.capture()
        if mode=='native':
            assert str(error.value.__cause__)==cause
        else:cause=str(error.value.__cause__)
        yield RunObservation(first)
        yield RunObservation(failed)
        fixture.neurons.y_[:]=[.1,.2,.3,.4]
        yield observe_run(fixture,.1)
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/str(mode=='native'),
        parameters=parameters,initial_state=dict(x=[.3,.7,1.1,1.7],y=[0.,0.,0.,0.]),exc=(),exc2=(),silenced=()),
        exercise,monkeypatch,guard=lambda fixture,patch:None,target='cython',float_dtype=np.float32)
