"""Unchanged Eon factory and actual CPU code objects versus native operators."""
from native_execution import forbid_cpu_steps, require_metal

import numpy as np
import pytest

b=pytest.importorskip("brian2")
mx = require_metal('Metal required', allow_module_level=True)

from mlx_backend.brian_device import original_api
from mlx_backend.model import DEFAULT_EQUATIONS,DEFAULT_RESET
from original_api_reference import build_original_api_fixture,cpu_runtime, original_runtime,assert_snapshot_equal
from original_api_protocol import (
    RunObservation, SingleRunComparison, check_original_api_runs, observe_run,
    set_floating_operator_cache,
)
from test_mlx_compiled_integer_adapter import _typed_fixture

pytestmark=pytest.mark.metal


def _original_pair(directory, monkeypatch, options, **runtime):
    set_floating_operator_cache(monkeypatch, b.prefs)
    return SingleRunComparison(lambda mode:build_original_api_fixture(directory/mode,**options),**runtime)


@pytest.mark.parametrize("target",["numpy","cython"])
@pytest.mark.parametrize("case",["finite","nonfinite"])
def test_original_eon_dynamic_floating_operators(tmp_path,monkeypatch,target,case):
    set_floating_operator_cache(monkeypatch, b.prefs)
    params=dict(eqs=DEFAULT_EQUATIONS+"x:1\ny:1\np:1\nr:1\nq:1\n",
        eq_th="True",eq_rst=DEFAULT_RESET+";p=x**y;r=x%y;q=x//y",
        w_syn=0*b.mV,t_rfc=0*b.ms)
    if case=="finite":
        initial=dict(x=np.array([.3,-.3,3.,-3.]),y=np.array([.1,-.1,.5,2.1]))
    else:
        initial=dict(x=np.array([np.inf,np.nan,-0.,np.finfo(float).max]),
                     y=np.array([3.,2.,2.,.75]))
    options=dict(parameters=params,exc=(),exc2=(),silenced=(),initial_state=initial)
    pair=SingleRunComparison(
        lambda mode:build_original_api_fixture(tmp_path/mode,**options),target=target)
    assert pair.cpu.neurons.resetter["spike"].codeobj.class_name==target
    with pair.native(guard=lambda native:forbid_cpu_steps(
            native.network, monkeypatch, 'A CPU timestep executed in the native operator test', patch_preparation=True)) as (native,device):
        assert device.native_steps==1


def build_boundary_fixture(dtype,reset=None):
    from test_mlx_floating_operators import remainder_cases
    x,y=remainder_cases(dtype)
    # Cython raises at a zero divisor. Its error path has a separate contract.
    valid=(y.view(np.uint64 if dtype==np.float64 else np.uint32)&
           (0x7fffffffffffffff if dtype==np.float64 else 0x7fffffff))!=0
    x,y=x[valid],y[valid]
    types={name:dtype for name in ("x","y","p","r","q")}
    return _typed_fixture(types,reset or DEFAULT_RESET+";p=x**y;r=x%y;q=x//y",
        dict(x=x,y=y),equations="x:1\ny:1\np:1\nr:1\nq:1\n",default_reset=False)


@pytest.mark.parametrize("target",["numpy","cython"])
@pytest.mark.parametrize("dtype",[np.float64,np.float32])
def test_actual_brian_operator_boundary_matrix(monkeypatch,target,dtype):
    set_floating_operator_cache(monkeypatch, b.prefs)
    pair=SingleRunComparison(lambda mode:build_boundary_fixture(dtype),target=target)
    with pair.native() as (native,device):
        assert device.native_steps==1


@pytest.mark.parametrize("target",["numpy","cython"])
def test_namespace_exponent_retains_runtime_power_order(tmp_path,monkeypatch,target):
    set_floating_operator_cache(monkeypatch, b.prefs)
    params=dict(eqs=DEFAULT_EQUATIONS+"x:1\np:1\n",eq_th="True",
        eq_rst=DEFAULT_RESET+";p=x**exponent",exponent=3.,w_syn=0*b.mV,t_rfc=0*b.ms)
    options=dict(parameters=params,exc=(),exc2=(),silenced=(),initial_state=dict(x=[.3,3.,1e100,-.3]))
    pair=SingleRunComparison(
        lambda mode:build_original_api_fixture(tmp_path/mode,**options),target=target)
    source=str(pair.cpu.neurons.resetter["spike"].codeobj.code.run)
    assert "exponent" in source
    pair.check_native()


@pytest.mark.parametrize('operation,root_message',[('%','float divmod()'),('//','float division')])
def test_failed_cython_operator_retains_partial_state_and_monitor(tmp_path,monkeypatch,operation,root_message):
    set_floating_operator_cache(monkeypatch, b.prefs)
    params=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\np:1\n',eq_th='True',
        eq_rst=DEFAULT_RESET+f';p=x{operation}y',w_syn=0*b.mV,t_rfc=0*b.ms)
    options=dict(parameters=params,exc=(),exc2=(),silenced=(),initial_state=dict(
        x=[.3,.7,3.,-3.],y=[2.,0.,2.,2.],v=np.array([-.04,-.041,-.042,-.043])*b.volt,
        g=np.array([.01,.02,.03,.04])*b.volt))
    def exercise(fixture,mode):
        yield observe_run(fixture,0.)
        failed=observe_run(fixture,.3,raises=b.BrianObjectException)
        assert str(failed.error.__cause__)==root_message
        yield failed
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/mode,**options),
        exercise,monkeypatch,target='cython',guard=lambda fixture,patch:None)


@pytest.mark.parametrize('target',['numpy','cython'])
def test_mixed_float32_integer_operands_keep_target_promotions(monkeypatch,target):
    set_floating_operator_cache(monkeypatch, b.prefs)
    def build():
        types={name:np.float32 for name in ('x','p','r','q')};types['y']=np.int32
        return _typed_fixture(types,'p=x**y;r=x%y;q=x//y',dict(
            x=np.array([-1.,33554432.,3.,-.3],np.float32),
            y=np.array([16777217,16777217,-2,-3],np.int32)),
            equations='x:1\ny:integer\np:1\nr:1\nq:1\n')
    SingleRunComparison(lambda mode:build(),target=target).check_native()


def test_failed_cython_operator_restores_rng_and_pending_queue_on_resume(tmp_path,monkeypatch):
    set_floating_operator_cache(monkeypatch, b.prefs)
    params=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\np:1\n',eq_th='i<2',
        eq_rst=DEFAULT_RESET+';p=x%y',t_rfc=0*b.ms,t_dly=.1*b.ms,
        r_poi=4900*b.Hz,r_poi2=3200*b.Hz)
    options=dict(parameters=params,exc=(2,2),exc2=(3,),silenced=(),initial_state=dict(
        x=[.3,.7,3.,-3.],y=[2.,0.,2.,2.],g=np.array([.01,.02,.03,.04])*b.volt))
    next_random=None
    def exercise(fixture,mode):
        nonlocal next_random
        yield observe_run(fixture,.3,raises=b.BrianObjectException,rng=True)
        fixture.neurons.y[1]=3.
        yield observe_run(fixture,.3,rng=True)
        if mode=='cpu':
            next_random=np.random.random(10)
        else:
            np.testing.assert_array_equal(np.random.random(10),next_random)
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/mode,**options),
        exercise,monkeypatch,target='cython',guard=lambda fixture,patch:None)


@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('case',['finite','nonfinite'])
def test_operator_results_in_compound_expressions(tmp_path,monkeypatch,target,case):
    params=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\np:1\nr:1\nq:1\n',eq_th='True',
        eq_rst=DEFAULT_RESET+';p=.123-(x%y);r=(x**y)+(x//y);q=(x%y)*(-1)',w_syn=0*b.mV,t_rfc=0*b.ms)
    initial=(dict(x=[.3,-.3,3.,-3.],y=[.1,-.1,.5,2.1]) if case=='finite' else
        dict(x=[np.inf,np.nan,-0.,3.],y=[3.,2.,2.,np.nan]))
    options=dict(parameters=params,exc=(),exc2=(),silenced=(),initial_state=initial)
    _original_pair(tmp_path,monkeypatch,options,target=target).check_native()


@pytest.mark.parametrize('target',['numpy','cython'])
def test_operator_results_in_fused_context(tmp_path,monkeypatch,target):
    params=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\ns:1\nz:1\np:1\nr:1\nq:1\n',eq_th='True',
        eq_rst=DEFAULT_RESET+';p=s*(x**y)+z;r=s*(x%y)+z;q=s*(x//y)+z',w_syn=0*b.mV,t_rfc=0*b.ms)
    options=dict(parameters=params,exc=(),exc2=(),silenced=(),initial_state=dict(
        x=[.3,3.,1.3,.1],y=[.1,.3,.7,2.1],s=[1.1,1.3,.1,1.7],z=[-.3,-1.,-.1,-.123]))
    _original_pair(tmp_path,monkeypatch,options,target=target).check_native()


@pytest.mark.parametrize('target',['numpy','cython'])
def test_float32_operator_results_in_fused_context(monkeypatch,target):
    set_floating_operator_cache(monkeypatch, b.prefs)
    def build():
        types={name:np.float32 for name in ('x','y','s','z','p','r','q')}
        return _typed_fixture(types,'p=s*(x**y)+z;r=s*(x%y)+z;q=s*(x//y)+z',dict(
            x=np.array([.3,3.,1.3,.1],np.float32),y=np.array([.1,.3,.7,2.1],np.float32),
            s=np.array([1.1,1.3,.1,1.7],np.float32),z=np.array([-.3,-1.,-.1,-.123],np.float32)),
            equations='x:1\ny:1\ns:1\nz:1\np:1\nr:1\nq:1\n')
    SingleRunComparison(lambda mode:build(),target=target).check_native()


@pytest.mark.parametrize('target',['numpy','cython'])
def test_operator_threshold_augmented_reset_and_partial_runs(tmp_path,monkeypatch,target):
    set_floating_operator_cache(monkeypatch, b.prefs)
    params=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\np:1\nr:1\nq:1\n',
        eq_th='(x**y>0.5) and (x%y>=0)',
        eq_rst=DEFAULT_RESET+';p+=x**y;r+=x%y;q+=x//y;x=x**y',t_rfc=0*b.ms)
    options=dict(parameters=params,exc=(),exc2=(),silenced=(),initial_state=dict(
        x=[.3,3.,1.3,.1],y=[.1,.3,.7,2.1]))
    durations=(0.,.15,.15,0.,.3)
    def exercise(fixture,mode):
        snapshots=[]
        for number,duration in enumerate(durations):
            if number==3:
                fixture.neurons.y=[.3,.7,.9,.5]
                fixture.neurons.rfc[1]=.1*b.ms
                fixture.synapses.w[:]*=-.5
            snapshots.append(fixture.run_ms(duration))
        for snapshot in snapshots:
            yield RunObservation(snapshot)
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/mode,**options),
        exercise,monkeypatch,target=target,guard=lambda fixture,patch:None)


@pytest.mark.parametrize('target',['numpy','cython'])
def test_operator_recording_off_retains_counts_and_state(tmp_path,monkeypatch,target):
    set_floating_operator_cache(monkeypatch, b.prefs)
    params=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\np:1\nr:1\nq:1\n',eq_th='x**y>0.5',
        eq_rst=DEFAULT_RESET+';p=x**y;r=x%y;q=x//y;x=p',t_rfc=0*b.ms)
    options=dict(parameters=params,exc=(),exc2=(),silenced=(),initial_state=dict(
        x=[.3,3.,1.3,.1],y=[.1,.3,.7,2.1]))
    def execute(directory):
        fixture=build_original_api_fixture(directory,**options)
        fixture.network.remove(fixture.monitor)
        monitor=b.SpikeMonitor(fixture.neurons,record=False)
        fixture.network.add(monitor)
        fixture.network.run(.3*b.ms,namespace={})
        assert not monitor.record
        state={name:np.array(variable.get_value(),copy=True) for name,variable in fixture.neurons.variables.items()
            if isinstance(variable,b.core.variables.ArrayVariable) and name!='_spikespace'}
        return state,np.array(monitor.count),np.array(monitor.variables['N'].get_value()),np.array(fixture.neurons.spikes)
    with cpu_runtime():
        b.prefs.codegen.target=target;expected=execute(tmp_path/'cpu')
    with original_runtime(original_api, target=target):observed=execute(tmp_path/'native')
    for name in expected[0]:
        a,c=observed[0][name],expected[0][name]
        np.testing.assert_array_equal(a.view(np.uint8),c.view(np.uint8),err_msg=name)
    for actual,reference in zip(observed[1:],expected[1:]):np.testing.assert_array_equal(actual,reference)


def test_failed_cython_operator_in_threshold_retains_completed_phase(tmp_path,monkeypatch):
    set_floating_operator_cache(monkeypatch, b.prefs)
    params=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\np:1\n',eq_th='x%y>=0',
        eq_rst=DEFAULT_RESET+';p=x**y',t_rfc=0*b.ms)
    options=dict(parameters=params,exc=(),exc2=(),silenced=(),initial_state=dict(
        x=[.3,.7,3.,-3.],y=[2.,0.,2.,2.],g=np.array([.01,.02,.03,.04])*b.volt))
    def exercise(fixture,mode):
        yield observe_run(fixture,.3,raises=b.BrianObjectException)
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/mode,**options),
        exercise,monkeypatch,target='cython',guard=lambda fixture,patch:None)


@pytest.mark.parametrize('expression',['1/(x**y)','floor(x/y)+x**y'])
def test_failed_cython_division_in_operator_context(tmp_path,monkeypatch,expression):
    set_floating_operator_cache(monkeypatch, b.prefs)
    params=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\np:1\n',eq_th='True',
        eq_rst=DEFAULT_RESET+';p='+expression,t_rfc=0*b.ms)
    initial=(dict(x=[.3,0.,3.,2.],y=[2.,2.,2.,2.]) if expression.startswith('1/') else
             dict(x=[.3,1.,3.,2.],y=[2.,0.,2.,2.]))
    options=dict(parameters=params,exc=(),exc2=(),silenced=(),initial_state=initial)
    def exercise(fixture,mode):
        yield observe_run(fixture,.3,raises=b.BrianObjectException)
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/mode,**options),
        exercise,monkeypatch,target='cython',guard=lambda fixture,patch:None)


@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('dtype',[np.float64,np.float32])
def test_power_identity_preserves_enclosing_nan_priority(monkeypatch,target,dtype):
    set_floating_operator_cache(monkeypatch, b.prefs)
    def build():
        types={name:dtype for name in ('x','y','p','r','q')}
        raw=(np.array([0x7ff8000000000123,0xfff0000000000456,0x7ff0000000000789,0xfff8000000000321],np.uint64)
             if dtype==np.float64 else np.array([0x7fc00123,0xff800456,0x7f800789,0xffc00321],np.uint32))
        a,c=np.meshgrid(raw,raw)
        return _typed_fixture(types,'p=(x**1)+y;r=(x**1)*y;q=(x**1)/y',
            dict(x=a.ravel().view(dtype),y=c.ravel().view(dtype)),equations='x:1\ny:1\np:1\nr:1\nq:1\n')
    SingleRunComparison(lambda mode:build(),target=target).check_native()


@pytest.mark.parametrize('threshold',['(i==0) and (x%y>=0)','(i>0) or (x%y>=0)'])
def test_cython_operator_error_checks_remain_short_circuited(tmp_path,monkeypatch,threshold):
    params=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\np:1\n',eq_th=threshold,
        eq_rst=DEFAULT_RESET+';p=x**.3',t_rfc=0*b.ms)
    options=dict(parameters=params,exc=(),exc2=(),silenced=(),initial_state=dict(
        x=[.3,.7,3.,2.],y=[2.,0.,0.,0.]))
    _original_pair(tmp_path,monkeypatch,options,target='cython',duration_ms=.2).check_native()


@pytest.mark.parametrize('ambient,actual',[('numpy','cython'),('cython','numpy')])
def test_operator_policy_retains_actual_code_object_identity(monkeypatch,ambient,actual):
    from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
    from brian2.codegen.runtime.cython_rt import CythonCodeObject
    set_floating_operator_cache(monkeypatch, b.prefs)
    def build():
        return _typed_fixture({},'p=x**y;r=x%y;q=x//y',
            dict(x=[1.,-1.,.3,-.3],y=[.1,.1,.1,.1]),equations='x:1\ny:1\np:1\nr:1\nq:1\n',
            codeobj_class={'numpy':NumpyCodeObject,'cython':CythonCodeObject}[actual])
    pair=SingleRunComparison(lambda mode:build(),target=ambient)
    assert pair.cpu.neurons.state_updater.codeobj.class_name==actual
    assert pair.expected.state['q'][0]==(9. if actual=='numpy' else 10.)
    pair.check_native()


def test_numpy_floating_error_policy_for_dynamic_power(tmp_path):
    params=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\np:1\n',eq_th='True',
        eq_rst=DEFAULT_RESET+';p=x**y',t_rfc=0*b.ms)
    options=dict(parameters=params,exc=(),exc2=(),silenced=(),initial_state=dict(
        x=[.3,0.,3.,2.],y=[2.,-1.,2.,2.]))
    def exercise(fixture,mode):
        with np.errstate(divide='raise'):
            with pytest.raises(b.BrianObjectException) as error:fixture.run_ms(.1)
        if mode=='cpu':
            assert isinstance(error.value.__cause__,FloatingPointError)
            np.testing.assert_array_equal(fixture.neurons.p_[:],np.zeros(4))
            assert fixture.capture().clock_timestep==0
        yield RunObservation(fixture.capture(),error.value)
        if mode=='native':
            device=b.get_device()
            assert device.native_steps==device.native_runs==0
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/mode,**options),
        exercise,None,target='numpy',guard=lambda fixture,patch:None)


@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('expression',['expm1(log(i+1)/64)','(i+1)**(1/64)-1'])
def test_brian_fractional_power_rewrite_preserves_actual_target(tmp_path,monkeypatch,target,expression):
    params=dict(eqs=DEFAULT_EQUATIONS+'p:1\n',eq_th='True',
        eq_rst=DEFAULT_RESET+';p='+expression,t_rfc=0*b.ms)
    options=dict(parameters=params,exc=(),exc2=(),silenced=())
    _original_pair(tmp_path,monkeypatch,options,target=target).check_native()


@pytest.mark.parametrize('target',['numpy','cython'])
def test_augmented_floating_operators_preserve_target(tmp_path,monkeypatch,target):
    params=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\np:1\nr:1\n',eq_th='True',
        eq_rst=DEFAULT_RESET+';x%=y;p=x**y;r=x//y',t_rfc=0*b.ms)
    x=[.3,-.3,3.,-3.]
    options=dict(parameters=params,exc=(),exc2=(),silenced=(),initial_state=dict(x=x,p=x,r=x,y=[2.,-2.,.7,-.7]))
    _original_pair(tmp_path,monkeypatch,options,target=target).check_native()


def test_failed_threshold_preserves_partial_spike_buffer_after_previous_run(tmp_path,monkeypatch):
    set_floating_operator_cache(monkeypatch, b.prefs)
    params=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\nz:1\np:1\n',eq_th='(x%y)>.5',
        eq_rst=DEFAULT_RESET+';p=x%z',t_rfc=0*b.ms)
    options=dict(parameters=params,exc=(),exc2=(),silenced=(),initial_state=dict(
        x=[.1,.1,1.,1.],y=[2.,2.,2.,2.],z=[2.,2.,0.,2.]))
    def exercise(fixture,mode):
        # A successful Network.run clears the spike count during after_run.
        # A reset failure retains the preceding complete threshold buffer.
        failed=observe_run(fixture,.1,raises=b.BrianObjectException)
        if mode=='cpu':
            np.testing.assert_array_equal(fixture.neurons.spikes,[2,3])
        yield failed
        fixture.neurons.x[0]=1.;fixture.neurons.y[1]=0.
        failed=observe_run(fixture,.1,raises=b.BrianObjectException)
        if mode=='cpu':
            np.testing.assert_array_equal(fixture.neurons.spikes,[0,3])
        yield failed
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/mode,**options),
        exercise,monkeypatch,target='cython',guard=lambda fixture,patch:None)


@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('dtype',[np.float64,np.float32])
def test_augmented_modulo_boundary_matrix(monkeypatch,target,dtype):
    set_floating_operator_cache(monkeypatch, b.prefs)
    reset=DEFAULT_RESET+';x%=y;r=x;p=x**y;q=x//y'
    SingleRunComparison(lambda mode:build_boundary_fixture(dtype,reset=reset),target=target).check_native()


@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('integer_dtype',[np.int32,np.int64,np.uint64])
def test_integer_base_float32_exponent_preserves_target_promotions(monkeypatch,target,integer_dtype,default_dtype=None,base_expression='x',context=False,ambient_target=None,operand_dtype=np.float32):
    set_floating_operator_cache(monkeypatch, b.prefs)
    def build():
        equations='x:integer\ny:1\np:1\nr:1\nq:1\n'
        reset=f'p=({base_expression})**y;r=({base_expression})%y;q=({base_expression})//y'
        bits=np.dtype(integer_dtype).itemsize*8;high=1<<(bits-1)
        midpoint=high+(2**39+1 if bits==64 else 1)
        values=([min(2**24+1,np.iinfo(integer_dtype).max),midpoint,1,high-1,0,np.iinfo(integer_dtype).max,high,3]
            if np.dtype(integer_dtype).kind=='u' else
            [2**24+1,-2**24-1,1,-1,0,np.iinfo(integer_dtype).max,np.iinfo(integer_dtype).min,3])
        x=np.array(values,integer_dtype)
        y=np.array([1.,1.,.5,.3,-2.,.3,.5,-.3],operand_dtype)
        types={name:operand_dtype for name in ('y','p','r','q')};types['x']=integer_dtype
        if context:
            equations+='s:1\nz:1\n';types.update(s=operand_dtype,z=operand_dtype)
            reset=f'p=s*(({base_expression})**y)+z;r=s*(({base_expression})%y)+z;q=s*(({base_expression})//y)+z'
        codeobj_class=None
        if ambient_target is not None:
            from brian2.codegen.runtime.cython_rt import CythonCodeObject
            from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
            codeobj_class={'numpy':NumpyCodeObject,'cython':CythonCodeObject}[target]
        initial=dict(x=x,y=y)
        if context:
            initial['s']=np.array([1.1,1.3,.1,1.7]*2,np.float32)
            initial['z']=np.array([-.3,-1.,-.1,-.123]*2,np.float32)
        return _typed_fixture(types,reset,initial,equations=equations,codeobj_class=codeobj_class)
    SingleRunComparison(lambda mode:build(),target=ambient_target or target,
        float_dtype=np.float64 if default_dtype is None else default_dtype).check_native()


@pytest.mark.parametrize('target,integer_dtype,options', [
    pytest.param(target,integer_dtype,options,
                 id=name+('-'+target if len(targets)>1 else ''))
    for name,targets,integer_dtype,options in [
        ('test_all_float32_uint64_base_operators_preserve_target',
         ('numpy','cython'),np.uint64,{'default_dtype':np.float32}),
        ('test_uint64_integer_expression_floating_operands_preserve_target',
         ('numpy','cython'),np.uint64,{'base_expression':'x+1'}),
        ('test_cython_uint64_unary_floating_operand_preserves_type',
         ('cython',),np.uint64,{'base_expression':'-x'}),
        ('test_uint8_floating_operand_preserves_actual_target_type',
         ('numpy','cython'),np.uint8,{}),
        ('test_numpy_uint8_operator_temporaries_preserve_outer_float32_rounding',
         ('numpy',),np.uint8,{'context':True}),
        ('test_all_float32_cython_byte_floating_operand_preserves_cpp_type',
         ('cython',),np.uint8,{'default_dtype':np.float32}),
        ('test_cython_binary64_byte_graph_preserves_actual_char_type',
         ('cython',),np.uint8,{'context':True,'operand_dtype':np.float64}),
    ]
    for target in targets
])
def test_integer_base_named_variants(monkeypatch,target,integer_dtype,options):
    test_integer_base_float32_exponent_preserves_target_promotions(
        monkeypatch,target,integer_dtype,**options)


@pytest.mark.parametrize('flag',['-fsigned-char','-funsigned-char'])
def test_cython_byte_floating_operand_captures_char_flags(monkeypatch,flag,default_dtype=None):
    flags=b.prefs.codegen.cpp.extra_compile_args
    if flags is None:flags=b.prefs.codegen.cpp.extra_compile_args_gcc
    monkeypatch.setitem(b.prefs,'codegen.cpp.extra_compile_args',[*flags,flag])
    test_integer_base_float32_exponent_preserves_target_promotions(
        monkeypatch,'cython',np.uint8,default_dtype=default_dtype)


def test_all_float32_cython_unsigned_byte_flag_selects_actual_conversion(monkeypatch):
    test_cython_byte_floating_operand_captures_char_flags(
        monkeypatch,'-funsigned-char',default_dtype=np.float32)


def test_explicit_cython_byte_policy_uses_actual_flags_with_numpy_ambient(monkeypatch):
    flags=b.prefs.codegen.cpp.extra_compile_args
    if flags is None:flags=b.prefs.codegen.cpp.extra_compile_args_gcc
    monkeypatch.setitem(b.prefs,'codegen.cpp.extra_compile_args',[*flags,'-funsigned-char'])
    test_integer_base_float32_exponent_preserves_target_promotions(
        monkeypatch,'cython',np.uint8,ambient_target='numpy')


def test_cython_error_kind_survives_native_window_continuation(tmp_path,monkeypatch):
    from mlx_backend import configured
    from mlx_backend.models import Connectome,Stimulus
    from mlx_backend.model import resolve_model
    set_floating_operator_cache(monkeypatch, b.prefs)
    params=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\np:1\n',eq_th='True',
        eq_rst=DEFAULT_RESET+';p=x//y',t_rfc=0*b.ms)
    with cpu_runtime(target='cython'):
        cpu=build_original_api_fixture(tmp_path/'cpu',parameters=params,exc=(),exc2=(),silenced=())
        with pytest.raises(b.BrianObjectException) as error:cpu.run_ms(.1)
        expected=str(error.value.__cause__)
        model=resolve_model(params)
    empty=np.empty(0,np.int32);weights=np.empty(0,np.float64)
    graph=Connectome(np.array(['0','1','2','3']),empty,empty,weights,weights)
    stimulus=Stimulus(np.zeros((1,2,0),np.uint8),(),(),(0,),0,0,None)
    prepared=configured.prepare(graph,stimulus,model)
    state=configured.initial_state(prepared)
    events=mx.zeros((1,),mx.uint8)
    for _ in range(2):state,_=configured.advance(prepared,state,events,2,1)
    with pytest.raises(ZeroDivisionError,match='^'+__import__('re').escape(expected)+'$'):
        configured.evaluate_state(state)


@pytest.mark.parametrize('target',['numpy','cython'])
def test_namespace_constant_operators_preserve_actual_target(tmp_path,monkeypatch,target):
    params=dict(eqs=DEFAULT_EQUATIONS+'p:1\nr:1\nq:1\n',eq_th='True',
        eq_rst=DEFAULT_RESET+';p=numerator**denominator;r=numerator%denominator;q=numerator//denominator',
        numerator=1.,denominator=.1,t_rfc=0*b.ms)
    options=dict(parameters=params,exc=(),exc2=(),silenced=())
    _original_pair(tmp_path,monkeypatch,options,target=target).check_native()


def test_failed_cython_operator_stops_submitting_requested_future(tmp_path,monkeypatch):
    set_floating_operator_cache(monkeypatch, b.prefs)
    params=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\np:1\n',eq_th='True',
        eq_rst=DEFAULT_RESET+';p=x//y',t_rfc=0*b.ms)
    options=dict(parameters=params,exc=(),exc2=(),silenced=())
    def exercise(fixture,mode):
        yield observe_run(fixture,300.,raises=b.BrianObjectException)
        if mode=='native':
            assert b.get_device().native_steps<3000
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/mode,**options),
        exercise,monkeypatch,target='cython',guard=lambda fixture,patch:None)


@pytest.mark.parametrize('target',['numpy','cython'])
def test_literal_operator_inside_power_retains_actual_target(tmp_path,monkeypatch,target):
    params=dict(eqs=DEFAULT_EQUATIONS+'x:1\np:1\nr:1\nq:1\n',eq_th='True',
        eq_rst=DEFAULT_RESET+';p=x**(1.//.1);r=x**(1.%.1);q=x**((1.//.1)+.3)',t_rfc=0*b.ms)
    options=dict(parameters=params,exc=(),exc2=(),silenced=(),initial_state=dict(x=[.3,-.3,3.,-3.]))
    _original_pair(tmp_path,monkeypatch,options,target=target).check_native()
