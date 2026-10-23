"""Independent scalar-width policies and native configured inverse trig."""
from state_assertions import assert_exact_state as assert_state
from native_execution import forbid_call
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from mlx_backend.numpy_half import Renderer as InverseRenderer, half_precision
from mlx_backend import binary64,binary32,math64,inverse_trigonometry32


def cases():
    rng=np.random.default_rng(20261018)
    centers=np.asarray([0.,-0.,2**-27,-2**-27,2**-26,-2**-26,2**-12,-2**-12,
                        2**-9,-2**-9,.5,-.5,1.,-1.,1.5893254712295857e-08,-1.0331603483848539e-07],np.float32).view(np.uint32)
    boundary=np.concatenate([np.arange(max(0,int(x)-32),min(0xffffffff,int(x)+33),dtype=np.uint32) for x in centers])
    return np.unique(np.r_[rng.uniform(-1,1,8192).astype(np.float32).view(np.uint32),
        rng.integers(0,0xffffffff,4096,dtype=np.uint32),boundary,
        np.arange(1,129,dtype=np.uint32),np.arange(1,129,dtype=np.uint32)|np.uint32(0x80000000),
        np.array([0x7f800000,0xff800000,0x7f800001,0xff800001,0x7fc12345,0xffc12345],np.uint32)])


@pytest.mark.metal
@pytest.mark.parametrize('name',['arcsin','arccos'])
def test_native_float32_bits_match_independent_numpy(name):
    import mlx.core as mx
    from brian2.core.functions import DEFAULT_FUNCTIONS
    bits=cases()
    with np.errstate(all='ignore'):wanted=np.asarray(DEFAULT_FUNCTIONS[name].pyfunc(bits.view(np.float32)),np.float32).view(np.uint32)
    header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+binary32.MSL_HEADER+inverse_trigonometry32.metal_inverse_trigonometry32_header({name})
    kernel=mx.fast.metal_kernel(name='inverse32_'+name,input_names=['x'],output_names=['fast','exact'],
        header=header,source=f'uint i=thread_position_in_grid.x;if(i<COUNT){{fast[i]=f_inverse_{name}(x[i]);exact[i]=_eon_asincos32_{name}_exact(x[i]);}}',compile_options={'math_mode':'safe'})
    values=kernel(inputs=[mx.array(bits)],template=[('COUNT',len(bits))],grid=(len(bits),1,1),threadgroup=(128,1,1),
        output_shapes=[bits.shape,bits.shape],output_dtypes=[mx.uint32,mx.uint32])
    mx.eval(*values)
    for value in values:np.testing.assert_array_equal(np.asarray(value),wanted,err_msg=name)


def model_for(promoted=False,ode=False):
    import brian2 as b
    from mlx_backend.model import resolve_model
    argument='x+(i+1)/16384' if promoted else 'x'
    fields=('da/dt=-a/tau+arcsin(x)/second:1\ndc/dt=-c/tau+arccos(x)/second:1\n' if ode else 'a:1\nc:1\n')
    reset='v=v_rst;g=0*mV;x+=1/32;phase+=1'
    if not ode:reset+=f';a+=arcsin({argument});c+=arccos({argument})'
    return resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\nx:1\nphase:integer\n'+fields,
        'eq_th':'True','t_rfc':0*b.ms,'eq_rst':reset})


@pytest.mark.metal
@pytest.mark.parametrize('target,promoted',[('numpy',False),('cython',False),('numpy',True),('cython',True)])
def test_default_target_accumulators_promotions_continuation_and_recording(monkeypatch,target,promoted):
    import brian2 as b
    import mlx.core as mx
    from configured_reference import reference
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    model=model_for(promoted)
    assert model.state_dtype=='float32' and model.reference_codegen_target==target
    graph,stimulus=component_inputs(7,9,model)
    wanted=reference(graph,stimulus,model,record_state=True)
    p=configured.prepare(graph,stimulus,model)
    name='d_inverse_arcsin(' if promoted else 'f_inverse_arcsin('
    assert name in p.program.reset_code
    state=configured.initial_state(p)
    for step in range(9):
        state,_=configured.advance(p,state,mx.zeros((1,),mx.uint8),9,1)
        configured.evaluate_state(state)
        assert_state(configured.host_state(state,p),{name:tr[:,step,:] for name,tr in wanted.traces.items()})
    for recording in (False,True):
        result,state=configured.run_prepared(p,stimulus,record=recording,return_state=True,chunk_steps=2)
        assert_state(configured.host_state(state,p),wanted.state)
        if recording:
            for name in ('trials','neurons','steps'):np.testing.assert_array_equal(getattr(result.spikes,name),getattr(wanted.spikes,name))
        else:assert result.spikes.steps.size==0


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
def test_float32_linear_ode_math_is_consumed_every_step(monkeypatch,target):
    import brian2 as b
    import mlx.core as mx
    from configured_reference import reference
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    model=model_for(ode=True);graph,stimulus=component_inputs(5,11,model)
    wanted=reference(graph,stimulus,model,record_state=True)
    p=configured.prepare(graph,stimulus,model)
    state=configured.initial_state(p)
    for step in range(11):
        state,_=configured.advance(p,state,mx.zeros((1,),mx.uint8),11,1)
        configured.evaluate_state(state)
        assert_state(configured.host_state(state,p),{name:tr[:,step,:] for name,tr in wanted.traces.items()})


@pytest.mark.metal
@pytest.mark.parametrize('name,operation,target',[('arcsin','>','numpy'),('arcsin','>=','cython'),
    ('arccos','>=','numpy'),('arccos','>','cython')])
def test_float32_endpoint_threshold_ties_use_actual_result_width(monkeypatch,name,operation,target):
    import brian2 as b
    from configured_reference import reference
    from component_cpu_reference import component_inputs
    from mlx_backend.model import resolve_model
    from mlx_backend import configured
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    seed=1. if name=='arcsin' else -1.
    cutoff=float(getattr(np,name)(np.asarray([seed],np.float32))[0])
    model=resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\nx:1\n',
        'eq_th':f'(t<dt) or ({name}(x){operation}cutoff)',
        'eq_rst':'v=v_rst;g=0*mV;x=seed','seed':seed,'cutoff':cutoff,'t_rfc':0*b.ms})
    graph,stimulus=component_inputs(1,7,model);wanted=reference(graph,stimulus,model)
    p=configured.prepare(graph,stimulus,model)
    result,state=configured.run_prepared(p,stimulus,record=True,return_state=True,chunk_steps=2)
    assert result.spikes.steps.size==(7 if operation=='>=' else 1)
    for field in ('trials','neurons','steps'):np.testing.assert_array_equal(getattr(result.spikes,field),getattr(wanted.spikes,field))
    assert_state(configured.host_state(state,p),wanted.state)


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('promoted',[False,True])
def test_float32_nonfinite_payloads_are_observable_per_step(monkeypatch,target,promoted):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.model import resolve_model
    from mlx_backend import configured
    from component_cpu_reference import component_inputs
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    argument='x+(i+1)/16384' if promoted else 'x'
    model=resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\nx:1\na:1\nc:1\n',
        'eq_th':'True','eq_rst':f'v=v_rst;g=0*mV;a=arcsin({argument});c=arccos({argument})','t_rfc':0*b.ms})
    bits=np.asarray([0x7f800001,0xff800001,0x7fc12345,0xffc12345,0x3f800000,0xbf800000,
        0x3f800001,0xbf800001,0x7f800000,0xff800000,1,0x80000001,0x80000000],np.uint32)
    clock=b.Clock(dt=model.dt_s*b.second)
    neurons=b.NeuronGroup(len(bits),model.eqs,method='linear',threshold=model.eq_th,reset=model.eq_rst,
        refractory='rfc',namespace=model.brian_namespace(),clock=clock,dtype={name:np.dtype(var.dtype) for name,var in model.analysis.variables.items()})
    neurons.v=model.v_0*b.volt;neurons.g=0;neurons.rfc=0*b.second;neurons.x=bits.view(np.float32)
    monitor=b.StateMonitor(neurons,('a','c','x'),record=True,when='end')
    b.Network(neurons,monitor).run(3*clock.dt,namespace={})
    graph,stimulus=component_inputs(len(bits),3,model);p=configured.prepare(graph,stimulus,model)
    state=configured.initial_state(p);state=replace(state,values={**state.values,'x':mx.array(bits)})
    for step in range(3):
        state,_=configured.advance(p,state,mx.zeros((1,),mx.uint8),3,1);configured.evaluate_state(state)
        actual=configured.host_state(state,p)
        assert_state({name:actual[name] for name in ('a','c','x')},
            {name:np.asarray(getattr(monitor,name+'_'))[:,step][None,:] for name in ('a','c','x')})


@pytest.mark.metal
def test_actual_default_cpp_float32_accumulators_and_promoted_calls(monkeypatch,tmp_path):
    import brian2 as b
    from brian2.devices.device import get_device
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    monkeypatch.setitem(b.prefs,'codegen.target','auto')
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'devices.cpp_standalone.extra_make_args_unix',['-j2'])
    previous=get_device();b.set_device('cpp_standalone',build_on_run=False)
    b.device.reinit();b.device.activate(build_on_run=False)
    try:
        model=model_for(True);assert model.reference_codegen_target=='cpp'
        graph,stimulus=component_inputs(7,9,model);p=configured.prepare(graph,stimulus,model)
        clock=b.Clock(dt=model.dt_s*b.second)
        neurons=b.NeuronGroup(7,model.eqs,method='linear',threshold=model.eq_th,reset=model.eq_rst,
            refractory='rfc',namespace=model.brian_namespace(),clock=clock,dtype={name:np.dtype(var.dtype) for name,var in model.analysis.variables.items()})
        neurons.v=model.v_0*b.volt;neurons.g=0;neurons.rfc=0*b.second
        spikes=b.SpikeMonitor(neurons);b.Network(neurons,spikes).run(9*clock.dt,namespace={})
        b.device.build(directory=str(tmp_path/'cpp'),run=True,with_output=False)
        result,state=configured.run_prepared(p,stimulus,return_state=True)
        np.testing.assert_array_equal(result.spikes.neurons,np.asarray(spikes.i[:]))
        np.testing.assert_array_equal(result.spikes.steps,np.rint(np.asarray(spikes.t_[:])/model.dt_s).astype(np.int64))
        got=configured.host_state(state,p)
        assert_state(got,{name:np.asarray(getattr(neurons,name+'_')[:]).copy()[None,:] for name in got})
    finally:b.set_device(previous)


def test_inverse_width_header_selection_is_local():
    select=inverse_trigonometry32.metal_inverse_trigonometry32_header
    assert select(())==''
    assert 'f_inverse_arcsin(' in select({'arcsin'})
    assert 'f_inverse_arccos(' not in select({'arcsin'})
    assert 'f_inverse_arcsin(' not in select({'arccos'})


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('argument_dtype',[np.float32,np.float64])
def test_invariant_argument_width_preserves_reference_constant_promotion(monkeypatch,target,argument_dtype):
    import brian2 as b
    from configured_reference import reference
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    model=resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\na:1\nc:1\n',
        'eq_th':'True','eq_rst':'v=v_rst;g=0*mV;a=arcsin(endpoint);c=arccos(endpoint)',
        'endpoint':argument_dtype(-1.),'t_rfc':0*b.ms})
    graph,stimulus=component_inputs(2,2,model)
    wanted=reference(graph,stimulus,model)
    p=configured.prepare(graph,stimulus,model)
    _,state=configured.run_prepared(p,stimulus,record=False,return_state=True)
    actual=configured.host_state(state,p)
    assert_state(actual,wanted.state)
    low=0xdb
    assert actual['a'].view(np.uint32)[0,0]==0xbfc90f00+low
    assert actual['c'].view(np.uint32)[0,0]==0x40490f00+low


@pytest.mark.metal
@pytest.mark.parametrize('state_dtype',[np.float32,np.float64])
@pytest.mark.parametrize('target,optimise',[('numpy',False),('numpy',True),('cython',False),('cython',True)])
def test_boolean_and_nested_inverse_temporaries_keep_cpu_precision(monkeypatch,state_dtype,target,optimise):
    import brian2 as b
    import mlx.core as mx
    from configured_reference import reference
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',state_dtype)
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',optimise)
    # Brian's statement-cache key omits the optimizer preference. Distinct
    # Boolean identifiers prevent one policy's translation serving another.
    flag=f'flag_{np.dtype(state_dtype).name}_{int(optimise)}'
    model=resolve_model({'eqs':f'v:volt\ng:volt\nrfc:second\na:1\nc:1\nmixed:1\n{flag}:boolean\n',
        'eq_th':'True','t_rfc':0*b.ms,
        'eq_rst':f'v=v_rst;g=0*mV;a+=arcsin({flag});c+=arccos({flag});'
                 f'mixed+=arccos(arcsin({flag})/2);{flag}=not {flag}'})
    graph,stimulus=component_inputs(3,4,model)
    wanted=reference(graph,stimulus,model,record_state=True)
    # A stale optimized translation would use the prepared half endpoint.
    if target=='cython' and not optimise:
        assert wanted.state['a'][0,0]!=state_dtype(3.140625)
    else:
        assert wanted.state['a'][0,0]==state_dtype(3.140625)
    assert model.reference_optimise==optimise
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',not optimise)
    p=configured.prepare(graph,stimulus,model)
    forbid_call(b.Network, 'run', 'CPU network execution during native simulation', monkeypatch)
    state=configured.initial_state(p)
    for step in range(4):
        state,_=configured.advance(p,state,mx.zeros((1,),mx.uint8),4,1)
        configured.evaluate_state(state)
        assert_state(configured.host_state(state,p),{name:tr[:,step,:] for name,tr in wanted.traces.items()})
    result,state=configured.run_prepared(p,stimulus,record=True,return_state=True,chunk_steps=2)
    assert_state(configured.host_state(state,p),wanted.state)
    for name in ('trials','neurons','steps'):
        np.testing.assert_array_equal(getattr(result.spikes,name),getattr(wanted.spikes,name))


@pytest.mark.metal
def test_actual_cpp_namespace_constant_widths_match_optimized_reference(monkeypatch,tmp_path):
    import brian2 as b
    from brian2.devices.device import get_device
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.target','auto')
    monkeypatch.setitem(b.prefs,'devices.cpp_standalone.extra_make_args_unix',['-j2'])
    previous=get_device();b.set_device('cpp_standalone',build_on_run=False)
    b.device.reinit();b.device.activate(build_on_run=False)
    try:
        model=resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\na:1\nc:1\nasin64:1\nacos64:1\n',
            'eq_th':'True','t_rfc':0*b.ms,
            'eq_rst':'v=v_rst;g=0*mV;a=arcsin(endpoint32);c=arccos(endpoint32);'
                     'asin64=arcsin(endpoint64);acos64=arccos(endpoint64)',
            'endpoint32':np.float32(-1.),'endpoint64':np.float64(-1.)})
        assert model.reference_codegen_target=='cpp'
        graph,stimulus=component_inputs(2,2,model);p=configured.prepare(graph,stimulus,model)
        clock=b.Clock(dt=model.dt_s*b.second)
        neurons=b.NeuronGroup(2,model.eqs,method='linear',threshold=model.eq_th,reset=model.eq_rst,
            refractory='rfc',namespace=model.brian_namespace(),clock=clock,
            dtype={name:np.dtype(var.dtype) for name,var in model.analysis.variables.items()})
        neurons.v=model.v_0*b.volt;neurons.g=0;neurons.rfc=0*b.second
        network=b.Network(neurons);network.run(2*clock.dt,namespace={})
        b.device.build(directory=str(tmp_path/'cpp'),run=True,with_output=False)
        _,state=configured.run_prepared(p,stimulus,record=False,return_state=True)
        actual=configured.host_state(state,p)
        assert_state(actual,{name:np.asarray(getattr(neurons,name+'_')[:]).copy()[None,:] for name in actual})
    finally:b.set_device(previous)


@pytest.mark.metal
@pytest.mark.parametrize('dtype',[np.int8,np.uint8,np.int16,np.uint16])
@pytest.mark.parametrize('name',['arcsin','arccos'])
def test_integer_ufunc_result_width_nested_and_scalar_promotion(dtype,name):
    import mlx.core as mx
    from brian2.core.functions import DEFAULT_FUNCTIONS
    from mlx_backend.binary16 import MSL_HEADER as HALF_HEADER,MSL_F32_HEADER
    limits=np.iinfo(dtype)
    values=np.arange(limits.min,limits.max+1,dtype=np.int64).astype(dtype)
    renderer=InverseRenderer(
        {'value':np.dtype(dtype)},{},{'value'},set(),'reset',used_math=set(),reference_target='numpy',
        half_ufuncs={fn:(fn+'_f32','f_inverse_'+fn) for fn in ('arcsin','arccos')})
    other='arccos' if name=='arcsin' else 'arcsin'
    expressions=[renderer.expression(text) for text in (
        f'{name}(value)',f'{name}(value)+1.0004',f'{other}({name}(value)/2)')]
    expected_half=np.dtype(dtype).itemsize==1
    assert half_precision(expressions[0])==expected_half
    assert expressions[0].narrow is True
    with np.errstate(all='ignore'):
        first=DEFAULT_FUNCTIONS[name].pyfunc(values)
        wanted=[first,first+1.0004,DEFAULT_FUNCTIONS[other].pyfunc(first/2)]
    assert first.dtype==np.dtype('float16' if expected_half else 'float32')
    header=binary64.MSL_HEADER+binary32.MSL_HEADER+math64.MSL_MATH_HEADER+HALF_HEADER+MSL_F32_HEADER
    header+=inverse_trigonometry32.metal_inverse_trigonometry32_header({'arcsin','arccos'})
    source='uint i=thread_position_in_grid.x;if(i<COUNT){long value=(long)x[i];'
    source+=''.join(f'out{j}[i]={expression.code};' for j,expression in enumerate(expressions))+'}'
    kernel=mx.fast.metal_kernel(name=f'inverse_integer_{np.dtype(dtype).name}_{name}',input_names=['x'],
        output_names=['out0','out1','out2'],header=header,source=source,compile_options={'math_mode':'safe'})
    outputs=kernel(inputs=[mx.array(values.astype(np.int32))],template=[('COUNT',len(values))],
        grid=(len(values),1,1),threadgroup=(128,1,1),output_shapes=[values.shape]*3,output_dtypes=[mx.uint64]*3)
    mx.eval(*outputs)
    for output,expected in zip(outputs,wanted):
        np.testing.assert_array_equal(np.asarray(output),expected.astype(np.float64).view(np.uint64))


@pytest.mark.metal
@pytest.mark.parametrize('name',['arcsin','arccos'])
def test_all_half_encodings_use_exact_scalar32_inverse_policy(name):
    import mlx.core as mx
    from brian2.core.functions import DEFAULT_FUNCTIONS
    from mlx_backend.binary16 import MSL_HEADER as HALF_HEADER,MSL_F32_HEADER
    from mlx_backend.numpy_half import Expression
    bits=np.arange(65536,dtype=np.uint16)
    renderer=InverseRenderer(
        {},{},{'value'},set(),'reset',used_math=set(),reference_target='numpy',
        half_ufuncs={fn:(fn+'_f32','f_inverse_'+fn) for fn in ('arcsin','arccos')})
    renderer.temporaries['value']=Expression('value','float',False,True,half=True)
    expression=renderer.expression(f'{name}(value)')
    assert expression.half is True
    with np.errstate(all='ignore'):
        wanted=np.asarray(DEFAULT_FUNCTIONS[name].pyfunc(bits.view(np.float16)),np.float16).astype(np.float64).view(np.uint64)
    header=binary64.MSL_HEADER+binary32.MSL_HEADER+math64.MSL_MATH_HEADER+HALF_HEADER+MSL_F32_HEADER
    header+=inverse_trigonometry32.metal_inverse_trigonometry32_header({name})
    kernel=mx.fast.metal_kernel(name='inverse_half_all_'+name,input_names=['x'],output_names=['out'],
        header=header,source=f'uint i=thread_position_in_grid.x;if(i<65536){{ulong value=h_to_d(x[i]);out[i]={expression.code};}}',
        compile_options={'math_mode':'safe'})
    output,=kernel(inputs=[mx.array(bits.astype(np.uint32))],grid=(65536,1,1),threadgroup=(128,1,1),
        output_shapes=[bits.shape],output_dtypes=[mx.uint64])
    mx.eval(output)
    np.testing.assert_array_equal(np.asarray(output),wanted)
