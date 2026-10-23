"""Exact binary64 inverse trig against independent CPU and actual Metal."""
from state_assertions import assert_exact_state as assert_state
import ctypes
from pathlib import Path

import numpy as np
import pytest
from native_execution import require_metal
from pinned_apple_math import require_apple_arm64_reference

from mlx_backend import binary64, inverse_trigonometry64, math64


def cases():
    centers=[0.,-0.,.4,-.4,.5,-.5,.6,-.6,1.,-1.,
             float.fromhex('0x1.7137449123ef5p-26'),
             float.fromhex('0x1.8d313198a2e03p-53'),
             np.finfo(np.float64).tiny,-np.finfo(np.float64).tiny]
    centers += [-x for x in centers[10:]]
    values=[np.nextafter(x,side) for x in centers for side in (-np.inf,np.inf)]+centers
    rng=np.random.default_rng(20261017)
    raw=rng.integers(0,2**64-1,4096,dtype=np.uint64)
    uniform=rng.uniform(-1,1,8192).view(np.uint64)
    endpoints=np.r_[np.arange(0x3ff0000000000000-256,0x3ff0000000000000+257,dtype=np.uint64),
                    np.arange(0xbff0000000000000-256,0xbff0000000000000+257,dtype=np.uint64)]
    special=np.array([0,1,2,0x000fffffffffffff,0x0010000000000000,
        0x8000000000000000,0x8000000000000001,0x800fffffffffffff,
        0x7fefffffffffffff,0xffefffffffffffff,0x7ff0000000000000,0xfff0000000000000,
        0x7ff0000000000001,0xfff0000000000001,0x7ff8000000000000,0xfff8000000000000,
        0x7ff8123456789abc,0xfff8123456789abc,0x7fffffffffffffff,0xffffffffffffffff],np.uint64)
    tiny=np.r_[np.arange(1,129,dtype=np.uint64),
               np.array([2.**k for k in range(-1074,1)],np.float64).view(np.uint64)]
    return np.unique(np.r_[np.asarray(values,np.float64).view(np.uint64),raw,uniform,endpoints,
                            special,tiny,tiny|np.uint64(0x8000000000000000)])


def expected(name,raw,*,scalar=False):
    require_apple_arm64_reference()
    if scalar:
        library=ctypes.CDLL('/usr/lib/libSystem.B.dylib')
        function=getattr(library,{'arcsin':'asin','arccos':'acos'}[name])
        function.argtypes,function.restype=[ctypes.c_double],ctypes.c_double
        return np.asarray([function(x) for x in raw.view(np.float64)],np.float64).view(np.uint64)
    from brian2.core.functions import DEFAULT_FUNCTIONS
    with np.errstate(all='ignore'):
        return np.asarray(DEFAULT_FUNCTIONS[name].pyfunc(raw.view(np.float64)),np.float64).view(np.uint64)


def assert_bits(actual,reference,raw,name):
    differences=np.flatnonzero(actual!=reference)
    if differences.size:
        index=differences[0]
        pytest.fail(f'{name}: input={int(raw[index]):#018x}, native={int(actual[index]):#018x}, '
                    f'CPU={int(reference[index]):#018x}; {len(differences)}/{len(raw)} differences')


@pytest.mark.metal
@pytest.mark.parametrize('name',['arcsin','arccos'])
def test_actual_metal_full_range_matches_independent_cpu(name):
    mx = require_metal('Metal required')
    raw=cases()
    header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+inverse_trigonometry64.metal_inverse_trigonometry_header({name})
    kernel=mx.fast.metal_kernel(name='inverse_trig64_'+name,input_names=['values'],output_names=['out'],
        header=header,source=f'uint i=thread_position_in_grid.x;if(i<COUNT)out[i]=d_inverse_{name}(values[i]);',
        compile_options={'math_mode':'safe'})
    out,=kernel(inputs=[mx.array(raw)],template=[('COUNT',len(raw))],grid=(len(raw),1,1),
        threadgroup=(128,1,1),output_shapes=[raw.shape],output_dtypes=[mx.uint64])
    mx.eval(out)
    for scalar in (False,True):
        assert_bits(np.asarray(out),expected(name,raw,scalar=scalar),raw,name)


def test_selective_header_has_no_forward_trig_dependency():
    select=inverse_trigonometry64.metal_inverse_trigonometry_header
    assert select(())==''
    assert 'd_inverse_arccos(' not in select({'arcsin'})
    assert 'd_inverse_arcsin(' not in select({'arccos'})
    assert 'd_inverse_arcsin(' in select({'arcsin','arccos'})
    assert 'd_inverse_arccos(' in select({'arcsin','arccos'})


def accumulator_model():
    import brian2 as b
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    return resolve_model({'eqs':DEFAULT_EQUATIONS+'x:1\na:1\nc:1\nphase:integer\n',
        'eq_th':'True','t_rfc':0*b.ms,
        'eq_rst':'v=v_rst;g=0*mV;a+=arcsin(x);c+=arccos(x);x=(i+phase-8)/16;phase+=1'})


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
def test_independent_brian_accumulators_every_step_continuation_and_recording(monkeypatch,target):
    import brian2 as b
    import mlx.core as mx
    from configured_reference import reference
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    model=accumulator_model()
    assert model.reference_codegen_target==target
    graph,stimulus=component_inputs(7,13,model)
    wanted=reference(graph,stimulus,model,record_state=True)
    p=configured.prepare(graph,stimulus,model)
    assert 'd_inverse_arcsin(' in p.program.reset_code
    assert 'd_inverse_arccos(' in p.program.reset_code
    state=configured.initial_state(p)
    events=mx.zeros((1,),mx.uint8)
    observed=[]
    for step in range(13):
        state,words=configured.advance(p,state,events,13,1)
        configured.evaluate_state(state)
        observed.extend((0,int(i),step) for i in np.flatnonzero(np.asarray(words)&1))
        actual=configured.host_state(state,p)
        for name,values in actual.items():
            expected_step=wanted.traces[name][:,step,:]
            assert_state({name:values},{name:expected_step})
    assert observed==list(zip(wanted.spikes.trials,wanted.spikes.neurons,wanted.spikes.steps))
    for record in (True,False):
        actual,state=configured.run_prepared(p,stimulus,record=record,return_state=True,chunk_steps=3,block_windows=1)
        if record:
            for name in ('trials','neurons','steps'):
                np.testing.assert_array_equal(getattr(actual.spikes,name),getattr(wanted.spikes,name))
        else:assert actual.spikes.steps.size==0
        assert_state(configured.host_state(state,p),wanted.state)


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
def test_nan_payloads_and_domain_results_are_observable_in_brian_states(monkeypatch,target):
    from dataclasses import replace
    import brian2 as b
    import mlx.core as mx
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    from component_cpu_reference import component_inputs
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    model=resolve_model({'eqs':DEFAULT_EQUATIONS+'x:1\na:1\nc:1\n','eq_th':'True',
                        'eq_rst':'v=v_rst;g=0*mV;a=arcsin(x);c=arccos(x)','t_rfc':0*b.ms})
    raw=np.array([0x7ff0000000000001,0xfff0000000000001,0x7ff8123456789abc,0xfff8123456789abc,
        0x3ff0000000000001,0xbff0000000000001,0x7fefffffffffffff,0xffefffffffffffff,
        0x7ff0000000000000,0xfff0000000000000,0x8000000000000000,1,0x8000000000000001],np.uint64)
    clock=b.Clock(dt=model.dt_s*b.second)
    neurons=b.NeuronGroup(len(raw),model.eqs,method='linear',threshold=model.eq_th,reset=model.eq_rst,
        refractory='rfc',namespace=model.brian_namespace(),clock=clock)
    neurons.v=model.v_0*b.volt;neurons.g=0;neurons.rfc=0*b.second;neurons.x=raw.view(np.float64)
    trace=b.StateMonitor(neurons,('a','c','x'),record=True,when='end')
    b.Network(neurons,trace).run(3*clock.dt,namespace={})
    graph,stimulus=component_inputs(len(raw),3,model)
    p=configured.prepare(graph,stimulus,model)
    state=configured.initial_state(p)
    state=replace(state,values={**state.values,'x':mx.array(raw)})
    for step in range(3):
        state,_=configured.advance(p,state,mx.zeros((1,),mx.uint8),3,1)
        configured.evaluate_state(state)
        actual=configured.host_state(state,p)
        for name in ('a','c','x'):
            assert_state({name:actual[name]},{name:np.asarray(getattr(trace,name+'_'))[:,step][None,:]})


@pytest.mark.metal
@pytest.mark.parametrize('name,operation,target',[
    ('arcsin','>','numpy'),('arcsin','>=','cython'),
    ('arccos','>=','numpy'),('arccos','>','cython')])
def test_function_threshold_exact_ties(monkeypatch,name,operation,target):
    import brian2 as b
    from configured_reference import reference
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    argument=.6
    cutoff=float(expected(name,np.array([argument]).view(np.uint64)).view(np.float64)[0])
    model=resolve_model({'eqs':DEFAULT_EQUATIONS+'x:1\n',
        'eq_th':f'(t<dt) or ({name}(x){operation}cutoff)',
        'eq_rst':'v=v_rst;g=0*mV;x=argument','argument':argument,'cutoff':cutoff,'t_rfc':0*b.ms})
    graph,stimulus=component_inputs(1,7,model)
    wanted=reference(graph,stimulus,model)
    p=configured.prepare(graph,stimulus,model)
    actual,state=configured.run_prepared(p,stimulus,record=True,return_state=True,chunk_steps=2)
    assert actual.spikes.steps.size==(7 if operation=='>=' else 1)
    for field in ('trials','neurons','steps'):
        np.testing.assert_array_equal(getattr(actual.spikes,field),getattr(wanted.spikes,field))
    assert_state(configured.host_state(state,p),wanted.state)


@pytest.mark.metal
def test_numpy_linear_ode_consumption_of_dynamic_inverse_functions(monkeypatch):
    import brian2 as b
    import mlx.core as mx
    from configured_reference import reference
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    model=resolve_model({'eqs':DEFAULT_EQUATIONS+'da/dt=-a/tau+arcsin(x)/second:1\n'
        'dc/dt=-c/tau+arccos(x)/second:1\nx:1\nphase:integer\n',
        'eq_th':'(t<dt) or (arcsin(x)>=0)','t_rfc':0*b.ms,
        'eq_rst':'v=v_rst;g=0*mV;x=(i+phase-2)/32;phase+=1'})
    graph,stimulus=component_inputs(5,21,model)
    wanted=reference(graph,stimulus,model,record_state=True)
    p=configured.prepare(graph,stimulus,model)
    for name in ('arcsin','arccos'):assert 'd_inverse_'+name+'(' in p.program.state_code
    state=configured.initial_state(p)
    for step in range(21):
        state,_=configured.advance(p,state,mx.zeros((1,),mx.uint8),21,1)
        configured.evaluate_state(state)
        assert_state(configured.host_state(state,p),{name:trace[:,step,:] for name,trace in wanted.traces.items()})


def test_reference_target_flags_and_snapshot_identity(monkeypatch):
    import brian2 as b
    from mlx_backend.compiler import compile_model
    models=[]
    for target in ('numpy','cython','auto'):
        monkeypatch.setitem(b.prefs,'codegen.target',target)
        models.append(accumulator_model())
    assert models[0].signature!=models[1].signature
    # The canonical integer policy preserves the requested target as well as
    # the resolved target. Auto and explicit Cython execute the same float
    # function but intentionally retain distinct cache identities.
    from dataclasses import replace
    assert models[1].reference_codegen_target==models[2].reference_codegen_target=='cython'
    assert models[1].signature!=models[2].signature
    assert replace(models[1].integer_policy,requested_target='auto').identity==models[2].integer_policy.identity
    before=compile_model(models[0])
    monkeypatch.setitem(b.prefs,'codegen.cpp.extra_compile_args',['-O0'])
    changed=accumulator_model()
    assert changed.signature!=models[2].signature
    assert compile_model(models[0]).reset_code==before.reset_code


@pytest.mark.metal
def test_actual_cpp_standalone_accumulator_and_spikes(monkeypatch,tmp_path):
    import brian2 as b
    from brian2.devices.device import get_device
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    monkeypatch.setitem(b.prefs,'codegen.target','auto')
    monkeypatch.setitem(b.prefs,'devices.cpp_standalone.extra_make_args_unix',['-j2'])
    previous=get_device()
    b.set_device('cpp_standalone',build_on_run=False)
    b.device.reinit();b.device.activate(build_on_run=False)
    try:
        model=accumulator_model()
        assert model.reference_codegen_target=='cpp'
        graph,stimulus=component_inputs(7,13,model)
        p=configured.prepare(graph,stimulus,model)
        clock=b.Clock(dt=model.dt_s*b.second)
        neurons=b.NeuronGroup(7,model.eqs,method='linear',threshold=model.eq_th,reset=model.eq_rst,
            refractory='rfc',namespace=model.brian_namespace(),clock=clock)
        neurons.v=model.v_0*b.volt;neurons.g=0;neurons.rfc=0*b.second
        spikes=b.SpikeMonitor(neurons)
        network=b.Network(neurons,spikes);network.schedule=list(model.schedule)
        network.run(13*clock.dt,namespace={})
        b.device.build(directory=str(tmp_path/'cpp'),compile=True,run=True,with_output=False)
        actual,state=configured.run_prepared(p,stimulus,return_state=True)
        np.testing.assert_array_equal(actual.spikes.neurons,np.asarray(spikes.i[:]))
        np.testing.assert_array_equal(actual.spikes.steps,np.rint(np.asarray(spikes.t_[:])/model.dt_s).astype(np.int64))
        got=configured.host_state(state,p)
        assert_state(got,{name:np.asarray(getattr(neurons,name+'_')[:]).copy()[None,:] for name in got})
    finally:b.set_device(previous)


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
def test_invariant_folding_preserves_domain_nan_signs(monkeypatch,target):
    import brian2 as b
    from configured_reference import reference
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    model=resolve_model({'eqs':DEFAULT_EQUATIONS+'a:1\nc:1\n','eq_th':'True',
        'eq_rst':'v=v_rst;g=0*mV;a=arcsin(argument);c=arccos(argument)',
        'argument':-2.,'t_rfc':0*b.ms})
    graph,stimulus=component_inputs(2,2,model)
    wanted=reference(graph,stimulus,model)
    p=configured.prepare(graph,stimulus,model)
    _,state=configured.run_prepared(p,stimulus,record=False,return_state=True)
    actual=configured.host_state(state,p)
    assert_state(actual,wanted.state)
    assert actual['a'].view(np.uint64)[0,0]==0xfff8000000000000
    assert actual['c'].view(np.uint64)[0,0]==0x7ff8000000000000


@pytest.mark.metal
def test_compact_normal_fma_matches_independent_scalar_fma():
    """Exercise alignment limits, signs and cancellation through actual Metal."""
    require_apple_arm64_reference()
    mx = require_metal('Metal required')
    rng=np.random.default_rng(20261018)
    operands=rng.uniform(-2.,2.,(3,4096))*np.exp2(rng.integers(-150,151,(3,4096)))
    a,b,c=operands
    # Close cancellation must retain the entire product, including residuals
    # below either rounded operand. Neighbors exercise the final tie decision.
    c[:1024]=-(a[:1024]*b[:1024])
    c[1024:2048]=np.nextafter(-(a[1024:2048]*b[1024:2048]),np.inf)
    # Exact powers cover shift=-65,-64,-63,20,21,22 and both sum signs.
    for index,shift in enumerate((-65,-64,-63,20,21,22)):
        for sign in (-1.,1.):
            offset=2048+2*index+int(sign>0)
            a[offset]=np.exp2(shift-22);b[offset]=1.;c[offset]=sign
    a[-4:]=[0.,-0.,1.,-1.];b[-4:]=[1.,1.,0.,-0.];c[-4:]=[-0.,-0.,0.,-0.]
    library=ctypes.CDLL('/usr/lib/libSystem.B.dylib')
    scalar=library.fma
    scalar.argtypes,scalar.restype=[ctypes.c_double]*3,ctypes.c_double
    wanted=np.asarray([scalar(x,y,z) for x,y,z in zip(a,b,c)],np.float64).view(np.uint64)
    header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+inverse_trigonometry64.metal_inverse_trigonometry_header({'arcsin'})
    kernel=mx.fast.metal_kernel(name='inverse_compact_fma',input_names=['a','b','c'],output_names=['out'],
        header=header,source='uint i=thread_position_in_grid.x;if(i<COUNT)out[i]=_eon_asincos64_fma_compact(a[i],b[i],c[i]);',
        compile_options={'math_mode':'safe'})
    actual,=kernel(inputs=[mx.array(x.view(np.uint64)) for x in (a,b,c)],template=[('COUNT',len(a))],
        grid=(len(a),1,1),threadgroup=(128,1,1),output_shapes=[a.shape],output_dtypes=[mx.uint64])
    mx.eval(actual)
    np.testing.assert_array_equal(np.asarray(actual),wanted)
