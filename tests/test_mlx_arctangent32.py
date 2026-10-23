"""Exact binary32 arctangent against actual NumPy and scalar CPU targets."""
from state_assertions import assert_recorded_spikes_equal, assert_spikes_equal
import ctypes
import numpy as np
import pytest
from native_execution import require_metal
from pinned_apple_math import require_apple_arm64_reference
from mlx_backend import arctangent32,binary32,binary64,math64


def cases():
    centers=np.array([0.,-0.,2.**-149,2.**-126,2.**-27,.5,1.,2.,7401785.5],np.float32)
    boundary=np.array([v for x in centers for sign in (-1,1)
        for v in [np.nextafter(np.float32(sign*x),np.float32(-np.inf)),np.float32(sign*x),
                  np.nextafter(np.float32(sign*x),np.float32(np.inf))]],np.float32).view(np.uint32)
    special=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x00800000,
        0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7f800001,0xff800001,
        0x7fc12345,0xffc12345,0x7fffffff,0xffffffff],np.uint32)
    rng=np.random.default_rng(20261020)
    raw=rng.integers(0,np.iinfo(np.uint32).max,16384,dtype=np.uint32)
    dense=rng.uniform(-4,4,16384).astype(np.float32).view(np.uint32)
    # Candidate inputs nearest output rounding midpoints expose double-rounding
    # and coefficient-order mistakes; tan is used only in reference preparation.
    values=rng.uniform(-1.5,1.5,4096).astype(np.float32)
    following=np.nextafter(values,np.float32(np.inf));mid=(values.astype(np.float64)+following.astype(np.float64))/2
    near=np.tan(mid).astype(np.float32)
    hard=np.r_[near,np.nextafter(near,np.float32(-np.inf)),np.nextafter(near,np.float32(np.inf))].view(np.uint32)
    return np.unique(np.r_[boundary,special,raw,dense,hard])


def expected(raw,target):
    require_apple_arm64_reference()
    if target=='numpy':
        with np.errstate(all='ignore'):return np.arctan(raw.view(np.float32)).view(np.uint32)
    library=ctypes.CDLL('/usr/lib/libSystem.B.dylib');fn=library.atanf
    fn.argtypes,fn.restype=[ctypes.c_float],ctypes.c_float
    return np.asarray([fn(ctypes.c_float.from_buffer_copy(v.tobytes())) for v in raw],np.float32).view(np.uint32)


@pytest.mark.metal
def test_direct_metal_matches_actual_float_overload_bits():
    mx = require_metal('Metal required')
    raw=cases()
    kernel=mx.fast.metal_kernel(name='direct_arctangent32',input_names=['x'],output_names=['out'],
        source='uint i=thread_position_in_grid.x;if(i<COUNT)out[i]=_eon_arctan32(x[i]);',
        header=binary64.MSL_HEADER+binary32.MSL_HEADER+math64.MSL_MATH_HEADER+
               arctangent32.metal_arctangent32_header({'arctan'}),compile_options={'math_mode':'safe'})
    output,=kernel(inputs=[mx.array(raw)],template=[('COUNT',len(raw))],grid=(len(raw),1,1),
        threadgroup=(128,1,1),output_shapes=[raw.shape],output_dtypes=[mx.uint32])
    mx.eval(output);actual=np.asarray(output)
    for target in ('numpy','cython'):
        wanted=expected(raw,target);bad=np.flatnonzero(actual!=wanted)
        if len(bad):
            i=bad[0];pytest.fail(f'{target}: arctan({int(raw[i]):#010x}) actual={int(actual[i]):#010x}, CPU={int(wanted[i]):#010x}; {len(bad)}/{len(raw)} differ')


def model_configuration(divisor=16384, *, mixed=False, periodic=False):
    import brian2 as b
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    extra='promoted:1\n' if mixed else ''
    index='i-int(i/4096)*4096' if periodic else 'i'
    reset=f'v=v_rst;g=0*mV;argument+=({index}+1)/{divisor};observer+=arctan(argument)'
    parameters={}
    if mixed:
        reset+=';promoted=arctan(argument+wide)';parameters['wide']=1e100
    return resolve_model({'eqs':DEFAULT_EQUATIONS+'argument:1\nobserver:1\n'+extra,
        'eq_th':'True','eq_rst':reset,'t_rfc':0*b.ms,**parameters})


def _assert_state(actual,wanted):
    assert actual.keys()==wanted.keys()
    for name,x in actual.items():
        assert x.dtype==wanted[name].dtype,name
        def bits(a):
            return a.view(np.uint32) if a.dtype==np.float32 else a.view(np.uint64) if a.dtype==np.float64 else a
        np.testing.assert_array_equal(bits(x),bits(wanted[name]),err_msg=name)


def _reference_dtype(model):
    # Brian's built-in lastspike retains float64 even for float32 equations.
    return {name:np.dtype(variable.dtype) for name,variable in model.analysis.variables.items()}


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('mixed',[False,True])
def test_runtime_actual_and_promoted_arguments(monkeypatch,tmp_path,target,mixed):
    mx = require_metal('Metal required')
    import brian2 as b
    from configured_reference import reference
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(tmp_path/'cython-default'))
    model=model_configuration(4,mixed=mixed);assert model.state_dtype=='float32'
    graph,stimulus=component_inputs(7,21,model)
    wanted=reference(graph,stimulus,model,record_state=True)
    prepared=configured.prepare(graph,stimulus,model)
    assert '_eon_arctan32' in prepared.program.reset_code
    for record in (True,False):
        result,state=configured.run_prepared(prepared,stimulus,record=record,return_state=True,chunk_steps=7)
        _assert_state(configured.host_state(state,prepared),wanted.state)
        assert_recorded_spikes_equal(result, wanted, record)
    # Every intermediate arctan value contributes to cumulative observer state.
    state=configured.initial_state(prepared);events=mx.array(np.zeros(1,np.uint8))
    for step in range(21):
        state,_=configured.advance(prepared,state,events,21,1)
        configured.evaluate_state(state)
        _assert_state(configured.host_state(state,prepared),{name:a[:,step] for name,a in wanted.traces.items()})


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
def test_runtime_special_argument_bits(monkeypatch,tmp_path,target):
    mx = require_metal('Metal required')
    from dataclasses import replace
    import brian2 as b
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(tmp_path/'cython-default'))
    raw=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x00800000,
        0x7f800000,0xff800000,0x7f7fffff,0xff7fffff,0x7f800001,0xff800001,
        0x7fc12345,0xffc12345,0x3f800000,0xbf800000,0x4ae1e273,0xcae1e273],np.uint32)
    model=resolve_model({'eqs':DEFAULT_EQUATIONS+'argument:1\nobserver:1\n',
        'eq_th':'True','eq_rst':'v=v_rst;g=0*mV;observer=arctan(argument)','t_rfc':0*b.ms})
    graph,stimulus=component_inputs(len(raw),3,model)
    clock=b.Clock(dt=model.dt_s*b.second)
    neurons=b.NeuronGroup(len(raw),model.eqs,method='linear',threshold=model.eq_th,
        reset=model.eq_rst,refractory='rfc',namespace=model.brian_namespace(),clock=clock,dtype=_reference_dtype(model))
    neurons.v=model.v_0*b.volt;neurons.g=0*b.volt;neurons.rfc=0*b.second
    neurons.argument=raw.view(np.float32)
    monitor=b.SpikeMonitor(neurons);network=b.Network(neurons,monitor)
    network.schedule=list(model.schedule);network.run(3*clock.dt,namespace={})
    prepared=configured.prepare(graph,stimulus,model)
    state=configured.initial_state(prepared)
    state=replace(state,values={**dict(state.values),'argument':mx.array(raw)})
    events=mx.array(np.zeros(1,np.uint8));observed=[]
    for step in range(3):
        state,words=configured.advance(prepared,state,events,3,1)
        configured.evaluate_state(state)
        observed.extend((int(i),step) for i in np.flatnonzero(np.asarray(words)&1))
    actual=configured.host_state(state,prepared)
    wanted={name:np.asarray(getattr(neurons,name+'_')[:]).copy().reshape(1,-1) for name in actual}
    _assert_state(actual,wanted)
    assert sorted(observed,key=lambda p:(p[1],p[0]))==list(zip(np.asarray(monitor.i[:]),np.rint(np.asarray(monitor.t_[:])/model.dt_s).astype(np.int64)))


def test_header_selection_is_disjoint():
    assert arctangent32.metal_arctangent32_header(())==''
    assert arctangent32.metal_arctangent32_header({'sin','cos','tan'})==''
    assert '_eon_arctan32' in arctangent32.metal_arctangent32_header({'arctan'})


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('operator',['>','>='])
def test_runtime_threshold_ties(monkeypatch,tmp_path,target,operator):
    mx = require_metal('Metal required')
    import brian2 as b
    from configured_reference import reference
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(tmp_path/'cython-default'))
    model=resolve_model({'eqs':DEFAULT_EQUATIONS+'argument:1\n',
        'eq_th':f'(t<dt) or (arctan(argument){operator}cutoff)',
        'eq_rst':'v=v_rst;g=0*mV;argument=1','t_rfc':0*b.ms,
        'cutoff':float(np.arctan(np.float32(1)))})
    graph,stimulus=component_inputs(1,7,model)
    wanted=reference(graph,stimulus,model)
    prepared=configured.prepare(graph,stimulus,model)
    result,state=configured.run_prepared(prepared,stimulus,return_state=True,chunk_steps=3)
    assert result.spikes.steps.size==(1 if operator=='>' else 7)
    assert_spikes_equal(result.spikes, wanted.spikes)
    _assert_state(configured.host_state(state,prepared),wanted.state)


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
def test_runtime_linear_state_arctan(monkeypatch,tmp_path,target):
    mx = require_metal('Metal required')
    import brian2 as b
    from configured_reference import reference
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(tmp_path/'cython-default'))
    model=resolve_model({'eqs':DEFAULT_EQUATIONS+'da/dt=(-a+arctan(argument))/dt:1\nargument:1\n',
        'eq_th':'True','eq_rst':'v=v_rst;g=0*mV;argument+=0.25','t_rfc':0*b.ms})
    graph,stimulus=component_inputs(3,9,model)
    wanted=reference(graph,stimulus,model)
    prepared=configured.prepare(graph,stimulus,model)
    assert '_eon_arctan32' in prepared.program.state_code
    result,state=configured.run_prepared(prepared,stimulus,return_state=True,chunk_steps=3)
    _assert_state(configured.host_state(state,prepared),wanted.state)
    assert_spikes_equal(result.spikes, wanted.spikes)


@pytest.mark.metal
def test_runtime_actual_cpp_standalone(monkeypatch,tmp_path):
    mx = require_metal('Metal required')
    import brian2 as b
    from brian2.devices.device import get_device
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.target','auto')
    monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(tmp_path/'cython-default'))
    previous=get_device();b.set_device('cpp_standalone',build_on_run=False)
    b.device.reinit();b.device.activate(build_on_run=False)
    b.prefs.devices.cpp_standalone.extra_make_args_unix=['-j2']
    try:
        model=model_configuration(4);assert model.reference_codegen_target=='cpp'
        graph,stimulus=component_inputs(3,5,model)
        prepared=configured.prepare(graph,stimulus,model)
        clock=b.Clock(dt=model.dt_s*b.second)
        neurons=b.NeuronGroup(3,model.eqs,method='linear',threshold=model.eq_th,reset=model.eq_rst,
            refractory='rfc',namespace=model.brian_namespace(),clock=clock,dtype=_reference_dtype(model))
        neurons.v=model.v_0*b.volt;neurons.g=0*b.volt;neurons.rfc=0*b.second
        monitor=b.SpikeMonitor(neurons);network=b.Network(neurons,monitor)
        network.schedule=list(model.schedule);network.run(5*clock.dt,namespace={})
        b.device.build(directory=str(tmp_path/'cpp-default'),run=True,with_output=False)
        result,state=configured.run_prepared(prepared,stimulus,return_state=True)
        np.testing.assert_array_equal(result.spikes.neurons,np.asarray(monitor.i[:]))
        np.testing.assert_array_equal(result.spikes.steps,np.rint(np.asarray(monitor.t_[:])/model.dt_s).astype(np.int64))
        actual=configured.host_state(state,prepared)
        wanted={name:np.asarray(getattr(neurons,name+'_')[:]).copy().reshape(1,-1) for name in actual}
        _assert_state(actual,wanted)
    finally:b.set_device(previous)


@pytest.mark.metal
@pytest.mark.parametrize("target", ["numpy", "cython"])
@pytest.mark.parametrize("state_dtype", [np.float32, np.float64])
def test_original_api_arctan_composes_stored_and_promoted_widths(tmp_path, target, state_dtype):
    """Unchanged application objects preserve mixed widths and continuation."""
    import brian2 as b
    from brian2.devices.device import get_device
    from original_api_protocol import check_original_api_runs, observe_run
    from original_api_reference import build_original_api_fixture
    parameters = {
        "eqs": "v:volt\ng:volt\nrfc:second\nargument:1\nobserved:1\nwide:1\nchecksum:integer\n",
        "eq_th": "True", "t_rfc": 0*b.ms, "r_poi": 0*b.Hz,
        "r_poi2": 0*b.Hz, "big": 1e100,
        "eq_rst": "v=v_rst;g=0*mV;argument+=1/32;observed=arctan(argument);"
                  "wide=arctan(argument+big);checksum+=int(observed*1024)",
    }
    def build(mode):
        return build_original_api_fixture(tmp_path/str(mode == 'native'), parameters=parameters,
            exc=(), exc2=(), silenced=(),
            initial_state={"argument": np.array([-.75, -.25, .5, 1.5], state_dtype)})
    def exercise(fixture, mode):
        b.prefs.core.default_float_dtype = np.float64
        for duration in (.1, .2, 0., .1):
            observation = observe_run(fixture, duration)
            assert observation.snapshot.state["argument"].dtype == state_dtype
            yield observation
        if mode == 'native':
            device = get_device()
            assert device.native_runs == 4 and device.native_steps == 4
    check_original_api_runs(build, exercise, None, guard=lambda fixture, patch: None,
        target=target, float_dtype=state_dtype)


@pytest.mark.metal
@pytest.mark.parametrize('dtype',[np.float32,np.float64],ids=['float32','float64'])
@pytest.mark.parametrize('operator',['>','>='])
def test_boolean_half_threshold_ties(monkeypatch,tmp_path,dtype,operator):
    mx = require_metal('Metal required')
    import brian2 as b
    from configured_reference import reference
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',dtype)
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',True)
    monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(tmp_path/'cython-default'))
    cutoff=float((np.arctan(np.array([True],bool))+0.1)[0])
    model=resolve_model({'eqs':DEFAULT_EQUATIONS+'boolean_tie:boolean\n',
        'eq_th':f'(t<dt) or (arctan(boolean_tie)+0.1{operator}cutoff)',
        'eq_rst':'v=v_rst;g=0*mV;boolean_tie=True','t_rfc':0*b.ms,'cutoff':cutoff})
    graph,stimulus=component_inputs(3,7,model)
    wanted=reference(graph,stimulus,model,cpu_target='numpy')
    prepared=configured.prepare(graph,stimulus,model)
    result,state=configured.run_prepared(prepared,stimulus,return_state=True,chunk_steps=3)
    assert result.spikes.steps.size==(3 if operator=='>' else 21)
    assert_spikes_equal(result.spikes, wanted.spikes)
    _assert_state(configured.host_state(state,prepared),wanted.state)


@pytest.mark.metal
@pytest.mark.parametrize('dtype',[np.float32,np.float64],ids=['float32','float64'])
@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('optimise',[False,True])
def test_boolean_half_temporaries_match_actual_cpu(monkeypatch,tmp_path,dtype,target,optimise):
    mx = require_metal('Metal required')
    import brian2 as b
    from configured_reference import reference
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',dtype)
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',optimise)
    monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(tmp_path/'cython-default'))
    # Brian's statement cache omits this optimisation preference. Distinct
    # identifiers ensure each actual CPU reference uses the selected policy.
    flag=f'boolean{np.dtype(dtype).itemsize*8}_{target}_{int(optimise)}'
    parameters={'eqs':DEFAULT_EQUATIONS+f'{flag}:boolean\nobserver:1\nweak:1\nnested:1\n',
        'eq_th':'True','t_rfc':0*b.ms,
        'eq_rst':f'v=v_rst;g=0*mV;{flag}=not {flag};observer+=arctan({flag});'
                 f'weak=arctan({flag})+0.1;nested=arctan(arctan({flag}))'}
    model=resolve_model(parameters)
    assert model.reference_optimise is optimise
    graph,stimulus=component_inputs(3,9,model)
    wanted=reference(graph,stimulus,model,record_state=True,cpu_target=target)
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',not optimise)
    assert resolve_model(parameters).signature!=model.signature
    prepared=configured.prepare(graph,stimulus,model)
    assert b.prefs.codegen.loop_invariant_optimisations is (not optimise)
    for record in (True,False):
        result,state=configured.run_prepared(prepared,stimulus,record=record,return_state=True,chunk_steps=4)
        _assert_state(configured.host_state(state,prepared),wanted.state)
        assert_recorded_spikes_equal(result, wanted, record)
    state=configured.initial_state(prepared);events=mx.array(np.zeros(1,np.uint8))
    for step in range(9):
        state,_=configured.advance(prepared,state,events,9,1)
        configured.evaluate_state(state)
        _assert_state(configured.host_state(state,prepared),{name:a[:,step] for name,a in wanted.traces.items()})


@pytest.mark.metal
@pytest.mark.parametrize('integer_dtype',['int8','uint8','int16','uint16'])
def test_numpy_small_integer_loops_match_original_eon_cpu(tmp_path,integer_dtype):
    mx = require_metal('Metal required')
    from trigonometry_reference import original_reference
    from mlx_backend.numpy_half import Renderer,half_precision
    from mlx_backend import binary16,operator32,operator64
    info=np.iinfo(integer_dtype)
    inputs=np.arange(info.min,info.max+1,dtype=np.int64).astype(integer_dtype)
    expected_loop=np.dtype('float16' if info.bits==8 else 'float32')
    assert np.arctan(inputs).dtype==expected_loop
    symbol='arctan_phase_'+integer_dtype
    expression=f'arctan({symbol})+0.1'
    wanted,reference_info=original_reference('numpy','arctan',inputs,tmp_path/'cpu',
        state_inputs=True,dtype=np.float64,phase_dtype='integer',phase_symbol=symbol,
        integer_dtype=integer_dtype,optimise=True,result_expression=expression,
        cache_suffix='-arctan-'+integer_dtype+'-default')
    assert reference_info['target_resolved']=='NumpyCodeObject'
    assert reference_info['loop_optimisations'] is True
    np.save(tmp_path/'actual-eon-output.npy',wanted)
    renderer=Renderer({symbol:np.dtype(integer_dtype)},{},set(),set(),'reset',
        half_ufuncs={'arctan':('arctan_f32','_eon_arctan32')})
    call=renderer.expression(f'arctan({symbol})')
    assert call.narrow is True and half_precision(call)==(info.bits==8)
    result=renderer.expression(expression)
    header=(binary64.MSL_HEADER+binary32.MSL_HEADER+math64.MSL_MATH_HEADER+
        binary16.MSL_HEADER+binary16.MSL_F32_HEADER+
        operator64.MSL_REMAINDER_HEADER+operator32.MSL_REMAINDER_HEADER+
        operator32.MSL_UINT_CAST_HEADER+
        arctangent32.metal_arctangent32_header({'arctan'}))
    kernel=mx.fast.metal_kernel(name='arctan_numpy_small_integer',input_names=['values'],output_names=['out'],
        header=header,source='uint i=thread_position_in_grid.x;if(i<N){long '+symbol+'=values[i];out[i]='+result.code+';}',
        compile_options={'math_mode':'safe'})
    (actual,)=kernel(inputs=[mx.array(inputs)],template=[('N',len(inputs))],grid=(len(inputs),1,1),
        threadgroup=(128,1,1),output_shapes=[inputs.shape],output_dtypes=[mx.uint64])
    mx.eval(actual);raw=np.asarray(actual)
    np.save(tmp_path/'actual-metal-output-bits.npy',raw)
    np.testing.assert_array_equal(raw,wanted.view(np.uint64),err_msg=integer_dtype)
