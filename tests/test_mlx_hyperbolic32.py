"""Exact float32 function policies against independent CPU and actual Metal."""
import ast
import ctypes
from pathlib import Path
import subprocess
import numpy as np
import pytest
from native_execution import require_metal
from mlx_backend import binary64,math64,hyperbolic32
from pinned_apple_math import require_apple_arm64_reference
NAMES=('log10','sinh','cosh','tanh_scalar','tanh_numpy')

def comparison(spikes,state,expected_spikes,expected_state):
    differences={}
    for name,wanted in expected_state.items():
        actual=state[name]
        if actual.dtype != wanted.dtype:
            differences[name]={"actual_dtype":str(actual.dtype),"expected_dtype":str(wanted.dtype)}
            continue
        raw=lambda x:x.view(np.uint32 if x.dtype.itemsize==4 else np.uint64) if x.dtype.kind=='f' else x
        different=raw(actual)!=raw(wanted)
        if np.any(different):
            index=tuple(np.argwhere(different)[0])
            differences[name]={"actual":str(actual[index]),"expected":str(wanted[index])}
    return {"spikes_exact":bool(np.array_equal(spikes,expected_spikes)),"raw_state_exact":not differences,
            "state_differences":differences}

def cases():
    centers=[0.,-0.,.125,.5,.75,1.,1.1875,1.4375,2.,8.,9.,10.,89.,2**-27,2**-20,2**-12]
    centers+=[np.array(x,np.uint32).view(np.float32).item() for x in
        (0x42b2d4fd,0x411a3902,0x411b4359,0x41102cb4,0x4057ffff,0x3ff40000,0x39a9d80b)]
    centers+=[2.**(k/2-5) for k in range(17)]
    centers+=[(1+j/64)*2.**k for k in (-126,-1,0,1,100,127) for j in range(65)]
    centers+=[(j+.5)*np.log(2)/128 for j in range(128)]
    with np.errstate(all='ignore'):
        centers=np.array([s*v for v in centers for s in (-1,1)],np.float32)
        boundary=np.concatenate((np.nextafter(centers,np.float32(-np.inf)),centers,np.nextafter(centers,np.float32(np.inf)))).view(np.uint32)
    special=np.array([0,1,0x80000000,0x80000001,0x7fffff,0x800000,0x7f7fffff,
        0x7f800000,0xff800000,0x7f800001,0xff800001,0x7fc12345,0xffc00001],np.uint32)
    random=np.random.default_rng(20261018).integers(0,0xffffffff,4096,dtype=np.uint32)
    return np.unique(np.concatenate((boundary,special,random)))

def reference(name,raw,host=None):
    require_apple_arm64_reference()
    if name != 'tanh_numpy' and host is not None:
        out=np.empty_like(raw);getattr(host.scalar,'run_'+name.split('_')[0])(raw,out,len(raw));return out
    if name=='tanh_scalar':
        lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib');fn=lib.tanhf
        fn.argtypes,fn.restype=[ctypes.c_float],ctypes.c_float
        return np.asarray([fn(v) for v in raw.view(np.float32)],np.float32).view(np.uint32)
    with np.errstate(all='ignore'):
        return np.asarray(getattr(np,name.split('_')[0])(raw.view(np.float32)),np.float32).view(np.uint32)

def assert_bits(actual,expected,raw,name):
    indices=np.flatnonzero(actual!=expected)
    if len(indices):
        i=indices[0]
        pytest.fail(f'{name} input={int(raw[i]):#010x} actual={int(actual[i]):#010x} reference={int(expected[i]):#010x}; {len(indices)}/{len(raw)} differ')

@pytest.fixture(scope='module')
def host(tmp_path_factory):
    require_apple_arm64_reference()
    tree=ast.parse(Path(__file__).with_name('test_mlx_math64.py').read_text())
    prefix=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='HOST_PREFIX' for t in n.targets))
    out=tmp_path_factory.mktemp('hyperbolic32-host');src,so=out/'math.cpp',out/'math.so'
    header=hyperbolic32.metal_hyperbolic32_header({'log10','sinh','cosh','tanh_scalar','tanh_numpy'},'numpy')
    wrappers='extern "C" {\n'+''.join(f'void run_{n}(const uint*x,uint*y,int count){{for(int i=0;i<count;++i)y[i]=f_hyper_{n}(x[i]);}}\n' for n in NAMES)+'}\n'
    src.write_text(prefix+'\n#define constant static const\n#define fma std::fma\n'+binary64.MSL_HEADER+math64.MSL_MATH_HEADER+header+wrappers)
    compiled=subprocess.run(['clang++','-std=c++17','-shared','-fPIC','-O2','-fno-fast-math','-ffp-contract=off',str(src),'-o',str(so)],capture_output=True,text=True,timeout=30)
    assert compiled.returncode==0,compiled.stderr
    lib=ctypes.CDLL(str(so));ptr=np.ctypeslib.ndpointer(dtype=np.uint32,flags='C_CONTIGUOUS')
    for n in NAMES:
        fn=getattr(lib,'run_'+n);fn.argtypes,fn.restype=[ptr,ptr,ctypes.c_int],None
    from brian2.codegen.cpp_prefs import get_compiler_and_args
    scalar_source,scalar_so=out/'reference.cpp',out/'reference.so'
    scalar_source.write_text(prefix+'\nextern "C" {\n'+''.join(
        f'void run_{n}(const uint*x,uint*y,int count){{for(int i=0;i<count;++i)y[i]=as_type<uint>(std::{n}(as_type<float>(x[i])));}}\n'
        for n in ('log10','sinh','cosh','tanh'))+'}\n')
    _,flags=get_compiler_and_args()
    compiled=subprocess.run(['clang++','-std=c++17','-shared','-fPIC',*flags,str(scalar_source),'-o',str(scalar_so)],capture_output=True,text=True,timeout=30)
    assert compiled.returncode==0,compiled.stderr
    lib.scalar=ctypes.CDLL(str(scalar_so))
    for n in ('log10','sinh','cosh','tanh'):
        fn=getattr(lib.scalar,'run_'+n);fn.argtypes,fn.restype=[ptr,ptr,ctypes.c_int],None
    return lib

def test_host_exact_independent_float32_policies(host):
    raw=cases()
    for n in NAMES:
        result=np.empty_like(raw);getattr(host,'run_'+n)(raw,result,len(raw))
        assert_bits(result,reference(n,raw,host),raw,n)
        if n in ('log10','sinh','cosh'): assert_bits(result,reference(n,raw),raw,n+'_numpy')

@pytest.mark.metal
def test_metal_exact_independent_float32_policies(host):
    mx = require_metal('Metal required')
    raw=cases()
    for target,names in [('cython',NAMES[:-1]),('numpy',('tanh_numpy',))]:
        header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+hyperbolic32.metal_hyperbolic32_header({n.split('_')[0] for n in names},target)
        for n in names:
            kernel=mx.fast.metal_kernel(name='hyper32_'+n,input_names=['x'],output_names=['out'],header=header,
                source=f'uint i=thread_position_in_grid.x;if(i<COUNT)out[i]=f_hyper_{n}(x[i]);',compile_options={'math_mode':'safe'})
            out,=kernel(inputs=[mx.array(raw)],template=[('COUNT',len(raw))],grid=(len(raw),1,1),threadgroup=(128,1,1),output_shapes=[raw.shape],output_dtypes=[mx.uint32])
            mx.eval(out);assert_bits(np.asarray(out),reference(n,raw,host),raw,n)

def model_configuration(promoted=False):
    import brian2 as b
    from mlx_backend.model import resolve_model
    argument='x+(i+1)/16384' if promoted else 'x'
    return resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\nx:1\nchecksum:integer\na:1\nb:1\nc:1\nd:1\nf_hyper_log10:1\nf_hyper_sinh:1\nf_hyper_cosh:1\nf_hyper_tanh:1\n',
        'eq_th':'True','t_rfc':0*b.ms,
        'eq_rst':f'v=v_rst;g=0*mV;x+=1/32;a=log10(1+{argument});b=sinh({argument});c=cosh({argument});d=tanh({argument});f_hyper_log10+=1;f_hyper_sinh+=1;f_hyper_cosh+=1;f_hyper_tanh+=1;checksum+=int(a*1024)+int(b*1024)+int(c*1024)+int(d*1024)'})

@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('promoted',[False,True])
def test_actual_brian_float32_model(monkeypatch,tmp_path,target,promoted):
    import brian2 as b
    b.BrianLogger.log_level_error()
    mx=pytest.importorskip('mlx.core')
    from mlx_backend import configured
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    from component_cpu_reference import HyperbolicComponentCpu as ComponentCpu,component_inputs
    model=model_configuration(promoted)
    assert model.state_dtype=='float32' and model.reference_codegen_target==target
    cache=Path(__file__).resolve().parents[1]/'executions/mlx-feature-parity/hyperbolic32-cython'/tmp_path.name
    cpu=ComponentCpu(7,5,model,target,cache)
    _,spikes,reference=cpu.run()
    graph,stimulus=component_inputs(7,5,model);prepared=configured.prepare(graph,stimulus,model)
    result,state=configured.run_prepared(prepared,stimulus,record=False,return_state=True)
    actual=configured.host_state(state,prepared)
    diff=comparison(np.column_stack((result.spikes.neurons,result.spikes.steps)),actual,spikes,reference)
    if not (diff['spikes_exact'] and diff['raw_state_exact']):
        pytest.fail(str(diff['state_differences']))
    assert ('_eon_hyper_' if promoted else '_eon_hyper32_') in prepared.program.reset_code

@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
def test_float32_tanh_threshold_tie_continuation_and_no_record(monkeypatch,tmp_path,target):
    import brian2 as b
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    mx=pytest.importorskip('mlx.core');b.BrianLogger.log_level_error()
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    from component_cpu_reference import HyperbolicComponentCpu as ComponentCpu,component_inputs
    seed=float(np.uint32(0x39a9d80b).view(np.float32))
    boundary=float(np.uint32(0x39a9d80a).view(np.float32))
    model=resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\nx:1\ncount:integer\n',
        'eq_th':'(t<dt) or (tanh(x)>boundary)', 'eq_rst':'v=v_rst;g=0*mV;x=seed;count+=1',
        'seed':seed,'boundary':boundary,'t_rfc':0*b.ms})
    cache=Path(__file__).resolve().parents[1]/'executions/mlx-feature-parity/hyperbolic32-cython'/tmp_path.name
    cpu=ComponentCpu(3,5,model,target,cache)
    monitor=b.SpikeMonitor(cpu.neurons);cpu.network.add(monitor);cpu.network.store('initial')
    _,_,reference=cpu.run()
    expected=np.column_stack((np.asarray(monitor.i[:],np.int64),np.rint(np.asarray(monitor.t_[:])/model.dt_s).astype(np.int64)))
    assert len(expected)==(3 if target=='numpy' else 15)
    graph,stimulus=component_inputs(3,5,model);prepared=configured.prepare(graph,stimulus,model)
    assert '_eon_hyper32_tanh_'+('numpy' if target=='numpy' else 'scalar') in prepared.program.threshold_code
    for record in (True,False):
        result,state=configured.run_prepared(prepared,stimulus,record=record,return_state=True,chunk_steps=2)
        wanted_spikes=expected if record else np.empty((0,2),np.int64)
        diff=comparison(np.column_stack((result.spikes.neurons,result.spikes.steps)),configured.host_state(state,prepared),wanted_spikes,reference)
        assert diff['spikes_exact'] and diff['raw_state_exact'],str(diff)
    state=configured.initial_state(prepared);events=mx.array(np.zeros(1,np.uint8));spikes=[]
    for length in (2,3):
        begin=state.step;state,words=configured.advance(prepared,state,events,5,length)
        configured.evaluate_state(state)
        for bit in range(length):
            for row in np.flatnonzero(np.asarray(words)&(1<<bit)):spikes.append((int(row),begin+bit))
    actual=np.asarray(sorted(spikes,key=lambda row:(row[1],row[0])),np.int64)
    diff=comparison(actual,configured.host_state(state,prepared),expected,reference)
    assert diff['spikes_exact'] and diff['raw_state_exact'],str(diff)

@pytest.mark.metal
def test_float32_actual_cpp_standalone(monkeypatch,tmp_path):
    import brian2 as b
    from brian2.devices.device import get_device
    from mlx_backend import configured
    mx=pytest.importorskip('mlx.core');b.BrianLogger.log_level_error()
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.target','auto')
    from component_cpu_reference import component_inputs
    previous=get_device();b.set_device('cpp_standalone',build_on_run=False)
    b.device.reinit();b.device.activate(build_on_run=False)
    try:
        model=model_configuration();assert model.reference_codegen_target=='cpp'
        graph,stimulus=component_inputs(3,5,model);prepared=configured.prepare(graph,stimulus,model)
        clock=b.Clock(dt=model.dt_s*b.second)
        neurons=b.NeuronGroup(3,model.eqs,method='linear',threshold=model.eq_th,reset=model.eq_rst,
            refractory='rfc',namespace=model.brian_namespace(),clock=clock,dtype={name:np.dtype(var.dtype) for name,var in model.analysis.variables.items()})
        neurons.v=model.v_0*b.volt;neurons.g=0*b.volt;neurons.rfc=0*b.second
        monitor=b.SpikeMonitor(neurons);network=b.Network(neurons,monitor);network.schedule=list(model.schedule)
        network.run(5*model.dt_s*b.second,namespace={})
        b.device.build(directory=str(tmp_path/'cpp'),compile=True,run=True,clean=True,debug=False,with_output=False)
        result,state=configured.run_prepared(prepared,stimulus,return_state=True)
        np.testing.assert_array_equal(result.spikes.neurons,np.asarray(monitor.i[:]))
        np.testing.assert_array_equal(result.spikes.steps,np.rint(np.asarray(monitor.t_[:])/model.dt_s).astype(np.int64))
        for name,values in configured.host_state(state,prepared).items():
            wanted=np.asarray(getattr(neurons,name+'_')[:]).copy().reshape(1,-1)
            assert values.dtype==wanted.dtype,name
            raw=lambda x:x.view(np.uint32 if x.dtype.itemsize==4 else np.uint64) if x.dtype.kind=='f' else x
            np.testing.assert_array_equal(raw(values),raw(wanted),err_msg=name)
    finally:b.set_device(previous)

def test_float32_header_selection():
    assert hyperbolic32.metal_hyperbolic32_header(())==''
    assert 'hyper32_exp_table' not in hyperbolic32.metal_hyperbolic32_header({'log10'})
    assert 'hyper32_exp_table' not in hyperbolic32.metal_hyperbolic32_header({'tanh'},'numpy')
    assert 'f_hyper_tanh_scalar' in hyperbolic32.metal_hyperbolic32_header({'tanh'},'cython')
    half_only=hyperbolic32.metal_hyperbolic32_header({'tanh_scalar'},'numpy')
    assert '_eon_hyper32_tanh_scalar' in half_only
    assert 'f_hyper_tanh_numpy' not in half_only

def test_float32_invariant_width_matches_pinned_reference(host):
    from mlx_backend.compiler import _Renderer
    from mlx_backend.numpy32 import Renderer
    raw=np.array([0x4057ffff,0x3ff40000,0x39a9d80b],np.uint32)
    values={'seed':raw.view(np.float32)[-1]}
    for target in ('numpy','cython','cpp'):
        result=_Renderer({},values,reference_target=target).expression('tanh(seed)')
        wanted=reference('tanh_numpy' if target=='numpy' else 'tanh_scalar',raw[-1:],host)[0]
        assert result.kind=='float32'
        assert result.code==f'{int(wanted)}u'
    numpy=Renderer({},values,set(),set(),'reset',reference_target='numpy')
    result=numpy.expression('sinh(seed)')
    assert result.narrow is True

@pytest.mark.metal
@pytest.mark.parametrize('active',[True,False])
def test_merged_float32_input_zero_draw_and_skipped_runner(monkeypatch,active):
    import brian2 as b
    from dataclasses import replace
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    mx=pytest.importorskip('mlx.core');b.BrianLogger.log_level_error()
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    model=resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\n','eq_th':'False',
        'eq_rst':'v=v_rst;g=0*mV','v_0':-0.0*b.volt,'t_rfc':0*b.ms})
    clock=b.Clock(dt=model.dt_s*b.second)
    neurons=b.NeuronGroup(1,model.eqs,method='linear',threshold=model.eq_th,reset=model.eq_rst,
        refractory='rfc',namespace=model.brian_namespace(),clock=clock,
        dtype={name:np.dtype(var.dtype) for name,var in model.analysis.variables.items()})
    neurons.v=-0.0*b.volt;neurons.g=0*b.volt;neurons.rfc=0*b.second
    poisson=b.PoissonInput(neurons,'v',N=1,rate=0*b.Hz,weight=1*b.mV);poisson.active=active
    network=b.Network(neurons,poisson);network.run(model.dt_s*b.second,namespace={})
    expected=np.asarray(neurons.v_[:],np.float32).view(np.uint32)
    np.testing.assert_array_equal(expected,[0 if active else 0x80000000])
    graph,stimulus=component_inputs(1,1,model)
    events=np.array([[[0 if active else 2]]],np.uint8)
    stimulus=replace(stimulus,events=events,targets=(0,),amplitudes_mv=(1.0,))
    prepared=configured.prepare(graph,stimulus,model)
    state=configured.initial_state(prepared)
    state,_=configured.advance(prepared,state,mx.array(events.reshape(-1)),1,1)
    configured.evaluate_state(state)
    actual=configured.host_state(state,prepared)['v'].reshape(-1).view(np.uint32)
    np.testing.assert_array_equal(actual,expected)

@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
def test_original_api_float32_functions_keep_actual_objects_and_policy(tmp_path,target):
    """Four actual application neurons, later preference changes and continuation."""
    import brian2 as b
    from original_api_protocol import check_original_api_runs,observe_run
    from original_api_reference import build_original_api_fixture
    b.BrianLogger.log_level_error()
    params={'eqs':'v:volt\ng:volt\nrfc:second\nx:1\na:1\nb:1\nc:1\nd:1\nchecksum:integer\n',
        'eq_th':'True','t_rfc':0*b.ms,'r_poi':0*b.Hz,'r_poi2':0*b.Hz,
        'eq_rst':'v=v_rst;g=0*mV;x+=1/32;a=log10(1+x);b=sinh(x);c=cosh(x);d=tanh(x);checksum+=int(a*1024)+int(b*1024)+int(c*1024)+int(d*1024)'}
    def build(mode):
        return build_original_api_fixture(tmp_path/str(mode=='native'),parameters=params,
            exc=(),exc2=(),silenced=(),initial_state={'x':np.arange(4,dtype=np.float32)/16})
    def exercise(fixture,mode):
        assert fixture.neurons.variables['x'].dtype==np.float32
        # These later globals affect code-generation temporaries, not the
        # already allocated state. The bridge must retain both facts.
        b.prefs.core.default_float_dtype=np.float64
        for duration in (.1,.2):
            observation=observe_run(fixture,duration)
            assert observation.snapshot.state['x'].dtype==np.float32
            yield observation
    check_original_api_runs(build,exercise,None,guard=lambda fixture,patch:None,
        target=target,float_dtype=np.float32)

@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
def test_float32_function_state_update_context(monkeypatch,tmp_path,target):
    """Storage and actual function-call width can differ within an ODE."""
    import brian2 as b
    from component_cpu_reference import HyperbolicComponentCpu,component_inputs
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    b.BrianLogger.log_level_error()
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    model=resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\nx:1\nda/dt=-a/tau+(log10(1+x)+sinh(x)+cosh(x)+tanh(x))/second:1\nchecksum:integer\n',
        'eq_th':'True','t_rfc':0*b.ms,
        'eq_rst':'v=v_rst;g=0*mV;x+=1/32;checksum+=int(a*65536)'})
    cpu=HyperbolicComponentCpu(3,7,model,target,tmp_path/'cython')
    _,spikes,expected=cpu.run()
    graph,stimulus=component_inputs(3,7,model)
    prepared=configured.prepare(graph,stimulus,model)
    result,state=configured.run_prepared(prepared,stimulus,return_state=True,record=False)
    actual=configured.host_state(state,prepared)
    difference=comparison(np.column_stack((result.spikes.neurons,result.spikes.steps)),actual,spikes,expected)
    assert difference['spikes_exact'] and difference['raw_state_exact'],difference
    for name in ('log10','sinh','cosh','tanh'):
        assert any('_eon_hyper'+prefix+name in prepared.program.dynamics_code for prefix in ('_','32_'))

@pytest.mark.metal
def test_numpy_header_keeps_scalar_and_vector_tanh_policies_distinct(host):
    """Half ufunc scalar32 and ordinary NumPy32 calls can share one kernel."""
    mx=pytest.importorskip('mlx.core')
    raw=np.array([0x39a9d80b,0x3f800000,0],np.uint32)
    header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+hyperbolic32.metal_hyperbolic32_header(
        {'tanh_scalar','tanh_numpy'},'numpy')
    kernel=mx.fast.metal_kernel(name='mixed_tanh_policies',input_names=['x'],output_names=['out'],
        header=header,source='uint i=thread_position_in_grid.x;if(i<COUNT){out[2*i]=_eon_hyper32_tanh_scalar(x[i]);out[2*i+1]=_eon_hyper32_tanh_numpy(x[i]);}',
        compile_options={'math_mode':'safe'})
    out,=kernel(inputs=[mx.array(raw)],template=[('COUNT',len(raw))],grid=(len(raw),1,1),
        threadgroup=(32,1,1),output_shapes=[(len(raw),2)],output_dtypes=[mx.uint32])
    actual=np.asarray(out)
    assert_bits(actual[:,0],reference('tanh_scalar',raw,host),raw,'scalar')
    assert_bits(actual[:,1],reference('tanh_numpy',raw),raw,'numpy')
    assert actual[0,0]!=actual[0,1]

def test_numpy_half_function_scalar32_policy_all_encodings(host):
    """Pin the scalar function policy independently of half renderer plumbing."""
    raw=np.arange(65536,dtype=np.uint16)
    values=raw.view(np.float16)
    with np.errstate(all='ignore'):
        source=values.astype(np.float32).view(np.uint32)
    for name in ('log10','sinh','cosh','tanh'):
        result=np.empty_like(source)
        getattr(host,'run_'+name+('_scalar' if name=='tanh' else ''))(source,result,len(source))
        with np.errstate(all='ignore'):
            actual=result.view(np.float32).astype(np.float16).view(np.uint16)
            expected=getattr(np,name)(values).view(np.uint16)
        different=np.flatnonzero(actual!=expected)
        if len(different):
            index=different[0]
            pytest.fail(f'{name} half input={int(raw[index]):#06x} actual={int(actual[index]):#06x} '
                        f'NumPy={int(expected[index]):#06x}; {len(different)}/{len(raw)} differ')

@pytest.mark.metal
def test_numpy_half_function_loops_all_encodings():
    """Half ufuncs use scalar32 math followed by the shared half conversion."""
    from mlx_backend import binary16,binary32
    mx=pytest.importorskip('mlx.core')
    raw=np.arange(65536,dtype=np.uint16)
    values=raw.view(np.float16)
    source=raw.astype(np.uint32)
    header=(binary64.MSL_HEADER+binary32.MSL_HEADER+math64.MSL_MATH_HEADER+
        binary16.MSL_HEADER+binary16.MSL_F32_HEADER+
        hyperbolic32.metal_hyperbolic32_header({'log10','sinh','cosh','tanh_scalar'},'numpy'))
    for name in ('log10','sinh','cosh','tanh'):
        suffix='_scalar' if name=='tanh' else ''
        with np.errstate(all='ignore'):
            expected=getattr(np,name)(values).view(np.uint16)
        kernel=mx.fast.metal_kernel(name='hyper_half_'+name,input_names=['x'],output_names=['out'],
            header=header,source=f'uint i=thread_position_in_grid.x;if(i<COUNT)out[i]=h_from_d(h_f32_to_d(_eon_hyper32_{name}{suffix}(h_d_to_f(h_to_d(x[i])))));',
            compile_options={'math_mode':'safe'})
        out,=kernel(inputs=[mx.array(source)],template=[('COUNT',len(raw))],grid=(len(raw),1,1),
            threadgroup=(128,1,1),output_shapes=[raw.shape],output_dtypes=[mx.uint32])
        actual=np.asarray(out).astype(np.uint16)
        different=np.flatnonzero(actual!=expected)
        if len(different):
            index=different[0]
            pytest.fail(f'{name} half input={int(raw[index]):#06x} actual={int(actual[index]):#06x} '
                        f'NumPy={int(expected[index]):#06x}; {len(different)}/{len(raw)} differ')
