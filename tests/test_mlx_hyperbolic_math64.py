"""Exact target-specific functions against independent CPU and actual Metal."""
from state_assertions import assert_exact_state, assert_array_bits_equal
import ast
import ctypes
from pathlib import Path
import subprocess

import numpy as np
import pytest
from native_execution import require_metal
from pinned_apple_math import require_apple_arm64_reference

from mlx_backend import binary64, hyperbolic_math64, math64

NAMES = ("log10", "sinh", "cosh", "tanh_scalar", "tanh_numpy")


def cases():
    import math
    centers = [0., -0., .1, -.1, 1., -1., .125, -.125, 2., -2., 21., -21.,
        2**-27, -2**-27, 2**-30, -2**-30, 709., 710., -709., -710.,
        float.fromhex("0x1.30fc1931f09cap+4"),
        float.fromhex("0x1.62e4200000000p-2"),
        float.fromhex("0x1.633ce8fb9f87ep+9")]
    centers += [sign*j/4 for sign in (-1, 1) for j in range(1, 9)]
    centers += [sign*2**(j/2-3) for sign in (-1, 1) for j in range(17)]
    centers += [(1+j/128)*2.**k for k in (-1022, -1, 0, 1, 52, 53, 1022)
                for j in (0, 1, 63, 64, 65, 126, 127)]
    centers += [sign*(k+j/128)*math.log(2) for sign in (-1,1)
                for k in (0, 1, 30, 1023, 1024) for j in (0, 1, 63, 64, 127)]
    values = [y for x in centers for y in
              (math.nextafter(x, -math.inf), x, math.nextafter(x, math.inf))]
    special = np.array([0, 1, 0x8000000000000000, 0x8000000000000001,
        0x7ff0000000000000, 0xfff0000000000000, 0x7ff0000000000001,
        0xfff0000000000001, 0x7ff8123456789abc, 0xfff8000000000001], np.uint64)
    random = np.random.default_rng(20261018).integers(0, np.iinfo(np.uint64).max,
                                                    4096, dtype=np.uint64)
    # Same dyadic grid as the component, through every step's phase.
    grid = (np.arange(1, 4096+129, dtype=np.float64)/16384).view(np.uint64)
    return np.unique(np.r_[np.array(values).view(np.uint64), special, random, grid])


def expected(name, raw):
    require_apple_arm64_reference()
    from brian2.core.functions import DEFAULT_FUNCTIONS
    if name == "tanh_scalar":
        lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
        fn = lib.tanh
        fn.argtypes, fn.restype = [ctypes.c_double], ctypes.c_double
        return np.asarray([fn(x) for x in raw.view(np.float64)]).view(np.uint64)
    with np.errstate(all="ignore"):
        return np.asarray(DEFAULT_FUNCTIONS[name.split('_')[0]].pyfunc(raw.view(np.float64)),
                          np.float64).view(np.uint64)


def assert_bits(actual, reference, raw, name):
    different = np.flatnonzero(actual != reference)
    if len(different):
        i = different[0]
        pytest.fail(f"{name} input={int(raw[i]):#018x}: actual={int(actual[i]):#018x}, "
                    f"reference={int(reference[i]):#018x}; {len(different)}/{len(raw)} differ")


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    module = ast.parse(Path(__file__).with_name("test_mlx_math64.py").read_text())
    prefix = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "HOST_PREFIX" for t in n.targets))
    out = tmp_path_factory.mktemp("hyperbolic-host")
    source, library = out/"math.cpp", out/"math.so"
    header = hyperbolic_math64.metal_hyperbolic_header({"log10","sinh","cosh","tanh"}, "cython")
    header += hyperbolic_math64._TANH_NUMPY_HEADER
    wrappers = '\nextern "C" {\n' + ''.join(
        f"void run_{name}(const ulong*x,ulong*y,int count) "
        f"{{for(int i=0;i<count;++i)y[i]=d_hyper_{name}(x[i]);}}\n" for name in NAMES)+r'''
void run_fma(const ulong*a,const ulong*b,const ulong*c,ulong*y,int count) {
    for(int i=0;i<count;++i)y[i]=hyper_fma_normal(a[i],b[i],c[i]);
}
void reference_fma(const ulong*a,const ulong*b,const ulong*c,ulong*y,int count) {
    for(int i=0;i<count;++i)y[i]=as_type<ulong>(std::fma(
        as_type<double>(a[i]),as_type<double>(b[i]),as_type<double>(c[i])));
}
}
'''
    source.write_text(prefix+'\n#define constant static const\n'+binary64.MSL_HEADER
                      +math64.MSL_MATH_HEADER+header+wrappers)
    subprocess.run(["clang++","-std=c++17","-shared","-fPIC","-O2","-fno-fast-math",
                    "-ffp-contract=off",str(source),"-o",str(library)],
                   check=True,capture_output=True,text=True,timeout=30)
    result = ctypes.CDLL(str(library))
    pointer = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
    for name in NAMES:
        function = getattr(result,"run_"+name)
        function.argtypes, function.restype = [pointer,pointer,ctypes.c_int], None
    for name in ("run_fma","reference_fma"):
        function=getattr(result,name)
        function.argtypes,function.restype=[pointer,pointer,pointer,pointer,ctypes.c_int],None
    return result


def normal_fma_cases():
    from test_mlx_stable_math64 import fma_cases
    raw=fma_cases()
    exponents=(raw>>np.uint64(52))&np.uint64(2047)
    normal=((exponents>0)&(exponents<2047))|((raw<<np.uint64(1))==0)
    raw=raw[np.all(normal,axis=1)]
    zero=[0,0x8000000000000000]
    nonzero=[0x10000000000000,0x3ff0000000000000,0xbff0000000000000,0x7fefffffffffffff]
    extra=np.array([(a,b,c) for a in zero+nonzero for b in zero+nonzero
                    for c in zero+nonzero if a in zero or b in zero or c in zero],np.uint64)
    return np.ascontiguousarray(np.concatenate((raw,extra)))


def test_host_normal_only_fma_matches_hardware_including_signed_zeros(host):
    raw=normal_fma_cases()
    cols=[np.ascontiguousarray(raw[:,i]) for i in range(3)]
    actual,reference=np.empty(len(raw),np.uint64),np.empty(len(raw),np.uint64)
    host.run_fma(*cols,actual,len(raw));host.reference_fma(*cols,reference,len(raw))
    np.testing.assert_array_equal(actual,reference)


@pytest.mark.metal
def test_actual_metal_normal_only_fma_matches_hardware(host):
    mx = require_metal('Metal required')
    raw=normal_fma_cases()
    cols=[np.ascontiguousarray(raw[:,i]) for i in range(3)]
    reference=np.empty(len(raw),np.uint64)
    host.reference_fma(*cols,reference,len(raw))
    kernel=mx.fast.metal_kernel(name="hyper_normal_fma",input_names=["a","b","c"],output_names=["out"],
        source="uint i=thread_position_in_grid.x;if(i<COUNT)out[i]=hyper_fma_normal(a[i],b[i],c[i]);",
        header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+hyperbolic_math64.metal_hyperbolic_header({"log10"}),
        compile_options={"math_mode":"safe"})
    output,=kernel(inputs=[mx.array(v) for v in cols],template=[("COUNT",len(raw))],grid=(len(raw),1,1),
        threadgroup=(128,1,1),output_shapes=[reference.shape],output_dtypes=[mx.uint64])
    mx.eval(output);np.testing.assert_array_equal(np.asarray(output),reference)


def test_host_matches_independent_target_bits(host):
    raw=cases()
    for name in NAMES:
        actual=np.empty_like(raw)
        getattr(host,"run_"+name)(raw,actual,len(raw))
        assert_bits(actual,expected(name,raw),raw,name)


@pytest.mark.metal
def test_actual_metal_matches_independent_target_bits():
    mx = require_metal('Metal required')
    raw=cases()
    for target,names in (("cython",NAMES[:-1]),("numpy",("tanh_numpy",))):
        header=hyperbolic_math64.metal_hyperbolic_header({n.split('_')[0] for n in names},target)
        for name in names:
            kernel=mx.fast.metal_kernel(name="direct_hyper_"+name,input_names=["x"],output_names=["out"],
                source=f"uint i=thread_position_in_grid.x;if(i<COUNT)out[i]=d_hyper_{name}(x[i]);",
                header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+header,
                compile_options={"math_mode":"safe"})
            output,=kernel(inputs=[mx.array(raw)],template=[("COUNT",len(raw))],
                grid=(len(raw),1,1),threadgroup=(128,1,1),output_shapes=[raw.shape],output_dtypes=[mx.uint64])
            mx.eval(output)
            assert_bits(np.asarray(output),expected(name,raw),raw,name)


def test_header_selector_preserves_target_and_only_used_dependencies():
    assert hyperbolic_math64.metal_hyperbolic_header(()) == ""
    log=hyperbolic_math64.metal_hyperbolic_header({"log10"})
    assert "d_hyper_log10(" in log and "hyper_exp_table" not in log
    numpy=hyperbolic_math64.metal_hyperbolic_header({"tanh"},"numpy")
    assert "d_hyper_tanh_numpy(" in numpy and "hyper_exp_table" not in numpy
    scalar=hyperbolic_math64.metal_hyperbolic_header({"tanh"},"cython")
    assert "d_hyper_tanh_scalar(" in scalar and "d_hyper_tanh_numpy(" not in scalar


def model_configuration():
    import brian2 as b
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    return resolve_model({"eqs":DEFAULT_EQUATIONS+"a:1\nb:1\nc:1\nd:1\nphase:integer\n",
        "eq_th":"True","t_rfc":0*b.ms,
        "eq_rst":"v=v_rst;g=0*mV;a=log10((i+phase+1)/32);b=sinh((i-1+phase)/32)"
                 ";c=cosh((i-1+phase)/32);d=tanh((i-1+phase)/32);phase+=1"})


@pytest.mark.metal
@pytest.mark.parametrize("target",["numpy","cython","auto"])
def test_independent_target_model_resets_and_no_record(monkeypatch,tmp_path,target):
    mx = require_metal('Metal required')
    import brian2 as b
    from brian2.devices.device import get_device
    from mlx_backend import configured
    monkeypatch.setitem(b.prefs,"codegen.target",target)
    from component_cpu_reference import ComponentCpu,component_inputs
    model=model_configuration()
    resolved="numpy" if target=="numpy" else "cython"
    assert model.reference_codegen_target==resolved
    cache=Path(__file__).resolve().parents[1]/"executions/mlx-feature-parity/hyperbolic-cython"/tmp_path.name
    cpu=ComponentCpu(7,5,model,resolved,cache)
    _,_,reference=cpu.run()
    graph,stimulus=component_inputs(7,5,model)
    prepared=configured.prepare(graph,stimulus,model)
    suffix="numpy" if resolved=="numpy" else "scalar"
    assert "_eon_hyper_tanh_"+suffix in prepared.program.reset_code
    for recording in (False,True):
        result,state=configured.run_prepared(prepared,stimulus,return_state=True,record=recording)
        actual=configured.host_state(state,prepared)
        assert result.spikes.steps.size==(35 if recording else 0)
        assert_exact_state(actual,reference)


def test_tanh_target_identity_does_not_alias(monkeypatch):
    import brian2 as b
    from mlx_backend.compiler import compile_model
    models=[]
    for target in ("numpy","cython","auto"):
        monkeypatch.setitem(b.prefs,"codegen.target",target)
        model=model_configuration();models.append(model)
        program=compile_model(model)
        assert "_eon_hyper_tanh_"+("numpy" if target=="numpy" else "scalar") in program.reset_code
    assert models[0].signature!=models[1].signature
    assert models[1].signature != models[2].signature
    explicit = dict(models[1].integer_policy.identity)
    automatic = dict(models[2].integer_policy.identity)
    assert explicit.pop("requested_target") == "cython"
    assert automatic.pop("requested_target") == "auto"
    assert explicit == automatic
    assert models[1].reference_codegen_target == models[2].reference_codegen_target == "cython"


def test_tanh_scalar_and_numpy_reference_rounding_differ():
    require_apple_arm64_reference()
    raw=np.array([float.fromhex("0x1.fffffffffffffp-2")]).view(np.uint64)
    assert expected("tanh_numpy",raw)[0]!=expected("tanh_scalar",raw)[0]


@pytest.mark.metal
def test_numpy_dynamic_functions_in_odes_thresholds_and_manual_continuation(monkeypatch):
    mx = require_metal('Metal required')
    import brian2 as b
    from configured_reference import reference
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    from test_mlx_configured import graph,inputs
    monkeypatch.setitem(b.prefs,"codegen.target","numpy")
    model=resolve_model({"eqs":DEFAULT_EQUATIONS+
        "da/dt=-a*sinh((i+1)/32)/tau:1\n"
        "db/dt=-b*cosh((i+1)/32)/tau:1\n"
        "dc/dt=-c*tanh((i+1)/32)/tau:1\n"
        "dd/dt=-d*(1+log10(1+(i+1)/32))/tau:1\n",
        "eq_th":"(t<dt) or ((sinh(a)>1) and (cosh(b)>1) and (tanh(c)>0) and (log10(1+d)>0))",
        "eq_rst":"v=v_rst;g=0*mV;a+=1;b+=1;c+=1;d+=1","t_rfc":0*b.ms})
    g,stimulus=graph(),inputs(model,targets=(),steps=21,trials=2)
    wanted=reference(g,stimulus,model)
    prepared=configured.prepare(g,stimulus,model)
    for function in ("log10","sinh","cosh","tanh"):
        assert "_eon_hyper_"+function in prepared.program.threshold_code
    for record in (True,False):
        actual,state=configured.run_prepared(prepared,stimulus,record=record,
            return_state=True,chunk_steps=7,block_windows=1)
        if record:
            for field in ("trials","neurons","steps"):
                np.testing.assert_array_equal(getattr(actual.spikes,field),getattr(wanted.spikes,field))
        else:assert actual.spikes.steps.size==0
        assert_exact_state(configured.host_state(state,prepared),wanted.state)
    events=mx.array(np.zeros(1,np.uint8))
    state=configured.initial_state(prepared,2)
    lengths=(1,3,7,2);index=0;spikes=[]
    while state.step<21:
        begin=state.step
        length=min(lengths[index%4],prepared.window-begin%prepared.window,21-begin)
        state,words=configured.advance(prepared,state,events,21,length)
        configured.evaluate_state(state)
        for bit in range(length):
            for row in np.flatnonzero(np.asarray(words)&(1<<bit)):
                spikes.append((stimulus.trial_indices[row//prepared.neurons],row%prepared.neurons,begin+bit))
        index+=1
    assert sorted(spikes,key=lambda row:(row[0],row[2],row[1]))==list(zip(
        wanted.spikes.trials,wanted.spikes.neurons,wanted.spikes.steps))
    assert_exact_state(configured.host_state(state,prepared),wanted.state)


def test_cpp_target_identity_selects_scalar_policy_without_changing_device(monkeypatch):
    import brian2 as b
    from brian2.devices.device import get_device
    from mlx_backend.compiler import compile_model
    previous=get_device()
    monkeypatch.setitem(b.prefs,"codegen.target","auto")
    b.set_device("cpp_standalone",build_on_run=False)
    try:
        model=model_configuration()
        assert model.reference_codegen_target=="cpp"
        assert get_device().__class__.__name__=="CPPStandaloneDevice"
        assert "_eon_hyper_tanh_scalar" in compile_model(model).reset_code
        assert get_device().__class__.__name__=="CPPStandaloneDevice"
    finally:
        b.set_device(previous)


@pytest.mark.metal
@pytest.mark.parametrize("target",["numpy","cython"])
def test_invariant_tanh_preserves_resolved_cpu_target(monkeypatch,tmp_path,target):
    mx = require_metal('Metal required')
    import brian2 as b
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    monkeypatch.setitem(b.prefs,"codegen.target",target)
    from component_cpu_reference import HyperbolicComponentCpu as ComponentCpu,component_inputs
    model=resolve_model({"eqs":DEFAULT_EQUATIONS+"a:1\n","eq_th":"True","t_rfc":0*b.ms,
        "eq_rst":"v=v_rst;g=0*mV;a=tanh(argument)",
        "argument":float.fromhex("0x1.fffffffffffffp-2")})
    cache=Path(__file__).resolve().parents[1]/"executions/mlx-feature-parity/hyperbolic-cython"/tmp_path.name
    cpu=ComponentCpu(1,1,model,target,cache)
    _,_,wanted=cpu.run()
    graph,stimulus=component_inputs(1,1,model)
    prepared=configured.prepare(graph,stimulus,model)
    _,state=configured.run_prepared(prepared,stimulus,return_state=True,record=False)
    actual=configured.host_state(state,prepared)
    assert_exact_state(actual,wanted)


@pytest.mark.metal
@pytest.mark.parametrize("target",["numpy","cython"])
@pytest.mark.parametrize("operation",[">",">="])
def test_target_specific_tanh_threshold_ties(monkeypatch,tmp_path,target,operation):
    mx = require_metal('Metal required')
    import brian2 as b
    from configured_reference import reference
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    monkeypatch.setitem(b.prefs,"codegen.target",target)
    monkeypatch.setitem(b.prefs,"codegen.runtime.cython.cache_dir",str(tmp_path/"cython"))
    from component_cpu_reference import component_inputs
    argument=float.fromhex("0x1.fffffffffffffp-2")
    raw=np.array([argument]).view(np.uint64)
    cutoff=float(expected("tanh_numpy" if target=="numpy" else "tanh_scalar",raw).view(np.float64)[0])
    model=resolve_model({"eqs":DEFAULT_EQUATIONS+"a:1\n","t_rfc":0*b.ms,
        "eq_th":f"(t<dt) or (tanh(a){operation}cutoff)",
        "eq_rst":"v=v_rst;g=0*mV;a=argument","argument":argument,"cutoff":cutoff})
    graph,stimulus=component_inputs(1,7,model)
    wanted=reference(graph,stimulus,model)
    prepared=configured.prepare(graph,stimulus,model)
    actual,state=configured.run_prepared(prepared,stimulus,return_state=True,chunk_steps=3)
    assert actual.spikes.steps.size==(1 if operation==">" else 7)
    for field in ("trials","neurons","steps"):
        np.testing.assert_array_equal(getattr(actual.spikes,field),getattr(wanted.spikes,field))
    assert_exact_state(configured.host_state(state,prepared),wanted.state)


@pytest.mark.metal
def test_actual_cpp_standalone_functions_match_native(monkeypatch,tmp_path):
    mx = require_metal('Metal required')
    import brian2 as b
    from brian2.devices.device import get_device
    from mlx_backend import configured
    monkeypatch.setitem(b.prefs,"codegen.target","auto")
    from component_cpu_reference import component_inputs
    previous=get_device()
    b.set_device("cpp_standalone",build_on_run=False)
    b.device.reinit()
    b.device.activate(build_on_run=False)
    try:
        model=model_configuration()
        assert model.reference_codegen_target=="cpp"
        graph,stimulus=component_inputs(3,5,model)
        prepared=configured.prepare(graph,stimulus,model)
        clock=b.Clock(dt=model.dt_s*b.second)
        neurons=b.NeuronGroup(3,model.eqs,method="linear",threshold=model.eq_th,
            reset=model.eq_rst,refractory="rfc",namespace=model.brian_namespace(),
            clock=clock,dtype=np.float64)
        neurons.v=model.v_0*b.volt;neurons.g=0*b.volt;neurons.rfc=0*b.second
        monitor=b.SpikeMonitor(neurons)
        network=b.Network(neurons,monitor);network.schedule=list(model.schedule)
        network.run(5*model.dt_s*b.second,namespace={})
        b.device.build(directory=str(tmp_path/"cpp"),compile=True,run=True,clean=True,
                       debug=False,with_output=False)
        actual,state=configured.run_prepared(prepared,stimulus,return_state=True)
        np.testing.assert_array_equal(actual.spikes.neurons,np.asarray(monitor.i[:]))
        np.testing.assert_array_equal(actual.spikes.steps,np.rint(np.asarray(monitor.t_[:])/model.dt_s).astype(np.int64))
        for name,values in configured.host_state(state,prepared).items():
            expected=np.asarray(getattr(neurons,name+"_")[:]).copy().reshape(1,-1)
            assert_array_bits_equal(values,expected,name)
    finally:
        b.set_device(previous)

