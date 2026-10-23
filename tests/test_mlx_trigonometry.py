"""Focused original Brian API and native forward-trigonometry qualification."""
from state_assertions import assert_state_bytes_equal, assert_recorded_spikes_equal
from trigonometry_checks import OriginalComparison
from native_execution import forbid_call, require_metal
from pathlib import Path

import numpy as np
import pytest

b=pytest.importorskip("brian2")
from mlx_backend.compiler import compile_model
from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
from trigonometry_reference import sine_inputs


@pytest.mark.parametrize("function",["sin","cos","tan"])
def test_forward_codegen_policy_identity_and_no_cpu_preparation(function,monkeypatch):
    parameters={"eqs":DEFAULT_EQUATIONS+"sample : 1\n", "eq_th":"t>=0*ms",
                "eq_rst":f"v=v_rst; sample={function}(v/mV)"}
    monkeypatch.setitem(b.prefs,"codegen.target","numpy")
    numpy=resolve_model(parameters)
    monkeypatch.setitem(b.prefs,"codegen.target","cython")
    compiled=resolve_model(parameters)
    assert numpy.reference_codegen_target=="numpy"
    assert compiled.reference_codegen_target=="cython"
    assert numpy.signature!=compiled.signature
    assert compiled.reference_compile_args
    forbid_call(b.Network,"run","Compiler ran a Brian timestep",monkeypatch)
    program=compile_model(numpy)
    from brian2.core.functions import DEFAULT_FUNCTIONS
    # The preceding v=v_rst makes this original RHS a scalar constant.
    value = DEFAULT_FUNCTIONS[function].pyfunc(np.float64(1000.) * np.float64(numpy.v_rst))
    expected = int(np.asarray(value, np.float64).view(np.uint64))
    assert f"_eon_numpy__eon_var_73616d706c65 = {expected}ul;" in program.reset_code
    assert not program.header_code
    assert not compile_model(resolve_model()).header_code
    assert "d_cosine_numpy(" not in compile_model(compiled).reset_code
    # Read v before resetting it to retain the original dynamic-helper checks.
    dynamic = dict(parameters, eq_rst=f"sample={function}(v/mV);v=v_rst")
    dynamic_numpy = compile_model(resolve_model(dynamic, cpu_target="numpy"))
    assert {"sin":"d_sine(","cos":"d_cosine_numpy(","tan":"d_tangent("}[function] in dynamic_numpy.reset_code
    assert "constant ulong _eon_trig_range" in dynamic_numpy.header_code
    assert "d_cosine_numpy(" not in compile_model(resolve_model(dynamic, cpu_target="cython")).reset_code


@pytest.mark.metal
@pytest.mark.parametrize("target",["numpy","cython"])
@pytest.mark.parametrize("function",["sin","cos","tan"])
def test_original_api_forward_raw_state_spikes_continuation_without_cpu_steps(tmp_path,target,function):
    from brian2.devices.device import get_device
    mx = require_metal('Metal required')
    previous_device = get_device()
    # Includes large arguments, subnormals, signed zero and NaN payloads. Every
    # neuron spikes on every step; sample stores the function's exact result.
    inputs=sine_inputs()
    comparison=OriginalComparison(target, function, inputs, tmp_path, run_steps=(1, 2, 0, 3))
    assert get_device() is previous_device
    with comparison.native() as (actual,native,device):
        assert device.native_steps==6
        assert device.native_runs==4
    assert get_device() is previous_device
    if target == "cython":
        assert comparison.cpu["compiled_module"] is not None
        assert native["compiled_module"] is None


@pytest.mark.metal
@pytest.mark.parametrize("target", ["numpy", "cython"])
def test_forward_stable_math_composition_with_synapses(monkeypatch, target):
    """Composed helpers retain exact state, arrivals and recording semantics."""
    from configured_reference import reference
    from mlx_backend import configured
    from test_mlx_configured import graph, inputs

    monkeypatch.setitem(b.prefs, "codegen.target", target)
    model = resolve_model({
        "eqs": DEFAULT_EQUATIONS+"x:1\na:1\nb:1\nc:1\nd:1\nfifth:1\nsixth:1\nchecksum:integer\n",
        "eq_th": "True", "t_rfc": 0*b.ms,
        "eq_rst": "v=v_rst;x+=1/32;a=sin(x);b=cos(x);c=tan(x);d=expm1(x);"
                  "fifth=tanh(x);sixth=arctan(x);"
                  "checksum+=int(a*1024)+int(b*1024)+int(c*1024)+int(d*1024)+int(fifth*1024)+int(sixth*1024)",
    })
    connectome = graph()
    stimulus = inputs(model, steps=35, trials=2)
    expected = reference(connectome, stimulus, model, cpu_target=target)
    prepared = configured.prepare(connectome, stimulus, model)
    for record in (True, False):
        result, state = configured.run_prepared(prepared, stimulus, record=record,
                                               return_state=True, chunk_steps=19)
        assert_recorded_spikes_equal(result, expected, record)
        actual = configured.host_state(state, prepared)
        assert_state_bytes_equal({name: actual[name] for name in expected.state}, expected.state)


@pytest.mark.metal
@pytest.mark.parametrize("target",["numpy","cython"])
@pytest.mark.parametrize("function",["sin","cos","tan"])
def test_original_float32_forward_target_state_and_spike_context(tmp_path,target,function):
    mx = require_metal('Metal required')
    # A sine comparison need not use the same scalar call width as a casted
    # reset assignment. The actual original target is the independent oracle.
    centers=np.array([np.pi/6,np.pi/3,np.pi/2,-np.pi/6,0.,-0.,120.,2**26],np.float32)
    inputs=np.r_[centers,np.nextafter(centers,np.float32(-np.inf)),
                 np.nextafter(centers,np.float32(np.inf))]
    options=dict(state_inputs=True,dtype="float32",run_steps=(1,2,0,3),
                 threshold_expression=function+"(phase)>.5")
    comparison=OriginalComparison(target, function, inputs, tmp_path, **options)
    with comparison.native() as (actual,native,device):
        assert device.native_steps==6


@pytest.mark.metal
@pytest.mark.parametrize("function",["sin","cos","tan"])
@pytest.mark.parametrize("optimise",[True,False])
def test_original_numpy_boolean_trig_argument(tmp_path,function,optimise):
    mx = require_metal('Metal required')
    symbol="phase_optimised" if optimise else "phase_unoptimised"
    options=dict(state_inputs=True,phase_dtype="boolean",optimise=optimise,phase_symbol=symbol,
                 result_expression=function+"("+symbol+")+.1")
    inputs=np.array([False,True,True,False])
    comparison=OriginalComparison('numpy', function, inputs, tmp_path, **options)
    source=(tmp_path/"cpu/reset-source.txt").read_text()
    assert ("_numpy.where" in source)==optimise
    comparison.check_native()


@pytest.mark.metal
@pytest.mark.parametrize("dtype,expression",[
    ("float64","sin(sin(phase_nested))+.1"),
    ("float32","cos(phase_nested)+t/ms"),
    ("float64","tan(phase_nested)*.1"),
    ("float64","sin(phase_nested)-.1"),
    ("float64","sin(phase_nested)/3."),
    ("float64","clip(sin(phase_nested),.1,.7)"),
    ("float64","int(sin(phase_nested))+.1"),
    ("float64","floor(sin(phase_nested))+.1"),
    ("float64","ceil(sin(phase_nested))+.1"),
    ("float32","sqrt(sin(phase_nested))+.1"),
    ("float32","(cos(phase_nested)+t/second*1e10)-t/second*1e10"),
])
def test_original_numpy_half_trig_dataflow_continuation(tmp_path,dtype,expression):
    mx = require_metal('Metal required')
    options=dict(state_inputs=True,phase_dtype="boolean",phase_symbol="phase_nested",dtype=dtype,
                 optimise=False,result_expression=expression,run_steps=(1,2,0,3))
    inputs=np.array([False,True,True,False])
    comparison=OriginalComparison('numpy', 'sin', inputs, tmp_path, **options)
    with comparison.native() as (actual,native,device):
        assert device.native_steps==6 and device.native_runs==4


@pytest.mark.metal
@pytest.mark.parametrize("function,boundary",[("sin",.8412),("cos",.5405),("tan",1.5575)])
def test_original_numpy_half_trig_threshold_rounding(tmp_path,function,boundary):
    mx = require_metal('Metal required')
    symbol="phase_bound_"+function
    options=dict(state_inputs=True,phase_dtype="boolean",phase_symbol=symbol,optimise=False,
        result_expression=function+"("+symbol+")+.1",run_steps=(1,2,0,3),
        threshold_expression=function+"("+symbol+")>"+repr(boundary))
    inputs=np.array([False,True,True,False])
    comparison=OriginalComparison('numpy', function, inputs, tmp_path, **options)
    comparison.check_native()


@pytest.mark.metal
@pytest.mark.parametrize("function",["sin","cos","tan"])
def test_original_numpy_boolean_argument_keeps_floating_scalar_width(tmp_path,function):
    mx = require_metal('Metal required')
    symbol="phase_argument_"+function
    options=dict(state_inputs=True,phase_dtype="boolean",phase_symbol=symbol,dtype="float32",
                 optimise=False,argument_expression=symbol+"+.1",run_steps=(1,2,0,3))
    inputs=np.array([False,True,True,False])
    comparison=OriginalComparison('numpy', function, inputs, tmp_path, **options)
    comparison.check_native()


@pytest.mark.metal
@pytest.mark.parametrize("function",["sin","cos","tan"])
def test_original_numpy_int16_trig_ufunc_dtype(tmp_path,function):
    from trigonometry_checks import check_original_small_integer_loop
    check_original_small_integer_loop(tmp_path,"int16",function)


@pytest.mark.metal
@pytest.mark.parametrize("function",["cos","tan"])
def test_original_cpp_float32_forward_result_width(tmp_path,function):
    """The accepted standalone float argument selects the actual float overload."""
    mx = require_metal('Metal required')
    centers=np.array([0.,-0.,np.pi/4,-np.pi/2,120.,2**26],np.float32)
    inputs=np.r_[centers,np.nextafter(centers,np.float32(-np.inf)),
        np.nextafter(centers,np.float32(np.inf)),
        np.array([0x7f800001,0xffc00123,0x7f800000,0xff800000],np.uint32).view(np.float32)]
    options=dict(state_inputs=True,dtype="float32")
    comparison=OriginalComparison('cpp', function, inputs, tmp_path, **options)
    comparison.check_native(target='cython')
