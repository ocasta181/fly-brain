"""Actual optimized CPU may combine sibling sine/cosine calls into sincos."""
from trigonometry_checks import OriginalComparison
import ctypes
import subprocess
import numpy as np
import pytest
from native_execution import require_metal
from trigonometry_reference import sine_inputs

pytestmark = pytest.mark.metal


@pytest.mark.parametrize("dtype",["float64","float32"])
def test_original_compiled_sibling_sine_cosine_calls(tmp_path,dtype):
    mx = require_metal('Metal required')
    inputs=sine_inputs()
    if dtype=="float32":
        centers=np.array([0.,-0.,np.pi/6,np.pi/4,np.pi/2,120.,2**26],np.float32)
        inputs=np.r_[centers,np.nextafter(centers,np.float32(-np.inf)),
            np.nextafter(centers,np.float32(np.inf)),
            np.array([0x7f800001,0xffc00123,0x7f800000,0xff800000],np.uint32).view(np.float32)]
    options=dict(state_inputs=True,dtype=dtype,phase_symbol="phase_sincos",
        cache_suffix="-paired-sincos",result_expression="sin(phase_sincos)+cos(phase_sincos)")
    comparison=OriginalComparison('cython', 'sin', inputs, tmp_path, **options)
    audit=subprocess.run(["nm","-u",comparison.cpu["compiled_module"]],check=True,capture_output=True,text=True,timeout=5)
    (tmp_path/"cpu/imports.txt").write_text(audit.stdout)
    comparison.check_native()


def test_paired_scalar32_full_range_matches_actual_cpu_abi(tmp_path):
    """Independent live CPU pair ABI, not the standalone sinf/cosf oracle."""
    mx = require_metal('Metal required')
    from mlx_backend import binary64,binary32,math64
    from mlx_backend.trigonometry32 import metal_trigonometry32_header
    class Pair(ctypes.Structure):
        _fields_=[("sine",ctypes.c_float),("cosine",ctypes.c_float)]
    function=getattr(ctypes.CDLL(None),"__sincosf_stret")
    function.argtypes=[ctypes.c_float];function.restype=Pair
    rng=np.random.default_rng(20261029)
    raw=rng.integers(0,1<<32,32768,dtype=np.uint32)
    centers=np.array([0,1,0x7fffff,0x800000,0x39800000,0x3f490fdb,
        0x42f00000,0x4c800000,0x7f7fffff,0x7f800000,0x7f800001,
        0x7fc00123],np.uint32)
    boundary=np.r_[centers-1,centers,centers+1]
    quadrants=(np.arange(-1000,1001,dtype=np.float64)*np.pi/2).astype(np.float32)
    values=np.r_[raw,boundary,boundary^np.uint32(1<<31),quadrants.view(np.uint32),
        np.nextafter(quadrants,np.float32(-np.inf)).view(np.uint32),
        np.nextafter(quadrants,np.float32(np.inf)).view(np.uint32)]
    expected=np.array([(result.sine,result.cosine)for result in
        (function(float(x))for x in values.view(np.float32))],np.float32).view(np.uint32)
    header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+binary32.MSL_HEADER+metal_trigonometry32_header({"sin","cos"})
    kernel=mx.fast.metal_kernel(name="trig32_paired_actual_abi",input_names=["raw"],output_names=["out"],
        source="uint i=thread_position_in_grid.x; auto pair=_eon_trig32_sincos_scalar(raw[i]); out[2*i]=pair.sine; out[2*i+1]=pair.cosine;",
        header=header,compile_options={"math_mode":"safe"})
    result,=kernel(inputs=[mx.array(values)],grid=(values.size,1,1),threadgroup=(128,1,1),
        output_shapes=[(values.size,2)],output_dtypes=[mx.uint32]);mx.eval(result)
    np.testing.assert_array_equal(np.asarray(result),expected)


@pytest.mark.parametrize("case",["canonical-argument","different-argument","short-circuit","cpp-pair"])
def test_original_compiled_pairing_context(tmp_path,case):
    mx = require_metal('Metal required')
    centers=np.array([-2**26,-120.,-1.,-0.,0.,1.,120.,2**26],np.float32)
    values=np.r_[centers,np.nextafter(centers,np.float32(-np.inf)),
                 np.nextafter(centers,np.float32(np.inf))]
    symbol="phase_"+case.replace("-","_")
    expression={"canonical-argument":f"sin({symbol})+cos(-{symbol})",
        "different-argument":f"sin({symbol})+cos({symbol}+0.125)",
        "short-circuit":f"sin({symbol})",
        "cpp-pair":f"sin({symbol})+cos({symbol})"}[case]
    options=dict(state_inputs=True,dtype="float32",phase_symbol=symbol,
        cache_suffix="-paired-context-"+case,result_expression=expression,run_steps=(1,2,0,1))
    if case=="short-circuit":
        options["threshold_expression"]=f"sin({symbol})>0 and cos({symbol})>0"
    target="cpp" if case=="cpp-pair" else "cython"
    if target=="cpp":options["run_steps"]=(1,)
    comparison=OriginalComparison(target, 'sin', values, tmp_path, **options)
    # Both compiled targets declare these physical state operands float;
    # this case has no namespace Constant/codegen-language conversion.
    comparison.check_native(target='cython')


def test_original_builtin_pairing_preserves_flag_identity(tmp_path,monkeypatch):
    """Explicit C++ builtins still pair under no-builtin library-name flags."""
    import brian2 as b
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    mx = require_metal('Metal required')
    monkeypatch.setitem(b.prefs,"core.default_float_dtype",np.float32)
    monkeypatch.setitem(b.prefs,"codegen.target","cython")
    params=dict(eqs=DEFAULT_EQUATIONS+"sample : 1\nphase_separate : 1\n",
        eq_th="t>=0*ms",eq_rst="v=v_rst; sample=sin(phase_separate)+cos(phase_separate)")
    default=resolve_model(params)
    flags=b.prefs.codegen.cpp.extra_compile_args
    if flags is None:flags=b.prefs.codegen.cpp.extra_compile_args_gcc
    monkeypatch.setitem(b.prefs,"codegen.cpp.extra_compile_args",
        [*flags,"-fno-builtin-sinf","-fno-builtin-cosf"])
    separate=resolve_model(params)
    assert default.signature!=separate.signature
    assert default.float32_compile_args!=separate.float32_compile_args
    values=np.array([2**26,-2**26,120.,-120.,1.,-1.],np.float32)
    options=dict(state_inputs=True,dtype="float32",phase_symbol="phase_separate",
        cache_suffix="-paired-separate-flags",result_expression="sin(phase_separate)+cos(phase_separate)")
    comparison=OriginalComparison('cython', 'sin', values, tmp_path, **options)
    imports=subprocess.run(["nm","-u",comparison.cpu["compiled_module"]],check=True,capture_output=True,text=True,timeout=5).stdout
    assert "___sincosf_stret" in imports
    comparison.check_native()
