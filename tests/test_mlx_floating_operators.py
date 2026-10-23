"""Independent CPU operators versus integer-backed native floating helpers."""
import ast
import ctypes
from pathlib import Path
import subprocess

import numpy as np
import pytest
from native_execution import require_metal

from mlx_backend import binary32, binary64, math64, operator32, operator64


def power_cases(dtype=np.float64):
    info=np.finfo(dtype)
    centers = np.array([0., -0., 1., -1., 2., -2., .5, -.5, 3., -3.,
        np.inf, -np.inf, np.nan, info.tiny, info.max, np.nextafter(dtype(0.), dtype(1.)),
        np.nextafter(dtype(1.), dtype(0.)), np.nextafter(dtype(1.), dtype(2.))], dtype=dtype)
    rawtype=np.uint64 if dtype==np.float64 else np.uint32
    nan_raw=([0x7ff8123456789abc,0xfff8123456789abc,0x7ff0000000000001,0xfff0000000012345]
             if dtype==np.float64 else [0x7fc12345,0xffc12345,0x7f800001,0xff812345])
    centers=np.r_[centers,np.array(nan_raw,rawtype).view(dtype)]
    exponent = np.array([0., -0., 1., -1., 2., -2., .5, -.5, 1/64, -1/3,
        1.3, 2.1, np.inf, -np.inf, np.nan, 2**-65, 2**64,
        1023., 1024., -1074., -1075., 2**24-1,2**24,2**53-1, 2**53], dtype=dtype)
    a,b = np.meshgrid(centers, exponent)
    rng = np.random.default_rng(461093)
    extent=700 if dtype==np.float64 else 85
    random_base = np.exp(rng.uniform(-extent, extent, 2048)).astype(dtype)
    random_exponent = rng.uniform(-4, 4, 2048).astype(dtype)
    a = np.r_[a.ravel(), random_base]
    b = np.r_[b.ravel(), random_exponent]
    return a,b


def assert_same_bits(actual, expected, a, b):
    dtype = np.uint64 if expected.dtype == np.float64 else np.uint32
    mismatch = np.flatnonzero(actual != expected.view(dtype))
    if len(mismatch):
        i=int(mismatch[0])
        pytest.fail(f"{len(mismatch)}/{len(a)} differ; first {float(a[i]).hex()} ** "
                    f"{float(b[i]).hex()}: Metal/software {int(actual[i]):#x}, CPU {int(expected.view(dtype)[i]):#x}")


@pytest.mark.metal
@pytest.mark.parametrize('dtype',[np.float64,np.float32])
def test_exp2_lowering_matches_actual_platform_function(dtype):
    mx=pytest.importorskip('mlx.core')
    wide=dtype==np.float64;rawtype=np.uint64 if wide else np.uint32
    lower,upper=(-1075,1024) if wide else (-150,128)
    exponent=np.array([0.,-0.,1.,-1.,upper-1,upper,lower+1,lower,
        np.inf,-np.inf,np.nan,.1,.3,.7],dtype)
    # Table cells, scale transitions, rounding neighbors and raw exponents
    # independently probe the constant-base lowering across the exponent range.
    boundary=(np.arange(lower*128,upper*128+1,127,dtype=np.float64)/128).astype(dtype)
    exponent=np.r_[exponent,boundary,np.nextafter(boundary,dtype(np.inf)),np.nextafter(boundary,dtype(-np.inf))]
    rng=np.random.default_rng(613119)
    raw=rng.integers(0,np.iinfo(rawtype).max,4096,dtype=rawtype)
    nan=([0x7ff8123456789abc,0xfff8123456789abc,0x7ff0000000000001,0xfff0000000012345]
         if wide else [0x7fc12345,0xffc12345,0x7f800001,0xff812345])
    raw=np.r_[exponent.view(rawtype),raw,np.array(nan,rawtype)]
    function=getattr(ctypes.CDLL(None),'exp2' if wide else 'exp2f')
    function.argtypes=[ctypes.c_double if wide else ctypes.c_float]
    function.restype=ctypes.c_double if wide else ctypes.c_float
    expected=np.array([function(value) for value in raw.view(dtype)],dtype)
    prefix='d' if wide else 'f'
    kernel=mx.fast.metal_kernel(name='operator_exp2_platform_reference_'+prefix,input_names=['raw'],output_names=['out'],
        header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+binary32.MSL_HEADER+
            (operator64.MSL_EXP2_HEADER+operator32.MSL_EXP2_HEADER if wide else
             operator32.MSL_EXP2_HEADER+operator64.MSL_EXP2_HEADER),
        source=f'uint i=thread_position_in_grid.x;if(i<COUNT)out[i]={prefix}_operator_exp2(raw[i]);',
        compile_options={'math_mode':'safe'})
    result,=kernel(inputs=[mx.array(raw)],template=[('COUNT',len(raw))],grid=(len(raw),1,1),
        threadgroup=(128,1,1),output_shapes=[expected.shape],output_dtypes=[mx.uint64 if wide else mx.uint32])
    mx.eval(result)
    assert_same_bits(np.asarray(result),expected,np.full(len(raw),2.),raw.view(dtype))


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    module = ast.parse((root / "tests/test_mlx_math64.py").read_text())
    prefix = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "HOST_PREFIX" for t in n.targets))
    directory = tmp_path_factory.mktemp("floating-operator-host")
    source, library = directory / "operators.cpp", directory / "operators.so"
    # Metal provides max for the binary32 helper; host C++ needs its adapter.
    source.write_text(prefix+"\n#include <algorithm>\nusing std::max;\n#define constant static const\n"+binary64.MSL_HEADER+
        math64.MSL_MATH_HEADER+binary32.MSL_HEADER+operator64.MSL_POWER_HEADER+operator32.MSL_POWER_HEADER+
        operator64.MSL_REMAINDER_HEADER+operator32.MSL_REMAINDER_HEADER+'''
extern "C" void power(const ulong *a,const ulong *b,ulong *y,int count) {
    for(int i=0;i<count;++i)y[i]=d_operator_pow(a[i],b[i]);
}
extern "C" void power32(const uint *a,const uint *b,uint *y,int count) {
    for(int i=0;i<count;++i)y[i]=f_operator_pow(a[i],b[i]);
}
extern "C" void remainder64(const ulong *a,const ulong *b,ulong *y,int count) {
    for(int i=0;i<count;++i) {
        y[3*i]=d_operator_fmod(a[i],b[i]);
        y[3*i+1]=d_operator_mod_numpy(a[i],b[i]);
        y[3*i+2]=d_operator_floordiv_numpy(a[i],b[i]);
    }
}
extern "C" void remainder32(const uint *a,const uint *b,uint *y,int count) {
    for(int i=0;i<count;++i) {
        y[3*i]=f_operator_fmod(a[i],b[i]);
        y[3*i+1]=f_operator_mod_numpy(a[i],b[i]);
        y[3*i+2]=f_operator_floordiv_numpy(a[i],b[i]);
    }
}
extern "C" void polynomial_fma(const ulong *raw,ulong *y,int count) {
    for(int i=0;i<count;++i)y[i]=_eon_operator_polynomial_fma(raw[3*i],raw[3*i+1],raw[3*i+2]);
}
extern "C" void reference_fma(const ulong *raw,ulong *y,int count) {
    for(int i=0;i<count;++i)y[i]=as_type<ulong>(std::fma(as_type<double>(raw[3*i]),
        as_type<double>(raw[3*i+1]),as_type<double>(raw[3*i+2])));
}
extern "C" void operator_fma32(const uint *raw,uint *y,int count) {
    for(int i=0;i<count;++i)y[i]=f_operator_fma(raw[3*i],raw[3*i+1],raw[3*i+2]);
}
extern "C" void reference_fma32(const uint *raw,uint *y,int count) {
    for(int i=0;i<count;++i)y[i]=as_type<uint>(std::fma(as_type<float>(raw[3*i]),
        as_type<float>(raw[3*i+1]),as_type<float>(raw[3*i+2])));
}
extern "C" void binary_nan64(const ulong *a,const ulong *b,ulong *y,int count) {
    for(int i=0;i<count;++i){y[4*i]=d_add(a[i],b[i]);y[4*i+1]=d_sub(a[i],b[i]);
        y[4*i+2]=d_mul(a[i],b[i]);y[4*i+3]=d_div(a[i],b[i]);}
}
extern "C" void binary_nan32(const uint *a,const uint *b,uint *y,int count) {
    for(int i=0;i<count;++i){y[4*i]=f_add(a[i],b[i]);y[4*i+1]=f_sub(a[i],b[i]);
        y[4*i+2]=f_mul(a[i],b[i]);y[4*i+3]=f_div(a[i],b[i]);}
}
''')
    compilation=subprocess.run(["clang++", "-std=c++17", "-shared", "-fPIC", "-O2", "-fno-fast-math",
                    "-ffp-contract=off", str(source), "-o", str(library)],
                   capture_output=True, text=True, timeout=30)
    assert compilation.returncode==0,compilation.stderr
    result = ctypes.CDLL(str(library))
    pointer = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
    result.power.argtypes = [pointer,pointer,pointer,ctypes.c_int]
    result.power.restype = None
    pointer32 = np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS")
    result.power32.argtypes = [pointer32,pointer32,pointer32,ctypes.c_int]
    result.power32.restype = None
    for name,pointer_type in (("remainder64",pointer),("remainder32",pointer32)):
        getattr(result,name).argtypes=[pointer_type,pointer_type,pointer_type,ctypes.c_int]
        getattr(result,name).restype=None
    for name in ("polynomial_fma","reference_fma"):
        getattr(result,name).argtypes=[pointer,pointer,ctypes.c_int]
        getattr(result,name).restype=None
    for name in ('operator_fma32','reference_fma32'):
        getattr(result,name).argtypes=[pointer32,pointer32,ctypes.c_int]
        getattr(result,name).restype=None
    for name,pointer_type in (('binary_nan64',pointer),('binary_nan32',pointer32)):
        getattr(result,name).argtypes=[pointer_type,pointer_type,pointer_type,ctypes.c_int]
        getattr(result,name).restype=None
    return result


def polynomial_fma_cases():
    from test_mlx_math64 import fma_cases
    original=fma_cases()
    rng=np.random.default_rng(992015)
    rows=[]
    for gap in (3,4,5,9,53,64,128,1024):
        for _ in range(512):
            ea,eb=(int(x) for x in rng.integers(1,2047,2))
            ec=ea+eb-1023+gap
            if not 1<=ec<=2046: continue
            mant=rng.integers(0,1<<52,3,dtype=np.uint64)
            signs=rng.integers(0,2,3,dtype=np.uint64)
            exponents=np.array([ea,eb,ec],np.uint64)
            rows.append((signs<<np.uint64(63))|(exponents<<np.uint64(52))|mant)
    return np.ascontiguousarray(np.concatenate((original,np.array(rows,np.uint64))))


@pytest.mark.metal
def test_polynomial_fma_matches_independent_cpu_fma(host):
    mx=pytest.importorskip("mlx.core")
    raw=polynomial_fma_cases()
    expected=np.empty(len(raw),np.uint64)
    observed=np.empty_like(expected)
    host.reference_fma(raw.reshape(-1),expected,len(raw))
    host.polynomial_fma(raw.reshape(-1),observed,len(raw))
    valid=~np.isnan(expected.view(np.float64))
    np.testing.assert_array_equal(observed[valid],expected[valid])
    kernel=mx.fast.metal_kernel(name="operator_polynomial_fma_reference",input_names=["raw"],
        output_names=["out"],header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+operator64._OPERATOR_FMA_HEADER,
        source="uint i=thread_position_in_grid.x;if(i<COUNT)out[i]=_eon_operator_polynomial_fma(raw[3*i],raw[3*i+1],raw[3*i+2]);",
        compile_options={"math_mode":"safe"})
    result,=kernel(inputs=[mx.array(raw.reshape(-1))],template=[("COUNT",len(raw))],
        grid=(len(raw),1,1),threadgroup=(128,1,1),output_shapes=[expected.shape],output_dtypes=[mx.uint64])
    mx.eval(result)
    np.testing.assert_array_equal(np.asarray(result)[valid],expected[valid])


@pytest.mark.metal
def test_float32_operator_fma_rounds_directly_once(host):
    mx=pytest.importorskip('mlx.core')
    rng=np.random.default_rng(191513)
    random=rng.integers(0,1<<32,(8192,3),dtype=np.uint32)
    boundary=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f800000,
        0xbf800000,0x7f7fffff,0x7f800000,0xff800000,0x7fc00123,0xff800456],np.uint32)
    grid=np.array(np.meshgrid(boundary,boundary,boundary)).reshape(3,-1).T.copy()
    raw=np.ascontiguousarray(np.concatenate((random,grid)).reshape(-1))
    count=len(raw)//3;expected=np.empty(count,np.uint32);observed=np.empty_like(expected)
    host.reference_fma32(raw,expected,count);host.operator_fma32(raw,observed,count)
    np.testing.assert_array_equal(observed,expected)
    kernel=mx.fast.metal_kernel(name='float32_operator_fma_once',input_names=['raw'],output_names=['out'],
        header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+binary32.MSL_HEADER+
            operator64.MSL_REMAINDER_HEADER+operator32.MSL_REMAINDER_HEADER,
        source='uint i=thread_position_in_grid.x;if(i<COUNT)out[i]=f_operator_fma(raw[3*i],raw[3*i+1],raw[3*i+2]);',
        compile_options={'math_mode':'safe'})
    result,=kernel(inputs=[mx.array(raw)],template=[('COUNT',count)],grid=(count,1,1),threadgroup=(128,1,1),
        output_shapes=[expected.shape],output_dtypes=[mx.uint32])
    mx.eval(result);np.testing.assert_array_equal(np.asarray(result),expected)


@pytest.mark.metal
def test_exact_53bit_product_matches_independent_integer_reference():
    mx=pytest.importorskip('mlx.core');rng=np.random.default_rng(24915)
    raw=rng.integers(1<<52,1<<53,(8192,2),dtype=np.uint64)
    raw[:4]=[[1<<52,1<<52],[(1<<53)-1,(1<<53)-1],[(1<<52)+1,(1<<53)-1],[(1<<53)-1,(1<<52)+1]]
    products=[int(a)*int(c) for a,c in raw]
    expected=np.array([[p&((1<<64)-1),p>>64] for p in products],np.uint64)
    kernel=mx.fast.metal_kernel(name='exact_operator_mul53',input_names=['raw'],output_names=['out'],
        header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+operator64._OPERATOR_FMA_HEADER,
        source='uint i=thread_position_in_grid.x;if(i<COUNT){ulong high;out[2*i]=_eon_operator_mul53(raw[2*i],raw[2*i+1],&high);out[2*i+1]=high;}',
        compile_options={'math_mode':'safe'})
    result,=kernel(inputs=[mx.array(raw.reshape(-1))],template=[('COUNT',len(raw))],grid=(len(raw),1,1),
        threadgroup=(128,1,1),output_shapes=[expected.shape],output_dtypes=[mx.uint64])
    mx.eval(result);np.testing.assert_array_equal(np.asarray(result),expected)


@pytest.mark.metal
@pytest.mark.parametrize('dtype',[np.float64,np.float32])
def test_binary_arithmetic_nan_priority_matches_numpy(host,dtype):
    mx=pytest.importorskip('mlx.core');wide=dtype==np.float64
    rawtype=np.uint64 if wide else np.uint32
    raw=np.array([0x7ff8000000000123,0xfff0000000000456,0x7ff0000000000789,0xfff8000000000321]
        if wide else [0x7fc00123,0xff800456,0x7f800789,0xffc00321],rawtype)
    a,c=np.meshgrid(raw,raw);a=np.ascontiguousarray(a.ravel());c=np.ascontiguousarray(c.ravel())
    with np.errstate(all='ignore'):
        expected=np.stack([operation(a.view(dtype),c.view(dtype)) for operation in
            (np.add,np.subtract,np.multiply,np.divide)],axis=1).view(rawtype)
    observed=np.empty_like(expected)
    getattr(host,'binary_nan64' if wide else 'binary_nan32')(a,c,observed,len(a))
    np.testing.assert_array_equal(observed,expected)
    prefix='d' if wide else 'f'
    source='uint i=thread_position_in_grid.x;if(i<COUNT){'+''.join(
        f'out[4*i+{number}]={prefix}_{name}(a[i],b[i]);' for number,name in enumerate(('add','sub','mul','div')))+'}'
    kernel=mx.fast.metal_kernel(name='binary_nan_priority_'+str(np.dtype(dtype)),input_names=['a','b'],output_names=['out'],
        header=binary64.MSL_HEADER+binary32.MSL_HEADER,source=source,compile_options={'math_mode':'safe'})
    result,=kernel(inputs=[mx.array(a),mx.array(c)],template=[('COUNT',len(a))],grid=(len(a),1,1),threadgroup=(32,1,1),
        output_shapes=[expected.shape],output_dtypes=[mx.uint64 if wide else mx.uint32])
    mx.eval(result);np.testing.assert_array_equal(np.asarray(result),expected)






def _check_host_power(host,dtype):
    a,b = power_cases(dtype)
    rawtype=np.uint64 if dtype==np.float64 else np.uint32
    observed = np.empty(len(a),rawtype)
    getattr(host,'power' if dtype==np.float64 else 'power32')(
        a.view(rawtype),b.view(rawtype),observed,len(a))
    with np.errstate(all="ignore"):
        expected=np.power(a,b)
    assert_same_bits(observed,expected,a,b)


def test_binary64_power_matches_independent_numpy(host):
    _check_host_power(host,np.float64)


def test_binary32_power_matches_independent_numpy(host):
    _check_host_power(host,np.float32)


def _check_metal_power(dtype):
    mx = require_metal('Metal required')
    wide=dtype==np.float64;rawtype=np.uint64 if wide else np.uint32
    a,b=power_cases(dtype)
    with np.errstate(all="ignore"):
        expected=np.power(a,b)
    operator=operator64 if wide else operator32
    header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+('' if wide else binary32.MSL_HEADER)
    kernel=mx.fast.metal_kernel(name=f"operator{64 if wide else 32}_power_reference",input_names=["a","b"],
        output_names=["out"],header=header+operator.MSL_POWER_HEADER,
        source=f"uint i=thread_position_in_grid.x;if(i<COUNT)out[i]={'d' if wide else 'f'}_operator_pow(a[i],b[i]);",
        compile_options={"math_mode":"safe"})
    observed,=kernel(inputs=[mx.array(a.view(rawtype)),mx.array(b.view(rawtype))],
        template=[("COUNT",len(a))],grid=(len(a),1,1),threadgroup=(128,1,1),
        output_shapes=[a.shape],output_dtypes=[mx.uint64 if wide else mx.uint32])
    mx.eval(observed)
    assert_same_bits(np.asarray(observed),expected,a,b)


@pytest.mark.metal
def test_actual_metal_binary64_power_matches_independent_numpy():
    _check_metal_power(np.float64)


@pytest.mark.metal
def test_actual_metal_binary32_power_matches_independent_numpy():
    _check_metal_power(np.float32)


def remainder_cases(dtype):
    bits=np.uint64 if dtype==np.float64 else np.uint32
    info=np.finfo(dtype)
    values=np.array([0.,-0.,1.,-1.,3.,-3.,.3,-.3,np.inf,-np.inf,
        info.max,-info.max,info.tiny,-info.tiny,np.nextafter(dtype(0),dtype(1)),
        np.nextafter(dtype(1),dtype(0)),np.nextafter(dtype(1),dtype(2))],dtype)
    nan_raw=([0x7ff8123456789abc,0xfff8123456789abc,0x7ff0000000000001,0xfff0000000012345]
             if dtype==np.float64 else [0x7fc12345,0xffc12345,0x7f800001,0xff812345])
    values=np.r_[values,np.array(nan_raw,bits).view(dtype)]
    a,b=np.meshgrid(values,values)
    rng=np.random.default_rng(781515)
    raw=rng.integers(0,np.iinfo(bits).max,(2,2048),dtype=bits).view(dtype)
    return np.ascontiguousarray(np.r_[a.ravel(),raw[0]]),np.ascontiguousarray(np.r_[b.ravel(),raw[1]])


def numpy_remainder_expected(a,b):
    with np.errstate(all="ignore"):
        return np.ascontiguousarray(np.stack((np.fmod(a,b),np.remainder(a,b),np.floor_divide(a,b)),axis=1))


def assert_remainder_bits(observed,expected,a,b):
    bits=np.uint64 if expected.dtype==np.float64 else np.uint32
    mismatch=np.argwhere(observed!=expected.view(bits))
    if len(mismatch):
        i,j=(int(v) for v in mismatch[0])
        names=("fmod","remainder","floor_divide")
        pytest.fail(f"{len(mismatch)}/{observed.size} differ; {names[j]}({float(a[i]).hex()}, {float(b[i]).hex()}): "
                    f"native {int(observed[i,j]):#x}, CPU {int(expected.view(bits)[i,j]):#x}")


@pytest.mark.parametrize("dtype",[np.float64,np.float32])
def test_remainder_matches_independent_numpy(host,dtype):
    bits=np.uint64 if dtype==np.float64 else np.uint32
    a,b=remainder_cases(dtype)
    observed=np.empty((len(a),3),bits)
    getattr(host,"remainder64" if dtype==np.float64 else "remainder32")(
        a.view(bits),b.view(bits),observed.ravel(),len(a))
    assert_remainder_bits(observed,numpy_remainder_expected(a,b),a,b)


@pytest.mark.metal
@pytest.mark.parametrize("dtype",[np.float64,np.float32])
def test_actual_metal_remainder_matches_independent_numpy(dtype):
    mx=pytest.importorskip("mlx.core")
    bits=np.uint64 if dtype==np.float64 else np.uint32
    a,b=remainder_cases(dtype)
    expected=numpy_remainder_expected(a,b)
    prefix="d" if dtype==np.float64 else "f"
    width="64" if dtype==np.float64 else "32"
    header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+binary32.MSL_HEADER+operator64.MSL_REMAINDER_HEADER
    if dtype==np.float32: header+=operator32.MSL_REMAINDER_HEADER
    kernel=mx.fast.metal_kernel(name="operator_remainder_reference"+width,input_names=["a","b"],
        output_names=["out"],header=header,
        source="uint i=thread_position_in_grid.x;if(i<COUNT){"+
            f"out[3*i]={prefix}_operator_fmod(a[i],b[i]);"+
            f"out[3*i+1]={prefix}_operator_mod_numpy(a[i],b[i]);"+
            f"out[3*i+2]={prefix}_operator_floordiv_numpy(a[i],b[i]);}}",
        compile_options={"math_mode":"safe"})
    observed,=kernel(inputs=[mx.array(a.view(bits)),mx.array(b.view(bits))],template=[("COUNT",len(a))],
        grid=(len(a),1,1),threadgroup=(128,1,1),output_shapes=[expected.shape],
        output_dtypes=[mx.uint64 if dtype==np.float64 else mx.uint32])
    mx.eval(observed)
    assert_remainder_bits(np.asarray(observed),expected,a,b)


@pytest.mark.metal
def test_numpy_half_operators_use_scalar32_then_half_rounding():
    """Check half-ufunc operator values; the half engine owns explicit narrowing.

    Metal native half casts canonicalize NaNs and flush some small results, so
    comparison narrows the returned raw32 through the independent NumPy cast.
    This proves the scalar32-operator contract, not the separate half converter.
    """
    mx=pytest.importorskip('mlx.core')
    half=np.arange(1<<16,dtype=np.uint16).view(np.float16)
    exponents=np.array([.3,-.3,0.,.5,1.,2.,np.inf,-np.inf],np.float16)
    a=np.tile(half,len(exponents));b=np.repeat(exponents,len(half))
    with np.errstate(all='ignore'):
        expected=np.stack([operation(a,b) for operation in
            (np.power,np.remainder,np.floor_divide)],axis=1).view(np.uint16)
        wide_a=a.astype(np.float32).view(np.uint32)
        wide_b=b.astype(np.float32).view(np.uint32)
    header=(binary64.MSL_HEADER+math64.MSL_MATH_HEADER+binary32.MSL_HEADER+
        operator64.MSL_REMAINDER_HEADER+operator32.MSL_POWER_HEADER+operator32.MSL_REMAINDER_HEADER)
    kernel=mx.fast.metal_kernel(name='half_operator_adapter_contract',input_names=['a','b'],output_names=['out'],
        header=header,source='''uint i=thread_position_in_grid.x;if(i<COUNT){
          uint p=f_operator_pow(a[i],b[i]);
          uint r=f_operator_mod_numpy(a[i],b[i]);
          uint q=f_operator_floordiv_numpy(a[i],b[i]);
          out[3*i]=p;out[3*i+1]=r;out[3*i+2]=q;
        }''',compile_options={'math_mode':'safe'})
    observed,=kernel(inputs=[mx.array(wide_a),mx.array(wide_b)],template=[('COUNT',len(a))],
        grid=(len(a),1,1),threadgroup=(128,1,1),output_shapes=[expected.shape],output_dtypes=[mx.uint32])
    mx.eval(observed)
    with np.errstate(all='ignore'):
        actual=np.asarray(observed).view(np.float32).astype(np.float16).view(np.uint16)
    mismatch=np.argwhere(actual!=expected)
    if len(mismatch):
        i,j=(int(value) for value in mismatch[0])
        pytest.fail(f'{len(mismatch)}/{expected.size} raw half outputs differ; '
            f'{("power","remainder","floor_divide")[j]}({int(a.view(np.uint16)[i]):#x},'
            f'{int(b.view(np.uint16)[i]):#x}): native={int(actual[i,j]):#x}, CPU={int(expected[i,j]):#x}')


@pytest.mark.metal
def test_operator_safe_division_matches_independent_numpy():
    mx=pytest.importorskip('mlx.core')
    a,b=remainder_cases(np.float64)
    rng=np.random.default_rng(192417)
    random=rng.integers(0,1<<64,(2,16384),dtype=np.uint64)
    a=np.ascontiguousarray(np.r_[a,random[0].view(np.float64)])
    b=np.ascontiguousarray(np.r_[b,random[1].view(np.float64)])
    with np.errstate(all='ignore'):expected=np.divide(a,b).view(np.uint64)
    kernel=mx.fast.metal_kernel(name='operator_safe_division_reference',input_names=['a','b'],output_names=['out'],
        header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+operator64.MSL_REMAINDER_HEADER,
        source='uint i=thread_position_in_grid.x;if(i<COUNT)out[i]=d_operator_div(a[i],b[i]);',
        compile_options={'math_mode':'safe'})
    result,=kernel(inputs=[mx.array(a.view(np.uint64)),mx.array(b.view(np.uint64))],template=[('COUNT',len(a))],
        grid=(len(a),1,1),threadgroup=(128,1,1),output_shapes=[expected.shape],output_dtypes=[mx.uint64])
    mx.eval(result);actual=np.asarray(result)
    mismatch=np.flatnonzero(actual!=expected)
    if len(mismatch):
        i=int(mismatch[0])
        pytest.fail(f'{len(mismatch)}/{len(a)} division bits differ; inputs={int(a.view(np.uint64)[i]):#x},'
            f'{int(b.view(np.uint64)[i]):#x}, native={int(actual[i]):#x}, CPU={int(expected[i]):#x}')


@pytest.mark.metal
def test_operator_fma_guard_composes_with_guarded_stable_dependency():
    """Exercise both include orders with the completed guarded dependency.

    Older checkouts do not guard stable FMA yet. Bind that guarded dependency
    into the authoritative header construction so the test covers the merged
    source layout without importing another worktree or mutating modules.
    """
    from mlx_backend.stable_math64 import _STABLE_FMA_HEADER
    mx=pytest.importorskip('mlx.core')
    stable=_STABLE_FMA_HEADER
    if 'EON_STABLE_FMA_INCLUDED' not in stable:
        stable='#ifndef EON_STABLE_FMA_INCLUDED\n#define EON_STABLE_FMA_INCLUDED\n'+stable+'\n#endif\n'
    module=ast.parse(Path(operator64.__file__).read_text())
    expression=next(node.value for node in module.body if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='_OPERATOR_FMA_HEADER' for target in node.targets))
    owned=eval(compile(ast.Expression(expression),'<operator-fma-guard-fixture>','eval'),
        {'__builtins__':{}},{'_STABLE_FMA_HEADER':stable})
    power64=operator64.MSL_POWER_HEADER.replace(operator64._OPERATOR_FMA_HEADER,owned,1)
    power32=operator32.MSL_POWER_HEADER.replace(operator32._OPERATOR_FMA_HEADER,owned,1)
    a=np.array([.3,1.3,2.,10.],np.float64);b=np.array([.1,.7,-.3,1/64],np.float64)
    a32=a.astype(np.float32);b32=b.astype(np.float32)
    expected=np.stack([np.multiply(a,b).view(np.uint64),np.multiply(a,b).view(np.uint64),
        np.power(a,b).view(np.uint64),np.power(a32,b32).view(np.uint32).astype(np.uint64)])
    for number,dependencies in enumerate((stable+power64+power32,power64+power32+stable)):
        kernel=mx.fast.metal_kernel(name='operator_guard_composition_'+str(number),input_names=['a','b','fa','fb'],
            output_names=['out'],header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+binary32.MSL_HEADER+dependencies,
            source='''uint i=thread_position_in_grid.x;if(i<N){
              out[i]=stable_fma(a[i],b[i],0ul);
              out[N+i]=_eon_operator_fma(a[i],b[i],0ul);
              out[2*N+i]=d_operator_pow(a[i],b[i]);
              out[3*N+i]=ulong(f_operator_pow(fa[i],fb[i]));
            }''',compile_options={'math_mode':'safe'})
        observed,=kernel(inputs=[mx.array(a.view(np.uint64)),mx.array(b.view(np.uint64)),
            mx.array(a32.view(np.uint32)),mx.array(b32.view(np.uint32))],template=[('N',len(a))],
            grid=(len(a),1,1),threadgroup=(128,1,1),output_shapes=[expected.shape],output_dtypes=[mx.uint64])
        mx.eval(observed);np.testing.assert_array_equal(np.asarray(observed),expected)


@pytest.mark.metal
def test_operator_div_guard_composes_with_guarded_stable_dependency():
    from mlx_backend.stable_math64 import _STABLE_DIV_HEADER
    mx=pytest.importorskip('mlx.core')
    stable=_STABLE_DIV_HEADER
    if 'EON_STABLE_DIV_INCLUDED' not in stable:
        stable='#ifndef EON_STABLE_DIV_INCLUDED\n#define EON_STABLE_DIV_INCLUDED\n'+stable+'\n#endif\n'
    module=ast.parse(Path(operator64.__file__).read_text())
    expression=next(node.value for node in module.body if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='_OPERATOR_DIV_HEADER' for target in node.targets))
    owned=eval(compile(ast.Expression(expression),'<operator-div-guard-fixture>','eval'),
        {'__builtins__':{}},{'_STABLE_DIV_HEADER':stable})
    remainder=operator64.MSL_REMAINDER_HEADER.replace(operator64._OPERATOR_DIV_HEADER,owned,1)
    a=np.array([.3,-1.3,2.,10.],np.float64);b=np.array([.1,.7,-.3,1/64],np.float64)
    expected=np.stack([np.divide(a,b).view(np.uint64),np.divide(a,b).view(np.uint64),
        np.remainder(a,b).view(np.uint64),np.floor_divide(a,b).view(np.uint64)])
    for number,dependencies in enumerate((stable+remainder,remainder+stable)):
        kernel=mx.fast.metal_kernel(name='operator_div_guard_composition_'+str(number),
            input_names=['a','b'],output_names=['out'],header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+dependencies,
            source='''uint i=thread_position_in_grid.x;if(i<N){
              out[i]=stable_div(a[i],b[i]);
              out[N+i]=_eon_operator_div(a[i],b[i]);
              out[2*N+i]=d_operator_mod_numpy(a[i],b[i]);
              out[3*N+i]=d_operator_floordiv_numpy(a[i],b[i]);
            }''',compile_options={'math_mode':'safe'})
        observed,=kernel(inputs=[mx.array(a.view(np.uint64)),mx.array(b.view(np.uint64))],template=[('N',len(a))],
            grid=(len(a),1,1),threadgroup=(128,1,1),output_shapes=[expected.shape],output_dtypes=[mx.uint64])
        mx.eval(observed);np.testing.assert_array_equal(np.asarray(observed),expected)
