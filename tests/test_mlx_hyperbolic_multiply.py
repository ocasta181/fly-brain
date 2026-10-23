"""Known-width multiplication normalization against independent hardware."""
import ctypes
from pathlib import Path
import subprocess

import numpy as np
import pytest
from native_execution import require_metal

from mlx_backend import binary64,hyperbolic_math64
from test_mlx_math64 import HOST_PREFIX


def multiply_inputs():
    random=np.random.default_rng(2026101802).integers(0,np.iinfo(np.uint64).max,(4096,2),np.uint64)
    mantissas=(0,1,2,3,(1<<26)-1,1<<26,(1<<27)-1,1<<51,(1<<52)-2,(1<<52)-1)
    exponents=(1,2,1022,1023,1024,2045,2046)
    regular=np.array([[(ea<<52)|ma,(eb<<52)|mb] for ea in exponents for eb in exponents
        for ma in mantissas for mb in mantissas],np.uint64)
    regular=np.r_[regular,regular^np.array([1<<63,0],np.uint64),
        regular^np.array([0,1<<63],np.uint64)]
    special=(0,1,0x8000000000000000,0x8000000000000001,
        0x7ff0000000000000,0xfff0000000000000,0x7ff0000000000001,
        0xfff8000012345678,0x3ff0000000000000,0x0010000000000000)
    exceptional=np.array([(a,b) for a in special for b in special],np.uint64)
    return np.r_[random,regular,exceptional]


@pytest.fixture(scope='module')
def multiply_host(tmp_path_factory):
    directory=tmp_path_factory.mktemp('hyperbolic-multiply')
    source,library=directory/'multiply.cpp',directory/'multiply.so'
    source.write_text(HOST_PREFIX+binary64.MSL_HEADER+hyperbolic_math64._NORMAL_MUL_HEADER+r'''
extern "C" {
void products(const ulong*a,const ulong*b,ulong*y,ulong*baseline,ulong*hardware,int count) {
    for(int i=0;i<count;++i) {
        y[i]=hyper_mul(a[i],b[i]);baseline[i]=d_mul(a[i],b[i]);
        hardware[i]=as_type<ulong>(as_type<double>(a[i])*as_type<double>(b[i]));
    }
}
}
''')
    subprocess.run(['clang++','-std=c++17','-shared','-fPIC','-O2','-fno-fast-math',
        '-ffp-contract=off',str(source),'-o',str(library)],check=True,capture_output=True,text=True,timeout=30)
    result=ctypes.CDLL(str(library))
    pointer=np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS')
    result.products.argtypes=[pointer]*5+[ctypes.c_int]
    result.products.restype=None
    return result


def independent_products(host):
    raw=multiply_inputs()
    a,b=(np.ascontiguousarray(raw[:,i]) for i in range(2))
    actual,baseline,hardware=(np.empty(len(raw),np.uint64) for _ in range(3))
    host.products(a,b,actual,baseline,hardware,len(raw))
    np.testing.assert_array_equal(actual,baseline)
    finite=np.all(((raw>>np.uint64(52))&np.uint64(2047))!=2047,axis=1)
    np.testing.assert_array_equal(actual[finite],hardware[finite])
    return a,b,baseline


def test_known_width_multiply_matches_independent_host(multiply_host):
    independent_products(multiply_host)


@pytest.mark.metal
def test_known_width_multiply_matches_actual_metal(multiply_host):
    mx = require_metal('Metal required')
    a,b,reference=independent_products(multiply_host)
    kernel=mx.fast.metal_kernel(name='hyper_known_width_multiply',input_names=['a','b'],output_names=['out'],
        source='uint i=thread_position_in_grid.x;if(i<N)out[i]=hyper_mul(a[i],b[i]);',
        header=binary64.MSL_HEADER+hyperbolic_math64._NORMAL_MUL_HEADER,compile_options={'math_mode':'safe'})
    (actual,)=kernel(inputs=[mx.array(a),mx.array(b)],template=[('N',len(a))],grid=(len(a),1,1),
        threadgroup=(128,1,1),output_shapes=[reference.shape],output_dtypes=[mx.uint64])
    mx.eval(actual)
    np.testing.assert_array_equal(np.asarray(actual),reference)
