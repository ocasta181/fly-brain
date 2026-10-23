"""Actual default-compiler unsigned floating casts against raw native words."""
import ctypes
import subprocess

import brian2 as b
import numpy as np
import pytest
from native_execution import require_metal

from mlx_backend import binary32,binary64,operator32
from mlx_backend.operator_policy import compiler_command


def unsigned_cast_cases():
    values={0,1,2**24-1,2**24,2**24+1,2**63-1,2**63,2**64-1}
    for leading in range(24,64):
        shift=leading-23
        for mantissa in (2**23,2**23+1,2**23+2,2**24-2,2**24-1):
            midpoint=(mantissa<<shift)+(1<<(shift-1))
            values.update(midpoint+delta for delta in (-2,-1,0,1,2)
                          if 0<=midpoint+delta<2**64)
    raw=np.array(sorted(values),np.uint64)
    random=np.random.default_rng(491009).integers(0,2**64,8192,dtype=np.uint64)
    return np.r_[raw,random]


@pytest.mark.metal
def test_unsigned_floating_casts_match_default_compiler_and_metal(tmp_path):
    mx = require_metal('Metal required')
    raw=unsigned_cast_cases()
    source=tmp_path/'casts.cpp';library=tmp_path/'casts.so'
    source.write_text('''#include <stdint.h>
#include <string.h>
extern "C" void reference(const uint64_t* in,uint32_t* f32,uint64_t* f64,int n) {
    for(int i=0;i<n;++i) {
        float a=(float)in[i];double b=(double)in[i];
        memcpy(f32+i,&a,sizeof(a));memcpy(f64+i,&b,sizeof(b));
    }
}
''')
    flags=b.prefs.codegen.cpp.extra_compile_args
    if flags is None:flags=b.prefs.codegen.cpp.extra_compile_args_gcc
    flags=list(flags)
    command=[*compiler_command(),*flags,'-shared','-fPIC',str(source),'-o',str(library)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    reference=ctypes.CDLL(str(library)).reference
    pointer64=np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS')
    pointer32=np.ctypeslib.ndpointer(dtype=np.uint32,flags='C_CONTIGUOUS')
    reference.argtypes=[pointer64,pointer32,pointer64,ctypes.c_int]
    expected32=np.empty(raw.size,np.uint32);expected64=np.empty(raw.size,np.uint64)
    reference(raw,expected32,expected64,raw.size)
    # A rounded binary64 intermediate is demonstrably insufficient here.
    assert np.any(expected32!=raw.astype(np.float64).astype(np.float32).view(np.uint32))
    kernel=mx.fast.metal_kernel(name='operator_unsigned_float_cast',input_names=['raw'],
        output_names=['f32','f64'],header=binary64.MSL_HEADER+binary32.MSL_HEADER+operator32.MSL_UINT_CAST_HEADER,
        source='uint i=thread_position_in_grid.x;if(i<COUNT){f32[i]=f_operator_from_uint(raw[i]);f64[i]=d_from_uint(raw[i]);}',
        compile_options={'math_mode':'safe'})
    actual32,actual64=kernel(inputs=[mx.array(raw)],template=[('COUNT',raw.size)],
        grid=(raw.size,1,1),threadgroup=(128,1,1),output_shapes=[raw.shape,raw.shape],
        output_dtypes=[mx.uint32,mx.uint64])
    mx.eval(actual32,actual64)
    np.testing.assert_array_equal(np.asarray(actual32),expected32)
    np.testing.assert_array_equal(np.asarray(actual64),expected64)
