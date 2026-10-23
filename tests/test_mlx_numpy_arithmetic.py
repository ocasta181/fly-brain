"""Independent NumPy arithmetic versus raw-word carrier lowering on Metal."""
import ast

import numpy as np
import pytest

from mlx_backend import binary32,binary64,math64
from mlx_backend.numpy_arithmetic import render_numpy_arithmetic


# A test-only representation fixture. Finite values use the existing exact
# conversions; NaN fields are moved without a numeric conversion. Production
# uses these shared h_* names from its existing representation registry.
_CARRIER_FIXTURE=r'''
inline ulong h_f32_to_d(uint x) {
    if((x&0x7fffffffu)>0x7f800000u)
        return (ulong(x&0x80000000u)<<32)|0x7ff0000000000000ul|(ulong(x&0x7fffffu)<<29);
    return f_to_d(x);
}
inline uint h_d_to_f(ulong x) {
    if(d_is_nan(x))return uint((x>>32)&0x80000000ul)|0x7f800000u|uint(d_mant(x)>>29);
    return f_from_d(x);
}
'''


def _values(dtype):
    words=([0x7fc00123,0xffc00234,0x7f800345,0xff800456] if dtype==np.float32 else
           [0x7ff8000000000123,0xfff8000000000234,0x7ff0000000000345,0xfff0000000000456])
    raw=np.uint32 if dtype==np.float32 else np.uint64
    return np.r_[np.array(words,raw).view(dtype),np.array([0.,-0.,1.,-1.,.3,-.3,np.inf,-np.inf,
        np.finfo(dtype).tiny,np.nextafter(dtype(0),dtype(1)),np.finfo(dtype).max],dtype)]


def _carrier(array):
    if array.dtype==np.float64:return array.view(np.uint64)
    with np.errstate(all='ignore'):out=array.astype(np.float64).view(np.uint64)
    words=array.view(np.uint32);mask=(words&0x7fffffff)>0x7f800000
    out[mask]=((words[mask].astype(np.uint64)&0x80000000)<<32)|0x7ff0000000000000|((words[mask].astype(np.uint64)&0x7fffff)<<29)
    return out


@pytest.mark.metal
@pytest.mark.parametrize('left_dtype,right_dtype',[
    (np.float32,np.float32),(np.float32,np.float64),
    (np.float64,np.float32),(np.float64,np.float64)])
def test_numpy_arithmetic_preserves_operand_representation(left_dtype,right_dtype):
    mx=pytest.importorskip('mlx.core')
    left,right=_values(left_dtype),_values(right_dtype)
    operations=[ast.Add(),ast.Sub(),ast.Mult(),ast.Div()]
    functions=[np.add,np.subtract,np.multiply,np.divide]
    expected=[];narrow=[]
    with np.errstate(all='ignore'):
        for function in functions:
            rows=[]
            for value in right:
                argument=np.full(left.shape,value,right_dtype)
                result=function(left,argument)
                rows.append(_carrier(result))
                if len(expected)==0:narrow.append(result.dtype==np.float32)
            expected.append(rows)
    expected=np.array(expected,np.uint64)
    # All NaNs produced by arithmetic are quiet; the carrier equals an exact
    # float32->float64 cast of the result, while input carriers remain raw.
    bodies=[]
    for number,operation in enumerate(operations):
        code,_=render_numpy_arithmetic(operation,'a[row]','b[column]',
            'narrow[column]!=0u',left_narrow=left_dtype==np.float32,right_narrow=right_dtype==np.float32,
            right_scalar=False)
        bodies.append(f'if(operation=={number}u)out[gid]={code};')
    kernel=mx.fast.metal_kernel(name='numpy_arithmetic_carrier_'+str(int(left_dtype==np.float32))+
        str(int(right_dtype==np.float32)),input_names=['a','b','narrow'],output_names=['out'],
        header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+binary32.MSL_HEADER+_CARRIER_FIXTURE,
        source='uint gid=thread_position_in_grid.x;if(gid<COUNT){uint row=gid%ROWS,column=(gid/ROWS)%COLS,operation=gid/(ROWS*COLS);'+''.join(bodies)+'}',
        compile_options={'math_mode':'safe'})
    observed,=kernel(inputs=[mx.array(_carrier(left)),mx.array(_carrier(right)),mx.array(narrow,mx.uint32)],
        template=[('COUNT',expected.size),('ROWS',len(left)),('COLS',len(right))],
        grid=(expected.size,1,1),threadgroup=(128,1,1),output_shapes=[expected.shape],output_dtypes=[mx.uint64])
    mx.eval(observed);np.testing.assert_array_equal(np.asarray(observed),expected)
