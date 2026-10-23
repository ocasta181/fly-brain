"""Independent raw NaN priority for the shared arithmetic primitives."""
import numpy as np
import pytest
from mlx_backend import binary64,binary32


@pytest.mark.metal
@pytest.mark.parametrize('dtype',[np.float64,np.float32])
def test_binary_arithmetic_nan_priority_matches_numpy(dtype):
    mx=pytest.importorskip('mlx.core');wide=dtype==np.float64
    rawtype=np.uint64 if wide else np.uint32
    raw=np.array([0x7ff8000000000123,0xfff0000000000456,0x7ff0000000000789,0xfff8000000000321]
        if wide else [0x7fc00123,0xff800456,0x7f800789,0xffc00321],rawtype)
    a,c=np.meshgrid(raw,raw);a=np.ascontiguousarray(a.ravel());c=np.ascontiguousarray(c.ravel())
    with np.errstate(all='ignore'):
        expected=np.stack([operation(a.view(dtype),c.view(dtype)) for operation in
            (np.add,np.subtract,np.multiply,np.divide)],axis=1).view(rawtype)
    prefix='d' if wide else 'f'
    source='uint i=thread_position_in_grid.x;if(i<COUNT){'+''.join(
        f'out[4*i+{number}]={prefix}_{name}(a[i],b[i]);' for number,name in enumerate(('add','sub','mul','div')))+'}'
    kernel=mx.fast.metal_kernel(name='binary_nan_priority_'+str(np.dtype(dtype)),input_names=['a','b'],output_names=['out'],
        header=binary64.MSL_HEADER+binary32.MSL_HEADER,source=source,compile_options={'math_mode':'safe'})
    result,=kernel(inputs=[mx.array(a),mx.array(c)],template=[('COUNT',len(a))],grid=(len(a),1,1),threadgroup=(32,1,1),
        output_shapes=[expected.shape],output_dtypes=[mx.uint64 if wide else mx.uint32])
    mx.eval(result);np.testing.assert_array_equal(np.asarray(result),expected)
