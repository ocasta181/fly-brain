"""Pinned NumPy source-precision cast rules, including transient half values."""
import numpy as np
import pytest
from mlx_backend.binary64 import MSL_HEADER


@pytest.mark.metal
@pytest.mark.parametrize('bits',[16,32,64])
@pytest.mark.parametrize('length',[1,12,32])
def test_numpy_int32_cast_source_width_and_vector_context(bits,length):
    mx=pytest.importorskip('mlx.core')
    original=np.array([np.inf,np.nan,-np.inf,-3.75,3.75,-2.**31,2.**31,
                       2.**32+1,-2.**63,2.**63,2.**64,2.**63-1024])
    with np.errstate(all='ignore'):
        values=np.resize(original,length).astype(f'float{bits}')
        expected=np.int32(values)
        encoded=values.astype(np.float64).view(np.uint64)
    kernel=mx.fast.metal_kernel(name='integer_source_precision_cast',input_names=['values'],output_names=['result'],
        source='uint i=thread_position_in_grid.x; if(i<SIZE) result[i]=n_to_int32(values[i],i,SIZE,SOURCE_BITS);',
        header=MSL_HEADER,compile_options={'math_mode':'safe'})
    result,=kernel(inputs=[mx.array(encoded)],template=[('SIZE',length),('SOURCE_BITS',bits)],
        grid=(length,1,1),threadgroup=(min(256,length),1,1),output_shapes=[(length,)],output_dtypes=[mx.int32])
    mx.eval(result);actual=np.asarray(result)
    np.testing.assert_array_equal(actual,expected)
