"""The installed complex cast loop's scalar tail and invalid status."""
import warnings

import numpy as np
import pytest
from numpy_callback import RestoreCallback


@pytest.mark.metal
def test_complex_int32_cast_keeps_final_scalar_block():
    mx=pytest.importorskip('mlx.core')
    from mlx_backend import binary64,binary32,binary16,math64
    from mlx_backend.complex_math64 import MSL_BASE_HEADER
    from mlx_backend.numpy_error_policy import metal_header
    from mlx_backend.numpy_narrow_status import metal_header as narrow_header
    records=[];groups=[]
    with RestoreCallback(np.geterrcall()):
        for width in (64,32):
            for count in (8,9,16):
                with np.errstate(all='ignore'):
                    values=np.resize(np.array([2.**40,-2.**40,np.nan,np.inf,-np.inf,
                        2.**31,-2.**31-2.,3.75]),count).astype(
                        np.complex128 if width==64 else np.complex64)
                calls=[]
                np.seterrcall(lambda text,flags:calls.append(flags))
                with warnings.catch_warnings(),np.errstate(all='call'):
                    warnings.simplefilter('ignore',np.ComplexWarning)
                    expected=np.int32(values)
                start=len(records)
                for pos,(value,result) in enumerate(zip(values,expected)):
                    records.append((np.float64(value.real).view(np.uint64),pos,count,width,int(result)))
                groups.append((start,count,0 if not calls else calls[-1]))
    kernel=mx.fast.metal_kernel(name='complex_integer_final_scalar_block',
        input_names=['raw','position','count','width'],output_names=['out','flags'],
        header=(binary64.MSL_HEADER+binary32.MSL_HEADER+binary16.MSL_HEADER+
            binary16.MSL_F32_HEADER+math64.MSL_MATH_HEADER+MSL_BASE_HEADER+
            metal_header()+narrow_header(())),
        source='''uint i=thread_position_in_grid.x;
        out[i]=_eon_complex_int32(raw[i],position[i],count[i],width[i]);
        flags[i]=_eon_numpy_int32_cast_status(raw[i],position[i],count[i]-1L,width[i]);
        ''',compile_options={'math_mode':'safe'})
    output,flags=kernel(inputs=[mx.array(np.array([r[0] for r in records],np.uint64)),
        mx.array([r[1] for r in records],mx.int64),mx.array([r[2] for r in records],mx.int64),
        mx.array([r[3] for r in records],mx.uint32)],grid=(len(records),1,1),
        threadgroup=(32,1,1),output_shapes=[(len(records),),(len(records),)],
        output_dtypes=[mx.int32,mx.uint32])
    np.testing.assert_array_equal(np.asarray(output),[r[4] for r in records])
    flags=np.asarray(flags)
    for start,count,expected in groups:
        assert int(np.bitwise_or.reduce(flags[start:start+count]))==expected
