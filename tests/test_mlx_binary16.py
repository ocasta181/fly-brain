"""Independent NumPy cast checks for half temporaries, including all encodings."""
import numpy as np
import pytest
from native_execution import require_metal


@pytest.mark.metal
def test_raw_half_conversions_match_numpy_on_metal():
    mx = require_metal('Metal required')
    from mlx_backend.binary64 import MSL_HEADER
    from mlx_backend.binary16 import MSL_HEADER as half_header
    raw=np.arange(65536,dtype=np.uint32)
    with np.errstate(all="ignore"):
        decoded=raw.astype(np.uint16).view(np.float16).astype(np.float64).view(np.uint64)
    decode=mx.fast.metal_kernel(name="half_decode",input_names=["values"],output_names=["out"],
        header=MSL_HEADER+half_header,source="uint i=thread_position_in_grid.x;if(i<N)out[i]=h_to_d(values[i]);",
        compile_options={"math_mode":"safe"})
    (actual,)=decode(inputs=[mx.array(raw)],template=[("N",len(raw))],grid=(len(raw),1,1),
        threadgroup=(128,1,1),output_shapes=[raw.shape],output_dtypes=[mx.uint64])
    mx.eval(actual);np.testing.assert_array_equal(np.asarray(actual),decoded)
    positive=np.arange(0x7bff,dtype=np.uint16).view(np.float16).astype(np.float64)
    upper=np.arange(1,0x7c00,dtype=np.uint16).view(np.float16).astype(np.float64)
    centers=(positive+upper)*.5
    cases=np.r_[centers,np.nextafter(centers,-np.inf),np.nextafter(centers,np.inf),
                -centers,-np.nextafter(centers,-np.inf),-np.nextafter(centers,np.inf),
                decoded.view(np.float64),65520.,np.nextafter(65520.,-np.inf),
                -65520.,np.nextafter(-65520.,np.inf),
                np.random.default_rng(20261018).integers(0,2**64,8192,dtype=np.uint64).view(np.float64)]
    with np.errstate(all="ignore"):expected=cases.astype(np.float16).view(np.uint16).astype(np.uint32)
    encode=mx.fast.metal_kernel(name="half_encode",input_names=["values"],output_names=["out"],
        header=MSL_HEADER+half_header,source="uint i=thread_position_in_grid.x;if(i<N)out[i]=h_from_d(values[i]);",
        compile_options={"math_mode":"safe"})
    (actual,)=encode(inputs=[mx.array(cases.view(np.uint64))],template=[("N",len(cases))],grid=(len(cases),1,1),
        threadgroup=(128,1,1),output_shapes=[cases.shape],output_dtypes=[mx.uint32])
    mx.eval(actual);np.testing.assert_array_equal(np.asarray(actual),expected)
