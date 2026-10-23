"""Actual system asinhf versus native raw binary32 results."""
import ctypes
import json
import os
from pathlib import Path

import numpy as np
import pytest
from native_execution import require_metal

from mlx_backend import basic_math32, binary32, binary64, math64
from mlx_backend.complex_asinh_math32 import metal_complex_asinh_header


def _cases():
    boundaries=[0,1,0x007fffff,0x00800000,0x32000000,
                0x3e800000,0x40800000,0x7f7fffff,0x7f800000,
                0x7fc12345,0x7f812345]
    words=[(value+delta)&0x7fffffff for value in boundaries
           for delta in (-1,0,1) if value+delta>=0]
    # Every log-table transition and adjacent words in the large-input branch.
    words += [0x40800000+(index<<16)+delta for index in range(128)
              for delta in (-0x8001,-0x8000,-0x7fff,0)
              if 0x40800000+(index<<16)+delta>=0x40800000]
    words=np.array(words, np.uint32)
    dense=np.linspace(.125,4.125,257,dtype=np.float32).view(np.uint32)
    random=np.random.default_rng(20261022).integers(0,2**32,1024,dtype=np.uint32)
    return np.concatenate((words,words|np.uint32(0x80000000),dense,
                           dense|np.uint32(0x80000000),random))


@pytest.mark.metal
def test_complex_real_asinh32_matches_actual_system_bits():
    mx = require_metal('Actual Metal is required')
    raw=_cases()
    library=ctypes.CDLL(None)
    operation=library.asinhf
    operation.argtypes=[ctypes.c_float]
    operation.restype=ctypes.c_float
    # Pass raw words through ctypes unions so signalingNaN does not first widen
    # through a Python float; the actual CPU call establishes quieting/payload.
    class Word(ctypes.Union):
        _fields_=[('raw',ctypes.c_uint32),('value',ctypes.c_float)]
    # A c_float view passes the original storage directly through libffi.
    # asinhf quiets signalingNaNs before its return widens to a Python float.
    expected=np.array([operation(ctypes.c_float.from_buffer(Word(raw=int(v))))
                       for v in raw],np.float32).view(np.uint32)
    header=(binary64.MSL_HEADER+binary32.MSL_HEADER+math64.MSL_MATH_HEADER
            +basic_math32.MSL_HEADER+metal_complex_asinh_header())
    kernel=mx.fast.metal_kernel(name='complex_real_asinh32',input_names=['raw'],
        output_names=['out'],header=header,
        source=f'uint i=thread_position_in_grid.x;if(i<{len(raw)}u)out[i]=_eon_complex_real_asinh32(raw[i]);',
        ensure_row_contiguous=True)
    out=kernel(inputs=[mx.array(raw)],grid=(len(raw),1,1),threadgroup=(64,1,1),
        output_shapes=[raw.shape],output_dtypes=[mx.uint32])[0]
    mx.eval(out)
    actual=np.asarray(out)
    differences=np.flatnonzero(actual!=expected)
    result=dict(rows=len(raw),mismatches=len(differences),first_mismatches=[
        dict(input=hex(int(raw[i])),actual=hex(int(actual[i])),expected=hex(int(expected[i])))
        for i in differences[:20]])
    destination=os.environ.get('EON_COMPLEX_ASINH_PROBE')
    if destination:Path(destination).write_text(json.dumps(result,indent=2)+'\n')
    assert not len(differences),result
