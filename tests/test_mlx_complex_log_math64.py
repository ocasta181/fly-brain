"""Raw-bit NumPy and actual Metal coverage for complex logarithms."""
import ctypes
import json
import os
from pathlib import Path

import numpy as np
import pytest
from native_execution import require_metal

from mlx_backend import binary64, binary32, math64, basic_math32
from mlx_backend.complex_log_math64 import metal_complex_log_header
from mlx_backend.stable_math64 import metal_stable_math_header
from mlx_backend.transcendentals import metal_math_header


def _cases(single):
    words = np.array([
        0, 0x8000000000000000, 1, 0x8000000000000001,
        0x0010000000000000, 0x3fefffffffffffff, 0x3ff0000000000000,
        0x3ff0000000000001, 0xbff0000000000000, 0x3fd0000000000000,
        0x3fe0000000000000, 0x4000000000000000, 0x7fefffffffffffff,
        0x7ff0000000000000, 0xfff0000000000000,
        0x7ff8000012345678, 0xfff8000087654321,
        0x7ff0000012345678, 0xfff0000087654321,
    ], dtype=np.uint64)
    edge = np.array([(a, b) for a in words for b in words], dtype=np.uint64)
    random = np.random.default_rng(20261021).integers(0, 2**64, (256, 2), dtype=np.uint64)
    reduction = np.random.default_rng(731).uniform(-4., 4., (128, 2)).view(np.uint64)
    scaling = np.array([(exponent << 52, (exponent-gap) << 52)
                        for exponent in (511, 512, 1534, 1535, 1536)
                        for gap in range(1, 6)], np.uint64)
    scaling = np.concatenate((scaling, scaling[:, ::-1]))
    raw = np.concatenate((edge, random, reduction, scaling))
    if single:
        with np.errstate(all='ignore'):
            values = raw.view(np.float64).astype(np.float32).astype(np.float64)
        raw = values.view(np.uint64)
        float_edges = np.array([1, 0x80000001, 0x00800000, 0x3f7fffff,
                               0x3f800001, 0x7f7fffff, 0x7fc12345,
                               0xffc54321], np.uint32).view(np.float32)
        more = np.array([(a, b) for a in float_edges for b in float_edges], np.float32)
        random32 = np.random.default_rng(810).integers(0, 2**32, (256, 2), dtype=np.uint32)
        with np.errstate(all='ignore'):
            random32 = random32.view(np.float32).astype(np.float64).view(np.uint64)
        raw = np.concatenate((raw, more.astype(np.float64).view(np.uint64), random32))
    return raw


def _probe(single):
    mx = require_metal('Actual Metal is required')
    raw = _cases(single)
    dtype = np.complex64 if single else np.complex128
    values = raw.view(np.float64).astype(np.float32 if single else np.float64).view(dtype).reshape(-1)
    scalar_kind = np.float32 if single else np.float64
    expected = np.empty((len(raw), 8), np.float64)
    lib = ctypes.CDLL(None)
    atan = getattr(lib, 'atan2f' if single else 'atan2')
    hypot = getattr(lib, 'hypotf' if single else 'hypot')
    ctype = ctypes.c_float if single else ctypes.c_double
    for function in (atan, hypot):
        function.argtypes = [ctype, ctype]
        function.restype = ctype
    with np.errstate(all='ignore'):
        for i, name in enumerate(('log', 'log1p', 'log10')):
            reference = getattr(np, name)(values)
            expected[:, 2*i] = reference.real
            expected[:, 2*i+1] = reference.imag
            # Scalar and contiguous-array ufuncs are separately observable.
            scalar = np.array([getattr(np, name)(v) for v in values], dtype=dtype)
            np.testing.assert_array_equal(scalar.view(np.uint32 if single else np.uint64),
                                          reference.view(np.uint32 if single else np.uint64))
        expected[:, 6] = [atan(float(z.imag), float(z.real)) for z in values]
        expected[:, 7] = [hypot(float(scalar_kind(z.real+scalar_kind(1))), float(z.imag))
                          for z in values]
    suffix = '32' if single else ''
    calls = ','.join('_eon_complex_'+name+suffix+'(z)' for name in ('log', 'log1p', 'log10'))
    source = f'''
    uint i=thread_position_in_grid.x;if(i>={len(raw)}u)return;
    _eon_complex z={{raw[2*i],raw[2*i+1]}};
    _eon_complex result[3]={{{calls}}};
    for(uint j=0;j<3;++j){{out[8*i+2*j]=result[j].real;out[8*i+2*j+1]=result[j].imag;}}
    out[8*i+6]=_eon_complex_atan2{suffix}(z.imag,z.real);
    out[8*i+7]=_eon_complex_log_hypot{suffix}(
        {'f_to_d(f_add(f_from_d(z.real),0x3f800000u))' if single else 'd_add(z.real,0x3ff0000000000000ul)'},z.imag);
    '''
    header = (binary64.MSL_HEADER+binary32.MSL_HEADER+math64.MSL_MATH_HEADER
              +metal_math_header({'log'})+metal_stable_math_header({'log1p'})
              +basic_math32.MSL_HEADER+'\nstruct _eon_complex {ulong real,imag;};\n'
              +metal_complex_log_header())
    kernel = mx.fast.metal_kernel(name='complex_log_probe_'+suffix, input_names=['raw'],
        output_names=['out'], header=header, source=source, ensure_row_contiguous=True)
    out = kernel(inputs=[mx.array(raw.reshape(-1))], grid=(len(raw), 1, 1),
        threadgroup=(64, 1, 1), output_shapes=[expected.shape], output_dtypes=[mx.uint64])[0]
    mx.eval(out)
    actual = np.asarray(out)
    expected_raw = expected.view(np.uint64)
    mismatch = np.argwhere(actual != expected_raw)
    result = dict(single=single, numpy=np.__version__, rows=len(raw),
        compared_components=expected.size, mismatches=len(mismatch),
        first_mismatches=[dict(row=int(i), column=int(j),
            input=[hex(int(v)) for v in raw[i]], actual=hex(int(actual[i,j])),
            expected=hex(int(expected_raw[i,j]))) for i,j in mismatch[:20]])
    destination = os.environ.get('EON_COMPLEX_LOG_PROBE')
    if destination:
        Path(destination+('-32' if single else '-64')+'.json').write_text(json.dumps(result, indent=2)+'\n')
    assert not len(mismatch), result


@pytest.mark.metal
@pytest.mark.parametrize('single', [False, True])
def test_complex_logarithms_match_scalar_array_and_native_bits(single):
    _probe(single)
