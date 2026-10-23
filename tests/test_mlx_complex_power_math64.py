"""Focused raw-bit power dispatch and installed complex-product recovery."""
import ctypes
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mlx_backend import binary64, binary32, math64, basic_math32
from mlx_backend.complex_log_math64 import metal_complex_log_header
from mlx_backend.complex_power_math64 import metal_complex_power_header
from mlx_backend.stable_math64 import metal_stable_math_header
from mlx_backend.transcendentals import metal_math_header
from mlx_backend.trigonometry64 import metal_trigonometry_header


def _inputs(single, recovery):
    one, half, inf = 0x3ff0000000000000, 0x3fe0000000000000, 0x7ff0000000000000
    sign, qnan, other = 0x8000000000000000, 0x7ff8000012345678, 0xfff8000087654321
    if recovery:
        rows = [(inf, 0, one, 0), (inf, qnan, one, 0),
                (one, 0, inf, other), (inf, inf, inf, inf),
                (qnan, other, half, one), (one, one, qnan, other),
                (0x7fefffffffffffff, 0x7fefffffffffffff, one, one),
                (0, sign, sign, 0), (qnan ^ (1 << 51), other, half, one)]
    else:
        rows = [(0, sign, 0, sign), (0, sign, one, qnan),
                (0, sign, sign | one, 0), (0, 0, 0, one),
                (0, 0, qnan, 0), (qnan, other, 0, 0),
                (qnan, other, one, 0), (inf, 0, one, 0),
                (inf, qnan, 0x4000000000000000, 0),
                (inf, other, 0x4008000000000000, 0),
                (one, sign, 0x4058c00000000000, 0),
                (one, sign, 0xc058c00000000000, 0),
                (qnan, other, half, one), (one, one, qnan, other),
                (inf, 0, half, 0), (inf, inf, half, 0),
                (one, 0, inf, 0), (one, 0, 0, inf),
                (sign | one, 0, inf, 0), (qnan ^ (1 << 51), other, half, one)]
    raw = np.array(rows, np.uint64)
    if single:
        with np.errstate(all='ignore'):
            raw = raw.view(np.float64).astype(np.float32).astype(np.float64).view(np.uint64)
    return raw


def _expected(raw, single, recovery):
    if recovery:
        kind = ctypes.c_float if single else ctypes.c_double
        class Pair(ctypes.Structure):
            _fields_ = [('real', kind), ('imag', kind)]
        function = getattr(ctypes.CDLL(None), '__mulsc3' if single else '__muldc3')
        function.argtypes, function.restype = [kind]*4, Pair
        result = [function(*map(float, row)) for row in raw.view(np.float64)]
        return np.array([(z.real, z.imag) for z in result], np.float64)
    values = raw.view(np.float64).astype(np.float32 if single else np.float64)
    values = values.view(np.complex64 if single else np.complex128).reshape(-1, 2)
    with np.errstate(all='ignore'):
        result = np.power(values[:, 0], values[:, 1])
    return np.column_stack((result.real, result.imag)).astype(np.float64)


@pytest.mark.metal
@pytest.mark.parametrize('single', [False, True])
@pytest.mark.parametrize('recovery', [True, False], ids=['product-recovery', 'numpy-wrapper'])
def test_complex_power_boundaries_match_raw_bits(single, recovery):
    mx = pytest.importorskip('mlx.core')
    complex_math = pytest.importorskip('mlx_backend.complex_math64')
    if not mx.metal.is_available():
        pytest.skip('Actual Metal is required')
    raw = _inputs(single, recovery)
    expected = _expected(raw, single, recovery).view(np.uint64)
    if recovery:
        call = f'_eon_complex_power_recover(a,b,{str(single).lower()})'
    else:
        call = '_eon_complex_power'+('32' if single else '')+'(a,b)'
    header = (binary64.MSL_HEADER+binary32.MSL_HEADER+math64.MSL_MATH_HEADER
              +metal_math_header({'exp', 'log'})
              +metal_stable_math_header({'expm1', 'log1p'})
              +basic_math32.MSL_HEADER+metal_trigonometry_header({'sin', 'cos'})
              +complex_math.metal_complex_header({'_complex', '_complex_basic', '_complex_trig'})
              +metal_complex_log_header()+metal_complex_power_header())
    source = f'''uint i=thread_position_in_grid.x;if(i>={len(raw)}u)return;
    _eon_complex a={{raw[4*i],raw[4*i+1]}},b={{raw[4*i+2],raw[4*i+3]}};
    _eon_complex value={call};out[2*i]=value.real;out[2*i+1]=value.imag;'''
    kernel = mx.fast.metal_kernel(name='power_boundary_'+str(single)+'_'+str(recovery),
        input_names=['raw'], output_names=['out'], header=header, source=source)
    out = kernel(inputs=[mx.array(raw.reshape(-1))], grid=(len(raw), 1, 1),
        threadgroup=(32, 1, 1), output_shapes=[expected.shape], output_dtypes=[mx.uint64])[0]
    mx.eval(out)
    actual = np.asarray(out)
    mismatches = np.argwhere(actual != expected)
    result = dict(single=single, recovery=recovery, rows=len(raw), mismatches=len(mismatches),
        first_mismatches=[dict(row=int(i), column=int(j), input=[hex(int(x)) for x in raw[i]],
            actual=hex(int(actual[i, j])), expected=hex(int(expected[i, j])))
            for i, j in mismatches[:12]])
    destination = os.environ.get('EON_COMPLEX_POWER_PROBE')
    if destination:
        Path(destination+f'-{single}-{recovery}.json').write_text(json.dumps(result, indent=2)+'\n')
    assert not len(mismatches), result
