"""Exact-output scalar sine guard, including its rounding-cell boundaries."""
import ctypes
from dataclasses import dataclass
from fractions import Fraction
import platform
import re
import struct
import subprocess

import numpy as np
import pytest
from native_execution import require_metal


_PINNED_COEFFICIENTS = (
    0xbf29943e0fc4818f, 0x3f811073afd145eb, 0xbfc555545268a02a,
    0x3ef99169fc938f4a, 0xbf56c07f16945df4, 0x3fa55553c7899699,
    0xbfdffffffcb82e97, 0x3ff0000000000000,
)


def _coefficient_errors():
    """Read the emitted literals, rather than posit their representation error."""
    from mlx_backend.trigonometry32 import _SINE_HEADER
    from mlx_backend.trigonometry32_certificate import MSL_HEADER
    target_words = []
    for name in ('sine', 'cosine'):
        start = _SINE_HEADER.index('_eon_trig32_'+name+'_poly(')
        body = _SINE_HEADER[start:_SINE_HEADER.index('\n}', start)]
        target_words.extend(int(word, 16) for word in re.findall(r'0x([0-9a-f]+)ul', body))
    assert tuple(target_words) == _PINNED_COEFFICIENTS
    literal = r"([-+]?(?:0x[0-9a-f.]+p[-+]?\d+|\d+\.\d+))f"
    pairs = re.findall(r"_eon_sin32_dd\{"+literal+","+literal+r"\}", MSL_HEADER)
    assert len(pairs) == len(_PINNED_COEFFICIENTS)
    result = {}
    for raw, pair in zip(_PINNED_COEFFICIENTS, pairs):
        target = Fraction(struct.unpack("d", struct.pack("Q", raw))[0])
        values = [float.fromhex(word) if '0x' in word else float(word) for word in pair]
        # Every actual high/low literal is exactly a binary32 value. The pair
        # must be RNE32(target), RNE32(target-high), not merely nearby constants.
        assert all(struct.unpack('f', struct.pack('f', value))[0] == value for value in values)
        high, low = map(Fraction, values)
        assert values[0] == struct.unpack('f', struct.pack('f', float(target)))[0]
        remainder = target-high
        assert Fraction(float(remainder)) == remainder  # Exact Sterbenz residual.
        assert values[1] == struct.unpack('f', struct.pack('f', float(remainder)))[0]
        error = abs(high+low-target)
        assert abs(low) <= abs(high)/2**24
        assert error <= abs(target)/2**48
        result[raw] = error
    return result


def test_sine_certificate_actual_msl_coefficients():
    assert len(_coefficient_errors()) == 8


def test_sine_certificate_graph_error_and_comparison_margin():
    """Rational propagation includes the pinned CPU's double rounding nodes.

    A normalized two-float primitive has a conservative local error below
    16*u32**2 times its operand magnitudes; 2**-120 covers flushed tails.
    Multiplication discards the low*low term, and addition includes the two
    rounded tail additions. Each target multiply/FMA adds its own u64 error.
    """
    u, d, tiny = Fraction(1, 2**24), Fraction(1, 2**53), Fraction(1, 2**120)
    coefficient_errors = _coefficient_errors()

    @dataclass(frozen=True)
    class Bound:
        magnitude: Fraction
        error: Fraction = Fraction(0)

    def mul(a, b):
        m = a.magnitude*b.magnitude
        propagated = a.magnitude*b.error+b.magnitude*a.error+a.error*b.error
        local = 16*u*u*(a.magnitude+a.error)*(b.magnitude+b.error)+tiny
        return Bound(m*(1+d), propagated+local+d*m)

    def fma(a, b, c):
        product = a.magnitude*b.magnitude
        propagated = a.magnitude*b.error+b.magnitude*a.error+a.error*b.error+c.error
        aa, bb, cc = a.magnitude+a.error, b.magnitude+b.error, c.magnitude+c.error
        local_mul = 16*u*u*aa*bb+tiny
        local_add = 16*u*u*(aa*bb+local_mul+cc)+tiny
        return Bound((product+c.magnitude)*(1+d),
                     propagated+local_mul+local_add+d*(product+c.magnitude))

    def coefficient(raw):
        magnitude = abs(Fraction(struct.unpack("d", struct.pack("Q", raw))[0]))
        return Bound(magnitude, coefficient_errors[raw])

    r = Bound(Fraction(1))  # Both accepted reductions have |r| < pi/4 < 1.
    q = mul(r, r)
    cube, fifth = mul(r, q), mul(q, mul(r, q))
    p = fma(q, coefficient(0xbf29943e0fc4818f), coefficient(0x3f811073afd145eb))
    sine = fma(fifth, p, fma(cube, coefficient(0xbfc555545268a02a), r))
    p = fma(q, coefficient(0x3ef99169fc938f4a), coefficient(0xbf56c07f16945df4))
    leading = fma(q, coefficient(0x3fa55553c7899699), coefficient(0xbfdffffffcb82e97))
    cosine = fma(q, fma(mul(q, q), p, leading), Bound(Fraction(1)))
    error = max(sine.error, cosine.error)
    assert max(sine.magnitude+sine.error, cosine.magnitude+cosine.error) < 2
    assert error < Fraction(1, 2**42)
    # |lo| <= 2**-24 in this output range. The floating addition used by the
    # guard errs by at most 2**-48, safely inside the 2**-41 certificate margin.
    assert error+Fraction(1, 2**48) < Fraction(1, 2**41)
    from mlx_backend.trigonometry32 import metal_trigonometry32_header
    scalar = metal_trigonometry32_header({"sin"})
    assert "fabs(value.lo)+0x1p-42f>=radius" in scalar
    assert "_eon_trig32_sine_exact(raw)" in scalar
    assert "#pragma METAL fp contract(off)" in scalar
    assert "_eon_sin32_dd" not in metal_trigonometry32_header({"cos"})


def test_branchless_horner_error_against_pinned_cpu_graph():
    """Both graphs enclose the same exact polynomial, with separate errors."""
    u, d, tiny = Fraction(1, 2**24), Fraction(1, 2**53), Fraction(1, 2**120)
    errors = _coefficient_errors()

    @dataclass(frozen=True)
    class Bound:
        magnitude: Fraction
        error: Fraction = Fraction(0)

    def operation(a, b, c=None, *, native):
        product = a.magnitude*b.magnitude
        propagation = a.magnitude*b.error+b.magnitude*a.error+a.error*b.error
        if not native:
            magnitude = product+(c.magnitude if c else 0)
            propagation += c.error if c else 0
            return Bound(magnitude, propagation+d*(magnitude+propagation))
        aa, bb = a.magnitude+a.error, b.magnitude+b.error
        multiply_error = 16*u*u*aa*bb+tiny
        error = propagation+multiply_error
        if c is None:
            return Bound(product, error)
        add_error = 16*u*u*(aa*bb+multiply_error+c.magnitude+c.error)+tiny
        return Bound(product+c.magnitude, error+c.error+add_error)

    def coefficient(raw, native):
        return Bound(abs(Fraction(struct.unpack('d', struct.pack('Q', raw))[0])),
                     errors[raw] if native else Fraction(0))

    def original(cosine):
        mul = lambda a,b: operation(a,b,native=False)
        fma = lambda a,b,c: operation(a,b,c,native=False)
        coef = lambda raw: coefficient(raw, False)
        r = Bound(Fraction(1));q = mul(r,r)
        if cosine:
            q2 = mul(q,q)
            p = fma(q,coef(_PINNED_COEFFICIENTS[3]),coef(_PINNED_COEFFICIENTS[4]))
            leading = fma(q,coef(_PINNED_COEFFICIENTS[5]),coef(_PINNED_COEFFICIENTS[6]))
            return fma(q,fma(q2,p,leading),Bound(Fraction(1)))
        cube = mul(r,q);fifth = mul(q,cube)
        p = fma(q,coef(_PINNED_COEFFICIENTS[0]),coef(_PINNED_COEFFICIENTS[1]))
        return fma(fifth,p,fma(cube,coef(_PINNED_COEFFICIENTS[2]),r))

    def candidate(cosine):
        mul = lambda a,b: operation(a,b,native=True)
        fma = lambda a,b,c: operation(a,b,c,native=True)
        coef = lambda raw: coefficient(raw, True)
        r = Bound(Fraction(1));q = mul(r,r);cube = mul(r,q)
        c4 = coef(_PINNED_COEFFICIENTS[3]) if cosine else Bound(Fraction(0))
        c3,c2,c1 = [coef(raw) for raw in (_PINNED_COEFFICIENTS[4:7] if cosine
                                        else _PINNED_COEFFICIENTS[:3])]
        p = fma(q,c4,c3);p = fma(q,p,c2);p = fma(q,p,c1)
        return fma(q if cosine else cube,p,Bound(Fraction(1)) if cosine else r)

    for cosine in (False, True):
        exact_cpu, native_dd = original(cosine), candidate(cosine)
        error = exact_cpu.error+native_dd.error
        assert native_dd.magnitude+native_dd.error < 2
        assert error < Fraction(1, 2**42)
        assert error+Fraction(1, 2**48) < Fraction(1, 2**41)
    from mlx_backend.trigonometry32_certificate import MSL_HEADER
    start = MSL_HEADER.index('inline _eon_sin32_dd _eon_sin32_dd_polynomial(')
    body = MSL_HEADER[start:MSL_HEADER.index('\n}', start)]
    assert body.count('_eon_sin32_dd_mul(') == 2
    assert body.count('_eon_sin32_dd_fma(') == 4
    assert 'if(' not in body and '?_eon_sin32_dd' not in body
    assert '_eon_sin32_dd value=_eon_sin32_dd_polynomial(reduced,bool(count&1L));' in MSL_HEADER


def _integer_rne32(raw_integer):
    """Independent integer quotient/remainder RNE; no hardware float conversion."""
    magnitude = abs(int(raw_integer))
    if not magnitude:
        return 0
    exponent = magnitude.bit_length()-1
    shift = max(0, exponent-23)
    significant, remainder = divmod(magnitude, 1 << shift)
    if shift and (2*remainder > 1 << shift or
                  (2*remainder == 1 << shift and significant & 1)):
        significant += 1
    if significant == 1 << 24:
        significant >>= 1
        exponent += 1
    elif exponent < 23:
        significant <<= 23-exponent
    return (int(raw_integer < 0) << 31) | ((exponent+127) << 23) | (significant & 0x7fffff)


def _fraction32(raw):
    sign = -1 if raw >> 31 else 1
    exponent = (raw >> 23) & 255
    significant = (raw & 0x7fffff) | ((1 << 23) if exponent else 0)
    power = (exponent or 1)-150
    return sign*Fraction(significant)*Fraction(2)**power


def _decomposition_witnesses():
    values = {0, 1, (1 << 49)-1}
    for exponent in range(49):
        values.update((1 << exponent)+delta for delta in (-1, 0, 1))
        if exponent < 24:
            continue
        shift = exponent-23
        for significant in (1 << 23, (1 << 23)+1, (1 << 23)+2,
                            3 << 22, (1 << 24)-2, (1 << 24)-1):
            for offset in (0, 1 << (shift-1)):
                values.update((significant << shift)+offset+delta for delta in (-1, 0, 1))
    return sorted({sign*value for value in values for sign in (-1, 1) if 0 <= value < 1 << 49})


def _decomposition_bits(value):
    high = _integer_rne32(value)
    residual = value-int(_fraction32(high))
    low = _integer_rne32(residual)
    # Each nonzero component before scaling is an integer >=1. Multiplication
    # by 2^-49 therefore remains normal and is exact, even in an FTZ mode.
    return tuple(word-(49 << 23) if word & 0x7fffffff else word for word in (high, low))


def test_medium_reduction_dd_exact_decomposition_bound():
    """Proof for every |N|<2^49, with binade/tie/normal-limit witnesses.

    RNE32(N)'s spacing is at most 2^25, including rounding to 2^49 at
    the upper endpoint. Its integer residual thus has |residual|<=2^24,
    representable exactly in binary32. Scaling either integer component
    by 2^-49 is exact and normal (or zero); no reduced DD input can flush.
    The RNE residual also gives |lo|<=u32*|hi|. Scalar subnormal inputs
    take the pre-existing exact fallback before this decomposition.
    """
    for value in _decomposition_witnesses():
        high, low = map(_fraction32, _decomposition_bits(value))
        assert high+low == Fraction(value, 1 << 49)
        assert abs(low) <= abs(high)/2**24
        for component in (high, low):
            assert not component or abs(component) >= Fraction(1, 2**49)
        assert abs(value-int(_fraction32(_integer_rne32(value)))) <= 1 << 24
    from mlx_backend.trigonometry32_certificate import MSL_HEADER
    assert 'absolute<0x39800000u||absolute>=0x42f00000u' in MSL_HEADER


@pytest.mark.metal
def test_medium_reduction_actual_dd_decomposition_on_metal():
    mx = require_metal('Metal required')
    from mlx_backend import binary32, binary64, math64
    from mlx_backend.trigonometry32 import metal_trigonometry32_header
    from mlx_backend.trigonometry32_certificate import MSL_HEADER
    # Exercise the actual production conversion/scaling body over independent
    # synthetic remainders spanning all binades and RNE ties, not only those
    # remainders conveniently generated by a small scalar-input sample.
    start = MSL_HEADER.index('    float high=', MSL_HEADER.index('_eon_sin32_dd_medium('))
    tail = MSL_HEADER[start:MSL_HEADER.index('\n}', start)]
    witness = '\ninline _eon_sin32_dd decomposition_witness(long reduced){\n'+tail+'\n}\n'
    header = binary64.MSL_HEADER+binary32.MSL_HEADER+math64.MSL_MATH_HEADER
    header += metal_trigonometry32_header({'sin'})+witness
    values = np.array(_decomposition_witnesses(), np.int64)
    expected = np.array([_decomposition_bits(int(value)) for value in values], np.uint32)
    kernel = mx.fast.metal_kernel(name='sine_certificate_decomposition', input_names=['values'],
        output_names=['out'], header=header, source='''
        uint i=thread_position_in_grid.x;
        _eon_sin32_dd pair=decomposition_witness(values[i]);
        out[2*i]=as_type<uint>(pair.hi);out[2*i+1]=as_type<uint>(pair.lo);
        ''', compile_options={'math_mode': 'safe'})
    result, = kernel(inputs=[mx.array(values)], grid=(len(values), 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[expected.shape], output_dtypes=[mx.uint32])
    mx.eval(result)
    np.testing.assert_array_equal(np.asarray(result), expected)


def _inputs():
    # Tight-guard fallback inputs found in the retained 65K x8 component.
    fallback = np.array([
        0x3d91a205, 0x40096a0e, 0x40456daa, 0x40490631, 0x40490fde,
        0x40d45487, 0x40e36f46, 0x4116afe5, 0x4116cbcd, 0x4116cbd3,
        0x41490fc1, 0x41490ff6, 0x417b09b7, 0x4196cbce, 0x41afedee,
        0x41c90fd3, 0x41c90fde, 0xc048c78d, 0xc049137d, 0xc049186f,
        0xc0c91633, 0xc116cbea, 0xc1490a3f, 0xc17b09b7, 0xc196cbf7,
        0xc196ccc4, 0xc1afedd1, 0xc1afee33], np.uint32)
    edges = np.array([0, 1, 0x7fffff, 0x800000, 0x39800000, 0x3f490fdb,
                      0x42f00000, 0x4c800000, 0x7f7fffff, 0x7f800000,
                      0x7f800001, 0x7fc00123], np.uint32)
    quads = (np.arange(-77, 78)*np.pi/2).astype(np.float32)
    centers = np.r_[fallback, edges, edges ^ np.uint32(1 << 31), quads.view(np.uint32)]
    neighbors = np.concatenate([centers+np.uint32(delta & 0xffffffff)
                                for delta in range(-8, 9)])
    return np.unique(np.r_[neighbors, np.random.default_rng(20261031).integers(
        0, 1 << 32, 8192, dtype=np.uint32)])


@pytest.mark.metal
@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64",
                    reason="Pinned Apple ARM64 scalar libm contract")
def test_scalar_sine_tight_guard_matches_actual_default_cpu_bits(tmp_path):
    mx = require_metal('Metal required')
    from brian2.codegen.cpp_prefs import get_compiler_and_args
    from mlx_backend import binary32, binary64, math64
    from mlx_backend.trigonometry32 import metal_trigonometry32_header
    raw = _inputs()
    source = tmp_path/"reference.cpp"
    source.write_text('#include <cmath>\nextern "C" void reference(const float* x,float* y,int n)'
                      '{for(int i=0;i<n;++i)y[i]=std::sin(x[i]);}\n')
    _, flags = get_compiler_and_args()
    subprocess.run(["clang++", *flags, "-shared", "-fPIC", str(source), "-o",
                    str(tmp_path/"reference.so")], check=True, capture_output=True, timeout=20)
    imports = subprocess.run(["nm", "-u", str(tmp_path/"reference.so")], check=True,
                             capture_output=True, text=True, timeout=5).stdout
    assert "_sinf" in imports
    reference = ctypes.CDLL(str(tmp_path/"reference.so")).reference
    words = np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS")
    reference.argtypes, reference.restype = [words, words, ctypes.c_int], None
    expected = np.empty_like(raw)
    reference(raw, expected, len(raw))
    header = binary64.MSL_HEADER+binary32.MSL_HEADER+math64.MSL_MATH_HEADER
    header += metal_trigonometry32_header({"sin"})
    kernel = mx.fast.metal_kernel(name="scalar_sine_tight_guard", input_names=["raw"],
        output_names=["out"], header=header,
        source="uint i=thread_position_in_grid.x;out[i]=_eon_trig32_sine_scalar(raw[i]);",
        compile_options={"math_mode": "safe"})
    output, = kernel(inputs=[mx.array(raw)], grid=(len(raw), 1, 1), threadgroup=(128, 1, 1),
                     output_shapes=[raw.shape], output_dtypes=[mx.uint32])
    mx.eval(output)
    np.testing.assert_array_equal(np.asarray(output), expected)


@pytest.mark.metal
@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64",
                    reason="Pinned Apple ARM64 scalar libm contract")
def test_actual_brian_scalar_sine_trajectory_and_continuation(tmp_path):
    mx = require_metal('Metal required')
    from trigonometry_checks import OriginalComparison
    values = _inputs()[::43].view(np.float32)
    options = dict(state_inputs=True, dtype="float32", phase_symbol="certified_phase",
        result_expression="sin(certified_phase); certified_phase+=1/32",
        run_steps=(1, 2, 0, 3), cache_suffix="-certified-sine")
    comparison = OriginalComparison("cython", "sin", values, tmp_path, **options)
    with comparison.native() as (_, _, device):
        assert device.native_steps == 6 and device.native_runs == 4
