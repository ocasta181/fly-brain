"""Independent boundary checks for the exact small-count binary64 reducer."""
import ctypes
from fractions import Fraction

import numpy as np
import pytest
from native_execution import require_metal

from mlx_backend import binary64, math64, trigonometry64


def _header():
    return (binary64.MSL_HEADER + math64.MSL_MATH_HEADER
            + trigonometry64.metal_trigonometry_header({"sin"}))


def test_small_count_products_and_sterbenz_bounds():
    high, mid, tail, reciprocal = np.array([
        0x3ff921fb54440000, 0x3d868c234c4c0000,
        0x3b298a2e03707345, 0x3fe45f306dc9c883,
    ], np.uint64).view(np.float64)
    for count in range(1, 32):
        assert Fraction(count * high) == Fraction(count * 0x6487ed511, 1 << 34)
        assert Fraction(count * mid) == Fraction(count * 0x5a308d313, 1 << 73)
        # This padding exceeds the binary64 reciprocal-product rounding error
        # throughout the small-count interval, including its cell boundaries.
        lower = (Fraction(count) - Fraction(1, 2) - Fraction(1, 1 << 47)) / Fraction(reciprocal)
        upper = (Fraction(count) + Fraction(1, 2) + Fraction(1, 1 << 47)) / Fraction(reciprocal)
        assert Fraction(1, 2) < lower / Fraction(count * high)
        assert upper / Fraction(count * high) < 2
        rounded = count * tail
        center = Fraction(rounded)
        exact = Fraction(count) * Fraction(tail)
        assert (Fraction(np.nextafter(rounded, -np.inf)) + center) / 2 <= exact
        assert exact <= (Fraction(np.nextafter(rounded, np.inf)) + center) / 2


@pytest.mark.metal
def test_small_fixed_reduction_and_guard_boundaries():
    mx = require_metal('Metal required')
    rng = np.random.default_rng(20261018)
    high, mid, reciprocal = np.array([
        0x3ff921fb54440000, 0x3d868c234c4c0000, 0x3fe45f306dc9c883,
    ], np.uint64).view(np.float64)
    centers = np.r_[np.arange(1, 34) * high, np.arange(1, 34) * (high + mid),
                    (np.arange(1, 34) + .5) * np.pi / 2]
    values = np.r_[rng.uniform(np.pi / 4, 51, 4096), centers,
                   np.nextafter(centers, -np.inf), np.nextafter(centers, np.inf)]
    counts = np.rint(values * reciprocal).astype(np.uint32)
    remaining = values - counts.astype(np.float64) * high
    wanted_high = remaining - counts.astype(np.float64) * mid
    wanted_error = (remaining - wanted_high) - counts.astype(np.float64) * mid
    expected = np.column_stack([wanted_high.view(np.uint64), wanted_error.view(np.uint64)])
    kernel = mx.fast.metal_kernel(
        name="test_sine_fixed_reduction", input_names=["x", "counts"], output_names=["out"],
        header=_header(), compile_options={"math_mode": "safe"},
        source="uint i=thread_position_in_grid.x;auto p=_eon_trig_small_fixed(x[i],counts[i]);"
               "out[3*i]=p.high;out[3*i+1]=p.error;out[3*i+2]=ulong(p.valid);")
    result, = kernel(inputs=[mx.array(values.view(np.uint64)), mx.array(counts)],
        grid=(values.size, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(values.size, 3)], output_dtypes=[mx.uint64])
    mx.eval(result)
    actual = np.asarray(result)
    valid = actual[:, 2] != 0
    assert valid.sum() > 3900
    assert (~valid).sum() > 100
    assert np.all(np.abs(remaining[valid]) >= counts[valid].astype(np.float64) * mid)
    np.testing.assert_array_equal(actual[valid, :2], expected[valid])
    # A full sine call exercises the guarded fallback, signs and nonfinite
    # handling against the independent scalar CPU libm, not another GPU helper.
    raw = np.r_[values.view(np.uint64), (-values).view(np.uint64),
                np.array([0, 1 << 63, 0x7ff0000000000000, 0x7ff0000000000001,
                          0x7fefffffffffffff], np.uint64)]
    lib = ctypes.CDLL(None)
    lib.sin.argtypes = [ctypes.c_double]
    lib.sin.restype = ctypes.c_double
    expected_sine = np.array([lib.sin(float(x)) for x in raw.view(np.float64)], np.float64).view(np.uint64)
    sine = mx.fast.metal_kernel(name="test_sine_fixed_full", input_names=["x"], output_names=["out"],
        header=_header(), compile_options={"math_mode": "safe"},
        source="uint i=thread_position_in_grid.x;out[i]=d_sine(x[i]);")
    result, = sine(inputs=[mx.array(raw)], grid=(raw.size, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[raw.shape], output_dtypes=[mx.uint64])
    mx.eval(result)
    np.testing.assert_array_equal(np.asarray(result), expected_sine)


@pytest.mark.metal
def test_dominant_sum_lower_binade_midpoints():
    mx = require_metal('Metal required')
    powers = np.arange(55, 2047, dtype=np.uint64) << np.uint64(52)
    a = np.repeat(powers, 9)
    b = np.column_stack([powers - (np.uint64(gap) << np.uint64(52)) + np.uint64(delta)
        for gap in (53, 54, 55) for delta in (0, 1, 0xfffffffffffff)]).reshape(-1)
    a, b = np.r_[a, a | np.uint64(1 << 63)], np.r_[b | np.uint64(1 << 63), b]
    expected = (a.view(np.float64) + b.view(np.float64)).view(np.uint64)
    kernel = mx.fast.metal_kernel(name="test_sine_dominant_sum", input_names=["a", "b"], output_names=["out"],
        header=_header(), compile_options={"math_mode": "safe"},
        source="uint i=thread_position_in_grid.x;out[i]=_eon_trig_dominant_sum(a[i],b[i]);")
    result, = kernel(inputs=[mx.array(a), mx.array(b)], grid=(a.size, 1, 1),
        threadgroup=(128, 1, 1), output_shapes=[a.shape], output_dtypes=[mx.uint64])
    mx.eval(result)
    np.testing.assert_array_equal(np.asarray(result), expected)
