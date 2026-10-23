"""Power-local direct product packing versus independent CPU multiplication."""
import numpy as np
import pytest

from mlx_backend import binary64, math64, operator64


@pytest.mark.metal
def test_power_local_product_rounds_exactly_at_normal_and_gradual_boundaries():
    mx = pytest.importorskip("mlx.core")
    centers = np.array([
        0, 1, 2, 0x000fffffffffffff, 0x0010000000000000,
        0x0010000000000001, 0x3fdfffffffffffff, 0x3fe0000000000000,
        0x3fefffffffffffff, 0x3ff0000000000000, 0x3ff0000000000001,
        0x3ff0000000000002, 0x3ff8000000000000, 0x3fffffffffffffff,
        0x4000000000000000, 0x7fdfffffffffffff, 0x7fe0000000000000,
        0x7fefffffffffffff, 0x7ff0000000000000, 0x7ff0000000000001,
        0x7ff8123456789abc,
    ], np.uint64)
    centers = np.r_[centers, centers ^ np.uint64(1 << 63)]
    a, b = np.meshgrid(centers, centers)
    rng = np.random.default_rng(616119)
    a = np.r_[a.ravel(), rng.integers(0, np.iinfo(np.uint64).max, 16384, dtype=np.uint64)]
    b = np.r_[b.ravel(), rng.integers(0, np.iinfo(np.uint64).max, 16384, dtype=np.uint64)]
    # Every exponent plus exact/neighbor products around underflow/overflow.
    exponent = np.arange(1, 2047, dtype=np.uint64) << np.uint64(52)
    boundary = np.r_[exponent - np.uint64(1), exponent, exponent + np.uint64(1)]
    for factor in (0x3fefffffffffffff, 0x3ff0000000000001, 0x3ff8000000000000):
        a = np.r_[a, boundary]
        b = np.r_[b, np.full(len(boundary), factor, np.uint64)]
    kernel = mx.fast.metal_kernel(
        name="operator_power_local_exact_product",
        input_names=["a", "b"], output_names=["out"],
        header=binary64.MSL_HEADER + math64.MSL_MATH_HEADER + operator64.MSL_POWER_HEADER,
        source="""uint i=thread_position_in_grid.x;if(i<COUNT){
          out[i]=_eon_operator_mul(a[i],b[i]);out[COUNT+i]=d_mul(a[i],b[i]);
        }""", compile_options={"math_mode": "safe"})
    result, = kernel(inputs=[mx.array(a), mx.array(b)], template=[("COUNT", len(a))],
        grid=(len(a), 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(2, len(a))], output_dtypes=[mx.uint64])
    mx.eval(result)
    result = np.asarray(result)
    # Exceptional payload policy remains the already qualified canonical one.
    np.testing.assert_array_equal(result[0], result[1])
    with np.errstate(all="ignore"):
        expected = a.view(np.float64) * b.view(np.float64)
    finite_operands = np.isfinite(a.view(np.float64)) & np.isfinite(b.view(np.float64))
    np.testing.assert_array_equal(result[0, finite_operands], expected.view(np.uint64)[finite_operands])
