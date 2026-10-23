"""Bitwise host and Metal checks for native binary64 rounding and FMA.

Acceptance is zero ULP against the pinned Brian/NumPy behavior. The
exp/log helpers have separate exact-reference tests; correct rounding alone
is insufficient to establish reference agreement.
"""
from state_assertions import assert_raw_state, assert_spikes_equal

import ctypes
from hashlib import sha256
import math
import random
import shutil
import struct
import subprocess

import numpy as np
import pytest
from native_execution import require_metal

from mlx_backend import binary64, math64
from mlx_backend.compiler import compile_model
from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
from mlx_backend.models import Connectome, Stimulus


def bits(value):
    return struct.unpack("Q", struct.pack("d", value))[0]


def value(raw):
    return struct.unpack("d", struct.pack("Q", raw))[0]


HOST_PREFIX = r"""
#include <cstdint>
#include <cstring>
#include <cmath>
using uint = uint32_t;
using ulong = uint64_t;
#define thread
inline uint clz(ulong x) { return x ? __builtin_clzll(x) : 64u; }
inline float rsqrt(float x) { return 1.0f / std::sqrt(x); }
template<class To, class From> To as_type(From x) {
    To result;
    std::memcpy(&result, &x, sizeof(result));
    return result;
}
"""


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None or ctypes.sizeof(ctypes.c_long) != 8:
        pytest.skip("Host math checks require C++ and 64-bit long")
    directory = tmp_path_factory.mktemp("math64-host")
    source, library = directory / "math.cpp", directory / "math.so"
    source.write_text(HOST_PREFIX + binary64.MSL_HEADER + math64.MSL_MATH_HEADER + r"""
extern "C" {
ulong floor_bits(ulong x) { return d_floor(x); }
ulong ceil_bits(ulong x) { return d_ceil(x); }
ulong roundeven_bits(ulong x) { return d_roundeven(x); }
ulong fma_bits(ulong a, ulong b, ulong c) { return d_fma(a, b, c); }
ulong reference_fma(ulong a, ulong b, ulong c) {
    return as_type<ulong>(std::fma(as_type<double>(a), as_type<double>(b), as_type<double>(c)));
}
}
""")
    subprocess.run([compiler, "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
                    "-shared", "-fPIC", str(source), "-o", str(library)],
                   check=True, capture_output=True, text=True, timeout=30)
    result = ctypes.CDLL(str(library))
    for name in ("floor_bits", "ceil_bits", "roundeven_bits"):
        function = getattr(result, name)
        function.argtypes, function.restype = [ctypes.c_uint64], ctypes.c_uint64
    for name in ("fma_bits", "reference_fma"):
        function = getattr(result, name)
        function.argtypes, function.restype = [ctypes.c_uint64] * 3, ctypes.c_uint64
    return result


def rounding_cases():
    result = [0, 1, 0x8000000000000000, 0x8000000000000001,
              0x7ff0000000000000, 0xfff0000000000000,
              0x7ff0000000000001, 0xfff8000012345678]
    for exponent in range(-1074, 1024, 17):
        x = math.ldexp(1.0, exponent)
        result.extend(bits(y) for sign in (1, -1) for y in
                      (sign*x, math.nextafter(sign*x, -math.inf), math.nextafter(sign*x, math.inf)))
    for x in (-2**53, -2**24-.25, -2.5, -1.5, -.5, .5, 1.5, 2.5, 2**24+.25, 2**53):
        result.extend(bits(y) for y in (x, math.nextafter(x, -math.inf), math.nextafter(x, math.inf)))
    rng = random.Random(20261017)
    result.extend(rng.getrandbits(64) for _ in range(4096))
    return np.array(result, dtype=np.uint64)


@pytest.mark.parametrize("name", ["floor", "ceil", "roundeven"])
def test_host_rounding_matches_numpy_bits_including_signed_zero(host, name):
    raw = rounding_cases()
    operation = np.rint if name == "roundeven" else getattr(np, name)
    with np.errstate(invalid="ignore"):
        expected = operation(raw.view(np.float64))
    actual = np.array([getattr(host, name+"_bits")(int(x)) for x in raw], dtype=np.uint64)
    np.testing.assert_array_equal(actual[~np.isnan(expected)], expected.view(np.uint64)[~np.isnan(expected)])
    assert np.all(np.isnan(actual.view(np.float64)[np.isnan(expected)]))


def fma_cases():
    rng = random.Random(20261017)
    result = [tuple(rng.getrandbits(64) for _ in range(3)) for _ in range(4096)]
    # Every exponent, exact/near cancellation, ties, underflow, and overflow.
    for exponent in range(-1074, 1024, 13):
        x = math.ldexp(1., exponent)
        for y in (1., math.nextafter(1., math.inf), math.nextafter(1., 0.), -1.):
            for z in (-x, math.nextafter(-x, -math.inf), math.nextafter(-x, math.inf)):
                result.append((bits(x), bits(y), bits(z)))
    special = (0., -0., math.inf, -math.inf, math.nan, 1., -1., math.ulp(0.))
    result.extend((bits(a), bits(b), bits(c)) for a in special for b in special for c in special)
    return np.array(result, dtype=np.uint64)


def test_fma_rounds_once_and_matches_independent_host_fma(host):
    raw = fma_cases()
    for a, b, c in raw:
        a, b, c = int(a), int(b), int(c)
        expected, actual = host.reference_fma(a, b, c), host.fma_bits(a, b, c)
        if math.isnan(value(expected)):
            assert math.isnan(value(actual))
        else:
            assert actual == expected, tuple(hex(v) for v in (a, b, c, actual, expected))
    # Separate multiply/add loses a unit here; a fused result retains it.
    a, b, c = bits(2**27+1.), bits(2**27-1.), bits(-float(2**54))
    assert host.fma_bits(a, b, c) == bits(-1.)


def test_dynamic_rounding_emits_binary64_calls():
    program = compile_model(resolve_model({
        "eqs": DEFAULT_EQUATIONS + "a : 1\nb : 1\n",
        "eq_th": "floor(t/dt) >= 0",
        "eq_rst": "v=v_rst; g=0*mV; a=floor(i/2 + t/(4*dt)); b=ceil(-i/3 - t/(4*dt))",
    }))
    assert "d_floor(" in program.threshold_code
    assert "d_floor(" in program.reset_code
    assert "d_ceil(" in program.reset_code
    assert "float(" not in program.reset_code


def metal():
    mx = require_metal('Metal is required')
    return mx


@pytest.mark.metal
@pytest.mark.parametrize("name", ["floor", "ceil", "roundeven"])
def test_metal_rounding_matches_numpy_bits(name):
    mx = metal()
    raw = rounding_cases()
    operation = np.rint if name == "roundeven" else getattr(np, name)
    with np.errstate(invalid="ignore"):
        expected = operation(raw.view(np.float64))
    kernel = mx.fast.metal_kernel(name="math64_"+name, input_names=["raw"], output_names=["out"],
        header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER,
        source=f"uint i=thread_position_in_grid.x; if(i<{raw.size}u) out[i]=d_{name}(raw[i]);",
        ensure_row_contiguous=True)
    out = kernel(inputs=[mx.array(raw)], grid=(raw.size, 1, 1), threadgroup=(128, 1, 1),
                 output_shapes=[raw.shape], output_dtypes=[mx.uint64])[0]
    mx.eval(out)
    actual = np.asarray(out)
    np.testing.assert_array_equal(actual[~np.isnan(expected)], expected.view(np.uint64)[~np.isnan(expected)])
    assert np.all(np.isnan(actual.view(np.float64)[np.isnan(expected)]))


@pytest.mark.metal
def test_metal_fma_matches_independent_host_fma(host):
    mx = metal()
    raw = fma_cases()
    expected = np.array([host.reference_fma(*(int(v) for v in row)) for row in raw], dtype=np.uint64)
    kernel = mx.fast.metal_kernel(name="math64_fma", input_names=["raw"], output_names=["out"],
        header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER,
        source=f"uint i=thread_position_in_grid.x; if(i<{len(raw)}u) out[i]=d_fma(raw[3*i],raw[3*i+1],raw[3*i+2]);",
        ensure_row_contiguous=True)
    out = kernel(inputs=[mx.array(raw)], grid=(len(raw), 1, 1), threadgroup=(128, 1, 1),
                 output_shapes=[(len(raw),)], output_dtypes=[mx.uint64])[0]
    mx.eval(out)
    actual = np.asarray(out)
    regular = ~np.isnan(expected.view(np.float64))
    np.testing.assert_array_equal(actual[regular], expected[regular])
    assert np.all(np.isnan(actual.view(np.float64)[~regular]))


@pytest.mark.metal
def test_floor_ceil_configured_runtime_matches_independent_brian_network():
    metal()
    b = pytest.importorskip("brian2")
    from configured_reference import reference
    from mlx_backend import configured
    model = resolve_model({
        "eqs": DEFAULT_EQUATIONS + "a : 1\nb : 1\n",
        "eq_th": "floor(t/dt) >= 0",
        "eq_rst": "v=v_rst; g=0*mV; a=floor(i/2 + t/(4*dt)); b=ceil(-i/3 - t/(4*dt))",
        "t_rfc": 0*b.ms,
    })
    c = Connectome(np.array(["first", "second", "third"]), np.array([], np.int32),
                   np.array([], np.int32), np.array([], np.float64), np.array([], np.float64))
    events = np.zeros((1, 19, 0), np.uint8)
    stimulus = Stimulus(events, (), (), (7,), 0, 0, sha256(events.tobytes()).hexdigest(), model.dt_ms)
    expected = reference(c, stimulus, model)
    prepared = configured.prepare(c, stimulus, model)
    actual, device_state = configured.run_prepared(prepared, stimulus, return_state=True,
                                                  chunk_steps=7, block_windows=2)
    assert_spikes_equal(actual.spikes, expected.spikes)
    assert len(actual.spikes.steps) == 19*3
    states = configured.host_state(device_state, prepared)
    assert_raw_state(states, expected.state)
