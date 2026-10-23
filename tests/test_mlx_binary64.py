"""Host-only checks for binary64 literals, provenance, and added integer helpers."""

import ctypes
import hashlib
import math
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess

import pytest

from mlx_backend import binary64


def _bits(value):
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def _value(bits):
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


@pytest.mark.parametrize("value, bits", [
    (0.0, 0x0000000000000000),
    (-0.0, 0x8000000000000000),
    (1.0, 0x3FF0000000000000),
    (-1.0, 0xBFF0000000000000),
    (math.inf, 0x7FF0000000000000),
    (-math.inf, 0xFFF0000000000000),
    (math.ulp(0.0), 0x0000000000000001),
    (float.fromhex("0x1.fffffffffffffp+1023"), 0x7FEFFFFFFFFFFFFF),
    (1.0 + 1e-9, 0x3FF000000044B830),
    (2**53 + 1, 0x4340000000000000),
])
def test_literal_preserves_binary64_bits(value, bits):
    assert binary64.literal(value) == f"{bits}ul"


def test_literal_preserves_nan_payload_and_sign():
    for bits in [0x7FF8000000000001, 0xFFF8000012345678]:
        assert binary64.literal(_value(bits)) == f"{bits}ul"


def test_arithmetic_derivation_is_pinned():
    digest = hashlib.sha256(binary64._ARROWMETAL_HEADER.encode()).hexdigest()
    assert digest == "d2201622862f82028cb401d6ef6d56f233a7cbe6a0ad8b034c2c59a1b4d179d5"
    assert binary64.MSL_HEADER == binary64._ARROWMETAL_HEADER + binary64._LOCAL_HEADER
    names = set(re.findall(r"inline\s+(?:bool|ulong|long)\s+(d_\w+)\(", binary64.MSL_HEADER))
    assert {
        "d_add", "d_sub", "d_mul", "d_div", "d_sqrt", "d_from_float",
        "d_gt", "d_ge", "d_lt", "d_le", "d_eq", "d_abs", "d_neg",
        "d_from_int", "d_to_int",
    } <= names


def test_arrowmetal_license_and_provenance_are_packaged():
    third_party = Path(binary64.__file__).parent / "third_party"
    license_text = (third_party / "ArrowMetal-LICENSE").read_text()
    notice = (third_party / "ArrowMetal-NOTICE").read_text()
    assert "Apache License" in license_text
    assert "Copyright 2026 ArrowMetal contributors" in license_text
    assert "https://github.com/singhpratech/ArrowMetal" in notice
    assert "1f2111540e68edfdf1adf87df8821ca9cd1958c0" in notice
    assert "Extracted MSL SHA-256: 08e9f6da055f9af13839cf561fa30f870f1b9c68ddc40880522bb59973b116a5" in notice
    assert hashlib.sha256(binary64._ARROWMETAL_HEADER.encode()).hexdigest() in notice


@pytest.fixture(scope="module")
def host_helpers(tmp_path_factory):
    """Compile the actual MSL text with host shims; no MLX or Metal execution.

    Only the appended bit/integer helpers run here. The arithmetic's float
    reciprocal seeds need the separate native Metal conformance check.
    """
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None or ctypes.sizeof(ctypes.c_long) != 8:
        pytest.skip("Host helper checks need a C++ compiler and 64-bit long")
    directory = tmp_path_factory.mktemp("binary64-host")
    source = directory / "helpers.cpp"
    library = directory / "helpers.so"
    source.write_text("""
#include <cstdint>
#include <cstring>
#include <cmath>
using uint = uint32_t;
using ulong = uint64_t;
#define thread
inline uint clz(ulong x) { return x ? __builtin_clzll(x) : 64u; }
inline float rsqrt(float x) { return 1.0f / std::sqrt(x); }
template<class To, class From> To as_type(From value) {
    static_assert(sizeof(To) == sizeof(From));
    To result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
""" + binary64.MSL_HEADER + """
extern "C" {
    ulong from_int(long x) { return d_from_int(x); }
    long to_int(ulong x) { return d_to_int(x); }
    ulong negate(ulong x) { return d_neg(x); }
    ulong absolute(ulong x) { return d_abs(x); }
    bool gt(ulong a, ulong b) { return d_gt(a, b); }
    bool ge(ulong a, ulong b) { return d_ge(a, b); }
    bool lt(ulong a, ulong b) { return d_lt(a, b); }
    bool le(ulong a, ulong b) { return d_le(a, b); }
    bool eq(ulong a, ulong b) { return d_eq(a, b); }
}
""")
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-fno-fast-math", "-shared", "-fPIC",
         str(source), "-o", str(library)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    helpers = ctypes.CDLL(str(library))
    helpers.from_int.argtypes = [ctypes.c_int64]
    helpers.from_int.restype = ctypes.c_uint64
    helpers.to_int.argtypes = [ctypes.c_uint64]
    helpers.to_int.restype = ctypes.c_int64
    for name in ["negate", "absolute"]:
        getattr(helpers, name).argtypes = [ctypes.c_uint64]
        getattr(helpers, name).restype = ctypes.c_uint64
    for name in ["gt", "ge", "lt", "le", "eq"]:
        getattr(helpers, name).argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        getattr(helpers, name).restype = ctypes.c_bool
    return helpers


def test_signed_integer_conversion_uses_full_precision_and_ties_even(host_helpers):
    values = [0, 1, -1, 2**63 - 1, -(2**63)]
    for exponent in range(64):
        values.extend(
            sign * (2**exponent + offset)
            for offset in [-3, -1, 0, 1, 3]
            for sign in [-1, 1]
            if -(2**63) <= sign * (2**exponent + offset) < 2**63
        )
    rng = random.Random(20261017)
    values.extend(rng.randrange(-(2**63), 2**63) for _ in range(2000))
    for value in values:
        assert host_helpers.from_int(value) == _bits(value), value


def test_to_int_truncates_and_saturates_without_float32_rounding(host_helpers):
    values = [0.0, -0.0, 0.999, -0.999, 1.999, -1.999,
              2**31 + 1.75, -(2**31 + 1.75), 2**52 + 1.0,
              float(2**63), -float(2**63),
              math.nextafter(float(2**63), 0),
              math.nextafter(-float(2**63), -math.inf),
              math.inf, -math.inf, math.nan]
    rng = random.Random(20261017)
    values.extend(_value(rng.getrandbits(64)) for _ in range(2000))
    for value in values:
        if math.isnan(value):
            expected = 0
        elif value >= 2**63:
            expected = 2**63 - 1
        elif value <= -(2**63):
            expected = -(2**63)
        else:
            expected = int(value)
        assert host_helpers.to_int(_bits(value)) == expected, value


def test_comparisons_match_ieee_including_nan_and_signed_zero(host_helpers):
    values = [0.0, -0.0, 1.0, -1.0, math.ulp(0.0), -math.ulp(0.0),
              math.inf, -math.inf, math.nan, _value(0xFFF8000000000001)]
    rng = random.Random(20261017)
    values.extend(_value(rng.getrandbits(64)) for _ in range(64))
    for a in values:
        for b in values:
            aa, bb = _bits(a), _bits(b)
            assert host_helpers.gt(aa, bb) == (a > b)
            assert host_helpers.ge(aa, bb) == (a >= b)
            assert host_helpers.lt(aa, bb) == (a < b)
            assert host_helpers.le(aa, bb) == (a <= b)
            assert host_helpers.eq(aa, bb) == (a == b)


def test_sign_helpers_preserve_all_payload_bits(host_helpers):
    rng = random.Random(20261017)
    values = [0, 1, 0x8000000000000000, 0x7FF0000000000000,
              0xFFF0000000000000, 0x7FF0000000000001, 0xFFF8000000000001]
    values.extend(rng.getrandbits(64) for _ in range(1000))
    for bits in values:
        assert host_helpers.negate(bits) == bits ^ 0x8000000000000000
        assert host_helpers.absolute(bits) == bits & 0x7FFFFFFFFFFFFFFF
