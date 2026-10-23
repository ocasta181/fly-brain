"""Software IEEE-754 binary64 for Metal, represented by raw ``ulong`` bits.

The arithmetic header is derived from ArrowMetal's
DoubleMath.swift at commit 1f2111540e68edfdf1adf87df8821ca9cd1958c0
(https://github.com/singhpratech/ArrowMetal). Copyright 2026 ArrowMetal
contributors; Apache-2.0. See third_party/ArrowMetal-LICENSE and
third_party/ArrowMetal-NOTICE for the license and pinned provenance.

Local changes preserve signaling-NaN operand priority and subtraction signs.
Local additions provide ordered comparisons, sign operations, and integer
conversion. ``d_exp`` in the original header extracts the exponent field;
it does not calculate the exponential function. Use safe Metal math mode.
"""

import struct

__all__ = ["MSL_HEADER", "literal"]


def literal(value: float) -> str:
    """Return an MSL ``ulong`` literal containing the binary64 bits of value."""
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return f"{bits}ul"


_DIVISION_HEADER = r'''
    inline ulong @NAME@(ulong a, ulong b) {
        ulong s = (a ^ b) >> 63;
        int ea = (int)d_exp(a), eb = (int)d_exp(b);
        ulong ma = d_mant(a), mb = d_mant(b);
@PRIORITY@        if (d_is_nan(a)) return a | (1ul << 51);
        if (d_is_nan(b)) return b | (1ul << 51);
        if (ea == 0x7FF) return (eb == 0x7FF) ? D_QNAN : ((s << 63) | D_INF);
        if (eb == 0x7FF) return s << 63;
        if (d_is_zero(b)) return d_is_zero(a) ? D_QNAN : ((s << 63) | D_INF);
        if (d_is_zero(a)) return s << 63;
        if (ea) ma |= 1ul << 52; else { int z = (int)clz(ma) - 11; ma <<= z; ea = 1 - z; }
        if (eb) mb |= 1ul << 52; else { int z = (int)clz(mb) - 11; mb <<= z; eb = 1 - z; }
        // V ~= 2^126 / D for the normalised divisor D, so V lands in (2^62, 2^63] and never overflows.
        ulong D = mb << 11;                             // [2^63, 2^64)
        float rf = 1.0f / (float)((uint)(D >> 32));
        ulong V = ((ulong)(uint)(rf * 18014398509481984.0f)) << 40;   // rf * 2^54, about 23 good bits
        for (int i = 0; i < @STEPS@; i++) {                   // 23 -> 45 -> past the 62 bits V can hold
            long e = (long)(1ul << 62) - (long)d_mulhi(D, V);
            V = (e >= 0) ? (V + 4ul * d_mulhi(V, (ulong)e)) : (V - 4ul * d_mulhi(V, (ulong)(-e)));
        }
        // N = ma * 2^67 = n1 * 2^64 with n1 < 2^56 <= D, so N / D is the 57-bit quotient wanted.
        ulong n1 = ma << 3;
        ulong ph, pl;
        pl = d_umul128(n1, V, &ph);
        ulong q = (ph << 2) | (pl >> 62);
        // rh:rl = N - q * D, exactly, in two's complement. |q - floor(N/D)| <= 1, so rh is 0 or all ones.
        ulong qh, ql;
        ql = d_umul128(q, D, &qh);
        ulong rl = 0ul - ql;
        ulong rh = n1 - qh - ((ql != 0ul) ? 1ul : 0ul);
        if ((rh >> 63) != 0ul) {                        // q was one too big
            q--;
            ulong t = rl + D;
            rh += (t < rl) ? 1ul : 0ul;
            rl = t;
        } else if (rh != 0ul || rl >= D) {              // q was one too small
            q++;
            rh -= (rl < D) ? 1ul : 0ul;
            rl -= D;
        }
        if (rl != 0ul) q |= 1ul;                        // sticky
        // q = floor(ma * 2^56 / mb) (57 bits at most)  =>  value = (q / 2^55) * 2^(ea - eb - 1)
        return d_finish(s, ea - eb + 1022, q);
    }
'''


def _division_header(name: str, *, steps: int, signaling_priority: bool) -> str:
    priority = (
        "        if (d_is_nan(a) && !(ma & (1ul << 51))) return a | (1ul << 51);\n"
        "        if (d_is_nan(b) && !(mb & (1ul << 51))) return b | (1ul << 51);\n"
        if signaling_priority else ""
    )
    return (_DIVISION_HEADER.removeprefix("\n").replace("@NAME@", name)
            .replace("@STEPS@", str(steps)).replace("@PRIORITY@", priority))


# Derived from the pinned upstream MSL; local changes are recorded in its NOTICE.
_ARROWMETAL_HEADER = r'''

    #define D_INF  0x7FF0000000000000ul
    #define D_QNAN 0x7FF8000000000000ul
    inline ulong d_exp(ulong x) { return (x >> 52) & 0x7FFul; }
    inline ulong d_mant(ulong x) { return x & 0xFFFFFFFFFFFFFul; }
    inline bool d_is_nan(ulong x) { return d_exp(x) == 0x7FFul && d_mant(x) != 0ul; }
    inline bool d_is_zero(ulong x) { return (x & 0x7FFFFFFFFFFFFFFFul) == 0ul; }

    // Rounds and packs: sign s, biased exponent e for a significand whose leading one sits at bit 55
    // (three guard bits below the 53-bit significand; bit 0 carries sticky).
    //
    // The exponent is carried as an `int`, not a `long`: every value it can hold fits in sixteen bits,
    // and a 64-bit add or compare is two instructions on a GPU whose ALUs are 32 bits wide. The
    // normalisation is one `clz` and one shift rather than the two shift loops it used to be — those
    // loops ran up to fifty times after a cancelling subtraction, and a loop with a data-dependent trip
    // count costs the whole SIMD group, not just the lane that needed it.
    inline ulong d_finish(ulong s, long e64, ulong m) {
        if (m == 0ul) return s << 63;
        int e = (int)e64;
        int sh = (int)clz(m) - 8;                       // leading one to bit 55, either direction
        if (sh > 0) { m <<= sh; e -= sh; }
        else if (sh < 0) {
            int r = -sh;
            ulong lost = m & ((1ul << r) - 1ul);
            m = (m >> r) | (lost ? 1ul : 0ul);
            e += r;
        }
        if (e <= 0) {
            int shn = 1 - e;
            if (shn > 60) { m = 1ul; }                  // m is non-zero, so only the sticky bit survives
            else { ulong st = (m & ((1ul << shn) - 1ul)) ? 1ul : 0ul; m = (m >> shn) | st; }
            e = 0;
        }
        ulong low = m & 7ul; m >>= 3;
        if (low > 4ul || (low == 4ul && (m & 1ul))) m++;
        if ((m >> 53) != 0ul) { m >>= 1; e++; }
        if (e == 0 && (m >> 52) != 0ul) e = 1;
        if (e >= 0x7FF) return (s << 63) | D_INF;
        return (s << 63) | ((ulong)e << 52) | (m & 0xFFFFFFFFFFFFFul);
    }

    inline ulong d_add(ulong a, ulong b) {
        ulong sa = a >> 63, sb = b >> 63;
        int ea = (int)d_exp(a), eb = (int)d_exp(b);
        ulong ma = d_mant(a), mb = d_mant(b);
        if (ea == 0x7FF || eb == 0x7FF) {
            if (ea == 0x7FF && ma && !(ma & (1ul << 51))) return a | (1ul << 51);
            if (eb == 0x7FF && mb && !(mb & (1ul << 51))) return b | (1ul << 51);
            if (ea == 0x7FF && ma) return a | (1ul << 51);
            if (eb == 0x7FF && mb) return b | (1ul << 51);
            if (ea == 0x7FF && eb == 0x7FF) return (sa == sb) ? a : D_QNAN;
            return (ea == 0x7FF) ? a : b;
        }
        bool za = (ea == 0 && ma == 0ul), zb = (eb == 0 && mb == 0ul);
        if (za && zb) return (sa & sb) << 63;
        if (za) return b;
        if (zb) return a;
        if (ea) ma |= 1ul << 52; else ea = 1;
        if (eb) mb |= 1ul << 52; else eb = 1;
        ma <<= 3; mb <<= 3;
        if (ea < eb || (ea == eb && ma < mb)) {
            ulong tm = ma; ma = mb; mb = tm; int te = ea; ea = eb; eb = te; ulong ts = sa; sa = sb; sb = ts;
        }
        int d = ea - eb;
        if (d > 0) {
            if (d >= 64) mb = 1ul;
            else { ulong st = (mb & ((1ul << d) - 1ul)) ? 1ul : 0ul; mb = (mb >> d) | st; }
        }
        ulong m = (sa == sb) ? (ma + mb) : (ma - mb);
        if (m == 0ul) return 0ul;
        return d_finish(sa, ea, m);
    }
    // Exact widening of a float bit pattern to a double bit pattern (subnormals preserved: no float arithmetic).
    inline ulong d_from_float(float f) {
        uint b = as_type<uint>(f);
        ulong s = (ulong)(b >> 31); ulong e = (b >> 23) & 0xFFu; ulong m = b & 0x7FFFFFu;
        if (e == 0xFFu) return (s << 63) | 0x7FF0000000000000ul | (m ? ((1ul << 51) | (m << 29)) : 0ul);
        if (e == 0u) {
            if (m == 0u) return s << 63;
            long ee = 1; while ((m >> 23) == 0ul) { m <<= 1; ee--; }
            m &= 0x7FFFFFu;
            return (s << 63) | ((ulong)(ee - 127 + 1023) << 52) | (m << 29);
        }
        return (s << 63) | ((e - 127 + 1023) << 52) | (m << 29);
    }
    inline ulong d_sub(ulong a, ulong b) {
        // Fsub propagates an operand NaN before applying the subtraction
        // sign. Negating b first changes observable NaN signs and payload
        // priority when a floating operator feeds an enclosing expression.
        if(d_is_nan(a) || d_is_nan(b)) {
            if(d_is_nan(a) && !(a & (1ul << 51))) return a | (1ul << 51);
            if(d_is_nan(b) && !(b & (1ul << 51))) return b | (1ul << 51);
            return (d_is_nan(a) ? a : b) | (1ul << 51);
        }
        return d_add(a, b ^ 0x8000000000000000ul);
    }

    inline ulong d_mul(ulong a, ulong b) {
        ulong s = (a ^ b) >> 63;
        int ea = (int)d_exp(a), eb = (int)d_exp(b);
        ulong ma = d_mant(a), mb = d_mant(b);
        if (ea == 0x7FF || eb == 0x7FF) {
            if (ea == 0x7FF && ma && !(ma & (1ul << 51))) return a | (1ul << 51);
            if (eb == 0x7FF && mb && !(mb & (1ul << 51))) return b | (1ul << 51);
            if (ea == 0x7FF && ma) return a | (1ul << 51);
            if (eb == 0x7FF && mb) return b | (1ul << 51);
            if (d_is_zero(a) || d_is_zero(b)) return D_QNAN;
            return (s << 63) | D_INF;
        }
        if (d_is_zero(a) || d_is_zero(b)) return s << 63;
        if (ea) ma |= 1ul << 52; else { int z = (int)clz(ma) - 11; ma <<= z; ea = 1 - z; }
        if (eb) mb |= 1ul << 52; else { int z = (int)clz(mb) - 11; mb <<= z; eb = 1 - z; }
        // 53 x 53 bits as four 32 x 32 products. Apple's ALUs are 32 bits wide, so a `mulhi(ulong,
        // ulong)` next to a `ma * mb` is lowered to two independent emulation sequences over the same
        // partial products; writing them out once shares them and drops the redundant halves.
        // `a1`, `b1` are under 2^21, so `mid` stays under 2^54 and cannot carry out of 64 bits.
        uint a1 = (uint)(ma >> 32), a0 = (uint)ma;
        uint b1 = (uint)(mb >> 32), b0 = (uint)mb;
        ulong p00 = (ulong)a0 * (ulong)b0;
        ulong mid = (ulong)a0 * (ulong)b1 + (ulong)a1 * (ulong)b0;
        ulong lo = p00 + (mid << 32);
        ulong hi = (ulong)a1 * (ulong)b1 + (mid >> 32) + ((lo < p00) ? 1ul : 0ul);
        ulong m = (hi << 15) | (lo >> 49);
        m |= (lo & ((1ul << 49) - 1ul)) ? 1ul : 0ul;
        return d_finish(s, ea + eb - 1023, m);
    }

    // Full 64 x 64 -> 128 product, out of four 32 x 32 ones. Apple's ALUs are 32 bits wide, so this is
    // what the hardware does anyway; writing it out shares the partial products between the two halves
    // instead of computing them twice for a `mulhi` and a `*`.
    inline ulong d_umul128(ulong a, ulong b, thread ulong* hiOut) {
        uint a1 = (uint)(a >> 32), a0 = (uint)a;
        uint b1 = (uint)(b >> 32), b0 = (uint)b;
        ulong p00 = (ulong)a0 * (ulong)b0;
        ulong mid = (ulong)a0 * (ulong)b1 + (p00 >> 32);        // cannot carry out of 64 bits
        ulong mid2 = mid + (ulong)a1 * (ulong)b0;               // this one can
        ulong carry = (mid2 < mid) ? (1ul << 32) : 0ul;
        *hiOut = (ulong)a1 * (ulong)b1 + (mid2 >> 32) + carry;
        return (mid2 << 32) | (p00 & 0xFFFFFFFFul);
    }
    inline ulong d_mulhi(ulong a, ulong b) { ulong h; d_umul128(a, b, &h); return h; }

    // Correctly rounded division: a Newton reciprocal of the divisor, one multiply for the quotient,
    // and an **exact** remainder to settle the last bit.
    //
    // This used to be a 57-step restoring long division — one quotient bit per iteration, each with a
    // 64-bit compare, subtract and shift, and a data-dependent branch that the whole SIMD group pays
    // for. The reciprocal costs one hardware `float` division for about 22 bits and two Newton steps
    // (`V += 4 * mulhi(V, 2^62 - mulhi(D, V))`, which doubles the correct bits each time) to reach the
    // 62 the quotient needs. That leaves the quotient at most one off, and the remainder
    // `N - q * D` — computed exactly, in 128 bits — says which way, so the answer is still the
    // correctly rounded one and not an approximation with a good reputation. The correction never took
    // more than a single step over 200k random significand pairs, and `DoubleMathTests` compares the
    // result against Swift's `Double` bit for bit.
    //
    // Three Newton steps, not the two the `mathMode = .safe` reciprocal needs. Under safe math the seed
    // is good to about 2^-22 and two steps already saturate the 63 bits `V` holds; the third costs
    // around 25 instructions in a kernel that is memory bound anyway, and it keeps the seed requirement
    // down at 2^-12 — so the quotient stays within the one correction step this code allows for even if
    // the compile options ever move to fast math. A numerical kernel that quietly stops being correctly
    // rounded when a build flag changes is not worth the instructions saved.
''' + _division_header("d_div", steps=3, signaling_priority=True) + r'''
    // Correctly rounded square root: a hardware `rsqrt` seed, three Newton steps in fixed point, and
    // an **exact** 128-bit remainder that settles the last bit — the same shape as `d_div` above.
    //
    // Write a = m * 2^k with m an integer and k **even** (halving an odd exponent is what loses the
    // last bit, so the significand absorbs the odd one), which puts m in [2^52, 2^54). Everything the
    // rounding needs is then
    //     q = floor(sqrt(m * 2^54))            53 significand bits and one guard bit
    //     m * 2^54 - q^2 == 0 ?                the sticky flag
    // and the whole job is to produce that q. It used to come out of 54 restoring steps, one bit at a
    // time, each with a 64-bit shift, compare and subtract — around 800 instructions in a kernel whose
    // arithmetic is otherwise a single load and store.
    //
    // Instead, normalise A = m << 10 into [2^62, 2^64), so a = A * 2^-64 lies in [0.25, 1) and
    // sqrt(m * 2^54) = sqrt(a) * 2^54. One `rsqrt` on the top 24 bits of A seeds y ~ 1/sqrt(a) in Q62;
    // three Newton steps `y += y * (1 - a * y^2) / 2` (each doubling the correct bits, and each three
    // 64 x 64 -> high-64 products) take it to about 2^-58, past the 57 bits the answer needs; and
    // `q = (a * y) >> 8` then lands within a couple of units of the true floor. The correction loop
    // makes it exact rather than merely close: it holds the full 128-bit remainder N - q^2, so it
    // *knows* which side of the answer q is on. Over 40M random doubles it has never taken more than
    // two steps, and the loop is bounded at four.
    //
    // Three Newton steps rather than two for the reason `d_div` gives: two saturate the seed that safe
    // math delivers, but the third keeps the seed requirement down at 2^-12, so the routine does not
    // quietly stop being correctly rounded if the compile options ever move off safe math.
    inline ulong d_sqrt(ulong a) {
        if (d_is_nan(a)) return a | (1ul << 51);
        if (d_is_zero(a)) return a;                     // sqrt(-0) is -0, as IEEE-754 says
        if ((a >> 63) != 0ul) return D_QNAN;
        if (d_exp(a) == 0x7FFul) return a;              // +inf
        long k;
        ulong m;
        if (d_exp(a) == 0ul) {                          // subnormal: normalise, no arithmetic needed
            m = d_mant(a); int z = (int)clz(m) - 11; m <<= z; k = -1074 - (long)z;
        } else {
            m = d_mant(a) | (1ul << 52); k = (long)d_exp(a) - 1075;
        }
        if (k & 1) { m <<= 1; k--; }                    // m now spans [2^52, 2^54) and k is even
        ulong A = m << 10;                              // [2^62, 2^64): a = A * 2^-64 in [0.25, 1)
        float af = (float)(uint)(A >> 40) * (1.0f / 16777216.0f);   // the top 24 bits, exactly
        ulong Y = (ulong)(rsqrt(af) * 4611686018427387904.0f);      // 1/sqrt(a) in Q62, in (2^62, 2^63]
        for (int i = 0; i < 3; i++) {
            ulong y2 = d_mulhi(Y, Y);                                   // y^2 in Q60, in [2^60, 2^62]
            long e = (long)(1ul << 60) - (long)d_mulhi(A, y2);          // (1 - a y^2) in Q60, small
            Y = (e >= 0) ? (Y + (d_mulhi(Y, (ulong)e) << 3))
                         : (Y - (d_mulhi(Y, (ulong)(-e)) << 3));
        }
        ulong q = d_mulhi(A, Y) >> 8;                   // sqrt(a) in Q62, then floor to Q54
        // N = m * 2^54 as a 128-bit pair, and the exact remainder N - q^2.
        ulong nhi = m >> 10, nlo = m << 54;
        ulong qh, ql;
        ql = d_umul128(q, q, &qh);
        ulong rl = nlo - ql;
        ulong rh = nhi - qh - ((nlo < ql) ? 1ul : 0ul);
        for (int i = 0; i < 4; i++) {
            if ((rh >> 63) != 0ul) {                    // q is one too big: N - (q-1)^2 = r + 2(q-1) + 1
                q--;
                ulong t = 2ul * q + 1ul, nl = rl + t;
                rh += (nl < rl) ? 1ul : 0ul;
                rl = nl;
            } else if (rh != 0ul || rl > 2ul * q) {     // q is one too small: N - (q+1)^2 = r - 2q - 1
                ulong t = 2ul * q + 1ul;
                rh -= (rl < t) ? 1ul : 0ul;
                rl -= t;
                q++;
            } else {
                break;
            }
        }
        ulong s = q >> 1;
        // A set guard bit always comes with a non-zero remainder: an exact halfway result would need a
        // 54-bit root, whose square has 107 significant bits and so cannot be a double. The tie term is
        // written out anyway, because relying on that silently would be worse than paying for one `and`.
        if ((q & 1ul) && ((rl | rh) != 0ul || (s & 1ul))) s++;
        long e = (k >> 1) + 26 + 1023;                  // sqrt of any finite double is normal
        if ((s >> 53) != 0ul) { s >>= 1; e++; }
        return ((ulong)e << 52) | (s & 0xFFFFFFFFFFFFFul);
    }
    '''


# These helpers operate on bit patterns, never through a hardware float cast.
_LOCAL_HEADER = r'''
    // Local additions to ArrowMetal's unchanged binary64 arithmetic header.
    inline ulong d_neg(ulong a) { return a ^ 0x8000000000000000ul; }
    inline ulong d_abs(ulong a) { return a & 0x7FFFFFFFFFFFFFFFul; }

    inline bool d_eq(ulong a, ulong b) {
        if (d_is_nan(a) || d_is_nan(b)) return false;
        return a == b || (d_is_zero(a) && d_is_zero(b));
    }
    inline bool d_gt(ulong a, ulong b) {
        if (d_is_nan(a) || d_is_nan(b)) return false;
        if (d_is_zero(a) && d_is_zero(b)) return false;
        ulong ka = (a >> 63) ? ~a : (a | 0x8000000000000000ul);
        ulong kb = (b >> 63) ? ~b : (b | 0x8000000000000000ul);
        return ka > kb;
    }
    inline bool d_lt(ulong a, ulong b) { return d_gt(b, a); }
    inline bool d_ge(ulong a, ulong b) { return d_gt(a, b) || d_eq(a, b); }
    inline bool d_le(ulong a, ulong b) { return d_gt(b, a) || d_eq(a, b); }

    // NumPy's signed integer loops use floor division and the divisor's sign
    // for the remainder. Zero and MIN/-1 have defined wrapping loop results.
    inline long n_integer_floor_div(long a, long b) {
        if (b == 0l) return 0l;
        if (a == (-9223372036854775807l - 1l) && b == -1l) return a;
        long q = a / b, r = a % b;
        return q - (r != 0l && ((r < 0l) != (b < 0l)));
    }
    inline long n_integer_mod(long a, long b) {
        if (b == 0l || b == -1l) return 0l;
        long r = a % b;
        return r != 0l && ((r < 0l) != (b < 0l)) ? r + b : r;
    }
    inline ulong n_integer_pow(ulong base, ulong exponent) {
        ulong result = 1ul;
        while (exponent) {
            if (exponent & 1ul) result *= base;
            exponent >>= 1;
            if (exponent) base *= base;
        }
        return result;
    }

    // Correctly rounded signed-integer conversion, without float intermediates.
    // Unsigned subtraction also handles the magnitude of LONG_MIN exactly.
    inline ulong d_from_int(long value) {
        if (value == 0l) return 0ul;
        ulong sign = value < 0l ? 1ul : 0ul;
        ulong magnitude = sign ? 0ul - (ulong)value : (ulong)value;
        int leading = 63 - (int)clz(magnitude);
        ulong significand;
        if (leading <= 55) {
            significand = magnitude << (55 - leading);
        } else {
            int shift = leading - 55;
            ulong lost = magnitude & ((1ul << shift) - 1ul);
            significand = (magnitude >> shift) | (lost ? 1ul : 0ul);
        }
        return d_finish(sign, leading + 1023, significand);
    }

    // Keep uint64 state above LONG_MAX positive and round only once.
    inline ulong d_from_uint(ulong value) {
        if (value == 0ul) return 0ul;
        int leading = 63 - (int)clz(value);
        ulong significand;
        if (leading <= 55) {
            significand = value << (55 - leading);
        } else {
            int shift = leading - 55;
            ulong lost = value & ((1ul << shift) - 1ul);
            significand = (value >> shift) | (lost ? 1ul : 0ul);
        }
        return d_finish(0ul, leading + 1023, significand);
    }

    // Truncate toward zero. Saturate out-of-range values and infinities;
    // NaN converts to zero. No hardware floating-point conversion is used.
    inline long d_to_int(ulong value) {
        const long maximum = 9223372036854775807l;
        const long minimum = -9223372036854775807l - 1l;
        if (d_is_nan(value)) return 0l;
        int exponent = (int)d_exp(value) - 1023;
        if (exponent < 0) return 0l;
        bool negative = (value >> 63) != 0ul;
        if (exponent >= 63) return negative ? minimum : maximum;
        ulong magnitude = d_mant(value) | (1ul << 52);
        if (exponent >= 52) magnitude <<= exponent - 52;
        else magnitude >>= 52 - exponent;
        return negative ? -(long)magnitude : (long)magnitude;
    }
    // The frozen arm64 CPU's scalar FCVTZS32 differs from a 64-bit
    // conversion followed by narrowing outside the signed-32 range.
    inline int d_to_int32_sat(ulong value) {
        long result = d_to_int(value);
        return result > 2147483647L ? 2147483647 :
               result < (-2147483647L-1L) ? (-2147483647-1) : int(result);
    }
    // NumPy1.26 arm64's contiguous f64->i32 cast loop handles blocks of
    // eight via FCVTZS64+narrow, and the remainder via scalar FCVTZS32.
    inline int n_to_int32(ulong value, long position, long length, int source_bits=64) {
        return source_bits==64 && position < (length/8)*8 ? int(d_to_int(value)) : d_to_int32_sat(value);
    }


    inline ulong d_to_uint_sat(ulong value) {
        if((value >> 63) || d_is_nan(value)) return 0UL;
        int exponent = int(d_exp(value))-1023;
        if(exponent < 0) return 0UL;
        if(exponent >= 64) return ~0UL;
        ulong magnitude = d_mant(value) | (1UL << 52);
        return exponent >= 52 ? magnitude << (exponent-52) : magnitude >> (52-exponent);
    }
    inline uint d_to_uint32_sat(ulong value) {
        ulong result = d_to_uint_sat(value);
        return result > 0xffffffffUL ? 0xffffffffU : uint(result);
    }
    inline long n_to_signed(ulong value, int bits, long position, long length) {
        if(bits==64) return d_to_int(value);
        int block=bits==16 ? 32 : 8;
        return position < (length/block)*block ? d_to_int(value) : long(d_to_int32_sat(value));
    }
    inline ulong n_to_unsigned(ulong value, int bits, long position, long length) {
        if(bits==64) return d_to_uint_sat(value);
        return position < (length/8)*8 ? d_to_uint_sat(value) : ulong(d_to_uint32_sat(value));
    }

'''

MSL_HEADER: str = _ARROWMETAL_HEADER + _LOCAL_HEADER
