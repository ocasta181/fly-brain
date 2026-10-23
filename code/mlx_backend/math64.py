"""Binary64 rounding and fused multiply-add implemented with integer bits.

The Metal source extends :mod:`binary64` and requires its ``MSL_HEADER`` first.
``floor`` and ``ceil`` round the original binary64 operand directly; they never
narrow the value to a hardware float. The fused multiply-add keeps its full
product before rounding once, including subnormal outputs and cancellation.

Native exp/log helpers are defined separately in :mod:`transcendentals`.
"""

__all__ = ["MSL_MATH_HEADER"]

_FMA_HEADER = r'''
    // Three limbs retain the exact 106-bit product and a 53-bit addend
    // throughout cancellation. Far-separated addends need only sticky bits.
    struct d_u192 { ulong lo; ulong mid; ulong hi; };
    inline d_u192 d_shr192(d_u192 a, int n) {
        if (n <= 0) return a;
        if (n >= 192) return {ulong((a.lo | a.mid | a.hi) != 0ul), 0ul, 0ul};
        ulong sticky = 0ul;
        while (n >= 64) {
            sticky |= a.lo;
            a = {a.mid, a.hi, 0ul};
            n -= 64;
        }
        if (n) {
            sticky |= a.lo << (64 - n);
            a = {(a.lo >> n) | (a.mid << (64 - n)),
                 (a.mid >> n) | (a.hi << (64 - n)), a.hi >> n};
        }
        a.lo |= ulong(sticky != 0ul);
        return a;
    }
    inline d_u192 d_shl192(d_u192 a, int n) {
        while (n >= 64) { a = {0ul, a.lo, a.mid}; n -= 64; }
        if (n) a = {a.lo << n, (a.mid << n) | (a.lo >> (64 - n)),
                    (a.hi << n) | (a.mid >> (64 - n))};
        return a;
    }
    inline d_u192 d_add192(d_u192 a, d_u192 b) {
        ulong l = a.lo + b.lo, carry = ulong(l < a.lo);
        ulong m = a.mid + b.mid, h = a.hi + b.hi + ulong(m < a.mid);
        ulong mm = m + carry;
        return {l, mm, h + ulong(mm < m)};
    }
    inline d_u192 d_sub192(d_u192 a, d_u192 b) {
        ulong l = a.lo - b.lo, borrow = ulong(a.lo < b.lo);
        ulong m = a.mid - b.mid, h = a.hi - b.hi - ulong(a.mid < b.mid);
        ulong mm = m - borrow;
        return {l, mm, h - ulong(m < borrow)};
    }
    inline bool d_lt192(d_u192 a, d_u192 b) {
        return a.hi != b.hi ? a.hi < b.hi :
               a.mid != b.mid ? a.mid < b.mid : a.lo < b.lo;
    }
    inline int d_top192(d_u192 a) {
        return a.hi ? 191 - int(clz(a.hi)) :
               a.mid ? 127 - int(clz(a.mid)) : 63 - int(clz(a.lo));
    }
    inline ulong d_fma(ulong a, ulong b, ulong c) {
        if (d_is_nan(a)) return a | 0x0008000000000000ul;
        if (d_is_nan(b)) return b | 0x0008000000000000ul;
        if (d_is_nan(c)) return c | 0x0008000000000000ul;
        ulong s = (a ^ b) >> 63;
        if (d_exp(a) == 0x7fful || d_exp(b) == 0x7fful) {
            if (d_is_zero(a) || d_is_zero(b)) return D_QNAN;
            if (d_exp(c) == 0x7fful && (c >> 63) != s) return D_QNAN;
            return (s << 63) | D_INF;
        }
        if (d_exp(c) == 0x7fful) return c;
        if (d_is_zero(a) || d_is_zero(b)) return d_add(s << 63, c);
        ulong ma = d_mant(a), mb = d_mant(b), mc = d_mant(c);
        int ea = int(d_exp(a)), eb = int(d_exp(b)), ec = int(d_exp(c));
        if (ea) ma |= 1ul << 52; else ea = 1;
        if (eb) mb |= 1ul << 52; else eb = 1;
        if (ec) mc |= 1ul << 52; else ec = 1;
        ulong ph, pl = d_umul128(ma, mb, &ph);
        d_u192 product = {pl, ph, 0ul};
        int top = d_top192(product);
        int pe = ea + eb - 2150 + top;
        product = d_shl192(product, 190 - top);
        d_u192 addend = {mc, 0ul, 0ul};
        int ce = -100000;
        if (mc) {
            int ct = 63 - int(clz(mc));
            ce = ec - 1075 + ct;
            addend = d_shl192(addend, 190 - ct);
        }
        int exponent = pe > ce ? pe : ce;
        product = d_shr192(product, exponent - pe);
        addend = d_shr192(addend, exponent - ce);
        d_u192 result;
        if (s == (c >> 63)) result = d_add192(product, addend);
        else if (d_lt192(product, addend)) {
            result = d_sub192(addend, product); s = c >> 63;
        } else result = d_sub192(product, addend);
        if ((result.lo | result.mid | result.hi) == 0ul) return 0ul;
        int rt = d_top192(result);
        d_u192 rounded = rt > 55 ? d_shr192(result, rt - 55)
                                 : d_shl192(result, 55 - rt);
        return d_finish(s, exponent + rt - 190 + 1023, rounded.lo);
    }
    inline ulong d_roundeven(ulong x) {
        int exponent = int(d_exp(x)) - 1023;
        if (exponent >= 52) return x;
        if (exponent < -1) return x & 0x8000000000000000ul;
        ulong sign = x & 0x8000000000000000ul;
        if (exponent == -1) return sign | (d_mant(x) ? 0x3ff0000000000000ul : 0ul);
        int n = 52 - exponent;
        ulong mask = (1ul << n) - 1ul, frac = x & mask, midpoint = 1ul << (n - 1);
        ulong value = x & ~mask;
        if (frac > midpoint || (frac == midpoint && ((value >> n) & 1ul))) value += 1ul << n;
        return value;
    }
'''

_ROUNDING_HEADER = r'''
    inline ulong d_floor(ulong x) {
        if (d_is_nan(x)) return x | 0x0008000000000000ul;
        int exponent = int(d_exp(x)) - 1023;
        if (exponent >= 52 || d_is_zero(x)) return x;
        bool negative = (x >> 63) != 0ul;
        if (exponent < 0) return negative ? 0xbff0000000000000ul : 0ul;
        int shift = 52 - exponent;
        ulong mask = (1ul << shift) - 1ul;
        if ((x & mask) == 0ul) return x;
        ulong integral = x & ~mask;
        return negative ? integral + (1ul << shift) : integral;
    }
    inline ulong d_ceil(ulong x) {
        if (d_is_nan(x)) return x | 0x0008000000000000ul;
        int exponent = int(d_exp(x)) - 1023;
        if (exponent >= 52 || d_is_zero(x)) return x;
        bool negative = (x >> 63) != 0ul;
        if (exponent < 0) return negative ? 0x8000000000000000ul : 0x3ff0000000000000ul;
        int shift = 52 - exponent;
        ulong mask = (1ul << shift) - 1ul;
        if ((x & mask) == 0ul) return x;
        ulong integral = x & ~mask;
        return negative ? integral : integral + (1ul << shift);
    }
'''

MSL_MATH_HEADER = _FMA_HEADER + _ROUNDING_HEADER
