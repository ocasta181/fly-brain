"""Stable binary64 functions for the pinned Apple/Brian reference.

The source uses only integer-backed binary64 operations. Its rounded operation
order and coefficient bits reproduce macOS ARM64 libsystem_m, UUID 83b173cd7f3239018e67c0b2676be015.
The Python module does not call system math at execution time. Include
``binary64.MSL_HEADER`` and ``math64.MSL_MATH_HEADER`` before these helpers.
Compile with math_mode="safe"; the division specialization relies on a safe
float reciprocal seed, then corrects its binary64 quotient exactly.
"""

from .binary64 import _division_header
from .scalar_math_tables import exponential64_segment_table, logarithm64_table

__all__ = ["metal_stable_math_header", "MSL_STABLE_MATH_HEADER"]

# A binary64 product has at most 106 significant bits. Aligning its leading bit
# at 126 keeps the entire product for close cancellation. If exponent separation
# discards low bits, magnitudes differ by at least 2**22, so cancellation cannot
# expose those bits: a sticky bit suffices for the final 53-bit rounding.
_STABLE_FMA_HEADER = r'''
#ifndef EON_STABLE_FMA_INCLUDED
#define EON_STABLE_FMA_INCLUDED
struct stable_u128 { ulong lo; ulong hi; };
inline stable_u128 stable_shr128(stable_u128 a,int n) {
    if(n<=0) return a;
    if(n>=128) return {ulong((a.lo|a.hi)!=0ul),0ul};
    if(n>=64) {
        ulong sticky=a.lo;
        a={a.hi,0ul}; n-=64;
        if(n) { sticky|=a.lo<<(64-n); a.lo>>=n; }
        a.lo|=ulong(sticky!=0ul); return a;
    }
    ulong sticky=a.lo<<(64-n);
    return {(a.lo>>n)|(a.hi<<(64-n))|ulong(sticky!=0ul),a.hi>>n};
}
inline stable_u128 stable_shl128(stable_u128 a,int n) {
    if(n>=64) return {0ul,a.lo<<(n-64)};
    if(n) return {a.lo<<n,(a.hi<<n)|(a.lo>>(64-n))};
    return a;
}
inline int stable_top128(stable_u128 a) {
    return a.hi ? 127-int(clz(a.hi)) : 63-int(clz(a.lo));
}
inline ulong stable_fma(ulong a,ulong b,ulong c) {
    int ea=int(d_exp(a)),eb=int(d_exp(b)),ec=int(d_exp(c));
    if(uint(ea-1)>=2046u || uint(eb-1)>=2046u || uint(ec-1)>=2046u) {
        if(ec==0 && d_is_zero(c) && uint(ea-1)<2046u && uint(eb-1)<2046u)
            return d_mul(a,b);
        return d_fma(a,b,c);
    }
    ulong sign=(a^b)>>63;
    ulong ma=d_mant(a)|(1ul<<52),mb=d_mant(b)|(1ul<<52),mc=d_mant(c)|(1ul<<52);
    ulong ph,pl=d_umul128(ma,mb,&ph);
    // Normal 53-bit operands put the product's highest bit at 104 or 105. Normalizing this
    // known width avoids variable leading-zero work in every polynomial FMA.
    int top=(ph>>41) ? 105:104;
    int pe=ea+eb-2150+top;
    stable_u128 product=stable_shl128({pl,ph},126-top);
    int ce=ec-1023;
    stable_u128 addend={0ul,mc<<10};
    int exponent=pe>ce ? pe:ce;
    product=stable_shr128(product,exponent-pe);
    addend=stable_shr128(addend,exponent-ce);
    stable_u128 result;
    if(sign==(c>>63)) {
        ulong lo=product.lo+addend.lo;
        result={lo,product.hi+addend.hi+ulong(lo<product.lo)};
    } else {
        bool smaller=product.hi!=addend.hi ? product.hi<addend.hi : product.lo<addend.lo;
        stable_u128 large=smaller ? addend:product,small=smaller ? product:addend;
        result={large.lo-small.lo,large.hi-small.hi-ulong(large.lo<small.lo)};
        if(smaller) sign=c>>63;
    }
    if((result.lo|result.hi)==0ul) return 0ul;
    int rt=stable_top128(result);
    stable_u128 rounded=rt>55 ? stable_shr128(result,rt-55):stable_shl128(result,55-rt);
    int result_exponent=exponent+rt-126+1023;
    if(result_exponent<=0 || result_exponent>=2047)
        return d_finish(sign,result_exponent,rounded.lo);
    // rounded.lo already has its leading bit at55. Keep ArrowMetal's final
    // round-to-even/pack rules, without normalizing this known width again.
    ulong mantissa=rounded.lo>>3,low=rounded.lo&7ul;
    if(low>4ul || (low==4ul && (mantissa&1ul))) ++mantissa;
    if(mantissa>>53) { mantissa>>=1; ++result_exponent; }
    if(result_exponent>=2047) return (sign<<63)|D_INF;
    return (sign<<63)|(ulong(result_exponent)<<52)|(mantissa&0xffffffffffffful);
}
#endif // EON_STABLE_FMA_INCLUDED
'''

# Division derives from ArrowMetal DoubleMath.swift, commit
# 1f2111540e68edfdf1adf87df8821ca9cd1958c0, copyright 2026 ArrowMetal contributors,
# Apache-2.0; see third_party/ArrowMetal-LICENSE and ArrowMetal-NOTICE.
# Modification: two Newton steps under explicitly required safe math. The float
# seed has at least 22 correct bits; two steps exceed the quotient's 57 bits.
# Exact remainder correction, sticky bits and final rounding still decide the
# binary64 answer. The global division helper is unchanged.
_STABLE_DIV_HEADER = "\n" + _division_header("stable_div", steps=2, signaling_priority=False)

_EXPM1_HEADER = r'''
constant ulong stable_expm1_coeff[7]={
    0x40671547652b82feul,0x3ce777d0ffda0d24ul,0x3d25e38de424a447ul,0x3dc3b2a9845c31ccul,
    0x3e5c6b08d73824d6ul,0x3eeebfbdff82c694ul,0x3f762e42fefa39eful,
};
constant ulong stable_expm1_tiny[6]={
    0x3f2a01d1ca3a507aul,0x3f56c1999aee6814ul,0x3f81111110ee6827ul,0x3fa555555527cfeful,
    0x3fc5555555555576ul,0x3fe0000000000017ul,
};
''' + exponential64_segment_table('stable_expm1_table', four_columns=True) + r'''

inline ulong d_expm1(ulong x) {
    ulong a=d_abs(x);
    if(a >= 0x40862e42fefa39f0ul || x >= 0xc0430fc100000000ul) {
        if(d_lt(x,0ul)) return 0xbff0000000000000ul;
        return d_mul(x,0x7feffffffffffffful);
    }
    if(a < 0x3fa0000000000000ul) {
        if(a < 0x3ca0000000000000ul) return x;
        ulong p=stable_fma(x,stable_expm1_tiny[0],stable_expm1_tiny[1]);
        ulong square=d_mul(x,x);
        p=stable_fma(x,p,stable_expm1_tiny[2]);
        p=stable_fma(x,p,stable_expm1_tiny[3]);
        p=stable_fma(x,p,stable_expm1_tiny[4]);
        p=stable_fma(x,p,stable_expm1_tiny[5]);
        return stable_fma(square,p,x);
    }
    ulong t=d_mul(x,stable_expm1_coeff[0]);
    long k=d_to_int(t);
    if(d_lt(t,d_from_int(k))) --k;
    ulong r=d_sub(t,d_from_int(k));
    ulong error=stable_fma(x,stable_expm1_coeff[0],d_neg(t));
    ulong delta=stable_fma(x,stable_expm1_coeff[1],error);
    uint index=uint(k)&127u;
    ulong base=stable_expm1_table[2*index],tail=stable_expm1_table[2*index+1];
    ulong p=stable_fma(r,stable_expm1_coeff[2],stable_expm1_coeff[3]);
    p=stable_fma(r,p,stable_expm1_coeff[4]);
    r=d_add(r,d_add(tail,delta));
    p=stable_fma(r,p,stable_expm1_coeff[5]);
    p=stable_fma(r,p,stable_expm1_coeff[6]);
    ulong factor=d_mul(r,p);
    long scale=k>>7;
    ulong shift=ulong(scale)<<52;
    ulong negative_one=scale>=1023 ? 0x8000000000000000ul : 0xbff0000000000000ul-shift;
    ulong y;
    if(scale < -1) {
        ulong high=d_add(base,negative_one);
        ulong low=d_sub(base,d_sub(high,negative_one));
        y=d_add(stable_fma(base,factor,low),high);
    } else if(scale > 52) {
        y=d_add(base,stable_fma(base,factor,negative_one));
    } else {
        y=stable_fma(base,factor,d_add(base,negative_one));
    }
    return y+shift;
}
// Brian reserves leading underscores for generated identifiers. Keep model
// call sites there, so a valid state named d_expm1 cannot shadow this helper.
inline ulong _eon_stable_expm1(ulong x) { return d_expm1(x); }
'''

_LOG1P_HEADER = r'''
constant ulong stable_log1p_coeff[6]={
    0xbfc7b47c30037039ul,0x3fc249449239443dul,0x3ff9de61616cdf2bul,0x3ff609d14c7ee9beul,
    0x3ffab7d4f0547db7ul,0xbff3f6ab6ec10818ul,
};
''' + logarithm64_table(
    'stable_log1p_table', padded=True, separator=",", trailing_comma=True
) + r'''

inline ulong d_log1p(ulong x) {
    if(x==D_INF || d_is_nan(x)) return x;
    if(d_eq(x,0xbff0000000000000ul)) return 0xfff0000000000000ul;
    if(d_lt(x,0xbff0000000000000ul)) return D_QNAN;
    if(d_le(d_abs(x),0x3e40000000000000ul)) {
        ulong half_argument=(d_abs(x)>>32) < 0x200000ul ? x : d_mul(x,0x3fe0000000000000ul);
        return stable_fma(d_neg(x),half_argument,x);
    }
    ulong sum=d_add(x,0x3ff0000000000000ul);
    ulong fraction=sum&0xffffffffffffful;
    long k=as_type<long>(sum+0xc018100000000000ul)>>52;
    uint index=uint((fraction+0x100000000000ul)>>45);
    ulong m=fraction|0x3ff0000000000000ul;
    ulong inv=stable_log1p_table[4*index],hi=stable_log1p_table[4*index+1],lo=stable_log1p_table[4*index+2];
    ulong r=stable_fma(m,inv,0xbff0000000000000ul);
    ulong correction=d_mul(d_sub(x,d_sub(sum,0x3ff0000000000000ul)),inv);
    long adjustment=1024-k-(fraction<0x7f00000000000ul ? 1 : 0);
    ulong scale=adjustment<0 ? 0ul : ulong(adjustment)<<52;
    ulong p1=stable_fma(r,stable_log1p_coeff[1],stable_log1p_coeff[0]);
    ulong b=d_add(r,stable_log1p_coeff[3]),c=d_add(r,stable_log1p_coeff[5]);
    correction=d_mul(correction,scale);
    r=d_add(r,correction);
    ulong square=d_mul(r,r);
    ulong p2=stable_fma(r,b,stable_log1p_coeff[2]);
    ulong p3=stable_fma(r,c,stable_log1p_coeff[4]);
    ulong p=d_mul(d_mul(d_mul(square,p1),p2),p3);
    if(k!=0) {
        lo=stable_fma(d_from_int(k),0x3e5df473de6af279ul,lo);
        hi=stable_fma(d_from_int(k),0x3fe62e42f0000000ul,hi);
        return d_add(hi,d_add(r,d_add(lo,p)));
    }
    ulong prod=d_mul(inv,m),rounded=d_sub(prod,0x3ff0000000000000ul);
    ulong error=stable_fma(inv,m,d_neg(prod));
    ulong high=rounded&0xfffffffff8000000ul;
    ulong tail=d_add(error,d_sub(rounded,high));
    tail=d_add(tail,correction);
    tail=d_add(tail,lo);
    tail=d_add(tail,p);
    return d_add(d_add(high,hi),tail);
}
inline ulong _eon_stable_log1p(ulong x) { return d_log1p(x); }
'''

_EXPREL_HEADER = r'''
inline ulong d_exprel(ulong x) {
    if(d_lt(d_abs(x),0x3c9cd2b297d889bcul)) return 0x3ff0000000000000ul;
    if(d_gt(x,0x4086680000000000ul)) return D_INF;
    return stable_div(d_expm1(x),x);
}
inline ulong _eon_stable_exprel(ulong x) { return d_exprel(x); }
inline ulong _eon_stable_exprel_cached(ulong x,ulong expm1_x) {
    if(d_lt(d_abs(x),0x3c9cd2b297d889bcul)) return 0x3ff0000000000000ul;
    if(d_gt(x,0x4086680000000000ul)) return D_INF;
    return stable_div(expm1_x,x);
}
'''

def metal_stable_math_header(functions, *, expm1_fma='stable_fma'):
    """Emit only the called functions and the expm1 dependency of exprel."""
    names = set(functions)
    return ((_STABLE_FMA_HEADER if names & {"expm1", "log1p", "exprel"} else "")
            + (_STABLE_DIV_HEADER if names & {"log1p", "exprel"} else "")
            + (_EXPM1_HEADER.replace('stable_fma(',expm1_fma+'(')
               if names & {"expm1", "exprel"} else "")
            + (_LOG1P_HEADER if "log1p" in names else "")
            + (_EXPREL_HEADER if "exprel" in names else ""))


MSL_STABLE_MATH_HEADER = _STABLE_FMA_HEADER + _STABLE_DIV_HEADER + _EXPM1_HEADER + _LOG1P_HEADER + _EXPREL_HEADER
