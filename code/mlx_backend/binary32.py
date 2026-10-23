"""Binary32 rounding on raw bits, including subnormals and ties to even.

Metal's floating point denormal modes must not change Brian state. Reuse the
binary64 arithmetic for exceptional intermediates and round explicitly to
binary32. Normal finite operations use IEEE hardware rounding; denormals,
overflow and signed zero retain the integer implementation.
"""

MSL_HEADER = r'''
inline uint f_from_d(ulong value) {
    uint sign = uint(value >> 32) & 0x80000000u;
    int exponent = int(d_exp(value));
    ulong mantissa = d_mant(value);
    if(exponent == 2047)
        return sign | 0x7f800000u | (mantissa ? (0x400000u | uint(mantissa >> 29)) : 0u);
    if(exponent == 0) return sign;
    mantissa |= 1ul << 52;
    exponent -= 896;
    if(exponent >= 255) return sign | 0x7f800000u;
    int shift = exponent > 0 ? 29 : 30-exponent;
    if(shift > 53) return sign;
    ulong rounded = mantissa >> shift;
    ulong remainder = mantissa & ((1ul << shift)-1ul);
    ulong halfway = 1ul << (shift-1);
    if(remainder > halfway || (remainder == halfway && (rounded & 1ul))) ++rounded;
    if(exponent <= 0) return sign | uint(rounded);
    if(rounded == (1ul << 24)) { rounded >>= 1; ++exponent; }
    if(exponent >= 255) return sign | 0x7f800000u;
    return sign | (uint(exponent) << 23) | (uint(rounded) & 0x7fffffu);
}
inline ulong f_to_d(uint value) { return d_from_float(as_type<float>(value)); }
inline uint f_from_int(long value) {
    uint sign = value < 0 ? 0x80000000u : 0u;
    ulong magnitude = value < 0 ? ulong(-(value+1L))+1ul : ulong(value);
    if(!magnitude) return 0u;
    int leading = 63-int(clz(magnitude));
    int exponent = leading+127;
    ulong rounded;
    if(leading <= 23) rounded = magnitude << (23-leading);
    else {
        int shift = leading-23;
        rounded = magnitude >> shift;
        ulong remainder = magnitude & ((1ul << shift)-1ul);
        ulong halfway = 1ul << (shift-1);
        if(remainder > halfway || (remainder == halfway && (rounded & 1ul))) ++rounded;
        if(rounded == (1ul << 24)) { rounded >>= 1; ++exponent; }
    }
    return sign | (uint(exponent) << 23) | (uint(rounded) & 0x7fffffu);
}
inline bool _eon_f_is_nan(uint value) {return (value&0x7fffffffu)>0x7f800000u;}
inline uint _eon_f_nan2(uint a,uint b) {
    if(_eon_f_is_nan(a) && !(a&0x00400000u)) return a|0x00400000u;
    if(_eon_f_is_nan(b) && !(b&0x00400000u)) return b|0x00400000u;
    return _eon_f_is_nan(a) ? a:b;
}
inline bool f_native_input(uint a) {
    uint magnitude=a & 0x7fffffffu;
    return magnitude==0u || (magnitude>=0x00800000u && magnitude<0x7f800000u);
}
inline bool f_native_result(uint a) {
    uint magnitude=a & 0x7fffffffu;
    return magnitude>=0x00800000u && magnitude<0x7f800000u;
}
inline uint f_add(uint a, uint b) {
    if(f_native_input(a) && f_native_input(b)) {
        uint result=as_type<uint>(as_type<float>(a)+as_type<float>(b));
        if(f_native_result(result)) return result;
    }
    if(_eon_f_is_nan(a)||_eon_f_is_nan(b)) return _eon_f_nan2(a,b);
    return f_from_d(d_add(f_to_d(a), f_to_d(b)));
}
inline uint f_sub(uint a, uint b) {
    if(f_native_input(a) && f_native_input(b)) {
        uint result=as_type<uint>(as_type<float>(a)-as_type<float>(b));
        if(f_native_result(result)) return result;
    }
    if(_eon_f_is_nan(a)||_eon_f_is_nan(b)) return _eon_f_nan2(a,b);
    return f_from_d(d_sub(f_to_d(a), f_to_d(b)));
}
inline uint f_mul(uint a, uint b) {
    if(f_native_input(a) && f_native_input(b)) {
        uint result=as_type<uint>(as_type<float>(a)*as_type<float>(b));
        if(f_native_result(result)) return result;
    }
    if(_eon_f_is_nan(a)||_eon_f_is_nan(b)) return _eon_f_nan2(a,b);
    return f_from_d(d_mul(f_to_d(a), f_to_d(b)));
}
inline uint f_div(uint a, uint b) { if(_eon_f_is_nan(a)||_eon_f_is_nan(b)) return _eon_f_nan2(a,b);
    return f_from_d(d_div(f_to_d(a), f_to_d(b))); }
inline uint f_sqrt(uint a) { return f_from_d(d_sqrt(f_to_d(a))); }
// A fused multiply/add retains a 106-bit product until the final rounding.
// Both precisions use the same integer accumulator; Metal's float denormal
// mode and its contraction heuristics cannot change the result.
struct clang_wide { ulong hi; ulong lo; };
inline clang_wide clang_right_jam(clang_wide a, int distance) {
    if(distance <= 0) return a;
    if(distance >= 128) return {0ul, (a.hi | a.lo) != 0ul};
    if(distance >= 64) {
        int n=distance-64;
        ulong sticky=a.lo != 0ul;
        if(n) sticky |= (a.hi << (64-n)) != 0ul;
        return {0ul, (a.hi >> n) | sticky};
    }
    return {a.hi >> distance,
        (a.lo >> distance) | (a.hi << (64-distance)) | ((a.lo << (64-distance)) != 0ul)};
}
inline ulong clang_fma(ulong a, ulong b, ulong c, bool single) {
    uint sa=uint(a>>63), sb=uint(b>>63), sc=uint(c>>63);
    int ea=int(d_exp(a)), eb=int(d_exp(b)), ec=int(d_exp(c));
    ulong ma=d_mant(a), mb=d_mant(b), mc=d_mant(c);
    if(ea==2047 || eb==2047 || ec==2047 || d_is_zero(a) || d_is_zero(b) || d_is_zero(c)) {
        ulong result=d_add(d_mul(a,b),c);
        return single ? ulong(f_from_d(result)) : result;
    }
    if(ea) ma|=1ul<<52; else {int z=int(clz(ma))-11; ma<<=z; ea=1-z;}
    if(eb) mb|=1ul<<52; else {int z=int(clz(mb))-11; mb<<=z; eb=1-z;}
    if(ec) mc|=1ul<<52; else {int z=int(clz(mc))-11; mc<<=z; ec=1-z;}
    ulong hi;
    ulong lo=d_umul128(ma,mb,&hi);
    int leading=(hi & (1ul<<41)) ? 105 : 104;
    int shift=126-leading;
    clang_wide product={(hi<<shift)|(lo>>(64-shift)),lo<<shift};
    clang_wide addend={mc<<10,0ul};
    int ep=ea+eb-2046+leading-104;
    int ef=ec-1023;
    int exponent=max(ep,ef);
    product=clang_right_jam(product,exponent-ep);
    addend=clang_right_jam(addend,exponent-ef);
    uint sign=sa^sb;
    clang_wide sum;
    if(sign==sc) {
        sum.lo=product.lo+addend.lo;
        sum.hi=product.hi+addend.hi+(sum.lo<product.lo);
    } else {
        if(product.hi<addend.hi || (product.hi==addend.hi && product.lo<addend.lo)) {
            clang_wide temporary=product; product=addend; addend=temporary; sign=sc;
        }
        sum.lo=product.lo-addend.lo;
        sum.hi=product.hi-addend.hi-(product.lo<addend.lo);
    }
    if(!(sum.hi|sum.lo)) return 0ul;
    leading=sum.hi ? 127-int(clz(sum.hi)) : 63-int(clz(sum.lo));
    int target=single ? 26 : 55;
    ulong significand;
    if(leading>target) significand=clang_right_jam(sum,leading-target).lo;
    else significand=sum.lo<<(target-leading);
    if(!single) return d_finish(sign,exponent+leading-126+1023,significand);
    int e=exponent+leading-126+127;
    if(e<=0) {
        int distance=1-e;
        significand=distance>=64 ? 1ul :
            (significand>>distance)|((significand<<(64-distance))!=0ul);
        e=0;
    }
    ulong remainder=significand&7ul;
    significand>>=3;
    if(remainder>4ul || (remainder==4ul && (significand&1ul))) ++significand;
    if(significand & (1ul<<24)) {significand>>=1; ++e;}
    if(e==0 && (significand & (1ul<<23))) e=1;
    if(e>=255) return ulong((sign<<31)|0x7f800000u);
    return ulong((sign<<31)|(uint(e)<<23)|(uint(significand)&0x7fffffu));
}
inline ulong clang_d_fma(ulong a, ulong b, ulong c) {return clang_fma(a,b,c,false);}
inline uint f_fma(uint a, uint b, uint c) {
    if(f_native_input(a) && f_native_input(b) && f_native_input(c)) {
        uint result=as_type<uint>(fma(as_type<float>(a),as_type<float>(b),as_type<float>(c)));
        if(f_native_result(result)) return result;
    }
    return uint(clang_fma(f_to_d(a),f_to_d(b),f_to_d(c),true));
}
inline uint f_neg(uint a) { return a ^ 0x80000000u; }
inline uint f_abs(uint a) { return a & 0x7fffffffu; }
inline bool f_gt(uint a, uint b) { return d_gt(f_to_d(a), f_to_d(b)); }
inline bool f_ge(uint a, uint b) { return d_ge(f_to_d(a), f_to_d(b)); }
inline bool f_lt(uint a, uint b) { return d_lt(f_to_d(a), f_to_d(b)); }
inline bool f_le(uint a, uint b) { return d_le(f_to_d(a), f_to_d(b)); }
inline bool f_eq(uint a, uint b) { return d_eq(f_to_d(a), f_to_d(b)); }
'''
