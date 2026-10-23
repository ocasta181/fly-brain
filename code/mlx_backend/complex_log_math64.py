"""Complex logarithms with the pinned Apple/NumPy operation sequences.

Include the binary64, binary32, math64, real log/log1p, basic_math32 and
complex base headers first. Inputs and outputs carry binary64 bits; the32
entry points explicitly retain each binary32 rounding boundary. All numerical
operations execute on Metal. The atan2 polynomial follows the installed
libsystem_m target rather than replacing atan2 with atan(y/x).
"""

__all__ = ["metal_complex_log_header"]

_HEADER = r'''
inline ulong _eon_complex_atan2(ulong y,ulong x) {
    if(d_is_nan(y)||d_is_nan(x))return d_add(y,x);
    ulong one=0x3ff0000000000000ul,half_value=0x3fe0000000000000ul;
    ulong sign=(y>>63)?0xbff0000000000000ul:one;
    ulong hi=y&0x8000000000000000ul,lo=hi;
    ulong numerator=d_abs(y),denominator=x;
    if(x>>63) {
        hi=d_fma(sign,0x400921fb54440000ul,hi);
        lo=d_fma(sign,0x3d968c234c4c6629ul,lo);
        sign=d_neg(sign);denominator=d_abs(x);
    }
    if(d_eq(denominator,numerator)) {
        if(!d_is_zero(numerator)) {
            hi=d_fma(sign,0x3fe921fb54440000ul,hi);
            lo=d_fma(sign,0x3d768c234c4c6629ul,lo);
        }
        return d_add(lo,hi);
    }
    if(d_lt(denominator,numerator)) {
        hi=d_fma(sign,0x3ff921fb54440000ul,hi);
        lo=d_fma(sign,0x3d868c234c4c6629ul,lo);
        sign=d_neg(sign);
        ulong swap=denominator;denominator=numerator;numerator=swap;
    }
    int exponent=int(d_exp(denominator));
    if(exponent<=0x1ff) {
        denominator=d_mul(denominator,0x4ff0000000000000ul);
        numerator=d_mul(numerator,0x4ff0000000000000ul);
    } else if(exponent>=0x5ff && d_exp(numerator)>=0x5fbul) {
        denominator=d_mul(denominator,0x2ff0000000000000ul);
        numerator=d_mul(numerator,0x2ff0000000000000ul);
    }
    ulong denominator_low=0ul;
    if(d_ge(numerator,d_mul(denominator,half_value))) {
        ulong total=d_fma(numerator,half_value,denominator);
        denominator_low=d_fma(numerator,half_value,d_sub(denominator,total));
        numerator=d_fma(denominator,d_neg(half_value),numerator);
        denominator_low=d_add(denominator_low,0ul);
        hi=d_fma(sign,0x3fddac6705600000ul,hi);
        lo=d_fma(sign,0x3d9bb4f68adfc88cul,lo);
        denominator=total;
    }
    if(d_ge(numerator,d_mul(denominator,0x3fd0000000000000ul))) {
        ulong total=d_fma(numerator,0x3fd0000000000000ul,denominator);
        ulong tail=d_fma(numerator,0x3fd0000000000000ul,d_sub(denominator,total));
        numerator=d_fma(denominator,0xbfd0000000000000ul,numerator);
        denominator_low=d_add(denominator_low,tail);
        hi=d_fma(sign,0x3fcf5b75f92c0000ul,hi);
        lo=d_fma(sign,0x3d701bac55b71e7cul,lo);
        denominator=total;
    }
    if(d_ge(numerator,d_mul(denominator,0x3fc0000000000000ul))) {
        ulong total=d_fma(numerator,0x3fc0000000000000ul,denominator);
        ulong tail=d_fma(numerator,0x3fc0000000000000ul,d_sub(denominator,total));
        numerator=d_fma(denominator,0xbfc0000000000000ul,numerator);
        denominator_low=d_add(denominator_low,tail);
        hi=d_fma(sign,0x3fbfd5ba9aac0000ul,hi);
        lo=d_fma(sign,0x3d47b6e32c89798aul,lo);
        denominator=total;
    }
    int gap=int(d_exp(denominator))-int(d_exp(numerator));
    if(uint(gap)>30u) {
        if(uint(gap)>=61u&&!d_is_zero(hi))return d_add(lo,hi);
        ulong ratio=d_div(d_mul(sign,numerator),denominator);
        ratio=d_mul(ratio,0x47f0000000000000ul);
        ulong tail=d_fma(ratio,0xbbf0000000000000ul,ratio);
        tail=d_fma(tail,0x37f0000000000000ul,lo);
        if(!d_is_zero(ratio))lo=tail;
        return d_add(lo,hi);
    }
    numerator=d_mul(sign,numerator);
    ulong ratio=d_div(numerator,denominator),square=d_mul(ratio,ratio);
    ulong error=d_fma(d_neg(ratio),denominator,numerator);
    error=d_fma(d_neg(denominator_low),ratio,error);
    error=d_div(error,denominator);
    ulong cubic=d_mul(square,0xbfd555555555548ful);
    ulong linear=d_add(cubic,one);
    cubic=d_add(cubic,d_sub(one,linear));
    cubic=d_add(cubic,0xbc4027f9f7e28f03ul);
    ulong product=d_mul(ratio,linear);
    ulong tail=d_fma(ratio,linear,d_neg(product));
    tail=d_fma(ratio,cubic,tail);
    error=d_fma(error,linear,tail);
    ratio=d_mul(ratio,0x3fb2c93f48b79173ul);
    ulong p=d_fma(d_add(square,0xc000bd56e27c9fd4ul),square,0x3ffaf261ae1512b6ul);
    ulong q=d_fma(d_add(square,0x3feb594d97787601ul),square,0x3ff9e459e5bf3b17ul);
    p=d_mul(p,square);q=d_mul(q,square);
    ratio=d_mul(d_mul(ratio,q),p);
    ulong sum=d_add(hi,product);
    ulong roundoff=d_add(product,d_sub(hi,sum));
    lo=d_add(lo,ratio);error=d_add(lo,error);
    return d_add(sum,d_add(roundoff,error));
}

// atan2f uses a separate widened polynomial, including a biased endpoint.
inline ulong _eon_complex_atan32_small(ulong ratio) {
    ulong z=d_mul(ratio,ratio);
    ulong p=d_fma(d_add(z,0x4002609372076e2cul),z,0x40038f4c070b63c1ul);
    ulong q=d_fma(d_add(z,0xc00792d5bf17c0bcul),z,0x4015e43168d7c2b6ul);
    ulong r=d_fma(d_add(z,0x3f953db49b9fcf6bul),z,0x400e4fa479729042ul);
    ulong s=d_fma(d_add(z,0xc013e370146726b3ul),z,0x401ae0fb08f2dc00ul);
    p=d_mul(p,r);q=d_mul(q,s);
    p=d_mul(p,0x3f680bc0fe23203dul);q=d_mul(q,ratio);
    return d_mul(p,q);
}
inline ulong _eon_complex_atan2_32(ulong y,ulong x) {
    ulong value;
    if(d_is_nan(y)||d_is_nan(x))return f_to_d(f_add(f_from_d(y),f_from_d(x)));
    if(d_lt(y,x)) {
        if(d_ge(d_neg(x),y)) {
            value=d_eq(d_neg(x),y)?0xbfe921fb54442d18ul:
                d_sub(0xbff921fb54442d18ul,_eon_complex_atan32_small(d_div(x,y)));
        }else value=_eon_complex_atan32_small(d_div(y,x));
    }else if(d_eq(y,x)) {
        if(d_lt(y,0ul))value=0xc002d97c7f3321d2ul;
        else if(d_gt(y,0ul))value=0x3fe921fb54442d18ul;
        else if(!(x>>63))return y;
        else value=(y&0x8000000000000000ul)|0x400921fb40080000ul;
    }else if(d_ge(d_neg(x),y)) {
        if(d_eq(d_neg(x),y))value=0x4002d97c7f3321d2ul;
        else {
            ulong ratio=d_div(y,x);
            if(d_lt(d_abs(ratio),0x3e90000000000000ul))
                value=(y&0x8000000000000000ul)|0x400921fb40080000ul;
            else value=d_add(_eon_complex_atan32_small(ratio),
                (y&0x8000000000000000ul)|0x400921fb54442d18ul);
        }
    }else value=d_sub(0x3ff921fb54442d18ul,_eon_complex_atan32_small(d_div(x,y)));
    return f_to_d(f_from_d(value));
}
inline ulong _eon_complex_atan232(ulong y,ulong x) {
    return _eon_complex_atan2_32(y,x);
}

inline ulong _eon_complex_log_hypot(ulong a,ulong b) {
    a=d_abs(a);b=d_abs(b);
    if((d_is_nan(a)&&!(a&0x0008000000000000ul))||
       (d_is_nan(b)&&!(b&0x0008000000000000ul)))return d_add(a,b);
    ulong high,low;
    if(d_is_nan(a)){high=d_is_nan(b)?a:b;low=a;}
    else if(d_is_nan(b)){high=a;low=b;}
    else {high=d_gt(a,b)?a:b;low=d_gt(a,b)?b:a;}
    int exponent=int(d_exp(high));
    if(exponent==2047)return d_add(high,high);
    if(d_is_zero(low))return high;
    if(d_is_nan(low))return d_add(low,low);
    if(exponent-int(d_exp(low))>27)low=0ul;
    ulong scale=0x3ff0000000000000ul;
    if(exponent>=0x5fe) {
        high=d_mul(high,0x0ff0000000000000ul);low=d_mul(low,0x0ff0000000000000ul);
        scale=0x6ff0000000000000ul;
    }else if(exponent<0x21b) {
        high=d_mul(high,0x6ff0000000000000ul);low=d_mul(low,0x6ff0000000000000ul);
        scale=0x0ff0000000000000ul;
    }
    return d_mul(d_sqrt(d_fma(low,low,d_mul(high,high))),scale);
}
inline ulong _eon_complex_log_hypot32(ulong a,ulong b) {
    if(d_abs(a)>=D_INF) {
        if(d_abs(a)==D_INF||d_abs(b)==D_INF)return D_INF;
        return f_to_d(f_mul(f_from_d(a),0x3f800000u));
    }
    return f_to_d(f_from_d(d_sqrt(d_add(d_mul(a,a),d_mul(b,b)))));
}
inline _eon_complex _eon_complex_log(_eon_complex z) {
    ulong angle=_eon_complex_atan2(z.imag,z.real);
    ulong a=d_abs(z.real),b=d_abs(z.imag);
    if(a==D_INF||b==D_INF)return {D_INF,angle};
    if(d_is_nan(z.real))return {z.real,(z.real&0x7ffffffffffffffful)|(z.imag&0x8000000000000000ul)};
    if(d_is_nan(z.imag))return {z.imag,z.imag};
    ulong high=d_lt(a,b)?b:a,low=d_lt(a,b)?a:b;
    ulong value=d_logarithm(high);
    if(!d_is_zero(high)&&(high==0x3ff0000000000000ul||int(d_exp(high))-int(d_exp(low))<53)) {
        ulong ratio=d_div(low,high);
        value=d_fma(_eon_stable_log1p(d_mul(ratio,ratio)),0x3fe0000000000000ul,value);
    }
    return {value,angle};
}
inline _eon_complex _eon_complex_log32(_eon_complex z) {
    ulong angle=_eon_complex_atan2_32(z.imag,z.real);
    ulong a=d_abs(z.real),b=d_abs(z.imag);
    if(a==D_INF||b==D_INF)return {D_INF,angle};
    if(d_is_nan(z.real))return {z.real,(z.real&0x7ffffffffffffffful)|(z.imag&0x8000000000000000ul)};
    if(d_is_nan(z.imag))return {z.imag,z.imag};
    ulong high=d_lt(a,b)?b:a,low=d_lt(a,b)?a:b;
    uint value=_eon_basic32_log(f_from_d(high));
    if(!d_is_zero(high)&&(high==0x3ff0000000000000ul||int(d_exp(high))-int(d_exp(low))<24)) {
        uint ratio=f_div(f_from_d(low),f_from_d(high));
        value=f_fma(_eon_basic32_log1p(f_mul(ratio,ratio)),0x3f000000u,value);
    }
    return {f_to_d(value),angle};
}
inline _eon_complex _eon_complex_log1p(_eon_complex z) {
    ulong x=d_add(z.real,0x3ff0000000000000ul);
    return {d_logarithm(_eon_complex_log_hypot(x,z.imag)),_eon_complex_atan2(z.imag,x)};
}
inline _eon_complex _eon_complex_log1p32(_eon_complex z) {
    ulong x=f_to_d(f_add(f_from_d(z.real),0x3f800000u));
    return {f_to_d(_eon_basic32_log(f_from_d(_eon_complex_log_hypot32(x,z.imag)))),
        _eon_complex_atan2_32(z.imag,x)};
}
inline _eon_complex _eon_complex_log10(_eon_complex z) {
    z=_eon_complex_log(z);
    return {d_mul(z.real,0x3fdbcb7b1526e50eul),d_mul(z.imag,0x3fdbcb7b1526e50eul)};
}
inline _eon_complex _eon_complex_log1032(_eon_complex z) {
    z=_eon_complex_log32(z);
    return {f_to_d(f_mul(f_from_d(z.real),0x3ede5bd9u)),
        f_to_d(f_mul(f_from_d(z.imag),0x3ede5bd9u))};
}
'''


def metal_complex_log_header():
    """Return the raw-bit logarithm family and actual-target atan2 helpers."""
    return _HEADER
