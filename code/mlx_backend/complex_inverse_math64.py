"""Native ports of the installed complex inverse operation graphs.

All inputs and outputs are raw binary64 carriers. The single-precision path
rounds every arithmetic operation at the same points as its scalar32 target.
"""

MSL_HEADER = r'''
inline ulong _eon_complex_real_asinh(ulong x) {
    if(d_is_nan(x))return d_add(x,x);
    ulong a=d_abs(x),result;
    if(d_lt(a,0x3e40000000000000ul)) {
        if(d_is_zero(x))return x;
        result=d_mul(d_add(d_mul(a,0x4360000000000000ul),0x8010000000000000ul),0x3c80000000000000ul);
    }else if(d_le(a,0x3ff5555555555555ul)) {
        ulong reciprocal=d_div(0x3ff0000000000000ul,a);
        ulong root=d_sqrt(d_fma(reciprocal,reciprocal,0x3ff0000000000000ul));
        result=_eon_stable_log1p(d_add(a,d_div(a,d_add(reciprocal,root))));
    }else if(d_le(a,0x41a0000000000000ul)) {
        ulong root=d_sqrt(d_fma(x,x,0x3ff0000000000000ul));
        result=d_logarithm(d_add(d_add(a,a),d_div(0x3ff0000000000000ul,d_add(a,root))));
    }else result=d_add(d_logarithm(a),0x3fe62e42fefa39eful);
    return d_lt(x,0ul)?d_neg(result):result;
}
inline ulong _eon_cinv_add(ulong a,ulong b,bool single) {return single?f_to_d(f_add(f_from_d(a),f_from_d(b))):d_add(a,b);}
inline ulong _eon_cinv_sub(ulong a,ulong b,bool single) {return single?f_to_d(f_sub(f_from_d(a),f_from_d(b))):d_sub(a,b);}
inline ulong _eon_cinv_mul(ulong a,ulong b,bool single) {return single?f_to_d(f_mul(f_from_d(a),f_from_d(b))):d_mul(a,b);}
inline ulong _eon_cinv_div(ulong a,ulong b,bool single) {return single?f_to_d(f_div(f_from_d(a),f_from_d(b))):d_div(a,b);}
inline ulong _eon_cinv_sqrt(ulong a,bool single) {return single?f_to_d(f_sqrt(f_from_d(a))):d_sqrt(a);}
inline ulong _eon_cinv_fma(ulong a,ulong b,ulong c,bool single) {return single?f_to_d(f_fma(f_from_d(a),f_from_d(b),f_from_d(c))):d_fma(a,b,c);}
inline ulong _eon_cinv_asinh(ulong a,bool single) {return single?f_to_d(_eon_complex_real_asinh32(f_from_d(a))):_eon_complex_real_asinh(a);}
inline ulong _eon_cinv_atan2(ulong y,ulong x,bool single) {return single?_eon_complex_atan232(y,x):_eon_complex_atan2(y,x);}
#define CADD(a,b) _eon_cinv_add(a,b,single)
#define CSUB(a,b) _eon_cinv_sub(a,b,single)
#define CMUL(a,b) _eon_cinv_mul(a,b,single)
#define CDIV(a,b) _eon_cinv_div(a,b,single)
#define CSQRT(a) _eon_cinv_sqrt(a,single)
#define CFMA(a,b,c) _eon_cinv_fma(a,b,c,single)
inline _eon_complex _eon_complex_asinh(_eon_complex z,bool single=false) {
    const ulong one=0x3ff0000000000000ul,half_value=0x3fe0000000000000ul;
    ulong x=d_abs(z.real),y=d_abs(z.imag),real,imag;
    ulong small=single?0x3f20000000000000ul:0x3e40000000000000ul;
    ulong large=single?0x40c0000000000000ul:0x41a0000000000000ul;
    if(x==D_INF||y==D_INF) {
        real=D_INF;imag=_eon_cinv_atan2(y,x,single);
    }else if(d_is_nan(x)||d_is_nan(y)) {
        ulong sum=CADD(x,y);
        real=d_is_zero(z.imag)?z.real:sum;imag=d_is_zero(z.imag)?z.imag:sum;
    }else if(d_lt(x,small)&&d_lt(y,small)) {
        real=x;imag=y;
    }else if(d_gt(x,large)||d_gt(y,large)) {
        _eon_complex logarithm=single?_eon_complex_log32({x,y}):_eon_complex_log({x,y});
        real=CADD(logarithm.real,single?f_to_d(0x3f317218u):0x3fe62e42fefa39eful);
        imag=logarithm.imag;
    }else {
        ulong square=d_lt(x,single?f_to_d(0x25800000u):0x3950000000000000ul)?0ul:CMUL(z.real,z.real);
        ulong difference=CSUB(one,y),a,b;
        if(d_is_zero(difference)) {
            a=CMUL(CSQRT(x),single?f_to_d(0x3f3504f3u):0x3fe6a09e667f3bcdul);b=a;
        }else {
            ulong root=CSQRT(CMUL(CADD(d_abs(difference),CSQRT(CFMA(difference,difference,square))),half_value));
            ulong other=CMUL(CDIV(x,root),half_value);
            a=d_lt(difference,0ul)?other:root;b=d_lt(difference,0ul)?root:other;
        }
        ulong plus=CADD(y,one);
        ulong root=CSQRT(CMUL(CADD(plus,CSQRT(CFMA(plus,plus,square))),half_value));
        ulong other=CDIV(x,CADD(root,root));
        real=_eon_cinv_asinh(CFMA(a,other,CMUL(root,b)),single);
        imag=_eon_cinv_atan2(y,CFMA(a,root,CMUL(other,b)),single);
    }
    return {_eon_complex_copysign(real,z.real),_eon_complex_copysign(imag,z.imag)};
}
inline _eon_complex _eon_complex_arcsin(_eon_complex z,bool single=false) {
    _eon_complex result=_eon_complex_asinh({d_neg(z.imag),z.real},single);
    return {result.imag,d_neg(result.real)};
}
inline _eon_complex _eon_complex_arcsin32(_eon_complex z) {return _eon_complex_arcsin(z,true);}
inline _eon_complex _eon_complex_arccos(_eon_complex z,bool single=false) {
    const ulong one=0x3ff0000000000000ul,half_value=0x3fe0000000000000ul;
    ulong x=d_abs(z.real),y=d_abs(z.imag),angle,imag;
    ulong pi=single?f_to_d(0x40490fdbu):0x400921fb54442d18ul;
    ulong pi2=single?f_to_d(0x3fc90fdbu):0x3ff921fb54442d18ul;
    ulong small=single?0x3f20000000000000ul:0x3e40000000000000ul;
    ulong large=single?0x40c0000000000000ul:0x41a0000000000000ul;
    if(x==D_INF||y==D_INF) {
        angle=_eon_cinv_atan2(y,x,single);imag=0xfff0000000000000ul;
    }else if(d_is_nan(x)||d_is_nan(y)) {
        imag=CADD(x,y);angle=d_is_zero(z.real)?pi2:imag;
    }else if(d_lt(x,small)&&d_lt(y,small)) {
        angle=CSUB(pi2,x);imag=d_neg(y);
    }else if(d_gt(x,large)||d_gt(y,large)) {
        _eon_complex logarithm=single?_eon_complex_log32({x,y}):_eon_complex_log({x,y});
        ulong length=CADD(logarithm.real,single?f_to_d(0x3f317218u):0x3fe62e42fefa39eful);
        angle=CADD(logarithm.imag,CMUL(length,0x8000000000000000ul));
        imag=CSUB(CMUL(logarithm.imag,0x8000000000000000ul),length);
    }else {
        ulong square=d_lt(y,single?f_to_d(0x25800000u):0x3950000000000000ul)?0ul:CMUL(z.imag,z.imag);
        ulong plus=CADD(x,one);
        ulong root=CSQRT(CMUL(CADD(plus,CSQRT(CFMA(plus,plus,square))),half_value));
        ulong other=CMUL(CDIV(y,root),half_value);
        ulong difference=CSUB(one,x),a,b;
        if(d_is_zero(difference)) {
            a=CMUL(CSQRT(y),single?f_to_d(0x3f3504f3u):0x3fe6a09e667f3bcdul);b=a;
        }else {
            ulong second=CSQRT(CMUL(CADD(d_abs(difference),CSQRT(CFMA(difference,difference,square))),half_value));
            ulong partner=CMUL(CDIV(y,second),half_value);
            a=d_lt(difference,0ul)?second:partner;b=d_lt(difference,0ul)?partner:second;
        }
        ulong atan=_eon_cinv_atan2(b,root,single);angle=CADD(atan,atan);
        imag=_eon_cinv_asinh(CFMA(d_neg(root),a,d_neg(CMUL(b,other))),single);
    }
    return {d_lt(z.real,0ul)?CSUB(pi,angle):angle,_eon_complex_copysign(imag,d_neg(z.imag))};
}
inline _eon_complex _eon_complex_arccos32(_eon_complex z) {return _eon_complex_arccos(z,true);}
inline _eon_complex _eon_complex_atanh(_eon_complex z,bool single=false) {
    const ulong one=0x3ff0000000000000ul,half_value=0x3fe0000000000000ul;
    ulong sign=_eon_complex_copysign(one,z.real),x=CMUL(z.real,sign),y=d_neg(CMUL(sign,z.imag));
    ulong absolute=d_abs(y),real,angle;
    ulong limit=single?f_to_d(0x5e7fffffu):0x5fcffffffffffffful;
    // ARM FCMP/B.LE sends unordered operands through the ordinary graph.
    // Negating an ordered <= comparison incorrectly chooses the large path.
    if(d_gt(x,limit)||d_gt(absolute,limit)) {
        angle=_eon_complex_copysign(single?f_to_d(0x3fc90fdbu):0x3ff921fb54442d18ul,y);
        ulong ratio=CDIV(x,y);
        ulong first=CDIV(ratio,CFMA(ratio,x,y));
        ratio=CDIV(y,x);
        ulong second=x==D_INF?0ul:CDIV(one,CFMA(y,ratio,x));
        real=d_lt(x,absolute)?first:second;
    }else {
        ulong adjusted=CADD(absolute,single?f_to_d(0x20800001u):0x2010000000000000ul);
        if(d_eq(x,one)) {
            ulong root=CDIV(CSQRT(CSQRT(CFMA(adjusted,adjusted,0x4010000000000000ul))),CSQRT(absolute));
            real=single?f_to_d(_eon_basic32_log(f_from_d(root))):d_logarithm(root);
            ulong ratio=CDIV(0x4000000000000000ul,adjusted);
            ulong remainder=d_sub(0x400921fb54442d18ul,_eon_arctan64(ratio));
            if(single)remainder=f_to_d(f_from_d(remainder));
            angle=CMUL(_eon_complex_copysign(remainder,y),half_value);
        }else {
            ulong difference=CSUB(one,x),square=CMUL(adjusted,adjusted);
            ulong quotient=CDIV(CMUL(x,0x4010000000000000ul),CFMA(difference,difference,square));
            ulong logarithm=single?f_to_d(_eon_basic32_log1p(f_from_d(quotient))):_eon_stable_log1p(quotient);
            real=CMUL(logarithm,0x3fd0000000000000ul);
            ulong denominator=CFMA(difference,CADD(x,one),d_neg(square));
            angle=CMUL(_eon_cinv_atan2(CADD(y,y),denominator,single),half_value);
        }
    }
    return {CMUL(sign,real),CMUL(angle,d_neg(sign))};
}
inline _eon_complex _eon_complex_arctan(_eon_complex z,bool single=false) {
    _eon_complex result=_eon_complex_atanh({d_neg(z.imag),z.real},single);
    return {result.imag,d_neg(result.real)};
}
inline _eon_complex _eon_complex_arctan32(_eon_complex z) {return _eon_complex_arctan(z,true);}
#undef CADD
#undef CSUB
#undef CMUL
#undef CDIV
#undef CSQRT
#undef CFMA
'''


def metal_complex_inverse_header():
    return MSL_HEADER
