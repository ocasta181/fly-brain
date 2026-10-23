"""Integer-backed complex temporaries with target real arithmetic helpers.

The installed NumPy calls Apple csinh/ccosh, whose paired sincos and expm1
operation graph differs from standalone real sinh/cosh. The retained symbol
inspection binds this graph to the same libsystem_m as the real helpers.
"""

MSL_BASE_HEADER = r'''
struct _eon_complex { ulong real; ulong imag; };
inline _eon_complex _eon_complex_round32(_eon_complex z) {
    return {f_to_d(f_from_d(z.real)),f_to_d(f_from_d(z.imag))};
}
inline bool _eon_complex_truth(_eon_complex z) {
    return !d_is_zero(z.real)||!d_is_zero(z.imag);
}
inline int _eon_complex_int32(ulong real,long position,long length,uint width) {
    // CDOUBLE_to_INT's vectorizer keeps its final 1..8 elements scalar,
    // including an entire final block when length is divisible by eight.
    long prefix=((length>0L?length-1L:0L)/8L)*8L;
    return width==64u&&position<prefix?int(d_to_int(real)):d_to_int32_sat(real);
}
inline ulong _eon_complex_integer_cast(ulong real,long position,long length,
                                      uint source_width,uint target_width,bool target_unsigned) {
    if(target_width==64u)return target_unsigned?d_to_uint_sat(real):ulong(d_to_int(real));
    long prefix8=((length>0L?length-1L:0L)/8L)*8L;
    if(source_width==64u&&position<prefix8)
        return target_unsigned&&target_width==32u?d_to_uint_sat(real):ulong(d_to_int(real));
    if(target_unsigned&&target_width==32u)return ulong(d_to_uint32_sat(real));
    long prefix16=((length>0L?length-1L:0L)/16L)*16L;
    if(source_width==32u&&target_unsigned&&target_width==16u&&position<prefix16)
        return ulong(d_to_uint32_sat(real));
    return ulong(d_to_int32_sat(real));
}
inline _eon_complex _eon_complex_neg(_eon_complex z) {
    return {d_neg(z.real),d_neg(z.imag)};
}
inline _eon_complex _eon_complex_add(_eon_complex a,_eon_complex b) {
    return {d_add(a.real,b.real),d_add(a.imag,b.imag)};
}
inline _eon_complex _eon_complex_sub(_eon_complex a,_eon_complex b) {
    return {d_sub(a.real,b.real),d_sub(a.imag,b.imag)};
}
inline _eon_complex _eon_complex_mul(_eon_complex a,_eon_complex b) {
    return {d_sub(d_mul(a.real,b.real),d_mul(a.imag,b.imag)),
            d_add(d_mul(a.real,b.imag),d_mul(a.imag,b.real))};
}
// NumPy's ARM ndarray loop rounds the cross product before FMLA. Keep
// scalar multiplication above separate: cpow and NumPy scalar ops use it.
inline _eon_complex _eon_complex_multiply_numpy(_eon_complex a,_eon_complex b) {
    return {d_fma(b.real,a.real,d_neg(d_mul(b.imag,a.imag))),
            d_fma(b.imag,a.real,d_mul(b.real,a.imag))};
}
inline _eon_complex _eon_complex_div(_eon_complex a,_eon_complex b) {
    ulong ar=d_abs(b.real),ai=d_abs(b.imag),ratio,scale;
    if(d_ge(ar,ai)) {
        if(d_is_zero(ar)&&d_is_zero(ai))return {d_div(a.real,ar),d_div(a.imag,ai)};
        ratio=d_div(b.imag,b.real);
        scale=d_div(0x3ff0000000000000ul,d_add(b.real,d_mul(b.imag,ratio)));
        return {d_mul(d_add(a.real,d_mul(a.imag,ratio)),scale),
                d_mul(d_sub(a.imag,d_mul(a.real,ratio)),scale)};
    }
    ratio=d_div(b.real,b.imag);
    scale=d_div(0x3ff0000000000000ul,d_add(b.imag,d_mul(b.real,ratio)));
    return {d_mul(d_add(d_mul(a.real,ratio),a.imag),scale),
            d_mul(d_sub(d_mul(a.imag,ratio),a.real),scale)};
}
inline _eon_complex _eon_complex_add32(_eon_complex a,_eon_complex b) {
    return {f_to_d(f_add(f_from_d(a.real),f_from_d(b.real))),
            f_to_d(f_add(f_from_d(a.imag),f_from_d(b.imag)))};
}
inline _eon_complex _eon_complex_sub32(_eon_complex a,_eon_complex b) {
    return {f_to_d(f_sub(f_from_d(a.real),f_from_d(b.real))),
            f_to_d(f_sub(f_from_d(a.imag),f_from_d(b.imag)))};
}
inline _eon_complex _eon_complex_mul32(_eon_complex a,_eon_complex b) {
    uint ar=f_from_d(a.real),ai=f_from_d(a.imag),br=f_from_d(b.real),bi=f_from_d(b.imag);
    return {f_to_d(f_sub(f_mul(ar,br),f_mul(ai,bi))),
            f_to_d(f_add(f_mul(ar,bi),f_mul(ai,br)))};
}
inline _eon_complex _eon_complex_multiply_numpy32(_eon_complex a,_eon_complex b) {
    uint ar=f_from_d(a.real),ai=f_from_d(a.imag),br=f_from_d(b.real),bi=f_from_d(b.imag);
    return {f_to_d(f_fma(br,ar,f_mul(bi,ai)^0x80000000u)),
            f_to_d(f_fma(bi,ar,f_mul(br,ai)))};
}
inline _eon_complex _eon_complex_div32(_eon_complex a,_eon_complex b) {
    uint ar=f_from_d(a.real),ai=f_from_d(a.imag),br=f_from_d(b.real),bi=f_from_d(b.imag);
    uint abr=br&0x7fffffffu,abi=bi&0x7fffffffu,ratio,scale;
    if(f_ge(abr,abi)) {
        if(abr==0u&&abi==0u)return {f_to_d(f_div(ar,abr)),f_to_d(f_div(ai,abi))};
        ratio=f_div(bi,br);scale=f_div(0x3f800000u,f_add(br,f_mul(bi,ratio)));
        return {f_to_d(f_mul(f_add(ar,f_mul(ai,ratio)),scale)),
                f_to_d(f_mul(f_sub(ai,f_mul(ar,ratio)),scale))};
    }
    ratio=f_div(br,bi);scale=f_div(0x3f800000u,f_add(bi,f_mul(br,ratio)));
    return {f_to_d(f_mul(f_add(f_mul(ar,ratio),ai),scale)),
            f_to_d(f_mul(f_sub(f_mul(ai,ratio),ar),scale))};
}
inline bool _eon_complex_eq(_eon_complex a,_eon_complex b) {
    return d_eq(a.real,b.real)&&d_eq(a.imag,b.imag);
}
inline bool _eon_complex_ne(_eon_complex a,_eon_complex b) {
    return !_eon_complex_eq(a,b);
}
inline bool _eon_complex_lt(_eon_complex a,_eon_complex b) {
    return !d_is_nan(a.imag)&&!d_is_nan(b.imag)&&
        (d_lt(a.real,b.real)||(d_eq(a.real,b.real)&&d_lt(a.imag,b.imag)));
}
inline bool _eon_complex_le(_eon_complex a,_eon_complex b) {
    return _eon_complex_lt(a,b)||_eon_complex_eq(a,b);
}
inline bool _eon_complex_gt(_eon_complex a,_eon_complex b) {return _eon_complex_lt(b,a);}
inline bool _eon_complex_ge(_eon_complex a,_eon_complex b) {return _eon_complex_le(b,a);}
inline _eon_complex _eon_complex_sign(_eon_complex z) {
    const _eon_complex zero={0ul,0ul};
    return {_eon_complex_gt(z,zero)?0x3ff0000000000000ul:
        _eon_complex_lt(z,zero)?0xbff0000000000000ul:
        _eon_complex_eq(z,zero)?0ul:D_QNAN,0ul};
}
inline _eon_complex _eon_complex_clip(_eon_complex value,_eon_complex low,_eon_complex high) {
    _eon_complex lower=(d_is_nan(value.real)||d_is_nan(value.imag)||_eon_complex_ge(value,low))?value:low;
    return (d_is_nan(lower.real)||d_is_nan(lower.imag)||_eon_complex_le(lower,high))?lower:high;
}
inline _eon_complex _eon_complex_powi(_eon_complex z,long n) {
    const _eon_complex one={0x3ff0000000000000ul,0ul};
    if(n==0)return one;
    if(!_eon_complex_truth(z))return n>0?_eon_complex{0ul,0ul}:_eon_complex{D_QNAN,D_QNAN};
    if(n==1)return z;
    if(n==2)return _eon_complex_mul(z,z);
    if(n==3)return _eon_complex_mul(z,_eon_complex_mul(z,z));
    bool negative=n<0;ulong magnitude=negative?0ul-ulong(n):ulong(n);
    _eon_complex result=one,power=z;
    while(magnitude){
        if(magnitude&1ul)result=_eon_complex_mul(result,power);
        magnitude>>=1;
        if(magnitude)power=_eon_complex_mul(power,power);
    }
    return negative?_eon_complex_div(one,result):result;
}
inline _eon_complex _eon_complex_powi32(_eon_complex z,long n) {
    const _eon_complex one={0x3ff0000000000000ul,0ul};
    if(n==0)return one;
    if(!_eon_complex_truth(z))return n>0?_eon_complex{0ul,0ul}:_eon_complex{D_QNAN,D_QNAN};
    if(n==1)return z;
    if(n==2)return _eon_complex_mul32(z,z);
    if(n==3)return _eon_complex_mul32(z,_eon_complex_mul32(z,z));
    bool negative=n<0;ulong magnitude=negative?0ul-ulong(n):ulong(n);
    _eon_complex result=one,power=z;
    while(magnitude){
        if(magnitude&1ul)result=_eon_complex_mul32(result,power);
        magnitude>>=1;
        if(magnitude)power=_eon_complex_mul32(power,power);
    }
    return negative?_eon_complex_div32(one,result):result;
}
'''

MSL_TRIG_HEADER = r'''
inline ulong _eon_complex_copysign(ulong value,ulong sign) {
    return (value&0x7ffffffffffffffful)|(sign&0x8000000000000000ul);
}
inline _eon_complex _eon_complex_hyper(_eon_complex z,bool cosine,bool single=false) {
    ulong x=z.real,y=z.imag,absolute=d_abs(x);
    if(d_is_nan(x))return {x,d_is_zero(y)?y:_eon_complex_copysign(x,y)};
    _eon_trig64_pair pair=_eon_trig64_sincos_scalar(y);
    ulong sine=pair.sine,cos=pair.cosine;
    ulong signed_sine=d_mul(sine,_eon_complex_copysign(0x3ff0000000000000ul,x));
    if(absolute==D_INF && (d_is_nan(y)||y==D_INF||d_is_zero(y)))
        return {cosine?D_INF:x,cosine?signed_sine:sine};
    if(d_is_zero(x))return cosine?_eon_complex{cos,d_mul(x,_eon_complex_copysign(0x3ff0000000000000ul,y))}:_eon_complex{x,sine};
    ulong input=absolute;
    ulong first=cosine?cos:d_mul(cos,_eon_complex_copysign(0x3ff0000000000000ul,x));
    ulong second=cosine?signed_sine:sine;
    if(!single && absolute<0x40962e42fefa39eful && d_gt(absolute,0x40862e42fefa39eful)) {
        input=d_add(absolute,0xc0862e42fefa39eful);
        first=d_mul(first,0x7fefffffffffff2aul);
        second=d_mul(second,0x7fefffffffffff2aul);
    }
    ulong t=_eon_stable_expm1(input);
    if(absolute<0x3e40000000000000ul)
        return cosine?_eon_complex{first,d_mul(absolute,second)}:_eon_complex{d_mul(first,absolute),second};
    ulong denominator=d_add(t,0x3ff0000000000000ul);
    if(d_gt(absolute,0x4033000000000000ul)) {
        ulong magnitude=d_mul(denominator,0x3fe0000000000000ul);
        return {d_mul(first,magnitude),d_is_zero(y)?second:d_mul(second,magnitude)};
    }
    denominator=d_add(denominator,denominator);
    ulong cosh_factor=d_add(d_div(d_mul(t,t),denominator),0x3ff0000000000000ul);
    ulong sinh_factor=d_fma(t,0x3fe0000000000000ul,d_div(t,denominator));
    return cosine?_eon_complex{d_mul(first,cosh_factor),d_mul(second,sinh_factor)}:
        _eon_complex{d_mul(first,sinh_factor),d_mul(second,cosh_factor)};
}
inline _eon_complex _eon_complex_sinh(_eon_complex z) {return _eon_complex_hyper(z,false);}
inline _eon_complex _eon_complex_cosh(_eon_complex z) {return _eon_complex_hyper(z,true);}
inline _eon_complex _eon_complex_sin(_eon_complex z) {
    _eon_complex rotated=_eon_complex_sinh({d_neg(z.imag),z.real});
    return {rotated.imag,d_neg(rotated.real)};
}
inline _eon_complex _eon_complex_cos(_eon_complex z) {
    return _eon_complex_cosh({d_neg(z.imag),z.real});
}
inline _eon_complex _eon_complex_sinh32(_eon_complex z) {return _eon_complex_hyper(z,false,true);}
inline _eon_complex _eon_complex_cosh32(_eon_complex z) {return _eon_complex_hyper(z,true,true);}
inline _eon_complex _eon_complex_sin32(_eon_complex z) {
    _eon_complex rotated=_eon_complex_sinh32({d_neg(z.imag),z.real});
    return {rotated.imag,d_neg(rotated.real)};
}
inline _eon_complex _eon_complex_cos32(_eon_complex z) {
    return _eon_complex_cosh32({d_neg(z.imag),z.real});
}
'''

MSL_BASIC_HEADER = r'''
// NumPy1.26.4 loops_unary_complex.dispatch.c.src:29-63. The expression
// ufunc allocates a separate real result and takes its ASIMD path even for
// a partial vector; this differs from the system hypot used by scalar cabs.
inline ulong _eon_complex_abs_numpy(_eon_complex z) {
    ulong a=d_abs(z.real),b=d_abs(z.imag);
    if(a==D_INF||b==D_INF)return D_INF;
    if(d_is_nan(a)||d_is_nan(b))return D_QNAN;
    ulong high=d_gt(a,b)?a:b,low=d_gt(a,b)?b:a;
    ulong ratio=d_is_zero(high)?0ul:d_div(low,high);
    return d_mul(d_sqrt(d_fma(ratio,ratio,0x3ff0000000000000ul)),high);
}
inline ulong _eon_complex_abs_numpy32(_eon_complex z) {
    uint a=f_from_d(z.real)&0x7fffffffu,b=f_from_d(z.imag)&0x7fffffffu;
    if(a==0x7f800000u||b==0x7f800000u)return D_INF;
    if(a>0x7f800000u||b>0x7f800000u)return D_QNAN;
    uint high=f_gt(a,b)?a:b,low=f_gt(a,b)?b:a;
    uint ratio=high==0u?0u:f_div(low,high);
    return f_to_d(f_mul(f_sqrt(f_fma(ratio,ratio,0x3f800000u)),high));
}
inline ulong _eon_complex_hypot(ulong a,ulong b) {
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
    if(d_is_nan(low))return low;
    if(exponent-int(d_exp(low))>27)low=0ul;
    ulong scale=0x3ff0000000000000ul;
    if(exponent>=0x5fe){
        high=d_mul(high,0x0ff0000000000000ul);low=d_mul(low,0x0ff0000000000000ul);
        scale=0x6ff0000000000000ul;
    }else if(exponent<0x21b){
        high=d_mul(high,0x6ff0000000000000ul);low=d_mul(low,0x6ff0000000000000ul);
        scale=0x0ff0000000000000ul;
    }
    return d_mul(d_sqrt(d_fma(low,low,d_mul(high,high))),scale);
}
inline _eon_complex _eon_complex_sqrt(_eon_complex z) {
    ulong x=z.real,y=z.imag,root,other,scale=0x3ff0000000000000ul;
    int ex=int(d_exp(x)),ey=int(d_exp(y));
    if(d_abs(y)==D_INF)return {D_INF,y};
    if(d_is_nan(x)||d_is_nan(y)) {
        if(d_abs(x)==D_INF)return (x>>63)?_eon_complex{d_abs(y),_eon_complex_copysign(D_INF,y)}:
            _eon_complex{D_INF,_eon_complex_copysign(0ul,y)};
        ulong value=d_add(x,y);return {value,value};
    }
    if(d_abs(x)==D_INF)return (x>>63)?_eon_complex{0ul,_eon_complex_copysign(D_INF,y)}:
        _eon_complex{D_INF,_eon_complex_copysign(0ul,y)};
    if(d_is_zero(x)&&d_is_zero(y))return {0ul,y};
    if(ex<0x200||ex>0x5fe||ey<0x200||ey>0x5fe) {
        if(ex-ey>54)y=_eon_complex_copysign(0ul,y);
        if(ey-ex>54)x=_eon_complex_copysign(0ul,x);
        int maximum=ex>ey?ex:ey;
        if(maximum>0x5fd){
            x=d_mul(x,0x0ff0000000000000ul);y=d_mul(y,0x0ff0000000000000ul);
            scale=0x57f0000000000000ul;
        }else if(maximum<=0x235){
            x=d_mul(x,0x6ff0000000000000ul);y=d_mul(y,0x6ff0000000000000ul);
            scale=0x27f0000000000000ul;
        }
    }
    root=d_sqrt(d_mul(d_add(d_abs(x),d_sqrt(d_fma(x,x,d_mul(y,y)))),0x3fe0000000000000ul));
    other=d_div(d_mul(y,0x3fe0000000000000ul),root);
    root=d_mul(root,scale);other=d_mul(other,scale);
    return (x>>63)?_eon_complex{d_abs(other),_eon_complex_copysign(root,z.imag)}:_eon_complex{root,other};
}
inline _eon_complex _eon_complex_sqrt32(_eon_complex z) {
    uint x=f_from_d(z.real),y=f_from_d(z.imag),root,other;
    uint ax=x&0x7fffffffu,ay=y&0x7fffffffu;
    if(ay==0x7f800000u)return {D_INF,z.imag};
    if(ax>0x7f800000u||ay>0x7f800000u){
        if(ax==0x7f800000u)return (x>>31)?_eon_complex{d_abs(z.imag),_eon_complex_copysign(D_INF,z.imag)}:
            _eon_complex{D_INF,_eon_complex_copysign(0ul,z.imag)};
        ulong value=f_to_d(f_add(x,y));return {value,value};
    }
    if(ax==0x7f800000u)return (x>>31)?_eon_complex{0ul,_eon_complex_copysign(D_INF,z.imag)}:
        _eon_complex{D_INF,_eon_complex_copysign(0ul,z.imag)};
    if(ax==0u&&ay==0u)return {0ul,z.imag};
    uint ex=(ax>>23),ey=(ay>>23);
    if(ex>=0x40u&&ex<=0xbeu&&ey>=0x40u&&ey<=0xbeu)
        root=f_sqrt(f_mul(f_add(ax,f_sqrt(f_fma(x,x,f_mul(y,y)))),0x3f000000u));
    else root=f_from_d(d_sqrt(d_mul(d_add(d_abs(z.real),d_sqrt(d_fma(z.real,z.real,d_mul(z.imag,z.imag)))),0x3fe0000000000000ul)));
    other=f_div(f_mul(y,0x3f000000u),root);
    return (x>>31)?_eon_complex{f_to_d(other&0x7fffffffu),_eon_complex_copysign(f_to_d(root),z.imag)}:
        _eon_complex{f_to_d(root),f_to_d(other)};
}
inline _eon_complex _eon_complex_exp(_eon_complex z,bool single=false) {
    ulong x=z.real,y=z.imag;
    if(d_is_nan(x))return {x,d_is_zero(y)?y:_eon_complex_copysign(x,y)};
    if(x==0xfff0000000000000ul&&d_exp(y)==2047ul)
        return {0ul,_eon_complex_copysign(0ul,y)};
    if(d_is_zero(y))return {single?f_to_d(_eon_basic32_exp(f_from_d(x))):d_exponential(x),0ul};
    ulong sine=d_sine(y);
    if(x==D_INF&&(y==D_INF||d_is_nan(y)))return {D_INF,sine};
    ulong cosine=d_cosine(y);
    if(!single && d_lt(x,0x40962e42fefa39eful)&&d_gt(x,0x40862e42fefa39eful)) {
        x=d_add(x,0xc0862e42fefa39eful);
        sine=d_mul(sine,0x7fefffffffffff2aul);cosine=d_mul(cosine,0x7fefffffffffff2aul);
    }
    ulong magnitude=d_exponential(x);
    return {d_mul(cosine,magnitude),d_mul(sine,magnitude)};
}
inline _eon_complex _eon_complex_exp32(_eon_complex z) {return _eon_complex_exp(z,true);}
'''

MSL_EXTENDED_HEADER = r'''
// NumPy's nc_expm1 loops preserve each scalar operation; this is not cexp-1.
inline _eon_complex _eon_complex_expm1(_eon_complex z) {
    ulong half_sine=d_sine(d_mul(z.imag,0x3fe0000000000000ul));
    ulong real=d_sub(d_mul(_eon_stable_expm1(z.real),d_cosine(z.imag)),
                     d_mul(half_sine,d_add(half_sine,half_sine)));
    return {real,d_mul(d_exponential(z.real),d_sine(z.imag))};
}
inline _eon_complex _eon_complex_expm132(_eon_complex z) {
    uint x=f_from_d(z.real),y=f_from_d(z.imag);
    uint half_sine=_eon_trig32_sine_scalar(f_mul(y,0x3f000000u));
    uint real=f_sub(f_mul(_eon_basic32_expm1(x),_eon_trig32_cosine_scalar(y)),
                    f_mul(half_sine,f_add(half_sine,half_sine)));
    return {f_to_d(real),f_to_d(f_mul(_eon_basic32_exp(x),_eon_trig32_sine_scalar(y)))};
}
// Actual system ctanh/ctanhf use different scalar widths before a common
// compensated double graph. Keep the float scalar calls and their boundaries.
inline _eon_complex _eon_complex_tanh(_eon_complex z,bool single=false) {
    ulong x=z.real,y=z.imag,absolute=d_abs(x),ay=d_abs(y),real,imag;
    if(absolute==D_INF || (!d_is_nan(absolute)&&d_gt(absolute,0x4033000000000000ul))) {
        ulong twice=single?f_to_d(f_add(f_from_d(ay),f_from_d(ay))):d_add(ay,ay);
        ulong sine=single?f_to_d(_eon_trig32_sine_scalar(f_from_d(twice))):d_sine(twice);
        real=0x3ff0000000000000ul;imag=_eon_complex_copysign(0ul,sine);
    }else if(d_is_nan(x)||d_is_nan(y)) {
        ulong total=d_add(x,y);
        real=d_is_zero(y)?x:total;imag=d_is_zero(y)?0ul:total;
    }else if(ay==D_INF) {
        real=D_QNAN;imag=D_QNAN;
    }else {
        ulong tangent=single?f_to_d(_eon_trig32_tangent_scalar(f_from_d(ay))):d_tangent(ay);
        ulong beta=d_fma(tangent,tangent,0x3ff0000000000000ul);
        ulong sinh_mag=absolute,cosh_factor=0x3ff0000000000000ul;
        if(d_ge(absolute,single?0x3f20000000000000ul:0x3e40000000000000ul)) {
            ulong t=single?f_to_d(_eon_basic32_expm1(f_from_d(absolute))):_eon_stable_expm1(absolute);
            ulong denominator=d_add(t,0x3ff0000000000000ul);
            cosh_factor=d_add(d_div(d_mul(d_mul(t,t),0x3fe0000000000000ul),denominator),0x3ff0000000000000ul);
            sinh_mag=d_mul(d_add(t,d_div(t,denominator)),0x3fe0000000000000ul);
        }
        ulong denominator=d_fma(d_mul(beta,sinh_mag),sinh_mag,0x3ff0000000000000ul);
        real=d_div(d_mul(d_mul(beta,cosh_factor),sinh_mag),denominator);
        imag=d_div(tangent,denominator);
        if(single){real=f_to_d(f_from_d(real));imag=f_to_d(f_from_d(imag));}
    }
    real=_eon_complex_copysign(real,x);
    imag=single?f_to_d(f_mul(f_from_d(imag),f_from_d(_eon_complex_copysign(0x3ff0000000000000ul,y)))):
        d_mul(imag,_eon_complex_copysign(0x3ff0000000000000ul,y));
    return {real,imag};
}
inline _eon_complex _eon_complex_tanh32(_eon_complex z) {return _eon_complex_tanh(z,true);}
inline _eon_complex _eon_complex_tan(_eon_complex z) {
    _eon_complex value=_eon_complex_tanh({d_neg(z.imag),z.real});
    return {value.imag,d_neg(value.real)};
}
inline _eon_complex _eon_complex_tan32(_eon_complex z) {
    _eon_complex value=_eon_complex_tanh32({d_neg(z.imag),z.real});
    return {value.imag,d_neg(value.real)};
}
'''


def metal_complex_header(used_math):
    if '_complex' not in used_math:
        return ''
    return (MSL_BASE_HEADER + (MSL_TRIG_HEADER if '_complex_trig' in used_math else '')
            + (MSL_BASIC_HEADER if '_complex_basic' in used_math else '')
            + (MSL_EXTENDED_HEADER if '_complex_extended' in used_math else ''))
