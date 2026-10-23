"""NumPy power dispatch and the installed system cpow exceptional product.

The ordinary product retains cpow's four scalar multiplies.  When both
components are NaN, cpow recomputes with __muldc3/__mulsc3 before the shared
cexp tail call.  Recovery has a different imaginary addend order and fused
operation graph.
"""

MSL_HEADER = r'''
inline ulong _eon_complex_power_signed(ulong magnitude,ulong sign) {
    return (magnitude&0x7ffffffffffffffful)|(sign&0x8000000000000000ul);
}
inline _eon_complex _eon_complex_power_recover(_eon_complex a,_eon_complex b,bool single=false) {
    ulong ar=a.real,ai=a.imag,br=b.real,bi=b.imag;
    ulong ac,bd,ad,bc,real,imag;
    if(single) {
        ac=f_to_d(f_mul(f_from_d(ar),f_from_d(br)));
        bd=f_to_d(f_mul(f_from_d(ai),f_from_d(bi)));
        ad=f_to_d(f_mul(f_from_d(ar),f_from_d(bi)));
        bc=f_to_d(f_mul(f_from_d(ai),f_from_d(br)));
        real=f_to_d(f_sub(f_from_d(ac),f_from_d(bd)));
        imag=f_to_d(f_add(f_from_d(bc),f_from_d(ad)));
    } else {
        ac=d_mul(ar,br);bd=d_mul(ai,bi);ad=d_mul(ar,bi);bc=d_mul(ai,br);
        real=d_sub(ac,bd);imag=d_add(bc,ad);
    }
    if(!d_is_nan(real)||!d_is_nan(imag))return {real,imag};
    bool recalculate=false;
    if(d_abs(ar)==D_INF||d_abs(ai)==D_INF) {
        ar=_eon_complex_power_signed(d_abs(ar)==D_INF?0x3ff0000000000000ul:0ul,ar);
        ai=_eon_complex_power_signed(d_abs(ai)==D_INF?0x3ff0000000000000ul:0ul,ai);
        if(d_is_nan(br))br=_eon_complex_power_signed(0ul,br);
        if(d_is_nan(bi))bi=_eon_complex_power_signed(0ul,bi);
        recalculate=true;
    }
    if(d_abs(br)==D_INF||d_abs(bi)==D_INF) {
        br=_eon_complex_power_signed(d_abs(br)==D_INF?0x3ff0000000000000ul:0ul,br);
        bi=_eon_complex_power_signed(d_abs(bi)==D_INF?0x3ff0000000000000ul:0ul,bi);
        if(d_is_nan(ar))ar=_eon_complex_power_signed(0ul,ar);
        if(d_is_nan(ai))ai=_eon_complex_power_signed(0ul,ai);
        recalculate=true;
    }
    if(!recalculate&&(d_abs(ac)==D_INF||d_abs(bd)==D_INF||d_abs(ad)==D_INF||d_abs(bc)==D_INF)) {
        if(d_is_nan(ar))ar=_eon_complex_power_signed(0ul,ar);
        if(d_is_nan(ai))ai=_eon_complex_power_signed(0ul,ai);
        if(d_is_nan(br))br=_eon_complex_power_signed(0ul,br);
        if(d_is_nan(bi))bi=_eon_complex_power_signed(0ul,bi);
        recalculate=true;
    }
    if(recalculate) {
        if(single) {
            uint a_real=f_from_d(ar),a_imag=f_from_d(ai),b_real=f_from_d(br),b_imag=f_from_d(bi);
            real=f_to_d(f_mul(f_fma(b_real,a_real,f_mul(b_imag^0x80000000u,a_imag)),0x7f800000u));
            imag=f_to_d(f_mul(f_fma(b_imag,a_real,f_mul(a_imag,b_real)),0x7f800000u));
        } else {
            real=d_mul(d_fma(br,ar,d_mul(d_neg(bi),ai)),D_INF);
            imag=d_mul(d_fma(bi,ar,d_mul(ai,br)),D_INF);
        }
    }
    return {real,imag};
}
inline _eon_complex _eon_complex_power(_eon_complex a,_eon_complex b) {
    if(!_eon_complex_truth(b))return {0x3ff0000000000000ul,0ul};
    if(!_eon_complex_truth(a))return d_gt(b.real,0ul)?_eon_complex{0ul,0ul}:_eon_complex{D_QNAN,D_QNAN};
    if(d_is_zero(b.imag)&&d_lt(d_abs(b.real),0x4059000000000000ul)) {
        long exponent=d_to_int(b.real);
        if(d_eq(d_from_int(exponent),b.real))return _eon_complex_powi(a,exponent);
    }
    _eon_complex logarithm=_eon_complex_log(a);
    _eon_complex product=_eon_complex_mul(b,logarithm);
    if(d_is_nan(product.real)&&d_is_nan(product.imag))product=_eon_complex_power_recover(b,logarithm);
    return _eon_complex_exp(product);
}
inline _eon_complex _eon_complex_power32(_eon_complex a,_eon_complex b) {
    if(!_eon_complex_truth(b))return {0x3ff0000000000000ul,0ul};
    if(!_eon_complex_truth(a))return d_gt(b.real,0ul)?_eon_complex{0ul,0ul}:_eon_complex{D_QNAN,D_QNAN};
    if(d_is_zero(b.imag)&&d_lt(d_abs(b.real),0x4059000000000000ul)) {
        long exponent=d_to_int(b.real);
        if(d_eq(d_from_int(exponent),b.real))return _eon_complex_powi32(a,exponent);
    }
    _eon_complex logarithm=_eon_complex_log32(a);
    _eon_complex product=_eon_complex_mul32(b,logarithm);
    if(d_is_nan(product.real)&&d_is_nan(product.imag))product=_eon_complex_power_recover(b,logarithm,true);
    return _eon_complex_exp32(product);
}
'''


def metal_complex_power_header():
    return MSL_HEADER
