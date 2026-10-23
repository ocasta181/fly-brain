"""Software binary64 exp/log with the frozen Apple arm64 rounding sequence.

Table reduction and factored polynomials preserve the reference libm operation
order, including fused products, subnormal inputs and gradual underflow. Tables
contain numerical coefficients only. No host math runs during native execution.
Include binary64.MSL_HEADER and math64.MSL_MATH_HEADER before these fragments.
"""

from .scalar_math_tables import exponential64_segment_tables, logarithm64_table

_LOG_HEADER = r'''
constant ulong _eon_log_coefficients[6]={
    0xbfc7b47c30037039ul, 0x3fc249449239443dul, 0x3ff9de61616cdf2bul, 0x3ff609d14c7ee9beul, 0x3ffab7d4f0547db7ul, 0xbff3f6ab6ec10818ul
};
''' + logarithm64_table(
    '_eon_log_segments', padded=False, separator=", ", trailing_comma=True
) + r'''

inline ulong _eon_log_core(ulong bits, int adjust) {
 if(d_is_zero(bits)) return 0xfff0000000000000ul;
 long k=(as_type<long>(bits+13841831046909263872ul)>>52)+adjust;
 ulong fraction=bits&0xffffffffffffful;
 uint idx=uint((fraction+0x100000000000ul)>>45);
 ulong m=fraction|4607182418800017408ul,inv=_eon_log_segments[3*idx],hi=_eon_log_segments[3*idx+1],lo=_eon_log_segments[3*idx+2];
 ulong r=d_fma(m,inv,13830554455654793216ul);
 ulong p16=d_fma(r,_eon_log_coefficients[1],_eon_log_coefficients[0]);
 ulong p18=d_fma(r,d_add(r,_eon_log_coefficients[3]),_eon_log_coefficients[2]);
 ulong p20=d_fma(r,d_add(r,_eon_log_coefficients[5]),_eon_log_coefficients[4]);
 ulong p=d_mul(d_mul(d_mul(d_mul(r,r),p16),p18),p20);
 if(k!=0) {
  lo=d_fma(d_from_int(k),4494016781675590265ul,lo);
  hi=d_fma(d_from_int(k),4604418534062161920ul,hi);
  return d_add(hi,d_add(r,d_add(lo,p)));
 }
 ulong prod=d_mul(inv,m);
 ulong rounded=d_sub(prod,4607182418800017408ul);
 ulong error=d_fma(m,inv,d_neg(prod));
 ulong high=rounded&18446744073575333888ul;
 ulong tail=d_add(error,d_sub(rounded,high));
 return d_add(d_add(high,hi),d_add(d_add(tail,lo),p));
}


inline ulong d_logarithm(ulong x) {
    if(d_is_nan(x)) return x | 0x0008000000000000ul;
    if(d_is_zero(x)) return 0xfff0000000000000ul;
    if(x >> 63) return D_QNAN;
    if(d_exp(x) == 0x7fful) return x;
    if(d_exp(x) == 0ul)
        return _eon_log_core(d_mul(x,0x4350000000000000ul),-54);
    return _eon_log_core(x,0);
}



'''

_EXP_HEADER = "\n" + exponential64_segment_tables('_eon_exp_coefficients', '_eon_exp_segments') + r'''

inline ulong _eon_exp_core(ulong x) {
 ulong t=d_mul(x,_eon_exp_coefficients[0]);
 long k=d_to_int(t);if(d_lt(t,d_from_int(k)))--k;
 ulong r=d_sub(t,d_from_int(k));
 ulong error=d_fma(x,_eon_exp_coefficients[0],d_neg(t));
 ulong delta=d_fma(x,_eon_exp_coefficients[1],error);
 uint idx=uint(k)&127u;
 ulong base=_eon_exp_segments[2*idx],tail=_eon_exp_segments[2*idx+1];
 ulong a=d_add(r,_eon_exp_coefficients[2]),b=d_add(r,_eon_exp_coefficients[3]);
 r=d_add(r,d_add(tail,delta));
 ulong q=d_fma(r,a,_eon_exp_coefficients[4]),s=d_fma(r,b,_eon_exp_coefficients[5]);
 ulong factor=d_mul(d_mul(d_mul(r,_eon_exp_coefficients[6]),q),s);
 
 long scale=k>>7;
 if(scale >= -1022) return d_fma(factor,base,base)+(ulong(scale)<<52);
 base+=ulong(scale+1022)<<52;
 ulong rounded=d_add(base,0x3ff0000000000000ul);
 ulong error2=d_sub(base,d_sub(rounded,0x3ff0000000000000ul));
 ulong tail2=d_fma(base,factor,error2);
 return d_mul(d_sub(d_add(rounded,tail2),0x3ff0000000000000ul),0x0010000000000000ul);

}


inline ulong d_exponential(ulong x) {
    if(d_is_nan(x)) return x | 0x0008000000000000ul;
    if(d_ge(x,0x40862e42fefa39f0ul)) return d_mul(x,0x7feffffffffffffful);
    if(d_le(x,0xc0874a0000000000ul)) return 0ul;
    return _eon_exp_core(x);
}

'''

def metal_math_header(functions):
    """Include only the helpers used by dynamic model statements."""
    headers = {"exp": _EXP_HEADER, "log": _LOG_HEADER}
    return "\n".join(headers[name] for name in sorted(functions))
