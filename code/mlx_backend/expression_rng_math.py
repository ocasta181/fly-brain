"""Binary64 operation sequence for the frozen Apple arm64 CPU normal transform.

The numeric reciprocal/log split table and factored polynomial are pinned to
macOS 26.4 (25E246), with Brian 2.8.0 / NumPy 1.26.4. These helpers keep the libm rounding points used
by legacy NumPy RandomState's polar normal transform, including fused products.
They do not introduce a one-ULP acceptance mode or replace model math helpers.
"""

from .scalar_math_tables import exponential64_segment_tables, logarithm64_table

FMA_HEADER = r'''

struct RngWide { ulong hi; ulong lo; };
inline RngWide rng_shift_jam(RngWide a,uint shift) {
 if(!shift)return a;
 if(shift>=128)return {0ul,ulong(a.hi||a.lo)};
 if(shift>=64) {
  uint s=shift-64;
  ulong tail=s ? a.hi&((1ul<<s)-1ul):0ul;
  return {0ul,(a.hi>>s)|ulong(tail||a.lo)};
 }
 return {a.hi>>shift,(a.lo>>shift)|(a.hi<<(64-shift))|
         ulong((a.lo&((1ul<<shift)-1ul))!=0ul)};
}
inline ulong expr_fma(ulong a,ulong b,ulong c) {
 int ea=int(d_exp(a)),eb=int(d_exp(b)),ec=int(d_exp(c));
 if(ea==2047||eb==2047||ec==2047)return d_add(d_mul(a,b),c);
 if(d_is_zero(a)||d_is_zero(b))return d_add(d_mul(a,b),c);
 if(d_is_zero(c))return d_mul(a,b);
 ulong ma=d_mant(a),mb=d_mant(b),mc=d_mant(c);
 if(ea)ma|=1ul<<52;else {int z=int(clz(ma))-11;ma<<=z;ea=1-z;}
 if(eb)mb|=1ul<<52;else {int z=int(clz(mb))-11;mb<<=z;eb=1-z;}
 if(ec)mc|=1ul<<52;else {int z=int(clz(mc))-11;mc<<=z;ec=1-z;}
 ulong hi,lo=d_umul128(ma,mb,&hi);
 uint shift=uint(clz(hi))-1u;
 RngWide p={(hi<<shift)|(lo>>(64-shift)),lo<<shift};
 int pe=ea+eb-2046+22-int(shift);
 RngWide q={mc<<10,0ul};
 int ce=ec-1023,e=max(pe,ce);
 p=rng_shift_jam(p,uint(e-pe));q=rng_shift_jam(q,uint(e-ce));
 ulong sign=(a^b)>>63,cs=c>>63;
 if(sign==cs) {
  ulong low=p.lo+q.lo;p={p.hi+q.hi+ulong(low<p.lo),low};
 } else {
  if(p.hi<q.hi||(p.hi==q.hi&&p.lo<q.lo)) {RngWide tmp=p;p=q;q=tmp;sign=cs;}
  ulong low=p.lo-q.lo;p={p.hi-q.hi-ulong(p.lo<q.lo),low};
 }
 if(!(p.hi||p.lo))return 0ul;
 int leading=p.hi ?127-int(clz(p.hi)):63-int(clz(p.lo));
 // Exact cancellation can expose low bits; normalize before rounding once.
 if(leading<55) return d_finish(sign,e+leading-126+1023,p.lo<<(55-leading));
 RngWide rounded=rng_shift_jam(p,uint(leading-55));
 return d_finish(sign,e+leading-126+1023,rounded.lo);
}

'''

NORMAL_MATH_HEADER = FMA_HEADER + r'''
constant ulong expr_log_coeff[6]={
    0xbfc7b47c30037039ul, 0x3fc249449239443dul, 0x3ff9de61616cdf2bul, 0x3ff609d14c7ee9beul, 0x3ffab7d4f0547db7ul, 0xbff3f6ab6ec10818ul
};
''' + logarithm64_table(
    'expr_log_table', padded=True, separator=", ", trailing_comma=False
) + r'''

inline ulong expr_apple_log(ulong bits) {
 if(d_is_zero(bits)) return 0xfff0000000000000ul;
 long k=as_type<long>(bits+13841831046909263872ul)>>52;
 ulong fraction=bits&0xffffffffffffful;
 uint idx=uint((fraction+0x100000000000ul)>>45);
 ulong m=fraction|4607182418800017408ul,inv=expr_log_table[4*idx],hi=expr_log_table[4*idx+1],lo=expr_log_table[4*idx+2];
 ulong r=expr_fma(m,inv,13830554455654793216ul);
 ulong p16=expr_fma(r,expr_log_coeff[1],expr_log_coeff[0]);
 ulong p18=expr_fma(r,d_add(r,expr_log_coeff[3]),expr_log_coeff[2]);
 ulong p20=expr_fma(r,d_add(r,expr_log_coeff[5]),expr_log_coeff[4]);
 ulong p=d_mul(d_mul(d_mul(d_mul(r,r),p16),p18),p20);
 if(k!=0) {
  lo=expr_fma(d_from_int(k),4494016781675590265ul,lo);
  hi=expr_fma(d_from_int(k),4604418534062161920ul,hi);
  return d_add(hi,d_add(r,d_add(lo,p)));
 }
 ulong prod=d_mul(inv,m);
 ulong rounded=d_sub(prod,4607182418800017408ul);
 ulong error=expr_fma(m,inv,d_neg(prod));
 ulong high=rounded&18446744073575333888ul;
 ulong tail=d_add(error,d_sub(rounded,high));
 return d_add(d_add(high,hi),d_add(d_add(tail,lo),p));
}

'''


POISSON_EXP_HEADER = "\n\n" + exponential64_segment_tables('expr_exp_coeff', 'expr_exp_table') + r'''

inline ulong expr_poisson_exp_negative(ulong x) {
 ulong t=d_mul(x,expr_exp_coeff[0]);
 long k=d_to_int(t);if(d_lt(t,d_from_int(k)))--k;
 ulong r=d_sub(t,d_from_int(k));
 ulong error=expr_fma(x,expr_exp_coeff[0],d_neg(t));
 ulong delta=expr_fma(x,expr_exp_coeff[1],error);
 uint idx=uint(k)&127u;
 ulong base=expr_exp_table[2*idx],tail=expr_exp_table[2*idx+1];
 ulong a=d_add(r,expr_exp_coeff[2]),b=d_add(r,expr_exp_coeff[3]);
 r=d_add(r,d_add(tail,delta));
 ulong q=expr_fma(r,a,expr_exp_coeff[4]),s=expr_fma(r,b,expr_exp_coeff[5]);
 ulong factor=d_mul(d_mul(d_mul(r,expr_exp_coeff[6]),q),s);
 return expr_fma(factor,base,base)+(ulong(k>>7)<<52);
}

'''
