"""Native binary64 arcsin/arccos for the frozen Apple arm64 CPU targets.

Polynomial coefficients and rounded operation sequences were independently
reconstructed from installed libsystem_m (macOS26.4), UUID
83b173cd7f3239018e67c0b2676be015. No libm code executes
on the CPU during native simulation. Include binary64.MSL_HEADER and
math64.MSL_MATH_HEADER first. The companion inverse_trigonometry32 module
preserves the distinct scalar-float endpoints and NaN policy.

Apple NumPy1.26.4 and scalar libm use the same functions here, including NaN
payload/sign handling; compiler target and flags remain part of model identity.
"""

# Reuses the exact 128-bit accumulator and normal-operand FMA from
# stable_math64/hyperbolic_math64. Entry guards keep inverse polynomial
# operands normal or zero, including compensated residuals. The full product
# and sticky alignment preserve ties and cancellation; final rounding is the
# same round-to-even operation as the general binary64 FMA. Distinct symbols
# and an include guard allow both widths and unrelated math headers together.
from .stable_math64 import _STABLE_FMA_HEADER
from .hyperbolic_math64 import _NORMAL_FMA_HEADER, _NORMAL_MUL_HEADER as _HYPER_MUL_HEADER

_ARITHMETIC_HEADER = (
    _STABLE_FMA_HEADER.split('inline ulong stable_fma(', 1)[0]
        .replace('EON_STABLE_FMA_INCLUDED', 'EON_ASINCOS64_ARITH_INCLUDED')
        .replace('stable_', '_eon_asincos64_')
        .replace('#define EON_ASINCOS64_ARITH_INCLUDED\n',
                 '#define EON_ASINCOS64_ARITH_INCLUDED\n\n')
    + _NORMAL_FMA_HEADER.replace('hyper_', '_eon_asincos64_')
    + '\n#endif\n')

# Retain this specialization's namespace, guard and original source layout.
# Apache-2.0; see third_party/ArrowMetal-hyperbolic-multiplication-NOTICE.
_NORMAL_MUL_HEADER = ('\n' + _HYPER_MUL_HEADER
    .replace('EON_HYPER_MUL_INCLUDED', 'EON_INVERSE_MUL_INCLUDED')
    .replace('hyper_mul(', '_eon_asincos64_mul('))

# Align at the addend exponent only while both complete significands fit.
# Same-sign sums have their leading bit at126or127; cancellation retains the
# original full-width normalization. Out-of-range alignment uses the unchanged
# normal FMA above. No approximation or additional rounding is introduced.
_COMPACT_FMA_HEADER = r'''
#ifndef EON_ASINCOS64_COMPACT_INCLUDED
#define EON_ASINCOS64_COMPACT_INCLUDED

inline ulong _eon_asincos64_fma_compact(ulong a,ulong b,ulong c) {
    if(d_is_zero(a)||d_is_zero(b)||d_is_zero(c)) return _eon_asincos64_fma_normal(a,b,c);
    int ec=int(d_exp(c));
    int shift=int(d_exp(a))+int(d_exp(b))-ec-1001;
    // Both significands are 53 bits: their full product has at most106.
    // This range keeps its high bit <=126 at the addend exponent, so one
    // alignment replaces normalizing both operands and aligning them again.
    if(shift < -64 || shift > 21) return _eon_asincos64_fma_normal(a,b,c);
    ulong ph,pl=d_umul128(d_mant(a)|(1ul<<52),d_mant(b)|(1ul<<52),&ph);
    _eon_asincos64_u128 product;
    if(shift>=0) product=shift ? _eon_asincos64_u128{pl<<shift,(ph<<shift)|(pl>>(64-shift))}:_eon_asincos64_u128{pl,ph};
    else if(shift==-64) product={ph|ulong(pl!=0ul),0ul};
    else { int n=-shift;product={(pl>>n)|(ph<<(64-n))|ulong((pl<<(64-n))!=0ul),ph>>n}; }
    _eon_asincos64_u128 addend={0ul,(d_mant(c)|(1ul<<52))<<10};
    ulong sign=(a^b)>>63;
    _eon_asincos64_u128 result;
    int rt; ulong normalized;
    if(sign==(c>>63)) { ulong low=product.lo+addend.lo;result={low,product.hi+addend.hi+ulong(low<product.lo)};
        // One operand starts at126; equal signs cannot cancel. Reuse the
        // fixed-width normalization of shared FMA commit8b3aaab.
        rt=(result.hi>>63)?127:126;
        int down=rt-119;
        normalized=(result.hi>>down)|ulong((result.lo|(result.hi<<(64-down)))!=0ul);
    }
    else {
        bool smaller=product.hi!=addend.hi ? product.hi<addend.hi:product.lo<addend.lo;
        _eon_asincos64_u128 large=smaller?addend:product,small=smaller?product:addend;
        result={large.lo-small.lo,large.hi-small.hi-ulong(large.lo<small.lo)};
        if(smaller)sign=c>>63;
        if((result.lo|result.hi)==0ul)return 0ul;
        rt=_eon_asincos64_top128(result);
        normalized=(rt>55?_eon_asincos64_shr128(result,rt-55):_eon_asincos64_shl128(result,55-rt)).lo;
    }
    int exponent=ec+rt-126;
    if(exponent<=0||exponent>=2047)return d_finish(sign,exponent,normalized);
    ulong mantissa=normalized>>3,low=normalized&7ul;
    if(low>4ul||(low==4ul&&(mantissa&1ul)))++mantissa;
    if(mantissa>>53){mantissa>>=1;++exponent;}
    if(exponent>=2047)return (sign<<63)|D_INF;
    return (sign<<63)|(ulong(exponent)<<52)|(mantissa&0xffffffffffffful);
}

#endif
'''

_COMMON_HEADER = r'''
// A compensated sqrt(1-x) pair, in the scalar CPU operation order.
inline void _eon_inverse_root_pair(ulong x, thread ulong *hi, thread ulong *lo) {
    ulong z=d_sub(0x3ff0000000000000ul,x);
    ulong r=d_div(0x3ff0000000000000ul,d_sqrt(z));
    ulong h=d_mul(z,r);
    ulong l=_eon_asincos64_fma_normal(r,z,d_neg(h));
    ulong e=_eon_asincos64_fma_normal(r,h,0xbff0000000000000ul);
    e=_eon_asincos64_fma_normal(r,l,e);
    e=d_mul(d_mul(h,0xbfe0000000000000ul),e);
    *hi=d_add(h,e);
    e=d_add(e,d_sub(h,*hi));
    *lo=d_add(l,e);
}
'''

_ASIN_HEADER = r'''
constant ulong _eon_inverse_asin_small[12]={
    0x3fa036813a32683dul, 0xbf903e23c29bea05ul, 0x3f93c1e11fadf3bful, 0x3f7b18a981ad46e5ul,
    0x3f88e14135547bb9ul, 0x3f8c71986b9cc17eul, 0x3f91c6a65f51a9f2ul, 0x3f96e8a0f27f87e2ul,
    0x3f9f1c72b02273fdul, 0x3fa6db6db469a3ecul, 0x3fb3333333367712ul, 0x3fc55555555553d1ul
};
constant ulong _eon_inverse_asin_outer[12]={
    0xbee96a54345fba36ul, 0x3f21237515adbe1aul, 0xbf45f673b87be906ul, 0x3f61f355490991e8ul,
    0xbf7551cf57cf534bul, 0x3f840b1e002b9fedul, 0xbf900d00dbcdcd24ul, 0x3f978a41eaa0a83eul,
    0xbfa0fe896e15227bul, 0x3fa9f0e3d3944f91ul, 0xbfb6ca67147357d8ul, 0x3fcb77fdafad0fc7ul
};

inline ulong _eon_inverse_asin_positive_outer(ulong x) {
    ulong p=_eon_asincos64_fma_normal(x,_eon_inverse_asin_outer[0],_eon_inverse_asin_outer[1]);
    for(uint j=2;j<12;++j) p=_eon_asincos64_fma_normal(p,x,_eon_inverse_asin_outer[j]);
    p=d_mul(x,p);
    ulong h,l;
    _eon_inverse_root_pair(x,&h,&l);
    ulong q=d_add(p,0xbff921fb31e97d96ul);
    ulong tail=d_add(p,d_sub(0xbff921fb31e97d96ul,q));
    tail=d_add(tail,0x3c9eab77149ad27cul);
    ulong product=d_mul(h,q);
    ulong error=_eon_asincos64_fma_normal(h,q,d_neg(product));
    error=_eon_asincos64_fma_normal(h,tail,error);
    error=_eon_asincos64_fma_normal(l,q,error);
    ulong result=d_add(product,0x3ff921fb54442d18ul);
    ulong correction=d_add(product,d_sub(0x3ff921fb54442d18ul,result));
    error=d_add(correction,error);
    error=d_add(error,0x3c91a62633145c07ul);
    return d_add(result,error);
}
inline ulong d_inverse_arcsin(ulong x) {
    ulong a=d_abs(x),sign=x&0x8000000000000000ul;
    if(d_is_nan(x)) return x|0x0008000000000000ul;
    if(a>0x3ff0000000000000ul) return D_QNAN|sign;
    if(a==0x3ff0000000000000ul) return 0x3ff921fb54442d18ul|sign;
    if(a>0x3fe0000000000000ul) return _eon_inverse_asin_positive_outer(a)^sign;
    // The CPU's tiny arithmetic rounds to x, including subnormals and -0.
    if(a<=0x3e57137449123ef5ul) return x;
    ulong z=d_mul(x,x);
    ulong p=_eon_asincos64_fma_normal(z,_eon_inverse_asin_small[0],_eon_inverse_asin_small[1]);
    for(uint j=2;j<12;++j) p=_eon_asincos64_fma_normal(p,z,_eon_inverse_asin_small[j]);
    return _eon_asincos64_fma_normal(d_mul(z,p),x,x);
}
'''

_ACOS_HEADER = r'''
constant ulong _eon_inverse_acos_small[10]={
    0x3f93fb59550c32c6ul, 0x3f73247de513273ful, 0x3f8a2bb96b7a9a27ul, 0x3f8c3b748342976eul,
    0x3f91c951a069c364ul, 0x3f96e8769bc3cc51ul, 0x3f9f1c7452fc3c22ul, 0x3fa6db6daf9e68ccul,
    0x3fb33333333dd551ul, 0x3fc5555555554f7dul
};
constant ulong _eon_inverse_acos_positive[11]={
    0xbef0b930081a7021ul, 0x3f26afe1a4c8d680ul, 0xbf4d29684a4dbb2eul, 0x3f67db25aac9ff95ul,
    0xbf7c59eacc8789cful, 0x3f8ac501bff82c7ful, 0xbf95c3d9c3b1d6a7ul, 0x3fa09088dea5a941ul,
    0xbfa9cb1ed2332914ul, 0x3fb6c609ab6ca660ul, 0xbfcb77afa2b16541ul
};
constant ulong _eon_inverse_acos_negative[12]={
    0x3ef183f42cd7c67dul, 0x3f278bbe82eecf3aul, 0x3f4e0176f0fe6d68ul, 0x3f685a242113be25ul,
    0x3f7cbd21c17d0ac6ul, 0x3f8afb184a1c7e2eul, 0x3f95d8d8a7514918ul, 0x3fa09656752cdb57ul,
    0x3fa9cd5bdbcf0e69ul, 0x3fb6c6531de3c9a2ul, 0x3fcb77b5441ad6c0ul, 0x3ff921fa9ef1a025ul
};
constant ulong _eon_inverse_acos_middle[12]={
    0x3fc6ed2e902f3321ul, 0xbfd7cc5c41dc4d17ul, 0x3fda96b55af07319ul, 0xbfd0b2b89e9a0155ul,
    0x3fc11ada9fdcafeeul, 0xbf99903713a9edb1ul, 0x3f9ae64e0ba95f92ul, 0x3f95694f21873452ul,
    0x3f9f487f82c10e4cul, 0x3fa6d9be2b774268ul, 0x3fb3333d167d72acul, 0x3fc555553b0bf834ul
};

inline ulong _eon_inverse_acos_mid(ulong x) {
    ulong z=d_mul(x,x);
    ulong p=_eon_asincos64_fma_normal(z,_eon_inverse_acos_middle[0],_eon_inverse_acos_middle[1]);
    for(uint j=2;j<12;++j) p=_eon_asincos64_fma_normal(p,z,_eon_inverse_acos_middle[j]);
    ulong l=d_mul(x,d_mul(z,p));
    ulong h=d_add(x,l),e=d_sub(h,x);
    ulong result=d_sub(0x3ff921fb54442d18ul,h);
    ulong correction=d_sub(d_sub(0x3ff921fb54442d18ul,result),h);
    e=d_sub(e,l);
    e=d_add(e,correction);
    e=d_add(e,0x3c91a62633145c07ul);
    return d_add(result,e);
}
inline ulong d_inverse_arccos(ulong x) {
    ulong a=d_abs(x);
    // Scalar arccos quiets a NaN and reverses its sign, preserving payload.
    if(d_is_nan(x)) return (x^0x8000000000000000ul)|0x0008000000000000ul;
    if(a>0x3ff0000000000000ul) return D_QNAN;
    if(x==0x3ff0000000000000ul) return 0ul;
    if(x==0xbff0000000000000ul) return 0x400921fb54442d18ul;
    if(a<=0x3fd999999999999aul) {
        if(a<=0x3ca8d313198a2e03ul) return 0x3ff921fb54442d18ul;
        ulong z=d_mul(x,x);
        ulong p=_eon_asincos64_fma_normal(z,_eon_inverse_acos_small[0],_eon_inverse_acos_small[1]);
        for(uint j=2;j<10;++j) p=_eon_asincos64_fma_normal(p,z,_eon_inverse_acos_small[j]);
        return d_sub(0x3ff921fb54442d18ul,_eon_asincos64_fma_normal(d_mul(z,p),x,x));
    }
    if(a<=0x3fe3333333333333ul) return _eon_inverse_acos_mid(x);
    if(x>>63) {
        ulong h=d_sqrt(d_add(x,0x3ff0000000000000ul));
        ulong p=_eon_asincos64_fma_normal(x,_eon_inverse_acos_negative[0],_eon_inverse_acos_negative[1]);
        for(uint j=2;j<12;++j) p=_eon_asincos64_fma_normal(p,x,_eon_inverse_acos_negative[j]);
        ulong y=d_mul(h,p),result=d_sub(0x400921fb54442d18ul,y);
        ulong correction=d_sub(d_sub(0x400921fb54442d18ul,result),y);
        return d_add(result,d_add(correction,0x3ca1a62633145c07ul));
    }
    ulong p=_eon_asincos64_fma_normal(x,_eon_inverse_acos_positive[0],_eon_inverse_acos_positive[1]);
    for(uint j=2;j<11;++j) p=_eon_asincos64_fma_normal(p,x,_eon_inverse_acos_positive[j]);
    p=d_mul(x,p);
    ulong h,l;
    _eon_inverse_root_pair(x,&h,&l);
    ulong q=d_add(p,0x3ff921fa926d2f24ul);
    ulong tail=d_add(p,d_sub(0x3ff921fa926d2f24ul,q));
    tail=d_add(tail,0x3c4b4a23d0ecbb40ul);
    ulong y=d_mul(l,q);
    y=_eon_asincos64_fma_normal(h,tail,y);
    return _eon_asincos64_fma_normal(h,q,y);
}
'''


def metal_inverse_trigonometry_header(names):
    """Select only named binary64 functions; no forward-trig dependency."""
    used=set(names)&{'arcsin','arccos'}
    if not used:
        return ''
    arithmetic = _ARITHMETIC_HEADER.replace('d_mul(', '_eon_asincos64_mul(')
    functions = (_COMMON_HEADER
                 + (_ASIN_HEADER if 'arcsin' in used else '')
                 + (_ACOS_HEADER if 'arccos' in used else ''))
    functions = functions.replace('d_mul(', '_eon_asincos64_mul(')
    functions = functions.replace('_eon_asincos64_fma_normal(',
                                  '_eon_asincos64_fma_compact(')
    return _NORMAL_MUL_HEADER + arithmetic + _COMPACT_FMA_HEADER + functions
