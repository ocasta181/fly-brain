"""Certified scalar32 sine evaluation of the pinned Apple operation graph.

The native pair-of-floats polynomial encloses the existing software64 graph;
only an enclosure wholly inside one binary32 rounding cell bypasses that graph.
This is an exact-output optimization, not a tolerance or an approximate mode.
The coefficient graph and dyadic reduction come from trigonometry32.py. The
companion rational-bound test records every propagation step and the margin.
"""

MSL_HEADER = r'''
#pragma METAL fp contract(off)
struct _eon_sin32_dd{float hi;float lo;};
inline _eon_sin32_dd _eon_sin32_dd_sum(float a,float b){
    float s=a+b,z=s-a;
    return _eon_sin32_dd{s,(a-(s-z))+(b-z)};
}
inline _eon_sin32_dd _eon_sin32_dd_add(_eon_sin32_dd a,_eon_sin32_dd b){
    _eon_sin32_dd s=_eon_sin32_dd_sum(a.hi,b.hi);
    float tail=(a.lo+b.lo)+s.lo;
    return _eon_sin32_dd_sum(s.hi,tail);
}
inline _eon_sin32_dd _eon_sin32_dd_mul(_eon_sin32_dd a,_eon_sin32_dd b){
    float product=a.hi*b.hi;
    float tail=fma(a.hi,b.hi,-product);
    tail=fma(a.hi,b.lo,tail);tail=fma(a.lo,b.hi,tail);
    // For normalized finite pairs, |tail| <= (3*u+O(u*u))*|product|.
    // FastTwoSum then gives the same exact normalization with three fewer
    // additions. The comparison keeps the general path for cancellation or
    // exceptional intermediates outside that proof, including FTZ cases.
    if(isfinite(product)&&fabs(product)>=fabs(tail)){
        float sum=product+tail;
        return _eon_sin32_dd{sum,tail-(sum-product)};
    }
    return _eon_sin32_dd_sum(product,tail);
}
inline _eon_sin32_dd _eon_sin32_dd_fma(_eon_sin32_dd a,_eon_sin32_dd b,_eon_sin32_dd c){
    return _eon_sin32_dd_add(_eon_sin32_dd_mul(a,b),c);
}
template<bool Relative=false>
inline bool _eon_sin32_dd_certify(_eon_sin32_dd value,thread uint& result,float reduced=1.0f){
    uint raw=as_type<uint>(value.hi),absolute=raw&0x7fffffffu;
    uint exponent=absolute>>23;
    if(exponent<26u||exponent>252u)return false;
    uint shift=(absolute&0x7fffffu)?24u:25u;
    float radius=as_type<float>((exponent-shift)<<23);
    // Independent rational propagation bounds both the CPU/DD graph error
    // and this comparison's rounding by less than 2^-42.
    if(Relative){
        // The sine graph error scales with |reduced|. Its relative bound is
        // <2^-43 for |reduced|>=2^-49, including flushed-tail allowances.
        // 2^-42 also covers DD reduction and comparison rounding.
        float error=fabs(reduced)*0x1p-42f;
        if(fabs(value.lo)+error>=radius)return false;
    }else if(fabs(value.lo)+0x1p-42f>=radius)return false;
    result=raw;return true;
}
// Metal permits either nearest-even or toward-zero integer casts. A cast
// rounded toward zero can be corrected by one adjacent magnitude word.
// |value|<2^49 makes the residual fit 25 bits and every shift fit uint32.
inline float _eon_sin32_integer_nearest(long value,float high,thread long& residual){
    uint bits=as_type<uint>(high),exponent=(bits>>23)&255u;
    uint halfway=1u<<max(int(exponent)-151,0);
    uint distance=uint(residual<0?-residual:residual);
    if((residual<0)==(value<0)
            &&(distance>halfway||(distance==halfway&&(bits&1u)))){
        high=as_type<float>(bits+1u);
        residual=value-long(high);
    }
    return high;
}
inline _eon_sin32_dd _eon_sin32_dd_medium(uint raw,thread long& count){
    long reduced=_eon_trig32_medium_integer(raw,count);
    // Correct either permitted cast mode to nearest-even. The remaining
    // integer residual is <=2^24, so its float conversion is exact in both
    // modes. Scaling preserves normal exact components of the certificate.
    float high=float(reduced);
    long residual=reduced-long(high);
    high=_eon_sin32_integer_nearest(reduced,high,residual);
    return _eon_sin32_dd{high*0x1p-49f,float(residual)*0x1p-49f};
}
// One polynomial graph for both quadrants avoids executing two divergent DD
// bodies in a mixed-quadrant SIMD group. These are the same real polynomials;
// the separate certificate proof accounts for the changed DD rounding graph.
inline _eon_sin32_dd _eon_sin32_s3(){return _eon_sin32_dd{-0x1.9943e00000000p-13f,-0x1.f890320000000p-38f};}
inline _eon_sin32_dd _eon_sin32_s2(){return _eon_sin32_dd{0x1.11073a0000000p-7f,0x1.fa28be0000000p-32f};}
inline _eon_sin32_dd _eon_sin32_s1(){return _eon_sin32_dd{-0x1.5555460000000p-3f,0x1.b2ebfa0000000p-28f};}
inline _eon_sin32_dd _eon_sin32_c4(){return _eon_sin32_dd{0x1.9916a00000000p-16f,-0x1.b6385c0000000p-43f};}
inline _eon_sin32_dd _eon_sin32_c3(){return _eon_sin32_dd{-0x1.6c07f20000000p-10f,0x1.2d74420000000p-35f};}
inline _eon_sin32_dd _eon_sin32_c2(){return _eon_sin32_dd{0x1.55553c0000000p-5f,0x1.e265a60000000p-31f};}
inline _eon_sin32_dd _eon_sin32_c1(){return _eon_sin32_dd{-0x1.0000000000000p-1f,0x1.a3e8b40000000p-29f};}
inline _eon_sin32_dd _eon_sin32_one(){return _eon_sin32_dd{1.0f,0.0f};}
inline _eon_sin32_dd _eon_sin32_dd_select(_eon_sin32_dd a,_eon_sin32_dd b,bool condition){
    return _eon_sin32_dd{select(a.hi,b.hi,condition),select(a.lo,b.lo,condition)};
}
inline _eon_sin32_dd _eon_sin32_dd_polynomial(_eon_sin32_dd value,bool cosine){
    _eon_sin32_dd q=_eon_sin32_dd_mul(value,value),cube=_eon_sin32_dd_mul(value,q);
    _eon_sin32_dd c4=_eon_sin32_c4();
    c4.hi=select(0.0f,c4.hi,cosine);c4.lo=select(0.0f,c4.lo,cosine);
    _eon_sin32_dd c3=_eon_sin32_dd_select(_eon_sin32_s3(),_eon_sin32_c3(),cosine);
    _eon_sin32_dd c2=_eon_sin32_dd_select(_eon_sin32_s2(),_eon_sin32_c2(),cosine);
    _eon_sin32_dd c1=_eon_sin32_dd_select(_eon_sin32_s1(),_eon_sin32_c1(),cosine);
    _eon_sin32_dd p=_eon_sin32_dd_fma(q,c4,c3);
    p=_eon_sin32_dd_fma(q,p,c2);p=_eon_sin32_dd_fma(q,p,c1);
    _eon_sin32_dd product=_eon_sin32_dd_select(cube,q,cosine);
    _eon_sin32_dd leading=_eon_sin32_dd_select(value,_eon_sin32_one(),cosine);
    return _eon_sin32_dd_fma(product,p,leading);
}

inline uint _eon_trig32_sine_scalar(uint raw){
    uint absolute=raw&0x7fffffffu;
    if(absolute<0x39800000u||absolute>=0x42f00000u)return _eon_trig32_sine_exact(raw);
    long count=0;
    _eon_sin32_dd reduced;
    if(absolute<0x3f490fdbu)reduced=_eon_sin32_dd{as_type<float>(raw),0.0f};
    else reduced=_eon_sin32_dd_medium(raw,count);

    _eon_sin32_dd value=_eon_sin32_dd_polynomial(reduced,bool(count&1L));
    uint result;
    bool certified=(count&1L)?_eon_sin32_dd_certify(value,result)
        :_eon_sin32_dd_certify<true>(value,result,reduced.hi);
    if(certified)return result^((uint(count)>>1)<<31);
    return _eon_trig32_sine_exact(raw);
}

'''
