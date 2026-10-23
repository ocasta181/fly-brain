"""Native binary32 inverse trig with an enclosed two-float fast path.

Frozen Apple libsystem_m UUID83b173cd7f3239018e67c0b2676be015 uses
binary64 polynomial stages even for float results. Two-float evaluation supplies
the same final binary32 rounding unless its error enclosure meets a rounding
boundary; those inputs execute the exact CPU operation sequence in native
software binary64. Endpoints and NaN policies follow the float library, not a
cast of the binary64 inverse-trig functions. No CPU simulation fallback.

"""

from .inverse_trigonometry64 import _ARITHMETIC_HEADER

_COMMON_HEADER = r'''


struct _eon_asincos32_pair { float hi; float lo; };
inline _eon_asincos32_pair _eon_asincos32_normalize(float hi,float lo) {
    float rounded=hi+lo;
    return {rounded,lo-(rounded-hi)};
}
inline _eon_asincos32_pair _eon_asincos32_neg(_eon_asincos32_pair a) { return {-a.hi,-a.lo}; }
inline _eon_asincos32_pair _eon_asincos32_add(_eon_asincos32_pair a,_eon_asincos32_pair b) {
    float hi=a.hi+b.hi, split=hi-a.hi;
    float lo=(a.hi-(hi-split))+(b.hi-split);
    lo=lo+(a.lo+b.lo);
    return _eon_asincos32_normalize(hi,lo);
}
inline _eon_asincos32_pair _eon_asincos32_mul(_eon_asincos32_pair a,_eon_asincos32_pair b) {
    float hi=a.hi*b.hi;
    float lo=fma(a.hi,b.hi,-hi);
    lo=fma(a.hi,b.lo,lo);
    lo=fma(a.lo,b.hi,lo);
    return _eon_asincos32_normalize(hi,lo);
}
inline _eon_asincos32_pair _eon_asincos32_fma(_eon_asincos32_pair a,_eon_asincos32_pair b,_eon_asincos32_pair c) {
    return _eon_asincos32_add(_eon_asincos32_mul(a,b),c);
}
inline _eon_asincos32_pair _eon_asincos32_root(float z) {
    float hi=precise::sqrt(z);
    float lo=fma(-hi,hi,z)/(2.0f*hi);
    return _eon_asincos32_normalize(hi,lo);
}
inline bool _eon_asincos32_ambiguous(_eon_asincos32_pair a) {
    uint bits=as_type<uint>(a.hi);
    uint next=((bits>>31)!=uint(a.lo>0.0f)) ? bits+1u:bits-1u;
    float half_distance=fabs(as_type<float>(next)-a.hi)*0.5f;
    // Conservative error enclosure for six two-float polynomial stages,
    // coefficient splitting and one Newton sqrt correction (normal operands).
    return fabs(a.lo)>=half_distance-fabs(a.hi)*0x1p-38f;
}
constant ulong _eon_asincos32_small_exact[6]={0x3fa15e1a450e8b98ul,0x3f9169f766637e01ul,0x3f9fe10bae9583baul,0x3fa6d55e6da5eec2ul,0x3fb3334481d0bc2ful,0x3fc555554d58f2f9ul};
constant _eon_asincos32_pair _eon_asincos32_small_pair[6]={{0x1.15e1a40000000p-5f,0x1.43a2e60000000p-31f},{0x1.169f760000000p-6f,0x1.98df800000000p-32f},{0x1.fe10ba0000000p-6f,0x1.d2b0780000000p-31f},{0x1.6d55e60000000p-5f,0x1.b4bdd80000000p-30f},{0x1.3334480000000p-4f,0x1.d0bc300000000p-32f},{0x1.5555540000000p-3f,0x1.ab1e600000000p-28f}};
constant ulong _eon_asincos32_outer_exact[6]={0x3f5ee93cb2957053ul,0x3f6140186da86810ul,0x3f807e5fa0a35d8bul,0x3f9b21ec14e82132ul,0x3fbe2b8cc3173f9eul,0x3ff6a09e64c7c6e1ul};
constant _eon_asincos32_pair _eon_asincos32_outer_pair[6]={{0x1.ee93cc0000000p-10f,-0x1.ad51f60000000p-35f},{0x1.1401860000000p-9f,0x1.b50d020000000p-34f},{0x1.07e5fa0000000p-7f,0x1.46bb160000000p-36f},{0x1.b21ec20000000p-6f,-0x1.62fbda0000000p-31f},{0x1.e2b8cc0000000p-4f,0x1.8b9fd00000000p-31f},{0x1.6a09e60000000p+0f,0x1.31f1b80000000p-26f}};
constant _eon_asincos32_pair _eon_asincos32_half_pi={0x1.921fb60000000p+0f,-0x1.777a5c0000000p-25f};
constant _eon_asincos32_pair _eon_asincos32_pi={0x1.921fb60000000p+1f,-0x1.777a5c0000000p-24f};

inline _eon_asincos32_pair _eon_asincos32_poly_pair(_eon_asincos32_pair z,bool outer) {
    _eon_asincos32_pair a,b,c;
    if(outer) {
        a=_eon_asincos32_fma(z,_eon_asincos32_outer_pair[0],_eon_asincos32_outer_pair[1]);
        b=_eon_asincos32_fma(z,_eon_asincos32_outer_pair[2],_eon_asincos32_outer_pair[3]);
        c=_eon_asincos32_fma(z,_eon_asincos32_outer_pair[4],_eon_asincos32_outer_pair[5]);
    } else {
        a=_eon_asincos32_fma(z,_eon_asincos32_small_pair[0],_eon_asincos32_small_pair[1]);
        b=_eon_asincos32_fma(z,_eon_asincos32_small_pair[2],_eon_asincos32_small_pair[3]);
        c=_eon_asincos32_fma(z,_eon_asincos32_small_pair[4],_eon_asincos32_small_pair[5]);
    }
    auto square=_eon_asincos32_mul(z,z);
    b=_eon_asincos32_fma(square,a,b);
    return _eon_asincos32_fma(square,b,c);
}
inline ulong _eon_asincos32_poly_exact(ulong z,bool outer) {
    ulong a,b,c;
    if(outer) {
        a=_eon_asincos64_fma_normal(z,_eon_asincos32_outer_exact[0],_eon_asincos32_outer_exact[1]);
        b=_eon_asincos64_fma_normal(z,_eon_asincos32_outer_exact[2],_eon_asincos32_outer_exact[3]);
        c=_eon_asincos64_fma_normal(z,_eon_asincos32_outer_exact[4],_eon_asincos32_outer_exact[5]);
    } else {
        a=_eon_asincos64_fma_normal(z,_eon_asincos32_small_exact[0],_eon_asincos32_small_exact[1]);
        b=_eon_asincos64_fma_normal(z,_eon_asincos32_small_exact[2],_eon_asincos32_small_exact[3]);
        c=_eon_asincos64_fma_normal(z,_eon_asincos32_small_exact[4],_eon_asincos32_small_exact[5]);
    }
    ulong square=d_mul(z,z);
    b=_eon_asincos64_fma_normal(square,a,b);
    return _eon_asincos64_fma_normal(square,b,c);
}
inline ulong _eon_asincos32_small_value_exact(ulong x) {
    ulong z=d_mul(x,x),p=_eon_asincos32_poly_exact(z,false);
    return _eon_asincos64_fma_normal(x,d_mul(z,p),x);
}
inline _eon_asincos32_pair _eon_asincos32_small_value_pair(float x) {
    _eon_asincos32_pair a={x,0.0f};
    auto z=_eon_asincos32_mul(a,a);
    return _eon_asincos32_fma(a,_eon_asincos32_mul(z,_eon_asincos32_poly_pair(z,false)),a);
}


'''

_ASIN_HEADER = r'''


inline uint _eon_asincos32_arcsin_exact(uint x) {
    uint a=x&0x7fffffffu,sign=x&0x80000000u;
    if(a>0x7f800000u) return x|0x400000u;
    if(a>0x3f800000u) return 0x7fc00000u;
    if(a==0x3f800000u) return 0x3fc90fdau|sign;
    if(a<0x32000000u) return x;
    ulong v=f_to_d(a),result;
    if(a>0x3f000000u) {
        ulong z=f_to_d(as_type<uint>(1.0f-as_type<float>(a)));
        result=_eon_asincos64_fma_normal(d_neg(_eon_asincos32_poly_exact(z,true)),d_sqrt(z),0x3ff921fb54442d18ul);
    } else result=_eon_asincos32_small_value_exact(v);
    return f_from_d(result)|sign;
}
inline uint f_inverse_arcsin(uint x) {
    uint a=x&0x7fffffffu,sign=x&0x80000000u;
    if(a>=0x3f800000u) return _eon_asincos32_arcsin_exact(x);
    // |asin(x)-x| rounds away below this bound, including every subnormal.
    if(a<=0x39800000u) return x;
    float v=as_type<float>(a);
    _eon_asincos32_pair y;
    if(a>0x3f000000u) {
        float z=1.0f-v;
        auto p=_eon_asincos32_poly_pair({z,0.0f},true);
        y=_eon_asincos32_add(_eon_asincos32_half_pi,_eon_asincos32_neg(_eon_asincos32_mul(p,_eon_asincos32_root(z))));
    } else y=_eon_asincos32_small_value_pair(v);
    if(_eon_asincos32_ambiguous(y)) return _eon_asincos32_arcsin_exact(x);
    return as_type<uint>(y.hi)|sign;
}
inline uint _eon_asincos32_arcsin(uint x) { return f_inverse_arcsin(x); }


'''

_ACOS_HEADER = r'''


inline uint _eon_asincos32_arccos_exact(uint x) {
    uint a=x&0x7fffffffu;
    if(a>0x7f800000u) return x|0x400000u;
    if(a>0x3f800000u) return 0x7fc00000u;
    if(x==0x3f800000u) return 0u;
    if(x==0xbf800000u) return 0x40490fdau;
    ulong v=f_to_d(x),result;
    if(a>0x3f000000u) {
        ulong z=f_to_d(as_type<uint>(1.0f-as_type<float>(a)));
        ulong p=_eon_asincos32_poly_exact(z,true),root=d_sqrt(z);
        result=(x>>31) ? _eon_asincos64_fma_normal(d_neg(p),root,0x400921fb54442d18ul):d_mul(p,root);
    } else {
        if(a>=0x3b000000u) v=_eon_asincos32_small_value_exact(v);
        result=d_sub(0x3ff921fb54442d18ul,v);
    }
    return f_from_d(result);
}
inline uint f_inverse_arccos(uint x) {
    uint a=x&0x7fffffffu;
    if(a>=0x3f800000u) return _eon_asincos32_arccos_exact(x);
    if(a<=0x32800000u) return 0x3fc90fdbu;
    float v=as_type<float>(x);
    _eon_asincos32_pair y;
    if(a>0x3f000000u) {
        float z=1.0f-as_type<float>(a);
        y=_eon_asincos32_mul(_eon_asincos32_poly_pair({z,0.0f},true),_eon_asincos32_root(z));
        if(x>>31) y=_eon_asincos32_add(_eon_asincos32_pi,_eon_asincos32_neg(y));
    } else {
        y=a<0x3b000000u ? _eon_asincos32_pair{v,0.0f}:_eon_asincos32_small_value_pair(v);
        y=_eon_asincos32_add(_eon_asincos32_half_pi,_eon_asincos32_neg(y));
    }
    if(_eon_asincos32_ambiguous(y)) return _eon_asincos32_arccos_exact(x);
    return as_type<uint>(y.hi);
}
inline uint _eon_asincos32_arccos(uint x) { return f_inverse_arccos(x); }


'''

def metal_inverse_trigonometry32_header(names):
    """Select this vertical's float32 helpers; arithmetic headers precede it."""
    used=set(names)&{'arcsin','arccos'}
    if not used:return ''
    return _ARITHMETIC_HEADER+_COMMON_HEADER+(_ASIN_HEADER if 'arcsin' in used else '')+(_ACOS_HEADER if 'arccos' in used else '')
