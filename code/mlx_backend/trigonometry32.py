"""Exact pinned scalar/NumPy binary32 forward trigonometry.

Scalar coefficients/operation order follow macOS26.4 ARM64 libsystem_m UUID
83b173cd7f3239018e67c0b2676be015. NumPy sin/cos follow v1.26.4 ARM NEON
loops_trigonometric.dispatch.c.src. Include binary64, binary32 and math64.
The dynamic compiler selects the actual scalar, NumPy32 or NumPy-half policy.
"""
from .stable_math64 import _STABLE_FMA_HEADER, _STABLE_DIV_HEADER

_SINE_HEADER = r'''
#pragma clang fp contract(off)
constant ulong _eon_trig32_pi_reciprocal[128]={
    0x3fd45f306e000000ul,0xbdeb1bbead603d8bul,0x3fd45f306e000000ul,0xbdeb1bbead603d8bul,
    0x3fd45f306e000000ul,0xbdeb1bbead603d8bul,0x3fd45f306e000000ul,0xbdeb1bbead603d8bul,
    0x3fd45f306e000000ul,0xbdeb1bbead603d8bul,0x3fd45f306e000000ul,0xbdeb1bbead603d8bul,
    0x3fd45f306e000000ul,0xbdeb1bbead603d8bul,0x3fd45f306e000000ul,0xbdeb1bbead603d8bul,
    0x3fd45f306e000000ul,0xbdeb1bbead603d8bul,0x3fd45f306e000000ul,0xbdeb1bbead603d8bul,
    0x3fd45f306e000000ul,0xbdeb1bbead603d8bul,0x3fd45f306e000000ul,0xbdeb1bbead603d8bul,
    0xbfc7419f24000000ul,0xbdeb1bbead603d8bul,0xbfad067c92000000ul,0x3dc391054a7f09d6ul,
    0x3f77cc1b72000000ul,0x3d9c882a53f84eb0ul,0xbf6067c91b000000ul,0xbd6bbead603d8a83ul,
    0xbf09f246c7000000ul,0x3d0054a7f09d5f48ul,0xbf09f246c7000000ul,0x3d0054a7f09d5f48ul,
    0xbf09f246c7000000ul,0x3d0054a7f09d5f48ul,0x3ee836e4e4000000ul,0x3d0054a7f09d5f48ul,
    0xbecf246c6f000000ul,0x3ca529fc2757d1f5ul,0x3e7b727221000000ul,0xbc95ac07b1505c16ul,
    0x3e7b727221000000ul,0xbc95ac07b1505c16ul,0xbe5236377d000000ul,0xbc76b01ec5417056ul,
    0x3e4b939105000000ul,0x3c629fc2757d1f53ul,0xbe21b1bbeb000000ul,0x3c34fe13abe8fa9aul,
    0xbdeb1bbead000000ul,0xbc080f62a0b82b2dul,0xbdeb1bbead000000ul,0xbc080f62a0b82b2dul,
    0x3dc391054a000000ul,0x3befc2757d1f534eul,0x3d9c882a54000000ul,0xbb7ec54170565912ul,
    0xbd6bbead60000000ul,0xbb7ec54170565912ul,0xbd6bbead60000000ul,0xbb7ec54170565912ul,
    0x3d41054a7f000000ul,0x3b33abe8fa9a6ee0ul,0x3d0054a7f1000000ul,0xbb28a82e0acb223ful,
    0x3d0054a7f1000000ul,0xbb28a82e0acb223ful,0x3ca529fc27000000ul,0x3ac5f47d4d377037ul,
    0x3ca529fc27000000ul,0x3ac5f47d4d377037ul,0x3ca529fc27000000ul,0x3ac5f47d4d377037ul,
    0x3c84a7f09d000000ul,0x3aa7d1f534ddc0dbul,0x3c629fc275000000ul,0x3a8f47d4d377036eul,
    0x3c34fe13ac000000ul,0xba370565911f924ful,0xbc2603d8a8000000ul,0xba370565911f924ful,
    0xbc080f62a1000000ul,0x3a21f534ddc0db63ul,0x3befc2757d000000ul,0x39ef534ddc0db629ul,
    0xbb7ec54170000000ul,0xb99596447e493ad5ul,0xbb7ec54170000000ul,0xb99596447e493ad5ul,
    0xbb7ec54170000000ul,0xb99596447e493ad5ul,0x3b33abe8fb000000ul,0xb9596447e493ad4dul,
    0x3b33abe8fb000000ul,0xb9596447e493ad4dul,0xbb28a82e0b000000ul,0x393a6ee06db14acdul,
    0x3b0d5f47d5000000ul,0xb916447e493ad4ceul,0xbad505c159000000ul,0xb8f911f924eb5336ul,
    0x3ac5f47d4d000000ul,0x38dbb81b6c52b328ul,0x3aa7d1f535000000ul,0xb8b11f924eb53362ul,
    0x3a8f47d4d3000000ul,0x38adc0db6295993cul,0xba37056591000000ul,0xb83f924eb53361deul,
    0xba37056591000000ul,0xb83f924eb53361deul,0x3a21f534de000000ul,0xb83f924eb53361deul,
    0x39ef534ddc000000ul,0x37db6c52b3278872ul,0x39ef534ddc000000ul,0x37db6c52b3278872ul,
    0xb99596447e000000ul,0xb7b24eb53361de38ul,0xb99596447e000000ul,0xb7b24eb53361de38ul,
    0x3984d37703000000ul,0x37ab6295993c4390ul,0x39634ddc0e000000ul,0xb78275a99b0ef1bful,

};
inline ulong _eon_trig32_sine_poly(ulong value){
    ulong q=d_mul(value,value),cube=d_mul(value,q);
    ulong p=stable_fma(q,0xbf29943e0fc4818ful,0x3f811073afd145ebul);
    ulong fifth=d_mul(q,cube);
    ulong leading=stable_fma(cube,0xbfc555545268a02aul,value);
    return stable_fma(fifth,p,leading);
}
inline ulong _eon_trig32_cosine_poly(ulong value){
    ulong q=d_mul(value,value),q2=d_mul(q,q);
    ulong p=stable_fma(q,0x3ef99169fc938f4aul,0xbf56c07f16945df4ul);
    ulong leading=stable_fma(q,0x3fa55553c7899699ul,0xbfdffffffcb82e97ul);
    p=stable_fma(q2,p,leading);
    return stable_fma(q,p,0x3ff0000000000000ul);
}
inline uint _eon_trig32_large(ulong value,uint absolute,bool cosine){
    uint row=(absolute>>24)-0x40u;
    ulong high=d_mul(value,_eon_trig32_pi_reciprocal[2*row]);
    ulong low=d_mul(value,_eon_trig32_pi_reciprocal[2*row+1]);
    if(cosine)high=d_add(high,0x3fe0000000000000ul);
    ulong nearest=d_roundeven(high);
    long n=d_to_int(nearest);
    ulong reduced=d_add(d_sub(high,nearest),low),signed_reduced=reduced^(ulong(n)<<63);
    ulong q=d_mul(reduced,reduced);
    ulong a=d_add(q,0xc010a8b9fc97b22cul),b=d_add(q,0xc00d31f4a0a55230ul);
    ulong c=stable_fma(q,a,0x402ef88f0b1250c1ul),d=stable_fma(q,b,0x400537b6744fa779ul);
    c=d_mul(signed_reduced,c);d=d_mul(0x3fb39502de83d49cul,d);
    return f_from_d(d_mul(c,d));
}
inline long _eon_trig32_medium_integer(uint raw,thread long& count){
    uint absolute=raw&0x7fffffffu;
    // The float estimate chooses a quadrant only when its complete rounding
    // interval excludes either half-integer boundary. Its conservative error
    // is below2^(E-22), including the frozen double reciprocal's rounding.
    float estimate=as_type<float>(absolute)*0x1.45f306p-1f;
    float nearest=rint(estimate);
    float error=as_type<float>(((absolute>>23)-22u)<<23);
    long magnitude;
    if(fabs(estimate-nearest)<0.5f-error)magnitude=long(nearest);
    else magnitude=d_to_int(d_roundeven(d_mul(f_to_d(absolute),0x3fe45f306dc9c883ul)));
    count=(raw>>31)?-magnitude:magnitude;
    // pi/2's frozen double is exactly0x3243f6a8885a3 *2^-49.
    // For pi/4<=|x|<120, exponent E is[-1,6] and magnitude<=76.
    // Both scaled operands fit signed57bits; their exact difference is below
    // 2^49, so its conversion and scaling to double are exact. This reproduces
    // the CPU's single-round FMA, including cancellation, without emulating it.
    uint exponent=(absolute>>23)&255u;
    ulong significand=ulong((absolute&0x7fffffu)|0x800000u);
    long reduced=long(significand<<int(exponent-101u))-magnitude*0x3243f6a8885a3L;
    if(raw>>31)reduced=-reduced;
    return reduced;
}
inline ulong _eon_trig32_medium(uint raw,thread long& count){
    long reduced=_eon_trig32_medium_integer(raw,count);
    return reduced?d_from_int(reduced)-(49ul<<52):0ul;
}
// Clang combines same-block sinf/cosf calls into __sincosf_stret. Its
// paired polynomial differs from the individually called scalar functions.
struct _eon_trig32_pair { uint sine; uint cosine; };
inline _eon_trig32_pair _eon_trig32_sincos_scalar(uint x){
    uint absolute=x&0x7fffffffu;
    if(absolute>=0x7f800000u){uint value=f_sub(x,x);return {value,value};}
    if(absolute<0x39800000u)return {x,0x3f800000u};
    ulong value=f_to_d(x),high,low=0ul;
    if(absolute<0x42f00000u)high=d_mul(value,0x3fe45f306dc9c883ul);
    else{
        uint row=(absolute>>24)-0x40u;
        value=d_add(value,value);
        high=d_mul(value,_eon_trig32_pi_reciprocal[2*row]);
        low=d_mul(value,_eon_trig32_pi_reciprocal[2*row+1]);
    }
    ulong nearest=d_roundeven(high);
    long n=d_to_int(nearest);
    ulong reduced=d_sub(high,nearest);
    if(absolute>=0x42f00000u)reduced=d_add(reduced,low);
    ulong q=d_mul(reduced,reduced);
    ulong ps=stable_fma(q,d_add(q,0xc02c3d3ee4f8713ful),0x40444921a30f0d80ul);
    ulong pc=stable_fma(q,d_add(q,0xc020bc0396540c13ul),0x401d78212789bc11ul);
    ulong zs=stable_fma(q,d_add(q,0xc02ef09ab49e3a5bul),0x406e97264e36bd93ul);
    ulong zc=stable_fma(q,d_add(q,0xc02d6f5ade6c8c37ul),0x4062c6b9c7f88fa0ul);
    uint sine=f_from_d(d_mul(d_mul(d_mul(reduced,0x3f24bc899499342cul),ps),zs));
    uint cosine=f_from_d(d_mul(d_mul(0x3f4d9c364e9721d5ul,pc),zc));
    if(n&2L){sine^=0x80000000u;cosine^=0x80000000u;}
    return (n&1L)?_eon_trig32_pair{cosine,sine^0x80000000u}:_eon_trig32_pair{sine,cosine};
}
inline uint _eon_trig32_sine_scalar(uint x){
    uint absolute=x&0x7fffffffu;
    if(absolute>=0x7f800000u)return f_sub(x,x);
    // Both tiny scalar instructions round back to the unchanged input, even
    // for subnormals. Returning raw bits avoids Metal denormal flushing.
    if(absolute<0x39800000u)return x;
    ulong value=f_to_d(x);
    if(absolute<0x3f490fdbu)return f_from_d(_eon_trig32_sine_poly(value));
    if(absolute<0x42f00000u){
        long n;ulong reduced=_eon_trig32_medium(x,n);
        ulong result=(n&1L)?_eon_trig32_cosine_poly(reduced):_eon_trig32_sine_poly(reduced);
        return f_from_d(result^((ulong(n)>>1)<<63));
    }
    return _eon_trig32_large(value,absolute,false);
}

inline uint _eon_trig32_cosine_scalar(uint x){
    uint absolute=x&0x7fffffffu;
    if(absolute>=0x7f800000u)return f_sub(absolute,absolute);
    if(absolute<0x39800000u)return 0x3f800000u;
    ulong value=f_to_d(absolute);
    if(absolute<0x3f490fdbu)return f_from_d(_eon_trig32_cosine_poly(value));
    if(absolute<0x42f00000u){
        long n;ulong reduced=_eon_trig32_medium(absolute,n);++n;
        ulong result=(n&1L)?_eon_trig32_cosine_poly(reduced):_eon_trig32_sine_poly(reduced);
        return f_from_d(result^((ulong(n)>>1)<<63));
    }
    if(absolute<0x4c800000u){
        ulong nearest=d_roundeven(d_mul(value,0x3fe45f306dc9c883ul));
        long n=d_to_int(nearest)+1L;
        ulong reduced=stable_fma(d_neg(nearest),0x3ff921fb54442d18ul,value);
        if(absolute>=0x42f00000u)reduced=stable_fma(d_neg(nearest),0x3c91a62633145c07ul,reduced);
        ulong result=(n&1L)?_eon_trig32_cosine_poly(reduced):_eon_trig32_sine_poly(reduced);
        return f_from_d(result^((ulong(n)>>1)<<63));
    }
    return _eon_trig32_large(value,absolute,true);
}
inline uint _eon_trig32_cosine_half(uint raw){
    // NumPy's half loop preserves a NaN input's sign/payload while quieting
    // it; scalar Apple cosf instead operates on the absolute argument.
    if((raw&0x7fffffffu)>0x7f800000u)return raw|0x400000u;
    return _eon_trig32_cosine_scalar(raw);
}
template<bool Cosine> inline uint _eon_trig32_numpy(uint raw){
    uint absolute=raw&0x7fffffffu;
    if(absolute>0x7f800000u)return 0x7fc00000u;
    if(absolute<0x39800000u)return Cosine?0x3f800000u:raw;
    if(absolute>(Cosine?0x478b9a08u:0x47e55dffu))
        return Cosine?_eon_trig32_cosine_scalar(raw):_eon_trig32_sine_scalar(raw);
    float x=as_type<float>(raw);
    float q=x*0x1.45f306p-1f;
    q=q+0x1.800000p+23f;q=q-0x1.800000p+23f;
    float r=fma(q,-0x1.921fb0p+0f,x);
    r=fma(q,-0x1.5110b4p-22f,r);r=fma(q,-0x1.846988p-48f,r);
    float square=r*r;
    float cosine=fma(0x1.98e616p-16f,square,-0x1.6c06dcp-10f);
    cosine=fma(cosine,square,0x1.55553cp-5f);
    cosine=fma(cosine,square,-0x1p-1f);cosine=fma(cosine,square,1.0f);
    float sine=fma(0x1.7d3bbcp-19f,square,-0x1.a06bbap-13f);
    sine=fma(sine,square,0x1.11119ap-7f);sine=fma(sine,square,-0x1.555556p-3f);
    sine=fma(sine,square,0.0f);sine=fma(sine,r,r);
    int quadrant=int(q)+int(Cosine);
    float result=(quadrant&1)?cosine:sine;
    if(quadrant&2)result=0.0f-result;
    return as_type<uint>(result);
}
inline uint _eon_trig32_sine_numpy(uint raw){return _eon_trig32_numpy<false>(raw);}
inline uint _eon_trig32_cosine_numpy(uint raw){return _eon_trig32_numpy<true>(raw);}
'''

_TAN_HEADER = r'''
constant ulong _eon_trig32_tan_small[7]={
    0x3f6fe5da804efbdcul,0x3f51e9c887f6c8a1ul,0x3f8487da95c9f173ul,0x3f961eb51f86747cul,
    0x3faba586a7f44ae7ul,0x3fc1110746e09b07ul,0x3fd555555d8d8930ul,
};
constant ulong _eon_trig32_tan_coeff[8]={
    0x40151d4a014005d8ul,0x400e91f74c2eeaabul,0xbfd609394a49adf1ul,0xbff88f856cde0b64ul,
    0x3fed3764811481d2ul,0x3fe81ba4203c502dul,0x3fedb0d5be6b368bul,0x3fe2d17a5cc6b70bul,
};
constant ulong _eon_trig32_tan_reciprocal[106]={
    0x3fe45f306e000000ul,0xbdfb1bbead603d8bul,0x3fc17cc1b7000000ul,0x3dd391054a7f09d6ul,
    0x3f87cc1b72000000ul,0x3dac882a53f84eb0ul,0x3f87cc1b72000000ul,0x3dac882a53f84eb0ul,
    0x3f6f306dca000000ul,0xbd7bbead603d8a83ul,0xbf19f246c7000000ul,0x3d1054a7f09d5f48ul,
    0xbf19f246c7000000ul,0x3d1054a7f09d5f48ul,0x3ef836e4e4000000ul,0x3d1054a7f09d5f48ul,
    0xbedf246c6f000000ul,0x3cb529fc2757d1f5ul,0x3e8b727221000000ul,0xbca5ac07b1505c16ul,
    0x3e8b727221000000ul,0xbca5ac07b1505c16ul,0x3e8b727221000000ul,0xbca5ac07b1505c16ul,
    0xbe6236377d000000ul,0xbc86b01ec5417056ul,0xbe31b1bbeb000000ul,0x3c44fe13abe8fa9aul,
    0x3e2c9c882a000000ul,0x3c44fe13abe8fa9aul,0xbdfb1bbead000000ul,0xbc180f62a0b82b2dul,
    0x3dd391054a000000ul,0x3bffc2757d1f534eul,0xbdc8ddf56b000000ul,0xbb8ec54170565912ul,
    0x3dac882a54000000ul,0xbb8ec54170565912ul,0xbd7bbead60000000ul,0xbb8ec54170565912ul,
    0x3d51054a7f000000ul,0x3b43abe8fa9a6ee0ul,0xbd4df56b02000000ul,0x3b43abe8fa9a6ee0ul,
    0x3d1054a7f1000000ul,0xbb38a82e0acb223ful,0xbd0f56b01f000000ul,0x3b1d5f47d4d37703ul,
    0x3cb529fc27000000ul,0x3ad5f47d4d377037ul,0x3cb529fc27000000ul,0x3ad5f47d4d377037ul,
    0xbca5ac07b1000000ul,0xbac4170565911f92ul,0xbc86b01ec5000000ul,0xbaa05c1596447e49ul,
    0xbc6ac07b15000000ul,0xba470565911f924ful,0x3c44fe13ac000000ul,0xba470565911f924ful,
    0x3c23f84eb0000000ul,0xba470565911f924ful,0x3bffc2757d000000ul,0x39ff534ddc0db629ul,
    0xbb8ec54170000000ul,0xb9a596447e493ad5ul,0xbb8ec54170000000ul,0xb9a596447e493ad5ul,
    0xbb8ec54170000000ul,0xb9a596447e493ad5ul,0xbb8ec54170000000ul,0xb9a596447e493ad5ul,
    0x3b43abe8fb000000ul,0xb9696447e493ad4dul,0x3b43abe8fb000000ul,0xb9696447e493ad4dul,
    0x3b1d5f47d5000000ul,0xb926447e493ad4ceul,0xbae505c159000000ul,0xb90911f924eb5336ul,
    0xbae505c159000000ul,0xb90911f924eb5336ul,0xbac4170566000000ul,0x38ebb81b6c52b328ul,
    0xbaa05c1596000000ul,0xb8c11f924eb53362ul,0xba47056591000000ul,0xb84f924eb53361deul,
    0xba47056591000000ul,0xb84f924eb53361deul,0xba47056591000000ul,0xb84f924eb53361deul,
    0xba2c159644000000ul,0xb84f924eb53361deul,0x39ff534ddc000000ul,0x37eb6c52b3278872ul,
    0xb9a596447e000000ul,0xb7c24eb53361de38ul,0xb9a596447e000000ul,0xb7c24eb53361de38ul,
    0xb9a596447e000000ul,0xb7c24eb53361de38ul,0xb9865911f9000000ul,0xb79275a99b0ef1bful,
    0xb9696447e5000000ul,0x378b14acc9e21c82ul,
};
inline uint _eon_trig32_tangent_scalar(uint x){
    uint absolute=x&0x7fffffffu;
    if(absolute>=0x7f800000u)return f_sub(x,x);
    if(absolute<0x39800000u)return x;
    ulong value=f_to_d(x);
    if(absolute<0x3f490fdbu){
        ulong q=d_mul(value,value),q2=d_mul(q,q),q4=d_mul(q2,q2);
        ulong a=stable_fma(q,_eon_trig32_tan_small[1],_eon_trig32_tan_small[2]);
        ulong b=stable_fma(q,_eon_trig32_tan_small[3],_eon_trig32_tan_small[4]);
        ulong c=stable_fma(q,_eon_trig32_tan_small[5],_eon_trig32_tan_small[6]);
        a=stable_fma(q2,_eon_trig32_tan_small[0],a);
        c=stable_fma(q2,b,c);c=stable_fma(q4,a,c);
        return f_from_d(stable_fma(value,d_mul(q,c),value));
    }
    ulong nearest,reduced;
    if(absolute<0x4c800000u){
        ulong high=d_mul(value,0x3fe45f306e000000ul);
        nearest=d_roundeven(high);
        reduced=stable_fma(value,0xbdfb1bbead603d8bul,d_sub(high,nearest));
    }else{
        int row=max(int(absolute>>24)-0x4b,0);
        ulong high=d_mul(value,_eon_trig32_tan_reciprocal[2*row]);
        ulong low=d_mul(value,_eon_trig32_tan_reciprocal[2*row+1]);
        nearest=d_sub(d_add(high,0x4338000000000000ul),0x4338000000000000ul);
        reduced=d_add(d_sub(high,nearest),low);
    }
    long n=d_to_int(nearest);
    ulong q=d_mul(reduced,reduced);
    ulong a=stable_fma(q,_eon_trig32_tan_coeff[0],_eon_trig32_tan_coeff[1]);
    ulong b=d_add(q,_eon_trig32_tan_coeff[2]);
    ulong c=d_add(q,_eon_trig32_tan_coeff[3]);
    ulong d=d_add(q,_eon_trig32_tan_coeff[4]);
    a=d_mul(reduced,a);
    b=stable_fma(q,b,_eon_trig32_tan_coeff[5]);
    c=stable_fma(q,c,_eon_trig32_tan_coeff[6]);
    d=stable_fma(q,d,_eon_trig32_tan_coeff[7]);
    a=d_mul(a,b);c=d_mul(c,d);
    ulong result=d_mul(a,c);
    if(n&1L)result=stable_div(0xbff0000000000000ul,result);
    return f_from_d(result);
}
inline uint _eon_trig32_tangent_numpy(uint x){return _eon_trig32_tangent_scalar(x);}
'''


def metal_trigonometry32_header(functions):
    names=set(functions)
    if names-{"sin","cos","tan"}:
        raise NotImplementedError("Binary32 trigonometry development currently implements sine/cosine/tangent")
    sine_header = _SINE_HEADER if names & {"sin", "cos"} else ""
    if "sin" in names:
        from .trigonometry32_certificate import MSL_HEADER
        sine_header = sine_header.replace(
            "inline uint _eon_trig32_sine_scalar(uint x)",
            "__attribute__((noinline)) uint _eon_trig32_sine_exact(uint x)")
        marker = "inline uint _eon_trig32_cosine_scalar(uint x)"
        sine_header = sine_header.replace(marker, MSL_HEADER+"\n"+marker)
    return ((_STABLE_FMA_HEADER if names else "")
            + (_STABLE_DIV_HEADER+_TAN_HEADER if "tan" in names else "")
            + sine_header)


def scalar_sine_threadgroup(program, size):
    """Use smaller groups for small shaders that actually call scalar sinf."""
    stages = (program.scalar_code, program.state_code, program.threshold_code,
              program.reset_code, program.refractory_code, program.dynamics_code,
              program.subexpression_code)
    scalar_sine = size <= 16384 and any(
        "_eon_trig32_sine_scalar(" in code for code in stages)
    return min(64 if scalar_sine else 256, size)
