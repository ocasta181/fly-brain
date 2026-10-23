"""Raw IEEE binary16 temporary conversions; persistent state keeps its dtype.

Integer conversion preserves subnormals under Metal's floating-point FTZ mode.
Round-to-nearest-even and NaN payloads follow the pinned NumPy casts.
Include binary64.MSL_HEADER before this selective header.
"""

MSL_HEADER = r'''
#ifndef EON_BINARY16_INCLUDED
#define EON_BINARY16_INCLUDED
inline ulong h_to_d(uint raw){
    ulong sign=ulong(raw&0x8000u)<<48;
    uint exponent=(raw>>10)&31u,mantissa=raw&1023u;
    if(exponent==31u)return sign|D_INF|(ulong(mantissa)<<42);
    if(exponent==0u){
        if(!mantissa)return sign;
        int leading=31-int(clz(mantissa));
        return sign|(ulong(leading+999)<<52)|((ulong(mantissa)<<(52-leading))&0xffffffffffffful);
    }
    return sign|(ulong(exponent+1008u)<<52)|(ulong(mantissa)<<42);
}
inline uint h_from_d(ulong raw){
    uint sign=uint(raw>>48)&0x8000u;
    int source=int(d_exp(raw)),exponent=source-1008;
    ulong mantissa=d_mant(raw);
    if(source==2047){
        uint payload=uint(mantissa>>42);
        if(mantissa&&!payload)payload=1u;
        return sign|0x7c00u|payload;
    }
    if(exponent>=31)return sign|0x7c00u;
    if(exponent < -10)return sign;
    mantissa|=1ul<<52;
    int shift=exponent>0?42:43-exponent;
    ulong rounded=mantissa>>shift,remainder=mantissa&((1ul<<shift)-1ul);
    ulong halfway=1ul<<(shift-1);
    rounded+=ulong(remainder>halfway||(remainder==halfway&&(rounded&1ul)));
    if(exponent<=0)return sign|uint(rounded);
    if(rounded==2048ul){rounded>>=1;++exponent;}
    return sign|(uint(exponent)<<10)|(uint(rounded)&1023u);
}
inline ulong h_round(ulong raw){return h_to_d(h_from_d(raw));}
#endif // EON_BINARY16_INCLUDED
'''

# A half ufunc calls scalar32 math on the exact widened half value. Ordinary
# f_from_d/f_to_d intentionally quiet NaNs; the half loop's bit casts preserve
# signaling payloads when the scalar function itself preserves them (log10f).
# Include binary32.MSL_HEADER and MSL_HEADER before this selective extension.
MSL_F32_HEADER = r'''
#ifndef EON_BINARY16_F32_INCLUDED
#define EON_BINARY16_F32_INCLUDED
inline uint h_d_to_f(ulong value){
    if(d_exp(value)==2047ul){
        ulong mantissa=d_mant(value);
        uint payload=uint(mantissa>>29);
        if(mantissa&&!payload)payload=1u;
        return (uint(value>>32)&0x80000000u)|0x7f800000u|payload;
    }
    return f_from_d(value);
}
inline ulong h_f32_to_d(uint value){
    if((value&0x7f800000u)==0x7f800000u)
        return (ulong(value&0x80000000u)<<32)|D_INF|(ulong(value&0x7fffffu)<<29);
    return f_to_d(value);
}
#endif // EON_BINARY16_F32_INCLUDED
'''
