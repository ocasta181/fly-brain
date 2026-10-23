"""Actual Apple asinhf operation graph for complex64 inverse functions.

Inputs/outputs are raw binary32 words. Include binary64, binary32, math64
and basic_math32 headers first. Coefficients and log-table identity follow
the pinned ARM64 libsystem_m target; each widened arithmetic boundary is
explicit and all numerical evaluation runs on Metal.
"""

__all__ = ["metal_complex_asinh_header"]

_HEADER = r'''
constant ulong _eon_complex_asinh32_small[4]={
    0x3fd135b26ed287f7ul,0x3f9c6f62ea698057ul,
    0x4009a2cd459a8b35ul,0xbfaaa1182374f4d4ul
};
constant ulong _eon_complex_asinh32_middle[14]={
    0x3f26c17c7b263d18ul,0x3f001e11059bddcaul,
    0x3f6a9bd617bd81f1ul,0x3f5135659e34c549ul,
    0x3f807d5f61f237fcul,0x3f74ccecc3a9e525ul,
    0x3f8b5062b37338e9ul,0x3f8417abb8190857ul,
    0x3f85b32a2f11b40aul,0x3f8e26416e925090ul,
    0x3f80f6e561f06785ul,0x3f85b29a3e277523ul,
    0x3e4328ccef61bd30ul,0x3f80f6f9cf323b3cul
};
constant ulong _eon_complex_asinh32_large[4]={
    0x3fda7158db696b2eul,0xbfa006d5bd0916b3ul,
    0x3fb09db8e7cdd0ceul,0x400ed040335c49d5ul
};
inline uint _eon_complex_real_asinh32(uint bits) {
    uint magnitude=bits&0x7fffffffu,sign=bits&0x80000000u;
    if(magnitude>=0x7f800000u)return f_add(bits,bits);
    if(magnitude<0x32000000u)
        return f_mul(f_fma(bits,0x4d000000u,bits),0x32000000u);
    if(magnitude<0x3e800000u) {
        ulong x=f_to_d(bits),square=d_mul(x,x);
        ulong first=d_add(square,_eon_complex_asinh32_small[0]);
        ulong second=d_fma(square,_eon_complex_asinh32_small[1],_eon_complex_asinh32_small[3]);
        first=d_fma(square,first,_eon_complex_asinh32_small[2]);
        second=d_mul(square,second);
        return f_from_d(d_fma(x,d_mul(first,second),x));
    }
    ulong x=f_to_d(magnitude),value;
    if(magnitude<0x40800000u) {
        ulong numerator=d_fma(x,_eon_complex_asinh32_middle[0],_eon_complex_asinh32_middle[2]);
        ulong denominator=d_fma(x,_eon_complex_asinh32_middle[1],_eon_complex_asinh32_middle[3]);
        for(uint index=4;index<14;index+=2) {
            numerator=d_fma(x,numerator,_eon_complex_asinh32_middle[index]);
            denominator=d_fma(x,denominator,_eon_complex_asinh32_middle[index+1]);
        }
        value=d_div(numerator,denominator);
    }else {
        int exponent=as_type<int>(magnitude-0x3ebf8000u)>>23;
        uint fraction=magnitude&0x7fffffu;
        uint index=(fraction+0x8000u)>>16;
        ulong mantissa=f_to_d(fraction|0x3f800000u);
        ulong residual=d_fma(mantissa,_eon_basic32_p_logf_table[2*index],0xbff0000000000000ul);
        ulong inverse_square=d_div(0x3ff0000000000000ul,d_mul(x,x));
        ulong polynomial=d_fma(residual,_eon_basic32_p_logf_coefficients[0],_eon_basic32_p_logf_coefficients[1]);
        value=d_fma(d_from_int(exponent),_eon_basic32_p_logf_coefficients[3],_eon_basic32_p_logf_table[2*index+1]);
        ulong first=d_fma(inverse_square,_eon_complex_asinh32_large[1],_eon_complex_asinh32_large[2]);
        ulong second=d_fma(inverse_square,d_add(inverse_square,_eon_complex_asinh32_large[0]),_eon_complex_asinh32_large[3]);
        value=d_fma(residual,_eon_basic32_p_logf_coefficients[2],value);
        value=d_fma(d_mul(residual,residual),polynomial,value);
        value=d_fma(inverse_square,d_mul(first,second),value);
    }
    return f_from_d(value)|sign;
}
'''


def metal_complex_asinh_header():
    """Return the exact float32 real asinh helper used by inverse complex math."""
    return _HEADER
