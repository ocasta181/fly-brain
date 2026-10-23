"""Binary32 arctangent preserving the actual macOS ARM64 atanf result.

The CPU float overload evaluates its polynomial in binary64 before narrowing.
This independent Metal helper retains that operation order and its biased
large-input endpoint, including signed zeros and NaN payloads. Include the
binary64, binary32 and math64 headers first, with safe Metal math.
"""

__all__ = ["metal_arctangent32_header"]

_HEADER = r"""
constant ulong arctan32_small[8]={0x4002609372076e2cul,0x3f953db49b9fcf6bul,0xc00792d5bf17c0bcul,0xc013e370146726b3ul,0x40038f4c070b63c1ul,0x400e4fa479729042ul,0x4015e43168d7c2b6ul,0x401ae0fb08f2dc00ul};
constant ulong arctan32_large[8]={0x3fed6b74d6295266ul,0xbf75505d3cf68fb0ul,0xbfe0dfcb5ed76ca4ul,0xbfe70baa17f25493ul,0x3fd98dc25f02f8e9ul,0x3fd022d8b8d718a9ul,0x3fc626645186b1ceul,0x3fc207f5efa5231dul};

inline ulong arctan32_product(ulong z,constant ulong (&c)[8]) {
    ulong q0=d_fma(z,d_add(z,c[0]),c[4]);
    ulong q1=d_fma(z,d_add(z,c[1]),c[5]);
    ulong q2=d_fma(z,d_add(z,c[2]),c[6]);
    ulong q3=d_fma(z,d_add(z,c[3]),c[7]);
    return d_mul(d_mul(q0,q1),d_mul(q2,q3));
}
inline uint _eon_arctan32(uint input) {
    uint a=input&0x7fffffffu, sign=input&0x80000000u;
    if(a>0x7f800000u) return input|0x00400000u;
    // These are the reference's actual float branch boundaries. Its endpoint
    // is one float below nearest pi/2, including at infinity; do not replace
    // this observable CPU behavior with a mathematically rounded constant.
    if(a>=0x4ae1e273u) return sign|0x3fc90fdau;
    if(a<0x32000000u) return input;
    if(a<=0x3f800000u) {
        ulong x=f_to_d(input),z=d_mul(x,x);
        ulong scaled=d_mul(x,0x3f680bc0fe23203dul);
        return f_from_d(d_mul(scaled,arctan32_product(z,arctan32_small)));
    }
    ulong x=f_to_d(a),inv=d_div(0x3ff0000000000000ul,x);
    ulong power=d_mul(inv,inv);
    power=d_mul(power,power);power=d_mul(power,power);power=d_mul(power,power);
    ulong weighted=d_mul(inv,arctan32_product(d_mul(x,x),arctan32_large));
    ulong result=d_fma(weighted^0x8000000000000000ul,power,0x3ff921fb54442d18ul);
    return sign|f_from_d(result);
}
"""


def metal_arctangent32_header(names):
    """Return only this function's binary32 implementation."""
    if "arctan" not in set(names):
        return ""
    from .stable_math64 import _STABLE_FMA_HEADER
    # The reference float overload evaluates its polynomial in binary64;
    # share the exact guarded two-limb FMA used by the other stable helpers.
    return _STABLE_FMA_HEADER + _HEADER.replace("d_fma(", "stable_fma(")
