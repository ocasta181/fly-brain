"""Native binary64 arctangent for the pinned macOS ARM64 CPU reference.

Inputs and outputs are raw IEEE-754 bits. The polynomial coefficients and
rounded operation order follow the actual scalar libsystem_m arctangent;
NumPy/Brian uses the same results on this reference platform. All arithmetic
executes on Metal via the existing integer binary64 and fused arithmetic.
Include binary64.MSL_HEADER and math64.MSL_MATH_HEADER first; use safe math.
No host evaluation or forward-trigonometry implementation is required.
"""

__all__ = ["metal_arctangent_header"]

_HEADER = r"""
constant ulong arctan64_small[12]={
    0xbf9c0eb85f543412ul,0x3f84a9f5c4724056ul,0x3fa56cdb5d887934ul,0xbfaa38cf590469ecul,
    0x3fadfe7b9674ae37ul,0xbfb10f31279ec05dul,0x3fb3b113f18ac049ul,0xbfb745cf51795b21ul,
    0x3fbc71c7096c2725ul,0xbfc2492492179ca3ul,0x3fc99999999918d8ul,0xbfd555555555551bul
};
constant ulong arctan64_mid0[13]={
    0xbf8d88d31abc3ae5ul,0x3f710f3e20f6a2e2ul,0x3f8950e69dcdd967ul,0x3f7ff3b7c531aa4aul,
    0xbfa09bc0ab7f914cul,0x3fa094966be2b531ul,0x3f8a759263f377f2ul,0xbfb519e110f61b54ul,
    0x3fba1247ca5d9475ul,0x3f95d0b7e9e69054ul,0xbfd4af2b78215a1bul,0x3fe702e05c0b8155ul,
    0x3fe1e00babdefed0ul
};
constant ulong arctan64_mid1[12]={
    0xbf7265ef51b17db7ul,0x3f1ca7cde5de9bd7ul,0x3f8152311b180e6cul,0xbf7ac1645739e676ul,
    0xbf791f7e2a7a338ful,0x3f9c90e92ac8d86cul,0xbfa63b543effa4eful,0x3f959bc93f81895aul,
    0x3fb41b15e5e8dcd0ul,0xbfd1f6a8499714a2ul,0x3fe21fb781196ac3ul,0x3fe700a7c580ea7eul
};
constant ulong arctan64_mid2[13]={
    0x3f4a89224ff69018ul,0xbf3883d8959134b3ul,0xbf50120e602f6336ul,0x3f2cbd49da316282ul,
    0x3f62fcdacdd6e5b5ul,0xbf7c0b993a09ce31ul,0x3f873a9328786665ul,0xbf82b01fc60cc37aul,
    0xbf8bc317394714b7ul,0x3fb3fd2b9f586a67ul,0xbfcac97826d58470ul,0x3fdb1b1b1b1b1b3dul,
    0x3feb96e5a78c5c40ul
};
constant ulong arctan64_mid3[12]={
    0x3f1b4e30d1ba3819ul,0x3f00280537f097f3ul,0xbf3e4eef429eb680ul,0x3f525bad4d85cbe1ul,
    0xbf5f4ac1342182d2ul,0x3f5dfaa5e77b7375ul,0x3f613a254d6e5b7cul,0xbf9135a0938ec462ul,
    0x3fac963c83985742ul,0xbfc22d719c06115eul,0x3fd3b13b13b13b0cul,0x3fef730bd281f69dul
};
constant ulong arctan64_mid4[11]={
    0xbf23045b70e93129ul,0x3f100b6a460ac05dul,0x3f310f884eac0e0aul,0xbf32b9ad13db35a8ul,
    0xbf33182219e21362ul,0x3f68cb82a74e0699ul,0xbf8881ec3d15241ful,0x3fa2b090aad5f9dcul,
    0xbfb8ad3c44f10dc3ul,0x3fcd59ae78c11c49ul,0x3ff124a85750fb5cul
};
constant ulong arctan64_large[11]={
    0x3fa007733e06ceb3ul,0xbf883dafda7bd3fdul,0xbfa81d33e401836dul,0x3fad78252fa69c1cul,
    0xbfb104146b1a1ae8ul,0x3fb3afd7a0e6eb75ul,0xbfb745b7f2d72663ul,0x3fbc71c5edfed480ul,
    0xbfc249248e1422e3ul,0x3fc999999989ebcaul,0xbfd5555555554a51ul
};

// The first two coefficients form c[1]*x+c[0]. Every subsequent
// multiply-add rounds once, including the final linear reconstruction.
template<int N>
inline ulong arctan64_polynomial(ulong x,constant ulong (&c)[N]) {
    ulong p=d_fma(x,c[1],c[0]);
    for(int i=2;i<N;++i) p=d_fma(p,x,c[i]);
    return p;
}
inline ulong _eon_arctan64(ulong input) {
    if(d_is_nan(input)) return input|0x0008000000000000ul;
    ulong sign=input&0x8000000000000000ul;
    ulong x=d_abs(input),r;
    if(x<=0x3fe0000000000000ul) {
        if(x<=0x3e4d12ed0af1a27ful) return input;
        ulong z=d_mul(x,x);
        r=d_fma(d_mul(z,arctan64_polynomial(z,arctan64_small)),x,x);
    } else if(x<=0x3fe8000000000000ul) {
        r=arctan64_polynomial(d_add(x,0xbfe4000000000027ul),arctan64_mid0);
    } else if(x<=0x3ff0000000000000ul) {
        r=arctan64_polynomial(d_add(x,0xbfec0000000f4213ul),arctan64_mid1);
    } else if(x<=0x3ff5555555555555ul) {
        r=arctan64_polynomial(d_add(x,0xbff2aaaaaaaaaa96ul),arctan64_mid2);
    } else if(x<=0x3ffaaaaaaaaaaaabul) {
        r=arctan64_polynomial(d_add(x,0xbff8000000000003ul),arctan64_mid3);
    } else if(x<=0x4000000000000000ul) {
        r=arctan64_polynomial(d_add(x,0xbffd555555461337ul),arctan64_mid4);
    } else {
        if(x>=0x4340000000000000ul) r=0x3ff921fb54442d18ul;
        else {
            ulong inv=d_div(0x3ff0000000000000ul,x),z=d_mul(inv,inv);
            ulong tail=d_fma(d_mul(z,arctan64_polynomial(z,arctan64_large)),inv,inv);
            r=d_add(d_sub(0x3c91a62633145c07ul,tail),0x3ff921fb54442d18ul);
        }
    }
    return r^sign;
}
"""


def metal_arctangent_header(names):
    """Select the independent binary64 arctangent helper only when requested."""
    if "arctan" not in set(names):
        return ""
    from .stable_math64 import _STABLE_FMA_HEADER
    # Reuse the exact 106-bit-product FMA. Its exceptional-operand fallback
    # preserves signed zero, subnormals and NaNs without the general
    # three-limb alignment cost in every normal polynomial operation.
    return _STABLE_FMA_HEADER + _HEADER.replace("d_fma(", "stable_fma(")
