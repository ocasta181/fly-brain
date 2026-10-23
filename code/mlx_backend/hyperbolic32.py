"""Pinned Apple scalar/NumPy float32 log10 and hyperbolic operation sequences.

The scalar implementation follows installed ARM64libsystem_m UUID
83b173cd7f3239018e67c0b2676be015, including its binary64 intermediates.
NumPy tanh follows NumPy1.26.4 loops_hyperbolic.dispatch.c.src;
see third_party/NumPy-hyperbolic-NOTICE and BSD/SVML licenses.
Raw uint interfaces preserve subnormals without Metal float FTZ.
"""

from .scalar_math_tables import exponential32_table

_BASE_HEADER = r'''
inline uint hyper32_round(ulong value) {
    uint sign=uint(value>>32)&0x80000000u;
    int exponent=int(d_exp(value));ulong mantissa=d_mant(value);
    if(exponent==2047) return sign|0x7f800000u|(mantissa ? (0x400000u|uint(mantissa>>29)):0u);
    if(exponent==0) return sign;
    mantissa|=1ul<<52;exponent-=896;
    if(exponent>=255) return sign|0x7f800000u;
    int shift=exponent>0 ? 29:30-exponent;
    if(shift>53) return sign;
    ulong rounded=mantissa>>shift,remainder=mantissa&((1ul<<shift)-1ul),halfway=1ul<<(shift-1);
    if(remainder>halfway || (remainder==halfway && (rounded&1ul))) ++rounded;
    if(exponent<=0) return sign|uint(rounded);
    if(rounded==(1ul<<24)) {rounded>>=1;++exponent;}
    if(exponent>=255) return sign|0x7f800000u;
    return sign|(uint(exponent)<<23)|(uint(rounded)&0x7fffffu);
}
inline ulong hyper32_to_d(uint x) {return d_from_float(as_type<float>(x));}
'''

_EXP_HEADER = "\n" + exponential32_table('hyper32_exp_table') + r'''
constant ulong hyper32_exp_coeff[2]={
    0x3eeebfbdff30d656ul,0x3f762e4453e10daeul,
};

inline ulong hyper32_reduced(ulong x,thread long *k) {
    ulong t=d_mul(x,0x40671547652b82feul);
    *k=d_to_int(t);
    ulong fraction=d_sub(t,d_from_int(*k));
    if(d_abs(fraction)>=0x3fe0000000000000ul) *k+=(t>>63) ? -1:1;
    return d_sub(t,d_from_int(*k));
}
inline ulong hyper32_exp_factor(ulong r,ulong base) {
    ulong q=d_mul(hyper32_fma_normal(hyper32_exp_coeff[0],r,hyper32_exp_coeff[1]),r);
    return hyper32_fma_normal(base,q,base);
}
inline ulong hyper32_half_exp(ulong x) {
    long k;ulong r=hyper32_reduced(x,&k);
    ulong base=hyper32_exp_table[uint(k)&127u]+(ulong(k-128)<<45);
    return hyper32_exp_factor(r,base);
}
'''

_SINH_HEADER = r'''
constant ulong hyper32_sinh_coeff[4]={
    0x3ec7a5f2981bf340ul,0x3f29fff52c4b25fcul,0x3f811112bd052503ul,0x3fc55555544953eeul,
};

inline uint f_hyper_sinh(uint x) {
    uint a=x&0x7fffffffu,sign=x&0x80000000u;
    if(a>0x7f800000u) return x|0x400000u;
    if(a>=0x42b2d4fdu) return sign|0x7f800000u;
    if(a>=0x411a3902u) return hyper32_round(hyper32_half_exp(hyper32_to_d(a)))|sign;
    if(a<0x3f980000u) {
        if(a<0x32000000u) return x;
        ulong v=hyper32_to_d(x),square=d_mul(v,v),p=hyper32_sinh_coeff[0];
        for(uint i=1;i<4;++i) p=hyper32_fma_normal(square,p,hyper32_sinh_coeff[i]);
        return hyper32_round(hyper32_fma_normal(d_mul(v,square),p,v));
    }
    long k;ulong r=hyper32_reduced(hyper32_to_d(x),&k);
    ulong pos=hyper32_exp_table[uint(k)&127u]+(ulong(k)<<45);
    ulong neg=hyper32_exp_table[uint(-k)&127u]+(ulong(-k)<<45);
    pos=hyper32_exp_factor(r,pos);neg=hyper32_exp_factor(d_neg(r),neg);
    return hyper32_round(d_mul(d_sub(pos,neg),0x3fe0000000000000ul));
}
'''

_COSH_HEADER = r'''
constant ulong hyper32_cosh_coeff[4]={
    0x3e92eb3402f9f08ful,0x3efa0005f8e397b6ul,0x3f56c16e044b54a7ul,0x3fa55555549b8f53ul,
};

inline uint f_hyper_cosh(uint x) {
    uint a=x&0x7fffffffu;
    if(a>0x7f800000u) return a|0x400000u;
    if(a>=0x42b2d4fdu) return 0x7f800000u;
    if(a>=0x411b4359u) return hyper32_round(hyper32_half_exp(hyper32_to_d(a)));
    if(a<0x3fb80000u) {
        if(a<0x39800000u) return 0x3f800000u;
        ulong v=hyper32_to_d(x),square=d_mul(v,v),p=hyper32_cosh_coeff[0];
        for(uint i=1;i<4;++i) p=hyper32_fma_normal(square,p,hyper32_cosh_coeff[i]);
        ulong q=d_mul(v,0x3fe0000000000000ul);
        q=hyper32_fma_normal(v,d_mul(square,p),q);
        return hyper32_round(hyper32_fma_normal(v,q,0x3ff0000000000000ul));
    }
    long k;ulong r=hyper32_reduced(hyper32_to_d(x),&k);
    ulong pos=hyper32_exp_table[uint(k)&127u]+(ulong(k)<<45);
    ulong neg=hyper32_exp_table[uint(-k)&127u]+(ulong(-k)<<45);
    pos=hyper32_exp_factor(r,pos);neg=hyper32_exp_factor(d_neg(r),neg);
    return hyper32_round(d_mul(d_add(pos,neg),0x3fe0000000000000ul));
}
'''

_LOG10_HEADER = r'''
constant ulong hyper32_log10_coeff[4]={
    0xbfbbcbea4c258daaul,0x3fc287d3dcaf476dul,0xbfcbcb7b14dcbd85ul,0x3fdbcb7b151bc6b3ul,
};
constant ulong hyper32_log10_table[130]={
    0x3ff0000000000000ul,0x0000000000000000ul,0x3fef81f81f81a3e5ul,0x3f7b9476a50f6475ul,
    0x3fef07c1f07c3108ul,0x3f8b5e908eaf3398ul,0x3fee9131abf09b55ul,0x3f945f4f5acd2447ul,
    0x3fee1e1e1e1e0ae1ul,0x3f9af5f92b020224ul,0x3fedae6076b8f76bul,0x3fa0ba01a81b0d1dul,
    0x3fed41d41d410288ul,0x3fa3ed1199ac1d02ul,0x3fecd8568902fb3ful,0x3fa714834281fa6cul,
    0x3fec71c71c70f481ul,0x3faa30a9d6105e4bul,0x3fec0e0703826caaul,0x3fad41d5164a5b23ul,
    0x3febacf914c177b9ul,0x3fb02428c1f18d98ul,0x3feb4e81b4e7b01ful,0x3fb1a2344551cc7eul,
    0x3feaf286bca2397cul,0x3fb31b3055c23663ul,0x3fea98ef60696bd7ul,0x3fb48f3ed1e3553aul,
    0x3fea41a41a40b6e9ul,0x3fb5fe80488ee159ul,0x3fe9ec8e95113011ul,0x3fb769140a20ed4ful,
    0x3fe9999999992278ul,0x3fb8cf1838884d6eul,0x3fe948b0fcd69907ul,0x3fba30a9d60b536dul,
    0x3fe8f9c18f9b4e11ul,0x3fbb8de4d3aec4d9ul,0x3fe8acb90f6c68a1ul,0x3fbce6e41e442e9eul,
    0x3fe861861862484bul,0x3fbe3bc1ab0aa46ful,0x3fe8181818173ae4ul,0x3fbf8c96834a7e48ul,
    0x3fe7d05f417c0774ul,0x3fc06cbd67a915cdul,0x3fe78a4c81784397ul,0x3fc11142f081f8dbul,
    0x3fe745d1745de7baul,0x3fc1b3e71ec75d90ul,0x3fe702e05c0c18e3ul,0x3fc254b4d35d0f5ful,
    0x3fe6c16c16c06c9bul,0x3fc2f3b691c81020ul,0x3fe681681681a1f6ul,0x3fc390f684497c90ul,
    0x3fe642c8590aa5eful,0x3fc42c7e7fe5304eul,0x3fe605816057fe92ul,0x3fc4c65807e96e6bul,
    0x3fe5c9882b922e81ul,0x3fc55e8c518d5130ul,0x3fe58ed230825723ul,0x3fc5f5244722cd0ful,
    0x3fe555555555be48ul,0xbfbffbfc2bbe9af4ul,0x3fe51d07eae25fe5ul,0xbfbed50a4a23c997ul,
    0x3fe4e5e0a72ea8caul,0xbfbdb11ed764c030ul,0x3fe4afd6a051e80ful,0xbfbc902a19e1cbfdul,
    0x3fe47ae147ad26e5ul,0xbfbb721cd16c4e1cul,0x3fe446f8656250caul,0xbfba56e8325c6c50ul,
    0x3fe4141414137fceul,0xbfb93e7de0f90978ul,0x3fe3e22cbce5a6b7ul,0xbfb828cfed2f2caful,
    0x3fe3b13b13b04947ul,0xbfb715d0ce3125d3ul,0x3fe3813813803a55ul,0xbfb605735ee3df92ul,
    0x3fe3521cfb2b1096ul,0xbfb4f7aad9b97441ul,0x3fe323e34a2baef3ul,0xbfb3ec6ad5440f5aul,
    0x3fe2f684bda0a126ul,0xbfb2e3a740b43dfful,0x3fe2c9fb4d8060d5ul,0xbfb1dd5460c3fb8bul,
    0x3fe29e4129e3187dul,0xbfb0d966cc5f2b50ul,0x3fe27350b881d871ul,0xbfafafa6d3a047d3ul,
    0x3fe249249249e9cful,0xbfadb11ed7700a68ul,0x3fe21fb781227951ul,0xbfabb7209d242ce5ul,
    0x3fe1f7047dc1c36bul,0xbfa9c197abf7fb7aul,0x3fe1cf06ada1ee29ul,0xbfa7d070145824feul,
    0x3fe1a7b961199740ul,0xbfa5e3966b7356cbul,0x3fe181181181a457ul,0xbfa3faf7c669fbf0ul,
    0x3fe15b1e5f74ca8ful,0xbfa21681b5c42118ul,0x3fe135c81135efedul,0xbfa0362241e83be8ul,
    0x3fe1111111104099ul,0xbf9cb38fccc387a4ul,0x3fe0ecf56be75b6eul,0xbf9902c31d763f47ul,
    0x3fe0c9714fbdd2a2ul,0xbf9559bd2420786bul,0x3fe0a6810a68f2a2ul,0xbf91b85d605c7d78ul,
    0x3fe084210841501dul,0xbf8c3d08374fd2e0ul,0x3fe0624dd2f253e0ul,0xbf851824c77c8587ul,
    0x3fe04104104183eeul,0xbf7c03a80b1c8a49ul,0x3fe0204081017ac2ul,0xbf6be76bd7050239ul,
    0x3fe0000000000000ul,0x0000000000000000ul,
};

inline uint f_hyper_log10(uint x) {
    uint a=x&0x7fffffffu;
    if(a>0x7f800000u) return x;
    if(a==0u) return 0xff800000u;
    if(x>>31) return 0x7fc00000u;
    if(a==0x7f800000u) return x;
    int adjust=0;
    if(a<0x800000u) {
        int shift=int(clz(ulong(a)))-40;
        x=((a<<shift)&0x7fffffu)|0x3f800000u;
        adjust=-126-shift;
    }
    int k=(as_type<int>(x+0xc0c10000u)>>23)+adjust;
    uint frac=x&0x7fffffu,index=(frac+0x10000u)>>17;
    ulong m=hyper32_to_d(frac|0x3f800000u);
    ulong delta=hyper32_fma_normal(m,hyper32_log10_table[2*index],0xbff0000000000000ul);
    ulong p=hyper32_log10_coeff[0];
    for(uint i=1;i<4;++i) p=hyper32_fma_normal(delta,p,hyper32_log10_coeff[i]);
    ulong hi=hyper32_fma_normal(d_from_int(k),0x3fd34413509f79fful,hyper32_log10_table[2*index+1]);
    return hyper32_round(hyper32_fma_normal(delta,p,hi));
}
'''

_TANH_SCALAR_HEADER = r'''
constant uint hyper32_tanh_tiny[2]={
    0x3e075604u,0xbeaaa9edu,
};
constant uint hyper32_tanh_coeff[10]={
    0xbbd621d2u,0x3cadd2b4u,0xbd5cb058u,0x3e088701u,
    0xbeaaaaa6u,0xbb883eccu,0x3c991673u,0xbd585d60u,
    0x3e0851d2u,0xbeaaa8aeu,
};

inline uint f_hyper_tanh_scalar(uint x) {
    uint a=x&0x7fffffffu,sign=x&0x80000000u;
    if(a>0x7f800000u) return x|0x400000u;
    if(a>=0x41102cb4u) return sign|0x3f800000u;
    if(a<0x32000000u) return x;
    if(a>=0x3f400000u) {
        ulong v=d_neg(d_mul(hyper32_to_d(a),0x4000000000000000ul));
        long k;ulong r=hyper32_reduced(v,&k);
        ulong base=hyper32_exp_table[uint(k)&127u]+(ulong(k)<<45);
        ulong e=hyper32_exp_factor(r,base);
        return hyper32_round(hyper32_div(d_sub(0x3ff0000000000000ul,e),d_add(0x3ff0000000000000ul,e)))|sign;
    }
    float v=as_type<float>(x),square=v*v,p;
    if(a>=0x3e000000u) {
        uint index=a>=0x3f000000u ? 5u:0u;
        p=as_type<float>(hyper32_tanh_coeff[index]);
        for(uint i=1;i<5;++i) p=fma(square,p,as_type<float>(hyper32_tanh_coeff[index+i]));
    } else p=fma(square,as_type<float>(hyper32_tanh_tiny[0]),as_type<float>(hyper32_tanh_tiny[1]));
    float q=square*p;
    return as_type<uint>(fma(v,q,v));
}
'''

_TANH_NUMPY_HEADER = r'''
constant uint hyper32_tanh_numpy_table[256]={
    0xbc0e2f66u,0x3e0910e9u,0xb76dd6b9u,0xbeaaaaa5u,
    0xb0343c7bu,0x3f800000u,0x00000000u,0x00000000u,
    0x460bda12u,0x43761143u,0xbe1c276du,0xbeab0612u,
    0xbd6ee69du,0x3f7f1f84u,0x3d6fb9c9u,0x3d700000u,
    0x43d638efu,0x4165ecdcu,0x3c1dcf2fu,0xbea7f01fu,
    0xbd8f0da7u,0x3f7ebd11u,0x3d8fc35fu,0x3d900000u,
    0xc3e11c3eu,0xc190f756u,0x3dc1a78du,0xbea4e120u,
    0xbdae477du,0x3f7e1e5fu,0x3daf9169u,0x3db00000u,
    0xc2baa4e9u,0xc08c097du,0x3d96f985u,0xbea387b7u,
    0xbdcd2a1fu,0x3f7d609fu,0x3dcf49abu,0x3dd00000u,
    0xc249da2du,0xc02ba813u,0x3da2b61bu,0xbea15962u,
    0xbdeba80du,0x3f7c842du,0x3deee849u,0x3df00000u,
    0xc1859b82u,0xbf7f6bdau,0x3dc13397u,0xbe9d57f7u,
    0xbe0c443bu,0x3f7b00e5u,0x3e0f0ee8u,0x3e100000u,
    0x40dd5b57u,0x3f2b1dc0u,0x3dd2f670u,0xbe976b5au,
    0xbe293cf3u,0x3f789580u,0x3e2e4984u,0x3e300000u,
    0x40494640u,0x3ece105du,0x3df48a0au,0xbe90230du,
    0xbe44f282u,0x3f75b8adu,0x3e4d2f8eu,0x3e500000u,
    0x40c730a8u,0x3f426a94u,0x3e06c5a8u,0xbe880dffu,
    0xbe5f3651u,0x3f726fd9u,0x3e6bb32eu,0x3e700000u,
    0xbf0f160eu,0xbadb0dc4u,0x3e1a3abau,0xbe7479b3u,
    0xbe81c7c0u,0x3f6cc59bu,0x3e8c51cdu,0x3e900000u,
    0x3e30e76fu,0x3da43b17u,0x3e27c405u,0xbe4c3d88u,
    0xbe96d7cau,0x3f63fb92u,0x3ea96163u,0x3eb00000u,
    0xbea81387u,0xbd51ab88u,0x3e2e78d0u,0xbe212482u,
    0xbea7fb8eu,0x3f59ff97u,0x3ec543f1u,0x3ed00000u,
    0xbdb26a1cu,0xbcaea23du,0x3e2c3e44u,0xbdeb8cbau,
    0xbeb50e9eu,0x3f4f11d7u,0x3edfd735u,0x3ef00000u,
    0xbd351e57u,0xbd3b6d8du,0x3e1d3097u,0xbd5e78adu,
    0xbec12efeu,0x3f3d7573u,0x3f028438u,0x3f100000u,
    0xbb4c01a0u,0xbd6caaadu,0x3df4a8f4u,0x3c6b5e6eu,
    0xbec4be92u,0x3f24f360u,0x3f18abf0u,0x3f300000u,
    0x3c1d7bfbu,0xbd795bedu,0x3da38508u,0x3d839143u,
    0xbebce070u,0x3f0cbfe7u,0x3f2bc480u,0x3f500000u,
    0x3c722cd1u,0xbd5fdddau,0x3d31416au,0x3dc21ee1u,
    0xbead510eu,0x3eec1a69u,0x3f3bec1cu,0x3f700000u,
    0x3c973f1cu,0xbd038f3bu,0x3b562657u,0x3de347afu,
    0xbe8ef7d6u,0x3eb0a801u,0x3f4f2e5bu,0x3f900000u,
    0x3c33a31bu,0xbc1cad63u,0xbcaeeac9u,0x3dcbec96u,
    0xbe4b8704u,0x3e6753a2u,0x3f613c53u,0x3fb00000u,
    0x3b862ef4u,0x3abb4766u,0xbcce9419u,0x3d99ef2du,
    0xbe083237u,0x3e132f1au,0x3f6ce37du,0x3fd00000u,
    0x3a27b3d0u,0x3b95f10bu,0xbcaaeac4u,0x3d542ea1u,
    0xbdaf7449u,0x3db7e7d3u,0x3f743c4fu,0x3ff00000u,
    0xba3b5907u,0x3b825873u,0xbc49e7d0u,0x3cdde701u,
    0xbd2e1ec4u,0x3d320845u,0x3f7a5febu,0x40100000u,
    0xba0efc22u,0x3afaea66u,0xbba71dddu,0x3c2cca67u,
    0xbc83bf06u,0x3c84d3d4u,0x3f7dea85u,0x40300000u,
    0xb97f9f0fu,0x3a49f878u,0xbb003b0eu,0x3b81cb27u,
    0xbbc3e0b5u,0x3bc477b7u,0x3f7f3b3du,0x40500000u,
    0xb8c8af50u,0x39996bf3u,0xba3f9a05u,0x3ac073a1u,
    0xbb10aadcu,0x3b10d3dau,0x3f7fb78cu,0x40700000u,
    0xb7bdddfbu,0x388f3e6cu,0xb92c08a7u,0x39ac3032u,
    0xba0157dbu,0x3a01601eu,0x3f7fefd4u,0x40900000u,
    0xb64f2950u,0x371bb0e3u,0xb7ba9232u,0x383a94d9u,
    0xb88c18f2u,0x388c1a3bu,0x3f7ffdd0u,0x40b00000u,
    0xb4e085b1u,0x35a8a5e6u,0xb64a0b0fu,0x36ca081du,
    0xb717b096u,0x3717b0dau,0x3f7fffb4u,0x40d00000u,
    0xb3731dfau,0x34369b17u,0xb4dac169u,0x355abd4cu,
    0xb5a43baeu,0x35a43bceu,0x3f7ffff6u,0x40f00000u,
    0xb15a1f04u,0x322487b0u,0xb2ab78acu,0x332b3cb6u,
    0xb383012cu,0x338306c6u,0x3f7fffffu,0x41100000u,
    0x00000000u,0x00000000u,0x00000000u,0x00000000u,
    0x00000000u,0x00000000u,0x3f800000u,0x00000000u,
};

inline uint f_hyper_tanh_numpy(uint x) {
    uint a=x&0x7fffffffu,sign=x&0x80000000u;
    if(a>0x7f800000u) return 0x7fc00000u;
    // Below2^-20, |tanh(x)-x|<|x|*2^-40/3; the NumPy cell0 Horner
    // rounds its coefficient to1 and returns x. Keep subnormals in integers.
    if(a<0x35800000u) return x;
    uint ndnan=x&0x7fe00000u;
    if(ndnan>0x7f000000u) return sign|0x3f800000u;
    int idx=as_type<int>(ndnan)-int(0x3d400000u);
    idx=idx<0 ? 0:idx>int(0x3e00000u) ? int(0x3e00000u):idx;
    uint offset=uint(idx>>21)*8u;
    float y=as_type<float>(a)-as_type<float>(hyper32_tanh_numpy_table[offset+7]);
    float p=as_type<float>(hyper32_tanh_numpy_table[offset]);
    for(uint i=1;i<7;++i) p=fma(p,y,as_type<float>(hyper32_tanh_numpy_table[offset+i]));
    return as_type<uint>(p)|sign;
}
'''

def metal_hyperbolic32_header(functions, target="numpy"):
    """Select only requested helpers; binary64 core must precede this header."""
    from .stable_math64 import _STABLE_FMA_HEADER, _STABLE_DIV_HEADER
    from .hyperbolic_math64 import _NORMAL_FMA_HEADER
    requested=set(functions)
    names=requested&{"log10", "sinh", "cosh", "tanh"}
    # NumPy's half ufunc loops use scalar float32 math, while its float32
    # tanh array loop uses a different SIMD polynomial. Both can occur in
    # one expression block and therefore need independently selected helpers.
    scalar_tanh="tanh_scalar" in requested or "tanh" in names and target!="numpy"
    numpy_tanh="tanh_numpy" in requested or "tanh" in names and target=="numpy"
    if scalar_tanh or numpy_tanh:names.add("tanh")
    if not names:
        return ""
    needs64=bool(names&{"log10", "sinh", "cosh"}) or scalar_tanh
    tanh_suffixes=(["_scalar"] if scalar_tanh else [])+(["_numpy"] if numpy_tanh else [])
    wrappers = "".join(f"inline uint _eon_hyper32_{name}{suffix}(uint x) {{ return f_hyper_{name}{suffix}(x); }}\n"
        for name in sorted(names) for suffix in (tanh_suffixes if name == "tanh" else [""]))
    return (_BASE_HEADER
        + ((_STABLE_FMA_HEADER.split("inline ulong stable_fma(",1)[0]
            .replace("EON_STABLE_FMA_INCLUDED", "EON_HYPER32_FMA_INCLUDED")
            .replace("stable_", "hyper32_")
            + _NORMAL_FMA_HEADER.replace("hyper_", "hyper32_")
            + "\n#endif // EON_HYPER32_FMA_INCLUDED\n") if needs64 else "")
        + (_STABLE_DIV_HEADER.replace("stable_", "hyper32_") if scalar_tanh else "")
        + (_EXP_HEADER if names&{"sinh", "cosh"} or scalar_tanh else "")
        + (_LOG10_HEADER if "log10" in names else "")
        + (_SINH_HEADER if "sinh" in names else "")
        + (_COSH_HEADER if "cosh" in names else "")
        + (_TANH_SCALAR_HEADER if scalar_tanh else "")
        + (_TANH_NUMPY_HEADER if numpy_tanh else "") + wrappers)
