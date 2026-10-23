"""Pinned ARM64 floating binary32 operators, with explicit IEEE rounding.

The scalar powf implementation computes its log2 and exp2 polynomials in
binary64. This source preserves those operations before one binary32 round;
using binary64 pow and narrowing would change float32 NumPy/CPP results.
"""

from .scalar_math_tables import exponential32_table

from .operator64 import _OPERATOR_FMA_HEADER, MSL_REMAINDER_HEADER as _REMAINDER64

MSL_UINT_CAST_HEADER=r'''
#ifndef EON_OPERATOR_UINT_CAST_INCLUDED
#define EON_OPERATOR_UINT_CAST_INCLUDED
inline uint f_operator_from_uint(ulong value) {
    if(value<0x8000000000000000ul) return f_from_int(long(value));
    // Halve with a sticky low bit before the signed conversion. This retains
    // the original tie direction, unlike rounding through binary64.
    ulong halved=(value>>1)|(value&1ul);
    return f_from_int(long(halved))+0x00800000u;
}
#endif
'''
# A single direct binary32 rounding avoids a binary64 midpoint double rounding.
_FLOAT_FMA_HEADER=r'''
inline ulong _eon_operator_float_jam(ulong value,int shift) {
    if(shift<=0) return value;
    if(shift>=64) return ulong(value!=0ul);
    return (value>>shift)|ulong((value<<(64-shift))!=0ul);
}
inline uint _eon_operator_float_fma(uint a,uint b,uint c) {
    int ea=int((a>>23)&255u),eb=int((b>>23)&255u),ec=int((c>>23)&255u);
    if(ea==255 || eb==255 || ec==255)
        return f_from_d(d_fma(f_to_d(a),f_to_d(b),f_to_d(c)));
    uint ma=a&0x7fffffu,mb=b&0x7fffffu,mc=c&0x7fffffu;
    uint sign=(a^b)>>31;
    if((ea|ma)==0 || (eb|mb)==0) return f_add(sign<<31,c);
    if((ec|mc)==0) return f_mul(a,b);
    if(ea) ma|=1u<<23;else {int shift=int(clz(ulong(ma)))-40;ma<<=shift;ea=1-shift;}
    if(eb) mb|=1u<<23;else {int shift=int(clz(ulong(mb)))-40;mb<<=shift;eb=1-shift;}
    if(ec) mc|=1u<<23;else {int shift=int(clz(ulong(mc)))-40;mc<<=shift;ec=1-shift;}
    // A binary32 product has48 significant bits. Aligning it at bit62
    // retains its complete product during close cancellation. Once a shift
    // loses any product bit, exponent separation exceeds15bits, so opposite
    // signs cannot expose a lost tail at the24-bit output precision.
    ulong product=ulong(ma)*ulong(mb);
    int top=(product>>47) ? 47:46;
    int pe=ea+eb-300+top,ce=ec-127,exponent=pe>ce ? pe:ce;
    product=_eon_operator_float_jam(product<<(62-top),exponent-pe);
    ulong addend=_eon_operator_float_jam(ulong(mc)<<39,exponent-ce),sum;
    if(sign==(c>>31)) sum=product+addend;
    else {
        if(product<addend) {ulong temporary=product;product=addend;addend=temporary;sign=c>>31;}
        sum=product-addend;
    }
    if(sum==0ul) return 0u;
    int leading=63-int(clz(sum));
    ulong significand=leading>26 ? _eon_operator_float_jam(sum,leading-26):(sum<<(26-leading));
    int output_exponent=exponent+leading-62+127;
    if(output_exponent<=0) {
        int shift=1-output_exponent;
        significand=shift>=64 ? ulong(significand!=0ul):
            ((significand>>shift)|ulong((significand<<(64-shift))!=0ul));
        output_exponent=0;
    }
    ulong remainder=significand&7ul;significand>>=3;
    significand+=ulong(remainder>4ul || (remainder==4ul && (significand&1ul)));
    if(significand>>24) {significand>>=1;++output_exponent;}
    if(output_exponent==0 && (significand>>23)) output_exponent=1;
    if(output_exponent>=255) return (sign<<31)|0x7f800000ul;
    return (sign<<31)|(uint(output_exponent)<<23)|uint(significand&0x7ffffful);
}
'''

_POW_TABLES = r'''
constant ulong _eon_powf_log_coeff[4]={
    0xc0471559dca827ebul,0x404ec71c53b3b1e8ul,0xc05715476529654cul,0x40671547652aef42ul,
};
constant ulong _eon_powf_log_table[256]={
    0x3ff6c16c16c16c17ul,0xc05042bd4b9a7c99ul,0x3ff6a13cd1537290ul,0xc050014332be0033ul,
    0x3ff6816816816817ul,0xc04f804ae8d0cd02ul,0x3ff661ec6a5122f9ul,0xc04efec61b011f85ul,
    0x3ff642c8590b2164ul,0xc04e7df5fe538ab3ul,0x3ff623fa77016240ul,0xc04dfdd89d586e2bul,
    0x3ff6058160581606ul,0xc04d7e6c0abc3579ul,0x3ff5e75bb8d015e7ul,0xc04cffae611ad12bul,
    0x3ff5c9882b931057ul,0xc04c819dc2d45fe4ul,0x3ff5ac056b015ac0ul,0xc04c043859e2fdb3ul,
    0x3ff58ed2308158edul,0xc04b877c57b1b070ul,0x3ff571ed3c506b3aul,0xc04b0b67f4f46810ul,
    0x3ff5555555555555ul,0xc04a8ff971810a5eul,0x3ff5390948f40febul,0xc04a152f142981b4ul,
    0x3ff51d07eae2f815ul,0xc0499b072a96c6b2ul,0x3ff5015015015015ul,0xc04921800924dd3bul,
    0x3ff4e5e0a72f0539ul,0xc048a8980abfbd32ul,0x3ff4cab88725af6eul,0xc048304d90c11fd3ul,
    0x3ff4afd6a052bf5bul,0xc047b89f02cf2aadul,0x3ff49539e3b2d067ul,0xc047418acebbf18ful,
    0x3ff47ae147ae147bul,0xc046cb0f6865c8eaul,0x3ff460cbc7f5cf9aul,0xc046552b49986277ul,
    0x3ff446f86562d9fbul,0xc045dfdcf1eeae0eul,0x3ff42d6625d51f87ul,0xc0456b22e6b578e5ul,
    0x3ff4141414141414ul,0xc044f6fbb2cec598ul,0x3ff3fb013fb013fbul,0xc0448365e695d797ul,
    0x3ff3e22cbce4a902ul,0xc044106017c3eca3ul,0x3ff3c995a47babe7ul,0xc0439de8e1559f6ful,
    0x3ff3b13b13b13b14ul,0xc0432bfee370ee68ul,0x3ff3991c2c187f63ul,0xc042baa0c34be1ecul,
    0x3ff3813813813814ul,0xc04249cd2b13cd6cul,0x3ff3698df3de0748ul,0xc041d982c9d52708ul,
    0x3ff3521cfb2b78c1ul,0xc04169c05363f158ul,0x3ff33ae45b57bcb2ul,0xc040fa848044b351ul,
    0x3ff323e34a2b10bful,0xc0408bce0d95fa38ul,0x3ff30d190130d190ul,0xc0401d9bbcfa61d4ul,
    0x3ff2f684bda12f68ul,0xc03f5fd8a9063e35ul,0x3ff2e025c04b8097ul,0xc03e857d3d361368ul,
    0x3ff2c9fb4d812ca0ul,0xc03dac22d3e441d3ul,0x3ff2b404ad012b40ul,0xc03cd3c712d31109ul,
    0x3ff29e4129e4129eul,0xc03bfc67a7fff4ccul,0x3ff288b01288b013ul,0xc03b2602497d5346ul,
    0x3ff27350b8812735ul,0xc03a5094b54d2828ul,0x3ff25e22708092f1ul,0xc0397c1cb13c7ec1ul,
    0x3ff2492492492492ul,0xc038a8980abfbd32ul,0x3ff23456789abcdful,0xc037d60496cfbb4cul,
    0x3ff21fb78121fb78ul,0xc037046031c79f85ul,0x3ff20b470c67c0d9ul,0xc03633a8bf437ce1ul,
    0x3ff1f7047dc11f70ul,0xc03563dc29ffacb2ul,0x3ff1e2ef3b3fb874ul,0xc03494f863b8df35ul,
    0x3ff1cf06ada2811dul,0xc033c6fb650cde51ul,0x3ff1bb4a4046ed29ul,0xc032f9e32d5bfdd1ul,
    0x3ff1a7b9611a7b96ul,0xc0322dadc2ab3497ul,0x3ff19453808ca29cul,0xc03162593186da70ul,
    0x3ff1811811811812ul,0xc03097e38ce60649ul,0x3ff16e0689427379ul,0xc02f9c95dc1d1165ul,
    0x3ff15b1e5f75270dul,0xc02e0b1ae8f2fd56ul,0x3ff1485f0e0acd3bul,0xc02c7b528b70f1c5ul,
    0x3ff135c81135c811ul,0xc02aed391ab6674eul,0x3ff12358e75d3033ul,0xc02960caf9abb7caul,
    0x3ff1111111111111ul,0xc027d60496cfbb4cul,0x3ff0fef010fef011ul,0xc0264ce26c067157ul,
    0x3ff0ecf56be69c90ul,0xc024c560fe68af88ul,0x3ff0db20a88f4696ul,0xc0233f7cde14cf5aul,
    0x3ff0c9714fbcda3bul,0xc021bb32a600549dul,0x3ff0b7e6ec259dc8ul,0xc020387efbca869eul,
    0x3ff0a6810a6810a7ul,0xc01d6ebd1f1febfeul,0x3ff0953f39010954ul,0xc01a6f9c377dd31bul,
    0x3ff0842108421084ul,0xc0177394c9d958d5ul,0x3ff073260a47f7c6ul,0xc0147aa07357704ful,
    0x3ff0624dd2f1a9fcul,0xc01184b8e4c56af8ul,0x3ff05197f7d73404ul,0xc00d23afc49139f9ul,
    0x3ff0410410410410ul,0xc00743ee861f3556ul,0x3ff03091b51f5e1aul,0xc0016a21e20a0a45ul,
    0x3ff0204081020408ul,0xbff72c7ba20f7327ul,0x3ff0101010101010ul,0xbfe720d9c06a835ful,
    0x3ff0000000000000ul,0x0000000000000000ul,0x3fefc07f01fc07f0ul,0x3ff6fe50b6ef0851ul,
    0x3fef81f81f81f820ul,0x4006e79685c2d22aul,0x3fef44659e4a4271ul,0x40111cd1d5133413ul,
    0x3fef07c1f07c1f08ul,0x4016bad3758efd87ul,0x3feecc07b301ecc0ul,0x401c4dfab90aab5ful,
    0x3fee9131abf0b767ul,0x4020eb389fa29f9bul,0x3fee573ac901e574ul,0x4023aa2fdd27f1c3ul,
    0x3fee1e1e1e1e1e1eul,0x402663f6fac91316ul,0x3fede5d6e3f8868aul,0x402918a16e46335bul,
    0x3fedae6076b981dbul,0x402bc84240adabbaul,0x3fed77b654b82c34ul,0x402e72ec117fa5b2ul,
    0x3fed41d41d41d41dul,0x40308c588cda79e4ul,0x3fed0cb58f6ec074ul,0x4031dcd197552b7bul,
    0x3fecd85689039b0bul,0x40332ae9e278ae1aul,0x3feca4b3055ee191ul,0x403476a9f983f74dul,
    0x3fec71c71c71c71cul,0x4035c01a39fbd688ul,0x3fec3f8f01c3f8f0ul,0x40370742d4ef027ful,
    0x3fec0e070381c0e0ul,0x40384c2bd02f03b3ul,0x3febdd2b899406f7ul,0x40398edd077e70dful,
    0x3febacf914c1bad0ul,0x403acf5e2db4ec94ul,0x3feb7d6c3dda338bul,0x403c0db6cdd94deeul,
    0x3feb4e81b4e81b4ful,0x403d49ee4c325970ul,0x3feb2036406c80d9ul,0x403e840be74e6a4dul,
    0x3feaf286bca1af28ul,0x403fbc16b902680aul,0x3feac5701ac5701bul,0x4040790adbb03009ul,
    0x3fea98ef606a63beul,0x40411307dad30b76ul,0x3fea6d01a6d01a6dul,0x4041ac05b291f070ul,
    0x3fea41a41a41a41aul,0x40424407ab0e073aul,0x3fea16d3f97a4b02ul,0x4042db10fc4d9aaful,
    0x3fe9ec8e951033d9ul,0x40437124cea4cdedul,0x3fe9c2d14ee4a102ul,0x404406463b1b0449ul,
    0x3fe999999999999aul,0x40449a784bcd1b8bul,0x3fe970e4f80cb872ul,0x40452dbdfc4c96b3ul,
    0x3fe948b0fcd6e9e0ul,0x4045c01a39fbd688ul,0x3fe920fb49d0e229ul,0x4046518fe4677ba7ul,
    0x3fe8f9c18f9c18faul,0x4046e221cd9d0cdeul,0x3fe8d3018d3018d3ul,0x404771d2ba7efb3cul,
    0x3fe8acb90f6bf3aaul,0x404800a563161c54ul,0x3fe886e5f0abb04aul,0x40488e9c72e0b226ul,
    0x3fe8618618618618ul,0x40491bba891f1709ul,0x3fe83c977ab2beddul,0x4049a802391e232ful,
    0x3fe8181818181818ul,0x404a33760a7f6051ul,0x3fe7f405fd017f40ul,0x404abe18797f1f49ul,
    0x3fe7d05f417d05f4ul,0x404b47ebf73882a1ul,0x3fe7ad2208e0ecc3ul,0x404bd0f2e9e79031ul,
    0x3fe78a4c8178a4c8ul,0x404c592fad295b56ul,0x3fe767dce434a9b1ul,0x404ce0a4923a587dul,
    0x3fe745d1745d1746ul,0x404d6753e032ea0ful,0x3fe724287f46debcul,0x404ded3fd442364cul,
    0x3fe702e05c0b8170ul,0x404e726aa1e754d2ul,0x3fe6e1f76b4337c7ul,0x404ef6d67328e220ul,
};
constant ulong _eon_powf_exp_coeff[2]={
    0x3eeebfbdff30d656ul,0x3f762e4453e10daeul,
};
''' + exponential32_table('_eon_powf_exp_table') + "\n\n"

_POW_HEADER = r'''
inline uint f_operator_pow(uint x,uint y) {
    const uint one=0x3f800000u;
    uint ax=x&0x7fffffffu,ay=y&0x7fffffffu,sign=0u;
    if(x==one) return one;
    if(y==one) return f_mul(x,one);
    if(ay==0u) return one;
    if(ay>0x7f800000u) return f_mul(y,one);
    if(ax>0x7f800000u) return f_mul(x,one);
    if(ay==0x7f800000u) {
        if(ax==one) return one;
        return ((ax>one) != bool(y>>31)) ? 0x7f800000u:0u;
    }
    bool integral=ay>=0x4b800000u;
    bool odd=false;
    // Only negative bases consume integral/parity below. Keep the positive
    // finite path out of software binary64 rounding/conversion.
    if((x>>31) && ay>=one && ay<0x4b800000u) {
        ulong wide=f_to_d(ay);
        integral=d_floor(wide)==wide;
        odd=integral && bool(d_to_int(wide)&1L);
    }
    if(ax==0u || ax==0x7f800000u) {
        uint result=odd ? x:ax;
        return (y>>31) ? f_div(one,result):result;
    }
    if(x>>31) {
        if(!integral) return 0x7fc00000u;
        if(odd) sign=0x80000000u;
        x=ax;
    }
    if(ax<0x00800000u) x=f_sub(x|one,one)-0x3f000000u;
    uint difference=x-0x3f338000u;
    uint index=(difference>>16)&127u;
    uint scale=difference&0xff800000u;
    int exponent=as_type<int>(scale)>>16;
    ulong m=f_to_d(x-scale);
    ulong r=d_fma(m,_eon_powf_log_table[2*index],0xbff0000000000000ul);
    ulong hi=d_add(_eon_powf_log_table[2*index+1],d_from_int(exponent));
    ulong square=d_mul(r,r);
    ulong p0=d_fma(r,_eon_powf_log_coeff[0],_eon_powf_log_coeff[1]);
    ulong p1=d_fma(r,_eon_powf_log_coeff[2],_eon_powf_log_coeff[3]);
    ulong logarithm=d_fma(d_fma(p0,square,p1),r,hi);
    ulong product=d_mul(f_to_d(y),logarithm);
    if(d_gt(product,0x40e0000000000000ul)) product=0x40e0000000000000ul;
    if(d_lt(product,0xc0e0000000000000ul)) product=0xc0e0000000000000ul;
    long k=d_to_int(product);
    ulong fraction=d_sub(product,d_from_int(k));
    if(d_ge(fraction,0x3fe0000000000000ul)) ++k;
    else if(d_le(fraction,0xbfe0000000000000ul)) --k;
    fraction=d_sub(product,d_from_int(k));
    ulong base=_eon_powf_exp_table[uint(k)&127u]+(ulong(k)<<45);
    ulong p=d_fma(_eon_powf_exp_coeff[0],fraction,_eon_powf_exp_coeff[1]);
    ulong result=d_fma(d_mul(p,fraction),base,base);
    return f_from_d(result)^sign;
}
'''

MSL_POWER_HEADER = (_OPERATOR_FMA_HEADER + _POW_TABLES
                    + _POW_HEADER.replace("d_fma(", "_eon_operator_polynomial_fma("))

MSL_INTEGER_EXPONENT_HEADER = r'''
inline uint f_operator_powi(uint base,int exponent) {
    uint magnitude=exponent<0 ? uint(-long(exponent)):uint(exponent);
    uint result=0x3f800000u;
    while(true) {
        if(magnitude&1u) result=f_mul(result,base);
        magnitude>>=1;
        if(magnitude==0u) break;
        base=f_mul(base,base);
    }
    return exponent<0 ? f_div(0x3f800000u,result):result;
}
'''

def _remainder32_source(first: str, last: str) -> str:
    first = 'inline ulong d_operator_' + first
    last = 'inline ulong d_operator_' + last
    source = first + _REMAINDER64.split(first, 1)[1].split(last, 1)[0]
    source = (source.replace('if(!d_is_zero(r))', 'if(r&0x7fffffffu)')
              .replace('if(!d_is_zero(quotient))', 'if(quotient&0x7fffffffu)')
              .replace('!d_is_zero(r)', '(r&0x7fffffffu)')
              .replace('!d_is_zero(quotient)', '(quotient&0x7fffffffu)')
              .replace('d_is_zero(y)', '(y&0x7fffffffu)==0u')
              .replace('0x3ff0000000000000ul', '0x3f800000u')
              .replace('0x3fe0000000000000ul', '0x3f000000u')
              .replace('0x8000000000000000ul', '0x80000000u'))
    return (source.replace('ulong', 'uint').replace('d_operator_', 'f_operator_')
            .replace('d_lt(', 'f_lt(').replace('d_gt(', 'f_gt(')
            .replace('0ul', '0u')
            .replace('d_floor(quotient)', 'f_from_d(d_floor(f_to_d(quotient)))')
            .replace('\n            result=', ' result=').rstrip('\n'))


MSL_REMAINDER_HEADER = _FLOAT_FMA_HEADER+r'''
inline bool f_operator_is_nan(uint a) {return (a&0x7fffffffu)>0x7f800000u;}
inline uint f_operator_nan2(uint a,uint b) {
    if(f_operator_is_nan(a) && !(a&0x00400000u)) return a|0x00400000u;
    if(f_operator_is_nan(b) && !(b&0x00400000u)) return b|0x00400000u;
    return f_operator_is_nan(a) ? a:b;
}
inline uint f_operator_add(uint a,uint b) {
    return f_operator_is_nan(a)||f_operator_is_nan(b) ? f_operator_nan2(a,b):f_add(a,b);
}
inline uint f_operator_sub(uint a,uint b) {
    return f_operator_is_nan(a)||f_operator_is_nan(b) ? f_operator_nan2(a,b):f_sub(a,b);
}
inline uint f_operator_mul(uint a,uint b) {
    return f_operator_is_nan(a)||f_operator_is_nan(b) ? f_operator_nan2(a,b):f_mul(a,b);
}
inline uint f_operator_div(uint a,uint b) {
    return f_operator_is_nan(a)||f_operator_is_nan(b) ? f_operator_nan2(a,b):f_div(a,b);
}
inline uint f_operator_fma(uint a,uint b,uint c) {
    if(f_operator_is_nan(c) && !(c&0x00400000u)) return c|0x00400000u;
    if(f_operator_is_nan(a) && !(a&0x00400000u)) return a|0x00400000u;
    if(f_operator_is_nan(b) && !(b&0x00400000u)) return b|0x00400000u;
    if(((a&0x7fffffffu)==0x7f800000u && (b&0x7fffffffu)==0u) ||
       ((b&0x7fffffffu)==0x7f800000u && (a&0x7fffffffu)==0u)) return 0x7fc00000u;
    if(f_operator_is_nan(c)) return c;
    if(f_operator_is_nan(a)) return a;
    if(f_operator_is_nan(b)) return b;
    return _eon_operator_float_fma(a,b,c);
}
inline uint f_operator_fmod(uint x,uint y) {
    uint ax=x&0x7fffffffu,ay=y&0x7fffffffu;
    if(ax>0x7f800000u && !(x&0x00400000u)) return x|0x00400000u;
    if(ay>0x7f800000u && !(y&0x00400000u)) return y|0x00400000u;
    return f_from_d(d_operator_fmod(f_to_d(x),f_to_d(y)));
}
''' + _remainder32_source(
    'mod_numpy(', 'mod_cython(') + r'''
inline uint f_operator_mod_adjust(uint y,bool adapt,uint c) {
    if(f_operator_is_nan(c) && !(c&0x00400000u)) return c|0x00400000u;
    if(f_operator_is_nan(y) && !(y&0x00400000u)) return y|0x00400000u;
    if(!adapt && (y&0x7fffffffu)==0x7f800000u) return 0x7fc00000u;
    if(f_operator_is_nan(c)) return c;
    if(f_operator_is_nan(y)) return y;
    return f_add(adapt ? y:(y&0x80000000u),c);
}
''' + _remainder32_source(
    'mod_cython_fma(', 'mod_cython_once(') + r'''
inline uint f_operator_mod_cython(uint x,uint y,thread bool &zero_error) {
    zero_error|=(y&0x7fffffffu)==0u;
    return f_operator_cython_mod_once(f_operator_add(f_operator_cython_mod_once(x,y),y),y);
}
''' + _remainder32_source(
    'mod_cython_once(', 'floordiv_cython(') + r'''
// Cython's generated C++ invokes double floor on a float quotient. Preserve
// that result type for any surrounding expression, before assignment rounding.
inline ulong f_operator_floordiv_cython(uint x,uint y,thread bool &zero_error) {
    zero_error|=(y&0x7fffffffu)==0u;
    return d_floor(f_to_d(f_operator_div(x,y)));
}
inline uint f_operator_mod_cpp(uint x,uint y) {
    return f_from_d(d_operator_mod_cpp(f_to_d(x),f_to_d(y)));
}
inline uint f_operator_mod_cpp_fma(uint x,uint y) {
    return f_from_d(d_operator_mod_cpp_fma(f_to_d(x),f_to_d(y)));
}
inline uint f_operator_floordiv_cpp(uint x,uint y) {
    return f_from_d(d_operator_floordiv_cpp(f_to_d(x),f_to_d(y)));
}
''' + _remainder32_source(
    'mod_cython(ulong x,ulong y,thread bool &error,', 'floordiv_cython(') + r'''
inline ulong f_operator_floordiv_cython(uint x,uint y,thread bool &error,thread int &kind) {
    if((y&0x7fffffffu)==0u && !error) kind=2;
    return f_operator_floordiv_cython(x,y,error);
}
'''
del _remainder32_source, _REMAINDER64


# Frozen installed ARM64 exp2f; code SHA256 e40604af9451a401ac89b4f21980510dd5994e503ea7639ca83d69d6b3954c4f.
_EXP2_TABLES = r'''
constant ulong _eon_exp2_32_coeff[2]={
    0x3fcebfbdff30d656ul,0x3fe62e4453e10daeul,
};
''' + exponential32_table('_eon_exp2_32_table') + "\n"

MSL_EXP2_HEADER = _OPERATOR_FMA_HEADER+_EXP2_TABLES+r'''
inline uint f_operator_exp2(uint x) {
    if(((x>>23)&255u)>=135u) {
        if((x&0x7fffffffu)>0x7f800000u || !(x>>31)) return f_mul(x,0x7f7fffffu);
        return 0u;
    }
    ulong value=f_to_d(x),magic=0x42c8000000000000ul;
    ulong biased=d_add(value,magic);
    ulong residual=d_sub(value,d_sub(biased,magic));
    ulong base=_eon_exp2_32_table[uint(biased)&127u]+(biased<<45);
    ulong polynomial=_eon_operator_polynomial_fma(_eon_exp2_32_coeff[0],residual,_eon_exp2_32_coeff[1]);
    polynomial=d_mul(polynomial,residual);
    return f_from_d(_eon_operator_polynomial_fma(polynomial,base,base));
}
'''
