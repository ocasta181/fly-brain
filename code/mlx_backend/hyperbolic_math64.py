"""Pinned target-aware binary64 log10 and hyperbolic Metal functions.

Apple scalar operation order/tables are reconstructed from installed ARM64
libsystem_m UUID83b173cd7f3239018e67c0b2676be015. NumPy tanh follows
NumPy1.26.4 loops_hyperbolic.dispatch.c.src (BSD3; see third_party notices).
Only software binary64 runs at execution time; no CPU libm fallback.
"""

from .scalar_math_tables import exponential64_tables

_EXP_SHARED_HEADER = "\n" + exponential64_tables('hyper_exp_coeff', 'hyper_exp_table') + r'''

inline ulong hyper_exp_reduced(ulong x,thread long *k,thread ulong *delta) {
    ulong t=d_mul(x,0x3ff71547652b82feul);
    *k=d_to_int(d_mul(t,0x4060000000000000ul));
    ulong rounded=d_mul(d_from_int(*k),0x3f80000000000000ul);
    ulong r=d_sub(t,rounded);
    ulong error=hyper_fma_normal(x,0x3ff71547652b82feul,d_neg(t));
    *delta=hyper_fma_normal(x,0x3c7777d0ffda0d24ul,error);
    return r;
}
inline ulong hyper_exp_factor(ulong r,ulong delta,ulong tail) {
    ulong p=d_add(r,hyper_exp_coeff[0]),q=d_add(r,hyper_exp_coeff[1]);
    r=d_add(r,d_add(tail,delta));
    p=hyper_fma_normal(r,p,hyper_exp_coeff[2]);
    q=hyper_fma_normal(r,q,hyper_exp_coeff[3]);
    return d_mul(d_mul(d_mul(r,hyper_exp_coeff[4]),p),q);
}
inline ulong hyper_large_half_exp(ulong x) {
    long k;ulong delta;
    ulong r=hyper_exp_reduced(x,&k,&delta);
    uint i=uint(k)&127u;
    ulong base=hyper_exp_table[2*i];
    ulong factor=hyper_exp_factor(r,delta,hyper_exp_table[2*i+1]);
    ulong y=hyper_fma_normal(factor,base,base);
    ulong scale=0x3fe0000000000000ul+(ulong(k>>7)<<52);
    return d_mul(y,scale);
}
'''

_SINH_HEADER = r'''
constant ulong hyper_sinh_tiny[6]={
    0x3de61bfa2e91919aul,0x3e5ae642c86bfbc7ul,0x3ec71de3a5aa7055ul,0x3f2a01a01a0196aeul,
    0x3f81111111111111ul,0x3fc5555555555555ul,
};

inline ulong d_hyper_sinh(ulong x) {
    ulong a=d_abs(x),sign=x&0x8000000000000000ul;
    if(d_is_nan(x)) return d_mul(x,0x7feffffffffffffful);
    if(a>=0x4035000000000000ul) {
        if(a>=0x408633ce8fb9f87eul) return d_mul(x,0x7feffffffffffffful);
        return hyper_large_half_exp(a)|sign;
    }
    if(a<0x3fd62e4200000000ul) {
        if(a<0x3e40000000000000ul)
            return d_mul(hyper_fma_any(x,0x4360000000000000ul,x),0x3c80000000000000ul);
        ulong square=d_mul(x,x),p=hyper_sinh_tiny[0];
        for(uint i=1;i<6;++i) p=hyper_fma_normal(square,p,hyper_sinh_tiny[i]);
        return hyper_fma_normal(d_mul(x,square),p,x);
    }
    long k;ulong delta;
    ulong r=hyper_exp_reduced(a,&k,&delta);
    uint i=uint(k)&127u,j=127u-i;
    ulong negative_r=d_sub(0x3f80000000000000ul,r);
    ulong positive_factor=hyper_exp_factor(r,delta,hyper_exp_table[2*i+1]);
    ulong negative_factor=hyper_exp_factor(negative_r,d_neg(delta),hyper_exp_table[2*j+1]);
    long scale=k>>7;
    ulong positive=hyper_exp_table[2*i]+(ulong(scale-1)<<52);
    ulong negative=hyper_exp_table[2*j]-(ulong(scale+2)<<52);
    ulong high=d_sub(positive,negative);
    ulong correction=d_add(d_sub(high,positive),negative);
    correction=hyper_fma_normal(negative,negative_factor,correction);
    correction=hyper_fma_normal(d_neg(positive),positive_factor,correction);
    return d_sub(high,correction)|sign;
}
inline ulong _eon_hyper_sinh(ulong x) { return d_hyper_sinh(x); }
'''

_COSH_HEADER = r'''
constant ulong hyper_cosh_tiny[6]={
    0x3e21f7f20c7f8ea9ul,0x3e927e4d41cf7cf1ul,0x3efa01a01a9e8422ul,0x3f56c16c16c15a6cul,
    0x3fa5555555555557ul,0x3fe0000000000000ul,
};

inline ulong d_hyper_cosh(ulong x) {
    ulong a=d_abs(x);
    if(d_is_nan(x)) return d_mul(a,0x7feffffffffffffful);
    if(a>=0x4035000000000000ul) {
        if(a>=0x408633ce8fb9f87eul) return d_mul(a,0x7feffffffffffffful);
        return hyper_large_half_exp(a);
    }
    if(a<0x3fd62e4200000000ul) {
        if(a<0x3e10000000000000ul)
            return d_mul(hyper_fma_any(a,a,0x4000000000000000ul),0x3fe0000000000000ul);
        ulong square=d_mul(a,a),p=hyper_cosh_tiny[0];
        for(uint i=1;i<5;++i) p=hyper_fma_normal(square,p,hyper_cosh_tiny[i]);
        ulong q=d_mul(a,hyper_cosh_tiny[5]);
        q=hyper_fma_normal(a,d_mul(square,p),q);
        return hyper_fma_normal(a,q,0x3ff0000000000000ul);
    }
    long k;ulong delta;
    ulong r=hyper_exp_reduced(a,&k,&delta);
    uint i=uint(k)&127u,j=127u-i;
    ulong negative_r=d_sub(0x3f80000000000000ul,r);
    ulong positive_factor=hyper_exp_factor(r,delta,hyper_exp_table[2*i+1]);
    ulong negative_factor=hyper_exp_factor(negative_r,d_neg(delta),hyper_exp_table[2*j+1]);
    long scale=k>>7;
    ulong positive=hyper_exp_table[2*i]+(ulong(scale-1)<<52);
    ulong negative=hyper_exp_table[2*j]-(ulong(scale+2)<<52);
    ulong high=d_add(positive,negative);
    ulong correction=d_sub(d_sub(high,positive),negative);
    correction=hyper_fma_normal(negative,negative_factor,d_neg(correction));
    correction=hyper_fma_normal(positive,positive_factor,correction);
    return d_add(high,correction);
}
inline ulong _eon_hyper_cosh(ulong x) { return d_hyper_cosh(x); }
'''

_LOG10_HEADER = r'''
constant ulong hyper_log10_coeff[6]={
    0xbfb49704e9bc168eul,0x3fafc431f82b9b9dul,0x3ff9de61616cdf2bul,0x3ff609d14c7ee9beul,
    0x3ffab7d4f0547db7ul,0xbff3f6ab6ec10818ul,
};
constant ulong hyper_log10_table[258]={
    0x3ff0000000000000ul,0x0000000000000000ul,0x3fefc07f01fb38d4ul,0x3f6bafd4727c9364ul,
    0x3fef81f81f81a3e5ul,0x3f7b9476a50f6475ul,0x3fef44659e49d2b1ul,0x3f849b085150a16dul,
    0x3fef07c1f07c3108ul,0x3f8b5e908eaf3398ul,0x3feecc07b3017601ul,0x3f910a83a84b1f39ul,
    0x3fee9131abf09b55ul,0x3f945f4f5acd2447ul,0x3fee573ac902a11ful,0x3f97adc3df306138ul,
    0x3fee1e1e1e1e0ae1ul,0x3f9af5f92b020224ul,0x3fede5d6e3f7cbaful,0x3f9e3806acc7e0f9ul,
    0x3fedae6076b8f76bul,0x3fa0ba01a81b0d1dul,0x3fed77b654b7b0b1ul,0x3fa25502c0ffd549ul,
    0x3fed41d41d410288ul,0x3fa3ed1199ac1d02ul,0x3fed0cb58f6e27bcul,0x3fa58238eeb7e4d6ul,
    0x3fecd8568902fb3ful,0x3fa714834281fa6cul,0x3feca4b3055ed9baul,0x3fa8a3fadeb884d0ul,
    0x3fec71c71c70f481ul,0x3faa30a9d6105e4bul,0x3fec3f8f01c3ce79ul,0x3fabba9a058f4bc9ul,
    0x3fec0e0703826caaul,0x3fad41d5164a5b23ul,0x3febdd2b89930e8eul,0x3faec6647ebd4665ul,
    0x3febacf914c177b9ul,0x3fb02428c1f18d98ul,0x3feb7d6c3dd95825ul,0x3fb0e3d29d848db3ul,
    0x3feb4e81b4e7b01ful,0x3fb1a2344551cc7eul,0x3feb2036406badd7ul,0x3fb25f5215eeba23ul,
    0x3feaf286bca2397cul,0x3fb31b3055c23663ul,0x3feac5701ac4d7c0ul,0x3fb3d5d335c7aa31ul,
    0x3fea98ef60696bd7ul,0x3fb48f3ed1e3553aul,0x3fea6d01a6cfc64ful,0x3fb547773198a06bul,
    0x3fea41a41a40b6e9ul,0x3fb5fe80488ee159ul,0x3fea16d3f97b18daul,0x3fb6b45df6f07593ul,
    0x3fe9ec8e95113011ul,0x3fb769140a20ed4ful,0x3fe9c2d14ee59e3bul,0x3fb81ca63d015f6cul,
    0x3fe9999999992278ul,0x3fb8cf1838884d6eul,0x3fe970e4f80c4ba4ul,0x3fb9806d94167d88ul,
    0x3fe948b0fcd69907ul,0x3fba30a9d60b536dul,0x3fe920fb49d1de51ul,0x3fbadfd074126264ul,
    0x3fe8f9c18f9b4e11ul,0x3fbb8de4d3aec4d9ul,0x3fe8d3018d2f1d8bul,0x3fbc3aea4a60d466ul,
    0x3fe8acb90f6c68a1ul,0x3fbce6e41e442e9eul,0x3fe886e5f0abeab2ul,0x3fbd91d58669a0d9ul,
    0x3fe861861862484bul,0x3fbe3bc1ab0aa46ful,0x3fe83c977ab327e6ul,0x3fbee4aba60f1033ul,
    0x3fe8181818173ae4ul,0x3fbf8c96834a7e48ul,0x3fe7f405fd00d922ul,0x3fc019c2a066360cul,
    0x3fe7d05f417c0774ul,0x3fc06cbd67a915cdul,0x3fe7ad2208e0dbe0ul,0x3fc0bf3d0937ebc3ul,
    0x3fe78a4c81784397ul,0x3fc11142f081f8dbul,0x3fe767dce433bafdul,0x3fc162d082aed3feul,
    0x3fe745d1745de7baul,0x3fc1b3e71ec75d90ul,0x3fe724287f468d6dul,0x3fc204881def4ac9ul,
    0x3fe702e05c0c18e3ul,0x3fc254b4d35d0f5ful,0x3fe6e1f76b42428ful,0x3fc2a46e8caa0de1ul,
    0x3fe6c16c16c06c9bul,0x3fc2f3b691c81020ul,0x3fe6a13cd1528b5eul,0x3fc3428e2542415bul,
    0x3fe681681681a1f6ul,0x3fc390f684497c90ul,0x3fe661ec6a5049d6ul,0x3fc3def0e6e1faceul,
    0x3fe642c8590aa5eful,0x3fc42c7e7fe5304eul,0x3fe623fa7701094ful,0x3fc479a07d3c4360ul,
    0x3fe605816057fe92ul,0x3fc4c65807e96e6bul,0x3fe5e75bb8cfdcfbul,0x3fc512a64429fcb4ul,
    0x3fe5c9882b922e81ul,0x3fc55e8c518d5130ul,0x3fe5ac056b01183eul,0x3fc5aa0b4b0a3393ul,
    0x3fe58ed230825723ul,0x3fc5f5244722cd0ful,0x3fe571ed3c514831ul,0x3fc63fd857fa0cf1ul,
    0x3fe555555555be48ul,0xbfbffbfc2bbe9af4ul,0x3fe5390948f33af7ul,0xbfbf68216c986b90ul,
    0x3fe51d07eae25fe5ul,0xbfbed50a4a23c997ul,0x3fe501501501bd58ul,0xbfbe42b4c16eed44ul,
    0x3fe4e5e0a72ea8caul,0xbfbdb11ed764c030ul,0x3fe4cab88726a83cul,0xbfbd204698d07526ul,
    0x3fe4afd6a051e80ful,0xbfbc902a19e1cbfdul,0x3fe49539e3b27047ul,0xbfbc00c776701e92ul,
    0x3fe47ae147ad26e5ul,0xbfbb721cd16c4e1cul,0x3fe460cbc7f6ce28ul,0xbfbae428550f01d9ul,
    0x3fe446f8656250caul,0xbfba56e8325c6c50ul,0x3fe42d6625d45a4eul,0xbfb9ca5aa16e608dul,
    0x3fe4141414137fceul,0xbfb93e7de0f90978ul,0x3fe3fb013fb1102ful,0xbfb8b3503651ddb3ul,
    0x3fe3e22cbce5a6b7ul,0xbfb828cfed2f2caful,0x3fe3c995a47afe53ul,0xbfb79efb57ad28baul,
    0x3fe3b13b13b04947ul,0xbfb715d0ce3125d3ul,0x3fe3991c2c17b662ul,0xbfb68d4eaf2263a3ul,
    0x3fe3813813803a55ul,0xbfb605735ee3df92ul,0x3fe3698df3de9d20ul,0xbfb57e3d47c709aful,
    0x3fe3521cfb2b1096ul,0xbfb4f7aad9b97441ul,0x3fe33ae45b573065ul,0xbfb471ba8a7ab790ul,
    0x3fe323e34a2baef3ul,0xbfb3ec6ad5440f5aul,0x3fe30d190131cd5bul,0xbfb367ba3aafd5eeul,
    0x3fe2f684bda0a126ul,0xbfb2e3a740b43dfful,0x3fe2e025c04c5430ul,0xbfb2603072a73dd8ul,
    0x3fe2c9fb4d8060d5ul,0xbfb1dd5460c3fb8bul,0x3fe2b404ad01d56cul,0xbfb15b11a098953aul,
    0x3fe29e4129e3187dul,0xbfb0d966cc5f2b50ul,0x3fe288b012898ed6ul,0xbfb05852837b79c2ul,
    0x3fe27350b881d871ul,0xbfafafa6d3a047d3ul,0x3fe25e22707fbbf8ul,0xbfaeafd0502b92c7ul,
    0x3fe249249249e9cful,0xbfadb11ed7700a68ul,0x3fe23456789bb267ul,0xbfacb38fcce476e8ul,
    0x3fe21fb781227951ul,0xbfabb7209d242ce5ul,0x3fe20b470c676cacul,0xbfaabbcebd80ef57ul,
    0x3fe1f7047dc1c36bul,0xbfa9c197abf7fb7aul,0x3fe1e2ef3b3fd2c9ul,0xbfa8c878eeb197cbul,
    0x3fe1cf06ada1ee29ul,0xbfa7d070145824feul,0x3fe1bb4a4045ee1ful,0xbfa6d97ab3addfcful,
    0x3fe1a7b961199740ul,0xbfa5e3966b7356cbul,0x3fe19453808caf9ful,0xbfa4eec0e24633c5ul,
    0x3fe181181181a457ul,0xbfa3faf7c669fbf0ul,0x3fe16e0689430a4cul,0xbfa30838cdca801eul,
    0x3fe15b1e5f74ca8ful,0xbfa21681b5c42118ul,0x3fe1485f0e0bc882ul,0xbfa125d0433b42f6ul,
    0x3fe135c81135efedul,0xbfa0362241e83be8ul,0x3fe12358e75dd419ul,0xbf9e8eeb0a0393c4ul,
    0x3fe1111111104099ul,0xbf9cb38fccc387a4ul,0x3fe0fef010fe3855ul,0xbf9ada2e8e2b8cccul,
    0x3fe0ecf56be75b6eul,0xbf9902c31d763f47ul,0x3fe0db20a89031a0ul,0xbf972d4956e25992ul,
    0x3fe0c9714fbdd2a2ul,0xbf9559bd2420786bul,0x3fe0b7e6ec259ec9ul,0xbf93881a7b81aa5eul,
    0x3fe0a6810a68f2a2ul,0xbf91b85d605c7d78ul,0x3fe0953f3900460bul,0xbf8fd503c36764eful,
    0x3fe084210841501dul,0xbf8c3d08374fd2e0ul,0x3fe073260a48f47ful,0xbf88a8c06be7337eul,
    0x3fe0624dd2f253e0ul,0xbf851824c77c8587ul,0x3fe05197f7d73005ul,0xbf818b2dc8d1e1bful,
    0x3fe04104104183eeul,0xbf7c03a80b1c8a49ul,0x3fe03091b51e60a3ul,0xbf74f82051c84636ul,
    0x3fe0204081017ac2ul,0xbf6be76bd7050239ul,0x3fe01010100fc088ul,0xbf5bd96a1cf3fdd8ul,
    0x3fe0000000000000ul,0x0000000000000000ul,
};

inline ulong hyper_log10_core(ulong x,int adjust) {
    long k=(as_type<long>(x+0xc018100000000000ul)>>52)+adjust;
    ulong fraction=x&0xffffffffffffful;
    uint index=uint((fraction+0x100000000000ul)>>45);
    ulong m=fraction|0x3ff0000000000000ul;
    ulong inv=hyper_log10_table[2*index],hi=hyper_log10_table[2*index+1];
    ulong r=hyper_fma_normal(m,inv,0xbff0000000000000ul);
    ulong p=hyper_fma_normal(r,hyper_log10_coeff[1],hyper_log10_coeff[0]);
    ulong q=hyper_fma_normal(r,d_add(r,hyper_log10_coeff[3]),hyper_log10_coeff[2]);
    ulong s=hyper_fma_normal(r,d_add(r,hyper_log10_coeff[5]),hyper_log10_coeff[4]);
    ulong poly=d_mul(d_mul(d_mul(d_mul(r,r),p),q),s);
    if(k!=0) {
        ulong high=hi&0xfffffffff8000000ul;
        ulong low=d_add(d_sub(hi,high),poly);
        high=hyper_fma_normal(d_from_int(k),0x3fd3441350000000ul,high);
        ulong result=d_mul(d_from_int(k),0x3e03ef3fde623e25ul);
        result=d_add(result,d_mul(r,0x3fdbcb7b1526e50eul));
        return d_add(d_add(result,low),high);
    }
    ulong product=d_mul(inv,m);
    ulong rounded=d_sub(product,0x3ff0000000000000ul);
    ulong error=hyper_fma_normal(inv,m,d_neg(product));
    ulong high=r&0xfffffffff8000000ul;
    ulong tail=d_add(error,d_sub(rounded,high));
    ulong high_scaled=d_mul(high,0x3fdbcb7b18000000ul);
    ulong low_scaled=d_mul(r,0xbe26c8d78e6acaa4ul);
    ulong sum=d_add(hi,high_scaled);
    ulong sum_error=d_sub(high_scaled,d_sub(sum,hi));
    low_scaled=hyper_fma_normal(tail,0x3fdbcb7b18000000ul,low_scaled);
    return d_add(d_add(d_add(low_scaled,sum_error),poly),sum);
}
inline ulong d_hyper_log10(ulong x) {
    if(d_is_nan(x)) return x;
    if(d_is_zero(x)) return 0xfff0000000000000ul;
    if(x>>63) return D_QNAN;
    if(x==D_INF) return x;
    if(d_exp(x)==0ul) return hyper_log10_core(d_mul(x,0x4350000000000000ul),-54);
    return hyper_log10_core(x,0);
}
inline ulong _eon_hyper_log10(ulong x) { return d_hyper_log10(x); }
'''

_TANH_SCALAR_HEADER = r'''
constant ulong hyper_tanh_large[8]={
    0xbf8dc165dad1eab3ul,0x3f9fef26c0bbebc1ul,0xbfafffbdaa5e7368ul,0x3fbfffff6ee83673ul,
    0xbfcfffffff536d53ul,0x3fdfffffffff9b22ul,0xbfefffffffffffeaul,0x3ff0000000000000ul,
};
constant ulong hyper_tanh_tiny[9]={
    0x400e0dabc398b233ul,0x3fe8671c1eaf3374ul,0xc0053f1e5aacd98cul,0xc01391fe3509b143ul,
    0x4018aa753b569bd0ul,0x4019755323ab1cfdul,0x401a5c956c18b14eul,0x401af4617f8ecc54ul,
    0xbf29111400a41fbful,
};
constant ulong hyper_tanh_table[128]={
    0xbfbd20c2f0233edbul,0xc03343b38896f0b4ul,0x40062a6dd642e5f2ul,0x3ffd6a2b08287816ul,
    0xbff82c7d74f3eec6ul,0xc007215acc9fb7f0ul,0x3f53c077627b845cul,0x403dfa52b8813743ul,
    0x4000f1eaf9d69209ul,0x400230c40dbb23a6ul,0x40058dfd62bc4a9cul,0x4006d7ac95bb7ebbul,
    0xbf32c5dbe59ba738ul,0xbf2285e830f09106ul,0x3feef20000000000ul,0x3f6e000000000000ul,
    0x3f8c8ea9f23e804cul,0x400760f0b7993b69ul,0x400150bd04d7a1c5ul,0xbfecbb7fbf48be5bul,
    0xc002b4f4c7166628ul,0xc00967616f164262ul,0xbf7f0e0a35e743eeul,0x4002bfcc207fb647ul,
    0x40044f947dc5b290ul,0x40052d29d5a7c63dul,0x40056dbce8bf8267ul,0x4005933cf97f4ab1ul,
    0xbf63769443462e8eul,0xbf66371574fff23bul,0x3febfc0000000000ul,0x3f9cc00000000000ul,
    0x3f805149baef85eful,0x4007dd95197635ccul,0x400401a9f73d2b4cul,0xbfdecab664089381ul,
    0xc002118c95acd0c0ul,0xc00d4f1ee6952a16ul,0xbf8083470d069524ul,0x40029f96c2560ddbul,
    0x4004d508bdb79078ul,0x4008e8711607d045ul,0x400b8b9b0e8d997bul,0x400fe3aaa6cdddedul,
    0x3f465a05505fff51ul,0xbf583538a7c2274bul,0x3fe6400000000000ul,0x3fbdf00000000000ul,
    0x3ffcea67384b11e9ul,0xbffb61dca98942f1ul,0x4008f484bbf139fdul,0xbf9b3edb87fc9a2bul,
    0xbffce191d611f94ful,0xc008f93f06e92964ul,0xbfc4581fb52271d2ul,0xbfc52176c275a3abul,
    0x400aad35d2a7a8bbul,0x4009a5abbcfde960ul,0x4009c6f32a11f93ful,0x4009f0bf5d144f71ul,
    0x3f3815b62267a35aul,0x3f4e91115d517686ul,0x3fe0400000000000ul,0x3fd0700000000000ul,
    0x3f79d573ac48a801ul,0x400b2f66c7d28529ul,0x3fe1984c3d542023ul,0xbff5c0811965f6aful,
    0xc00849d4c45d855dul,0xc0100ca2f60e8c0dul,0xbf7fc955f6cf414ful,0x400d178e287f7e41ul,
    0x400ee3aaa95874bcul,0x401023e37f392380ul,0x4010c20ad086484eul,0x40111aae4ef8eaa5ul,
    0xbf2311e422ef2c53ul,0xbf3212bb7f586ef1ul,0x3fd6400000000000ul,0x3fdaa00000000000ul,
    0x3f77ce7bc458b46cul,0x4010c82b6cec1ffdul,0x3fec3c0474639899ul,0xbff21c2bc70cf0daul,
    0xc00725d4ba0681dful,0xc00f5a29bc4b0e92ul,0xbf7d8d1f70cb1f6ful,0x4013ddf5a34313a8ul,
    0x401078e54f22290aul,0x40104dd7ea5a025aul,0x40105f93a44aaaf5ul,0x40106e0d61650779ul,
    0xbf0c60bd458a1834ul,0xbf22a7763e951b37ul,0x3fcd280000000000ul,0x3fe2160000000000ul,
    0xc015516bf006503cul,0x40118016dbc7636eul,0x3ff5514cc150e4b9ul,0xbfedebdb00c05dd1ul,
    0xc009ba10bb2b0a0aul,0xc0133877476ec678ul,0x3fdbae9cc13f99d1ul,0x4014740dbcf7677dul,
    0x401476a144da27f9ul,0x40158fe768dfbc76ul,0x401757547139120dul,0x40192330f539ad04ul,
    0x3ed4a72990f33fe0ul,0x3e9d115702afdd8dul,0x3fc2900000000000ul,0x3fe60a0000000000ul,
    0xc00354b959b1dff0ul,0x4013a94fb28e608bul,0x3ffd3038014972b9ul,0xbfd51fa3cad3528eul,
    0xc003c48aa812bf6eul,0xc0103fdd839363fcul,0x3fcb8de916d04f6dul,0x401976f1f99190cful,
    0x4016028448cc199ful,0x40155491205f7adaul,0x40157086c28e6266ul,0x40158eb8b8d53c8aul,
    0x3edcc89f858814a5ul,0x3ea62d021453335ful,0x3fb7300000000000ul,0x3fe9120000000000ul,
};

inline ulong d_hyper_tanh_scalar(ulong x) {
    ulong a=d_abs(x),sign=x&0x8000000000000000ul;
    if(d_is_nan(x)) return d_add(x,x);
    if(a>=0x40330fc1931f09caul) return 0x3ff0000000000000ul|sign;
    if(a>=0x4000000000000000ul) {
        ulong arg=d_neg(d_add(a,a));
        long k;ulong delta;
        ulong r=hyper_exp_reduced(arg,&k,&delta);
        uint i=uint(k)&127u;
        ulong base=hyper_exp_table[2*i];
        ulong factor=hyper_exp_factor(r,delta,hyper_exp_table[2*i+1]);
        ulong y=hyper_fma_normal(factor,base,base)+(ulong((k>>7)+1)<<52);
        ulong p=hyper_tanh_large[0];
        for(uint j=1;j<8;++j) p=hyper_fma_normal(y,p,hyper_tanh_large[j]);
        return p|sign;
    }
    if(a>=0x3fc0000000000000ul) {
        uint index=uint(d_to_int(d_mul(a,0x4010000000000000ul)));
        ulong center=d_add(d_mul(d_from_int(index),0x3fd0000000000000ul),0x3fc0000000000000ul);
        ulong y=d_sub(a,center);
        uint off=16*index;
        ulong p[7];
        for(uint j=0;j<6;++j)
            p[j]=hyper_fma_normal(y,d_add(y,hyper_tanh_table[off+j]),hyper_tanh_table[off+6+j]);
        p[6]=hyper_fma_normal(y,hyper_tanh_table[off+12],hyper_tanh_table[off+13]);
        ulong product=d_mul(d_mul(d_mul(p[0],p[1]),d_mul(p[2],p[3])),d_mul(d_mul(p[4],p[5]),p[6]));
        ulong high=a&0xfffffffff8000000ul,low=d_sub(a,high);
        product=hyper_fma_normal(low,hyper_tanh_table[off+14],product);
        ulong tail=hyper_fma_normal(high,hyper_tanh_table[off+14],hyper_tanh_table[off+15]);
        return d_add(product,tail)|sign;
    }
    if(a<0x3e40000000000000ul)
        return d_mul(hyper_fma_any(x,0x4360000000000000ul,x),0x3c80000000000000ul);
    ulong square=d_mul(x,x),p[4];
    for(uint j=0;j<4;++j) p[j]=hyper_fma_normal(square,d_add(square,hyper_tanh_tiny[j]),hyper_tanh_tiny[4+j]);
    ulong product=d_mul(d_mul(p[0],p[1]),d_mul(p[2],p[3]));
    ulong cube=d_mul(x,square);
    return hyper_fma_normal(d_mul(cube,hyper_tanh_tiny[8]),product,x);
}
inline ulong _eon_hyper_tanh_scalar(ulong x) { return d_hyper_tanh_scalar(x); }
'''

_TANH_NUMPY_HEADER = r'''
constant ulong hyper_numpy_tanh_table[288]={
    0x0000000000000000ul,0x0000000000000000ul,0x3ff0000000000000ul,0xbbf0b3ea3fdfaa19ul,
    0xbfd5555555555555ul,0xbce6863ee44ed636ul,0x3fc1111111112ab5ul,0xbda1ea19ddddb3b4ul,
    0xbfaba1ba1990520bul,0xbe351ca7f096011ful,0x3f9664f94e6ac14eul,0xbea8c4c1fd7852feul,
    0xbf822404577aa9ddul,0xbefdd99a221ed573ul,0x3f6e3be689423841ul,0xbf2a1306713a4f3aul,
    0xbf55d7e76dc56871ul,0x3f35e67ab76a26e7ul,0x3fcc000000000000ul,0x3fcb8fd0416a7c92ul,
    0x3fee842ca3f08532ul,0xbfca48aaeb53bc21ul,0xbfd183afc292ba11ul,0x3fc04dcd0476c75eul,
    0x3fb5c19efdfc08adul,0xbfb0b8df995ce4dful,0xbf96e37bba52f6fcul,0x3f9eaaf3320c3851ul,
    0xbf94d3343bae39ddul,0xbfccce16b1046f13ul,0x403d8b07f7a82aa3ul,0x4070593a3735bab4ul,
    0xc0d263511f5baac1ul,0xc1045e509116b066ul,0x41528c38809c90c7ul,0x41848ee0627d8206ul,
    0x3fd4000000000000ul,0x3fd35f98a0ea650eul,0x3fed11574af58f1bul,0xbfd19921f4329916ul,
    0xbfcc1a4b039c9bfaul,0x3fc43d3449a80f08ul,0x3fa74c98dc34fbacul,0xbfb2955cf41e8164ul,
    0x3ecff7df18455399ul,0x3f9cf823fe761fc1ul,0xbf7bc748e60df843ul,0xbf81a16f224bb7b6ul,
    0xbf9f44ab92fbab0aul,0xbfccab654e44835eul,0x40169f73b15ebe5cul,0x4041fab9250984ceul,
    0xc076d57fb5190b02ul,0xc0a216d618b489ecul,0x3fdc000000000000ul,0x3fda5729ee488037ul,
    0x3fea945b9c24e4f9ul,0xbfd5e0f09bef8011ul,0xbfc16e1e6d8d0be6ul,0x3fc5c26f3699b7e7ul,
    0xbf790d6a8eff0a77ul,0xbfaf9d05c309f7c6ul,0x3f97362834d33a4eul,0x3f9022271754ff1ful,
    0xbf8c89372b43ba85ul,0xbf62cbf00406bc09ul,0x3fb2eac604473d6aul,0x3fd13ed80037dbacul,
    0xc025c1dd41cd6cb5ul,0xc0458d090ec3de95ul,0x4085f09f888f8adaul,0x40a5b89107c8af4ful,
    0x3fe4000000000000ul,0x3fe1bf47eabb8f95ul,0x3fe6284c3374f815ul,0xbfd893b59c35c882ul,
    0xbf92426c751e48a2ul,0x3fc1a686f6ab2533ul,0xbfac3c021789a786ul,0xbf987d27ccff4291ul,
    0x3f9e7f8380184b45ul,0xbf731fe77c9c60aful,0xbf8129a092de747aul,0x3f75b29bb02cf69bul,
    0x3f45f87d903aaac8ul,0xbf6045b9076cc487ul,0xbf58fd89fe05e0d1ul,0xbf74949d60113d63ul,
    0x3fa246332a2fcba5ul,0x3fb69d8374520edaul,0x3fec000000000000ul,0x3fe686650b8c2015ul,
    0x3fe02500a09f8d6eul,0xbfd6ba7cb7576538ul,0x3fb4f152b2bad124ul,0x3faf203c316ce730ul,
    0xbfae2196b7326859ul,0x3f8b2ca62572b098ul,0x3f869543e7c420d4ul,0xbf84a6046865ec7dul,
    0x3f60c85b4d538746ul,0x3f607df0f9f90c17ul,0xbf5e104671036300ul,0x3f2085ee7e8ac170ul,
    0x3f73f7af01d5af7aul,0x3f7c9fd6200d0adeul,0xbfb29d851a896fcdul,0xbfbded519f981716ul,
    0x3ff4000000000000ul,0x3feb2523bb6b2deeul,0x3fd1f25131e3a8c0ul,0xbfce7291743d7555ul,
    0x3fbbba40cbef72beul,0xbf89c7a02788557cul,0xbf93a7a011ff8c2aul,0x3f8f1cf6c7f5b00aul,
    0xbf7326bd4914222aul,0xbf4ca3f1f2b9192bul,0x3f5be9392199ec18ul,0xbf4b852a6e0758d5ul,
    0x3f19bc98ddf0f340ul,0x3f23524622610430ul,0xbf1e40bdead17e6bul,0x3f02cd40e0ad0a9ful,
    0x3ed9065ae369b212ul,0xbef02d288b5b3371ul,0x3ffc000000000000ul,0x3fee1fbf97e33527ul,
    0x3fbd22ca1c24a139ul,0xbfbb6d85a01efb80ul,0x3fb01ba038be6a3dul,0xbf98157e26e0d541ul,
    0x3f6e4709c7e8430eul,0x3f60379811e43dd5ul,0xbf5fc15b0a9d98faul,0x3f4c77dee0afd227ul,
    0xbf2a0c68a4489f10ul,0xbf0078c63d1b8445ul,0x3f0d4304bc9246e8ul,0xbeff12a6626911b4ul,
    0x3ee224cd6c4513e5ul,0xbe858ab8e019f311ul,0xbeb8e1ba4c98a030ul,0x3eb290981209c1a6ul,
    0x4004000000000000ul,0x3fef9258260a71c2ul,0x3f9b3afe1fba5c76ul,0xbf9addae58c7141aul,
    0x3f916df44871efc8ul,0xbf807b55c1c7d278ul,0x3f67682afa611151ul,0xbf4793826f78537eul,
    0x3f14cffcfa69fbb6ul,0x3f04055bce68597aul,0xbf00462601dc2faaul,0x3eec12eadd55be7aul,
    0xbed13c415f7b9d41ul,0x3eab9008bca408aful,0xbe24b645e68eeaa3ul,0xbe792fa6323b7cf8ul,
    0x3e6ffd0766ad4016ul,0xbe567e924bf5ff6eul,0x400c000000000000ul,0x3feff112c63a9077ul,
    0x3f6dd37d19b22b21ul,0xbf6dc59376c7aa19ul,0x3f63c6869dfc8870ul,0xbf53a18d5843190ful,
    0x3f3ef2ee77717cbful,0xbf2405695e36240ful,0x3f057e48e5b79d10ul,0xbee2bf0cb4a71647ul,
    0x3eb7b6a219dea9f4ul,0xbe6fa600f593181bul,0xbe722b8d9720cdb0ul,0x3e634df71865f620ul,
    0xbe4abfebfb72bc83ul,0x3e2df04d67876402ul,0xbe0c63c29f505f5bul,0x3de3f7f7de6b0eb6ul,
    0x4014000000000000ul,0x3fefff419668df11ul,0x3f27ccec13a9ef96ul,0xbf27cc5e74677410ul,
    0x3f1fb9aef915d828ul,0xbf0fb6bbc89b1a5bul,0x3ef95a4482f180b7ul,0xbee0e08de39ce756ul,
    0x3ec33b66d7d77264ul,0xbea31eaafe73efd5ul,0x3e80cbcc8d4c5c8aul,0xbe5a3c935dce3f7dul,
    0x3e322666d739bec0ul,0xbe05bb1bcf83ca73ul,0x3dd51c38f8695ed3ul,0xbd95c72be95e4d2cul,
    0xbd7fab216b9e0e49ul,0x3d69ed18bae3ebbcul,0x401c000000000000ul,0x3feffffc832750f2ul,
    0x3ecbe6c3f33250aeul,0xbecbe6c0e8b4cc87ul,0x3ec299d1e27c6e11ul,0xbeb299c9c684a963ul,
    0x3e9dc2c27da3b603ul,0xbe83d709ba5f714eul,0x3e66ac4e578b9b10ul,0xbe46abb02c4368edul,
    0x3e2425bb231a5e29ul,0xbe001c6d95e3ae96ul,0x3dd76a553d7e7918ul,0xbdaf2ac143fb6762ul,
    0x3d8313ac38c6832bul,0xbd55a89c30203106ul,0x3d2826b62056aa27ul,0xbcf7534c4f3dfa71ul,
    0x4024000000000000ul,0x3feffffffdc96f35ul,0x3e41b4865394f75ful,0xbe41b486526b0565ul,
    0x3e379b5ddcca334cul,0xbe279b5dd4fb3d01ul,0x3e12e2afd9f7433eul,0xbdf92e3fc5ee63e0ul,
    0x3ddcc74b8d3d5c42ul,0xbdbcc749ca8079ddul,0x3d9992a4beac8662ul,0xbd74755a00ea1fd3ul,
    0x3d4de0fa59416a39ul,0xbd23eae52a3dbf57ul,0x3cf7787935626685ul,0xbccad6b3bb9eff65ul,
    0x3ca313e31762f523ul,0xbc730b73f1eaff20ul,0x402c000000000000ul,0x3fefffffffffcf58ul,
    0x3d8853f01bda5f28ul,0xbd8853f01bef63a4ul,0x3d8037f57bc62c9aul,0xbd7037f57ae72aa6ul,
    0x3d59f320348679baul,0xbd414cc030f2110eul,0x3d23c589137f92b4ul,0xbd03c5883836b9d2ul,
    0x3ce191ba5ed3fb67ul,0xbcbc1c6c063bb7acul,0x3c948716cf3681b4ul,0xbc6b5e3e9ca0955eul,
    0x3c401ffc49c6bc29ul,0xbc12705ccd3dd884ul,0x3bea37aa21895319ul,0xbbba2cff8135d462ul,
    0x4034000000000000ul,0x3ff0000000000000ul,0x3c73953c0197ef58ul,0xbc73955be519be31ul,
    0x3c6a2d4b50a2cff7ul,0xbc5a2ca2bba78e86ul,0x3c44b61d9bbcc940ul,0xbc2ba022e8d82a87ul,
    0x3c107f8e2c8707a1ul,0xbbf07a5416264aecul,0x3bc892450bad44c4ul,0xbba3be9a4460fe00ul,
    0x3b873f9f2d2fda99ul,0xbb5eca68e2c1ba2eul,0xbabf0b21acfa52abul,0xba8e0a4c47ae75f5ul,
    0x3ae5c7f1fd871496ul,0xbab5a71b5f7d9035ul,0x0000000000000000ul,0x3ff0000000000000ul,
    0x0000000000000000ul,0x0000000000000000ul,0x0000000000000000ul,0x0000000000000000ul,
    0x0000000000000000ul,0x0000000000000000ul,0x0000000000000000ul,0x0000000000000000ul,
    0x0000000000000000ul,0x0000000000000000ul,0x0000000000000000ul,0x0000000000000000ul,
    0x0000000000000000ul,0x0000000000000000ul,0x0000000000000000ul,0x0000000000000000ul,
};

inline ulong d_hyper_tanh_numpy(ulong x) {
    if(d_is_nan(x)) return D_QNAN;
    // Here the NumPy Horner sequence rounds its linear coefficient to one.
    // Returning x also keeps subnormal operands outside the normal-only FMA.
    if(d_abs(x)<0x3e40000000000000ul) return x;
    ulong sign=x&0x8000000000000000ul;
    ulong ndnan=x&0x7ff8000000000000ul;
    if(ndnan>0x7fe0000000000000ul) return 0x3ff0000000000000ul|sign;
    long index=as_type<long>(ndnan-0x3fc0000000000000ul)>>51;
    index=index<0 ? 0:(index>15 ? 15:index);
    uint offset=uint(index)*18u;
    ulong y=d_sub(d_abs(x),hyper_numpy_tanh_table[offset]);
    ulong result=hyper_numpy_tanh_table[offset+17];
    for(int j=16;j>=1;--j) result=hyper_fma_normal(result,y,hyper_numpy_tanh_table[offset+j]);
    return result|sign;
}
inline ulong _eon_hyper_tanh_numpy(ulong x) { return d_hyper_tanh_numpy(x); }
'''

# The public branch guards bound every polynomial operand away from subnormal
# range (unless exactly zero): sinh >=2^-27, cosh >=2^-30, NumPy tanh >=2^-27;
# table reductions retain at most106 significant product bits. Tiny paths use
# hyper_fma_any. The integer106-bit product and rounding core are unchanged.
_NORMAL_FMA_HEADER = r'''
inline ulong hyper_fma_normal(ulong a,ulong b,ulong c) {
    int ea=int(d_exp(a)),eb=int(d_exp(b)),ec=int(d_exp(c));
    if(d_is_zero(a) || d_is_zero(b)) {
        if(d_is_zero(c)) return (a^b)&c&0x8000000000000000ul;
        return c;
    }
    if(d_is_zero(c)) return d_mul(a,b);
    ulong sign=(a^b)>>63;
    ulong ma=d_mant(a)|(1ul<<52),mb=d_mant(b)|(1ul<<52),mc=d_mant(c)|(1ul<<52);
    ulong ph,pl=d_umul128(ma,mb,&ph);
    // Normal 53-bit operands put the product's highest bit at 104 or 105. Normalizing this
    // known width avoids variable leading-zero work in every polynomial FMA.
    int top=(ph>>41) ? 105:104;
    int pe=ea+eb-2150+top;
    hyper_u128 product=hyper_shl128({pl,ph},126-top);
    int ce=ec-1023;
    hyper_u128 addend={0ul,mc<<10};
    int exponent=pe>ce ? pe:ce;
    product=hyper_shr128(product,exponent-pe);
    addend=hyper_shr128(addend,exponent-ce);
    hyper_u128 result;
    if(sign==(c>>63)) {
        ulong lo=product.lo+addend.lo;
        result={lo,product.hi+addend.hi+ulong(lo<product.lo)};
    } else {
        bool smaller=product.hi!=addend.hi ? product.hi<addend.hi : product.lo<addend.lo;
        hyper_u128 large=smaller ? addend:product,small=smaller ? product:addend;
        result={large.lo-small.lo,large.hi-small.hi-ulong(large.lo<small.lo)};
        if(smaller) sign=c>>63;
    }
    if((result.lo|result.hi)==0ul) return 0ul;
    int rt=hyper_top128(result);
    hyper_u128 rounded=rt>55 ? hyper_shr128(result,rt-55):hyper_shl128(result,55-rt);
    int result_exponent=exponent+rt-126+1023;
    if(result_exponent<=0 || result_exponent>=2047)
        return d_finish(sign,result_exponent,rounded.lo);
    // rounded.lo already has its leading bit at55. Keep ArrowMetal's final
    // round-to-even/pack rules, without normalizing this known width again.
    ulong mantissa=rounded.lo>>3,low=rounded.lo&7ul;
    if(low>4ul || (low==4ul && (mantissa&1ul))) ++mantissa;
    if(mantissa>>53) { mantissa>>=1; ++result_exponent; }
    if(result_exponent>=2047) return (sign<<63)|D_INF;
    return (sign<<63)|(ulong(result_exponent)<<52)|(mantissa&0xffffffffffffful);
}
'''

# ArrowMetal's existing 53x53 product and rounding rules, specialized only in
# normalization: the retained product's leading bit is known to be55or56.
# Exceptional operands delegate unchanged; subnormal/overflowing products use
# the existing finish path. No approximate floating-point seed is involved.
# Apache-2.0; see third_party/ArrowMetal-hyperbolic-multiplication-NOTICE.
_NORMAL_MUL_HEADER = r'''
#ifndef EON_HYPER_MUL_INCLUDED
#define EON_HYPER_MUL_INCLUDED
inline ulong hyper_mul(ulong a,ulong b) {
    int ea=int(d_exp(a)),eb=int(d_exp(b));
    if(uint(ea-1)>=2046u || uint(eb-1)>=2046u) return d_mul(a,b);
    ulong sign=(a^b)>>63;
    ulong ma=d_mant(a)|(1ul<<52),mb=d_mant(b)|(1ul<<52);
    uint a1=uint(ma>>32),a0=uint(ma),b1=uint(mb>>32),b0=uint(mb);
    ulong p00=ulong(a0)*ulong(b0);
    ulong mid=ulong(a0)*ulong(b1)+ulong(a1)*ulong(b0);
    ulong lo=p00+(mid<<32);
    ulong hi=ulong(a1)*ulong(b1)+(mid>>32)+ulong(lo<p00);
    ulong m=(hi<<15)|(lo>>49)|ulong((lo&((1ul<<49)-1ul))!=0ul);
    int exponent=ea+eb-1023;
    if(m>>56) {m=(m>>1)|(m&1ul);++exponent;}
    if(exponent<=0 || exponent>=2047) return d_finish(sign,exponent,m);
    ulong mantissa=m>>3,low=m&7ul;
    if(low>4ul || (low==4ul && (mantissa&1ul))) ++mantissa;
    if(mantissa>>53) {mantissa>>=1;++exponent;}
    if(exponent>=2047) return (sign<<63)|D_INF;
    return (sign<<63)|(ulong(exponent)<<52)|(mantissa&0xffffffffffffful);
}
#endif // EON_HYPER_MUL_INCLUDED
'''

def metal_hyperbolic_header(functions, target="numpy"):
    """Select only requested functions and their exact arithmetic dependencies."""
    from .stable_math64 import _STABLE_FMA_HEADER
    names=set(functions)&{"log10","sinh","cosh","tanh"}
    if not names:
        return ""
    numpy_tanh=target=="numpy"
    header=(_STABLE_FMA_HEADER.replace("EON_STABLE_FMA_INCLUDED", "EON_HYPER_FMA_INCLUDED")
        .replace("stable_", "hyper_").replace("inline ulong hyper_fma(", "inline ulong hyper_fma_any(")
        + _NORMAL_FMA_HEADER
        + (_EXP_SHARED_HEADER if names&{"sinh","cosh"} or "tanh" in names and not numpy_tanh else "")
        + (_LOG10_HEADER if "log10" in names else "")
        + (_SINH_HEADER if "sinh" in names else "")
        + (_COSH_HEADER if "cosh" in names else "")
        + ((_TANH_NUMPY_HEADER if numpy_tanh else _TANH_SCALAR_HEADER) if "tanh" in names else ""))
    return _NORMAL_MUL_HEADER+header.replace('d_mul(', 'hyper_mul(')
