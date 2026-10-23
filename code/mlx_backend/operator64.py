"""Floating operators on raw binary64, preserving the pinned CPU target.

Power follows macOS ARM64 libsystem_m's compensated log2/product/exp2
sequence; it is not exp(y*log(x)). Numerical tables are frozen below. The
helpers require binary64.MSL_HEADER and math64.MSL_MATH_HEADER first.
"""

from .scalar_math_tables import exponential64_tables

from .stable_math64 import _STABLE_FMA_HEADER, _STABLE_DIV_HEADER

# The existing completed 128-bit FMA retains every product bit for nearby
# cancellation and falls back to the general FMA at exponent boundaries. Keep
# private names so a model using stable functions and power includes it once.
_OPERATOR_FMA_HEADER = (r'''
#ifndef EON_OPERATOR_FMA_DEFINED
#define EON_OPERATOR_FMA_DEFINED
// Normal binary64 significands have only53bits. Their cross-product sum is
// less than2^54, so it cannot overflow64bits and needs no general carry word.
inline ulong _eon_operator_mul53(ulong a,ulong b,thread ulong* high) {
    uint a0=uint(a),a1=uint(a>>32),b0=uint(b),b1=uint(b>>32);
    ulong p00=ulong(a0)*ulong(b0);
    ulong middle=ulong(a0)*ulong(b1)+ulong(a1)*ulong(b0);
    ulong low=p00+(middle<<32);
    *high=ulong(a1)*ulong(b1)+(middle>>32)+ulong(low<p00);
    return low;
}
inline ulong _eon_operator_mul(ulong a,ulong b) {
    int ea=int(d_exp(a)),eb=int(d_exp(b));
    if(uint(ea-1)>=2046u || uint(eb-1)>=2046u) return d_mul(a,b);
    ulong high,low=_eon_operator_mul53(d_mant(a)|(1ul<<52),d_mant(b)|(1ul<<52),&high);
    int carry=int(high>>41),exponent=ea+eb-1023+carry;
    // A normal53x53-bit product has its leading bit at104or105. Round that
    // exact106-bit product directly; generic normalization is unnecessary.
    // Keep the canonical gradual-underflow path before any rounding occurs.
    if(exponent<=0) return d_mul(a,b);
    ulong sign=(a^b)&0x8000000000000000ul;
    if(exponent>=2047) return sign|D_INF;
    int shift=52+carry;
    ulong mantissa=(high<<(64-shift))|(low>>shift);
    ulong tail=low&((1ul<<shift)-1ul),halfway=1ul<<(shift-1);
    mantissa+=ulong(tail>halfway || (tail==halfway && (mantissa&1ul)));
    if(mantissa>>53) {mantissa>>=1;++exponent;}
    if(exponent>=2047) return sign|D_INF;
    return sign|(ulong(exponent)<<52)|(mantissa&0xffffffffffffful);
}
'''+ _STABLE_FMA_HEADER.replace("EON_STABLE_FMA_INCLUDED", "EON_OPERATOR_STABLE_FMA_INCLUDED").replace("stable_", "_eon_operator_").replace(
    'd_umul128(ma,mb,&ph)', '_eon_operator_mul53(ma,mb,&ph)') + r'''
inline ulong _eon_operator_polynomial_fma(ulong a,ulong b,ulong c) {
    int ea=int(d_exp(a)),eb=int(d_exp(b)),ec=int(d_exp(c));
    // With a normal addend at least four binary exponents above the product,
    // the addend's53bits fit exactly at bit61 and cancellation loses at most
    // one leading bit. Only the product loses bits: shift-right-jam retains
    // every rounding decision with at least eight extra bits. No two rounded
    // tails are added. All close cancellation uses the full128-bit path.
    if(uint(ea-1)<2046u && uint(eb-1)<2046u && uint(ec-1)<2046u
            && ec-ea-eb+1022>=4) {
        ulong ma=d_mant(a)|(1ul<<52),mb=d_mant(b)|(1ul<<52),mc=d_mant(c)|(1ul<<52);
        ulong high,low=_eon_operator_mul53(ma,mb,&high);
        int carry=int(high>>41),shift=43+carry;
        ulong product=(high<<(64-shift))|(low>>shift)|ulong((low&((1ul<<shift)-1ul))!=0ul);
        int gap=ec-ea-eb+1023-carry;
        if(gap>=64) product=1ul;
        else product=(product>>gap)|ulong((product<< (64-gap))!=0ul);
        ulong addend=mc<<9;
        ulong result=((a^b^c)>>63) ? addend-product:addend+product;
        // The addend dominates by at least four exponent positions. The
        // result's highest bit is therefore60,61or62; no arbitrary-width
        // normalization is needed. Keep the general underflow path at ec=1.
        if(ec==1) return d_finish(c>>63,ec-6,result);
        int pack_shift=(result>>62) ? 10:((result>>61) ? 9:8);
        int exponent=ec+pack_shift-9;
        ulong mantissa=result>>pack_shift,tail=result&((1ul<<pack_shift)-1ul);
        ulong halfway=1ul<<(pack_shift-1);
        mantissa+=ulong(tail>halfway || (tail==halfway && (mantissa&1ul)));
        if(mantissa>>53) {mantissa>>=1;++exponent;}
        if(exponent>=2047) return (c&0x8000000000000000ul)|D_INF;
        return (c&0x8000000000000000ul)|(ulong(exponent)<<52)|(mantissa&0xffffffffffffful);
    }
    return _eon_operator_fma(a,b,c);
}
#endif
''')

_POW_TABLES = r'''
constant ulong _eon_pow_log_coeff[11]={
    0xbfc714b312ffc98eul,0x3fca619f928980a6ul,0xbfcec709dd391af8ul,0x3fd2776c50e28d61ul,
    0xbfd71547652b82c7ul,0x3fdec709dc3a0400ul,0x3c5746f307c00c8eul,0xbfe71547652b82feul,
    0xbc677aa5cbc36d0ful,0x3ff71547652b82feul,0x3c7777cbae0ed1aaul,
};
constant ulong _eon_pow_log_table[512]={
    0x000000003ff00000ul,0x0000000000000000ul,0xbc0ebef93fefc068ul,0x3f8706adaf949f49ul,
    0x3bab0b5d3fef81a8ul,0x3f96f6435f8f3269ul,0x3c2ec7643fef445dul,0x3fa11d9d6edefdb8ul,
    0x3bbbc33d3fef0871ul,0x3fa6aa8c69f82995ul,0x3b8261bf3feecc32ul,0x3fac4a04287d6529ul,
    0xbb6669b23fee920ful,0x3fb0e0c68e424aceul,0xbc0ca7313fee56eeul,0x3fb3add69167a803ul,
    0xbc20d8513fee1e52ul,0x3fb6617ac26d3c85ul,0xbbc630d63fede50ful,0x3fb92246d1c79540ul,
    0x3b74d1953fedad92ul,0x3fbbd24b77dbb164ul,0x3c043bb83fed7712ul,0x3fbe7af7c8933e03ul,
    0x3c6e54c13fed411bul,0x3fc090e9025eddd4ul,0x3c1ef8a03fed0cfeul,0x3fc1db051c140737ul,
    0xbc117e843fecd75dul,0x3fc33127811d49f6ul,0xbc15ffe33feca3c2ul,0x3fc47cbbf2792a04ul,
    0x3bf265a13fec72bbul,0x3fc5b9eaf8e6028bul,0x3c423d8f3fec404bul,0x3fc70275f1837cd6ul,
    0x3c438f233fec0df8ul,0x3fc84c8ea3631e5bul,0x3bca3acf3febdc79ul,0x3fc9937c528da973ul,
    0xbb9531f63febac9aul,0x3fcad1d89ecd0361ul,0x3c6563203feb7d57ul,0x3fcc0e457f1a4606ul,
    0xbc0c163d3feb4e23ul,0x3fcd4c6ec7f0fa9bul,0xbbba40e13feb2043ul,0x3fce83b51de1e130ul,
    0xbbc7e2273feaf2f4ul,0x3fcfb929fcb47417ul,0xbc1c14b63feac4f8ul,0x3fd07aa91b2ca230ul,
    0x3bd5ff933fea985dul,0x3fd1150405368efcul,0x3c1cf38b3fea6c72ul,0x3fd1adfba4172d24ul,
    0x3c054efc3fea41fcul,0x3fd242d294235c3aul,0xbba4d3ba3fea1613ul,0x3fd2ddbbfb7f3c85ul,
    0xbc5100723fe9ec65ul,0x3fd371b8e9222827ul,0xbc7cd2013fe9c1f1ul,0x3fd4096a3fa76078ul,
    0x3bdf75713fe999e1ul,0x3fd49976c7cfb747ul,0xbc041a1c3fe971c7ul,0x3fd52a89b6126155ul,
    0xbbd3ca043fe9495aul,0x3fd5bdb10e1494ceul,0x3c039bdd3fe92122ul,0x3fd65101a7359962ul,
    0x3b93fcad3fe8fa0aul,0x3fd6e11602431b0eul,0x3c3c1b243fe8d2f8ul,0x3fd771f64145b29ful,
    0x3c2f706a3fe8ada3ul,0x3fd7fd3a0cd1729bul,0x3bd918b83fe887a8ul,0x3fd88bc1f2b8fde1ul,
    0xbc2608613fe861f5ul,0x3fd91a168bad4093ul,0xbbfe1d823fe83bacul,0x3fd9ab83601c2e99ul,
    0xbc2aee253fe81812ul,0x3fda338d65098fbful,0x3c1a69003fe7f4e7ul,0x3fdabab52f19a9cbul,
    0xbc0f037e3fe7d08aul,0x3fdb47463d06a5f0ul,0x3c2340b73fe7adabul,0x3fdbcedccd4da020ul,
    0xbbed72fc3fe78ac8ul,0x3fdc574b4f99ab6dul,0xbc2e1aae3fe7674eul,0x3fdce2d8491411daul,
    0xbc5131033fe74516ul,0x3fdd6a3ba17a98beul,0xbc118bbe3fe7237eul,0x3fddefe824a6fe07ul,
    0x3c2e68a13fe701f8ul,0x3fde760f0bc2be8bul,0x3c3ea09e3fe6e2c1ul,0x3fdef3a91e17bbc9ul,
    0xbc6f56bb3fe6c240ul,0x3fdf77299fcad5deul,0x3c4a86963fe6a196ul,0x3fdffc0dbd620500ul,
    0xbc25c2883fe6810cul,0x3fe04097743d5c6ful,0x3c7fac703fe66156ul,0x3fe081d335d14e25ul,
    0x3c33a9f83fe6421bul,0x3fe0c26c86b500dful,0x3c6397df3fe6239ful,0x3fe101d26a8d3ec4ul,
    0x3c4eed683fe6057dul,0x3fe140d32757ef1bul,0xbbdc85393fe5e6c8ul,0x3fe181602d908fc1ul,
    0x3c257ab03fe5c88ful,0x3fe1c1412597d32dul,0x3bc639043fe5ac59ul,0x3fe1fd31c7e401e5ul,
    0x3c5c090a3fe58e2ful,0x3fe23d9f51b99eb2ul,0x3c167a3e3fe572b0ul,0x3fe278a8c472c40ful,
    0xbbf2e2123fe5552cul,0xbfda8f468bea6babul,0x3c50b09f3fe538c4ul,0xbfda1401a35e2207ul,
    0x3c3d724e3fe51da3ul,0xbfd99dad548d5854ul,0x3bdb2df33fe5014ful,0xbfd9217b478247c6ul,
    0x3c7dc05c3fe4e5f6ul,0xbfd8a8f65b8148dcul,0xbc6280263fe4cb44ul,0xbfd832b8e69895b4ul,
    0xbc42ec4d3fe4af97ul,0xbfd7b78304d09278ul,0x3be07f8f3fe494dful,0xbfd73ff3131de57aul,
    0x3c5aa8043fe47a6eul,0xbfd6c907a7936f76ul,0xbc23a9d13fe46037ul,0xbfd6528920cd0775ul,
    0x3c157e153fe44758ul,0xbfd5e190433aef0eul,0xbc1c5e8c3fe42d10ul,0xbfd56998ac049478ul,
    0xbc26d1ef3fe414a2ul,0xbfd4f9884d6cfb2ful,0x3c57b5cf3fe3faa2ul,0xbfd481adba3bc479ul,
    0x3c4c44373fe3e1a4ul,0xbfd40de517e6879cul,0xbc702b7b3fe3c990ul,0xbfd39dce8d0c6f05ul,
    0x3c7803ce3fe3b1bbul,0xbfd32e56a8c00a1eul,0xbc76ff603fe398b0ul,0xbfd2b8a31c011109ul,
    0x3bd74b4a3fe380a5ul,0xbfd24714e481adf7ul,0x3bd6010c3fe36a2cul,0xbfd1dc7279db548ful,
    0x3c48b8973fe351dcul,0xbfd16889c7153834ul,0xbc2a84b23fe339e9ul,0xbfd0f5cd7fa7752cul,
    0xbbd88c373fe323acul,0xbfd08ac35435c943ul,0xbbc6fbf93fe30c53ul,0xbfd019dc059d0e54ul,
    0x3bf111ca3fe2f6dbul,0xbfcf6320a3e9eb20ul,0xbbfb7cc43fe2e05bul,0xbfce87862d120f6bul,
    0xbbc4f1023fe2ca3aul,0xbfcdae8b05ec1659ul,0x3c056e0a3fe2b461ul,0xbfccd75698d83d8eul,
    0x3bf96f3e3fe29f22ul,0xbfcc051d7f25d8c7ul,0xbbec0f453fe28808ul,0xbfcb1f77901c26d4ul,
    0xbbf3bc703fe27389ul,0xbfca52c7fa334e41ul,0xbbbf1f003fe25d32ul,0xbfc972ab200ae538ul,
    0xbc6a2c6d3fe249c6ul,0xbfc8aef62d76aa72ul,0x3c41beab3fe233c6ul,0xbfc7d04afe225801ul,
    0xbc54f6153fe21f57ul,0xbfc70088de1f5b2dul,0x3c290e853fe20b41ul,0xbfc6336ad8a92825ul,
    0xbba6ad5e3fe1f655ul,0xbfc55cd01eaa9487ul,0x3bc5753a3fe1e203ul,0xbfc48b713902f7feul,
    0xbbdbbf393fe1ce6eul,0xbfc3c0cc1d1b1692ul,0xbb8ae1d33fe1bb6eul,0xbfc2fb577a193876ul,
    0xbbd4d76f3fe1a8acul,0xbfc2379731e9df8ful,0x3c0bb1b03fe194f0ul,0xbfc168c50866af19ul,
    0x3c3a48a53fe18154ul,0xbfc09a5bcb833e9eul,0x3ba5a0983fe16ed7ul,0xbfbfadd6b7f005b1ul,
    0x3c0b41903fe15b12ul,0xbfbe0a139ddec9d0ul,0xbc4080e43fe14803ul,0xbfbc73a3428818bcul,
    0x3bc9b8cb3fe1358aul,0xbfbae8051978c11bul,0xbc2d8b833fe12303ul,0xbfb9598fa7feff2bul,
    0xbba7a1293fe11141ul,0xbfb7da11dd576bbeul,0xbbe388f73fe0fe07ul,0xbfb63919388d8212ul,
    0x3bedc80c3fe0ed3aul,0xbfb4cb396209a15aul,0x3c597adf3fe0da57ul,0xbfb32e39f0a97dd1ul,
    0x3bfa2e1c3fe0ca09ul,0xbfb1c83bbb91b133ul,0x3b44b8943fe0b80bul,0xbfb03b9bf822af8eul,
    0x3bc4ae6c3fe0a76cul,0xbfad977378a15b0cul,0xbc473d493fe095ddul,0xbfaa8b0f9d7eed2ful,
    0x3bfc4c603fe084a8ul,0xbfa78b28a5f18eb0ul,0xbbc08d1e3fe07294ul,0xbfa461023c6c2b3ful,
    0x3b9e5abc3fe062faul,0xbfa1a30ab127d27ful,0xbbf9605e3fe05187ul,0xbf9d1dafa56778aful,
    0x3bcf176c3fe04027ul,0xbf96f5700660f22dul,0x3c1200bd3fe02fe4ul,0xbf912c35b1e2912eul,
    0xbc1b6a1c3fe01f94ul,0xbf86b1027d8347eaul,0x3b806a723fe00f11ul,0xbf75b2440dc383d5ul,
    0x000000003fe00000ul,0x0000000000000000ul,0x0000000000000000ul,0x3d25e52272171bcdul,
    0x3dc3b2a8abdac87bul,0x3e5c6b08d783107aul,0x3eeebfbdff82bda7ul,0x3f762e42fefa39eful,
    0x3ff0000000000000ul,0x0000000000000000ul,0x3ff0163da9fb3335ul,0xbd13a4bc88ab6f0bul,
    0x3ff02c9a3e778061ul,0x3ce91137d36a5dacul,0x3ff04315e86e7f85ul,0x3d079d9482b6e550ul,
    0x3ff059b0d3158574ul,0xbd04ca556a1b2fa8ul,0x3ff0706b29ddf6deul,0x3d040ee9a808de1dul,
    0x3ff0874518759bc8ul,0xbcd87a10f84f53b7ul,0x3ff09e3ecac6f383ul,0xbd0801b99f4f5b18ul,
    0x3ff0b5586cf9890ful,0xbd1106d9ba253401ul,0x3ff0cc922b7247f7ul,0xbd06269675ff4e35ul,
    0x3ff0e3ec32d3d1a2ul,0xbcb62d2d983e6844ul,0x3ff0fb66affed31bul,0x3ce2c399e6b423b5ul,
    0x3ff11301d0125b51ul,0x3d0ec8769e7386f7ul,0x3ff12abdc06c31ccul,0x3cc7cf4b389b60d3ul,
    0x3ff1429aaea92de0ul,0x3d09a8cab70c1007ul,0x3ff15a98c8a58e51ul,0xbcf846f8d141b1d0ul,
    0x3ff172b83c7d517bul,0x3cf73c5aec1a0547ul,0x3ff18af9388c8deaul,0x3d0673a724b46d55ul,
    0x3ff1a35beb6fcb75ul,0xbd03dd2c819e0908ul,0x3ff1bbe084045cd4ul,0x3d107b9a3e9b6a86ul,
    0x3ff1d4873168b9aaul,0xbd136c3949dfc34eul,0x3ff1ed5022fcd91dul,0x3d07039d1ea652dcul,
    0x3ff2063b88628cd6ul,0xbd03118ae698d206ul,0x3ff21f49917ddc96ul,0xbcf7c551db68456ful,
    0x3ff2387a6e756238ul,0xbd1045c6a6d3eecbul,0x3ff251ce4fb2a63ful,0xbd00db2bd14acf32ul,
    0x3ff26b4565e27cddul,0xbcf77bf4b6a48214ul,0x3ff284dfe1f56381ul,0x3d1063b457a17edbul,
    0x3ff29e9df51fdee1ul,0xbcfb5d8e0885ab1bul,0x3ff2b87fd0dad990ul,0x3cb5038687737beeul,
    0x3ff2d285a6e4030bul,0xbd03a1fee8c9225cul,0x3ff2ecafa93e2f56ul,0xbce5b2c9eff60911ul,
    0x3ff306fe0a31b715ul,0xbcfbd9046b69ea24ul,0x3ff32170fc4cd831ul,0xbd000e3e54fb7d75ul,
    0x3ff33c08b26416fful,0xbd06fc3180aac683ul,0x3ff356c55f929ff1ul,0x3d005491f94cc017ul,
    0x3ff371a7373aa9cbul,0x3d0a6409cdf25d3aul,0x3ff38cae6d05d866ul,0x3d120e5f6745f2e3ul,
    0x3ff3a7db34e59ff7ul,0x3ce9b570deeb4717ul,0x3ff3c32dc313a8e5ul,0x3d121abaf1248342ul,
    0x3ff3dea64c123422ul,0xbcff31b9fb32e2e4ul,0x3ff3fa4504ac801cul,0x3d0b83c7e85e695ful,
    0x3ff4160a21f72e2aul,0x3cd1c8d3f3908470ul,0x3ff431f5d950a897ul,0x3cf452cbd4449600ul,
    0x3ff44e086061892dul,0xbcbbf96ee8afb9fbul,0x3ff46a41ed1d0057ul,0xbd10283b91b051a3ul,
    0x3ff486a2b5c13cd0ul,0xbce637bd8150db7dul,0x3ff4a32af0d7d3deul,0xbd0cd9f521d33a36ul,
    0x3ff4bfdad5362a27ul,0xbcf04715f70da237ul,0x3ff4dcb299fddd0dul,0xbd0b942d586961a7ul,
    0x3ff4f9b2769d2ca7ul,0x3d06c785171c0daaul,0x3ff516daa2cf6642ul,0x3d013802b6c9135aul,
    0x3ff5342b569d4f82ul,0x3cf1f0a88738ea6aul,0x3ff551a4ca5d920ful,0x3cffd7a06db1d675ul,
    0x3ff56f4736b527daul,0xbd0bb5c29f573dd8ul,0x3ff58d12d497c7fdul,0xbcf3e81d7478cc39ul,
    0x3ff5ab07dd485429ul,0xbd07a560f05eeddbul,0x3ff5c9268a5946b7ul,0xbcadfa723ae1bf17ul,
    0x3ff5e76f15ad2148ul,0xbd0d2407ce0f6433ul,0x3ff605e1b976dc09ul,0x3d04d74bbcd33652ul,
    0x3ff6247eb03a5585ul,0x3d0457f6a88ab15ful,0x3ff6434634ccc320ul,0x3cfd5305865039dful,
    0x3ff6623882552225ul,0x3d0c93b642e08a78ul,0x3ff68155d44ca973ul,0xbcd0a350f225390eul,
    0x3ff6a09e667f3bcdul,0x3d0c6cdcb8cfcc24ul,0x3ff6c012750bdabful,0x3ce2cec326ee0b73ul,
    0x3ff6dfb23c651a2ful,0x3cdbff379bac560eul,0x3ff6ff7df9519484ul,0x3cf85304766e7062ul,
    0x3ff71f75e8ec5f74ul,0x3cf1669aa6aea2d3ul,0x3ff73f9a48a58174ul,0x3d008a8544a7c319ul,
    0x3ff75feb564267c9ul,0x3cffe19bb8ecd7c4ul,0x3ff780694fde5d3ful,0xbd07f77c42eaf147ul,
    0x3ff7a11473eb0187ul,0x3cf39ea4b4e0a91dul,0x3ff7c1ed0130c132ul,0xbd0e308095c8fab2ul,
    0x3ff7e2f336cf4e62ul,0xbcdfa03bd08bcda3ul,0x3ff80427543e1a12ul,0x3d01c4a6d87e2162ul,
    0x3ff82589994cce13ul,0x3d0c01c85abda967ul,0x3ff8471a4623c7adul,0x3cf79da32bfd8963ul,
    0x3ff868d99b4492edul,0x3d0e0ce7d31c17b4ul,0x3ff88ac7d98a6699ul,0xbd080f72957b0484ul,
    0x3ff8ace5422aa0dbul,0xbd056f6cd720fba3ul,0x3ff8cf3216b5448cul,0x3cdf530b22c146b2ul,
    0x3ff8f1ae99157736ul,0xbcf42bc69235780ful,0x3ff9145b0b91ffc6ul,0x3d0b766e6aa0a1dful,
    0x3ff93737b0cdc5e5ul,0x3cd565b633aaadeeul,0x3ff95a44cbc8520ful,0x3ce44c91b60b1f8dul,
    0x3ff97d829fde4e50ul,0x3d0a58fa485235a1ul,0x3ff9a0f170ca07baul,0x3cff6ff7608a3078ul,
    0x3ff9c49182a3f090ul,0xbce984806660caa6ul,0x3ff9e86319e32323ul,0xbce582eb35cac3f3ul,
    0x3ffa0c667b5de565ul,0x3d0125694bb46c7eul,0x3ffa309bec4a2d33ul,0xbd038e88295250eful,
    0x3ffa5503b23e255dul,0x3d099593455a26b5ul,0x3ffa799e1330b358ul,0xbd083bde4707301aul,
    0x3ffa9e6b5579fdbful,0xbcfd72d3f1e40a93ul,0x3ffac36bbfd3f37aul,0x3cfb3ad1802c5436ul,
    0x3ffae89f995ad3adul,0xbd0445b6038d7018ul,0x3ffb0e07298db666ul,0x3d07c789b4248453ul,
    0x3ffb33a2b84f15fbul,0x3ccf667a98d32ac1ul,0x3ffb59728de5593aul,0x3d0801f60c88f9bdul,
    0x3ffb7f76f2fb5e47ul,0x3ce1eb0404780ad1ul,0x3ffba5b030a1064aul,0x3d09df3f08eda153ul,
    0x3ffbcc1e904bc1d2ul,0xbcee4bb29d39c3a3ul,0x3ffbf2c25bd71e09ul,0x3d0998b2be294a3ful,
    0x3ffc199bdd85529cul,0xbcec08f260a0042aul,0x3ffc40ab5fffd07aul,0xbd0647d6355e9b74ul,
    0x3ffc67f12e57d14bul,0xbcfe1e92a1c04f0dul,0x3ffc8f6d9406e7b5ul,0xbcdc920234dbf34dul,
    0x3ffcb720dcef9069ul,0xbce0e49a8304ef1ful,0x3ffcdf0b555dc3faul,0x3cf7dc824b04c033ul,
    0x3ffd072d4a07897cul,0x3d06d9ab77ff22eful,0x3ffd2f87080d89f2ul,0x3d072904080a5c48ul,
    0x3ffd5818dcfba487ul,0xbcedc682d68358e2ul,0x3ffd80e316c98398ul,0x3cfaca05497de9c9ul,
    0x3ffda9e603db3285ul,0xbd05e51917091778ul,0x3ffdd321f301b460ul,0xbcfd2ec7bfc90fd9ul,
    0x3ffdfc97337b9b5ful,0x3cfb2b6b3f445f87ul,0x3ffe264614f5a129ul,0x3d02277152860520ul,
    0x3ffe502ee78b3ff6ul,0xbcede1319abb2ae2ul,0x3ffe7a51fbc74c83ul,0xbcfc86cfb6cd46e4ul,
    0x3ffea4afa2a490daul,0x3d070ed6ccceec4bul,0x3ffecf482d8e67f1ul,0x3d05693ab0188765ul,
    0x3ffefa1bee615a27ul,0xbd063124664aaf80ul,0x3fff252b376bba97ul,0xbcfd19790d732ab4ul,
};
constant ulong _eon_pow_exp_coeff[5]={
    0x3d25e52272171bcdul,0x3dc3b2a8abdac87bul,0x3e5c6b08d783107aul,0x3eeebfbdff82bda7ul,
    0x3f762e42fefa39eful,
};
constant ulong _eon_pow_exp_table[256]={
    0x3ff0000000000000ul,0x0000000000000000ul,0x3ff0163da9fb3335ul,0xbd13a4bc88ab6f0bul,
    0x3ff02c9a3e778061ul,0x3ce91137d36a5dacul,0x3ff04315e86e7f85ul,0x3d079d9482b6e550ul,
    0x3ff059b0d3158574ul,0xbd04ca556a1b2fa8ul,0x3ff0706b29ddf6deul,0x3d040ee9a808de1dul,
    0x3ff0874518759bc8ul,0xbcd87a10f84f53b7ul,0x3ff09e3ecac6f383ul,0xbd0801b99f4f5b18ul,
    0x3ff0b5586cf9890ful,0xbd1106d9ba253401ul,0x3ff0cc922b7247f7ul,0xbd06269675ff4e35ul,
    0x3ff0e3ec32d3d1a2ul,0xbcb62d2d983e6844ul,0x3ff0fb66affed31bul,0x3ce2c399e6b423b5ul,
    0x3ff11301d0125b51ul,0x3d0ec8769e7386f7ul,0x3ff12abdc06c31ccul,0x3cc7cf4b389b60d3ul,
    0x3ff1429aaea92de0ul,0x3d09a8cab70c1007ul,0x3ff15a98c8a58e51ul,0xbcf846f8d141b1d0ul,
    0x3ff172b83c7d517bul,0x3cf73c5aec1a0547ul,0x3ff18af9388c8deaul,0x3d0673a724b46d55ul,
    0x3ff1a35beb6fcb75ul,0xbd03dd2c819e0908ul,0x3ff1bbe084045cd4ul,0x3d107b9a3e9b6a86ul,
    0x3ff1d4873168b9aaul,0xbd136c3949dfc34eul,0x3ff1ed5022fcd91dul,0x3d07039d1ea652dcul,
    0x3ff2063b88628cd6ul,0xbd03118ae698d206ul,0x3ff21f49917ddc96ul,0xbcf7c551db68456ful,
    0x3ff2387a6e756238ul,0xbd1045c6a6d3eecbul,0x3ff251ce4fb2a63ful,0xbd00db2bd14acf32ul,
    0x3ff26b4565e27cddul,0xbcf77bf4b6a48214ul,0x3ff284dfe1f56381ul,0x3d1063b457a17edbul,
    0x3ff29e9df51fdee1ul,0xbcfb5d8e0885ab1bul,0x3ff2b87fd0dad990ul,0x3cb5038687737beeul,
    0x3ff2d285a6e4030bul,0xbd03a1fee8c9225cul,0x3ff2ecafa93e2f56ul,0xbce5b2c9eff60911ul,
    0x3ff306fe0a31b715ul,0xbcfbd9046b69ea24ul,0x3ff32170fc4cd831ul,0xbd000e3e54fb7d75ul,
    0x3ff33c08b26416fful,0xbd06fc3180aac683ul,0x3ff356c55f929ff1ul,0x3d005491f94cc017ul,
    0x3ff371a7373aa9cbul,0x3d0a6409cdf25d3aul,0x3ff38cae6d05d866ul,0x3d120e5f6745f2e3ul,
    0x3ff3a7db34e59ff7ul,0x3ce9b570deeb4717ul,0x3ff3c32dc313a8e5ul,0x3d121abaf1248342ul,
    0x3ff3dea64c123422ul,0xbcff31b9fb32e2e4ul,0x3ff3fa4504ac801cul,0x3d0b83c7e85e695ful,
    0x3ff4160a21f72e2aul,0x3cd1c8d3f3908470ul,0x3ff431f5d950a897ul,0x3cf452cbd4449600ul,
    0x3ff44e086061892dul,0xbcbbf96ee8afb9fbul,0x3ff46a41ed1d0057ul,0xbd10283b91b051a3ul,
    0x3ff486a2b5c13cd0ul,0xbce637bd8150db7dul,0x3ff4a32af0d7d3deul,0xbd0cd9f521d33a36ul,
    0x3ff4bfdad5362a27ul,0xbcf04715f70da237ul,0x3ff4dcb299fddd0dul,0xbd0b942d586961a7ul,
    0x3ff4f9b2769d2ca7ul,0x3d06c785171c0daaul,0x3ff516daa2cf6642ul,0x3d013802b6c9135aul,
    0x3ff5342b569d4f82ul,0x3cf1f0a88738ea6aul,0x3ff551a4ca5d920ful,0x3cffd7a06db1d675ul,
    0x3ff56f4736b527daul,0xbd0bb5c29f573dd8ul,0x3ff58d12d497c7fdul,0xbcf3e81d7478cc39ul,
    0x3ff5ab07dd485429ul,0xbd07a560f05eeddbul,0x3ff5c9268a5946b7ul,0xbcadfa723ae1bf17ul,
    0x3ff5e76f15ad2148ul,0xbd0d2407ce0f6433ul,0x3ff605e1b976dc09ul,0x3d04d74bbcd33652ul,
    0x3ff6247eb03a5585ul,0x3d0457f6a88ab15ful,0x3ff6434634ccc320ul,0x3cfd5305865039dful,
    0x3ff6623882552225ul,0x3d0c93b642e08a78ul,0x3ff68155d44ca973ul,0xbcd0a350f225390eul,
    0x3ff6a09e667f3bcdul,0x3d0c6cdcb8cfcc24ul,0x3ff6c012750bdabful,0x3ce2cec326ee0b73ul,
    0x3ff6dfb23c651a2ful,0x3cdbff379bac560eul,0x3ff6ff7df9519484ul,0x3cf85304766e7062ul,
    0x3ff71f75e8ec5f74ul,0x3cf1669aa6aea2d3ul,0x3ff73f9a48a58174ul,0x3d008a8544a7c319ul,
    0x3ff75feb564267c9ul,0x3cffe19bb8ecd7c4ul,0x3ff780694fde5d3ful,0xbd07f77c42eaf147ul,
    0x3ff7a11473eb0187ul,0x3cf39ea4b4e0a91dul,0x3ff7c1ed0130c132ul,0xbd0e308095c8fab2ul,
    0x3ff7e2f336cf4e62ul,0xbcdfa03bd08bcda3ul,0x3ff80427543e1a12ul,0x3d01c4a6d87e2162ul,
    0x3ff82589994cce13ul,0x3d0c01c85abda967ul,0x3ff8471a4623c7adul,0x3cf79da32bfd8963ul,
    0x3ff868d99b4492edul,0x3d0e0ce7d31c17b4ul,0x3ff88ac7d98a6699ul,0xbd080f72957b0484ul,
    0x3ff8ace5422aa0dbul,0xbd056f6cd720fba3ul,0x3ff8cf3216b5448cul,0x3cdf530b22c146b2ul,
    0x3ff8f1ae99157736ul,0xbcf42bc69235780ful,0x3ff9145b0b91ffc6ul,0x3d0b766e6aa0a1dful,
    0x3ff93737b0cdc5e5ul,0x3cd565b633aaadeeul,0x3ff95a44cbc8520ful,0x3ce44c91b60b1f8dul,
    0x3ff97d829fde4e50ul,0x3d0a58fa485235a1ul,0x3ff9a0f170ca07baul,0x3cff6ff7608a3078ul,
    0x3ff9c49182a3f090ul,0xbce984806660caa6ul,0x3ff9e86319e32323ul,0xbce582eb35cac3f3ul,
    0x3ffa0c667b5de565ul,0x3d0125694bb46c7eul,0x3ffa309bec4a2d33ul,0xbd038e88295250eful,
    0x3ffa5503b23e255dul,0x3d099593455a26b5ul,0x3ffa799e1330b358ul,0xbd083bde4707301aul,
    0x3ffa9e6b5579fdbful,0xbcfd72d3f1e40a93ul,0x3ffac36bbfd3f37aul,0x3cfb3ad1802c5436ul,
    0x3ffae89f995ad3adul,0xbd0445b6038d7018ul,0x3ffb0e07298db666ul,0x3d07c789b4248453ul,
    0x3ffb33a2b84f15fbul,0x3ccf667a98d32ac1ul,0x3ffb59728de5593aul,0x3d0801f60c88f9bdul,
    0x3ffb7f76f2fb5e47ul,0x3ce1eb0404780ad1ul,0x3ffba5b030a1064aul,0x3d09df3f08eda153ul,
    0x3ffbcc1e904bc1d2ul,0xbcee4bb29d39c3a3ul,0x3ffbf2c25bd71e09ul,0x3d0998b2be294a3ful,
    0x3ffc199bdd85529cul,0xbcec08f260a0042aul,0x3ffc40ab5fffd07aul,0xbd0647d6355e9b74ul,
    0x3ffc67f12e57d14bul,0xbcfe1e92a1c04f0dul,0x3ffc8f6d9406e7b5ul,0xbcdc920234dbf34dul,
    0x3ffcb720dcef9069ul,0xbce0e49a8304ef1ful,0x3ffcdf0b555dc3faul,0x3cf7dc824b04c033ul,
    0x3ffd072d4a07897cul,0x3d06d9ab77ff22eful,0x3ffd2f87080d89f2ul,0x3d072904080a5c48ul,
    0x3ffd5818dcfba487ul,0xbcedc682d68358e2ul,0x3ffd80e316c98398ul,0x3cfaca05497de9c9ul,
    0x3ffda9e603db3285ul,0xbd05e51917091778ul,0x3ffdd321f301b460ul,0xbcfd2ec7bfc90fd9ul,
    0x3ffdfc97337b9b5ful,0x3cfb2b6b3f445f87ul,0x3ffe264614f5a129ul,0x3d02277152860520ul,
    0x3ffe502ee78b3ff6ul,0xbcede1319abb2ae2ul,0x3ffe7a51fbc74c83ul,0xbcfc86cfb6cd46e4ul,
    0x3ffea4afa2a490daul,0x3d070ed6ccceec4bul,0x3ffecf482d8e67f1ul,0x3d05693ab0188765ul,
    0x3ffefa1bee615a27ul,0xbd063124664aaf80ul,0x3fff252b376bba97ul,0xbcfd19790d732ab4ul,
    0x3fff50765b6e4540ul,0xbd0309e68da0499ful,0x3fff7bfdad9cbe14ul,0x3d05cc2651e079fbul,
    0x3fffa7c1819e90d8ul,0xbcf0fa4931906b80ul,0x3fffd3c22b8f71f1ul,0xbccb71923d729650ul,
};

'''

_POW_HEADER = r'''
inline ulong d_operator_pow(ulong x, ulong y) {
    const ulong one=0x3ff0000000000000ul;
    ulong ax=x&0x7ffffffffffffffful,ay=y&0x7ffffffffffffffful,sign=0ul;
    if(x==one) return one;
    if(y==one) return d_is_nan(x) ? d_add(x,x):x;
    if(ay==0ul) return one;
    if(d_is_nan(y)) return d_add(y,y);
    if(d_is_nan(x)) return d_add(x,x);
    if(ay==D_INF) {
        if(ax==one) return one;
        return ((ax>one) != bool(y>>63)) ? D_INF:0ul;
    }
    bool odd=false;
    // Exponent parity only changes negative-base signs/domain. Avoid the
    // software rounding/integer conversion on the positive finite path.
    if((x>>63) && ay>=one && ay<0x4340000000000000ul) {
        ulong integral=d_floor(ay);
        odd=integral==ay && bool(d_to_int(ay)&1L);
    }
    if(ax==0ul || ax==D_INF) {
        ulong result=(odd ? x:ax);
        return (y>>63) ? d_div(one,result):result;
    }
    if(x>>63) {
        if(ay<one || (ay<0x4340000000000000ul && d_floor(ay)!=ay)) return D_QNAN;
        if(odd) sign=0x8000000000000000ul;
        x=ax;
    }
    if(ay>0x43f0000000000000ul) {
        if(ax==one) return sign|one;
        return sign|(((ax>one) != bool(y>>63)) ? D_INF:0ul);
    }
    if(ay<0x3be0000000000000ul) return sign|one;
    // The CPU normalizes subnormals with an exact 1+x subtraction and adjusts
    // the encoded exponent, before the same table reduction as normal inputs.
    if(ax<0x0010000000000000ul)
        x=d_sub(x|one,one)+0xc020000000000000ul;
    ulong fraction=d_mant(x),product,error;
    if(fraction==0ul) {
        ulong log_scaled=d_mul(d_from_int(as_type<long>(x-one)),0x3d20000000000000ul);
        product=d_mul(log_scaled,y);
        error=d_fma(log_scaled,y,d_neg(product));
    } else {
        uint index=uint(((fraction+0x100000000000ul)&0x1fe00000000000ul)>>45);
        long exponent=as_type<long>(x+0xc018100000000000ul)>>52;
        ulong inv=ulong(uint(_eon_pow_log_table[2*index]))<<32;
        ulong m=fraction|one,m_high=m&0xffffffffffe00000ul;
        ulong value=d_mul(inv,m);
        ulong low=d_fma(inv,m_high,d_neg(value));
        low=d_fma(inv,d_sub(m,m_high),low);
        ulong r=d_sub(value,one);
        ulong p=d_fma(r,_eon_pow_log_coeff[0],_eon_pow_log_coeff[1]);
        p=d_fma(r,p,_eon_pow_log_coeff[2]);
        p=d_fma(r,p,_eon_pow_log_coeff[3]);
        p=d_fma(r,p,_eon_pow_log_coeff[4]);
        ulong square=d_mul(r,r);
        p=d_mul(d_mul(square,square),p);
        ulong hi0=d_fma(_eon_pow_log_coeff[5],r,_eon_pow_log_coeff[7]);
        ulong lo0=d_fma(_eon_pow_log_coeff[6],r,_eon_pow_log_coeff[8]);
        ulong delta=d_fma(_eon_pow_log_coeff[5],r,d_neg(d_sub(hi0,_eon_pow_log_coeff[7])));
        lo0=d_add(delta,d_fma(_eon_pow_log_coeff[5],low,lo0));
        ulong hi1=d_fma(hi0,r,_eon_pow_log_coeff[9]);
        ulong lo1=d_fma(lo0,r,_eon_pow_log_coeff[10]);
        delta=d_fma(hi0,r,d_neg(d_sub(hi1,_eon_pow_log_coeff[9])));
        lo1=d_add(delta,d_fma(hi0,low,lo1));
        ulong hi2=d_mul(hi1,r);
        ulong lo2=d_add(d_fma(hi1,r,d_neg(hi2)),d_fma(hi1,low,d_mul(lo1,r)));
        ulong e=d_from_int(exponent),table_hi=_eon_pow_log_table[2*index+1];
        ulong log_hi=d_add(d_add(d_add(e,table_hi),hi2),p)&0xfffffffffffffffeul;
        ulong log_lo=d_add(d_add(d_add(d_sub(e,log_hi),table_hi),hi2),p);
        log_lo=d_add(_eon_pow_log_table[2*index]&0xffffffff00000000ul,log_lo);
        log_lo=d_add(log_lo,lo2);
        y=d_mul(y,0x4060000000000000ul);
        product=d_mul(y,log_hi);
        error=d_fma(y,log_hi,d_neg(product));
        error=d_fma(y,log_lo,error);
    }
    if(d_gt(product,0x4100040000000000ul)) return sign|D_INF;
    if(d_lt(product,0xc100cc0000000000ul)) return sign;
    long k=d_to_int(d_floor(product));
    ulong r=d_sub(product,d_from_int(k));
    uint index=uint(k)&127u;
    ulong base=_eon_pow_exp_table[2*index],tail=_eon_pow_exp_table[2*index+1];
    ulong corrected=d_add(d_sub(error,tail),r);
    ulong p=d_fma(r,_eon_pow_exp_coeff[0],_eon_pow_exp_coeff[1]);
    p=d_fma(r,p,_eon_pow_exp_coeff[2]);
    p=d_fma(corrected,p,_eon_pow_exp_coeff[3]);
    p=d_fma(corrected,p,_eon_pow_exp_coeff[4]);
    p=d_mul(corrected,p);
    ulong scale0=(ulong(k)<<44)&0xfff0000000000000ul;
    ulong scale1=((ulong(k)<<45)&0xfff0000000000000ul)-scale0;
    base+=(scale0^sign);
    return d_mul(d_fma(base,p,base),scale1+one);
}
'''

MSL_POWER_HEADER = (_OPERATOR_FMA_HEADER + _POW_TABLES
                    + _POW_HEADER.replace("d_fma(", "_eon_operator_polynomial_fma(")
                                 .replace("d_mul(", "_eon_operator_mul("))

MSL_INTEGER_EXPONENT_HEADER = r'''
inline ulong d_operator_powi(ulong base,int exponent) {
    uint magnitude=exponent<0 ? uint(-long(exponent)):uint(exponent);
    ulong result=0x3ff0000000000000ul;
    while(true) {
        if(magnitude&1u) result=d_mul(result,base);
        magnitude>>=1;
        if(magnitude==0u) break;
        base=d_mul(base,base);
    }
    return exponent<0 ? d_div(0x3ff0000000000000ul,result):result;
}
'''

# The completed safe-math reciprocal uses two Newton refinements followed by
# an exact integer remainder correction. Keep this specialization local;
# operator kernels already require math_mode="safe".
_OPERATOR_DIV_HEADER = (_STABLE_DIV_HEADER
    .replace('EON_STABLE_DIV_INCLUDED','EON_OPERATOR_STABLE_DIV_INCLUDED')
    .replace('stable_div(', '_eon_operator_div('))

MSL_REMAINDER_HEADER = _OPERATOR_DIV_HEADER+r'''
// Arithmetic instructions prioritize signaling NaNs, and subtraction does not
// negate a NaN's sign. Keep this operator policy local to the audited operators.
inline ulong d_operator_nan2(ulong a,ulong b) {
    if(d_is_nan(a) && !(a&0x0008000000000000ul)) return a|0x0008000000000000ul;
    if(d_is_nan(b) && !(b&0x0008000000000000ul)) return b|0x0008000000000000ul;
    return d_is_nan(a) ? a:b;
}
inline ulong d_operator_add(ulong a,ulong b) {
    return d_is_nan(a)||d_is_nan(b) ? d_operator_nan2(a,b):d_add(a,b);
}
inline ulong d_operator_sub(ulong a,ulong b) {
    return d_is_nan(a)||d_is_nan(b) ? d_operator_nan2(a,b):d_sub(a,b);
}
inline ulong d_operator_mul(ulong a,ulong b) {
    return d_is_nan(a)||d_is_nan(b) ? d_operator_nan2(a,b):d_mul(a,b);
}
inline ulong d_operator_div(ulong a,ulong b) {
    return d_is_nan(a)||d_is_nan(b) ? d_operator_nan2(a,b):_eon_operator_div(a,b);
}
inline ulong d_operator_fmod(ulong x,ulong y) {
    ulong ax=d_abs(x),ay=d_abs(y),sign=x&0x8000000000000000ul;
    // ARM arithmetic quiets the first signaling NaN before quiet operands.
    if(d_is_nan(x) && !(x&0x0008000000000000ul)) return x|0x0008000000000000ul;
    if(d_is_nan(y) && !(y&0x0008000000000000ul)) return y|0x0008000000000000ul;
    if(d_is_nan(x)) return x;
    if(d_is_nan(y)) return y;
    if(ax==D_INF || ay==0ul) return D_QNAN;
    if(ax<ay) return x;
    if(ax==ay) return sign;
    int ex=int(d_exp(ax)),ey=int(d_exp(ay));
    ulong mx=d_mant(ax),my=d_mant(ay);
    if(ex) mx|=1ul<<52;
    else {int shift=int(clz(mx))-11;mx<<=shift;ex=1-shift;}
    if(ey) my|=1ul<<52;
    else {int shift=int(clz(my))-11;my<<=shift;ey=1-shift;}
    while(ex>ey) {
        if(mx>=my) mx-=my;
        if(mx==0ul) return sign;
        mx<<=1;--ex;
    }
    if(mx>=my) mx-=my;
    return d_finish(sign>>63,ey,mx<<3);
}
inline ulong d_operator_mod_numpy(ulong x,ulong y) {
    ulong r=d_operator_fmod(x,y);
    if(!d_is_zero(r)) {
        if(d_lt(y,0ul)!=d_lt(r,0ul)) r=d_operator_add(r,y);
    } else r=y&0x8000000000000000ul;
    return r;
}
inline ulong d_operator_floordiv_numpy(ulong x,ulong y) {
    ulong r=d_operator_fmod(x,y);
    if(d_is_zero(y)) return d_operator_div(x,y);
    ulong quotient=d_operator_div(d_operator_sub(x,r),y);
    if(!d_is_zero(r)) {
        if(d_lt(y,0ul)!=d_lt(r,0ul)) quotient=d_operator_sub(quotient,0x3ff0000000000000ul);
    }
    if(!d_is_zero(quotient)) {
        ulong result=d_floor(quotient);
        if(d_gt(d_operator_sub(quotient,result),0x3fe0000000000000ul))
            result=d_operator_add(result,0x3ff0000000000000ul);
        return result;
    }
    return d_operator_div(x,y)&0x8000000000000000ul;
}
inline ulong d_operator_cython_mod_once(ulong x,ulong y) {
    ulong r=d_operator_fmod(x,y);
    bool adapt=!d_is_zero(r) && (d_lt(r,0ul)!=d_lt(y,0ul));
    return d_operator_add(r,d_operator_mul(adapt ? 0x3ff0000000000000ul:0ul,y));
}
inline ulong d_operator_mod_cython(ulong x,ulong y,thread bool &zero_error) {
    zero_error|=d_is_zero(y);
    return d_operator_cython_mod_once(d_operator_add(d_operator_cython_mod_once(x,y),y),y);
}
// The optimized Cython helper fuses a divisor times a Boolean with an addend.
// Finite multiplication by zero/one is exact, so one addition implements the
// fused operation. ARM checks signaling NaNs before the infinity*zero invalid
// case, then prioritizes the addend's quiet NaN over the product operands.
inline ulong d_operator_mod_adjust(ulong y,bool adapt,ulong c) {
    if(d_is_nan(c) && !(c&0x0008000000000000ul)) return c|0x0008000000000000ul;
    if(d_is_nan(y) && !(y&0x0008000000000000ul)) return y|0x0008000000000000ul;
    if(!adapt && d_abs(y)==D_INF) return D_QNAN;
    if(d_is_nan(c)) return c;
    if(d_is_nan(y)) return y;
    return d_add(adapt ? y:(y&0x8000000000000000ul),c);
}
inline ulong d_operator_mod_cython_fma(ulong x,ulong y,thread bool &zero_error) {
    zero_error|=d_is_zero(y);
    ulong r=d_operator_fmod(x,y);
    bool adapt=!d_is_zero(r) && (d_lt(r,0ul)!=d_lt(y,0ul));
    r=d_operator_fmod(d_operator_mod_adjust(y,adapt,d_operator_add(r,y)),y);
    adapt=!d_is_zero(r) && (d_lt(r,0ul)!=d_lt(y,0ul));
    return d_operator_mod_adjust(y,adapt,r);
}
inline ulong d_operator_mod_cython_once(ulong x,ulong y,thread bool &zero_error) {
    zero_error|=d_is_zero(y);
    return d_operator_cython_mod_once(x,y);
}
inline ulong d_operator_mod_cython_once_fma(ulong x,ulong y,thread bool &zero_error) {
    zero_error|=d_is_zero(y);
    ulong r=d_operator_fmod(x,y);
    bool adapt=!d_is_zero(r) && (d_lt(r,0ul)!=d_lt(y,0ul));
    return d_operator_mod_adjust(y,adapt,r);
}
inline ulong d_operator_floordiv_cython(ulong x,ulong y,thread bool &zero_error) {
    zero_error|=d_is_zero(y);
    return d_floor(d_operator_div(x,y));
}
inline ulong d_operator_mod_cpp(ulong x,ulong y) {
    return d_operator_sub(x,d_operator_mul(y,d_floor(d_operator_div(x,y))));
}
inline ulong d_operator_fma(ulong a,ulong b,ulong c) {
    if(d_is_nan(c) && !(c&0x0008000000000000ul)) return c|0x0008000000000000ul;
    if(d_is_nan(a) && !(a&0x0008000000000000ul)) return a|0x0008000000000000ul;
    if(d_is_nan(b) && !(b&0x0008000000000000ul)) return b|0x0008000000000000ul;
    if((d_abs(a)==D_INF && d_is_zero(b)) || (d_abs(b)==D_INF && d_is_zero(a))) return D_QNAN;
    if(d_is_nan(c)) return c;
    if(d_is_nan(a)) return a;
    if(d_is_nan(b)) return b;
    return d_fma(a,b,c);
}
inline ulong d_operator_mod_cpp_fma(ulong x,ulong y) {
    // Actual default CPP emits FMSUB quotient,y,x. Its negation applies to
    // the quotient before ARM's addend-first NaN selection, including payloads.
    return d_operator_fma(d_neg(d_floor(d_operator_div(x,y))),y,x);
}
inline ulong d_operator_floordiv_cpp(ulong x,ulong y) {
    return d_floor(d_operator_div(x,y));
}
inline ulong d_operator_mod_cython(ulong x,ulong y,thread bool &error,thread int &kind) {
    if(d_is_zero(y) && !error) kind=1;
    return d_operator_mod_cython(x,y,error);
}
inline ulong d_operator_mod_cython_fma(ulong x,ulong y,thread bool &error,thread int &kind) {
    if(d_is_zero(y) && !error) kind=1;
    return d_operator_mod_cython_fma(x,y,error);
}
inline ulong d_operator_mod_cython_once(ulong x,ulong y,thread bool &error,thread int &kind) {
    if(d_is_zero(y) && !error) kind=1;
    return d_operator_mod_cython_once(x,y,error);
}
inline ulong d_operator_mod_cython_once_fma(ulong x,ulong y,thread bool &error,thread int &kind) {
    if(d_is_zero(y) && !error) kind=1;
    return d_operator_mod_cython_once_fma(x,y,error);
}
inline ulong d_operator_floordiv_cython(ulong x,ulong y,thread bool &error,thread int &kind) {
    if(d_is_zero(y) && !error) kind=2;
    return d_operator_floordiv_cython(x,y,error);
}
inline ulong d_operator_checked_division(ulong value,ulong divisor,thread bool &error,thread int &kind) {
    if(d_is_zero(divisor)) {if(!error) kind=2;error=true;}
    return value;
}
inline uint f_operator_checked_division(uint value,ulong divisor,thread bool &error,thread int &kind) {
    if(d_is_zero(divisor)) {if(!error) kind=2;error=true;}
    return value;
}
'''


# Frozen installed ARM64 exp2; code SHA256 39a9fcbda051283b38b386ae4e93376ce5fce9f655569a1b9c2f989551360d28.
_EXP2_TABLES = (
    "\n" + exponential64_tables('_eon_exp2_64_coeff', '_eon_exp2_64_table') + "\n"
)

MSL_EXP2_HEADER = _OPERATOR_FMA_HEADER+_EXP2_TABLES+r'''
inline ulong d_operator_exp2(ulong x) {
    ulong magnitude=d_abs(x);
    ulong limit=(x>>63) ? 0x4091000000000000ul:0x4090000000000000ul;
    if(magnitude>=limit) {
        if(d_is_nan(x) || !(x>>63)) return d_mul(x,0x7feffffffffffffful);
        return 0ul;
    }
    const ulong one=0x3ff0000000000000ul;
    long k=d_to_int(d_mul(x,0x4060000000000000ul))-long(x>>63);
    ulong rounded=d_mul(d_from_int(k),0x3f80000000000000ul);
    ulong residual=d_sub(x,rounded);
    uint index=uint(k)&127u;
    ulong base=_eon_exp2_64_table[2*index],tail=_eon_exp2_64_table[2*index+1];
    ulong a=d_add(residual,_eon_exp2_64_coeff[0]);
    ulong b=d_add(residual,_eon_exp2_64_coeff[1]);
    residual=d_add(residual,tail);
    a=_eon_operator_polynomial_fma(residual,a,_eon_exp2_64_coeff[2]);
    b=_eon_operator_polynomial_fma(residual,b,_eon_exp2_64_coeff[3]);
    ulong polynomial=d_mul(d_mul(d_mul(residual,_eon_exp2_64_coeff[4]),a),b);
    long scale=k>>7;
    if(scale>=-1022)
        return _eon_operator_polynomial_fma(polynomial,base,base)+(ulong(scale)<<52);
    // The platform routine compensates the addition before scaling into
    // subnormal space, avoiding the double rounding of a normal exp2 value.
    base+=ulong(scale+1022)<<52;
    ulong high=d_add(base,one);
    ulong low=d_sub(base,d_sub(high,one));
    low=_eon_operator_polynomial_fma(base,polynomial,low);
    return d_mul(d_sub(d_add(high,low),one),0x0010000000000000ul);
}
'''
