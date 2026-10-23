"""Integer-backed binary64 trigonometry with the frozen Apple arm64 sequence.

The constants and operation order reproduce the installed reference libm
(UUID83b173cd7f3239018e67c0b2676be015, macOS26.4). No host math executes in a
native timestep. Include binary64.MSL_HEADER and math64.MSL_MATH_HEADER first.
The dynamic model compiler selects the NumPy or compiled cosine NaN policy.
"""

from .stable_math64 import _STABLE_FMA_HEADER

_SIN_HEADER = r'''
constant ulong _eon_trig_range[19] = {
    0x35fdafd88fc6ae84ul, 0x9e839cfbc5294975ul, 0xba93dd63f5f2f8bdul,
    0xa7a31fb34f2ff516ul, 0xb69b3f6793e584dbul, 0xf79788c5ad05368ful,
    0x8ffc4bffef02cc07ul, 0x4e422fc5defc941dul, 0x9cc8eb1cc1a99cfaul,
    0x74ce38135a2fbf20ul, 0x74411afa975da242ul, 0x7f0ef58e5894d39ful,
    0x0324977504e8c90eul, 0xdb92371d2126e970ul, 0xff28b1d5ef5de2b0ul,
    0x6db14acc9e21c820ul, 0xfe13abe8fa9a6ee0ul, 0x517cc1b727220a94ul,
    0x0000000000000000ul
};
constant ulong _eon_sin_coeff[6] = {
    0x3de5d8fd1fd19ccdul, 0xbe5ae5e5a9291f5dul, 0x3ec71de3567d48a1ul,
    0xbf2a01a019bfdf03ul, 0x3f8111111110f7d0ul, 0xbfc5555555555548ul
};
constant ulong _eon_cos_coeff[7] = {
    0xbda8fa49a0861a9bul, 0x3e21ee9d7b4e3f05ul, 0xbe927e4f7eac4bc6ul,
    0x3efa01a019c844f5ul, 0xbf56c16c16c14f91ul, 0x3fa555555555554bul,
    0xbfe0000000000000ul
};
inline uint _eon_trig_mulhi32(uint a,uint b){return mulhi(a,b);}
struct _eon_trig_words {uint w0,w1,w2,w3;};
inline _eon_trig_words _eon_trig_product53(ulong a,ulong b){
    uint a0=uint(a),a1=(uint(a>>32)&0xfffffu)|0x100000u;
    uint b0=uint(b),b1=(uint(b>>32)&0xfffffu)|0x100000u;
    uint p0=a0*b0,carry0=_eon_trig_mulhi32(a0,b0);
    uint cross1=a0*b1,cross2=a1*b0;
    uint p1=carry0+cross1,carry1=uint(p1<carry0);
    uint previous=p1;p1+=cross2;carry1+=uint(p1<previous);
    uint p2=a1*b1,carry2=0u;
    uint extra=_eon_trig_mulhi32(a0,b1);
    previous=p2;p2+=extra;carry2+=uint(p2<previous);
    extra=_eon_trig_mulhi32(a1,b0);
    previous=p2;p2+=extra;carry2+=uint(p2<previous);
    previous=p2;p2+=carry1;carry2+=uint(p2<previous);
    return {p0,p1,p2,_eon_trig_mulhi32(a1,b1)+carry2};
}
inline ulong _eon_trig_multiply(ulong a,ulong b){
    uint ah=uint(a>>32),bh=uint(b>>32);
    int ea=int((ah>>20)&2047u),eb=int((bh>>20)&2047u),exponent=ea+eb-1023;
    if(uint(ea-1)>=2046u||uint(eb-1)>=2046u||exponent<=0||exponent>=2046)return d_mul(a,b);
    _eon_trig_words p=_eon_trig_product53(a,b);
    bool upper=(p.w3&0x200u)!=0u;
    uint drop=upper?21u:20u;
    uint low=(p.w1>>drop)|(p.w2<<(32u-drop));
    uint high=(p.w2>>drop)|(p.w3<<(32u-drop));
    uint discarded=p.w1&((1u<<drop)-1u),halfway=1u<<(drop-1u);
    if(discarded>halfway||(discarded==halfway&&(p.w0!=0u||(low&1u)))){
        ++low;high+=uint(low==0u);
    }
    exponent+=int(upper);
    if((high&0x200000u)!=0u){low=(low>>1)|(high<<31);high>>=1;++exponent;}
    return (ulong(((ah^bh)&0x80000000u)|(uint(exponent)<<20)|(high&0xfffffu))<<32)|ulong(low);
}
inline ulong _eon_trig_add32(ulong a,ulong b){
    uint ah=uint(a>>32),al=uint(a),bh=uint(b>>32),bl=uint(b);
    int ea=int((ah>>20)&2047u),eb=int((bh>>20)&2047u);
    if(uint(ea-1)>=2046u||uint(eb-1)>=2046u)return d_add(a,b);
    if((ah&0x7fffffffu)<(bh&0x7fffffffu)||((ah&0x7fffffffu)==(bh&0x7fffffffu)&&al<bl)){
        uint t=ah;ah=bh;bh=t;t=al;al=bl;bl=t;int e=ea;ea=eb;eb=e;
    }
    uint a0=al<<3,a1=(((ah&0xfffffu)|0x100000u)<<3)|(al>>29);
    uint b0=bl<<3,b1=(((bh&0xfffffu)|0x100000u)<<3)|(bl>>29);
    int gap=ea-eb;
    if(gap>=64){b0=1u;b1=0u;}
    else if(gap>=32){
        int n=gap-32;uint sticky=b0|(b1&((1u<<n)-1u));
        b0=(b1>>n)|uint(sticky!=0u);b1=0u;
    }else if(gap){
        uint sticky=b0&((1u<<gap)-1u);
        b0=(b0>>gap)|(b1<<(32-gap))|uint(sticky!=0u);b1>>=gap;
    }
    uint lo,hi;
    if(((ah^bh)&0x80000000u)==0u){lo=a0+b0;hi=a1+b1+uint(lo<a0);}
    else{lo=a0-b0;hi=a1-b1-uint(a0<b0);}
    if((lo|hi)==0u)return 0ul;
    int shift=hi!=0u?int(clz(hi))-8:24+int(clz(lo));
    if(shift>=32){hi=lo<<(shift-32);lo=0u;ea-=shift;}
    else if(shift>0){hi=(hi<<shift)|(lo>>(32-shift));lo<<=shift;ea-=shift;}
    else if(shift<0){lo=(lo>>1)|(hi<<31)|(lo&1u);hi>>=1;++ea;}
    if(ea<=0)return d_add(a,b);
    uint tail=lo&7u;
    lo=(lo>>3)|(hi<<29);hi>>=3;
    if(tail>4u||(tail==4u&&(lo&1u))){++lo;hi+=uint(lo==0u);}
    if((hi&0x200000u)!=0u){lo=(lo>>1)|(hi<<31);hi>>=1;++ea;}
    if(ea>=2047)return (ulong(ah&0x80000000u)<<32)|D_INF;
    return (ulong((ah&0x80000000u)|(uint(ea)<<20)|(hi&0xfffffu))<<32)|ulong(lo);
}
inline ulong _eon_trig_sub32(ulong a,ulong b){
    if(d_is_nan(a)||d_is_nan(b))return d_sub(a,b);
    return _eon_trig_add32(a,b^0x8000000000000000ul);
}

inline _eon_trig_words _eon_trig_shift_coefficient(uint c2,uint c3,int n){
    if(n>=128)return {1u,0u,0u,0u};
    if(n>=96){
        int amount=n-96;
        uint sticky=uint((c2|(c3&((1u<<amount)-1u)))!=0u);
        return {(c3>>amount)|sticky,0u,0u,0u};
    }
    if(n>=64){
        int amount=n-64;
        uint low=(c2>>amount)|(amount==0?0u:c3<<(32-amount));
        low|=uint((c2&((1u<<amount)-1u))!=0u);
        return {low,c3>>amount,0u,0u};
    }
    if(n>=32){
        int amount=n-32;
        return {amount==0?0u:c2<<(32-amount),
                (c2>>amount)|(amount==0?0u:c3<<(32-amount)),c3>>amount,0u};
    }
    return {0u,c2<<(32-n),(c2>>n)|(c3<<(32-n)),c3>>n};
}
inline uint _eon_trig_add_word(uint a,uint b,thread uint* carry){
    uint first=a+b,result=first+*carry;
    *carry=uint(first<a||result<first);return result;
}
inline uint _eon_trig_sub_word(uint a,uint b,thread uint* borrow){
    uint first=a-b,result=first-*borrow;
    *borrow=uint(a<b||first<*borrow);return result;
}
// A normal product exponent at least two above the addend cannot cancel by more
// than one exponent bit. Preserve all 106 product bits; sticky alignment is needed
// only at large exponent separations, where discarded bits cannot be exposed.
inline ulong _eon_trig_product_fma32(ulong a,ulong b,ulong c){
    int ea=int(d_exp(a)),eb=int(d_exp(b)),ec=int(d_exp(c));
    if(d_is_zero(c)&&uint(ea-1)<2046u&&uint(eb-1)<2046u)return _eon_trig_multiply(a,b);
    if(uint(ea-1)>=2046u||uint(eb-1)>=2046u||uint(ec-1)>=2046u)return d_fma(a,b,c);
    _eon_trig_words p=_eon_trig_product53(a,b);
    int top=(p.w3&0x200u)!=0u?105:104;
    int exponent=ea+eb+top-1127,gap=exponent-ec;
    if(gap<2)return d_fma(a,b,c);
    int shift=126-top;
    p={(p.w0<<shift),(p.w1<<shift)|(p.w0>>(32-shift)),
       (p.w2<<shift)|(p.w1>>(32-shift)),(p.w3<<shift)|(p.w2>>(32-shift))};
    uint clow=uint(c),chigh=(uint(c>>32)&0xfffffu)|0x100000u;
    _eon_trig_words q=_eon_trig_shift_coefficient(clow<<10,(chigh<<10)|(clow>>22),gap);
    uint carry=0u;
    if(((a^b^c)>>63)==0ul){
        p.w0=_eon_trig_add_word(p.w0,q.w0,&carry);p.w1=_eon_trig_add_word(p.w1,q.w1,&carry);
        p.w2=_eon_trig_add_word(p.w2,q.w2,&carry);p.w3=_eon_trig_add_word(p.w3,q.w3,&carry);
    }else{
        p.w0=_eon_trig_sub_word(p.w0,q.w0,&carry);p.w1=_eon_trig_sub_word(p.w1,q.w1,&carry);
        p.w2=_eon_trig_sub_word(p.w2,q.w2,&carry);p.w3=_eon_trig_sub_word(p.w3,q.w3,&carry);
    }
    int highest=(p.w3&0x80000000u)!=0u?127:((p.w3&0x40000000u)!=0u?126:125);
    exponent+=highest-126;
    if(exponent<=0||exponent>=2047)return d_fma(a,b,c);
    int drop=highest-119;
    uint low=(p.w2>>drop)|(p.w3<<(32-drop)),high=p.w3>>drop;
    low|=uint((p.w0|p.w1|(p.w2&((1u<<drop)-1u)))!=0u);
    uint mantissa_low=(low>>3)|(high<<29),mantissa_high=high>>3,tail=low&7u;
    if(tail>4u||(tail==4u&&(mantissa_low&1u))){++mantissa_low;mantissa_high+=uint(mantissa_low==0u);}
    if(mantissa_high&0x200000u){
        mantissa_low=(mantissa_low>>1)|(mantissa_high<<31);mantissa_high>>=1;++exponent;
    }
    uint sign=uint((a^b)>>32)&0x80000000u;
    return (ulong(sign|(uint(exponent)<<20)|(mantissa_high&0xfffffu))<<32)|ulong(mantissa_low);
}
// The bounded Horner path keeps the 106-bit product in four 32-bit words.
// Truncation retains four rounding guards and a sticky remainder; no product
// bits capable of changing the final binary64 rounding are approximated.
inline ulong _eon_trig_horner32(ulong a,ulong b,ulong c){
    uint ah=uint(a>>32),bh=uint(b>>32),ch=uint(c>>32);
    int ea=int((ah>>20)&2047u),eb=int((bh>>20)&2047u),ec=int((ch>>20)&2047u);
    if(ea==0)return c;
    int shift=ea+eb-ec-1001;
    if(shift<=-34)return c;
    _eon_trig_words product=_eon_trig_product53(a,b);
    uint p0=product.w0,p1=product.w1,p2=product.w2,p3=product.w3;
    int cut=70-shift;
    uint q0,q1;bool sticky;
    if(cut<64){
        int amount=cut-32;
        q0=(p1>>amount)|(p2<<(32-amount));
        q1=(p2>>amount)|(p3<<(32-amount));
        sticky=(p0|(p1&((1u<<amount)-1u)))!=0u;
    }else if(cut<96){
        int amount=cut-64;
        q0=amount==0?p2:(p2>>amount)|(p3<<(32-amount));
        q1=p3>>amount;
        sticky=(p0|p1|(p2&((1u<<amount)-1u)))!=0u;
    }else{
        int amount=cut-96;
        q0=p3>>amount;q1=0u;
        sticky=(p0|p1|p2|(p3&((1u<<amount)-1u)))!=0u;
    }
    uint c0=uint(c)<<4,c1=(((ch&0xfffffu)|0x100000u)<<4)|(uint(c)>>28);
    uint low=(c0-q0-uint(sticky))|uint(sticky);
    uint high=c1-q1-uint(c0<q0||(c0==q0&&sticky));
    bool upper=(high&0x1000000u)!=0u;
    int exponent=ec-int(!upper);
    if(upper){low=(low>>1)|(high<<31)|(low&1u);high>>=1;}
    uint mantissa_low=(low>>3)|(high<<29),mantissa_high=high>>3;
    uint tail=low&7u;
    if(tail>4u||(tail==4u&&(mantissa_low&1u))){++mantissa_low;mantissa_high+=uint(mantissa_low==0u);}
    if(mantissa_high&0x200000u){
        mantissa_low=(mantissa_low>>1)|(mantissa_high<<31);mantissa_high>>=1;++exponent;
    }
    return (ulong((ch&0x80000000u)|(uint(exponent)<<20)|(mantissa_high&0xfffffu))<<32)|ulong(mantissa_low);
}
// Horner stages have a normal coefficient dominating an oppositely signed
// product. Retain the exact high product and a sticky remainder at four guard
// bits below the coefficient. Subtraction uses ceil(product), so its truncated
// integer and sticky bit represent the exact positive difference. A one-bit
// renormalization leaves the usual three guards for the single IEEE rounding.
// Guard every specialization assumption and use the general FMA otherwise.
template<bool Bounded> inline ulong _eon_trig_horner_impl(ulong a,ulong b,ulong c) {
    if(Bounded)return _eon_trig_horner32(a,b,c);
    int ea=int(d_exp(a)),eb=int(d_exp(b)),ec=int(d_exp(c));
    if(Bounded&&ea==0)return c;
    if(!Bounded&&(uint(ea-1)>=2046u||uint(eb-1)>=2046u||uint(ec-1)>=2046u||((a^b^c)>>63)==0ul))
        return d_fma(a,b,c);
    int shift=ea+eb-ec-1001;
    if(!Bounded&&shift>21) return d_fma(a,b,c);
    if(shift<=-34)return c;
    ulong ma=d_mant(a)|(1ul<<52),mb=d_mant(b)|(1ul<<52),mc=d_mant(c)|(1ul<<52);
    ulong hi,lo=d_umul128(ma,mb,&hi);
    int cut=70-shift;
    ulong product;bool sticky;
    if(cut>=64){
        product=hi>>(cut-64);
        sticky=(lo|(hi&((1ul<<(cut-64))-1ul)))!=0ul;
    }else{
        product=(hi<<(64-cut))|(lo>>cut);
        sticky=(lo&((1ul<<cut)-1ul))!=0ul;
    }
    ulong coefficient=mc<<4;
    if(!Bounded&&product>=coefficient)return d_fma(a,b,c);
    ulong rounded=(coefficient-product-ulong(sticky))|ulong(sticky);
    if(!Bounded&&(rounded&0x0180000000000000ul)==0ul)return d_fma(a,b,c);
    bool upper=(rounded&0x0100000000000000ul)!=0ul;
    int exponent=ec-(upper?0:1);
    if(!Bounded&&exponent<=0)return d_fma(a,b,c);
    if(upper)rounded=(rounded>>1)|(rounded&1ul);
    ulong mantissa=rounded>>3,tail=rounded&7ul;
    if(tail>4ul||(tail==4ul&&(mantissa&1ul)))++mantissa;
    if(mantissa>>53){mantissa>>=1;++exponent;}
    return (c&0x8000000000000000ul)|(ulong(exponent)<<52)|(mantissa&0xffffffffffffful);
}
inline ulong _eon_trig_horner(ulong a,ulong b,ulong c){return _eon_trig_horner_impl<false>(a,b,c);}
inline ulong _eon_trig_uint_scaled(ulong value, uint fractional) {
    if(value == 0ul) return 0ul;
    int top = 63-int(clz(value));
    ulong mantissa;
    if(top <= 55) mantissa = value << (55-top);
    else {
        int shift = top-55;
        mantissa = (value >> shift) | ulong((value & ((1ul << shift)-1ul)) != 0ul);
    }
    return d_finish(0ul, top+1023, mantissa) - (ulong(fractional) << 52);
}
// Medium reduction rounds the reciprocal product to binary64, then to an
// integer. Retain both roundings: rounding the exact product once can choose a
// different quadrant at an integer-plus-half boundary.
inline uint _eon_trig_nearest_count_exact(ulong absolute) {
    ulong hi,lo=d_umul128(d_mant(absolute)|(1ul<<52),0x145f306dc9c883ul,&hi);
    int top=(hi>>41)!=0ul?105:104;
    int shift=top-52,exponent=int(d_exp(absolute))+top-1128;
    ulong significand=(hi<<(64-shift))|(lo>>shift);
    ulong discarded=lo&((1ul<<shift)-1ul),halfway=1ul<<(shift-1);
    if(discarded>halfway||(discarded==halfway&&(significand&1ul)))++significand;
    if(significand>>53){significand>>=1;++exponent;}
    int drop=52-exponent;
    ulong whole=significand>>drop;
    discarded=significand&((1ul<<drop)-1ul);halfway=1ul<<(drop-1);
    return uint(whole+ulong(discarded>halfway||(discarded==halfway&&(whole&1ul))));
}
inline uint _eon_trig_nearest_count(ulong absolute){
    // A float estimate only selects a quadrant when its complete error interval
    // excludes both nearest-integer cell boundaries. Otherwise use both exact
    // binary64/integer roundings above. For E=floor(log2(x)), float conversion,
    // reciprocal representation, float multiply and the reference binary64
    // multiply together contribute strictly less than 2^(E-22). The interval
    // subtraction and distance are exact binary32 operations in this range.
    uint high=uint(absolute>>32),low=uint(absolute);
    uint converted=((high>>20)-896u)<<23;
    converted|=((high&0xfffffu)<<3)|(low>>29);
    uint discarded=low&0x1fffffffu;
    converted+=uint(discarded>0x10000000u||(discarded==0x10000000u&&(converted&1u)));
    float estimate=as_type<float>(converted)*as_type<float>(0x3f22f983u);
    float nearest=rint(estimate);
    float error=as_type<float>(uint(int(high>>20)-918)<<23);
    if(fabs(estimate-nearest)<0.5f-error)return uint(nearest);
    return _eon_trig_nearest_count_exact(absolute);
}
inline ulong _eon_trig_count_bits(uint count,ulong sign) {
    if(count==0u)return sign;
    int top=63-int(clz(ulong(count)));
    return sign|(ulong(top+1023)<<52)|((ulong(count)<<(52-top))&0xffffffffffffful);
}
// count is at most 333774; both split constants have only 35 significant
// bits. Their exact integer products fit in 54 bits, with at most one bit to
// round before the exact power-of-two scale.
inline ulong _eon_trig_split_product(ulong value,uint fractional,ulong sign) {
    if(value==0ul)return sign;
    int top=63-int(clz(value));
    ulong significand=top<=52?value<<(52-top):(value>>1)+ulong((value&3ul)==3ul);
    if(significand>>53){significand>>=1;++top;}
    return sign|(ulong(top+1023-int(fractional))<<52)|(significand&0xffffffffffffful);
}
inline ulong _eon_trig_sin_poly(ulong high, ulong low, ulong sign) {
    ulong total=_eon_trig_add32(high,low), square=_eon_trig_multiply(total,total);
    ulong p=d_fma(square,_eon_sin_coeff[0],_eon_sin_coeff[1]);
    for(int i=2;i<6;++i) p=d_fma(square,p,_eon_sin_coeff[i]);
    return _eon_trig_add32(d_fma(total,_eon_trig_multiply(square,p),low),high)^sign;
}
inline ulong _eon_trig_cos_poly(ulong square, ulong sign) {
    ulong p=d_fma(square,_eon_cos_coeff[0],_eon_cos_coeff[1]);
    for(int i=2;i<7;++i) p=d_fma(square,p,_eon_cos_coeff[i]);
    return d_fma(square,p,0x3ff0000000000000ul)+sign;
}
template<bool Bounded> inline ulong _eon_trig_shared_poly_impl(ulong high,ulong low,ulong total,ulong square,
                                   ulong sign,bool cosine) {
    ulong p=_eon_trig_horner_impl<Bounded>(square,cosine?_eon_cos_coeff[0]:_eon_sin_coeff[0],
                         cosine?_eon_cos_coeff[1]:_eon_sin_coeff[1]);
    #pragma unroll
    for(int i=2;i<6;++i) p=_eon_trig_horner_impl<Bounded>(square,p,cosine?_eon_cos_coeff[i]:_eon_sin_coeff[i]);
    if(cosine) p=_eon_trig_horner_impl<Bounded>(square,p,_eon_cos_coeff[6]);
    ulong a=cosine?square:total;
    ulong b=cosine?p:_eon_trig_multiply(square,p);
    ulong c=cosine?0x3ff0000000000000ul:low;
    ulong result;
    // The final cosine term has coefficient 1 and product magnitude <= 0.375.
    // In the unreduced sine branch c==a, and the product's relative magnitude
    // is <=0.125. These retain the same proved coefficient-dominant rounding.
    if(Bounded&&(cosine||(high==0ul&&low==total&&!d_is_zero(total))))
        result=_eon_trig_horner32(a,b,c);
    else result=_eon_trig_product_fma32(a,b,c);
    return cosine?result+sign:_eon_trig_add32(result,high)^sign;
}
inline ulong _eon_trig_shared_poly(ulong high,ulong low,ulong total,ulong square,ulong sign,bool cosine){
    // Exact rational interval bounds for the frozen coefficients prove each
    // Horner product opposes its coefficient and is below half its magnitude
    // for 0 <= square <= 0.75. Each intermediate stays normal, shift <= 21,
    // and at most one normalization bit is needed. A subnormal square cannot
    // change any coefficient's rounding. Retain a guarded path outside this
    // independently checked interval rather than assuming a reduction bound.
    if(square<=0x3fe8000000000000ul)
        return _eon_trig_shared_poly_impl<true>(high,low,total,square,sign,cosine);
    return _eon_trig_shared_poly_impl<false>(high,low,total,square,sign,cosine);
}
// For medium inputs with 1 <= count < 32, H=count*0x6487ed511*2^-34
// and M=count*0x5a308d313*2^-73 are exact binary64 products: their integer
// numerators have at most 40 bits. The first subtraction is exact by the
// existing medium-reduction Sterbenz bound. Where |x-H| >= |M|, FastTwoSum
// therefore makes ((x-H)-RN((x-H)-M))-M the exact error of the rounded high.
// Form that same unrounded difference in three 32-bit words at scale 2^-73,
// then round once and encode its exact residual. The guard retains the original
// operations whenever the dominance condition or fixed-point bounds do not hold.
struct _eon_trig_small_pair {ulong high,error;bool valid;};
inline _eon_trig_small_pair _eon_trig_small_fixed(ulong absolute,uint count){
    uint ah=uint(absolute>>32),al=uint(absolute),exponent=ah>>20;
    if(count==0u||count>=32u||exponent<1022u||exponent>1028u)return {0ul,0ul,false};
    uint shift=exponent-1002u,mantissa=(ah&0xfffffu)|0x100000u;
    uint x0=al<<shift,x1=(mantissa<<shift)|(al>>(32u-shift)),x2=mantissa>>(32u-shift);
    uint hp0=count*0x487ed511u,hp1=mulhi(count,0x487ed511u)+6u*count;
    uint h1=hp0<<7,h2=(hp1<<7)|(hp0>>25);
    uint r0=x0,r1=x1-h1,r2=x2-h2-uint(x1<h1);
    uint m0=count*0xa308d313u,m1=mulhi(count,0xa308d313u)+5u*count;
    if(r2==0u&&(r1<m1||(r1==m1&&r0<m0)))return {0ul,0ul,false};
    if(r2==0xffffffffu){
        uint a0=0u-r0,a1=~r1+uint(r0==0u);
        if(a1<m1||(a1==m1&&a0<m0))return {0ul,0ul,false};
    }
    uint w0=r0-m0,borrow=uint(r0<m0);
    uint first=r1-m1,w1=first-borrow,w2=r2-uint(r1<m1)-uint(first<borrow);
    uint sign=w2&0x80000000u;
    if(sign){
        w0=0u-w0;
        uint carry=uint(w0==0u),before=~w1;w1=before+carry;
        w2=~w2+uint(carry&&w1==0u);
    }
    if((w0|w1|w2)==0u)return {0ul,0ul,true};
    int top=w2!=0u?95-int(clz(w2)):(w1!=0u?63-int(clz(w1)):31-int(clz(w0)));
    // The overwhelmingly common range needs only a <=20-bit right shift.
    // Very small residues use the original path, preserving every input bit.
    if(top<52||top>73)return {0ul,0ul,false};
    uint drop=uint(top-52),low=drop==0u?w0:(w0>>drop)|(w1<<(32u-drop));
    uint high=drop==0u?w1:(w1>>drop)|(w2<<(32u-drop));
    uint remainder=w0&((1u<<drop)-1u),halfway=drop==0u?0u:1u<<(drop-1u);
    bool up=drop!=0u&&(remainder>halfway||(remainder==halfway&&(low&1u)));
    low+=uint(up);high+=uint(up&&low==0u);
    int out_exp=top+950;
    if(high&0x200000u){low=(low>>1)|(high<<31);high>>=1;++out_exp;}
    ulong rounded=(ulong(sign|(uint(out_exp)<<20)|(high&0xfffffu))<<32)|ulong(low);
    if(remainder==0u)return {rounded,0ul,true};
    uint residue=up?(1u<<drop)-remainder:remainder;
    int residual_top=31-int(clz(residue));
    ulong error=(ulong(sign^(uint(up)<<31))<<32)|(ulong(residual_top+950)<<52)
                |((ulong(residue)<<(52-residual_top))&0xffffffffffffful);
    return {rounded,error,true};
}

constant ulong _eon_trig_small_tails[32]={0x0ul,0x3b298a2e03707345ul,0x3b398a2e03707345ul,0x3b4327a282945674ul,0x3b498a2e03707345ul,0x3b4fecb9844c9016ul,0x3b5327a282945674ul,0x3b5658e8430264dcul,0x3b598a2e03707345ul,0x3b5cbb73c3de81aeul,0x3b5fecb9844c9016ul,0x3b618effa25d4f3ful,0x3b6327a282945674ul,0x3b64c04562cb5da8ul,0x3b6658e8430264dcul,0x3b67f18b23396c11ul,0x3b698a2e03707345ul,0x3b6b22d0e3a77a79ul,0x3b6cbb73c3de81aeul,0x3b6e5416a41588e2ul,0x3b6fecb9844c9016ul,0x3b70c2ae3241cba5ul,0x3b718effa25d4f3ful,0x3b725b511278d2daul,0x3b7327a282945674ul,0x3b73f3f3f2afda0eul,0x3b74c04562cb5da8ul,0x3b758c96d2e6e142ul,0x3b7658e8430264dcul,0x3b772539b31de877ul,0x3b77f18b23396c11ul,0x3b78bddc9354efabul};

// For two normal binary64 numbers separated by at least 54 exponents,
// |b| is strictly below half an ulp of a. The sole tighter spacing occurs
// immediately below an exact power of two with opposite signs. At gap54,
// b is at least half that lower spacing; a nonzero mantissa crosses the tie
// and selects a's immediate predecessor. The exact tie selects even a.
inline ulong _eon_trig_dominant_sum(ulong a,ulong b){
    int ea=int(d_exp(a)),eb=int(d_exp(b)),gap=ea-eb;
    if(uint(ea-1)<2046u&&uint(eb-1)<2046u&&gap>=54){
        if(((a^b)>>63)!=0ul&&d_mant(a)==0ul&&gap==54&&d_mant(b)!=0ul)return a-1ul;
        return a;
    }
    return _eon_trig_add32(a,b);
}
// Doubling a normal value below the last finite binade changes only its
// exponent and is exact. All subnormal, zero, overflow and nonfinite cases
// keep the same addition operation as the original sequence.
inline ulong _eon_trig_double(ulong a){
    if(uint(d_exp(a)-1ul)<2045u)return a+0x0010000000000000ul;
    return _eon_trig_add32(a,a);
}

struct _eon_trig_large {
    ulong significand;
    ulong sign;
    ulong cosine_sign;
    ulong tangent_sign;
    uint leading;
    bool cosine;
};
inline _eon_trig_large _eon_trig_reduce_large(ulong x) {
    int exponent=int(d_exp(x))-1023;
    ulong mantissa=(x << 11)|0x8000000000000000ul;
    uint shift=uint(exponent)&63u;
    int base=15-(exponent >> 6);
    ulong a,b,c;
    if(shift == 0u) {
        a=_eon_trig_range[base+1]; b=_eon_trig_range[base+2]; c=_eon_trig_range[base+3];
    } else {
        ulong w0=_eon_trig_range[base],w1=_eon_trig_range[base+1];
        ulong w2=_eon_trig_range[base+2],w3=_eon_trig_range[base+3];
        a=(w1 << shift)|(w0 >> (64u-shift));
        b=(w2 << shift)|(w1 >> (64u-shift));
        c=(w3 << shift)|(w2 >> (64u-shift));
    }
    ulong lower=d_mulhi(a,mantissa), product=b*mantissa;
    ulong low=lower+product;
    ulong high=d_mulhi(b,mantissa)+c*mantissa+ulong(low<lower);
    ulong sign=(x^high)&0x8000000000000000ul;
    ulong cosine_sign=(high^(high<<1))&0x8000000000000000ul;
    ulong tangent_sign=(x^(high<<1))&0x8000000000000000ul;
    bool cosine=((((high << 2)^(high << 1)) >> 63) != 0ul);
    ulong mask=0ul-((high >> 61)&1ul);
    high=(high^mask)&0x1ffffffffffffffful;
    low^=mask;
    uint leading=uint(clz(high));
    ulong significand=leading==64u ? low : (high << leading)|(low >> (64u-leading));
    return {significand,sign,cosine_sign,tangent_sign,leading,cosine};
}
struct _eon_trig_reduced {ulong high,low,total,square,sign;bool cosine;};
template<bool IntegerReduction,bool Cosine=false> inline _eon_trig_reduced _eon_trig_prepare(ulong x) {
    const ulong sign_bit=0x8000000000000000ul;
    ulong absolute=x&0x7ffffffffffffffful;
    ulong high,low,total,square,sign=0ul;
    bool cosine=Cosine;
    if(absolute <= 0x3fe921fb54442d18ul) {
        high=0ul;low=x;total=x;square=_eon_trig_multiply(x,x);
    } else if(absolute < 0x412000013be57a40ul) {
        ulong nearest,high_product,mid_product;long n;
        if(IntegerReduction){
            uint count=_eon_trig_nearest_count(absolute);
            ulong input_sign=x&sign_bit;
            _eon_trig_small_pair pair=_eon_trig_small_fixed(absolute,count);
            if(pair.valid){
                high=pair.high^input_sign;
                ulong error=pair.error==0ul?0ul:pair.error^input_sign;
                low=_eon_trig_sub32(error,_eon_trig_small_tails[count]|input_sign);
                long small_n=input_sign!=0ul?-long(count):long(count);
                sign=Cosine?((ulong(small_n)^(ulong(small_n)>>1))<<63):((ulong(small_n)<<62)&sign_bit);
                cosine=Cosine?((small_n&1l)==0l):((small_n&1l)!=0l);
                total=_eon_trig_dominant_sum(high,low);
                if(cosine){ulong cross=_eon_trig_multiply(high,low);square=_eon_trig_product_fma32(high,high,_eon_trig_double(cross));}
                else square=_eon_trig_multiply(total,total);
                return {high,low,total,square,sign,cosine};
            }
            nearest=_eon_trig_count_bits(count,input_sign);
            n=input_sign!=0ul?-long(count):long(count);
            high_product=_eon_trig_split_product(ulong(count)*0x6487ed511ul,34u,input_sign);
            mid_product=_eon_trig_split_product(ulong(count)*0x5a308d313ul,73u,input_sign);
        }else{
            nearest=d_roundeven(_eon_trig_multiply(x,0x3fe45f306dc9c883ul));
            n=d_to_int(nearest);
            high_product=_eon_trig_multiply(nearest,0x3ff921fb54440000ul);
            mid_product=_eon_trig_multiply(nearest,0x3d868c234c4c0000ul);
        }
        ulong tail_product=_eon_trig_multiply(nearest,0x3b298a2e03707345ul);
        ulong remaining=_eon_trig_sub32(x,high_product);
        high=_eon_trig_sub32(remaining,mid_product);
        low=_eon_trig_sub32(_eon_trig_sub32(_eon_trig_sub32(remaining,high),mid_product),tail_product);
        sign=Cosine?((ulong(n)^(ulong(n)>>1))<<63):((ulong(n)<<62)&sign_bit);
        cosine=Cosine?((n&1l)==0l):((n&1l)!=0l);
        total=_eon_trig_add32(high,low);
        if(cosine){ulong cross=_eon_trig_multiply(high,low);square=_eon_trig_product_fma32(high,high,_eon_trig_add32(cross,cross));}
        else square=_eon_trig_multiply(total,total);
    } else {
        _eon_trig_large r=_eon_trig_reduce_large(x);
        ulong product=d_mulhi(0xc90fdaa22168c235ul,r.significand);
        cosine=Cosine?!r.cosine:r.cosine;sign=Cosine?r.cosine_sign:r.sign;
        if(cosine) {
            ulong square_product=d_mulhi(product,product);
            square=_eon_trig_uint_scaled(square_product,58u)-(ulong(2u*r.leading) << 52);
            high=0ul;low=0ul;total=0ul;
        } else {
            high=_eon_trig_uint_scaled(product&0xffff000000000000ul,61u)-(ulong(r.leading) << 52);
            low=_eon_trig_uint_scaled(product&0x0000fffffffffffful,61u)-(ulong(r.leading) << 52);
            total=_eon_trig_uint_scaled(product,61u)-(ulong(r.leading) << 52);
            square=_eon_trig_multiply(total,total);
        }
    }
    return {high,low,total,square,sign,cosine};
}
template<bool IntegerReduction> inline ulong _eon_trig_sine_impl(ulong x) {
    if(d_is_zero(x))return x;
    if(d_exp(x)==0x7fful)return _eon_trig_sub32(x,x);
    _eon_trig_reduced r=_eon_trig_prepare<IntegerReduction>(x);
    return _eon_trig_shared_poly(r.high,r.low,r.total,r.square,r.sign,r.cosine);
}
inline ulong d_sine(ulong x){return _eon_trig_sine_impl<true>(x);}
inline ulong d_sine_integer_reduction(ulong x){return _eon_trig_sine_impl<true>(x);}
inline ulong d_cosine(ulong x){
    x&=0x7ffffffffffffffful;
    if(d_exp(x)==0x7fful)return _eon_trig_sub32(x,x);
    _eon_trig_reduced r=_eon_trig_prepare<true,true>(x);
    return _eon_trig_shared_poly(r.high,r.low,r.total,r.square,r.sign,r.cosine);
}
// The paired ABI enters the split sine polynomial with high=x, low=+0
// for |x| <= pi/4. Standalone sin instead uses high=+0, low=x. The
// former rounds x*square*p before adding x; the latter fuses that addition.
// Preserve the installed ABI's operation order, including both signed zeros.
struct _eon_trig64_pair {ulong sine,cosine;};
inline _eon_trig64_pair _eon_trig64_sincos_scalar(ulong x){
    if(d_is_zero(x))return {x,0x3ff0000000000000ul};
    ulong sine;
    if((x&0x7ffffffffffffffful)<=0x3fe921fb54442d18ul){
        ulong square=_eon_trig_multiply(x,x);
        sine=_eon_trig_shared_poly(x,0ul,x,square,0ul,false);
    }else sine=d_sine(x);
    return {sine,d_cosine(x)};
}
inline ulong d_cosine_numpy(ulong x){
    if(d_is_nan(x))return _eon_trig_sub32(x,x);
    return d_cosine(x);
}
inline ulong d_tangent(ulong x){
    if(d_is_zero(x))return x;
    if(d_exp(x)==0x7fful)return _eon_trig_sub32(x,x);
    ulong absolute=x&0x7ffffffffffffffful,total=x,sign=0ul;
    bool reciprocal=false;
    if(absolute>=0x412000013be57a40ul){
        _eon_trig_large r=_eon_trig_reduce_large(x);
        ulong product=d_mulhi(0xc90fdaa22168c235ul,r.significand);
        total=_eon_trig_uint_scaled(product,61u)-(ulong(r.leading)<<52);
        sign=r.tangent_sign;reciprocal=r.cosine;
    }else if(absolute>=0x3fe921fb54442d18ul){
        uint count=_eon_trig_nearest_count(absolute);
        ulong input_sign=x&0x8000000000000000ul;
        ulong nearest=_eon_trig_count_bits(count,input_sign);
        ulong high_product=_eon_trig_split_product(ulong(count)*0x6487ed511ul,34u,input_sign);
        ulong mid_product=_eon_trig_split_product(ulong(count)*0x5a308d313ul,73u,input_sign);
        ulong tail_product=_eon_trig_multiply(nearest,0x3b298a2e03707345ul);
        ulong remaining=_eon_trig_sub32(x,high_product),high=_eon_trig_sub32(remaining,mid_product);
        ulong low=_eon_trig_sub32(_eon_trig_sub32(_eon_trig_sub32(remaining,high),mid_product),tail_product);
        total=_eon_trig_add32(high,low);reciprocal=(count&1u)!=0u;sign=ulong(count)<<63;
    }
    ulong square=_eon_trig_multiply(total,total);
    // Tangent uses the same frozen coefficients, but its final sine has no
    // split low correction and its cosine squares the rounded total. Taking
    // the quotient of separately rounded d_sine/d_cosine changes the result.
    ulong sine=_eon_trig_shared_poly(0ul,total,total,square,sign,false);
    ulong cosine=_eon_trig_shared_poly(0ul,0ul,0ul,square,0ul,true);
    return reciprocal?d_div(cosine,sine):d_div(sine,cosine);
}
'''


def metal_trigonometry_header(functions):
    names=set(functions)
    if names-{"sin","cos","tan"}:
        raise NotImplementedError("Trigonometric candidate currently implements sine, cosine and tangent")
    return _STABLE_FMA_HEADER+_SIN_HEADER.replace("d_fma(","stable_fma(") if names else ""
