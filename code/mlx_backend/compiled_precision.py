"""Native temporary precision is separate from a model's reference identity.

Exhaustion requests a larger GPU specialization and replay of an immutable
frontier. It is never exposed as a simulated Python exception.
"""
from dataclasses import dataclass

ERROR_NAME = '_NativePrecisionRetry'
ERROR_MESSAGE = 'Native temporary integer precision exhausted'
DEFAULT_LIMBS = 8


class PrecisionRetry(RuntimeError):
    """Private request to replay the current native frontier with more limbs."""


@dataclass(frozen=True)
class PrecisionPolicy:
    limbs: int = DEFAULT_LIMBS

    def __post_init__(self):
        if self.limbs < 2:
            raise ValueError('Native integer precision needs at least two limbs')

    def grow(self):
        return type(self)(self.limbs * 2)


MSL_HEADER = r'''
#ifndef EON_COMPILED_PRECISION_INCLUDED
#define EON_COMPILED_PRECISION_INCLUDED
// The existing magnitude primitives remain authoritative. Only temporary
// capacity detection is added here; a failed attempt is never committed.
template<uint N> inline ci_py<N> ci_precision_add(thread const ci_py<N>& a,
        thread const ci_py<N>& b, thread bool& exhausted) {
    ci_py<N> result=ci_py_add(a,b);
    if(a.negative==b.negative && ci_py_ucmp(result,a)<0)exhausted=true;
    return result;
}
template<uint N> inline ci_py<N> ci_precision_sub(thread const ci_py<N>& a,
        thread const ci_py<N>& b, thread bool& exhausted) {
    ci_py<N> opposite=b;opposite.negative=!b.negative&&!ci_py_iszero(b);
    return ci_precision_add(a,opposite,exhausted);
}
template<uint N> inline ci_py<N> ci_precision_mul(thread const ci_py<N>& a,
        thread const ci_py<N>& b, thread bool& exhausted) {
    ci_py<2*N> product=ci_py_zero<2*N>();
    #pragma clang loop unroll(disable)
    for(uint i=0;i<N;i++) {
        ulong carry=0UL;
        #pragma clang loop unroll(disable)
        for(uint j=0;j<N;j++) {
            ulong value=ulong(a.limb[i])*ulong(b.limb[j])+ulong(product.limb[i+j])+carry;
            product.limb[i+j]=uint(value);carry=value>>32;
        }
        product.limb[i+N]=uint(carry);
    }
    uint high=0u;for(uint i=N;i<2*N;i++)high|=product.limb[i];
    exhausted|=high!=0u;
    ci_py<N> result=ci_py_zero<N>();
    for(uint i=0;i<N;i++)result.limb[i]=product.limb[i];
    result.negative=(a.negative!=b.negative)&&!ci_py_iszero(result);
    return result;
}
template<uint N> inline ci_py<N> ci_precision_pow(thread const ci_py<N>& base,
        thread const ci_py<N>& exponent, thread bool& exhausted) {
    ci_py<N> result=ci_py_from<N>(1UL,false),power=base;
    uint bits=ci_py_bits(exponent);
    #pragma clang loop unroll(disable)
    for(uint bit=0;bit<bits&&!exhausted;bit++) {
        if((exponent.limb[bit/32]>>(bit%32))&1u)result=ci_precision_mul(result,power,exhausted);
        if(bit+1<bits&&!exhausted)power=ci_precision_mul(power,power,exhausted);
    }
    return result;
}
#endif
'''
