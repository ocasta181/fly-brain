"""Typed Cython Python-integer expressions with statically bounded native limbs.

The frontend supplies real boxing nodes. C arithmetic before boxing remains in
its original Cython/Clang block; only unbounded Python-integer intermediates use
private exact limbs. No Python numeric code executes during native timesteps.
"""
from __future__ import annotations
from hashlib import sha256
import json
from .compiled_cache import cache_lock, atomic_write
from pathlib import Path
import re
import numpy as np
from .compiler import UnsupportedModelError

MSL_HEADER=r'''
#ifndef EON_COMPILED_PYTHON_INTEGER_INCLUDED
#define EON_COMPILED_PYTHON_INTEGER_INCLUDED
template<uint N> struct ci_py { uint limb[N]; bool negative; };
template<uint N> inline ci_py<N> ci_py_zero() { ci_py<N> r; r.negative=false; for(uint i=0;i<N;i++)r.limb[i]=0u;return r; }
template<uint N> inline bool ci_py_iszero(thread const ci_py<N>& a) {uint bits=0u;for(uint i=0;i<N;i++)bits|=a.limb[i];return bits==0u;}
template<uint N> inline ci_py<N> ci_py_from(ulong v,bool negative) {ci_py<N> r=ci_py_zero<N>();r.limb[0]=uint(v);if(N>1)r.limb[1]=uint(v>>32);r.negative=negative&&v!=0UL;return r;}
template<uint N> inline ci_py<N> ci_py_abs(thread const ci_py<N>& a) {ci_py<N> result=a;result.negative=false;return result;}
template<uint N> inline ci_py<N> ci_py_neg(thread const ci_py<N>& a) {ci_py<N> result=a;result.negative=!a.negative&&!ci_py_iszero(a);return result;}
template<uint N> inline int ci_py_ucmp(thread const ci_py<N>& a,thread const ci_py<N>& b) {for(uint i=N;i>0;i--){if(a.limb[i-1]!=b.limb[i-1])return a.limb[i-1]<b.limb[i-1]?-1:1;}return 0;}
template<uint N> inline int ci_py_cmp(thread const ci_py<N>& a,thread const ci_py<N>& b) {if(a.negative!=b.negative)return a.negative?-1:1;int v=ci_py_ucmp(a,b);return a.negative?-v:v;}
template<uint N> inline ci_py<N> ci_py_uadd(thread const ci_py<N>& a,thread const ci_py<N>& b) {ci_py<N> r=ci_py_zero<N>();ulong carry=0UL;for(uint i=0;i<N;i++){ulong v=ulong(a.limb[i])+ulong(b.limb[i])+carry;r.limb[i]=uint(v);carry=v>>32;}return r;}
template<uint N> inline ci_py<N> ci_py_usub(thread const ci_py<N>& a,thread const ci_py<N>& b) {ci_py<N> r=ci_py_zero<N>();ulong borrow=0UL;for(uint i=0;i<N;i++){ulong v=ulong(b.limb[i])+borrow;r.limb[i]=uint(ulong(a.limb[i])-v);borrow=ulong(a.limb[i])<v;}return r;}
template<uint N> inline ci_py<N> ci_py_add(thread const ci_py<N>& a,thread const ci_py<N>& b) {ci_py<N> r; if(a.negative==b.negative){r=ci_py_uadd(a,b);r.negative=a.negative;}else if(ci_py_ucmp(a,b)>=0){r=ci_py_usub(a,b);r.negative=a.negative;}else{r=ci_py_usub(b,a);r.negative=b.negative;}if(ci_py_iszero(r))r.negative=false;return r;}
template<uint N> inline ci_py<N> ci_py_sub(thread const ci_py<N>& a,thread const ci_py<N>& b) {return ci_py_add(a,ci_py_neg(b));}
template<uint N> inline ci_py<N> ci_py_mul(thread const ci_py<N>& a,thread const ci_py<N>& b) {
    ci_py<N> r=ci_py_zero<N>();uint aa=N,bb=N;
    while(aa&&!a.limb[aa-1])aa--;while(bb&&!b.limb[bb-1])bb--;
    #pragma clang loop unroll(disable)
    for(uint i=0;i<aa;i++){
        ulong carry=0UL;
        #pragma clang loop unroll(disable)
        for(uint j=0;j<min(bb,N-i);j++){
            ulong v=ulong(a.limb[i])*ulong(b.limb[j])+ulong(r.limb[i+j])+carry;
            r.limb[i+j]=uint(v);carry=v>>32;
        }
        if(i+bb<N)r.limb[i+bb]=uint(carry);
    }
    r.negative=(a.negative!=b.negative)&&!ci_py_iszero(r);return r;
}
template<uint N> inline ci_py<N> ci_py_add_bounded(thread const ci_py<N>& a,thread const ci_py<N>& b) {
    ci_py<N> r=ci_py_add(a,b);
    if(a.negative==b.negative && ci_py_ucmp(r,a)<0)for(uint j=0;j<N;j++)r.limb[j]=~0u;
    return r;
}
template<uint N> inline ci_py<N> ci_py_mul_bounded(thread const ci_py<N>& a,thread const ci_py<N>& b) {
    ci_py<2*N> aa=ci_py_zero<2*N>(),bb=ci_py_zero<2*N>();
    for(uint i=0;i<N;i++){aa.limb[i]=a.limb[i];bb.limb[i]=b.limb[i];}
    aa.negative=a.negative;bb.negative=b.negative;ci_py<2*N> product=ci_py_mul(aa,bb);
    uint high=0u;for(uint i=N;i<2*N;i++)high|=product.limb[i];
    ci_py<N> result=ci_py_zero<N>();for(uint i=0;i<N;i++)result.limb[i]=high?~0u:product.limb[i];
    result.negative=product.negative;return result;
}
template<uint N> inline ci_py<N> ci_py_pow_bounded(thread const ci_py<N>& input,thread const ci_py<N>& exponent) {
    ci_py<N> unit=ci_py_from<N>(1UL,false);
    if(ci_py_iszero(exponent))return unit;
    if(ci_py_iszero(input))return ci_py_zero<N>();
    if(ci_py_ucmp(input,unit)==0){unit.negative=input.negative&&(exponent.limb[0]&1u);return unit;}
    ci_py<N> a=input;
    ci_py<N> result=ci_py_from<N>(1UL,false);uint bits=0u;
    for(uint i=0;i<N;i++){uint word=exponent.limb[i];if(word)bits=i*32u+32u-clz(word);}
    for(uint bit=0;bit<bits;bit++){
        if((exponent.limb[bit/32]>>(bit%32))&1u)result=ci_py_mul_bounded(result,a);
        uint missing=0u;for(uint j=0;j<N;j++)missing|=~result.limb[j];
        if(!missing){result.negative=input.negative&&(exponent.limb[0]&1u);return result;}
        if(bit+1<bits){
            a=ci_py_mul_bounded(a,a);missing=0u;for(uint j=0;j<N;j++)missing|=~a.limb[j];
            if(!missing){a.negative=input.negative&&(exponent.limb[0]&1u);return a;}
        }
    }
    return result;
}
// Keep the wide private workspace behind a call boundary: the composed
// Metal shader otherwise sporadically loses an earlier live scalar result.
template<uint N> __attribute__((noinline)) void ci_py_divmod_wide(thread const ci_py<N>& a,thread const ci_py<N>& b,thread ci_py<N>& q,thread ci_py<N>& r) {
        // Wide private arrays must stay loops: full scalarization makes the
        // Metal compiler expand a quadratic-size selection network.
        uint bits=0u,denominator_words=0u,denominator_ones=0u,shift=0u;
        for(uint i=0;i<N;i++){
            if(a.limb[i])bits=i*32u+32u-clz(a.limb[i]);
            if(b.limb[i]){denominator_words=i+1u;denominator_ones+=popcount(b.limb[i]);shift=i*32u+31u-clz(b.limb[i]);}
        }
        if(denominator_words==1u){
            // Base-2**32 long division. The carried remainder is less than
            // the divisor, so each quotient word fits exactly in uint.
            ulong remainder=0UL;
            #pragma clang loop unroll(disable)
            for(uint i=(bits+31u)/32u;i>0;i--){
                ulong value=(remainder<<32)|ulong(a.limb[i-1u]);
                q.limb[i-1u]=uint(value/ulong(b.limb[0]));remainder=value%ulong(b.limb[0]);
            }
            r.limb[0]=uint(remainder);
        }else if(denominator_ones==1u){
            uint words=shift/32u,offset=shift%32u;
            #pragma clang loop unroll(disable)
            for(uint i=0;i<N;i++){
                if(i+words<N)q.limb[i]=a.limb[i+words]>>offset;
                if(offset&&i+words+1u<N)q.limb[i]|=a.limb[i+words+1u]<<(32u-offset);
                if(i<words)r.limb[i]=a.limb[i];
                else if(i==words)r.limb[i]=a.limb[i]&((1u<<offset)-1u);
            }
        }else{
          #pragma clang loop unroll(disable)
          for(uint bit=bits;bit>0;bit--) {
            uint carry=(a.limb[(bit-1)/32]>>((bit-1)%32))&1u;
            #pragma clang loop unroll(disable)
            for(uint j=0;j<N;j++){uint next=r.limb[j]>>31;r.limb[j]=(r.limb[j]<<1)|carry;carry=next;}
            if(ci_py_ucmp(r,b)>=0){r=ci_py_usub(r,b);q.limb[(bit-1)/32]|=1u<<((bit-1)%32);}
          }
        }
        bool opposite=a.negative!=b.negative;
        if(opposite&&!ci_py_iszero(r)){q=ci_py_uadd(q,ci_py_from<N>(1UL,false));r=ci_py_usub(b,r);}
        q.negative=opposite&&!ci_py_iszero(q);r.negative=b.negative&&!ci_py_iszero(r);return;
}
template<uint N> inline void ci_py_divmod(thread const ci_py<N>& a,thread const ci_py<N>& b,thread ci_py<N>& q,thread ci_py<N>& r) {q=ci_py_zero<N>();r=ci_py_zero<N>();if(ci_py_iszero(b))return;
    if(N>32u){ci_py_divmod_wide(a,b,q,r);return;}
    // Small modular-power workspaces also contain mostly zero words.
    // Divide by one word directly and skip absent numerator bits otherwise.
    uint bits=0u,denominator_words=0u;
    for(uint i=0;i<N;i++){
        if(a.limb[i])bits=i*32u+32u-clz(a.limb[i]);
        if(b.limb[i])denominator_words=i+1u;
    }
    if(denominator_words==1u){
        ulong remainder=0UL;
        for(uint i=(bits+31u)/32u;i>0;i--){
            ulong value=(remainder<<32)|ulong(a.limb[i-1u]);
            q.limb[i-1u]=uint(value/ulong(b.limb[0]));remainder=value%ulong(b.limb[0]);
        }
        r.limb[0]=uint(remainder);
    }else{
        for(uint bit=bits;bit>0;bit--){
            uint carry=(a.limb[(bit-1u)/32u]>>((bit-1u)%32u))&1u;
            for(uint j=0;j<N;j++){uint next=r.limb[j]>>31;r.limb[j]=(r.limb[j]<<1)|carry;carry=next;}
            if(ci_py_ucmp(r,b)>=0){r=ci_py_usub(r,b);q.limb[(bit-1u)/32u]|=1u<<((bit-1u)%32u);}
        }
    }
    bool opposite=a.negative!=b.negative;
    if(opposite&&!ci_py_iszero(r)){q=ci_py_uadd(q,ci_py_from<N>(1UL,false));r=ci_py_usub(b,r);}
    q.negative=opposite&&!ci_py_iszero(q);r.negative=b.negative&&!ci_py_iszero(r);
}
template<uint N> inline ci_py<N> ci_py_mod(thread const ci_py<N>& a,thread const ci_py<N>& b) {ci_py<N> q,r;ci_py_divmod(a,b,q,r);return r;}
template<uint N> inline ci_py<N> ci_py_mod_product(thread const ci_py<N>& a,thread const ci_py<N>& b,thread const ci_py<N>& modulus) {
    ci_py<2*N> aa=ci_py_zero<2*N>(),bb=ci_py_zero<2*N>(),mm=ci_py_zero<2*N>();
    for(uint i=0;i<N;i++){aa.limb[i]=a.limb[i];bb.limb[i]=b.limb[i];mm.limb[i]=modulus.limb[i];}
    aa.negative=a.negative;bb.negative=b.negative;mm.negative=modulus.negative;
    ci_py<2*N> remainder=ci_py_mod(ci_py_mul(aa,bb),mm);ci_py<N> result=ci_py_zero<N>();
    for(uint i=0;i<N;i++)result.limb[i]=remainder.limb[i];result.negative=remainder.negative;return result;
}
// Keep the modular-power workspace in its own call frame as well. In a
// composed reset, inlining it can lose its scalar result across later wide
// calls; retaining the call boundary preserves the earlier exact value.
template<uint N> __attribute__((noinline)) ci_py<N> ci_py_pow_mod(thread const ci_py<N>& input,thread const ci_py<N>& exponent,thread const ci_py<N>& modulus) {
    ci_py<N> base=input;
    ci_py<N> result=ci_py_mod(ci_py_from<N>(1UL,false),modulus);base=ci_py_mod(base,modulus);
    uint bits=0u;for(uint i=0;i<N;i++)if(exponent.limb[i])bits=i*32u+32u-clz(exponent.limb[i]);
    for(uint bit=0;bit<bits;bit++){
        if((exponent.limb[bit/32u]>>(bit%32u))&1u)result=ci_py_mod_product(result,base,modulus);
        if(bit+1<bits)base=ci_py_mod_product(base,base,modulus);
    }
    return result;
}
template<uint N> inline ci_py<N> ci_py_floordiv(thread const ci_py<N>& a,thread const ci_py<N>& b) {ci_py<N> q,r;ci_py_divmod(a,b,q,r);return q;}
template<uint N> inline uint ci_py_bits(thread const ci_py<N>& a) {uint bits=0u;for(uint i=0;i<N;i++)if(a.limb[i])bits=i*32u+32u-clz(a.limb[i]);return bits;}
template<uint N> inline ulong ci_py_to_double(thread const ci_py<N>& a) {
    uint bits=ci_py_bits(a);ulong sign=a.negative?(1UL<<63):0UL;
    if(!bits)return sign;if(bits>1024u)return sign|0x7ff0000000000000UL;
    ulong significand=0UL;
    if(bits<=53u) {
        significand=(ulong(a.limb[0])|(ulong(a.limb[1])<<32))<<(53u-bits);
    } else {
        uint shift=bits-53u;
        for(uint bit=0;bit<53u;bit++)significand|=ulong((a.limb[(shift+bit)/32u]>>((shift+bit)%32u))&1u)<<bit;
        bool guard=((a.limb[(shift-1u)/32u]>>((shift-1u)%32u))&1u)!=0u,sticky=false;
        for(uint bit=0;bit+1u<shift;bit++)sticky|=((a.limb[bit/32u]>>(bit%32u))&1u)!=0u;
        if(guard&&(sticky||(significand&1UL)))significand++;
        if(significand==(1UL<<53)){significand>>=1;bits++;}
    }
    uint exponent=bits+1022u;if(exponent>=2047u)return sign|0x7ff0000000000000UL;
    return sign|(ulong(exponent)<<52)|(significand&0xfffffffffffffUL);
}
template<uint N> inline ci_py<N> ci_py_shift(thread const ci_py<N>& a,uint shift) {
    ci_py<N> result=ci_py_zero<N>();uint words=shift/32u,bits=shift%32u;
    for(uint i=0;i<N;i++)if(i+words<N){result.limb[i+words]|=a.limb[i]<<bits;if(bits&&i+words+1<N)result.limb[i+words+1]|=a.limb[i]>>(32u-bits);}
    result.negative=a.negative;return result;
}
template<uint N> inline ulong ci_py_truediv_double(thread const ci_py<N>& a,thread const ci_py<N>& b) {
    ulong sign=(a.negative!=b.negative)?(1UL<<63):0UL;
    if(ci_py_iszero(a)||ci_py_iszero(b))return sign;
    int exponent=int(ci_py_bits(a))-int(ci_py_bits(b));
    int compare=exponent>=0?ci_py_ucmp(a,ci_py_shift(b,uint(exponent))):ci_py_ucmp(ci_py_shift(a,uint(-exponent)),b);
    if(compare<0)exponent--;
    if(exponent>1023)return sign|0x7ff0000000000000UL;
    if(exponent< -1075)return sign;
    int shift=exponent< -1022?1074:52-exponent;
    // Keep caller inputs read-only and give shifted operands distinct storage.
    // The wide private-array ABI regression exercises these nested calls.
    ci_py<N> dividend=shift>=0?ci_py_shift(a,uint(shift)):a;
    ci_py<N> divisor=shift<0?ci_py_shift(b,uint(-shift)):b;
    dividend.negative=false;divisor.negative=false;
    ci_py<N> quotient,remainder;ci_py_divmod(dividend,divisor,quotient,remainder);
    ulong significand=ulong(quotient.limb[0])|(ulong(quotient.limb[1])<<32);
    compare=ci_py_ucmp(ci_py_shift(remainder,1u),divisor);
    if(compare>0||(compare==0&&(significand&1UL)))significand++;
    if(exponent< -1022)return sign|significand;
    if(significand==(1UL<<53)){significand>>=1;exponent++;}
    if(exponent>1023)return sign|0x7ff0000000000000UL;
    return sign|(ulong(exponent+1023)<<52)|(significand&0xfffffffffffffUL);
}
template<uint N> inline ci_py<N> ci_py_from_double(ulong raw) {
    ci_py<N> result=ci_py_zero<N>();int exponent=int((raw>>52)&2047UL)-1023;
    if(exponent<0)return result;
    ulong significand=(raw&0xfffffffffffffUL)|(1UL<<52);
    if(exponent<int(N*32u)) {
        if(exponent<52)result=ci_py_from<N>(significand>>uint(52-exponent),false);
        else result=ci_py_shift(ci_py_from<N>(significand,false),uint(exponent-52));
    } else for(uint i=0;i<N;i++)result.limb[i]=~0u;
    result.negative=(raw>>63)!=0UL&&!ci_py_iszero(result);return result;
}
template<uint N> inline int ci_py_cmp_double(thread const ci_py<N>& a,ulong raw) {
    ulong magnitude=raw&0x7fffffffffffffffUL;bool negative=(raw>>63)!=0UL;
    if(magnitude>0x7ff0000000000000UL)return 2;
    int exponent=int(magnitude>>52)-1023;
    if(magnitude==0x7ff0000000000000UL||exponent>=int(N*32u))return negative?1:-1;
    int comparison=ci_py_cmp(a,ci_py_from_double<N>(raw));
    if(comparison)return comparison;
    bool fractional=exponent<0?magnitude!=0UL:
        exponent<52?(magnitude&((1UL<<uint(52-exponent))-1UL))!=0UL:false;
    return fractional?(negative?1:-1):0;
}
template<uint N> inline ci_py<N> ci_py_truediv_integer(thread const ci_py<N>& numerator,thread const ci_py<N>& denominator) {
    bool negative=numerator.negative!=denominator.negative;
    ci_py<N> a=numerator,b=denominator;a.negative=false;b.negative=false;
    ci_py<N> quotient,remainder;ci_py_divmod(a,b,quotient,remainder);
    if(ci_py_iszero(b))return quotient;
    ci_py<N> unit=ci_py_from<N>(1UL,false);uint bits=ci_py_bits(quotient);
    if(bits>=53u) {
        uint shift=bits-53u;bool upward=false;
        if(shift) {
            bool guard=((quotient.limb[(shift-1)/32u]>>((shift-1)%32u))&1u)!=0u;
            bool sticky=!ci_py_iszero(remainder);for(uint bit=0;bit+1<shift;bit++)sticky|=((quotient.limb[bit/32u]>>(bit%32u))&1u)!=0u;
            upward=guard&&(sticky||((quotient.limb[shift/32u]>>(shift%32u))&1u));
            for(uint bit=0;bit<shift;bit++)quotient.limb[bit/32u]&=~(1u<<(bit%32u));
        } else {
            int comparison=ci_py_ucmp(ci_py_shift(remainder,1u),b);
            upward=comparison>0||(comparison==0&&(quotient.limb[0]&1u));
        }
        if(upward)quotient=ci_py_uadd(quotient,ci_py_shift(unit,shift));
    } else if(!ci_py_iszero(remainder)) {
        // Only rounding across the next integer changes truncation. Its lower
        // midpoint is next - 2**(-precision); powers of two have half-sized
        // predecessor spacing. The next integer has an even mantissa here.
        ci_py<N> next=ci_py_uadd(quotient,unit);uint next_bits=ci_py_bits(next);
        ci_py<N> lower=ci_py_usub(next,unit);uint overlap=0u;
        for(uint i=0;i<N;i++)overlap|=next.limb[i]&lower.limb[i];
        uint precision=54u-next_bits+(overlap==0u?1u:0u);
        if(ci_py_ucmp(ci_py_shift(ci_py_usub(b,remainder),precision),b)<=0)quotient=next;
    }
    quotient.negative=negative&&!ci_py_iszero(quotient);return quotient;
}
#endif
'''


def _dtype(type_):
    if getattr(type_,'is_bint',False):return 'bool'
    if getattr(type_,'is_float',False):return 'float32' if str(type_)=='float' else 'float64'
    if getattr(type_,'is_int',False):
        rank=getattr(type_,'rank',2);bits={0:8,1:16,2:32,3:64,4:64}.get(rank)
        if bits is not None:return ('int' if getattr(type_,'signed',True) else 'uint')+str(bits)
    if getattr(type_,'is_bint',False):return 'bool'
    return None


def _facts(cpp):
    """Retain Cython's own typed nodes before code generation, under its lock."""
    from Cython.Compiler import Main,Pipeline
    from Cython.Compiler.Nodes import Node
    from .compiled_integer_frontend import _LOCK
    path=cpp.with_suffix('.python-types-v3.json')
    with _LOCK, cache_lock(path):
        if path.exists():return json.loads(path.read_text())
        factory=Pipeline.generate_pyx_code_stage_factory;captured=[]
        def serialize(node):
            row=dict(node=type(node).__name__,type=str(getattr(node,'type','')),
                     dtype=_dtype(getattr(node,'type',None)))
            for name in ('name','value','operator','cname'):
                value=getattr(node,name,None)
                if isinstance(value,(str,int,float,bool)):row[name]=value
            for name in getattr(node,'child_attrs',()):
                value=getattr(node,name,None)
                if isinstance(value,Node):row[name]=serialize(value)
                elif isinstance(value,(list,tuple)):row[name]=[serialize(item) for item in value if isinstance(item,Node)]
            return row
        def assignments(node):
            if type(node).__name__=='SingleAssignmentNode' and getattr(node.lhs,'name',None) is not None:
                captured.append(dict(name=node.lhs.name,line=node.pos[1],rhs=serialize(node.rhs)))
            for field in getattr(node,'child_attrs',()):
                child=getattr(node,field,None)
                if isinstance(child,Node):assignments(child)
                elif isinstance(child,(list,tuple)):
                    for item in child:
                        if isinstance(item,Node):assignments(item)
        def functions(node):
            if type(node).__name__=='CFuncDefNode' and getattr(getattr(node,'entry',None),'name',None)=='integer_block':
                assignments(node.body);return
            for field in getattr(node,'child_attrs',()):
                child=getattr(node,field,None)
                if isinstance(child,Node):functions(child)
                elif isinstance(child,(list,tuple)):
                    for item in child:
                        if isinstance(item,Node):functions(item)
        def wrapping(options,result):
            generate=factory(options,result)
            def stage(tree):functions(tree);return generate(tree)
            return stage
        Pipeline.generate_pyx_code_stage_factory=wrapping
        try:
            result=Main.compile(str(cpp.with_suffix('.pyx')),options=Main.CompilationOptions(
                Main.default_options,cplus=True,language_level=3,output_file=str(cpp.with_suffix('.typed.cpp')),cache=False))
            if result.num_errors:raise UnsupportedModelError('Original Cython Python-integer typed inspection was rejected')
        finally:Pipeline.generate_pyx_code_stage_factory=factory
        atomic_write(path,json.dumps(captured,indent=2)+'\n')
        return captured


def rewrites(compiled,statements,types,policy):
    cpp,_,_,_,_,_=compiled
    facts=_facts(cpp);by_name={}
    for row in sorted(facts,key=lambda row:row['line']):by_name.setdefault(row['name'],[]).append(row['rhs'])
    result={}
    for index,statement in enumerate(statements):
        rows=by_name.get(statement.var)
        if not rows:continue
        root=rows.pop(0)
        def contains_python(node):
            return ('object' in node.get('type','') or any(contains_python(child)
                for value in node.values() for child in (value if isinstance(value,list) else [value])
                if isinstance(child,dict)))
        if not contains_python(root):continue
        parameters=[];dtypes=[]
        def c_expression(node):
            kind=node['node']
            if kind=='NameNode':return node['name']
            if kind in ('IntNode','BoolNode','FloatNode'):return str(node['value'])
            if kind=='SimpleCallNode' and node['function'].get('name'):
                return node['function']['name']+'('+','.join(c_expression(arg) for arg in node['args'])+')'
            if 'operator' in node:
                if 'operand1' in node:return '('+c_expression(node['operand1'])+node['operator']+c_expression(node['operand2'])+')'
                if 'operand' in node:return '('+node['operator']+c_expression(node['operand'])+')'
            if kind in ('CoerceToTempNode','CloneNode'):return c_expression(node['arg'])
            raise UnsupportedModelError('Native Cython boxing operand needs typed source: '+kind)
        def expression(node):
            kind=node['node']
            if kind in ('CoerceToPyTypeNode','CoerceIntToBytesNode'):
                child=node['arg'];dtype=child.get('dtype')
                if child.get('type')=='char':dtype='uint8' if policy.unsigned_char else 'int8'
                if dtype is None:raise UnsupportedModelError('Unsupported Cython boxing operand '+child['type'])
                def checked(item):
                    return (item.get('operator') in ('/','//','%') or any(checked(value)
                        for child in item.values()
                        for value in (child if isinstance(child,list) else [child])
                        if isinstance(value,dict)))
                if checked(child):
                    # Keep checked C evaluation at its actual Python operand
                    # position; a function argument would run before the
                    # earlier boxed operand, changing the first exception.
                    leaves={}
                    def capture(item):
                        if item['node']=='NameNode' and item.get('dtype') is not None:
                            if item['name'] not in leaves:
                                dt=item['dtype']
                                if item.get('type')=='char':dt='uint8' if policy.unsigned_char else 'int8'
                                leaves[item['name']]=(len(parameters),dt)
                                parameters.append(item['name']);dtypes.append(dt)
                            return
                        for value in item.values():
                            if isinstance(value,dict):capture(value)
                            elif isinstance(value,list):
                                for value_ in value:
                                    if isinstance(value_,dict):capture(value_)
                    capture(child)
                    bindings=tuple(item for name,(position,dt) in leaves.items() for item in (name,position,dt))
                    return ('native_c',c_expression(child),dtype,*bindings)
                position=len(parameters);parameters.append(c_expression(child));dtypes.append(dtype)
                return ('argument',position,dtype)
            if kind in ('IntNode','BoolNode'):return ('constant',int(node['value']))
            if kind=='FloatNode':return ('floating',float(node['value']))
            if kind=='SimpleCallNode' and node['function'].get('name')=='abs' and len(node['args'])==1:
                return ('abs',expression(node['args'][0]))
            if kind=='PowNode' and node['operand2']['node']=='IntNode':
                exponent=int(node['operand2']['value'])
                if exponent>=0:return ('pow',expression(node['operand1']),exponent)
            if kind=='PowNode':return ('pow_dynamic',expression(node['operand1']),expression(node['operand2']))
            if kind=='PythonCapiCallNode':
                name=node['function'].get('cname','')
                match=re.fullmatch(r'__Pyx_Py(?:Long|Float)_(Add|Subtract|Multiply|Remainder|FloorDivide|TrueDivide|Divide)(ObjC|CObj)',name)
                if not match:raise UnsupportedModelError('Unlowered Cython Python integer intrinsic '+name)
                op={'Add':'add','Subtract':'sub','Multiply':'mul','Remainder':'mod','FloorDivide':'floordiv','TrueDivide':'truediv','Divide':'truediv'}[match[1]]
                a,b=(expression(item) for item in node['args'][:2])
                # Both Cython utility variants take original op1/op2 pointers;
                # CObj only selects which operand also has a native scalar.
                return (op,a,b)
            if 'operator' in node:
                operations={'+':'add','-':'sub','*':'mul','%':'mod','//':'floordiv','/':'truediv','>':'gt','>=':'ge','<':'lt','<=':'le','==':'eq','!=':'ne'}
                if 'operand1' in node and node['operator'] in operations:
                    return (operations[node['operator']],expression(node['operand1']),expression(node['operand2']))
                if 'operand' in node and node['operator'] in ('+','-'):
                    return ('pos' if node['operator']=='+' else 'neg',expression(node['operand']))
            raise UnsupportedModelError('Unlowered Cython Python integer expression '+kind)
        specifications=[]
        def helper(node,destination):
            nonlocal parameters,dtypes
            parameters=[];dtypes=[]
            tree=expression(node)
            if not parameters:parameters=['0'];dtypes=['int64']
            descriptor=('python_expr',tree,dtypes[0],destination,*dtypes[1:])
            number=len(specifications);specifications.append((descriptor,parameters))
            return f'@_eon_python_helper_{number}@'
        def source(node):
            kind=node['node']
            if kind in ('ProxyNode','BoolBinopResultNode','CoerceToBooleanNode','CoerceToTempNode','CloneNode'):
                return source(node['arg'])
            if kind=='NotNode':return '(not '+source(node['operand'])+')'
            if kind=='BoolBinopNode':
                return '('+source(node['operand1'])+' '+node['operator']+' '+source(node['operand2'])+')'
            if kind=='PrimaryCmpNode':
                def python_value(item):
                    return 'object' in item.get('type','') or any(python_value(child) for child in item.values() if isinstance(child,dict))
                return helper(node,'bool') if python_value(node) else c_expression(node)
            if kind=='CoerceFromPyTypeNode':
                destination=node.get('dtype')
                if destination not in ('float32','float64'):destination=np.dtype(statement.dtype).name
                return helper(node['arg'],destination)
            if kind=='SimpleCallNode' and node['function'].get('name'):
                return node['function']['name']+'('+','.join(source(arg) for arg in node['args'])+')'
            if 'operator' in node:
                if 'operand1' in node:return '('+source(node['operand1'])+node['operator']+source(node['operand2'])+')'
                if 'operand' in node:return '('+node['operator']+source(node['operand'])+')'
            return c_expression(node)
        rewritten=source(root)
        if specifications:result[index]=(specifications,rewritten)
    return result


def render(tree,arguments,destination,policy,prefix,register_error,*,used_math=None,
           _limbs_override=None,_raw_integer=False,_native_operand=None,_native_precision_limbs=0):
    """Return exact private-limb statements and raw checked destination bits."""
    def integer_tree(node):
        if node[0]=='argument':return np.dtype(node[2]).kind in 'iub'
        if node[0]=='constant':return True
        return (node[0] in ('abs','neg','pos','add','sub','mul','mod','floordiv','pow')
                and all(integer_tree(child) for child in node[1:] if isinstance(child,(tuple,list))))
    def operand_key(node):
        if node[0]=='argument':return ('argument',node[2],arguments[node[1]])
        return tuple(operand_key(child) if isinstance(child,(tuple,list)) else child for child in node)
    power_mod=None
    if tree[0]=='mod':
        inner=tree[1];wrapped=False
        if inner[0]=='add' and inner[1][0]=='mod':
            if operand_key(tree[2])==operand_key(inner[2])==operand_key(inner[1][2]):
                inner=inner[1][1];wrapped=True
        if inner[0]=='pow_dynamic' and all(integer_tree(child) for child in (*inner[1:],tree[2])):
            power_mod=(*inner[1:],tree[2],wrapped)
    float_powers=set()
    if tree[0]=='pow_dynamic' and np.dtype(destination).kind=='f':float_powers.add(id(tree))
    def known_float(node):
        op=node[0]
        if op=='native_c':return np.dtype(node[2]).kind=='f'
        if op=='argument':return np.dtype(node[2]).kind=='f'
        if op in ('floating','truediv'):return True
        if op in ('constant','pow_dynamic','gt','ge','lt','le','eq','ne'):return False
        return any(known_float(child) for child in node[1:] if isinstance(child,(tuple,list)))
    # A surrounding float operation cannot coerce its integer power operand
    # until BOTH operands have evaluated. Nonroot powers retain their runtime
    # Python numeric type in the contextual renderer instead of casting early.
    def unresolved(node):
        return ((node[0]=='pow_dynamic' and node is not tree and id(node) not in float_powers)
                or any(unresolved(child) for child in node[1:] if isinstance(child,(tuple,list))))
    def dynamic_render():
        from .compiled_python_dynamic import render as render_dynamic
        if used_math is None:raise UnsupportedModelError('Dynamic boxed values need canonical native math dependencies')
        return render_dynamic(tree,arguments,destination,policy,prefix,register_error,
            used_math=used_math,native_operand=_native_operand,limbs=_native_precision_limbs or 8)
    if tree[0]=='pow_dynamic' and not all(integer_tree(child) for child in tree[1:]):
        return dynamic_render()
    if not power_mod and unresolved(tree):
        from .compiled_python_context import fast_render
        fast=fast_render(tree,arguments,destination,policy,prefix,register_error,
                         used_math=used_math,native_operand=_native_operand)
        if fast is not None:return fast
        from .compiled_python_context import render as render_context
        try:
            return render_context(tree,arguments,destination,policy,prefix,register_error,
                                  used_math=used_math,native_operand=_native_operand)
        except UnsupportedModelError:
            return dynamic_render()
    all_bounds=[]
    def width(node):
        op=node[0]
        if node is tree and power_mod:
            a,b,q=(width(child) for child in power_mod[:3])
            if max(a,b)>64:raise UnsupportedModelError('Boxed modular power needs exact wide operand float conversions')
            bits=max(a,b,q)+2
        elif op=='native_power_integer':bits=node[4]
        elif op=='native_integer':bits=node[2]
        elif op=='native_power_float':bits=64
        elif op in ('argument','native_c'):bits=np.dtype(node[2]).itemsize*8
        elif op=='constant':bits=abs(node[1]).bit_length()
        elif op=='floating':bits=64
        elif op in ('abs','neg','pos'):bits=width(node[1])
        elif op=='pow':bits=max(1,width(node[1])*node[2])
        elif op=='pow_dynamic':
            a,b=width(node[1]),width(node[2])
            if (node is not tree and id(node) not in float_powers) or max(a,b)>64:
                raise UnsupportedModelError('Boxed dynamic power composition needs an exact contextual magnitude policy')
            # Saturation is admitted only immediately before conversion. For
            # floats,1056bits retain every finite binary64 conversion boundary;
            # larger magnitudes necessarily overflow CPython's float conversion.
            # No following arithmetic can cancel or inspect discarded bits.
            bits=max(a,b,1025 if id(node) in float_powers else np.dtype(destination).itemsize*8+1)
        elif op=='truediv':
            bits=max(width(node[1]),width(node[2]))+55
        else:
            a,b=width(node[1]),width(node[2]);bits=(1 if op in ('gt','ge','lt','le','eq','ne') else a+b if op=='mul' else max(a,b)+1 if op in ('add','sub') else b if op=='mod' else a+1)
        all_bounds.append(bits);return bits
    width(tree);limbs=max(2,(max(all_bounds)+2+31)//32,_limbs_override or 0)
    if tree[0]=='native_integer':limbs=max(2,(tree[2]+31)//32)
    type_=f'ci_py<{limbs}>'
    lines=[];sequence=0
    def floating(node):
        op=node[0]
        if node is tree and power_mod:return np.dtype(destination).kind=='f'
        if op=='native_power_float':return True
        if op=='native_power_integer':return False
        if op in ('argument','native_c'):return np.dtype(node[2]).kind=='f'
        if op=='floating':return True
        if op=='truediv':return node is not tree or np.dtype(destination).kind=='f' or floating(node[1]) or floating(node[2])
        if op in ('constant','gt','ge','lt','le','eq','ne'):return False
        if op=='pow_dynamic':return id(node) in float_powers
        if op=='pow':return floating(node[1])
        return any(floating(child) for child in node[1:] if isinstance(child,(tuple,list)))
    def integer_float(value):
        nonlocal sequence
        name=f'{prefix}_float_{sequence}';sequence+=1
        lines.append(f'const ulong {name}=ci_py_to_double({value});')
        code=register_error('OverflowError','int too large to convert to float')
        lines.append(f'if(({name}&0x7ff0000000000000UL)==0x7ff0000000000000UL && !_eon_integer_error) _eon_integer_error={code}u;')
        return name
    def to_float(node):
        value=emit(node)
        return value if floating(node) else integer_float(value)
    def float_pair(left,right):
        a,b=emit(left),emit(right)
        return (a if floating(left) else integer_float(a),
                b if floating(right) else integer_float(b))
    def emit(node):
        nonlocal sequence
        name=f'{prefix}_big_{sequence}';sequence+=1;op=node[0]
        if node is tree and power_mod:
            base,exponent=(emit(child) for child in power_mod[:2])
            if used_math is None:raise UnsupportedModelError('Boxed modular power requires canonical math dependencies')
            used_math.update(('operator_pow_float','operator_remainder_float'))
            dtype=np.dtype(destination)
            lines.append(f'{"ulong" if dtype.kind=="f" else type_} {name}='+('0UL;' if dtype.kind=='f' else f'ci_py_zero<{limbs}>();'))
            lines.append(f'if({exponent}.negative) {{')
            af,bf=integer_float(base),integer_float(exponent)
            code=register_error('ZeroDivisionError','0.0 cannot be raised to a negative power')
            lines.append(f'if(ci_py_iszero({base}) && !_eon_integer_error) _eon_integer_error={code}u;')
            real=name+'_real';lines.append(f'ulong {real}=d_operator_pow({af},{bf});')
            modulus=emit(power_mod[2])
            qf=integer_float(modulus);code=register_error('ZeroDivisionError','float modulo')
            lines.append(f'if(d_is_zero({qf}) && !_eon_integer_error) _eon_integer_error={code}u;')
            lines.append(f'{real}=d_operator_mod_numpy({real},{qf});')
            if power_mod[3]:lines.append(f'{real}=d_operator_mod_numpy(d_add({real},{qf}),{qf});')
            if dtype.kind=='f':lines.append(f'{name}={real};')
            elif dtype.kind=='b':lines.append(f'{name}=ci_py_from<{limbs}>(ulong(!d_is_zero({real})),false);')
            else:lines.append(f'{name}=ci_py_from_double<{limbs}>({real});')
            lines.append('} else {')
            modulus=emit(power_mod[2])
            code=register_error('ZeroDivisionError','integer division or modulo by zero')
            lines.append(f'if(ci_py_iszero({modulus}) && !_eon_integer_error) _eon_integer_error={code}u;')
            result=name+'_integer';lines.append(f'{type_} {result}=ci_py_pow_mod({base},{exponent},{modulus});')
            result=integer_float(result) if dtype.kind=='f' else result
            lines.append(f'{name}={result};'+'}')
            return name
        elif op=='native_integer':return node[1]
        elif op=='native_power_integer':
            base,exponent,capacity=node[1:4]
            small=name+'_base';power=name+'_power';ee=name+'_exponent'
            lines.append(f'ci_py<{capacity}> {small}=ci_py_zero<{capacity}>(),{ee}=ci_py_zero<{capacity}>();')
            lines.append(f'for(uint j=0;j<{capacity}u;j++){{{small}.limb[j]={base}.limb[j];{ee}.limb[j]={exponent}.limb[j];}}')
            lines.append(f'{small}.negative={base}.negative;')
            lines.append(f'ci_py<{capacity}> {power}=ci_py_pow_bounded({small},{ee});')
            lines.append(f'{type_} {name}=ci_py_zero<{limbs}>();')
            lines.append(f'for(uint j=0;j<{capacity}u;j++){name}.limb[j]={power}.limb[j];{name}.negative={power}.negative;')
            return name
        elif op=='native_power_float':
            a,b=node[1:3];af,bf=integer_float(a),integer_float(b)
            if used_math is None:raise UnsupportedModelError('Boxed contextual power requires canonical math dependencies')
            used_math.add('operator_pow_float');code=register_error('ZeroDivisionError','0.0 cannot be raised to a negative power')
            lines.append(f'if(ci_py_iszero({a}) && !_eon_integer_error) _eon_integer_error={code}u;')
            lines.append(f'const ulong {name}=d_operator_pow({af},{bf});');return name
        elif op in ('argument','native_c'):
            dtype=np.dtype(node[2])
            if op=='native_c':
                if _native_operand is None:raise UnsupportedModelError('Deferred Cython operand needs its native typed graph')
                source,value=_native_operand(node,arguments,name+'_native')
                lines.extend(source)
            else:value=arguments[node[1]]
            if dtype.kind=='f':
                lines.append(f'const ulong {name}='+('f_to_d('+value+')' if dtype.itemsize==4 else value)+';')
                return name
            negative=f'(long({value})<0)' if dtype.kind=='i' else 'false'
            expression=f'ci_py_from<{limbs}>({negative} ? 0UL-ulong({value}) : ulong({value}),{negative})'
        elif op=='constant':
            value=abs(node[1]);lines.append(f'{type_} {name}=ci_py_zero<{limbs}>();')
            for i in range(limbs):
                part=(value>>(32*i))&0xffffffff
                if part:lines.append(f'{name}.limb[{i}]={part}u;')
            if node[1]<0:lines.append(f'{name}.negative=true;')
            return name
        elif op=='floating':
            bits=int(np.asarray(node[1],np.float64).view(np.uint64))
            lines.append(f'const ulong {name}={bits}UL;');return name
        elif op=='truediv' and (node is not tree or np.dtype(destination).kind=='f' or floating(node[1]) or floating(node[2])):
            if floating(node[1]) or floating(node[2]):
                a,b=float_pair(node[1],node[2]);zero=f'd_is_zero({b})';expression=f'd_div({a},{b})';message='float division by zero'
            else:
                a,b=emit(node[1]),emit(node[2]);zero=f'ci_py_iszero({b})';expression=f'ci_py_truediv_double({a},{b})';message='division by zero'
            code=register_error('ZeroDivisionError',message)
            lines.append(f'if({zero} && !_eon_integer_error) _eon_integer_error={code}u;')
            lines.append(f'const ulong {name}={expression};')
            if not floating(node[1]) and not floating(node[2]):
                code=register_error('OverflowError','integer division result too large for a float')
                lines.append(f'if(({name}&0x7fffffffffffffffUL)==0x7ff0000000000000UL && !_eon_integer_error) _eon_integer_error={code}u;')
            return name
        elif op=='pow_dynamic' and id(node) in float_powers:
            a,b=emit(node[1]),emit(node[2]);lines.append(f'ulong {name}=0UL;')
            af,bf=integer_float(a),integer_float(b)
            if used_math is None:raise UnsupportedModelError('Boxed floating power requires canonical math dependencies')
            used_math.add('operator_pow_float')
            code=register_error('ZeroDivisionError','0.0 cannot be raised to a negative power')
            lines.append(f'if({b}.negative) {{')
            lines.append(f'if(ci_py_iszero({a}) && !_eon_integer_error) _eon_integer_error={code}u;')
            lines.append(f'{name}=d_operator_pow({af},{bf});')
            lines.append('} else {')
            power=name+'_integer';lines.append(f'{type_} {power}=ci_py_pow_bounded({a},{b});')
            converted=integer_float(power);lines.append(f'{name}={converted};'+'}')
            return name
        elif op=='pow_dynamic' and np.dtype(destination).kind=='b':
            a,b=emit(node[1]),emit(node[2]);af,bf=integer_float(a),integer_float(b)
            if used_math is None:raise UnsupportedModelError('Boxed power truth requires canonical math dependencies')
            used_math.add('operator_pow_float')
            code=register_error('ZeroDivisionError','0.0 cannot be raised to a negative power')
            lines.append(f'if({b}.negative && ci_py_iszero({a}) && !_eon_integer_error) _eon_integer_error={code}u;')
            expression=f'ci_py_from<{limbs}>(ulong({b}.negative?!d_is_zero(d_operator_pow({af},{bf})):(!ci_py_iszero({a})||ci_py_iszero({b}))),false)'
        elif floating(node):
            if op=='pos':return to_float(node[1])
            if op in ('abs','neg'):
                a=to_float(node[1]);expression=f'({a}&0x7fffffffffffffffUL)' if op=='abs' else f'({a}^(1UL<<63))'
            elif op in ('add','sub','mul'):
                a,b=float_pair(node[1],node[2]);expression=f'd_{op}({a},{b})'
            elif op in ('mod','floordiv'):
                a,b=float_pair(node[1],node[2])
                if used_math is None:raise UnsupportedModelError('Boxed floating remainder requires canonical math dependencies')
                used_math.add('operator_remainder_float')
                message='float modulo' if op=='mod' else 'float floor division by zero'
                code=register_error('ZeroDivisionError',message)
                lines.append(f'if(d_is_zero({b}) && !_eon_integer_error) _eon_integer_error={code}u;')
                expression=f'd_operator_{op}_numpy({a},{b})'
            else:raise UnsupportedModelError('Boxed floating composition needs exact '+op+' policy')
            lines.append(f'const ulong {name}={expression};');return name
        elif op=='pos':return emit(node[1])
        elif op in ('abs','neg'):expression=f'ci_py_{op}({emit(node[1])})'
        elif op=='pow':
            # Python's nonnegative integer exponent keeps exact intermediates.
            # Exponentiation by squaring is emitted at preparation, retaining
            # arbitrarily sized literal exponents without a narrowed runtime ABI.
            base=emit(node[1]);exponent=node[2]
            lines.append(f'{type_} {name}=ci_py_from<{limbs}>(1UL,false);')
            power=name+'_power';lines.append(f'{type_} {power}={base};')
            while exponent:
                if exponent&1:lines.append(f'{name}=ci_py_mul({name},{power});')
                exponent>>=1
                if exponent:lines.append(f'{power}=ci_py_mul({power},{power});')
            return name
        elif op=='pow_dynamic':
            base,exponent=emit(node[1]),emit(node[2]);unit=name+'_unit'
            lines.append(f'{type_} {unit}=ci_py_from<{limbs}>(1UL,false);')
            lines.append(f'{type_} {name}=ci_py_zero<{limbs}>();')
            code=register_error('ZeroDivisionError','0.0 cannot be raised to a negative power')
            # Both operands are <=64-bit magnitudes, so CPython's preliminary
            # integer-to-double conversions cannot overflow. For negative integer
            # exponents, |base|>=2 has magnitude<1 and converts exactly to zero;
            # +/-1 are exact, but CPython first rounds the exponent to binary64;
            # every integer at/above2**53 is even after that conversion.
            lines.append(f'if({exponent}.negative) {{')
            lines.append(f'if(ci_py_iszero({base}) && !_eon_integer_error) _eon_integer_error={code}u;')
            lines.append(f'if(ci_py_ucmp({base},{unit})==0) {{{name}={unit};{name}.negative={base}.negative&&(ci_py_bits({exponent})<=53u)&&(({exponent}.limb[0]&1u)!=0u);}}')
            lines.append(f'}} else {name}=ci_py_pow_bounded({base},{exponent});')
            return name
        else:
            a,b=emit(node[1]),emit(node[2])
            if op in ('mod','floordiv','truediv'):
                message='division by zero' if op=='truediv' else 'integer division or modulo by zero'
                kind=register_error('ZeroDivisionError',message)
                lines.append(f'if(ci_py_iszero({b}) && !_eon_integer_error) _eon_integer_error={kind}u;')
            if op=='truediv':
                real=name+'_real'
                lines.append(f'const ulong {real}=ci_py_truediv_double({a},{b});')
                code=register_error('OverflowError','integer division result too large for a float')
                lines.append(f'if(({real}&0x7fffffffffffffffUL)==0x7ff0000000000000UL && !_eon_integer_error) _eon_integer_error={code}u;')
                lines.append(f'{type_} {name}=ci_py_from_double<{limbs}>({real});')
                return name
            if op in ('gt','ge','lt','le','eq','ne'):
                compare={'gt':'>','ge':'>=','lt':'<','le':'<=','eq':'==','ne':'!='}[op]
                af,bf=floating(node[1]),floating(node[2])
                if af or bf:
                    ordering=name+'_ordering'
                    if af and bf:ordered=f'(d_is_nan({a})||d_is_nan({b})?2:d_lt({a},{b})?-1:d_gt({a},{b})?1:0)'
                    elif bf:ordered=f'ci_py_cmp_double({a},{b})'
                    else:ordered=f'ci_py_cmp_double({b},{a})'
                    lines.append(f'int {ordering}={ordered};')
                    if af and not bf:lines.append(f'if({ordering}!=2){ordering}=-{ordering};')
                    predicate=f'({ordering}==2?'+('true' if op=='ne' else 'false')+f':{ordering}{compare}0)'
                    expression=f'ci_py_from<{limbs}>(ulong({predicate}),false)'
                else:expression=f'ci_py_from<{limbs}>(ulong(ci_py_cmp({a},{b}){compare}0),false)'
            else:expression=f'ci_py_{op}({a},{b})'
        lines.append(f'{type_} {name}={expression};');return name
    value=emit(tree);declared_dtype=np.dtype(destination);dtype=declared_dtype
    if _raw_integer:
        if floating(tree):raise UnsupportedModelError('Internal boxed integer context contains a floating value')
        return lines,value
    if dtype.kind=='f':
        raw=value if floating(tree) else integer_float(value)
        return lines,f'f_from_d({raw})' if dtype.itemsize==4 else raw
    if dtype.kind=='b':return lines,f'!d_is_zero({value})' if floating(tree) else f'!ci_py_iszero({value})'
    if floating(tree):
        code=register_error('ValueError','cannot convert float NaN to integer')
        lines.append(f'if(d_is_nan({value}) && !_eon_integer_error) _eon_integer_error={code}u;')
        code=register_error('OverflowError','cannot convert float infinity to integer')
        lines.append(f'if(({value}&0x7fffffffffffffffUL)==0x7ff0000000000000UL && !_eon_integer_error) _eon_integer_error={code}u;')
        integer=prefix+'_integer';lines.append(f'{type_} {integer}=ci_py_from_double<{limbs}>({value});');value=integer
    if dtype==np.dtype('uint8') and not policy.unsigned_char:dtype=np.dtype('int8')
    from brian2.codegen.generators.cython_generator import get_cpp_dtype
    maximum=int(np.iinfo(dtype).max);minimum=int(np.iinfo(dtype).min)
    low=f'(ulong({value}.limb[0])|(ulong({value}.limb[1])<<32))'
    high='|'.join(f'{value}.limb[{i}]' for i in range(2,limbs)) or '0u'
    magnitude=prefix+'_magnitude';lines.append(f'const ulong {magnitude}={low};')
    negative=f'({value}.negative && {magnitude}!=0UL)'
    outside=(f'({value}.negative ? {magnitude}>{-minimum}UL : {magnitude}>{maximum}UL)' if dtype.kind=='i' else
             f'({value}.negative || {magnitude}>{maximum}UL)')
    outside=f'(({high})!=0u || {outside})'
    message=('Python int too large to convert to C long' if dtype==np.dtype('int64') else
             'Python int too large to convert to C unsigned long' if dtype==np.dtype('uint64') else
             'value too large to convert to '+get_cpp_dtype(declared_dtype.type))
    code=register_error('OverflowError',message)
    if dtype.kind=='u':
        negative_code=register_error('OverflowError',"can't convert negative value to "+get_cpp_dtype(declared_dtype.type))
        code=f'({value}.negative ? {negative_code}u : {code}u)'
    else:code=str(code)+'u'
    lines.append(f'if({outside} && !_eon_integer_error) _eon_integer_error={code};')
    return lines,f'({value}.negative ? 0UL-{magnitude} : {magnitude})'
