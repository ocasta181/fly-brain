"""Exact common-power polynomial contexts, with Python's runtime numeric type.

A positive integer power stays integer through subsequent integer operations;
a negative exponent produces a Python float before those operations. Both
branches reuse the canonical boxed renderer and the existing binary64 helpers.

For the integer branch, a statically bounded surrogate is used only beyond a
proved coercion boundary. The coefficient bound sums absolute polynomial
coefficients (never their possibly cancelling values). Once P exceeds 2**K,
where K exceeds that bound plus the binary64 domain, a nonconstant polynomial
cannot return to a representable result. Constant cancellations are evaluated
exactly at the surrogate with enough limbs for the entire polynomial degree.
For an integer ratio A(P)/B(P), comparison against any binary64 rounding
midpoint n/d is the sign of the integer polynomial d*A(P)-n*B(P). Those
midpoints admit |n|<2**1025 and d<=2**1075. The same coefficient bound plus
1076 bits puts the surrogate past every such polynomial's Cauchy root bound,
including cancellation, exact ties, subnormal and overflow boundaries.
Integer remainder/floor-division and distinct powers need separate policies.
"""
from __future__ import annotations
import numpy as np
from .compiler import UnsupportedModelError


def render(tree,arguments,destination,policy,prefix,register_error,*,used_math,native_operand=None):
    from .compiled_python_integer import render as boxed
    powers=[];bounds=[]
    def children(node):return [child for child in node[1:] if isinstance(child,(tuple,list))]
    def collect(node):
        if node[0]=='pow_dynamic':powers.append(node)
        for child in children(node):collect(child)
    collect(tree)
    def key(node):
        if node[0]=='argument':return ('argument',node[2],arguments[node[1]])
        return tuple(key(child) if isinstance(child,(tuple,list)) else child for child in node)
    if not powers or any(key(node)!=key(powers[0]) for node in powers):
        raise UnsupportedModelError('Distinct boxed dynamic powers need an exact joint magnitude policy')
    def pure(node):
        if node[0]=='argument':return np.dtype(node[2]).kind in 'iub'
        if node[0]=='constant':return True
        return node[0] in ('abs','neg','pos','add','sub','mul','pow') and all(pure(child) for child in children(node))
    base,exponent=powers[0][1:]
    if not pure(base) or not pure(exponent):
        raise UnsupportedModelError('Contextual boxed power needs a pure integer base and exponent')
    def known_float(node):
        if node[0] in ('argument','native_c'):return np.dtype(node[2]).kind=='f'
        if node[0]=='floating':return True
        if node[0]=='pow_dynamic':return False
        if node[0]=='truediv':return True
        return any(known_float(child) for child in children(node))
    def shape(node):
        op=node[0]
        if op=='pow_dynamic':degree,bits=1,1
        elif op in ('argument','native_c'):degree,bits=0,np.dtype(node[2]).itemsize*8
        elif op=='constant':degree,bits=0,abs(node[1]).bit_length()
        elif op=='floating':degree,bits=0,64
        elif op in ('abs','neg','pos'):degree,bits=shape(node[1])
        elif op=='pow':
            degree,bits=shape(node[1])
            if degree:raise UnsupportedModelError('A further boxed power needs its conditional floating policy')
            bits=max(1,bits*node[2])
        else:
            da,a=shape(node[1]);db,b=shape(node[2])
            float_operation=(op=='truediv' or (op in ('mod','floordiv') and
                             (known_float(node[1]) or known_float(node[2]))))
            if (da or db) and op not in ('add','sub','mul','gt','ge','lt','le','eq','ne') and not float_operation:
                raise UnsupportedModelError('Boxed power context needs exact '+op+' magnitude/type handling')
            degree=da+db if op=='mul' else max(da,db)
            bits=(a+b if op=='mul' else max(a,b)+1 if op in ('add','sub') else
                  max(a,b)+55 if op=='truediv' else b if op=='mod' else max(a,b)+1)
        bounds.append((degree,bits));return degree,bits
    # Only the nonnegative-exponent branch has Python integer ring semantics.
    # Canonicalize that branch before allocating private precision. In
    # particular, exact cancellation must not force huge unused intermediates
    # through Metal's template optimizer. Treat non-ring operations as opaque
    # atoms; they are never discarded unless they are proved error-free.
    atoms={}
    def ring(node):
        op=node[0]
        if op=='constant':return {():int(node[1])} if node[1] else {}
        if op in ('neg','pos'):
            result=ring(node[1])
            return {term:(-value if op=='neg' else value) for term,value in result.items()}
        if op in ('add','sub','mul'):
            left,right=ring(node[1]),ring(node[2]);result=dict(left) if op!='mul' else {}
            if op=='mul':
                for a,ca in left.items():
                    for b,cb in right.items():
                        term=tuple(sorted(a+b));result[term]=result.get(term,0)+ca*cb
            else:
                sign=-1 if op=='sub' else 1
                for term,value in right.items():result[term]=result.get(term,0)+sign*value
            return {term:value for term,value in result.items() if value}
        identity=repr(key(node));atoms[identity]=node
        return {(identity,):1}
    def safe(node):
        if node[0]=='pow_dynamic':return True  # nonnegative exponent, pure operands
        return pure(node) or (node[0] in ('abs','neg','pos','add','sub','mul','pow')
                             and all(safe(child) for child in children(node)))
    def normalize(node):
        node=tuple(normalize(child) if isinstance(child,(tuple,list)) else child for child in node)
        if not safe(node) or node[0] not in ('add','sub','mul','neg','pos'):return node
        terms=[]
        for monomial,coefficient in sorted(ring(node).items()):
            factors=[atoms[atom] for atom in monomial]
            if not factors:value=('constant',coefficient)
            else:
                value=factors[0]
                for factor in factors[1:]:value=('mul',value,factor)
                if coefficient==-1:value=('neg',value)
                elif coefficient!=1:value=('mul',('constant',coefficient),value)
            terms.append(value)
        result=terms[0] if terms else ('constant',0)
        for term in terms[1:]:result=('add',result,term)
        return result
    positive=normalize(tree)
    # Negative powers are floats: retain their original evaluation/rounding
    # order and only count their two integer operand preparations here.
    def floating_placeholder(node):
        if node[0]=='pow_dynamic':return ('floating',0.0)
        return tuple(floating_placeholder(child) if isinstance(child,(tuple,list)) else child for child in node)
    shape(positive);shape(floating_placeholder(tree));shape(base);shape(exponent)
    growth=max(bits for degree,bits in bounds)+8
    degree=max(degree for degree,bits in bounds)
    capacity=(1076+growth+31)//32
    # Round the surrogate width first: all integer polynomial intermediates,
    # including coefficients that later cancel, fit without modular wrapping.
    limbs=max(2,(degree*capacity*32+growth+8+31)//32)
    lines,base_value=boxed(base,arguments,'int64',policy,prefix+'_base',register_error,
        used_math=used_math,_limbs_override=limbs,_raw_integer=True,_native_operand=native_operand)
    source,exponent_value=boxed(exponent,arguments,'int64',policy,prefix+'_exponent',register_error,
        used_math=used_math,_limbs_override=limbs,_raw_integer=True,_native_operand=native_operand)
    lines.extend(source)
    def replace(node,negative):
        if node[0]=='pow_dynamic':
            return (('native_power_float',base_value,exponent_value) if negative else
                    ('native_power_integer',base_value,exponent_value,capacity,capacity*32))
        return tuple(replace(child,negative) if isinstance(child,(tuple,list)) else child for child in node)
    dtype=np.dtype(destination);result=prefix+'_contextual_result'
    ctype='uint' if dtype.kind=='f' and dtype.itemsize==4 else 'bool' if dtype.kind=='b' else 'ulong'
    lines.append(f'{ctype} {result};')
    lines.append(f'if({exponent_value}.negative) {{')
    source,value=boxed(replace(tree,True),arguments,destination,policy,prefix+'_negative',register_error,
        used_math=used_math,_limbs_override=limbs,_native_operand=native_operand)
    lines.extend(source);lines.append(f'{result}={ctype}({value});')
    lines.append('} else {')
    source,value=boxed(replace(positive,False),arguments,destination,policy,prefix+'_positive',register_error,
        used_math=used_math,_limbs_override=limbs,_native_operand=native_operand)
    lines.extend(source);lines.append(f'{result}={ctype}({value});'+'}')
    return lines,result


def fast_render(tree,arguments,destination,policy,prefix,register_error,*,used_math,native_operand=None):
    """Narrow proofs reuse checked numeric dispatch and canonical limb helpers.

    Nonnegative sums/products cannot recover from a magnitude above binary64's
    conversion range. A common-base difference factors exact cancellation before
    that saturation. Every negative-exponent branch keeps Python float order.
    General trees retain the existing precision frontier.
    """
    def children(node):return [child for child in node[1:] if isinstance(child,(tuple,list))]
    def integer(node):
        if node[0] in ('argument','native_c'):return np.dtype(node[2]).kind in 'iub' and node[0]!='native_c'
        if node[0]=='constant':return True
        return node[0] in ('abs','neg','pos','add','sub','mul','pow') and all(integer(child) for child in children(node))
    def width(node):
        if node[0]=='argument':return np.dtype(node[2]).itemsize*8
        if node[0]=='constant':return abs(node[1]).bit_length()
        if node[0] in ('abs','neg','pos'):return width(node[1])
        if node[0]=='pow':return max(1,width(node[1])*node[2])
        a,b=width(node[1]),width(node[2]);return a+b if node[0]=='mul' else max(a,b)+1
    def nonnegative(node):
        op=node[0]
        if op=='argument':return np.dtype(node[2]).kind in 'ub'
        if op=='constant':return node[1]>=0
        if op=='abs':return integer(node[1]) or nonnegative(node[1])
        if op=='pos':return nonnegative(node[1])
        if op=='pow_dynamic':return integer(node[1]) and nonnegative(node[1]) and integer(node[2])
        return op in ('add','mul') and all(nonnegative(child) for child in children(node))
    def key(node):
        if node[0]=='argument':return ('argument',node[2],arguments[node[1]])
        return tuple(key(child) if isinstance(child,(tuple,list)) else child for child in node)
    powers=[]
    def collect(node):
        if node[0]=='pow_dynamic':powers.append(node)
        for child in children(node):collect(child)
    collect(tree)
    common=(tree[0]=='sub' and len(powers)==2 and tree[1][0]==tree[2][0]=='pow_dynamic'
            and key(tree[1][1])==key(tree[2][1])
            and all(integer(child) for node in powers for child in node[1:]))
    positive=len(powers)>1 and nonnegative(tree)
    division=(tree[0] in ('floordiv','truediv') and tree[1][0]=='pow_dynamic'
              and all(integer(child) for child in (*tree[1][1:],tree[2])))
    if not (common or positive or division):return None
    operands=[child for power in powers for child in power[1:]]
    bounds=max([width(child) for child in operands]+[64])
    # Division needs the denominator's entire magnitude in addition to every
    # finite quotient boundary. Two further words hold rounding workspace.
    bits=max(1056,bounds+2,width(tree[2])+1088 if division else 0)
    limbs=(bits+31)//32
    if common:tree=('power_difference',tree[1][1],tree[1][2],tree[2][2])
    from .compiled_python_dynamic import render as dynamic
    return dynamic(tree,arguments,destination,policy,prefix,register_error,
        used_math=used_math,native_operand=native_operand,limbs=limbs,
        bounded_powers=True,bounded_nonnegative=positive)
