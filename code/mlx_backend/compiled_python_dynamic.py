"""Conditional Python numeric types with checked native bigint temporaries.

This is the unresolved dynamic-type extension of the existing boxed adapter.
Static coercion/polynomial proofs retain their smaller qualified path. All
numeric operations here reuse its integer and canonical binary64 helpers.
"""
from dataclasses import dataclass
import errno
import os
import numpy as np
from .compiler import UnsupportedModelError
from .compiled_precision import DEFAULT_LIMBS, ERROR_NAME, ERROR_MESSAGE


@dataclass(frozen=True)
class _Value:
    integer: str
    real: str
    floating: str


def render(tree, arguments, destination, policy, prefix, register_error, *,
           used_math, native_operand=None, limbs=DEFAULT_LIMBS, bounded_powers=False, bounded_nonnegative=False):
    """Emit one left-to-right expression; retry is a private catalog entry."""
    limbs=max(2,int(limbs));type_=f'ci_py<{limbs}>'
    used_math.update(('_compiled_python_integer','_compiled_precision'))
    precision=register_error(ERROR_NAME,ERROR_MESSAGE)
    lines=[];sequence=0

    def error(name,message,predicate):
        code=register_error(name,message)
        return f'if(({predicate}) && !_eon_integer_error)_eon_integer_error={code}u;'

    def temporary():
        nonlocal sequence
        name=f'{prefix}_dynamic_{sequence}';sequence+=1
        value=_Value(name+'_integer',name+'_real',name+'_floating')
        lines.extend([f'{type_} {value.integer}=ci_py_zero<{limbs}>();',
                      f'ulong {value.real}=0UL;',f'bool {value.floating}=false;'])
        return value

    def check_capacity(flag):
        return f'if({flag} && !_eon_integer_error)_eon_integer_error={precision}u;'

    def to_float(value,code):
        nonlocal sequence
        name=f'{prefix}_coerce_{sequence}';sequence+=1
        code.append(f'ulong {name}={value.real};')
        code.append(f'if(!{value.floating} && !_eon_integer_error){{')
        code.append(f'{name}=ci_py_to_double({value.integer});')
        code.append(error('OverflowError','int too large to convert to float',
                          f'({name}&0x7ff0000000000000UL)==0x7ff0000000000000UL'))
        code.append('}')
        return name

    def integer_only(node):
        op=node[0]
        if op=='value':return True
        if op in ('argument','native_c'):return np.dtype(node[2]).kind in 'iub'
        if op=='constant':return True
        if op in ('floating','truediv','pow_dynamic'):return False
        return all(integer_only(child) for child in node[1:] if isinstance(child,(tuple,list)))

    def nonnegative(node):
        if node[0]=='abs':return True
        if node[0] in ('constant','floating'):return node[1]>=0
        if node[0] in ('argument','native_c'):return np.dtype(node[2]).kind in 'ub'
        if node[0]=='pow':return node[2]%2==0 or nonnegative(node[1])
        return False

    def emit(node):
        op=node[0]
        if op=='value':return node[1]
        if op=='power_difference':
            base,left,right=(emit(child) for child in node[1:]);value=temporary()
            lines.append(f'if(!{left.integer}.negative&&!{right.integer}.negative){{')
            delta=value.integer+'_delta';minimum=value.integer+'_minimum';common=value.integer+'_common';residue=value.integer+'_residue'
            lines.extend([f'{type_} {delta}=ci_py_sub({left.integer},{right.integer});',
                f'{type_} {minimum}={delta}.negative?{left.integer}:{right.integer};',
                f'{delta}.negative=false;',
                f'{type_} {common}=ci_py_pow_bounded({base.integer},{minimum});',
                f'{type_} {residue}=ci_py_sub(ci_py_pow_bounded({base.integer},{delta}),ci_py_from<{limbs}>(1UL,false));',
                f'{value.integer}=ci_py_mul_bounded({common},{residue});',
                f'if(ci_py_cmp({left.integer},{right.integer})<0&&!ci_py_iszero({value.integer})){value.integer}.negative=!{value.integer}.negative;',
                '}else{'])
            result=emit(('sub',('pow_dynamic',('value',base),('value',left)),('pow_dynamic',('value',base),('value',right))))
            lines.extend([f'{value.integer}={result.integer};{value.real}={result.real};{value.floating}={result.floating};','}'])
            return value
        if op in ('argument','native_c','constant','floating'):
            value=temporary();code=[]
            if op in ('argument','native_c'):
                dtype=np.dtype(node[2])
                if op=='native_c':
                    if native_operand is None:raise UnsupportedModelError('Deferred Cython operand needs its selected typed graph')
                    source,argument=native_operand(node,arguments,value.real+'_native')
                    code.extend(source)
                else:argument=arguments[node[1]]
                if dtype.kind=='f':
                    raw=f'f_to_d({argument})' if dtype.itemsize==4 else argument
                    code.extend([f'{value.floating}=true;',f'{value.real}={raw};'])
                else:
                    negative=f'(long({argument})<0)' if dtype.kind=='i' else 'false'
                    code.append(f'{value.integer}=ci_py_from<{limbs}>({negative}?0UL-ulong({argument}):ulong({argument}),{negative});')
            elif op=='floating':
                raw=int(np.asarray(node[1],np.float64).view(np.uint64))
                code.extend([f'{value.floating}=true;',f'{value.real}={raw}UL;'])
            else:
                magnitude=abs(node[1])
                if magnitude.bit_length()>limbs*32:code.append(check_capacity('true'))
                else:
                    for j in range(limbs):
                        word=(magnitude>>(32*j))&0xffffffff
                        if word:code.append(f'{value.integer}.limb[{j}]={word}u;')
                    if node[1]<0:code.append(f'{value.integer}.negative=true;')
            lines.extend(['if(!_eon_integer_error){',*code,'}']);return value
        if op in ('pos','neg','abs'):
            argument=emit(node[1])
            if op=='pos':return argument
            value=temporary();code=[f'{value.floating}={argument.floating};',
                f'if({argument.floating}){{',
                f'{value.real}='+ (f'({argument.real}&0x7fffffffffffffffUL);' if op=='abs' else f'({argument.real}^(1UL<<63));'),
                '}else{',f'{value.integer}=ci_py_{op}({argument.integer});','}']
            lines.extend(['if(!_eon_integer_error){',*code,'}']);return value
        if op=='pow':
            # Preserve a literal exponent as an integer operand. Its float
            # conversion, when needed, follows CPython before libm dispatch.
            right_node=('constant',node[2]);left=emit(node[1]);right=emit(right_node)
        else:
            if op not in ('add','sub','mul','mod','floordiv','truediv','pow_dynamic',
                          'gt','ge','lt','le','eq','ne'):
                raise UnsupportedModelError('Dynamic boxed numeric operation needs native '+op+' policy')
            left=emit(node[1]);right=emit(node[2]);right_node=node[2]
        value=temporary();code=[]
        if op in ('gt','ge','lt','le','eq','ne'):
            comparison={'gt':'>','ge':'>=','lt':'<','le':'<=','eq':'==','ne':'!='}[op]
            ordering=value.real+'_ordering'
            code.extend([f'int {ordering}=0;',f'if(!{left.floating}&&!{right.floating}){ordering}=ci_py_cmp({left.integer},{right.integer});',
                f'else if(!{left.floating}){ordering}=ci_py_cmp_double({left.integer},{right.real});',
                f'else if(!{right.floating}){{{ordering}=ci_py_cmp_double({right.integer},{left.real});if({ordering}!=2){ordering}=-{ordering};}}',
                f'else {ordering}=d_is_nan({left.real})||d_is_nan({right.real})?2:d_lt({left.real},{right.real})?-1:d_gt({left.real},{right.real})?1:0;'])
            predicate=f'({ordering}==2?'+('true' if op=='ne' else 'false')+f':{ordering}{comparison}0)'
            code.append(f'{value.integer}=ci_py_from<{limbs}>(ulong({predicate}),false);')
        else:
            power=op in ('pow','pow_dynamic')
            float_condition=f'({left.floating}||{right.floating}'+(f'||{right.integer}.negative' if power else '')+')'
            if op=='truediv':float_condition='true'
            code.extend([f'if({float_condition}){{',f'{value.floating}=true;'])
            if op=='truediv':
                code.append(f'if(!{left.floating}&&!{right.floating}){{')
                code.append(error('ZeroDivisionError','division by zero',f'ci_py_iszero({right.integer})'))
                # Two extra limbs hold the rounding workspace. Input capacity
                # therefore does not fail merely because division needs 53 bits.
                wide=limbs+2;a=value.real+'_numerator';b=value.real+'_denominator'
                code.extend([f'ci_py<{wide}> {a}=ci_py_zero<{wide}>(),{b}=ci_py_zero<{wide}>();',
                    f'for(uint j=0;j<{limbs}u;j++){{{a}.limb[j]={left.integer}.limb[j];{b}.limb[j]={right.integer}.limb[j];}}',
                    f'{a}.negative={left.integer}.negative;{b}.negative={right.integer}.negative;',
                    f'if(!_eon_integer_error){value.real}=ci_py_truediv_double({a},{b});',
                    error('OverflowError','integer division result too large for a float',
                          f'({value.real}&0x7fffffffffffffffUL)==0x7ff0000000000000UL'),'}else{'])
            a,b=to_float(left,code),to_float(right,code)
            if op in ('add','sub','mul'):code.append(f'if(!_eon_integer_error){value.real}=d_{op}({a},{b});')
            elif op in ('truediv','mod','floordiv'):
                message={'truediv':'float division by zero','mod':'float modulo','floordiv':'float floor division by zero'}[op]
                code.append(error('ZeroDivisionError',message,f'd_is_zero({b})'))
                helper='d_div' if op=='truediv' else f'd_operator_{op}_numpy'
                if op!='truediv':used_math.add('operator_remainder_float')
                code.append(f'if(!_eon_integer_error){value.real}={helper}({a},{b});')
                if op=='truediv':code.append('}')
            else:
                # Complex output is a separate pending domain decision. No
                # real-only numerical substitution is made for that case.
                if not nonnegative(node[1]) and not integer_only(right_node):
                    raise UnsupportedModelError('Boxed fractional power may need the pending complex-number policy')
                used_math.add('operator_pow_float')
                code.append(error('ZeroDivisionError','0.0 cannot be raised to a negative power',f'd_is_zero({a}) && d_lt({b},0UL)'))
                code.append(f'if(!_eon_integer_error){value.real}=d_operator_pow({a},{b});')
                overflow_message=str(OverflowError(errno.ERANGE,os.strerror(errno.ERANGE)))
                code.append(error('OverflowError',overflow_message,
                    f'({value.real}&0x7fffffffffffffffUL)==0x7ff0000000000000UL && ({a}&0x7fffffffffffffffUL)<0x7ff0000000000000UL && ({b}&0x7fffffffffffffffUL)<0x7ff0000000000000UL'))
            code.append('}else{')
            if op in ('add','sub','mul','pow','pow_dynamic'):
                if (power and bounded_powers) or (bounded_nonnegative and op in ('add','mul')):
                    helper='pow' if power else op
                    code.append(f'{value.integer}=ci_py_{helper}_bounded({left.integer},{right.integer});')
                else:
                    checked='pow' if power else op;flag=value.real+'_exhausted'
                    code.extend([f'bool {flag}=false;',
                        f'{value.integer}=ci_precision_{checked}({left.integer},{right.integer},{flag});',check_capacity(flag)])
            elif op in ('mod','floordiv'):
                code.append(error('ZeroDivisionError','integer division or modulo by zero',f'ci_py_iszero({right.integer})'))
                code.append(f'if(!_eon_integer_error){value.integer}=ci_py_{op}({left.integer},{right.integer});')
            code.append('}')
        lines.extend(['if(!_eon_integer_error){',*code,'}']);return value

    value=emit(tree);dtype=np.dtype(destination)
    if dtype.kind=='f':
        code=[];raw=to_float(value,code);lines.extend(code)
        return lines,f'f_from_d({raw})' if dtype.itemsize==4 else raw
    if dtype.kind=='b':return lines,f'({value.floating}?!d_is_zero({value.real}):!ci_py_iszero({value.integer}))'
    # Reuse the existing checked C store conversion, not another integer policy.
    from .compiled_python_integer import render as boxed
    raw=prefix+'_store'
    lines.append(f'ulong {raw}=0UL;')
    for real in (True,False):
        lines.append(f'if({value.floating}){{' if real else '}else{')
        if real:
            child=('argument',0,'float64');arguments_=[value.real]
        else:
            # Existing boxed renderer accepts its internal native integer node
            # only for contextual powers, so a narrow direct carrier node is
            # supplied by the integration hook below.
            child=('native_integer',value.integer,limbs*32);arguments_=[]
        source,result=boxed(child,arguments_,destination,policy,prefix+('_real_store' if real else '_integer_store'),
            register_error,used_math=used_math,_limbs_override=limbs)
        lines.extend(source);lines.append(f'{raw}=ulong({result});')
    lines.append('}')
    return lines,raw
