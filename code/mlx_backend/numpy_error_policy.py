"""NumPy ufunc floating status carried by native arithmetic, in source order.

The binary64 result routines remain the qualified shared implementation.  A
private copy of their rounding boundary additionally reports IEEE underflow
and overflow; it does not ask the CPU to evaluate a model operation.
"""
import ast
from dataclasses import replace
from functools import lru_cache
import re
import numpy as np


KINDS = ((1, "divide", "divide by zero"), (2, "over", "overflow"),
         (4, "under", "underflow"), (8, "invalid", "invalid value"))
OPERATORS = {ast.Add: (0, "add"), ast.Sub: (1, "subtract"),
             ast.Mult: (2, "multiply"), ast.Div: (3, "divide"),
             ast.Mod: (4, "remainder"), ast.FloorDiv: (5, "floor_divide"),
             ast.Pow: (6, "power")}


def binary_status_code(renderer,operation,left,right,arguments,result,width):
    op,name=operation
    ordinary=(f'_eon_numpy_binary_status({op}u,{arguments[0]},'
              f'{arguments[1]},{result},{width})')
    if op<4:
        return ordinary
    scalar='true' if numpy_array_exponent(left,right) else 'false'
    original=renderer.convert(right,'float').code
    if width not in ('16u','32u'):
        renderer.used_math.add('_numpy_operator_'+name)
        renderer.used_math.add('operator_pow_float' if op==6 else 'operator_remainder_float')
        exact=(f'_eon_numpy_operator64_status({op}u,{arguments[0]},'
               f'{arguments[1]},{scalar},{original})')
    if width=='64u':return exact
    renderer.used_math.update(('_numpy_operator32_'+name,'_half','_half_f32'))
    renderer.used_math.add('operator_pow_float32' if op==6 else 'operator_remainder_float32')
    narrow=(f'_eon_numpy_operator32_status({op}u,h_d_to_f({arguments[0]}),'
            f'h_d_to_f({arguments[1]}),{scalar},{original})')
    half=(f'_eon_numpy_operator16_status({op}u,ushort(h_from_d({arguments[0]})),'
          f'ushort(h_from_d({arguments[1]})),{scalar},{original})')
    if width=='16u':return half
    if width=='32u':return narrow
    return f'({width}==16u?{half}:{width}==32u?{narrow}:{exact})'


def width_code(narrow,half=False):
    choose=lambda predicate,yes,no:yes if predicate is True else no if predicate is False else f'({predicate}?{yes}:{no})'
    return choose(half,'16u',choose(narrow,'32u','64u'))


def numpy_array_exponent(left,right):
    scalar_operation=(left.scalar and right.scalar and
        not left.array_object and not right.array_object and
        not(left.python_scalar and right.python_scalar))
    return right.scalar and not right.array_object and not scalar_operation


def binary_action_name(name,left,right):
    scalar_operation=(left.scalar and right.scalar and
        not left.array_object and not right.array_object and
        not(left.python_scalar and right.python_scalar))
    if scalar_operation and name=='divide':
        def dtype(value):
            if value.kind=='float':
                half=getattr(value,'half',False)
                if not isinstance(value.narrow,bool) or not isinstance(half,bool):
                    return None
                return np.dtype('float16' if half else 'float32' if value.narrow else 'float64')
            if value.kind=='int':
                return np.dtype(('uint' if value.integer_unsigned else 'int')+str(value.integer_bits))
            return np.dtype('bool') if value.kind=='bool' else None
        types=dtype(left),dtype(right)
        if all(value is not None for value in types):
            # NumPy scalar division falls back to its ufunc when neither
            # NumPy operand can safely accept the other scalar's dtype.
            # Python float / NumPy integer is one such reflected operation.
            scalar_operation=any(not value.python_scalar and np.can_cast(other,own,casting='safe')
                for value,own,other in ((left,types[0],types[1]),(right,types[1],types[0])))
    return 'scalar '+name if scalar_operation else name


def unary_status_value(renderer,name,arg,result,status):
    """Native value/flags with exactly the selected NumPy ufunc width."""
    width=width_code(result.narrow,getattr(result,'half',False))
    raw=renderer.convert(arg,'float').code
    if width not in ('16u','32u'):
        renderer.used_math.add('_numpy_unary_'+name)
        renderer.used_math.add(name)
        regular=f'_eon_numpy_{name}_status({raw},{status})'
    if width=='64u':return regular
    renderer.used_math.update(('_numpy_narrow_'+name,'_half','_half_f32'))
    if name not in ('abs','sign','floor','ceil','sqrt'):
        renderer.used_math.add(name+'_f32')
    single=renderer.convert(arg,'float32').code
    narrow=f'h_f32_to_d(_eon_numpy_{name}32_status({single},{status}))'
    half=(f'({status}|=_eon_numpy_{name}16_flags(ushort(h_from_d({raw}))),'
          f'{result.code})')
    if width=='16u':return half
    if width=='32u':return narrow
    return f'({width}==16u?{half}:{width}==32u?{narrow}:{regular})'


def int32_status_code(renderer,arg,raw):
    renderer.used_math.update(('_numpy_narrow_cast','_half','_half_f32'))
    width=width_code(arg.narrow,getattr(arg,'half',False))
    rank,count=renderer._array_indices(scalar=arg.scalar)
    return f'_eon_numpy_int32_cast_status({raw},{rank},{count},{width})'


def floor_expression(renderer,name,arg):
    from .numpy32 import ufunc_float32_precision
    raw=renderer.convert(arg,'float').code
    return replace(arg,code=f'd_{name}({raw})',kind='float',
                   narrow=ufunc_float32_precision(arg),value=None,python_scalar=False)


def python_int_status(raw):
    return (f'(d_is_nan({raw})?1u:d_exp({raw})==2047ul?2u:'
            f'(d_lt({raw},0xc3e0000000000000ul)||d_ge({raw},0x43e0000000000000ul))?3u:0u)')


def int64_status_code(raw):
    return (f'((d_exp({raw})==2047ul||d_lt({raw},0xc3e0000000000000ul)'
            f'||d_ge({raw},0x43e0000000000000ul))?8u:0u)')


def floating_store_status(renderer,name,expression,destination):
    if expression.kind!='float' or destination.kind not in 'fiu':
        return None
    if destination.kind in 'iu' and expression.scalar and expression.python_scalar:
        # The base finish path already preserves Python scalar conversion
        # exceptions, independently of NumPy's configurable error policy.
        return None
    source=renderer.scalar_store_sources.get(name,expression) if expression.scalar else expression
    if source.kind!='float':source=expression
    width=width_code(source.narrow,getattr(source,'half',False))
    if destination.kind=='f' and ((destination==np.dtype('float32') and width in ('16u','32u'))
            or(destination==np.dtype('float64') and width in ('16u','64u'))):
        return None
    renderer.used_math.update(('_numpy_narrow_cast','_half','_half_f32','_numpy_floating_status'))
    raw=renderer.convert(source,'float').code
    if destination.kind in 'iu':
        original=getattr(renderer,'complex_store_sources',{}).get(name)
        if original is not None:
            width=width_code(original.narrow)
        rank,count=renderer._array_indices(scalar=source.scalar)
        unsigned='true' if destination.kind=='u' else 'false'
        complex_source='true' if original is not None else 'false'
        return (f'_eon_numpy_integer_cast_status({raw},{destination.itemsize*8}u,'
                f'{unsigned},{rank},{count},{width},{complex_source})',source.scalar)
    if destination==np.dtype('float32'):
        # Same-dtype copies and half widening do not execute a narrowing
        # instruction. Value-dependent promotion must select this per row.
        return (f'({width}==64u?_eon_numpy_cast_status({raw},32u):0u)',source.scalar)
    if destination==np.dtype('float64'):
        # The actual float32 widening instruction quiets signaling NaNs;
        # half's software expansion and raw64 copies preserve their bits.
        return (f'({width}==32u&&_eon_status_snan({raw})?8u:0u)',source.scalar)
    return None


class NumpyFloatingStatus(FloatingPointError):
    """Internal native trap; operation-segment replay performs policy delivery."""


def _function(source, name):
    match = re.search(r"\binline ulong " + name + r"\([^)]*\)\s*\{", source)
    if match is None:
        raise RuntimeError("Missing shared arithmetic helper " + name)
    begin = match.end() - 1
    depth = 1
    end = begin + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


def _status_calls(source, names):
    insertions = []
    for match in re.finditer(r"\b(" + "|".join(names) + r")\(", source):
        position = match.end()
        depth = 1
        while depth:
            depth += (source[position] == "(") - (source[position] == ")")
            position += 1
        declaration = source[max(0, match.start()-13):match.start()].endswith("inline ulong ")
        insertions.append((position-1, ", thread uint& _status" if declaration else ", _status"))
    for position, value in reversed(insertions):
        source = source[:position] + value + source[position:]
    return re.sub(r"\b(" + "|".join(names) + r")\b", r"_eon_status_\1", source)


@lru_cache(maxsize=1)
def metal_header():
    from .binary64 import MSL_HEADER
    names = ("d_finish", "d_add", "d_sub", "d_mul", "d_div")
    source = "\n".join(_function(MSL_HEADER, name) for name in names)
    for name in ('d_add','d_sub','d_mul','d_div'):
        begin=source.index('{',source.index('inline ulong '+name+'('))+1
        flags=('if((d_is_nan(a)&&!(a&(1ul<<51))) || '
               '(d_is_nan(b)&&!(b&(1ul<<51)))) _status|=8u;')
        if name in ('d_add','d_sub'):
            flags+=(f'if(d_abs(a)==D_INF && d_abs(b)==D_INF && '
                    f'(((a^b)>>63)=={1 if name=="d_add" else 0}ul)) _status|=8u;')
        elif name=='d_mul':
            flags+=('if((d_is_zero(a)&&d_abs(b)==D_INF) || '
                    '(d_is_zero(b)&&d_abs(a)==D_INF)) _status|=8u;')
        else:
            flags+=('if((d_is_zero(a)&&d_is_zero(b)) || '
                    '(d_abs(a)==D_INF&&d_abs(b)==D_INF)) _status|=8u;'
                    'else if(d_is_zero(b)&&!d_is_zero(a)&&d_exp(a)!=2047ul) _status|=1u;')
        source=source[:begin]+flags+source[begin:]
    source = source.replace("ulong low = m & 7ul; m >>= 3;",
        "ulong low = m & 7ul; bool tiny = e == 0; m >>= 3;")
    source = source.replace("if (e >= 0x7FF) return (s << 63) | D_INF;",
        "if (tiny && low != 0ul) _status |= 4u;\n"
        "if (e >= 0x7FF) { _status |= 2u; return (s << 63) | D_INF; }")
    source = _status_calls(source, names)
    return source + r'''
inline bool _eon_status_snan(ulong x) {
    return d_is_nan(x) && !(x & (1ul << 51));
}
inline uint _eon_numpy_binary_status(uint op, ulong a, ulong b, ulong result, uint width) {
    // A bit-range proof avoids evaluating ordinary arithmetic twice merely
    // because NumPy's default action is warn. For add/sub the finest exact
    // cancellation lattice is 2^(min_exp-1075), still normal in this range.
    // Multiplication/division bounds keep the entire exact result normal and
    // finite. Narrow-loop rounding still needs its separate cast check.
    uint ea=uint(d_exp(a)),eb=uint(d_exp(b));
    if(width==64u && ea!=2047u && eb!=2047u && op<=3u) {
        bool za=d_is_zero(a),zb=d_is_zero(b);
        if(op<=1u && (za||zb || (min(ea,eb)>=54u && max(ea,eb)<=2045u))) return 0u;
        if(op==2u && (za||zb || (ea && eb && ea+eb>=1025u && ea+eb<=3067u))) return 0u;
        if(op==3u && !zb && (za || (ea && eb && int(ea)-int(eb)>=-1020 && int(ea)-int(eb)<=1020))) return 0u;
    }
    uint status = (_eon_status_snan(a) || _eon_status_snan(b)) ? 8u : 0u;
    bool finite_a = d_exp(a) != 2047ul, finite_b = d_exp(b) != 2047ul;
    bool zero_a = d_is_zero(a), zero_b = d_is_zero(b);
    bool nan_a = d_is_nan(a), nan_b = d_is_nan(b);
    ulong exact = result;
    uint rounded_status = 0;
    if (op == 0) exact = _eon_status_d_add(a, b, rounded_status);
    if (op == 1) exact = _eon_status_d_sub(a, b, rounded_status);
    if (op == 2) exact = _eon_status_d_mul(a, b, rounded_status);
    if (op == 3) exact = _eon_status_d_div(a, b, rounded_status);
    if (op <= 3) status |= rounded_status;
    if (!nan_a && !nan_b) {
        bool invalid = (op <= 1 && !finite_a && !finite_b && (((a^b)>>63) != ulong(op==1)))
            || (op == 2 && ((zero_a && !finite_b) || (zero_b && !finite_a)))
            || (op == 3 && ((zero_a && zero_b) || (!finite_a && !finite_b)))
            || (op == 4 && (zero_b || (!finite_a && finite_b)))
            || (op == 5 && ((zero_a && zero_b) || (!finite_a && !finite_b)))
            || (op == 6 && (a>>63) && !zero_a && finite_a && finite_b && !d_eq(b,d_floor(b)));
        if (invalid) status |= 8u;
        if ((op == 3 || op == 5) && zero_b && !zero_a && finite_a) status |= 1u;
        if (op == 6 && zero_a && finite_b && d_lt(b,0ul)) status |= 1u;
        if (finite_a && finite_b && !(status & 9u) && d_exp(result)==2047ul && !d_is_nan(result)) status |= 2u;
        if (op == 6 && finite_a && finite_b && !zero_a && !d_is_zero(b) && !(status & 9u)) {
            if (d_eq(b,0x3ff0000000000000ul)) {}
            else if (d_eq(b,0x4000000000000000ul)) {
                exact = _eon_status_d_mul(a,a,rounded_status); status |= rounded_status;
            } else if (d_eq(b,0xbff0000000000000ul)) {
                exact = _eon_status_d_div(0x3ff0000000000000ul,a,rounded_status); status |= rounded_status;
            } else if (d_exp(result)==0ul) status |= 4u;
        }
        if (width != 64 && op <= 3 && finite_a && finite_b) {
            ulong minimum = width == 32 ? 0x3810000000000000ul : 0x3f10000000000000ul;
            status &= ~(rounded_status & 6u);
            if (!d_is_zero(exact) && d_lt(d_abs(exact),minimum) && !d_eq(exact,result)) status |= 4u;
            if (d_exp(result)==2047ul && !d_is_nan(result) && !(status & 9u)) status |= 2u;
        }
    }
    return status;
}
'''


@lru_cache(maxsize=16)
def policy_renderer(base):
    class PolicyRenderer(base):
        def __init__(self, *args, numpy_error_actions=(), **kwargs):
            super().__init__(*args, **kwargs)
            self.numpy_error_actions = dict(numpy_error_actions)
            self.numpy_status_scope = 'true'

        def evaluate(self,node):
            parts=tuple(ast.walk(node))
            python_only=(not any(isinstance(part,ast.Call) for part in parts)
                and all(type(self._invariant_values().get(part.id)) in (int,float,bool,complex)
                        for part in parts if isinstance(part,ast.Name)))
            if (not python_only and any(isinstance(part,(ast.BinOp,ast.Call)) for part in parts)):
                raise KeyError('NumPy floating actions execute at runtime')
            return super().evaluate(node)

        def _exprel(self,node,arg):
            # Brian slices away its small/large branches before invoking
            # expm1/divide. Their unselected arithmetic cannot raise a policy
            # action merely because the native expression uses a ternary.
            from .binary64 import literal
            previous=self.numpy_status_scope
            value=self.convert(arg,'float').code
            self.numpy_status_scope=(f'({previous} && !d_lt(d_abs({value}),{literal(1e-16)})'
                                     f' && !d_gt({value},{literal(717.)}))')
            try:
                return super()._exprel(node,arg)
            finally:
                self.numpy_status_scope=previous

        def _status_trap(self,name,status):
            self.runtime_errors.append(('NumpyFloatingStatus',name))
            kind=len(self.runtime_errors)
            flag=self.index_error_name
            enabled=sum(bit for bit,key,_ in KINDS
                        if self.numpy_error_actions.get(key,'ignore')!='ignore')
            return (f'if(!{flag} && {self.numpy_status_scope} && ({status}&{enabled}u))'
                    f'{{{flag}_kind={kind};{flag}=true;}}')

        def finish(self):
            lines=super().finish()
            if not any(self.numpy_error_actions.get(key,'ignore')!='ignore' for _,key,_ in KINDS):
                return lines
            result=[]
            for index,line in enumerate(lines):
                info=self.finish_stores.get(index)
                status=floating_store_status(self,*info) if info is not None else None
                if status is not None:
                    result.append(self._status_trap('cast',status[0]))
                result.append(line)
            if self.runtime_errors:
                flag=self.index_error_name
                result=[line if line.startswith(f'if(!{flag}') else f'if(!{flag}){{{line}}}' for line in result]
            return result

        def expression(self,source):
            if not isinstance(source,(str,ast.AST)):
                return super().expression(source)
            node=ast.parse(source,mode='eval').body if isinstance(source,str) else source
            cache=getattr(self,'_numpy_expression_cache',None)
            outer=cache is None
            if outer:self._numpy_expression_cache=cache={}
            # Rendering a call first renders its arguments through the base
            # renderer. Status instrumentation then needs those same operands;
            # reuse their rendered values instead of emitting their actions a
            # second time at every nesting level.
            key=(node,self.numpy_status_scope)
            try:
                if key not in cache:cache[key]=self._policy_expression(node)
                return cache[key]
            finally:
                if outer:self._numpy_expression_cache=None

        def _policy_expression(self,node):
            if (isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                    and node.func.id=='timestep' and len(node.args)==2):
                from .numpy32 import Expression
                # Brian's wrapper casts after this exact offset/add/divide
                # sequence and returns a Python int for a scalar result.
                offset=self._binary(self.expression('0.001'),self.expression(node.args[1]),ast.Mult())
                rounded=self._binary(self._binary(self.expression(node.args[0]),offset,ast.Add()),
                                     self.expression(node.args[1]),ast.Div())
                raw=self.convert(rounded,'float').code
                if rounded.python_scalar:
                    check=self._python_store_check(rounded,np.dtype('int64'),())
                    if check:self.prelude.append(check)
                elif any(self.numpy_error_actions.get(key,'ignore')!='ignore' for _,key,_ in KINDS):
                    self.prelude.append(self._status_trap('cast',int64_status_code(raw)))
                return Expression(f'd_to_int({raw})','int',rounded.scalar,
                                  integer_bits=64,python_scalar=rounded.scalar)
            result=super().expression(node)
            from .numpy_math_status import ROOTS
            if (isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                    and node.func.id in ('int','floor','ceil') and len(node.args)==1):
                name=node.func.id
                arg=self.expression(node.args[0])
                if arg.kind=='float' or name in ('floor','ceil') and arg.kind in ('int','bool'):
                    if name=='int' and arg.python_scalar:
                        check=self._python_store_check(arg,np.dtype('int32'),())
                        if check:self.prelude.append(check)
                        return result
                    if not any(self.numpy_error_actions.get(key,'ignore')!='ignore' for _,key,_ in KINDS):
                        return result
                    operand=self.convert(arg,'float').code
                    cast_arg=arg
                    if name in ('floor','ceil'):
                        temporary=f'_eon_numpy_floor_result_{self.temporary_index}'
                        self.temporary_index+=1
                        status=temporary+'_flags'
                        cast_arg=floor_expression(self,name,arg)
                        code=unary_status_value(self,name,arg,cast_arg,status)
                        self.prelude.extend((f'uint {status}=0u;',f'const ulong {temporary}={code};',
                                             self._status_trap(name,status)))
                        operand=temporary
                    self.used_math.add('_numpy_floating_status')
                    self.prelude.append(self._status_trap('cast',int32_status_code(self,cast_arg,operand)))
                    return replace(result,code=self._numpy_int32(cast_arg,operand))
            if (not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name)
                    or node.func.id not in ROOTS or len(node.args)!=1
                    or result.kind!='float' or result.value is not None
                    or not any(self.numpy_error_actions.get(key,'ignore')!='ignore'
                               for _,key,_ in KINDS)):
                return result
            name=node.func.id
            arg=self.expression(node.args[0])
            temporary=f'_eon_numpy_unary_result_{self.temporary_index}'
            self.temporary_index+=1
            status=temporary+'_flags'
            self.used_math.add('_numpy_floating_status')
            code=unary_status_value(self,name,arg,result,status)
            self.prelude.extend((f'uint {status}=0u;',
                f'const ulong {temporary}={code};'))
            self.prelude.append(self._status_trap(name,status))
            return replace(result,code=temporary)

        def _binary(self, left, right, operation):
            result = super()._binary(left, right, operation)
            active = tuple((bit, key, text) for bit,key,text in KINDS
                           if self.numpy_error_actions.get(key,"ignore") != "ignore")
            if (not active or type(operation) not in OPERATORS or result.kind != "float"
                    or result.python_scalar or result.value is not None):
                return result
            op, name = OPERATORS[type(operation)]
            narrow = result.narrow
            half = getattr(result, "half", False)
            arguments = [(self._precision_operand(arg,narrow,half)
                          if hasattr(self,'_precision_operand') else self._operand(arg,narrow))
                         for arg in (left,right)]
            choose = lambda predicate, yes, no: yes if predicate is True else no if predicate is False else f"({predicate}?{yes}:{no})"
            width = choose(half, "16u", choose(narrow, "32u", "64u"))
            while True:
                temporary = f"_eon_numpy_status_result_{self.temporary_index}"
                self.temporary_index += 1
                if temporary not in self.types and temporary not in self.values:
                    break
            self.used_math.add("_numpy_floating_status")
            self.prelude.append(f"const ulong {temporary} = {result.code};")
            status = temporary + "_flags"
            code=binary_status_code(self,(op,name),left,right,arguments,temporary,width)
            self.prelude.append(f"const uint {status} = {code};")
            self.prelude.append(self._status_trap(binary_action_name(name,left,right),status))
            return replace(result, code=temporary)
    return PolicyRenderer
