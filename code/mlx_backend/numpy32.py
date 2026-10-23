"""NumPy's vector/scalar promotion rules for Brian float32 statements.

Expressions carry exact binary64 bits and a binary32 precision predicate.
This also handles value-dependent promotion of scalar clocks. State remains
binary32; NumPy temporaries retain the dtype produced by their expression.
"""

import ast
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

from .compiler import _Expression, _Renderer, _TimedArray, _evaluate, _kind, _literal, _PREP_FUNCTIONS, UnsupportedModelError


# These ufuncs share width selection and raw32-to-raw64 result transport.
# Arctan retains its distinct numeric-cast path below.
from .unary_math_policy import UFUNC_HELPERS as _UFUNC_HELPERS


@dataclass(frozen=True)
class Expression(_Expression):
    scalar: bool = False
    narrow: bool | str = False
    value: object = None
    integer_bits: int = 64
    integer_unsigned: bool = field(default=False, kw_only=True)
    python_scalar: bool = field(default=False, kw_only=True)
    array_object: bool = field(default=False, kw_only=True)


def _and(predicates):
    if False in predicates:
        return False
    dynamic = list(dict.fromkeys(item for item in predicates if item is not True))
    return "(" + " && ".join(dynamic) + ")" if dynamic else True


def _choose(predicate, yes, no):
    if predicate is True:
        return yes
    if predicate is False:
        return no
    return f"({predicate} ? {yes} : {no})"


def _round(code):
    return f"f_to_d(f_from_d({code}))"


def ufunc_float32_precision(expression):
    """NumPy ufunc floating loop after the registered half loop is handled."""
    if expression.kind=="float":return expression.narrow
    return expression.kind=="int" and expression.integer_bits<=16


def basic_wrapper_profile():
    """Read the selected Brian NumPy wrappers without registering callbacks."""
    import brian2 as b
    from brian2.core.functions import DEFAULT_FUNCTIONS
    from brian2.codegen.generators.numpy_generator import NumpyCodeGenerator
    result=[]
    for name in ('exp','log','expm1','log1p','exprel'):
        function=DEFAULT_FUNCTIONS[name]
        try:code=function.implementations[NumpyCodeGenerator].get_code(None)
        except KeyError:
            if not b.prefs.codegen.runtime.numpy.discard_units:
                result.append((name,True));continue
            code=getattr(function.pyfunc,'_orig_func',function.pyfunc)
        with np.errstate(all='ignore'):value=code(np.float32(0.))
        result.append((name,isinstance(value,np.ndarray)))
    return tuple(result)


@lru_cache(maxsize=1)
def _scalar_boundary():
    # Derive the exact installed NumPy boundary instead of assuming FLT_MAX:
    # NumPy 1.26 uses a deliberately conservative min_scalar_type boundary.
    low, high = 0, int(np.float64(np.finfo(np.float32).max).view(np.uint64))
    vector = np.empty(1, np.float32)
    while low < high:
        mid = (low+high+1)//2
        if np.result_type(vector, np.uint64(mid).view(np.float64)) == np.float32:
            low = mid
        else:
            high = mid-1
    return _literal(np.uint64(low).view(np.float64), "float")


@lru_cache(maxsize=1)
def integer_cast_block():
    """Capture the installed NumPy's integer64 cast loop, including its tail."""
    midpoint=np.int64(2**62+2**38+1)
    direct=np.float32(midpoint).view(np.uint32)
    block=0
    for size in range(1,65):
        bits=np.full(size,midpoint,np.int64).astype(np.float32).view(np.uint32)
        if np.any(bits!=direct):
            block=size
            break
    if block:
        for size in (block,block+1,2*block-1,2*block,2*block+1):
            for value in (midpoint,-midpoint):
                raw=np.full(size,value,np.int64)
                expected=np.full(size,np.float32(value),np.float32)
                prefix=size//block*block
                expected[:prefix]=raw[:prefix].astype(np.float64).astype(np.float32)
                if not np.array_equal(raw.astype(np.float32).view(np.uint32),expected.view(np.uint32)):
                    raise UnsupportedModelError('Unrecognized NumPy integer64-to-float32 cast loop')
    return block


class Renderer(_Renderer):
    def __init__(self, types, values, persistent, scalars, stage, cast_block=0, *, wrapper_arrays=None, store_order=(), runtime_error_writes=None, **kwargs):
        super().__init__(types, values, **kwargs)
        self.persistent = persistent
        self.scalars = scalars
        self.temporaries = {}
        self.prologue = []
        self.shadows = {}
        self.storage_versions = {}
        self.written = set()
        self.stage = stage
        self.collective_reset=False
        self.cast_block=cast_block
        self.store_order=tuple(store_order)
        self.runtime_error_writes=runtime_error_writes if runtime_error_writes is not None else {}
        self.empty_store_checks=[]
        self.scalar_store_sources={}
        self.wrapper_arrays=dict(basic_wrapper_profile() if wrapper_arrays is None else wrapper_arrays)

    def evaluate(self,node):
        return _evaluate(node,self._invariant_values(),'numpy',numpy_objects=True,numpy_wrappers=self.wrapper_arrays)

    def _name(self, name):
        if name in self.temporaries:
            return self.temporaries[name]
        kind = _kind(self.types[name])
        if kind == "float32":
            self.used_math.add('_half_f32')
            return Expression(f"h_f32_to_d({self.code_name(name)})", "float", name in self.scalars, True)
        return Expression(self.code_name(name), kind, name in self.scalars,
                          integer_bits=np.dtype(self.types[name]).itemsize*8,
                          integer_unsigned=np.dtype(self.types[name]).kind=='u')

    def _timed_result(self, code, args):
        return Expression(code, 'float', all(arg.scalar for arg in args))

    def convert(self, expression, kind):
        if (kind == 'bool' and expression.scalar and expression.value is not None
                and np.iscomplexobj(expression.value)):
            return _Expression('true' if bool(expression.value) else 'false', 'bool')
        if (expression.kind == 'int' and getattr(expression, 'integer_unsigned', False)
                and expression.integer_bits == 64 and kind == 'float'):
            return _Expression(f'd_from_uint((ulong)({expression.code}))', 'float')
        if kind=='float32' and expression.kind=='float':
            narrow=getattr(expression,'narrow',False)
            if narrow is not False:
                self.used_math.add('_half_f32')
                return _Expression(_choose(narrow,f'h_d_to_f({expression.code})',
                    f'f_from_d({expression.code})'),'float32')
        return super().convert(expression,kind)

    def _shadow(self, name, *, expression=None, replace=False):
        if name not in self.shadows or replace:
            current = self._name(name) if expression is None else expression
            version = self.storage_versions.get(name, 0)
            self.storage_versions[name] = version + 1
            local = f"_eon_numpy_{self.code_name(name)}" + (f"_{version}" if version else "")
            encoded = self._storage_zero(current) if expression is not None else self._storage_encode(current)
            self.prologue.append(f"{self._storage_type(current)} {local} = {encoded};")
            self.shadows[name] = local
            self._bind(name, current)

    def _bind(self, name, expression):
        local = self.shadows[name]
        self.temporaries[name] = self._storage_expression(local, expression)

    def _storage_type(self, expression):
        return 'ulong'

    def _storage_zero(self, expression):
        return '0ul'

    def _storage_encode(self, expression):
        return expression.code if expression.kind == 'float' else f'(ulong)({expression.code})'

    def _temporary_type(self, expression):
        return {"float": "ulong", "int": "ulong" if expression.integer_unsigned else "long", "bool": "bool"}[expression.kind]

    def _requires_typed_storage(self,expression):
        return False

    def _storage_expression(self, local, expression):
        integer = f"({'ulong' if expression.integer_unsigned else 'long'})({local})"
        code = (local if expression.kind == "float" else integer if expression.kind == "int"
                else f"({local} != 0ul)")
        return Expression(code, expression.kind, expression.scalar, expression.narrow,
                                            None if expression.array_object else expression.value, expression.integer_bits,
                                            integer_unsigned=expression.integer_unsigned,
                                            array_object=expression.array_object,python_scalar=expression.python_scalar)

    @staticmethod
    def _integer_dtype(args):
        operands = [arg.value if arg.value is not None else
                    np.empty(() if arg.scalar else 1, np.dtype('bool') if arg.kind=='bool' else
                        np.dtype(('uint' if arg.integer_unsigned else 'int')+str(arg.integer_bits)))
                    for arg in args]
        return np.result_type(*operands)

    @staticmethod
    def _integer_bits(args):
        return Renderer._integer_dtype(args).itemsize*8

    @staticmethod
    def _integer_round(code, bits, unsigned=False):
        # NumPy evaluates narrow integer ufuncs at their selected width,
        # before an enclosing floating ufunc chooses its own input loop.
        casts={8:'uchar' if unsigned else 'char',16:'ushort' if unsigned else 'short',
               32:'uint' if unsigned else 'int'}
        return (f"(long)({casts[bits]})({code})" if bits in casts else
                f"({'ulong' if unsigned else 'long'})({code})")

    def _python_store_check(self, expression, destination, committed):
        if expression.value is not None:
            try:
                probe=np.empty(1,destination)
                probe[0]=expression.value
            except (ValueError,OverflowError,TypeError) as error:
                errors=[('true',(type(error).__name__,str(error)))]
            else:return ''
        else:
            code=expression.code
            upper=2.**64 if destination.kind=='u' and destination.itemsize>=4 else 2.**63
            errors=[(f'd_is_nan({code})',('ValueError','cannot convert float NaN to integer')),
                    (f'd_exp({code})==2047ul',('OverflowError','cannot convert float infinity to integer')),
                    (f'(d_lt({code},{_literal(-2.**63,"float")}) || !d_lt({code},{_literal(upper,"float")}))',
                     ('OverflowError','Python int too large to convert to C long'))]
        checks=[]
        for condition,error in errors:
            self.runtime_errors.append(error)
            kind=len(self.runtime_errors)
            self.runtime_error_writes[kind]=tuple(committed)
            flag=self.index_error_name
            checks.append(f'if(!{flag} && ({condition})){{{flag}_kind={kind};{flag}=true;}}')
        check=''.join(checks)
        if self.stage=='reset':
            self.collective_reset=True
            self.empty_store_checks.append(check)
        return check

    def _array_indices(self, scalar=False):
        if scalar:
            return '0L', '0L'
        if self.stage == 'reset':
            self.collective_reset = True
            return '_numpy_reset_rank', '_numpy_reset_count'
        return 'i', 'N'

    def finish(self):
        lines=[];committed=[]
        self.finish_stores={}
        order=tuple(name for name in self.store_order if name in self.shadows)
        order+=tuple(name for name in self.shadows if name not in order)
        for name in order:
            if name not in self.persistent or self.stage == 'reset' and name not in self.written:
                # Reset locals are indexed copies. An alias can mutate that
                # copy without writing its read-only source array back.
                continue
            expression=self._name(name)
            target=_kind(self.types[name])
            code=self.convert(expression,target).code
            if target == 'int' and expression.kind == 'float':
                destination = np.dtype(self.types[name])
                unsigned = destination.kind == 'u' and destination.itemsize >= 4
                function = 'n_to_unsigned' if unsigned else 'n_to_signed'
                half = getattr(expression, 'half', False)
                unsigned16 = destination == np.dtype('uint16') and half is not True
                vector_cast = (not expression.scalar and destination.itemsize < 8
                               and (expression.narrow is not True or unsigned16))
                rank, count = self._array_indices(scalar=not vector_cast)
                # Brian delays ordinary local rebinding conversions until
                # the persistent array store. The actual selected reset
                # vector determines NumPy's SIMD prefix and scalar tail.
                from .compiler import _metal_type
                wide = f'{function}({expression.code}, {destination.itemsize*8}, {rank}, {count})'
                # Float32/half loops convert at the destination's arithmetic
                # width. A float64/int64 intermediate changes saturation
                # before the final narrowing, even for finite values.
                narrow_function = ('d_to_uint_sat' if destination.itemsize == 8 else 'd_to_uint32_sat') if unsigned else (
                    'd_to_int' if destination.itemsize == 8 else 'd_to_int32_sat')
                narrow = f'{narrow_function}({expression.code})'
                if unsigned16 and not expression.scalar:
                    # NumPy's float32->uint16 loop uses unsigned conversion
                    # for a 16-element SIMD prefix and signed conversion for
                    # its scalar tail. The half loop stays signed throughout.
                    vector = (f'(({rank}) < (({count})/16L)*16L ? '
                              f'd_to_uint32_sat({expression.code}) : {narrow})')
                    narrow = _choose(half, narrow, vector)
                code = f'({_metal_type(destination)})({_choose(expression.narrow,narrow,wide)})'
                if expression.scalar and expression.python_scalar:
                    scalar_cast='d_to_uint_sat' if unsigned else 'd_to_int'
                    code=f'({_metal_type(destination)})({scalar_cast}({expression.code}))'
                    if unsigned:
                        code=f'({_metal_type(destination)})(d_lt({expression.code},0ul) ? ulong(d_to_int({expression.code})) : d_to_uint_sat({expression.code}))'
                if expression.scalar and expression.python_scalar:
                    check=self._python_store_check(expression,destination,committed)
                    if check:lines.append(check)
            if (target in ('float','float32') and expression.python_scalar
                    and expression.value is not None and type(expression.value) is complex):
                check=self._python_store_check(expression,np.dtype(self.types[name]),committed)
                if check:lines.append(check)
            if target=='float' and expression.kind=='float':
                # The raw64 carrier is representation, not a NumPy cast.
                # A genuine float32-array -> float64-array store quiets an
                # sNaN; an exact half -> float64 cast preserves its payload.
                half=getattr(expression,'half',False)
                widen32=_and([expression.narrow,False if half is True else
                    True if half is False else f'!({half})'])
                if widen32 is not False:
                    self.used_math.add('_half_f32')
                    code=_choose(widen32,f'f_to_d(h_d_to_f({code}))',code)
            if (target=='float32' and expression.kind=='int' and not expression.scalar
                    and expression.integer_bits==64 and self.cast_block):
                block=self.cast_block
                rank,count=self._array_indices()
                code=f'(({rank}) < (({count})/{block}L)*{block}L ? f_from_d(d_from_int((long)({expression.code}))) : {code})'
            self.finish_stores[len(lines)]=(name,expression,np.dtype(self.types[name]))
            lines.append(f'{self.code_name(name)} = {code};')
            committed.append(name)
        if self.runtime_errors:
            flag=self.index_error_name
            lines=[line if line.startswith(f'if(!{flag}') else f'if(!{flag}){{{line}}}' for line in lines]
        return lines

    def _promotion(self, args):
        if all(arg.scalar for arg in args):
            return _and([arg.narrow if arg.kind == "float" else
                         arg.kind == "bool" or arg.integer_bits <= 16 for arg in args])
        if all(arg.scalar or arg.kind == "bool" for arg in args):
            # A Boolean array does not supply a floating-point dtype for
            # NumPy's value-based scalar promotion. In particular bool[:] +
            # float64(.1) is float64, even though .1 fits in float32. Preserve
            # the scalar's actual width until a floating array anchors it.
            return _and([arg.narrow if arg.kind == "float" else False
                         for arg in args if arg.kind != "bool"])
        predicates = []
        floating_array = any(arg.kind == "float" and not arg.scalar for arg in args)
        for arg in args:
            if arg.kind == "bool":
                continue
            if not arg.scalar or not floating_array:
                # Integer arrays up to 16 bits fit NumPy's float32 loop.
                # Without a floating array, scalars retain their own dtype;
                # value-based weak-scalar promotion needs that array anchor.
                predicates.append(arg.narrow if arg.kind == "float" else arg.integer_bits <= 16)
            elif arg.value is not None:
                predicates.append(np.result_type(np.empty(1, np.float32), arg.value) == np.float32)
            elif arg.kind == "float":
                predicates.append(f"(d_le(d_abs({arg.code}), {_scalar_boundary()}) || d_exp({arg.code}) == 2047ul)")
            else:
                predicates.append(f"(({arg.code}) >= -32768L && ({arg.code}) <= 65535L)")
        return _and(predicates)

    def _operand(self, arg, narrow):
        value = self.convert(arg, "float").code
        return _choose(narrow, _round(value), value)

    def _numpy_int32(self,arg,code):
        # Reuse the owner-qualified scalar/vector cast primitive. A half or
        # single source uses FCVTZS32 throughout; binary64 arrays have an
        # eight-element FCVTZS64+narrow prefix and a scalar32 tail.
        if arg.scalar:return f"d_to_int32_sat({code})"
        if arg.kind=="float":
            half=getattr(arg,'half',False)
            bits=_choose(half,"16",_choose(arg.narrow,"32","64"))
            may_double=half is not True and arg.narrow is not True
        else:
            bits="16" if arg.kind=="bool" or arg.integer_bits<=8 else "32" if arg.integer_bits<=16 else "64"
            may_double=bits=="64"
        rank,count=self._array_indices(scalar=not may_double)
        return f"n_to_int32({code}, {rank}, {count}, {bits})"

    def _binary(self, left, right, operation):
        if any(arg.value is not None and np.iscomplexobj(arg.value) for arg in (left,right)):
            if left.value is not None and right.value is not None:
                import operator
                function={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,
                          ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv,
                          ast.Mod:operator.mod,ast.Pow:operator.pow}[type(operation)]
                return self._constant_expression(function(left.value,right.value))
            raise UnsupportedModelError('Native state-dependent complex arithmetic is not implemented')
        result=self._binary_value(left,right,operation)
        if left.scalar and right.scalar and left.python_scalar and right.python_scalar:
            from dataclasses import replace
            value=None
            if left.value is not None and right.value is not None:
                import operator
                function={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,
                          ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv,
                          ast.Mod:operator.mod,ast.Pow:operator.pow}.get(type(operation))
                if function is not None:
                    try:value=function(left.value,right.value)
                    except (ArithmeticError,ValueError):pass
            result=replace(result,python_scalar=True,value=value)
        return result

    def _binary_value(self, left, right, operation):
        if isinstance(operation, (ast.Pow, ast.Mod, ast.FloorDiv)) and 'float' in (left.kind, right.kind):
            from .numpy_half_operators import render_registered_operator
            from .operator_policy import render_numpy_operator
            node = ast.BinOp(ast.Name('_left', ast.Load()), operation, ast.Name('_right', ast.Load()))
            return render_registered_operator(self, node, left, right, render_numpy_operator)
        if left.kind != "float" and right.kind != "float" and not isinstance(operation, ast.Div):
            symbols = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}
            if type(operation) not in symbols and not isinstance(operation, (ast.FloorDiv, ast.Mod, ast.Pow)):
                raise UnsupportedModelError(f"Native dynamic operator {type(operation).__name__} is not implemented")
            dtype = self._integer_dtype((left, right))
            if dtype.kind == 'f':
                # NumPy promotes mixed int64/uint64 arrays to binary64.
                # Preserve each operand's signedness before that conversion.
                if isinstance(operation, (ast.FloorDiv, ast.Mod, ast.Pow)):
                    from .numpy_half_operators import render_registered_operator
                    from .operator_policy import render_numpy_operator
                    node = ast.BinOp(ast.Name('_left', ast.Load()), operation, ast.Name('_right', ast.Load()))
                    return render_registered_operator(self, node, left, right, render_numpy_operator)
                operation_name = {ast.Add: 'add', ast.Sub: 'sub', ast.Mult: 'mul'}[type(operation)]
                a, b = (self.convert(arg, 'float').code for arg in (left, right))
                return Expression(f'd_{operation_name}({a}, {b})', 'float', left.scalar and right.scalar)
            bits,unsigned=dtype.itemsize*8,dtype.kind=='u'
            if isinstance(operation, ast.Pow):
                a, b = (self._integer_round(arg.code, bits, unsigned) for arg in (left, right))
                code = f'n_integer_pow((ulong)({a}), (ulong)({b}))'
                nonnegative_constant = right.scalar and right.value is not None and int(right.value) >= 0
                if not unsigned and not nonnegative_constant:
                    error = ('ValueError', 'Integers to negative integer powers are not allowed.')
                    self.runtime_errors.append(error)
                    kind = len(self.runtime_errors)
                    flag = self.index_error_name
                    if flag is None:
                        raise UnsupportedModelError('Integer power requires a native error flag')
                    code = (f'(({b}) < 0l ? (({flag} ? 0 : ({flag}_kind = {kind})), '
                            f'{flag} = true, 0ul) : {code})')
                    # Materialize checked operations in Python evaluation
                    # order, before later eager calls or C++ conditionals.
                    while True:
                        temporary = f'_eon_integer_result_{self.temporary_index}'
                        self.temporary_index += 1
                        if temporary not in self.types and temporary not in self.values: break
                    self.types[temporary] = dtype
                    self.prelude.append(f'const long {temporary} = {self._integer_round(code,bits,unsigned)};')
                    code = temporary
                return Expression(self._integer_round(code,bits,unsigned), 'int',
                    left.scalar and right.scalar,integer_bits=bits,integer_unsigned=unsigned)
            if isinstance(operation, (ast.FloorDiv, ast.Mod)):
                # Cast operands to the selected ufunc loop before dividing;
                # result narrowing alone cannot reproduce integer promotion.
                a, b = (self._integer_round(arg.code, bits, unsigned) for arg in (left, right))
                if unsigned:
                    symbol = '/' if isinstance(operation, ast.FloorDiv) else '%'
                    code = f'(({b}) == 0ul ? 0ul : (ulong)({a}) {symbol} (ulong)({b}))'
                else:
                    function = 'n_integer_floor_div' if isinstance(operation, ast.FloorDiv) else 'n_integer_mod'
                    code = f'{function}((long)({a}), (long)({b}))'
                return Expression(self._integer_round(code, bits, unsigned), 'int',
                    left.scalar and right.scalar, integer_bits=bits, integer_unsigned=unsigned)
            if dtype.kind=='b' and isinstance(operation,(ast.Add,ast.Mult)):
                # NumPy's Boolean add/multiply are eager OR/AND ufuncs,
                # retaining Boolean dtype for subsequent promotion.
                symbol='|' if isinstance(operation,ast.Add) else '&'
                return Expression(f"bool(({left.code}) {symbol} ({right.code}))","bool",
                    left.scalar and right.scalar,integer_bits=bits)
            # NumPy integer ufuncs wrap. Unsigned intermediates make that
            # defined even for signed64 extrema under Metal optimisation.
            code = self._integer_round(f"((ulong)({left.code}) {symbols[type(operation)]} (ulong)({right.code}))", bits,unsigned)
            return Expression(code, "int", left.scalar and right.scalar, integer_bits=bits,
                              integer_unsigned=unsigned)
        narrow = self._promotion((left, right)) if "float" in (left.kind, right.kind) else False
        names = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}
        if type(operation) not in names:
            raise UnsupportedModelError(f"Native dynamic operator {type(operation).__name__} is not implemented")
        if left.kind == right.kind == "float" and not left.scalar and not right.scalar:
            # Recover existing float32 words without a numeric conversion.
            # Scalar NaN priority also depends on NumPy's actual loop/alias
            # context and is not inferred by this vector-only path.
            from .numpy_arithmetic import render_numpy_arithmetic
            code, dependencies = render_numpy_arithmetic(operation, left.code, right.code, narrow,
                left_narrow=left.narrow, right_narrow=right.narrow)
            self.used_math.update(dependencies)
            return Expression(code, "float", False, narrow)
        a, b = (self._operand(arg, narrow) for arg in (left, right))
        code = f"d_{names[type(operation)]}({a}, {b})"
        return Expression(_choose(narrow, _round(code), code), "float", left.scalar and right.scalar, narrow)

    def _select(self, condition, a, b):
        if any(arg.value is not None and np.iscomplexobj(arg.value) for arg in (a,b)):
            raise UnsupportedModelError('Native state-dependent complex selection is not implemented')
        scalar = condition.scalar and a.scalar and b.scalar
        condition = self.convert(condition, "bool").code
        if "float" not in (a.kind, b.kind):
            dtype = self._integer_dtype((a, b)) if "int" in (a.kind, b.kind) else np.dtype('int64')
            return Expression(f"({condition} ? {a.code} : {b.code})", a.kind, scalar,
                              integer_bits=dtype.itemsize*8,integer_unsigned=dtype.kind=='u')
        narrow = self._promotion((a, b))
        return Expression(f"({condition} ? {self._operand(a, narrow)} : {self._operand(b, narrow)})",
                          "float", scalar, narrow)

    def _invariant_values(self):
        # A later vector rebind invalidates any invariant with the same name.
        # Its old complex (or real) scalar must not survive in preparation.
        return {**{name:value for name,value in self.values.items() if name not in self.temporaries},
            **{name:value.value for name,value in self.temporaries.items()
               if value.scalar and value.value is not None}}

    def _constant_expression(self,value):
        from .compiler import _numpy_scalar_carrier
        carrier=_numpy_scalar_carrier(value)
        kind = _kind(np.asarray(carrier).dtype)
        narrow=kind=='float32'
        if narrow:
            self.used_math.add('_half_f32')
            code=f'h_f32_to_d({_literal(carrier, "float32")})'
            kind='float'
        else:code=_literal(carrier,kind)
        return Expression(code,kind,True,narrow,value,np.asarray(carrier).itemsize*8,
            integer_unsigned=np.asarray(carrier).dtype.kind=='u',
            python_scalar=type(value) in (int,float,bool,complex),array_object=isinstance(value,np.ndarray))

    def _exprel(self,node,arg):
        if arg.kind=='bool' and not arg.scalar:
            return Expression('true','bool',False)
        numerator=self.expression(ast.Call(ast.Name('expm1',ast.Load()),[node.args[0]],[]))
        ratio=self._binary(numerator,arg,ast.Div())
        if arg.kind=='int':
            if arg.integer_unsigned:absolute=arg
            else:
                minimum=-(1<<(arg.integer_bits-1))
                literal=f'({minimum+1}L-1L)' if arg.integer_bits==64 else f'{minimum}L'
                code=f'(({arg.code})=={literal} ? ({arg.code}) : (({arg.code})<0L ? -({arg.code}) : ({arg.code})))'
                absolute=replace(arg,code=code,value=None,array_object=False)
        elif arg.kind=='bool':absolute=arg
        else:absolute=replace(arg,code=f'd_abs({arg.code})',value=None,array_object=False)
        def compare(left,right,operation):
            narrow=self._promotion((left,right))
            half=self._half_promotion((left,right)) if hasattr(self,'_half_promotion') else False
            def operand(value):
                return self._precision_operand(value,narrow,half) if hasattr(self,'_precision_operand') else self._operand(value,narrow)
            return f'd_{operation}({operand(left)},{operand(right)})'
        small=compare(absolute,self.expression('1e-16'),'lt')
        big=compare(arg,self.expression('717'),'gt')
        code=f'({small} ? {_literal(1.,"float")} : ({big} ? {_literal(np.inf,"float")} : {ratio.code}))'
        narrow=ratio.narrow;half=getattr(ratio,'half',False)
        if arg.scalar:
            narrow=_and([narrow,f'(!{small} && !{big})'])
            half=_and([half,f'(!{small} && !{big})'])
        elif arg.kind=='int':narrow,half=False,False
        flags={'half':half} if hasattr(ratio,'half') else {}
        return replace(ratio,code=code,narrow=narrow,value=None,
                       array_object=self.wrapper_arrays.get('exprel',False),**flags)

    def expression(self, source):
        if isinstance(source, Expression):
            return source
        node = ast.parse(source, mode="eval").body if isinstance(source, str) else source
        if self.expression_random is not None and isinstance(node,ast.Call)and isinstance(node.func,ast.Name)and node.func.id==self.expression_random.function:
            return Expression(self.expression_random.result,'float',False,False)
        try:
            value = self.evaluate(node)
        except KeyError:
            pass
        else:
            return self._constant_expression(value)
        if isinstance(node, ast.Name):
            return self._name(node.id)
        if isinstance(node, ast.Compare):
            args=[self.expression(item) for item in (node.left,*node.comparators)]
            if any(arg.value is not None and np.iscomplexobj(arg.value) for arg in args):
                raise UnsupportedModelError('Native state-dependent complex comparison is not implemented')
        if isinstance(node, ast.BoolOp):
            operands = [self.expression(item) for item in node.values]
            symbol = ' & ' if isinstance(node.op, ast.And) else ' | '
            # Brian's NumPy renderer uses eager Boolean ufuncs. Preserve
            # scalar provenance so loop-invariant Boolean temporaries are
            # declared before their use, rather than as vector-array shadows.
            code = symbol.join(f'({self.convert(item, "bool").code})' for item in operands)
            return Expression(f'bool({code})', 'bool', all(item.scalar for item in operands))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = self.expression(node.operand)
            return Expression(f'(!{self.convert(value, "bool").code})', 'bool', value.scalar)
        if isinstance(node, ast.BinOp):
            return self._binary(self.expression(node.left), self.expression(node.right), node.op)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self.expression(node.operand)
            code = value.code if isinstance(node.op, ast.UAdd) else (
                f"d_neg({value.code})" if value.kind == "float" else f"(0ul-(ulong)({value.code}))")
            if value.kind == "int":
                code = self._integer_round(code, value.integer_bits,value.integer_unsigned)
            folded=(value.value if isinstance(node.op,ast.UAdd) else -value.value) if value.value is not None else None
            return Expression(code, value.kind, value.scalar, value.narrow,folded,
                              integer_bits=value.integer_bits,integer_unsigned=value.integer_unsigned,
                              python_scalar=value.python_scalar)
        if isinstance(node, ast.Compare):
            operands = [self.expression(node.left)] + [self.expression(item) for item in node.comparators]
            names = {ast.Gt: "gt", ast.GtE: "ge", ast.Lt: "lt", ast.LtE: "le", ast.Eq: "eq", ast.NotEq: "eq"}
            comparisons = []
            for op, a, b in zip(node.ops, operands, operands[1:]):
                integer_float = ('float' not in (a.kind, b.kind)
                                 and self._integer_dtype((a, b)).kind == 'f')
                if "float" in (a.kind, b.kind) or integer_float:
                    precision = self._promotion((a, b))
                    code = f"d_{names[type(op)]}({self._operand(a, precision)}, {self._operand(b, precision)})"
                    comparisons.append(f"(!{code})" if isinstance(op, ast.NotEq) else code)
                else:
                    symbols = {ast.Gt: ">", ast.GtE: ">=", ast.Lt: "<", ast.LtE: "<=", ast.Eq: "==", ast.NotEq: "!="}
                    comparisons.append(f"({a.code} {symbols[type(op)]} {b.code})")
            return Expression("(" + " && ".join(comparisons) + ")", "bool", all(arg.scalar for arg in operands))
        if isinstance(node, ast.IfExp):
            return self._select(self.expression(node.test),
                                self.expression(node.body), self.expression(node.orelse))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if isinstance(self.values.get(name), _TimedArray):
                return super().expression(node)
            # With Brian's loop optimiser disabled, linear updates leave
            # invariant exponentials gated by the refractory boolean in the
            # vector expression. Both exact invariant values can still be
            # prepared; this does not implement dynamic transcendental math.
            functions = {item.func.id for item in ast.walk(node)
                         if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)}
            names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)} - functions
            if (name in _PREP_FUNCTIONS and name not in ("int", "timestep", "abs", "sqrt", "sign", "clip")
                    and names - self.values.keys() == {"not_refractory"}):
                choices = _evaluate(node, dict(self.values,
                    not_refractory=np.array([False, True], bool)))
                return self._select(self._name("not_refractory"),
                                    self.expression(repr(float(choices[1]))),
                                    self.expression(repr(float(choices[0]))))
            args = [self.expression(arg) for arg in node.args]
            if name in ('floor','ceil') and len(args)==1:
                arg=args[0]
                code=self._numpy_int32(arg,f'd_{name}({self.convert(arg,"float").code})')
                return Expression(code,'int',arg.scalar,integer_bits=32)
            if name=='int' and len(args)==1 and args[0].kind=='float':
                arg=args[0]
                return Expression(self._numpy_int32(arg,arg.code),'int',arg.scalar,integer_bits=32)
            if name in _UFUNC_HELPERS and len(args)==1:
                arg=args[0];narrow=ufunc_float32_precision(arg)
                wide,single=_UFUNC_HELPERS[name]
                code64=f'{wide}({self.convert(arg,"float").code})'
                code32=f'h_f32_to_d({single}({self.convert(arg,"float32").code}))'
                if narrow is not False:self.used_math.update((name+'_f32','_half_f32'))
                if narrow is not True:self.used_math.add(name)
                return Expression(_choose(narrow,code32,code64),'float',arg.scalar,narrow,
                    array_object=self.wrapper_arrays.get(name,False) if name in ('exp','log','expm1','log1p') else False)
            if name=='exprel' and len(args)==1:
                return self._exprel(node,args[0])
            if name == "timestep":
                offset = self._binary(self.expression("0.001"), args[1], ast.Mult())
                result = self._binary(self._binary(args[0], offset, ast.Add()), args[1], ast.Div())
                scalar=all(a.scalar for a in args)
                return Expression(self.convert(result, "int").code, "int", scalar,python_scalar=scalar)
            if name in ("abs", "sqrt", "sign") and len(args) == 1:
                arg = args[0]
                if arg.kind == "float":
                    if name == "sign":
                        code = f"(d_gt({arg.code}, 0ul) ? {_literal(1., 'float')} : (d_lt({arg.code}, 0ul) ? {_literal(-1., 'float')} : (d_is_zero({arg.code}) ? 0ul : {arg.code})))"
                    else:
                        code = f"d_{name}({arg.code})"
                    return Expression(_choose(arg.narrow, _round(code), code), "float", arg.scalar, arg.narrow)
            if name == "arctan" and len(args) == 1:
                arg = args[0]
                narrow = ufunc_float32_precision(arg)
                value = self.convert(arg, "float").code
                code64 = f"_eon_arctan64({value})"
                code32 = f"f_to_d(_eon_arctan32(f_from_d({value})))"
                if narrow is not False: self.used_math.add("arctan_f32")
                if narrow is not True: self.used_math.add("arctan")
                return Expression(_choose(narrow, code32, code64), "float", arg.scalar, narrow)
            if name == "clip" and len(args) == 3:
                narrow = self._promotion(args)
                value, low, high = [self._operand(arg, narrow) for arg in args]
                return Expression(f"(d_lt({value}, {low}) ? {low} : (d_gt({value}, {high}) ? {high} : {value}))",
                                  "float", all(a.scalar for a in args), narrow)
        result = super().expression(node)
        return Expression(result.code, result.kind)

    def _inplace_value(self, previous, rendered):
        if previous.kind == "float":
            code = self.convert(rendered, "float").code
            return Expression(_choose(previous.narrow, _round(code), code),
                              "float", previous.scalar, previous.narrow,python_scalar=previous.python_scalar,array_object=previous.array_object)
        code = self.convert(rendered, previous.kind).code
        if previous.kind == "int":
            code = self._integer_round(code, previous.integer_bits, previous.integer_unsigned)
        return Expression(code, previous.kind, previous.scalar,
                          integer_bits=previous.integer_bits,
                          integer_unsigned=previous.integer_unsigned,python_scalar=previous.python_scalar,array_object=previous.array_object)

    def _temporary(self, name, expression):
        self.temporaries[name] = Expression(self.code_name(name), expression.kind, expression.scalar, expression.narrow,None if expression.array_object else expression.value,
            integer_bits=expression.integer_bits, integer_unsigned=expression.integer_unsigned,python_scalar=expression.python_scalar,array_object=expression.array_object)

    def assignment(self, statement, expression=None, declare=True, masked=False):
        source = statement.expr if expression is None else expression
        rendered = self.expression(source)
        if (self.stage == "state" and statement.var == "not_refractory"
                and not getattr(self, 'numpy_callback_locals', False)):
            value = self.convert(rendered, "bool").code
            self.temporaries[statement.var] = Expression(statement.var, "bool", rendered.scalar)
            return f"not_refractory = {value};"
        inplace = statement.op not in ("=", ":=")
        previous = self._name(statement.var) if inplace or masked else None
        if inplace:
            operations = {"+=": ast.Add, "-=": ast.Sub, "*=": ast.Mult, "/=": ast.Div, "%=": ast.Mod}
            rendered = self._binary(previous, rendered, operations[statement.op]())
        retain = masked or inplace and (not previous.scalar or previous.array_object)
        if retain:
            rendered = self._inplace_value(previous, rendered)
        if statement.var in self.persistent:
            self.written.add(statement.var)
            if rendered.scalar and not retain:
                self.scalar_store_sources[statement.var]=rendered
            else:self.scalar_store_sources.pop(statement.var,None)
        storage = (statement.var in self.persistent or statement.var in self.shadows
                   or not rendered.scalar or rendered.array_object
                   or self._requires_typed_storage(rendered)) and statement.var != "_cond"
        if storage:
            node = ast.parse(source, mode='eval').body if isinstance(source, str) else source
            if not retain and not inplace and isinstance(node, ast.Name) and (not rendered.scalar or rendered.array_object):
                # Python array assignment aliases storage. An in-place write
                # through either name is visible through every surviving alias.
                name, initialize = node.id, ''
                if name not in self.shadows:
                    if (name in self.values and rendered.array_object
                            or name in self.temporaries and name not in self.persistent):
                        self._shadow(name, expression=rendered)
                        value = self._storage_encode(rendered)
                        initialize = f'{self.shadows[name]} = {value};'
                        self.values.pop(name,None)
                    else:
                        self._shadow(name)
                self.shadows[statement.var] = self.shadows[name]
                self._bind(statement.var, self._name(name))
                return initialize
            if retain:
                self._shadow(statement.var)
            else:
                # A new array expression rebinds this name without changing
                # the old array still referenced by other local names.
                self._shadow(statement.var, expression=rendered, replace=True)
            self._bind(statement.var, rendered)
            value = self._storage_encode(rendered)
            return f"{self.shadows[statement.var]} = {value};"
        # Scalar auxiliary values have value semantics and no array aliases.
        self._temporary(statement.var, rendered)
        prefix = (self._temporary_type(rendered) + " "
                  if statement.op == ":=" and declare and statement.var != "_cond" else "")
        return f"{prefix}{self.code_name(statement.var)} = {rendered.code};"

    def statement(self, statement, conditional_write=None):
        self.prelude=[]
        # Match NumpyCodeGenerator's where() rewrite, including its cost test.
        used = getattr(statement, "used_boolean_variables", None)
        if (used and len(used) == 1 and _kind(statement.dtype) in ("float", "float32")
                and statement.complexity_std > sum(statement.complexities.values())):
            branches = {conditions[0][1]: self.expression(expr)
                        for conditions, expr in statement.boolean_simplified_expressions.items()}
            expression = self._select(self._name(used[0]), branches[True], branches[False])
            expression = replace(expression,array_object=True)
            code = self.assignment(statement, expression, masked=bool(conditional_write))
        else:
            code = self.assignment(statement, masked=bool(conditional_write))
        code = "\n".join(self.prelude+[code])
        if conditional_write:
            code = f"if ({self._name(conditional_write).code}) {{ {code} }}"
        return code
