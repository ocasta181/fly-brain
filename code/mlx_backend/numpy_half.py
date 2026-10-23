"""NumPy half temporaries introduced by registered Boolean ufuncs.

This extends the existing NumPy32 renderer with dtype flow; persistent state
stays binary32/binary64. Register a builtin name as (dependency, scalar32helper)
through ``half_ufuncs``. Other function owners can use the same interface without
duplicating half arithmetic/promotion or selecting their NumPy32 SIMD routine.
The ARM NumPy1.26 half ufunc loop calls scalar float32 libm then rounds to half:
numpy/core/src/umath/loops_umath_fp.dispatch.c.src:165–168.
"""
import ast
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .compiler import _evaluate,_literal,_PREP_FUNCTIONS,UnsupportedModelError
from .numpy32 import Renderer as NumpyRenderer,Expression as NumpyExpression,_and,_choose,_round


@dataclass(frozen=True)
class Expression(NumpyExpression):
    half: bool | str = False


def half_precision(expression):
    return getattr(expression,"half",False)


@lru_cache(maxsize=1)
def scalar_boundary():
    """Installed NumPy's value-dependent float16 weak-scalar boundary."""
    low,high=0,int(np.float64(np.finfo(np.float16).max).view(np.uint64))
    vector=np.empty(1,np.float16)
    while low<high:
        middle=(low+high+1)//2
        if np.result_type(vector,np.uint64(middle).view(np.float64))==np.float16:low=middle
        else:high=middle-1
    return _literal(np.uint64(low).view(np.float64),"float")


class Renderer(NumpyRenderer):
    def __init__(self,*args,half_ufuncs=None,half_binary_operators=None,
                 half_input_arrays=None,half_input_names=None,**kwargs):
        super().__init__(*args,**kwargs)
        from .numpy_basic_half import FUNCTIONS
        # sqrt already has a native scalar32 helper, and NumPy selects its
        # half loop for Boolean/int8 arguments just as for registered libm.
        self.half_ufuncs={"sqrt":("_half_sqrt","f_sqrt"),
                         **dict.fromkeys(FUNCTIONS),**dict(half_ufuncs or {})}
        self.half_binary_operators=dict(half_binary_operators or {})
        self.half_input_arrays=half_input_arrays if half_input_arrays is not None else {}
        self.half_input_names=half_input_names or {}
        self._expression_cache=None

    def _half_ufunc(self,name,arg):
        from .numpy_basic_half import FUNCTIONS,values
        if name in FUNCTIONS:
            from .compiler import CompiledInputArray
            input_name=self.half_input_names.get(name,'_eon_numpy_half_'+name)
            if input_name not in self.half_input_arrays:
                data=values(name)
                self.half_input_arrays[input_name]=CompiledInputArray(
                    'numpy-half-'+name,'float64','ulong',data.shape,data)
            self.used_math.add('_half')
            return f'{input_name}[h_from_d({self.convert(arg,"float").code})]'
        dependency,helper=self.half_ufuncs[name]
        self.used_math.update((dependency,'_half_f32'))
        return self._half_round(f'h_f32_to_d({helper}({self.convert(arg,"float32").code}))')

    def _half_round(self,code):
        self.used_math.add("_half")
        return f"h_round({code})"

    def _round_precision(self,code,narrow,half=False,operation=False):
        # HALF arithmetic loops compute in float32, then narrow to binary16.
        half_code=self._half_round(_round(code) if operation else code) if half is not False else code
        return _choose(half,half_code,_choose(narrow,_round(code),code))

    def _half_promotion(self,args):
        if all(arg.scalar for arg in args):
            return _and([half_precision(arg) if arg.kind=="float" else
                         arg.kind=="bool" or arg.integer_bits<=8 for arg in args])
        predicates=[]
        floating_array=any(arg.kind=="float" and not arg.scalar for arg in args)
        for arg in args:
            if arg.kind=="bool":continue
            if not arg.scalar or not floating_array:
                # Signed/unsigned 8-bit arrays promote with half to half;
                # wider integers require the ordinary float32/64 loop.
                # Scalar values weaken only beside a floating array.
                predicates.append(half_precision(arg) if arg.kind=="float" else arg.integer_bits<=8)
            elif arg.value is not None:
                predicates.append(np.result_type(np.empty(1,np.float16),arg.value)==np.float16)
            elif arg.kind=="float":
                predicates.append(f"(d_le(d_abs({arg.code}), {scalar_boundary()}) || d_exp({arg.code}) == 2047ul)")
            else:predicates.append(f"(({arg.code}) >= -128L && ({arg.code}) <= 255L)")
        return _and(predicates)

    def _precision_operand(self,arg,narrow,half):
        if arg.value is not None and np.iscomplexobj(arg.value):
            raise UnsupportedModelError('Native state-dependent complex arithmetic is not implemented')
        return self._round_precision(self.convert(arg,"float").code,narrow,half)

    def _binary(self,left,right,operation):
        if isinstance(operation, (ast.Pow, ast.Mod, ast.FloorDiv)):
            return super()._binary(left, right, operation)
        if not any(half_precision(arg) is not False for arg in (left,right)):
            return super()._binary(left,right,operation)
        names={ast.Add:"add",ast.Sub:"sub",ast.Mult:"mul",ast.Div:"div"}
        if type(operation) not in names:
            raise UnsupportedModelError(f"Native half dynamic operator {type(operation).__name__} is not implemented")
        narrow,half=self._promotion((left,right)),self._half_promotion((left,right))
        if half is False:return super()._binary(left,right,operation)
        a,z=(self._precision_operand(arg,True,True) for arg in (left,right))
        # The canonical scalar32 helpers preserve NaN priority and signs
        # before the existing explicit rounding to half precision.
        self.used_math.update(("operator_remainder_float32","_half_f32"))
        code=self._half_round(f"h_f32_to_d(f_operator_{names[type(operation)]}(h_d_to_f({a}), h_d_to_f({z})))")
        if half is True:return Expression(code,"float",left.scalar and right.scalar,True,half=True)
        regular=super()._binary(left,right,operation)
        return Expression(_choose(half,code,regular.code),"float",regular.scalar,regular.narrow,half=half)

    def _select(self,condition,a,z):
        if not any(half_precision(arg) is not False for arg in (a,z)):
            return super()._select(condition,a,z)
        narrow,half=self._promotion((a,z)),self._half_promotion((a,z))
        test=self.convert(condition,"bool").code
        code=f"({test} ? {self._precision_operand(a,narrow,half)} : {self._precision_operand(z,narrow,half)})"
        return Expression(code,"float",condition.scalar and a.scalar and z.scalar,narrow,half=half)

    def _bind(self,name,expression):
        super()._bind(name,expression)
        bound=self.temporaries[name]
        self.temporaries[name]=Expression(bound.code,bound.kind,bound.scalar,bound.narrow,bound.value,
                                         bound.integer_bits,half_precision(expression),
                                         integer_unsigned=bound.integer_unsigned,python_scalar=bound.python_scalar,array_object=bound.array_object)

    def expression(self,source):
        if isinstance(source,NumpyExpression):return source
        node=ast.parse(source,mode="eval").body if isinstance(source,str) else source
        # A superclass may render the same argument again while selecting its
        # ordinary-precision implementation. Reuse that expression so eager
        # TimedArray reads are emitted once, inside the original write mask.
        root=self._expression_cache is None
        if root:self._expression_cache={}
        try:
            if node not in self._expression_cache:
                self._expression_cache[node]=self._expression(node)
            return self._expression_cache[node]
        finally:
            if root:self._expression_cache=None

    def _expression(self,node):
        try:value=self.evaluate(node)
        except KeyError:pass
        else:
            if np.asarray(value).dtype==np.dtype("float16"):
                return Expression(_literal(np.float64(value),"float"),"float",True,True,value,half=True,
                                  array_object=isinstance(value,np.ndarray))
        if isinstance(node,ast.BinOp) and type(node.op) in self.half_binary_operators:
            from .numpy_half_operators import render_registered_operator
            left,right=self.expression(node.left),self.expression(node.right)
            if "float" in (left.kind,right.kind):
                return render_registered_operator(self,node,left,right,self.half_binary_operators[type(node.op)])
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            arg=self.expression(node.operand)
            if half_precision(arg) is not False:
                code=arg.code if isinstance(node.op,ast.UAdd) else f"d_neg({arg.code})"
                return Expression(code,arg.kind,arg.scalar,arg.narrow,half=half_precision(arg))
        if isinstance(node,ast.Compare):
            operands=[self.expression(node.left)]+[self.expression(item) for item in node.comparators]
            if any(half_precision(arg) is not False for arg in operands):
                names={ast.Gt:"gt",ast.GtE:"ge",ast.Lt:"lt",ast.LtE:"le",ast.Eq:"eq",ast.NotEq:"eq"}
                comparisons=[]
                for operation,a,z in zip(node.ops,operands,operands[1:]):
                    narrow,half=self._promotion((a,z)),self._half_promotion((a,z))
                    code=f"d_{names[type(operation)]}({self._precision_operand(a,narrow,half)}, {self._precision_operand(z,narrow,half)})"
                    comparisons.append(f"(!{code})" if isinstance(operation,ast.NotEq) else code)
                return Expression("("+" && ".join(comparisons)+")","bool",all(arg.scalar for arg in operands))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
            name=node.func.id
            args=[self.expression(item) for item in node.args]
            if name in ("floor","ceil") and len(args)==1:
                # Brian's NumPy wrappers cast the ufunc result to int32,
                # including for half inputs. Retaining half here changes the
                # dtype (and rounding) of subsequent weak-scalar arithmetic.
                arg=args[0]
                value=self.convert(arg,"float").code
                code=self._numpy_int32(arg,f"d_{name}({value})")
                return Expression(code,"int",arg.scalar,integer_bits=32)
            if name=="int" and len(args)==1 and args[0].kind=="float":
                arg=args[0]
                return Expression(self._numpy_int32(arg,self.convert(arg,"float").code),
                                  "int",arg.scalar,integer_bits=32)
            if name in self.half_ufuncs and len(args)==1:
                arg=args[0]
                if arg.kind=="bool":
                    # This is immutable preparation of the two builtin values,
                    # not a host callback or a timestep evaluation.
                    choices=_PREP_FUNCTIONS[name](np.array([False,True],bool))
                    if choices.dtype!=np.float16:
                        raise UnsupportedModelError(f"Unexpected installed NumPy Boolean {name} dtype {choices.dtype}")
                    code=f"({arg.code} ? {_literal(np.float64(choices[1]),'float')} : {_literal(np.float64(choices[0]),'float')})"
                    return Expression(code,"float",arg.scalar,True,half=True,
                                      array_object=self.wrapper_arrays.get(name,False))
                if arg.kind=="int" and arg.integer_bits<=8:
                    # Every signed/unsigned 8-bit integer is exactly a half.
                    # NumPy therefore selects the same scalar32->half loop as
                    # for Boolean/half inputs, rather than its float32 SIMD.
                    code=self._half_ufunc(name,arg)
                    return Expression(code,"float",arg.scalar,True,half=True,
                                      array_object=self.wrapper_arrays.get(name,False))
                half=half_precision(arg)
                if half is not False:
                    code=self._half_ufunc(name,arg)
                    if half is True:return Expression(code,"float",arg.scalar,True,half=True,
                        array_object=self.wrapper_arrays.get(name,False))
                    regular=super().expression(node)
                    return Expression(_choose(half,code,regular.code),"float",arg.scalar,regular.narrow,half=half,
                                      array_object=self.wrapper_arrays.get(name,False))
            if args and any(half_precision(arg) is not False for arg in args):
                if name in ("abs","sign") and len(args)==1:
                    arg=args[0];half=half_precision(arg)
                    regular=super().expression(node)
                    code=regular.code
                    return Expression(_choose(half,self._half_round(code),regular.code),"float",arg.scalar,arg.narrow,half=half)
                if name=="clip" and len(args)==3:
                    narrow,half=self._promotion(args),self._half_promotion(args)
                    # HALF_clip uses its own >=/<= comparisons, preserving
                    # the first operand on NaNs and equal signed zeros.
                    # NumPy1.26.4 numpy/core/src/umath/clip.cpp:32-40,104-107.
                    if half is False:return super().expression(node)
                    value,low,high=[self._precision_operand(arg,True,True) for arg in args]
                    lower=f"((d_is_nan({value}) || d_ge({value},{low})) ? {value} : {low})"
                    code=f"((d_is_nan({lower}) || d_le({lower},{high})) ? {lower} : {high})"
                    if half is True:return Expression(code,"float",all(a.scalar for a in args),True,half=True)
                    regular=super().expression(node)
                    return Expression(_choose(half,code,regular.code),"float",regular.scalar,regular.narrow,half=half)
        return super().expression(node)

    def _inplace_value(self, previous, rendered):
        if previous.kind == "float":
            code = self.convert(rendered, "float").code
            return Expression(self._round_precision(code, previous.narrow, half_precision(previous)),
                "float", previous.scalar, previous.narrow, half=half_precision(previous),
                array_object=previous.array_object,python_scalar=previous.python_scalar)
        return super()._inplace_value(previous, rendered)

    def _temporary(self, name, expression):
        self.temporaries[name] = Expression(self.code_name(name), expression.kind, expression.scalar, expression.narrow,None if expression.array_object else expression.value,
            integer_bits=expression.integer_bits, half=half_precision(expression),
            integer_unsigned=expression.integer_unsigned,python_scalar=expression.python_scalar,array_object=expression.array_object)
