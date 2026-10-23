"""Lower Brian's typed statements to exact floating-point Metal operations.

This preparation-only compiler does not advance a Brian network. Immutable
scalar expressions are evaluated in the resolved CPU target's operation order.
Dynamic expressions use raw binary32/binary64 helpers; native operation programs
preserve stateful draws and ordered NumPy policy boundaries. Unsupported
lowering raises UnsupportedModelError without executing a CPU simulation.
"""

from __future__ import annotations

import ast
import copy
import math
import platform
import struct
from fractions import Fraction
from dataclasses import dataclass, replace, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .symbols import variable_code_name, variable_code_names
from .model import _lif_refractory_write_variant, DEFAULT_EQUATIONS, DEFAULT_THRESHOLD, ResolvedModel, TimedArrayValue, _expression_key, _preparation_device, model_preferences


class UnsupportedModelError(NotImplementedError):
    """Brian accepted a model whose native lowering is not implemented yet."""


def _compiled_math_policy(model: ResolvedModel) -> tuple[str, bool]:
    """Relevant effective flags, preserving their command-line order."""
    reciprocal, contract, unsafe = False, True, False
    optimized = model.reference_codegen_target == "cython"
    for flag in model.reference_compile_args:
        if flag == "-O0":
            optimized = False
        elif flag in ("-O1", "-O2", "-O3", "-Os", "-Oz", "-Ofast"):
            optimized = True
        if flag in ("-ffast-math", "-Ofast", "-funsafe-math-optimizations"):
            reciprocal, contract, unsafe = True, True, True
        elif flag in ("-fno-fast-math", "-fno-unsafe-math-optimizations"):
            reciprocal, contract, unsafe = False, False, False
        elif flag == "-freciprocal-math":
            reciprocal = True
        elif flag == "-fno-reciprocal-math":
            reciprocal = False
        elif flag == "-ffp-contract=off":
            contract = False
        elif flag in ("-ffp-contract=fast", "-ffp-contract=on"):
            contract = True
        elif flag == "-fno-associative-math":
            unsafe = False
    policy = "source" if not reciprocal or not optimized else "reciprocal_fma" if contract else "reciprocal"
    return policy, optimized and unsafe and contract


@dataclass(frozen=True)
class CompiledVariable:
    name: str
    dtype: str
    metal_type: str
    scalar: bool
    constant: bool
    read_only: bool
    conditional_write: str | None
    persistent: bool

    @property
    def storage_dtype(self):
        # Brian's CPP generator declares Boolean array elements as char.
        # Preserve its byte (and later signed promotion), not C++bool's0/1.
        return 'uint8' if self.dtype=='bool' and self.metal_type in ('char','uchar') else self.dtype

    @property
    def code_name(self):
        return ("_eon_shared_state_"+self.name if self.scalar and self.persistent
                else variable_code_name(self.name))


@dataclass(frozen=True,eq=False)
class ScalarStage:
    scalar_code: str
    variables: tuple[CompiledVariable,...]
    vector_code: str
    dynamics_code: str
    scalar_errors: tuple[str,...]=()
    vector_error: bool=False


@dataclass(frozen=True, eq=False)
class CompiledInputArray:
    """Immutable preparation data; binary64 values upload as their raw bits."""

    source_name: str
    dtype: str
    metal_type: str
    shape: tuple[int, ...]
    values: np.ndarray


@dataclass(frozen=True, eq=False)
class CompiledModel:
    variables: Mapping[str, CompiledVariable]
    scalar_code: str
    state_code: str
    threshold_code: str
    reset_code: str
    # Brian typed statements before target-specific compiled transformations.
    # The emitted stage code includes those transformations when applicable.
    scalar_values: Mapping[str, Any]
    state_statements: tuple[Any, ...]
    threshold_statements: tuple[Any, ...]
    reset_statements: tuple[Any, ...]
    refractory_code: str
    dynamics_code: str
    # A bit-exact fixed point of the emitted dynamics, when proven. The caller
    # must compare every listed state's raw bits before skipping that block.
    equilibrium: Mapping[str, Any] | None
    # Smallest nonnegative binary64 elapsed time satisfying Brian's static-rfc
    # predicate. Valid for elapsed>=0 while timestep conversion is in int64
    # range; canonical equations/reset cannot move lastspike ahead of t.
    refractory_elapsed_threshold: float | None
    input_arrays: Mapping[str, CompiledInputArray]
    header_code: str
    # The caller declares this thread-local bool and propagates it between
    # windows. Materializing an invalid 2D access raises IndexError on the host.
    index_error_name: str | None
    # Brian evaluates the scalar reset section even when its spike vector is
    # empty. Keep the combined reset_code for host diagnostics, plus this split
    # so runtime scalar errors are observable before conditional vector writes.
    reset_scalar_code: str
    reset_vector_code: str
    # Target-specific release arithmetic for audited compiled model forms.
    refractory_policy: str = "elapsed"
    # Pure 1D calls whose time expression depends only on the clock and frozen
    # scalar constants. Runtime may evaluate these once per clock on GPU;
    # emitted stage code remains independently executable without that cache.
    shared_timed_calls: tuple[str, ...] = ()
    timed_cache_name: str | None = None
    # Index errors belong to NumPy TimedArrays; Cython floating divmod errors
    # use the same sticky transport. A model has one frozen CPU target.
    runtime_error_type: str = "index"
    operator_policies: tuple[Any, ...] = ()
    collective_reset: bool = False
    replay_phase: str | None = None
    runtime_errors: tuple[tuple[str, str], ...] = ()
    runtime_error_order: str = "row"
    runtime_error_writes: tuple[tuple[int,tuple[str,...]],...] = ()
    disabled_stages: tuple[str,...] = ()
    compiled_integer_stages: tuple[str,...] = ()
    scalar_stages: Mapping[str, ScalarStage] | None = None
    subexpression_code: str = ""
    cpu_refractory_guard: bool = False
    scalar_parallel: bool = False
    scalar_input_independent: bool = False
    native_precision_limbs: int = 0
    random_program: object | None = None
    shared_expression_inputs: bool = False
    complex_warning_sites: object | None = None
    # A selected custom Cython state block restored to canonical LIF arithmetic.
    canonical_custom_lif: bool = False


# Brian 2.8's NumPy TimedArray implementation rounds to nearest-even *before*
# dividing by K, then casts to int64 and clips. Its C++ implementation instead
# adds .5, so copying that implementation changes half-step boundaries. Default
# compiled fast-math folds the affine lookup to reciprocal/FMA; strict flags
# preserve the source division/addition sequence instead.
_TIMED_ARRAY_HEADER = r"""
inline ulong _eon_timed_round_even(ulong value) {
    ulong magnitude = value & 0x7ffffffffffffffful;
    ulong sign = value & 0x8000000000000000ul;
    int exponent = int((magnitude >> 52) & 0x7fful) - 1023;
    if(exponent >= 52) return value;
    if(exponent < -1) return sign;
    if(exponent == -1)
        return sign | (magnitude > 0x3fe0000000000000ul ? 0x3ff0000000000000ul : 0ul);
    int shift = 52-exponent;
    ulong mask = (1ul << shift)-1ul;
    ulong integral = magnitude & ~mask;
    ulong remainder = magnitude & mask;
    ulong halfway = 1ul << (shift-1);
    if(remainder > halfway || (remainder == halfway && ((integral >> shift) & 1ul)))
        integral += 1ul << shift;
    return sign | integral;
}
inline long _eon_timed_index_numpy(ulong t, ulong epsilon, ulong factor, long count) {
    ulong scaled = d_div(_eon_timed_round_even(d_div(t, epsilon)), factor);
    // Apple silicon NumPy casts saturate overflow/infinities, and NaN becomes
    // zero. Guard the cast explicitly, including for the host source checks.
    if(!d_gt(scaled, 0ul)) return 0L;
    if(d_ge(scaled, 0x43e0000000000000ul)) return count-1L;
    long index = d_to_int(scaled);
    return index >= count ? count-1L : index;
}
inline long _eon_timed_int32(ulong value) {
    long converted = d_to_int(value);
    return converted < -2147483648L ? -2147483648L :
           (converted > 2147483647L ? 2147483647L : converted);
}
inline long _eon_timed_clamp_compiled(long index, long count) {
    return index < 0L ? 0L : (index >= count ? count-1L : index);
}
inline long _eon_timed_index_compiled(ulong t, ulong epsilon, ulong factor, long count) {
    ulong scaled = d_div(d_add(d_div(t, epsilon), 0x3fe0000000000000ul), factor);
    long index = _eon_timed_int32(scaled);
    return _eon_timed_clamp_compiled(index, count);
}
"""


@dataclass(frozen=True, eq=False)
class _TimedArray:
    input_name: str
    helper_name: str
    array: CompiledInputArray
    epsilon: np.float64
    factor: int
    target: str
    math_policy: str
    reciprocal: np.float64
    offset: np.float64

    def evaluate(self, args: list[Any]) -> Any:
        dimensions = len(self.array.shape)
        if len(args) != dimensions:
            raise UnsupportedModelError(f"TimedArray {self.array.source_name} expects {dimensions} arguments")
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            if self.target == "numpy":
                scaled = np.round(args[0] / self.epsilon) / self.factor
            elif self.math_policy == "source":
                scaled = (args[0] / self.epsilon + np.float64(.5)) / self.factor
            elif self.math_policy == "reciprocal_fma" and all(np.isfinite(value) for value in
                                                             (args[0], self.reciprocal, self.offset)):
                exact = (Fraction(float(args[0])) * Fraction(float(self.reciprocal)) +
                         Fraction(float(self.offset)))
                try:
                    scaled = np.float64(float(exact))
                except OverflowError:
                    scaled = np.float64(np.inf if exact > 0 else -np.inf)
            else:
                scaled = np.float64(args[0])*self.reciprocal + self.offset
            if self.target == "numpy":
                index = np.int64(scaled)
            else:
                index = (0 if np.isnan(scaled) else 2147483647 if scaled >= 2147483647 else
                         -2147483648 if scaled <= -2147483648 else math.trunc(scaled))
            index = np.clip(index, 0, self.array.shape[0]-1)
        if dimensions == 1:
            return self.array.values[index]
        if self.target != "numpy":
            column = args[1]
            if self.target == "cython" and np.asarray(column).dtype.kind == "f":
                raise UnsupportedModelError(f"TimedArray {self.array.source_name} requires an integer column index for Cython")
            if np.asarray(column).dtype.kind == "f":
                column = (0 if np.isnan(column) else 2147483647 if column >= 2147483647 else
                          -2147483648 if column <= -2147483648 else math.trunc(column))
            else:
                column = int(column) & 0xffffffff
                column = column-(1 << 32) if column >= 1 << 31 else column
            return self.array.values[index, column] if 0 <= column < self.array.shape[1] else np.float64(np.nan)
        if np.asarray(args[1]).dtype.kind not in "iu":
            raise UnsupportedModelError(f"TimedArray {self.array.source_name} requires an integer column index")
        try:
            return self.array.values[index, args[1]]
        except IndexError:
            # Even an invariant invalid access fails only when its code runs;
            # a zero-step call must remain valid.
            raise KeyError(self.array.source_name) from None

    def header(self) -> str:
        policy = "numpy" if self.target == "numpy" else "compiled"
        if self.target != "numpy" and self.math_policy != "source":
            operation = (f"d_fma(t, {_literal(self.reciprocal, 'float')}, {_literal(self.offset, 'float')})"
                         if self.math_policy == "reciprocal_fma" else
                         f"d_add(d_mul(t, {_literal(self.reciprocal, 'float')}), {_literal(self.offset, 'float')})")
            index = (f"_eon_timed_clamp_compiled(_eon_timed_int32({operation}), "
                     f"{self.array.shape[0]}L)")
        else:
            index = (f"_eon_timed_index_{policy}(t, {_literal(self.epsilon, 'float')}, "
                     f"{_literal(self.factor, 'float')}, {self.array.shape[0]}L)")
        if len(self.array.shape) == 1:
            return (f"template<typename Values>\ninline ulong {self.helper_name}(Values values, ulong t) {{\n"
                    f"    return values[{index}];\n}}\n")
        columns = self.array.shape[1]
        if self.target != "numpy":
            return f"""
template<typename Values>
inline ulong {self.helper_name}(Values values, ulong t, long column) {{
    if(column < 0L || column >= {columns}L) return 0x7ff8000000000000ul;
    return values[({index})*{columns}L + column];
}}
"""
        return f"""
template<typename Values>
inline ulong {self.helper_name}(Values values, ulong t, long column,
                               thread bool& index_error) {{
    if(column < 0L) column += {columns}L;
    if(column < 0L || column >= {columns}L) {{
        index_error = true;
        return 0x7ff8000000000000ul;
    }}
    return values[({index})*{columns}L + column];
}}
"""


def _numpy_scalar_carrier(value):
    """Encode an invariant complex object's real component without losing its type.

    The NumPy renderer retains the original object for expression folding and
    destination conversion. Only its native carrier is projected here;
    state-dependent complex arithmetic must never consume that carrier alone.
    """
    if np.ndim(value) == 0 and np.iscomplexobj(value):
        return np.asarray(value).real[()]
    return value


def _kind(dtype: Any) -> str:
    kind = np.dtype(dtype).kind
    if kind == "b":
        return "bool"
    if kind in "iu":
        return "int"
    if kind == "f" and np.dtype(dtype).itemsize == 8:
        return "float"
    if kind == "f" and np.dtype(dtype).itemsize == 4:
        return "float32"
    raise UnsupportedModelError(f"Native equation compiler does not implement dtype {np.dtype(dtype)}")


def _metal_type(dtype: Any) -> str:
    kind = _kind(dtype)
    if kind == "float":
        return "ulong"
    if kind == "float32":
        return "uint"
    if kind == "bool":
        return "bool"
    dtype = np.dtype(dtype)
    if dtype.kind not in "iu" or dtype.itemsize not in (1, 2, 4, 8):
        raise UnsupportedModelError(f"Native equation compiler does not implement integer dtype {np.dtype(dtype)}")
    return {1: ("char", "uchar"), 2: ("short", "ushort"),
            4: ("int", "uint"), 8: ("long", "ulong")}[dtype.itemsize][dtype.kind == "u"]


def _literal(value: Any, kind: str) -> str:
    if kind == "float32":
        return f"{int(np.float32(value).view(np.uint32))}u"
    if kind == "float":
        from .binary64 import literal
        return literal(float(value))
    if kind == "bool":
        return "true" if value else "false"
    integer = int(value)
    if integer == -(1 << 63):
        return "(-9223372036854775807L - 1L)"
    if integer >= 1 << 63:
        return f"{integer}UL"
    return f"{integer}L"


def _cast_value(value: Any, dtype: Any) -> Any:
    return np.dtype(dtype).type(value)


# Evaluating only this whitelist avoids invoking arbitrary namespace callbacks.
# np scalar operands preserve the statement AST's binary64 rounding points.
def _prep_exprel(value):
    # Brian defines exprel's small/large-argument branches itself. Reuse that
    # invariant-only definition instead of dividing expm1(0) by zero or running
    # a host function for any state-dependent expression.
    from brian2.units.unitsafefunctions import exprel
    return exprel(value)


_PREP_FUNCTIONS = {
    "exp": np.exp, "expm1": np.expm1, "log": np.log, "log10": np.log10,
    "log1p": np.log1p, "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos,
    "tan": np.tan, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "abs": np.abs, "ceil": np.ceil, "floor": np.floor, "sign": np.sign,
    "clip": np.clip, "int": np.int64,
    "exprel": _prep_exprel,
    "timestep": lambda t, dt: np.int64((t + np.float64(1e-3) * dt) / dt),
}


def _evaluate(node: ast.AST, values: Mapping[str, Any], reference_target: str = "numpy", *, numpy_objects=False, numpy_wrappers=None) -> Any:
    """Evaluate a side-effect-free invariant, or raise KeyError if dynamic.

    NumPy source keeps Python literal/Constant objects until an actual ufunc
    or array operation converts them. This opt-in preserves that object flow;
    compiled-target preparation retains its existing typed scalar policy.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
        if numpy_objects:return node.value
        return (np.bool_(node.value) if isinstance(node.value, bool) else
                np.int64(node.value) if isinstance(node.value, int) else np.float64(node.value))
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, values, reference_target, numpy_objects=numpy_objects, numpy_wrappers=numpy_wrappers)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.Not):
            return np.bool_(not value)
    if isinstance(node, ast.BinOp):
        left, right = _evaluate(node.left, values, reference_target, numpy_objects=numpy_objects, numpy_wrappers=numpy_wrappers), _evaluate(node.right, values, reference_target, numpy_objects=numpy_objects, numpy_wrappers=numpy_wrappers)
        # C's usual arithmetic conversions keep float32 when the other operand
        # is an integer. NumPy scalar promotion would otherwise widen to double.
        if not numpy_objects and isinstance(left, np.float32) and isinstance(right, (np.integer, np.bool_)):
            right = np.float32(right)
        elif not numpy_objects and isinstance(right, np.float32) and isinstance(left, (np.integer, np.bool_)):
            left = np.float32(left)
        operations = {ast.Add: lambda: left + right, ast.Sub: lambda: left - right,
                      ast.Mult: lambda: left * right, ast.Div: lambda: left / right,
                      ast.FloorDiv: lambda: left // right, ast.Mod: lambda: left % right,
                      ast.Pow: lambda: left ** right}
        if type(node.op) in operations:
            return operations[type(node.op)]()
    if isinstance(node, ast.BoolOp):
        operands = [_evaluate(item, values, reference_target, numpy_objects=numpy_objects, numpy_wrappers=numpy_wrappers) for item in node.values]
        return np.bool_(all(operands) if isinstance(node.op, ast.And) else any(operands))
    if isinstance(node, ast.Compare):
        operands = [_evaluate(node.left, values, reference_target, numpy_objects=numpy_objects, numpy_wrappers=numpy_wrappers)] + [_evaluate(item, values, reference_target, numpy_objects=numpy_objects, numpy_wrappers=numpy_wrappers) for item in node.comparators]
        comparisons = {ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
                       ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
                       ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b}
        return np.bool_(all(comparisons[type(op)](a, b) for op, a, b in
                            zip(node.ops, operands, operands[1:])))
    if isinstance(node, ast.IfExp):
        return _evaluate(node.body if _evaluate(node.test, values, reference_target, numpy_objects=numpy_objects, numpy_wrappers=numpy_wrappers) else node.orelse, values, reference_target, numpy_objects=numpy_objects, numpy_wrappers=numpy_wrappers)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in('rand','randn','poisson'):
            raise KeyError('Stateful model draw cannot be folded during preparation')
        # Evaluate arguments first: state-dependent unimplemented math remains
        # dynamic; an unimplemented invariant function is explicitly unsupported.
        args = [_evaluate(arg, values, reference_target, numpy_objects=numpy_objects, numpy_wrappers=numpy_wrappers) for arg in node.args]
        timed_array = values.get(node.func.id)
        if isinstance(timed_array, _TimedArray) and not node.keywords:
            return timed_array.evaluate(args)
        if node.func.id == "int" and len(args) == 1 and isinstance(args[0], np.ndarray):
            # Brian's NumPy implementation preserves vectors and uses int32.
            return np.int32(args[0])
        function = _PREP_FUNCTIONS.get(node.func.id)
        if numpy_objects and node.func.id in ('int','floor','ceil'):
            from brian2.core.functions import DEFAULT_FUNCTIONS
            from brian2.codegen.generators.numpy_generator import NumpyCodeGenerator
            function=DEFAULT_FUNCTIONS[node.func.id].implementations[NumpyCodeGenerator].get_code(None)
        if (node.func.id in ("sin", "cos", "tan") and len(args)==1
                and reference_target in ("cython", "cpp") and np.ndim(args[0])==0):
            # Preparation-only scalar folding follows the actual compiled
            # overload too. In particular scalar cosine clears a negative NaN
            # sign, whereas NumPy's binary64 cosine preserves it.
            import ctypes
            single=isinstance(args[0],np.float32)
            scalar=getattr(ctypes.CDLL(None),node.func.id+("f" if single else ""))
            ctype=ctypes.c_float if single else ctypes.c_double
            scalar.argtypes,scalar.restype=[ctype],ctype
            return (np.float32 if single else np.float64)(scalar(args[0]))
        if node.func.id == "arctan" and reference_target in ("cython", "cpp"):
            if len(args) == 1 and isinstance(args[0], np.float32):
                import ctypes
                scalar = ctypes.CDLL(None).atanf
                scalar.argtypes, scalar.restype = [ctypes.c_float], ctypes.c_float
                return np.float32(scalar(args[0]))
            function = lambda value: np.float64(math.atan(value))
        if (node.func.id in ("log10", "sinh", "cosh", "tanh") and len(args) == 1
                and isinstance(args[0], np.float32) and reference_target in ("cython", "cpp")):
            # Preparation-only scalar32 folding follows the actual float overload.
            import ctypes
            scalar = getattr(ctypes.CDLL(None), node.func.id+"f")
            scalar.argtypes, scalar.restype = [ctypes.c_float], ctypes.c_float
            return np.float32(scalar(args[0]))
        if node.func.id == "tanh" and reference_target in ("cython", "cpp"):
            # Preparation-only invariant folding must use the scalar CPU math
            # policy too; the NumPy SIMD tanh can differ by one binary64 bit.
            function = lambda value: np.float64(math.tanh(value))
        if function is not None and not node.keywords:
            result=function(*args)
            if numpy_wrappers is not None and node.func.id in numpy_wrappers:
                return np.asarray(result) if numpy_wrappers[node.func.id] else np.asarray(result)[()]
            return result
        raise UnsupportedModelError(f"Native equation compiler does not implement function {node.func.id}")
    raise UnsupportedModelError(f"Native equation compiler does not implement {ast.dump(node)}")


@dataclass(frozen=True)
class _Expression:
    code: str
    kind: str
    integer_unsigned: bool = field(default=False,kw_only=True)


def _floating(kind):
    return kind in ("float", "float32")

def _prefix(kind):
    return "f" if kind == "float32" else "d"

def _common_kind(*expressions):
    kinds = {expression.kind for expression in expressions}
    return "float" if "float" in kinds else "float32" if "float32" in kinds else "int"


class _Renderer:
    def __init__(self, types: dict[str, Any], values: dict[str, Any],
                 used_arrays: set[str] | None = None, index_error_name: str | None = None,
                 used_math: set[str] | None = None,
                 shared_timed_calls: list[str] | None = None,
                 reference_target: str = "numpy", binary32: bool = False,
                 reference_flags: tuple[str, ...] = (),
                 operator_policies: dict[str, Any] | None = None,
                 zero_division_error_name: str | None = None,integer_policy=None,
                 runtime_errors: list[tuple[str, str]] | None = None, code_names=None):
        self.types = types
        self.expression_random = None
        self.code_names = {} if code_names is None else code_names
        self.values = values
        self.binary32 = binary32
        self.used_arrays = used_arrays if used_arrays is not None else set()
        self.index_error_name = index_error_name
        self.used_math = used_math if used_math is not None else set()
        if any(np.dtype(dtype).kind=='u' for dtype in types.values()):
            self.used_math.add('operator_uint_float32')
        self.prelude: list[str] = []
        self.temporary_index = 0
        self.shared_timed_calls = shared_timed_calls
        self.reference_target = reference_target
        self.reference_flags = reference_flags
        self.compiled_unsigned_char=bool(integer_policy is not None and integer_policy.unsigned_char)
        self.runtime_errors = runtime_errors if runtime_errors is not None else []
        self.operator_policies = operator_policies if operator_policies is not None else {}
        self.zero_division_error_name = zero_division_error_name or "_eon_operator_zero_error"
        self.expm1_results: dict[str, tuple[str, set[str]]] = {}
        from .hyperbolic_pair import StatementPairs
        self.hyperbolic_pairs = StatementPairs((), self.used_math)

    def plan_hyperbolic_pairs(self, statements):
        from .hyperbolic_pair import StatementPairs
        self.hyperbolic_pairs = StatementPairs(statements, self.used_math)

    def code_name(self, name):
        return self.code_names.get(name, name)

    def _timed_result(self, code, args):
        return _Expression(code, 'float')

    def compiled_operator_expression(self,node):
        if self.reference_target not in ('cython','cpp'):
            return False
        operators=(ast.Pow,ast.Mod,ast.FloorDiv)
        if not any(isinstance(child,ast.BinOp) and isinstance(child.op,operators) for child in ast.walk(node)):
            return False
        from types import SimpleNamespace
        from brian2.core.functions import DEFAULT_FUNCTIONS
        from brian2.parsing.bast import BrianASTRenderer
        variables={name:SimpleNamespace(dtype=dtype,scalar=True) for name,dtype in self.types.items()}
        variables.update(DEFAULT_FUNCTIONS)
        for name,value in self.values.items():
            if name not in variables and np.asarray(value).dtype.kind in 'fiub':
                variables[name]=SimpleNamespace(dtype=np.asarray(value).dtype,scalar=True)
        typed=BrianASTRenderer(variables).render_node(copy.deepcopy(node))
        return any(isinstance(child,ast.BinOp) and isinstance(child.op,operators)
                   and child.dtype=='float' for child in ast.walk(typed))

    def numpy_operand(self,node):
        """Infer NumPy array/scalar promotion without inspecting a state value."""
        try:
            return _evaluate(node,self.values)
        except KeyError:
            pass
        if isinstance(node,ast.Name):
            return np.ones((1,),dtype=self.types[node.id])
        if isinstance(node,ast.BinOp):
            a,b=self.numpy_operand(node.left),self.numpy_operand(node.right)
            operations={ast.Add:np.add,ast.Sub:np.subtract,ast.Mult:np.multiply,ast.Div:np.divide,
                        ast.FloorDiv:np.floor_divide,ast.Mod:np.remainder,ast.Pow:np.power}
            with np.errstate(all="ignore"):
                return operations[type(node.op)](a,b)
        if isinstance(node,ast.UnaryOp):
            return self.numpy_operand(node.operand)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
            if isinstance(self.values.get(node.func.id),_TimedArray):
                return np.ones((1,),np.float64)
            function=_PREP_FUNCTIONS.get(node.func.id)
            if function is not None:
                with np.errstate(all="ignore"):
                    return function(*(self.numpy_operand(arg) for arg in node.args))
        if isinstance(node,ast.IfExp):
            return np.ones((1,),np.result_type(self.numpy_operand(node.body),self.numpy_operand(node.orelse)))
        return np.ones((1,),np.float64)

    def floating_operator(self,node,left,right,*,result_kind=None,numpy_exponent_scalar=None,
                          numpy_original_exponent=None):
        kind=result_kind or _common_kind(left,right)
        target=self.reference_target
        if target not in ("numpy","cython","cpp"):
            raise UnsupportedModelError(f"Native floating operators do not implement CPU target {target}")
        if target=="numpy" and result_kind is None:
            kind=_kind(np.asarray(self.numpy_operand(node)).dtype)
        elif (target=="cpp" and isinstance(node.op,ast.Pow) and kind=="float32"
              and (not _floating(left.kind) or not _floating(right.kind))):
            # std::pow's arithmetic overload promotes an integer to double.
            kind="float"
        if target=='cpp':
            names={child.id for child in ast.walk(node) if isinstance(child,ast.Name)}
            functions={child.func.id for child in ast.walk(node)
                       if isinstance(child,ast.Call) and isinstance(child.func,ast.Name)}
            if names-functions<=self.values.keys():
                from brian2.parsing.rendering import CPPNodeRenderer
                from brian2.codegen.generators.cpp_generator import c_data_type
                from .operator_policy import constant_cpp_operator_policy
                cpp=CPPNodeRenderer()
                constants=tuple((name,c_data_type(np.asarray(self.values[name]).dtype),
                    cpp.render_expr(repr(np.asarray(self.values[name]).reshape(-1)[0].item())))
                    for name in sorted(names-functions))
                policy=constant_cpp_operator_policy(cpp.render_node(node),constants,kind,tuple(self.reference_flags))
                self.operator_policies[policy.name]=policy
                if policy.library_power:self.used_math.add('operator_pow_'+kind)
                if policy.library_powi:self.used_math.add('operator_powi_'+kind)
                return _Expression(f'{policy.name}()',kind)
        a,b=self.convert(left,kind).code,self.convert(right,kind).code
        prefix=_prefix(kind)
        if isinstance(node.op,ast.Pow):
            try:
                exponent=(None if target=="numpy" and (numpy_exponent_scalar is False or numpy_original_exponent is not None)
                          else _evaluate(node.right,self.values))
            except KeyError:
                exponent=None
            if exponent is not None:
                value=float(exponent)
                if target=="numpy" and numpy_exponent_scalar is not False and value in (-1.,0.,.5,1.,2.):
                    if value==0.: return _Expression(_literal(1.,kind),kind)
                    if value==1.: return self.convert(left,kind)
                    if value==-1.: return _Expression(f"{prefix}_div({_literal(1.,kind)},{a})",kind)
                    if value==2.: return _Expression(f"{prefix}_mul({a},{a})",kind)
                    square_root=f"d_sqrt({self.convert(left,'float').code})"
                    return _Expression(f"f_from_d({square_root})" if kind=="float32" else square_root,kind)
                # A scalar namespace or prepared temporary is loaded at
                # runtime by compiled Brian code. Only a literal expression
                # can receive Clang's constant-exponent rewrites.
                literal_exponent=not any(isinstance(child,ast.Name) for child in ast.walk(node.right))
                if target in ("cython","cpp") and literal_exponent:
                    from .operator_policy import constant_power_policy
                    input_kind=left.kind if _floating(left.kind) else kind
                    policy=constant_power_policy(input_kind,kind,value,tuple(self.reference_flags))
                    self.operator_policies[policy.name]=policy
                    if policy.library_power: self.used_math.add("operator_pow_"+kind)
                    if policy.library_powi: self.used_math.add("operator_powi_"+kind)
                    return _Expression(f"{policy.name}({self.convert(left,input_kind).code})",kind)
            if target in ("cython","cpp") and _floating(left.kind) and right.kind in ("int","bool"):
                from .operator_policy import integer_power_policy
                integer_types=[np.dtype(self.types[child.id]) for child in ast.walk(node.right)
                    if isinstance(child,ast.Name) and child.id in self.types
                    and np.dtype(self.types[child.id]).kind in "iu"]
                exponent_dtype="int64" if any(dtype.itemsize==8 for dtype in integer_types) else "int32"
                policy=integer_power_policy(left.kind,kind,exponent_dtype,tuple(self.reference_flags))
                self.operator_policies[policy.name]=policy
                self.used_math.add(("operator_powi_" if policy.library_powi else "operator_pow_")+kind)
                return _Expression(f"{policy.name}({left.code},{self.convert(right,'int').code})",kind)
            self.used_math.add("operator_pow_"+kind)
            code=f"{prefix}_operator_pow({a},{b})"
            if target=="numpy" and exponent is None and numpy_exponent_scalar is True:
                # ndarray ** scalar has five value-specific shortcuts. Compare
                # the original scalar before weak-scalar operand narrowing.
                scalar=self.convert(numpy_original_exponent or right,'float').code
                root=f"d_sqrt({self.convert(left,'float').code})"
                square_root=f"f_from_d({root})" if kind=="float32" else root
                choices=((0.,_literal(1.,kind)),(1.,a),(-1.,f"{prefix}_div({_literal(1.,kind)},{a})"),
                         (.5,square_root),(2.,f"{prefix}_mul({a},{a})"))
                for value,shortcut in reversed(choices):
                    code=f"(d_eq({scalar},{_literal(value,'float')}) ? {shortcut} : {code})"
            return _Expression(code,kind)
        self.used_math.add("operator_remainder_"+kind)
        operation="mod" if isinstance(node.op,ast.Mod) else "floordiv"
        # CPP's template promotes the division to double before applying the
        # higher-type return conversion, including a float32/integer pair.
        cpp_narrow=target=="cpp" and kind=="float32"
        if cpp_narrow:
            a,b=self.convert(left,"float").code,self.convert(right,"float").code
            prefix="d"
        arguments=f"{a},{b}"
        if target=="cython":
            self.used_math.add("operator_zero_division")
            arguments+=f",{self.zero_division_error_name},{self.zero_division_error_name}_kind"
        elif target not in ("numpy","cpp"):
            raise UnsupportedModelError(f"Native floating operators do not implement CPU target {target}")
        if target=="cython" and operation=="mod":
            from types import SimpleNamespace
            policy=SimpleNamespace(reference_codegen_target=target,reference_compile_args=self.reference_flags)
            operation+="_cython"
            if getattr(node,"_eon_augmented",False): operation+="_once"
            if _compiled_math_policy(policy)[1]: operation+="_fma"
        elif target=="cpp" and operation=="mod":
            # The default compiled template contracts x-y*floor(x/y).
            class Policy:
                reference_codegen_target=target
                reference_compile_args=self.reference_flags
            if _compiled_math_policy(Policy())[1]: operation+="_cpp_fma"
            else: operation+="_cpp"
        else: operation+="_"+target
        result_kind="float" if target=="cython" and kind=="float32" and isinstance(node.op,ast.FloorDiv) else kind
        code=f"{prefix}_operator_{operation}({arguments})"
        if cpp_narrow: code=f"f_from_d({code})"
        return _Expression(code,result_kind)

    def checked_division(self,code,denominator,kind):
        """Cython checks the source denominator before an optimized division.

        This also covers FloorDiv rewritten by Brian's symbolic state updater
        into floor(x/y), and a division enclosing a newly accepted power.
        Evaluation remains inside its original short-circuit branch.
        """
        if self.reference_target!='cython':return _Expression(code,kind)
        self.used_math.update(('operator_remainder_float','operator_zero_division'))
        divisor=self.convert(denominator,'float').code
        checked=f'{_prefix(kind)}_operator_checked_division({code},{divisor},{self.zero_division_error_name},{self.zero_division_error_name}_kind)'
        return _Expression(checked,kind)

    @staticmethod
    def convert(expression: _Expression, kind: str) -> _Expression:
        source, code = expression.kind, expression.code
        if source == kind:
            return expression
        if source=='int' and expression.integer_unsigned and kind in ('float','float32'):
            function='d_from_uint' if kind=='float' else 'f_operator_from_uint'
            return _Expression(f'{function}((ulong)({code}))',kind)
        if kind == "float":
            if source == "float32":
                return _Expression(f"f_to_d({code})", kind)
            return _Expression(f"d_from_int((long)({code}))", kind)
        if kind == "float32":
            if source in ("int", "bool"):
                return _Expression(f"f_from_int((long)({code}))", kind)
            return _Expression(f"f_from_d({_Renderer.convert(expression, 'float').code})", kind)
        if kind == "int":
            if source == "float32":
                return _Expression(f"d_to_int(f_to_d({code}))", kind)
            return _Expression(f"d_to_int({code})" if source == "float" else f"(long)({code})", kind)
        if source == "float32":
            return _Expression(f"((({code}) & 0x7fffffffu) != 0u)", kind)
        return _Expression(f"(!d_eq({code}, {_literal(0., 'float')}))" if source == "float" else
                           f"(({code}) != 0)", kind)

    def expression(self, source: str | ast.AST) -> _Expression:
        node = ast.parse(source, mode="eval").body if isinstance(source, str) else source
        if self.expression_random is not None and isinstance(node,ast.Call)and isinstance(node.func,ast.Name)and node.func.id==self.expression_random.function:
            return _Expression(self.expression_random.result,'float')
        try:
            if self.compiled_operator_expression(node):
                raise KeyError('compiled floating operator retains target semantics')
            value = _evaluate(node, self.values, self.reference_target)
        except KeyError:
            pass
        else:
            kind = _kind(np.asarray(value).dtype)
            return _Expression(_literal(value, kind), kind,
                integer_unsigned=np.asarray(value).dtype.kind=='u')
        if isinstance(node, ast.Name):
            if node.id not in self.types:
                raise UnsupportedModelError(f"Native equation compiler cannot resolve variable {node.id}")
            dtype=np.dtype(self.types[node.id]);code=self.code_name(node.id)
            if self.reference_target=='cython' and dtype==np.dtype('uint8'):
                from .operator_policy import compiled_integer_operand_dtype
                dtype=compiled_integer_operand_dtype(node,self.types,target=self.reference_target,
                    unsigned_char=self.compiled_unsigned_char)
                code=f"({_metal_type(dtype)})({code})"
            return _Expression(code, _kind(dtype),integer_unsigned=dtype.kind=='u')
        if isinstance(node, ast.UnaryOp):
            value = self.expression(node.operand)
            if isinstance(node.op, ast.Not):
                return _Expression(f"(!{self.convert(value, 'bool').code})", "bool")
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.USub):
                unsigned=value.integer_unsigned
                if value.kind=='int' and self.reference_target in ('cython','cpp'):
                    from .operator_policy import compiled_integer_operand_dtype
                    dtype=compiled_integer_operand_dtype(node,self.types,target=self.reference_target,
                        unsigned_char=self.compiled_unsigned_char)
                    unsigned=dtype is not None and dtype.kind=='u'
                return _Expression(f"{_prefix(value.kind)}_neg({value.code})" if _floating(value.kind) else f"(-{value.code})", value.kind,
                    integer_unsigned=unsigned)
        if isinstance(node, ast.BinOp):
            if (self.reference_target in ('cython','cpp') and isinstance(node.op,(ast.Add,ast.Sub))
                    and any(isinstance(child,ast.BinOp) and isinstance(child.op,(ast.Pow,ast.Mod,ast.FloorDiv))
                            for child in ast.walk(node))):
                products=[(node.left,node.right,True),(node.right,node.left,False)]
                for product,addend,product_left in products:
                    if not isinstance(product,ast.BinOp) or not isinstance(product.op,ast.Mult):continue
                    if isinstance(addend,ast.BinOp) and isinstance(addend.op,ast.Mult):continue
                    operands=[self.expression(product.left),self.expression(product.right),self.expression(addend)]
                    kind=_common_kind(_Expression('',_common_kind(*operands[:2])),operands[2])
                    if not _floating(kind):continue
                    operands=[operand if _floating(operand.kind) else self.convert(operand,kind) for operand in operands]
                    from .operator_policy import product_add_policy
                    policy=product_add_policy(tuple(operand.kind for operand in operands),kind,
                        '+' if isinstance(node.op,ast.Add) else '-',product_left,tuple(self.reference_flags))
                    self.operator_policies[policy.name]=policy
                    self.used_math.add('operator_remainder_float')
                    if any(operand.kind=='float32' for operand in operands):self.used_math.add('operator_remainder_float32')
                    return _Expression(f'{policy.name}('+','.join(operand.code for operand in operands)+')',kind)
            left, right = self.expression(node.left), self.expression(node.right)
            kind = _common_kind(left, right)
            if _floating(kind) and isinstance(node.op,(ast.Pow,ast.Mod,ast.FloorDiv)):
                return self.floating_operator(node,left,right)
            if _floating(kind) and self.reference_target=="numpy":
                kind=_kind(np.asarray(self.numpy_operand(node)).dtype)
            if kind == "int" and isinstance(node.op, ast.Div):
                kind = "float"
            if _floating(kind) and self.reference_target in ("cython","cpp") and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
                for constant_node,variable,constant_left in ((node.left,right,True),(node.right,left,False)):
                    if any(isinstance(child,ast.Name) for child in ast.walk(constant_node)):
                        continue
                    try:
                        constant=_evaluate(constant_node,self.values)
                    except KeyError:
                        continue
                    if not _floating(variable.kind):
                        continue
                    from .operator_policy import constant_binary_policy
                    symbol={ast.Add:"+",ast.Sub:"-",ast.Mult:"*",ast.Div:"/"}[type(node.op)]
                    policy=constant_binary_policy(variable.kind,kind,symbol,float(constant),constant_left,
                                                  tuple(self.reference_flags))
                    self.operator_policies[policy.name]=policy
                    code=f"{policy.name}({variable.code})"
                    return self.checked_division(code,right,kind) if isinstance(node.op,ast.Div) else _Expression(code,kind)
            denominator=right
            left, right = self.convert(left, kind).code, self.convert(right, kind).code
            operations = {ast.Add: ("d_add", "+"), ast.Sub: ("d_sub", "-"),
                          ast.Mult: ("d_mul", "*"), ast.Div: ("d_div", "/")}
            if type(node.op) in operations:
                function, symbol = operations[type(node.op)]
                function = function.replace("d_", _prefix(kind) + "_")
                code=f"{function}({left}, {right})" if _floating(kind) else f"({left} {symbol} {right})"
                if isinstance(node.op,ast.Div) and _floating(kind):
                    return self.checked_division(code,denominator,kind)
                unsigned=False
                if kind=='int' and self.reference_target in ('cython','cpp'):
                    from .operator_policy import compiled_integer_operand_dtype
                    dtype=compiled_integer_operand_dtype(node,self.types,target=self.reference_target,
                        unsigned_char=self.compiled_unsigned_char)
                    unsigned=dtype is not None and dtype.kind=='u'
                return _Expression(code,kind,integer_unsigned=unsigned)
            raise UnsupportedModelError(f"Native dynamic operator {type(node.op).__name__} is not implemented")
        if isinstance(node, ast.BoolOp):
            operands = [self.convert(self.expression(item), "bool").code for item in node.values]
            return _Expression("(" + (" && " if isinstance(node.op, ast.And) else " || ").join(operands) + ")", "bool")
        if isinstance(node, ast.Compare):
            operands = [self.expression(node.left)] + [self.expression(item) for item in node.comparators]
            comparisons = {ast.Gt: ("d_gt", ">"), ast.GtE: ("d_ge", ">="),
                           ast.Lt: ("d_lt", "<"), ast.LtE: ("d_le", "<="),
                           ast.Eq: ("d_eq", "=="), ast.NotEq: ("d_eq", "!=")}
            result = []
            for operation, left, right in zip(node.ops, operands, operands[1:]):
                kind = _common_kind(left, right)
                a, b = self.convert(left, kind).code, self.convert(right, kind).code
                function, symbol = comparisons[type(operation)]
                function = function.replace("d_", _prefix(kind) + "_")
                code = f"{function}({a}, {b})" if _floating(kind) else f"({a} {symbol} {b})"
                result.append(f"(!{code})" if _floating(kind) and isinstance(operation, ast.NotEq) else code)
            return _Expression("(" + " && ".join(result) + ")", "bool")
        if isinstance(node, ast.IfExp):
            condition = self.convert(self.expression(node.test), "bool").code
            a, b = self.expression(node.body), self.expression(node.orelse)
            kind = _common_kind(a, b)
            return _Expression(f"({condition} ? {self.convert(a, kind).code} : {self.convert(b, kind).code})", kind)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            args = [self.expression(arg) for arg in node.args]
            timed_array = self.values.get(name)
            if isinstance(timed_array, _TimedArray):
                dimensions = len(timed_array.array.shape)
                if len(args) != dimensions:
                    raise UnsupportedModelError(f"TimedArray {name} expects {dimensions} arguments")
                self.used_arrays.add(name)
                arguments = [timed_array.input_name, self.convert(args[0], "float").code]
                if dimensions == 2:
                    if ((timed_array.target == "numpy" and args[1].kind != "int") or
                            (timed_array.target == "cython" and _floating(args[1].kind))):
                        raise UnsupportedModelError(f"TimedArray {name} requires an integer column index")
                    if timed_array.target == "numpy":
                        arguments.extend([args[1].code, self.index_error_name])
                    else:
                        arguments.append(f"_eon_timed_int32({self.convert(args[1], 'float').code})" if _floating(args[1].kind) else
                                         f"(long)((int)({args[1].code}))")
                code = f"{timed_array.helper_name}({', '.join(arguments)})"
                if dimensions == 1 and self.shared_timed_calls is not None:
                    names = {item.id for item in ast.walk(node.args[0]) if isinstance(item, ast.Name)}
                    frozen = {key for key, value in self.values.items()
                              if not isinstance(value, _TimedArray) and np.asarray(value).size == 1}
                    if names <= frozen | {"t", "dt", "t_in_timesteps"}:
                        if code not in self.shared_timed_calls:
                            self.shared_timed_calls.append(code)
                if dimensions == 2 and timed_array.target == "numpy":
                    # NumPy evaluates all function arguments before clip/boolean
                    # expressions. Hoist the call so a C++ conditional cannot
                    # suppress an invalid access in an unselected argument.
                    while True:
                        temporary = f"_eon_timed_result_{self.temporary_index}"
                        self.temporary_index += 1
                        if temporary not in self.types and temporary not in self.values:
                            break
                    self.types[temporary] = np.float64
                    # Each NumPy ufunc/call runs over the complete vector
                    # before the next operation. Retain its callsite identity
                    # so an earlier failure on a later neuron wins globally.
                    self.runtime_errors.append(('IndexError', 'TimedArray column index is out of bounds'))
                    kind = len(self.runtime_errors)
                    flag = f"{temporary}_error"
                    arguments[-1] = flag
                    code = f"{timed_array.helper_name}({', '.join(arguments)})"
                    self.prelude.extend((f"bool {flag} = false;",
                        f"const ulong {temporary} = {code};",
                        f"if ({flag} && !{self.index_error_name}) {{ "
                        f"{self.index_error_name}_kind = {kind}; {self.index_error_name} = true; }}"))
                    code = temporary
                return self._timed_result(code, args)
            if name == "int" and len(args) == 1:
                return self.convert(args[0], "int")
            if name == "timestep" and len(args) == 2:
                a, dt = [self.convert(arg, "float").code for arg in args]
                return _Expression(f"d_to_int(d_div(d_add({a}, d_mul({_literal(.001, 'float')}, {dt})), {dt}))", "int")
            if name in ("abs", "sqrt") and len(args) == 1:
                value = args[0]
                if value.kind == "float32":
                    return _Expression(f"f_{name}({value.code})", "float32")
                if _floating(value.kind) or name == "sqrt":
                    return _Expression(f"d_{name}({self.convert(value, 'float').code})", "float")
                return _Expression(f"abs({value.code})", "int")
            if name in ("floor", "ceil") and len(args) == 1:
                value = self.convert(args[0], "float").code
                return _Expression(f"d_{name}({value})", "float")
            if (name in ('arcsin', 'arccos', 'arctan', 'log10', 'sinh', 'cosh', 'tanh',
                         'exp', 'log', 'expm1', 'log1p', 'exprel', 'sin', 'cos', 'tan')
                    and len(args) == 1):
                from .unary_math_policy import unary_helper
                single = args[0].kind == 'float32' and name != 'exprel'
                kind = 'float32' if single else 'float'
                if not single and name in ('sinh', 'cosh'):
                    paired = self.hyperbolic_pairs.expression(name, node.args[0],
                        self.convert(args[0], 'float').code, self.prelude)
                    if paired is not None:
                        return _Expression(paired, 'float')
                self.used_math.add(name + '_f32' if single else name)
                function = unary_helper(name, single, self.reference_target)
                argument = (args[0].code if single and name in (
                    'exp', 'log', 'expm1', 'log1p', 'sin', 'cos', 'tan') else
                    self.convert(args[0], kind).code)
                cached = self.expm1_results.get(ast.dump(node.args[0])) if name == 'exprel' else None
                if cached is not None:
                    return _Expression(f"_eon_stable_exprel_cached({argument}, {self.code_name(cached[0])})", 'float')
                return _Expression(f'{function}({argument})', kind)
            if name == "clip" and len(args) == 3:
                if args[0].kind == "float32":
                    value, low, high = [self.convert(arg, "float").code for arg in args]
                    return _Expression(f"(d_lt({value}, {low}) ? f_from_d({low}) : (d_gt({value}, {high}) ? f_from_d({high}) : {args[0].code}))", "float32")
                kind = "float" if any(_floating(arg.kind) for arg in args) else "int"
                value, low, high = [self.convert(arg, kind).code for arg in args]
                if kind == "float":
                    return _Expression(f"(d_lt({value}, {low}) ? {low} : (d_gt({value}, {high}) ? {high} : {value}))", kind)
                return _Expression(f"min(max({value}, {low}), {high})", kind)
            if name == "sign" and len(args) == 1:
                value = self.convert(args[0], "float").code
                zero = _literal(0., "float")
                if self.binary32:
                    return _Expression(f"(d_gt({value}, {zero}) ? 1L : (d_lt({value}, {zero}) ? -1L : 0L))", "int")
                return _Expression(f"(d_gt({value}, {zero}) ? {_literal(1., 'float')} : (d_lt({value}, {zero}) ? {_literal(-1., 'float')} : {value}))", "float")
            raise UnsupportedModelError(f"Native state-dependent function {name} is not implemented")
        raise UnsupportedModelError(f"Native equation compiler does not implement {ast.dump(node)}")

    def assignment(self, statement: Any, expression: str | None = None, declare: bool = True) -> str:
        self.types[statement.var] = statement.dtype
        expression = statement.expr if expression is None else expression
        self.prelude = []
        operator_assignment=statement.op in ("**=","%=","//=")
        if operator_assignment:
            node=ast.parse(f"{statement.var} {statement.op[:-1]} ({expression})",mode="eval").body
            node._eon_augmented=True
            rendered=self.expression(node)
        else:
            rendered=self.expression(expression)
        if statement.op not in ("=", ":=") and not operator_assignment:
            operation = {"+=": "d_add", "-=": "d_sub", "*=": "d_mul", "/=": "d_div"}.get(statement.op)
            if operation is None:
                raise UnsupportedModelError(f"Native assignment operator {statement.op} is not implemented")
            target = _Expression(self.code_name(statement.var), _kind(statement.dtype))
            kind = _common_kind(target, rendered)
            left, right = (self.convert(arg, kind).code for arg in (target, rendered))
            rendered = _Expression(
                f"{operation.replace('d_', _prefix(kind)+'_')}({left}, {right})" if _floating(kind) else
                f"({left} {statement.op[0]} {right})", kind)
        value = self.convert(rendered, _kind(statement.dtype)).code
        prefix = ""
        if statement.op == ":=" and declare and statement.var != "_cond":
            prefix = ("const " if statement.constant else "") + _metal_type(statement.dtype) + " "
        return "\n".join(self.prelude + [f"{prefix}{self.code_name(statement.var)} = {value};"])

    def statement(self, statement: Any, conditional_write: str | None = None) -> str:
        self.hyperbolic_pairs.statement = statement
        branches = getattr(statement, "boolean_simplified_expressions", None)
        if branches or conditional_write:
            # A conditional assignment need not execute, so no cached value may
            # escape into or out of it based on one statically selected branch.
            self.expm1_results.clear()
        if branches:
            self.types[statement.var] = statement.dtype
            lines = []
            if statement.op == ":=" and statement.var != "_cond":
                lines.append(f"{_metal_type(statement.dtype)} {self.code_name(statement.var)};")
            for index, (conditions, expression) in enumerate(branches.items()):
                condition = " && ".join(self.code_name(name) if value else f"!{self.code_name(name)}" for name, value in conditions)
                lines.append(f"{'if' if index == 0 else 'else if'} ({condition}) {{")
                lines.append("    " + self.assignment(statement, expression, declare=False))
                lines.append("}")
            code = "\n".join(lines)
        else:
            code = self.assignment(statement)
        if conditional_write:
            code = f"if ({self.code_name(conditional_write)}) {{\n" + "\n".join("    " + line for line in code.splitlines()) + "\n}"
        # A cached result and every name in its argument must still have the
        # same value. Invalidate after rendering the RHS (which sees old state).
        self.expm1_results = {key: entry for key, entry in self.expm1_results.items()
                              if statement.var != entry[0] and statement.var not in entry[1]}
        if not branches and not conditional_write and statement.op in ("=", ":="):
            root = ast.parse(statement.expr, mode="eval").body
            if (isinstance(root, ast.Call) and isinstance(root.func, ast.Name)
                    and root.func.id == "expm1" and len(root.args) == 1
                    and np.dtype(statement.dtype) == np.dtype(np.float64)):
                dependencies = {node.id for node in ast.walk(root.args[0]) if isinstance(node, ast.Name)}
                if statement.var not in dependencies:
                    self.expm1_results[ast.dump(root.args[0])] = (statement.var, dependencies)
        return code


def _block(lines: list[str]) -> str:
    return "{\n" + "\n".join("    " + line for line in "\n".join(lines).splitlines()) + "\n}"


def _static_lif(model: ResolvedModel, reset_statements: tuple[Any, ...]) -> bool:
    """Canonical autonomous dynamics may carry independent parameter state.

    Extra parameters cannot advance on their own. Their values may change on
    resets, but cannot affect the canonical dynamics, threshold or refractory
    predicate. The later equilibrium proof additionally checks the actual
    lowered persistent writes, so an extra updater cannot be skipped silently.
    """
    if any(statement.var in {"rfc", "lastspike", "not_refractory"}
           for statement in reset_statements):
        return False
    if _expression_key(model.eqs) == _expression_key(DEFAULT_EQUATIONS):
        return True
    import brian2 as b

    equations, canonical = b.Equations(model.eqs), b.Equations(DEFAULT_EQUATIONS)
    for name, expected in canonical.items():
        if name not in equations:
            return False
        actual = equations[name]
        if (actual.type != expected.type or actual.var_type != expected.var_type
                or actual.dim != expected.dim or set(actual.flags) != set(expected.flags)):
            return False
        if expected.expr is not None and _expression_key(actual.expr.code) != _expression_key(expected.expr.code):
            return False
    extras = set(equations) - set(canonical)
    if any(equations[name].type != "parameter"
           or set(equations[name].flags) - {"constant"} for name in extras):
        return False
    for statement in reset_statements:
        if statement.var not in extras:
            continue
        nodes = tuple(ast.walk(ast.parse(statement.expr, mode="eval")))
        if (any(isinstance(node, ast.Name) and node.id in {"t", "t_in_timesteps"} for node in nodes)
                or any(isinstance(node, ast.Call)
                       and (not isinstance(node.func, ast.Name) or node.func.id not in _PREP_FUNCTIONS)
                       for node in nodes)):
            return False
    dependencies = {"rfc", "lastspike", "not_refractory", "t", "dt", "t_in_timesteps"}
    for source in [model.eq_th, *(equation.expr.code for equation in canonical.values()
                                  if equation.expr is not None)]:
        dependencies.update(node.id for node in ast.walk(ast.parse(source, mode="eval"))
                            if isinstance(node, ast.Name))
    return not (extras & dependencies)


def _compiled_lif_arithmetic(model: ResolvedModel, program: CompiledModel,
                             types: Mapping[str, Any], values: Mapping[str, Any]) -> CompiledModel:
    """Preserve the independently audited arm64 Clang canonical LIF update.

    Custom thresholds and resets do not alter Brian's canonical linear state
    kernel. Additional parameter states are passive in that kernel as well.
    The compiled target's default fast math contracts two voltage FMAs and
    reassociates the scalar coefficient and refractory timestep calculation.
    """
    if model.reference_codegen_target not in ("cython", "cpp"):
        return program
    if 'state' in program.disabled_stages:return program
    if (platform.system() != "Darwin" or platform.machine() != "arm64"
            or not _compiled_math_policy(model)[1]):
        return program
    import brian2 as b
    equations = b.Equations(model.eqs)
    canonical = b.Equations(DEFAULT_EQUATIONS)
    if (set(equations.diff_eq_names) != set(canonical.diff_eq_names)
            or equations["g"].expr != canonical["g"].expr
            or any(equations[name].flags != canonical[name].flags for name in canonical.diff_eq_names)):
        return program
    forcing = None
    if _expression_key(str(equations["v"].expr)) != _expression_key(str(canonical["v"].expr)):
        actual = ast.parse(str(equations["v"].expr), mode="eval").body
        expected = ast.parse(str(canonical["v"].expr), mode="eval").body
        # This audited extension retains the same LIF dynamics and adds a
        # TimedArray voltage drive. Other equation families keep their existing
        # lowering; this is not a general fast-math algebra transformation.
        if not (isinstance(actual, ast.BinOp) and isinstance(actual.op, ast.Div)
                and ast.dump(actual.right) == ast.dump(expected.right)
                and isinstance(actual.left, ast.BinOp) and isinstance(actual.left.op, ast.Add)
                and ast.dump(actual.left.left) == ast.dump(expected.left)):
            return program
        term = actual.left.right
        if not isinstance(term, ast.BinOp) or not isinstance(term.op, ast.Mult):
            return program
        for call, unit in ((term.left, term.right), (term.right, term.left)):
            if (isinstance(unit, ast.Name) and unit.id == "mV" and isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name) and isinstance(values.get(call.func.id), _TimedArray)):
                forcing = call
                break
        if forcing is None:
            return program
    dt, tau, tm = np.float64(model.dt_s), np.float64(model.tau), np.float64(model.t_mbr)
    if tau <= 0 or tm <= 0 or tau == tm:
        return program
    from .binary64 import literal
    rest, rest_code = "v_0", ""
    if forcing is not None:
        renderer = _Renderer(dict(types), dict(values), reference_target=model.reference_codegen_target or "numpy",
                             code_names=variable_code_names(types))
        sample_code = renderer.convert(renderer.expression(forcing), "float").code
        unit = renderer.convert(renderer.expression("mV"), "float").code
        rest = "_eon_compiled_rest"
        while rest in types or rest in values:
            rest += "_"
        sample = "_eon_compiled_drive"
        while sample in types or sample in values or sample == rest:
            sample += "_"
        rest_code = (f"const ulong {sample} = {sample_code};\n"
                     f"const ulong {rest} = d_fma({unit}, {sample}, v_0);")
    def c_exp(value):
        try:
            return np.float64(math.exp(float(value)))
        except OverflowError:
            return np.float64(np.inf)
    if model.reference_codegen_target == "cpp":
        # Standalone embeds tau/t_mbr as constants. Clang first folds their
        # reciprocals, then factors decay_v around the full voltage sum.
        tau_reciprocal, tm_reciprocal = np.float64(1.)/tau, np.float64(1.)/tm
        decay_g, decay_v = c_exp(-dt*tau_reciprocal), c_exp(-dt*tm_reciprocal)
        grow_g, grow_v = c_exp(dt*tau_reciprocal), c_exp(dt*tm_reciprocal)
        coefficient = (decay_g*(tau/(tm-tau)))*(grow_g-grow_v)
        if forcing is not None and len(values[forcing.func.id].array.shape) == 2:
            drive_voltage = f"d_mul({sample}, {unit})"
            driven = f"d_fma(d_fma({literal(coefficient)}, g, v), {literal(decay_v)}, {drive_voltage})"
            next_voltage = (f"d_add(d_fma(d_sub(d_neg(v_0), {drive_voltage}), {literal(decay_v)}, "
                            f"{driven}), v_0)")
        else:
            next_voltage = (f"d_fma(d_add(d_fma({literal(coefficient)}, g, d_neg({rest})), v), "
                            f"{literal(decay_v)}, {rest})")
    else:
        decay_g, decay_v = c_exp(-dt/tau), c_exp(-dt/tm)
        grow_g, grow_v = c_exp(dt/tau), c_exp(dt/tm)
        coefficient = ((decay_g*decay_v)*(np.float64(1.)/(tm-tau)))*((grow_g-grow_v)*tau)
        if forcing is not None and len(values[forcing.func.id].array.shape) == 2:
            # Per-neuron calls stay in Brian's vector statement. Clang keeps
            # v_0 as the final addition instead of moving it into the scalar
            # effective-rest term used for the one-dimensional call.
            next_voltage = (f"d_add(d_fma(d_sub(v, {rest}), {literal(decay_v)}, "
                            f"d_fma({sample}, {unit}, d_mul({literal(coefficient)}, g))), v_0)")
        else:
            next_voltage = (f"d_fma(d_sub(v, {rest}), {literal(decay_v)}, "
                            f"d_fma({literal(coefficient)}, g, {rest}))")
    offset, reciprocal = np.float64(.001)*dt, np.float64(1.)/dt
    refractory = ("not_refractory = d_to_int(d_mul(d_sub(d_add(t, " + literal(offset)
                  + "), lastspike), " + literal(reciprocal)
                  + ")) >= d_to_int(d_mul(d_add(rfc, " + literal(offset)
                  + "), " + literal(reciprocal) + "));")
    policy = "cpp_fast" if model.reference_codegen_target == "cpp" else "cython_fast"
    if (model.reference_codegen_target == "cpp" and forcing is not None
            and len(values[forcing.func.id].array.shape) == 2):
        refractory = (f"not_refractory = d_to_int(d_div(d_sub(d_add(t, {literal(offset)}), lastspike), dt)) "
                      f">= d_to_int(d_div(d_add(rfc, {literal(offset)}), dt));")
        policy = "cpp_vector_drive"
    if model.legacy_refractory_timing:
        # Compiled voltage arithmetic is independent of the release policy.
        # The legacy boundary is strict elapsed time, with no timestep cache.
        refractory = program.refractory_code
    dynamics = f"""
    {{
        {rest_code}
        if(not_refractory) {{
            ulong eon_next_v = {next_voltage};
            g = d_mul({literal(decay_g)}, g);
            v = eon_next_v;
        }}
    }}
    """
    if (model.reference_codegen_target == "cython" and forcing is None
            and model.analysis is not None and model.analysis.temporary_dtype == "float32"):
        # A TimedArray in threshold/reset does not change the state updater's
        # temporary precision. Share the independently audited original-API
        # lowering instead of replacing its binary32 assignments with doubles.
        from .brian_codegen import _cython_lif_binary32_dynamics
        dynamics = _cython_lif_binary32_dynamics(model)
    return replace(program, state_code=_block([refractory, dynamics]),
                   refractory_code=refractory, dynamics_code=dynamics,
                   refractory_policy=policy)


def _refractory_elapsed_threshold(model: ResolvedModel, reset_statements: tuple[Any, ...]) -> float | None:
    """Invert the exact rounded timestep predicate on positive binary64 bits.

    For positive dt, each rounded operation in (elapsed + .001*dt)/dt is
    monotone. Truncation is also monotone within signed-int64 range. Therefore
    its comparison with a fixed integer has a single boundary. Binary search
    finds that boundary over the actual binary64 values, retaining every
    rounding operation used by Brian rather than replacing it algebraically.
    """
    if (model.dt_s <= 0 or model.legacy_refractory_timing or np.ndim(model.t_rfc) != 0
            or isinstance(model.t_rfc, str) or not _static_lif(model, reset_statements)):
        return None
    dt, rfc = np.float64(model.dt_s), np.float64(model.t_rfc)
    offset = np.float64(.001) * dt
    with np.errstate(over="ignore", invalid="ignore"):
        if model.state_dtype == "float32":
            rfc = np.float64(np.float32(rfc))
            if not np.isfinite(rfc):
                return None
        if model.state_dtype == "float32" and model.float32_codegen == "numpy":
            # timestep(rfc, dt) sees a binary32 vector and a double scalar.
            # Use the independent helper's actual vector promotion/rounding.
            from brian2.core.functions import timestep
            quotient = timestep(np.array([rfc], np.float32), dt)[0]
        else:
            quotient = (rfc + offset) / dt
    if not np.isfinite(quotient) or not -(1 << 63) <= quotient < (1 << 63):
        return None
    refractory_steps = int(quotient)
    if refractory_steps <= 0:
        return 0.
    low = 0
    upper = rfc if model.state_dtype == "float64" else np.float64(refractory_steps+1)*dt
    if not np.isfinite(upper):
        return None
    high = struct.unpack("Q", struct.pack("d", upper))[0]
    while low < high:
        middle = (low + high) // 2
        elapsed = np.float64(struct.unpack("d", struct.pack("Q", middle))[0])
        quantized = math.trunc((elapsed + offset) / dt)
        if quantized >= refractory_steps:
            high = middle
        else:
            low = middle + 1
    return struct.unpack("d", struct.pack("Q", low))[0]


def _prove_equilibrium(model: ResolvedModel, inventory: Mapping[str, CompiledVariable],
                       constants: Mapping[str, Any], state_statements: tuple[Any, ...],
                       reset_statements: tuple[Any, ...]) -> Mapping[str, Any] | None:
    """Conservatively prove the canonical LIF rest point, using emitted order.

    This is a preparation proof of one exact fixed point, not a tolerance test.
    Restricting the equations and reset writes makes its assumptions visible
    without claiming to establish equilibrium for arbitrary model dynamics.
    """
    if not _static_lif(model, reset_statements):
        return None
    numpy32 = model.state_dtype == "float32" and model.float32_codegen == "numpy"
    if model.state_dtype == "float32" and (not numpy32 or not model.float32_optimise):
        return None
    initial = ({"v": np.array([model.v_0], np.float32), "g": np.zeros(1, np.float32)}
               if numpy32 else {"v": np.float64(model.v_0), "g": np.float64(0.)})
    values = dict(constants)
    values.update(initial)
    values["not_refractory"] = np.bool_(True)
    persistent_writes = set()
    for statement in state_statements:
        if statement.scalar or statement.var == "not_refractory":
            continue
        variable = inventory.get(statement.var)
        if variable is not None and variable.persistent:
            persistent_writes.add(statement.var)
            if variable.conditional_write and not values[variable.conditional_write]:
                continue
        expression = statement.expr
        branches = getattr(statement, "boolean_simplified_expressions", None)
        if branches:
            expression = next((expr for conditions, expr in branches.items()
                               if all(bool(values[name]) == value for name, value in conditions)), None)
            if expression is None:
                return None
        try:
            result = _evaluate(ast.parse(expression, mode="eval").body, values)
        except (KeyError, UnsupportedModelError):
            return None
        if statement.op not in ("=", ":="):
            operation = {"+=": np.add, "-=": np.subtract, "*=": np.multiply, "/=": np.divide}.get(statement.op)
            if operation is None:
                return None
            result = operation(values[statement.var], result)
        values[statement.var] = (result if numpy32 and variable is None else _cast_value(result, statement.dtype))
    if persistent_writes != set(initial):
        return None
    if any(not np.isfinite(values[name]) or values[name].tobytes() != value.tobytes()
           for name, value in initial.items()):
        return None
    return MappingProxyType({name: value[0] for name, value in initial.items()} if numpy32 else initial)


@_preparation_device()
def compile_model(model: ResolvedModel, *, neurons=None, native_precision_limbs=0,
                  shared_expression_inputs=False) -> CompiledModel:
    if neurons is None:
        neurons=(model.integer_policy.vector_length if model.integer_policy is not None
                 and model.integer_policy.vector_length is not None else 1)
    with model_preferences(model):
        return _compile_model(model, neurons=neurons,native_precision_limbs=native_precision_limbs,
                              shared_expression_inputs=shared_expression_inputs)


def _compile_model(model: ResolvedModel, *, neurons=1, native_precision_limbs=0,
                   shared_expression_inputs=False) -> CompiledModel:
    """Prepare typed source fragments; never run a CPU simulation timestep.

    The caller declares/loads persistent variables plus ``t, dt, i, N`` and
    ``bool _cond``. Emit scalar_code once in the per-neuron kernel, then each
    stage block at its Brian schedule position. Threshold side effects
    (``lastspike=t; not_refractory=false``) belong to the caller.

    Bind ``input_arrays`` through their generated pointer names, append
    ``header_code`` after the binary64 helpers, and declare ``index_error_name``
    as a sticky thread-local bool when present. Reset scalar preparation runs
    every timestep; only ``reset_vector_code`` is conditional on a spike.
    """
    if model.pending_namespace:
        raise UnsupportedModelError("Native namespace callbacks remain unresolved: " + ", ".join(model.pending_namespace))
    if model.state_dtype not in ("float64", "float32"):
        raise UnsupportedModelError(f"Native equation compiler does not implement state dtype {model.state_dtype}")
    import brian2 as b
    from brian2.codegen.translation import analyse_identifiers, make_statements
    from brian2.core.variables import ArrayVariable, Subexpression
    from brian2.input.timedarray import _find_K

    equations = b.Equations(model.eqs)
    dtype = {equation.varname: np.dtype(
        model.integer_dtype if equation.var_type == "integer" else
        bool if equation.var_type == "boolean" else model.state_dtype,
    ) for equation in equations.values()}
    if model.analysis is not None:
        dtype.update({name: np.dtype(variable.dtype) for name, variable in model.analysis.variables.items()})
    group = b.NeuronGroup(
        1, equations, method="linear", threshold=model.eq_th, reset=model.eq_rst,
        refractory="rfc", namespace=model.brian_namespace(), dt=model.dt_s * b.second,
        dtype=dtype, name="mlx_model_compiler*",
    )
    group._network = "mlx-model-compiler-only"
    # Brian overrides lastspike to clock precision for dictionary dtype input.
    # Preserve the independently analyzed scalar/original-group dtype here;
    # this temporary group is never executed or used as runtime state storage.
    group.variables["lastspike"].dtype = np.dtype(
        model.analysis.variables["lastspike"].dtype if model.analysis is not None else model.state_dtype)
    if model.analysis is None:
        # Brian's default helper consults the current legacy preference. This
        # model has already snapshotted timing and always uses rfc, so
        # bind that one helper on the preparation-only updater, without
        # mutating Brian's process-wide preferences or redoing its integrator.
        group.state_updater._get_refractory_code = lambda run_namespace: model.refractory_abstract
    inventory: dict[str, CompiledVariable] = {}
    types: dict[str, Any] = {}
    external = {"t", "dt", "t_in_timesteps", "i", "N", "_spikespace"}
    for name, variable in group.variables.items():
        if not hasattr(variable, "dtype") or name == "_spikespace":
            continue
        if isinstance(variable, Subexpression):
            continue
        persistent = isinstance(variable, ArrayVariable) and name not in external
        condition = getattr(variable, "conditional_write", None)
        dtype = np.dtype(variable.dtype).name
        metal_type=(('uchar' if model.integer_policy is not None and model.integer_policy.unsigned_char else 'char')
                    if dtype=='bool' and model.reference_codegen_target=='cpp' else _metal_type(variable.dtype))
        inventory[name] = CompiledVariable(
            name, dtype, metal_type, variable.scalar, variable.constant,
            variable.read_only, condition.name if condition is not None else None, persistent,
        )
        types[name] = variable.dtype
    has_shared = any(v.persistent and v.scalar for v in inventory.values())
    has_scalar_stages = has_shared or group.subexpression_updater is not None
    values: dict[str, Any] = {"dt": np.float64(model.dt_s)}

    # Allocate names against the model's entire namespace, including unreferenced
    # entries, so a user identifier never shadows a generated pointer/helper.
    reserved = set(types) | set(model.source_namespace)

    def unique_name(stem):
        candidate = stem
        while candidate in reserved:
            candidate += "_"
        reserved.add(candidate)
        return candidate

    from .expression_fixed import plan as random_plan,GeneralRandomPlan
    expression_random=random_plan(model,equations,unique_name('_eon_expression_random'))
    if expression_random is not None and shared_expression_inputs:
        expression_random=GeneralRandomPlan(expression_random.target)
    general_random=isinstance(expression_random,GeneralRandomPlan)
    if model.numpy_policy_ir:
        expression_random=GeneralRandomPlan('numpy')
        general_random=True
    random_statements={}

    index_error_name = unique_name("_eon_timed_index_error")
    zero_division_error_name = unique_name("_eon_operator_zero_error")
    timed_arrays: dict[str, _TimedArray] = {}
    timed_snapshots = {name: (number, snapshot)
                       for number, (name, snapshot) in enumerate(sorted(model.source_namespace.items()))
                       if isinstance(snapshot, TimedArrayValue)}

    from brian2.utils.stringtools import get_identifiers
    referenced=equations.identifiers | get_identifiers(model.eq_th) | get_identifiers(model.eq_rst)
    if (expression_random is not None and expression_random.target=='numpy'
            and any(name in referenced and snapshot.values.ndim==2
                    for name,(_,snapshot) in timed_snapshots.items())):
        expression_random=GeneralRandomPlan('numpy')
        general_random=True

    def register_timed_array(name):
        number, snapshot = timed_snapshots[name]
        if (snapshot.values.ndim not in (1, 2) or not all(snapshot.values.shape)
                or not np.isfinite(snapshot.dt_s) or snapshot.dt_s <= 0):
            raise UnsupportedModelError(f"TimedArray {name} requires nonempty 1D/2D values and positive finite dt")
        factor = _find_K(model.dt_s, snapshot.dt_s)
        target = model.reference_codegen_target
        if target not in ("numpy", "cython", "cpp"):
            raise UnsupportedModelError(f"Native TimedArray does not implement reference target {target}")
        # Brian embeds compiled TimedArray dt with exactly 18 decimal places.
        interval = snapshot.dt_s if target == "numpy" else float(f"{snapshot.dt_s:.18f}")
        math_policy = _compiled_math_policy(model)[0]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            reciprocal_value = (np.float64(1.) / np.float64(interval / factor)) / np.float64(factor)
        values_array = np.array(snapshot.values, dtype=np.float64, order="C", copy=True)
        values_array = np.frombuffer(values_array.tobytes(), np.float64).reshape(values_array.shape)
        compiled = CompiledInputArray(name, "float64", "ulong", values_array.shape, values_array)
        timed_arrays[name] = _TimedArray(unique_name(f"_eon_timed_values_{number}"),
                                        unique_name(f"_eon_timed_sample_{number}"), compiled,
                                        np.float64(interval / factor), factor, target, math_policy,
                                        reciprocal_value, np.float64(.5) / np.float64(factor))
        values[name] = timed_arrays[name]

    used_arrays: set[str] = set()
    used_math: set[str] = set()
    operator_policies: dict[str,Any] = {}
    runtime_errors: list[tuple[str, str]] = []
    half_input_arrays: dict[str,CompiledInputArray] = {}
    half_input_names={name:unique_name('_eon_numpy_half_'+name)
                      for name in ('exp','log','expm1','log1p')}
    runtime_error_writes={}
    disabled_stages=[]
    compiled_integer_stages=[]
    shared_timed_calls: list[str] = []
    timed_cache_name = unique_name("_eon_timed_cache")
    constants: dict[str, Any] = {}
    integer_cache_declarations=[]
    integer_read_only=()
    if model.integer_policy is not None and model.integer_policy.target=='cpp_standalone' and not has_scalar_stages:
        import textwrap
        reset_writes={node.id for statement in ast.walk(ast.parse(textwrap.dedent(model.eq_rst)))
            if isinstance(statement,(ast.Assign,ast.AugAssign))
            for target in (statement.targets if isinstance(statement,ast.Assign) else [statement.target])
            for node in ast.walk(target) if isinstance(node,ast.Name)}
        integer_read_only=({name for name,equation in equations.items() if equation.type=='parameter'}
            -reset_writes-{'v','g','rfc','lastspike','not_refractory'})|{'i','N'}
    scalar_lines: list[str] = []
    state_constants = None
    blocks: dict[str, str] = {}
    refractory_code = ""
    dynamics_code = ""
    statements: dict[str, tuple[Any, ...]] = {}
    reset_scalar_code = ""
    reset_vector_code = ""
    compiled_equilibrium = None
    compiled_refractory_safe = True
    compiled64_operator_stages=set()
    collective_reset=False
    scalar_stages = {}
    scalar_statements = {}
    cpu_refractory_guard = False
    stages = ["state", "threshold", "reset"]
    runners = [group.state_updater, group.thresholder["spike"], group.resetter["spike"]]
    if has_scalar_stages:
        stages.insert(0, "subexpression")
        runners.insert(0, group.subexpression_updater)
    for stage, runner in zip(stages, runners):
        if runner is None:
            scalar_stages[stage] = ScalarStage("", (), "", "")
            blocks[stage] = ""
            continue
        if model.analysis is None or stage == "subexpression":
            runner.update_abstract_code(run_namespace={})
            abstract_code = runner.abstract_code
        else:
            abstract_code = getattr(model.analysis, stage + "_abstract")
        additional = getattr(runner, "variables", None)
        known_variables = dict(group.variables)
        if additional is not None:
            known_variables.update(additional)
        _, known, unknown = analyse_identifiers(abstract_code, known_variables, recursive=True)
        resolved = group.resolve_all(known | unknown, run_namespace={}, additional_variables=additional,
                                     user_identifiers=set())
        for name in sorted(resolved):
            if name in timed_snapshots and name not in timed_arrays and isinstance(resolved[name], b.TimedArray):
                register_timed_array(name)
        for name, variable in resolved.items():
            if not hasattr(variable, "dtype"):
                continue
            types[name] = variable.dtype
            if name in external or isinstance(variable, ArrayVariable) or not variable.scalar or not variable.constant:
                continue
            try:
                value = variable.get_value()
            except TypeError:  # auxiliary variables have no preparation value
                continue
            array = np.asarray(value)
            if array.size != 1:
                raise UnsupportedModelError(f"Native namespace array {name} is not implemented")
            # Brian exposes namespace Constants as Python values. NumPy uses
            # that value directly; Cython also derives declarations from it.
            # CPP retains the dtype recorded on the variable itself.
            target = model.reference_codegen_target or (
                model.analysis.codegen_target if model.analysis is not None else None)
            if target in ("numpy", "cython"):
                from brian2.core.variables import Constant
                if isinstance(variable, Constant):
                    types[name] = array.dtype
            effective_dtype = types[name]
            value = ((value if isinstance(variable, Constant) else array.reshape(-1)[0]) if target == "numpy" else
                     _cast_value(array.reshape(-1)[0], effective_dtype))
            values[name] = value
            if name not in constants:
                constants[name] = value
                scalar_lines.append(f"const {_metal_type(effective_dtype)} {variable_code_name(name)} = {_literal(value, _kind(effective_dtype))};")
                inventory[name] = CompiledVariable(name, np.dtype(effective_dtype).name,
                                                  _metal_type(effective_dtype), True, True, True, None, False)
        temporary_dtype = (model.analysis.temporary_dtype if model.analysis is not None
                           and model.analysis.temporary_dtype else model.state_dtype)
        frozen = (model.analysis.typed_statements if model.analysis is not None else None)
        if frozen is not None and stage in frozen:
            scalar, vector = (list(items) for items in copy.deepcopy(frozen[stage]))
        else:
            optimise=(model.reference_optimise if model.reference_optimise is not None else
                      model.state_dtype!="float32" or model.float32_optimise)
            scalar, vector = make_statements(abstract_code, resolved, np.dtype(temporary_dtype),
                                             optimise=optimise, blockname=stage)
        if has_scalar_stages:
            from collections import defaultdict
            from types import SimpleNamespace
            from brian2.codegen.generators.base import CodeGenerator
            indices = defaultdict(lambda: "_idx", {name:"0" for name,var in resolved.items() if getattr(var,"scalar",False)})
            checker = SimpleNamespace(variables=resolved,variable_indices=indices,
                allows_scalar_write=stage in ("state","subexpression"))
            CodeGenerator.array_read_write(checker,scalar+vector)
        code_names = variable_code_names(set(types) | {stmt.var for stmt in scalar+vector})
        ordered_numpy=(general_random and expression_random.target=='numpy'
                       and any(action!='ignore' for _,action in model.numpy_error_actions))
        if model.numpy_policy_ir or ordered_numpy or general_random and any(isinstance(node,ast.Call)and isinstance(node.func,ast.Name)
                and node.func.id in('rand','randn','poisson')
                for statement in scalar+vector for node in ast.walk(ast.parse(statement.expr,mode='eval'))):
            random_statements[stage]=(tuple(scalar),tuple(vector))
            statements[stage]=tuple(scalar)+tuple(vector)
            blocks[stage]=''
            if has_scalar_stages:scalar_stages[stage]=ScalarStage('',(),'', '')
            continue
        code_names.update({name:var.code_name for name,var in inventory.items()})
        # Brian allocates _lio names independently for each runner. Their
        # folded values can differ, even though the Metal scalar scope is
        # shared by state, threshold and reset. Keep the central encoding.
        for statement in scalar:
            if statement.var.startswith('_lio_') and statement.var not in inventory:
                code_names[statement.var] = variable_code_name(
                    unique_name('_eon_phase_'+stage+'_'+statement.var))
        if has_scalar_stages:
            scalar_statements[stage] = tuple(scalar)
        numpy32 = model.state_dtype == "float32" and model.float32_codegen == "numpy"
        half_numpy=(model.reference_codegen_target=="numpy" and any(
            token[1] in {"exp","log","expm1","log1p","exprel","sqrt","log10","sinh","cosh","tanh","arctan","arcsin","arccos","sin","cos","tan"} for source in (model.eqs,model.eq_th,model.eq_rst)
            for token in _expression_key(source)))
        numpy_vectors = model.reference_codegen_target == "numpy"
        numpy_temporaries = model.analysis is not None and model.analysis.codegen_target == "numpy"
        if numpy_temporaries:
            # Brian's NumPy generator emits plain assignments for auxiliary
            # values. Their dtype follows NumPy operations, with no C-style
            # cast to the generated temporary declaration's dtype.
            vector = [copy.copy(statement) for statement in vector]
            for statement in vector:
                if statement.var not in inventory and np.dtype(statement.dtype).kind == "f":
                    statement.dtype = np.dtype(model.state_dtype)
        from .expression_fixed import checked_threshold
        fixed_float_comparison=(stage=='threshold' and
            checked_threshold(model,expression_random,shared=has_scalar_stages))
        # The proved comparison consumes an already-drawn float and an exact
        # literal. Sending its original randn() call to the isolated integer
        # frontend can produce NoCode and incorrectly disable the threshold.
        nondefault_cython_state=(stage=='state' and model.state_dtype=='float64'
            and model.reference_codegen_target=='cython'
            and not _compiled_math_policy(model)[1])
        frontend_policy=model.integer_policy
        if nondefault_cython_state and frontend_policy is None:
            from .integer_policy import snapshot_integer_policy
            # The frontend also owns floating-only Cython blocks. Preserve the
            # resolved flags rather than the preparation process's current flags.
            frontend_policy=replace(snapshot_integer_policy('cython'),flags=model.reference_compile_args)
        if (not fixed_float_comparison and frontend_policy is not None
                and frontend_policy.target in {'cython','cpp_standalone'}):
            from .compiled_integer import selected,lower as lower_integer,lower_shared,reset_scalars_are_unobservable
            persistent={name for name,var in inventory.items() if var.persistent}
            integer_state=any(var.persistent and np.dtype(var.dtype).kind in 'iu' for var in inventory.values())
            strict_state=stage=='state' and integer_state and '-ffast-math' not in frontend_policy.flags
            # The Cython division/error CFG changes coefficient reassociation
            # for independent refractory-write guards. Keep its actual frontend
            # for the complete state block, including the release predicate.
            refractory_frontend=(stage=='state' and model.state_dtype=='float64'
                and temporary_dtype=='float64' and frontend_policy.target=='cython'
                and _lif_refractory_write_variant(model.eqs))
            if nondefault_cython_state or strict_state or refractory_frontend or selected(
                    scalar+vector,types,persistent,target=frontend_policy.target):
                compiled_integer_stages.append(stage)
                # Keep the two existing floating operator codes stable when a
                # different phase shares this catalog with integer failures.
                if not runtime_errors:
                    runtime_errors.extend([('ZeroDivisionError','float divmod()'),
                                           ('ZeroDivisionError','float division')])
                guards={name:var.conditional_write for name,var in inventory.items()
                        if var.conditional_write and stage=='state'}
                unobservable_reset=(stage=='reset' and frontend_policy.target=='cython'
                    and reset_scalars_are_unobservable(scalar,persistent))
                selected_name=unique_name('_eon_integer_selected') if stage=='reset' and not unobservable_reset else None
                from .operator_graph import Binary64Registry
                function_registry=Binary64Registry(types,used_math,zero_division_error_name,
                    values=values,used_arrays=used_arrays,index_error_name=index_error_name,
                    target=model.reference_codegen_target)
                arguments=dict(
                    guards=guards,errors=runtime_errors,used_math=used_math,
                    operator_policies=operator_policies,error_name=zero_division_error_name,selected_name=selected_name,
                    function_registry=function_registry,original_cpp=(group,runner,abstract_code),
                    code_names=code_names,native_precision_limbs=native_precision_limbs,
                    read_only=integer_read_only,cache_declarations=integer_cache_declarations if integer_read_only else None)
                lowered_integer=(lower_shared(frontend_policy,scalar,vector,types,values,persistent,**arguments)
                    if has_scalar_stages else lower_integer(frontend_policy,scalar+vector,types,values,persistent,**arguments))
                from .compiled_integer_frontend import NoCode
                if isinstance(lowered_integer,NoCode):
                    disabled_stages.append(stage)
                    if has_scalar_stages:scalar_stages[stage]=ScalarStage("",(),"","")
                    blocks[stage]=_block(['// Brian Cython has no compiled module for phase: '+lowered_integer.key])
                else:
                    if has_scalar_stages:
                        scalar_stages[stage]=lowered_integer
                        lowered_integer=lowered_integer.scalar_code+'\n'+lowered_integer.vector_code
                    blocks[stage]=(_block([f'const bool {selected_name}=true;',lowered_integer])
                                   if selected_name else lowered_integer)
                    if stage=='reset':
                        if selected_name:
                            reset_scalar_code=_block([f'const bool {selected_name}=_spike;',lowered_integer])
                            reset_vector_code=''
                        else:
                            reset_scalar_code=''
                            reset_vector_code=lowered_integer
                statements[stage]=tuple(scalar)+tuple(vector)
                if stage=='state':
                    refractory_code=''
                    dynamics_code=blocks[stage]
                    compiled_refractory_safe=False
                continue
        # Selected scalar auxiliaries can appear in later vector expressions.
        # Brian's AST classifier otherwise assumes an unknown name is float,
        # including a hoisted Boolean used by a short-circuit threshold.
        for statement in scalar+vector:
            if statement.var not in inventory:
                types[statement.var] = np.dtype(statement.dtype).type
        compiled_operator_stage=has_scalar_stages and model.reference_codegen_target in ("cython","cpp")
        if model.reference_codegen_target in ('cython','cpp'):
            compiled_operator_stage |= stage == "state" and _lif_refractory_write_variant(model.eqs)
            # Custom Cython differential dynamics need the actual optimizer's
            # fused operations while retaining scalar/vector loop boundaries.
            compiled_operator_stage |= (stage == "state" and model.reference_codegen_target == "cython"
                and bool(equations.diff_eq_names) and not _static_lif(model, ()))
            probe=_Renderer(types,values,reference_target=model.reference_codegen_target)
            compiled_operator_stage=compiled_operator_stage or any(probe.compiled_operator_expression(ast.parse(stmt.expr,mode='eval').body)
                           for stmt in scalar+vector)
        mixed_storage=any(np.dtype(dtype)==np.dtype('float32') for dtype in types.values()) or any(
            np.dtype(statement.dtype)==np.dtype('float32') for statement in scalar+vector)
        compiled64=model.state_dtype=='float64' and compiled_operator_stage and not mixed_storage
        if compiled64:
            from .cpu_arithmetic import compiler_identity,lower_stage
            from .operator_graph import Binary64Registry
            target='cython' if model.reference_codegen_target=='cython' else 'cpp_standalone'
            identity=compiler_identity(target,flags_override=model.reference_compile_args,
                python_flags=(model.integer_policy.python_flags if model.integer_policy is not None
                              and model.integer_policy.target==target else None))
            graph,_=lower_stage(model,inventory,resolved,scalar,vector,group,runner,stage,
                neurons,identity,target,Binary64Registry(
                    {**types,**{statement.var:statement.dtype for statement in scalar+vector}},
                    used_math,zero_division_error_name,operator_policies,identity,
                    values=values,used_arrays=used_arrays,index_error_name=index_error_name,
                    target=model.reference_codegen_target))
            lowered=graph.render(guard=has_scalar_stages and stage=="state")
            if has_scalar_stages:
                scalar_stages[stage]=lowered
                cpu_refractory_guard |= stage=="state" and graph.guard_node is not None
            scalar_code=lowered.scalar_code
            if lowered.scalar_errors:
                used_math.add('operator_zero_division')
                scalar_code+='\n'+f'if({" || ".join(lowered.scalar_errors)}) {zero_division_error_name}=true;'
            vector_code=lowered.vector_code
            if lowered.vector_error:
                used_math.add('operator_zero_division')
                vector_code=vector_code.replace('_eon_cython_error',zero_division_error_name)
            blocks[stage]=_block([scalar_code,vector_code])
            statements[stage]=tuple(scalar)+tuple(vector)
            compiled64_operator_stages.add(stage)
            if stage=='state':
                refractory_code='';dynamics_code=blocks[stage];compiled_refractory_safe=False
            if stage=='reset':reset_scalar_code,reset_vector_code=scalar_code,vector_code
            continue
        if ((model.state_dtype=='float32' and not numpy32)
                or (model.state_dtype=='float64' and compiled_operator_stage and mixed_storage)):
            from .clang32 import Node, lower, specialize, render as render_graph
            from functools import partial
            import shlex
            graph_model=model
            if model.state_dtype=='float64':
                from .clang32 import compiler_identity
                graph_model=replace(model,float32_codegen=model.reference_codegen_target,
                    float32_compile_args=model.reference_compile_args,
                    float32_compiler_identity=compiler_identity())
                compiled64_operator_stages.add(stage)
            function_registry=None
            if any(isinstance(value,_TimedArray) for value in values.values()):
                from .operator_graph import Binary64Registry
                function_registry=Binary64Registry(types,used_math,zero_division_error_name,
                    values=values,used_arrays=used_arrays,index_error_name=index_error_name,
                    target=model.reference_codegen_target or model.float32_codegen)
            render=partial(render_graph,operator_policies=operator_policies,
                safe_float32_div=model.state_dtype=='float32',
                reference_flags=graph_model.float32_compile_args,
                operator_compiler=tuple(shlex.split(graph_model.float32_compiler_identity[0])+
                    shlex.split(graph_model.float32_compiler_identity[1])),function_registry=function_registry,code_names=code_names)
            original, lowered = lower(graph_model,inventory,scalar,vector,types,dict(values,N=neurons) if has_scalar_stages else values,stage,
                function_registry=function_registry)
            statements[stage]=tuple(scalar)+tuple(vector)
            if has_scalar_stages:
                scalar_outputs={name:node for name,node in lowered.items() if
                    name in inventory and inventory[name].persistent and inventory[name].scalar
                    or name.startswith('__eon_operator_scalar_failure_')}
                vector_outputs={name:node for name,node in lowered.items() if name not in scalar_outputs}
                scalar_source=render(scalar_outputs,used_math,zero_division_error_name)
                vector_source=render(vector_outputs,used_math,zero_division_error_name)
                scalar_stages[stage]=ScalarStage(scalar_source,(),vector_source,vector_source)
                blocks[stage]=scalar_source+'\n'+vector_source
                if stage=='state':
                    refractory_code='';dynamics_code=blocks[stage];compiled_refractory_safe=False
                if stage=='reset':reset_scalar_code,reset_vector_code=scalar_source,vector_source
                continue
            if stage=='state':
                clock=original.get('not_refractory')
                if clock is not None:
                    if clock.op=='zext' and clock.args[0].dtype=='bool':clock=clock.args[0]
                    fixed_clock=specialize({'not_refractory':clock},values)
                    def clock_arguments(node):
                        if node.op=='argument':return {node.value}
                        return set().union(*(clock_arguments(arg) for arg in node.args)) if node.args else set()
                    compiled_refractory_safe=(clock_arguments(fixed_clock['not_refractory'])
                        <= {'t','dt','lastspike','rfc','not_refractory'})
                    refractory_code=render(fixed_clock,used_math,zero_division_error_name)
                    dynamic={name:node for name,node in original.items() if name!='not_refractory'}
                    dynamic=specialize(dynamic,values,{clock:Node('argument','bool',value='not_refractory')})
                    dynamics_code=render(dynamic,used_math,zero_division_error_name)
                    blocks[stage]=refractory_code+'\n'+dynamics_code
                    # Prove exactly the same emitted graph at the canonical
                    # rest point before allowing idle-neuron compaction.
                    if set(dynamic)=={'g','v'} and _expression_key(model.eqs)==_expression_key(DEFAULT_EQUATIONS):
                        initial={'v':np.float32(model.v_0),'g':np.float32(0.)}
                        proof=specialize(dynamic,dict(initial,not_refractory=True))
                        if all(node.op=='constant' and np.asarray(node.value).tobytes()==initial[name].tobytes()
                               for name,node in proof.items()):
                            compiled_equilibrium=MappingProxyType(initial)
                else:
                    blocks[stage]=render(lowered,used_math,zero_division_error_name)
                    dynamics_code=blocks[stage]
            else:
                blocks[stage]=render(lowered,used_math,zero_division_error_name)
            if stage == 'reset':
                scalar_failures={name:node for name,node in lowered.items()
                    if name.startswith('__eon_operator_scalar_failure_')}
                vector_outputs={name:node for name,node in lowered.items() if name not in scalar_failures}
                reset_scalar_code=render(scalar_failures,used_math,zero_division_error_name) if scalar_failures else ''
                reset_vector_code=render(vector_outputs,used_math,zero_division_error_name)
                blocks[stage]=reset_scalar_code+'\n'+reset_vector_code
            continue
        if numpy_vectors:
            if half_numpy:
                from .numpy_complex import Renderer
                half_options={'half_input_arrays':half_input_arrays,'half_input_names':half_input_names,
                    'half_ufuncs':{name:(name+('_scalar' if name=='tanh' else '')+'_f32',
                    '_eon_hyper32_'+name+('_scalar' if name=='tanh' else ''))
                    for name in ('log10','sinh','cosh','tanh')}}
                half_options['half_ufuncs']['arctan']=('arctan_f32','_eon_arctan32')
                half_options['half_ufuncs'].update({name:(name+'_f32','_eon_trig32_'+helper)
                    for name,helper in (('sin','sine_scalar'),('cos','cosine_half'),('tan','tangent_scalar'))})
                half_options['half_ufuncs'].update({name:(name+'_f32','f_inverse_'+name)
                                                  for name in ('arcsin','arccos')})
                from .operator_policy import render_numpy_operator
                half_options['half_binary_operators'] = {
                    operation: render_numpy_operator
                    for operation in (ast.Pow, ast.Mod, ast.FloorDiv)}
            else:
                from .numpy_complex import BasicRenderer as Renderer
                half_options={}
            if model.numpy_error_actions:
                from .numpy_error_policy import policy_renderer
                Renderer = policy_renderer(Renderer)
                half_options['numpy_error_actions'] = model.numpy_error_actions
            store_order=(model.analysis.numpy_store_order or {}).get(stage) if model.analysis is not None else None
            if store_order is None:
                from .model import _numpy_write_order
                store_order=_numpy_write_order(resolved,vector)
            renderer = Renderer(types.copy(), values.copy(),
                {name for name,var in inventory.items() if var.persistent},
                {name for name,var in inventory.items() if var.scalar}, stage, model.float32_int_cast_block,
                used_arrays=used_arrays, index_error_name=index_error_name, used_math=used_math,
                shared_timed_calls=shared_timed_calls, reference_target="numpy",
                reference_flags=model.reference_compile_args,operator_policies=operator_policies,
                zero_division_error_name=zero_division_error_name,integer_policy=model.integer_policy,
                runtime_errors=runtime_errors,wrapper_arrays=model.numpy_basic_wrappers,store_order=store_order,
                runtime_error_writes=runtime_error_writes,code_names=code_names,**half_options)
        else:
            renderer = _Renderer(types.copy(), values.copy(), used_arrays, index_error_name, used_math, shared_timed_calls,
                                 reference_target=model.reference_codegen_target or "numpy",reference_flags=model.reference_compile_args,
                                 operator_policies=operator_policies,zero_division_error_name=zero_division_error_name,
                                 integer_policy=model.integer_policy,runtime_errors=runtime_errors,code_names=code_names)
        if stage=='threshold'and not general_random:renderer.expression_random=expression_random
        runtime_scalar: list[str] = []
        for statement in scalar:
            renderer.types[statement.var] = statement.dtype
            if statement.var in inventory and inventory[statement.var].persistent:
                runtime_scalar.append(renderer.statement(statement))
                continue
            try:
                node=ast.parse(statement.expr,mode="eval").body
                if renderer.compiled_operator_expression(node):
                    raise KeyError('compiled floating operator retains target semantics')
                value = (renderer.evaluate(node) if numpy_vectors else
                         _evaluate(node, renderer.values, renderer.reference_target))
                if numpy_temporaries and np.dtype(statement.dtype).kind == "f":
                    statement = copy.copy(statement)
                    statement.dtype = np.asarray(value).dtype
                if not numpy_vectors:
                    value = _cast_value(value, statement.dtype)
            except KeyError:
                runtime_scalar.append(renderer.statement(statement))
                continue
            if statement.op not in ("=", ":="):
                # Mutating scalar state has cross-neuron semantics.
                raise UnsupportedModelError(f"Native scalar assignment to {statement.var} is not implemented")
            renderer.values[statement.var] = value
            emitted_value = _numpy_scalar_carrier(value) if numpy_vectors else value
            emitted_dtype = np.asarray(emitted_value).dtype if numpy_vectors else statement.dtype
            if emitted_dtype==np.dtype('float16'):emitted_dtype=np.dtype('float64')
            line = f"const {_metal_type(emitted_dtype)} {renderer.code_name(statement.var)} = {_literal(emitted_value, _kind(emitted_dtype))};"
            if statement.var.startswith("_lio_"):
                scalar_lines.append(line)
                constants[statement.var] = value
            else:
                runtime_scalar.append(line)
        scalar_prologue=list(renderer.prologue) if numpy_vectors else []
        scalar_epilogue=renderer.finish() if numpy_vectors and has_scalar_stages else []
        if has_scalar_stages and numpy_vectors:
            renderer.persistent = renderer.persistent - {name for name,var in inventory.items() if var.scalar}
        scalar_setup=scalar_prologue+runtime_scalar
        scalar_source="\n".join(scalar_setup+scalar_epilogue)
        lines = list(scalar_setup)
        vector_lines = []
        if not numpy_vectors:
            renderer.plan_hyperbolic_pairs((statement,
                inventory[statement.var].conditional_write if stage=="state" and statement.var in inventory else None)
                for statement in vector)
        for statement in vector:
            condition = inventory.get(statement.var)
            # Brian resets override unless-refractory guards.
            guard = condition.conditional_write if condition is not None and stage == "state" else None
            vector_lines.append(renderer.statement(statement, guard))
        prologue = renderer.prologue if numpy_vectors else []
        epilogue = renderer.finish() if numpy_vectors else []
        if numpy_vectors:
            collective_reset |= renderer.collective_reset
        vector_prologue=prologue[len(scalar_prologue):]
        lines.extend(vector_prologue + vector_lines + epilogue)
        if has_scalar_stages:
            from .shared_state import declared_variables
            vector_source="\n".join(vector_prologue+vector_lines+epilogue)
            scalar_stages[stage]=ScalarStage(scalar_source,declared_variables(scalar_source),vector_source,vector_source)
        statements[stage] = tuple(scalar) + tuple(vector)
        blocks[stage] = _block(lines)
        if stage == "reset":
            reset_scalar_code = "\n".join(scalar_setup)
            if numpy_vectors and renderer.empty_store_checks:
                reset_scalar_code+='\nif(_numpy_reset_count==0){'+''.join(renderer.empty_store_checks)+'}'
            reset_vector_code = _block(vector_prologue + vector_lines + epilogue)
        if stage == "state":
            # The split is valid only if the refractory statement is first and
            # does not need declarations scoped with the remaining dynamics.
            runtime_names = {statement.var for statement in scalar if statement.var not in renderer.values}
            first_names = ({node.id for node in ast.walk(ast.parse(vector[0].expr, mode="eval"))
                            if isinstance(node, ast.Name)} if vector else set())
            if (vector and vector[0].var == "not_refractory" and vector[0].op == "="
                    and not runtime_names.intersection(first_names)):
                refractory_code = _block(vector_lines[:1])
                dynamics_code = _block(scalar_setup + vector_prologue + vector_lines[1:] + epilogue)
            else:
                dynamics_code = blocks[stage]
            state_constants = dict(constants)
    constants["dt"] = np.float64(model.dt_s)
    if state_constants is not None:
        state_constants["dt"] = constants["dt"]
    equilibrium = _prove_equilibrium(model, inventory, constants if state_constants is None else state_constants,
                                     statements["state"], statements["reset"])
    if model.state_dtype=='float32' and model.float32_codegen!='numpy' and _static_lif(model,statements['reset']):
        equilibrium=compiled_equilibrium
    refractory_elapsed_threshold = _refractory_elapsed_threshold(model, statements["reset"])
    if not compiled_refractory_safe:refractory_elapsed_threshold=None
    if '_complex_cast' in used_math:
        # Complex array casting has an ordered host warning boundary, even
        # when no random draw or floating-error action was requested.
        general_random=True
        expression_random=GeneralRandomPlan('numpy')
        for stage,items in statements.items():
            random_statements.setdefault(stage,(
                tuple(item for item in items if item.scalar),
                tuple(item for item in items if not item.scalar)))
    if general_random:
        from .expression_lowering import build_program
        expression_random=build_program(inventory,{**values,**constants},random_statements,
            floating_status=(model.numpy_policy_ir or
                (expression_random.target=='numpy' and
                 any(action!='ignore' for _,action in model.numpy_error_actions))),
            target=expression_random.target,renderer_options=dict(used_arrays=used_arrays,index_error_name=index_error_name,
                used_math=used_math,shared_timed_calls=shared_timed_calls,
                numpy_store_orders=(model.analysis.numpy_store_order or {}) if model.analysis is not None else {},
                reference_flags=model.reference_compile_args,operator_policies=operator_policies))
    inputs = {array.input_name: array.array for name, array in timed_arrays.items() if name in used_arrays}
    inputs.update(half_input_arrays)
    header_code = (_TIMED_ARRAY_HEADER + "\n".join(array.header() for name, array in timed_arrays.items()
                                                if name in used_arrays) if used_arrays else "")
    basic32={name[:-4] for name in used_math if name in {'exp_f32','log_f32','expm1_f32','log1p_f32'}}
    # d_expm1 guards nonfinite/large inputs and returns |x|<2^-53 before
    # entering either polynomial. All FMA operands there are normal or zero:
    # the smallest polynomial square is at least 2^-106; reduction residuals
    # of bounded binary64 inputs and table tails are also normal. Reuse the
    # existing exact known-width FMA without changing its operation graph.
    normal_expm1=model.state_dtype=='float32' and bool(used_math & {'expm1','exprel'})
    if basic32:
        from .basic_math32 import metal_basic_math_header
        header_code+=metal_basic_math_header(basic32)
    if normal_expm1:
        from .operator64 import _OPERATOR_FMA_HEADER
        header_code+=_OPERATOR_FMA_HEADER
    if used_math & {"exp", "log"}:
        from .transcendentals import metal_math_header
        header_code += metal_math_header(used_math & {"exp", "log"})
    complex_status=any(name.startswith('_numpy_complex_') for name in used_math)
    if '_half' in used_math or complex_status:
        from .binary16 import MSL_HEADER as half_header
        header_code+=half_header
    if '_compiled_integer' in used_math:
        from .compiled_integer import MSL_HEADER as integer_header
        header_code+=integer_header
    if '_compiled_python_integer' in used_math:
        from .compiled_python_integer import MSL_HEADER as python_integer_header
        header_code+=python_integer_header
    if '_compiled_precision' in used_math:
        from .compiled_precision import MSL_HEADER as PRECISION_HEADER
        header_code += PRECISION_HEADER

    if '_half_f32' in used_math or complex_status:
        from .binary16 import MSL_F32_HEADER as half_float_header
        header_code+=half_float_header
    if used_math & {"expm1", "log1p", "exprel"}:
        from .stable_math64 import metal_stable_math_header
        header_code += metal_stable_math_header(used_math,
            expm1_fma='_eon_operator_polynomial_fma' if normal_expm1 else 'stable_fma')
    if '_float32_safe_div' in used_math:
        from .basic_math32 import COMPILED_DIV_HEADER
        if not used_math & {'log1p','exprel'}:
            from .stable_math64 import _STABLE_DIV_HEADER
            header_code += _STABLE_DIV_HEADER
        header_code += COMPILED_DIV_HEADER
    if "arctan" in used_math:
        from .arctangent64 import metal_arctangent_header
        header_code += metal_arctangent_header(used_math)
    if used_math & {"log10", "sinh", "cosh", "tanh"}:
        from .hyperbolic_math64 import metal_hyperbolic_header
        header_code += metal_hyperbolic_header(used_math, model.reference_codegen_target or "numpy")
    if "_hyperbolic_pair" in used_math:
        from .hyperbolic_pair import MSL_HEADER
        header_code += MSL_HEADER
    if used_math & {"sin", "cos", "tan"}:
        from .trigonometry64 import metal_trigonometry_header
        header_code += metal_trigonometry_header(used_math & {"sin", "cos", "tan"})
    if "arctan_f32" in used_math:
        from .arctangent32 import metal_arctangent32_header
        header_code += metal_arctangent32_header({"arctan"})
    if used_math & {"arcsin", "arccos"}:
        from .inverse_trigonometry64 import metal_inverse_trigonometry_header
        header_code += metal_inverse_trigonometry_header(used_math)
    inverse32 = {name[:-4] for name in used_math if name in {"arcsin_f32", "arccos_f32"}}
    if inverse32:
        from .inverse_trigonometry32 import metal_inverse_trigonometry32_header
        header_code += metal_inverse_trigonometry32_header(inverse32)
    math32 = {name[:-4] for name in used_math
              if name in {"log10_f32", "sinh_f32", "cosh_f32", "tanh_f32",
                          "tanh_scalar_f32", "tanh_numpy_f32"}}
    if math32:
        from .hyperbolic32 import metal_hyperbolic32_header
        header_code += metal_hyperbolic32_header(math32, model.reference_codegen_target or "numpy")
    forward32 = {name[:-4] for name in used_math if name in {"sin_f32","cos_f32","tan_f32"}}
    if forward32:
        from .trigonometry32 import metal_trigonometry32_header
        forward_header=metal_trigonometry32_header(forward32)
        if used_math & {'log1p','exprel','_float32_safe_div'}:
            # These prior headers already define the shared exact divider.
            from .stable_math64 import _STABLE_DIV_HEADER
            forward_header=forward_header.replace(_STABLE_DIV_HEADER,'')
        header_code += forward_header
    if any(name.startswith("operator_") for name in used_math):
        from . import operator32,operator64
        if 'operator_uint_float32' in used_math: header_code+=operator32.MSL_UINT_CAST_HEADER
        if "operator_pow_float" in used_math: header_code+=operator64.MSL_POWER_HEADER
        if "operator_pow_float32" in used_math: header_code+=operator32.MSL_POWER_HEADER
        if "operator_powi_float" in used_math: header_code+=operator64.MSL_INTEGER_EXPONENT_HEADER
        if "operator_powi_float32" in used_math: header_code+=operator32.MSL_INTEGER_EXPONENT_HEADER
        if "operator_exp2_float" in used_math: header_code+=operator64.MSL_EXP2_HEADER
        if "operator_exp2_float32" in used_math: header_code+=operator32.MSL_EXP2_HEADER
        if used_math & {"operator_remainder_float","operator_remainder_float32"}:
            header_code+=operator64.MSL_REMAINDER_HEADER
            if "operator_remainder_float32" in used_math: header_code+=operator32.MSL_REMAINDER_HEADER
    if '_complex' in used_math:
        from .complex_math64 import metal_complex_header
        header_code += metal_complex_header(used_math)
    if '_complex_log' in used_math:
        from .complex_log_math64 import metal_complex_log_header
        header_code += metal_complex_log_header()
    if '_complex_power' in used_math:
        from .complex_power_math64 import metal_complex_power_header
        header_code += metal_complex_power_header()
    if '_complex_inverse' in used_math:
        from .complex_asinh_math32 import metal_complex_asinh_header
        from .complex_inverse_math64 import metal_complex_inverse_header
        header_code += metal_complex_asinh_header()+metal_complex_inverse_header()
    if '_python_complex' in used_math:
        from .python_complex_math64 import metal_python_complex_header
        header_code += metal_python_complex_header(power='_python_complex_power' in used_math)
    header_code+="\n".join(policy.header for policy in operator_policies.values())
    if '_numpy_floating_status' in used_math:
        from .numpy_error_policy import metal_header
        header_code += metal_header()
    numpy_unary=tuple(sorted(name[len('_numpy_unary_'):] for name in used_math
                             if name.startswith('_numpy_unary_')))
    if numpy_unary:
        from .numpy_math_status import metal_header as numpy_math_status_header
        header_code += numpy_math_status_header(numpy_unary)
    numpy_narrow=tuple(sorted(name[len('_numpy_narrow_'):] for name in used_math
                              if name.startswith('_numpy_narrow_') and name!='_numpy_narrow_cast'))
    if numpy_narrow or '_numpy_narrow_cast' in used_math:
        from .numpy_narrow_status import metal_header as numpy_narrow_status_header
        header_code += numpy_narrow_status_header(numpy_narrow)
    numpy_operators=tuple(sorted(name[len('_numpy_operator_'):] for name in used_math
                                 if name.startswith('_numpy_operator_')))
    if numpy_operators:
        from .numpy_operator_status import metal_header as numpy_operator_status_header
        header_code += numpy_operator_status_header(numpy_operators)
    numpy_operators32=tuple(sorted(name[len('_numpy_operator32_'):] for name in used_math
                                   if name.startswith('_numpy_operator32_')))
    if numpy_operators32:
        from .numpy_operator_status import metal_narrow_header
        header_code += metal_narrow_header(numpy_operators32)
    if '_numpy_complex_binary_status' in used_math:
        from .numpy_complex_error_status import metal_complex_error_header
        header_code += metal_complex_error_header()
    complex_unary=tuple(sorted(name[len('_numpy_complex_unary_'):] for name in used_math
                              if name.startswith('_numpy_complex_unary_')))
    if complex_unary or '_numpy_complex_power_status' in used_math:
        from .numpy_complex_unary_status import metal_complex_unary_header
        header_code += metal_complex_unary_header(complex_unary,
            power='_numpy_complex_power_status' in used_math)
    error_name = (index_error_name if any(len(timed_arrays[name].array.shape) == 2 and
                                         timed_arrays[name].target == "numpy" for name in used_arrays)
                  else None)
    runtime_error_type="index"
    if "operator_zero_division" in used_math:
        assert error_name is None # NumPy array indexing and Cython divmod have different frozen CPU targets.
        error_name=zero_division_error_name
        runtime_error_type="zero_division"
        scalar_lines.insert(0,f"int {zero_division_error_name}_kind = 0;")
    if runtime_errors:
        from .runtime_errors import ERROR_KIND_MASK
        if len(runtime_errors) > ERROR_KIND_MASK:
            raise UnsupportedModelError('Too many distinct native model errors')
        error_name = zero_division_error_name if '_compiled_integer' in used_math else index_error_name
        runtime_error_type = 'model'
        if f'int {error_name}_kind = 0;' not in scalar_lines:
            scalar_lines.insert(0, f'int {error_name}_kind = 0;')
    scalar_code=("\n".join(scalar_lines) if model.state_dtype == "float64" else
                 f"int {error_name}_kind = 0;" if runtime_error_type in ('zero_division','model') else '')
    scalar_code+='\n'+'\n'.join(dict.fromkeys(integer_cache_declarations)) if integer_cache_declarations else ''
    program = CompiledModel(MappingProxyType(inventory), scalar_code, blocks["state"],
                         blocks["threshold"], blocks["reset"], MappingProxyType(constants),
                         statements["state"], statements["threshold"], statements["reset"],
                         refractory_code, dynamics_code, equilibrium, refractory_elapsed_threshold,
                         MappingProxyType(inputs), header_code, error_name,
                         reset_scalar_code, reset_vector_code,
                         shared_timed_calls=tuple(shared_timed_calls),
                         timed_cache_name=timed_cache_name if shared_timed_calls else None,
                         runtime_error_type=runtime_error_type,operator_policies=tuple(operator_policies.values()),
                         disabled_stages=tuple(disabled_stages),
                         compiled_integer_stages=tuple(compiled_integer_stages),
                         native_precision_limbs=(native_precision_limbs or 8) if '_compiled_precision' in used_math else 0,
                         collective_reset=collective_reset,runtime_errors=tuple(runtime_errors),
                         runtime_error_writes=tuple(sorted(runtime_error_writes.items())),
                         runtime_error_order='operation' if runtime_errors and model.reference_codegen_target == 'numpy' else 'row')
    if expression_random is not None:
        if program.index_error_name is not None or program.native_precision_limbs:
            from .expression_fixed import checked_threshold
            from .expression_errors import COMPLEX_UFUNC_ERRORS
            # The speculative renderer records these faults too. Accept only
            # exact errors backed by an ordered IR action, not general TypeError.
            ordered_host_errors={COMPLEX_UFUNC_ERRORS[op.argument][1]
                for operations in getattr(expression_random,'stages',{}).values() for op in operations
                if getattr(op,'function',None)=='numpy_complex_ufunc_error'
                and op.argument in COMPLEX_UFUNC_ERRORS}
            ordered_general=(general_random and (
                (expression_random.target=='cython' and model.execution_phases==('groups','thresholds','synapses','resets'))
                or (expression_random.target=='numpy' and not program.native_precision_limbs
                    and all(kind in ('IndexError','NumpyFloatingStatus')
                            or (kind,message) in ordered_host_errors
                            for kind,message in runtime_errors))))
            if not ordered_general and not checked_threshold(model,expression_random,shared=has_scalar_stages):
                raise UnsupportedModelError('Checked random-expression replay integration is pending')
            expression_random=replace(expression_random,checked=True)
        program=replace(program,random_program=expression_random,
                        shared_expression_inputs=bool(shared_expression_inputs))
    if has_scalar_stages:
        from .shared_state import independent_steps
        from brian2.utils.stringtools import word_substitute
        aliases={name:var.code_name for name,var in inventory.items() if var.persistent and var.scalar}
        scalar_stages={name:replace(stage,scalar_code=word_substitute(stage.scalar_code,aliases),
            vector_code=word_substitute(stage.vector_code,aliases),dynamics_code=word_substitute(stage.dynamics_code,aliases))
            for name,stage in scalar_stages.items()}
        return replace(program,scalar_stages=MappingProxyType(scalar_stages),
            scalar_parallel=independent_steps(program,scalar_statements,model.execution_phases),
            scalar_input_independent=independent_steps(program,scalar_statements,model.execution_phases,error_transport=True),
            subexpression_code=scalar_stages['subexpression'].vector_code,
            equilibrium=None,refractory_elapsed_threshold=None,cpu_refractory_guard=cpu_refractory_guard,
            refractory_policy='cython_fast' if 'state' in compiled64_operator_stages else program.refractory_policy)
    if 'state' in compiled64_operator_stages or 'state' in compiled_integer_stages:
        program=replace(program,refractory_policy='cython_fast',equilibrium=None,
            refractory_elapsed_threshold=None)
        canonical=b.Equations(DEFAULT_EQUATIONS)
        if ('state' in compiled64_operator_stages
                and model.reference_codegen_target=='cython'
                and set(equations.diff_eq_names)==set(canonical.diff_eq_names)
                and all(equations[name].expr==canonical[name].expr for name in canonical.diff_eq_names)
                and all(np.asarray(value).ndim==0 and np.asarray(value).dtype.kind in 'fiu'
                        for value in (model.tau,model.t_mbr,model.v_0))):
            # A custom threshold does not change the canonical state updater.
            # Generic block folding rounds its Cython coupling coefficient
            # differently; retain the independently audited canonical tree.
            canonical_program = _compiled_lif_arithmetic(model,program,types,values)
            if canonical_program is program:
                return program
            # A custom threshold does not alter the physical fixed point or
            # refractory clock. Prove only those properties with the canonical
            # threshold; retain the actual threshold/reset in the program.
            proof_model = replace(model,eq_th=DEFAULT_THRESHOLD)
            proof_values = dict(constants)
            try:
                for statement in statements['state']:
                    if statement.scalar:
                        proof_values[statement.var] = _cast_value(
                            _evaluate(ast.parse(statement.expr,mode='eval').body,proof_values),
                            statement.dtype)
                equilibrium = _prove_equilibrium(proof_model,inventory,proof_values,
                    statements['state'],statements['reset'])
            except (KeyError,UnsupportedModelError):
                equilibrium = None
            return replace(canonical_program,equilibrium=equilibrium,
                refractory_elapsed_threshold=_refractory_elapsed_threshold(proof_model,statements['reset']),
                canonical_custom_lif=True)
        return program
    return (_compiled_lif_arithmetic(model, program, types, values)
            if model.state_dtype == "float64" else program)
