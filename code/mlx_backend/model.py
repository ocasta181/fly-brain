"""Canonical Eon model configuration, independent of the execution backend.

State and physical constants use SI units here.  The legacy fixed LIF kernel
uses millivolts and a resting-potential offset; its adapter must convert them.
Brian is imported only for explicitly supplied parameters/analysis, or by
``brian_namespace``.  Analysis builds equations and statements, never a CPU run.
"""

from __future__ import annotations

import hashlib
import copy
import io
import json
import math
import re
import sys
import tokenize
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


DEFAULT_EQUATIONS = """dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau : volt (unless refractory)
rfc : second
"""
DEFAULT_THRESHOLD = "v > v_th"
DEFAULT_RESET = "v = v_rst; w = 0; g = 0 * mV"

_DIMENSIONLESS = (0., 0., 0., 0., 0., 0., 0.)
_VOLT = (2., 1., -3., -1., 0., 0., 0.)
_SECOND = (0., 0., 1., 0., 0., 0., 0.)
_HERTZ = (0., 0., -1., 0., 0., 0., 0.)
_DIMENSIONS = {
    **dict.fromkeys(("v_0", "v_rst", "v_th", "w_syn"), _VOLT),
    **dict.fromkeys(("t_mbr", "tau", "t_rfc", "t_dly", "t_run"), _SECOND),
    "r_poi": _HERTZ, "r_poi2": _HERTZ,
    "f_poi": _DIMENSIONLESS, "n_run": _DIMENSIONLESS,
}
# Retain the rounding of the original unit multiplications, including -52*mV.
_DEFAULTS = {
    "v_0": -52 * .001, "v_rst": -52 * .001, "v_th": -45 * .001,
    "t_mbr": 20 * .001, "tau": 5 * .001, "t_rfc": 2.2 * .001,
    "t_dly": 1.8 * .001, "w_syn": .275 * .001,
    "r_poi": 150., "r_poi2": 0., "f_poi": 250,
    "t_run": 1000 * .001, "n_run": 30,
    "eqs": DEFAULT_EQUATIONS, "eq_th": DEFAULT_THRESHOLD,
    "eq_rst": DEFAULT_RESET,
}
_SCALARS = tuple(k for k in _DIMENSIONS if k not in ("t_run", "n_run"))
_INITIALIZERS = frozenset(("v_0", "t_rfc"))
_SCHEDULE = ("start", "groups", "thresholds", "synapses", "resets", "end")


@contextmanager
def model_preferences(model):
    """Bind Brian's preparation-time dtype decisions to the saved model.

    Brian's optimiser consults the global preference for folded constants and
    loop invariants even when a NeuronGroup has an explicit dtype mapping.
    Always restore the caller's preferences, including on preparation errors.
    """
    import brian2 as b

    saved = {name: b.prefs[name] for name in (
        "core.default_float_dtype", "core.default_integer_dtype", "codegen.loop_invariant_optimisations",
    )}
    try:
        temporary_dtype = (model.analysis.temporary_dtype if model.analysis is not None
                           and model.analysis.temporary_dtype else model.state_dtype)
        b.prefs.core.default_float_dtype = np.dtype(temporary_dtype).type
        b.prefs.core.default_integer_dtype = np.dtype(model.integer_dtype).type
        if model.reference_optimise is not None:
            b.prefs.codegen.loop_invariant_optimisations=model.reference_optimise
        yield
    finally:
        for name, value in saved.items():
            b.prefs[name] = value


@lru_cache(maxsize=64)
def _schedule_phases(schedule) -> tuple[str, ...]:
    """Resolve the application's active phases as Brian Network.sorted_objects.

    Default-schedule preferences bypass Network.schedule's explicit setter:
    duplicate slots and before_/after_ names are accepted there. Reproduce the
    automatic-slot updates, including last-occurrence precedence. The parent
    NeuronGroup/Synapses objects require start even though they execute no code;
    end has no objects in the frozen application and can be omitted.
    """
    positions = {slot: 1 + i * 3 for i, slot in enumerate(schedule)}
    positions.update((f"before_{slot}", i * 3) for i, slot in enumerate(schedule))
    positions.update((f"after_{slot}", 2 + i * 3) for i, slot in enumerate(schedule))
    positions["start"]
    phases = ("groups", "thresholds", "synapses", "resets")
    return tuple(sorted(phases, key=positions.__getitem__))


def _lif_refractory_write_variant(source: str) -> bool:
    """Recognize canonical LIF equations with changed conditional writes."""
    import brian2 as b
    equations, canonical = b.Equations(source), b.Equations(DEFAULT_EQUATIONS)
    if set(equations.diff_eq_names) != set(canonical.diff_eq_names):
        return False
    changed = False
    for name in canonical.diff_eq_names:
        actual, expected = equations[name], canonical[name]
        if (actual.expr != expected.expr or actual.dim != expected.dim
                or set(actual.flags) - {"unless refractory"}):
            return False
        changed |= set(actual.flags) != set(expected.flags)
    return changed


def _expression_key(source: str) -> tuple[tuple[int, str], ...]:
    # Preserve token boundaries: ``x and y`` and ``xandy`` must never share a
    # cache identity.  Formatting and comments do not change an equation.
    source = "\n".join(line.lstrip() for line in source.splitlines()).strip() + "\n"
    # Tokenization is pure. Bound retained source size as well as entry count;
    # larger expressions remain accepted through the same uncached tokenizer.
    return _cached_expression_tokens(source) if len(source) <= 8192 else _expression_tokens(source)


@lru_cache(maxsize=64)
def _cached_expression_tokens(source):
    return _expression_tokens(source)


def _expression_tokens(source):
    ignored = {tokenize.NL, tokenize.INDENT, tokenize.DEDENT,
               tokenize.COMMENT, tokenize.ENDMARKER}
    return tuple((token.type, token.string) for token in
                 tokenize.generate_tokens(io.StringIO(source).readline)
                 if token.type not in ignored)


def _array_snapshot(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError("Namespace arrays must contain real numeric or boolean values")
    # Immutable bytes own the storage, so setflags(write=True) cannot undo this.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _equations_source(equations: Any) -> str:
    # Brian's str(Equations) uses display aliases such as V/s and erases integer
    # types.  These are not accepted equation unit declarations on reconstruction.
    from brian2.units.fundamentalunits import get_unit

    lines = []
    for equation in equations.values():
        name = (f"d{equation.varname}/dt" if equation.type == "differential equation"
                else equation.varname)
        expression = f" = {equation.expr.code}" if equation.expr is not None else ""
        unit = (equation.var_type if equation.var_type in ("integer", "boolean") else
                "1" if tuple(equation.dim._dims) == _DIMENSIONLESS else repr(get_unit(equation.dim)))
        flags = f" ({', '.join(equation.flags)})" if equation.flags else ""
        lines.append(f"{name}{expression} : {unit}{flags}")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class TimedArrayValue:
    """An immutable SI snapshot; the caller's TimedArray is never cached."""

    values: np.ndarray = field(repr=False, compare=False)
    dt_s: float
    dimensions: tuple[float, ...]


@dataclass(frozen=True)
class ModelVariable:
    name: str
    dtype: str
    dimensions: tuple[float, ...]
    scalar: bool
    constant: bool
    read_only: bool
    conditional_write: str | None
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelAnalysis:
    """Brian-validated abstract code; this does not assert native execution."""

    state_abstract: str
    threshold_abstract: str
    reset_abstract: str
    variables: Mapping[str, ModelVariable]
    brian_version: str
    temporary_dtype: str | None = None
    codegen_target: str | None = None
    typed_statements: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)
    numpy_store_order: Mapping[str, tuple[str,...]] | None = field(default=None,repr=False,compare=False,kw_only=True)

    @property
    def compilation_key(self):
        """Complete selected statements and store order, independent of identity."""
        statements, stores = _numpy_analysis_identity(self)
        return repr(self), statements, stores


def _numpy_store_order(source):
    import ast
    stores=[node for node in ast.walk(ast.parse(str(source)))
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.Name)
            and any(isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name)
                    and target.value.id.startswith('_array_') for target in node.targets)]
    return tuple(dict.fromkeys(node.value.id for node in sorted(stores,key=lambda node:(node.lineno,node.col_offset))))


def _numpy_write_order(variables,statements):
    from collections import defaultdict
    from types import SimpleNamespace
    from brian2.codegen.generators.numpy_generator import NumpyCodeGenerator
    context=SimpleNamespace(variables=variables,variable_indices=defaultdict(lambda:'_idx'),allows_scalar_write=True)
    return tuple(NumpyCodeGenerator.array_read_write(context,statements)[1])


def _numpy_analysis_identity(analysis):
    selected=tuple((stage,tuple(tuple(tuple((name,repr(value)) for name,value in sorted(statement.__dict__.items()))
                    for statement in section) for section in sections))
                   for stage,sections in sorted((analysis.typed_statements or {}).items()))
    return selected,tuple(sorted((analysis.numpy_store_order or {}).items()))

@dataclass(frozen=True, eq=False)
class ResolvedModel:
    """Frozen model constants and namespace snapshots in SI units.

    ``signature`` is absent for opaque callbacks/objects whose changing host
    dependencies cannot be hashed safely.  Their objects remain in
    ``source_namespace`` and their names in ``pending_namespace``; this is an
    explicit unresolved execution capability, not a supported-domain exclusion.
    ``analysis`` describes accepted Brian equations, not implemented MLX code.
    ``state_specialization`` allows only input-rate differences from the default
    state kernel; ``default_specialization`` also requires the default rates.
    """

    v_0: float | np.ndarray | str
    v_rst: float
    v_th: float
    t_mbr: float
    tau: float
    t_rfc: float | np.ndarray | str
    t_dly: float
    w_syn: float | np.ndarray
    r_poi: float
    r_poi2: float
    f_poi: float
    dt_s: float
    state_dtype: str
    float32_codegen: str
    float32_optimise: bool
    float32_compile_args: tuple[str, ...]
    float32_compiler_identity: tuple[str, ...]
    float32_int_cast_block: int
    integer_dtype: str
    legacy_refractory_timing: bool
    schedule: tuple[str, ...]
    eqs: str
    eq_th: str
    eq_rst: str
    source_namespace: Mapping[str, Any] = field(repr=False, compare=False)
    namespace_dimensions: Mapping[str, tuple[float, ...]] = field(repr=False, compare=False)
    signature: str | None
    default_specialization: bool
    state_specialization: bool
    pending_namespace: tuple[str, ...] = ()
    analysis: ModelAnalysis | None = field(default=None, repr=False, compare=False)
    # TimedArray lookup and tanh rounding differ between Brian CPU targets.
    # Resolve auto through the active device before caching target-dependent
    # models. Other models retain the lazy, Brian-free path.
    reference_codegen_target: str | None = None
    reference_cpp_compiler: str | None = None
    reference_compile_args: tuple[str, ...] = ()
    reference_optimise: bool | None = None
    integer_policy: Any = field(default=None, repr=False, compare=False)
    source_signature: str | None = field(default=None, repr=False, compare=False)
    numpy_basic_wrappers: tuple[tuple[str,bool], ...] = ()
    numpy_error_actions: tuple[tuple[str,str], ...] = ()
    numpy_policy_ir: bool = False

    @property
    def random_target(self):
        return self.reference_codegen_target or 'numpy'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResolvedModel):
            return NotImplemented
        if self.signature is None or other.signature is None:
            return self is other
        return self.signature == other.signature

    def __hash__(self) -> int:
        if self.signature is None:
            raise TypeError("Models with pending namespace objects cannot be cached")
        return hash(self.signature)

    @property
    def dt_ms(self) -> float:
        return self.dt_s * 1000

    @property
    def poisson_weight_v(self) -> float:
        if self.state_dtype=='float32':
            return float(np.multiply(self.source_namespace['w_syn'],self.source_namespace['f_poi']))
        return self.w_syn * self.f_poi

    @property
    def delay_steps(self) -> int:
        """Brian's compiled synapse queue rounds nonnegative ties upward."""
        if self.dt_s < 0:
            # The fresh scalar queue quantizes delay/dt with a truncating
            # compiled cast. Nonpositive casts retain its single queue slot.
            if not math.isfinite(self.t_dly):
                return 0
            return max(0, int(self.t_dly / self.dt_s + .5))
        # A fresh Brian queue given NaN or a negative scalar retains its
        # single arrival slot, so accepted original-model runs deliver now.
        if math.isnan(self.t_dly) or self.t_dly < 0:
            return 0
        return math.floor(self.t_dly / self.dt_s + .5)

    @property
    def execution_phases(self) -> tuple[str, ...]:
        return _schedule_phases(self.schedule)

    @property
    def arrival_delay_steps(self) -> int:
        """Synapses before thresholds enqueue the previous step's eventspace."""
        phases = self.execution_phases
        return self.delay_steps + (phases.index("synapses") < phases.index("thresholds"))

    @property
    def reset_uses_previous_spikes(self) -> bool:
        phases = self.execution_phases
        return phases.index("resets") < phases.index("thresholds")

    @property
    def refractory_steps(self) -> int:
        """Modern Brian timestep() quantization; legacy has no fixed count."""
        if self.legacy_refractory_timing:
            raise ValueError("Legacy refractory timing has no clock-independent step count")
        return int((self.t_rfc + 1e-3 * self.dt_s) / self.dt_s)

    @property
    def refractory_abstract(self) -> str:
        """Snapshotted duration predicate for the application's per-neuron rfc.

        Brian 2.8 omits the assignment in its legacy string-duration branch.
        Supply it locally, using the same strict elapsed comparison as Brian's
        working Quantity-duration branch. The linear state updater and all
        conditional writes are still generated by Brian.
        """
        if self.legacy_refractory_timing:
            return "not_refractory = (t - lastspike) > rfc\n"
        return "not_refractory = timestep(t - lastspike, dt) >= timestep(rfc, dt)\n"

    @property
    def cacheable(self) -> bool:
        return self.signature is not None

    def computed_steps(self, duration_sec: float) -> int:
        """Fresh-clock horizon, including Brian's 0.0001*dt boundary tolerance."""
        if hasattr(duration_sec, "dim"):
            from brian2 import second
            from brian2.units.fundamentalunits import fail_for_dimension_mismatch
            fail_for_dimension_mismatch(duration_sec, second, "Duration must be a time")
        duration = float(duration_sec)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Duration must be finite and nonnegative")
        quotient = duration / self.dt_s
        if not math.isfinite(quotient) or quotient >= 2**63:
            raise OverflowError("Duration exceeds Brian's signed 64-bit clock")
        nearest = round(quotient)
        nearest_time = nearest * self.dt_s
        if nearest_time == duration or abs(nearest_time - duration) / self.dt_s <= 1e-4:
            return nearest
        return math.ceil(quotient)

    def brian_namespace(self) -> dict[str, Any]:
        """Reconstruct a separate unit-bearing Brian namespace lazily."""
        from brian2 import TimedArray, second
        from brian2.units.fundamentalunits import Quantity, get_or_create_dimension

        result = {}
        for name, value in self.source_namespace.items():
            if isinstance(value, TimedArrayValue):
                quantity = Quantity(value.values.copy(), dim=get_or_create_dimension(value.dimensions))
                result[name] = TimedArray(quantity, dt=value.dt_s * second)
            elif name in self.namespace_dimensions:
                dimensions = self.namespace_dimensions[name]
                copied = value.copy() if isinstance(value, np.ndarray) else value
                result[name] = (copied if dimensions == _DIMENSIONLESS else
                                Quantity(copied, dim=get_or_create_dimension(dimensions)))
            else:
                result[name] = value
        return result


def _numeric_identity(value: Any) -> Any:
    if isinstance(value, TimedArrayValue):
        return ["TimedArray", value.dt_s.hex(), value.dimensions, _numeric_identity(value.values)]
    if isinstance(value, np.ndarray):
        return ["array", value.dtype.str, value.shape, hashlib.sha256(value.tobytes()).hexdigest()]
    if isinstance(value,np.generic):
        if value.dtype.kind == "f" and not np.isfinite(value):
            return ["numpy-scalar-bits", value.dtype.str, value.tobytes().hex()]
        return ['numpy-scalar',value.dtype.str,_numeric_identity(value.item())]
    if isinstance(value, (bool, np.bool_)):
        return ["bool", bool(value)]
    if isinstance(value, (int, np.integer)):
        return ["int", int(value)]
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return ["float-bits", np.float64(value).tobytes().hex()]
        return ["float", float(value).hex()]
    return ["string", value] if isinstance(value, str) else None


@contextmanager
def _preparation_device():
    """Keep temporary analysis/compiler groups out of standalone builds."""
    import brian2 as b
    from brian2.devices.device import RuntimeDevice, get_device

    active = get_device()
    isolated = not isinstance(active, RuntimeDevice)
    options = dict(getattr(active, "build_options", {}))
    build_on_run = getattr(active, "build_on_run", True)
    if isolated:
        b.set_device(RuntimeDevice())
    try:
        yield
    finally:
        if isolated:
            b.set_device(active, build_on_run=build_on_run, **options)


@_preparation_device()
def _analyse(model: ResolvedModel) -> ModelAnalysis:
    import brian2 as b
    from brian2.codegen.translation import analyse_identifiers, make_statements
    from brian2.equations.unitcheck import check_units_statements
    from brian2.groups.neurongroup import check_subexpressions

    group = b.NeuronGroup(
        1, model.eqs, method="linear", threshold=model.eq_th, reset=model.eq_rst,
        refractory="rfc", namespace=model.brian_namespace(),
        dt=model.dt_s * b.second, dtype={
            equation.varname: np.dtype(
                model.integer_dtype if equation.var_type == "integer" else
                bool if equation.var_type == "boolean" else model.state_dtype,
            ) for equation in b.Equations(model.eqs).values()
        },
        name="mlx_model_analysis*",
    )
    # Analysis objects must not join a later caller's Brian magic network.
    group._network = "mlx-model-analysis-only"
    group.state_updater._get_refractory_code = lambda run_namespace: model.refractory_abstract
    # This one-neuron group validates declarations, not dataset-sized state.
    # The actual application assignments (including string expressions) happen
    # on the real group; never execute an initializer while analysing a model.
    group.v = 0 * b.volt
    group.g = 0
    group.rfc = 0 * b.second
    group.equations.check_units(group, run_namespace={})
    check_subexpressions(group, group.equations, {})
    runners = (group.state_updater, group.thresholder["spike"], group.resetter["spike"])
    typed = {};numpy_stores={}
    for stage, runner in zip(("state", "threshold", "reset"), runners):
        runner.update_abstract_code(run_namespace={})
        additional = getattr(runner, "variables", None)
        known_variables = dict(group.variables)
        if additional is not None:
            known_variables.update(additional)
        _, known, unknown = analyse_identifiers(runner.abstract_code, known_variables, recursive=True)
        resolved = group.resolve_all(
            known | unknown, run_namespace={}, additional_variables=additional,
            user_identifiers=set(),
        )
        check_units_statements(runner.abstract_code, resolved)
        if model.integer_policy is not None:
            if model.reference_codegen_target=='numpy':
                # This is the same cached request made by the actual NumPy
                # generator; its cache omits the mutable optimization preference.
                scalar,vector=make_statements(runner.abstract_code,resolved,
                    np.dtype(b.prefs.core.default_float_dtype),optimise=True,blockname='')
                numpy_stores[stage]=_numpy_write_order(resolved,vector)
            else:
                scalar, vector = make_statements(runner.abstract_code, resolved,
                    np.dtype(b.prefs.core.default_float_dtype),
                    optimise=bool(model.reference_optimise), blockname=stage)
            typed[stage] = copy.deepcopy((tuple(scalar), tuple(vector)))
    variables = {}
    for name, variable in group.variables.items():
        if not hasattr(variable, "dtype"):
            continue
        condition = getattr(variable, "conditional_write", None)
        equation = group.equations[name] if name in group.equations else None
        variables[name] = ModelVariable(
            name, np.dtype(variable.dtype).name, tuple(float(x) for x in variable.dim._dims),
            variable.scalar, variable.constant, variable.read_only,
            condition.name if condition is not None else None,
            tuple(equation.flags) if equation is not None else (),
        )
    return ModelAnalysis(*(r.abstract_code for r in runners), MappingProxyType(variables), b.__version__,
                         np.dtype(b.prefs.core.default_float_dtype).name,
                         model.reference_codegen_target or b.prefs.codegen.target,
                         MappingProxyType(typed) if typed else None,
                         numpy_store_order=MappingProxyType(numpy_stores) if numpy_stores else None)


def resolve_model(params: Mapping[str, Any] | None = None, dt_ms: Any = None,
                  *, state_dtype=None, cpu_target=None,random_target=None,
                  _prepared_analysis=None) -> ResolvedModel:
    """Resolve defaults plus upstream parameter overrides, without simulating.

    Dimensional overrides use Brian quantities.  ``dt_ms`` accepts a numeric
    millisecond value or a time quantity.  When omitted it uses the existing
    Brian defaultclock if Brian is already loaded, otherwise 0.1 ms.  A ``dt``
    namespace key has no clock-setting meaning, matching the upstream API.
    ``state_dtype`` can preserve existing storage when preparation uses a
    different default temporary precision.
    """
    if random_target is not None:
        if random_target not in('numpy','cpp','cython'):raise ValueError('Unknown random-expression reference target')
        if cpu_target is not None and {'cpp_standalone':'cpp'}.get(cpu_target,cpu_target)!=random_target:
            raise ValueError('Random-expression and model reference targets must agree')
        cpu_target=random_target
    brian = sys.modules.get("brian2")
    if cpu_target is not None and brian is None:
        import brian2 as brian
    dt_s = float(brian.defaultclock.dt_) if dt_ms is None and brian is not None else .1 * .001
    if dt_ms is not None:
        if hasattr(dt_ms, "dim"):
            from brian2 import second
            from brian2.units.fundamentalunits import fail_for_dimension_mismatch
            fail_for_dimension_mismatch(dt_ms, second, "dt_ms must be a time")
            dt_s = float(dt_ms)
        else:
            dt_s = float(dt_ms) * .001
    if not math.isfinite(dt_s) or dt_s == 0 or (dt_s < 0 and _prepared_analysis is None):
        raise ValueError("Timestep must be finite and positive")
    temporary_dtype = np.dtype(brian.prefs.core.default_float_dtype).name if brian is not None else "float64"
    state_dtype = np.dtype(temporary_dtype if state_dtype is None else state_dtype).name
    float32_codegen = ""
    float32_optimise = True
    float32_compile_args = ()
    float32_compiler_identity = ()
    float32_int_cast_block = 0
    if state_dtype == "float32":
        float32_optimise = bool(brian.prefs.codegen.loop_invariant_optimisations)
        if cpu_target is None:
            from brian2.devices.device import get_device
            from brian2.codegen.generators import CPPCodeGenerator
            code_class = get_device().code_object_class()
            target = ('cpp' if issubclass(code_class.generator_class,CPPCodeGenerator)
                      else code_class.class_name)
        else:
            target = cpu_target
        if target == 'cpp_standalone':
            target = 'cpp'
        if target == "auto":
            from brian2.devices.device import auto_target
            target = auto_target().class_name
        float32_codegen = target if isinstance(target, str) else target.class_name
        if float32_codegen=='numpy':
            from .numpy32 import integer_cast_block
            float32_int_cast_block=integer_cast_block()
        if float32_codegen != "numpy":
            from .clang32 import compiler_identity
            float32_compiler_identity=compiler_identity()
            flags = brian.prefs.codegen.cpp.extra_compile_args
            if flags is None:
                flags = brian.prefs.codegen.cpp.extra_compile_args_gcc
            float32_compile_args = tuple(flags)
    integer_dtype = np.dtype(brian.prefs.core.default_integer_dtype).name if brian is not None else "int32"
    legacy_refractory = bool(brian.prefs.legacy.refractory_timing) if brian is not None else False
    schedule = tuple(brian.prefs.core.network.default_schedule) if brian is not None else _SCHEDULE
    _schedule_phases(schedule)
    namespace, dimensions = dict(_DEFAULTS), dict(_DIMENSIONS)
    pending = []
    if params is not None:
        import brian2 as b
        from brian2.units.fundamentalunits import fail_for_dimension_mismatch, get_dimensions, get_or_create_dimension

        for name, value in params.items():
            if not isinstance(name, str):
                raise TypeError("Namespace keys must be strings")
            if name in ("eqs", "eq_th", "eq_rst"):
                if name == "eqs" and isinstance(value, b.Equations):
                    value = _equations_source(value)
                if not isinstance(value, str):
                    raise TypeError(f"{name} must be a Brian expression string")
                namespace[name] = value
            elif isinstance(value, b.TimedArray):
                namespace[name] = TimedArrayValue(
                    _array_snapshot(value.values), float(value.dt),
                    tuple(float(x) for x in value.dim._dims),
                )
                dimensions.pop(name, None)
            elif isinstance(value, (np.ndarray, list, tuple, int, float, bool, np.number)):
                if name in _DIMENSIONS:
                    fail_for_dimension_mismatch(
                        value, get_or_create_dimension(_DIMENSIONS[name]), f"Invalid dimensions for {name}",
                    )
                dimension = tuple(float(x) for x in get_dimensions(value)._dims)
                array = np.asarray(value)
                # The original model forwards these real values to Brian,
                # including NaN initializers, reset constants and weights.
                # Retain Brian's preparation/runtime behavior instead of
                # introducing a separate physical-value validity policy.
                ieee_scalar = name in (
                    "v_th", "f_poi", "r_poi", "r_poi2", "v_0", "v_rst",
                    "w_syn", "t_mbr", "tau", "t_rfc") or (
                        name == "t_dly" and array.dtype.kind in "biuf"
                        and (dt_s < 0 or not np.any(np.isposinf(array))))
                if array.dtype.kind not in "biuf" or (
                        name in _SCALARS and not ieee_scalar
                        and not np.all(np.isfinite(array))):
                    raise ValueError(f"{name} must contain finite real values")
                typed_scalar=isinstance(value,np.generic) or array.dtype==np.dtype('float32')
                namespace[name] = (array[()] if typed_scalar else array.item()) if array.shape == () else _array_snapshot(array)
                # A literal zero is dimension-polymorphic in Brian.
                dimensions[name] = _DIMENSIONS.get(name, dimension)
            elif isinstance(value, str):
                namespace[name] = value
                dimensions.pop(name, None)
            else:
                namespace[name] = value
                dimensions.pop(name, None)
                pending.append(name)
    for name in _SCALARS:
        if not isinstance(namespace[name], (int, float, np.number, np.bool_, np.ndarray)) and not (
                name in _INITIALIZERS and isinstance(namespace[name], str)):
            raise TypeError(f"{name} must be a numeric constant or initializer")
    scalar_values = {name: (namespace[name] if isinstance(namespace[name], (np.ndarray, str))
                            else float(namespace[name])) for name in _SCALARS}
    reference_codegen_target = None
    reference_cpp_compiler = None
    reference_compile_args = ()
    operator_source=" ".join(str(namespace[name]) for name in ("eqs","eq_th","eq_rst"))
    operator_policy=(any(token in operator_source for token in ("**","%","//"))
                     or re.search(r"\b(?:log|log1p)\s*\(",operator_source) is not None)
    expression_tokens = {token[1] for name in ("eqs", "eq_th", "eq_rst")
                         for token in _expression_key(namespace[name])}
    integer_contract = bool(params is not None and (
        any(e.var_type in ("integer", "boolean") for e in b.Equations(namespace["eqs"]).values())
        or expression_tokens & {"int", "i", "N"}))
    custom_language = any(_expression_key(namespace[name]) != _expression_key(_DEFAULTS[name])
                          for name in ("eqs", "eq_th", "eq_rst"))
    target_math = any(
        any(tokens[index][0] == tokenize.NAME
            and tokens[index][1] in {"log10", "sinh", "cosh", "tanh", "sin", "cos", "tan",
                                     "arcsin", "arccos", "arctan"}
            and tokens[index+1][1] == "("
            for index in range(len(tokens)-1))
        for tokens in (_expression_key(namespace[name]) for name in ("eqs", "eq_th", "eq_rst")))
    if cpu_target is not None or operator_policy or state_dtype == "float32" or schedule != _SCHEDULE or target_math or integer_contract or custom_language or any(isinstance(value, TimedArrayValue) for value in namespace.values()):
        from brian2.devices.device import get_device
        from brian2.codegen.generators import CPPCodeGenerator, CythonCodeGenerator, NumpyCodeGenerator

        code_class = get_device().code_object_class()
        generator = code_class.generator_class
        reference_codegen_target = next((name for base, name in (
            (NumpyCodeGenerator, "numpy"), (CythonCodeGenerator, "cython"), (CPPCodeGenerator, "cpp"),
        ) if issubclass(generator, base)), f"{code_class.__module__}.{code_class.__qualname__}")
        if cpu_target is not None:
            reference_codegen_target = {"cpp_standalone": "cpp"}.get(cpu_target, cpu_target)
        if reference_codegen_target in ("cython", "cpp"):
            from brian2.codegen.cpp_prefs import get_compiler_and_args
            reference_cpp_compiler, args = get_compiler_and_args()
            reference_compile_args = tuple(args)
    reference_optimise=bool(brian.prefs.codegen.loop_invariant_optimisations) if cpu_target is not None or state_dtype=="float32" or target_math or integer_contract or custom_language or operator_policy else None
    integer_policy = None
    # Freeze the actual Cython frontend for custom floating phases too.
    # Their long sums can acquire a final target tree that differs from a
    # C++ rendering, even when the model has no integer state or index use.
    if integer_contract or (custom_language and reference_codegen_target=='cython'):
        from .integer_policy import snapshot_integer_policy
        integer_policy = snapshot_integer_policy(cpu_target)
    canonical = {
        "format": "eon-resolved-model-si-v1", "dt_s": dt_s.hex(), "dtype": state_dtype,
        "temporary_dtype": temporary_dtype,
        "integer_dtype": integer_dtype, "legacy_refractory_timing": legacy_refractory,
        "schedule": schedule,
        "namespace": {name: [["expression", _expression_key(value)] if name in ("eqs", "eq_th", "eq_rst")
                             else _numeric_identity(value), dimensions.get(name)]
                      for name, value in sorted(namespace.items())},
    }
    if reference_codegen_target is not None:
        canonical.update(reference_codegen_target=reference_codegen_target,
                         reference_cpp_compiler=reference_cpp_compiler,
                         reference_compile_args=reference_compile_args)
    if state_dtype == "float32":
        canonical["float32_codegen"] = float32_codegen
        canonical["float32_optimise"] = float32_optimise
        canonical["float32_compile_args"] = float32_compile_args
        canonical["float32_compiler_identity"] = float32_compiler_identity
        canonical["float32_int_cast_block"] = float32_int_cast_block
        canonical['float32_numpy_version']=np.__version__
    if reference_optimise is not None:
        canonical['reference_optimise']=reference_optimise
    if integer_policy is not None:
        canonical['integer_policy'] = integer_policy.identity
    from .basic_math32 import POLICY_IDENTITY as basic_math32_identity
    canonical['basic_math32_policy']=basic_math32_identity
    numpy_basic_wrappers=()
    if reference_codegen_target=='numpy':
        from .numpy32 import basic_wrapper_profile
        numpy_basic_wrappers=basic_wrapper_profile()
        canonical['numpy_basic_wrappers']=(np.__version__,numpy_basic_wrappers)
    signature = None if pending else hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    # An explicitly selected CPU arithmetic target needs its matching float
    # lowering rather than the fixed integer-state specialization.
    fixed = (
        cpu_target is None and not pending and state_dtype == "float64" and dt_s == .1 * .001
        and not legacy_refractory and schedule == _SCHEDULE
        and all(np.ndim(value) == 0 and not isinstance(value, str)
                for value in scalar_values.values())
        and all(scalar_values[k] == _DEFAULTS[k] for k in _SCALARS if k not in ("r_poi", "r_poi2"))
        and all(_expression_key(namespace[k]) == _expression_key(_DEFAULTS[k])
                for k in ("eqs", "eq_th", "eq_rst"))
        and not (set(namespace) - set(_DEFAULTS))
    )
    model = ResolvedModel(
        **scalar_values, dt_s=dt_s, state_dtype=state_dtype, float32_codegen=float32_codegen,
        float32_optimise=float32_optimise,
        float32_compile_args=float32_compile_args,
        float32_compiler_identity=float32_compiler_identity,numpy_basic_wrappers=numpy_basic_wrappers,
        float32_int_cast_block=float32_int_cast_block,
        integer_dtype=integer_dtype,
        legacy_refractory_timing=legacy_refractory, schedule=schedule,
        **{k: namespace[k] for k in ("eqs", "eq_th", "eq_rst")},
        source_namespace=MappingProxyType(namespace), namespace_dimensions=MappingProxyType(dimensions),
        signature=signature,
        default_specialization=fixed and all(scalar_values[k] == _DEFAULTS[k] for k in ("r_poi", "r_poi2")),
        state_specialization=fixed, pending_namespace=tuple(sorted(pending)),
        reference_codegen_target=reference_codegen_target,
        reference_cpp_compiler=reference_cpp_compiler, reference_compile_args=reference_compile_args,
        reference_optimise=reference_optimise,
        integer_policy=integer_policy,
        source_signature=signature if integer_policy is not None else None,
    )
    if params is not None:
        # Pending opaque callbacks remain represented; their validation may need
        # host dependencies and must not imply an accepted native implementation.
        if not pending:
            analysis=_analyse(model) if _prepared_analysis is None else _prepared_analysis
            object.__setattr__(model, "analysis",analysis)
            if model.reference_codegen_target=='numpy':
                selected_signature=hashlib.sha256((model.signature+repr(_numpy_analysis_identity(analysis))).encode()).hexdigest()
                object.__setattr__(model,'signature',selected_signature)
                if model.source_signature is not None:object.__setattr__(model,'source_signature',selected_signature)
    return model
