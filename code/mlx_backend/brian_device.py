"""Opt-in MLX execution of the original Eon program's real Brian objects.

Select ``MLXDevice`` before calling the unchanged paper model's create_model or
run_exp.  Brian still owns units, indexed assignments, host preparation and
monitors.  Only neuron/synapse timesteps execute through the native engine.
N=1 Poisson draws preserve Brian's RNG and scheduler order. Positive delays use
a native target-neuron prepass per delay window; zero delay uses a native probe
per step. Neither path executes a CPU neuron timestep.
"""

from __future__ import annotations

import ast
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from time import time
from weakref import WeakKeyDictionary

import numpy as np

import brian2 as b
from brian2.devices.device import RuntimeDevice, reset_device
from brian2.units.fundamentalunits import check_units
from brian2.codegen.runtime.cython_rt.cython_rt import CythonCodeObject


def _prepare_cython_source(codeobj):
    codeobj.compiled_code = dict.fromkeys(("before_run", "run", "after_run"))


class _CythonPreparationFactory(CythonCodeObject):
    """Generate standard Cython source, returning the actual standard object.

    Only its construction-time compile call is replaced. The device restores
    the instance's real compile method immediately; target class, flags,
    variables and generated source retain Brian's original identities.
    """

    def __new__(cls, *args, **kwargs):
        from types import MethodType
        codeobj = CythonCodeObject(*args, **kwargs)
        codeobj.compile = MethodType(_prepare_cython_source, codeobj)
        return codeobj


class UnsupportedBrianNetwork(NotImplementedError):
    """The object graph needs a native lowering this device does not provide."""


def _count_spike_words(words, lengths, previous):
    """Count valid emitted bits, retaining Brian's modulo-32-bit counts."""
    from .window import count_words
    import mlx.core as mx

    if isinstance(previous, np.uint32):
        previous = np.full(words[0].size, previous, np.uint32)
    masked = [word if length == 32 else word & np.uint32((1 << length)-1)
              for word, length in zip(words, lengths)]
    return count_words(masked, mx.array(previous) if isinstance(previous, np.ndarray) else previous)


def _session_snapshot(values, caller_values, *, native_output):
    """Retain an owned native output without aliasing mutable Brian storage."""
    if native_output:
        import mlx.core as mx

        owner = values
        while isinstance(owner, np.ndarray):
            owner = owner.base
        if (isinstance(owner, memoryview) and isinstance(owner.obj, mx.array)
                and not np.shares_memory(values, caller_values)):
            # The base chain retains the MLX allocation. Native advancement
            # produces new output buffers; Brian received a separate copy.
            snapshot = values.view()
            snapshot.flags.writeable = False
            return snapshot
    # Zero-step/imported state and error replay do not prove new allocation.
    return values.copy()


@dataclass
class _Session:
    topology: tuple
    state: object
    sampler: object
    prepared: object
    host_state: dict
    fast_refractory: bool
    clock_rebase_pending: bool = False


def _rebase_event_words(words, delta, window):
    """Translate queued iteration coordinates without reading device words."""
    rebased = {}
    for index, word in words.items():
        target, shift = divmod(index*window+delta, window)
        low = word << shift if shift else word
        if window < 32:
            low = low & np.uint32((1 << window)-1)
        rebased[target] = rebased[target] | low if target in rebased else low
        if shift:
            high = word >> (window-shift)
            rebased[target+1] = rebased[target+1] | high if target+1 in rebased else high
    return rebased


class _PoissonSampler:
    """Brian-generated random draws only, with no user neuron state or clock run."""

    def __init__(self, inputs, clock):
        from brian2.core.functions import Function

        self.inputs = tuple(inputs)
        functions = []
        for source in inputs:
            functions.append(next(value for value in source.variables.values()
                                  if isinstance(value, Function)))
        self.functions = tuple(functions)
        self.group = b.NeuronGroup(
            len(inputs), "eon_draw : integer\neon_eligible : boolean",
            clock=clock, name="mlx_poisson_draws*",
        )
        self.group._network = "mlx-poisson-input-only"
        self.group.variables["eon_draw"].conditional_write = self.group.variables["eon_eligible"]

        def numpy_draw(indices):
            return np.asarray([functions[int(channel)].pyfunc()[0]
                               for channel in np.asarray(indices)], dtype=np.int32)

        function = Function(lambda: 0, arg_units=[], return_unit=1,
                            stateless=False, auto_vectorise=True)
        function.implementations.add_implementation("numpy", numpy_draw)
        calls = "\n".join(f"    if vectorisation_idx == {channel}:\n"
                          f"        return {draw.name}(vectorisation_idx)"
                          for channel, draw in enumerate(functions))
        function.implementations.add_implementation(
            "cython", "cdef int eon_original_poisson(const int vectorisation_idx):\n"
            + calls + "\n    return 0\n", name="eon_original_poisson",
            dependencies={draw.name: draw for draw in functions},
        )
        self.group.namespace["eon_original_poisson"] = function
        self.runner = self.group.run_regularly(
            "eon_draw = eon_original_poisson()", name="mlx_poisson_sampler*",
            codeobj_class=next((type(source.codeobj) for source in inputs if source.codeobj is not None), None),
        )
        self.runner.before_run({})

    def draw_prefix(self, eligible):
        """Original NumPy calls in source order, including empty masked calls."""
        events = np.full(len(self.inputs), 2, np.uint8)
        for channel, (source, function) in enumerate(zip(self.inputs, self.functions)):
            if not source.active:
                continue
            try:
                values = function.implementations["numpy"].get_code(source)(int(eligible[channel]))
            except ValueError as error:
                return events, (source, error)
            if eligible[channel]:
                events[channel] = values[0]
        return events, None

    def draw(self, eligible):
        self.group.variables["eon_eligible"].get_value()[:] = eligible
        self.runner.codeobj()
        # Two means no runner executed. A sampled zero must still add its
        # signed zero product, unlike an inactive or conditionally skipped input.
        return np.where(eligible, self.group.variables["eon_draw"].get_value(), 2).astype(np.uint8)


def _literal_amplitude(source):
    """Evaluate the literal Brian actually compiled, preserving its SI rounding."""
    from brian2.core.namespace import DEFAULT_UNITS

    if isinstance(source.weight, str):
        raise UnsupportedBrianNetwork("PoissonInput string weights need native expression lowering")
    expression = ast.parse(source.abstract_code, mode="exec").body
    if (len(expression) != 1 or not isinstance(expression[0], ast.AugAssign)
            or not isinstance(expression[0].op, ast.Add)
            or not isinstance(expression[0].target, ast.Name)
            or expression[0].target.id != "v"):
        raise UnsupportedBrianNetwork("Unsupported PoissonInput update code")
    class ReplaceDraw(ast.NodeTransformer):
        calls = 0

        def visit_Call(self, node):
            from brian2.input.binomial import BinomialFunction
            self.calls += 1
            if (node.args or node.keywords or not isinstance(node.func, ast.Name)
                    or node.func.id not in source.variables
                    or not isinstance(source.variables[node.func.id], BinomialFunction)):
                raise UnsupportedBrianNetwork("Unsupported PoissonInput draw expression")
            return ast.copy_location(ast.Constant(1), node)

    transformer = ReplaceDraw()
    literal = ast.Expression(body=transformer.visit(expression[0].value))
    if transformer.calls != 1:
        raise UnsupportedBrianNetwork("Unsupported PoissonInput literal weight")
    for node in ast.walk(literal):
        if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                                 ast.Name, ast.Load, ast.Add, ast.Sub, ast.Mult,
                                 ast.Div, ast.Pow, ast.UAdd, ast.USub)):
            raise UnsupportedBrianNetwork("Unsupported PoissonInput literal expression")
    namespace = dict(DEFAULT_UNITS)
    names = {node.id for node in ast.walk(literal) if isinstance(node, ast.Name)}
    # Quantity repr can contain inf or a namespace-resolved nan. Use Brian's
    # actual lookup rather than assuming the spelling fixes its value.
    for name, variable in source._group.resolve_all(names - namespace.keys(), run_namespace={}).items():
        namespace[name] = variable.get_value()
    amplitude = float(eval(compile(literal, "<Brian Poisson literal>", "eval"),
                           {"__builtins__": {}}, namespace))
    if (source.codeobj is not None and source.codeobj.class_name == "cython"
            and np.dtype(b.prefs.core.default_float_dtype) == np.dtype(np.float32)):
        from brian2.codegen.translation import make_statements
        from .compiler import _evaluate
        variables = dict(source.codeobj.variables)
        values = {}
        for name, variable in variables.items():
            if getattr(variable, "scalar", False) and getattr(variable, "constant", False):
                value = np.asarray(variable.get_value())
                if value.size == 1:
                    values[name] = np.dtype(variable.dtype).type(value.reshape(-1)[0])
        selected=getattr(source.codeobj,"_mlx_typed_statements",None)
        if selected is not None:
            if len(selected)!=1:raise UnsupportedBrianNetwork("PoissonInput needs one prepared statement block")
            scalar,vector=next(iter(selected.values()))
        else:
            scalar,vector=make_statements(source.abstract_code,variables,np.float32,optimise=True)
        updates=[statement for statement in scalar+vector if statement.var=="v" and statement.op=="+="]
        if len(updates)!=1:
            raise UnsupportedBrianNetwork("Unsupported prepared PoissonInput update")
        for statement in scalar:
            if statement is updates[0]:continue
            # A shared target puts the draw-containing update in scalar code.
            # Recover its deterministic increment; never execute a random call.
            expression=ReplaceDraw().visit(ast.parse(statement.expr,mode="eval").body)
            values[statement.var]=np.dtype(statement.dtype).type(_evaluate(expression,values))
        prepared_literal=ReplaceDraw().visit(ast.parse(updates[0].expr,mode="eval").body)
        amplitude=float(_evaluate(prepared_literal,values))
    return amplitude


class MLXDevice(RuntimeDevice):
    """Application-specific RuntimeDevice; real Brian allocation stays intact."""

    def __init__(self):
        super().__init__()
        # Brian's Cython DEFAULT_FUNCTIONS capture the registered runtime's
        # buffer namespace when the generator module imports. Share these
        # arrays so inherited seed/get_random_state address the actual draws.
        from brian2.devices.device import runtime_device
        for name in ("rand_buffer", "rand_buffer_index", "randn_buffer", "randn_buffer_index"):
            setattr(self, name, getattr(runtime_device, name))
        self._sessions = WeakKeyDictionary()
        self.native_runs = 0
        self.native_steps = 0
        self.last_timings = {}
        self._preparing_native = False

    def _finish_empty_network_run(self, network, objects, t_start, t_end,
                                  duration, callback, entry_time):
        """Preserve Brian's accepted clock-only completion without timesteps.

        The caller has run the actual clock and ``before_run`` preparation.
        No model lowering or CPU numerical code is needed when that clock
        establishes an empty interval for a nonfinite or negative horizon/dt.
        """
        from brian2.groups.group import Group
        began = time()
        if callback:
            callback(0*b.second, 0., t_start, duration)
        network.t_ = float(t_end)
        self._last_run_time = time()-began
        self._last_run_completed_fraction = (
            (network.t-t_start)/duration if duration > 0*b.second else 1.)
        for obj in objects:
            if isinstance(obj, Group):
                obj._check_for_invalid_states()
        if callback:
            callback(self._last_run_time*b.second, 1., t_start, duration)
        network.after_run()
        for obj in objects:
            if type(obj) is b.NeuronGroup:
                # Cython preparation retains source without compiling its
                # after_run block; preserve the thresholder's metadata clear.
                obj.variables["_spikespace"].get_value()[-1] = 0
        previous = self._sessions.get(network)
        if previous is not None:
            # Brian leaves its physical queue position unchanged on an empty
            # run, even if the requested network time rewinds the next clock.
            previous.clock_rebase_pending = True
            previous.fast_refractory = False
        network._profiling_info = None
        self.native_runs += 1
        self.last_timings = dict(
            host_preparation_s=began-entry_time, analysis_s=0., layout_s=0.,
            import_and_rng_setup_s=0., execution_s=0., writeback_s=time()-began,
            prepass_and_dependencies_s=0., general_submission_s=0.,
            recording_and_dependencies_s=0.)

    def code_object(self, owner, name, abstract_code, variables, template_name,
                    variable_indices, codeobj_class=None, **kwargs):
        target = self.code_object_class(codeobj_class)
        source_only = self._preparing_native and target is CythonCodeObject
        result = super().code_object(owner, name, abstract_code, variables,
            template_name, variable_indices,
            codeobj_class=_CythonPreparationFactory if source_only else target, **kwargs)
        if source_only:
            del result.compile
        if target.class_name == "numpy":
            from copy import deepcopy
            from brian2.codegen.translation import make_statements

            # Brian's cached translation consults a global optimisation
            # preference that is absent from its cache key. Preserve the
            # statements actually selected for this code object, including
            # a cached selection from an earlier preference value.
            result._mlx_typed_statements = {
                block: deepcopy(make_statements(code, variables,
                    b.prefs.core.default_float_dtype, optimise=True, blockname=block))
                for block, code in abstract_code.items()
            }
        return result

    @staticmethod
    def _objects(network):
        from brian2.groups.neurongroup import Resetter, StateUpdater, Thresholder
        from brian2.synapses.synapses import SynapticPathway

        objects = network.sorted_objects
        groups = [obj for obj in objects if type(obj) is b.NeuronGroup]
        synapses = [obj for obj in objects if type(obj) is b.Synapses]
        monitors = [obj for obj in objects if type(obj) is b.SpikeMonitor]
        inputs = [obj for obj in objects if type(obj) is b.PoissonInput]
        if len(groups) != 1 or len(synapses) != 1:
            raise UnsupportedBrianNetwork("MLXDevice needs one Eon group and one recurrent Synapses")
        group, syn = groups[0], synapses[0]
        runners = [group.state_updater, group.thresholder.get("spike"), group.resetter.get("spike")]
        supported = {group, syn, *monitors, *inputs, *runners, *syn._pathways}
        if group.subexpression_updater is not None:
            supported.add(group.subexpression_updater)
        if any(obj not in supported for obj in objects):
            names = [obj.name for obj in objects if obj not in supported]
            raise UnsupportedBrianNetwork(f"Unsupported Brian runners: {', '.join(names)}")
        if (set(group.events) != {"spike"} or any(runner is None for runner in runners[:2])
                or not all(isinstance(runner, cls) for runner, cls in zip(
                    runners, (StateUpdater, Thresholder, Resetter)) if runner is not None)):
            raise UnsupportedBrianNetwork("MLXDevice needs standard spike state/threshold runners and an optional reset")
        if (not isinstance(group._refractory, str) or group._refractory != "rfc"
                or group.state_updater.method_choice != "linear"):
            raise UnsupportedBrianNetwork("MLXDevice needs linear Eon equations and rfc")
        from .model import _schedule_phases
        _schedule_phases(tuple(network.schedule))
        if any((runner.when, runner.order) != expected for runner, expected in zip(
                runners, [("groups", 0), ("thresholds", 0), ("resets", 0)]) if runner is not None):
            raise UnsupportedBrianNetwork("Custom neuron runner phases/orders need native phase lowering")
        if any(not obj.active for obj in (group, syn, *runners, *syn._pathways) if obj is not None):
            raise UnsupportedBrianNetwork("Disabling neuron or synaptic runners needs native phase lowering")
        if (syn.source is not group or syn.target is not group or len(syn._pathways) != 1
                or not isinstance(getattr(syn, "pre", None), SynapticPathway) or syn.pre.event != "spike"
                or syn.pre.when != "synapses" or syn.pre.order != -1
                or "".join(syn.pre.code.split()) not in ("g+=w", "g_post+=w")
                or len(syn.equations) != 1 or "w" not in syn.equations):
            raise UnsupportedBrianNetwork("MLXDevice needs one recurrent on_pre='g += w' pathway")
        for monitor in monitors:
            if (monitor.source is not group or monitor.when != "thresholds" or monitor.order != 1
                    or set(monitor.record_variables) - {"i", "t"}):
                raise UnsupportedBrianNetwork("MLXDevice supports default whole-group spike monitors only")
        for source in inputs:
            subgroup = source._group
            if (getattr(subgroup, "source", None) != group or len(subgroup) != 1
                    or source.target_var != "v" or source.N != 1
                    or source.when != "synapses" or source.order != 0):
                raise UnsupportedBrianNetwork("MLXDevice needs literal N=1 single-neuron v PoissonInput channels")
            _literal_amplitude(source)
        clocks = {obj.clock for obj in objects}
        if len(clocks) != 1:
            raise UnsupportedBrianNetwork("Multiple Brian clocks need native clock scheduling")
        return objects, group, syn, monitors, inputs, next(iter(clocks))

    @staticmethod
    def _model(group, syn, network, namespace):
        from copy import deepcopy
        from dataclasses import replace
        from hashlib import sha256
        from types import MappingProxyType
        from .model import ModelAnalysis, ModelVariable, _DEFAULTS, _equations_source, resolve_model
        from brian2.utils.stringtools import get_identifiers

        delay = np.asarray(syn.pre.variables["delay"].get_value())
        if delay.size != 1 and delay.size and not np.all(delay == delay[0]):
            raise UnsupportedBrianNetwork("Heterogeneous synaptic delays need native queue lowering")
        delay_s = float(delay[0]) if delay.size else float(group.namespace.get("t_dly", 0*b.ms))
        equation_source = group.namespace.get("eqs")
        if equation_source is None or str(b.Equations(equation_source)) != str(group.user_equations):
            equation_source = _equations_source(group.user_equations)
        parameters = {**namespace, **group.namespace,
                      "eqs": equation_source,
                      "eq_th": group.events["spike"], "eq_rst": group.event_codes.get("spike", ""),
                      "t_dly": delay_s*b.second}
        referenced = (group.equations.identifiers | get_identifiers(group.events["spike"])
                      | get_identifiers(group.event_codes.get("spike", "")))
        for name in ("v_0", "t_rfc"):
            if isinstance(parameters.get(name), str):
                referenced |= get_identifiers(parameters[name])
        # Keep all Eon fields and bindings referenced by its actual equations,
        # threshold, reset or initializers. Other namespace annotations cannot
        # affect simulation; preserve them on the real Brian objects without
        # making their Python types native compiler obligations.
        parameters = {name: value for name, value in parameters.items()
                      if name in _DEFAULTS or name in referenced}
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
        runners = (group.state_updater, group.thresholder["spike"], group.resetter.get("spike"))
        typed = {};numpy_stores={}
        stage_runners=list(zip(("state", "threshold", "reset"), runners))
        if group.subexpression_updater is not None:
            stage_runners.append(("subexpression",group.subexpression_updater))
        for stage, runner in stage_runners:
            if runner is None:
                typed[stage] = ([], [])
                numpy_stores[stage] = ()
                continue
            if runner.codeobj.class_name=='numpy':
                from .model import _numpy_store_order
                numpy_stores[stage]=_numpy_store_order(runner.codeobj.code.run)
            selected = getattr(runner.codeobj, "_mlx_typed_statements", None)
            if selected is not None:
                if len(selected) != 1:
                    raise UnsupportedBrianNetwork("Neuron stages need one prepared statement block")
                typed[stage] = deepcopy(next(iter(selected.values())))
        analysis = ModelAnalysis(*(runner.abstract_code if runner is not None else "" for runner in runners),
                                 MappingProxyType(variables), b.__version__,
                                 np.dtype(b.prefs.core.default_float_dtype).name,
                                 group.state_updater.codeobj.class_name,
                                 MappingProxyType(typed) if typed else None,
                                 numpy_store_order=MappingProxyType(numpy_stores) if numpy_stores else None)
        # The actual objects have already passed Brian's before_run analysis.
        # Reuse that snapshot instead of analysing a disposable second group.
        model = resolve_model(parameters, dt_ms=group.clock.dt,
                              state_dtype=group.variables["v"].dtype,
                              _prepared_analysis=analysis)
        codeobj = group.state_updater.codeobj
        target = codeobj.class_name
        compiler = getattr(codeobj, "compiler", None)
        compile_args = tuple(getattr(codeobj, "extra_compile_args", ()))
        single_policy = {}
        if model.state_dtype == "float32":
            single_policy = dict(float32_codegen=target, float32_compile_args=compile_args,
                                 float32_compiler_identity=(), float32_int_cast_block=0)
            if target == "numpy":
                from .numpy32 import integer_cast_block
                single_policy["float32_int_cast_block"] = integer_cast_block()
            else:
                from .clang32 import compiler_identity
                single_policy["float32_compiler_identity"] = compiler_identity()
        # TimedArray models already capture their CPU target in resolve_model.
        # Real Brian objects must retain that identity even without a TimedArray.
        # ModelAnalysis intentionally omits its mutable statement objects from
        # repr/equality. Include all selected metadata explicitly: Boolean
        # simplification and scalar provenance affect emitted arithmetic too.
        statement_identity = analysis.compilation_key[1]
        signature = (sha256((model.signature+repr(analysis)
                            +repr((target, compiler, compile_args, single_policy, tuple(network.schedule),
                                   statement_identity,tuple(sorted(numpy_stores.items()))))).encode()).hexdigest()
                     if model.signature is not None else None)
        if signature is not None and group.resetter.get("spike") is None:
            # Empty native reset code must not erase the real None-reset
            # configuration from the original-object cache identity.
            signature = sha256((signature+";reset_runner_absent").encode()).hexdigest()
        return replace(model, analysis=analysis, schedule=tuple(network.schedule), signature=signature,
                       source_signature=signature if model.integer_policy is not None else model.source_signature,
                       reference_codegen_target=target, reference_cpp_compiler=compiler,
                       reference_compile_args=compile_args,
                       state_dtype=np.dtype(group.variables["v"].dtype).name,
                       integer_dtype=np.dtype(group.variables["i"].dtype).name,
                       **single_policy), delay_s

    @check_units(duration=b.second, report_period=b.second)
    def network_run(self, network, duration, report=None, report_period=10*b.second,
                    namespace=None, profile=None, level=0):
        import gc
        import sys
        from hashlib import sha256
        import mlx.core as mx
        from dataclasses import replace
        from brian2.core.namespace import get_local_namespace
        from brian2.core.network import TextReport
        from brian2.groups.group import Group
        from . import configured
        from .brian_poisson import NativePoisson
        from .brian_codegen import original_codegen_policy
        from .models import Connectome, Stimulus
        from .window import collect_words

        entry_time = time()
        entry_native_steps = self.native_steps
        if not mx.metal.is_available():
            raise RuntimeError("MLXDevice requires a Metal GPU")
        duration_s = float(duration)
        if np.isposinf(duration_s):
            raise ValueError("Duration must be finite and nonnegative")
        if profile is None:
            profile = self.build_options.get("profile", False)
        if profile:
            raise UnsupportedBrianNetwork("Brian object-level profiling does not describe fused native execution")
        if self.build_options.keys() - {"profile"}:
            raise UnsupportedBrianNetwork("Unsupported MLXDevice build options")
        callback = None
        if report is not None:
            if report in ("text", "stdout", "stderr"):
                callback = TextReport(sys.stderr if report == "stderr" else sys.stdout)
            elif callable(report):
                raise UnsupportedBrianNetwork("Mutable user report callbacks need native phase synchronization")
            else:
                raise ValueError("Unknown Brian report callback")
        if not np.isfinite(float(report_period)) or float(report_period) <= 0:
            raise ValueError("Report period must be finite and positive")
        objects, group, syn, monitors, inputs, clock = self._objects(network)
        if namespace is None:
            namespace = get_local_namespace(level=level+3)
        sources = np.array(syn.variables["_synaptic_pre"].get_value(), copy=True)
        destinations = np.array(syn.variables["_synaptic_post"].get_value(), copy=True)
        delays = np.array(syn.pre.variables["delay"].get_value(), copy=True)
        topology = (id(group), id(syn), len(group), sources.tobytes(), destinations.tobytes(),
                    delays.tobytes(), float(clock.dt), tuple(network.schedule), tuple(id(source) for source in inputs),
                    b.prefs.codegen.target, repr(b.prefs.codegen.cpp.extra_compile_args),
                    tuple(b.prefs.codegen.cpp.extra_compile_args_gcc))
        previous = self._sessions.get(network)
        if previous is not None and topology != previous.topology:
            raise UnsupportedBrianNetwork("Changing topology, delays, channels or dt during a native session is unsupported")
        if previous is None:
            gc.collect()
        network._clocks = {clock}
        t_start = network.t
        t_end = t_start + duration
        clock.set_interval(t_start, t_end)
        self._preparing_native = True
        try:
            network.before_run(namespace)
        finally:
            self._preparing_native = False
        if ((np.isnan(duration_s) or duration_s < 0
                or not np.isfinite(float(clock.dt)) or float(clock.dt) < 0)
                and int(clock._i_end) <= int(clock.variables["timestep"].get_value()[0])):
            return self._finish_empty_network_run(
                network, objects, t_start, t_end, duration, callback, entry_time)
        prepared_host_time = time()
        model, _ = self._model(group, syn, network, namespace)
        if model.reference_codegen_target == 'numpy':
            policy = tuple(sorted(np.geterr().items()))
            action_ir = getattr(self,'_numpy_policy_replay',False)
            signature = (sha256((model.signature + repr((policy,action_ir))).encode()).hexdigest()
                         if model.signature is not None else None)
            model = replace(model, numpy_error_actions=policy, numpy_policy_ir=action_ir,
                            signature=signature,
                            source_signature=signature if model.integer_policy is not None else model.source_signature)
        analysed_time = time()
        targets = tuple(int(source._group.start) for source in inputs)
        stimulus = Stimulus(np.empty((1, 0, len(inputs)), np.uint8), targets,
                           tuple(float(source.rate) for source in inputs), (0,), 0, 0, None,
                           model.dt_ms)
        # Actual SI weights are imported below; coefficient placeholders need
        # no per-edge allocation or physical count/weight conversion.
        unused_coefficients = np.broadcast_to(np.float64(0.), sources.shape)
        graph = Connectome(np.arange(len(group)), sources, destinations,
                           unused_coefficients, unused_coefficients)
        rfc = group.variables["rfc"].get_value()
        lastspike = group.variables["lastspike"].get_value()
        scalar_rfc = np.ndim(model.t_rfc) == 0 and not isinstance(model.t_rfc, str)
        canonical_rfc = (np.full(len(group), model.t_rfc, dtype=rfc.dtype)
                         if scalar_rfc else None)
        if canonical_rfc is not None:
            canonical_rfc[list(targets)] = 0.
        # The release-boundary inversion covers the declared state width and
        # Brian's binary64 lastspike clock. Other actual storage stays exact
        # through the general predicate rather than using this specialization.
        canonical_storage = (rfc.dtype == np.dtype(model.state_dtype)
                             and lastspike.dtype == np.dtype(np.float64))
        fast_refractory = (
            previous.fast_refractory and model.signature == previous.prepared.model.signature
            and rfc.tobytes() == previous.host_state["rfc"].tobytes()
            and lastspike.tobytes() == previous.host_state["lastspike"].tobytes()
        ) if previous is not None else (
            scalar_rfc and canonical_storage and np.array_equal(rfc, canonical_rfc)
            and np.all(lastspike == -1e4)
        )
        fast_refractory = (fast_refractory and model.dt_s > 0
                           and int(clock.variables["timestep"].get_value()[0]) >= 0)
        prepared = configured.prepare(
            graph, stimulus, model, weights_v=syn.variables["w"].get_value(),
            amplitudes_v=np.asarray([_literal_amplitude(source) for source in inputs]),
            # The state-dependent Poisson prepass consumes due masks. Its
            # target simulation must retain the canonical marking dispatch.
            optimize_refractory=(fast_refractory and not model.numpy_policy_ir
                and not any(action in ('call','log') for _,action in model.numpy_error_actions)),
            allow_independent=False,
            allow_local_steps=not inputs,
            allow_arrival_scan=False,
            shared_expression_inputs=bool(inputs),
            layout_key=(len(group), sha256(sources).digest(), sha256(destinations).digest()),
        )
        prepared = original_codegen_policy(prepared, group)
        from .numpy_complex_warnings import bind as bind_complex_warnings
        prepared = bind_complex_warnings(prepared,group)
        if prepared.program.native_precision_limbs and previous is not None and previous.prepared.program.native_precision_limbs>prepared.program.native_precision_limbs:
            from .compiled_precision_frontier import rebind
            prepared=rebind(prepared,previous.prepared.program.native_precision_limbs,
                            lambda item:original_codegen_policy(item,group))
        custom_phases = (model.execution_phases != ("groups", "thresholds", "synapses", "resets")
                         or bool(prepared.program.scalar_stages))
        layout_time = time()
        state_values = {}
        for variable in configured._persistent(prepared.program):
            value = np.array(group.variables[variable.name].get_value(), copy=True)
            if value.dtype != np.dtype(variable.dtype) or value.shape != (1 if variable.scalar else len(group),):
                raise UnsupportedBrianNetwork(f"Unsupported shape or dtype for {variable.name}")
            if value.dtype in (np.dtype("float32"), np.dtype("float64")):
                value = value.view(np.uint32 if value.dtype.itemsize == 4 else np.uint64)
            elif variable.dtype=='bool' and variable.storage_dtype=='uint8':
                value=value.view(np.uint8)
            state_values[variable.name] = mx.array(value)
        start_step = int(clock.variables["timestep"].get_value()[0])
        end_step = int(clock._i_end)
        if previous is not None and previous.clock_rebase_pending:
            delta = start_step-previous.state.step
            if delta:
                old = previous.state
                previous.state = replace(old, step=start_step,
                    history=_rebase_event_words(old.history, delta, previous.prepared.window),
                    due=_rebase_event_words(old.due, delta, previous.prepared.window),
                    release=None)
            previous.fast_refractory = False
            previous.clock_rebase_pending = False
        if previous is not None and previous.state.step != start_step:
            raise UnsupportedBrianNetwork("Network clock no longer matches the native session")
        state = configured.State(configured._pack_values(prepared.program, state_values),
                                 previous.state.history if previous else {},
                                 previous.state.due if previous else {}, start_step,
                                 previous.state.release if previous and fast_refractory else None,
                                 spikes=(mx.array(np.isin(np.arange(len(group)),group.spikes).astype(np.uint32))
                                         if configured._uses_previous_events(model,prepared.program) else None))
        expression_rng=prepared.program.random_program is not None
        shared_poisson=bool(expression_rng and inputs)
        if expression_rng:
            from .expression_poisson_input import import_random
            state=replace(state,random=import_random(prepared,self))
        run_initial_state=state
        operator_error=prepared.program.runtime_error_type in ("zero_division", "model")
        if operator_error and inputs:
            from .brian_errors import RandomSnapshot
            rng_snapshot=RandomSnapshot(self)
        else:
            rng_snapshot=None
        failure=None
        # Invalid inputs fail in their original scheduled phase, after prior
        # expression and input draws have committed.
        invalid_poisson = bool(model.reference_codegen_target == "numpy" and any(
            source.active and not 0 <= float(source.rate*clock.dt) <= 1 for source in inputs))
        if shared_poisson:
            from .expression_poisson_input import ExpressionPoisson
            expression_sampler=ExpressionPoisson(inputs,clock,prepared.model.random_target)
        sampler = expression_sampler if shared_poisson else previous.sampler if (
            previous is not None and isinstance(
                previous.sampler, NativePoisson if model.delay_steps > 0 else _PoissonSampler)) else (
            (NativePoisson(self, inputs, clock, prepared.window,
                           fallback_codeobj_class=type(group.state_updater.codeobj)) if model.delay_steps > 0
             else _PoissonSampler(inputs, clock)) if inputs else None)
        if (inputs and model.delay_steps > 0
                and (prepared.program.scalar_parallel or prepared.program.scalar_input_independent)
                and model.execution_phases == ('groups','thresholds','synapses','resets')
                and sampler.can_fuse(prepared)):
            custom_phases=False
        unconditional_inputs=bool(inputs and model.delay_steps>0
                                  and not prepared.program.variables['v'].conditional_write)
        if custom_phases and inputs and not unconditional_inputs:
            # Stateful shared phases retain exact prefix/draw ordering. Pure
            # scalar snapshots can share the existing fused target-row loop.
            prepared=replace(prepared,window=1)
        began = time()
        next_report = began + float(report_period)
        if callback:
            callback(0*b.second, 0., t_start, duration)
        spike_indices, spike_steps = [], []
        recording_ordered = True
        spike_words, word_starts, word_lengths = [], [], []
        recording = any(monitor.active and monitor.record for monitor in monitors)
        counting = any(monitor.active for monitor in monitors)
        # Uniform output decoding accumulates on the host. Starting its zero
        # count there avoids a separate lazy GPU zero-fill/readback dispatch.
        counts = (np.uint32(0) if recording and prepared.independent_neurons
                  else mx.zeros((len(group),), mx.uint32)) if counting else None
        last_bits, last_length = None, 0
        zero_events = mx.zeros((max(1, len(inputs)),), mx.uint8)
        conditional = prepared.program.variables["v"].conditional_write
        prepass_s = general_submission_s = recording_s = 0.

        precision=None
        precision_error=None
        if prepared.program.native_precision_limbs:
            from .compiled_precision_frontier import PrecisionFrontier
            from .brian_errors import RandomSnapshot
            precision=PrecisionFrontier(prepared,state,
                transform=lambda item:original_codegen_policy(item,group),sampler=sampler,
                rng_snapshot=(lambda:RandomSnapshot(self)) if inputs and model.delay_steps==0 else None)

        def submit_window(p,s,step,length,times,*,precision_epoch_begin=None):
            nonlocal prepass_s
            if inputs and custom_phases and not shared_poisson:
                phase_time = time()
                if unconditional_inputs:
                    # Draw count/order does not depend on state. Batch
                    # the existing inversion loop in timestep/channel
                    # order, then execute the exact shared phase trace.
                    flags=mx.ones((len(group),),mx.bool_)
                else:
                    probe, _ = configured.advance(p, s, zero_events, end_step, 1,
                        event_begin=step, event_steps=1, event_times=times,
                        stop_before_synapses=True,precision_epoch_begin=precision_epoch_begin)
                    flags = (probe.values[conditional] if conditional else
                             mx.ones((len(group),), mx.bool_))
                if model.delay_steps > 0:
                    events = sampler.draw_conditional(flags,steps=length)
                else:
                    eligible_flags = np.asarray(flags, dtype=bool)
                    eligible = np.asarray([source.active and eligible_flags[target]
                                           for source, target in zip(inputs, targets)], bool)
                    events = mx.array(sampler.draw(eligible))
                prepass_s += time()-phase_time
            elif fused_poisson:
                events = zero_events
            elif inputs and model.delay_steps > 0:
                phase_time = time()
                events = sampler.events(p, s, length, times)
                prepass_s += time()-phase_time
            elif inputs:
                # Eligibility is observed at synapses, before a later reset
                # can write not_refractory again.
                probe, bits = configured.advance(p, s, zero_events, end_step, 1,
                    event_begin=step, event_steps=1, event_times=times,
                    stop_before_synapses=True,precision_epoch_begin=precision_epoch_begin)
                flags = (np.asarray(probe.values[conditional], dtype=bool)
                         if conditional else np.ones(len(group), bool))
                eligible = np.asarray([source.active and flags[target]
                                       for source, target in zip(inputs, targets)], bool)
                events = mx.array(sampler.draw(eligible))
            else:
                events = zero_events
            phase_time = time()
            s, bits = configured.advance(p, s, events, end_step, length,
                event_begin=step, event_steps=length, event_times=times,
                native_poisson=sampler if fused_poisson else None,precision_epoch_begin=precision_epoch_begin)
            return s,bits,phase_time

        def precision_window(p,s,origin,step,length,times):
            nonlocal general_submission_s
            s,bits,phase_time=submit_window(p,s,step,length,times,
                precision_epoch_begin=origin)
            general_submission_s += time()-phase_time
            return s,bits

        def flush_spike_words():
            nonlocal spike_words, word_starts, word_lengths, recording_s, counts
            nonlocal recording_ordered
            nonlocal state,prepared,precision_error,last_bits,last_length
            if precision is not None and precision.calls:
                result=precision.evaluate()
                prepared,state=result.prepared,result.state
                precision_error=result.error_kind
                spike_words=[word for _,_,word in result.windows]
                word_starts=[begin for begin,_,_ in result.windows]
                word_lengths=[length for _,length,_ in result.windows]
                if result.windows:last_bits,last_length=result.windows[-1][2],result.windows[-1][1]
                if precision_error is not None:
                    spike_words,word_starts,word_lengths=[],[],[]
                    return
            if not spike_words:
                return
            phase_time = time()
            decoded=None
            compacted=None
            uniform=None
            if recording and prepared.independent_neurons:
                from .expression_recording import collect_host_words
                uniform=collect_host_words(spike_words,word_starts,word_lengths)
            if uniform is not None:
                decoded=uniform[:2]
                counts=(np.uint32((int(counts)+uniform[2]) & 0xffffffff)
                        if isinstance(counts, np.uint32) and isinstance(uniform[2], int)
                        else np.asarray(counts)+uniform[2])
            elif counting:
                counts = _count_spike_words(spike_words, word_lengths, counts)
            if recording and decoded is None:
                from .expression_recording import collect, ranked_reset
                from .expression_window import _shared_normal_threshold
                if (prepared.independent_neurons or ranked_reset(prepared.program) or prepared.program.shared_expression_inputs
                        and _shared_normal_threshold(prepared.program)):
                    decoded=collect(spike_words,word_starts,word_lengths)
                if decoded is None:
                    compacted=collect_words(spike_words,capacity=65536,allow_split=True)
                    masks=compacted[2]
                    # Dense active words otherwise produce a large host
                    # sort. Sparse words retain their existing compact path.
                    if masks.size*4 >= len(spike_words)*len(group):
                        decoded=collect(spike_words,word_starts,word_lengths)
            if decoded is not None:
                spike_indices.append(decoded[0]);spike_steps.append(decoded[1])
            elif recording:
                recording_ordered = False
                windows, rows, masks = compacted
                starts = np.asarray(word_starts, np.int64)
                if cython_limit==64 and np.count_nonzero(masks&(masks-np.uint32(1)))<masks.size//4:
                    # Sparse words need one pass per set bit, not per bit position.
                    # frexp's exponent is exact for every uint32 power of two.
                    remaining=masks.copy()
                    while np.any(remaining):
                        low=remaining&(-remaining)
                        select=low!=0
                        bit=np.frexp(low[select])[1]-1
                        spike_indices.append(rows[select].astype(np.int32))
                        spike_steps.append(starts[windows[select]]+bit)
                        remaining^=low
                else:
                    for bit in range(max(word_lengths)):
                        select = (masks & (1 << bit)) != 0
                        if np.any(select):
                            spike_indices.append(rows[select].astype(np.int32))
                            spike_steps.append(starts[windows[select]]+bit)
            elif not counting:
                configured.evaluate_state(state)
            spike_words, word_starts, word_lengths = [], [], []
            recording_s += time()-phase_time

        fused_poisson = shared_poisson or bool(inputs and not custom_phases and model.delay_steps > 0 and sampler.can_fuse(prepared))
        from .expression_window import cython_batch_eligible,cython_submission_limit
        cython_windows=(precision is None and (shared_poisson or not inputs)
            and model.delay_steps>0 and cython_batch_eligible(prepared))
        cython_limit=cython_submission_limit(prepared,sampler if shared_poisson else None)if cython_windows else 32
        tape_end = start_step
        if invalid_poisson and start_step < end_step:
            from .brian_errors import replay_failure
            failure = replay_failure(prepared, run_initial_state, end_step, inputs=inputs,
                clock=clock, device=self, rng_snapshot=rng_snapshot, group_spikes=group.spikes)
            state = failure.state
            spike_indices, spike_steps = [failure.indices], [failure.steps]
            fast_refractory = False
        from .numpy_policy_context import PolicyContext
        policy_ir=(model.numpy_policy_ir or bool(expression_rng and any(
            getattr(operation,'function',None) in ('numpy_status','numpy_complex_cast',
                'numpy_python_complex_store','numpy_complex_ufunc_error')
            for operations in getattr(prepared.program.random_program,'stages',{}).values()
            for operation in operations)))
        policy_context=(PolicyContext(network,group,syn,monitors,clock,self)
                        if policy_ir and any(action in ('call','log')
                                             for _,action in model.numpy_error_actions) else None)
        with ExitStack() as transaction:
            if policy_context is not None:transaction.enter_context(policy_context)
            while failure is None and state.step < end_step and not network._stopped and not b.Network._globally_stopped:
                step = state.step
                network.t_ = step * model.dt_s
                length = min((1 if step==start_step else cython_limit)if cython_windows else prepared.window-step % prepared.window, end_step-step)
                if inputs and model.delay_steps > 0 and not shared_poisson:
                    if step == tape_end:
                        if precision is not None and precision.calls:flush_spike_words()
                        if precision_error is not None:break
                        transaction.close()
                        tape_end = step + sampler.chunk_steps(end_step-step)
                        transaction.enter_context(sampler.transaction(tape_end-step))
                    length = min(length, tape_end-step)
                if inputs and model.delay_steps == 0:
                    length = 1
                times = mx.array(configured._clock_times(step, step+length, model.dt_s))
                if precision is not None:
                    operation=lambda pp,ss,origin,step=step,length=length,times=times:precision_window(pp,ss,origin,step,length,times)
                    state,bits=precision.submit(operation,length)
                else:
                    state,bits,phase_time=submit_window(prepared,state,step,length,times)
                    if policy_context is not None:
                        length=min(length,max(1,state.step-step+int(state.policy_error is not None)))
                        prepared=policy_context.refreshed(prepared)
                        policy_context.flush_monitors()
                    general_submission_s += time()-phase_time
                if cython_windows and length>32:
                    for begin in range(0,length,32):
                        word=bits[(begin//32)*len(group):(begin//32+1)*len(group)]
                        last_bits,last_length=word,min(32,length-begin)
                        spike_words.append(word);word_starts.append(step+begin);word_lengths.append(last_length)
                else:
                    last_bits, last_length = bits, length
                    spike_words.append(bits)
                    word_starts.append(step)
                    word_lengths.append(length)
                collected = len(spike_words) >= 64 or state.step == end_step
                if collected:
                    flush_spike_words()
                if state.policy_error is not None:
                    self.native_steps += state.step-step
                    from .brian_errors import FailureReplay
                    flush_spike_words()
                    stage,cause=state.policy_error
                    failure=FailureReplay(state,
                        np.concatenate(spike_indices) if spike_indices else np.empty(0,np.int32),
                        np.concatenate(spike_steps) if spike_steps else np.empty(0,np.int64),
                        np.flatnonzero(np.asarray(state.spikes)).astype(np.int32),stage,0,cause=cause)
                    break
                self.native_steps += length
                if collected and operator_error:
                    # Use the existing host collection boundary. A bad input
                    # must not submit its entire requested future horizon;
                    # normal runs keep their fused windows and no step polling.
                    try:
                        if precision_error is not None:
                            from .runtime_errors import model_exception
                            raise model_exception(prepared.program.runtime_errors,precision_error)
                        configured._check_index_error(state)
                    except (ZeroDivisionError, ValueError, TypeError, OverflowError, IndexError, FloatingPointError) as error:
                        from .runtime_errors import is_model_exception
                        from .expression_errors import ExpressionRuntimeError
                        if (not isinstance(error, (ZeroDivisionError, ExpressionRuntimeError)) and not is_model_exception(error)
                    and not (isinstance(error,IndexError) and expression_rng
                        and model.reference_codegen_target=='numpy')):
                            raise
                        break
                elapsed = time() - began
                if callback and time() >= next_report:
                    callback(elapsed*b.second, (step*model.dt_s-float(t_start))/duration_s,
                             t_start, duration)
                    next_report = time() + float(report_period)
                if self._maximum_run_time is not None and elapsed > float(self._maximum_run_time):
                    network._stopped = True
            flush_spike_words()
        try:
            if precision_error is not None:
                from .runtime_errors import model_exception
                raise model_exception(prepared.program.runtime_errors,precision_error)
            configured.evaluate_state(state)
        except (ZeroDivisionError, ValueError, TypeError, OverflowError, IndexError, FloatingPointError) as error:
            from .numpy_error_policy import NumpyFloatingStatus
            if isinstance(error,NumpyFloatingStatus):
                if getattr(self,'_numpy_policy_replay',False):
                    raise RuntimeError('A floating policy trap escaped operation-segment replay') from error
                if rng_snapshot is not None:
                    rng_snapshot.restore()
                network.t_=float(t_start)
                clock.variables['timestep'].get_value()[0]=start_step
                clock.variables['t'].get_value()[0]=start_step*model.dt_s
                if previous is not None:
                    self._sessions[network]=replace(previous,sampler=None)
                self.native_steps=entry_native_steps
                self._numpy_policy_replay=True
                try:
                    return self.network_run(network,duration,report,report_period,
                                            namespace=namespace,profile=profile,level=level)
                finally:
                    self._numpy_policy_replay=False
            from .runtime_errors import is_model_exception
            from .expression_errors import ExpressionRuntimeError
            if (not isinstance(error, (ZeroDivisionError, ExpressionRuntimeError)) and not is_model_exception(error)
                    and not (isinstance(error,IndexError) and expression_rng
                        and model.reference_codegen_target=='numpy')):
                raise
            from .brian_errors import replay_failure
            failure=replay_failure(prepared,run_initial_state,end_step,inputs=inputs,clock=clock,
                device=self,rng_snapshot=rng_snapshot,group_spikes=group.spikes)
            state=failure.state
            spike_indices=[failure.indices]
            spike_steps=[failure.steps]
            fast_refractory=False
        execution_end = time()
        if expression_rng:
            from .expression_poisson_input import export_random
            export_random(state.random,self)
        host_values = configured.host_state(state, prepared)
        for name, values in host_values.items():
            variable=prepared.program.variables[name]
            group.variables[name].get_value()[:] = values[0,:1] if variable.scalar else values[0]
        uniform_output = (failure is None and len(spike_indices) == 1
                          and spike_indices[0].ndim == 2)
        if not uniform_output:
            spike_indices = [rows.reshape(-1) for rows in spike_indices]
            spike_steps = [times.reshape(-1) for times in spike_steps]
        if recording_ordered and failure is None and len(spike_indices) == 1:
            indices, steps = spike_indices[0], spike_steps[0]
        else:
            indices = np.concatenate(spike_indices) if spike_indices else np.empty(0, np.int32)
            steps = np.concatenate(spike_steps) if spike_steps else np.empty(0, np.int64)
        from .expression_recording import ordered_chunks
        if (failure is not None or not recording_ordered
                or not ordered_chunks(spike_indices, spike_steps)):
            event_order = np.lexsort((indices, steps))
            indices, steps = indices[event_order], steps[event_order]
        count_values = (np.bincount(indices, minlength=len(group)) if failure is not None else
                        np.asarray(counts).astype(np.int64) if counting else np.zeros(len(group), np.int64))
        total_spikes = indices.size if recording else int(count_values.sum())
        for monitor in monitors:
            if policy_context is not None:
                continue
            if not monitor.active:
                continue
            old_size = int(monitor.variables["N"].get_value()[0])
            if monitor.record:
                monitor.resize(old_size + indices.size)
                rows = monitor.variables["i"].get_value()[old_size:]
                times = monitor.variables["t"].get_value()[old_size:]
                if uniform_output:
                    rows.reshape(indices.shape)[:] = indices
                    times.reshape(steps.shape)[:] = np.multiply(
                        steps[:, 0], model.dt_s, dtype=np.float64)[:, None]
                else:
                    rows[:] = indices
                    np.multiply(steps, model.dt_s, out=times, dtype=np.float64)
            monitor.variables["count"].get_value()[:] += count_values
            monitor.variables["N"].get_value()[0] = np.asarray(old_size+total_spikes).astype(np.int32)
        if failure is not None or state.step > start_step:
            final = (failure.group_spikes if failure is not None else
                     np.flatnonzero(np.asarray(last_bits) & (1 << (last_length-1))))
            spikespace = group.variables["_spikespace"].get_value()
            spikespace[:len(final)] = final
            spikespace[-1] = len(final)
        clock.variables["timestep"].get_value()[0] = state.step
        clock.variables["t"].get_value()[0] = state.step*model.dt_s
        network.t_ = float(clock.variables['t'].get_value()[0]) if failure is not None or network._stopped or b.Network._globally_stopped else float(t_end)
        self._last_run_time = time()-began
        self._last_run_completed_fraction = ((network.t-float(t_start)*b.second)/duration
                                             if duration_s > 0 else 1.)
        for obj in objects:
            if failure is None and isinstance(obj, Group):
                obj._check_for_invalid_states()
        if callback:
            callback(self._last_run_time*b.second, 1., t_start, duration)
        if failure is None:
            network.after_run()
            group.variables["_spikespace"].get_value()[-1] = 0
        network._profiling_info = None
        # Before-threshold synapses have not enqueued the final threshold
        # eventspace yet; Brian's thresholder after_run clears it. Remove only
        # that last emission, retaining all arrivals already queued at synapses.
        if failure is None and model.execution_phases.index("synapses") < model.execution_phases.index("thresholds") and state.step > start_step:
            emission = state.step-1
            q, bit = divmod(emission, prepared.window)
            history = dict(state.history)
            if q in history:
                history[q] = history[q] & np.uint32(0xffffffff ^ (1 << bit))
            q, bit = divmod(emission+model.arrival_delay_steps, prepared.window)
            due = dict(state.due)
            if q in due:
                due[q] = due[q] & np.uint32(0xffffffff ^ (1 << bit))
            state = replace(state, history=history, due=due)
        if failure is None:
            state = replace(state, spikes=(configured._empty_words(len(group))
                                          if configured._uses_previous_events(model,prepared.program) else None))
        self._sessions[network] = _Session(topology, state, sampler, prepared,
            {name: _session_snapshot(host_values[name][0], group.variables[name].get_value(),
                                    native_output=failure is None and state.step > start_step)
             for name in ("rfc", "lastspike")}, fast_refractory)
        self.native_runs += failure is None
        self.last_timings = dict(
            host_preparation_s=prepared_host_time-entry_time,
            analysis_s=analysed_time-prepared_host_time,
            layout_s=layout_time-analysed_time,
            import_and_rng_setup_s=began-layout_time,
            execution_s=execution_end-began,
            writeback_s=time()-execution_end,
            prepass_and_dependencies_s=prepass_s,
            general_submission_s=general_submission_s,
            recording_and_dependencies_s=recording_s,
        )
        if failure is not None:
            from brian2.core.base import BrianObjectException
            runner=(failure.source if failure.source is not None else {"subexpression":group.subexpression_updater,"state":group.state_updater,"threshold":group.thresholder["spike"],
                    "reset":group.resetter.get("spike")}[failure.stage])
            codeobj=runner.codeobj
            if failure.cause is not None:
                cause = failure.cause
            elif prepared.program.runtime_error_type == 'model':
                from .runtime_errors import model_exception
                cause = model_exception(prepared.program.runtime_errors, failure.kind)
            else:
                cause=ZeroDivisionError("float division" if failure.kind==2 else "float divmod()")
            message=("An exception occured during the execution of the "
                     f"'run' block of code object '{codeobj.name}'.\n")
            if failure.source is not None:
                from brian2.core.functions import Function
                function = next(value for value in failure.source.variables.values()
                                if isinstance(value, Function))
                line = next(line for line in codeobj.code.run.splitlines()
                            if function.name+'(' in line)
                message = ("An exception occured during the execution of the "
                           f"'run' block of code object {codeobj.name}.\n"
                           "The error was raised in the following line:\n"+line+"\n")
            raise BrianObjectException(message,codeobj.owner) from cause

    def network_store(self, *args, **kwargs):
        raise UnsupportedBrianNetwork("Native delayed events are not yet included in Network.store")

    def network_restore(self, *args, **kwargs):
        raise UnsupportedBrianNetwork("Native delayed events are not yet included in Network.restore")


def _worker_initialize(settings, previous_initializer, previous_initargs):
    """Install before loky deserializes/calls the original run_trial."""
    b.set_device(MLXDevice())
    for name, value in settings["preferences"].items():
        b.prefs[name] = value
    b.defaultclock.dt = settings["dt_s"]*b.second
    if previous_initializer is not None:
        previous_initializer(*previous_initargs)


@contextmanager
def original_api(*, workers=True):
    """Select the device and install its loky bootstrap for unchanged run_exp.

    Use this before constructing Brian objects. On exit the caller's previous
    device and joblib backend registration are restored. Existing worker pools
    are replaced when their initializer/settings differ, as loky normally does.
    """
    from joblib.parallel import BACKENDS, register_parallel_backend
    from joblib._parallel_backends import LokyBackend

    device = MLXDevice()
    original_dt = b.defaultclock.dt
    previous_backend = BACKENDS["loky"]
    preference_names = ("codegen.target", "core.default_float_dtype", "core.default_integer_dtype",
                        "legacy.refractory_timing", "core.network.default_schedule")
    settings = {"dt_s": float(original_dt),
                "preferences": {name: b.prefs[name] for name in preference_names}}

    class NativeLokyBackend(LokyBackend):
        def configure(self, n_jobs=1, parallel=None, prefer=None, require=None,
                      idle_worker_timeout=300, **memmapping_executor_kwargs):
            initializer = memmapping_executor_kwargs.pop("initializer", None)
            initargs = memmapping_executor_kwargs.pop("initargs", ())
            return super().configure(
                n_jobs=n_jobs, parallel=parallel, prefer=prefer, require=require,
                idle_worker_timeout=idle_worker_timeout,
                initializer=_worker_initialize, initargs=(settings, initializer, initargs),
                **memmapping_executor_kwargs,
            )

    b.set_device(device)
    if workers:
        register_parallel_backend("loky", NativeLokyBackend)
    try:
        yield device
    finally:
        if workers:
            BACKENDS["loky"] = previous_backend
        b.defaultclock.dt = original_dt
        reset_device()
