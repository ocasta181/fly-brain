"""Configurable native execution with ordered, typed synaptic delivery.

The default integer-count engine remains a guarded specialization.  This path
uses Brian's analysed model statements and exact raw-bit arithmetic on Metal.
Delivery uses marked windows, direct arrival scans, or proven independent
components. Incoming traversal preserves source/edge order without floating atomics.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
from operator import index
import re
from time import perf_counter
from typing import Mapping
from weakref import ref

import mlx.core as mx
import numpy as np

from .models import Connectome, SimulationRun, SpikeEvents, Stimulus
from .model import ResolvedModel, DEFAULT_THRESHOLD, _expression_key
from .window import _event_chunk
from .state_layout import StateLayout
from .stimuli import event_input_snapshot, remember_event_hashes


@dataclass(frozen=True)
class Prepared:
    model: ResolvedModel
    program: object
    neurons: int
    targets: tuple[int, ...]
    window: int
    incoming_offsets: mx.array
    incoming_sources: mx.array
    weights: mx.array
    outgoing_offsets: mx.array
    outgoing_destinations: mx.array
    channel_offsets: mx.array
    channels: mx.array
    amplitudes: mx.array
    amplitude_bits: bytes
    compact_idle: bool
    # Namespace data shares one raw uint64 buffer instead of consuming one
    # Metal buffer binding per TimedArray.
    input_arrays: tuple[mx.array, ...]
    independent_neurons: bool
    scan_arrivals: bool = False
    components: object | None = None
    local_synapses: bool = False


@dataclass(frozen=True)
class State:
    values: Mapping[str, mx.array]
    history: Mapping[int, mx.array]
    due: Mapping[int, mx.array]
    step: int = 0
    release: mx.array | None = None
    check_index_error: bool = False
    # Eventspace survives until thresholds, independently of spike recording.
    spikes: mx.array | None = None
    runtime_error_type: str = "index"
    runtime_errors: tuple[tuple[str, str], ...] = ()
    runtime_error_order: str = "row"
    program_identity: tuple = ()
    native_precision_limbs: int = 0
    random: object | None = None
    policy_error: object | None = None
    # Negative absolute clocks need one stable nonnegative error-epoch origin.
    # None preserves the existing positive-clock encoding and kernel identity.
    error_epoch_origin: int | None = None


_LAYOUTS = OrderedDict()
_PROGRAMS = OrderedDict()
# Successful probe shapes, without retaining model/namespace arrays. This is a
# bounded preparation cache; actual simulation outputs are never cached.
_WARMUPS = OrderedDict()
_TARGET_ROWS = OrderedDict()
_STAGED_INPUTS = OrderedDict()
_MAX_STAGED_INPUT_BYTES = 128 * 1024 * 1024
ARRIVAL_CACHE = 4


@lru_cache(maxsize=8)
def _empty_words(size):
    with mx.stream(mx.gpu):
        result = mx.array(np.zeros(size, np.uint32))
        mx.eval(result)
    return result


def _amplitude_values(stimulus, model):
    if stimulus.amplitudes_mv is None:
        if not stimulus.targets:
            return np.empty(0, np.float64)
        return np.full(len(stimulus.targets), model.poisson_weight_v, np.float64)
    values = np.asarray(stimulus.amplitudes_mv, np.float64)
    if values.shape != (len(stimulus.targets),) or not np.all(np.isfinite(values)):
        raise ValueError("Input amplitudes must be finite and match the channels")
    return values * .001


def _persistent(program):
    return tuple(v for v in program.variables.values() if v.persistent)


def _uses_previous_events(model,program):
    return model.reset_uses_previous_spikes or 'threshold' in program.disabled_stages


def _compile_expression_model(model,neurons,native_precision_limbs,shared_expression_inputs=False):
    from .compiler import compile_model
    from .expression_fixed import FixedRandomProgram
    from .model import model_preferences
    program=compile_model(model,neurons=neurons,native_precision_limbs=native_precision_limbs,
                          shared_expression_inputs=shared_expression_inputs)
    if program.random_program is not None and not isinstance(program.random_program,FixedRandomProgram):
        from .expression_profile import apply
        with model_preferences(model):program=apply(program,model,neurons)
    return program


def _compiled_program(model, neurons=1, *, native_precision_limbs=0,shared_expression_inputs=False):
    """Reuse preparation only for the model's complete immutable identity."""
    # Selected statements and their metadata distinguish independently
    # analysed snapshots while allowing equal immutable snapshots to reuse
    # compilation. Object identity would recompile every managed API call.
    if not model.cacheable:
        return _compile_expression_model(model,neurons,native_precision_limbs,shared_expression_inputs)
    import brian2 as b
    from brian2.devices.device import get_device

    # Analysis-free/default models need not bind every Brian preparation
    # preference. Conservatively include all ambient compiler/core policy,
    # including mutable flag lists, without changing the caller's settings.
    policy = (type(get_device()), tuple(
        (name, repr(b.prefs[name])) for name in sorted(b.prefs)
        if name.startswith(('core.', 'codegen.', 'legacy.'))))
    analysis_key=model.analysis.compilation_key if model.analysis is not None else None
    key = (model.signature, analysis_key, policy, neurons,native_precision_limbs,shared_expression_inputs)
    if key in _PROGRAMS:
        _PROGRAMS.move_to_end(key)
        return _PROGRAMS[key][1]
    program = _compile_expression_model(model,neurons,native_precision_limbs,shared_expression_inputs)
    _PROGRAMS[key] = (model.analysis, program)
    while len(_PROGRAMS) > 2:
        _PROGRAMS.popitem(last=False)
    return program


@lru_cache(maxsize=16)
def _state_layout(program):
    return StateLayout.from_program(program, scalar=False)


class _PackedValues(Mapping):
    """Named typed views sharing a fixed set of physical state buffers."""

    def __init__(self, layout, buffers, size, shared=None):
        self.layout, self.buffers, self.size = layout, tuple(buffers), size
        self.shared = {} if shared is None else dict(shared)
        self._views = {}
        if len(self.buffers) != len(layout.groups):
            raise ValueError("Packed state buffers differ from the model layout")
        for group, buffer, shape in zip(layout.groups, self.buffers, layout.shapes(size)):
            if buffer.shape != shape or buffer.dtype != _storage_dtype(group.dtype):
                raise ValueError("Packed state buffer shape or dtype differs from its layout")

    def __getitem__(self, name):
        if name in self.shared:return self.shared[name]
        if name not in self._views:
            group, slot = self.layout.slots[name]
            value=self.buffers[group][slot*self.size:(slot+1)*self.size]
            dtype=_storage_dtype(self.layout.dtypes[name])
            self._views[name]=value if value.dtype==dtype else value.view(dtype)
        return self._views[name]

    def __iter__(self):
        return iter(self.layout.names + tuple(self.shared))

    def __len__(self):
        return len(self.layout.names) + len(self.shared)


def _pack_values(program, values):
    layout = _state_layout(program)
    if isinstance(values, _PackedValues) and values.layout is layout:
        return values
    size = values[layout.names[0]].size
    layout.shapes(size)
    buffers = []
    for group in layout.groups:
        arrays = [values[name] for name in group.names]
        if any(array.shape != (size,) or array.dtype != _storage_dtype(layout.dtypes[name])
               for name,array in zip(group.names,arrays)):
            raise ValueError("State variable shape or dtype differs from the model")
        dtype=_storage_dtype(group.dtype)
        arrays=[array if array.dtype==dtype else array.view(dtype) for array in arrays]
        buffers.append(mx.concatenate(arrays) if len(arrays) > 1 else arrays[0])
    return _PackedValues(layout, buffers, size, {v.name:values[v.name] for v in _persistent(program) if v.scalar})


def _state_buffers(values):
    return values.buffers+tuple(values.shared.values()) if isinstance(values, _PackedValues) else tuple(values.values())


def _storage_dtype(dtype):
    return {"float64": mx.uint64, "float32": mx.uint32,
            "int8": mx.int8, "uint8": mx.uint8, "int16": mx.int16, "uint16": mx.uint16,
            "int32": mx.int32, "uint32": mx.uint32, "int64": mx.int64, "uint64": mx.uint64,
            "bool": mx.bool_}[dtype]


def _ordered_layout(sources, destinations, neurons):
    """Stable O(edges+neurons) buckets, retaining every duplicate edge.

    pandas already supplies the application's table loader. Its group-sort
    primitive is a stable counting scatter for dense integer labels; composing
    source then destination buckets gives Brian's source/original-edge order
    without a comparison sort over the full connectome.
    """
    from pandas._libs.algos import groupsort_indexer

    outgoing_order, out_counts = groupsort_indexer(np.asarray(sources, np.intp), neurons)
    destination_order, in_counts = groupsort_indexer(
        np.asarray(destinations[outgoing_order], np.intp), neurons,
    )
    incoming_order = outgoing_order[destination_order]
    in_offsets = np.r_[0, np.cumsum(in_counts[1:])].astype(np.int32)
    out_offsets = np.r_[0, np.cumsum(out_counts[1:])].astype(np.int32)
    return incoming_order, outgoing_order, in_offsets, out_offsets
def _dtype(variable):
    return _storage_dtype(variable.dtype)


def _target_rows(raw):
    """Reuse normalization only for immutable tuples of exact Python ints."""
    previous = _TARGET_ROWS.get(id(raw))
    if previous is not None and previous[0] is raw:
        _TARGET_ROWS.move_to_end(id(raw))
        return previous[1], ('target-identity', id(previous[1]))
    targets = tuple(map(index, raw))
    if (type(raw) is tuple and len(raw) <= 262144
            and all(type(row) is int for row in raw)):
        # Retain both tuples, preventing recycled ids. Foreign __index__
        # objects are normalized afresh even if they compare equal to ints.
        _TARGET_ROWS[id(raw)] = (raw, targets)
        while len(_TARGET_ROWS) > 4:
            _TARGET_ROWS.popitem(last=False)
        return targets, ('target-identity', id(targets))
    return targets, targets


def prepare(connectome: Connectome, stimulus: Stimulus, model: ResolvedModel,
            silenced=(), *, weights_v=None, amplitudes_v=None,
            optimize_refractory=True, layout_key=None, allow_independent=True,
            component=False, allow_arrival_scan=True,shared_expression_inputs=False,
            allow_local_steps=False) -> Prepared:
    n = connectome.neuron_ids.size
    if model.integer_policy is not None:
        policy = model.integer_policy.execution(n, stimulus.events.shape[0])
        identity = None if model.signature is None else sha256(
            ((model.source_signature or model.signature) + policy.signature).encode()).hexdigest()
        model = replace(model, integer_policy=policy, signature=identity)
    program = _compiled_program(model, n,shared_expression_inputs=shared_expression_inputs)
    if program.variables["g"].scalar:
        raise SyntaxError("Writing to scalar variable g_post not allowed in this context.")
    source_program = program
    if not optimize_refractory:
        # Imported real Brian state can contain edited rfc/lastspike values.
        # The resident release cache is only valid for canonical initialization.
        program = replace(program, refractory_elapsed_threshold=None,cpu_refractory_guard=False)
    targets, target_key = _target_rows(stimulus.targets)
    silenced = tuple(sorted(set(index(row) for row in silenced)))
    # No edges imply no cross-neuron delivery/synchronization boundary,
    # independently of state storage width. Fused RNG callers can opt out.
    local_steps = bool(allow_local_steps and not targets and model.arrival_delay_steps == 0
                       and model.execution_phases == ("groups", "thresholds", "synapses", "resets")
                       and not program.collective_reset and not program.scalar_stages
                       and not any(variable.scalar for variable in _persistent(program))
                       and program.random_program is None
                       and (program.index_error_name is None
                            or program.runtime_error_type in ("zero_division", "model"))
                       and program.replay_phase is None and not _uses_previous_events(model, program)
                       and np.array_equal(connectome.sources, connectome.destinations))
    independent_neurons = (allow_independent and connectome.sources.size == 0) or local_steps
    window = (1 if program.collective_reset else 32 if independent_neurons
              else max(1, min(32, model.arrival_delay_steps)))
    # Explicit amplitudes use the public mV interface; the model's shared
    # default amplitude retains the exact upstream SI multiplication.
    amplitude_v = (_amplitude_values(stimulus, model) if amplitudes_v is None
                   else np.asarray(amplitudes_v, np.float64))
    if amplitude_v.shape != (len(targets),):
        raise ValueError("Input amplitudes must match the channels")
    exact_weights = None if weights_v is None else np.asarray(weights_v, np.float64)
    identity = connectome.pin if layout_key is None else layout_key
    key = (identity, n, connectome.coefficient_dtype, model.signature, id(model.analysis), source_program, target_key, silenced, amplitude_v.tobytes(), allow_independent, allow_local_steps, bool(component), bool(allow_arrival_scan),
           None if exact_weights is None else sha256(exact_weights).digest(), optimize_refractory)
    if identity is not None and model.cacheable and key in _LAYOUTS:
        _LAYOUTS.move_to_end(key)
        return _LAYOUTS[key]
    if n <= 0 or any(row < 0 or row >= n for row in targets + silenced):
        raise ValueError("Invalid neuron count or input target")
    if model.arrival_delay_steps >= 2**63:
        raise OverflowError("Synaptic delay exceeds the signed 64-bit clock")
    sources = np.asarray(connectome.sources)
    destinations = np.asarray(connectome.destinations)
    counts = np.asarray(connectome.counts)
    if (sources.ndim != 1 or destinations.shape != sources.shape
            or counts.shape != sources.shape or sources.size > np.iinfo(np.int32).max
            or np.any(sources < 0) or np.any(sources >= n)
            or np.any(destinations < 0) or np.any(destinations >= n)
            or counts.dtype.kind not in 'buif'):
        raise ValueError("Invalid connectivity indices or coefficients")
    from .inputs import coefficient_weights_si

    weights = (coefficient_weights_si(connectome, model.source_namespace.get('w_syn', model.w_syn))
               if exact_weights is None else exact_weights.copy())
    if exact_weights is None and model.state_dtype == "float32":
        weights = weights.astype(np.float32).astype(np.float64)
    if weights.shape != sources.shape:
        raise ValueError("SI weights must match the edges")
    if silenced:
        weights[np.isin(sources, silenced)] = 0
    # Brian's presynaptic spike queue processes increasing source indices and
    # each source's edges in their original connection order.
    incoming_order, outgoing_order, in_offsets, out_offsets = _ordered_layout(sources, destinations, n)
    channel_order = np.argsort(targets, kind="stable").astype(np.int32)
    channel_offsets = np.r_[0, np.cumsum(np.bincount(targets, minlength=n))].astype(np.int32)
    all_targeted = bool(np.all(channel_offsets[:-1] != channel_offsets[1:]))
    scan_arrivals = bool(allow_arrival_scan and not independent_neurons and model.arrival_delay_steps > 0
                         and all_targeted and np.max(np.diff(in_offsets), initial=0) <= 8)
    components = None
    if (component and scan_arrivals and model.arrival_delay_steps <= 128
            and not program.collective_reset and not program.scalar_stages
            and not _uses_previous_events(model, program)
            and program.replay_phase is None and program.index_error_name is None):
        from .components import prepare_components
        components = prepare_components(connectome)
    nonempty = lambda array, dtype: np.asarray(array, dtype=dtype) if array.size else np.zeros(1, dtype)
    with mx.stream(mx.gpu):
        prepared = Prepared(
            model, program, n, targets, window,
            mx.array(in_offsets), mx.array(nonempty(sources[incoming_order], np.int32)),
            mx.array(nonempty(weights[incoming_order].view(np.uint64), np.uint64)),
            mx.array(out_offsets), mx.array(nonempty(destinations[outgoing_order], np.int32)),
            mx.array(channel_offsets), mx.array(nonempty(channel_order, np.int32)),
            mx.array(nonempty(amplitude_v.view(np.uint64), np.uint64)),
            amplitude_v.tobytes(),
            bool(program.equilibrium is not None
                 and program.refractory_elapsed_threshold is not None
                 and _expression_key(model.eq_th) == _expression_key(DEFAULT_THRESHOLD)
                 and model.v_0 <= model.v_th and model.arrival_delay_steps > 0
                 and model.execution_phases == ("groups", "thresholds", "synapses", "resets")
                 and program.index_error_name is None and not all_targeted),
            ((mx.array(np.concatenate([array.values.reshape(-1)
                                       for array in program.input_arrays.values()]).view(np.uint64)),)
             if program.input_arrays else ()), independent_neurons, scan_arrivals, components,
            local_steps and sources.size > 0,
        )
        if program.random_program is not None:
            from .expression_fixed import FixedRandomProgram
            if not isinstance(program.random_program,FixedRandomProgram):
                prepared=replace(prepared,window=32 if independent_neurons else window,compact_idle=False,
                    scan_arrivals=False,components=None)
        mx.eval(*_layout_arrays(prepared))
    if identity is not None and model.cacheable:
        _LAYOUTS[key] = prepared
        while len(_LAYOUTS) > 2:
            _LAYOUTS.popitem(last=False)
    return prepared


def _layout_arrays(p):
    return (p.incoming_offsets, p.incoming_sources, p.weights,
            p.outgoing_offsets, p.outgoing_destinations,
            p.channel_offsets, p.channels, p.amplitudes, *p.input_arrays,
            *((p.components.rows, p.components.positions) if p.components is not None else ()))


def initial_state(prepared: Prepared, trials=1,*,random_states=None):
    model = prepared.model
    n = prepared.neurons
    values = {}
    expression_initializers = None
    if isinstance(model.v_0, str) or isinstance(model.t_rfc, str):
        # Brian assignment code is host initialization, just as create_model
        # performs before a run. Execute it at the real shape, never on the
        # one-neuron analysis group and never as a CPU timestep fallback.
        import brian2 as b
        from .model import model_preferences
        expression_initializers = []
        with model_preferences(model):
            for _ in range(trials):
                group = b.NeuronGroup(n, model.eqs, method="linear",
                    threshold=model.eq_th, reset=model.eq_rst, refractory="rfc",
                    namespace=model.brian_namespace(), dt=model.dt_s*b.second,
                    dtype={variable.name: np.dtype(variable.dtype)
                           for variable in _persistent(prepared.program)},
                    name="mlx_initializer*")
                group._network = "mlx-host-initialization-only"
                namespace = model.brian_namespace()
                group.v = namespace["v_0"]
                group.g = 0
                group.rfc = namespace["t_rfc"]
                expression_initializers.append({name: group.variables[name].get_value().copy()
                                                for name in ("v", "g", "rfc")})
    for variable in _persistent(prepared.program):
        name = variable.name
        value = {"v": model.v_0, "g": 0., "rfc": model.t_rfc,
                 "lastspike": -1e4, "not_refractory": True}.get(name, 0)
        # Broadcast within one real neuron group first: a column vector must
        # not accidentally become valid when its length equals the trial count.
        if expression_initializers is not None and name in ("v", "g", "rfc"):
            array = np.stack([trial[name] for trial in expression_initializers])
        else:
            one_trial = np.broadcast_to(np.asarray(value, dtype=variable.dtype),
                                        (1 if variable.scalar else n,))
            array = np.broadcast_to(one_trial, (trials, one_trial.size)).copy()
        if name == "rfc" and prepared.targets:
            array[:, [0] if variable.scalar else list(prepared.targets)] = 0
        if variable.dtype in ("float64", "float32"):
            array = array.view(np.uint64 if variable.dtype == "float64" else np.uint32)
        elif variable.dtype=='bool' and variable.storage_dtype=='uint8':
            array=array.view(np.uint8)
        values[name] = mx.array(array.reshape(-1))
    state = State(_pack_values(prepared.program, values), {}, {}, program_identity=(model.signature,n), release=mx.zeros((trials*n,), mx.int64),
                  check_index_error=prepared.program.index_error_name is not None,
                  runtime_error_type=prepared.program.runtime_error_type,
                  runtime_errors=prepared.program.runtime_errors,
                  runtime_error_order=prepared.program.runtime_error_order,
                  spikes=(_empty_words(trials*n) if _uses_previous_events(model,prepared.program) else None))
    mx.eval(*_state_buffers(state.values))
    if prepared.program.random_program is not None:
        from .expression_runtime import initial
        state=replace(state,random=initial(prepared,trials,random_states))
    elif random_states is not None:
        raise ValueError('Random snapshots require a random-expression model')
    return state


def _variables_source(program, output_index="gid", packed=False, aliases=None):
    variables = _persistent(program)
    loads, stores = _state_layout(program).source(program, output_index, aliases=aliases)
    loads += "\n" + "\n".join(f"{v.metal_type} {(aliases or {}).get(v.name,v.code_name)} = 0;" for v in variables if v.scalar)
    return variables, loads, stores


def _state_aliases(program):
    # Keep the central code_name hook authoritative. The existing binary32
    # local prefix also keeps vector before-start fields apart from geometry
    # and fused-loop control names on the frozen canonical base.
    return {v.name:(v.code_name if v.code_name!=v.name else '_eon_state_'+v.name)
            for v in _persistent(program)}


def _state_source(code,aliases):
    if not aliases:return code
    pattern=r'(?<![.\w])('+ '|'.join(re.escape(name) for name in aliases)+r')\b(?!\s*\()'
    return re.sub(pattern,lambda match:aliases[match[1]],code)


def _poisson_increment(event):
    # A zero draw still multiplies its literal weight. Keep the finite
    # signed-zero shortcut, but preserve IEEE 0*infinity and NaN propagation.
    amplitude = "amplitudes[channel]"
    return (f"({event} ? {amplitude} : (({amplitude} & 0x7ff0000000000000ul) == "
            f"0x7ff0000000000000ul ? d_mul(0ul, {amplitude}) : ({amplitude} & (1ul << 63))))")


def _arrival_source(program, zero_delay, scalar_input32=False, aliases=None,
                    independent_neurons=False, *, poisson=True, scan_arrivals=False, component=False, local_synapses=False):
    aliases={**{name:var.code_name for name,var in program.variables.items()}, **(aliases or {})}
    g = program.variables["g"]
    v = program.variables["v"]
    condition_g = aliases.get(g.conditional_write,g.conditional_write) or "true"
    condition_v = aliases.get(v.conditional_write,v.conditional_write) or "true"
    g_name,v_name=aliases.get('g','g'),aliases.get('v','v')
    def add(variable, weight):
        if variable.dtype == "float32":
            if scalar_input32 and variable.name == "v":
                # Compiled PoissonInput stores its invariant weight in a
                # float32 temporary. NumPy instead multiplies its double
                # invariant by an integer event vector, retaining binary64.
                weight = f"f_to_d(f_from_d({weight}))"
            return f"f_from_d(d_add(f_to_d({aliases.get(variable.name,variable.name)}), {weight}))"
        return f"d_add({aliases.get(variable.name, variable.name)}, {weight})"
    source_bits = ("source_current[trial * N + source] & 1u" if zero_delay else
                   "((source_bit < W ? source_low[trial * N + source] : "
                   "source_high[trial * N + source]) >> (source_bit % W)) & 1u")
    marker = "true" if zero_delay or component else ("(_eon_arrival_bits & (1u << k))" if scan_arrivals else "(due_bits[gid] & (1u << k))")
    if component:
        from .components import _prefix
        prefix = _prefix(program)
        source_bits = f"{prefix}past[{prefix}slot*32+{prefix}positions[source]]"

    scan = f"""
            {"" if component else "int source_bit = SOURCE_OFFSET + k;"}
            for(int edge = incoming_offsets[row]; edge < incoming_offsets[row+1]; ++edge) {{
                int source = incoming_sources[edge];
                if({source_bits}) {g_name} = {add(g, 'weights[edge]')};
            }}
    """
    if not zero_delay and not component:
        cached = "\n".join(
            f"if(cached_mask_{j} & (1u << k)) {g_name} = {add(g, f'cached_weight_{j}')};"
            for j in range(ARRIVAL_CACHE))
        scan = f"if(arrival_overflow) {{ {scan} }} else {{ {cached} }}"
    if independent_neurons and local_synapses:
        # Only proven zero-delay self-local graphs enter this loop.
        # Retain every edge, including zero weights, in Brian's original order.
        marker = "_spike"
        scan = f"""
            for(int edge = incoming_offsets[row]; edge < incoming_offsets[row+1]; ++edge) {{
                {g_name} = {add(g, 'weights[edge]')};
            }}
        """
    input_event = "events[(trial * staged_steps + step - event_begin) * C + channel]"
    input_delivery = (f"if({input_event} != 2u) {{\n"
        f"ulong input_amplitude = {_poisson_increment(input_event + ' == 1u')};\n"
        f"{v_name} = {add(v, 'input_amplitude')};\n}}")
    synaptic_delivery='' if independent_neurons and not local_synapses else f"""
        if(({condition_g}) && ({marker})) {{
            {scan}
        }}
    """
    if v.scalar:
        from .shared_state import phase_load
        return synaptic_delivery+'\n'+phase_load(program,'synapses',aliases=aliases)
    if not poisson:
        return synaptic_delivery
    return synaptic_delivery+f"""
        if({condition_v}) {{
            for(int p = channel_offsets[row]; p < channel_offsets[row+1]; ++p) {{
                int channel = channels[p];
                {input_delivery}
            }}
        }}
    """


def _arrival_cache_source(scan_arrivals=False):
    declarations = "\n".join(f"uint cached_mask_{j}=0; ulong cached_weight_{j}=0;"
                             for j in range(ARRIVAL_CACHE))
    assignments = "\n".join(
        f"{'if' if j == 0 else 'else if'}(cached_count == {j}) "
        f"{{ cached_mask_{j}=bits; cached_weight_{j}=weights[edge]; }}"
        for j in range(ARRIVAL_CACHE))
    return f"""
        {declarations}
        bool arrival_overflow = false;
        int cached_count = 0;
        uint _eon_arrival_bits = 0u;
        if({"true" if scan_arrivals else "due_bits[gid]"}) {{
            for(int edge=incoming_offsets[row]; edge<incoming_offsets[row+1]; ++edge) {{
                long source=trial*N+incoming_sources[edge];
                uint bits=source_low[source] >> SOURCE_OFFSET;
                if(SOURCE_OFFSET > 0) bits |= source_high[source] << (W-SOURCE_OFFSET);
                bits &= 0xffffffffu >> (32-W);
                if(!bits) continue;
                {"_eon_arrival_bits |= bits;" if scan_arrivals else ""}
                if(cached_count == {ARRIVAL_CACHE}) {{ arrival_overflow=true; {"continue" if scan_arrivals else "break"}; }}
                {assignments}
                ++cached_count;
            }}
        }}
    """


@lru_cache(maxsize=16)
def _timed_clock_kernel(model, program):
    """Evaluate proven clock-only 1D samples once on GPU for a staged chunk."""
    from .binary64 import MSL_HEADER, literal
    from .binary32 import MSL_HEADER as MSL_BINARY32
    from .math64 import MSL_MATH_HEADER

    inputs = ["clock_times", "clock_begin", next(iter(program.input_arrays))]
    namespace_loads, offset = [], 0
    buffer = next(iter(program.input_arrays))
    for name, array in program.input_arrays.items():
        if name != buffer:
            namespace_loads.append(f"auto {name} = {buffer} + {offset}L;")
        offset += array.values.size
    samples = [f"samples[{query}L * STEPS + gid] = {call};"
               for query, call in enumerate(program.shared_timed_calls)]
    return mx.fast.metal_kernel(
        name="fly_timed_clock_"+model.signature[:12], input_names=inputs,
        output_names=["samples"], header=MSL_HEADER+MSL_BINARY32+MSL_MATH_HEADER+program.header_code,
        source=f"""
            uint gid = thread_position_in_grid.x;
            if(gid >= STEPS) return;
            const ulong dt = {literal(model.dt_s)};
            const long t_in_timesteps = clock_begin[0] + gid;
            const ulong t = clock_times[gid];
            {chr(10).join(namespace_loads)}
            {chr(10).join(samples)}
        """, compile_options={"math_mode": "safe"})


def _stage_timed_calls(prepared, clock_times, event_begin):
    if not prepared.program.shared_timed_calls:
        return None
    steps = clock_times.size
    return _timed_clock_kernel(prepared.model, prepared.program)(
        inputs=[clock_times, mx.array([event_begin], mx.int64), *prepared.input_arrays],
        template=[("STEPS", steps)], grid=(steps, 1, 1),
        threadgroup=(min(256, steps), 1, 1),
        output_shapes=[(steps*len(prepared.program.shared_timed_calls),)],
        output_dtypes=[mx.uint64])[0]


def _phase_parts(model, program, zero_delay, stop_before_synapses=False):
    if program.replay_phase is not None:
        return (program.replay_phase,), ()
    phases = model.execution_phases
    threshold = phases.index("thresholds")
    boundaries = [phases.index("synapses")] if zero_delay else []
    if program.collective_reset and phases.index("resets") > threshold:
        boundaries.append(min(phases.index(phase) for phase in ("resets", "synapses")
                              if phases.index(phase) > threshold))
    boundary = min(boundaries) if boundaries else len(phases)
    end = phases.index("synapses") if stop_before_synapses else len(phases)
    return phases[:min(boundary, end)], phases[boundary:end]


@lru_cache(maxsize=16)
def _neuron_kernel(model, program, zero_delay=False, finish=False, compact=False, fused_idle=False,
                   poisson=None, independent_neurons=False, stop_before_synapses=False, local_synapses=False,
                   scan_arrivals=False, component=False,expression_draws=False,expression_clock=False,
                   expression_prefix=False,expression_selection=False,expression_reset=None,expression_rank_flags=False,expression_reset_blocks=False,expression_due_fusion=False,expression_full_clock=False,
                   batched_lif_arithmetic=False,error_epoch_origin=None):
    from .binary64 import MSL_HEADER, literal
    from .binary32 import MSL_HEADER as SINGLE_HEADER
    from .compiler import _literal, _kind
    from .math64 import MSL_MATH_HEADER

    aliases=_state_aliases(program) if model.state_dtype=='float32' or program.scalar_stages else None
    names=aliases or {v.name:v.code_name for v in _persistent(program)}
    def state_code(code):
        return _state_source(code,aliases)
    variables, loads, stores = _variables_source(program, "work_index" if compact else "gid", aliases=aliases)
    groups = _state_layout(program).groups
    inputs = ["_eon_p_"+group.buffer_name for group in groups]+["p_release"]
    if _uses_previous_events(model,program): inputs += ["previous_spikes"]
    inputs += ["incoming_offsets", "incoming_sources", "weights", "channel_offsets",
               "channels", "amplitudes", "events", "clock_times", "params"]
    inputs += (["source_current"] + ([] if zero_delay else ["source_low", "source_high", "due_bits"])
               if finish else ["source_low", "source_high", "due_bits"])
    collective=program.collective_reset
    if collective:inputs += ['reset_ranks','reset_counts']
    if program.input_arrays: inputs += [next(iter(program.input_arrays))]
    if program.shared_timed_calls: inputs += [program.timed_cache_name]
    if program.scalar_stages: inputs += ["_eon_shared_trace"]
    if compact:
        inputs += ["active_ids", "active_count"]
    if expression_draws:
        inputs += (['_eon_expression_pool','_eon_expression_tail','_eon_expression_cursor',
                    '_eon_expression_done','_eon_expression_counts']if expression_prefix else['_eon_expression_draws'])
        if expression_prefix=='cython':
            inputs=inputs[:-5]+['_eon_expression_pool','_eon_expression_tail','_eon_expression_buffer','_eon_expression_metadata']
    outputs = ["_eon_q_" + group.buffer_name for group in groups] + ["q_release", "spike_bits"]
    if expression_selection:outputs+=['_eon_expression_ranks','_eon_expression_blocks']
    if expression_reset is not None:
        inputs+=['_eon_pending_draws','_eon_pending_ranks']
        if expression_reset_blocks:inputs+=['_eon_pending_blocks']
        outputs+=['_eon_pending_marker']
        if expression_due_fusion:inputs+=['_eon_pending_due']
    if poisson:
        assert not (zero_delay or finish or compact)
        inputs[inputs.index("channels")] = "uniforms"
        inputs[inputs.index("events")] = "poisson_meta"
        inputs += ["cursor_in", "overflow_in"]
        outputs += ["used", "overflow"]
    refractory = state_code(program.refractory_code)
    if program.refractory_elapsed_threshold is not None:
        refractory = f"""
        if(FAST_REFRACTORY)
            {names['not_refractory']} = step >= release_step;
        else {{ {refractory} }}
        """
    dynamics = state_code(program.dynamics_code)
    arithmetic_header = ""
    if batched_lif_arithmetic:
        # Reuse the independently checked integer-backed binary64 helpers.
        # They retain fused rounding and defer exceptional operands and
        # gradual underflow to the existing general arithmetic.
        from .operator64 import _OPERATOR_FMA_HEADER
        arithmetic_header = _OPERATOR_FMA_HEADER
        dynamics = dynamics.replace("d_fma(", "_eon_operator_polynomial_fma(")
        dynamics = dynamics.replace("d_mul(", "_eon_operator_mul(")
    if program.equilibrium is not None:
        at_equilibrium = " && ".join(f"{names[name]} == {_literal(value, _kind(np.asarray(value).dtype))}"
                                    for name, value in program.equilibrium.items())
        dynamics = f"if(!({at_equilibrium})) {{ {dynamics} }}"
    release_update = ""
    if program.refractory_elapsed_threshold is not None:
        distance = min(2**63-1, max(1, model.refractory_steps + 2))
        zero_rfc = (f"(({names['rfc']} & 0x7fffffffu) == 0u)" if program.variables["rfc"].dtype == "float32"
                    else f"d_eq({names['rfc']}, 0ul)")
        release_update = f"""
            if(FAST_REFRACTORY)
                release_step = {zero_rfc} ? step : fly_release_step(step, t, dt,
                    {literal(program.refractory_elapsed_threshold)}, {distance}L);
        """
    spike_time="f_from_d(t)" if program.variables["lastspike"].dtype == "float32" else "t"
    threshold = f"""
        bool _cond = false;
        {state_code(program.threshold_code)}
        if(_cond) {{ {names['lastspike']} = {spike_time}; {names['not_refractory']} = false; {release_update} }}
        _spike = _cond;
    """
    if 'threshold' in program.disabled_stages:threshold=program.threshold_code
    head, tail = _phase_parts(model, program, zero_delay, stop_before_synapses)
    phases = tail if finish else head
    if program.replay_phase is not None:
        phases = (program.replay_phase,)
    temporary_dtype = (model.analysis.temporary_dtype if model.analysis is not None
                       and model.analysis.temporary_dtype else model.state_dtype)
    arrivals = _arrival_source(program, zero_delay,
        model.float32_codegen != "numpy" and model.float32_optimise
        and temporary_dtype == "float32", aliases,
        independent_neurons, poisson=not poisson, scan_arrivals=scan_arrivals, component=component,
        local_synapses=local_synapses)
    reset = (state_code(program.reset_scalar_code)
             + f"\nif(_spike) {{ {state_code(program.reset_vector_code)} }}")
    blocks = {"groups": refractory+"\n"+dynamics, "thresholds": threshold,
              "synapses": arrivals, "resets": reset}
    if program.scalar_stages:
        from .shared_state import phase_source,phase_load
        blocks['subexpression']=state_code(phase_source(program,'subexpression',aliases=aliases))
        blocks['groups']=state_code(phase_source(program,'state',aliases=aliases))
        blocks['thresholds']=(state_code(phase_load(program,'threshold',aliases=aliases))+"\nbool _cond=false;\n"
            +state_code(program.scalar_stages['threshold'].vector_code)
            +f"\nif(_cond) {{ {names['lastspike']}={spike_time}; {names['not_refractory']}=false; {release_update} }}\n_spike=_cond;")
        blocks['resets']=state_code(phase_load(program,'reset',aliases=aliases))+"\nif(_spike) { "+state_code(program.scalar_stages['reset'].vector_code)+" }"
        if not program.variables['v'].scalar:
            blocks['synapses']=state_code(phase_load(program,'synapses',aliases=aliases))+"\n"+arrivals
    if 'threshold' in program.disabled_stages:
        blocks['thresholds']=state_code(program.threshold_code)
    indexing = ("if(work_index >= active_count[0]) return; uint gid = active_ids[work_index];"
                if compact else "uint gid = work_index;")
    output_index = "work_index" if compact else "gid"
    error_load = (f"bool {program.index_error_name} = p_release[gid] < 0L;"
                  if program.index_error_name is not None else "")
    # Cache release steps are nonnegative signed int64 values. Its unused sign
    # bit transports a sticky index error without another Metal buffer binding.
    release_load = ("p_release[gid] & 9223372036854775807L"
                    if program.index_error_name is not None else "p_release[gid]")
    release_store = (f"({program.index_error_name} ? (release_step | (-9223372036854775807L-1L)) : release_step)"
                     if program.index_error_name is not None else "release_step")
    error_kind_restore=""
    if program.runtime_error_type in ("zero_division", "model"):
        mask = 65535 if program.runtime_error_type == 'model' else 3
        error_kind_restore=(f"if(p_release[gid]<0L) {program.index_error_name}_kind = "
                            f"int(p_release[gid]&{mask}L);")
        release_store=(f"({program.index_error_name} ? ((-9223372036854775807L-1L) | "
                       f"long({program.index_error_name}_kind)) : release_step)")
    operation_errors = program.runtime_error_order == 'operation'
    precision_errors=bool(program.native_precision_limbs)
    if operation_errors or precision_errors:
        # The existing signed release word carries the first clock/phase and
        # vector operation as well as its exception kind. No extra binding.
        error_kind_restore += (f"ulong {program.index_error_name}_epoch = "
                               "(ulong(p_release[gid]) & 0x7ffffffffffffffful) >> 16;")
        release_store = (f"({program.index_error_name} ? long(0x8000000000000000ul | "
                         f"({program.index_error_name}_epoch << 16) | "
                         f"ulong({program.index_error_name}_kind)) : release_step)")
    namespace_loads = []
    namespace_offset = 0
    namespace_buffer = next(iter(program.input_arrays), "")
    for name, array in program.input_arrays.items():
        # MLX can promote small inputs to Metal's constant address space.
        if name != namespace_buffer:
            namespace_loads.append(f"auto {name} = {namespace_buffer} + {namespace_offset}L;")
        namespace_offset += array.values.size
    arrival_cache = (_arrival_cache_source(scan_arrivals) if not component and not independent_neurons and not zero_delay
                     and "synapses" in phases else "")
    reset_indices=('const long _numpy_reset_rank=long(reset_ranks[gid])-1L;\n'
                   'const long _numpy_reset_count=long(reset_counts[trial]);' if collective else '')
    spike_word=("""
            if(_spike) fired |= 1u << (k & 31);
            if((k & 31)==31 || k==length-1) {
                spike_bits[(k >> 5)*NEURONS*B+gid]=fired;
                fired=0;
            }
        """ if independent_neurons and not compact else "if(_spike) fired |= 1u << k;")
    spike_tail=("for(int word=(length+31)/32; word<(W+31)/32; ++word) spike_bits[word*NEURONS*B+gid]=0u;"
                if independent_neurons and not compact else f"spike_bits[{output_index}] = fired;")
    if not component:
        blocks["thresholds"] += "\n"+spike_word
    if operation_errors or precision_errors:
        epoch_step='ulong(step-params[4])' if precision_errors else 'ulong(step)'
        if operation_errors and not precision_errors and error_epoch_origin is not None:
            origin=f'({error_epoch_origin+1}L-1L)'
            epoch_step=f'ulong(step-{origin})'
        step_code = "\n".join(
            "{\n" + f"bool _had_model_error = {program.index_error_name};\n" + blocks[phase] +
            f"\nif (!_had_model_error && {program.index_error_name}) "
            f"{program.index_error_name}_epoch = ({epoch_step} << 2) | "
            f"{0 if phase=='subexpression' else model.execution_phases.index(phase)}ul;\n}}" for phase in phases)
    else:
        step_code = "\n".join("{\n"+blocks[phase]+"\n}" for phase in phases)
    previous = "previous_spikes[gid]" if _uses_previous_events(model,program) else "0u"
    spike_init = f"bool _spike = (({previous}) & 1u) != 0;"
    if finish: spike_init = "bool _spike = (source_current[gid] & 1u) != 0;"
    idle_branch = ""
    if fused_idle and not zero_delay and not finish:
        idle_branch = f"""
        if(FAST_REFRACTORY && v == {literal(model.v_0)} && g == 0ul
            && due_bits[gid] == 0u && channel_offsets[row] == channel_offsets[row+1]) {{
            not_refractory = start + length - 1 >= release_step;
            {stores}
            q_release[{output_index}] = release_step;
            spike_bits[{output_index}] = 0u;
            return;
        }}
        """
    poisson_body = ""
    if poisson:
        from .brian_poisson import _fused_body
        fused_release = release_update
        for name, alias in (aliases or {}).items():
            fused_release = re.sub(r'\b'+re.escape(alias)+r'\b', program.variables[name].code_name, fused_release)
        poisson_body = _fused_body(program, poisson[0], fused_release)
    shared_offset=''
    expression_source=(''if not expression_draws else
        f'const ulong {program.random_program.result}=_eon_expression_draws[ulong(k)*B*NEURONS+gid];')
    if expression_prefix:
        expression_source=f'''
        ulong {program.random_program.result}=0ul;
        if(row<_eon_expression_counts[trial]) {{
            {program.random_program.result}=row<_eon_expression_done[trial]
                ? _eon_expression_pool[ulong(trial)*EXPRESSION_CAPACITY+_eon_expression_cursor[trial]+row]
                : _eon_expression_tail[ulong(trial)*NEURONS+row-_eon_expression_done[trial]];
        }}
        '''
    if expression_prefix=='cython':
        expression_source=f'''
        ulong {program.random_program.result}=0ul;
        if(row<_eon_expression_counts[trial]) {{
            if(row<_eon_expression_remaining[trial])
                {program.random_program.result}=_eon_expression_buffer[ulong(trial)*20000+_eon_expression_index[trial]+row];
            else {{
                uint fresh=row-_eon_expression_remaining[trial];
                {program.random_program.result}=fresh<_eon_expression_done[trial]
                    ?_eon_expression_pool[ulong(trial)*EXPRESSION_CAPACITY+_eon_expression_cursor[trial]+fresh]
                    :_eon_expression_tail[ulong(trial)*EXPRESSION_PHYSICAL_CAPACITY+fresh-_eon_expression_done[trial]];
            }}
        }}
        '''
    if expression_prefix=='cython':
        for slot,name in enumerate(('cursor','done','counts','index','remaining')):
            expression_source=expression_source.replace(f'_eon_expression_{name}[trial]',f'_eon_expression_metadata[trial*5+{slot}]')
    if program.scalar_stages:
        from .shared_state import layout as shared_layout,phase_source
        slots=shared_layout(program)[2]
        shared_offset=f'uint _eon_shared_offset=(trial*(length+1)+k)*{slots};'
        before_start=state_code(phase_source(program,'subexpression',aliases=aliases))
        if operation_errors or precision_errors:
            error=program.index_error_name
            before_start=('{\n'+f'bool _had_shared_error={error};\n'+before_start
                +f'\nif(!_had_shared_error && {error}) {error}_epoch={epoch_step}<<2;\n}}')
        if not finish and program.replay_phase is None:
            step_code=before_start+'\n'+step_code
    pending_header=pending_source=pending_marker=''
    if expression_reset is not None:
        stage=expression_reset.random_program.compiled_stages['reset']
        reset_vector=stage.vector
        if expression_reset.input_arrays:
            from .expression_single_call import _namespace_arithmetic
            reset_vector=_namespace_arithmetic("{"+reset_vector+"}")[1:-1]
        from .expression_rng_math import NORMAL_MATH_HEADER,POISSON_EXP_HEADER
        pending_header=NORMAL_MATH_HEADER+POISSON_EXP_HEADER+"""
        struct SingleCallRng {ulong value;uint calls,error;};
        inline ulong compiled_rand(thread SingleCallRng& rng){++rng.calls;return rng.value;}
        inline ulong compiled_normal(thread SingleCallRng& rng){++rng.calls;return rng.value;}
        """
        pending_source=f"""
        {{
            const ulong t={"clock_times[start-1-event_begin]"if expression_full_clock else"ulong(params[3])"};const long t_in_timesteps=start-1;
            uint encoded=_eon_pending_ranks[gid];
            SingleCallRng _compiled_rng;_compiled_rng.calls=0;_compiled_rng.error=0;
            _compiled_rng.value=_eon_pending_draws[ulong(trial)*NEURONS+(encoded&0x7fffffffu)];
            bool _cond=false;
            {stage.scalar}
            if(encoded>>31){{{reset_vector}}}
        }}
        """
        pending_marker="""
        uint hit=0;
        for(int edge=incoming_offsets[row];edge<incoming_offsets[row+1];edge++)
            hit|=_eon_pending_ranks[trial*NEURONS+incoming_sources[edge]]>>31;
        _eon_pending_marker[gid]=hit<<PENDING_SHIFT;
        """
    if expression_due_fusion:
        pending_marker=pending_marker.replace('hit<<PENDING_SHIFT','((_eon_pending_due[gid] & (0xffffffffu << PENDING_CLEAR)) | (hit<<PENDING_SHIFT))')
    # With delay == W, source_high is the current history word. Pack its
    # update beside the marker to retain the Metal buffer binding limit.
    if expression_due_fusion=='history':
        pending_marker+='\n_eon_pending_marker[ulong(NEURONS)*B+gid]=(DUE_OFFSET?source_high[gid]:0u)|(fired<<DUE_OFFSET);'
    if expression_reset_blocks:
        pending_source=pending_source.replace('uint encoded=_eon_pending_ranks[gid];',
            'uint encoded=_eon_pending_ranks[gid];uint block=gid/256;'
            'uint before=block%(NEURONS/256)?_eon_pending_blocks[block-1]:0u;'
            'encoded=(encoded&0x80000000u)|((encoded&0x7fffffffu)+before);')
    source = f"""
        uint work_index = thread_position_in_grid.x;
        if(work_index >= NEURONS * B) return;
        {indexing}
        uint row = gid % NEURONS, trial = gid / NEURONS;
        const long N = NEURONS;
        const int i = int(row);
        {reset_indices}
        const ulong dt = {literal(model.dt_s)};
        long start = params[0];
        int length = {"1"if expression_clock else"int(params[1])"}, staged_steps = int(params[2]);
        long event_begin = params[{1 if expression_clock else 3}];
        {chr(10).join(namespace_loads)}
        {poisson_body}
        {loads}
        {pending_source}
        {error_load}
        long release_step = {release_load};
        {program.scalar_code}
        {error_kind_restore}
        {idle_branch}
        {arrival_cache}
        uint fired = {'source_current[gid]' if finish else '0u'};
        {spike_init}
        for(int k = 0; k < {'1' if not independent_neurons and model.arrival_delay_steps <= 1 else 'length'}; ++k) {{
            long step = start + k;
            const long t_in_timesteps = step;
            ulong t = clock_times[step - event_begin];
            {expression_source}
            {shared_offset}
            {step_code}
        }}
        {stores}
        q_release[{output_index}] = {release_store};
        {spike_tail}
        {pending_marker}
    """
    # Combined input events contain only this step, while the clock remains
    # the original staged array for both the current and pending reset times.
    if expression_full_clock:
        source=source.replace('events[(trial * staged_steps + step - event_begin) * C + channel]',
            'events[trial * C + channel]')
    if expression_due_fusion:
        source=source.replace('due_bits[gid]','(due_bits[gid] >> DUE_OFFSET)')
    if expression_selection:
        source+='''
        uint _eon_lane=thread_position_in_threadgroup.x;
        uint _eon_rank=simd_prefix_exclusive_sum(uint(_spike));
        uint _eon_count=simd_sum(uint(_spike));
        threadgroup uint _eon_warps[8];
        if(_eon_lane%32==0)_eon_warps[_eon_lane/32]=_eon_count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _eon_total=0;
        for(uint warp=0;warp<8;++warp){
            if(warp<_eon_lane/32)_eon_rank+=_eon_warps[warp];
            _eon_total+=_eon_warps[warp];
        }
        _eon_expression_ranks[gid]=_eon_rank;
        if(_eon_lane==0)_eon_expression_blocks[gid/256]=_eon_total;
        '''
    if expression_rank_flags:
        source=source.replace('_eon_expression_ranks[gid]=_eon_rank;',
            '_eon_expression_ranks[gid]=_eon_rank|(uint(_spike)<<31);')
    if component:
        assert not (zero_delay or finish or compact or poisson or collective or program.scalar_stages)
        from .components import kernel_source, _prefix
        prefix = _prefix(program)
        source = kernel_source(program, loads, stores, "\n".join(namespace_loads),
                               program.scalar_code, error_load, release_load,
                               error_kind_restore, release_store, spike_init, step_code)
        source = source.replace("DT_BITS", literal(model.dt_s))
        inputs = [prefix+"rows" if name=="source_low" else prefix+"positions" if name=="source_high"
                  else prefix+"history_in" if name=="due_bits" else name for name in inputs]
        outputs[-1] = prefix+"history_out"
    for query, call in enumerate(program.shared_timed_calls):
        source=source.replace(call,f"{program.timed_cache_name}[{query}L * staged_steps + (step-event_begin)]")
    release_predicate="return d_ge(d_sub(d_mul(dt, d_from_int(candidate)), spike), threshold);"
    # Do not evaluate out-of-range constants for an unused release helper.
    cached_release = (program.cpu_refractory_guard
                      or program.refractory_elapsed_threshold is not None)
    if cached_release and program.refractory_policy in ("cython_fast", "cpp_fast"):
        offset = float(np.float64(.001)*np.float64(model.dt_s))
        reciprocal = float(np.float64(1.)/np.float64(model.dt_s))
        rfc_steps = int((np.float64(model.t_rfc)+np.float64(offset))*np.float64(reciprocal))
        release_predicate = (f"return d_to_int(d_mul(d_sub(d_add(d_mul(dt, d_from_int(candidate)), "
                    f"{literal(offset)}), spike), {literal(reciprocal)})) >= {rfc_steps}L;")
    if (model.state_dtype=='float32' and model.float32_codegen!='numpy'
            and program.refractory_elapsed_threshold is not None):
        release_predicate=f"""
            const ulong t=d_mul(dt,d_from_int(candidate));
            const {program.variables['lastspike'].metal_type} lastspike={"f_from_d(spike)" if program.variables['lastspike'].dtype == 'float32' else "spike"};
            const uint rfc={_literal(np.float32(model.t_rfc),'float32')};
            bool not_refractory=false;
            {program.refractory_code}
            return not_refractory;
        """
    return mx.fast.metal_kernel(
        name="fly_configured_" + ("components" if component else "finish" if finish else "neurons") + ("_compact" if compact else "_fused_poisson" if poisson else "_fused_idle" if fused_idle else ""),
        input_names=inputs, output_names=outputs, source=source,
        header=MSL_HEADER + SINGLE_HEADER + MSL_MATH_HEADER + arithmetic_header + pending_header + program.header_code +
        "inline bool fly_released(long candidate, ulong spike, ulong dt, ulong threshold) {"+release_predicate+"}\n"+"""
        inline long fly_release_step(long now, ulong spike, ulong dt, ulong threshold, long distance) {
            const long maximum = 9223372036854775807L;
            long low = now;
            long high = distance > maximum-now ? maximum : now+distance;
            while(high < maximum && !fly_released(high, spike, dt, threshold)) {
                long gap = high-now;
                high = gap > (maximum-now)/2 ? maximum : now+gap*2;
            }
            while(low < high) {
                long middle = low + (high-low)/2;
                if(fly_released(middle, spike, dt, threshold)) high=middle;
                else low=middle+1;
            }
            return low;
        }
        """, compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=16)
def _idle_compaction_kernel(rest_bits):
    """Compact only a proved autonomous fixed point with no possible input."""
    return mx.fast.metal_kernel(
        name="fly_configured_active_neurons",
        input_names=["voltage", "synaptic", "due_masks", "channel_offsets"],
        output_names=["count", "ids", "positions"], atomic_outputs=True,
        source=f"""
        uint gid = thread_position_in_grid.x;
        if(gid >= N * B) return;
        uint row = gid % N;
        bool active = voltage[gid] != {rest_bits}ul || synaptic[gid] != 0ul
            || due_masks[gid] != 0u || channel_offsets[row] != channel_offsets[row+1];
        uint position = 0xffffffffu;
        if(active) {{
            position = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
            atomic_store_explicit(&ids[position], gid, memory_order_relaxed);
        }}
        atomic_store_explicit(&positions[gid], position, memory_order_relaxed);
        """, compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=16)
def _merge_idle_kernel(program):
    layout = _state_layout(program)
    names = [group.buffer_name for group in layout.groups]
    inputs = ["base_"+name for name in names] + ["base_release", "positions"]
    inputs += ["updated_"+name for name in names] + ["updated_release", "updated_bits", "params"]
    lines = []
    for group in layout.groups:
        name = group.buffer_name
        lines.append(f"for(ulong slot=0; slot<{len(group.names)}ul; ++slot) {{ "
                     f"q_{name}[slot*SIZE+gid] = active ? updated_{name}[slot*SIZE+position] "
                     f": base_{name}[slot*SIZE+gid]; }}")
    group, slot = layout.slots["not_refractory"]
    lines.append(f"if(!active) q_{layout.groups[group].buffer_name}[{slot}ul*SIZE+gid] "
                 "= params[0] >= base_release[gid];")
    return mx.fast.metal_kernel(
        name="fly_configured_merge_idle",
        input_names=inputs,
        output_names=["q_"+name for name in names]+["q_release", "spike_bits"],
        source="""
        uint gid = thread_position_in_grid.x;
        if(gid >= SIZE) return;
        uint position = positions[gid];
        bool active = position != 0xffffffffu;
        """ + "\n".join(lines) + """
        q_release[gid] = active ? updated_release[position] : base_release[gid];
        spike_bits[gid] = active ? updated_bits[position] : 0u;
        """, compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=1)
def _mark_kernel():
    return mx.fast.metal_kernel(
        name="fly_configured_mark_arrivals",
        input_names=["offsets", "destinations", "spikes"],
        output_names=["first", "second"], atomic_outputs=True,
        source="""
        uint gid = thread_position_in_grid.x;
        if(gid >= N * B) return;
        uint bits = spikes[gid];
        if(bits == 0) return;
        uint row = gid % N, trial = gid / N;
        for(int edge = offsets[row]; edge < offsets[row+1]; ++edge) {
            uint target = trial * N + destinations[edge];
            uint low = (bits << SHIFT) & (0xffffffffu >> (32 - W));
            uint high = SHIFT == 0 ? 0u : bits >> (W - SHIFT);
            if(low) atomic_fetch_or_explicit(&first[target], low, memory_order_relaxed);
            if(high) atomic_fetch_or_explicit(&second[target], high, memory_order_relaxed);
        }
        """, compile_options={"math_mode": "safe"},
    )


def _clock_times(begin, end, dt_s):
    # Brian converts each int64 clock step to binary64 before multiplying by
    # dt. Floating arange loses the increments once begin reaches 2**53.
    return (np.arange(begin, end, dtype=np.int64).astype(np.float64) * dt_s).view(np.uint64)


def advance(prepared: Prepared, state: State, events, steps, length=None,
            *, event_begin=0, event_steps=None, event_times=None, timed_calls=None,
            fused_idle=True, native_poisson=None, stop_before_synapses=False, precision_epoch_begin=None,
            _expression_draws=None,_expression_clock=None,_expression_selection=False,_expression_defer_markers=False,
            _expression_pending_reset=None,_expression_pending_shift=None,_expression_rank_flags=False,_expression_due_fusion=False,_expression_full_clock=False):
    p, model = prepared, prepared.model
    if p.program.native_precision_limbs and state.native_precision_limbs>p.program.native_precision_limbs:
        from .compiled_precision_frontier import rebind
        p=rebind(p,state.native_precision_limbs)
    identity=(model.signature,p.neurons)
    if state.program_identity and state.program_identity != identity:
        raise ValueError("Continuation state belongs to a different model or neuron count")
    w = p.window
    offset = state.step % w
    length = min(w-offset, steps-state.step) if length is None else length
    maximum=w-offset
    if (length>maximum and _expression_draws is None and not stop_before_synapses
            and precision_epoch_begin is None and _expression_clock is None
            and state.random is not None and state.random.buffers is not None):
        from .expression_window import cython_batch_eligible,cython_submission_limit
        if cython_batch_eligible(p):maximum=cython_submission_limit(p,native_poisson)
    if (not 1 <= length <= maximum
            or state.step + length > steps):
        raise ValueError("Invalid configured window boundary")
    if p.program.runtime_error_order == 'operation':
        from .runtime_errors import NUMPY_ERROR_STEP_LIMIT
        if state.step < 0 and state.error_epoch_origin is None:
            state=replace(state,error_epoch_origin=state.step)
        origin=state.error_epoch_origin or 0
        if not 0 <= state.step-origin or state.step + length-origin > NUMPY_ERROR_STEP_LIMIT:
            raise OverflowError('Native NumPy error clock exceeds its checked encoding range')
    event_steps = steps if event_steps is None else event_steps
    if state.step < event_begin or state.step + length > event_begin + event_steps:
        raise ValueError("Configured window lies outside the staged events")
    layout = _state_layout(p.program)
    packed = _pack_values(p.program, state.values)
    size = packed.size
    b = size // p.neurons
    window_index = state.step // w
    delay = model.arrival_delay_steps
    source_window, source_offset = divmod(state.step - delay, w)
    zeros = _empty_words(size)
    source_low = state.history.get(source_window, zeros)
    source_high = state.history.get(source_window + 1, zeros)
    due = state.due.get(window_index, zeros)
    if offset and not _expression_due_fusion:
        due = due >> offset
    if event_times is None:
        event_times = mx.array(_clock_times(event_begin, event_begin + event_steps, model.dt_s))
    if p.program.shared_timed_calls and timed_calls is None:
        with mx.stream(mx.gpu):
            timed_calls = _stage_timed_calls(p, event_times, event_begin)
    if p.program.random_program is not None and _expression_draws is None:
        from .expression_fixed import advance as random_advance,FixedRandomProgram
        if not isinstance(p.program.random_program,FixedRandomProgram):
            from .expression_general import advance as random_advance
            random_options=dict(clock_params=_expression_clock)
        else:random_options={}
        return random_advance(p,state,events,steps,length,event_begin=event_begin,event_steps=event_steps,
            event_times=event_times,timed_calls=timed_calls,fused_idle=fused_idle,native_poisson=native_poisson,
            stop_before_synapses=stop_before_synapses,precision_epoch_begin=precision_epoch_begin,**random_options)
    from .expression_rng_prefix import UniformPrefixView
    from .expression_cython import CythonUniformView
    expression_prefix=("cython"if isinstance(_expression_draws,CythonUniformView)else isinstance(_expression_draws,UniformPrefixView))
    expression_inputs=[]
    if expression_prefix:
        if length!=1 or _expression_draws.counts is None:
            raise ValueError('A uniform prefix view requires one step and explicit trial counts')
        expression_inputs=[_expression_draws.values,_expression_draws.tail,_expression_draws.cursor,
            _expression_draws.done,_expression_draws.counts]
        if expression_prefix=="cython":
            expression_inputs=[_expression_draws.values,_expression_draws.tail,_expression_draws.buffer,
                _expression_draws.metadata if _expression_draws.metadata is not None else
                mx.stack([_expression_draws.cursor,_expression_draws.done,_expression_draws.counts,
                    _expression_draws.index,_expression_draws.remaining],axis=1)]
    elif _expression_draws is not None:expression_inputs=[_expression_draws.reshape(-1)]
    parameters=[state.step,length,event_steps,event_begin]
    if p.program.native_precision_limbs:
        from .compiled_precision_frontier import FRONTIER_STEP_LIMIT
        origin=state.step if precision_epoch_begin is None else precision_epoch_begin
        if not 0<=state.step-origin<FRONTIER_STEP_LIMIT-length:
            raise ValueError('Native precision error epoch exceeds its relative frontier range')
        if len(model.execution_phases)>4:
            raise ValueError('Native precision error epoch needs at most four simulation phases')
        parameters.append(origin)
    direct_clock=(_expression_clock is not None and not p.program.scalar_stages
                  and not p.program.index_error_name)
    if _expression_clock is None:
        params = mx.array(parameters, dtype=mx.int64)
    elif direct_clock:
        if length!=1 or len(parameters)!=4:raise ValueError('Direct expression clock needs one ordinary model step')
        params=_expression_clock
    else:
        if len(parameters)!=4:raise ValueError('Expression clock replay needs the ordinary four-field protocol')
        params=mx.stack([_expression_clock[0],mx.array(length,mx.int64),
            _expression_clock[2],_expression_clock[1]])
    if _expression_pending_reset is not None:
        if not direct_clock or length!=1:raise ValueError('Pending reset requires an explicit single-step clock')
        if not _expression_full_clock:
            params=mx.concatenate([params,_expression_pending_reset.time.astype(mx.int64)])
    common = [p.incoming_offsets, p.incoming_sources, p.weights, p.channel_offsets,
              p.channels, p.amplitudes, events, event_times, params]
    if native_poisson is not None:
        assert b == 1 and model.delay_steps > 0
        common[4] = native_poisson.uniforms
        common[6] = native_poisson.metadata
    # Legacy elapsed comparisons have no integer release cache and always use
    # the compiled binary64 predicate. Modern precomputation assumes Brian's
    # timestep conversion stays in int64 range, including lastspike=-10000 s.
    max_elapsed_steps = (steps * model.dt_s + 1e4 + .001 * model.dt_s) / model.dt_s
    fast_refractory = (p.program.refractory_elapsed_threshold is not None
                       and state.step >= 0
                       and np.isfinite(max_elapsed_steps) and max_elapsed_steps < 2**63
                       and 1e4 >= p.program.refractory_elapsed_threshold)
    templates = [("NEURONS", p.neurons), ("B", b), ("C", len(p.targets)),
                 ("W", w), ("SOURCE_OFFSET", 0 if direct_clock and p.independent_neurons else source_offset),
                 ("FAST_REFRACTORY", bool(fast_refractory)),
                 ("FAST_CPU_REFRACTORY", bool(p.program.cpu_refractory_guard and state.step >= 0 and np.isfinite(max_elapsed_steps) and max_elapsed_steps < 2**62))]
    if _expression_due_fusion:
        pending_window=(state.step-1+delay)//w
        next_window,next_offset=divmod(state.step+length,w)
        clear=next_offset if pending_window==next_window else 0
        templates += [('DUE_OFFSET',offset),('PENDING_CLEAR',clear)]
    if expression_prefix:templates.append(('EXPRESSION_CAPACITY',_expression_draws.values.shape[-1]))
    if expression_prefix=='cython':templates.append(('EXPRESSION_PHYSICAL_CAPACITY',_expression_draws.physical_capacity))
    from .trigonometry32 import scalar_sine_threadgroup
    kwargs = dict(template=templates, grid=(size, 1, 1),
                  threadgroup=(scalar_sine_threadgroup(p.program, size), 1, 1),
                  output_shapes=[*layout.shapes(size), (size,), (size*((w+31)//32 if p.independent_neurons else 1),)],
                  output_dtypes=[_storage_dtype(group.dtype) for group in layout.groups]
                                + [mx.int64, mx.uint32])
    has_index_error = p.program.index_error_name is not None
    if _expression_selection:
        from .expression_lowering import speculative_rank_eligible
        checked_ranks=speculative_rank_eligible(p.program,fused_idle=fused_idle)
        if (length!=1 or (not p.independent_neurons and model.arrival_delay_steps<=0) or p.neurons%256 or p.compact_idle
                or scalar_sine_threadgroup(p.program,size)!=256
                or p.program.scalar_stages or (has_index_error and not checked_ranks) or native_poisson is not None
                or _expression_draws is None or _phase_parts(model,p.program,False,stop_before_synapses)[1]):
            raise ValueError('Expression ranks require a complete256-lane step without zero-delay splitting')
        kwargs['output_shapes'] += [(size,),(size//256,)]
        kwargs['output_dtypes'] += [mx.uint32,mx.uint32]
    if _expression_defer_markers and (length!=1 or _expression_draws is None or model.arrival_delay_steps<=0 or p.independent_neurons or p.scan_arrivals):
        raise ValueError('Deferred expression markers require a complete positive-delay graph step')
    if _expression_pending_reset is not None:
        templates.append(('PENDING_SHIFT',_expression_pending_shift))
        kwargs['output_shapes'].append((size*(2 if _expression_due_fusion=='history'else 1),));kwargs['output_dtypes'].append(mx.uint32)
    if native_poisson is not None:
        templates += [("CHANNELS", len(p.targets)), ("CAPACITY", native_poisson.capacity)]
        kwargs["output_shapes"] += [(1,), (1,)]
        kwargs["output_dtypes"] += [mx.int32, mx.int32]
    release = state.release if state.release is not None else mx.zeros((size,), mx.int64)
    previous_inputs = [state.spikes if state.spikes is not None else zeros] if _uses_previous_events(model,p.program) else []
    namespace_inputs = list(p.input_arrays)+([timed_calls] if p.program.shared_timed_calls else [])
    error_epoch_options=({'error_epoch_origin':state.error_epoch_origin}
        if state.error_epoch_origin is not None and not p.program.native_precision_limbs else {})
    with mx.stream(mx.gpu):
        trace=None
        if p.program.scalar_stages:
            from . import shared_state
            shared,_,slots=shared_state.layout(p.program)
            if b*(length+1)*slots>=2**31:raise OverflowError('Shared scalar trace exceeds safe MLX indexing')
            initial=(mx.stack([packed[v.name].astype(mx.uint64) for v in shared],axis=1).reshape(-1)
                     if shared else mx.zeros((1,),mx.uint64))
            shared_threads=b*length if p.program.scalar_parallel else b
            trace=shared_state.kernel(model,p.program,stop_before_synapses)(
                inputs=[initial,p.amplitudes,events,event_times,params,*namespace_inputs],
                template=[('NEURONS',p.neurons),('B',b),('C',len(p.targets))],
                grid=(shared_threads,1,1),threadgroup=(min(256,shared_threads),1,1),
                output_shapes=[(b*(length+1)*slots,)],output_dtypes=[mx.uint64])[0]
            namespace_inputs.append(trace)
        fuse_idle = p.compact_idle and fast_refractory and fused_idle and model.state_dtype == "float64"
        # Reuse the exact helpers for the demonstrated long batched path and
        # restored custom canonical state. Other short/default paths retain
        # their existing compiled-kernel identities.
        batched_lif_arithmetic = ((b >= 4 and steps >= 10000 and fuse_idle
            and model.reference_codegen_target == "cython"
            and p.program.refractory_policy == "cython_fast")
            or p.program.canonical_custom_lif)
        compact = p.compact_idle and fast_refractory and not fuse_idle and native_poisson is None
        active_inputs = []
        if compact:
            rest_bits = (int(np.float32(model.v_0).view(np.uint32)) if model.state_dtype == "float32"
                         else int(np.float64(model.v_0).view(np.uint64)))
            count, ids, positions = _idle_compaction_kernel(rest_bits)(
                inputs=[packed["v"], packed["g"], due, p.channel_offsets],
                template=[("N", p.neurons), ("B", b)], grid=(size, 1, 1),
                threadgroup=(min(256, size), 1, 1),
                output_shapes=[(1,), (size,), (size,)],
                output_dtypes=[mx.uint32]*3, init_value=0)
            active_inputs = [ids, count]
        zero_delay = delay == 0 and not p.independent_neurons
        collective = p.program.collective_reset
        rank_inputs = []
        if collective:
            reset_bits = previous_inputs[0] if _uses_previous_events(model,p.program) else zeros
            ranks = mx.cumsum((reset_bits != 0).reshape(b, p.neurons).astype(mx.int32), axis=1)
            rank_inputs = [ranks.reshape(-1), ranks[:, -1]]
        outputs = _neuron_kernel(model, p.program, zero_delay,
                                 compact=compact, fused_idle=fuse_idle, scan_arrivals=p.scan_arrivals,
                                 independent_neurons=p.independent_neurons, local_synapses=p.local_synapses,
                                 stop_before_synapses=stop_before_synapses,
                                 expression_draws=_expression_draws is not None,
                                 expression_clock=direct_clock,
                                 expression_prefix=expression_prefix,
                                 expression_selection=_expression_selection,
                                 expression_reset=None if _expression_pending_reset is None else _expression_pending_reset.program,
                                 expression_rank_flags=_expression_rank_flags,
                                 expression_due_fusion=_expression_due_fusion,
                                 expression_full_clock=_expression_full_clock,
                                 batched_lif_arithmetic=batched_lif_arithmetic,
                                 expression_reset_blocks=_expression_pending_reset is not None and _expression_pending_reset.blocks is not None,
                                 poisson=(native_poisson.unique_count, len(p.targets))
                                 if native_poisson is not None else None,**error_epoch_options)(
            inputs=[*packed.buffers, release, *previous_inputs, *common,
                    source_low, source_high, due, *rank_inputs, *namespace_inputs, *active_inputs,
                    *([native_poisson.used, native_poisson.overflow]
                      if native_poisson is not None else []),
                    *expression_inputs,*([]if _expression_pending_reset is None else
                        [_expression_pending_reset.draws,_expression_pending_reset.ranks]+
                        ([]if _expression_pending_reset.blocks is None else[_expression_pending_reset.blocks])+
                        ([state.due.get((state.step-1+delay)//w,zeros)]if _expression_due_fusion else[]))], **kwargs)
        pending_marker=None;expression_history=None
        if _expression_pending_reset is not None:
            pending_marker=outputs[-1];outputs=outputs[:-1]
            if _expression_due_fusion=='history':
                expression_history=pending_marker[size:];pending_marker=pending_marker[:size]
        selection=None
        if _expression_selection:
            selection=(outputs[-2],outputs[-1]);outputs=outputs[:-2]
        if native_poisson is not None:
            *outputs, native_poisson.used, native_poisson.overflow = outputs
        if compact:
            outputs = _merge_idle_kernel(p.program)(
                inputs=[*packed.buffers, release, positions,
                        *outputs, mx.array([state.step+length-1], dtype=mx.int64)],
                template=[("SIZE", size)], grid=(size, 1, 1),
                threadgroup=(min(256, size), 1, 1),
                output_shapes=kwargs["output_shapes"], output_dtypes=kwargs["output_dtypes"])
        bits = outputs[-1]
        if _phase_parts(model, p.program, zero_delay, stop_before_synapses)[1]:
            if collective and not _uses_previous_events(model,p.program):
                ranks = mx.cumsum((bits != 0).reshape(b, p.neurons).astype(mx.int32), axis=1)
                rank_inputs = [ranks.reshape(-1), ranks[:, -1]]
            sources = [] if zero_delay else [source_low, source_high, due]
            outputs = _neuron_kernel(model, p.program, zero_delay, True,
                                     independent_neurons=p.independent_neurons, local_synapses=p.local_synapses, scan_arrivals=p.scan_arrivals,
                                     stop_before_synapses=stop_before_synapses,
                                     expression_draws=_expression_draws is not None,expression_clock=direct_clock,
                                     expression_prefix=expression_prefix,**error_epoch_options)(
                inputs=[*outputs[:-1], *previous_inputs, *common, bits, *sources,
                        *rank_inputs, *namespace_inputs,
                        *expression_inputs], **kwargs)
        shared_values=({v.name:trace.reshape(b,length+1,slots)[:,-1,i].astype(_dtype(v)) for i,v in enumerate(shared)} if trace is not None else None)
        values = _PackedValues(layout, outputs[:-2], size, shared_values)
        new_step = state.step + length
        oldest = (new_step - delay) // w
        history = {q: bits_ for q, bits_ in state.history.items() if q >= oldest}
        next_window, next_offset = divmod(new_step, w)
        due_windows = {q: mask for q, mask in state.due.items() if q >= next_window}
        if next_offset and next_window in due_windows:
            remaining = mx.array((0xffffffff << next_offset) & 0xffffffff, dtype=mx.uint32)
            due_windows[next_window] = due_windows[next_window] & remaining
        if delay > 0 and not p.independent_neurons:
            if expression_history is not None:history[window_index]=expression_history
            else:
                word = bits << offset if offset else bits
                history[window_index] = (state.history[window_index] | word
                                         if offset and window_index in state.history else word)
            if not p.scan_arrivals and not _expression_defer_markers:
                distance, shift = divmod(delay+offset, w)
                markers = _mark_kernel()(
                    inputs=[p.outgoing_offsets, p.outgoing_destinations, bits],
                    template=[("N", p.neurons), ("B", b), ("W", w),
                              ("SHIFT", shift)],
                    grid=(size, 1, 1), threadgroup=(min(256, size), 1, 1),
                    output_shapes=[(size,), (size,)], output_dtypes=[mx.uint32, mx.uint32],
                    init_value=0)
                for q, mask in zip((window_index + distance, window_index + distance + 1), markers):
                    if q == window_index + distance + 1 and shift == 0:
                        continue
                    due_windows[q] = due_windows[q] | mask if q in due_windows else mask
    previous = None
    if _uses_previous_events(model,p.program):
        word = (length-1)//32 if p.independent_neurons else 0
        previous = (bits[word*size:(word+1)*size] >> ((length-1)%32)) & 1 if length != 1 else bits
    result=State(values, history, due_windows, new_step, outputs[-2], has_index_error,
                 previous, p.program.runtime_error_type, p.program.runtime_errors,
                 p.program.runtime_error_order, identity,p.program.native_precision_limbs,
                 error_epoch_origin=state.error_epoch_origin)
    returned=(result,bits,selection)if _expression_selection else(result,bits)
    return (*returned,pending_marker)if _expression_pending_reset is not None else returned


def evaluate_state(state):
    from .expression_runtime import arrays as random_arrays
    mx.eval(*_state_buffers(state.values), *state.history.values(), *state.due.values(), state.release,
            *(() if state.spikes is None else (state.spikes,)),*random_arrays(state.random))
    _check_index_error(state)
    if state.random is not None:
        errors=np.asarray(state.random.errors)
        if np.any(errors&2):raise ZeroDivisionError('float division')
        if np.any(errors&1):
            from .expression_errors import ExpressionRuntimeError
            raise ExpressionRuntimeError(int(errors[np.flatnonzero(errors&1)[0]]))


def _check_index_error(state):
    if not state.check_index_error:
        return
    # The caller requires a synchronous error decision either way. Small
    # release buffers are cheaper to inspect directly at this host boundary;
    # retain the GPU reduction for larger populations.
    failed = (bool(np.any(np.asarray(state.release) < 0)) if state.release.size <= 65536
              else bool(mx.any(state.release < 0).item()))
    if failed:
        if state.runtime_error_type == 'model':
            from .runtime_errors import first_error, model_exception
            _, kind = first_error(state.release, state.runtime_error_order)
            raise model_exception(state.runtime_errors, kind)
        if state.runtime_error_type=="zero_division":
            kinds=np.asarray(state.release)
            kind=int(kinds[np.flatnonzero(kinds<0)[0]]) & 3
            raise ZeroDivisionError("float division" if kind==2 else "float divmod()")
        raise IndexError("TimedArray column index is out of bounds")


def host_state(state, prepared):
    """Export persistent SI state while preserving every typed bit pattern."""
    _check_index_error(state)
    result = {}
    # Export each physical plane once. Named MLX slices otherwise require a
    # separate lazy view evaluation for every persistent variable.
    packed = state.values if isinstance(state.values, _PackedValues) else None
    buffers = tuple(np.asarray(buffer) for buffer in packed.buffers) if packed else ()
    for variable in _persistent(prepared.program):
        if packed is not None and variable.name not in packed.shared:
            group, slot = packed.layout.slots[variable.name]
            array = buffers[group][slot*packed.size:(slot+1)*packed.size]
            storage = packed.layout.dtypes[variable.name]
            dtype = np.dtype({'float64': 'uint64', 'float32': 'uint32'}.get(storage, storage))
            if array.dtype != dtype:
                array = array.view(dtype)
        else:
            array = np.asarray(state.values[variable.name])
        if variable.dtype in ("float64", "float32"):
            array = array.view(np.float64 if variable.dtype == "float64" else np.float32)
        elif variable.dtype=='bool' and variable.storage_dtype=='uint8':
            array=array.view(np.bool_)
        result[variable.name] = (np.broadcast_to(array.reshape(-1,1),(array.size,prepared.neurons)) if variable.scalar else array.reshape(-1,prepared.neurons))
    return result


@lru_cache(maxsize=1)
def _unit_pack_kernel():
    return mx.fast.metal_kernel(
        name="fly_configured_pack_units",
        input_names=["flags"], output_names=["words", "counts"],
        source="""
            uint gid = thread_position_in_grid.x;
            constexpr uint ITEMS = uint(N) * uint(RECORD_WINDOWS) * uint(RECORD_BITS);
            constexpr uint WORDS = ITEMS / 32u + uint(ITEMS % 32u != 0u);
            if(gid >= WORDS * uint(B)) return;
            uint trial = gid / WORDS, begin = (gid % WORDS) * 32u;
            uint mask = 0u;
            for(uint bit=0u; bit<min(32u, ITEMS-begin); ++bit) {
                uint offset=begin+bit, step=offset/uint(N), row=offset%uint(N);
                uint word=flags[(trial*uint(RECORD_WINDOWS)+step/uint(RECORD_BITS))*uint(N)+row];
                mask |= ((word >> (step%uint(RECORD_BITS))) & 1u) << bit;
            }
            words[gid] = mask;
            counts[gid] = popcount(mask);
        """, compile_options={"math_mode": "safe"})


@lru_cache(maxsize=1)
def _unit_record_kernel():
    return mx.fast.metal_kernel(
        name="fly_configured_record_units",
        input_names=["words", "prefix", "starts"],
        output_names=["q_neurons", "q_steps"],
        source="""
            uint gid = thread_position_in_grid.x;
            constexpr uint ITEMS = uint(N) * uint(RECORD_WINDOWS) * uint(RECORD_BITS);
            constexpr uint WORDS = ITEMS / 32u + uint(ITEMS % 32u != 0u);
            if(gid >= WORDS * uint(B)) return;
            uint mask = words[gid];
            uint destination = prefix[gid] - popcount(mask);
            while(mask != 0u) {
                uint offset = (gid % WORDS) * 32u + ctz(mask);
                q_neurons[destination] = long(offset % uint(N));
                q_steps[destination] = starts[offset / uint(N)];
                ++destination;
                mask &= mask - 1u;
            }
        """, compile_options={"math_mode": "safe"})



@lru_cache(maxsize=1)
def _unit_parallel_record_kernel():
    return mx.fast.metal_kernel(
        name="fly_configured_record_parallel_units",
        input_names=["words", "prefix", "starts"],
        output_names=["q_neurons", "q_steps"],
        source="""
            uint gid = thread_position_in_grid.x;
            constexpr uint ITEMS = uint(N) * uint(RECORD_WINDOWS) * uint(RECORD_BITS);
            constexpr uint WORDS = ITEMS / 32u + uint(ITEMS % 32u != 0u);
            if(gid >= ITEMS * uint(B)) return;
            uint trial = gid / ITEMS, offset = gid % ITEMS;
            uint word_index = trial * WORDS + offset / 32u, bit = offset % 32u;
            uint mask = words[word_index];
            if((mask & (1u << bit)) == 0u) return;
            uint below = mask & ((1u << bit) - 1u);
            uint destination = prefix[word_index] - popcount(mask) + popcount(below);
            q_neurons[destination] = long(offset % uint(N));
            q_steps[destination] = starts[offset / uint(N)];
        """, compile_options={"math_mode": "safe"})


@lru_cache(maxsize=64)
def _warm_unit_recording(neurons, trials, windows):
    flags = mx.zeros((neurons*trials*windows,), mx.uint32)
    words, prefix = _unit_record_prefix(flags, neurons, trials, windows)
    outputs = _unit_record_kernel()(
        inputs=[words, prefix, mx.arange(windows, dtype=mx.int64)],
        template=[("N", neurons), ("RECORD_WINDOWS", windows), ("RECORD_BITS", 1), ("B", trials)],
        grid=(1, 1, 1), threadgroup=(1, 1, 1),
        output_shapes=[(1,)]*2, output_dtypes=[mx.int64]*2, init_value=0)
    mx.eval(prefix, *outputs)


def _unit_record_prefix(flags, neurons, trials, windows, word_bits=1):
    size = trials * ((neurons * windows * word_bits + 31) // 32)
    words, counts = _unit_pack_kernel()(
        inputs=[flags], template=[("N", neurons), ("RECORD_WINDOWS", windows), ("RECORD_BITS", word_bits), ("B", trials)],
        grid=(size, 1, 1), threadgroup=(min(256, size), 1, 1),
        output_shapes=[(size,)]*2, output_dtypes=[mx.uint32]*2)
    return words, mx.cumsum(counts)


def _collect_small_unit_windows(block, starts, neurons, trial_ids, word_bits):
    """Decode a bounded compressed recording during host export.

    No model work occurs here: each bit is an already computed native spike.
    Small blocks avoid the scan/count/readback/expand/readback GPU round trips.
    Larger blocks retain GPU population scans and parallel expansion.
    """
    trials = len(trial_ids)
    arrays = [np.asarray(word).reshape(trials, neurons) for word in block]
    parts = []
    for trial in np.argsort(trial_ids):
        # For uniform native masks, fill final output storage directly.
        # Avoid allocating and concatenating a separate array for each bit.
        if all(np.all(array[trial] == array[trial, 0]) for array in arrays):
            times = [start+bit for array, start in zip(arrays, starts)
                     for bit in range(word_bits) if int(array[trial, 0]) & (1 << bit)]
            if times:
                selected = np.empty((len(times), neurons), np.int64)
                selected[:] = np.arange(neurons, dtype=np.int64)
                steps = np.empty_like(selected)
                steps[:] = np.asarray(times, np.int64)[:, None]
                parts.append((np.full(selected.size, trial_ids[trial], np.int64),
                              selected.reshape(-1), steps.reshape(-1)))
            continue
        neuron_parts, step_parts = [], []
        for array, start in zip(arrays, starts):
            words = array[trial]
            used = int(np.bitwise_or.reduce(words, initial=np.uint32(0))) & ((1 << word_bits)-1)
            if not used:
                continue
            # Uniform masks are common in synchronized populations. This is
            # checked from actual recorded bits, never inferred from a model.
            uniform = bool(np.all(words == words[0]))
            while used:
                bit = (used & -used).bit_length()-1
                selected = (np.arange(neurons, dtype=np.int64) if uniform else
                            np.flatnonzero(words & np.uint32(1 << bit)))
                neuron_parts.append(selected)
                step_parts.append(np.full(selected.size, start+bit, np.int64))
                used &= used-1
        if neuron_parts:
            selected = np.concatenate(neuron_parts)
            parts.append((np.full(selected.size, trial_ids[trial], np.int64),
                          selected, np.concatenate(step_parts)))
    return parts


@lru_cache(maxsize=1)
def _sparse_record_words_kernel():
    return mx.fast.metal_kernel(
        name="fly_configured_sparse_record_words",
        input_names=["flags"], output_names=["count", "overflow", "indices", "values"],
        atomic_outputs=True, source="""
        uint gid = thread_position_in_grid.x;
        if(gid >= uint(SIZE)) return;
        uint mask = flags[gid] & (0xffffffffu >> (32-uint(BITS)));
        if(!mask) return;
        if(atomic_load_explicit(count, memory_order_relaxed) >= uint(CAPACITY)) {
            atomic_store_explicit(overflow, 1u, memory_order_relaxed);
            return;
        }
        uint position = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
        if(position < uint(CAPACITY)) {
            atomic_store_explicit(&indices[position], gid, memory_order_relaxed);
            atomic_store_explicit(&values[position], mask, memory_order_relaxed);
        } else atomic_store_explicit(overflow, 1u, memory_order_relaxed);
        """, compile_options={"math_mode": "safe"})


def _collect_sparse_unit_windows(block, starts, neurons, trial_ids, word_bits,
                                 capacity=65536):
    """Export sparse packed words; an overflow retains the existing recorder."""
    trials, windows = len(trial_ids), len(block)
    size = trials * windows * neurons
    # MLX's integer template ABI is signed int32, even though Metal indices
    # are uint32. Larger legal blocks keep the existing split/prefix path.
    if size >= 2**31:
        return None
    flags = mx.stack([word.reshape(trials, neurons) for word in block], axis=1).reshape(-1)
    count, overflow, indices, values = _sparse_record_words_kernel()(
        inputs=[flags], template=[("SIZE", size), ("BITS", word_bits), ("CAPACITY", capacity)],
        grid=(size, 1, 1), threadgroup=(min(256, size), 1, 1),
        output_shapes=[(1,), (1,), (capacity,), (capacity,)],
        output_dtypes=[mx.uint32]*4, init_value=0)
    mx.eval(count, overflow, indices, values)
    if int(overflow.item()):
        return None
    active = int(count.item())
    if not active:
        return []
    locations = np.asarray(indices[:active]).astype(np.int64, copy=False)
    masks = np.asarray(values[:active])
    times = np.asarray(starts, np.int64)
    bits = np.arange(word_bits, dtype=np.uint32)
    parts = []
    # Expansion has a fixed tile bound even when every retained word is full.
    for begin in range(0, active, 16384):
        end = min(begin+16384, active)
        selected, offsets = np.nonzero((masks[begin:end, None] >> bits[None, :]) & np.uint32(1))
        location = locations[begin:end][selected]
        parts.append((trial_ids[location // (windows*neurons)], location % neurons,
                      times[(location // neurons) % windows] + offsets))
    arrays = [np.concatenate([part[i] for part in parts]).astype(np.int64, copy=False)
              for i in range(3)]
    order = np.lexsort((arrays[1], arrays[2], arrays[0]))
    arrays = [array[order] for array in arrays]
    # Keep one part per trial so the caller's stable block merge remains valid.
    ends = np.r_[np.flatnonzero(arrays[0][1:] != arrays[0][:-1])+1, len(arrays[0])]
    parts, begin = [], 0
    for end in ends:
        parts.append(tuple(array[begin:end] for array in arrays))
        begin = end
    return parts


def _collect_unit_windows(block, starts, neurons, trial_ids, word_bits=1, *, sparse=False):
    """Export bit windows in trial/step/neuron order within bounded blocks.

    Pack each trial's flags into 32-bit words before scanning their population
    counts. Stable word/bit order assigns chronological output indices. Work
    and memory are bounded by the block. The optional sparse path sorts only
    emitted events and falls back here when its fixed word capacity is full.
    """
    trials = len(trial_ids)
    # Prefix counters and kernel offsets are uint32. Split the compressed
    # block before expanding its possible events, preserving chronological
    # parts for the final stable trial grouping.
    limit = (2**32-1) // (neurons*trials*word_bits)
    if limit < 1:
        raise OverflowError('Configured recording block exceeds uint32 event indexing')
    if len(block) > limit:
        return [part for begin in range(0,len(block),limit)
                for part in _collect_unit_windows(block[begin:begin+limit],starts[begin:begin+limit],
                                                  neurons,trial_ids,word_bits,sparse=sparse)]
    if neurons*trials*len(block) <= 65536:
        return _collect_small_unit_windows(block, starts, neurons, trial_ids, word_bits)
    if sparse:
        parts = _collect_sparse_unit_windows(block, starts, neurons, trial_ids, word_bits)
        if parts is not None:
            return parts
    order = np.argsort(trial_ids)
    flags = mx.stack([word.reshape(trials, neurons) for word in block], axis=1)
    if not np.array_equal(order, np.arange(trials)):
        flags = mx.take(flags, mx.array(order.astype(np.int32)), axis=0)
    flags = flags.reshape(-1)
    words, prefix = _unit_record_prefix(flags, neurons, trials, len(block), word_bits)
    ends = np.asarray(prefix.reshape(trials, -1)[:, -1])
    count = int(ends[-1])
    if not count:
        return []
    times=mx.array(np.asarray(starts,np.int64)[:,None]+np.arange(word_bits),mx.int64).reshape(-1)
    # At most one event per packed word on average: keep word-based sparse
    # decoding. Dense outputs use coalesced per-event writes. The existing
    # prefix total selects the path without another count or synchronization.
    possible=neurons*trials*len(block)*word_bits
    parallel=count*32 >= possible
    grid=possible if parallel else words.size
    arrays = (_unit_parallel_record_kernel() if parallel else _unit_record_kernel())(
        inputs=[words,prefix,times],
        template=[("N", neurons), ("RECORD_WINDOWS", len(block)), ("RECORD_BITS", word_bits), ("B", trials)],
        grid=(grid, 1, 1), threadgroup=(min(256,grid), 1, 1),
        output_shapes=[(count,)]*2, output_dtypes=[mx.int64]*2)
    mx.eval(*arrays)
    arrays = [np.asarray(array) for array in arrays]
    parts = []
    begin = 0
    for trial_id,end in zip(trial_ids[order],ends):
        end = int(end)
        if end > begin:
            # Trial boundaries are already known from the prefix scan. Avoid
            # writing and transferring an identical GPU word for every spike.
            ids = (np.zeros(end-begin,np.int64) if trial_id == 0 else
                   np.full(end-begin,trial_id,np.int64))
            parts.append((ids,*(array[begin:end] for array in arrays)))
        begin = end
    return parts


def _stage_inputs(chunk, token, begin, end, dt_s):
    """Reuse read-only Metal inputs only for an exact immutable snapshot.

    The caller still validates every consumed chunk. Mutable owners take a
    fresh snapshot before this lookup; exact equality is required before the
    snapshot helper returns a previous bytes object. Keep that object alive
    here to prevent recycled identities. Streams and unpinned arrays bypass
    this cache. The byte limit counts both retained host bytes and GPU inputs.
    """
    snapshot = None if token is None else token[2]
    key = None if token is None else (
        id(snapshot), token[0][2], begin, end, np.float64(dt_s).tobytes())
    previous = _STAGED_INPUTS.get(key)
    if previous is not None and previous[0] is snapshot:
        _STAGED_INPUTS.move_to_end(key)
        return previous[1:3]
    with mx.stream(mx.gpu):
        events = mx.array(chunk.reshape(-1) if chunk.size else np.zeros(1, np.uint8))
        clock_times = mx.array(_clock_times(begin, end, dt_s))
        mx.eval(events, clock_times)
    if token is not None:
        cost = len(snapshot) + events.nbytes + clock_times.nbytes
        if cost <= _MAX_STAGED_INPUT_BYTES:
            _STAGED_INPUTS[key] = (snapshot, events, clock_times, cost)
            while (len(_STAGED_INPUTS) > 4 or
                   sum(item[3] for item in _STAGED_INPUTS.values()) > _MAX_STAGED_INPUT_BYTES):
                _STAGED_INPUTS.popitem(last=False)
    return events, clock_times


def run_prepared(prepared, stimulus, *, record=True, block_windows=64,
                 chunk_steps=18432, spike_sink=None, return_state=False,random_states=None):
    if spike_sink is not None and not record:
        raise ValueError("A spike sink requires recording")
    p = prepared
    raw = stimulus.events
    shape = tuple(index(d) for d in raw.shape)
    ids = tuple(index(t) for t in stimulus.trial_indices)
    if (raw.dtype != np.uint8 or raw.ndim != 3 or len(shape) != 3
            or shape[0] < 1 or shape[1] < 0 or shape[2] != len(p.targets)
            or shape[0] != len(ids) or len(set(ids)) != len(ids)
            or tuple(getattr(raw, "trial_indices", ids)) != ids
            or any(t < 0 or t > np.iinfo(np.int64).max for t in ids)
            or tuple(stimulus.targets) != p.targets
            or stimulus.dt_ms != p.model.dt_ms
            or getattr(raw, "dt_ms", stimulus.dt_ms) != stimulus.dt_ms
            or _amplitude_values(stimulus, p.model).tobytes() != p.amplitude_bits):
        raise ValueError("Invalid canonical stimulus for the resolved model")
    trials, steps, channels = shape
    if steps >= 2**63:
        raise OverflowError("Configured horizon exceeds the signed 64-bit clock")
    if isinstance(chunk_steps, bool) or isinstance(block_windows, bool):
        raise ValueError("Chunk/recording sizes must be positive integers")
    chunk_steps, block_windows = index(chunk_steps), index(block_windows)
    if chunk_steps < 1 or block_windows < 1:
        raise ValueError("Chunk/recording sizes must be positive integers")
    if p.independent_neurons and not p.compact_idle and not p.program.collective_reset:
        # No edges impose a cross-neuron synchronization boundary. Retain
        # every spike word while integrating more steps in one dispatch.
        target_window=max(32,min(1024,chunk_steps//32*32))
        if p.program.random_program is not None:
            from .expression_fixed import max_frames
            target_window=min(target_window,max_frames(p,trials))
        if p.window!=target_window:p=replace(p,window=target_window)
    size = trials * p.neurons
    spike_words = (p.window+31)//32 if p.independent_neurons else 1
    window_bytes = size*(24 + 4*spike_words
                         + sum(np.dtype(group.dtype).itemsize*len(group.names)
                               for group in _state_layout(p.program).groups))
    submission_windows = max(1,min(16,(64*1024*1024)//max(1,window_bytes)))
    component_steps = (min(1024,(64*1024*1024)//size)
                       if not record and p.scan_arrivals and p.components is not None and p.program.random_program is None else 0)
    if component_steps < min(p.window,max(1,steps)):
        component_steps = 0
    if size * max(1, block_windows, (p.window+31)//32) >= 2**32:
        raise OverflowError("Configured recording index exceeds uint32")
    sparse_recording = trials > 1 and steps > p.window * block_windows
    # Every internal chunk boundary is a complete delay window.  The final
    # horizon may be partial; arbitrary input-source slices remain supported.
    maximum_chunk = (2**31 - 1) // max(1, trials * channels)
    if maximum_chunk < p.window:
        raise OverflowError("Configured event window exceeds safe Metal indexing")
    chunk_steps = max(p.window, chunk_steps // p.window * p.window)
    chunk_steps = min(chunk_steps, maximum_chunk)
    chunk_steps = max(p.window, chunk_steps // p.window * p.window)
    mx.reset_peak_memory()
    started = perf_counter()
    if p.program.random_program is not None and random_states is None:
        from .expression_runtime import trial_snapshots
        random_states=trial_snapshots(stimulus)
    state = initial_state(p, trials,random_states=random_states)
    setup_s = perf_counter() - started
    jit_s = 0.
    if steps:
        staged = min(chunk_steps, steps)
        warmup_key = (ref(p.program), p.neurons, trials, p.window, steps, staged,
                      bool(record), id(prepared), chunk_steps, block_windows, component_steps)
        previous_warmup = _WARMUPS.get(warmup_key)
    if steps and (previous_warmup is None or previous_warmup() is not prepared):
        compiling = perf_counter()
        events = mx.zeros((max(1, trials * staged * channels),), mx.uint8)
        bits = None
        probe_length=min(p.window,steps)
        if p.program.random_program is not None:
            from .expression_fixed import max_frames
            probe_length=min(probe_length,max_frames(p,trials))
        if component_steps:
            from .components import advance_components
            probe = advance_components(p,state,events,steps,min(steps,staged,component_steps),event_steps=staged)
        elif p.program.native_precision_limbs:
            from .compiled_precision_frontier import PrecisionFrontier
            frontier=PrecisionFrontier(p,state)
            operation=lambda pp,ss,origin:advance(pp,ss,events,steps,min(pp.window,steps),
                event_steps=staged,precision_epoch_begin=origin)
            frontier.submit(operation,min(p.window,steps))
            attempt=frontier.evaluate();p,probe=attempt.prepared,attempt.state
            bits=attempt.windows[-1][2]
        else:
            probe, bits = advance(p, state, events, steps, probe_length, event_steps=staged)
        try:
            evaluate_state(probe)
        except Exception as error:
            # Synthetic inputs compile a disposable probe, not a simulation.
            # Actual execution below remains authoritative for model errors.
            from .runtime_errors import is_model_exception
            if not is_model_exception(error):
                raise
        if bits is not None:
            mx.eval(bits)
        if record:
            if p.window == 1:
                _warm_unit_recording(p.neurons, trials, min(block_windows, steps))
            else:
                _collect_unit_windows([bits[:size]], [0], p.neurons,
                                      np.asarray(ids,np.int64), 32, sparse=sparse_recording)
        jit_s = perf_counter() - compiling
        del probe, bits, events
        _WARMUPS[warmup_key] = ref(prepared)
        while len(_WARMUPS) > 32:
            _WARMUPS.popitem(last=False)
    elif steps:
        _WARMUPS.move_to_end(warmup_key)
    parts, block, starts = [], [], []
    frontier=None
    if p.program.native_precision_limbs:
        from .compiled_precision_frontier import PrecisionFrontier
        frontier=PrecisionFrontier(p,state)

    def accept_frontier():
        nonlocal p,state,block,starts
        attempt=frontier.evaluate();p,state=attempt.prepared,attempt.state
        if attempt.error_kind is not None:
            from .runtime_errors import model_exception
            raise model_exception(p.program.runtime_errors,attempt.error_kind)
        if record:
            block=[];starts=[]
            for begin,length,bits in attempt.windows:
                for word in range((length+31)//32 if p.independent_neurons else 1):
                    block.append(bits[word*size:(word+1)*size] if p.independent_neurons else bits)
                    starts.append(begin+word*32)

    started = perf_counter()
    raw, known_hashes, input_token = event_input_snapshot(raw)
    snapshot_s = perf_counter() - started
    hashes = [sha256() for _ in ids] if known_hashes is None else []
    trial_ids = np.asarray(ids, np.int64)
    collection_s = spool_io_s = 0.
    staging_s = snapshot_s
    component_passes = 0
    for event_begin in range(0, steps, chunk_steps):
        event_end = min(steps, event_begin + chunk_steps)
        staging = perf_counter()
        chunk = _event_chunk(raw, event_begin, event_end, trials, channels)
        for trial, hasher in enumerate(hashes):
            hasher.update(memoryview(chunk[trial]))
        events, clock_times = _stage_inputs(chunk,input_token,event_begin,event_end,p.model.dt_s)
        with mx.stream(mx.gpu):
            timed_calls = _stage_timed_calls(p,clock_times,event_begin)
            if timed_calls is not None:
                mx.eval(timed_calls)
        staging_s += perf_counter() - staging
        del chunk
        pending_windows = 0
        while state.step < event_end:
            if component_steps:
                from .components import advance_components
                state = advance_components(p,state,events,steps,min(event_end,state.step+component_steps),
                    event_begin=event_begin,event_steps=event_end-event_begin,event_times=clock_times,timed_calls=timed_calls)
                mx.async_eval(*_state_buffers(state.values),*state.history.values(),state.release)
                component_passes += 1
                continue
            begin = state.step
            length=min(p.window,event_end-begin)
            if p.program.random_program is not None:
                from .expression_fixed import max_frames
                length=min(length,p.window-state.step%p.window,max_frames(p,trials))
            if frontier is not None:
                operation=lambda pp,ss,origin,length=length,events=events,clock_times=clock_times,timed_calls=timed_calls,event_begin=event_begin,event_end=event_end:advance(
                    pp,ss,events,steps,length,event_begin=event_begin,event_steps=event_end-event_begin,
                    event_times=clock_times,timed_calls=timed_calls,precision_epoch_begin=origin)
                state,bits=frontier.submit(operation,length)
            else:
                state, bits = advance(p, state, events, steps, length,
                                      event_begin=event_begin, event_steps=event_end-event_begin,
                                      event_times=clock_times, timed_calls=timed_calls)
            pending_windows += 1
            if pending_windows >= submission_windows:
                mx.async_eval(*_state_buffers(state.values), *state.history.values(), *state.due.values(),
                              state.release, *(() if state.spikes is None else (state.spikes,)), bits)
                pending_windows = 0
            if record:
                for word in range((state.step-begin+31)//32 if p.independent_neurons else 1):
                    block.append(bits[word*size:(word+1)*size] if p.independent_neurons else bits)
                    starts.append(begin+word*32)
            if record and (len(block) >= block_windows or state.step == steps
                           or frontier is not None and state.step==event_end):
                if frontier is not None:accept_frontier()
                collecting = perf_counter()
                with mx.stream(mx.gpu):
                    parts.extend(_collect_unit_windows(block, starts, p.neurons, trial_ids,
                                                       1 if p.window == 1 else 32,
                                                       sparse=sparse_recording))
                io = 0.
                if spike_sink is not None and parts:
                    batch = SpikeEvents(*(np.concatenate([part[i] for part in parts]) for i in range(3)))
                    io = float(spike_sink.append(batch))
                    if not np.isfinite(io) or io < 0:
                        raise ValueError("Spike sink returned invalid I/O timing")
                    parts.clear()
                spool_io_s += io
                collection_s += perf_counter() - collecting - io
                block, starts = [], []
        if frontier is not None and frontier.calls:accept_frontier()
        evaluate_state(state)
        del events, clock_times, timed_calls
    evaluate_state(state)
    simulation_s = perf_counter() - started - spool_io_s
    collecting = perf_counter()
    # A stable ordering of the small block list groups trials, preserving
    # chronological windows and ascending neuron ids within each trial.
    parts.sort(key=lambda part: int(part[0][0]))
    arrays = [np.asarray(parts[0][i], dtype=np.int64) if len(parts) == 1 else
              np.concatenate([part[i] for part in parts]).astype(np.int64, copy=False)
              if parts else np.array([], np.int64) for i in range(3)]
    spikes = SpikeEvents(*arrays)
    final_collection_s = perf_counter() - collecting if record else 0.
    consumed_hashes = known_hashes if known_hashes is not None else tuple(h.hexdigest() for h in hashes)
    remember_event_hashes(input_token, consumed_hashes)
    result = SimulationRun(spikes, dict(setup_s=setup_s, simulation_wall_s=simulation_s,
        collection_s=collection_s+final_collection_s, final_collection_s=final_collection_s,
        input_staging_s=staging_s, jit_and_warmup_s=jit_s, spike_spool_io_s=spool_io_s, component_passes=component_passes),
        mx.get_peak_memory(), str(mx.device_info()["device_name"]),
        input_sha256=consumed_hashes, dt_s=p.model.dt_s)
    return (result, state) if return_state else result


def run(connectome, stimulus, silenced=(), *, model, record=True, spike_sink=None):
    started = perf_counter()
    prepared = prepare(connectome, stimulus, model, silenced, component=not record)
    layout_s = perf_counter() - started
    result = run_prepared(prepared, stimulus, record=record, spike_sink=spike_sink)
    result.timings["layout_s"] = layout_s
    result.timings["setup_s"] += layout_s
    return result
