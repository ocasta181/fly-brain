"""State-dependent original N=1 Poisson inputs in one native delay-window pass.

Within a window no newly emitted spike can arrive (window <= delay). Only the
small set of Poisson target neurons needs serial simulation to determine the
global Brian draw order. The regular native engine then uses the resulting
events for every neuron, including those targets. Unused prefetched uniforms
are rolled back before advancing the actual Brian RNG by the consumed count.
"""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from functools import lru_cache
import math
import re

import mlx.core as mx
import numpy as np

from .configured import _poisson_increment


class _UniformTape:
    def __init__(self, device, codeobj_class, clock, capacity):
        self.device = device
        self.cython = codeobj_class.class_name == "cython"
        self.capacity = capacity
        self.saved = None
        if self.cython:
            import brian2 as b
            self.group = b.NeuronGroup(
                capacity, "eon_uniform : 1\neon_take : boolean", clock=clock,
                dtype={"eon_uniform": np.float64}, name="mlx_uniform_tape*",
            )
            self.group._network = "mlx-uniform-input-only"
            self.group.variables["eon_uniform"].conditional_write = self.group.variables["eon_take"]
            self.runner = self.group.run_regularly(
                "eon_uniform = rand()", codeobj_class=codeobj_class, name="mlx_uniform_generator*",
            )
            self.runner.before_run({})

    def _buffer(self):
        pointer = int(self.device.rand_buffer[0])
        if not pointer:
            return None
        return np.ctypeslib.as_array(ctypes.cast(pointer, ctypes.POINTER(ctypes.c_double)),
                                    shape=(20000,))

    def _sample(self, count):
        if not self.cython:
            return np.random.random(count)
        flags = self.group.variables["eon_take"].get_value()
        flags[:] = False
        flags[:count] = True
        self.runner.codeobj()
        return self.group.variables["eon_uniform"].get_value()[:count].copy()

    def prefetch(self, count):
        buffer = self._buffer() if self.cython else None
        self.saved = (np.random.get_state(), int(self.device.rand_buffer_index[0]),
                      None if buffer is None else buffer.copy())
        return self._sample_many(count)

    def _sample_many(self, count):
        if not self.cython or count <= self.capacity:
            return self._sample(count)
        return np.concatenate([self._sample(min(self.capacity, count-offset))
                               for offset in range(0, count, self.capacity)])

    def commit(self, consumed):
        numpy_state, index, values = self.saved
        np.random.set_state(numpy_state)
        if self.cython:
            if values is not None:
                self._buffer()[:] = values
            elif consumed == 0:
                # A no-draw CPU window does not allocate its first buffer.
                # Brian transfers NumPy's allocation to a malloc/free-owned
                # pointer in its generated _rand function; undo our prefetch.
                pointer = int(self.device.rand_buffer[0])
                if pointer:
                    free = ctypes.CDLL(None).free
                    free.argtypes = [ctypes.c_void_p]
                    free.restype = None
                    free(pointer)
                    self.device.rand_buffer[0] = 0
            self.device.rand_buffer_index[0] = index
        self._sample_many(consumed)
        self.saved = None


def _addition(variable, value, weight):
    if variable.dtype == "float32":
        return f"f_from_d(d_add(f_to_d({value}), {weight}))"
    return f"d_add({value}, {weight})"


def _fused_body(program, unique_count, release_update):
    """One designated thread owns channel-order RNG and all target rows.

    Other rows need no current-window Poisson events: the positive delay keeps
    their evolution independent until this dispatch completes. Every output
    row has exactly one writer, and the following marking dispatch is the
    global barrier before any newly emitted spikes become visible.
    """
    from .configured import _persistent, _state_layout, _state_aliases, _state_source
    from .compiler import _kind, _literal

    variables = tuple(v for v in _persistent(program) if not v.scalar)
    names=(_state_aliases(program) if program.scalar_stages else
           {v.name:v.code_name for v in _persistent(program)})
    layout = _state_layout(program)
    def packed_slot(variable, prefix):
        group, slot = layout.slots[variable.name]
        return f"{prefix}{layout.groups[group].buffer_name}[{slot}ul*(NEURONS*B)+row]"
    declarations = "\n".join(f"{v.metal_type} _eon_local_{v.name}[TARGETS];" for v in variables)
    initialize = "\n".join(f"_eon_local_{v.name}[_eon_target_slot] = {layout.decode(program,v.name,packed_slot(v, '_eon_p_'))};" for v in variables)
    loads = "\n".join(f"{v.metal_type} {names[v.name]} = _eon_local_{v.name}[_eon_target_slot];" for v in variables)
    if program.scalar_stages:
        from .shared_state import variables as shared_variables
        loads += '\n'+'\n'.join(f'{v.metal_type} {v.code_name};' for v in shared_variables(program))
    stores = "\n".join(f"_eon_local_{v.name}[_eon_target_slot] = {names[v.name]};" for v in variables)
    final = "\n".join(f"{packed_slot(v, '_eon_q_')} = {layout.encode(program,v.name,'_eon_local_'+v.name+'[_eon_target_slot]')};" for v in variables)
    error_name = program.index_error_name
    error_declarations = "bool index_errors[TARGETS];" if error_name else ""
    error_initialize = "index_errors[_eon_target_slot] = p_release[row] < 0L;" if error_name else ""
    release_load = "p_release[row] & 9223372036854775807L" if error_name else "p_release[row]"
    release_store = ("(index_errors[_eon_target_slot] ? (releases[_eon_target_slot] | (-9223372036854775807L-1L)) : releases[_eon_target_slot])"
                     if error_name else "releases[_eon_target_slot]")
    if error_name:
        loads += f"\nbool {error_name} = index_errors[_eon_target_slot];"
        stores += f"\nindex_errors[_eon_target_slot] = {error_name};"
        if program.runtime_error_type in {'zero_division','model'}:
            mask=65535 if program.runtime_error_type=='model' else 3
            error_declarations+='\nint error_kinds[TARGETS];'
            error_initialize+=f'\nerror_kinds[_eon_target_slot]=int(p_release[row]&{mask}L);'
            loads+=f'\n{error_name}_kind=error_kinds[_eon_target_slot];'
            stores+=f'\nerror_kinds[_eon_target_slot]={error_name}_kind;'
            release_store='(index_errors[_eon_target_slot] ? ((-9223372036854775807L-1L)|long(error_kinds[_eon_target_slot])) : releases[_eon_target_slot])'
    precision=bool(program.native_precision_limbs)
    if precision:
        error_declarations+='\nulong error_epochs[TARGETS];'
        error_initialize+='\nerror_epochs[_eon_target_slot]=(ulong(p_release[row])&0x7fffffffffffffffUL)>>16;'
        loads+=f'\nulong {error_name}_epoch=error_epochs[_eon_target_slot];'
        stores+=f'\nerror_epochs[_eon_target_slot]={error_name}_epoch;'
        release_store=f'(index_errors[_eon_target_slot] ? long(0x8000000000000000UL|(error_epochs[_eon_target_slot]<<16)|ulong(error_kinds[_eon_target_slot])) : releases[_eon_target_slot])'
    def epoch(code,phase):
        if not precision:return code
        return ('{bool _had_precision_error='+error_name+';'+code+
            f'if(!_had_precision_error&&{error_name}){error_name}_epoch=(ulong(step-params[4])<<2)|{phase}UL;}}')
    input_condition = program.variables["v"].conditional_write
    input_condition = f"_eon_local_{input_condition}[_eon_target_slot]" if input_condition else "true"
    g_condition = program.variables["g"].conditional_write or "true"
    g_condition=names.get(g_condition,g_condition)
    dynamics = program.dynamics_code
    if program.equilibrium is not None:
        at_equilibrium = " && ".join(f"{program.variables[name].code_name} == {_literal(value, _kind(np.asarray(value).dtype))}"
                                    for name,value in program.equilibrium.items())
        dynamics = f"if(!({at_equilibrium})) {{ {dynamics} }}"
    refractory = program.refractory_code
    if program.refractory_elapsed_threshold is not None:
        refractory = f"if(FAST_REFRACTORY) not_refractory = step >= release_step; else {{ {refractory} }}"
    spike_time = "f_from_d(t)" if program.variables["lastspike"].dtype == "float32" else "t"
    before_start=shared_offset=''
    threshold=program.threshold_code
    reset=program.reset_scalar_code+'\nif(firing[_eon_target_slot]) { '+program.reset_vector_code+' }'
    if program.scalar_stages:
        from .shared_state import phase_source,phase_load,layout as shared_layout
        shared_offset=f'uint _eon_shared_offset=k*{shared_layout(program)[2]};'
        before_start=_state_source(phase_source(program,'subexpression',aliases=names),names)
        refractory=_state_source(phase_source(program,'state',aliases=names),names)
        dynamics=''
        threshold=_state_source('{\n'+phase_load(program,'threshold',aliases=names)+'\n'+program.scalar_stages['threshold'].vector_code+'\n}',names)
        reset=(phase_load(program,'reset',aliases=names)+'\nif(firing[_eon_target_slot]) { '
               +_state_source(program.scalar_stages['reset'].vector_code,names)+' }')
        release_update=_state_source(release_update,names)
    threshold_execution=f"""
        bool _cond=false;
        {threshold}
        if(_cond) {{
            {names['lastspike']} = {spike_time}; {names['not_refractory']} = false;
            {release_update}
            fired[_eon_target_slot] |= 1u << k;
        }}
        firing[_eon_target_slot]=_cond;
    """
    if 'threshold' in program.disabled_stages:
        threshold_execution='firing[_eon_target_slot]=(previous_spikes[row]&1u)!=0; if(firing[_eon_target_slot]) fired[_eon_target_slot]|=1u<<k;'
    g_add = _addition(program.variables["g"], names['g'], "weights[edge]")
    v_add = _addition(program.variables["v"], "_eon_local_v[_eon_target_slot]", "increment")
    overflow_return="used[0]=cursor; overflow[0]=1; return;"
    if precision:
        overflow_return=("for(int _eon_target_slot=0;_eon_target_slot<TARGETS;++_eon_target_slot) {"
            "int row=int(poisson_meta[_eon_target_slot]);"+final+
            "q_release[row]="+release_store+";spike_bits[row]=fired[_eon_target_slot];}"
            "used[0]=cursor;overflow[0]=1;return;")
    return f"""
        if(work_index == 0) {{
            const int TARGETS = {unique_count};
            {program.scalar_code}
            {declarations}
            bool firing[TARGETS];
            uint fired[TARGETS];
            long releases[TARGETS];
            {error_declarations}
            for(int _eon_target_slot=0; _eon_target_slot<TARGETS; ++_eon_target_slot) {{
                int row = int(poisson_meta[_eon_target_slot]);
                {initialize}
                releases[_eon_target_slot] = {release_load};
                {error_initialize}
                fired[_eon_target_slot] = 0u;
            }}
            int cursor = cursor_in[0];
            if(overflow_in[0]) {{ {overflow_return} }}
            for(int k=0; k<length; ++k) {{
                long step = start+k;
                const long t_in_timesteps = step;
                ulong t = clock_times[step-event_begin];
                {shared_offset}
                for(int _eon_target_slot=0; _eon_target_slot<TARGETS; ++_eon_target_slot) {{
                    int row = int(poisson_meta[_eon_target_slot]);
                    const int i = row;
                    long release_step = releases[_eon_target_slot];
                    {loads}
                    {epoch(before_start+chr(10)+refractory+chr(10)+dynamics,0)}
                    {epoch(threshold_execution,1)}
                    releases[_eon_target_slot] = release_step;
                    {stores}
                }}
                for(int _eon_target_slot=0; _eon_target_slot<TARGETS; ++_eon_target_slot) {{
                    int row = int(poisson_meta[_eon_target_slot]);
                    const int i = row;
                    {loads}
                    if(({g_condition}) && (due_bits[row] & (1u << k))) {{
                        int source_bit = SOURCE_OFFSET+k;
                        for(int edge=incoming_offsets[row]; edge<incoming_offsets[row+1]; ++edge) {{
                            int src = incoming_sources[edge];
                            uint word = source_bit < W ? source_low[src] : source_high[src];
                            if((word >> (source_bit % W)) & 1u) {names['g']} = {g_add};
                        }}
                    }}
                    {stores}
                }}
                for(int channel=0; channel<CHANNELS; ++channel) {{
                    int _eon_target_slot = int(poisson_meta[TARGETS+channel]);
                    if(poisson_meta[TARGETS+5*CHANNELS+channel] && ({input_condition})) {{
                        int X = 0;
                        ulong px = poisson_meta[TARGETS+CHANNELS+channel];
                        if(cursor >= CAPACITY) {{ {overflow_return} }}
                        ulong U = uniforms[cursor++];
                        while(d_gt(U, px)) {{
                            ++X;
                            if(X > 1) {{
                                X=0; px=poisson_meta[TARGETS+CHANNELS+channel];
                                if(cursor >= CAPACITY) {{ {overflow_return} }}
                                U=uniforms[cursor++];
                            }} else {{
                                U=d_sub(U, px);
                                px=d_div(d_mul(poisson_meta[TARGETS+2*CHANNELS+channel], px),
                                         poisson_meta[TARGETS+3*CHANNELS+channel]);
                            }}
                        }}
                        int event = poisson_meta[TARGETS+4*CHANNELS+channel] ? 1-X : X;
                        ulong increment = {_poisson_increment('event')};
                        _eon_local_v[_eon_target_slot] = {v_add};
                    }}
                }}
                for(int _eon_target_slot=0; _eon_target_slot<TARGETS; ++_eon_target_slot) {{
                    int row = int(poisson_meta[_eon_target_slot]);
                    const int i = row;
                    {loads}
                    const bool _spike=firing[_eon_target_slot];
                    {epoch(reset,3)}
                    {stores}
                }}
            }}
            for(int _eon_target_slot=0; _eon_target_slot<TARGETS; ++_eon_target_slot) {{
                int row = int(poisson_meta[_eon_target_slot]);
                {final}
                q_release[row] = {release_store};
                spike_bits[row] = fired[_eon_target_slot];
            }}
            used[0]=cursor; overflow[0]=0;
        }}
        if(channel_offsets[row] != channel_offsets[row+1]) return;
    """


@lru_cache(maxsize=32)
def _kernel(model, program, unique_count):
    from .binary64 import MSL_HEADER, literal
    from .binary32 import MSL_HEADER as MSL_BINARY32
    from .math64 import MSL_MATH_HEADER
    from .configured import _persistent, _state_layout

    variables = _persistent(program)
    layout = _state_layout(program)
    declarations = "\n".join(f"{v.metal_type} _eon_local_{v.name}[TARGETS];" for v in variables)
    initialize = []
    for variable in variables:
        group, slot = layout.slots[variable.name]
        buffer = layout.groups[group].buffer_name
        expression=layout.decode(program,variable.name,f'_eon_p_{buffer}[{slot}ul*NEURONS+row]')
        initialize.append(f"_eon_local_{variable.name}[_eon_target_slot] = {expression};")
    initialize = "\n".join(initialize)
    loads = "\n".join(f"{v.metal_type} {v.code_name} = _eon_local_{v.name}[_eon_target_slot];" for v in variables)
    stores = "\n".join(f"_eon_local_{v.name}[_eon_target_slot] = {v.code_name};" for v in variables)
    v_condition = program.variables["v"].conditional_write
    input_condition = f"_eon_local_{v_condition}[_eon_target_slot]" if v_condition else "true"
    g_guard = program.variables["g"].conditional_write
    g_condition = program.variables[g_guard].code_name if g_guard else "true"
    namespace_buffer = next(iter(program.input_arrays), "")
    namespace_loads, namespace_offset = [], 0
    for name, array in program.input_arrays.items():
        if name != namespace_buffer:
            namespace_loads.append(f"auto {name} = {namespace_buffer}+{namespace_offset}L;")
        namespace_offset += array.values.size
    index_error = f"bool {program.index_error_name} = false;" if program.index_error_name else ""
    spike_time = "f_from_d(t)" if program.variables["lastspike"].dtype == "float32" else "t"
    g_add = _addition(program.variables["g"], "g", "weights[edge]")
    v_add = _addition(program.variables["v"], "_eon_local_v[_eon_target_slot]", "increment")
    # Collective NumPy reset casts depend on all firing rows. Such models use
    # single-step prepasses; their reset executes in the full native network.
    reset_scalar = "" if program.collective_reset else program.reset_scalar_code
    reset_vector = "" if program.collective_reset else program.reset_vector_code
    threshold_execution=f"""bool _cond=false;{program.threshold_code}
        if(_cond) {{ lastspike={spike_time}; not_refractory=false; }} firing[_eon_target_slot]=_cond;"""
    if 'threshold' in program.disabled_stages:threshold_execution='firing[_eon_target_slot]=(previous_spikes[row]&1u)!=0;'
    source = f"""
        if(thread_position_in_grid.x != 0) return;
        const int TARGETS = {unique_count};
        const long N = NEURONS;
        const ulong dt = {literal(model.dt_s)};
        {chr(10).join(namespace_loads)}
        {index_error}
        {program.scalar_code}
        {declarations}
        bool firing[TARGETS];
        for(int _eon_target_slot=0; _eon_target_slot<TARGETS; ++_eon_target_slot) {{
            int row = target_rows[_eon_target_slot];
            {initialize}
        }}
        int cursor = cursor_in[0];
        int length = int(params[1]);
        for(int k=0; k<length*CHANNELS; ++k) events[k]=0;
        if(overflow_in[0]) {{ used[0]=cursor; overflow[0]=1; return; }}
        for(int k=0; k<length; ++k) {{
            long step = params[0]+k;
            const long t_in_timesteps = step;
            ulong t = clock_times[k];
            for(int _eon_target_slot=0; _eon_target_slot<TARGETS; ++_eon_target_slot) {{
                int row = target_rows[_eon_target_slot];
                const int i = row;
                {loads}
                {program.refractory_code}
                {program.dynamics_code}
                {threshold_execution}
                {stores}
            }}
            for(int _eon_target_slot=0; _eon_target_slot<TARGETS; ++_eon_target_slot) {{
                int row = target_rows[_eon_target_slot];
                const int i = row;
                {loads}
                if(({g_condition}) && (due_bits[row] & (1u << k))) {{
                    int source_bit = SOURCE_OFFSET+k;
                    for(int edge=incoming_offsets[row]; edge<incoming_offsets[row+1]; ++edge) {{
                        int src = incoming_sources[edge];
                        uint word = source_bit < W ? source_low[src] : source_high[src];
                        if((word >> (source_bit % W)) & 1u) g = {g_add};
                    }}
                }}
                {stores}
            }}
            for(int channel=0; channel<CHANNELS; ++channel) {{
                int _eon_target_slot = channel_targets[channel];
                int event = 2;
                if(active[channel] && ({input_condition})) {{
                    int X = 0;
                    ulong px = qn[channel];
                    if(cursor >= CAPACITY) {{ used[0]=cursor; overflow[0]=1; return; }}
                    ulong U = uniforms[cursor++];
                    while(d_gt(U, px)) {{
                        ++X;
                        if(X > 1) {{
                            X=0; px=qn[channel];
                            if(cursor >= CAPACITY) {{ used[0]=cursor; overflow[0]=1; return; }}
                            U=uniforms[cursor++];
                        }} else {{
                            U=d_sub(U, px);
                            px=d_div(d_mul(probability[channel], px), complement[channel]);
                        }}
                    }}
                    event = reverse[channel] ? 1-X : X;
                    ulong increment = {_poisson_increment('event')};
                    _eon_local_v[_eon_target_slot] = {v_add};
                }}
                events[k*CHANNELS+channel] = uchar(event);
            }}
            for(int _eon_target_slot=0; _eon_target_slot<TARGETS; ++_eon_target_slot) {{
                int row = target_rows[_eon_target_slot];
                const int i = row;
                {loads}
                const bool _spike=firing[_eon_target_slot];
                {reset_scalar}
                if(firing[_eon_target_slot]) {{ {reset_vector} }}
                {stores}
            }}
        }}
        used[0]=cursor; overflow[0]=0;
    """
    return mx.fast.metal_kernel(
        name="fly_original_poisson_window",
        input_names=[*("_eon_p_"+group.buffer_name for group in layout.groups),
                     *(['previous_spikes'] if 'threshold' in program.disabled_stages else []),
                     *([namespace_buffer] if namespace_buffer else []), "target_rows", "channel_targets",
                     "incoming_offsets", "incoming_sources", "weights", "source_low",
                     "source_high", "due_bits", "amplitudes", "active", "qn", "probability",
                     "complement", "reverse", "uniforms", "clock_times", "params",
                     "cursor_in", "overflow_in"],
        output_names=["events", "used", "overflow"], source=source,
        header=MSL_HEADER+MSL_BINARY32+MSL_MATH_HEADER+program.header_code,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=8)
def _conditional_draw_kernel():
    from .binary64 import MSL_HEADER
    return mx.fast.metal_kernel(
        name="fly_original_poisson_phase_draw", input_names=["flags", "targets", "active", "qn",
            "probability", "complement", "reverse", "uniforms", "cursor_in", "overflow_in"],
        output_names=["events", "used", "overflow"], header=MSL_HEADER,
        source="""
            if(thread_position_in_grid.x != 0) return;
            int cursor=cursor_in[0];
            for(int index=0; index<STEPS*CHANNELS; ++index) events[index]=2u;
            if(overflow_in[0]) { used[0]=cursor; overflow[0]=1; return; }
            for(int step=0; step<STEPS; ++step) {
            for(int channel=0; channel<CHANNELS; ++channel) {
                if(!active[channel] || !flags[targets[channel]]) continue;
                int X=0;
                ulong px=qn[channel];
                if(cursor >= CAPACITY) { used[0]=cursor; overflow[0]=1; return; }
                ulong U=uniforms[cursor++];
                while(d_gt(U, px)) {
                    ++X;
                    if(X > 1) {
                        X=0; px=qn[channel];
                        if(cursor >= CAPACITY) { used[0]=cursor; overflow[0]=1; return; }
                        U=uniforms[cursor++];
                    } else {
                        U=d_sub(U,px);
                        px=d_div(d_mul(probability[channel],px),complement[channel]);
                    }
                }
                events[step*CHANNELS+channel]=uchar(reverse[channel] ? 1-X : X);
            }
            }
            used[0]=cursor; overflow[0]=0;
        """, compile_options={"math_mode":"safe"})


class NativePoisson:
    # At most 8 MiB of raw uniforms per transaction, independent of run length.
    # One timestep remains the lower bound when the channel count is large.
    MAX_TAPE_DRAWS = 1 << 20

    def __init__(self, device, inputs, clock, window, *, fallback_codeobj_class):
        from brian2.core.functions import Function

        self.inputs = tuple(inputs)
        self.targets = tuple(int(source._group.start) for source in inputs)
        rows = tuple(sorted(set(self.targets)))
        self.unique_count = len(rows)
        self.target_rows = mx.array(rows, dtype=mx.int32)
        self.channel_targets = mx.array([rows.index(target) for target in self.targets], dtype=mx.int32)
        classes = {type(source.codeobj) for source in inputs if source.codeobj is not None}
        if not classes:
            classes = {fallback_codeobj_class}
        if len(classes) != 1 or next(iter(classes)).class_name not in ("numpy", "cython"):
            raise NotImplementedError("Original Poisson inputs need a common NumPy/Cython RNG target")
        codeobj_class = next(iter(classes))
        self.tape = _UniformTape(device, codeobj_class, clock, max(1, len(inputs)*window*2))
        thresholds, probabilities, complements, reversals = [], [], [], []
        for source in inputs:
            probability = float(source.rate*clock.dt)
            # NumPy validates p at runtime. Cython runs its emitted inversion
            # constants even outside [0, 1]; retain that target's draw order.
            if codeobj_class.class_name == "numpy" and not 0 <= probability <= 1:
                # Active invalid NumPy sources execute through phase replay.
                # Inactive sources and zero-duration runs must prepare without
                # evaluating the invalid binomial or consuming a uniform.
                probability = 0.
            reverse = probability > .5
            p = 1-probability if reverse else probability
            q = 1-p
            if codeobj_class.class_name == "cython":
                function = next(value for value in source.variables.values() if isinstance(value, Function))
                code = function.implementations["cython"].get_code(source)
                qn = float(re.search(r"cdef double px = ([-+.0-9eE]+)", code).group(1))
                # Brian 2.8's Cython generator uses the original p, including
                # when reversing the distribution. Preserve that literal.
                p = float(re.search(r"\(\(1-X\+1\) \* ([-+.0-9eE]+)", code).group(1))
                q = float(re.search(r"/\(X\*([-+.0-9eE]+)\)", code).group(1))
            else:
                # NumPy RandomState legacy binomial inversion computes its
                # threshold with scalar libm exp/log, not NumPy ufunc exp/log.
                qn = math.exp(math.log(q))
            thresholds.append(qn)
            probabilities.append(p)
            complements.append(q)
            reversals.append(reverse)
        self.qn = mx.array(np.asarray(thresholds, np.float64).view(np.uint64))
        self.probability = mx.array(np.asarray(probabilities, np.float64).view(np.uint64))
        self.complement = mx.array(np.asarray(complements, np.float64).view(np.uint64))
        self.reverse = mx.array(reversals, dtype=mx.bool_)
        self.metadata_static = np.concatenate((
            np.asarray(rows, np.uint64),
            np.asarray([rows.index(target) for target in self.targets], np.uint64),
            np.asarray(thresholds, np.float64).view(np.uint64),
            np.asarray(probabilities, np.float64).view(np.uint64),
            np.asarray(complements, np.float64).view(np.uint64),
            np.asarray(reversals, np.uint64),
        ))

    def can_fuse(self, prepared):
        from .configured import _persistent, _state_layout
        program = prepared.program
        slots = (2*len(_state_layout(program).groups)+19
                 +bool(program.input_arrays)+bool(program.shared_timed_calls)+bool(program.scalar_stages))
        # Bound target-thread register pressure as well as Metal buffer slots.
        # Wider models retain the native independent prepass, including all
        # packed state/namespace support, rather than a CPU fallback.
        # Error-bearing snapshots remain serial, but can still feed the
        # existing checked target loop when their values do not depend on
        # current-window input. Its error-kind transport/replay is unchanged.
        shared_safe=(program.scalar_parallel or
            (program.scalar_input_independent and program.compiled_integer_stages
             and program.runtime_error_type=='model' and program.runtime_error_order=='row'))
        return ((not program.scalar_stages or shared_safe) and not program.collective_reset and self.unique_count <= 8
                and sum(not v.scalar for v in _persistent(program)) <= 6 and slots <= 31)

    def chunk_steps(self, remaining):
        """Bound uniform storage without changing timestep or RNG order."""
        steps = max(1, self.MAX_TAPE_DRAWS // (2*len(self.inputs)))
        return min(remaining, steps)

    @contextmanager
    def transaction(self, steps):
        """Keep the draw cursor on GPU until a bounded tape/run boundary."""
        capacity = len(self.inputs)*steps*2
        if capacity > np.iinfo(np.int32).max:
            raise ValueError("Original Poisson tape exceeds its native cursor range")
        uniforms = self.tape.prefetch(capacity)
        self.uniforms = mx.array(uniforms.view(np.uint64))
        self.used = mx.zeros((1,), mx.int32)
        self.overflow = mx.zeros((1,), mx.int32)
        self.capacity = capacity
        self.metadata = mx.array(np.concatenate((self.metadata_static,
            np.asarray([source.active for source in self.inputs], np.uint64))))
        try:
            yield
            mx.eval(self.used, self.overflow)
            if int(self.overflow.item()):
                raise RuntimeError("Original Poisson inversion exhausted its bounded uniform tape")
            consumed = int(self.used.item())
        except BaseException:
            self.tape.commit(0)
            raise
        else:
            self.tape.commit(consumed)
        finally:
            self.uniforms = self.used = self.overflow = self.metadata = None

    def draw_conditional(self, flags, *, steps=1):
        active = mx.array([source.active for source in self.inputs], dtype=mx.bool_)
        targets = self.target_rows[self.channel_targets]
        events, self.used, self.overflow = _conditional_draw_kernel()(
            inputs=[flags, targets, active, self.qn, self.probability, self.complement,
                    self.reverse, self.uniforms, self.used, self.overflow],
            template=[("CHANNELS",len(self.inputs)),("CAPACITY",self.capacity),("STEPS",steps)],
            grid=(1,1,1), threadgroup=(1,1,1),
            output_shapes=[(steps*len(self.inputs),),(1,),(1,)],
            output_dtypes=[mx.uint8,mx.int32,mx.int32])
        return events

    def events(self, prepared, state, length, times):
        from .configured import _pack_values

        if prepared.program.collective_reset:
            assert length == 1

        window = prepared.window
        source_window, source_offset = divmod(state.step-prepared.model.delay_steps, window)
        zeros = mx.zeros((prepared.neurons,), mx.uint32)
        low = state.history.get(source_window, zeros)
        high = state.history.get(source_window+1, zeros)
        index, offset = divmod(state.step, window)
        due = state.due.get(index, zeros)
        if offset:
            due = due >> offset
        count = len(self.inputs)*length
        active = mx.array([source.active for source in self.inputs], dtype=mx.bool_)
        params = mx.array([state.step, length], dtype=mx.int64)
        outputs = _kernel(prepared.model, prepared.program, self.unique_count)(
                inputs=[*_pack_values(prepared.program, state.values).buffers,
                        *([state.spikes if state.spikes is not None else zeros]
                          if 'threshold' in prepared.program.disabled_stages else []), *prepared.input_arrays,
                        self.target_rows, self.channel_targets, prepared.incoming_offsets,
                        prepared.incoming_sources, prepared.weights, low, high, due,
                        prepared.amplitudes, active, self.qn, self.probability, self.complement,
                        self.reverse, self.uniforms, times, params, self.used, self.overflow],
                template=[("NEURONS", prepared.neurons), ("W", window),
                          ("SOURCE_OFFSET", source_offset), ("CHANNELS", len(self.inputs)),
                          ("CAPACITY", self.capacity)],
                grid=(1, 1, 1), threadgroup=(1, 1, 1),
                output_shapes=[(count,), (1,), (1,)],
                output_dtypes=[mx.uint8, mx.int32, mx.int32],
            )
        events, self.used, self.overflow = outputs
        return events
