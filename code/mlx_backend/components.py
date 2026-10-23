"""Bounded weak graph components sharing canonical packed state and phases."""
from dataclasses import dataclass, replace

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class ComponentLayout:
    rows: object
    positions: object
    groups: int


def prepare_components(connectome):
    n = connectome.neuron_ids.size
    if n > 262144:
        return None
    roots, sizes = list(range(n)), [1] * n

    def find(row):
        while roots[row] != row:
            roots[row] = roots[roots[row]]
            row = roots[row]
        return row

    # Include zero/silenced edges. The graph proof never splits a real edge.
    for source, destination in zip(connectome.sources, connectome.destinations):
        left, right = find(int(source)), find(int(destination))
        if left == right:
            continue
        if sizes[left] < sizes[right]:
            left, right = right, left
        roots[right] = left
        sizes[left] += sizes[right]
        if sizes[left] > 32:
            return None
    labels = np.fromiter((find(row) for row in range(n)), np.int32, count=n)
    members = np.argsort(labels, kind='stable')
    counts = np.bincount(labels, minlength=n)
    offsets = np.r_[0, np.cumsum(counts[counts > 0])]
    packed = []
    for begin, end in zip(offsets[:-1], offsets[1:]):
        component = members[begin:end].tolist()
        if len(packed) % 32 + len(component) > 32:
            packed.extend([-1] * (32 - len(packed) % 32))
        packed.extend(component)
    packed.extend([-1] * (-len(packed) % 32))
    rows = np.asarray(packed, np.int32)
    positions = np.empty(n, np.int32)
    valid = rows >= 0
    positions[rows[valid]] = np.flatnonzero(valid) % 32
    with mx.stream(mx.gpu):
        return ComponentLayout(mx.array(rows), mx.array(positions), rows.size // 32)


def _prefix(program):
    prefix = '_eon_component_'
    while any(name.startswith(prefix) for name in (*program.variables, *program.scalar_values)):
        prefix += 'x'
    return prefix


def kernel_source(program, loads, stores, namespace, scalar, error_load,
                  release_load, error_restore, release_store, spike_init, step_code):
    """Only geometry/history changes; all model code comes from configured."""
    p = _prefix(program)
    return f'''
        uint {p}lane = thread_position_in_threadgroup.x;
        uint {p}group = threadgroup_position_in_grid.x;
        int {p}row = {p}rows[({p}group % GROUPS)*32 + {p}lane];
        bool {p}valid = {p}row >= 0;
        uint trial = {p}group / GROUPS, row = max({p}row, 0);
        uint gid = trial*NEURONS + row;
        const long N = NEURONS;
        const int i = int(row);
        const ulong dt = DT_BITS;
        long start = params[0];
        int length = int(params[1]), staged_steps = int(params[2]);
        long event_begin = params[3];
        long {p}first_in = params[4], {p}count_in = params[5];
        long {p}first_out = params[6], {p}count_out = params[7];
        const long {p}size = NEURONS*B;
        {namespace}
        {loads}
        {error_load}
        long release_step = {release_load};
        {scalar}
        {error_restore}
        {spike_init}
        // Keep every timestep slot contiguous across the whole group.
        threadgroup uint {p}past[32*DELAY];
        for(uint {p}slot=0; {p}slot<DELAY; ++{p}slot)
            {p}past[{p}slot*32+{p}lane] = 0;
        if({p}valid) {{
            for(long {p}old=max(0l,start-DELAY); {p}old<start; ++{p}old) {{
                long {p}word = {p}old/W - {p}first_in;
                if({p}word >= 0 && {p}word < {p}count_in)
                    {p}past[({p}old%DELAY)*32+{p}lane] =
                        ({p}history_in[{p}word*{p}size+gid] >> ({p}old%W)) & 1u;
            }}
            for(long {p}word=0; {p}word<{p}count_out; ++{p}word) {{
                long {p}old={p}word+{p}first_out-{p}first_in;
                {p}history_out[{p}word*{p}size+gid] =
                    {p}old >= 0 && {p}old < {p}count_in ?
                    {p}history_in[{p}old*{p}size+gid] : 0u;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(int k=0; k<length; ++k) {{
            long step=start+k;
            const long t_in_timesteps=step;
            ulong t=clock_times[step-event_begin];
            uint {p}slot=step%DELAY;
            if({p}valid) {{
                {step_code}
                if(_spike) {{
                    long {p}word=step/W-{p}first_out;
                    if({p}word>=0 && {p}word<{p}count_out)
                        {p}history_out[{p}word*{p}size+gid] |= 1u << (step%W);
                }}
            }} else {{ _spike=false; }}
            // All padded lanes participate, and reads finish before overwrite.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {p}past[{p}slot*32+{p}lane]=_spike;
            // A different slot is read next when DELAY>1. The next
            // iteration's pre-store barrier publishes this write before
            // any later iteration can revisit its slot. Depth one must
            // publish immediately because the next read uses this slot.
            if(DELAY == 1) threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if({p}valid) {{
            {stores}
            q_release[gid]={release_store};
        }}
    '''


def advance_components(prepared, state, events, steps, end, *, event_begin=0,
                       event_steps=None, event_times=None, timed_calls=None):
    from .configured import (_PackedValues, _pack_values, _state_layout, _storage_dtype,
                             _neuron_kernel, _clock_times, _stage_timed_calls)

    p = prepared
    identity = (p.model.signature, p.neurons)
    if (p.components is None or not p.scan_arrivals or not 0 <= state.step < end <= steps
            or event_begin > state.step or (state.program_identity and state.program_identity != identity)):
        raise ValueError('Invalid component continuation')
    event_steps = steps if event_steps is None else event_steps
    if end > event_begin + event_steps:
        raise ValueError('Component continuation exceeds staged events')
    if end-state.step > 1024:
        raise ValueError('Component pass exceeds its bounded work budget')
    layout = _state_layout(p.program)
    packed = _pack_values(p.program, state.values)
    size, trials = packed.size, packed.size // p.neurons
    if (end-state.step)*size > 64*1024*1024:
        raise ValueError('Component pass exceeds its bounded work budget')
    first_in = min(state.history, default=0)
    count_in = max(state.history, default=-1)-first_in+1
    first_out = max(0, (end-p.model.arrival_delay_steps)//p.window)
    count_out = (end-1)//p.window-first_out+1
    if count_out*size > 2**31-1:
        raise OverflowError('Component history exceeds safe Metal indexing')
    with mx.stream(mx.gpu):
        history = (mx.concatenate([state.history.get(q, mx.zeros((size,),mx.uint32))
                                  for q in range(first_in, first_in+count_in)])
                   if count_in else mx.zeros((1,),mx.uint32))
        if event_times is None:
            event_times = mx.array(_clock_times(event_begin,event_begin+event_steps,p.model.dt_s))
        if p.program.shared_timed_calls and timed_calls is None:
            timed_calls = _stage_timed_calls(p,event_times,event_begin)
        params = mx.array([state.step,end-state.step,event_steps,event_begin,
                           first_in,count_in,first_out,count_out],mx.int64)
        elapsed = (steps*p.model.dt_s+1e4+.001*p.model.dt_s)/p.model.dt_s
        fast = (p.program.refractory_elapsed_threshold is not None and np.isfinite(elapsed)
                and elapsed < 2**63 and 1e4 >= p.program.refractory_elapsed_threshold)
        outputs = _neuron_kernel(p.model,p.program,component=True)(
            inputs=[*packed.buffers, state.release if state.release is not None else mx.zeros((size,),mx.int64),
                    p.incoming_offsets,p.incoming_sources,p.weights,p.channel_offsets,p.channels,
                    p.amplitudes,events,event_times,params,p.components.rows,p.components.positions,history,
                    *p.input_arrays,*([timed_calls] if p.program.shared_timed_calls else [])],
            template=[('NEURONS',p.neurons),('B',trials),('C',len(p.targets)),('W',p.window),
                      ('GROUPS',p.components.groups),('DELAY',p.model.arrival_delay_steps),
                      ('FAST_REFRACTORY',bool(fast)),
                      ('FAST_CPU_REFRACTORY',bool(p.program.cpu_refractory_guard and np.isfinite(elapsed) and elapsed<2**62))],
            grid=(p.components.rows.size*trials,1,1),threadgroup=(32,1,1),
            output_shapes=[*layout.shapes(size),(size,),(count_out*size,)],
            output_dtypes=[_storage_dtype(group.dtype) for group in layout.groups]+[mx.int64,mx.uint32])
    return replace(state,values=_PackedValues(layout,outputs[:-2],size),
                   history={first_out+q:outputs[-1][q*size:(q+1)*size] for q in range(count_out)},
                   due={},step=end,release=outputs[-2],program_identity=identity)
