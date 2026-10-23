"""Bounded replay of the existing mixed-RNG Metal operation graph.

Only host bookkeeping uses relative window indices. The absolute Brian clock,
staged-event origin and every mutable model/RNG array remain explicit inputs.
No draw, seed, simulated state or absolute-window specialization is cached.
"""
from collections import OrderedDict
from dataclasses import dataclass,replace
import math
import weakref
import mlx.core as mx


_PLANS=OrderedDict()
_OWNERS={}
_PROGRAM_KEYS=weakref.WeakKeyDictionary()
_MAX_PLANS=32
_MIN_NEURONS=2**16


@dataclass(frozen=True)
class _Layout:
    values: tuple
    history: tuple
    due: tuple
    release: bool
    temporary: tuple
    trials: int
    step: int
    width: int
    frames: int
    end: int
    previous: bool
    checked: bool
    error_type: str
    error_catalog: tuple
    error_order: str
    program_identity: tuple
    precision_limbs: int
    cython: bool
    pooled: bool
    packed_count: int
    packed_size: int
    normal_pool: bool
    error_epoch_origin: int | None


def _packet(state):
    from .configured import _PackedValues
    random=state.random;stream=random.stream
    cython=random.buffers is not None
    packed=isinstance(state.values,_PackedValues)and not state.values.shared and cython
    prefix=None if random.pool is None else random.pool.data
    data=None if prefix is None else prefix.data
    layout=_Layout(tuple(state.values),tuple(state.history),tuple(state.due),state.release is not None,
        tuple(random.temporaries),stream.trials,state.step,0 if data is None else data.width,0 if data is None else data.frames,
        0 if data is None else random.pool.end_step,
        state.spikes is not None,state.check_index_error,state.runtime_error_type,state.runtime_errors,
        state.runtime_error_order,state.program_identity,state.native_precision_limbs,cython,data is not None,
        len(state.values.buffers)if packed else 0,state.values.size if packed else 0,
        data is not None and data.normal is not None,state.error_epoch_origin)
    arrays=[*(state.values.buffers if packed else tuple(state.values[name]for name in layout.values)),*(state.history[q]for q in layout.history),
        *(state.due[q]for q in layout.due)]
    if layout.release:arrays.append(state.release)
    if layout.previous:arrays.append(state.spikes)
    arrays.extend((stream.words,stream.position,stream.has_gauss,stream.gauss,random.errors))
    arrays.extend(random.temporaries[name]for name in layout.temporary)
    if cython:
        buffers=random.buffers
        arrays.extend((buffers.uniform,buffers.uniform_index,buffers.normal,buffers.normal_index))
    if prefix is not None:
        initial=prefix.initial
        arrays.extend((data.values,data.words,data.positions,initial.words,initial.position,
            initial.has_gauss,initial.gauss,prefix.cursor))
        if data.normal is not None:arrays.append(data.normal)
    return tuple(arrays),layout


def _restore(layout,arrays,program):
    from .configured import State,_pack_values,_PackedValues,_state_layout
    from .expression_rng import MT19937State
    from .expression_rng_prefix import PrefixData,StreamPrefix
    from .expression_runtime import RandomState,MixedLookahead
    values=iter(arrays)
    model=(_PackedValues(_state_layout(program),tuple(next(values)for _ in range(layout.packed_count)),layout.packed_size)
        if layout.packed_count else{name:next(values)for name in layout.values})
    history={q:next(values)for q in layout.history};due={q:next(values)for q in layout.due}
    release=next(values)if layout.release else None
    previous=next(values)if layout.previous else None
    stream=MT19937State(next(values),next(values),layout.trials,next(values),next(values))
    errors=next(values);temporary={name:next(values)for name in layout.temporary}
    buffers=None;cache=None
    if layout.cython:
        from .expression_cython import CythonBuffers
        buffers=CythonBuffers(next(values),next(values),next(values),next(values))
    if layout.pooled:
        data=PrefixData(next(values),next(values),next(values),layout.width,layout.frames)
        initial=MT19937State(next(values),next(values),layout.trials,next(values),next(values))
        prefix=StreamPrefix(data,initial,next(values),stream)
        if layout.normal_pool:prefix=replace(prefix,data=replace(data,normal=next(values)))
        cache=MixedLookahead(prefix,program,layout.step,layout.end)
    sentinel=object()
    if next(values,sentinel)is not sentinel:raise ValueError('Unexpected compiled expression state fields')
    return State(_pack_values(program,model),history,due,layout.step,release,
        check_index_error=layout.checked,spikes=previous,runtime_error_type=layout.error_type,
        runtime_errors=layout.error_catalog,runtime_error_order=layout.error_order,
        program_identity=layout.program_identity,native_precision_limbs=layout.precision_limbs,
        random=RandomState(stream,temporary,errors,cache,buffers),
        error_epoch_origin=layout.error_epoch_origin)


def _owner(p):
    identity=id(p);reference=_OWNERS.get(identity)
    if reference is not None and reference()is p:return reference
    def remove(reference):
        if _OWNERS.get(identity)is reference:
            _OWNERS.pop(identity,None)
            for key in tuple(_PLANS):
                if key[0]==identity:_PLANS.pop(key,None)
    reference=weakref.ref(p,remove);_OWNERS[identity]=reference
    return reference


def _shared_program_key(program):
    """Equal immutable emitted programs may share a graph across preparations."""
    if program not in _PROGRAM_KEYS:
        from dataclasses import fields,is_dataclass
        from collections.abc import Mapping
        from hashlib import sha256
        import numpy as np
        def freeze(value):
            if is_dataclass(value):
                return (type(value).__module__,type(value).__name__,
                    tuple((f.name,freeze(getattr(value,f.name)))for f in fields(value)))
            if isinstance(value,Mapping):return tuple((freeze(k),freeze(v))for k,v in value.items())
            if isinstance(value,(tuple,list)):return tuple(map(freeze,value))
            if isinstance(value,np.ndarray):return ('array',value.dtype.str,value.shape,sha256(value.tobytes()).hexdigest())
            if isinstance(value,np.generic):return ('scalar',value.dtype.str,value.tobytes())
            if isinstance(value,np.dtype):return ('dtype',value.str)
            if isinstance(value,float):return ('float',value.hex())
            if value is None or isinstance(value,(str,int,bool,bytes,type)):return value
            if type(value).__module__=='brian2.codegen.statements':
                return (type(value).__name__,freeze(vars(value)))
            # Unknown metadata must never make two distinct preparations equal.
            return ('identity',type(value).__module__,type(value).__name__,id(value))
        _PROGRAM_KEYS[program]=sha256(repr(freeze(program)).encode()).hexdigest()
    return _PROGRAM_KEYS[program]


def _shared_normal_threshold(program):
    """Interleaved inputs prevent treating this draw as a standalone tape."""
    from .expression_runtime import _fixed_draw
    draw=_fixed_draw(program)
    return (program.shared_expression_inputs and draw is not None
        and draw.function=='randn' and not draw.scalar)


def cython_batch_eligible(p):
    """A sequential graph may cross delay windows; each inner step may not."""
    from .expression_runtime import _fixed_draw
    from .expression_general import _prefix
    from .expression_single_call import selected_primitive
    random=p.program.random_program
    return (random is not None and getattr(random,'target',None)=='cython'
        and p.neurons>=_MIN_NEURONS
        and (_fixed_draw(p.program)is None or p.program.shared_expression_inputs)
        and not p.program.scalar_stages and not p.program.index_error_name
        and not getattr(random,'checked',False)and p.program.replay_phase is None
        and p.model.execution_phases==('groups','thresholds','synapses','resets')
        and (_prefix(p.program)is not None or _shared_normal_threshold(p.program))
        and (selected_primitive(p.program,'reset')is not None or 'reset' not in random.stages))


def cython_submission_limit(p,native_poisson):
    """Use two recording words only for the namespace pending-reset route."""
    from .configured import _state_layout,_uses_previous_events
    from .expression_single_call import selected_primitive
    return (64 if native_poisson is not None and p.input_arrays and cython_batch_eligible(p)
        and selected_primitive(p.program,'reset')=='randn'
        and p.model.arrival_delay_steps>=2 and not p.independent_neurons and not p.scan_arrivals
        and p.outgoing_destinations.size<=p.neurons and p.neurons<2**31
        and not p.program.variables['v'].conditional_write
        and 2*len(_state_layout(p.program).groups)+24+len(p.input_arrays)
            +int(_uses_previous_events(p.model,p.program))<=31 else 32)


def advance(p,state,events,steps,length,*,event_begin,event_steps,event_times,native_poisson=None):
    from .expression_runtime import _fixed_draw,_mixed_prefix,_mixed_prefix_eligible,MixedLookahead
    from .expression_general import _advance_steps,_prefix
    cython=state.random is not None and state.random.buffers is not None
    numpy_checked=(not cython and p.program.index_error_name is not None
        and p.program.runtime_error_type=='index' and _prefix(p.program) is not None
        and p.program.replay_phase is None)
    if (state.random is None or p.neurons<_MIN_NEURONS
            or (_fixed_draw(p.program)is not None and native_poisson is None)
            or (not cython and not _mixed_prefix_eligible(p.program))
            or (_prefix(p.program)is None and not _shared_normal_threshold(p.program))
            or p.program.scalar_stages or (p.program.index_error_name and not numpy_checked)
            or (getattr(p.program.random_program,"checked",False) and not numpy_checked)):return None
    if cython and not cython_batch_eligible(p):return None
    # Relative bookkeeping must not turn an unsafe absolute refractory-clock
    # conversion into a fast-path proof. Such horizons keep the original path.
    elapsed=(steps*p.model.dt_s+1e4+.001*p.model.dt_s)/p.model.dt_s
    if not math.isfinite(elapsed)or elapsed>=2**63:return None
    random=_mixed_prefix(p,state.random,state.step,steps,length)
    cache=random.pool
    if cython:
        # Keep speculative physical uniforms across delay windows. A first
        # step gets its own epoch, so its usual normal-buffer allocation does
        # not discard a long prefix. Later refills still invalidate on device.
        valid=(isinstance(cache,MixedLookahead)and cache.program is p.program
            and cache.data.stream is random.stream and cache.next_step==state.step
            and state.step+length<=cache.end_step)
        if not valid:
            from .expression_rng_prefix import prefix_pool
            from .expression_runtime import POOL_VALUE_BYTES,POOL_CHECKPOINT_BYTES
            normal=_shared_normal_threshold(p.program)
            namespace=bool(p.input_arrays)and not normal
            epoch=64 if namespace else 32
            frames=min(steps-state.step,max(length,1 if state.step==0 else epoch))
            # Pending namespace resets consume whole physical uniform buffers.
            # Normal thresholds retain their own raw-normal lookahead budget.
            checkpoint=20000 if namespace else 2496
            limit=min(POOL_VALUE_BYTES//(random.stream.trials*8)//checkpoint,
                POOL_CHECKPOINT_BYTES//(random.stream.trials*2496))*checkpoint
            if limit<checkpoint:return None
            capacity=min((p.neurons*frames+20000)*(2 if normal else 1),limit-checkpoint+1)
            prefix=prefix_pool(random.stream,capacity,checkpoint=checkpoint)
            if normal:
                from .expression_normal_prefix import prepare
                prefix=prepare(prefix)
            cache=MixedLookahead(prefix,p.program,state.step,state.step+frames)
            random=replace(random,pool=cache)
    elif not isinstance(cache,MixedLookahead)or cache.end_step<state.step+length:return None
    window,offset=divmod(state.step,p.window)
    working=replace(state,step=offset,history={q-window:a for q,a in state.history.items()},
        due={q-window:a for q,a in state.due.items()},
        random=replace(random,pool=replace(cache,next_step=offset,end_step=offset+length)))
    inputs,layout=_packet(working)
    static_names=('incoming_offsets','incoming_sources','weights','outgoing_offsets',
        'outgoing_destinations','channel_offsets','channels','amplitudes')
    if native_poisson is None:
        key=(id(p),layout,length,tuple(events.shape),tuple(event_times.shape),None)
        reference=_owner(p);static=None;static_inputs=()
    else:
        # Reuse the graph across equal model preparations. Every mutable
        # connectivity/weight/amplitude array is an explicit graph argument.
        key=('shared',_shared_program_key(p.program)if cython else p.program,p.model.signature,p.neurons,p.targets,p.window,
            p.compact_idle,p.independent_neurons,p.scan_arrivals,layout,length,
            tuple(events.shape),tuple(event_times.shape),native_poisson.identity)
        static=replace(p,input_arrays=(),**{name:None for name in static_names})
        static_inputs=tuple(getattr(p,name)for name in static_names)+tuple(p.input_arrays)
        reference=None
    if key not in _PLANS:
        count=len(static_inputs)
        output={}
        def operation(*arguments):
            if static is None:
                prepared=reference();state_arguments=arguments[:-3]
                if prepared is None:raise RuntimeError('Prepared expression graph was released')
            else:
                data=arguments[-count-3:-3]
                prepared=replace(static,input_arrays=tuple(data[len(static_names):]),
                    **dict(zip(static_names,data)))
                state_arguments=arguments[:-count-3]
            state=_restore(layout,state_arguments,prepared.program)
            final,bits=_advance_steps(prepared,state,arguments[-3],offset+length,length,
                event_begin=offset,event_steps=event_steps,event_times=arguments[-2],clock_params=arguments[-1],
                native_poisson=native_poisson)
            arrays,output['layout']=_packet(final)
            return (*arrays,bits)
        _PLANS[key]=(mx.compile(operation),output)
        while len(_PLANS)>_MAX_PLANS:_PLANS.popitem(last=False)
    _PLANS.move_to_end(key)
    operation,output=_PLANS[key]
    clock=mx.array([state.step,event_begin,event_steps],mx.int64)
    arrays=operation(*inputs,*static_inputs,events,event_times,clock)
    final=_restore(output['layout'],arrays[:-1],p.program)
    final=replace(final,step=state.step+length,history={q+window:a for q,a in final.history.items()},
        due={q+window:a for q,a in final.due.items()},
        random=replace(final.random,pool=replace(final.random.pool,next_step=state.step+length,end_step=cache.end_step)))
    return final,arrays[-1]
