"""Run stateful draws at the canonical engine's native phase boundaries."""
from dataclasses import dataclass,replace
from functools import lru_cache
from types import MappingProxyType

import mlx.core as mx


@dataclass(frozen=True)
class _PhaseModel:
    model: object

    def __getattr__(self,name):return getattr(self.model,name)

    @property
    def reset_uses_previous_spikes(self):return True

    @property
    def arrival_delay_steps(self):return max(1,self.model.arrival_delay_steps)


@lru_cache(maxsize=64)
def _program(program,phase,commit=False,draw=False):
    options=dict(random_program=None,replay_phase=phase,equilibrium=None,
        refractory_elapsed_threshold=None)
    if draw:options['random_program']=program.random_program
    if commit:
        from .compiler import ScalarStage
        shared=MappingProxyType({name:ScalarStage('',(),'', '')for name in (program.scalar_stages or {})})
        options.update(threshold_code='',scalar_stages=shared,
            disabled_stages=tuple(set(program.disabled_stages)|{'threshold'}))
    return replace(program,**options)


def _active(p,state,random):
    active=random.errors==0
    if getattr(p.program.random_program,'checked',False)and state.release is not None:
        active=active & ~mx.any(state.release.reshape(random.stream.trials,p.neurons)<0,axis=1)
    return active


def _checked_merge(p,before,after,active):
    """Commit Cython's completed rows, retaining the first failed carrier.

    Each native row owns local variables until its code-object epilogue.
    A later row cannot commit after an earlier row throws on the CPU.
    All selection stays on device, including inactive independent trials.
    """
    from .configured import _pack_values
    rows=mx.arange(p.neurons,dtype=mx.int32)[None,:]
    errors=after.release.reshape(-1,p.neurons)<0
    first=mx.min(mx.where(errors,rows,p.neurons),axis=1)[:,None]
    writes=(active[:,None] & (rows<first)).reshape(-1)
    carriers=(active[:,None] & (rows<=first)).reshape(-1)
    values={name:mx.where(active if p.program.variables[name].scalar else writes,
        value,before.values[name])for name,value in after.values.items()}
    return replace(after,values=_pack_values(p.program,values),
        release=mx.where(carriers,after.release,before.release if before.release is not None else mx.zeros_like(after.release)))


def _fixed_draw(p,random,active):
    from .expression_cython import buffered_draws
    from .expression_runtime import RandomState
    site=p.program.random_program
    counts=mx.where(active,p.neurons,0).astype(mx.uint32)
    values,stream,buffers=buffered_draws(random.stream,random.buffers,site.function,p.neurons,counts)
    return values,RandomState(stream,random.temporaries,random.errors,None,buffers)


def _rebind_values(program,values):
    from .configured import _PackedValues,_state_layout,_pack_values
    layout=_state_layout(program)
    if isinstance(values,_PackedValues)and values.layout==layout:
        return values if values.layout is layout else _PackedValues(layout,values.buffers,values.size,values.shared)
    return _pack_values(program,values)


@lru_cache(maxsize=64)
def _prefix(program):
    """Keep pure dynamics/threshold/arrivals in the canonical fused kernel."""
    from .expression_lowering import Draw,Segment
    from .expression_fixed import FixedRandomProgram
    random=program.random_program
    speculative=(random.target=='numpy' and program.index_error_name is not None
        and program.runtime_error_type=='index')
    if program.scalar_stages or (program.index_error_name and not speculative):return None
    if any(name!='reset'or random.compiled_stages[name].single_call not in('rand','randn')
           for name in random.serial_required):return None
    if set(random.stages)-{'threshold','reset'} or (speculative and 'reset' not in random.stages):return None
    early=random.stages.get('threshold',())
    draws=[op for op in early if isinstance(op,Draw)]
    if len(draws)!=1:return None
    draw=draws[0]
    if draw.function!='rand'or draw.scalar or draw.guard is not None:return None
    if any(isinstance(op,Segment)and(op.scalar or (op.may_raise and not (speculative and op.index_fault)) or
           (set(op.writes)-{'_cond','lastspike','not_refractory'} if not speculative else
            {name for name in op.writes if name in program.variables and program.variables[name].persistent}
                -{'lastspike','not_refractory'}))for op in early):return None
    if any((isinstance(op,Segment)and op.may_raise)or
           (isinstance(op,Draw)and op.function not in('rand','randn'))
           for op in random.stages.get('reset',())):return None
    source='\n'.join(op.source for op in early if isinstance(op,Segment))
    if speculative:
        from .expression_lowering import speculative_threshold
        source=speculative_threshold(program,draw)
    fixed=FixedRandomProgram('rand',random.variables[draw.result].code_name or draw.result,random.target,checked=speculative)
    # Only random resets are executed separately below. Keep the original
    # compiled arithmetic for deterministic resets in the canonical kernel.
    reset=(dict(reset_code='',reset_scalar_code='',reset_vector_code='',collective_reset=False)
           if 'reset' in random.stages else {})
    return draw,replace(program,random_program=fixed,threshold_code=source,equilibrium=None,**reset)


@lru_cache(maxsize=64)
def _normal_prefix(program):
    """Use the canonical phase kernel for a pure shared normal threshold."""
    from .expression_runtime import _fixed_draw
    from .expression_lowering import Segment
    from .expression_fixed import FixedRandomProgram
    draw=_fixed_draw(program)
    if (not program.shared_expression_inputs or draw is None or draw.function!='randn'
            or draw.scalar or set(program.random_program.stages)!={'threshold'}
            or program.scalar_stages or program.index_error_name
            or program.replay_phase is not None):return None
    source='\n'.join(op.source for op in program.random_program.stages['threshold'] if isinstance(op,Segment))
    fixed=FixedRandomProgram('randn',program.random_program.variables[draw.result].code_name or draw.result,
        program.random_program.target)
    return draw,replace(program,random_program=fixed,threshold_code=source,equilibrium=None)


def _normal_prefix_draw(p,random,counts):
    from .expression_runtime import MixedLookahead
    from .expression_cython import buffered_draws
    from .expression_rng_prefix import prefix_normals
    from .expression_rng import normals
    stream,buffers,cache=random.stream,random.buffers,random.pool
    if buffers is not None:
        if isinstance(cache,MixedLookahead):
            packed,stream,buffers,prefix=buffered_draws(stream,buffers,'randn',p.neurons,counts,_prefix=cache.data)
            cache=replace(cache,data=prefix)
        else:packed,stream,buffers=buffered_draws(stream,buffers,'randn',p.neurons,counts)
    elif isinstance(cache,MixedLookahead):
        packed,stream,prefix=prefix_normals(stream,cache.data,p.neurons,counts,target=p.program.random_program.target)
        cache=replace(cache,data=prefix)
    else:packed,stream=normals(stream,p.neurons,counts,target=p.program.random_program.target)
    return packed,stream,buffers,cache


def _advance_prefix(p,state,events,steps,*,event_begin,event_steps,event_times,timed_calls,clock_params=None,native_poisson=None,
                    _pending_reset=None,_defer_reset=False,_pipeline=False,_normal=False):
    if p.model.execution_phases!=('groups','thresholds','synapses','resets'):return None
    selected=_normal_prefix(p.program)if _normal else _prefix(p.program)
    if selected is None:return None
    from .configured import advance as canonical_advance,_pack_values,_state_layout
    from .expression_runtime import RandomState,MixedLookahead,_mixed_prefix,_stage
    from .expression_rng import uniforms
    draw,program=selected
    random=_mixed_prefix(p,state.random,state.step,steps)
    counts=mx.where(random.errors==0,p.neurons,0).astype(mx.uint32)
    stream=random.stream;buffers=random.buffers;cache=random.pool
    pre_input=bool(native_poisson is not None and not p.program.variables['v'].conditional_write)
    combined_input=pre_input and isinstance(cache,MixedLookahead) and not _normal
    if _normal:
        packed,stream,buffers,cache=_normal_prefix_draw(p,random,counts)
    elif combined_input:
        deferred_normal=None if _pending_reset is None else _pending_reset.normal_counts
        result=native_poisson.draw(p,state,random,random.errors==0,uniform_counts=counts,pending_normal=deferred_normal)
        input_events,random,packed=result[:3]
        if deferred_normal is not None:
            _pending_reset=replace(_pending_reset,draws=result[3],normal_counts=None,
                blocks=result[4]if deferred_normal.ndim==2 else _pending_reset.blocks)
        stream,buffers,cache=random.stream,random.buffers,random.pool
    elif buffers is not None:
        from .expression_cython import buffered_draws
        if isinstance(cache,MixedLookahead):
            packed,stream,buffers,prefix=buffered_draws(stream,buffers,'rand',p.neurons,counts,_prefix=cache.data,_view=True)
            cache=replace(cache,data=prefix)
        else:packed,stream,buffers=buffered_draws(stream,buffers,'rand',p.neurons,counts)
    elif isinstance(cache,MixedLookahead):
        from .expression_rng_prefix import prefix_uniforms
        packed,stream,prefix=prefix_uniforms(stream,cache.data,p.neurons,counts,_view=True)
        cache=replace(cache,data=prefix)
    else:packed,stream=uniforms(stream,p.neurons,counts)
    random=RandomState(stream,random.temporaries,random.errors,cache,buffers)
    part=replace(p,program=program,compact_idle=False,components=None)
    prefix_events,prefix_begin,prefix_steps,prefix_times,prefix_clock=events,event_begin,event_steps,event_times,clock_params
    full_clock=bool(_pipeline and combined_input and p.input_arrays and clock_params is not None
        and p.program.random_program.target=='cython')
    if pre_input:
        # Eligibility has no model-state dependency. Threshold uniforms have
        # already advanced the stream, so inputs can feed the same phase kernel.
        if combined_input:prefix_events=input_events
        elif _normal and isinstance(random.pool,MixedLookahead):
            prefix_events,random,_=native_poisson.draw(p,state,random,random.errors==0,
                uniform_counts=mx.zeros((random.stream.trials,),mx.uint32))
        else:prefix_events,random=native_poisson.draw(p,state,random,random.errors==0)
        prefix_begin,prefix_steps=state.step,1
        if not full_clock:
            prefix_times=(event_times[state.step-event_begin:state.step-event_begin+1]if clock_params is None
                else mx.take(event_times,(clock_params[0]-clock_params[1]).reshape(1)))
            prefix_clock=None if clock_params is None else mx.stack([clock_params[0],clock_params[0],mx.array(1,mx.int64)])
    elif native_poisson is not None:
        part=replace(part,targets=(),channel_offsets=mx.zeros((p.neurons+1,),mx.int32))
        if 'reset' not in p.program.random_program.stages:
            part=replace(part,program=replace(program,reset_code='',reset_scalar_code='',reset_vector_code=''))
    working=replace(state,values=_rebind_values(program,state.values))
    from .expression_single_call import selected_primitive,advance_packed
    from .expression_numpy_packed import selected as numpy_selected,advance as numpy_advance
    numpy_reset=numpy_selected(p.program,'reset')
    cython_reset=selected_primitive(p.program,'reset')if p.program.random_program.target=='cython'else None
    cython_markers=cython_reset is not None and 2*len(_state_layout(p.program).groups)+11+len(p.input_arrays)<=31
    defer_markers=bool((numpy_reset is not None or cython_markers) and p.model.arrival_delay_steps>0 and not p.independent_neurons
        and not p.scan_arrivals and p.outgoing_destinations.size<=p.neurons)
    from .trigonometry32 import scalar_sine_threadgroup
    from .expression_lowering import speculative_rank_eligible
    select=((not program.index_error_name or speculative_rank_eligible(program)) and (p.independent_neurons or p.model.arrival_delay_steps>0) and p.neurons%256==0
        and scalar_sine_threadgroup(program,p.neurons*random.stream.trials)==256
        and (selected_primitive(p.program,'reset')is not None or numpy_reset is not None))
    from .configured import _uses_previous_events
    rank_flags=bool(select and _defer_reset and combined_input and p.input_arrays and cython_reset=='randn'
        and 2*len(_state_layout(p.program).groups)+25+len(p.input_arrays)+bool(p.program.shared_timed_calls)
            +int(_uses_previous_events(p.model,p.program))<=31)
    due_fusion=bool(_pipeline and combined_input and p.input_arrays and _pending_reset is not None
        and cython_reset=='randn' and 2*len(_state_layout(p.program).groups)+26+len(p.input_arrays)
            +bool(p.program.shared_timed_calls)+int(_uses_previous_events(p.model,p.program))<=31)
    if due_fusion and p.window==p.model.arrival_delay_steps:due_fusion='history'
    result=canonical_advance(part,working,prefix_events,steps,1,event_begin=prefix_begin,event_steps=prefix_steps,
        event_times=prefix_times,timed_calls=timed_calls,fused_idle=False,_expression_draws=packed,
        _expression_clock=prefix_clock,_expression_selection=select,_expression_defer_markers=defer_markers,
        _expression_rank_flags=rank_flags,_expression_due_fusion=due_fusion,_expression_full_clock=full_clock,
        _expression_pending_reset=_pending_reset,
        _expression_pending_shift=(state.step-1+p.model.arrival_delay_steps)%p.window if _pending_reset is not None else None)
    previous_marker=None
    if _pending_reset is not None:previous_marker=result[-1];result=result[:-1]
    after,bits,selection=result if select else(*result,None)
    if previous_marker is not None:
        q,shift=divmod(state.step-1+p.model.arrival_delay_steps,p.window)
        due=dict(after.due);due[q]=previous_marker if due_fusion else(due[q]|previous_marker if q in due else previous_marker)
        if shift and q+1 not in due:due[q+1]=mx.zeros_like(previous_marker)
        after=replace(after,due=due)
    if native_poisson is not None and not pre_input:
        # Pure dynamics, threshold and recurrent arrivals have completed.
        # Inputs consume this same stream before reset; do not replay arrivals.
        input_state=replace(after,step=state.step,spikes=bits)
        if _normal and isinstance(random.pool,MixedLookahead):
            input_events,random,_=native_poisson.draw(p,input_state,random,random.errors==0,
                uniform_counts=mx.zeros((random.stream.trials,),mx.uint32))
        else:input_events,random=native_poisson.draw(p,input_state,random,random.errors==0)
        input_program=_program(p.program,'synapses')
        input_part=replace(p,program=input_program,model=_PhaseModel(p.model),
            incoming_offsets=mx.zeros((p.neurons+1,),mx.int32),compact_idle=False,
            independent_neurons=False,components=None)
        input_state=replace(input_state,values=_rebind_values(input_program,input_state.values),history={},due={})
        updated,_=canonical_advance(input_part,input_state,input_events,steps,1,
            event_begin=state.step,event_steps=1,
            event_times=(event_times[state.step-event_begin:state.step-event_begin+1]if clock_params is None
                else mx.take(event_times,(clock_params[0]-clock_params[1]).reshape(1))),fused_idle=False,
            _expression_clock=(None if clock_params is None else
                mx.stack([clock_params[0],clock_params[0],mx.array(1,mx.int64)])))
        after=replace(after,values=updated.values)
        if 'reset' not in p.program.random_program.stages:
            # State-dependent inputs follow thresholding and precede reset.
            # Reuse the original reset phase without replaying arrivals or
            # committing the clock/history a second time.
            reset_program=_program(p.program,'resets')
            reset_part=replace(p,program=reset_program,model=_PhaseModel(p.model),
                compact_idle=False,independent_neurons=False,components=None)
            reset_state=replace(after,step=state.step,spikes=bits,
                values=_rebind_values(reset_program,after.values))
            updated,_=canonical_advance(reset_part,reset_state,events,steps,1,
                event_begin=event_begin,event_steps=event_steps,event_times=event_times,
                timed_calls=timed_calls,fused_idle=False,_expression_clock=clock_params)
            after=replace(after,values=updated.values,release=updated.release,
                          check_index_error=updated.check_index_error)
    values={**after.values,**random.temporaries,'_cond':bits.astype(mx.bool_)}
    params=clock_params if clock_params is not None else mx.array([state.step,event_begin,event_steps],mx.int64)
    stored=None;pending=None
    if _defer_reset:
        source=_rebind_values(p.program,after.values)
        random,pending=advance_packed(p,'reset',source,values['_cond'],random,event_times,params,selection,_defer=True,
            _defer_normal=bool(combined_input and p.input_arrays and selected_primitive(p.program,'reset')=='randn'),
            _rank_flags=rank_flags)
        stored=source
    elif numpy_reset is not None:
        source=_rebind_values(p.program,after.values)
        random=replace(random,temporaries={name:values[name]for name in random.temporaries})
        marker_window,marker_shift=divmod(state.step+p.model.arrival_delay_steps,p.window)
        random,stored,marker=numpy_advance(p,'reset',source,values['_cond'],random,event_times,params,selection,
            marker_shift if defer_markers else None)
        if defer_markers:
            due=dict(after.due)
            due[marker_window]=due[marker_window]|marker if marker_window in due else marker
            if marker_shift and marker_window+1 not in due:due[marker_window+1]=mx.zeros_like(marker)
            after=replace(after,due=due)
        values.update(random.temporaries)
    elif selected_primitive(p.program,'reset')is not None:
        source=_rebind_values(p.program,after.values)
        marker_window,marker_shift=divmod(state.step+p.model.arrival_delay_steps,p.window)
        result=advance_packed(p,'reset',source,values['_cond'],random,event_times,params,selection,
            marker_shift if defer_markers else None)
        random,stored=result[:2]
        if defer_markers:
            marker=result[2];due=dict(after.due)
            due[marker_window]=due[marker_window]|marker if marker_window in due else marker
            if marker_shift and marker_window+1 not in due:due[marker_window+1]=mx.zeros_like(marker)
            after=replace(after,due=due)
    else:random=_stage(p,'reset',values,random,random.stream.trials,event_times,params)
    cache=random.pool
    if isinstance(cache,MixedLookahead):cache=replace(cache,next_step=after.step)
    random=RandomState(random.stream,{name:values[name]for name in random.temporaries},
        random.errors,cache,random.buffers)
    returned=(replace(after,values=stored if stored is not None else _pack_values(p.program,{name:values[name]for name in after.values}),
        random=random),bits)
    return (*returned,pending)if _pipeline else returned


def _advance_one(p,state,events,steps,length,*,event_begin,event_steps,event_times,timed_calls,
            fused_idle,native_poisson,stop_before_synapses,precision_epoch_begin,clock_params=None):
    from .configured import advance as canonical_advance,_pack_values,_mark_kernel
    from .expression_runtime import _stage,RandomState
    if length!=1:raise ValueError('General model phases advance one step at a time')
    if stop_before_synapses:
        raise NotImplementedError('Split random-expression synapse continuation is pending')
    checked=getattr(p.program.random_program,'checked',False)
    replay=p.program.replay_phase
    speculative=(p.program.random_program.target=='numpy' and _prefix(p.program) is not None)
    if precision_epoch_begin is None and (not checked or speculative) and replay is None:
        if (p.model.execution_phases==('groups','thresholds','synapses','resets')
                and _normal_prefix(p.program)is not None):
            return _advance_prefix(p,state,events,steps,event_begin=event_begin,event_steps=event_steps,
                event_times=event_times,timed_calls=timed_calls,clock_params=clock_params,
                native_poisson=native_poisson,_normal=True)
        fused=_advance_prefix(p,state,events,steps,event_begin=event_begin,event_steps=event_steps,
            event_times=event_times,timed_calls=timed_calls,clock_params=clock_params,native_poisson=native_poisson)
        if fused is not None:return fused
    random=state.random
    if native_poisson is not None and replay is None:
        from .expression_runtime import _mixed_prefix
        random=_mixed_prefix(p,random,state.step,steps)
    if random is None:raise ValueError('Random model requires explicit native stream state')
    size=p.neurons*random.stream.trials;step=state.step
    if state.spikes is None:state=replace(state,spikes=mx.zeros((size,),mx.uint32))
    if checked:state=replace(state,spikes=mx.where(mx.repeat(_active(p,state,random),p.neurons),state.spikes,0))
    params=clock_params if clock_params is not None else mx.array([step,event_begin,event_steps],mx.int64)
    phases=(replay,)if replay is not None else (('subexpression',)if p.program.scalar_stages else ())+p.model.execution_phases
    phase_names={'subexpression':'subexpression','groups':'state','thresholds':'threshold','resets':'reset'}
    values={**state.values,**random.temporaries,'_cond':state.spikes.astype(mx.bool_)}
    from .numpy_policy_context import stage as policy_stage, threshold_completed, refreshed, failed_phase_state
    synapses_completed=False
    synapse_spikes=None
    for phase in phases:
        p=refreshed(p)
        name=phase_names.get(phase)
        before=state
        if phase=='synapses':synapse_spikes=state.spikes
        active=_active(p,state,random)
        stages=getattr(p.program.random_program,'stages',{})
        if name in stages:
            incoming_errors=random.errors
            stage_random=(replace(random,errors=mx.where(active,random.errors,mx.array(0x80000000,mx.uint32)))
                if checked else random)
            with policy_stage(state,name) as boundary:
                random=_stage(p,name,values,stage_random,random.stream.trials,event_times,params)
            if checked:random=replace(random,errors=mx.where(active,random.errors,incoming_errors))
            merged={key:mx.where(active if p.program.variables[key].scalar else mx.repeat(active,p.neurons),
                values[key],old)for key,old in before.values.items()}
            visible=(values['_cond'].astype(mx.uint32)if phase=='thresholds' and random.policy_error is None
                     else boundary.spikes if boundary.changed_spikes else before.spikes)
            state=replace(state,values=_pack_values(p.program,merged),spikes=visible)
            if random.policy_error is not None:
                random=replace(random,temporaries={key:values[key] for key in random.temporaries})
                state=replace(state,random=random,policy_error=random.policy_error)
                state=failed_phase_state(p,state,synapse_spikes,synapses_completed)
                threshold_complete=(phase!='subexpression' and
                    p.model.execution_phases.index(phase)>p.model.execution_phases.index('thresholds'))
                return state,(state.spikes if threshold_complete else mx.zeros_like(state.spikes))
        else:
            phase_events,phase_begin,phase_steps,phase_times=events,event_begin,event_steps,event_times
            phase_clock=clock_params
            if phase=='synapses'and native_poisson is not None:
                from .expression_runtime import MixedLookahead
                from .expression_window import _shared_normal_threshold
                if isinstance(random.pool,MixedLookahead)and _shared_normal_threshold(p.program):
                    phase_events,random,_=native_poisson.draw(p,state,random,active,
                        uniform_counts=mx.zeros((random.stream.trials,),mx.uint32))
                else:phase_events,random=native_poisson.draw(p,state,random,active)
                phase_begin,phase_steps=step,1
                phase_times=(event_times[step-event_begin:step-event_begin+1]if clock_params is None
                    else mx.take(event_times,(clock_params[0]-clock_params[1]).reshape(1)))
                phase_clock=(None if clock_params is None else
                    mx.stack([clock_params[0],clock_params[0],mx.array(1,mx.int64)]))
            execution=state
            draws=None
            from .expression_fixed import FixedRandomProgram
            if checked and phase=='thresholds' and isinstance(p.program.random_program,FixedRandomProgram):
                draws,random=_fixed_draw(p,random,active)
            if phase=='synapses'and p.model.arrival_delay_steps==0:
                # The synapse-only kernel reads this step's already-computed
                # eventspace through the existing delayed-arrival protocol.
                markers=_mark_kernel()(inputs=[p.outgoing_offsets,p.outgoing_destinations,state.spikes],
                    template=[('N',p.neurons),('B',random.stream.trials),('W',p.window),('SHIFT',step%p.window)],
                    grid=(size,1,1),threadgroup=(min(256,size),1,1),
                    output_shapes=[(size,)]*2,output_dtypes=[mx.uint32]*2,init_value=0)
                source,bit=divmod(step-1,p.window)
                old=state.history.get(source,mx.zeros((size,),mx.uint32))
                mask=mx.array((1<<bit),mx.uint32)
                visible=(old&~mask)|(state.spikes<<bit)
                execution=replace(state,history={**state.history,source:visible},
                    due={**state.due,step//p.window:markers[0]})
            part=replace(p,program=_program(p.program,phase,draw=draws is not None),model=_PhaseModel(p.model),
                compact_idle=False,independent_neurons=False,components=None)
            execution=replace(execution,values=_rebind_values(part.program,execution.values))
            after,bits=canonical_advance(part,execution,phase_events,steps,1,event_begin=phase_begin,
                event_steps=phase_steps,event_times=phase_times,timed_calls=timed_calls,
                fused_idle=False,precision_epoch_begin=precision_epoch_begin,_expression_clock=phase_clock,
                _expression_draws=draws)
            if checked:after=_checked_merge(p,before,after,active)
            merged={key:mx.where(active if p.program.variables[key].scalar else mx.repeat(active,p.neurons),
                value,before.values[key])for key,value in after.values.items()}
            visible=after.spikes if phase=='thresholds'else before.spikes
            if checked and phase=='thresholds':visible=mx.where(mx.repeat(active,p.neurons),visible,0)
            state=replace(before,values=_pack_values(p.program,merged),release=after.release,
                check_index_error=after.check_index_error,
                spikes=visible)
        values.update(state.values)
        values['_cond']=state.spikes.astype(mx.bool_)
        if phase=='thresholds':threshold_completed(state)
        if phase=='synapses':synapses_completed=True
    if replay is not None:
        random=RandomState(random.stream,{name:values[name]for name in random.temporaries},
            random.errors,random.pool,random.buffers,error_metadata=random.error_metadata)
        return replace(state,random=random),state.spikes
    # Commit spike history, outgoing markers and the clock exactly once,
    # through the same packed-state engine used by nonrandom models.
    part=replace(p,program=_program(p.program,'thresholds',True),compact_idle=False,
        independent_neurons=False,components=None)
    working=replace(state,values=_rebind_values(part.program,state.values))
    after,bits=canonical_advance(part,working,events,steps,1,event_begin=event_begin,event_steps=event_steps,
        event_times=event_times,timed_calls=timed_calls,fused_idle=False,
        precision_epoch_begin=precision_epoch_begin,_expression_clock=clock_params)
    pool=random.pool
    if pool is not None:
        from .expression_runtime import MixedLookahead
        if isinstance(pool,MixedLookahead):pool=replace(pool,next_step=after.step)
    random=RandomState(random.stream,{name:values[name]for name in random.temporaries},
        random.errors,pool,random.buffers,error_metadata=random.error_metadata)
    return replace(after,random=random),bits


def _advance_steps(p,state,events,steps,length,*,event_begin,event_steps,event_times,timed_calls=None,
                   fused_idle=False,native_poisson=None,stop_before_synapses=False,
                   precision_epoch_begin=None,clock_params=None):
    combined=mx.zeros((p.neurons*state.random.stream.trials,),mx.uint32)
    from .expression_window import cython_batch_eligible
    from .configured import _state_layout,_uses_previous_events
    pipeline=(length>1 and clock_params is not None and precision_epoch_begin is None
        and not stop_before_synapses and not fused_idle and cython_batch_eligible(p)
        and 'reset' in p.program.random_program.stages
        and p.model.arrival_delay_steps>=2 and not p.independent_neurons and not p.scan_arrivals
        and p.outgoing_destinations.size<=p.neurons and p.neurons<2**31
        and not p.program.variables['v'].conditional_write
        and 2*len(_state_layout(p.program).groups)+24+len(p.input_arrays)+int(_uses_previous_events(p.model,p.program))<=31)
    if pipeline:
        pending=None;words=[]
        for local in range(length):
            clock=clock_params+mx.array([local,0,0],mx.int64)
            state,bits,pending=_advance_prefix(p,state,events,steps,event_begin=event_begin,event_steps=event_steps,
                event_times=event_times,timed_calls=timed_calls,clock_params=clock,native_poisson=native_poisson,
                _pending_reset=pending,_defer_reset=local+1<length,_pipeline=True)
            if local and local%32==0:
                words.append(combined);combined=mx.zeros_like(combined)
            combined=combined|(bits.astype(mx.uint32)<<(local%32))
        return state,mx.concatenate([*words,combined])if words else combined
    words=[]
    for local in range(length):
        active=mx.repeat(_active(p,state,state.random),p.neurons)
        clock=None if clock_params is None else clock_params+mx.array([local,0,0],mx.int64)
        state,bits=_advance_one(p,state,events,steps,1,event_begin=event_begin,event_steps=event_steps,
            event_times=event_times,timed_calls=timed_calls,fused_idle=fused_idle,
            native_poisson=native_poisson,stop_before_synapses=stop_before_synapses,
            precision_epoch_begin=precision_epoch_begin,clock_params=clock)
        if local and local%32==0:
            words.append(combined);combined=mx.zeros_like(combined)
        combined=combined|(mx.where(active,bits,0).astype(mx.uint32)<<(local%32))
        from .numpy_policy_context import stopped
        if state.policy_error is not None or stopped():
            break
    return state,mx.concatenate([*words,combined])if words else combined


def advance(p,state,events,steps,length,*,event_begin,event_steps,event_times,timed_calls,
            fused_idle,native_poisson,stop_before_synapses,precision_epoch_begin,clock_params=None):
    if not stop_before_synapses and precision_epoch_begin is None and clock_params is None and p.program.replay_phase is None:
        from .expression_window import advance as window_advance
        result=window_advance(p,state,events,steps,length,event_begin=event_begin,
            event_steps=event_steps,event_times=event_times,native_poisson=native_poisson)
        if result is not None:return result
    return _advance_steps(p,state,events,steps,length,event_begin=event_begin,event_steps=event_steps,
        event_times=event_times,timed_calls=timed_calls,fused_idle=fused_idle,
        native_poisson=native_poisson,stop_before_synapses=stop_before_synapses,
        precision_epoch_begin=precision_epoch_begin,clock_params=clock_params)
