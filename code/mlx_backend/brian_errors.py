"""Native replay of a failed Cython operator run at Brian's phase boundaries.

The regular window kernel stays fused. Only after a device error is observed,
replay the run on Metal one phase at a time, retaining the completed prefix of
Brian's ascending neuron loop. No CPU neuron or synapse code object executes.
"""
from dataclasses import dataclass,replace
from functools import lru_cache
import ctypes

import mlx.core as mx
import numpy as np

from . import configured


class RandomSnapshot:
    """Copy Brian's actual buffer contents; Device.get_random_state holds pointers."""
    def __init__(self,device):
        self.device=device
        self.numpy=np.random.get_state()
        self.buffers={}
        for name in ('rand','randn'):
            pointer=int(getattr(device,name+'_buffer')[0])
            values=(np.ctypeslib.as_array(ctypes.cast(pointer,ctypes.POINTER(ctypes.c_double)),
                shape=(20000,)).copy() if pointer else None)
            self.buffers[name]=(int(getattr(device,name+'_buffer_index')[0]),values)

    def restore(self):
        np.random.set_state(self.numpy)
        libc=ctypes.CDLL(None)
        libc.free.argtypes=[ctypes.c_void_p];libc.free.restype=None
        libc.malloc.argtypes=[ctypes.c_size_t];libc.malloc.restype=ctypes.c_void_p
        for name,(index,values) in self.buffers.items():
            address=getattr(self.device,name+'_buffer')
            pointer=int(address[0])
            if values is None:
                if pointer:libc.free(pointer)
                address[0]=0
            else:
                if not pointer:
                    pointer=libc.malloc(values.nbytes)
                    if not pointer:raise MemoryError('Unable to restore Brian random buffer')
                    address[0]=pointer
                np.ctypeslib.as_array(ctypes.cast(pointer,ctypes.POINTER(ctypes.c_double)),
                    shape=(20000,))[:]=values
            getattr(self.device,name+'_buffer_index')[:]=index


@dataclass
class FailureReplay:
    state: object
    indices: np.ndarray
    steps: np.ndarray
    group_spikes: np.ndarray
    stage: str
    kind: int
    source: object = None
    cause: Exception = None


@lru_cache(maxsize=16)
def _programs(program):
    return {phase:replace(program,replay_phase=phase,equilibrium=None,
        refractory_elapsed_threshold=None) for phase in (('subexpression',) if program.scalar_stages else ())+('groups','thresholds','synapses','resets')}


@dataclass(frozen=True)
class _ReplayModel:
    model: object

    def __getattr__(self,name):return getattr(self.model,name)

    @property
    def reset_uses_previous_spikes(self):return True

    @property
    def arrival_delay_steps(self):
        # A single synapses-only kernel reads its staged eventspace below;
        # avoiding the regular zero-delay threshold/synapse split executes
        # exactly one native phase, without running a dummy threshold.
        return max(1,self.model.arrival_delay_steps)


def _without_arrivals(prepared,program):
    return replace(prepared,program=program,targets=(),
        incoming_offsets=mx.zeros((prepared.neurons+1,),mx.int32),
        outgoing_offsets=mx.zeros((prepared.neurons+1,),mx.int32),
        channel_offsets=mx.zeros((prepared.neurons+1,),mx.int32))


def _shift_queue_back(history,window):
    result={}
    for key,value in history.items():
        high=value>>1
        carry=(value&1)<<(window-1)
        result[key]=result[key]|high if key in result else high
        result[key-1]=result[key-1]|carry if key-1 in result else carry
    return result


def _rescale_words(words,old,new):
    if old==new:return dict(words)
    result={}
    for position,word in words.items():
        for bit in range(old):
            coordinate=position*old+bit
            target,shift=divmod(coordinate,new)
            value=((word>>bit)&1)<<shift
            result[target]=result[target]|value if target in result else value
    return result


def _enqueue(prepared,state,visible,step):
    model=prepared.model.model
    lag=int(model.execution_phases.index('synapses')<model.execution_phases.index('thresholds'))
    history=dict(state.history);history[step-lag]=visible
    due=dict(state.due)
    masks=configured._mark_kernel()(inputs=[prepared.outgoing_offsets,prepared.outgoing_destinations,visible],
        template=[('N',prepared.neurons),('B',1),('W',1),('SHIFT',0)],
        grid=(prepared.neurons,1,1),threadgroup=(min(256,prepared.neurons),1,1),
        output_shapes=[(prepared.neurons,)]*2,output_dtypes=[mx.uint32]*2,init_value=0)
    delivery=step+model.delay_steps
    due[delivery]=due[delivery]|masks[0] if delivery in due else masks[0]
    return replace(state,history=history,due=due)


def replay_failure(prepared,initial,end_step,*,inputs,clock,device,rng_snapshot,group_spikes):
    """Materialize the first failed phase and its committed CPU loop prefix."""
    from .brian_device import _PoissonSampler
    from .expression_errors import exception as expression_exception
    if rng_snapshot is not None:rng_snapshot.restore()
    expression_rng=prepared.program.random_program is not None
    if expression_rng and inputs:
        from .expression_poisson_input import ExpressionPoisson
        sampler=ExpressionPoisson(inputs,clock,prepared.model.random_target)
    else:sampler=_PoissonSampler(inputs,clock) if inputs else None
    size=prepared.neurons
    programs=_programs(prepared.program)
    base=replace(prepared,model=_ReplayModel(prepared.model),window=1,compact_idle=False,independent_neurons=False)
    phases={phase:(_without_arrivals(base,program) if phase!='synapses' else replace(base,program=program))
            for phase,program in programs.items()}
    state=replace(initial,history=_rescale_words(initial.history,prepared.window,1),
        due=_rescale_words(initial.due,prepared.window,1),release=mx.zeros((size,),mx.int64),check_index_error=False)
    indices,steps=[],[]
    current_spikes=np.asarray(group_spikes,np.int32).copy()
    state=replace(state,spikes=mx.array(np.isin(np.arange(size),current_spikes).astype(np.uint32)))
    zero=mx.zeros((max(1,len(inputs)),),mx.uint8)
    while state.step<end_step:
        step=state.step
        synapses_completed=False
        times=mx.array(np.array([step*prepared.model.dt_s],np.float64).view(np.uint64))
        for phase in (('subexpression',) if prepared.program.scalar_stages else ())+prepared.model.execution_phases:
            part=phases[phase]
            before=state
            events=zero
            execution=state
            input_failure=None
            if phase=='synapses':
                state=_enqueue(part,state,state.spikes,step)
                execution=state
                if prepared.model.arrival_delay_steps==0:
                    execution=replace(state,history={**state.history,step-1:state.spikes})
                if inputs:
                    if expression_rng:
                        events,random=sampler.draw(part,state,state.random,mx.ones((1,),mx.bool_))
                        state=replace(state,random=random)
                        execution=replace(execution,random=random)
                        input_failure=sampler.failure
                    condition=prepared.program.variables['v'].conditional_write
                    flags=(np.asarray(state.values[condition],bool) if condition else np.ones(size,bool))
                    eligible=np.array([source.active and flags[int(source._group.start)] for source in inputs],bool)
                    if expression_rng:pass
                    elif prepared.model.reference_codegen_target == 'numpy':
                        draws,input_failure=sampler.draw_prefix(eligible)
                        events=mx.array(draws)
                    else:
                        events=mx.array(sampler.draw(eligible))
            after,bits=configured.advance(part,execution,events,end_step,1,event_begin=step,event_steps=1,event_times=times)
            mx.eval(*configured._state_buffers(after.values),after.release,bits)
            errors=np.asarray(after.release)<0
            expression_kind=(int(np.asarray(after.random.errors)[0])if expression_rng and after.random is not None else 0)
            expression_fault=bool(expression_kind and not np.any(errors))
            if np.any(errors) or expression_fault:
                if expression_fault:row,kind=(0 if prepared.model.reference_codegen_target=='numpy'else size),expression_kind
                elif prepared.program.runtime_error_type == 'model':
                    from .runtime_errors import first_error
                    row,kind=first_error(after.release,prepared.program.runtime_error_order)
                else:
                    row=int(np.flatnonzero(errors)[0])
                    kind=int(np.asarray(after.release)[row])&3
                numpy = prepared.model.reference_codegen_target == 'numpy'
                if phase=='thresholds' and not numpy:
                    # Cython writes spike indices while scanning, but stores
                    # the new spike count only after its complete loop. An
                    # error leaves the old count and untouched old tail visible.
                    partial=np.flatnonzero(np.asarray(bits)[:row]&1).astype(np.int32)
                    count=min(len(current_spikes),len(partial))
                    current_spikes[:count]=partial[:count]
                # NumPy copies read/write neuron arrays (or indexed reset
                # locals), then stores them after its complete vector block.
                # Expression faults commit no neuron arrays. A scalar-store
                # fault can follow earlier whole-array epilogue writes.
                selected=mx.arange(size)<(0 if numpy and phase in ('subexpression','groups','resets','thresholds') else row)
                committed=(dict(prepared.program.runtime_error_writes).get(kind,()) if numpy else ())
                merged={name:(after.values[name] if name in committed or prepared.program.variables[name].scalar else
                    mx.where(selected,after.values[name],before.values[name])) for name in before.values}
                if expression_fault and not numpy:merged=dict(after.values)
                history,due=state.history,state.due
                if synapses_completed:
                    # Brian's synaptic queue advanced even though its clock did
                    # not tick. Shift queue coordinates for a resumed Network.run.
                    history=_shift_queue_back(history,1)
                    due=_shift_queue_back(due,1)
                visible=mx.array(np.isin(np.arange(size),current_spikes).astype(np.uint32))
                if (prepared.model.execution_phases.index('synapses')
                        < prepared.model.execution_phases.index('thresholds')):
                    # A failed run does not execute threshold.after_run, so
                    # its current eventspace survives. On resume Brian pushes
                    # it at synapses before any new threshold. Stage that
                    # pending push in native coordinates without applying it.
                    pending=_enqueue(phases['synapses'],replace(before,history=history,due=due),visible,step)
                    history,due=pending.history,pending.due
                random=after.random
                if expression_fault:random=replace(random,errors=mx.zeros_like(random.errors))
                state=replace(before,values=configured._pack_values(prepared.program,merged),random=random,
                    history=_rescale_words(history,1,prepared.window),due=_rescale_words(due,1,prepared.window),
                    spikes=visible,
                    step=step,release=mx.zeros((size,),mx.int64),check_index_error=False)
                configured.evaluate_state(state)
                return FailureReplay(state,np.concatenate(indices) if indices else np.empty(0,np.int32),
                    np.concatenate(steps) if steps else np.empty(0,np.int64),current_spikes,
                    {'subexpression':'subexpression','groups':'state','thresholds':'threshold','synapses':'synapses','resets':'reset'}[phase],kind,
                    cause=expression_exception(kind,None if after.random.error_metadata is None else np.asarray(after.random.error_metadata)[0])if expression_fault else None)
            if input_failure is not None:
                # Recurrent arrivals and all earlier input channels completed;
                # the synaptic queue advanced although the clock did not tick.
                history={q:word for q,word in state.history.items()
                         if q>=step-prepared.model.arrival_delay_steps+1}
                due={q:word for q,word in state.due.items() if q>step}
                state=replace(after,step=step,
                    history=_rescale_words(_shift_queue_back(history,1),1,prepared.window),
                    due=_rescale_words(_shift_queue_back(due,1),1,prepared.window),
                    spikes=state.spikes,release=mx.zeros((size,),mx.int64),check_index_error=False)
                configured.evaluate_state(state)
                source,cause=input_failure
                return FailureReplay(state,np.concatenate(indices) if indices else np.empty(0,np.int32),
                    np.concatenate(steps) if steps else np.empty(0,np.int64),current_spikes,
                    'poisson',0,source,cause)
            if phase=='thresholds':
                current_spikes=np.flatnonzero(np.asarray(bits)&1).astype(np.int32)
                if len(current_spikes):
                    indices.append(current_spikes)
                    steps.append(np.full(len(current_spikes),step,np.int64))
            if phase=='synapses':
                synapses_completed=True
                state=replace(state,history={q:word for q,word in state.history.items()
                    if q>=step-prepared.model.arrival_delay_steps+1},
                    due={q:word for q,word in state.due.items() if q>step})
            # Phase kernels share one Brian clock and the same pending arrivals.
            state=replace(after,step=step,history=state.history,due=state.due,
                spikes=bits if phase=='thresholds' else state.spikes,
                release=mx.zeros((size,),mx.int64),check_index_error=False)
        state=replace(state,step=step+1)
    raise RuntimeError('The native error did not reproduce during phase replay')
