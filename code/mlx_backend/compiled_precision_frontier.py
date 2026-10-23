"""Replay checked native temporaries at existing evaluated commit frontiers.

Advance stays lazy. Only callers already collecting/evaluating a bounded block
invoke this object; immutable states, inputs and RNG cursors belong to attempts.
"""
from dataclasses import dataclass,replace
import numpy as np
from .compiled_precision import ERROR_NAME,PrecisionPolicy

# Four schedule phases fit below the signed release carrier's47 epoch bits.
# Relative clocks permit continuation beyond any absolute epoch encoding range.
FRONTIER_STEP_LIMIT=1<<44


def rebind(prepared,limbs,transform=None):
    """Change only temporary precision; persistent buffers must remain compatible."""
    from . import configured
    old=prepared.program
    if not old.native_precision_limbs:
        raise ValueError('This model has no checked native temporary specialization')
    new=configured._compiled_program(prepared.model,neurons=prepared.neurons,native_precision_limbs=limbs,
                                    shared_expression_inputs=old.shared_expression_inputs)
    if old.replay_phase is not None:new=replace(new,replay_phase=old.replay_phase)
    if configured._state_layout(old)!=configured._state_layout(new):
        raise RuntimeError('Native precision changed the persistent storage layout')
    if old.runtime_errors!=new.runtime_errors:
        raise RuntimeError('Native precision changed the model error catalog')
    a,b=old.input_arrays,new.input_arrays
    if tuple(a)!=tuple(b) or any(a[name].dtype!=b[name].dtype or
            a[name].values.shape!=b[name].values.shape or
            a[name].values.tobytes()!=b[name].values.tobytes() for name in a):
        raise RuntimeError('Native precision changed immutable namespace inputs')
    result=replace(prepared,program=new)
    return transform(result) if transform is not None else result


@dataclass(frozen=True)
class FrontierResult:
    prepared: object
    state: object
    windows: tuple
    error_kind: int | None = None


class PrecisionFrontier:
    """Own a bounded sequence of native attempts, committing only after evaluation."""
    def __init__(self,prepared,state,*,transform=None,sampler=None,rng_snapshot=None):
        if not prepared.program.native_precision_limbs:
            raise ValueError('Precision frontiers are only for checked temporary models')
        self.prepared=prepared;self.state=state;self.transform=transform
        self.sampler=sampler;self.rng_snapshot=rng_snapshot
        self.calls=[];self.windows=[];self.checkpoint=state;self.begin=state.step
        self.mutable={};self.random=None;self.error_kind=None;self.retries=0

    def _start(self):
        self.checkpoint=self.state;self.begin=self.state.step
        self.mutable={name:getattr(self.sampler,name) for name in ('used','overflow')
                      if self.sampler is not None and getattr(self.sampler,name,None) is not None}
        self.random=self.rng_snapshot() if self.rng_snapshot is not None else None

    def _restore(self):
        for name,value in self.mutable.items():setattr(self.sampler,name,value)
        if self.random is not None:self.random.restore()

    def submit(self,operation,length):
        """Operation(p,state,relative_epoch_origin) builds lazy native work."""
        if self.error_kind is not None:raise RuntimeError('Cannot advance a failed precision frontier')
        if not self.calls:self._start()
        if not 1<=length or not 0<=self.state.step-self.begin<FRONTIER_STEP_LIMIT-length:
            raise ValueError('Native precision frontier exceeds its proven relative clock range')
        before=self.state.step
        self.state,bits=operation(self.prepared,self.state,self.begin)
        if self.state.step!=before+length:
            raise RuntimeError('Native precision attempt changed its scheduled boundary')
        self.calls.append((operation,length));self.windows.append((before,length,bits))
        return self.state,bits

    def evaluate(self):
        """Evaluate/retry at the caller's existing synchronization boundary."""
        import mlx.core as mx
        from . import configured
        if not self.calls:
            return FrontierResult(self.prepared,self.state,(),self.error_kind)
        while True:
            buffers=[*configured._state_buffers(self.state.values),self.state.release,
                     *self.state.history.values(),*self.state.due.values(),
                     *(word for _,_,word in self.windows)]
            if self.state.spikes is not None:buffers.append(self.state.spikes)
            from .expression_runtime import arrays as random_arrays
            buffers.extend(random_arrays(self.state.random))
            buffers.extend(getattr(self.sampler,name) for name in self.mutable)
            mx.eval(*buffers)
            # This replaces the existing model-error scalar status read. Normal
            # frontiers never transfer the neuron release array to the host.
            if not bool(mx.any(self.state.release<0).item()):
                self.state=replace(self.state,check_index_error=False)
                self.error_kind=None
                break
            release=np.asarray(self.state.release)
            rows=np.flatnonzero(release<0)
            unsigned=release.view(np.uint64)
            epochs=(unsigned&np.uint64(0x7fffffffffffffff))>>np.uint64(16)
            first=int(min(rows,key=lambda row:(int(epochs[row]),int(row))))
            kind=int(unsigned[first]&np.uint64(65535))
            catalog=self.prepared.program.runtime_errors
            # Each trial is a separate CPU execution. An earlier user failure
            # suppresses later capacity requests within that trial, but cannot
            # leave another independent trial's valid computation truncated.
            trial_first={}
            for row in sorted(rows,key=lambda row:(int(epochs[row]),int(row))):
                trial_first.setdefault(int(row)//self.prepared.neurons,int(row))
            retry=False
            for row in trial_first.values():
                trial_kind=int(unsigned[row]&np.uint64(65535))
                if trial_kind>len(catalog):
                    from .runtime_errors import model_exception
                    raise model_exception(catalog,trial_kind)
                retry|=trial_kind>0 and catalog[trial_kind-1][0]==ERROR_NAME
            if not retry:
                self.error_kind=kind;self._restore();break
            capacity=PrecisionPolicy(self.prepared.program.native_precision_limbs).grow().limbs
            self.prepared=rebind(self.prepared,capacity,self.transform)
            self._restore();self.state=self.checkpoint;self.windows=[];self.retries+=1
            for operation,length in self.calls:
                before=self.state.step
                self.state,bits=operation(self.prepared,self.state,self.begin)
                self.windows.append((before,length,bits))
        result=FrontierResult(self.prepared,self.state,tuple(self.windows),self.error_kind)
        self.calls=[];self.windows=[]
        return result
