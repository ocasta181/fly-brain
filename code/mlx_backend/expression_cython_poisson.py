"""Buffered Cython Poisson prefixes and physical refill checkpoints."""
from dataclasses import dataclass,replace
from operator import index
import math
import numpy as np
from .expression_cython import BUFFER_SIZE,CythonBuffers,buffered_draws,buffered_pool,poisson


@dataclass(frozen=True)
class PoissonPool:
    values: object
    words: object
    positions: object
    uniform: object
    indices: object
    initial: object
    buffers: object
    width: int
    frames: int
    trials: int
    errors: object | None = None

    def arrays(self):
        s=self.initial;b=self.buffers
        return (self.values,self.words,self.positions,self.uniform,self.indices,
            s.has_gauss,s.gauss,b.normal,b.normal_index,*(()if self.errors is None else(self.errors,)))

    def state_at(self,frame):
        from .expression_rng import MT19937State
        frame=index(frame)
        if not 0<=frame<self.frames:raise IndexError('Cython Poisson checkpoint out of range')
        return (MT19937State(self.words[:,frame,:].reshape(-1),self.positions[:,frame],self.trials,
            self.initial.has_gauss,self.initial.gauss),CythonBuffers(self.uniform[:,frame,:],
            self.indices[:,frame],self.buffers.normal,self.buffers.normal_index))


def constant_poisson(state,buffers,maximum,lam,counts=None,*,raw_capacity=None,frame_width=None,errors=None):
    import mlx.core as mx
    from .binary64 import literal
    from .expression_poisson_parallel import _product_prefix,_ptrs_prefix
    maximum=index(maximum);lam=float(lam);trials=state.trials
    if maximum<0 or maximum*trials>=2**32:raise ValueError('Invalid buffered Poisson capacity')
    if counts is None:counts=mx.full((trials,),maximum,mx.uint32)
    if counts.shape!=(trials,)or counts.dtype!=mx.uint32:raise ValueError('Invalid buffered Poisson counts')
    if frame_width is not None:
        frame_width=index(frame_width)
        if frame_width<1 or maximum%frame_width:raise ValueError('Invalid buffered Poisson frame width')
    if errors is None:errors=mx.zeros((trials,),mx.uint32)
    counts=mx.where(errors==0,counts,0).astype(mx.uint32)
    # MLX's Python scalar conversion rejects unsigned IEEE bit patterns above
    # INT64_MAX. A typed one-element array also preserves negative/nonfinite
    # accepted compiled-target rates without a signed intermediate.
    rate=mx.broadcast_to(mx.array(np.asarray(lam,np.float64).view(np.uint64)),(trials,maximum))
    if not maximum:return mx.zeros((trials,0),mx.int32),state,buffers,mx.zeros((trials,),mx.uint32)
    if not math.isfinite(lam):
        if frame_width is not None:raise ValueError('This buffered prefix requires a finite rate below10')
        return poisson(state,buffers,rate,counts,errors=errors)
    if lam<0:
        if frame_width is not None:
            pool=buffered_pool(state,buffers,'rand',frame_width,maximum//frame_width)
            return replace(pool,values=mx.zeros(pool.values.shape,mx.int32))
        _,end,cache=buffered_draws(state,buffers,'rand',maximum,counts)
        return mx.zeros((trials,maximum),mx.int32),end,cache,mx.zeros((trials,),mx.uint32)
    if not lam:
        if frame_width is None:return mx.zeros((trials,maximum),mx.int32),state,buffers,mx.zeros((trials,),mx.uint32)
        frames=maximum//frame_width
        return PoissonPool(mx.zeros((frames,trials,frame_width),mx.int32),
            mx.broadcast_to(state.words.reshape(trials,1,624),(trials,frames,624)),
            mx.broadcast_to(state.position[:,None],(trials,frames)),
            mx.broadcast_to(buffers.uniform[:,None,:],(trials,frames,BUFFER_SIZE)),
            mx.broadcast_to(buffers.uniform_index[:,None],(trials,frames)),state,buffers,frame_width,frames,trials)
    raw=(int(math.ceil(maximum*(1.5*(lam+1)if lam<10 else 3)))+BUFFER_SIZE if raw_capacity is None else index(raw_capacity))
    if raw<1:raise ValueError('Buffered Poisson needs a positive raw capacity')
    raw=min(raw,64*1024**2//trials);refills=(raw+BUFFER_SIZE-1)//BUFFER_SIZE;raw=refills*BUFFER_SIZE
    if raw*trials>=2**32:raise ValueError('Buffered raw pool exceeds safe Metal indexing')
    pool=buffered_pool(state,buffers,'rand',BUFFER_SIZE,refills)
    uniforms=mx.contiguous(pool.values.transpose(1,0,2)).reshape(-1)
    if lam<10:
        packed,consumed,done,ends=_product_prefix(uniforms,counts,maximum,lam,raw,trials,
            keep_ends=frame_width is not None)
    else:
        packed,consumed,done,ends=_ptrs_prefix(uniforms,counts,maximum,lam,raw,trials,
            keep_ends=frame_width is not None,target='cython')
    start,cache=pool.states_at(consumed[:,None])
    tail,end,cache,error=poisson(start,cache,rate,(counts-done).astype(mx.uint32),errors=errors)
    offset=mx.maximum(mx.arange(maximum)[None,:]-done[:,None],0)
    values=mx.where(mx.arange(maximum)[None,:]<done[:,None],packed,
        mx.take_along_axis(tail,offset,axis=1))
    values=mx.where(mx.arange(maximum)[None,:]<counts[:,None],values,0).astype(mx.int32)
    if frame_width is None:return values,end,cache,error
    frames=maximum//frame_width
    wanted=mx.minimum(counts[:,None],mx.arange(1,frames+1,dtype=mx.uint32)[None,:]*frame_width)
    effective=mx.minimum(wanted,done[:,None]);ends=ends.reshape(trials,maximum)
    consumed=mx.take_along_axis(ends,mx.maximum(effective,1)-1,axis=1)
    consumed=mx.where(effective>0,consumed,0).astype(mx.uint32)
    start,cache=pool.states_at(consumed)
    initial_errors=mx.broadcast_to(errors[:,None],(trials,frames)).reshape(-1)
    _,end,cache,error=poisson(start,cache,mx.array([int(literal(lam)[:-2])],mx.uint64),
        (wanted-effective).reshape(-1).astype(mx.uint32),scalar=True,state_only=True,errors=initial_errors)
    return PoissonPool(mx.contiguous(values.reshape(trials,frames,frame_width).transpose(1,0,2)),
        end.words.reshape(trials,frames,624),end.position.reshape(trials,frames),
        cache.uniform.reshape(trials,frames,BUFFER_SIZE),cache.uniform_index.reshape(trials,frames),
        state,buffers,frame_width,frames,trials,error.reshape(trials,frames)if lam>=10 else None)


def poisson_pool(state,buffers,width,frames,lam,*,raw_capacity=None,errors=None):
    width=index(width);frames=index(frames)
    if width<1 or frames<1 or frames*state.trials*BUFFER_SIZE>=2**32:
        raise ValueError('Invalid buffered Poisson checkpoint layout')
    return constant_poisson(state,buffers,width*frames,lam,raw_capacity=raw_capacity,frame_width=width,errors=errors)
