"""Bounded speculative MT prefixes shared by mixed uniform/normal phases.

The cursor and exhaustion decisions stay on Metal. A pool contains an exact
prefix of the incoming stream, never a separately seeded stream. Unconsumed
values are invisible, and the existing native samplers finish exhausted rows.
"""
from dataclasses import dataclass,replace
from operator import index
from .expression_rng import MT19937State


@dataclass(frozen=True)
class PrefixData:
    values: object
    words: object
    positions: object
    width: int
    frames: int
    normal: object | None = None


@dataclass(frozen=True)
class StreamPrefix:
    data: object
    initial: MT19937State
    cursor: object
    stream: MT19937State

    def arrays(self):
        return (self.data.values,self.data.words,self.data.positions,
                self.initial.words,self.initial.position,self.initial.has_gauss,
                self.initial.gauss,self.cursor)

    @property
    def capacity(self):return self.data.width*self.data.frames


@dataclass(frozen=True)
class UniformPrefixView:
    """Inputs for an existing model kernel to select prefix or native tail."""
    values: object
    tail: object
    cursor: object
    done: object
    counts: object | None = None


def prefix_pool(state,capacity,*,checkpoint=2496,parts=None):
    import mlx.core as mx
    from .expression_rng_jump import uniform_pool
    capacity=index(capacity);checkpoint=index(checkpoint)
    if capacity<1 or checkpoint<1:raise ValueError('Invalid mixed-stream prefix capacity')
    frames=(capacity+checkpoint-1)//checkpoint
    if parts is None:parts=32 if capacity>=2**20 else 1
    pool=uniform_pool(state,checkpoint,frames,parts=parts)
    # Keep one trial-major representation for all later phases. Rebuilding
    # this layout on every draw would copy the whole pool for batched trials.
    values=mx.contiguous(pool.values.transpose(1,0,2)).reshape(state.trials,-1)
    data=PrefixData(values,pool.words,pool.positions,checkpoint,frames)
    return StreamPrefix(data,state,mx.zeros((state.trials,),mx.uint32),state)


def _counts(state,maximum,counts,cache):
    import mlx.core as mx
    if cache.stream is not state:raise ValueError('Mixed prefix does not belong to this stream checkpoint')
    maximum=index(maximum)
    if not 0<=maximum<2**32 or maximum*state.trials>=2**32:
        raise ValueError('Mixed prefix draw capacity exceeds safe Metal indexing')
    if counts is None:counts=mx.full((state.trials,),maximum,mx.uint32)
    if counts.shape!=(state.trials,)or counts.dtype!=mx.uint32:
        raise ValueError('Mixed prefix counts must be a uint32 vector per stream')
    return maximum,counts


def _raw(cache,width):
    import mlx.core as mx
    offsets=mx.minimum(cache.cursor[:,None].astype(mx.uint64)+mx.arange(width,dtype=mx.uint64),cache.capacity-1)
    return mx.take_along_axis(cache.data.values,offsets,axis=1)


_RESTORE_INPUT_NAMES=('prefix_initial_words','prefix_initial_position','prefix_words',
    'prefix_position','prefix_cursor','prefix_consumed')


def _restore_source(lanes):
    """Initialize the unchanged native sampler directly at a selected prefix."""
    return r'''
        {
        bool active=prefix_cursor[trial]<PREFIX_WIDTH*PREFIX_FRAMES;
        uint block=prefix_consumed[trial]/PREFIX_WIDTH;
        if(!active) {
            pos=position[trial];
            for(uint j=lane;j<624;j+=LANES)mt[j]=state[trial*624+j];
        } else if(block) {
            pos=prefix_position[trial*PREFIX_FRAMES+block-1];
            for(uint j=lane;j<624;j+=LANES)mt[j]=prefix_words[(trial*PREFIX_FRAMES+block-1)*624+j];
        } else {
            pos=prefix_initial_position[trial];
            for(uint j=lane;j<624;j+=LANES)mt[j]=prefix_initial_words[trial*624+j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Uniforms consume exactly two tempered MT words. Tempering and
        // conversion do not affect the state, so only twists need execution.
        uint left=active?2u*(prefix_consumed[trial]%PREFIX_WIDTH):0u;
        while(left) {
            if(pos==624){mt_twist(mt,lane);pos=0;}
            uint take=min(left,624u-pos);left-=take;pos+=take;
        }
        }
        '''.replace('LANES',str(lanes))


def _restore_arguments(cache,consumed):
    return ([cache.initial.words,cache.initial.position,cache.data.words,cache.data.positions,
        cache.cursor,consumed],[('PREFIX_WIDTH',cache.data.width),('PREFIX_FRAMES',cache.data.frames)])


def _merge(packed,tail,done,counts,maximum):
    import mlx.core as mx
    rows=mx.arange(maximum,dtype=mx.uint32)[None,:]
    offsets=mx.maximum(rows.astype(mx.int64)-done[:,None],0)
    result=mx.where(rows<done[:,None],packed,mx.take_along_axis(tail,offsets,axis=1))
    return mx.where(rows<counts[:,None],result,0)


def prefix_uniforms(state,cache,maximum,counts=None,*,_view=False):
    import mlx.core as mx
    from .expression_rng import uniforms
    maximum,counts=_counts(state,maximum,counts,cache)
    if not maximum:return mx.zeros((state.trials,0),mx.uint64),state,cache
    available=cache.capacity-cache.cursor
    done=mx.minimum(counts,available).astype(mx.uint32)
    cursor=cache.cursor+done
    # Only the consumed fallback prefix is selected by _merge. That merge
    # supplies the public zero padding, so do not clear a discarded tail.
    tail,end=uniforms(state,maximum,(counts-done).astype(mx.uint32),_clear_unused=False,_prefix=(cache,cursor))
    values=(UniformPrefixView(cache.data.values,tail,cache.cursor,done,counts)if _view else
        _merge(_raw(cache,maximum),tail,done,counts,maximum))
    return values,end,replace(cache,cursor=cursor,stream=end)


def prefix_normals(state,cache,maximum,counts=None,*,target='numpy',candidate_count=None):
    import mlx.core as mx
    from .expression_rng import _serial_normals,_normal_target
    from .expression_rng_parallel import _normal_candidates,_normal_pack
    maximum,counts=_counts(state,maximum,counts,cache)
    cpp=_normal_target(target)
    if maximum and cache.data.normal is not None and not cpp:
        from .expression_normal_prefix import draw
        return draw(state,cache,maximum,counts)
    if not maximum:return mx.zeros((state.trials,0),mx.uint64),state,cache
    candidates=3*((maximum+1)//2)//2+128 if candidate_count is None else index(candidate_count)
    if candidates<1 or 2*candidates*state.trials>=2**32:
        raise ValueError('Invalid mixed-prefix normal candidate capacity')
    cached=mx.logical_and(state.has_gauss>0,counts>0).astype(mx.uint32)
    remaining=counts-cached
    limits=mx.minimum(candidates,mx.minimum((cache.capacity-cache.cursor)//2,
        mx.where(remaining>0,3*((remaining+1)//2)//2+128,0))).astype(mx.uint32)
    xs,ys,radii,valid=_normal_candidates()(
        inputs=[_raw(cache,2*candidates).reshape(-1),limits],
        template=[('B',state.trials),('CANDIDATES',candidates),('CPP_NORMAL',cpp)],
        grid=(state.trials*candidates,1,1),threadgroup=(256,1,1),
        output_shapes=[(state.trials*candidates,)]*4,
        output_dtypes=[mx.uint64,mx.uint64,mx.uint64,mx.uint32])
    prefix=mx.cumsum(valid.reshape(state.trials,candidates),axis=1)
    packed,_,last_consumed,saved=_normal_pack()(inputs=[xs,ys,radii,valid,prefix.reshape(-1),counts,state.has_gauss,state.gauss],
        template=[('B',state.trials),('CANDIDATES',candidates),('MAX_COUNT',maximum),('KEEP_SECONDS',False)],
        grid=(state.trials*candidates,1,1),threadgroup=(256,1,1),
        output_shapes=[(state.trials*maximum,),(1,),(state.trials,),(state.trials,)],
        output_dtypes=[mx.uint64,mx.uint64,mx.uint32,mx.uint64])
    done=cached+mx.minimum(remaining,2*prefix[:,-1])
    pairs=(remaining+1)//2;enough=pairs<=prefix[:,-1]
    consumed=mx.where(remaining==0,0,mx.where(enough,last_consumed,2*limits)).astype(mx.uint32)
    cursor=cache.cursor+consumed
    flag=mx.where(counts==0,state.has_gauss,(remaining%2)*enough).astype(mx.uint32)
    gauss=mx.where(remaining==0,state.gauss,saved)if cpp else mx.where(flag>0,saved,0)
    gauss=mx.where(counts==0,state.gauss,gauss)
    end=MT19937State(state.words,state.position,state.trials,flag,gauss)
    tail,end=_serial_normals(end,maximum,(counts-done).astype(mx.uint32),target=target,
        _clear_unused=False,_prefix=(cache,cursor))
    values=_merge(packed.reshape(state.trials,maximum),tail,done,counts,maximum)
    # An exhausted candidate budget can leave one unused uniform. Once the
    # native normal fallback has consumed beyond that point, invalidate this
    # row's prefix rather than reusing that stale value on the next call.
    cursor=mx.where(done<counts,cache.capacity,cursor).astype(mx.uint32)
    return values,end,replace(cache,cursor=cursor,stream=end)
