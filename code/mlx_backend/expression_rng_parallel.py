"""Parallel rejection sampling over an exact prefix of an existing MT stream.

Speculative uniforms never become observable stream consumption. Device-side
prefix counts identify the last accepted candidate, and MT checkpoints restore
that exact prefix before continuing. A bounded candidate pool may run out;
the ordinary native sampler then completes the remaining draws on the device.
"""
from functools import lru_cache
from dataclasses import dataclass
from operator import index
from .expression_rng import MT19937State


@lru_cache(maxsize=1)
def _normal_candidates():
    import mlx.core as mx
    from .binary64 import MSL_HEADER
    from .expression_rng_math import NORMAL_MATH_HEADER
    return mx.fast.metal_kernel(name='expression_normal_candidates',
        input_names=['uniform','limits'],output_names=['xs','ys','radii','valid'],
        header=MSL_HEADER+NORMAL_MATH_HEADER,source=r'''
        uint j=thread_position_in_grid.x;if(j>=B*CANDIDATES)return;
        if(j%CANDIDATES>=limits[j/CANDIDATES]){
            xs[j]=0ul;ys[j]=0ul;radii[j]=0ul;valid[j]=0u;return;
        }
        const ulong one=0x3ff0000000000000ul,two=0x4000000000000000ul;
        ulong x=d_sub(d_mul(two,uniform[2*j]),one);
        ulong y=d_sub(d_mul(two,uniform[2*j+1]),one);
        ulong rr=CPP_NORMAL ?expr_fma(y,y,d_mul(x,x)):d_add(d_mul(x,x),d_mul(y,y));
        bool accepted=d_lt(rr,one)&&!d_eq(rr,0ul);
        xs[j]=x;ys[j]=y;radii[j]=rr;valid[j]=uint(accepted);
        ''',compile_options={'math_mode':'safe'})


@lru_cache(maxsize=1)
def _normal_pack():
    import mlx.core as mx
    from .binary64 import MSL_HEADER
    from .expression_rng_math import NORMAL_MATH_HEADER
    return mx.fast.metal_kernel(name='expression_normal_pack',
        input_names=['xs','ys','radii','valid','prefix','counts','cached','saved'],
        output_names=['values','seconds','last_consumed','last_second'],header=MSL_HEADER+NORMAL_MATH_HEADER,source=r'''
        uint j=thread_position_in_grid.x;if(j>=B*CANDIDATES)return;
        uint trial=j/CANDIDATES,local=j%CANDIDATES;
        uint count=counts[trial],cache=uint(cached[trial]&&count);
        if(local==0&&cache)values[trial*MAX_COUNT]=saved[trial];
        // No accepted pair writes a witness for zero/cached-only counts or
        // an exhausted candidate prefix. Initialize just those scalar cells;
        // otherwise the unique final accepted pair owns both stores below.
        if(local==0&&(count<=cache||2*prefix[trial*CANDIDATES+CANDIDATES-1]+cache<count)){
            last_consumed[trial]=0;last_second[trial]=0ul;
        }
        if(valid[j]) {
            uint rank=2*(prefix[j]-1)+cache;
            if(rank<count) {
                ulong f=d_sqrt(d_div(d_mul(0xc000000000000000ul,expr_apple_log(radii[j])),radii[j]));
                ulong second=d_mul(f,xs[j]);if(KEEP_SECONDS)seconds[j]=second;
                values[trial*MAX_COUNT+rank]=d_mul(f,ys[j]);
                if(rank+1<count)values[trial*MAX_COUNT+rank+1]=second;
                // Exactly one accepted pair spans the final requested rank.
                // Its witness replaces a dependent binary search on prefix.
                if(rank+2>=count){last_consumed[trial]=2*(local+1);last_second[trial]=second;}
            }
        }
        ''',compile_options={'math_mode':'safe'})


def _prefix_state(initial,pool,consumed):
    """Restore a dynamically selected prefix from bounded coarse checkpoints."""
    import mlx.core as mx
    from .expression_rng import uniforms
    consumed=consumed.reshape(initial.trials,-1);frames=consumed.shape[1]
    width=pool.width;block=consumed//width;previous=mx.maximum(block,1)-1
    words=mx.take_along_axis(pool.words,previous[:,:,None],axis=1).reshape(-1)
    positions=mx.take_along_axis(pool.positions,previous,axis=1).reshape(-1)
    original=mx.broadcast_to(initial.words.reshape(initial.trials,1,624),
                            (initial.trials,frames,624)).reshape(-1)
    repeat=lambda v:mx.broadcast_to(v[:,None],(initial.trials,frames)).reshape(-1)
    words=mx.where(mx.repeat(block.reshape(-1)>0,624),words,original)
    positions=mx.where(block.reshape(-1)>0,positions,repeat(initial.position))
    start=MT19937State(words,positions,initial.trials*frames,
                       repeat(initial.has_gauss),repeat(initial.gauss))
    _,end=uniforms(start,width-1,(consumed.reshape(-1)%width).astype(mx.uint32))
    return end


def parallel_normals(state,maximum,counts=None,*,candidate_count=None,frame_width=None,target='numpy'):
    """Exact NumPy polar samples, including zero counts and cached odd pairs."""
    import mlx.core as mx
    from .expression_rng import _serial_normals as normals,_normal_target
    from .expression_rng_jump import uniform_pool
    maximum=index(maximum)
    if maximum<0 or maximum*state.trials>=2**32:
        raise ValueError('Normal draw capacity exceeds safe Metal indexing')
    cpp=_normal_target(target)
    if not maximum:return normals(state,0,counts,target=target)
    if counts is None:counts=mx.full((state.trials,),maximum,mx.uint32)
    if counts.shape!=(state.trials,)or counts.dtype!=mx.uint32:
        raise ValueError('Normal draw counts must be a uint32 vector per stream')
    candidates=(3*((maximum+1)//2)//2+128 if candidate_count is None else index(candidate_count))
    if candidates<1:raise ValueError('At least one normal candidate is required')
    # Large speculative pools checkpoint every eight MT blocks. Restoring a
    # model frame regenerates at most2495 uniforms, rather than retaining a
    # full624-word snapshot for every312 speculative values.
    checkpoint=312*8 if 2*candidates>=2**20 else 312
    frames=(2*candidates+checkpoint-1)//checkpoint;candidates=checkpoint*frames//2
    if (624*frames*state.trials>=2**32 or 2*candidates*state.trials>=2**32):
        if frame_width is not None:raise ValueError('Normal pool exceeds safe Metal indexing')
        return normals(state,maximum,counts,target=target)
    with mx.stream(mx.gpu):
        cached=mx.logical_and(state.has_gauss>0,counts>0).astype(mx.uint32)
        remaining=counts-cached
        limits=(mx.minimum(candidates,mx.where(remaining>0,3*((remaining+1)//2)//2+128,0)).astype(mx.uint32)
                if frame_width is None else mx.full((state.trials,),candidates,mx.uint32))
        pool=uniform_pool(state,checkpoint,frames,parts=32 if candidates>=2**19 else 1,
                          counts=2*limits if frame_width is None else None)
        raw=mx.contiguous(pool.values.transpose(1,0,2)).reshape(-1)
        xs,ys,radii,valid=_normal_candidates()(
            inputs=[raw,limits],template=[('B',state.trials),('CANDIDATES',candidates),('CPP_NORMAL',cpp)],
            grid=(state.trials*candidates,1,1),threadgroup=(256,1,1),
            output_shapes=[(state.trials*candidates,)]*4,
            output_dtypes=[mx.uint64,mx.uint64,mx.uint64,mx.uint32])
        valid=valid.reshape(state.trials,candidates)
        prefix=mx.cumsum(valid,axis=1)
        packed,second,last_consumed,last_second=_normal_pack()(inputs=[xs,ys,radii,valid.reshape(-1),prefix.reshape(-1),
                counts,state.has_gauss,state.gauss],
            template=[('B',state.trials),('CANDIDATES',candidates),('MAX_COUNT',maximum),('KEEP_SECONDS',frame_width is not None)],
            grid=(state.trials*candidates,1,1),threadgroup=(256,1,1),
            output_shapes=[(state.trials*maximum,),(state.trials*candidates if frame_width is not None else 1,),
                           (state.trials,),(state.trials,)],
            output_dtypes=[mx.uint64,mx.uint64,mx.uint32,mx.uint64],init_value=0)
        done=cached+mx.minimum(remaining,2*prefix[:,-1])
        if frame_width is None:
            pairs=(remaining+1)//2;enough=pairs<=prefix[:,-1]
            consumed=mx.where(remaining==0,0,mx.where(enough,last_consumed,2*limits)).astype(mx.uint32)
            end=_prefix_state(state,pool,consumed)
            has_gauss=mx.where(counts==0,state.has_gauss,(remaining%2)*enough).astype(mx.uint32)
            gauss=(mx.where(remaining==0,state.gauss,last_second)if cpp else mx.where(has_gauss>0,last_second,0))
            gauss=mx.where(counts==0,state.gauss,gauss)
            end=MT19937State(end.words,end.position,state.trials,has_gauss,gauss)
        else:
            # Only an exhausted pool uses this continuation. Exact per-frame
            # states below are the observable stream, so avoid restoring the
            # final accepted prefix twice in the ordinary nonexhausted case.
            end=pool.state_at(pool.frames-1)
            end=MT19937State(end.words,end.position,state.trials,
                mx.zeros((state.trials,),mx.uint32),mx.zeros((state.trials,),mx.uint64))
        tail,end=normals(end,maximum,(counts-done).astype(mx.uint32),target=target)
        offsets=mx.maximum(mx.arange(maximum)[None,:]-done[:,None],0)
        output=mx.where(mx.arange(maximum)[None,:]<done[:,None],
            packed.reshape(state.trials,maximum),mx.take_along_axis(tail,offsets,axis=1))
        output=mx.where(mx.arange(maximum)[None,:]<counts[:,None],output,0)
        if frame_width is None:return output,end
        return _normal_pool(state,pool,output,prefix,second.reshape(state.trials,candidates),
                            maximum,index(frame_width),target)


@dataclass(frozen=True)
class NormalPool:
    values: object
    words: object
    positions: object
    has_gauss: object
    gauss: object
    trials: int
    width: int
    frames: int
    target: str = 'numpy'

    def state_at(self,frame):
        frame=index(frame)
        if not 0<=frame<self.frames:raise IndexError('Normal checkpoint frame out of range')
        return MT19937State(self.words[:,frame,:].reshape(-1),self.positions[:,frame].reshape(-1),
            self.trials,self.has_gauss[:,frame].reshape(-1),self.gauss[:,frame].reshape(-1))


def _first_prefix(prefix,counts):
    import mlx.core as mx
    candidates=prefix.shape[1]
    lo=mx.zeros(counts.shape,mx.uint32);hi=mx.full(counts.shape,candidates,mx.uint32)
    for _ in range(candidates.bit_length()):
        middle=(lo+hi)//2
        value=mx.take_along_axis(prefix,mx.minimum(middle,candidates-1),axis=1)
        go_right=value<counts
        lo=mx.where(go_right,middle+1,lo);hi=mx.where(go_right,hi,middle)
    return mx.minimum(lo,candidates-1)


def _normal_pool(initial,uniform_pool,output,prefix,seconds,maximum,width,target):
    import mlx.core as mx
    from .expression_rng import _normal_states
    frames=maximum//width;trials=initial.trials;candidates=prefix.shape[1]
    counts=mx.broadcast_to(mx.arange(1,frames+1,dtype=mx.uint32)[None,:]*width,(trials,frames))
    cached=mx.broadcast_to((initial.has_gauss>0)[:,None],counts.shape).astype(mx.uint32)
    remaining=counts-cached;pairs=(remaining+1)//2
    last=_first_prefix(prefix,pairs);enough=pairs<=prefix[:,-1,None]
    consumed=mx.where(remaining==0,0,mx.where(enough,2*(last+1),2*candidates)).astype(mx.uint32)
    stream=_prefix_state(initial,uniform_pool,consumed)
    flags=((remaining%2)*enough).astype(mx.uint32)
    saved=mx.take_along_axis(seconds,last,axis=1)
    gauss=(mx.where(remaining==0,initial.gauss[:,None],saved)if target=='cpp'else mx.where(flags>0,saved,0))
    stream=MT19937State(stream.words,stream.position,stream.trials,flags.reshape(-1),gauss.reshape(-1))
    done=cached+mx.minimum(remaining,2*prefix[:,-1,None])
    stream=_normal_states(stream,(counts-done).reshape(-1).astype(mx.uint32),target=target)
    return NormalPool(mx.contiguous(output.reshape(trials,frames,width).transpose(1,0,2)),
        stream.words.reshape(trials,frames,624),stream.position.reshape(trials,frames),
        stream.has_gauss.reshape(trials,frames),stream.gauss.reshape(trials,frames),trials,width,frames,target)


def normal_pool(state,width,frames,*,candidate_count=None,target='numpy'):
    """Fixed-width lookahead with exact normal cache and MT state each frame."""
    width=index(width);frames=index(frames)
    if width<1 or frames<1 or frames*state.trials*624>=2**32:
        raise ValueError('Invalid normal pool layout')
    return parallel_normals(state,width*frames,candidate_count=candidate_count,frame_width=width,target=target)
