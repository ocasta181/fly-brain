"""Native MT19937 uniform/normal streams with immutable device state.

Only initial seeding imports NumPy's legacy RandomState. Draws and stream
continuation execute on Metal. This primitive does not replace the PCG64
canonical input-event generator, whose stream is a separate contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from operator import index

import numpy as np


@dataclass(frozen=True)
class MT19937State:
    """Complete legacy state, one independent stream per row."""

    words: object
    position: object
    trials: int
    has_gauss: object
    gauss: object


def from_numpy_states(states) -> MT19937State:
    """Copy explicit legacy states; never read or advance process-global RNG.

    The normal cache is copied with the MT words. Uniform generation itself
    does not consume or change that cache.
    """
    import mlx.core as mx

    words, positions, has_gauss, gauss = [], [], [], []
    for state in states:
        if len(state) != 5 or state[0] != "MT19937":
            raise ValueError("Expression RNG requires a NumPy MT19937 state")
        row = np.asarray(state[1])
        position = index(state[2])
        if row.dtype != np.uint32 or row.shape != (624,) or not 0 <= position <= 624:
            raise ValueError("Invalid MT19937 words or uniform-stream position")
        words.append(row.copy())
        positions.append(position)
        has_gauss.append(index(state[3]))
        gauss.append(float(state[4]))
    if not words:
        raise ValueError("At least one independent random stream is required")
    return MT19937State(mx.array(np.stack(words).reshape(-1)),
                        mx.array(positions, dtype=mx.uint32), len(words),
                        mx.array(has_gauss, dtype=mx.uint32),
                        mx.array(np.asarray(gauss, np.float64).view(np.uint64)))


def seeded(seeds) -> MT19937State:
    """Use NumPy's unchanged legacy integer/array seed initialization."""
    return from_numpy_states([np.random.RandomState(seed).get_state() for seed in seeds])


@lru_cache(maxsize=8)
def _uniform_kernel(snapshots=False,clear_unused=True,prefix=False):
    import mlx.core as mx
    extra=[];extra_header=''
    load='for(uint j=lane; j<624; j+=320) mt[j]=state[trial*624+j];\nuint pos=position[trial], count=counts[trial];'
    if prefix:
        if snapshots:raise ValueError('Mixed-prefix restoration does not emit speculative snapshots')
        from .expression_rng_prefix import _RESTORE_INPUT_NAMES,_restore_source
        extra=list(_RESTORE_INPUT_NAMES)
        load='uint pos=position[trial], count=counts[trial];\n'+_restore_source(320)
        extra_header=_MT_HEADER[_MT_HEADER.index('inline void mt_twist'):]
    checkpoint=r'''
        uint global_end=global_begin+begin;
        if(global_end%SNAP_WIDTH==0) {
            uint frame=global_end/SNAP_WIDTH-1;
            if(frame<SNAP_STEPS) {
                for(uint j=lane;j<624;j+=320)
                    snapshot_words[(snapshot_trial*SNAP_STEPS+frame)*624+j]=mt[j];
                if(lane==0) snapshot_positions[snapshot_trial*SNAP_STEPS+frame]=pos;
            }
        }
    ''' if snapshots else ''
    prologue=('uint snapshot_trial=trial/PARTS;uint global_begin=(trial%PARTS)*MAX_COUNT;'
              if snapshots else '')
    limit=('available=min(available,SNAP_WIDTH-((global_begin+begin)%SNAP_WIDTH));'
           if snapshots else '')
    return mx.fast.metal_kernel(
        name="fly_expression_mt19937_uniforms",
        input_names=["state", "position", "counts",*extra],
        output_names=["result", "next_state", "next_position"]+
                     (["snapshot_words","snapshot_positions"]if snapshots else []),
        source=r'''
        uint lane = thread_position_in_threadgroup.x;
        uint trial = threadgroup_position_in_grid.x;
        /*SNAP_PROLOGUE*/
        threadgroup uint mt[624];
        /*LOAD_STATE*/
        // Requested entries are all written below; initialize only unused
        // masked capacity. A full-population draw needs no zeroing pass.
        /*CLEAR_UNUSED*/
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint begin=0;
        while(begin<count) {
            bool carry=pos==623;
            uint first=carry ? mt[623] : 0u;
            if(carry) pos=624;
            if(pos==624) {
                // The reference twists in place: later ranges depend on the
                // new words from earlier ranges. Cache the old adjacent word
                // pairs before any writes, then respect those dependencies.
                uint y0=0, y1=0, y2=0;
                if(lane<227) y0=(mt[lane]&0x80000000u)|(mt[lane+1]&0x7fffffffu);
                if(lane<227) y1=(mt[227+lane]&0x80000000u)|(mt[228+lane]&0x7fffffffu);
                if(lane<169) y2=(mt[454+lane]&0x80000000u)|(mt[455+lane]&0x7fffffffu);
                uint last=mt[623];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if(lane<227) mt[lane]=mt[lane+397]^(y0>>1)^((y0&1)?0x9908b0dfu:0u);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if(lane<227) mt[227+lane]=mt[lane]^(y1>>1)^((y1&1)?0x9908b0dfu:0u);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if(lane<169) mt[454+lane]=mt[227+lane]^(y2>>1)^((y2&1)?0x9908b0dfu:0u);
                if(lane==0) {
                    uint y=(last&0x80000000u)|(mt[0]&0x7fffffffu);
                    mt[623]=mt[396]^(y>>1)^((y&1)?0x9908b0dfu:0u);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                pos=0;
            }
            if(carry) {
                if(lane==0) result[trial*MAX_COUNT+begin]=mt_uniform_bits(first,mt[0]);
                pos=1;
                ++begin;
                /*SNAP_CARRY*/
            }
            uint available=min((624-pos)/2, count-begin);
            /*SNAP_LIMIT*/
            for(uint j=lane; j<available; j+=320) {
                result[trial*MAX_COUNT+begin+j]=mt_uniform_bits(mt[pos+2*j],mt[pos+2*j+1]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            begin+=available;
            pos+=2*available;
            /*SNAP_POINT*/
        }
        for(uint j=lane; j<624; j+=320) next_state[trial*624+j]=mt[j];
        if(lane==0) next_position[trial]=pos;
        '''.replace('/*LOAD_STATE*/',load).replace('/*CLEAR_UNUSED*/',r'''for(ulong j=lane; j<ulong(MAX_COUNT)-count; j+=320)
            result[ulong(trial)*MAX_COUNT+count+j]=0ul;'''if clear_unused else'')
           .replace('/*SNAP_PROLOGUE*/',prologue).replace('/*SNAP_CARRY*/',checkpoint)
           .replace('/*SNAP_LIMIT*/',limit)
           .replace('/*SNAP_POINT*/',('if(available>0) {'+checkpoint+'}')if snapshots else ''), header=r'''
        inline uint mt_temper(uint a) {
            a^=a>>11; a^=(a<<7)&0x9d2c5680u; a^=(a<<15)&0xefc60000u; a^=a>>18;
            return a;
        }
        inline ulong mt_uniform_bits(uint a,uint b) {
            ulong mantissa=(ulong(mt_temper(a)>>5)<<26)|ulong(mt_temper(b)>>6);
            if(mantissa==0ul) return 0ul;
            uint shift=uint(clz(mantissa))-11u;
            return (ulong(1022u-shift)<<52)|((mantissa<<shift)&0x000ffffffffffffful);
        }
        '''+extra_header, compile_options={"math_mode": "safe"},
    )


def uniforms(state: MT19937State, maximum: int, counts=None,*,_clear_unused=True,_prefix=None):
    """Draw device-selected counts, bounded by maximum, without host reads.

    `counts` is a uint32 device vector in [0, maximum], one per trial. The
    enclosing model planner must prove this bound before dispatch. Unused
    output entries are zero. State inputs are never modified, allowing exact
    branching/replay from the same checkpoint.
    """
    import mlx.core as mx

    maximum = index(maximum)
    if not 0 <= maximum < 2**32 or maximum * state.trials >= 2**32:
        raise ValueError("Uniform draw capacity exceeds safe Metal indexing")
    if counts is None:
        counts = mx.full((state.trials,), maximum, mx.uint32)
    if counts.shape != (state.trials,) or counts.dtype != mx.uint32:
        raise ValueError("Uniform draw counts must be a uint32 vector per stream")
    if maximum == 0:
        return mx.zeros((state.trials, 0), mx.uint64), state
    extra=[];templates=[]
    if _prefix is not None:
        from .expression_rng_prefix import _restore_arguments
        extra,templates=_restore_arguments(*_prefix)
    with mx.stream(mx.gpu):
        result, words, position = _uniform_kernel(clear_unused=_clear_unused,prefix=_prefix is not None)(
            inputs=[state.words, state.position, counts,*extra], template=[("MAX_COUNT", maximum),*templates],
            grid=(state.trials*320, 1, 1), threadgroup=(320, 1, 1),
            output_shapes=[(state.trials*maximum,), (state.trials*624,), (state.trials,)],
            output_dtypes=[mx.uint64, mx.uint32, mx.uint32],
        )
    return result.reshape(state.trials, maximum), MT19937State(
        words, position, state.trials, state.has_gauss, state.gauss)


def _normal_source():
    return r'''
    uint lane=thread_position_in_threadgroup.x, trial=threadgroup_position_in_grid.x;
    threadgroup uint mt[624], boundary[4], warp_counts[8], end_position;
    threadgroup ulong saved_gauss;
    uint pos=position[trial], count=counts[trial], begin=0;
    uint cached=has_gauss[trial]; ulong saved=gauss[trial];
    for(uint j=lane;j<624;j+=256) mt[j]=state[trial*624+j];
    if(cached && count>0) {
        if(!STATE_ONLY&&lane==0) result[trial*MAX_COUNT]=saved;
        cached=0;if(!CPP_NORMAL)saved=0ul;begin=1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    while(begin<count) {
        if(pos==624) {mt_twist(mt,lane); pos=0;}
        uint start=pos;
        bool cross=624-pos<4;
        uint candidates;
        uint a=0,b=0,c=0,d=0;
        if(cross) {
            uint tail=624-pos;
            if(lane<tail) boundary[lane]=mt[pos+lane];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mt_twist(mt,lane);
            if(lane<4-tail) boundary[tail+lane]=mt[lane];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            pos=4-tail;
            candidates=1;
            if(lane==0) {a=boundary[0];b=boundary[1];c=boundary[2];d=boundary[3];}
        } else {
            candidates=min((624-pos)/4,(count-begin+1)/2);
            if(lane<candidates) {a=mt[pos+4*lane];b=mt[pos+4*lane+1];c=mt[pos+4*lane+2];d=mt[pos+4*lane+3];}
        }
        bool valid=false; ulong first=0ul,second=0ul;
        if(lane<candidates) {
            const ulong one=0x3ff0000000000000ul,two=0x4000000000000000ul;
            ulong x=d_sub(d_mul(two,mt_uniform_bits(a,b)),one);
            ulong y=d_sub(d_mul(two,mt_uniform_bits(c,d)),one);
            ulong rr=CPP_NORMAL ?expr_fma(y,y,d_mul(x,x)):d_add(d_mul(x,x),d_mul(y,y));
            valid=d_lt(rr,one) && !d_eq(rr,0ul);
            if(valid) {
                ulong factor=d_sqrt(d_div(d_mul(0xc000000000000000ul,expr_apple_log(rr)),rr));
                first=d_mul(factor,y);second=d_mul(factor,x);
            }
        }
        uint prefix=simd_prefix_exclusive_sum(uint(valid));
        uint warp_count=simd_sum(uint(valid));
        if(lane%32==0) warp_counts[lane/32]=warp_count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint accepted=0;
        for(uint j=0;j<8;++j) {if(j<lane/32)prefix+=warp_counts[j];accepted+=warp_counts[j];}
        uint rank=2*prefix, remaining=count-begin;
        if(valid && rank<remaining) {
            if(!STATE_ONLY)result[trial*MAX_COUNT+begin+rank]=first;
            if(rank+1<remaining) {if(!STATE_ONLY)result[trial*MAX_COUNT+begin+rank+1]=second;}
            if(rank+2>=remaining)saved_gauss=second;
            if(rank+2>=remaining) end_position=cross ? pos : start+4*(lane+1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(2*accepted>=remaining) {
            pos=end_position;
            cached=remaining%2;
            saved=(CPP_NORMAL||cached) ?saved_gauss:0ul;
            begin=count;
        } else {
            begin+=2*accepted;
            if(!cross) pos+=4*candidates;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint j=lane;j<624;j+=256) next_state[trial*624+j]=mt[j];
    if(lane==0) {next_position[trial]=pos;next_has_gauss[trial]=cached;next_gauss[trial]=saved;}
    '''


@lru_cache(maxsize=2)
def _normal_kernel(prefix=False):
    import mlx.core as mx
    from .binary64 import MSL_HEADER
    from .expression_rng_math import NORMAL_MATH_HEADER

    # Reuse precisely the same twist and temper operations as uniform draws.
    mt_header = _MT_HEADER
    source=_normal_source();extra=[]
    if prefix:
        from .expression_rng_prefix import _RESTORE_INPUT_NAMES,_restore_source
        extra=list(_RESTORE_INPUT_NAMES)
        load='for(uint j=lane;j<624;j+=256) mt[j]=state[trial*624+j];'
        if source.count(load)!=1:raise ValueError('Missing native normal state initialization')
        source=source.replace(load,_restore_source(256))
    return mx.fast.metal_kernel(
        name='fly_expression_mt19937_normals',
        input_names=['state','position','counts','has_gauss','gauss',*extra],
        output_names=['result','next_state','next_position','next_has_gauss','next_gauss'],
        source=source,header=MSL_HEADER+NORMAL_MATH_HEADER+mt_header,
        compile_options={'math_mode':'safe'},
    )


def _normal_target(target):
    if target not in ('numpy','cpp'):raise ValueError('Unknown normal RNG source target')
    return target=='cpp'


def _serial_normals(state: MT19937State, maximum: int, counts=None,*,target='numpy',_clear_unused=True,_prefix=None):
    """Legacy polar normal draws with exact cached-pair/MT consumption."""
    import mlx.core as mx
    maximum=index(maximum)
    cpp=_normal_target(target)
    if not 0<=maximum<2**32 or maximum*state.trials>=2**32:
        raise ValueError('Normal draw capacity exceeds safe Metal indexing')
    if counts is None:counts=mx.full((state.trials,),maximum,mx.uint32)
    if counts.shape!=(state.trials,) or counts.dtype!=mx.uint32:
        raise ValueError('Normal draw counts must be a uint32 vector per stream')
    if maximum==0:return mx.zeros((state.trials,0),mx.uint64),state
    extra=[];templates=[]
    if _prefix is not None:
        from .expression_rng_prefix import _restore_arguments
        extra,templates=_restore_arguments(*_prefix)
    with mx.stream(mx.gpu):
        result,words,position,has_gauss,gauss=_normal_kernel(prefix=_prefix is not None)(
            inputs=[state.words,state.position,counts,state.has_gauss,state.gauss,*extra],
            template=[('MAX_COUNT',maximum),('STATE_ONLY',False),('CPP_NORMAL',cpp),*templates],
            grid=(state.trials*256,1,1),threadgroup=(256,1,1),
            output_shapes=[(state.trials*maximum,),(state.trials*624,),(state.trials,),(state.trials,),(state.trials,)],
            output_dtypes=[mx.uint64,mx.uint32,mx.uint32,mx.uint32,mx.uint64],
            **({'init_value':0}if _clear_unused else{}),
        )
    return result.reshape(state.trials,maximum),MT19937State(words,position,state.trials,has_gauss,gauss)


def _normal_states(state,counts,*,target='numpy'):
    """Advance dynamic prefixes without allocating discarded sample vectors."""
    import mlx.core as mx
    with mx.stream(mx.gpu):
        _,words,position,has_gauss,gauss=_normal_kernel()(
            inputs=[state.words,state.position,counts,state.has_gauss,state.gauss],
            template=[('MAX_COUNT',1),('STATE_ONLY',True),('CPP_NORMAL',_normal_target(target))],
            grid=(state.trials*256,1,1),threadgroup=(256,1,1),
            output_shapes=[(1,),(state.trials*624,),(state.trials,),(state.trials,),(state.trials,)],
            output_dtypes=[mx.uint64,mx.uint32,mx.uint32,mx.uint32,mx.uint64])
    return MT19937State(words,position,state.trials,has_gauss,gauss)


def normals(state: MT19937State, maximum: int, counts=None,*,target='numpy'):
    """Exact NumPy or frozen default-clang Brian CPP polar normal policy."""
    maximum=index(maximum)
    _normal_target(target)
    if 2**16<=maximum and maximum*state.trials<=8*1024**2:
        from .expression_rng_parallel import parallel_normals
        return parallel_normals(state,maximum,counts,target=target)
    return _serial_normals(state,maximum,counts,target=target)


_MT_HEADER = r'''

    inline uint mt_temper(uint a) {
        a^=a>>11; a^=(a<<7)&0x9d2c5680u; a^=(a<<15)&0xefc60000u; a^=a>>18;
        return a;
    }
    inline ulong mt_uniform_bits(uint a,uint b) {
        ulong mantissa=(ulong(mt_temper(a)>>5)<<26)|ulong(mt_temper(b)>>6);
        if(mantissa==0ul) return 0ul;
        uint shift=uint(clz(mantissa))-11u;
        return (ulong(1022u-shift)<<52)|((mantissa<<shift)&0x000ffffffffffffful);
    }
    inline void mt_twist(threadgroup uint* mt,uint lane) {
        uint y0=0, y1=0, y2=0;
        if(lane<227) y0=(mt[lane]&0x80000000u)|(mt[lane+1]&0x7fffffffu);
        if(lane<227) y1=(mt[227+lane]&0x80000000u)|(mt[228+lane]&0x7fffffffu);
        if(lane<169) y2=(mt[454+lane]&0x80000000u)|(mt[455+lane]&0x7fffffffu);
        uint last=mt[623];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(lane<227) mt[lane]=mt[lane+397]^(y0>>1)^((y0&1)?0x9908b0dfu:0u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(lane<227) mt[227+lane]=mt[lane]^(y1>>1)^((y1&1)?0x9908b0dfu:0u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(lane<169) mt[454+lane]=mt[227+lane]^(y2>>1)^((y2&1)?0x9908b0dfu:0u);
        if(lane==0) {
            uint y=(last&0x80000000u)|(mt[0]&0x7fffffffu);
            mt[623]=mt[396]^(y>>1)^((y&1)?0x9908b0dfu:0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

inline ulong mt_next_double(threadgroup uint* mt,thread uint* pos,uint lane) {
 if(*pos==624) {mt_twist(mt,lane);*pos=0;}
 uint a=mt[*pos];++*pos;
 if(*pos==624) {mt_twist(mt,lane);*pos=0;}
 uint b=mt[*pos];++*pos;
 return mt_uniform_bits(a,b);
}

'''


def to_numpy_states(state: MT19937State):
    """Serialize an explicit checkpoint at a caller-selected synchronization."""
    import mlx.core as mx
    mx.eval(state.words,state.position,state.has_gauss,state.gauss)
    words=np.asarray(state.words).reshape(state.trials,624)
    position=np.asarray(state.position);flags=np.asarray(state.has_gauss)
    gauss=np.asarray(state.gauss).view(np.float64)
    return tuple(('MT19937',words[row].copy(),int(position[row]),int(flags[row]),float(gauss[row]))
                 for row in range(state.trials))
