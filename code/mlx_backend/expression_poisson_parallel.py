"""Bounded exact constant-rate Poisson sampling on a contiguous MT prefix.

Rejection formulas reuse NumPy1.26.4's algorithms (BSD-3-Clause; see
third_party/NumPy-LICENSE). Block traversal and checkpoint selection are local.
"""
from functools import lru_cache
from operator import index
import math
import numpy as np


@lru_cache(maxsize=32)
def _rhs_table(lam,target):
    """Cache exact device arithmetic for repeated integer log probabilities."""
    import mlx.core as mx
    from .binary64 import MSL_HEADER,literal
    from .expression_rng_math import NORMAL_MATH_HEADER,POISSON_EXP_HEADER
    from .expression_rng import _MT_HEADER
    if target in('cpp','cython'):
        from .expression_poisson_compiled import source
        header=source('uint ignored','0ul',target=target)
        expression=('expr_fma(d_from_int(k),loglam,d_neg(d_add(lam,compiled_gamma(int(k)))))'if target=='cython'else
            'd_sub(d_mul(d_from_int(k),loglam),d_add(lam,compiled_gamma(int(k))))')
    else:
        from .expression_poisson import _source
        header=_source()
        expression='d_sub(d_add(d_neg(lam),d_mul(d_from_int(k),loglam)),poisson_loggam(d_from_int(k+1)))'
    center=int(lam)if lam<=1e8 else 0
    radius=min(4096,max(512,int(math.ceil(16*math.sqrt(lam)))))if lam<=1e8 else 0
    begin=max(0,center-radius);size=2*radius+1 if radius else 1
    if target in('cpp','cython')and lam>2147483647:begin=0;size=1
    kernel=mx.fast.metal_kernel(name='expression_poisson_integer_probability_table',
        input_names=['unused'],output_names=['values'],
        header=MSL_HEADER+NORMAL_MATH_HEADER+POISSON_EXP_HEADER+_MT_HEADER+header,source=f'''
        uint j=thread_position_in_grid.x;if(j>=COUNT)return;long k=long(j)+{begin}L;
        const ulong lam={literal(lam)},loglam={literal(math.log(lam))};values[j]={expression};
        ''',compile_options={'math_mode':'safe'})
    values,=kernel(inputs=[mx.zeros((1,),mx.uint32)],template=[('COUNT',size)],
        grid=(size,1,1),threadgroup=(256,1,1),output_shapes=[(size,)],output_dtypes=[mx.uint64])
    return begin,size,values


@lru_cache(maxsize=32)
def _candidates(lam,target='numpy'):
    import mlx.core as mx
    from .binary64 import MSL_HEADER,literal
    from .expression_rng_math import NORMAL_MATH_HEADER
    from .expression_poisson import _source,_MathRenderer
    if lam<10:
        return mx.fast.metal_kernel(name='expression_poisson_product_candidates',
            input_names=['uniform'],output_names=['next'],header=MSL_HEADER,source=f'''
            uint j=thread_position_in_grid.x;if(j>=B*RAW)return;
            uint row=j/RAW,start=j%RAW,p=start;ulong prod={literal(1.)};
            do {{prod=d_mul(prod,uniform[row*RAW+p]);++p;}}
            while(p<RAW&&d_gt(prod,{literal(math.exp(-lam))}));
            next[j]=d_le(prod,{literal(math.exp(-lam))}) ? p:RAW+1;
            ''',compile_options={'math_mode':'safe'})
    if target in('cpp','cython'):
        from fractions import Fraction
        from .expression_poisson_compiled import source
        from .expression_rng_math import POISSON_EXP_HEADER
        product=2.53*math.sqrt(lam);b=product+.931
        a=float(Fraction(b)*Fraction(.02483)+Fraction(-.059))
        invalpha=1.1239+1.1328/(product+(.931-3.4));vr=.9277-3.6224/(product+(.931-2))
        constants='\n'.join(f'const ulong {name}={literal(value)};'for name,value in
            dict(lam=lam,b=b,a=a,vr=vr,loglam=math.log(lam),logalpha=math.log(invalpha),lam43=lam+.43).items())
        return mx.fast.metal_kernel(name='expression_cpp_poisson_ptrs_candidates',
            input_names=['uniform','log_probability'],output_names=['values','valid']+(['zero']if target=='cython'else[]),
            header=MSL_HEADER+NORMAL_MATH_HEADER+POISSON_EXP_HEADER+source('uint ignored','0ul',target=target),source=f'''
            uint j=thread_position_in_grid.x;if(j>=B*CANDIDATES)return;
            {constants}
            ulong u=d_sub(uniform[2*j],{literal(.5)}),v=uniform[2*j+1],us=d_sub({literal(.5)},d_abs(u));
            {'zero[j]=uint(d_is_zero(us));if(d_is_zero(us)){values[j]=0;valid[j]=0;return;}'if target=='cython'else''}
            int result=compiled_floor_int32(expr_fma(d_add(d_div(d_add(a,a),us),b),u,lam43));
            bool accepted=d_ge(us,{literal(.07)})&&d_le(v,vr);
            if(!accepted&&result>=0&&!(d_lt(us,{literal(.013)})&&d_gt(v,us))) {{
                ulong logv=compiled_log(v),logden=compiled_log(d_add(d_div(a,d_mul(us,us)),b));
                ulong left={'d_add(d_sub(logalpha,logden),logv)'if target=='cython'else'd_sub(d_add(logv,logalpha),logden)'};
                long offset=long(result)-TABLE_BEGIN;
                ulong right=(offset>=0&&offset<TABLE_SIZE)?log_probability[offset]:
                    {'expr_fma(d_from_int(result),loglam,d_neg(d_add(lam,compiled_gamma(result))))'if target=='cython'else'd_sub(d_mul(d_from_int(result),loglam),d_add(lam,compiled_gamma(result)))'};
                accepted=d_le(left,right);
            }}
            values[j]=long(result);valid[j]=uint(accepted);
            ''',compile_options={'math_mode':'safe'})
    renderer=_MathRenderer({name:np.float64 for name in
        'u v us candidate k a b lam invalpha vr loglam'.split()}, {})
    def e(expr):return renderer.convert(renderer.expression(expr),'float').code
    slam=math.sqrt(lam);b=.931+2.53*slam;a=-.059+.02483*b
    invalpha=1.1239+1.1328/(b-3.4);vr=.9277-3.6224/(b-2)
    constants='\n'.join(f'const ulong {name}={literal(value)};'for name,value in
        dict(lam=lam,a=a,b=b,invalpha=invalpha,vr=vr,loglam=math.log(lam)).items())
    # poisson_loggam is shared with the general exact NumPy sampler.
    from .expression_rng import _MT_HEADER
    from .expression_rng_math import POISSON_EXP_HEADER
    return mx.fast.metal_kernel(name='expression_poisson_ptrs_candidates',
        input_names=['uniform','log_probability'],output_names=['values','valid'],
        header=MSL_HEADER+NORMAL_MATH_HEADER+POISSON_EXP_HEADER+_MT_HEADER+_source(),source=f'''
        uint j=thread_position_in_grid.x;if(j>=B*CANDIDATES)return;
        {constants}
        ulong u=d_sub(uniform[2*j],{literal(.5)}),v=uniform[2*j+1],us={e('.5-abs(u)')};
        long result=poisson_floor({e('(2*a/us+b)*u+lam+.43')});
        bool accepted=d_ge(us,{literal(.07)})&&d_le(v,vr);
        if(!accepted&&result>=0&&!(d_lt(us,{literal(.013)})&&d_gt(v,us))) {{
            ulong left={e('log(v)+log(invalpha)-log(a/(us*us)+b)')};
            long offset=result-TABLE_BEGIN;
            ulong right=(offset>=0&&offset<TABLE_SIZE)?log_probability[offset]:
                d_sub(d_add(d_neg(lam),d_mul(d_from_int(result),loglam)),poisson_loggam(d_from_int(result+1)));
            accepted=d_le(left,right);
        }}
        values[j]=result;valid[j]=uint(accepted);
        ''',compile_options={'math_mode':'safe'})


@lru_cache(maxsize=1)
def _pack():
    import mlx.core as mx
    return mx.fast.metal_kernel(name='expression_poisson_pack',
        input_names=['candidates','valid','prefix','counts'],output_names=['values','ends'],
        source=r'''
        uint j=thread_position_in_grid.x;if(j>=B*CANDIDATES)return;
        uint row=j/CANDIDATES;
        if(valid[j]&&prefix[j]<=counts[row]) {
            values[row*MAX_COUNT+prefix[j]-1]=candidates[j];
            if(KEEP_ENDS)ends[row*MAX_COUNT+prefix[j]-1]=2*(j%CANDIDATES+1);
        }
        ''',compile_options={'math_mode':'safe'})


@lru_cache(maxsize=1)
def _product_links():
    import mlx.core as mx
    return mx.fast.metal_kernel(name='expression_poisson_product_links',
        input_names=['following'],output_names=['exits','sizes'],source=r'''
        uint block=thread_position_in_grid.x;if(block>=B*BLOCKS)return;
        uint row=block/BLOCKS,local=block%BLOCKS,begin=local*256,end=min(begin+256,uint(RAW));
        for(uint p=end;p>begin;) {
            --p;uint next=following[row*RAW+p];
            exits[row*RAW+p]=next<end ?exits[row*RAW+next]:next;
            sizes[row*RAW+p]=next>RAW ?0u:1u+(next<end ?sizes[row*RAW+next]:0u);
        }
        ''',compile_options={'math_mode':'safe'})


@lru_cache(maxsize=1)
def _super_links():
    import mlx.core as mx
    return mx.fast.metal_kernel(name='expression_poisson_super_links',
        input_names=['old_exits','old_sizes'],output_names=['exits','sizes'],source=r'''
        uint j=thread_position_in_grid.x;if(j>=B*RAW)return;
        uint row=j/RAW,start=j%RAW,next=old_exits[j];
        bool within=next<RAW&&next/4096==start/4096;
        exits[j]=within ?old_exits[row*RAW+next]:next;
        sizes[j]=old_sizes[j]+(within ?old_sizes[row*RAW+next]:0u);
        ''',compile_options={'math_mode':'safe'})


@lru_cache(maxsize=1)
def _super_entries():
    import mlx.core as mx
    return mx.fast.metal_kernel(name='expression_poisson_super_entries',
        input_names=['exits','sizes','counts'],output_names=['entries','ranks'],source=r'''
        uint row=thread_position_in_grid.x;if(row>=B)return;
        uint current=0,rank=0;
        while(current<RAW&&rank<counts[row]) {
            uint block=current/4096;
            entries[row*SUPERBLOCKS+block]=current+1;ranks[row*SUPERBLOCKS+block]=rank;
            rank+=sizes[row*RAW+current];current=exits[row*RAW+current];
        }
        ''',compile_options={'math_mode':'safe'})


@lru_cache(maxsize=1)
def _product_entries():
    import mlx.core as mx
    return mx.fast.metal_kernel(name='expression_poisson_product_entries',
        input_names=['exits','sizes','super_entries','super_ranks','counts'],
        output_names=['entries','ranks'],source=r'''
        uint super=thread_position_in_grid.x;if(super>=B*SUPERBLOCKS)return;
        uint entry=super_entries[super];if(!entry)return;
        uint row=super/SUPERBLOCKS,local=super%SUPERBLOCKS,current=entry-1,rank=super_ranks[super];
        uint end=min((local+1)*4096,uint(RAW));
        while(current<end&&rank<counts[row]) {
            uint block=current/256;
            entries[row*BLOCKS+block]=current+1;ranks[row*BLOCKS+block]=rank;
            rank+=sizes[row*RAW+current];current=exits[row*RAW+current];
        }
        ''',compile_options={'math_mode':'safe'})


@lru_cache(maxsize=1)
def _product_pack():
    import mlx.core as mx
    return mx.fast.metal_kernel(name='expression_poisson_product_pack',
        input_names=['following','entries','ranks','counts'],output_names=['values','consumed','done','ends'],
        source=r'''
        uint block=thread_position_in_grid.x;if(block>=B*BLOCKS)return;
        uint entry=entries[block];if(!entry)return;
        uint row=block/BLOCKS,local=block%BLOCKS,current=entry-1,rank=ranks[block];
        uint end=min((local+1)*256,uint(RAW)),count=counts[row];
        while(current<end&&rank<count) {
            uint next=following[row*RAW+current];
            if(next>RAW) {consumed[row]=current;done[row]=rank;break;}
            values[row*MAX_COUNT+rank]=long(next-current-1);
            if(KEEP_ENDS)ends[row*MAX_COUNT+rank]=next;
            ++rank;current=next;
            if(rank==count||current==RAW) {consumed[row]=current;done[row]=rank;}
        }
        ''',compile_options={'math_mode':'safe'})


def _product_prefix(uniforms,counts,maximum,lam,raw,trials,*,keep_ends):
    """Exact draw-position traversal shared by buffered and direct streams."""
    import mlx.core as mx
    following,=_candidates(lam)(inputs=[uniforms],template=[('B',trials),('RAW',raw)],
        grid=(trials*raw,1,1),threadgroup=(256,1,1),
        output_shapes=[(trials*raw,)],output_dtypes=[mx.uint32])
    # A linear-work block traversal replaces whole-array pointer
    # doubling. Independent local suffixes identify each block exit;
    # one small device walk selects the actual entry into each block,
    # then blocks emit their exact sequential samples independently.
    blocks=(raw+255)//256;supers=(raw+4095)//4096
    template=[('B',trials),('RAW',raw),('BLOCKS',blocks),('SUPERBLOCKS',supers)]
    exits,sizes=_product_links()(inputs=[following],template=template,
        grid=(trials*blocks,1,1),threadgroup=(256,1,1),
        output_shapes=[(trials*raw,)]*2,output_dtypes=[mx.uint32,mx.uint32])
    super_exits,super_sizes=exits,sizes
    for _ in range(4):
        super_exits,super_sizes=_super_links()(inputs=[super_exits,super_sizes],template=template,
            grid=(trials*raw,1,1),threadgroup=(256,1,1),
            output_shapes=[(trials*raw,)]*2,output_dtypes=[mx.uint32,mx.uint32])
    super_entries,super_ranks=_super_entries()(inputs=[super_exits,super_sizes,counts],template=template,
        grid=(trials,1,1),threadgroup=(min(256,trials),1,1),
        output_shapes=[(trials*supers,)]*2,output_dtypes=[mx.uint32,mx.uint32],init_value=0)
    entries,ranks=_product_entries()(inputs=[exits,sizes,super_entries,super_ranks,counts],template=template,
        grid=(trials*supers,1,1),threadgroup=(256,1,1),
        output_shapes=[(trials*blocks,)]*2,
        output_dtypes=[mx.uint32,mx.uint32],init_value=0)
    packed,consumed,done,ends=_product_pack()(inputs=[following,entries,ranks,counts],
        template=template+[('MAX_COUNT',maximum),('KEEP_ENDS',keep_ends)],
        grid=(trials*blocks,1,1),threadgroup=(256,1,1),
        output_shapes=[(trials*maximum,),(trials,),(trials,),
                       (trials*maximum if keep_ends else 1,)],
        output_dtypes=[mx.int64,mx.uint32,mx.uint32,mx.uint32],init_value=0)
    packed=packed.reshape(trials,maximum)
    return packed,consumed,done,ends


def _ptrs_prefix(uniforms,counts,maximum,lam,raw,trials,*,keep_ends,target):
    """Stable accepted prefixes; trailing candidates remain unconsumed."""
    import mlx.core as mx
    candidates=raw//2
    begin,size,probability=_rhs_table(lam,target)
    outputs=_candidates(lam,target)(inputs=[uniforms,probability],
        template=[('B',trials),('CANDIDATES',candidates),('TABLE_BEGIN',begin),('TABLE_SIZE',size)],
        grid=(trials*candidates,1,1),threadgroup=(256,1,1),
        output_shapes=[(trials*candidates,)]*(3 if target=='cython'else 2),
        output_dtypes=[mx.int64,mx.uint32]+([mx.uint32]if target=='cython'else[]))
    values,valid=outputs[:2]
    valid=valid.reshape(trials,candidates)
    if target=='cython':
        # An unused speculative zero must not raise. Stop the prefix before
        # the first zero U, then let the native sampler consume the exact
        # invalid candidates and raise only if the requested draws reach it.
        indices=mx.arange(candidates,dtype=mx.uint32)[None,:]
        first=mx.min(mx.where(outputs[2].reshape(trials,candidates)>0,indices,candidates),axis=1)
        valid=mx.where(indices<first[:,None],valid,0)
    prefix=mx.cumsum(valid,axis=1)
    done=mx.minimum(counts,prefix[:,-1]).astype(mx.uint32)
    selected=(valid>0)&(prefix<=counts[:,None])
    indices=mx.arange(candidates,dtype=mx.uint32)[None,:]
    consumed=mx.max(mx.where(selected,2*(indices+1),0),axis=1).astype(mx.uint32)
    packed,ends=_pack()(inputs=[values,valid.reshape(-1),prefix.reshape(-1),counts],
        template=[('B',trials),('CANDIDATES',candidates),('MAX_COUNT',maximum),
                  ('KEEP_ENDS',keep_ends)],
        grid=(trials*candidates,1,1),threadgroup=(256,1,1),
        output_shapes=[(trials*maximum,),(trials*maximum if keep_ends else 1,)],
        output_dtypes=[mx.int64,mx.uint32],init_value=0)
    packed=packed.reshape(trials,maximum)
    return packed,consumed,done,ends


def constant_poisson(state,maximum,lam,counts=None,*,raw_capacity=None,frame_width=None,target='numpy'):
    """Preserve exact variable consumption, with a device-only exhaustion path."""
    import mlx.core as mx
    from .binary64 import literal
    if target=='numpy':from .expression_poisson import poisson
    elif target=='cpp':
        from .expression_poisson_compiled import poisson as compiled_poisson
        def poisson(s,l,c=None):
            values,end=compiled_poisson(s,l,c)
            return values,end,mx.zeros((s.trials,),mx.uint32)
    else:raise ValueError('Unknown constant Poisson target')
    from .expression_rng_jump import uniform_pool
    from .expression_rng_parallel import _prefix_state
    maximum=index(maximum);lam=float(lam)
    if maximum<0 or maximum*state.trials>=2**32:
        raise ValueError('Poisson capacity exceeds safe Metal indexing')
    if frame_width is not None:
        frame_width=index(frame_width)
        if frame_width<1 or maximum%frame_width:
            raise ValueError('Invalid Poisson checkpoint width')
    if counts is None:counts=mx.full((state.trials,),maximum,mx.uint32)
    if counts.shape!=(state.trials,)or counts.dtype!=mx.uint32:
        raise ValueError('Poisson draw counts must be uint32 per stream')
    lambdas=lambda:mx.full((state.trials,maximum),int(literal(lam)[:-2]),mx.uint64)
    if not maximum or not math.isfinite(lam)or(target=='numpy'and
            (lam<0 or lam>=float(np.iinfo(np.int64).max)-10*math.sqrt(float(np.iinfo(np.int64).max)))):
        return poisson(state,lambdas(),counts)
    if target=='cpp'and lam<0:
        from .expression_rng import uniforms
        if frame_width is None:
            _,end=uniforms(state,maximum,counts)
            return mx.zeros((state.trials,maximum),mx.int32),end,mx.zeros((state.trials,),mx.uint32)
        pool=uniform_pool(state,index(frame_width),maximum//index(frame_width))
        from dataclasses import replace
        return replace(pool,values=mx.zeros(pool.values.shape,mx.int32))
    if not lam:
        if frame_width is not None:return _zero_pool(state,index(frame_width),maximum//index(frame_width),target)
        return mx.zeros((state.trials,maximum),mx.int32 if target=='cpp'else mx.int64),state,mx.zeros((state.trials,),mx.uint32)
    raw=(int(math.ceil(maximum*(1.5*(lam+1)if lam<10 else 3)))+312
         if raw_capacity is None else index(raw_capacity))
    raw=min(raw,64*1024**2//state.trials)
    checkpoint=312*8 if raw>=2**20 else 312
    frames=max(1,(raw+checkpoint-1)//checkpoint);raw=frames*checkpoint
    if raw*state.trials>=2**32 or frames*state.trials*624>=2**32:
        return poisson(state,lambdas(),counts)
    with mx.stream(mx.gpu):
        pool=uniform_pool(state,checkpoint,frames,parts=32 if raw>=2**20 else 1)
        uniforms=mx.contiguous(pool.values.transpose(1,0,2)).reshape(-1)
        if lam<10:
            packed,consumed,done,ends=_product_prefix(uniforms,counts,maximum,lam,raw,state.trials,
                keep_ends=frame_width is not None)
        else:
            packed,consumed,done,ends=_ptrs_prefix(uniforms,counts,maximum,lam,raw,state.trials,
                keep_ends=frame_width is not None,target=target)
        end=_prefix_state(state,pool,consumed)
        tail,end,errors=poisson(end,lambdas(),(counts-done).astype(mx.uint32))
        offsets=mx.maximum(mx.arange(maximum)[None,:]-done[:,None],0)
        output=mx.where(mx.arange(maximum)[None,:]<done[:,None],packed,
                        mx.take_along_axis(tail,offsets,axis=1))
        output=mx.where(mx.arange(maximum)[None,:]<counts[:,None],output,0)
        if target=='cpp':output=output.astype(mx.int32)
        if frame_width is None:return output,end,errors
        return _finish_pool(state,pool,output,ends.reshape(state.trials,maximum),done,lam,index(frame_width),target)


def _zero_pool(state,width,frames,target='numpy'):
    import mlx.core as mx
    from .expression_rng_jump import UniformPool
    words=mx.broadcast_to(state.words.reshape(state.trials,1,624),(state.trials,frames,624))
    positions=mx.broadcast_to(state.position[:,None],(state.trials,frames))
    return UniformPool(mx.zeros((frames,state.trials,width),mx.int32 if target=='cpp'else mx.int64),words,positions,
        state.trials,state.has_gauss,state.gauss,width,frames)


def _finish_pool(initial,raw_pool,values,ends,done,lam,width,target):
    import mlx.core as mx
    from .expression_rng_jump import UniformPool
    from .expression_rng_parallel import _prefix_state
    if target=='cpp':from .expression_poisson_compiled import _poisson_states
    else:from .expression_poisson import _poisson_states
    trials,maximum=values.shape;frames=maximum//width
    counts=mx.broadcast_to(mx.arange(1,frames+1,dtype=mx.uint32)[None,:]*width,(trials,frames))
    effective=mx.minimum(counts,done[:,None])
    consumed=mx.take_along_axis(ends,mx.maximum(effective,1)-1,axis=1)
    consumed=mx.where(effective>0,consumed,0).astype(mx.uint32)
    stream=_prefix_state(initial,raw_pool,consumed)
    stream,_=_poisson_states(stream,(counts-effective).reshape(-1).astype(mx.uint32),lam)
    return UniformPool(mx.contiguous(values.reshape(trials,frames,width).transpose(1,0,2)),
        stream.words.reshape(trials,frames,624),stream.position.reshape(trials,frames),
        trials,initial.has_gauss,initial.gauss,width,frames)


def poisson_pool(state,width,frames,lam,*,raw_capacity=None,target='numpy'):
    """Fixed-rate lookahead with exact state at every completed model frame."""
    width=index(width);frames=index(frames)
    if width<1 or frames<1 or frames*state.trials*624>=2**32:
        raise ValueError('Invalid Poisson pool layout')
    return constant_poisson(state,width*frames,lam,raw_capacity=raw_capacity,frame_width=width,target=target)
