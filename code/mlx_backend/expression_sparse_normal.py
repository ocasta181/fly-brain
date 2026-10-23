"""Exact small selected normal batches with shared-prefix consumption."""
from functools import lru_cache
from dataclasses import replace
import mlx.core as mx


@lru_cache(maxsize=1)
def _kernel():
    from .expression_rng import _normal_source,_MT_HEADER
    from .binary64 import MSL_HEADER
    from .expression_rng_math import NORMAL_MATH_HEADER
    source=_normal_source()
    source=source.replace('uint cached=has_gauss[trial];','ulong consumed=0ul;uint cached=has_gauss[trial];')
    source=source.replace('if(2*accepted>=remaining) {',
        'consumed+=cross?4ul:(2*accepted>=remaining?ulong(end_position-start):4ul*candidates);\n    if(2*accepted>=remaining) {')
    source=source.replace('next_gauss[trial]=saved;', 'next_gauss[trial]=saved;consumed_words[trial]=consumed;')
    return mx.fast.metal_kernel(name='expression_sparse_normal_consumption',
        input_names=['state','position','counts','has_gauss','gauss'],
        output_names=['result','next_state','next_position','next_has_gauss','next_gauss','consumed_words'],
        source=source,header=MSL_HEADER+NORMAL_MATH_HEADER+_MT_HEADER,compile_options={'math_mode':'safe'})


def draw(random,maximum,counts,*,target):
    from .expression_rng import MT19937State,_normal_target
    state=random.stream;trials=state.trials
    result,words,position,has_gauss,gauss,consumed=_kernel()(
        inputs=[state.words,state.position,counts,state.has_gauss,state.gauss],
        template=[('MAX_COUNT',maximum),('STATE_ONLY',False),('CPP_NORMAL',_normal_target(target))],
        grid=(trials*256,1,1),threadgroup=(256,1,1),
        output_shapes=[(trials*maximum,),(trials*624,),(trials,),(trials,),(trials,),(trials,)],
        output_dtypes=[mx.uint64,mx.uint32,mx.uint32,mx.uint32,mx.uint64,mx.uint64])
    stream=MT19937State(words,position,trials,has_gauss,gauss)
    prefix=random.pool.data
    cursor=mx.minimum(prefix.cursor.astype(mx.uint64)+consumed//2,prefix.capacity).astype(mx.uint32)
    pool=replace(random.pool,data=replace(prefix,cursor=cursor,stream=stream))
    return result.reshape(trials,maximum),stream,pool
