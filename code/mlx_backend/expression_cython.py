"""Portable device state for Brian2.8's separate20000-value Cython buffers."""
from dataclasses import dataclass
from operator import index
from functools import lru_cache
import numpy as np

BUFFER_SIZE=20000


@dataclass(frozen=True)
class CythonBuffers:
    uniform: object
    uniform_index: object
    normal: object
    normal_index: object
    # Optional compiled-phase cache: (owning stream, errors, packed fields).
    # Ordinary buffer replacements and public packet restore discard it.
    metadata: object | None = None


def initial_buffers(trials):
    import mlx.core as mx
    return CythonBuffers(mx.zeros((trials,BUFFER_SIZE),mx.uint64),mx.zeros((trials,),mx.uint32),
        mx.zeros((trials,BUFFER_SIZE),mx.uint64),mx.zeros((trials,),mx.uint32))


def from_buffer_snapshots(snapshots):
    """Copy actual values/indices, never a process-local C pointer."""
    import mlx.core as mx
    uniforms=[];normals=[];ui=[];ni=[]
    for snapshot in snapshots:
        for name,target,indices in(('uniform',uniforms,ui),('normal',normals,ni)):
            values=np.asarray(snapshot[name],np.float64);position=index(snapshot[name+'_index'])
            if values.shape!=(BUFFER_SIZE,)or not 0<=position<BUFFER_SIZE:
                raise ValueError('Invalid portable Cython RNG buffer snapshot')
            target.append(values.view(np.uint64).copy());indices.append(position)
    if not uniforms:raise ValueError('At least one Cython buffer snapshot is required')
    return CythonBuffers(mx.array(np.stack(uniforms)),mx.array(ui,mx.uint32),
                         mx.array(np.stack(normals)),mx.array(ni,mx.uint32))


def buffer_snapshots(buffers):
    """Explicit caller-selected checkpoint synchronization."""
    uniform=np.asarray(buffers.uniform).view(np.float64);normal=np.asarray(buffers.normal).view(np.float64)
    ui=np.asarray(buffers.uniform_index);ni=np.asarray(buffers.normal_index)
    return tuple(dict(uniform=u.copy(),uniform_index=int(i),normal=n.copy(),normal_index=int(j))
        for u,i,n,j in zip(uniform,ui,normal,ni))


@dataclass(frozen=True)
class CythonUniformView:
    values: object
    tail: object
    cursor: object
    done: object
    counts: object
    buffer: object
    index: object
    remaining: object
    physical_capacity: int
    metadata: object = None


@lru_cache(maxsize=1)
def _prefix_buffer_kernel():
    import mlx.core as mx
    return mx.fast.metal_kernel(name='expression_cython_prefix_buffer',
        input_names=['previous','pool','tail','cursor','done','refills'],output_names=['latest'],
        source=r"""
        uint gid=thread_position_in_grid.x;if(gid>=B*20000)return;
        uint trial=gid/20000,row=gid%20000;
        if(!refills[trial]){latest[gid]=previous[gid];return;}
        uint offset=(refills[trial]-1)*20000+row;
        latest[gid]=offset<done[trial]?pool[ulong(trial)*POOL_CAPACITY+cursor[trial]+offset]
            :tail[ulong(trial)*PHYSICAL_CAPACITY+offset-done[trial]];
        """,compile_options={'math_mode':'safe'})


def buffered_draws(state,buffers,function,maximum,counts=None,*,sparse=False,_prefix=None,_view=False,_clear_unused=True):
    """Refill complete CPU-sized buffers; return only requested logical draws."""
    import mlx.core as mx
    from .expression_rng import normals
    from .expression_rng_jump import uniform_pool
    from .expression_rng_parallel import _prefix_state
    maximum=index(maximum)
    if function not in('rand','randn'):raise ValueError('Unknown Cython buffered primitive')
    if maximum<0 or maximum*state.trials>=2**32:
        raise ValueError('Cython buffered draw capacity exceeds safe Metal indexing')
    if buffers.uniform.shape!=(state.trials,BUFFER_SIZE)or buffers.normal.shape!=(state.trials,BUFFER_SIZE):
        raise ValueError('Cython buffers must match stream trial count')
    if counts is None:counts=mx.full((state.trials,),maximum,mx.uint32)
    if counts.shape!=(state.trials,)or counts.dtype!=mx.uint32:
        raise ValueError('Cython draw counts must be uint32 per stream')
    if not maximum:return mx.zeros((state.trials,0),mx.uint64),state,buffers
    if sparse:return _sparse_buffered_draws(state,buffers,function,maximum,counts,_clear_unused=_clear_unused)
    capacity=((maximum+BUFFER_SIZE-1)//BUFFER_SIZE)*BUFFER_SIZE
    if capacity*state.trials>=2**32:
        raise ValueError('Cython refill capacity exceeds safe Metal indexing')
    with mx.stream(mx.gpu):
        values=getattr(buffers,'uniform'if function=='rand'else'normal')
        position=getattr(buffers,'uniform_index'if function=='rand'else'normal_index')
        remaining=mx.where(position>0,BUFFER_SIZE-position,0).astype(mx.uint32)
        refill_counts=((mx.maximum(counts.astype(mx.int64)-remaining,0)+BUFFER_SIZE-1)//BUFFER_SIZE).astype(mx.uint32)
        generated_counts=refill_counts*BUFFER_SIZE
        prefix=_prefix
        if prefix is not None:
            from .expression_rng_prefix import prefix_uniforms,prefix_normals
            if function=='rand':
                generated,end,prefix=prefix_uniforms(state,prefix,capacity,generated_counts,_view=_view)
            else:
                if _view:raise ValueError('Normal prefix refills require materialized values')
                generated,end,prefix=prefix_normals(state,prefix,capacity,generated_counts)
            if _view:
                latest,=_prefix_buffer_kernel()(
                    inputs=[values,generated.values,generated.tail,generated.cursor,generated.done,refill_counts],
                    template=[("B",state.trials),("POOL_CAPACITY",generated.values.shape[-1]),("PHYSICAL_CAPACITY",capacity)],
                    grid=(state.trials*BUFFER_SIZE,1,1),threadgroup=(256,1,1),
                    output_shapes=[(state.trials,BUFFER_SIZE)],output_dtypes=[mx.uint64])
                new=CythonBuffers(latest,((position+counts)%BUFFER_SIZE).astype(mx.uint32),buffers.normal,buffers.normal_index)
                view=CythonUniformView(generated.values,generated.tail,generated.cursor,generated.done,counts,
                    values,position,remaining,capacity)
                return view,end,new,prefix
        elif function=='rand':
            frames=(capacity+311)//312
            pool=uniform_pool(state,312,frames,parts=32 if capacity>=2**20 else 1)
            generated=mx.contiguous(pool.values.transpose(1,0,2)).reshape(state.trials,-1)[:,:capacity]
            end=_prefix_state(state,pool,generated_counts)
        else:generated,end=normals(state,capacity,generated_counts)
        offsets=mx.arange(maximum,dtype=mx.int64)[None,:]
        cached=mx.take_along_axis(values,mx.minimum(position[:,None]+offsets,BUFFER_SIZE-1),axis=1)
        fresh=mx.take_along_axis(generated,mx.maximum(offsets-remaining[:,None],0),axis=1)
        output=mx.where(offsets<remaining[:,None],cached,fresh)
        output=mx.where(offsets<counts[:,None],output,0)
        last=(mx.maximum(refill_counts,1)-1)[:,None]*BUFFER_SIZE+mx.arange(BUFFER_SIZE)[None,:]
        latest=mx.take_along_axis(generated,last,axis=1)
        latest=mx.where(refill_counts[:,None]>0,latest,values)
        next_index=((position+counts)%BUFFER_SIZE).astype(mx.uint32)
        if function=='rand':new=CythonBuffers(latest,next_index,buffers.normal,buffers.normal_index)
        else:new=CythonBuffers(buffers.uniform,buffers.uniform_index,latest,next_index)
        return (output,end,new,prefix)if _prefix is not None else(output,end,new)


@lru_cache(maxsize=1)
def _sparse_buffered_kernel():
    """Consume cached chunks and generate a physical refill only when needed."""
    import mlx.core as mx
    from .binary64 import MSL_HEADER
    from .expression_rng import _MT_HEADER
    from .expression_rng_math import NORMAL_MATH_HEADER
    from .expression_compiled import _CONTEXT
    return mx.fast.metal_kernel(name='expression_cython_selected_buffered_draws',
        input_names=['state','position','has_gauss','gauss','buffer','index','counts'],
        output_names=['values','next_state','next_position','next_has_gauss','next_gauss',
                      'next_buffer','next_index','refilled'],
        header=MSL_HEADER+NORMAL_MATH_HEADER+_MT_HEADER+_CONTEXT+compiled_header(),source=r'''
        uint lane=thread_position_in_threadgroup.x,trial=threadgroup_position_in_grid.x;
        threadgroup uint mt[624],boundary[4],warps[8],end;threadgroup ulong pair;
        for(uint j=lane;j<624;j+=256)mt[j]=state[trial*624+j];
        CompiledRng rng;rng.mt=mt;rng.pos=position[trial];rng.cached=has_gauss[trial];
        rng.saved=gauss[trial];rng.lane=lane;rng.error=0;rng.boundary=boundary;
        rng.warps=warps;rng.end=&end;rng.pair=&pair;
        rng.ubuf=next_buffer+trial*20000;rng.nbuf=rng.ubuf;
        uint offset=index[trial],done=0;bool generated=false;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        while(done<counts[trial]) {
            if(!offset){
                if(NORMAL)cython_normal_refill(rng);else cython_uniform_refill(rng);
                generated=true;
            }
            uint size=min(20000-offset,counts[trial]-done);
            for(uint j=lane;j<size;j+=256)values[trial*MAX_COUNT+done+j]=
                generated?rng.ubuf[offset+j]:buffer[trial*20000+offset+j];
            threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
            done+=size;offset=(offset+size)%20000;
        }
        for(uint j=lane;j<624;j+=256)next_state[trial*624+j]=mt[j];
        if(lane==0){next_position[trial]=rng.pos;next_has_gauss[trial]=rng.cached;
            next_gauss[trial]=rng.saved;next_index[trial]=offset;refilled[trial]=uint(generated);}
        ''',compile_options={'math_mode':'safe'})


def _sparse_buffered_draws(state,buffers,function,maximum,counts,*,_clear_unused=True):
    import mlx.core as mx
    from .expression_rng import MT19937State
    name='normal'if function=='randn'else'uniform';old=getattr(buffers,name)
    with mx.stream(mx.gpu):
        values,words,pos,cache,saved,fresh,offset,refilled=_sparse_buffered_kernel()(
            inputs=[state.words,state.position,state.has_gauss,state.gauss,old,getattr(buffers,name+'_index'),counts],
            template=[('MAX_COUNT',maximum),('NORMAL',function=='randn')],
            grid=(state.trials*256,1,1),threadgroup=(256,1,1),
            output_shapes=[(state.trials*maximum,),(state.trials*624,),(state.trials,),(state.trials,),
                           (state.trials,),(state.trials*BUFFER_SIZE,),(state.trials,),(state.trials,)],
            output_dtypes=[mx.uint64,mx.uint32,mx.uint32,mx.uint32,mx.uint64,mx.uint64,mx.uint32,mx.uint32],
            **({'init_value':0}if _clear_unused else{}))
        latest=mx.where(refilled[:,None]>0,fresh.reshape(state.trials,BUFFER_SIZE),old)
        new=(CythonBuffers(latest,offset,buffers.normal,buffers.normal_index)if function=='rand'else
             CythonBuffers(buffers.uniform,buffers.uniform_index,latest,offset))
        return values.reshape(state.trials,maximum),MT19937State(words,pos,state.trials,cache,saved),new


def compiled_header():
    """Reuse the exact cooperative sampler for CPU-sized normal refills."""
    from .expression_rng import _normal_source
    body=_normal_source()
    body=body[body.index('    uint pos='):body.index('    for(uint j=lane;j<624;j+=256) next_state')]
    body=body.replace('position[trial]','rng.pos').replace('counts[trial]','20000u')
    body=body.replace('has_gauss[trial]','rng.cached').replace('gauss[trial]','rng.saved')
    body=body.replace('    for(uint j=lane;j<624;j+=256) mt[j]=state[trial*624+j];','')
    body=body.replace('trial*MAX_COUNT+','').replace('trial*MAX_COUNT]','0]')
    body=body.replace('CPP_NORMAL','false').replace('STATE_ONLY','false')
    body=body.replace('end_position','(*rng.end)').replace('saved_gauss','(*rng.pair)')
    normal=r'''
    inline void cython_normal_refill(thread CompiledRng& rng) {
        uint lane=rng.lane;threadgroup uint* mt=rng.mt;
        threadgroup uint* boundary=rng.boundary;threadgroup uint* warp_counts=rng.warps;
        device ulong* result=rng.nbuf;
    '''+body+r'''
        rng.pos=pos;rng.cached=cached;rng.saved=saved;
        threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
    }
    inline ulong compiled_normal(thread CompiledRng& rng) {
        if(rng.error)return 0ul;
        if(!rng.ni)cython_normal_refill(rng);
        ulong value=rng.nbuf[rng.ni];if(++rng.ni==20000)rng.ni=0;return value;
    }
    '''
    uniform=r'''
    inline void cython_uniform_refill(thread CompiledRng& rng) {
        uint lane=rng.lane,pos=rng.pos,begin=0;
        threadgroup uint* mt=rng.mt;
        while(begin<20000) {
            bool carry=pos==623;uint first=carry?mt[623]:0u;if(carry)pos=624;
            if(pos==624){mt_twist(mt,lane);pos=0;}
            if(carry){if(lane==0)rng.ubuf[begin]=mt_uniform_bits(first,mt[0]);pos=1;++begin;}
            uint available=min((624-pos)/2,20000-begin);
            for(uint j=lane;j<available;j+=256)rng.ubuf[begin+j]=mt_uniform_bits(mt[pos+2*j],mt[pos+2*j+1]);
            threadgroup_barrier(mem_flags::mem_threadgroup);begin+=available;pos+=2*available;
        }
        rng.pos=pos;threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
    }
    inline ulong compiled_rand(thread CompiledRng& rng) {
        if(rng.error)return 0ul;
        if(!rng.ui)cython_uniform_refill(rng);
        ulong value=rng.ubuf[rng.ui];if(++rng.ui==20000)rng.ui=0;return value;
    }
    '''
    division=r'''
    inline ulong compiled_div(ulong a,ulong b,thread CompiledRng& rng) {
        if(rng.error)return 0ul;
        if(d_eq(b,0ul)){rng.error|=2u;return 0ul;}
        return d_div(a,b);
    }
    '''
    return uniform+normal+division


@dataclass(frozen=True)
class BufferedPool:
    """Logical lookahead backed by whole physical Cython refills."""
    values: object
    generated: object
    initial: object
    buffers: object
    function: str
    width: int
    frames: int
    trials: int

    def arrays(self):
        p=self.generated;s=self.initial;b=self.buffers
        return (self.values,p.values,p.words,p.positions,p.has_gauss,p.gauss,
            s.words,s.position,s.has_gauss,s.gauss,b.uniform,b.uniform_index,b.normal,b.normal_index)

    def state_at(self,frame):
        import mlx.core as mx
        frame=index(frame)
        if not 0<=frame<self.frames:raise IndexError('Buffered checkpoint frame out of range')
        return self.states_at(mx.full((self.trials,1),(frame+1)*self.width,mx.uint32))

    def states_at(self,counts):
        """Dynamic logical prefixes, including several checkpoints per trial."""
        import mlx.core as mx
        from .expression_rng import MT19937State
        counts=counts.reshape(self.trials,-1);frames=counts.shape[1];rows=self.trials*frames
        if rows*BUFFER_SIZE>=2**32:raise ValueError('Buffered checkpoint layout exceeds safe Metal indexing')
        name='uniform'if self.function=='rand'else'normal'
        position=getattr(self.buffers,name+'_index')[:,None]
        remaining=mx.where(position>0,BUFFER_SIZE-position,0)
        refills=((mx.maximum(counts.astype(mx.int64)-remaining,0)+BUFFER_SIZE-1)//BUFFER_SIZE).astype(mx.uint32)
        selected=mx.maximum(refills,1)-1;pool=self.generated
        repeat=lambda array:mx.broadcast_to(array[:,None],(self.trials,frames)).reshape(-1)
        pick=lambda array:mx.take_along_axis(array,selected,axis=1).reshape(-1)
        words=mx.take_along_axis(pool.words,selected[:,:,None],axis=1)
        words=mx.where(refills[:,:,None]>0,words,self.initial.words.reshape(self.trials,1,624)).reshape(-1)
        pos=mx.where(refills.reshape(-1)>0,pick(pool.positions),repeat(self.initial.position))
        cache=repeat(pool.has_gauss)if pool.has_gauss.ndim==1 else pick(pool.has_gauss)
        saved=repeat(pool.gauss)if pool.gauss.ndim==1 else pick(pool.gauss)
        cache=mx.where(refills.reshape(-1)>0,cache,repeat(self.initial.has_gauss))
        saved=mx.where(refills.reshape(-1)>0,saved,repeat(self.initial.gauss))
        generated=pool.values.transpose(1,0,2)
        latest=mx.take_along_axis(generated,selected[:,:,None],axis=1)
        latest=mx.where(refills[:,:,None]>0,latest,getattr(self.buffers,name)[:,None,:]).reshape(rows,BUFFER_SIZE)
        next_index=((position+counts)%BUFFER_SIZE).reshape(-1).astype(mx.uint32)
        other='normal'if self.function=='rand'else'uniform'
        previous=mx.broadcast_to(getattr(self.buffers,other)[:,None,:],(self.trials,frames,BUFFER_SIZE)).reshape(rows,BUFFER_SIZE)
        previous_index=repeat(getattr(self.buffers,other+'_index'))
        if self.function=='rand':buffers=CythonBuffers(latest,next_index,previous,previous_index)
        else:buffers=CythonBuffers(previous,previous_index,latest,next_index)
        return MT19937State(words,pos,rows,cache,saved),buffers


def buffered_pool(state,buffers,function,width,frames):
    """Speculative physical refills never enter the observable checkpoint."""
    import mlx.core as mx
    from .expression_rng_jump import uniform_pool
    from .expression_rng_parallel import normal_pool
    width=index(width);frames=index(frames);maximum=width*frames
    if width<1 or frames<1 or maximum*state.trials>=2**32:
        raise ValueError('Invalid Cython buffered pool layout')
    if function not in('rand','randn'):raise ValueError('Unknown Cython buffered pool function')
    refills=(maximum+BUFFER_SIZE-1)//BUFFER_SIZE
    if refills*BUFFER_SIZE*state.trials>=2**32:raise ValueError('Cython physical pool exceeds safe Metal indexing')
    pool=(uniform_pool(state,BUFFER_SIZE,refills)if function=='rand'else normal_pool(state,BUFFER_SIZE,refills))
    fresh=mx.contiguous(pool.values.transpose(1,0,2)).reshape(state.trials,-1)
    name='uniform'if function=='rand'else'normal';position=getattr(buffers,name+'_index')
    remaining=mx.where(position>0,BUFFER_SIZE-position,0)
    offsets=mx.arange(maximum,dtype=mx.int64)[None,:]
    old=mx.take_along_axis(getattr(buffers,name),mx.minimum(position[:,None]+offsets,BUFFER_SIZE-1),axis=1)
    new=mx.take_along_axis(fresh,mx.maximum(offsets-remaining[:,None],0),axis=1)
    values=mx.where(offsets<remaining[:,None],old,new)
    return BufferedPool(mx.contiguous(values.reshape(state.trials,frames,width).transpose(1,0,2)),
        pool,state,buffers,function,width,frames,state.trials)


@lru_cache(maxsize=1)
def _poisson_kernel():
    import mlx.core as mx
    from .binary64 import MSL_HEADER
    from .expression_rng import _MT_HEADER
    from .expression_rng_math import NORMAL_MATH_HEADER,POISSON_EXP_HEADER
    from .expression_compiled import _CONTEXT
    from .expression_poisson_compiled import source
    return mx.fast.metal_kernel(name='expression_cython_buffered_poisson',
        input_names=['state','position','uniform','index','lambdas','counts','errors'],
        output_names=['values','next_state','next_position','next_uniform','next_index','next_errors'],
        header=MSL_HEADER+NORMAL_MATH_HEADER+POISSON_EXP_HEADER+_MT_HEADER+_CONTEXT+compiled_header()+
            source('thread CompiledRng& rng','compiled_rand(rng)',target='cython',
                   error='rng.error|=2u;',check='if(rng.error)return 0;'),source=r'''
        uint lane=thread_position_in_threadgroup.x,trial=threadgroup_position_in_grid.x;
        threadgroup uint mt[624];
        for(uint j=lane;j<624;j+=256)mt[j]=state[trial*624+j];
        for(uint j=lane;j<20000;j+=256)next_uniform[trial*20000+j]=uniform[trial*20000+j];
        CompiledRng rng;rng.mt=mt;rng.pos=position[trial];rng.lane=lane;
        rng.ubuf=next_uniform+trial*20000;rng.ui=index[trial];rng.error=errors[trial];
        threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
        ulong scalar_lambda=LAMBDA_SCALAR?lambdas[0]:0ul;
        for(uint j=0;j<counts[trial]&&!rng.error;++j) {
            int value=compiled_poisson(LAMBDA_SCALAR?scalar_lambda:lambdas[trial*MAX_COUNT+j],rng);
            if(!STATE_ONLY&&!rng.error&&lane==0)values[trial*MAX_COUNT+j]=value;
        }
        for(uint j=lane;j<624;j+=256)next_state[trial*624+j]=mt[j];
        if(lane==0){next_position[trial]=rng.pos;next_index[trial]=rng.ui;next_errors[trial]=rng.error;}
        ''',compile_options={'math_mode':'safe'})


def poisson(state,buffers,lambdas,counts=None,*,state_only=False,scalar=False,errors=None):
    """Stock compiled Cython Poisson, including physical refills and errors."""
    import mlx.core as mx
    from .expression_rng import MT19937State
    if scalar:
        if lambdas.shape!=(1,)or counts is None:raise ValueError('Scalar Poisson needs one rate and per-trial counts')
        maximum=1
    else:
        if lambdas.ndim!=2 or lambdas.shape[0]!=state.trials:raise ValueError('Invalid buffered Poisson rates')
        maximum=lambdas.shape[1]
    if lambdas.dtype!=mx.uint64 or maximum*state.trials>=2**32:raise ValueError('Invalid buffered Poisson rate storage')
    if counts is None:counts=mx.full((state.trials,),maximum,mx.uint32)
    if counts.shape!=(state.trials,)or counts.dtype!=mx.uint32:raise ValueError('Invalid buffered Poisson counts')
    if buffers.uniform.shape!=(state.trials,BUFFER_SIZE):raise ValueError('Uniform buffers must match stream trial count')
    if errors is None:errors=mx.zeros((state.trials,),mx.uint32)
    if errors.shape!=(state.trials,)or errors.dtype!=mx.uint32:raise ValueError('Invalid buffered Poisson errors')
    if not maximum:return mx.zeros((state.trials,0),mx.int32),state,buffers,errors
    values,words,pos,uniform,ui,error=_poisson_kernel()(
        inputs=[state.words,state.position,buffers.uniform.reshape(-1),buffers.uniform_index,lambdas.reshape(-1),counts,errors],
        template=[('MAX_COUNT',maximum),('LAMBDA_SCALAR',scalar),('STATE_ONLY',state_only)],
        grid=(256*state.trials,1,1),threadgroup=(256,1,1),
        output_shapes=[(1 if state_only else state.trials*maximum,),(state.trials*624,),(state.trials,),
            (state.trials*BUFFER_SIZE,),(state.trials,),(state.trials,)],
        output_dtypes=[mx.int32,mx.uint32,mx.uint32,mx.uint64,mx.uint32,mx.uint32],init_value=0)
    return (None if state_only else values.reshape(state.trials,maximum),
        MT19937State(words,pos,state.trials,state.has_gauss,state.gauss),
        CythonBuffers(uniform.reshape(state.trials,BUFFER_SIZE),ui,buffers.normal,buffers.normal_index),error)
