"""Original N=1 inputs consume the model expression's native RNG at synapses."""
from dataclasses import replace
from functools import lru_cache
import math
import re
import numpy as np
import mlx.core as mx


def import_random(prepared,device):
    from .brian_errors import RandomSnapshot
    from .expression_runtime import initial
    snapshot=RandomSnapshot(device)
    if prepared.model.random_target=='cython':
        portable={'numpy_state':snapshot.numpy}
        for name,source in(('uniform','rand'),('normal','randn')):
            index,values=snapshot.buffers[source]
            portable[name]=np.zeros(20000,np.float64)if values is None else values
            # Index zero refills before reading. This impossible RNG output
            # records an absent CPython allocation without a host step read.
            if values is None:portable[name].view(np.uint64)[0]=np.uint64(0xffffffffffffffff)
            portable[name+'_index']=index
    else:portable=snapshot.numpy
    return initial(prepared,1,[portable])


def export_random(random,device):
    from .brian_errors import RandomSnapshot
    from .expression_runtime import snapshots
    value=snapshots(random)[0]
    if random.buffers is None:np.random.set_state(value);return
    snapshot=RandomSnapshot(device);snapshot.numpy=value['numpy_state']
    for name,source in(('uniform','rand'),('normal','randn')):
        index=value[name+'_index'];values=value[name]
        absent=(snapshot.buffers[source][1]is None and index==0
                and values.view(np.uint64)[0]==np.uint64(0xffffffffffffffff))
        snapshot.buffers[source]=(index,None if absent else values)
    snapshot.restore()


@lru_cache(maxsize=3)
def _kernel(cython,prefix=False):
    from .binary64 import MSL_HEADER
    from .expression_rng import _MT_HEADER
    from .expression_compiled import _CONTEXT,_NORMAL_CPP
    from .expression_rng_math import NORMAL_MATH_HEADER
    from .expression_cython import compiled_header
    from .expression_rng_prefix import _RESTORE_INPUT_NAMES,_restore_source
    restore=''
    # Replace only the standalone fallback names, not prefix_position inputs.
    if prefix:
        restore=_restore_source(256).replace('pos=position[trial];','pos=uint(packet[trial*630+624]);').replace('mt[j]=state[trial*624+j];','mt[j]=uint(packet[trial*630+j]);')
    source=f"""
        uint lane=thread_position_in_threadgroup.x,trial=threadgroup_position_in_grid.x;
        threadgroup uint mt[624],boundary[4],warps[8],end;threadgroup ulong pair;
        for(uint j=lane;j<624;j+=256)mt[j]=uint(packet[trial*630+j]);
        uint pos=uint(packet[trial*630+624]);
        {restore}
        CompiledRng rng;rng.mt=mt;rng.pos=pos;
        rng.cached=uint(packet[trial*630+625]);rng.saved=packet[trial*630+626];
        rng.error=uint(packet[trial*630+627]);rng.lane=lane;
        rng.boundary=boundary;rng.warps=warps;rng.end=&end;rng.pair=&pair;
        {'''for(uint j=lane;j<40000;j+=256)next_buffers[trial*40000+j]=buffers[trial*40000+j];
        rng.ubuf=next_buffers+trial*40000;rng.nbuf=rng.ubuf+20000;
        rng.ui=uint(packet[trial*630+628]);rng.ni=uint(packet[trial*630+629]);
        threadgroup_barrier(mem_flags::mem_device);'''if cython else'rng.ui=0;rng.ni=0;'}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {"for(uint j=0;j<prefix_remaining[trial];j++){ulong value=compiled_rand(rng);if(!lane)uniform_tail[trial*MAX_COUNT+j]=value;}"if prefix else""}
        uint draws=0;
        for(uint channel=0;channel<CHANNELS;channel++) {{
            uint event=2u;
            if(!rng.error&&active[channel]&&flags[trial*NEURONS+targets[channel]]) {{
                int X=0;ulong px=qn[channel],U=compiled_rand(rng);++draws;
                while(d_gt(U,px)) {{
                    ++X;
                    if(X>1){{X=0;px=qn[channel];U=compiled_rand(rng);++draws;}}
                    else{{U=d_sub(U,px);px=d_div(d_mul(probability[channel],px),complement[channel]);}}
                }}
                event=reverse[channel]?1-X:X;
            }}
            if(!lane)events[trial*CHANNELS+channel]=uchar(event);
        }}
        for(uint j=lane;j<624;j+=256)next_packet[trial*630+j]=ulong(mt[j]);
        if(!lane){{consumed[trial]=draws;next_packet[trial*630+624]=rng.pos;next_packet[trial*630+625]=rng.cached;
        next_packet[trial*630+626]=rng.saved;next_packet[trial*630+627]=rng.error;
        next_packet[trial*630+628]=rng.ui;next_packet[trial*630+629]=rng.ni;}}
        """
    if prefix:
        for offset,name in ((624,'position'),(625,'cached'),(626,'saved'),(627,'error')):
            source=source.replace(f'packet[trial*630+{offset}]',f'{name}[trial]')
        source=source.replace('next_packet[trial*630+j]=ulong(mt[j]);','next_words[trial*624+j]=mt[j];')
        source=source.replace('packet[trial*630+j]','words[trial*624+j]')
        for offset,name in ((624,'position'),(625,'cached'),(626,'saved'),(627,'error')):
            source=source.replace(f'next_packet[trial*630+{offset}]',f'next_{name}[trial]')
        source=source.replace('next_packet[trial*630+628]=rng.ui;next_packet[trial*630+629]=rng.ni;','')
    return mx.fast.metal_kernel(name='expression_original_poisson_prefix'if prefix else'expression_original_poisson',
        input_names=(['words','position','cached','saved','error']if prefix else['packet'])+
            ['flags','targets','active','qn','probability','complement','reverse']+(['buffers']if cython else[])+([*_RESTORE_INPUT_NAMES,'prefix_remaining']if prefix else[]),
        output_names=(['events','next_words','next_position','next_cached','next_saved','next_error','consumed','uniform_tail']if prefix else
            ['events','next_packet','consumed']+(['next_buffers']if cython else[])),
        header=MSL_HEADER+NORMAL_MATH_HEADER+_MT_HEADER+_CONTEXT+(compiled_header()if cython else _NORMAL_CPP),
        source=source,compile_options={'math_mode':'safe'})


@lru_cache(maxsize=1)
def _cython_first_mass_kernel():
    from .binary64 import MSL_HEADER
    return mx.fast.metal_kernel(name='cython_input_first_mass',
        input_names=['qn','probability','complement'],output_names=['mass'],
        header=MSL_HEADER,source='''
        uint channel=thread_position_in_grid.x;
        if(channel<CHANNELS)mass[channel]=d_div(d_mul(probability[channel],qn[channel]),complement[channel]);
        ''',compile_options={'math_mode':'safe'})


@lru_cache(maxsize=2)
def _cython_prefix_kernel(pending_normal=False,normal_blocks=False):
    """Restore physical threshold refills and run original inputs together."""
    from .binary64 import MSL_HEADER
    from .expression_rng import _MT_HEADER
    from .expression_compiled import _CONTEXT
    from .expression_rng_math import NORMAL_MATH_HEADER
    from .expression_cython import compiled_header
    from .expression_rng_prefix import _RESTORE_INPUT_NAMES,_restore_source
    source=f'''
        uint lane=thread_position_in_threadgroup.x,trial=threadgroup_position_in_grid.x;
        threadgroup uint mt[624],boundary[4],warps[8],end;threadgroup ulong pair;
        uint pos=position[trial];
        {_restore_source(256)}
        CompiledRng rng;rng.mt=mt;rng.pos=pos;rng.cached=has_gauss[trial];rng.saved=gauss[trial];
        rng.error=errors[trial];rng.lane=lane;rng.boundary=boundary;rng.warps=warps;rng.end=&end;rng.pair=&pair;
        rng.ubuf=next_buffer+trial*20000;rng.nbuf=rng.ubuf;rng.ni=0;
        // Prefix consumption is always a whole physical refill. Any remaining
        // refills use the unchanged cooperative sampler from this exact MT state.
        for(uint offset=0;offset<draw_counts[trial*3+1]-draw_counts[trial*3+2];offset+=20000){{
            cython_uniform_refill(rng);
            for(uint j=lane;j<20000;j+=256)tail[ulong(trial)*PHYSICAL_CAPACITY+offset+j]=rng.ubuf[j];
            threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
        }}
        if(draw_counts[trial*3+1]==draw_counts[trial*3+2]){{
            for(uint j=lane;j<20000;j+=256)rng.ubuf[j]=draw_counts[trial*3+1]
                ?pool[ulong(trial)*POOL_CAPACITY+prefix_cursor[trial]+draw_counts[trial*3+1]-20000+j]
                :buffer[trial*20000+j];
        }}
        rng.ui=(index[trial]+draw_counts[trial*3])%20000;
        threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
        uint draws=0;
        for(uint channel=0;channel<CHANNELS;channel++){{
            uint event=2u;
            if(!rng.error&&active[channel]&&flags[trial*NEURONS+targets[channel]]){{
                int X=0;ulong px=qn[channel],U=compiled_rand(rng);++draws;
                while(d_gt(U,px)){{
                    ++X;
                    if(X>1){{X=0;px=qn[channel];U=compiled_rand(rng);++draws;}}
                    else{{U=d_sub(U,px);px=d_div(d_mul(probability[channel],px),complement[channel]);}}
                }}
                event=reverse[channel]?1-X:X;
            }}
            if(!lane)events[trial*CHANNELS+channel]=uchar(event);
        }}
        for(uint j=lane;j<624;j+=256)next_words[trial*624+j]=mt[j];
        if(!lane){{next_position[trial]=rng.pos;next_cached[trial]=rng.cached;next_saved[trial]=rng.saved;
            next_index[trial]=rng.ui;consumed[trial]=draws;}}
        '''
    inputs=['state','position','has_gauss','gauss','errors','buffer','index','draw_counts','pool','flags','targets','active','qn','probability','complement','reverse',*_RESTORE_INPUT_NAMES]
    outputs=['events','next_words','next_position','next_cached','next_saved','next_buffer','next_index','consumed','tail']
    if pending_normal:
        import re
        # X==1 always starts from immutable qn; later retries restore qn.
        source=source.replace('d_div(d_mul(probability[channel],px),complement[channel])','probability[channel]')
        inputs=[name for name in ['state','gauss',*inputs[5:],'normal_buffer']if name!='prefix_consumed']
        outputs=[name for name in outputs if name!='next_index']+['normal_draws','next_normal','normal_refilled','view_metadata']
        restore=_restore_source(256)
        source=source.replace(restore,'')
        source=source.replace('        rng.ubuf=next_buffer+trial*20000;',r'''        for(uint j=lane;j<624;j+=256)mt[j]=state[trial*624+j];
        uint normal_offset=draw_counts[trial*3+6],normal_done=0;
        uint normal_count=errors[trial]?0u:draw_counts[trial*3+7];bool normal_generated=false;
        rng.nbuf=next_normal+trial*20000;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        while(normal_done<normal_count){
            if(!normal_offset){cython_normal_refill(rng);normal_generated=true;}
            uint size=min(20000-normal_offset,normal_count-normal_done);
            for(uint j=lane;j<size;j+=256)normal_draws[trial*NEURONS+normal_done+j]=
                normal_generated?rng.nbuf[normal_offset+j]:normal_buffer[trial*20000+normal_offset+j];
            threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
            normal_done+=size;normal_offset=(normal_offset+size)%20000;
        }
        pos=rng.pos;
        if(!normal_generated){'''+restore+r'''}
        rng.pos=pos;
        uint effective_done=normal_generated?0u:draw_counts[trial*3+2];
        rng.ubuf=next_buffer+trial*20000;''')
        # Keep physical refill and original-input order after the prior normal.
        source=source.replace('draw_counts[trial*3+2]','effective_done')
        source=source.replace('uint effective_done=normal_generated?0u:effective_done;',
            'uint effective_done=normal_generated?0u:draw_counts[trial*3+2];')
        for name,column in (('position',3),('has_gauss',4),('errors',5)):
            source=re.sub(r'\b'+name+r'\[trial\]',f'draw_counts[trial*3+{column}]',source)
        source=source.replace('draw_counts[trial*3','draw_counts[trial*8')
        source=source.replace('next_index[trial]=rng.ui;','')
        source=source.replace('consumed[trial]=draws;',
            'consumed[trial*4]=draws;consumed[trial*4+1]=normal_offset;'
            'consumed[trial*4+2]=effective_done;consumed[trial*4+3]=rng.ui;'
            'normal_refilled[trial]=normal_generated;')
        # Physical-buffer bookkeeping belongs to this kernel, not a chain of
        # scalar MLX launches between adjacent model phases.
        for old,new in ((6,3),(7,4),(3,0),(4,1),(5,2)):
            source=source.replace(f'draw_counts[trial*8+{old}]',f'draw_counts[trial*5+{new}]')
        source=source.replace('draw_counts[trial*8+1]','physical').replace('draw_counts[trial*8+2]','planned_done')
        source=source.replace('draw_counts[trial*8]','threshold_count')
        source=source.replace('prefix_consumed[trial]','prefix_consumed_value')
        source=source.replace('        uint pos=',r'''        uint threshold_count=draw_counts[trial*5+2]?0u:uint(NEURONS);
        uint original_index=index[trial],remaining=original_index?20000-original_index:0;
        uint physical=((threshold_count>remaining?threshold_count-remaining:0u)+19999u)/20000u*20000u;
        uint planned_done=min(physical,(POOL_CAPACITY-prefix_cursor[trial])/20000u*20000u);
        uint prefix_consumed_value=prefix_cursor[trial]+planned_done;
        uint pos=''',1)
        source=source.replace('consumed[trial*4','consumed[trial*5')
        source=source.replace('normal_refilled[trial]=normal_generated;',r'''normal_refilled[trial]=normal_generated;
            uint after_threshold=(original_index+threshold_count)%20000u;
            uint input_available=after_threshold?20000u-after_threshold:0u;
            uint input_physical=((draws>input_available?draws-input_available:0u)+19999u)/20000u*20000u;
            consumed[trial*5+4]=normal_generated?POOL_CAPACITY:
                uint(min(ulong(prefix_cursor[trial])+physical+input_physical,ulong(POOL_CAPACITY)));
            view_metadata[trial*5]=prefix_cursor[trial];view_metadata[trial*5+1]=effective_done;
            view_metadata[trial*5+2]=threshold_count;view_metadata[trial*5+3]=original_index;
            view_metadata[trial*5+4]=remaining;''')

    if normal_blocks:
        assert pending_normal
        inputs=[name if name!='flags'else'normal_blocks'for name in inputs]
        source=source.replace('flags[trial*NEURONS+targets[channel]]','true')
        source=source.replace('draw_counts[trial*5+4]','normal_total')
        source=source.replace('draw_counts[trial*5','draw_counts[trial*4')
        source=source.replace('        uint threshold_count=',r'''        uint normal_total=0;
        for(uint base=0;base<NORMAL_BLOCKS;base+=256){
            uint block=base+lane;
            uint value=block<NORMAL_BLOCKS?normal_blocks[trial*NORMAL_BLOCKS+block]:0u;
            uint prefix=simd_prefix_exclusive_sum(value)+value;
            uint subtotal=simd_sum(value);
            if(lane%32==0)warps[lane/32]=subtotal;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint total=0;
            for(uint warp=0;warp<8;++warp){
                if(warp<lane/32)prefix+=warps[warp];
                total+=warps[warp];
            }
            if(block<NORMAL_BLOCKS)view_metadata[5*TRIALS+trial*NORMAL_BLOCKS+block]=normal_total+prefix;
            normal_total+=total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint threshold_count=''',1)
        source+=r'''
        if(!lane){
            uint at=TRIALS*(5+NORMAL_BLOCKS)+trial*4;
            view_metadata[at]=rng.pos;view_metadata[at+1]=rng.cached;
            view_metadata[at+2]=rng.error;view_metadata[at+3]=normal_offset;
        }
        '''

    return mx.fast.metal_kernel(name='expression_cython_threshold_inputs',input_names=inputs,output_names=outputs,
        header=MSL_HEADER+NORMAL_MATH_HEADER+_MT_HEADER+_CONTEXT+compiled_header(),source=source,compile_options={'math_mode':'safe'})


def _cython_prefix_draw(owner,prepared,state,random,active,counts,pending_normal=None):
    from .expression_rng import MT19937State
    from .expression_rng_prefix import _counts,_restore_arguments
    from .expression_cython import CythonBuffers,CythonUniformView
    stream=random.stream;buffers=random.buffers;prefix=random.pool.data;trials=stream.trials
    maximum,counts=_counts(stream,prepared.neurons,counts,prefix)
    capacity=((maximum+19999)//20000)*20000
    if capacity*trials>=2**32:raise ValueError('Cython refill capacity exceeds safe Metal indexing')
    remaining=physical=done=None
    if pending_normal is None:
        remaining=mx.where(buffers.uniform_index>0,20000-buffers.uniform_index,0).astype(mx.uint32)
        physical=((mx.maximum(counts.astype(mx.int64)-remaining,0)+19999)//20000).astype(mx.uint32)*20000
        done=mx.minimum(physical,((prefix.capacity-prefix.cursor)//20000)*20000).astype(mx.uint32)
    restore,templates=_restore_arguments(prefix,prefix.cursor if pending_normal is not None else prefix.cursor+done)
    block_counts=pending_normal is not None and pending_normal.ndim==2
    conditional=prepared.program.variables['v'].conditional_write
    if block_counts:
        assert not conditional and pending_normal.shape==(trials,prepared.neurons//256)
        flags=pending_normal.reshape(-1)
    else:
        flags=state.values[conditional]if conditional else mx.ones((trials*prepared.neurons,),mx.bool_)
        flags=flags&mx.repeat(active,prepared.neurons)
    shared_arguments=[prefix.data.values,flags,owner.targets,mx.array(owner.active,mx.bool_),owner.qn,
        owner.first_mass if pending_normal is not None and owner.first_mass is not None else owner.probability,owner.complement,owner.reverse]
    if pending_normal is None:
        arguments=[stream.words,stream.position,stream.has_gauss,stream.gauss,random.errors,buffers.uniform,
            buffers.uniform_index,mx.stack([counts,physical,done],axis=1),*shared_arguments,*restore]
    else:
        metadata=buffers.metadata
        if block_counts and metadata is not None and metadata[0] is stream and metadata[1] is random.errors:
            draw_metadata=metadata[2]
        else:
            draw_metadata=mx.stack([stream.position,stream.has_gauss,random.errors,buffers.normal_index]+
                ([]if block_counts else[pending_normal]),axis=1)
        arguments=[stream.words,stream.gauss,buffers.uniform,buffers.uniform_index,
            draw_metadata,
            *shared_arguments,*restore[:-1],buffers.normal]
    shapes=[(trials*owner.channels,),(trials*624,),(trials,),(trials,),(trials,),
        (trials,20000),(trials,),(trials,),(trials,capacity)]
    dtypes=[mx.uint8,mx.uint32,mx.uint32,mx.uint32,mx.uint64,mx.uint64,mx.uint32,mx.uint32,mx.uint64]
    if pending_normal is not None:
        shapes=[*shapes[:6],(trials,5),shapes[8],(trials,prepared.neurons),(trials,20000),(trials,),(trials,5)]
        dtypes=[*dtypes[:6],mx.uint32,mx.uint64,mx.uint64,mx.uint64,mx.bool_,mx.uint32]
    if block_counts:
        shapes[-1]=(trials*(9+prepared.neurons//256),)
        templates += [('TRIALS',trials),('NORMAL_BLOCKS',prepared.neurons//256)]
    kernel=(_cython_prefix_kernel(True,True)if block_counts else _cython_prefix_kernel(pending_normal is not None))
    result=kernel(inputs=arguments,
        template=[('NEURONS',prepared.neurons),('CHANNELS',owner.channels),('PHYSICAL_CAPACITY',capacity),
            ('POOL_CAPACITY',prefix.capacity),*templates],grid=(trials*256,1,1),threadgroup=(256,1,1),
        output_shapes=shapes,output_dtypes=dtypes)
    normal_draws=None;normal_refilled=None;view_metadata=None;cursor=None;block_prefix=None;next_metadata=None
    if pending_normal is not None:
        normal_draws=result[8]
        normal_refilled=result[10]
        buffers=CythonBuffers(buffers.uniform,buffers.uniform_index,
            mx.where(normal_refilled[:,None],result[9],buffers.normal),result[6][:,1])
        view_metadata=result[11]
        if block_counts:
            end=trials*(5+prepared.neurons//256)
            block_prefix=view_metadata[trials*5:end].reshape(trials,prepared.neurons//256)
            next_metadata=view_metadata[end:].reshape(trials,4)
            view_metadata=view_metadata[:trials*5].reshape(trials,5)
        done=view_metadata[:,1];counts=view_metadata[:,2];remaining=view_metadata[:,4]
        cursor=result[6][:,4]
        result=[*result[:6],result[6][:,3],result[6][:,0],result[7]]
    end=MT19937State(result[1],result[2],trials,result[3],result[4])
    if pending_normal is None:
        index=(buffers.uniform_index+counts)%20000
        available=mx.where(index>0,20000-index,0).astype(mx.uint64)
        input_physical=((mx.maximum(result[7].astype(mx.uint64),available)-available+19999)//20000)*20000
        cursor=mx.minimum(prefix.cursor.astype(mx.uint64)+physical+input_physical,prefix.capacity).astype(mx.uint32)
    metadata=None if next_metadata is None else(end,random.errors,next_metadata)
    updated=replace(random,stream=end,buffers=CythonBuffers(result[5],result[6],buffers.normal,buffers.normal_index,metadata),
        pool=replace(random.pool,data=replace(prefix,cursor=cursor,stream=end)))
    view=CythonUniformView(prefix.data.values,result[8],prefix.cursor,done,counts,buffers.uniform,
        buffers.uniform_index,remaining,capacity,view_metadata)
    returned=(result[0],updated,view)
    if block_counts:return (*returned,normal_draws,block_prefix)
    return (*returned,normal_draws)if pending_normal is not None else returned


class ExpressionPoisson:
    """Metadata matches the original target; no independent uniform tape."""
    def __init__(self,inputs,clock,target):
        from brian2.core.functions import Function
        self.channels=len(inputs);self.cython=target=='cython'
        self.active=tuple(bool(source.active)for source in inputs)
        if target not in('numpy','cython'):raise ValueError('Original input target must be NumPy or Cython')
        self.targets=mx.array([int(source._group.start)for source in inputs],mx.int32)
        thresholds=[];probabilities=[];complements=[];reversals=[]
        self.failure=None;self.prefix_channels=len(inputs)
        for source in inputs:
            probability=float(source.rate*clock.dt)
            if not self.cython and not 0<=probability<=1:
                if source.active and self.failure is None:
                    # Query the original NumPy validator on an empty private
                    # stream: no model timestep or shared draw runs on CPU.
                    try:np.random.RandomState(0).binomial(1,probability,0)
                    except ValueError as error:self.failure=(source,error)
                    self.prefix_channels=len(thresholds)
                probability=0.
            reverse=probability>.5;p=1-probability if reverse else probability;q=1-p
            if self.cython:
                function=next(value for value in source.variables.values()if isinstance(value,Function))
                code=function.implementations['cython'].get_code(source)
                qn=float(re.search(r'cdef double px = ([-+.0-9eE]+)',code)[1])
                p=float(re.search(r'\(\(1-X\+1\) \* ([-+.0-9eE]+)',code)[1])
                q=float(re.search(r'/\(X\*([-+.0-9eE]+)\)',code)[1])
            else:qn=math.exp(math.log(q))
            thresholds.append(qn);probabilities.append(p);complements.append(q);reversals.append(reverse)
        self.qn,self.probability,self.complement=[mx.array(np.asarray(v,np.float64).view(np.uint64))
            for v in(thresholds,probabilities,complements)]
        self.first_mass=(_cython_first_mass_kernel()(inputs=[self.qn,self.probability,self.complement],
            template=[('CHANNELS',self.channels)],grid=(self.channels,1,1),
            threadgroup=(min(self.channels,256),1,1),output_shapes=[(self.channels,)],output_dtypes=[mx.uint64])[0]
            if self.cython and self.channels else None)
        self.reverse=mx.array(reversals,mx.bool_)
        self.identity=(target,tuple(int(source._group.start)for source in inputs),self.active,
            *(np.asarray(v,np.float64).tobytes()for v in(thresholds,probabilities,complements)),tuple(reversals))

    def can_fuse(self,prepared):return False

    def draw(self,prepared,state,random,active,*,uniform_counts=None,pending_normal=None):
        if self.cython and uniform_counts is not None:
            return _cython_prefix_draw(self,prepared,state,random,active,uniform_counts,pending_normal)
        from .expression_rng import MT19937State
        from .expression_cython import CythonBuffers
        s=random.stream;trials=s.trials;buffers=random.buffers;zero=mx.zeros((trials,),mx.uint32)
        if self.cython!=(buffers is not None):raise ValueError('Input/model RNG targets differ')
        metadata=[s.position,s.has_gauss,s.gauss,random.errors,
            zero if buffers is None else buffers.uniform_index,zero if buffers is None else buffers.normal_index]
        packet=None if uniform_counts is not None else mx.concatenate([s.words.reshape(trials,624).astype(mx.uint64),
            *(v[:,None].astype(mx.uint64)for v in metadata)],axis=1)
        conditional=prepared.program.variables['v'].conditional_write
        flags=state.values[conditional]if conditional else mx.ones((trials*prepared.neurons,),mx.bool_)
        flags=flags&mx.repeat(active,prepared.neurons)
        extras=[]if buffers is None else[mx.concatenate([buffers.uniform,buffers.normal],axis=1)]
        prefix_args=[];prefix_templates=[]
        if uniform_counts is not None:
            from .expression_rng_prefix import _restore_arguments,UniformPrefixView,_counts
            if self.cython:raise ValueError('Uniform prefix fusion requires NumPy')
            prefix=random.pool.data
            maximum,uniform_counts=_counts(s,prepared.neurons,uniform_counts,prefix)
            done=mx.minimum(uniform_counts,prefix.capacity-prefix.cursor).astype(mx.uint32)
            prefix_args,prefix_templates=_restore_arguments(prefix,prefix.cursor+done)
            prefix_args.append((uniform_counts-done).astype(mx.uint32))
            prefix_templates.append(('MAX_COUNT',maximum))
        direct=uniform_counts is not None
        rng_inputs=[s.words,s.position,s.has_gauss,s.gauss,random.errors]if direct else[packet]
        shapes=([(trials*self.channels,),(trials*624,),*[(trials,)]*5,(trials,maximum)]if direct else
            [(trials*self.channels,),(trials*630,),(trials,)]+([(trials*40000,)]if self.cython else[]))
        dtypes=([mx.uint8,mx.uint32,mx.uint32,mx.uint32,mx.uint64,mx.uint32,mx.uint32,mx.uint64]if direct else
            [mx.uint8,mx.uint64,mx.uint32]+([mx.uint64]if self.cython else[]))
        result=_kernel(self.cython,direct)(inputs=[*rng_inputs,flags,self.targets,
            mx.array([active and channel<self.prefix_channels for channel,active in enumerate(self.active)],mx.bool_),self.qn,self.probability,self.complement,self.reverse,*extras,*prefix_args],
            template=[('CHANNELS',self.channels),('NEURONS',prepared.neurons),*prefix_templates],
            grid=(trials*256,1,1),threadgroup=(256,1,1),
            output_shapes=shapes,output_dtypes=dtypes)
        if direct:
            stream=MT19937State(result[1],result[2],trials,result[3],result[4])
            errors,consumed=result[5],result[6]
        else:
            packet=result[1].reshape(trials,630)
            stream=MT19937State(packet[:,:624].reshape(-1).astype(mx.uint32),packet[:,624].astype(mx.uint32),
                trials,packet[:,625].astype(mx.uint32),packet[:,626])
            errors,consumed=packet[:,627].astype(mx.uint32),result[2]
        if buffers is not None:
            values=result[3].reshape(trials,40000)
            buffers=CythonBuffers(values[:,:20000],packet[:,628].astype(mx.uint32),values[:,20000:],packet[:,629].astype(mx.uint32))
        pool=random.pool
        if pool is not None:
            from .expression_runtime import MixedLookahead
            if isinstance(pool,MixedLookahead):
                prefix=pool.data
                # These native draws consume the identical raw prefix. Advance
                # its cursor by actual calls, retaining exact fallback at end.
                physical=consumed
                if self.cython:
                    old_index=random.buffers.uniform_index
                    remaining=mx.where(old_index>0,20000-old_index,0).astype(mx.uint64)
                    physical=((mx.maximum(consumed.astype(mx.uint64),remaining)-remaining+19999)//20000)*20000
                cursor=mx.minimum(prefix.cursor.astype(mx.uint64)+physical+(0 if uniform_counts is None else done),prefix.capacity).astype(mx.uint32)
                pool=replace(pool,data=replace(prefix,cursor=cursor,stream=stream))
            else:pool=None
        updated=replace(random,stream=stream,errors=errors,buffers=buffers,pool=pool)
        if uniform_counts is not None:
            original=random.pool.data
            view=UniformPrefixView(original.data.values,result[7],original.cursor,done,uniform_counts)
            return result[0],updated,view
        return result[0],updated
