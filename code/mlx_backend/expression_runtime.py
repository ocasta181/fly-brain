"""Native phase execution for Brian's stateful model expressions.

All masks, draw counts, stable neuron ranks, RNG state, and model state stay
on Metal between normal chunk collection boundaries. The existing configured
engine continues to own delays, ordered arrivals, recording, and streaming.
"""
from dataclasses import dataclass,replace
from functools import lru_cache
import mlx.core as mx
import numpy as np
from .expression_lowering import Segment
from .expression_rng import seeded,uniforms,normals


@dataclass(frozen=True)
class RandomState:
    stream: object
    temporaries: object
    errors: object
    pool: object | None = None
    buffers: object | None = None
    error_metadata: object | None = None
    policy_error: object | None = None


@dataclass(frozen=True)
class UniformLookahead:
    data: object
    program: object
    begin: int
    next_step: int
    stream: object
    function: str = 'rand'
    constant_argument: float | None = None
    target: str = 'numpy'
    buffers: object | None = None


@dataclass(frozen=True)
class MixedLookahead:
    data: object
    program: object
    next_step: int
    end_step: int


POOL_VALUE_BYTES=256*1024**2
POOL_CHECKPOINT_BYTES=64*1024**2
POOL_MAX_FRAMES=1024
POOL_NORMAL_BYTES=2*1024**3
POOL_POISSON_BYTES=2*1024**3


def initial(p,trials,snapshots=None):
    from .expression_rng import from_numpy_states
    buffers=None
    if p.program.random_program.target=='cython':
        from .expression_cython import initial_buffers,from_buffer_snapshots
        if snapshots is not None and any(isinstance(s,dict)for s in snapshots):
            if not all(isinstance(s,dict)and 'numpy_state'in s for s in snapshots):
                raise ValueError('Cython snapshots require MT state and both portable buffers')
            buffers=from_buffer_snapshots(snapshots)
            snapshots=[s['numpy_state']for s in snapshots]
        else:buffers=initial_buffers(trials)
    stream=seeded(range(trials)) if snapshots is None else from_numpy_states(snapshots)
    if stream.trials!=trials:raise ValueError('Random snapshots must match trial count')
    from .configured import _dtype
    temporary={}
    for name,v in p.program.random_program.variables.items():
        if name in p.program.variables and p.program.variables[name].persistent:continue
        temporary[name]=mx.zeros((trials if v.scalar else trials*p.neurons,),_dtype(v))
    return RandomState(stream,temporary,mx.zeros((trials,),mx.uint32),buffers=buffers)


def snapshots(random):
    """Explicit portable checkpoints include Cython's physical refill state."""
    from .expression_rng import to_numpy_states
    states=to_numpy_states(random.stream)
    if random.buffers is None:return states
    from .expression_cython import buffer_snapshots
    return tuple(dict(s,numpy_state=state)for s,state in zip(buffer_snapshots(random.buffers),states))


def trial_snapshots(stimulus):
    """Explicit reproducible MT streams keyed by canonical input identities.

    These are distinct from PCG64 input events. Callers preserving an ambient
    Brian stream pass its actual get_state() snapshot instead of deriving one.
    """
    return [np.random.RandomState(np.random.SeedSequence([
        stimulus.seed,stimulus.generator_code,int(trial)]).generate_state(4)).get_state()
        for trial in stimulus.trial_indices]


def arrays(random):
    if random is None:return ()
    s=random.stream
    data=None if random.pool is None else random.pool.data
    pool=(()if data is None else data.arrays()if hasattr(data,'arrays')else
        (data.values,data.words,data.positions,data.has_gauss,data.gauss))
    buffers=()if random.buffers is None else (random.buffers.uniform,random.buffers.uniform_index,
        random.buffers.normal,random.buffers.normal_index)
    return (s.words,s.position,s.has_gauss,s.gauss,random.errors,*random.temporaries.values(),*pool,*buffers,*( () if random.error_metadata is None else (random.error_metadata,)))


def _segment_reset_ranks(segment):
    return ('_numpy_reset_rank' in segment.source
            or '_numpy_reset_count' in segment.source)


@lru_cache(maxsize=128)
def _segment_kernel(program,segment,dt):
    from .expression_compiled import namespace_bindings
    from .compiler import _metal_type
    from .binary64 import MSL_HEADER,literal
    from .binary32 import MSL_HEADER as SINGLE_HEADER
    from .math64 import MSL_MATH_HEADER
    from .expression_rng_math import NORMAL_MATH_HEADER,POISSON_EXP_HEADER
    variables=program.random_program.variables
    namespace_inputs,namespace_source=namespace_bindings(program)
    collective=_segment_reset_ranks(segment)
    packed=len(variables)+len(segment.writes)+3+int(segment.may_raise)+int(segment.index_fault is not None)+len(namespace_inputs)+int(collective)>31
    offsets={};width='0';output_offsets={};output_width='0'
    for name,v in variables.items():
        offsets[name]=width;width+='+'+('1'if v.scalar else'NEURONS')
    for name in segment.writes:
        output_offsets[name]=output_width;output_width+='+'+('1'if variables[name].scalar else'NEURONS')
    location=lambda name,offsets,width: f'_expression_trial*({width})+({offsets[name]})'+(''if variables[name].scalar else'+i')
    loads='\n'.join(f'{_metal_type(v.dtype)} {v.code_name or name}='+
        (f'{_metal_type(v.dtype)}(_expression_values[{location(name,offsets,width)}]);'if packed else
         f'_expression_p_{name}[{"_expression_trial" if v.scalar else "_expression_gid"}];')for name,v in variables.items())
    stores='\n'.join((f'_expression_next_values[{location(name,output_offsets,output_width)}]=ulong({variables[name].code_name or name});'if packed else
        f'_expression_q_{name}[{"_expression_trial"if variables[name].scalar else"_expression_gid"}]={variables[name].code_name or name};')for name in segment.writes)
    source=segment.source
    if (program.equilibrium is not None and {'g','v','not_refractory'}<=set(segment.writes)
            and source.startswith('not_refractory =')):
        refractory,dynamics=source.split('\n',1)
        rest=' && '.join(f'{name}=={literal(value)}' for name,value in program.equilibrium.items())
        source=refractory+'\nif(!('+rest+')) {\n'+dynamics+'\n}'
    return mx.fast.metal_kernel(name='fly_expression_segment',
        input_names=(['_expression_values']if packed else['_expression_p_'+name for name in variables])+['_expression_selected','_expression_clock_times','_expression_params']+list(namespace_inputs)+(['_expression_reset_ranks']if collective else[]),
        output_names=(['_expression_next_values']if packed else['_expression_q_'+name for name in segment.writes])+(
            ['_expression_segment_errors']if segment.may_raise else[])+(['_expression_index_metadata']if segment.index_fault else[]),
        source=f'''
        uint _expression_gid=thread_position_in_grid.x;
        if(_expression_gid>={"B" if segment.scalar else "B*NEURONS"})return;
        uint _expression_trial={"_expression_gid" if segment.scalar else "_expression_gid/NEURONS"};
        const long N=NEURONS;const int i={"0" if segment.scalar else "int(_expression_gid%N)"};
        const ulong dt={literal(dt)},t=_expression_clock_times[_expression_params[0]-_expression_params[1]];
        const long t_in_timesteps=_expression_params[0];
        {('const long _numpy_reset_rank=long(_expression_reset_ranks[_expression_trial*NEURONS+i])-1L;'
          'const long _numpy_reset_count=long(_expression_reset_ranks[(_expression_trial+1)*NEURONS-1]);')if collective else''}
        {namespace_source}
        {loads}
        {"uint _expression_local_error=0; uint _profile_error=0;"if segment.may_raise else""}
        if(_expression_selected[_expression_gid]) {{ {source}
            {"_expression_local_error=_profile_error;"if segment.may_raise else""}
        }}
        {stores}
        {f'_expression_index_metadata[2*_expression_gid]=long({segment.index_fault[0]}); _expression_index_metadata[2*_expression_gid+1]={segment.index_fault[1]};' if segment.index_fault else ''}
        {"_expression_segment_errors[_expression_gid]=_expression_local_error;"if segment.may_raise else""}
        ''',
        header=MSL_HEADER+SINGLE_HEADER+MSL_MATH_HEADER+NORMAL_MATH_HEADER+POISSON_EXP_HEADER+program.header_code,compile_options={'math_mode':'safe'})


def _segment(p,op,values,mask,trials,clock_times,params,reset_selected=None):
    from .configured import _dtype
    variables=p.program.random_program.variables
    size=trials if op.scalar else trials*p.neurons
    collective=_segment_reset_ranks(op)
    packed=len(variables)+len(op.writes)+3+int(op.may_raise)+int(op.index_fault is not None)+bool(p.input_arrays)+int(collective)>31
    data=([mx.concatenate([values[name].reshape(trials,-1).astype(mx.uint64)for name in variables],axis=1)]
          if packed else[values[name]for name in variables])
    from .expression_compiled import _layout
    layout,total=_layout([(name,variables[name].scalar)for name in op.writes],p.neurons)
    outputs=_segment_kernel(p.program,op,p.model.dt_s)(
        inputs=[*data,mask.reshape(-1),clock_times,params,*p.input_arrays,
                *([mx.cumsum(reset_selected.reshape(trials,p.neurons).astype(mx.int32),axis=1).reshape(-1)]if collective else[])],
        template=[('NEURONS',p.neurons),('B',trials)],grid=(size,1,1),threadgroup=(min(256,size),1,1),
        output_shapes=([(trials*total,)]if packed else[(trials if variables[name].scalar else trials*p.neurons,) for name in op.writes])+([(size,)]if op.may_raise else[])+([(size,2)]if op.index_fault else[]),
        output_dtypes=([mx.uint64]if packed else[_dtype(variables[name]) for name in op.writes])+([mx.uint32]if op.may_raise else[])+([mx.int64]if op.index_fault else[]))
    metadata=outputs[-1].reshape(trials,-1,2)if op.index_fault else None
    if metadata is not None:outputs=outputs[:-1]
    errors=outputs[-1].reshape(trials,-1)if op.may_raise else None
    if errors is not None:
        width=errors.shape[1];rows=mx.arange(width,dtype=mx.uint32)[None,:]
        first=mx.min(mx.where(errors>0,rows,width),axis=1)
        commit=rows<=first[:,None]
        outputs=outputs[:-1]
        before={name:values[name]for name in op.writes}
    if packed:
        result=outputs[0].reshape(trials,total)
        for name in op.writes:
            width=1 if variables[name].scalar else p.neurons
            values[name]=result[:,layout[name]:layout[name]+width].reshape(-1).astype(_dtype(variables[name]))
    else:values.update(zip(op.writes,outputs))
    if errors is not None:
        for name in op.writes:
            selected=commit[:,:1]if variables[name].scalar else commit
            values[name]=mx.where(selected.reshape(-1),values[name],before[name])
        result=mx.max(errors,axis=1)
        if metadata is not None:
            metadata=mx.take_along_axis(metadata,mx.minimum(first,width-1)[:,None,None],axis=1)[:,0,:]
            return result,metadata
        return result


@lru_cache(maxsize=32)
def _fixed_draw(program):
    """Prove that every step consumes one unconditional fixed-size draw.

    Reset selection, conditional writes, mixed primitives and state-dependent
    Poisson parameters retain the general phase-by-phase execution path.
    """
    if program.random_program.serial_required:return None
    if any(op.may_raise for operations in program.random_program.stages.values()
           for op in operations if isinstance(op,Segment)):return None
    draws=[(stage,op) for stage,operations in program.random_program.stages.items()
           for op in operations if not isinstance(op,Segment)]
    if len(draws)==1:
        stage,op=draws[0]
        if (stage!='reset' and op.guard is None and (op.function in ('rand','randn')
                or(op.function=='poisson'and op.constant_argument is not None))):
            return op
    return None


@lru_cache(maxsize=32)
def _mixed_prefix_eligible(program):
    random=program.random_program
    if random.target=='cython':return False
    for stage in random.serial_required:
        if random.compiled_stages[stage].single_call not in('rand','randn'):return False
    return all(op.function in('rand','randn')for operations in random.stages.values()
               for op in operations if not isinstance(op,Segment))


def _mixed_prefix(p,random,step,steps,length=1):
    """Refresh bounded raw lookahead at host-known model-step boundaries.

    The capacity is a speculative work budget, not a bound on model draws.
    Dense/conditional rejection can exhaust it; native fallback remains exact.
    """
    if p.neurons<2**16 or not _mixed_prefix_eligible(p.program):return random
    cache=random.pool
    if (isinstance(cache,MixedLookahead)and cache.program is p.program
            and cache.data.stream is random.stream and cache.next_step==step and step+length<=cache.end_step):return random
    from .expression_rng_prefix import prefix_pool
    trials=random.stream.trials;checkpoint=2496
    if 2*(3*((p.neurons+1)//2)//2+128)*trials>=2**32:return replace(random,pool=None)
    capacity=min(POOL_VALUE_BYTES//(trials*8)//checkpoint,
                 POOL_CHECKPOINT_BYTES//(trials*624*4))*checkpoint
    fixed=_fixed_draw(p.program)
    shared_normal=(p.program.shared_expression_inputs and fixed is not None
                   and fixed.function=='randn')
    raw_budget=2 if shared_normal else 1
    if any(_segment_reset_ranks(op) for op in p.program.random_program.stages.get('reset',())
           if isinstance(op,Segment)):
        # Each uniform site can consume a whole selected vector. A one-vector
        # budget exhausts before the promised frame boundary and serializes
        # later draws. Normal rejection still has its exact native fallback.
        raw_budget=max(raw_budget,sum(2 if op.function=='randn' else 1
            for operations in p.program.random_program.stages.values()
            for op in operations if not isinstance(op,Segment)))
    frames=min(steps-step,POOL_MAX_FRAMES,capacity//(p.neurons*raw_budget))
    if frames<1:return replace(random,pool=None)
    prefix=prefix_pool(random.stream,p.neurons*frames*raw_budget,checkpoint=checkpoint)
    if shared_normal:
        from .expression_normal_prefix import prepare
        prefix=prepare(prefix)
    return replace(random,pool=MixedLookahead(prefix,p.program,step,step+frames))


def uniform_lookahead(program,stream,cache,width,step,steps,length,*,function='rand',constant_argument=None,target='numpy'):
    """Device-only fixed-call lookahead shared by execution and component tests."""
    trials=stream.trials
    valid=(cache is not None and cache.program is program
           and cache.stream is stream and cache.next_step==step and cache.function==function
           and cache.constant_argument==constant_argument
           and cache.target==target
           and cache.data.width==width and cache.data.trials==trials
           and cache.begin<=step and step+length<=cache.begin+cache.data.frames)
    if not valid:
        if function=='poisson':
            limit=float(np.iinfo(np.int64).max)-10*np.sqrt(float(np.iinfo(np.int64).max))
            if (constant_argument is None or not np.isfinite(constant_argument)or
                    (target=='numpy'and(constant_argument<0 or constant_argument>=limit))):return None,stream,None
            cost=max(128,int(64*(constant_argument+1)))if constant_argument<10 else 128
            value_frames=POOL_POISSON_BYTES//(trials*width*cost)
        else:
            value_frames=(POOL_NORMAL_BYTES//(trials*width*128)if function=='randn'
                          else POOL_VALUE_BYTES//(trials*width*8))
        frames=min(steps-step,POOL_MAX_FRAMES,value_frames,POOL_CHECKPOINT_BYTES//(trials*624*4))
        cache=None
        if frames>=length:
            if function=='randn':
                from .expression_rng_parallel import normal_pool
                data=normal_pool(stream,width,frames,target=target)
            elif function=='poisson':
                from .expression_poisson_parallel import poisson_pool
                data=poisson_pool(stream,width,frames,constant_argument,target=target)
            else:
                from .expression_rng_jump import uniform_pool
                data=uniform_pool(stream,width,frames)
            cache=UniformLookahead(data,program,step,step,stream,function,constant_argument,target)
    if cache is None:return None,stream,None
    offset=step-cache.begin
    draws=cache.data.values[offset:offset+length]
    stream=cache.data.state_at(offset+length-1)
    return draws,stream,UniformLookahead(cache.data,program,cache.begin,step+length,stream,function,constant_argument,target)


def cython_lookahead(program,stream,buffers,cache,width,step,steps,length,*,function,constant_argument=None,errors=None):
    """Keep CPU-sized physical buffers and logical model steps independent."""
    trials=stream.trials
    valid=(cache is not None and cache.program is program and cache.stream is stream
        and cache.buffers is buffers and cache.next_step==step and cache.function==function
        and cache.constant_argument==constant_argument
        and cache.target=='cython'and cache.data.width==width and cache.data.trials==trials
        and cache.begin<=step and step+length<=cache.begin+cache.data.frames)
    if not valid:
        cost=(max(128,int(128*(constant_argument+1)))if function=='poisson'and constant_argument<10
              else 128 if function in('randn','poisson')else 24)
        frames=min(steps-step,POOL_MAX_FRAMES,POOL_NORMAL_BYTES//(trials*width*cost),
            POOL_CHECKPOINT_BYTES//(trials*624*4))
        if function=='poisson':frames=min(frames,256*1024**2//(trials*20000*8))
        cache=None
        if frames>=length:
            from .expression_cython import buffered_pool
            if function=='poisson':
                from .expression_cython_poisson import poisson_pool
                data=poisson_pool(stream,buffers,width,frames,constant_argument,errors=errors)
            else:data=buffered_pool(stream,buffers,function,width,frames)
            cache=UniformLookahead(data,program,step,step,stream,function,constant_argument,target='cython',buffers=buffers)
    if cache is None:return None,stream,buffers,None
    offset=step-cache.begin;draws=cache.data.values[offset:offset+length]
    stream,buffers=cache.data.state_at(offset+length-1)
    return draws,stream,buffers,UniformLookahead(cache.data,program,cache.begin,step+length,
        stream,function,constant_argument,target='cython',buffers=buffers)


def _stage(p,name,values,random,trials,clock_times,params,prefetched=None,selection=None):
    ordered_poisson=(p.program.random_program.target=='cython'and any(
        not isinstance(op,Segment)and op.function=='poisson'for op in p.program.random_program.stages.get(name,())))
    if name in p.program.random_program.serial_required or ordered_poisson:
        from .expression_single_call import primitive
        if primitive(p.program,name) is not None:
            from .expression_single_call import stage
        else:from .expression_compiled import stage
        result=stage(p,name,values,random,trials,clock_times,params)
        if name=='threshold'and not(p.program.random_program.target=='cython'and p.program.replay_phase is not None):
            values['_cond']=mx.where(mx.repeat(result.errors==0,p.neurons),values['_cond'],False)
        return result
    stream=random.stream;errors=random.errors;buffers=random.buffers;cache=random.pool
    metadata=random.error_metadata
    selected=values['_cond'] if name=='reset' else None
    for op in p.program.random_program.stages.get(name,()):
        width=1 if op.scalar else p.neurons
        if isinstance(op,Segment):
            mask=(selected.reshape(trials,width) if name=='reset' and not op.scalar
                  else mx.ones((trials,width),mx.bool_))
            mask=mx.logical_and(mask,(errors==0)[:,None])
            error=_segment(p,op,values,mask,trials,clock_times,params,reset_selected=selected)
            if isinstance(error,tuple):
                error,detail=error
                metadata=detail if metadata is None else mx.where((error!=0)[:,None],detail,metadata)
            if error is not None:errors=errors|error
            continue
        if op.function in ('numpy_complex_cast','numpy_python_complex_store','numpy_complex_ufunc_error'):
            from .numpy_complex_warnings import warn_cast,python_store_error,ufunc_error
            from .numpy_policy_context import host_action
            for trial in range(trials):
                if int(np.asarray(errors)[trial])!=0:continue
                current=RandomState(stream,random.temporaries,errors,cache,buffers,metadata)
                action=((lambda:warn_cast(p,name,op.result)) if op.function=='numpy_complex_cast' else
                    (lambda:python_store_error(op.argument)) if op.function=='numpy_python_complex_store' else
                    (lambda:ufunc_error(op.argument)))
                current,error=host_action(p,values,current,op.host_views,action)
                stream,buffers,cache=current.stream,current.buffers,current.pool
                if error is not None:
                    return RandomState(stream,random.temporaries,errors,cache,buffers,metadata,(name,error))
            continue
        if op.function in ('numpy_status','numpy_python_int'):
            from .numpy_policy_context import deliver
            flags=np.asarray(values[op.result]).reshape(trials,width)
            mask=(np.asarray(selected).reshape(trials,width) if name=='reset' and not op.scalar
                  else np.ones((trials,width),bool))
            if op.guard:
                mask &= np.asarray(values[op.guard]).reshape(trials,-1)
            for trial in range(trials):
                if int(np.asarray(errors)[trial])!=0:continue
                status=int(np.bitwise_or.reduce(flags[trial][mask[trial]],initial=np.uint32(0)))
                if op.function=='numpy_python_int':
                    if status:
                        error=(ValueError('cannot convert float NaN to integer') if status==1 else
                               OverflowError('cannot convert float infinity to integer') if status==2 else
                               OverflowError('Python int too large to convert to C long'))
                        return RandomState(stream,random.temporaries,errors,cache,buffers,metadata,(name,error))
                    continue
                current=RandomState(stream,random.temporaries,errors,cache,buffers,metadata)
                current,error=deliver(p,op.argument,status,values,current,op.host_views)
                stream,buffers,cache=current.stream,current.buffers,current.pool
                if error is not None:
                    return RandomState(stream,random.temporaries,errors,cache,buffers,metadata,
                                       (name,error))
            continue
        if prefetched is not None and op is prefetched[0]:
            values[op.result]=prefetched[1]
            continue
        unmasked=op.guard is None and (name!='reset' or op.scalar)
        mask=ranks=None
        counts=mx.where(errors==0,width,0).astype(mx.uint32)
        if not unmasked:
            mask=(selected.reshape(trials,width) if name=='reset' and not op.scalar
                  else mx.ones((trials,width),mx.bool_))
            if op.guard:
                guard=values[op.guard].reshape(trials,-1)
                mask=mx.logical_and(mask,mx.broadcast_to(guard,mask.shape))
            mask=mx.logical_and(mask,(errors==0)[:,None])
            if selection is not None and name=='reset'and not op.scalar and op.guard is None:
                counts,ranks=selection
            else:
                counts=mx.sum(mask,axis=1).astype(mx.uint32)
                ranks=mx.maximum(mx.cumsum(mask.astype(mx.int32),axis=1)-1,0)
        if buffers is not None and op.function in('rand','randn'):
            from .expression_cython import buffered_draws
            if (isinstance(cache,MixedLookahead) and p.program.shared_expression_inputs
                    and op.function=='randn' and _fixed_draw(p.program) is op):
                packed,stream,buffers,prefix=buffered_draws(stream,buffers,op.function,width,counts,
                    _prefix=cache.data)
                cache=replace(cache,data=prefix)
            else:
                packed,stream,buffers=buffered_draws(stream,buffers,op.function,width,counts,sparse=not unmasked)
        elif isinstance(cache,MixedLookahead)and op.function in('rand','randn'):
            from .expression_rng_prefix import prefix_uniforms,prefix_normals
            from .expression_single_call import _normal_candidate_budget
            budget=_normal_candidate_budget(p.model.eq_th,p.neurons)if name=='reset'and op.function=='randn'else None
            if op.function=='randn'and p.targets and budget is not None and budget<=256:
                from .expression_sparse_normal import draw
                packed,stream,cache=draw(replace(random,stream=stream,pool=cache),width,counts,
                    target=p.program.random_program.target)
            else:
                if op.function=='rand':packed,stream,prefix=prefix_uniforms(stream,cache.data,width,counts)
                else:packed,stream,prefix=prefix_normals(stream,cache.data,width,counts,
                    target=p.program.random_program.target,candidate_count=budget)
                cache=replace(cache,data=prefix)
        elif op.function=='rand':packed,stream=uniforms(stream,width,counts)
        elif op.function=='randn':packed,stream=normals(stream,width,counts,target=p.program.random_program.target)
        else:
            target=p.program.random_program.target
            if target=='cython':
                from .expression_cython import poisson as cython_poisson
                def poisson(s,l,c):
                    nonlocal buffers
                    packed,end,buffers,error=cython_poisson(s,buffers,l,c)
                    return packed,end,error
            elif target=='cpp':
                from .expression_poisson_compiled import poisson as cpp_poisson
                def poisson(s,l,c):
                    packed,end=cpp_poisson(s,l,c)
                    return packed,end,mx.zeros((s.trials,),mx.uint32)
            else:
                from .expression_poisson import poisson as numpy_poisson
                def poisson(s,l,c):
                    return numpy_poisson(s,l,c,_detailed=True,_scalar=op.argument_scalar,
                        _enabled=errors==0)
            lambdas=mx.broadcast_to(values[op.argument].reshape(trials,-1),(trials,width))
            if not unmasked:
                indices=mx.broadcast_to(mx.arange(width),mask.shape)
                order=mx.argsort(mx.where(mask,indices,indices+width),axis=1)
                lambdas=mx.take_along_axis(lambdas,order,axis=1)
            if (op.constant_argument is not None and width>=2**14
                    and (target!='numpy' or np.isfinite(op.constant_argument) and 0<=op.constant_argument<=float(np.iinfo(np.int64).max)-10*np.sqrt(float(np.iinfo(np.int64).max)))):
                if target=='cython':
                    from .expression_cython_poisson import constant_poisson
                    packed,stream,buffers,error=constant_poisson(stream,buffers,width,op.constant_argument,counts)
                else:
                    from .expression_poisson_parallel import constant_poisson
                    packed,stream,error=constant_poisson(stream,width,op.constant_argument,counts,target=target)
            else:packed,stream,error=poisson(stream,lambdas,counts)
            errors=errors|error
        values[op.result]=(packed if unmasked else
            mx.where(mask,mx.take_along_axis(packed,ranks,axis=1),0)).reshape(-1)
    if name=='threshold'and not(p.program.random_program.target=='cython'and p.program.replay_phase is not None):
        values['_cond']=mx.where(mx.repeat(errors==0,p.neurons),values['_cond'],False)
    return RandomState(stream,random.temporaries,errors,cache,buffers,metadata)
