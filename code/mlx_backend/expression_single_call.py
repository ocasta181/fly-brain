"""Stable device compaction for a single conditional compiled random call.

Eligibility can be evaluated without drawing: before the only call, control
flow depends solely on model inputs. A second evaluation uses the actual draw
at that cell's stable rank and commits the original compiled statements.
"""
from functools import lru_cache
from dataclasses import dataclass


@dataclass(frozen=True)
class PendingReset:
    """Internal pipeline state; deferred RNG is consumed before the next threshold."""
    program: object
    draws: object
    ranks: object
    time: object
    normal_counts: object = None
    blocks: object = None


@lru_cache(maxsize=128)
def primitive(program,name):
    return program.random_program.compiled_stages[name].single_call


@lru_cache(maxsize=128)
def _kernel(program,name,dt,apply,packed=False,block_ranks=False,markers=False):
    from .expression_compiled import namespace_bindings
    import mlx.core as mx
    from .binary64 import MSL_HEADER,literal
    from .binary32 import MSL_HEADER as SINGLE_HEADER
    from .math64 import MSL_MATH_HEADER
    from .expression_rng_math import NORMAL_MATH_HEADER,POISSON_EXP_HEADER
    variables={n:v for n,v in program.variables.items()if v.persistent}
    stage=program.random_program.compiled_stages[name]
    vector=(_namespace_arithmetic('{'+stage.vector+'}')[1:-1]
            if packed and apply and program.input_arrays else stage.vector)
    namespace_inputs,namespace_source=namespace_bindings(program)
    # Symbolic offsets keep one compiled kernel for every batch size.
    offsets={};width='0'
    for n,v in variables.items():
        offsets[n]=width;width+='+'+('1'if v.scalar else'NEURONS')
    output_offsets={};out_width='0'
    for n in stage.writes:
        output_offsets[n]=out_width;out_width+='+'+('1'if n in variables and variables[n].scalar else'NEURONS')
    loads='\n'.join(f'{v.metal_type} {v.code_name}={v.metal_type}(_single_values[_single_trial*({width})+({offsets[n]})'
        +(''if v.scalar else'+_single_row')+']);'for n,v in variables.items())
    stores=[]
    for n in stage.writes:
        scalar=n in variables and variables[n].scalar
        stores.append(('if(!_single_row)'if scalar else'')+
            f'_single_next_values[_single_trial*({out_width})+({output_offsets[n]})'+
            (''if scalar else'+_single_row')+f']=ulong({variables[n].code_name if n in variables else n});')
    input_values=['_single_values'];output_values=['_single_next_values']
    if packed:
        from .configured import _state_layout
        layout=_state_layout(program)
        loads,packed_stores=layout.source(program)
        stores=[packed_stores]
        input_values=['_eon_p_'+group.buffer_name for group in layout.groups]
        output_values=['_eon_q_'+group.buffer_name for group in layout.groups]
    return mx.fast.metal_kernel(name='expression_single_call_apply'if apply else'expression_single_call_eligibility',
        input_names=[*input_values,'_single_selected','_single_errors','_single_clock_times','_single_params']+
            (['_single_draws','_single_ranks']if apply else[])+(['_single_block_prefix']if block_ranks else[])+(['incoming_offsets','incoming_sources','spikes']if markers else[])+list(namespace_inputs),
        output_names=(output_values+(['arrival_marker']if markers else[]))if apply else['_single_eligible'],
        header=MSL_HEADER+SINGLE_HEADER+MSL_MATH_HEADER+NORMAL_MATH_HEADER+POISSON_EXP_HEADER+program.header_code+r'''
        struct SingleCallRng {ulong value;uint calls,error;};
        inline ulong compiled_rand(thread SingleCallRng& rng){++rng.calls;return rng.value;}
        inline ulong compiled_normal(thread SingleCallRng& rng){++rng.calls;return rng.value;}
        ''',source=f'''
        uint _single_gid=thread_position_in_grid.x;if(_single_gid>=NEURONS*B)return;
        const uint gid=_single_gid;
        uint _single_trial=_single_gid/NEURONS,_single_row=_single_gid%NEURONS;
        const long N=NEURONS,t_in_timesteps=_single_params[0];int i=0;
        const ulong dt={literal(dt)},t=_single_clock_times[_single_params[0]-_single_params[1]];
        {namespace_source}
        {loads}
        SingleCallRng _compiled_rng;_compiled_rng.calls=0;_compiled_rng.error=_single_errors[_single_trial];
        {"uint _single_block=_single_gid/256;uint _single_before=_single_block%(NEURONS/256)?_single_block_prefix[_single_block-1]:0u;"if block_ranks else""}
        _compiled_rng.value={"_single_draws[_single_trial*NEURONS+_single_ranks[_single_gid]"+(' + _single_before'if block_ranks else'')+"]"if apply else"0ul"};
        bool _cond=false;
        if(!_compiled_rng.error){{
            {stage.scalar}
            i=int(_single_row);
            if(_single_selected[_single_gid]){{{vector}}}
        }}
        {''.join(stores)if apply else'_single_eligible[_single_gid]=uint(_compiled_rng.calls>0);'}
        {"uint hit=0;for(int edge=incoming_offsets[_single_row];edge<incoming_offsets[_single_row+1];edge++)hit|=uint(spikes[_single_trial*NEURONS+incoming_sources[edge]]);arrival_marker[_single_gid]=hit<<SHIFT;"if markers else""}
        ''',compile_options={'math_mode':'safe'})


@lru_cache(maxsize=128)
def selected_primitive(program,name):
    """Prove that each selected row calls the same primitive exactly once."""
    from .expression_lowering import Draw,Segment
    random=program.random_program
    if (program.scalar_stages or program.index_error_name or random.target=='numpy'
            or any(v.persistent and v.dtype=='float32'for v in program.variables.values())
            or name not in random.compiled_stages):return None
    if random.compiled_stages[name].may_raise:return None
    operations=random.stages.get(name,())
    draws=[op for op in operations if isinstance(op,Draw)]
    if len(draws)!=1 or draws[0].scalar or draws[0].guard is not None:return None
    if any(op.scalar or op.may_raise for op in operations if isinstance(op,Segment)):return None
    result=primitive(program,name)
    return result if result==draws[0].function and result in('rand','randn')else None


@lru_cache(maxsize=128)
def _normal_candidate_budget(threshold,neurons):
    """Size speculative parallel work; native fallback owns any remainder."""
    import ast,math
    node=ast.parse(threshold,mode='eval').body
    if not(isinstance(node,ast.Compare)and len(node.ops)==len(node.comparators)==1
        and isinstance(node.left,ast.Call)and isinstance(node.left.func,ast.Name)
        and node.left.func.id=='rand'and not node.left.args and not node.left.keywords
        and isinstance(node.comparators[0],ast.Constant)
        and type(node.comparators[0].value)in(int,float)):return None
    probability=node.comparators[0].value
    if isinstance(node.ops[0],(ast.Gt,ast.GtE)):probability=1-probability
    elif not isinstance(node.ops[0],(ast.Lt,ast.LtE)):return None
    if isinstance(probability,float)and not math.isfinite(probability):return None
    probability=min(1.,max(0.,probability))
    mean=neurons*probability
    predicted=min(neurons,math.ceil(mean+8*math.sqrt(mean*(1-probability))+32))
    full=3*((neurons+1)//2)//2+128
    candidates=3*((predicted+1)//2)//2+128
    # This is a performance budget, never a limit on accepted model draws.
    return min(full,max(128,(candidates+255)//256*256))


def _draw(p,random,function,counts):
    import mlx.core as mx
    from .expression_runtime import MixedLookahead
    from dataclasses import replace
    from .expression_rng import uniforms,normals
    buffers=random.buffers;cache=random.pool
    if buffers is not None:
        from .expression_cython import buffered_draws
        draws,stream,buffers=buffered_draws(random.stream,buffers,function,p.neurons,counts,sparse=True,_clear_unused=False)
        if isinstance(cache,MixedLookahead):
            # Logical cached draws leave MT untouched. A physical refill can
            # include rejection/cached-Gaussian transitions: invalidate instead
            # of guessing its raw consumption.
            old=random.buffers
            index=old.normal_index if function=='randn'else old.uniform_index
            remaining=mx.where(index>0,20000-index,0)
            cursor=mx.where(counts>remaining,cache.data.capacity,cache.data.cursor).astype(mx.uint32)
            cache=replace(cache,data=replace(cache.data,cursor=cursor,stream=stream))
    elif isinstance(cache,MixedLookahead):
        from .expression_rng_prefix import prefix_uniforms,prefix_normals
        if function=='rand':draws,stream,prefix=prefix_uniforms(random.stream,cache.data,p.neurons,counts)
        else:draws,stream,prefix=prefix_normals(random.stream,cache.data,p.neurons,counts,
            target=p.program.random_program.target,candidate_count=_normal_candidate_budget(p.model.eq_th,p.neurons))
        cache=replace(cache,data=prefix)
    elif function=='rand':draws,stream=uniforms(random.stream,p.neurons,counts)
    else:draws,stream=normals(random.stream,p.neurons,counts,target=p.program.random_program.target)
    return draws,stream,buffers,cache


def advance_packed(p,name,values,selected,random,clock_times,params,selection=None,marker_shift=None,*,_defer=False,_defer_normal=False,_rank_flags=False):
    """Use canonical buffers and a proved selection instead of copying rows."""
    import mlx.core as mx
    from .configured import _PackedValues,_state_layout,_storage_dtype
    from .expression_runtime import RandomState
    function=selected_primitive(p.program,name)
    if function is None:raise ValueError('Packed random consumer needs an unconditional selected draw')
    trials=random.stream.trials;size=trials*p.neurons
    eligible=None if _rank_flags else selected.reshape(trials,p.neurons)&(random.errors==0)[:,None]
    extra=[]
    if selection is None:
        counts=mx.sum(eligible,axis=1).astype(mx.uint32)
        ranks=mx.maximum(mx.cumsum(eligible.astype(mx.int32),axis=1)-1,0).astype(mx.uint32)
    elif _rank_flags and _defer_normal:
        ranks,blocks=selection
        counts=blocks.reshape(trials,p.neurons//256);prefix=None
    else:
        ranks,blocks=selection
        prefix=mx.cumsum(blocks.reshape(trials,p.neurons//256),axis=1).astype(mx.uint32)
        counts=prefix[:,-1]if _rank_flags else mx.where(random.errors==0,prefix[:,-1],0).astype(mx.uint32)
        extra=[prefix]
    if _defer_normal:
        assert _defer and function=='randn'
        draws=None;stream=random.stream;buffers=random.buffers;cache=random.pool
    else:draws,stream,buffers,cache=_draw(p,random,function,counts)
    if _defer:
        if _rank_flags:
            assert _defer_normal and selection is not None
            time=mx.take(clock_times,(params[0]-params[1]).reshape(1))
            return random,PendingReset(p.program,None,ranks.reshape(-1),time,counts,prefix)
        if selection is not None:
            before=mx.concatenate([mx.zeros((trials,1),mx.uint32),prefix[:,:-1]],axis=1)
            ranks=ranks.reshape(trials,p.neurons)+mx.repeat(before,256,axis=1)
        encoded=ranks.reshape(-1)|(eligible.reshape(-1).astype(mx.uint32)<<31)
        time=mx.take(clock_times,(params[0]-params[1]).reshape(1))
        return RandomState(stream,random.temporaries,random.errors,cache,buffers),PendingReset(p.program,draws,encoded,time,counts if _defer_normal else None)
    layout=_state_layout(p.program)
    outputs=_kernel(p.program,name,p.model.dt_s,True,packed=True,block_ranks=selection is not None,markers=marker_shift is not None)(
        inputs=[*values.buffers,eligible.reshape(-1),random.errors,clock_times,params,draws,ranks.reshape(-1),*extra,*([p.incoming_offsets,p.incoming_sources,selected]if marker_shift is not None else[]),*p.input_arrays],
        template=[('NEURONS',p.neurons),('B',trials),*([('SHIFT',marker_shift)]if marker_shift is not None else[])],grid=(size,1,1),threadgroup=(min(256,size),1,1),
        output_shapes=[*layout.shapes(size),*([(size,)]if marker_shift is not None else[])],
        output_dtypes=[*[_storage_dtype(group.dtype)for group in layout.groups],*([mx.uint32]if marker_shift is not None else[])])
    marker=outputs[-1]if marker_shift is not None else None
    if marker_shift is not None:outputs=outputs[:-1]
    result=(RandomState(stream,random.temporaries,random.errors,cache,buffers),_PackedValues(layout,outputs,size))
    return (*result,marker)if marker_shift is not None else result


def stage(p,name,values,random,trials,clock_times,params):
    import mlx.core as mx
    from .configured import _dtype
    from .expression_compiled import _layout
    from .expression_runtime import RandomState
    function=primitive(p.program,name)
    if function is None:raise ValueError('Stage must contain exactly one vector random call')
    variables={n:v for n,v in p.program.variables.items()if v.persistent}
    selected=values['_cond']if name=='reset'else mx.ones((trials*p.neurons,),mx.bool_)
    data=mx.concatenate([values[n].reshape(trials,-1).astype(mx.uint64)for n in variables],axis=1)
    inputs=[data,selected,random.errors,clock_times,params]
    dispatch=dict(template=[('NEURONS',p.neurons),('B',trials)],
        grid=(trials*p.neurons,1,1),threadgroup=(min(256,trials*p.neurons),1,1))
    eligible=_kernel(p.program,name,p.model.dt_s,False)(inputs=[*inputs,*p.input_arrays],
        output_shapes=[(trials*p.neurons,)],output_dtypes=[mx.uint32],**dispatch)[0].reshape(trials,p.neurons)
    counts=mx.sum(eligible,axis=1).astype(mx.uint32)
    ranks=mx.maximum(mx.cumsum(eligible.astype(mx.int32),axis=1)-1,0).astype(mx.uint32)
    draws,stream,buffers,cache=_draw(p,random,function,counts)
    op=p.program.random_program.compiled_stages[name]
    layout,total=_layout([(n,n in variables and variables[n].scalar)for n in op.writes],p.neurons)
    output=_kernel(p.program,name,p.model.dt_s,True)(inputs=[*inputs,draws,ranks.reshape(-1),*p.input_arrays],
        output_shapes=[(trials*total,)],output_dtypes=[mx.uint64],**dispatch)[0].reshape(trials,total)
    for n in op.writes:
        width=1 if n in variables and variables[n].scalar else p.neurons
        dtype=mx.bool_ if n=='_cond'else _dtype(variables[n])
        values[n]=output[:,layout[n]:layout[n]+width].reshape(-1).astype(dtype)
    return RandomState(stream,random.temporaries,random.errors,cache,buffers)


# The Metal functions below are value-only helpers. In particular, no loads,
# RNG calls, error helpers, or user/model functions are eligible.
_NAMESPACE_PURE = frozenset(('expr_fma', 'd_to_int32_sat', 'd_from_int64',
                   'd_from_int32', 'd_from_uint64', 'd_from_uint32',
                   'd_add', 'd_sub', 'd_mul', 'd_from_int', 'as_type<int>',
                   'long', 'int', 'uint', 'ulong'))


def _namespace_arithmetic(source):
    from collections import Counter
    import re
    parts = source.split(';')
    serial = 0
    for k, part in enumerate(parts):
        brace = part.rfind('{')
        if brace < 0:
            continue
        prefix, statement = part[:brace + 1], part[brace + 1:]
        if not re.match(r'\s*[A-Za-z_]\w*\s*=', statement):
            continue
        declarations = []
        while True:
            calls = []
            for match in re.finditer(r'\b(?:expr_fma|d_to_int32_sat)\(', statement):
                depth, end = 1, match.end()
                while end < len(statement) and depth:
                    depth += (statement[end] == '(') - (statement[end] == ')')
                    end += 1
                call = statement[match.start():end]
                if depth or any(token in call for token in ('[', ']', '++', '--', '=', '?', '&&', '||')):
                    continue
                if not set(re.findall(r'\b([A-Za-z_]\w*(?:<[^<>]*>)?)\s*\(', call)) <= _NAMESPACE_PURE:
                    continue
                calls.append(call)
            repeated = [call for call, count in Counter(calls).items() if count > 1]
            if not repeated:
                break
            call = max(repeated, key=len)
            while True:
                name = f'_eon_pending_namespace_cse_{serial}'
                serial += 1
                if name not in source:
                    break
            declarations.append(f'const auto {name}={call};')
            statement = statement.replace(call, name)
        parts[k] = prefix + ''.join(declarations) + statement
    return ';'.join(parts)
