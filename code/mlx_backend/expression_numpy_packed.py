"""Pack one selected NumPy draw without changing its typed vector arithmetic."""
from functools import lru_cache
from dataclasses import replace
import mlx.core as mx
import re
from .expression_lowering import Draw,Segment


@lru_cache(maxsize=128)
def _reset_ranked(program,name):
    from .expression_runtime import _segment_reset_ranks
    return any(_segment_reset_ranks(op) for op in program.random_program.stages.get(name,())
               if isinstance(op,Segment))


@lru_cache(maxsize=128)
def _bindings(program,name):
    """Bind exact emitted identifiers, including shadow and encoded aliases."""
    from .configured import _state_layout
    operations=program.random_program.stages.get(name,())
    identifiers=set().union(*(set(re.findall(r'\b[A-Za-z_]\w*\b',op.source))
        if isinstance(op,Segment)else {program.random_program.variables[op.result].code_name or op.result}
        for op in operations))
    temporary=tuple(n for n,v in program.random_program.variables.items()
        if (n not in program.variables or not program.variables[n].persistent)
        and (v.code_name or n) in identifiers)
    changed={op.result for op in operations if isinstance(op,Draw)}
    changed.update(n for op in operations if isinstance(op,Segment)for n in op.writes)
    written=tuple(n for n in temporary if n in changed)
    # Include block ranks and fused arrival markers, the largest call layout.
    maximum=2*len(_state_layout(program).groups)+len(temporary)+len(written)+11+bool(program.input_arrays)+int(_reset_ranked(program,name))
    return temporary,written,maximum


@lru_cache(maxsize=128)
def _normal_budget(program,threshold,neurons):
    """Choose work from frozen namespace ranges; never cap actual draws."""
    import ast,math
    from .expression_single_call import _normal_candidate_budget
    ordinary=_normal_candidate_budget(threshold,neurons)
    if ordinary is not None:return ordinary
    arrays={a.source_name:a for a in program.input_arrays.values()}
    def bounds(node):
        if isinstance(node,ast.Constant)and type(node.value)in(int,float):
            value=float(node.value);result=(value,value)
        elif isinstance(node,ast.Call)and isinstance(node.func,ast.Name)and node.func.id in arrays:
            array=arrays[node.func.id]
            if node.keywords or len(node.args)!=len(array.shape):raise ValueError
            result=(float(array.values.min()),float(array.values.max()))
        elif isinstance(node,ast.UnaryOp)and isinstance(node.op,(ast.UAdd,ast.USub)):
            lo,hi=bounds(node.operand)
            result=(lo,hi)if isinstance(node.op,ast.UAdd)else(-hi,-lo)
        elif isinstance(node,ast.BinOp)and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult)):
            lo,hi=bounds(node.left);low,high=bounds(node.right)
            if isinstance(node.op,ast.Add):result=(lo+low,hi+high)
            elif isinstance(node.op,ast.Sub):result=(lo-high,hi-low)
            else:
                products=(lo*low,lo*high,hi*low,hi*high);result=(min(products),max(products))
            result=(math.nextafter(result[0],-math.inf),math.nextafter(result[1],math.inf))
        else:raise ValueError
        if not all(math.isfinite(x)for x in result):raise ValueError
        return result
    try:
        node=ast.parse(threshold,mode='eval').body
        if not(isinstance(node,ast.Compare)and len(node.ops)==len(node.comparators)==1
            and isinstance(node.left,ast.Call)and isinstance(node.left.func,ast.Name)
            and node.left.func.id=='rand'and not node.left.args and not node.left.keywords):return None
        low,high=bounds(node.comparators[0])
        if isinstance(node.ops[0],(ast.Lt,ast.LtE)):probability=high
        elif isinstance(node.ops[0],(ast.Gt,ast.GtE)):probability=1-low
        else:return None
        return _normal_candidate_budget('rand()<'+repr(probability),neurons)
    except (ValueError,OverflowError):return None


@lru_cache(maxsize=128)
def selected(program,name):
    random=program.random_program
    if (random.target!='numpy'or program.scalar_stages
            or any(v.persistent and (v.dtype=='float32'or v.scalar)for v in program.variables.values())
            or any(v.scalar for v in random.variables.values())):return None
    from .configured import _state_layout
    if _bindings(program,name)[2]>31:return None
    operations=random.stages.get(name,())
    draws=[op for op in operations if isinstance(op,Draw)]
    if len(draws)!=1 or draws[0].scalar or draws[0].guard is not None or draws[0].function not in('rand','randn'):return None
    if any(op.scalar or op.may_raise for op in operations if isinstance(op,Segment)):return None
    return draws[0]


@lru_cache(maxsize=128)
def _kernel(program,name,dt,temporaries,written,block_ranks,markers):
    from .expression_compiled import namespace_bindings
    from .configured import _state_layout
    from .compiler import _metal_type
    from .binary64 import MSL_HEADER,literal
    from .binary32 import MSL_HEADER as SINGLE_HEADER
    from .math64 import MSL_MATH_HEADER
    from .expression_rng_math import NORMAL_MATH_HEADER,POISSON_EXP_HEADER
    layout=_state_layout(program);variables=program.random_program.variables
    namespace_inputs,namespace_source=namespace_bindings(program)
    loads,stores=layout.source(program)
    loads+='\n'+'\n'.join(f'{_metal_type(variables[n].dtype)} {variables[n].code_name or n}=temp_in_{j}[gid];'for j,n in enumerate(temporaries))
    draw=selected(program,name);draw_name=variables[draw.result].code_name or draw.result
    collective=_reset_ranked(program,name)
    source=[]
    for op in program.random_program.stages[name]:
        source.append(f'{draw_name}=draws[trial*NEURONS+rank];'if isinstance(op,Draw)else '{'+op.source+'}')
    temp_stores='\n'.join(f'temp_out_{j}[gid]={variables[n].code_name or n};'for j,n in enumerate(written))
    return mx.fast.metal_kernel(name='expression_numpy_packed_selected',
        input_names=['_eon_p_'+group.buffer_name for group in layout.groups]+
            [f'temp_in_{j}'for j in range(len(temporaries))]+['selected','errors','clock_times','params','draws','ranks']+(['blocks']if block_ranks else[])+(['incoming_offsets','incoming_sources','spikes']if markers else[])+list(namespace_inputs)+(['reset_counts']if collective else[]),
        output_names=['_eon_q_'+group.buffer_name for group in layout.groups]+[f'temp_out_{j}'for j in range(len(written))]+(['arrival_marker']if markers else[]),
        header=MSL_HEADER+SINGLE_HEADER+MSL_MATH_HEADER+NORMAL_MATH_HEADER+POISSON_EXP_HEADER+program.header_code,
        source=f'''
        uint gid=thread_position_in_grid.x;if(gid>=B*NEURONS)return;
        uint trial=gid/NEURONS;const int i=int(gid%NEURONS);const long N=NEURONS;
        const ulong dt={literal(dt)},t=clock_times[params[0]-params[1]];
        const long t_in_timesteps=params[0];
        {namespace_source}
        {loads}
        uint rank=ranks[gid];
        {'uint block=gid/256;rank+=block%(NEURONS/256)?blocks[block-1]:0u;'if block_ranks else''}
        {'const long _numpy_reset_rank=long(rank);const long _numpy_reset_count=long(reset_counts[trial]);'if collective else''}
        {draw_name}=0ul;
        if(selected[gid]&&!errors[trial]){{{chr(10).join(source)}}}
        {stores}
        {temp_stores}
        {"uint hit=0;for(int edge=incoming_offsets[i];edge<incoming_offsets[i+1];edge++)hit|=uint(spikes[trial*NEURONS+incoming_sources[edge]]);arrival_marker[gid]=hit<<SHIFT;"if markers else""}
        ''',compile_options={'math_mode':'safe'})


def advance(p,name,values,chosen,random,times,params,selection,marker_shift=None):
    from .configured import _state_layout,_storage_dtype,_PackedValues,_dtype
    from .expression_runtime import RandomState
    from .expression_single_call import _draw
    op=selected(p.program,name);trials=random.stream.trials
    eligible=chosen.reshape(trials,p.neurons)&(random.errors==0)[:,None]
    extra=[]
    if selection is None:
        counts=mx.sum(eligible,axis=1).astype(mx.uint32)
        ranks=mx.maximum(mx.cumsum(eligible.astype(mx.int32),axis=1)-1,0).astype(mx.uint32)
    else:
        ranks,blocks=selection
        prefix=mx.cumsum(blocks.reshape(trials,p.neurons//256),axis=1).astype(mx.uint32)
        counts=mx.where(random.errors==0,prefix[:,-1],0).astype(mx.uint32);extra=[prefix]
    budget=_normal_budget(p.program,p.model.eq_th,p.neurons)if op.function=='randn'else None
    if random.pool is not None and op.function=='randn'and p.targets and budget is not None and budget<=256:
        from .expression_sparse_normal import draw
        draws,stream,pool=draw(random,p.neurons,counts,target='numpy');buffers=random.buffers
    else:draws,stream,buffers,pool=_draw(p,random,op.function,counts)
    temporaries,written,_=_bindings(p.program,name)
    layout=_state_layout(p.program);size=trials*p.neurons
    if any(p.program.random_program.variables[n].scalar for n in temporaries):raise ValueError('Packed NumPy vector temporaries must be per-neuron')
    result=_kernel(p.program,name,p.model.dt_s,temporaries,written,selection is not None,marker_shift is not None)(
        inputs=[*values.buffers,*(random.temporaries[n]for n in temporaries),eligible.reshape(-1),random.errors,times,params,draws,ranks.reshape(-1),*extra,*([p.incoming_offsets,p.incoming_sources,chosen]if marker_shift is not None else[]),*p.input_arrays,*([counts]if _reset_ranked(p.program,name)else[])],
        template=[('NEURONS',p.neurons),('B',trials),*([('SHIFT',marker_shift)]if marker_shift is not None else[])],grid=(size,1,1),threadgroup=(min(256,size),1,1),
        output_shapes=[*layout.shapes(size),*[(size,)for n in written],*([(size,)]if marker_shift is not None else[])],
        output_dtypes=[*[_storage_dtype(group.dtype)for group in layout.groups],*[_dtype(p.program.random_program.variables[n])for n in written],*([mx.uint32]if marker_shift is not None else[])])
    markers=result[-1]if marker_shift is not None else None
    if marker_shift is not None:result=result[:-1]
    count=len(layout.groups)
    return RandomState(stream,{**random.temporaries,**dict(zip(written,result[count:]))},random.errors,pool,buffers),_PackedValues(layout,result[:count],size),markers
