"""Once-per-trial storage and bounded phase snapshots for shared Brian state."""
from functools import lru_cache
import ast
import re

import mlx.core as mx

from .compiler import CompiledVariable


def variables(program):
    return tuple(v for v in program.variables.values() if v.persistent and v.scalar)


def enabled(program):
    return bool(program.scalar_stages)


def independent_steps(program, statements, phases, *, error_transport=False):
    """Prove scalar snapshots need no value written by a previous timestep.

    Inspect Brian's selected scalar statements in execution order. Parameters
    that are never written remain common inputs. A written field must be
    overwritten before its first read in every timestep. Error transport and
    shared Poisson voltage retain the serial path.
    """
    if (program.index_error_name and not error_transport) or program.variables['v'].scalar:
        return False
    fields={v.name for v in variables(program)}
    written={statement.var for block in statements.values() for statement in block
             if statement.var in fields}
    defined=set()
    stages={'groups':'state','thresholds':'threshold','synapses':'synapses','resets':'reset'}
    for stage in ('subexpression',)+tuple(stages[phase] for phase in phases):
        for statement in statements.get(stage,()):
            reads={node.id for node in ast.walk(ast.parse(statement.expr,mode='eval'))
                   if isinstance(node,ast.Name)}
            if statement.op not in ('=',':='):reads.add(statement.var)
            if (reads & written)-defined:return False
            if statement.var in written:
                if program.variables[statement.var].conditional_write:return False
                defined.add(statement.var)
    return True


def declared_variables(source):
    """Carry the actual renderer's scalar locals, including NumPy aliases.

    Declarations inside an inner branch cannot escape that branch and do not
    need snapshots. The surrounding renderer retains their bindings/precision.
    """
    dtypes = {'ulong':'uint64','long':'int64','uint':'uint32','int':'int32',
              'ushort':'uint16','short':'int16','uchar':'uint8','char':'int8','bool':'bool'}
    result = {}
    for match in re.finditer(r'\b(?:const\s+)?(ulong|long|uint|int|ushort|short|uchar|char|bool)\s+(\w+)\s*(?:=|;)',source):
        prefix=source[:match.start()]
        if prefix.count('{') != prefix.count('}'):
            continue
        metal,name=match.groups()
        result[name]=CompiledVariable(name,dtypes[metal],metal,True,False,True,None,False)
    return tuple(result.values())


@lru_cache(maxsize=16)
def layout(program):
    shared=variables(program)
    offset=len(shared)
    phases={}
    for name in ('subexpression','state','threshold','synapses','reset'):
        stage=program.scalar_stages.get(name)
        temps=stage.variables if stage is not None else ()
        fields=shared+tuple(temps)
        phases[name]=tuple((v,offset+i) for i,v in enumerate(fields))
        offset+=len(fields)
        if program.index_error_name:
            offset+=2
    return shared,phases,max(1,offset)


def error_offset(program,name):
    shared,phases,_=layout(program)
    offset=len(shared)
    for phase,fields in phases.items():
        if phase==name:return offset+len(fields)
        offset+=len(fields)+(2 if program.index_error_name else 0)
    raise KeyError(name)


def phase_load(program,name, *, aliases=None):
    if not enabled(program):return ''
    aliases=aliases or {}
    _,phases,_=layout(program)
    lines=[]
    for v,slot in phases[name]:
        # Scalar temporaries are already emitted identifiers. Only persistent
        # fields cross from logical host names through the code_name hook.
        local=aliases.get(v.name,v.code_name) if v.persistent else v.name
        declaration='' if v.persistent else v.metal_type+' '
        lines.append(f'{declaration}{local} = {v.metal_type}(_eon_shared_trace[_eon_shared_offset+{slot}]);')
    if program.index_error_name:
        end=error_offset(program,name)
        error=program.index_error_name
        kind=(f'{error}_kind=int(_eon_shared_trace[_eon_shared_offset+{end+1}]);'
              if program.runtime_error_type in ('zero_division','model') else '')
        lines.append(f'if (!{error} && _eon_shared_trace[_eon_shared_offset+{end}]) {{ {error}=true; {kind} }}')
    return '\n'.join(lines)


def phase_source(program,name,*,aliases=None):
    stage=program.scalar_stages[name]
    return '{\n'+phase_load(program,name,aliases=aliases)+'\n'+stage.vector_code+'\n}'


@lru_cache(maxsize=16)
def kernel(model,program,stop_before_synapses=False):
    from .binary64 import MSL_HEADER,literal
    from .binary32 import MSL_HEADER as SINGLE_HEADER
    from .math64 import MSL_MATH_HEADER
    from .expression_compiled import namespace_bindings

    shared,phases,slots=layout(program)
    loads='\n'.join(f'{v.metal_type} {v.code_name} = {v.metal_type}(_eon_shared_initial[trial*{len(shared)}+{i}]);'
                    for i,v in enumerate(shared))
    stores='\n'.join(f'_eon_shared_trace[(base+length)*{slots}+{i}] = ulong({v.code_name});'
                     for i,v in enumerate(shared))
    inputs=['_eon_shared_initial','amplitudes','events','clock_times','params']
    namespace_inputs,namespace_source=namespace_bindings(program)
    inputs.extend(namespace_inputs)
    if program.shared_timed_calls:inputs.append(program.timed_cache_name)
    error=(f'bool {program.index_error_name}=false;' if program.index_error_name else '')

    def phase(name):
        stage=program.scalar_stages.get(name)
        code=stage.scalar_code if stage is not None else ''
        samples='\n'.join(f'_eon_shared_trace[_eon_shared_offset+{slot}]=ulong({v.code_name if v.persistent else v.name});'
                          for v,slot in phases[name])
        if program.index_error_name:
            end=error_offset(program,name)
            samples+=f'\n_eon_shared_trace[_eon_shared_offset+{end}]=ulong({program.index_error_name});'
            kind=program.index_error_name+'_kind' if program.runtime_error_type in ('zero_division','model') else '0'
            samples+=f'\n_eon_shared_trace[_eon_shared_offset+{end+1}]=ulong({kind});'
        backups=restore=''
        if program.index_error_name:
            backups='\n'.join(f'{v.metal_type} _eon_shared_backup_{i}={v.code_name};' for i,v in enumerate(shared))
            restore='if('+program.index_error_name+') {'+' '.join(f'{v.code_name}=_eon_shared_backup_{i};' for i,v in enumerate(shared))+'}'
        return '{\n'+backups+'\n'+code+'\n'+restore+'\n'+samples+'\n}'

    delivery=''
    if program.variables['v'].scalar:
        v=program.variables['v']
        add='d_add('+v.code_name+', weight)'
        if v.dtype=='float32':
            # A scalar stateupdate retains the double namespace weight.
            # The compiled vector input's float32 invariant does not exist.
            add=f'f_from_d(d_add(f_to_d({v.code_name}), weight))'
        delivery=f'''for(int channel=0;channel<C;++channel) {{
            uchar event=events[(trial*staged_steps+step-event_begin)*C+channel];
            if(event!=2u) {{ ulong weight=event==1u ? amplitudes[channel] : (amplitudes[channel] & (1ul<<63));
                {v.code_name}={add}; }}
        }}'''
    stage_names={'groups':'state','thresholds':'threshold','synapses':'synapses','resets':'reset','subexpression':'subexpression'}
    names=(program.replay_phase,) if program.replay_phase is not None else model.execution_phases
    if stop_before_synapses:names=names[:names.index('synapses')]
    before=phase('subexpression') if program.replay_phase is None else ''
    body=before+'\n'+'\n'.join((delivery+'\n' if name=='synapses' else '')+phase(stage_names[name]) for name in names)
    parallel=program.scalar_parallel
    source=f'''
        int length=int(params[1]),staged_steps=int(params[2]);
        uint point=thread_position_in_grid.x;
        uint trial={'point/length' if parallel else 'point'};
        if(trial>=B)return;
        const long N=NEURONS;
        const ulong dt={literal(model.dt_s)};
        long start=params[0],event_begin=params[3];
        int first={'point%length' if parallel else '0'};
        uint base=trial*(length+1);
        {loads}
        {namespace_source}
        {error}
        {program.scalar_code}
        for(int k=first;k<{'first+1' if parallel else 'length'};++k) {{
            long step=start+k,t_in_timesteps=step;
            ulong t=clock_times[step-event_begin];
            uint _eon_shared_offset=(base+k)*{slots};
            {body}
        }}
        {'if(first==length-1)' if parallel else ''} {{ {stores} }}
    '''
    for query,call in enumerate(program.shared_timed_calls):
        source=source.replace(call,f'{program.timed_cache_name}[{query}L*staged_steps+(step-event_begin)]')
    return mx.fast.metal_kernel(name='fly_shared_phase_trace',input_names=inputs,
        output_names=['_eon_shared_trace'],source=source,
        header=MSL_HEADER+SINGLE_HEADER+MSL_MATH_HEADER+program.header_code,compile_options={'math_mode':'safe'})
