"""Neuron-major native execution for Brian's compiled random expressions.

The ordinary vector path remains available where statement/neuron ordering
commutes. Stateful short-circuit expressions and mixed/sequential calls use a
cooperative MT group per _compiled_trial, with no host-dependent selection or sync.
"""
import ast
import copy
import re
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
import numpy as np
from .compiler import _Renderer,_Expression,_metal_type,UnsupportedModelError


@dataclass(frozen=True)
class CompiledStage:
    scalar: str
    vector: str
    writes: tuple
    single_call: str | None = None
    may_raise: bool = False
    reference_call_aliases: tuple = ()
    scalar_reference_call_aliases: tuple = ()
    reference_call_guards: tuple = ()


class _OrderedRenderer(_Renderer):
    def __init__(self,types,values,target,options=None):
        super().__init__(types,values,reference_target=target,**(options or {}));self.lines=[];self.serial=0;self.target=target
        self.guards=[];self.draw_guards=[];self.random_ids={};self.guard_inputs=set()

    def record_guard(self,node):
        from .expression_guard import expression
        self.random_ids.setdefault(node,len(self.random_ids))
        self.draw_guards.append(('and',tuple(expression(part,self.types,self.random_ids,self.guard_inputs)for part in self.guards)))

    def temporary(self):
        name=f'_compiled_draw_{self.serial}';self.serial+=1;return name

    def assignment(self,statement,expression=None,declare=True):
        if self.target=='cython'and statement.op=='/=':
            expression=statement.expr if expression is None else expression
            replacement=copy.copy(statement);replacement.op='='
            return super().assignment(replacement,f'{statement.var}/({expression})',declare)
        previous=getattr(self,'integer_assignment',False)
        self.integer_assignment=np.dtype(statement.dtype).kind in 'iu'
        try:return super().assignment(statement,expression,declare)
        finally:self.integer_assignment=previous

    def expression(self,source):
        node=ast.parse(source,mode='eval').body if isinstance(source,str)else source
        if self.target=='cython':
            from .expression_integer_segments import render
            result=render(self,node,ordered=True,force=getattr(self,'integer_assignment',False))
            if result is not None:return result
            from .expression_profile_boxed import lower_large_literal
            result=lower_large_literal(self,node)
            if result is not None:return result
        if self.target=='cython'and isinstance(node,ast.BinOp)and isinstance(node.op,ast.Div):
            # Emit the check now, before visiting a subsequent sibling draw.
            # Deferring it inside a final expression would consume that draw
            # even when Cython raises while evaluating the earlier operand.
            a=self.convert(self.expression(node.left),'float').code
            b=self.convert(self.expression(node.right),'float').code
            result=self.temporary()
            self.lines.append(f'const ulong {result}=compiled_div({a},{b},_compiled_rng);')
            return _Expression(result,'float')
        if isinstance(node,ast.Call)and isinstance(node.func,ast.Name):
            name=node.func.id
            if name in ('rand','randn','poisson'):
                if node.keywords or len(node.args)!=(1 if name=='poisson'else 0):
                    raise ValueError(f'Invalid built-in random call {name}')
                result=self.temporary()
                if name=='poisson':
                    argument=self.convert(self.expression(node.args[0]),'float').code
                    call=f'compiled_poisson({argument},_compiled_rng)'
                    self.record_guard(node)
                    self.lines.append(f'const int {result}={call};')
                    return _Expression(result,'int')
                else:
                    call=f'{"compiled_rand"if name=="rand"else"compiled_normal"}(_compiled_rng)'
                    self.record_guard(node)
                    self.lines.append(f'const ulong {result}={call};')
                    return _Expression(result,'float')
        if isinstance(node,ast.BoolOp):
            result=self.temporary();first=self.convert(self.expression(node.values[0]),'bool')
            self.lines.append(f'bool {result}={first.code};')
            for number,operand in enumerate(node.values[1:],1):
                self.lines.append(f'if({result if isinstance(node.op,ast.And)else "!"+result}) {{')
                prefix=node.values[0]if number==1 else ast.BoolOp(op=node.op,values=node.values[:number])
                self.guards.append(prefix if isinstance(node.op,ast.And)else ast.UnaryOp(op=ast.Not(),operand=prefix))
                value=self.convert(self.expression(operand),'bool')
                self.guards.pop()
                self.lines.extend((f'{result}={value.code};','}'))
            return _Expression(result,'bool')
        if isinstance(node,ast.IfExp):
            condition=self.convert(self.expression(node.test),'bool')
            current=self.lines;self.lines=[];self.guards.append(node.test)
            a=self.expression(node.body);yes=self.lines;self.guards.pop()
            self.lines=[];self.guards.append(ast.UnaryOp(op=ast.Not(),operand=node.test))
            b=self.expression(node.orelse);no=self.lines;self.guards.pop();self.lines=current
            kind='float'if 'float'in(a.kind,b.kind)else a.kind;name=self.temporary()
            dtype={'float':'ulong','int':'long','bool':'bool'}[kind]
            self.lines.extend((f'{dtype} {name};',f'if({condition.code}) {{',*yes,
                f'{name}={self.convert(a,kind).code};','} else {',*no,
                f'{name}={self.convert(b,kind).code};','}'))
            return _Expression(name,kind)
        return super().expression(node)

    def emit(self,statement,guard):
        self.types[statement.var]=statement.dtype
        branches=getattr(statement,'boolean_simplified_expressions',None)
        if guard:
            self.lines.append(f'if({self.code_name(guard)}) {{');self.guards.append(ast.parse(guard,mode='eval').body)
        if branches:
            if statement.op==':='and statement.var!='_cond':
                self.lines.append(f'{_metal_type(statement.dtype)} {self.code_name(statement.var)};')
            for number,(conditions,expression)in enumerate(branches.items()):
                condition=' && '.join(self.code_name(name) if value else '!'+self.code_name(name) for name,value in conditions)
                self.lines.append(f'{"if"if number==0 else"else if"}({condition}) {{')
                self.guards.append(ast.BoolOp(op=ast.And(),values=[ast.Name(id=name,ctx=ast.Load())if value else
                    ast.UnaryOp(op=ast.Not(),operand=ast.Name(id=name,ctx=ast.Load()))for name,value in conditions]))
                code=self.assignment(statement,expression,declare=False)
                self.guards.pop()
                if self.target=='cython'and statement.var in self.persistent:
                    code=f'if(!_compiled_rng.error){{{code}}}'
                self.lines.extend((code,'}'))
        else:
            code=self.assignment(statement)
            if self.target=='cython'and statement.var in self.persistent:
                code=f'if(!_compiled_rng.error){{{code}}}'
            self.lines.append(code)
        if guard:self.lines.append('}');self.guards.pop()


def build_stages(inventory,constants,statements,*,target='cpp',renderer_options=None):
    from .compiler import _evaluate
    from .symbols import variable_code_names
    code_names=variable_code_names({name for name in inventory}|{
        statement.var for scalar,vector in statements.values()for statement in (*scalar,*vector)})
    code_names.update({name:v.code_name for name,v in inventory.items()if v.persistent})
    code_names.update({name:name for name in ('_cond','t','dt','N','i','t_in_timesteps')})
    options=dict(renderer_options or {},code_names=code_names)
    options.pop('numpy_store_orders',None)
    result={};required=[]
    randoms=lambda node:[item for item in ast.walk(node)if isinstance(item,ast.Call)
        and isinstance(item.func,ast.Name)and item.func.id in('rand','randn','poisson')]
    def conditional_calls(node,conditional=False):
        if isinstance(node,ast.BoolOp):
            return any(conditional_calls(value,conditional or i>0)for i,value in enumerate(node.values))
        if isinstance(node,ast.IfExp):
            return (conditional_calls(node.test,conditional)or conditional_calls(node.body,True)
                    or conditional_calls(node.orelse,True))
        if (conditional and isinstance(node,ast.Call)and isinstance(node.func,ast.Name)
                and node.func.id in('rand','randn','poisson')):return True
        return any(conditional_calls(child,conditional)for child in ast.iter_child_nodes(node))
    for stage,(scalar,vector)in statements.items():
        writes=[];sources=[];serial=False;block_calls=[];may_raise=False;block_aliases=[];block_guards=[]
        carried_types={name:np.dtype(v.dtype)for name,v in inventory.items()}
        for block in(scalar,vector):
            renderer=_OrderedRenderer(carried_types.copy(),dict(constants),target,options)
            renderer.guard_inputs={name for name,v in inventory.items()if v.persistent or name in('i','N','t_in_timesteps')}
            renderer.persistent={name for name,v in inventory.items()if v.persistent}|{'_cond'}
            calls=0;functions=[];aliases=[]
            for statement in block:
                if statement.var in constants:continue
                node=ast.parse(statement.expr,mode='eval').body;calls+=len(randoms(node))
                functions.extend(item.func.id for item in randoms(node))
                checked=(ast.BinOp(left=ast.Name(id=statement.var,ctx=ast.Load()),op=ast.Div(),right=node)
                         if statement.op=='/='else node)
                may_raise|=target=='cython'and any(isinstance(part,ast.BinOp)and isinstance(part.op,ast.Div)
                                                for part in ast.walk(checked))
                for item in randoms(node):
                    if item.func.id=='poisson':
                        try:
                            rate=float(_evaluate(item.args[0],constants))
                            constant=np.isfinite(rate)and(target=='cpp'or(target=='cython'and(rate<10 or stage=='threshold')))
                        except KeyError:constant=False
                        serial|=not constant
                serial|=conditional_calls(node)
                variable=inventory.get(statement.var)
                guard=variable.conditional_write if stage=='state'and variable is not None else None
                # Brian's Boolean optimization can duplicate an identical
                # stateful call sequence into mutually exclusive arms. Keep
                # one lexical sequence for the arithmetic registry, while
                # retaining aliases to every unchanged reference callsite.
                def trace(part):
                    result=[]
                    for child in ast.iter_child_nodes(part):result.extend(trace(child))
                    if isinstance(part,ast.Call)and isinstance(part.func,ast.Name)and part.func.id in('rand','randn','poisson'):
                        result.append(ast.dump(part,include_attributes=False))
                    return result
                original=trace(node);branches=getattr(statement,'boolean_simplified_expressions',None)
                shared=(len(original)>1 and branches and all(trace(ast.parse(expr,mode='eval').body)==original for expr in branches.values()))
                emitted=copy.copy(statement)if shared else statement
                if shared:emitted.boolean_simplified_expressions=None
                count_draws=lambda:sum(bool(re.search(r'=compiled_(?:rand|normal|poisson)\(',line))for line in renderer.lines)
                before=count_draws();renderer.emit(emitted,guard);after=count_draws()
                aliases.extend([before+i for _ in branches for i in range(after-before)]if shared else range(before,after))
                if (variable is not None and variable.persistent)or statement.var=='_cond':writes.append(statement.var)
            may_raise|=getattr(renderer,'integer_may_raise',False)
            serial|=calls>1;sources.append('\n'.join(renderer.lines));block_calls.append(functions);block_aliases.append(tuple(aliases))
            block_guards.append(tuple(renderer.draw_guards))
            carried_types.update(renderer.types)
        if stage=='threshold':
            sources[1]+=f'\nif(_cond){{{code_names["lastspike"]}=t;{code_names["not_refractory"]}=false;}}'
            writes.extend(('lastspike','not_refractory'))
        # Boolean simplification can duplicate a call textually in mutually
        # exclusive branches. Count original statement calls, not emitted text.
        single=(block_calls[1][0]if not may_raise and not block_calls[0]and len(block_calls[1])==1
                and block_calls[1][0]in('rand','randn')else None)
        result[stage]=CompiledStage(*sources,tuple(dict.fromkeys(writes)),single,may_raise,block_aliases[1],block_aliases[0],block_guards[1])
        if serial or may_raise:required.append(stage)
    return MappingProxyType(result),tuple(required)


_CONTEXT=r'''
struct CompiledRng {
    threadgroup uint *mt,*boundary,*warps,*end;
    threadgroup ulong *pair;
    device ulong *ubuf,*nbuf;
    uint pos,cached,lane,error,ui,ni;
    ulong saved;
};
'''

_NORMAL_CPP=r'''
inline ulong compiled_rand(thread CompiledRng& rng) {
 return mt_next_double(rng.mt,&rng.pos,rng.lane);
}
inline ulong compiled_normal(thread CompiledRng& rng) {
 if(rng.cached){rng.cached=0;return rng.saved;}
 const ulong one=0x3ff0000000000000ul,two=0x4000000000000000ul;
 while(true) {
  ulong x=d_sub(d_mul(two,compiled_rand(rng)),one);
  ulong y=d_sub(d_mul(two,compiled_rand(rng)),one);
  ulong r=expr_fma(y,y,d_mul(x,x));
  if(d_ge(r,one)||d_eq(r,0ul))continue;
  ulong f=d_sqrt(d_div(d_mul(0xc000000000000000ul,expr_apple_log(r)),r));
  rng.saved=d_mul(f,x);rng.cached=1;return d_mul(f,y);
 }
}
'''


def _layout(variables,neurons):
    offsets={};width=0
    for name,scalar in variables:
        offsets[name]=width;width+=1 if scalar else neurons
    return offsets,width


def namespace_bindings(program):
    """Bind the packed namespace without assuming a Metal address space."""
    if not program.input_arrays:return (),''
    buffer=next(iter(program.input_arrays));offset=0;aliases=[]
    for name,array in program.input_arrays.items():
        if name!=buffer:aliases.append(f'auto {name}={buffer}+{offset}L;')
        offset+=array.values.size
    return (buffer,),'\n'.join(aliases)


@lru_cache(maxsize=128)
def _kernel(program,name,dt):
    import mlx.core as mx
    from .binary64 import MSL_HEADER,literal
    from .binary32 import MSL_HEADER as SINGLE_HEADER
    from .math64 import MSL_MATH_HEADER
    from .expression_rng import _MT_HEADER
    from .expression_rng_math import NORMAL_MATH_HEADER,POISSON_EXP_HEADER
    from .expression_poisson_compiled import source as poisson_source
    stage=program.random_program.compiled_stages[name]
    namespace_inputs,namespace_source=namespace_bindings(program)
    cython=program.random_program.target=='cython'
    if cython:
        from .expression_cython import compiled_header
        samplers=compiled_header()
    else:samplers=_NORMAL_CPP
    variables={n:v for n,v in program.variables.items()if v.persistent}
    offset={};width='0'
    for n,v in variables.items():
        offset[n]=width;width+='+'+('1'if v.scalar else'NEURONS')
    input_width=width;offset_out={};width='0'
    for n in stage.writes:
        offset_out[n]=width;width+='+'+('1'if n in variables and variables[n].scalar else'NEURONS')
    output_width='('+width+')'
    load=lambda scalar:'\n'.join(f'{v.metal_type} {v.code_name}={v.metal_type}(_compiled_p_values[_compiled_trial*({input_width})+({offset[n]})'+
        (''if scalar else'+i')+']);'for n,v in variables.items()if v.scalar==scalar)
    def stores(scalar):
        lines=[]
        for n in stage.writes:
            if bool(n in variables and variables[n].scalar)!=scalar:continue
            value=f'ulong({variables[n].code_name if n in variables else n})'
            if cython:
                # Brian writes this neuron's locals back only after its
                # complete block succeeds; draws before a fault still count.
                old=(f'_compiled_p_values[_compiled_trial*({input_width})+({offset[n]})'+
                     (''if scalar else'+i')+']')if n in variables else'0ul'
                error='_compiled_scalar_error'if scalar else'_compiled_rng.error'
                value=f'({error}?{old}:{value})'
            lines.append(f'_compiled_q_values[_compiled_trial*{output_width}+({offset_out[n]})'+
                (''if scalar else'+i')+f']={value};')
        return'\n'.join(lines)
    return mx.fast.metal_kernel(name='expression_compiled_stage',
        input_names=['_compiled_p_values','_compiled_packet','_compiled_selected','_compiled_clock_times','_compiled_params']+(['_compiled_buffers']if cython else[])+list(namespace_inputs),
        output_names=['_compiled_q_values','_compiled_next_packet']+(['_compiled_next_buffers']if cython else[]),
        header=MSL_HEADER+SINGLE_HEADER+MSL_MATH_HEADER+NORMAL_MATH_HEADER+POISSON_EXP_HEADER+program.header_code+_MT_HEADER+_CONTEXT+samplers+poisson_source(
            'thread CompiledRng& rng','compiled_rand(rng)',target=program.random_program.target,error='rng.error|=2u;',
            check='if(rng.error)return 0;'),source=f'''
        uint _compiled_lane=thread_position_in_threadgroup.x,_compiled_trial=threadgroup_position_in_grid.x;
        threadgroup uint _compiled_mt[624],_boundary[4],_warps[8],_end;threadgroup ulong _pair;
        for(uint j=_compiled_lane;j<624;j+=256)_compiled_mt[j]=uint(_compiled_packet[_compiled_trial*630+j]);
        CompiledRng _compiled_rng;
        _compiled_rng.mt=_compiled_mt;_compiled_rng.pos=uint(_compiled_packet[_compiled_trial*630+624]);
        _compiled_rng.cached=uint(_compiled_packet[_compiled_trial*630+625]);_compiled_rng.saved=_compiled_packet[_compiled_trial*630+626];
        _compiled_rng.error=uint(_compiled_packet[_compiled_trial*630+627]);_compiled_rng.lane=_compiled_lane;
        _compiled_rng.boundary=_boundary;_compiled_rng.warps=_warps;_compiled_rng.end=&_end;_compiled_rng.pair=&_pair;
        {"""for(uint j=_compiled_lane;j<40000;j+=256)_compiled_next_buffers[_compiled_trial*40000+j]=_compiled_buffers[_compiled_trial*40000+j];
        _compiled_rng.ubuf=_compiled_next_buffers+_compiled_trial*40000;_compiled_rng.nbuf=_compiled_next_buffers+_compiled_trial*40000+20000;
        _compiled_rng.ui=uint(_compiled_packet[_compiled_trial*630+628]);_compiled_rng.ni=uint(_compiled_packet[_compiled_trial*630+629]);
        threadgroup_barrier(mem_flags::mem_device);"""if cython else"_compiled_rng.ui=0;_compiled_rng.ni=0;"}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const long N=NEURONS,t_in_timesteps=_compiled_params[0];const int i=0;
        const ulong dt={literal(dt)},t=_compiled_clock_times[_compiled_params[0]-_compiled_params[1]];
        {namespace_source}
        {load(True)}
        {stage.scalar}
        {"uint _compiled_scalar_error=_compiled_rng.error;"if cython else""}
        for(int i=0;i<NEURONS;++i) {{
            uint _compiled_gid=_compiled_trial*NEURONS+i;{load(False)}bool _cond=false;
            if({"_compiled_selected[_compiled_gid]"if name=='reset'else'true'}) {{ {stage.vector} }}
            if(_compiled_lane==0){{ {stores(False)} }}
        }}
        if(_compiled_lane==0){{ {stores(True)} }}
        for(uint j=_compiled_lane;j<624;j+=256)_compiled_next_packet[_compiled_trial*630+j]=ulong(_compiled_mt[j]);
        if(_compiled_lane==0){{_compiled_next_packet[_compiled_trial*630+624]=_compiled_rng.pos;_compiled_next_packet[_compiled_trial*630+625]=_compiled_rng.cached;
            _compiled_next_packet[_compiled_trial*630+626]=_compiled_rng.saved;_compiled_next_packet[_compiled_trial*630+627]=_compiled_rng.error;
            _compiled_next_packet[_compiled_trial*630+628]=_compiled_rng.ui;_compiled_next_packet[_compiled_trial*630+629]=_compiled_rng.ni;}}
        ''',compile_options={'math_mode':'safe'})


def stage(p,name,values,random,trials,clock_times,params):
    import mlx.core as mx
    from .configured import _dtype
    from .expression_rng import MT19937State
    from .expression_runtime import RandomState
    program=p.program;op=program.random_program.compiled_stages[name]
    variables={n:v for n,v in program.variables.items()if v.persistent}
    selected=values['_cond']if name=='reset'else mx.ones((trials*p.neurons,),mx.bool_)
    s=random.stream;buffers=random.buffers
    zero=mx.zeros((trials,),mx.uint32)
    metadata=[s.position,s.has_gauss,s.gauss,random.errors,
        zero if buffers is None else buffers.uniform_index,zero if buffers is None else buffers.normal_index]
    packet=mx.concatenate([s.words.reshape(trials,624).astype(mx.uint64),
        *(v[:,None].astype(mx.uint64)for v in metadata)],axis=1)
    data=mx.concatenate([values[n].reshape(trials,-1).astype(mx.uint64)for n in variables],axis=1)
    extras=[]if buffers is None else[mx.concatenate([buffers.uniform,buffers.normal],axis=1)]
    layout,total=_layout([(n,n in variables and variables[n].scalar)for n in op.writes],p.neurons)
    output=_kernel(program,name,p.model.dt_s)(
        inputs=[data,packet,selected,clock_times,params,*extras,*p.input_arrays],
        template=[('NEURONS',p.neurons)],grid=(trials*256,1,1),threadgroup=(256,1,1),
        output_shapes=[(trials*max(1,total),),(trials*630,)]+([(trials*40000,)]if buffers is not None else[]),
        output_dtypes=[mx.uint64,mx.uint64]+([mx.uint64]if buffers is not None else[]))
    written=output[0].reshape(trials,max(1,total))
    for n in op.writes:
        width=1 if n in variables and variables[n].scalar else p.neurons
        dtype=mx.bool_ if n=='_cond'else _dtype(variables[n])
        values[n]=written[:,layout[n]:layout[n]+width].reshape(-1).astype(dtype)
    packet=output[1].reshape(trials,630)
    stream=MT19937State(packet[:,:624].reshape(-1).astype(mx.uint32),packet[:,624].astype(mx.uint32),
        trials,packet[:,625].astype(mx.uint32),packet[:,626])
    if buffers is not None:
        from .expression_cython import CythonBuffers
        data=output[2].reshape(trials,40000)
        buffers=CythonBuffers(data[:,:20000],packet[:,628].astype(mx.uint32),
            data[:,20000:],packet[:,629].astype(mx.uint32))
    return RandomState(stream,random.temporaries,packet[:,627].astype(mx.uint32),buffers=buffers)
