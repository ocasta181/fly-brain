"""Split Brian statements only at stateful draw dependencies.

Brian supplies typed, optimized scalar and vector statements. Each segment is
lowered by the existing binary64 renderer. Random calls become explicit vector
operations, preserving NumPy's left-to-right eager evaluation and statement
order; a device mask determines the number and assignment of selected draws.
"""
import ast
import copy
import re
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
import numpy as np


@dataclass(frozen=True)
class Temporary:
    dtype: str
    scalar: bool
    code_name: str = ''


@dataclass(frozen=True,eq=False)
class Segment:
    source: str
    writes: tuple[str,...]
    scalar: bool
    may_raise: bool = False
    index_fault: tuple | None = None


@dataclass(frozen=True,eq=False)
class Draw:
    function: str
    result: str
    argument: str | None
    scalar: bool
    guard: str | None
    constant_argument: float | None = None
    argument_scalar: bool = False
    host_views: tuple[tuple[str, str], ...] = ()


def _numpy_status_draw(result, argument, scalar, guard, renderer, *, function='numpy_status'):
    # NumPy copies any whole-vector array written in this code block; reset
    # indexing and scalar indexing also copy. Only read-only vector sources
    # remain actual host-array views while a user error handler executes.
    views = () if scalar or renderer.stage == 'reset' else tuple(
        (name, renderer.shadows.get(name, name)) for name in renderer.persistent
        if name not in renderer.scalars and name not in renderer.numpy_stage_writes)
    return Draw(function, result, argument, scalar, guard, host_views=views)


@dataclass(frozen=True,eq=False)
class RandomProgram:
    stages: object
    variables: object
    target: str = 'numpy'
    compiled_stages: object = None
    serial_required: tuple = ()
    statements: object = None
    checked: bool = False


def build_program(inventory,constants,stage_statements,*,target='numpy',renderer_options=None,
                  floating_status=False):
    from .compiler import _Renderer,_evaluate,_Expression,_TimedArray
    from .symbols import variable_code_name
    from brian2.utils.stringtools import word_substitute
    variables={name:Temporary(v.dtype,v.scalar,v.code_name) for name,v in inventory.items() if v.persistent}
    types={name:np.dtype(v.dtype) for name,v in inventory.items()}
    code_names={name:v.code_name for name,v in inventory.items()}
    code_names.update({name:name for name in ('_cond','t','dt','N','i','t_in_timesteps')})
    options=dict(renderer_options or {})
    numpy_store_orders=options.pop('numpy_store_orders',None) or {}
    options['reference_target']=target
    class SegmentRenderer(_Renderer):
        def compiled_operator_expression(self,node):
            if target=='cython'and any(isinstance(part,ast.Constant)and type(part.value)is int
                    and not -(2**63)<=part.value<2**63 for part in ast.walk(node)):
                return True
            return super().compiled_operator_expression(node)

        def expression(self,source):
            node=ast.parse(source,mode='eval').body if isinstance(source,str)else source
            if target=='cython':
                from .expression_integer_segments import render
                result=render(self,node)
                if result is not None:return result
                from .expression_profile_boxed import lower_large_literal
                result=lower_large_literal(self,node)
                if result is not None:return result
            if target=='cython'and isinstance(node,ast.BinOp)and isinstance(node.op,ast.Div):
                # Cython division executes through the cooperative stage's
                # sticky RNG/error context, never this unused token segment.
                a,b=(self.convert(self.expression(part),'float').code for part in(node.left,node.right))
                return _Expression(f'd_div({a},{b})','float')
            return super().expression(node)
    # Typed array/scalar promotion and integer ufuncs also apply to binary64 models.
    numpy_locals=target=='numpy'
    stages={};serial=0
    def bind_temporary(name,dtype,scalar):
        code_names[name]=variable_code_name(name)
        variables[name]=Temporary(dtype,scalar,code_names[name])
        renderer.types[name]=types[name]=np.dtype(dtype)

    for stage,(scalar,vector) in stage_statements.items():
        operations=[]
        aliases={}
        for statement in (*scalar,*vector):
            name=statement.var
            if (name in aliases or (name in inventory and inventory[name].persistent)
                    or name in constants or (stage=='threshold' and name=='_cond')):
                continue
            candidate=f'_expr_{stage}_{name}'
            while candidate in types or candidate in aliases.values():candidate+='_'
            aliases[name]=candidate
        renderer=None;seen_prologue=0;complex_shadows={}
        for is_scalar,statements in ((True,scalar),(False,vector)):
            if numpy_locals and renderer is None:
                from .numpy_complex import BasicRenderer as Renderer
                if floating_status:
                    class StatusRenderer(Renderer):
                        numpy_operation_boundaries=True
                        def evaluate(self,node):
                            parts=tuple(ast.walk(node))
                            python_only=(not any(isinstance(part,ast.Call) for part in parts)
                                and all(type(self._invariant_values().get(part.id))
                                    in (int,float,bool,complex) for part in parts
                                    if isinstance(part,ast.Name)))
                            if (not python_only and any(isinstance(part,(ast.BinOp,ast.Call))
                                                        for part in parts)):
                                raise KeyError('NumPy ufunc policy executes at its native operation boundary')
                            return super().evaluate(node)
                    Renderer=StatusRenderer
                renderer=Renderer(types.copy(),dict(constants),
                    {name for name,v in inventory.items()if v.persistent},
                    {name for name,v in inventory.items()if v.scalar},stage,
                    code_names=code_names,store_order=numpy_store_orders.get(stage,()),**options)
            elif not numpy_locals:renderer=SegmentRenderer(types.copy(),dict(constants),code_names=code_names,**options)
            else:renderer.persistent=renderer.persistent-{name for name,v in inventory.items()if v.scalar}
            if numpy_locals:
                renderer.numpy_stage_writes = frozenset(s.var for s in statements)
                if floating_status:
                    # These are the actual NumPy block-entry copies. Creating
                    # them before the first operation prevents a callback's
                    # physical-array write from changing an existing local.
                    referenced=renderer.numpy_stage_writes | frozenset(
                        part.id for statement in statements
                        for part in ast.walk(ast.parse(statement.expr,mode='eval'))
                        if isinstance(part,ast.Name))
                    for name in sorted(renderer.persistent & referenced):
                        if ((name in renderer.scalars)==is_scalar and
                                (is_scalar or stage=='reset' or name in renderer.numpy_stage_writes)):
                            renderer._shadow(name)
                    renderer.numpy_callback_locals=True
            lines=[];writes=[]
            def flush(index_fault=None,scalar_mode=None):
                nonlocal seen_prologue
                mode=is_scalar if scalar_mode is None else scalar_mode
                if lines:
                    mode=is_scalar if scalar_mode is None else scalar_mode
                    initializers=[]
                    if numpy_locals and renderer.index_error_name:
                        flag=renderer.index_error_name
                        initializers.append(f'bool {flag}=false; int {flag}_kind=0;')
                    if numpy_locals:
                        new_complex=set()
                        for declaration in renderer.prologue[seen_prologue:]:
                            pair=re.fullmatch(r'_eon_complex (\w+) = (.*);',declaration)
                            if pair is not None:
                                name,value=pair.groups()
                                components=tuple(name+'_'+part for part in ('real','imag'))
                                complex_shadows[name]=components
                                for component in components:
                                    code_names[component]=component;types[component]=np.dtype('uint64')
                                    variables[component]=Temporary('uint64',mode,component)
                                initializers.append(declaration);new_complex.add(name)
                                continue
                            match=re.fullmatch(r'ulong (\w+) = (.*);',declaration)
                            if match is None:raise ValueError('Unknown NumPy random local declaration: '+declaration)
                            name,value=match.groups()
                            code_names[name]=name;types[name]=np.dtype('uint64')
                            variables[name]=Temporary('uint64',mode,name)
                            initializers.append(name+'='+value+';');writes.append(name)
                        seen_prologue=len(renderer.prologue)
                        # A complex local lives inside one kernel. Only its
                        # two raw component words cross native boundaries.
                        for name,(real,imag) in complex_shadows.items():
                            if mode and not variables[real].scalar:continue
                            if name not in new_complex:
                                initializers.append(f'_eon_complex {name} = {{{real},{imag}}};')
                            lines.extend((f'{real}={name}.real;',f'{imag}={name}.imag;'))
                            writes.extend((real,imag))
                        writes.extend(name for name in renderer.shadows.values()
                            if name not in complex_shadows and (not mode or variables[name].scalar))
                    faults=getattr(renderer,'expression_fault_bits',{})
                    if faults:
                        # Use the existing native error channel, including its
                        # consumed-RNG and phase-prefix replay boundary.
                        flag=renderer.index_error_name
                        bits='257u'
                        for kind,code in reversed(tuple(faults.items())):
                            bits=f'({flag}_kind=={kind}?{code|1}u:{bits})'
                        lines.append(f'_profile_error=({flag}?{bits}:0u);')
                    operations.append(Segment('\n'.join(initializers+lines),tuple(dict.fromkeys(writes)),mode,index_fault is not None or bool(faults),index_fault))
                    lines.clear();writes.clear()
            if (numpy_locals and floating_status and not is_scalar
                    and renderer.prologue[seen_prologue:]):
                lines.append('// NumPy block-entry copies precede scalar actions too.')
                flush()
            for original in statements:
                if original.var in constants:continue
                variable=inventory.get(original.var)
                guard=(variable.conditional_write if stage=='state' and variable else None)
                statement=copy.copy(original)
                statement.var=aliases.get(statement.var,statement.var)
                statement.expr=word_substitute(statement.expr,aliases)
                branches=getattr(statement,'boolean_simplified_expressions',None)
                if branches:
                    statement.boolean_simplified_expressions={
                        tuple((aliases.get(name,name),value) for name,value in conditions):
                            word_substitute(expression,aliases) for conditions,expression in branches.items()}
                used=getattr(statement,'used_boolean_variables',None)
                if (floating_status and used and len(used)==1
                        and np.dtype(statement.dtype).kind=='f'
                        and statement.complexity_std>sum(statement.complexities.values())):
                    choices={conditions[0][1]:expression for conditions,expression in
                             statement.boolean_simplified_expressions.items()}
                    statement.expr=ast.unparse(ast.IfExp(
                        test=ast.Name(aliases.get(used[0],used[0]),ast.Load()),
                        body=ast.parse(choices[True],mode='eval').body,
                        orelse=ast.parse(choices[False],mode='eval').body))
                    statement.boolean_simplified_expressions=None
                    statement.used_boolean_variables=()
                if statement.op==':=':statement.op='='
                renderer.types[statement.var]=statement.dtype
                code_names.setdefault(statement.var,variable_code_name(statement.var))
                # NumPy auxiliary values carry raw64 bits even when Brian's
                # declared temporary dtype follows float32 physical state.
                # Their actual ufunc width stays in renderer.temporaries;
                # narrowing this transport truncates dt/tau before exp().
                local_dtype=(np.dtype('float64') if numpy_locals
                    and np.dtype(statement.dtype).kind=='f'
                    and not(statement.var in inventory and inventory[statement.var].persistent)
                    else np.dtype(statement.dtype))
                variables[statement.var]=Temporary(local_dtype.name,is_scalar,code_names[statement.var])
                types[statement.var]=statement.dtype
                class Hoist(ast.NodeTransformer):
                    def visit_BinOp(self,node):
                        node=self.generic_visit(node)
                        from .numpy_error_policy import OPERATORS
                        if type(node.op) not in OPERATORS:return node
                        result=renderer.expression(node)
                        from .numpy_complex import is_complex
                        if is_complex(result):
                            if result.python_scalar:
                                return self.scalar_complex(node,result) if result.value is None else node
                            if not floating_status:return node
                            return self.complex_floating(node,renderer._complex(result),OPERATORS[type(node.op)])
                        if not floating_status:return node
                        if result.kind!='float' or result.python_scalar:return node
                        return self.floating(node,result,OPERATORS[type(node.op)])

                    def scalar_complex(self,node,result):
                        nonlocal serial
                        if not is_scalar and renderer.prologue[seen_prologue:]:lines.append(';')
                        flush()
                        name=f'_python_complex_result_{serial}';serial+=1
                        bind_temporary(name,'float64',True)
                        temporary=SimpleNamespace(var=name,expr=ast.unparse(node),op='=',
                            dtype=np.dtype('float64'),constant=False)
                        lines.append(renderer.statement(temporary,guard))
                        flush(scalar_mode=True)
                        return ast.copy_location(ast.Name(name,ast.Load()),node)

                    def complex_floating(self,node,result,operation,status_scope=None):
                        nonlocal serial
                        scalar_operation=is_scalar or result.scalar
                        if scalar_operation and not is_scalar and renderer.prologue[seen_prologue:]:
                            lines.append(';')
                        flush()
                        result_name=f'_complex_result_{serial}'
                        status_name=f'_complex_status_{serial}'
                        serial+=1
                        for name,dtype in ((result_name,'float64'),(status_name,'uint32')):
                            bind_temporary(name,dtype,scalar_operation)
                        temporary=SimpleNamespace(var=result_name,expr=ast.unparse(node),op='=',
                            dtype=np.dtype('float64'),constant=False)
                        lines.append(renderer.statement(temporary,guard))
                        value=renderer.expression(result_name)
                        packed=renderer._complex(value).code
                        choose=lambda p,y,n:y if p is True else n if p is False else f'({p}?{y}:{n})'
                        narrow=result.narrow
                        if operation[0] is None:
                            argument=renderer._complex(renderer.expression(node.args[0]))
                            narrow=argument.narrow
                            from .numpy_complex_unary_status import FUNCTIONS
                            code=FUNCTIONS.index(operation[1])
                            status=f'_eon_numpy_complex_unary_status({code}u,{argument.code},{packed},WIDTH)'
                            renderer.used_math.add('_numpy_complex_unary_'+operation[1])
                        else:
                            operands=[renderer.expression(part) for part in (node.left,node.right)]
                            arguments=[renderer._complex(arg).code for arg in operands]
                            if operation[0]==6:
                                status=f'_eon_numpy_complex_power_status({arguments[0]},{arguments[1]},{packed},WIDTH)'
                                renderer.used_math.add('_numpy_complex_power_status')
                            else:
                                array_loop=str(any(not arg.scalar or arg.array_object for arg in operands)).lower()
                                status=(f'_eon_numpy_complex_binary_status({operation[0]}u,{arguments[0]},'
                                        f'{arguments[1]},{packed},WIDTH,{array_loop})')
                                renderer.used_math.add('_numpy_complex_binary_status')
                        status=status.replace('WIDTH',choose(narrow,'32u','64u'))
                        if guard:status=f'({renderer.expression(guard).code}?{status}:0u)'
                        if status_scope is not None:status=f'({status_scope}?{status}:0u)'
                        lines.append(code_names[status_name]+'='+status+';')
                        writes.append(status_name)
                        # Complex locals cross this operation boundary as two
                        # raw words through flush(), never as a real projection.
                        if value.kind!='complex':
                            lines.append(code_names[result_name]+'='+value.code+';')
                            writes.append(result_name)
                        renderer.used_math.add('_numpy_floating_status')
                        flush(scalar_mode=scalar_operation)
                        operations.append(_numpy_status_draw(status_name,operation[1],scalar_operation,guard,renderer))
                        return ast.copy_location(ast.Name(result_name,ast.Load()),node)

                    def floating(self,node,result,operation,status_scope=None,raw_result=False):
                        nonlocal serial
                        from .numpy_complex import is_complex
                        if is_complex(result):
                            return self.complex_floating(node,renderer._complex(result),operation,status_scope)
                        flush()
                        scalar_operation=is_scalar or result.scalar
                        result_name=f'_floating_result_{serial}'
                        status_name=f'_floating_status_{serial}'
                        serial+=1
                        while result_name in types or status_name in types:
                            result_name+='_';status_name+='_'
                        for name,dtype in ((result_name,'float64'),(status_name,'uint32')):
                            bind_temporary(name,dtype,scalar_operation)
                        temporary=SimpleNamespace(var=result_name,expr=result if raw_result else ast.unparse(node),op='=',
                            dtype=np.dtype('float64'),constant=False)
                        lines.append(renderer.statement(temporary,guard))
                        value=renderer.convert(renderer.expression(result_name),'float').code
                        lines.append(code_names[result_name]+'='+value+';')
                        narrow=result.narrow;half=getattr(result,'half',False)
                        choose=lambda predicate,yes,no:yes if predicate is True else no if predicate is False else f'({predicate}?{yes}:{no})'
                        width=choose(half,'16u',choose(narrow,'32u','64u'))
                        if operation[0] is None:
                            from .numpy_error_policy import unary_status_value
                            argument=renderer.expression(node.args[0])
                            local_status=f'_unary_status_{serial}'
                            lines.extend((f'uint {local_status}=0u;',
                                f'(void)({unary_status_value(renderer,operation[1],argument,result,local_status)});'))
                            status=local_status
                        else:
                            from .numpy_error_policy import binary_status_code,binary_action_name
                            left,right=renderer.expression(node.left),renderer.expression(node.right)
                            arguments=[renderer._operand(arg,narrow) for arg in (left,right)]
                            status=binary_status_code(renderer,operation,left,right,arguments,value,width)
                            action_name=binary_action_name(operation[1],left,right)
                        if guard:
                            status=f'({renderer.expression(guard).code}?{status}:0u)'
                        if status_scope is not None:
                            status=f'({status_scope}?{status}:0u)'
                        lines.append(code_names[status_name]+'='+status+';')
                        writes.extend((result_name,status_name))
                        renderer.used_math.add('_numpy_floating_status')
                        flush(scalar_mode=scalar_operation)
                        operations.append(_numpy_status_draw(status_name,
                            operation[1] if operation[0] is None else action_name,
                            scalar_operation,guard,renderer))
                        return ast.copy_location(ast.Name(id=result_name,ctx=ast.Load()),node)

                    def integer(self,node,arg,bits=32):
                        nonlocal serial
                        from .numpy32 import Expression
                        from .numpy_error_policy import int32_status_code,int64_status_code,python_int_status
                        flush()
                        scalar_operation=is_scalar or arg.scalar
                        result_name=f'_int_result_{serial}'
                        status_name=f'_int_status_{serial}'
                        serial+=1
                        for name,dtype in ((result_name,f'int{bits}'),(status_name,'uint32')):
                            bind_temporary(name,dtype,scalar_operation)
                        operand=renderer.convert(arg,'float').code
                        expression=Expression((f'd_to_int({operand})' if bits==64 else
                            renderer._numpy_int32(arg,operand)),'int',arg.scalar,integer_bits=bits,
                            python_scalar=bits==64 and arg.scalar)
                        temporary=SimpleNamespace(var=result_name,expr=expression,op='=',
                            dtype=np.dtype(f'int{bits}'),constant=False)
                        lines.append(renderer.statement(temporary,guard))
                        lines.append(code_names[result_name]+'='+renderer.expression(result_name).code+';')
                        status=(python_int_status(operand) if arg.python_scalar else
                                int64_status_code(operand) if bits==64 else
                                int32_status_code(renderer,arg,operand))
                        if guard:status=f'({renderer.expression(guard).code}?{status}:0u)'
                        lines.append(code_names[status_name]+'='+status+';')
                        writes.extend((result_name,status_name));flush(scalar_mode=scalar_operation)
                        operations.append(Draw('numpy_python_int',status_name,None,scalar_operation,guard)
                            if arg.python_scalar else _numpy_status_draw(status_name,'cast',scalar_operation,guard,renderer))
                        return ast.copy_location(ast.Name(result_name,ast.Load()),node)

                    def visit_Call(self,node):
                        nonlocal serial
                        node=self.generic_visit(node)
                        from .numpy_complex import is_complex
                        if (numpy_locals and isinstance(node.func,ast.Name)
                                and node.func.id in ('floor','ceil') and len(node.args)==1
                                and is_complex(renderer.expression(node.args[0]))):
                            flush()
                            name=f'_complex_ufunc_error_{serial}';serial+=1
                            operations.append(_numpy_status_draw(name,node.func.id,True,None,renderer,
                                function='numpy_complex_ufunc_error'))
                            renderer.used_math.add('_complex_cast')
                            return ast.copy_location(ast.Constant(0),node)
                        if (numpy_locals and isinstance(node.func,ast.Name)
                                and node.func.id in ('int','timestep')
                                and any(is_complex(renderer.expression(arg)) for arg in node.args)):
                            source_node=node.args[0]
                            if node.func.id=='timestep':
                                source_node=ast.BinOp(ast.BinOp(node.args[0],ast.Add(),
                                    ast.BinOp(ast.Constant(.001),ast.Mult(),node.args[1])),
                                    ast.Div(),node.args[1])
                                if floating_status:source_node=self.visit(source_node)
                            source=renderer._complex(renderer.expression(source_node))
                            if node.func.id=='int' and source.python_scalar:
                                # Brian's NumPy int() is np.int32(value). Python
                                # complex rejects that conversion before NumPy
                                # can emit ComplexWarning, even on empty resets.
                                flush()
                                name=f'_python_complex_int_error_{serial}';serial+=1
                                operations.append(_numpy_status_draw(name,'i',True,None,renderer,
                                    function='numpy_python_complex_store'))
                                renderer.used_math.add('_complex_cast')
                                return ast.copy_location(ast.Constant(0),node)
                            value=renderer.expression(node)
                            scalar_operation=is_scalar or value.scalar
                            if scalar_operation and not is_scalar and renderer.prologue[seen_prologue:]:
                                lines.append(';')
                            flush()
                            name=f'_complex_cast_result_{serial}';serial+=1
                            code_names[name]=variable_code_name(name)
                            dtype='int32' if node.func.id=='int' else 'int64'
                            variables[name]=Temporary(dtype,scalar_operation,code_names[name])
                            renderer.types[name]=types[name]=np.dtype(dtype)
                            temporary=SimpleNamespace(var=name,expr=ast.unparse(node),op='=',
                                dtype=np.dtype(dtype),constant=False)
                            raw_name=name+'_raw'
                            if floating_status:
                                bind_temporary(raw_name,'float64',scalar_operation)
                                lines.append(code_names[raw_name]+f'=({source.code}).real;')
                                writes.append(raw_name)
                            lines.append(renderer.statement(temporary,guard))
                            lines.append(code_names[name]+'='+renderer.expression(name).code+';')
                            writes.append(name);flush(scalar_mode=scalar_operation)
                            operations.append(_numpy_status_draw(name,node.func.id,scalar_operation,guard,renderer,
                                function='numpy_complex_cast'))
                            if floating_status:
                                status_name=name+'_status'
                                bind_temporary(status_name,'uint32',scalar_operation)
                                raw=code_names[raw_name]
                                if node.func.id=='int':
                                    rank,count,width=renderer._complex_int32_arguments(source)
                                    status=(f'_eon_numpy_int32_cast_status({raw},{rank},'
                                        f'({count}>0L?{count}-1L:0L),{width})')
                                    renderer.used_math.add('_numpy_narrow_cast')
                                else:
                                    status=(f'((d_exp({raw})==2047ul||d_ge({raw},0x43e0000000000000ul)'
                                        f'||d_lt({raw},0xc3e0000000000000ul))?8u:0u)')
                                if guard:status=f'({renderer.expression(guard).code}?{status}:0u)'
                                lines.append(code_names[status_name]+'='+status+';')
                                writes.append(status_name)
                                renderer.used_math.add('_numpy_floating_status')
                                flush(scalar_mode=scalar_operation)
                                operations.append(_numpy_status_draw(status_name,'cast',scalar_operation,guard,renderer))
                            return ast.copy_location(ast.Name(name,ast.Load()),node)
                        if isinstance(node.func,ast.Name) and len(node.args)==1:
                            name=node.func.id
                            if name=='int':
                                argument=renderer.expression(node.args[0])
                                if argument.kind=='float' and (floating_status or argument.python_scalar):
                                    return self.integer(node,argument)
                            if floating_status and name in ('floor','ceil'):
                                from .numpy_error_policy import floor_expression
                                argument=renderer.expression(node.args[0])
                                if argument.kind in ('float','int','bool'):
                                    value=floor_expression(renderer,name,argument)
                                    rounded=self.floating(node,value,(None,name),raw_result=True)
                                    return self.integer(ast.Call(ast.Name('int',ast.Load()),[rounded],[]),
                                                        renderer.expression(rounded))
                        if (floating_status and isinstance(node.func,ast.Name)
                                and node.func.id=='timestep' and len(node.args)==2):
                            offset=ast.BinOp(ast.Constant(.001),ast.Mult(),node.args[1])
                            ratio=ast.BinOp(ast.BinOp(node.args[0],ast.Add(),offset),ast.Div(),node.args[1])
                            rounded=self.visit_BinOp(ratio)
                            return self.integer(node,renderer.expression(rounded),bits=64)
                        if floating_status and isinstance(node.func,ast.Name):
                            from .numpy_math_status import ROOTS
                            if node.func.id=='exprel' and len(node.args)==1:
                                argument=node.args[0]
                                text=ast.unparse(argument)
                                mask=ast.parse(f'not (abs({text}) < 1e-16) and not ({text} > 717)',mode='eval').body
                                scope=renderer.expression(mask).code
                                exponential=ast.Call(ast.Name('expm1',ast.Load()),[argument],[])
                                numerator=self.floating(exponential,renderer.expression(exponential),
                                                        (None,'expm1'),scope)
                                quotient=ast.BinOp(numerator,ast.Div(),argument)
                                ratio=self.floating(quotient,renderer.expression(quotient),
                                                    (3,'divide'),scope)
                                return ast.copy_location(ast.IfExp(
                                    ast.parse(f'abs({text}) < 1e-16',mode='eval').body,
                                    ast.Constant(1.),ast.IfExp(
                                        ast.parse(f'{text} > 717',mode='eval').body,
                                        ast.Constant(float('inf')),ratio)),node)
                            if node.func.id in ROOTS or node.func.id in ('abs','sign'):
                                if len(node.args)!=1:return node
                                result=renderer.expression(node)
                                from .numpy_complex import is_complex
                                if is_complex(renderer.expression(node.args[0])):
                                    return self.complex_floating(node,result,(None,node.func.id))
                                if result.kind=='float':
                                    return self.floating(node,result,(None,node.func.id))
                        timed=renderer.values.get(node.func.id) if isinstance(node.func,ast.Name) else None
                        if target=='numpy' and isinstance(timed,_TimedArray) and len(timed.array.shape)==2:
                            flush()
                            result=f'_rng_timed_{serial}';serial+=1
                            while result in types:result += '_'
                            code_names[result]=variable_code_name(result)
                            temporary=SimpleNamespace(var=result,expr=ast.unparse(node),op='=',dtype=np.dtype('float64'),constant=False)
                            column=renderer.expression(node.args[1]).code
                            variables[result]=Temporary('float64',is_scalar,code_names[result])
                            renderer.types[result]=types[result]=np.dtype('float64')
                            flag=renderer.index_error_name
                            lines.append(renderer.statement(temporary))
                            lines.append(f'_profile_error=({flag}?257u:0u);')
                            writes.append(result)
                            flush((column,timed.array.shape[1]))
                            return ast.copy_location(ast.Name(id=result,ctx=ast.Load()),node)
                        if not isinstance(node.func,ast.Name) or node.func.id not in ('rand','randn','poisson'):
                            return node
                        name=node.func.id
                        if node.keywords or len(node.args)!=(1 if name=='poisson' else 0):
                            raise ValueError(f'Invalid built-in random call {name}')
                        argument=None
                        constant_argument=None
                        argument_scalar=False
                        if name=='poisson':
                            functions={id(part.func)for part in ast.walk(node.args[0])if isinstance(part,ast.Call)}
                            argument_scalar=all(part.id in renderer.values or
                                (part.id in variables and variables[part.id].scalar)
                                for part in ast.walk(node.args[0])if isinstance(part,ast.Name)and id(part)not in functions)
                            try:constant_argument=float(_evaluate(node.args[0],renderer.values))
                            except KeyError:pass
                            argument=f'_rng_argument_{serial}'
                            while argument in types:argument+='_'
                            code_names[argument]=variable_code_name(argument)
                            temporary=SimpleNamespace(var=argument,expr=ast.unparse(node.args[0]),op='=',
                                dtype=np.dtype('float64'),constant=False)
                            code=renderer.statement(temporary,guard)
                            if numpy_locals:
                                # A sampler consumes this value now, before the
                                # deferred NumPy persistent-array writeback.
                                code+='\n'+code_names[argument]+'='+renderer.convert(renderer.expression(argument),'float').code+';'
                            variables[argument]=Temporary('float64',is_scalar or (argument_scalar and target=='numpy'),code_names[argument])
                            types[argument]=np.dtype('float64')
                            if argument_scalar and not is_scalar and target=='numpy':
                                flush()
                                operations.append(Segment(code,(argument,),True))
                            else:lines.append(code);writes.append(argument)
                        flush()
                        result=f'_rng_result_{serial}';serial+=1
                        while result in types:result+='_'
                        dtype=('int64'if target=='numpy'else'int32')if name=='poisson'else'float64'
                        bind_temporary(result,dtype,is_scalar)
                        operations.append(Draw(name,result,argument,is_scalar,guard,constant_argument,argument_scalar))
                        return ast.copy_location(ast.Name(id=result,ctx=ast.Load()),node)
                node=ast.parse(statement.expr,mode='eval').body
                changed=Hoist().visit(node)
                if ast.dump(changed)!=ast.dump(ast.parse(statement.expr,mode='eval').body):
                    statement.expr=ast.unparse(changed)
                    # A stateful call must not be duplicated in optimized
                    # Boolean branches; its draw has already occurred above.
                    statement.boolean_simplified_expressions=None
                    statement.used_boolean_variables=()
                statement_value=renderer.expression(changed) if numpy_locals else None
                scalar_complex=(numpy_locals and statement_value.kind=='complex' and statement_value.scalar
                                and not getattr(statement,'used_boolean_variables',()))
                if scalar_complex:
                    if not is_scalar and renderer.prologue[seen_prologue:]:lines.append(';')
                    flush()
                lines.append(renderer.statement(statement,guard));writes.append(statement.var)
                if scalar_complex:flush(scalar_mode=True)
            if numpy_locals:
                epilogue=renderer.finish()
                casts=getattr(renderer,'complex_cast_stores',{})
                python_casts=getattr(renderer,'python_complex_cast_stores',{})
                for index,line in enumerate(epilogue):
                    if index in python_casts:
                        flush()
                        name,kind=python_casts[index]
                        operations.append(_numpy_status_draw(name,kind,True,None,renderer,
                            function='numpy_python_complex_store'))
                    if index in casts:
                        flush()
                        operations.append(_numpy_status_draw(casts[index],None,is_scalar,None,renderer,
                            function='numpy_complex_cast'))
                    # Scalar packing can fail before assignment; array cast
                    # loops commit their destination before delivering status.
                    after_store=None
                    info=renderer.finish_stores.get(index)
                    if floating_status and info is not None:
                        from .numpy_error_policy import floating_store_status
                        check=floating_store_status(renderer,*info)
                        if check is not None:
                            flush()
                            status_name=f'_store_cast_status_{serial}';serial+=1
                            bind_temporary(status_name,'uint32',check[1])
                            lines.append(code_names[status_name]+'='+check[0]+';')
                            writes.append(status_name);flush(scalar_mode=check[1])
                            action=_numpy_status_draw(status_name,'cast',check[1],None,renderer)
                            if check[1]:operations.append(action)
                            else:after_store=action
                    lines.append(line)
                    writes.extend(name for name in renderer.persistent
                                  if f'{renderer.code_name(name)} = ' in line)
                    if after_store is not None:
                        flush()
                        operations.append(after_store)
            flush()
        stages[stage]=tuple(operations)
        if stage=='threshold':
            last=operations[-1]
            if not isinstance(last,Segment) or '_cond' not in last.writes:
                raise ValueError('Threshold lowering must end with its event condition')
            bookkeeping=f'\nif(_cond) {{ {code_names["lastspike"]}=t; {code_names["not_refractory"]}=false; }}'
            stages[stage]=tuple(operations[:-1])+(
                Segment(last.source+bookkeeping,last.writes+('lastspike','not_refractory'),last.scalar),)
    # Keep stateful calls even when their result is unused. Pure auxiliary
    # assignments such as Eon's undeclared reset ``w = 0`` have no observable
    # output and must not turn into an extra GPU dispatch every timestep.
    protected={name for name,v in inventory.items() if v.persistent}|{'_cond'}
    all_operations=[op for operations in stages.values() for op in operations]
    originals={code:name for name,code in code_names.items()}
    references={id(op):({originals.get(name,name)for name in re.findall(r'\b[A-Za-z_]\w*\b',op.source)} if isinstance(op,Segment)
                       else {op.argument,op.guard,*(storage for _,storage in op.host_views)}) for op in all_operations}
    for op in all_operations:
        if isinstance(op,Draw) and op.function in ('numpy_status','numpy_python_int'):
            # Policy delivery consumes this native flag buffer. Preserve its
            # producer even when that segment has no persistent-state writes.
            references[id(op)].add(op.result)
    dead=set()
    for op in all_operations:
        if isinstance(op,Segment) and not protected.intersection(op.writes):
            others=set().union(*(references[id(other)] for other in all_operations if other is not op))
            if not others.intersection(op.writes):dead.add(id(op))
    stages={stage:tuple(op for op in operations if id(op) not in dead)
            for stage,operations in stages.items()}
    used=protected|set().union(*(references[id(op)] for op in all_operations if id(op) not in dead))
    variables={name:v for name,v in variables.items() if name in used
               or any(isinstance(op,Draw) and op.result==name for op in all_operations)}
    # NumPy locals can be aliases with no materialized original storage.
    # Do not export an unused alias after liveness removes its allocation.
    if numpy_locals:
        from dataclasses import replace
        stages={name:tuple(replace(op,writes=tuple(v for v in op.writes if v in variables))
            if isinstance(op,Segment)else op for op in operations)for name,operations in stages.items()}
    compiled_stages=None;serial_required=()
    if target in ('cpp','cython'):
        from .expression_compiled import build_stages
        compiled_stages,serial_required=build_stages(inventory,constants,stage_statements,target=target,
            renderer_options=renderer_options)
    elif target!='numpy':
        from .compiler import UnsupportedModelError
        raise UnsupportedModelError('Cython RNG buffering is not implemented yet')
    return RandomProgram(MappingProxyType(stages),MappingProxyType(variables),target,compiled_stages,serial_required,
        MappingProxyType({name:(tuple(scalar),tuple(vector))for name,(scalar,vector)in stage_statements.items()}))


def speculative_threshold(program,draw):
    """Reuse checked vector code; faults discard this execution through replay.

    The ordinary fast path may consume the threshold draw before a checked
    call. Its sticky release flag forces restoration of the run's original
    RNG and state and exact operation-ordered phase replay before export.
    """
    from .compiler import _metal_type
    random=program.random_program
    declarations=[]
    for name,v in random.variables.items():
        if name==draw.result or name in program.variables and program.variables[name].persistent:continue
        if name=='_cond':continue
        declarations.append(f'{_metal_type(v.dtype)} {v.code_name or name}=0;')
    source=[]
    flag=program.index_error_name
    declarations.append(f'int {flag}_kind=0;')
    for op in random.stages['threshold']:
        if not isinstance(op,Segment):continue
        part=op.source.replace(f'bool {flag}=false; int {flag}_kind=0;','')
        part=part.replace(f'_profile_error=({flag}?257u:0u);','')
        source.append('{'+part+'}')
    return '\n'.join(declarations+source)


def speculative_rank_eligible(program,*,fused_idle=False):
    """Prove all lanes reach the rank barrier in the speculative kernel."""
    from .expression_fixed import FixedRandomProgram
    random=program.random_program
    return (not fused_idle and isinstance(random,FixedRandomProgram)
        and random.target=='numpy' and random.checked
        and program.index_error_name is not None and program.runtime_error_type=='index'
        and not program.runtime_errors and program.replay_phase is None
        and not program.scalar_stages and not re.search(r'\breturn\b',
            program.state_code+program.threshold_code+program.reset_code+program.scalar_code))
