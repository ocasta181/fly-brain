"""Preparation-only Brian Cython frontend for compiled integer blocks.

The qualified frontend from8073219 is kept separate from the canonical runtime
and math renderer. Its checked Python-integer descriptors are lowered natively;
no generated CPU code object is run.
"""
from __future__ import annotations
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import threading
from dataclasses import dataclass
import numpy as np
from .compiled_cache import cache_lock, atomic_write

_LOCK=threading.RLock()
_CACHE=Path(__file__).resolve().parents[2]/'executions/compiled-integer-frontend-v2'


@dataclass(frozen=True)
class NoCode:
    key:str
    diagnostics:str


def output_pointer_names(outputs,reserved):
    """Private ABI roles cannot redeclare a legal original data symbol."""
    names={};occupied=set(reserved)
    for index,name in enumerate(outputs):
        candidate=f'_eon_integer_output_{index}'
        while candidate in occupied:
            candidate += "_"
        occupied.add(candidate);names[name]=candidate
    return names


def compile_block(statements, types, policy, persistent, *, guards=None, native_python=False,
                  selected_name=None,function_registry=None,native_expressions=None):
    """Render/check a complete numeric block with Brian's actual Cython rules.

    The generated public C function is an audited preparation artifact; no CPU
    simulation is performed. Its optimized IR can preserve Cython-specific
    promotions, helper operations, and the selected compiler's rewrites.
    """
    import brian2 as b
    from brian2.codegen.generators.cython_generator import CythonCodeGenerator,get_cpp_dtype
    from brian2.codegen.runtime.cython_rt import CythonCodeObject
    from brian2.core.functions import DEFAULT_FUNCTIONS
    from brian2.core.variables import Variable
    from Cython.Compiler import Main

    types=dict(types)
    if selected_name is not None:types[selected_name]=np.dtype('bool')
    used=set();functions=set()
    for statement in statements:
        types[statement.var]=statement.dtype
        expressions=[statement.expr,*((getattr(statement,'boolean_simplified_expressions',None) or {}).values())]
        nodes=[node for expression in expressions for node in ast.walk(ast.parse(expression,mode='eval'))]
        used|={n.id for n in nodes if isinstance(n,ast.Name)}
        functions|={n.func.id for n in nodes if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        if statement.op!=':=':used.add(statement.var)
    if selected_name is not None:used.add(selected_name)
    locals_={s.var for s in statements if s.op==':='}
    inputs=sorted((used-locals_-functions)&types.keys())
    outputs=sorted({s.var for s in statements if s.var in persistent or s.var=='_cond'})
    generator=object.__new__(CythonCodeGenerator)
    generator.variables={name:Variable(name=name,dtype=np.dtype(dtype)) for name,dtype in types.items()}
    generator.func_name_replacements={};generator.auto_vectorise={}
    template=Path(b.__file__).resolve().parent/'codegen/runtime/cython_rt/templates/common_group.pyx'
    macros=CythonCodeObject.templater.env.from_string(template.read_text()).module
    source=str(macros.cython_directives())+'\n'+str(macros.imports())+'\n'
    for name in sorted(functions):
        original=name
        if native_expressions is not None and name not in DEFAULT_FUNCTIONS:
            # Typed Cython trees already contain Brian's implementation names
            # (int_, _clip, ...). Keep those names in the native expression,
            # but obtain their declarations from the original registry.
            for public,function in DEFAULT_FUNCTIONS.items():
                try:implementation=function.implementations[CythonCodeGenerator]
                except KeyError:continue
                if implementation.name==name:original=public;break
        if function_registry is not None and name in function_registry.timed:
            symbol=next(symbol for symbol,original in function_registry.timed_intrinsics.items() if original==name)
            dims=len(function_registry.timed[name].array.shape)
            signature='double' if dims==1 else 'double,int32_t'
            prototype=f'extern "C" __attribute__((const)) double {symbol}({signature});'
            source+=f'\ncdef extern from *:\n    """{prototype}"""\n    double {symbol}({signature}) noexcept\n'
            generator.func_name_replacements[name]=symbol
            continue
        if original not in DEFAULT_FUNCTIONS:
            raise NotImplementedError(f'Cython integer block function {name} needs native function lowering')
        implementation=DEFAULT_FUNCTIONS[original].implementations[CythonCodeGenerator]
        if implementation.name:generator.func_name_replacements[name]=implementation.name
        code=implementation.get_code(None)
        if isinstance(code,str):source+=code+'\n'
    helpers={};rewrites={}
    if native_python:
        def declare_helper(descriptor):
            nonlocal source
            helper='_eon_pyint_'+hashlib.sha256(repr(descriptor).encode()).hexdigest()[:16]
            if helper not in helpers:
                helpers[helper]=descriptor
                outtype=get_cpp_dtype(np.dtype(descriptor[3]).type)
                def helper_type(dtype):
                    name=np.dtype(dtype).name
                    return {'int8':'signed char','uint8':'unsigned char','int16':'short','uint16':'unsigned short'}.get(name) or get_cpp_dtype(np.dtype(dtype).type)
                intypes=[helper_type(dtype) for dtype in (descriptor[2],*descriptor[4:])]
                parameters=', '.join(intypes)
                source+=f'\ncdef extern from *:\n    """extern "C" {outtype} {helper}({parameters});"""\n    {outtype} {helper}({parameters}) except *\n'
            return helper

        def integer_operand(node):
            from .operator_policy import compiled_integer_operand_dtype
            if any(isinstance(n,ast.Constant) and (not isinstance(n.value,int) or not -(1<<31)<=n.value<(1<<31)) for n in ast.walk(node)):return None
            if any(isinstance(n,ast.Call) and (not isinstance(n.func,ast.Name) or n.func.id!='int') for n in ast.walk(node)):return None
            try:
                return compiled_integer_operand_dtype(node,types,target='cython',
                                                       unsigned_char=policy.unsigned_char)
            except (KeyError,TypeError,ValueError):return None

        def constant_arithmetic(node,destination):
            if not isinstance(node,ast.BinOp):return None
            operation={ast.Add:'add',ast.Sub:'sub',ast.Mult:'mul'}.get(type(node.op))
            if operation is None:return None
            operand,constant=node.left,node.right
            try:literal=ast.literal_eval(constant)
            except (ValueError,TypeError):
                if operation not in ('add','mul'):return None
                operand,constant=constant,operand
                try:literal=ast.literal_eval(constant)
                except (ValueError,TypeError):return None
            dtype=integer_operand(operand)
            if dtype is None or not isinstance(literal,int) or -(1<<31)<=literal<(1<<31):return None
            helper=declare_helper((operation,literal,dtype.name,destination.name))
            bounds=np.iinfo(dtype);lo,hi=int(bounds.min),int(bounds.max)
            low,high=((lo+literal,hi+literal) if operation=='add' else
                      (lo-literal,hi-literal) if operation=='sub' else sorted((lo*literal,hi*literal)))
            return f'{helper}({ast.unparse(operand)})',low,high

        class PythonComparisons(ast.NodeTransformer):
            def visit_Compare(self,node):
                node=self.generic_visit(node)
                if len(node.ops)!=1:return node
                operand=node.left;constant=node.comparators[0];operation=type(node.ops[0]).__name__
                if not isinstance(operand,ast.Name):
                    operand,constant=constant,operand
                    operation={'Lt':'Gt','LtE':'GtE','Gt':'Lt','GtE':'LtE','Eq':'Eq','NotEq':'NotEq'}.get(operation)
                if not isinstance(operand,ast.Name) or operation is None:return node
                try:literal=ast.literal_eval(constant)
                except (ValueError,TypeError):return node
                dtype=np.dtype(types.get(operand.id,float))
                if dtype.kind not in 'iu' or not isinstance(literal,int) or -(1<<31)<=literal<(1<<31):return node
                helper=declare_helper(('compare_'+operation,literal,dtype.name,'bool'))
                return ast.copy_location(ast.Call(func=ast.Name(id=helper,ctx=ast.Load()),args=[operand],keywords=[]),node)

        for k,statement in enumerate(statements):
            node=ast.parse(statement.expr,mode='eval').body
            arithmetic=node
            if statement.op in ('+=','-=','*='):
                operation={'+=':ast.Add,'-=':ast.Sub,'*=':ast.Mult}[statement.op]
                arithmetic=ast.BinOp(ast.Name(statement.var,ast.Load()),operation(),node)
            destination=np.dtype(types[statement.var])
            if destination.kind in 'iu':
                lowered=constant_arithmetic(arithmetic,destination)
                if lowered is not None:
                    rewritten=copy.copy(statement);rewritten.op='=';rewritten.expr=lowered[0];rewrites[k]=rewritten
                    continue
                if statement.op=='+=':
                    inner=constant_arithmetic(node,np.dtype('int64'))
                    if inner is not None and -(1<<63)<=inner[1] and inner[2]<(1<<63):
                        helper=declare_helper(('add2',None,destination.name,destination.name,'int64'))
                        rewritten=copy.copy(statement);rewritten.op='='
                        rewritten.expr=f'{helper}({statement.var},{inner[0]})';rewrites[k]=rewritten
                        continue
            transformed=PythonComparisons().visit(copy.deepcopy(node))
            if ast.dump(transformed)!=ast.dump(node):
                rewritten=copy.copy(statement);rewritten.expr=ast.unparse(transformed)
                rewrites[k]=rewritten
            if statement.op not in ('=',':='):continue
            operation=None;divisor=None;operand=None
            if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Mod) and isinstance(node.left,ast.Name):
                try:divisor=ast.literal_eval(node.right)
                except (ValueError,TypeError):continue
                if not (isinstance(divisor,int) and abs(divisor)>(1<<31)-1 and divisor):continue
                operation='mod';operand=node.left
            elif (isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='abs'
                  and len(node.args)==1):
                operation='abs';operand=node.args[0]
            elif (isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub))
                  and isinstance(node.left,ast.Call) and isinstance(node.left.func,ast.Name)
                  and node.left.func.id=='abs' and len(node.left.args)==1):
                try:divisor=ast.literal_eval(node.right)
                except (ValueError,TypeError):continue
                if not isinstance(divisor,int):continue
                if isinstance(node.op,ast.Sub):divisor=-divisor
                operation='abs_add';operand=node.left.args[0]
            if operation is None:continue
            names={n.id for n in ast.walk(operand) if isinstance(n,ast.Name)}
            if len(names)!=1:continue
            source_dtype=np.dtype(types[next(iter(names))]);output_dtype=np.dtype(types[statement.var])
            if not isinstance(operand,ast.Name):
                # Preserve Cython's C arithmetic inside abs before boxing.
                if any(not isinstance(n,(ast.BinOp,ast.Name,ast.Constant,ast.Load,ast.Add,ast.Sub,ast.Mult,ast.UnaryOp,ast.USub,ast.UAdd)) for n in ast.walk(operand)):continue
                if any(isinstance(n,ast.Constant) and (not isinstance(n.value,int) or not -(1<<31)<=n.value<(1<<31)) for n in ast.walk(operand)):continue
                if source_dtype.itemsize<4:source_dtype=np.dtype('int32')
            if source_dtype.kind not in 'iu' or output_dtype.kind not in 'iu':continue
            descriptor=(operation,divisor,source_dtype.name,output_dtype.name)
            helper=declare_helper(descriptor)
            rewritten=copy.copy(statement);rewritten.expr=f'{helper}({ast.unparse(operand)})'
            rewrites[k]=rewritten
    if isinstance(native_python,dict):
        for index,(specifications,expression) in native_python.items():
            for number,(descriptor,parameters_) in enumerate(specifications):
                helper=declare_helper(descriptor)
                expression=expression.replace(f'@_eon_python_helper_{number}@',helper+'('+','.join(parameters_)+')')
            rewritten=copy.copy(statements[index]);rewritten.op='='
            rewritten.expr=expression;rewritten.used_boolean_variables=None
            rewritten.boolean_simplified_expressions=None;rewrites[index]=rewritten
    pointers=output_pointer_names(outputs,types)
    parameters=[f'{get_cpp_dtype(np.dtype(types[name]).type)} {name}' for name in inputs]
    parameters += [f'{get_cpp_dtype(np.dtype(types[name]).type)} *{pointers[name]}' for name in outputs]
    source+='\ncdef public void integer_block('+', '.join(parameters)+') except *:\n'
    lines=[f'cdef {get_cpp_dtype(np.dtype(types[name]).type)} {name}' for name in sorted(locals_)]
    for k,statement in enumerate(statements):
        statement=copy.copy(rewrites.get(k,statement))
        # Brian's Boolean detector compares dtype singletons by identity.
        # An immutable snapshot can deepcopy a dtype without changing its
        # numerical meaning; pass its scalar type at this generator boundary.
        statement.dtype=np.dtype(statement.dtype).type
        if native_expressions is not None and k in native_expressions:
            if statement.op!='=':raise ValueError('A translated native operand needs a plain assignment')
            # This expression came from the actual typed Cython tree AFTER
            # Brian rendering. Reapplying Brian's modulo expansion changes
            # signed-width overflow (and therefore even finite values).
            code=statement.var+' = '+native_expressions[k]
        else:code=generator.translate_statement(statement)
        guard=(guards or {}).get(statement.var)
        if selected_name is not None and not statement.scalar:
            guard=f'({selected_name}) and ({guard})' if guard else selected_name
        if guard:code='if '+guard+':\n'+'\n'.join('    '+s for s in code.splitlines())
        lines.extend(code.splitlines())
    vector_writes={statement.var for statement in statements if not statement.scalar}
    for name in outputs:
        store=f'{pointers[name]}[0] = {name}'
        if selected_name is not None and name in vector_writes:store=f'if {selected_name}:\n    {store}'
        lines.extend(store.splitlines())
    if not lines:lines=['pass']
    source+='\n'.join('    '+line for line in lines)+'\n'
    # Type checking is independent of vector lengths and trial count; keeping
    # those in the enclosing model identity still prevents execution aliasing.
    identity=policy.identity
    identity.pop('vector_length',None);identity.pop('trials',None)
    key=hashlib.sha256(json.dumps([identity,source],sort_keys=True).encode()).hexdigest()
    _CACHE.mkdir(parents=True,exist_ok=True)
    status=_CACHE/('block_'+key+'.json');cpp=_CACHE/('block_'+key+'.cpp')
    with _LOCK, cache_lock(status):
        result=json.loads(status.read_text()) if status.exists() else None
        if result is None or (result['accepted'] and not cpp.exists()):
            # A rebuild must invalidate any old success marker before writing.
            status.unlink(missing_ok=True)
            pyx=_CACHE/('block_'+key+'.pyx');pyx.write_text(source)
            messages=io.StringIO()
            with contextlib.redirect_stderr(messages):
                compilation=Main.compile(str(pyx),options=Main.CompilationOptions(
                    Main.default_options,cplus=True,language_level=3,output_file=str(cpp),cache=False))
            result={'accepted':compilation.num_errors==0,'diagnostics':messages.getvalue()}
            atomic_write(status,json.dumps(result,indent=2)+'\n')
    # Brian's extension manager catches Cython CompileError and returns None;
    # CythonCodeObject.run_block silently skips that entire phase. Preserve
    # this observed preparation result, rather than inventing a runtime error.
    if not result['accepted']:return NoCode(key,result['diagnostics'])
    return cpp,inputs,outputs,types,key,helpers
