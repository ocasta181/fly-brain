"""Adapter from actual Brian compiled-integer frontends to canonical Metal.

Brian renders the C++ statements; the selected CPU compiler and unchanged flags
perform its rewrites. LLVM SSA is then translated to bit-preserving native
operations, never executed on the CPU. Auditable sources/IR are private cache
artifacts. Unhandled IR is an explicit preparation error, not a claimed pass.
"""
from __future__ import annotations
import ast
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import subprocess
from .compiled_cache import cache_output
import sysconfig
import numpy as np
from .compiler import UnsupportedModelError, _kind, _literal, _metal_type, _block

_CACHE = Path(__file__).resolve().parents[2]/'executions/compiled-integer-ir-v2'

MSL_HEADER = r'''
#ifndef EON_COMPILED_INTEGER_INCLUDED
#define EON_COMPILED_INTEGER_INCLUDED
inline ulong ci_roundtrip_f64(ulong x) {
    int exponent = int((x >> 52) & 2047UL) - 1023;
    if (exponent >= 52) return ((x & 0x7fffffffffffffffUL) > 0x7ff0000000000000UL) ? x | 0x0008000000000000UL : x;
    if (exponent < 0) return x & 0x8000000000000000UL;
    return x & ~((1UL << (52-exponent))-1UL);
}
inline uint ci_roundtrip_f32(uint x) {
    int exponent = int((x >> 23) & 255U) - 127;
    if (exponent >= 23) return ((x & 0x7fffffffU) > 0x7f800000U) ? x | 0x00400000U : x;
    if (exponent < 0) return x & 0x80000000U;
    return x & ~((1U << (23-exponent))-1U);
}
inline ulong ci_mul_high(ulong a,ulong b) {
    uint al=uint(a),ah=uint(a>>32),bl=uint(b),bh=uint(b>>32);
    uint cross0=ah*bl,cross1=al*bh,low_high=mulhi(al,bl);
    uint sum=cross0+low_high,carry=uint(sum<cross0);
    uint next=sum+cross1;carry+=uint(next<sum);
    ulong high=ulong(ah*bh)|(ulong(mulhi(ah,bh))<<32);
    return high+ulong(mulhi(ah,bl))+ulong(mulhi(al,bh))+ulong(carry);
}
inline long ci_trunc_div(long a,long b) {
    if(b==0L) return 0L;
    if(a==(-9223372036854775807L-1L) && b==-1L) return a;
    return a/b;
}
inline long ci_trunc_rem(long a,long b) {
    if(b==0L) return a;
    if(b==-1L) return 0L;
    return a%b;
}
#endif
'''


def selected(statements,types,persistent,*,target=None):
    """Select phases needing the actual compiled language frontend.

    Refractory's built-in timestep calls stay with the qualified clock path.
    The selected unit is always the entire phase, never individual operators.
    Long float32 sums also need Cython's final target addition tree: a
    C++ rendering of its expressions can select a different balanced tree.
    """
    types=dict(types)
    types.update({statement.var:statement.dtype for statement in statements})
    integral={name for name in persistent if np.dtype(types[name]).kind in 'iu'}
    booleans={name for name in persistent if np.dtype(types[name]).kind=='b' and name!='not_refractory'}
    from .operator_policy import compiled_integer_operand_dtype
    # Final target selection can fuse separate binary64 sine/cosine stores.
    # Coexistence only selects the complete CPU frontend; assembly and same-
    # block SSA argument identity below decide whether any calls actually pair.
    if not any(np.dtype(dtype)==np.dtype('float32') for dtype in types.values()):
        functions={node.func.id for statement in statements
            for node in ast.walk(ast.parse(statement.expr,mode='eval'))
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
        if {'sin','cos'}<=functions:return True
    for statement in statements:
        if statement.var in integral|booleans:return True
        nodes=list(ast.walk(ast.parse(statement.expr,mode='eval')))
        if target=='cython' and np.dtype(statement.dtype)==np.dtype('float32'):
            def terms(node):
                return (terms(node.left)+terms(node.right)
                        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add) else 1)
            if any(isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add)
                   and terms(node)>=4 for node in nodes):
                return True
        floating_operators=any(isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Pow,ast.Mod,ast.FloorDiv))
            and (compiled_integer_operand_dtype(node.left,types) is None or compiled_integer_operand_dtype(node.right,types) is None)
            for node in nodes)
        if not floating_operators and any(isinstance(node,ast.Name) and node.id in integral for node in nodes):return True
        for node in nodes:
            if (target=='cython' and isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                    and node.func.id=='abs' and len(node.args)==1
                    and compiled_integer_operand_dtype(node.args[0],types) is not None):
                # Cython boxes integer abs into a genuine Python integer even
                # when a later floating operation determines the store dtype.
                return True
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='int':
                arg=node.args[0]
                if not ((isinstance(arg,ast.Name) and np.dtype(types[arg.id]).kind=='b')
                        or isinstance(arg,(ast.Compare,ast.BoolOp))):return True
            if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd,ast.Invert)):
                if compiled_integer_operand_dtype(node.operand,types) is not None:return True
            if isinstance(node,ast.Compare):
                if all(compiled_integer_operand_dtype(arg,types) is not None for arg in (node.left,*node.comparators)):
                    return True
            if isinstance(node,ast.BinOp) and not isinstance(node.op,ast.Div):
                if (compiled_integer_operand_dtype(node.left,types) is not None and
                        compiled_integer_operand_dtype(node.right,types) is not None):return True
    return False


def reset_scalars_are_unobservable(statements,persistent):
    """Prove skipped scalar reset work is only safe local literal assignment.

    This admits no clocks, namespace reads, functions, RNG, persistent stores,
    checked out-of-range literals or compound assignments. Other scalar phases
    retain the whole masked block, including errors when no neuron spikes.
    """
    for statement in statements:
        if not statement.scalar:continue
        if statement.var in persistent or statement.op not in ('=',':='):return False
        try:value=ast.literal_eval(statement.expr)
        except (ValueError,TypeError,SyntaxError):return False
        dtype=np.dtype(statement.dtype)
        if dtype.kind in 'iu':
            if not isinstance(value,(int,bool)) or not np.iinfo(dtype).min<=value<=np.iinfo(dtype).max:return False
        elif dtype.kind=='b':
            if value not in (False,True,0,1):return False
        elif dtype.kind=='f':
            if not isinstance(value,(int,float,bool)):return False
            try:finite=float(value)
            except (OverflowError,TypeError,ValueError):return False
            if not np.isfinite(finite) or abs(finite)>float(np.finfo(dtype).max):return False
        else:return False
    return True


def lower(policy,statements,types,values,persistent,*,guards=None,errors,used_math,
          operator_policies,error_name,selected_name=None,function_registry=None,original_cpp=None,code_names=None,native_precision_limbs=0,read_only=(),cache_declarations=None):
    """Lower one genuine selected-compiler phase; return scoped native code."""
    used_math.add('_compiled_integer')
    if policy.target=='cython':
        from .compiled_integer_frontend import compile_block,NoCode
        prepared=compile_block(statements,types,policy,persistent,guards=guards,selected_name=selected_name,function_registry=function_registry)
        if isinstance(prepared,NoCode):return prepared
        return optimized_cython_block(policy,prepared,errors,statements=statements,values=values,
            used_math=used_math,operator_policies=operator_policies,error_name=error_name,
            guards=guards,selected_name=selected_name,function_registry=function_registry,code_names=code_names,
            native_precision_limbs=native_precision_limbs)
    return optimized_block(policy,statements,types,values,persistent,guards,
        errors=errors,used_math=used_math,operator_policies=operator_policies,error_name=error_name,
        selected_name=selected_name,function_registry=function_registry,original_cpp=original_cpp,code_names=code_names,
        read_only=read_only,cache_declarations=cache_declarations)


def lower_shared(policy,scalar,vector,types,values,persistent,**options):
    """Use the selected frontend for Brian's two execution sections.

    The scalar section executes once per trial. Its declared auxiliaries are
    explicit typed outputs/inputs across the CPU scalar-to-vector boundary,
    then phase snapshots transport those same values on Metal.
    """
    from .compiler import CompiledVariable,ScalarStage
    from .compiled_integer_frontend import compile_block,NoCode
    if policy.target=='cython':
        whole=compile_block(scalar+vector,types,policy,persistent,
            guards=options.get('guards'),selected_name=options.get('selected_name'),
            function_registry=options.get('function_registry'))
        if isinstance(whole,NoCode):return whole
    types=dict(types)
    for statement in scalar+vector:types[statement.var]=statement.dtype
    temporary={s.var for s in scalar if s.var not in persistent}
    values={name:value for name,value in values.items() if name not in temporary}
    names=options.get('code_names') or {}
    carried=[];declarations=[]
    for name in sorted(temporary):
        dtype=np.dtype(types[name]);metal=_metal_type(dtype)
        if policy.target=='cpp_standalone' and dtype.kind=='b':metal='char'
        emitted=names.get(name,name)
        declarations.append(f'{metal} {emitted}=0;')
        carried.append(CompiledVariable(emitted,dtype.name,metal,True,False,True,None,False))
    def section(statements,outputs,*,selected):
        if not statements:return ''
        arguments=dict(options,selected_name=selected)
        result=lower(policy,statements,types,values,outputs,**arguments)
        if isinstance(result,NoCode):
            raise UnsupportedModelError('Selected complete phase compiled but its execution section did not: '+result.diagnostics)
        return result
    scalar_code='\n'.join(declarations+[section(scalar,persistent|temporary,selected=None)])
    # The runtime visits this section only for selected spike rows. Scalar
    # reset work remains outside that mask, as in Brian's actual template.
    vector_code=section(vector,persistent,selected=None)
    return ScalarStage(scalar_code,tuple(carried),vector_code,vector_code)


def _ctype(dtype):
    from brian2.codegen.generators.cpp_generator import c_data_type
    return c_data_type(np.dtype(dtype).type)


def _abi_type(dtype):
    return 'char' if np.dtype(dtype).kind == 'b' else _metal_type(dtype)


def _paired_math_ir(policy,cpp,ir):
    body=re.search(r'define[^\n]*integer_block[^\n]*\{\n(.*?)\n\}',ir,re.S)
    widths=[width for width in (32,64) if body and all('@llvm.'+name+f'.f{width}(' in body[1] for name in ('sin','cos'))]
    if not widths:return ir
    key=hashlib.sha256(repr((policy.identity,str(cpp))).encode()).hexdigest()
    assembly=_CACHE/('paired_'+key+'.s')
    with cache_output(assembly) as pending:
        if pending is not None:
            command=[*policy.compiler,*(policy.python_flags if policy.target=='cython' else ()),
                     *policy.flags,*policy.compiler_options]
            if policy.target=='cython':
                import brian2 as b
                command += ['-I'+sysconfig.get_path('include'),'-I'+np.get_include(),
                            '-I'+str(Path(b.__file__).resolve().parent/'synapses')]
            result=subprocess.run([*command,'-S',str(cpp),'-o',str(pending)],capture_output=True,text=True,timeout=30)
            if result.returncode:raise UnsupportedModelError('CPU paired-function target preparation rejected: '+result.stderr)
    for width in widths:
        suffix='f' if width==32 else ''
        if re.search(r'\bbl\s+___sincos'+suffix+r'_stret\b',assembly.read_text()):
            ir+='\n; eon.target-paired-sincos'+suffix+'\n'
    return ir


def _cpp_function_support(registry):
    """Expose Brian's known invalid-column branch to the selected optimizer."""
    declarations=[]
    for symbol,name in list(registry.timed_intrinsics.items()):
        timed=registry.timed[name]
        if len(timed.array.shape)==1:
            declarations.append(f'extern "C" __attribute__((const)) double {symbol}(double);')
            continue
        opaque=symbol+'_value'
        while opaque in registry.types or opaque in registry.values:opaque+='_'
        registry.timed_intrinsics[opaque]=name
        declarations.append(f'extern "C" __attribute__((const)) double {opaque}(double,int32_t);')
        declarations.append(f'inline double {symbol}(double t,int32_t i) {{ if(i<0 || i>={timed.array.shape[1]}) return NAN; return {opaque}(t,i); }}')
    return '\n'.join(declarations)


def optimized_block(policy, statements, types, values, persistent, guards=None, *,
                    errors=None,used_math=None,operator_policies=None,error_name='_eon_integer_error',selected_name=None,function_registry=None,original_cpp=None,code_names=None,read_only=(),cache_declarations=None):
    from brian2.codegen.generators.cpp_generator import (CPPCodeGenerator,
        _universal_support_code, clip_code, sign_code, timestep_code, abs_code)
    from brian2.core.functions import DEFAULT_FUNCTIONS
    from brian2.parsing.rendering import CPPNodeRenderer
    if 'clang' not in policy.compiler_version.lower():
        raise UnsupportedModelError('Optimized integer IR currently needs the frozen Clang CPU compiler')
    values=dict(values)
    if policy.vector_length is not None:
        values['N']=np.int64(policy.vector_length)
    generator=object.__new__(CPPCodeGenerator)
    generator.c_data_type=lambda dtype:_ctype(dtype)
    generator.auto_vectorise={}
    generator.func_name_replacements={}
    for name,function in DEFAULT_FUNCTIONS.items():
        try:implementation=function.implementations[CPPCodeGenerator]
        except (KeyError, TypeError):continue
        if implementation.name:generator.func_name_replacements[name]=implementation.name
    if function_registry is not None:
        generator.func_name_replacements.update({name:symbol for symbol,name in function_registry.timed_intrinsics.items()})
    types=dict(types)
    if selected_name is not None:types[selected_name]=np.dtype('bool')
    for statement in statements:types[statement.var]=statement.dtype
    used=set()
    for statement in statements:
        expressions=[statement.expr,*((getattr(statement,'boolean_simplified_expressions',None) or {}).values())]
        used|={node.id for expression in expressions for node in ast.walk(ast.parse(expression,mode='eval')) if isinstance(node,ast.Name)}
        if statement.op!=':=':used.add(statement.var)
    if selected_name is not None:used.add(selected_name)
    written={s.var for s in statements if s.var in persistent or s.var=='_cond'}
    local={s.var for s in statements if s.op==':='}
    inputs=sorted((used-local-values.keys())&types.keys())
    outputs=sorted(written)
    declarations=[]
    for name in sorted((used&values.keys()&types.keys())-local):
        value=values[name];dtype=np.dtype(types.get(name,np.asarray(value).dtype))
        literal=float(value).hex() if dtype.kind=='f' else _literal(value,_kind(dtype))
        declarations.append(f'const {_ctype(dtype)} {name} = ({_ctype(dtype)})({literal});')
    lines=[];selected_lines=[]
    for s in statements:
        code=generator.translate_statement(s)
        guard=(guards or {}).get(s.var)
        destination=selected_lines if selected_name is not None and not s.scalar else lines
        destination.append(f'if({guard}) {{ {code} }}' if guard else code)
    if selected_lines:
        # Vector locals must remain visible to later statements in the same
        # selected reset, just as in Brian's complete per-neuron block.
        lines.append(f'if({selected_name}) {{\n'+'\n'.join(selected_lines)+'\n}')
    from .compiled_integer_frontend import output_pointer_names
    pointers=output_pointer_names(outputs,types)
    parameters=[f'{_ctype(types[name])} {name}' for name in inputs]
    parameters += [f'{_ctype(types[name])} *__restrict {pointers[name]}' for name in outputs]
    header=('#include <cstdint>\n#include <cmath>\n'+_universal_support_code+
            clip_code+sign_code+timestep_code+abs_code+
            '\ntemplate<class T> inline int int_(T x) {return (int)x;}\n')
    if function_registry is not None:header+='\n'+_cpp_function_support(function_registry)
    source=header+'\nextern "C" void integer_block('+','.join(parameters)+') {\n'
    vector_writes={s.var for s in statements if not s.scalar}
    stores=[(f'if({selected_name}) {{ *{pointers[name]}={name}; }}'
             if selected_name is not None and name in vector_writes else f'*{pointers[name]}={name};') for name in outputs]
    source+='\n'.join(declarations+lines+stores)+'\n}\n'
    identity=json.dumps([policy.identity,source],sort_keys=True)
    key=hashlib.sha256(identity.encode()).hexdigest()
    _CACHE.mkdir(parents=True,exist_ok=True)
    cpp=_CACHE/(key+'.cpp');ll=_CACHE/(key+'.ll')
    with cache_output(ll) as pending:
        if pending is not None:
            cpp.write_text(source)
            result=subprocess.run([*policy.compiler,*policy.flags,*policy.compiler_options,'-fno-discard-value-names',
                                   '-S','-emit-llvm',str(cpp),'-o',str(pending)],capture_output=True,text=True,timeout=30)
            if result.returncode:
                raise UnsupportedModelError(f'Frozen Brian C++ integer block rejected: {result.stderr}')
    ir=_paired_math_ir(policy,cpp,ll.read_text())
    from .compiled_integer_machine import poison_constants
    poison=poison_constants(policy,cpp,ir,inputs,outputs,types,_CACHE,key,original_cpp=original_cpp)
    return _lower_ir(ir,inputs,outputs,types,key,policy=policy,errors=errors,poison=poison,
                     used_math=used_math,operator_policies=operator_policies,error_name=error_name,
                     function_registry=function_registry,code_names=code_names,read_only=read_only,cache_declarations=cache_declarations)


def optimized_cython_block(policy, compiled, errors, *, statements=(),values=None,
                           used_math=None,operator_policies=None,error_name='_eon_integer_error',
                           guards=None,selected_name=None,function_registry=None,code_names=None,native_precision_limbs=0):
    import brian2 as b
    cpp,inputs,outputs,types,key,helpers=compiled
    _CACHE.mkdir(parents=True,exist_ok=True)
    ll=_CACHE/('cython_'+key+'.ll')
    with cache_output(ll) as pending:
        if pending is not None:
            command=[*policy.compiler,*policy.python_flags,*policy.flags,*policy.compiler_options,'-fno-discard-value-names',
                     '-I'+sysconfig.get_path('include'),'-I'+np.get_include(),
                     '-I'+str(Path(b.__file__).resolve().parent/'synapses'),
                     '-S','-emit-llvm',str(cpp),'-o',str(pending)]
            result=subprocess.run(command,capture_output=True,text=True,timeout=30)
            if result.returncode:raise UnsupportedModelError('Frozen Cython C++ compilation rejected: '+result.stderr)
    ir=_paired_math_ir(policy,cpp,ll.read_text())
    body=re.search(r'define[^\n]*integer_block[^\n]*\{\n(.*?)\n\}',ir,re.S)
    if body and any(fn in body.group(1) for fn in ('@PyNumber_Remainder','@PyNumber_Absolute','@PyLong_From')) and not helpers:
        from .compiled_integer_frontend import compile_block,NoCode
        from .compiled_python_integer import rewrites as typed_python_rewrites
        typed_rewrites=typed_python_rewrites(compiled,statements,types,policy)
        alternate=compile_block(statements,types,policy,set(outputs),native_python=typed_rewrites or True,
            guards=guards,selected_name=selected_name,function_registry=function_registry)
        if isinstance(alternate,NoCode):
            raise UnsupportedModelError('Native Python-integer rewrite rejected: '+alternate.diagnostics)
        if alternate[-1]:return optimized_cython_block(policy,alternate,errors,statements=statements,
            values=values,used_math=used_math,operator_policies=operator_policies,error_name=error_name,
            guards=guards,selected_name=selected_name,function_registry=function_registry,code_names=code_names,
            native_precision_limbs=native_precision_limbs)
    # Cython's PyLong ((a%b)+b)%b is floor remainder with unbounded
    # intermediates. A typed opaque helper retains the complete block's CPU
    # optimization context; native lowering also preserves conversion errors.
    from .compiled_integer_machine import floating_add_reassociations
    additions=floating_add_reassociations(policy,cpp,ir,_CACHE,'cython_'+key)
    from .compiled_roundtrip import floating_roundtrips
    roundtrips=floating_roundtrips(policy,cpp,ir,_CACHE,'cython_'+key)
    return _lower_ir(ir,inputs,outputs,types,key,errors=errors,policy=policy,helpers=helpers,
                     values=values,used_math=used_math,operator_policies=operator_policies,error_name=error_name,
                     function_registry=function_registry,code_names=code_names,native_precision_limbs=native_precision_limbs,
                     target_additions=additions,target_roundtrips=roundtrips)


def _constant_powi(policy,exponent,argument,kind,operator_policies,used_math):
    """Reuse the canonical scalar machine-code recipe for an LLVM intrinsic."""
    from .operator_policy import _translate,ConstantPowerPolicy
    ctype='float' if kind=='float32' else 'double'
    builtin='__builtin_powif' if kind=='float32' else '__builtin_powi'
    source=f'extern "C" {ctype} eon_power({ctype} x) {{ return {builtin}(x,{int(exponent)}); }}\n'
    flags=policy.python_flags+policy.flags if policy.target=='cython' else policy.flags
    key=hashlib.sha256(repr((source,policy.identity)).encode()).hexdigest()
    cpp=_CACHE/('powi_'+key+'.cpp');assembly=cpp.with_suffix('.s')
    with cache_output(assembly) as pending:
        if pending is not None:
            cpp.write_text(source)
            result=subprocess.run([*policy.compiler,*flags,*policy.compiler_options,'-S',str(cpp),'-o',str(pending)],
                capture_output=True,text=True,timeout=30)
            if result.returncode:raise UnsupportedModelError('CPU powi expansion rejected: '+result.stderr)
    name='_eon_compiled_integer_powi_'+key[:20]
    header,power,powi=_translate(assembly.read_text(),kind,kind,name)
    frozen=ConstantPowerPolicy(name,header,source,assembly.read_text(),policy.compiler,flags,kind,kind,power,powi)
    operator_policies[name]=frozen
    used_math.add('operator_remainder_float')
    if kind=='float32':used_math.add('operator_remainder_float32')
    if power:used_math.add('operator_pow_'+kind)
    if powi:used_math.add('operator_powi_'+kind)
    return name+'('+argument+')'


def _itype(t,signed=False):
    if t=='double':return 'ulong'
    if t=='float':return 'uint'
    if t=='ptr':return 'ulong'
    bits=int(t[1:])
    if bits==1:return 'bool'
    return {8:('char','uchar'),16:('short','ushort'),32:('int','uint'),64:('long','ulong')}[bits][not signed]


def _lower_ir(ir,inputs,outputs,types,key,errors=None,policy=None,helpers=None,*,values=None,
              used_math=None,operator_policies=None,error_name='_eon_integer_error',function_registry=None,poison=None,code_names=None,native_precision_limbs=0,read_only=(),cache_declarations=None,target_additions=None,target_roundtrips=None):
    code_names=code_names or {}
    match=re.search(r'define[^\n]*@[^\s(]*integer_block[^\s(]*\(([^\n]*)\)[^\n]*\{\n(.*?)\n\}',ir,re.S)
    if not match:raise UnsupportedModelError('CPU compiler did not emit the expected integer block')
    body=re.sub(r'(switch [^\n]*\[)(.*?)\]',
                lambda m:m.group(1)+' '+m.group(2).replace('\n',' ')+']',match.group(2),flags=re.S)
    arguments={};known={}
    lines=[]
    formal=re.findall(r'(%[\w.]+)\s*(?:,|$)',match.group(1))
    if len(formal)!=len(inputs)+len(outputs):
        raise UnsupportedModelError(f'Unexpected CPU integer ABI: {match.group(1)}')
    readonly_formals={parameter for name,parameter in zip(inputs,formal) if name in read_only}
    integer_casts=dict(re.findall(r'(%[\w.]+) = (?:sitofp|uitofp) (?:nneg )?i\d+ (%[\w.]+) to double',body))
    integer_absolute=dict(re.findall(r'(%[\w.]+) = (?:tail )?call[^@\n]*@llvm.abs.i\d+\(i\d+ (%[\w.]+),',body))
    floating_absolute=dict(re.findall(r'(%[\w.]+) = (?:tail )?call[^@\n]*@llvm.fabs.f64\(double (%[\w.]+)\)',body))
    def invariant_integer_float(operand):
        operand=floating_absolute.get(operand,operand)
        if not operand.startswith('%'):
            from .clang32 import _literal as llvm_literal
            try:number=float(llvm_literal(operand,'float64').value)
            except (ValueError,OverflowError):return False
            return np.isfinite(number) and number.is_integer() and abs(number)<=2.**64
        source=integer_casts.get(operand)
        return integer_absolute.get(source,source) in readonly_formals
    for name,parameter in zip(inputs,formal):
        alias='_cpu_input_'+name
        source=_literal(values[name],_kind(types[name])) if values is not None and name in values else code_names.get(name,name)
        lines.append(f'const {_abi_type(types[name])} {alias}={source};')
        arguments[parameter]=alias
        if values is not None and name in values:
            known[parameter]=np.dtype(types[name]).type(values[name])
    pointers={parameter:name for name,parameter in zip(outputs,formal[len(inputs):])}
    errors=[] if errors is None else errors
    def register_error(name,message):
        item=(name,message)
        if item not in errors:errors.append(item)
        return errors.index(item)+1
    strings={name:re.sub(r'\\([0-9A-Fa-f]{2})',lambda m:chr(int(m[1],16)),text).rstrip('\0')
             for name,text in re.findall(r'^(@[\w.]+) = .*?c\"(.*?)\"',ir,re.M)}
    exception_sources={}
    string_ids={name:index+1 for index,name in enumerate(strings)}
    body_strings={token for token in re.findall(r'@[\w.]+',body) if token in strings}
    body_exceptions=set(re.findall(r'@PyExc_([\w]+)',body))

    def error_value(name):
        message={'ZeroDivisionError':'integer division or modulo by zero',
                 'OverflowError':'value too large to perform division',
                 'ValueError':'invalid integer operation'}.get(name,'integer operation failed')
        return str(register_error(name,message))+'UL'

    def error_helper(function,arguments_text):
        """Recognize Cython's outlined exception-only handler, never numerics."""
        helper=re.search(r'define[^\n]*@'+re.escape(function)+r'\(([^\n]*)\)[^\n]*\{\n(.*?)\n\}',ir,re.S)
        if not helper:return None
        content=helper.group(2)
        calls=re.findall(r'call[^@\n]*@([^\s(]+)',content)
        if any(fn not in ('PyErr_SetString','PyErr_Occurred') and '__Pyx_AddTraceback' not in fn
               and not fn.startswith('llvm.') for fn in calls):return None
        formal=re.findall(r'(%[\w.]+)\s*(?:,|$)',helper.group(1))
        args=[typed(arg) for arg in arguments_text.split(', ')] if arguments_text else []
        # Exception outlining may store its traceback line through a formal
        # pointer. It must not hide a model/output store in the cold helper.
        for pointer in re.findall(r'\bstore [^\n,]+, ptr ([^, ]+)',content):
            if pointer not in formal:return None
            actual=re.sub(r'^(?:noundef |nonnull )+','',args[formal.index(pointer)][1])
            if actual not in metadata_slots:return None
        if '@PyErr_SetString' not in content:
            return '0UL' if any('__Pyx_AddTraceback' in fn for fn in calls) else None
        set_error=re.search(r'@PyErr_SetString\(ptr (?:noundef )?(%[\w.]+), ptr (?:(?:noundef|nonnull) )*([%@][\w.]+)',content)
        if not set_error:return None
        loaded=re.search(re.escape(set_error.group(1))+r' = load ptr, ptr ([^, ]+)',content)
        if not loaded:return None
        pointer=loaded.group(1)
        message=strings.get(set_error.group(2))
        if pointer.startswith('@PyExc_') and message is not None:
            return str(register_error(pointer[len('@PyExc_'):],message))+'UL'
        if pointer not in formal:return None
        t,operand=args[formal.index(pointer)]
        operand=re.sub(r'^(?:noundef |nonnull )+','',operand)
        exception=value(operand,t)
        message_parameter=set_error.group(2)
        if message_parameter in formal:
            mt,mv=args[formal.index(message_parameter)]
            message=value(re.sub(r'^(?:noundef |nonnull )+','',mv),mt)
            result=exception
            for symbol in sorted(body_exceptions):
                base=error_value(symbol)
                for token in sorted(body_strings):
                    kind=register_error(symbol,strings[token])
                    result=f'(({exception})=={base} && ({message})=={string_ids[token]}UL ? {kind}UL : {result})'
            return result
        return exception
    registers={};decls={};blocks=[('entry',[])];current=blocks[0][1]
    for raw in body.splitlines():
        raw=raw.split(';',1)[0].strip()
        if not raw:continue
        if raw.endswith(':'):
            name=raw[:-1]
            if name=='entry' and not blocks[0][1]:continue
            current=[];blocks.append((name,current));continue
        current.append(raw)
    ids={name:k for k,(name,_) in enumerate(blocks)}
    # Clang outlines Cython's exception-only paths using a stack slot for the
    # traceback line. These values do not participate in model arithmetic.
    # Prove their uses before omitting them; arbitrary stack memory remains
    # unsupported rather than silently losing numeric stores.
    instructions=[raw for _,code in blocks for raw in code]
    metadata_slots={m[1] for raw in instructions if (m:=re.match(
        r'(%__pyx_(?:c?lineno)[\w.]*) = alloca i32,',raw))}
    metadata_values=set(metadata_slots)
    changed=True
    while changed:
        changed=False
        for raw in instructions:
            m=re.match(r'(%[\w.]+) = (?:load i32, ptr |phi i32 )(.*)',raw)
            if m and set(re.findall(r'%[\w.]+',m[2]))&metadata_values and m[1] not in metadata_values:
                metadata_values.add(m[1]);changed=True
    for raw in instructions:
        m=re.match(r'(%[\w.]+) = (.*)',raw)
        uses=set(re.findall(r'%[\w.]+',m[2] if m else raw))&metadata_values
        if not uses:continue
        if raw.startswith('#dbg_') or '@llvm.lifetime.' in raw or '__Pyx_AddTraceback' in raw:continue
        if m and m[1] in metadata_values:continue
        # The normal call lowering still requires a recognized exception-only
        # helper before it can omit this traceback argument.
        if '.cold.' in raw:continue
        if raw.startswith('store i32 ') and uses<=metadata_slots:continue
        raise UnsupportedModelError(f'Cython traceback storage escapes into numeric IR: {raw}; {key}.ll')
    uses=Counter()
    for _,code in blocks:
        for instruction in code:
            if instruction.startswith('#dbg_') or '@llvm.dbg.' in instruction:continue
            uses.update(re.findall(r'%[\w.]+',instruction.split(' = ',1)[-1]))
    products={};powers={};paired_outputs={}
    paired_arguments=set()
    for width,ctype,suffix in ((32,'float','f'),(64,'double','')):
        if '; eon.target-paired-sincos'+suffix+'\n' not in ir:continue
        for label,code in blocks:
            calls={}
            for raw in code:
                call=re.search(ctype+r' @llvm\.(sin|cos)\.f'+str(width)+r'\('+ctype+r' (%[\w.]+)\)',raw)
                if call:calls.setdefault(call[2],set()).add(call[1])
            paired_arguments.update((label,arg,width) for arg,names in calls.items() if names=={'sin','cos'})

    def powi_product(arguments,t):
        from .operator_graph import literal_powi_product_policy
        for index in (0,1):
            if arguments[index] not in powers or uses[arguments[index]]!=1:continue
            at,av,exponent=powers[arguments[index]]
            kinds=('float32' if at=='float' else 'float',*(['float32' if t=='float' else 'float']*2))
            flags=policy.python_flags+policy.flags if policy.target=='cython' else policy.flags
            frozen=literal_powi_product_policy(kinds,'float32' if t=='float' else 'float',exponent,flags,policy.compiler)
            operator_policies[frozen.name]=frozen
            used_math.add('operator_remainder_float')
            if 'float32' in kinds:used_math.add('operator_remainder_float32')
            if frozen.library_power:used_math.add('operator_pow_'+kinds[0])
            if frozen.library_powi:used_math.add('operator_powi_'+kinds[0])
            return frozen.name+'('+','.join((value(av,at),value(arguments[1-index],t),value(arguments[2],t)))+')'
        return None

    def reg(name):
        if name in arguments:return arguments[name]
        if name not in registers:
            registers[name]='_cpu_ir_'+re.sub(r'[^a-zA-Z0-9_]','_',name[1:])
        return registers[name]

    def value(token,t):
        token=token.strip()
        if token.startswith('%'):return reg(token)
        if token in ('poison','undef'):
            raise UnsupportedModelError(f'Frozen CPU integer IR has {token}; artifact {key}.ll needs concrete machine-code analysis')
        if token in ('true','false'):return token
        if token=='null':return '0UL'
        if t=='ptr' and token.startswith('@PyExc_'):return error_value(token[len('@PyExc_'):])
        if t=='ptr' and token in string_ids:return str(string_ids[token])+'UL'
        if t=='ptr' and token.startswith('@__pyx_'):return '0UL'
        if t in ('double','float'):
            if t=='float':
                from .clang32 import _literal as llvm_literal
                return _literal(llvm_literal(token,'float32').value,'float32')
            if token.startswith('0x'):return token+'UL'
            from .binary64 import literal
            return literal(float(token))
        return _literal(int(token),'int')

    def typed(text):
        m=re.match(r'(i\d+|double|float|ptr)\s+(.*)',text.strip())
        if not m:raise UnsupportedModelError(f'Unparsed typed CPU IR operand {text}')
        return m.group(1),m.group(2)

    def cast(token,t,signed=False):
        if t=='i1' and signed:return f'(-long({value(token,t)}))'
        return f'{_itype(t,signed)}({value(token,t)})'

    def known_value(token,t):
        token=token.strip()
        if token.startswith('%'):return known[token]
        if token in ('poison','undef') or t=='ptr':raise KeyError(token)
        if token in ('true','false'):return token=='true'
        if t in ('float','double'):
            from .clang32 import _literal as llvm_literal
            return llvm_literal(token,'float32' if t=='float' else 'float64').value
        return np.dtype('uint'+t[1:]).type(int(token)%(1<<int(t[1:]))) if t!='i1' else bool(int(token))

    def remember_float(target,operation,t,tokens):
        from .clang32 import evaluate
        try:args=[known_value(token,t) for token in tokens]
        except KeyError:return
        with np.errstate(all='ignore'):
            known[target]=evaluate(operation,'float32' if t=='float' else 'float64',args)

    def target_addition(tree):
        if tree[0]=='leaf':return value(tree[2],tree[1])
        prefix='f' if tree[1]=='float' else 'd'
        return prefix+'_add('+target_addition(tree[2])+','+target_addition(tree[3])+')'

    translated=[]
    for label,code in blocks:
        out=[];phis=[]
        from .hyperbolic_pair import IrPairs
        hyperbolic_pairs=IrPairs(code,used_math)
        for raw in code:
            raw=re.sub(r',\s*!\w+\s*!\d+','',raw)
            if raw.startswith('#dbg_') or '@llvm.dbg.' in raw or '@llvm.lifetime.' in raw:continue
            definition=re.match(r'(%[\w.]+) = ',raw)
            if definition and definition[1] in metadata_values:continue
            if raw.startswith('store i32 ') and set(re.findall(r'%[\w.]+',raw))&metadata_slots:continue
            if phis and not re.match(r'%[\w.]+ = phi ',raw):
                out.extend(phis);phis=[]
            side_effect=re.match(r'(?:tail )?call (?:fastcc )?void @([^\s(]+)\((.*)\)',raw)
            if side_effect:
                if '@PyErr_SetString' in raw or '@PyErr_Format' in raw:
                    args=raw.split('(',1)[1].rsplit(')',1)[0]
                    first=args.split(', ',1)[0]
                    _,operand=typed(first)
                    operand=operand.split()[-1]
                    code=value(operand,'ptr')
                    message_match=re.search(r', ptr [^@\n]*(@[\w.]+)',args)
                    if operand in exception_sources and message_match and message_match[1] in strings:
                        code=str(register_error(exception_sources[operand],strings[message_match[1]]))+'UL'
                    out.append(f'if(!_eon_integer_error) _eon_integer_error=uint({code});')
                elif any(fn in raw for fn in ('__Pyx_AddTraceback','__Pyx_WriteUnraisable')):
                    pass  # CPython traceback bookkeeping; error already retained.
                else:
                    error=error_helper(*side_effect.groups())
                    if error is None:raise UnsupportedModelError(f'Unhandled CPU side-effect call {raw}; {key}.ll')
                    out.append(f'if(!_eon_integer_error) _eon_integer_error=uint({error});')
                continue
            if raw.startswith('store '):
                m=re.match(r'store\s+(i\d+|double|float)\s+([^,]+),\s+ptr\s+(%[\w.]+)',raw)
                if not m or m.group(3) not in pointers:
                    raise UnsupportedModelError(f'Unhandled CPU integer store {raw}; {key}.ll')
                name=pointers[m.group(3)]
                out.append(f'{code_names.get(name,name)}=({_abi_type(types[name])})({value(m.group(2),m.group(1))});')
                continue
            if raw=='ret void':out.append('_cpu_pc=-1; break;');continue
            if raw.startswith('switch '):
                m=re.match(r'switch (i\d+) (.*?), label %([\w.]+) \[(.*)\]',raw)
                if not m:raise UnsupportedModelError(f'Unhandled CPU switch {raw}; {key}.ll')
                t,operand,default,cases=m.groups();branch=str(ids[default])
                for case_value,destination in reversed(re.findall(r'i\d+ ([^,]+), label %([\w.]+)',cases)):
                    branch=f'({cast(operand,t)}=={cast(case_value.strip(),t)} ? {ids[destination]} : {branch})'
                out.append(f'_cpu_predecessor={ids[label]}; _cpu_pc={branch}; break;');continue
            if raw.startswith('br '):
                m=re.match(r'br i1 (.*), label %([\w.]+), label %([\w.]+)',raw)
                if m:branch=f'{value(m.group(1),"i1")} ? {ids[m.group(2)]} : {ids[m.group(3)]}'
                else:
                    m=re.match(r'br label %([\w.]+)',raw)
                    if not m:raise UnsupportedModelError(f'Unhandled CPU branch {raw}')
                    branch=str(ids[m.group(1)])
                out.append(f'_cpu_predecessor={ids[label]}; _cpu_pc={branch}; break;');continue
            m=re.match(r'(%[\w.]+) = (.*)',raw)
            if not m:raise UnsupportedModelError(f'Unhandled CPU integer instruction {raw}; {key}.ll')
            target,instruction=m.groups();name=reg(target)
            instruction=re.sub(r'\brange\([^)]*\)\s*','',instruction)
            instruction=re.sub(r'\b(nsw|nuw|nneg|samesign|exact|disjoint|reassoc|nnan|ninf|nsz|arcp|contract|afn|fast)\s+','',instruction)
            op,rest=instruction.split(' ',1);t=None;expr=None
            if op=='load':
                m=re.match(r'(i\d+|double|float|ptr), ptr ([^, ]+)',rest)
                if not m:raise UnsupportedModelError(f'Unhandled CPU load {raw}; {key}.ll')
                t,pointer=m.groups()
                if pointer.startswith('@PyExc_'):
                    error=pointer[len('@PyExc_'):]
                    exception_sources[target]=error
                    expr=error_value(error)
                elif t=='ptr' and pointer.startswith('@__pyx_'):
                    expr='0UL'  # traceback/module bookkeeping only
                else:raise UnsupportedModelError(f'Unhandled CPU numeric load {raw}; {key}.ll')
            elif op=='getelementptr':
                # Inputs are scalar and outputs are only element zero. Other
                # pointer arithmetic is CPython error-message/traceback data.
                t='ptr';expr='0UL'
            elif op in ('add','sub','mul','and','or','xor','shl','ashr','lshr','sdiv','udiv','srem','urem'):
                t,operands=typed(rest);a,b=operands.split(', ')
                try:divisor=int(b)
                except ValueError:divisor=None
                if t=='i64' and op in ('sdiv','srem','udiv','urem') and divisor not in (None,0,1,-1):
                    signed=op.startswith('s')
                    magnitude=abs(divisor) if signed else divisor&((1<<64)-1)
                    source=cast(a,t,signed);prefix=name+'_divide'
                    out.append(f'const ulong {prefix}_n='+
                        (f'({source})<0 ? 0UL-ulong({source}) : ulong({source});' if signed else f'ulong({source});'))
                    if magnitude&(magnitude-1)==0:
                        out.append(f'const ulong {prefix}_q={prefix}_n>>{magnitude.bit_length()-1};')
                        out.append(f'const ulong {prefix}_r={prefix}_n&{magnitude-1}UL;')
                    else:
                        # floor(n*floor(2**64/d)/2**64) underestimates n/d by
                        # at most one. One exact remainder correction suffices.
                        reciprocal=(1<<64)//magnitude
                        out.append(f'ulong {prefix}_q=ci_mul_high({prefix}_n,{reciprocal}UL);')
                        out.append(f'ulong {prefix}_r={prefix}_n-{prefix}_q*{magnitude}UL;')
                        out.append(f'if({prefix}_r>={magnitude}UL) {{ {prefix}_r-={magnitude}UL; ++{prefix}_q; }}')
                    result=prefix+('_r' if op.endswith('rem') else '_q')
                    if signed:
                        negative=f'({source})<0' if op=='srem' or divisor>0 else f'({source})>=0'
                        expr=f'({negative} ? 0UL-{result} : {result})'
                    else:expr=result
                elif op in ('sdiv','srem'):
                    aa,bb=cast(a,t,True),cast(b,t,True)
                    function='ci_trunc_rem' if op=='srem' else 'ci_trunc_div'
                    expr=f'{function}((long)({aa}),(long)({bb}))'
                else:
                    symbol={'add':'+','sub':'-','mul':'*','and':'&','or':'|','xor':'^','shl':'<<','ashr':'>>','lshr':'>>','udiv':'/','urem':'%'}[op]
                    aa=cast(a,t,op=='ashr');bb=cast(b,t)
                    if t in ('i8','i16') and op!='ashr':aa,bb=f'((uint)({aa}))',f'((uint)({bb}))'
                    expr=f'(({aa}) {symbol} ({bb}))'
                    if op in ('udiv','urem'):
                        # Frozen arm64 UDIV returns zero at a zero divisor;
                        # remainder is its MSUB sequence, hence the dividend.
                        expr=f'(({bb})==0 ? {aa if op=="urem" else "0UL"} : {expr})'
            elif op in ('fadd','fsub','fmul','fdiv','fneg'):
                t,operands=typed(rest)
                prefix='f' if t=='float' else 'd'
                args=operands.split(', ');fn=prefix+'_'+op[1:]
                expr=fn+'('+','.join(value(v,t) for v in args)+')'
                contract=bool(re.search(r'\b(?:contract|fast)\b',raw))
                if op=='fmul':products[target]=(args,contract)
                elif op in ('fadd','fsub') and contract:
                    for index,operand in enumerate(args):
                        if operand not in products or uses[operand]!=1 or not products[operand][1]:continue
                        factors=products[operand][0]
                        aa,bb=(value(v,t) for v in factors);cc=value(args[1-index],t)
                        if op=='fsub':
                            if index==0:cc=f'{prefix}_neg({cc})'
                            else:aa=f'{prefix}_neg({aa})'
                        expr=(powi_product((*factors,args[1-index]),t) if op=='fadd' else None) or f'{"f_fma" if t=="float" else "clang_d_fma"}({aa},{bb},{cc})';break
                remember_float(target,op,t,args)
                if op=='fadd' and contract:
                    for index,operand in enumerate(args):
                        if operand in products and uses[operand]==1 and products[operand][1]:
                            remember_float(target,'call:llvm.fma',t,[*products[operand][0],args[1-index]]);break
                if op=='fadd' and target in (target_additions or {}):
                    expr=target_addition(target_additions[target])
                    known.pop(target,None)
            elif op in ('sext','zext','trunc','fptosi','fptoui','sitofp','uitofp','bitcast','fpext','fptrunc'):
                source,dest=rest.rsplit(' to ',1);st,operand=typed(source);t=dest
                floating=value(operand,st)
                if st=='float' and op in ('fptosi','fptoui'):floating=f'f_to_d({floating})'
                if op in ('fpext','fptrunc','sitofp','uitofp'):
                    try:
                        source_value=known_value(operand,st)
                        if op=='sitofp':
                            bits=int(st[1:]);source_value=(int(source_value)+(1<<(bits-1)))%(1<<bits)-(1<<(bits-1))
                        known[target]=np.dtype('float32' if t=='float' else 'float64').type(source_value)
                    except KeyError:pass
                if op=='fpext':expr=f'f_to_d({floating})'
                elif op=='fptrunc':expr=f'f_from_d({floating})'
                elif op=='fptosi':expr=f'{"d_to_int" if t=="i64" else "d_to_int32_sat"}({floating})'
                elif op=='fptoui':
                    # LLVM's arm64 lowering uses signed32 FCVTZS for narrow
                    # unsigned results; valid values fit, then XT[N] narrows.
                    fn='d_to_uint_sat' if t=='i64' else 'd_to_uint32_sat' if t=='i32' else 'd_to_int32_sat'
                    expr=f'{fn}({floating})'
                elif op in ('sitofp','uitofp'):
                    fn=('f_from_int' if t=='float' else 'd_from_int') if op=='sitofp' else (
                        'f_operator_from_uint' if t=='float' else 'd_from_uint')
                    if fn=='f_operator_from_uint':used_math.add('operator_uint_float32')
                    expr=f'{fn}({cast(operand,st,op=="sitofp")})'
                    if target in (target_roundtrips or {}):
                        rt,rv=target_roundtrips[target]
                        expr=f'ci_roundtrip_f{32 if rt == "float" else 64}({value(rv,rt)})'
                        known.pop(target,None)
                elif op=='bitcast':expr=value(operand,st)
                elif op=='trunc' and t=='i1':expr=f'(ulong({value(operand,st)})&1UL)'
                else:expr=cast(operand,st,op=='sext')
            elif op in ('icmp','fcmp'):
                predicate,rest=rest.split(' ',1);st,operands=typed(rest);a,b=operands.split(', ');t='i1'
                symbol={'eq':'==','ne':'!=','slt':'<','sle':'<=','sgt':'>','sge':'>=','ult':'<','ule':'<=','ugt':'>','uge':'>='}
                if op=='icmp':expr=f'({cast(a,st,predicate.startswith("s"))} {symbol[predicate]} {cast(b,st,predicate.startswith("s"))})'
                else:
                    aa,bb=value(a,st),value(b,st)
                    p='f' if st=='float' else 'd'
                    nan_fn='_eon_f_is_nan' if st=='float' else 'd_is_nan'
                    fn={'oeq':'d_eq','one':'d_eq','olt':'d_lt','ole':'d_le','ogt':'d_gt','oge':'d_ge','une':'d_eq'}.get(predicate)
                    if fn:
                        expr=f'{fn.replace("d_",p+"_")}({aa},{bb})'
                        if predicate in ('one','une'):expr=f'(!{expr})'
                        if predicate=='one':expr=f'(!{nan_fn}({aa}) && !{nan_fn}({bb}) && {expr})'
                    elif predicate in ('ueq','ult','ule','ugt','uge'):
                        fn={'ueq':'d_eq','ult':'d_lt','ule':'d_le','ugt':'d_gt','uge':'d_ge'}[predicate]
                        expr=f'({nan_fn}({aa}) || {nan_fn}({bb}) || {fn.replace("d_",p+"_")}({aa},{bb}))'
                    elif predicate=='false':expr='false'
                    elif predicate=='true':expr='true'
                    elif predicate=='ord':expr=f'(!{nan_fn}({aa}) && !{nan_fn}({bb}))'
                    elif predicate=='uno':expr=f'({nan_fn}({aa}) || {nan_fn}({bb}))'
                    else:raise UnsupportedModelError(f'Unhandled CPU float predicate {predicate}')
            elif op=='select':
                condition,a,b=rest.split(', ',2);ct,cv=typed(condition);t,av=typed(a);bt,bv=typed(b)
                expr=f'({value(cv,ct)} ? {value(av,t)} : {value(bv,bt)})'
            elif op=='phi':
                t,rest=typed(rest);pairs=re.findall(r'\[\s*(.*?),\s*%([\w.]+)\s*\]',rest)
                if not pairs:raise UnsupportedModelError(f'Unhandled CPU phi {raw}')
                def incoming(operand,predecessor):
                    if operand in ('poison','undef') and poison is not None and (target,predecessor) in poison:
                        return f'0x{poison[(target,predecessor)]:x}UL'
                    return value(operand,t)
                expr=incoming(*pairs[-1])
                for operand,predecessor in reversed(pairs[:-1]):
                    expr=f'(_cpu_predecessor=={ids[predecessor]} ? {incoming(operand,predecessor)} : {expr})'
                # LLVM phi nodes read the predecessor's values simultaneously.
                temporary=name+'_incoming'
                decls[temporary]=_itype(t)
                phis.append(f'{name}={temporary};')
                decls[name]=_itype(t)
                out.append(f'{temporary}={_itype(t)}({expr});')
                continue
            elif op in ('tail','call'):
                call=rest if op=='tail' else instruction
                m=re.search(r'call\s+(?:(?:fastcc|noundef|signext|zeroext)\s+)*(i\d+|double|float|ptr)\s+@([\w.]+)\((.*)\)',call)
                if not m:raise UnsupportedModelError(f'Unhandled CPU call {raw}; {key}.ll')
                t,function,argtext=m.groups();args=[typed(s) for s in argtext.split(', ')] if argtext.strip() else [];args=[(at,re.sub(r'^(?:(?:noundef|immarg|signext|zeroext|nonnull)\s+)+','',av)) for at,av in args]
                outlined_error=error_helper(function,argtext) if '.cold.' in function else None
                if outlined_error is not None and t=='i1':
                    # These checked Cython helpers set an exception before
                    # querying PyErr_Occurred; its branch result is true.
                    out.append(f'if(!_eon_integer_error) _eon_integer_error=uint({outlined_error});')
                    expr='true'
                elif function_registry is not None and function in function_registry.timed_intrinsics:
                    expr=function_registry.intrinsic(function,'float64',[value(av,at) for at,av in args])
                elif function in (helpers or {}):
                    operation,divisor,input_dtype,output_dtype,*extra=helpers[function]
                    if operation=='python_expr':
                        from .compiled_python_integer import render as render_python_integer
                        parameter_types=[input_dtype,*extra]
                        operands=[cast(av,at,np.dtype(dtype).kind=='i') for (at,av),dtype in zip(args,parameter_types)]
                        def native_operand(node,arguments_,prefix):
                            from brian2.codegen.statements import Statement
                            from .compiled_integer_frontend import compile_block,NoCode
                            output=prefix+'_result';failure=prefix+'_failure'
                            dtype=np.dtype(node[2]);types_={output:dtype}
                            names={output:output};captures=[]
                            for number,(local,index,dt) in enumerate(zip(node[3::3],node[4::3],node[5::3])):
                                captured=prefix+'_input_'+str(number)
                                types_[local]=np.dtype(dt);names[local]=captured
                                captures.append(f'const {_abi_type(dt)} {captured}={arguments_[index]};')
                            statement=Statement(output,'=',node[1],'',dtype=dtype.type)
                            prepared=compile_block([statement],types_,policy,{output},
                                function_registry=function_registry,native_expressions={0:node[1]})
                            if isinstance(prepared,NoCode):
                                raise UnsupportedModelError('Deferred Cython operand was rejected: '+prepared.diagnostics)
                            source_=optimized_cython_block(policy,prepared,errors,statements=[statement],values={},
                                used_math=used_math,operator_policies=operator_policies,
                                error_name=failure,function_registry=function_registry,code_names=names,native_precision_limbs=native_precision_limbs)
                            code=[*captures,f'{_abi_type(dtype)} {output}=0;',
                                  f'bool {failure}=false; int {failure}_kind=0;',
                                  'if(!_eon_integer_error) {',source_,
                                  f'if({failure}) _eon_integer_error=uint({failure}_kind);','}']
                            return code,output
                        source,result=render_python_integer(divisor,operands,output_dtype,policy,name,register_error,
                            used_math=used_math,_native_operand=native_operand,_native_precision_limbs=native_precision_limbs)
                        used_math.add('_compiled_python_integer')
                        decls[name]=_itype(t);out.extend(source);out.append(f'{name}={_itype(t)}({result});');continue
                    source=np.dtype(input_dtype);destination=np.dtype(output_dtype)
                    if source==np.dtype('uint8'):source=np.dtype('uint8' if policy.unsigned_char else 'int8')
                    if destination==np.dtype('uint8'):destination=np.dtype('uint8' if policy.unsigned_char else 'int8')
                    at,av=args[0];a=cast(av,at,source.kind=='i')
                    if operation.startswith('compare_'):
                        import operator
                        compare={'Lt':operator.lt,'LtE':operator.le,'Gt':operator.gt,'GtE':operator.ge,'Eq':operator.eq,'NotEq':operator.ne}[operation[8:]]
                        bounds=np.iinfo(source)
                        if divisor<bounds.min or divisor>bounds.max:
                            expr=str(int(compare(int(bounds.min),divisor)))
                        else:
                            symbol={'Lt':'<','LtE':'<=','Gt':'>','GtE':'>=','Eq':'==','NotEq':'!='}[operation[8:]]
                            expr=f'({a} {symbol} {_literal(divisor,"uint" if source.kind=="u" else "int")})'
                        decls[name]=_itype(t);out.append(f'{name}={_itype(t)}({expr});')
                        continue
                    temporary=name+'_pyint'
                    bounds=np.iinfo(destination);conditions=[];negative_result='false'
                    if operation=='add2':
                        second=np.dtype(extra[0])
                        if second==np.dtype('uint8'):second=np.dtype('uint8' if policy.unsigned_char else 'int8')
                        bt,bv=args[1];b=cast(bv,bt,second.kind=='i')
                        for suffix,operand,dtype in (('a',a,source),('b',b,second)):
                            negative=f'(({operand})<0)' if dtype.kind=='i' else 'false'
                            out.append(f'const bool {temporary}_{suffix}_negative={negative};')
                            out.append(f'const ulong {temporary}_{suffix}={temporary}_{suffix}_negative ? 0UL-ulong({operand}) : ulong({operand});')
                        aa,bb=temporary+'_a',temporary+'_b'
                        out.append(f'const bool {temporary}_same={aa}_negative=={bb}_negative;')
                        out.append(f'const ulong {temporary}_sum={aa}+{bb};')
                        out.append(f'const ulong {temporary}_magnitude={temporary}_same ? {temporary}_sum : ({aa}>={bb} ? {aa}-{bb} : {bb}-{aa});')
                        out.append(f'const bool {temporary}_negative={temporary}_same ? {aa}_negative : ({aa}>={bb} ? {aa}_negative : {bb}_negative);')
                        conditions=[f'({temporary}_same && {temporary}_sum<{aa})']
                        negative_result=f'({temporary}_negative && {temporary}_magnitude!=0UL)'
                        if destination.kind=='i':
                            conditions.append(f'({temporary}_negative ? {temporary}_magnitude>{-int(bounds.min)}UL : {temporary}_magnitude>{int(bounds.max)}UL)')
                        else:
                            conditions.append(f'({temporary}_negative && {temporary}_magnitude!=0UL)')
                            if bounds.max<(1<<64)-1:conditions.append(f'{temporary}_magnitude>{int(bounds.max)}UL')
                        out.append(f'const ulong {temporary}={temporary}_negative ? 0UL-{temporary}_magnitude : {temporary}_magnitude;')
                    elif operation in ('add','sub','mul'):
                        if operation=='mul':
                            if divisor>0:low=-((-int(bounds.min))//divisor);high=int(bounds.max)//divisor
                            else:low=-((-int(bounds.max))//divisor);high=int(bounds.min)//divisor
                            symbol='*'
                        else:
                            offset=divisor if operation=='add' else -divisor
                            low=int(bounds.min)-offset;high=int(bounds.max)-offset
                            symbol='+' if operation=='add' else '-'
                        if destination.kind=='u':
                            if operation=='mul':
                                negative_result=f'(({a})!=0 && '+(f'({a})<0' if divisor>0 else f'({a})>0')+')' if source.kind=='i' else ('false' if divisor>0 else f'({a})!=0')
                            else:
                                zero_crossing=-offset
                                limits0=np.iinfo(source)
                                negative_result=('true' if zero_crossing>limits0.max else 'false' if zero_crossing<=limits0.min else
                                    f'({a})<{_literal(zero_crossing,"uint" if source.kind=="u" else "int")}')
                        limits=np.iinfo(source)
                        if low>limits.max or high<limits.min:conditions=['true']
                        else:
                            if low>limits.min:conditions.append(f'({a})<{_literal(low,"int")}')
                            if high<limits.max:conditions.append(f'({a})>{_literal(high,"int")}')
                        residue=divisor% (1<<64)
                        out.append(f'const ulong {temporary}=ulong({a}) {symbol} {residue}UL;')
                    elif operation=='abs_add':
                        magnitude=f'(ulong({a}))' if source.kind=='u' else f'(({a})<0 ? 0UL-ulong({a}) : ulong({a}))'
                        out.append(f'const ulong {temporary}_magnitude={magnitude};')
                        if destination.kind=='u' and divisor<0:
                            negative_result='true' if -divisor>(1<<64)-1 else f'{temporary}_magnitude<{-divisor}UL'
                        low=int(bounds.min)-divisor;high=int(bounds.max)-divisor
                        if high<0 or low>(1<<64)-1:conditions=['true']
                        else:
                            if low>0:conditions.append(f'{temporary}_magnitude<{low}UL')
                            if high<(1<<64)-1:conditions.append(f'{temporary}_magnitude>{high}UL')
                        out.append(f'const ulong {temporary}={temporary}_magnitude+{divisor%(1<<64)}UL;')
                    elif operation=='abs':
                        magnitude=f'(ulong({a}))' if source.kind=='u' else f'(({a})<0 ? 0UL-ulong({a}) : ulong({a}))'
                        out.append(f'const ulong {temporary}={magnitude};')
                        if bounds.max<(1<<64)-1:conditions.append(f'{temporary}>{bounds.max}UL')
                    else:
                        assert operation=='mod' and divisor
                        if abs(divisor)<=(1<<64):
                            negative=f'(({a})<0)' if source.kind=='i' else 'false'
                            out.append(f'const bool {temporary}_source_negative={negative};')
                            out.append(f'const ulong {temporary}_source_magnitude={temporary}_source_negative ? 0UL-ulong({a}) : ulong({a});')
                            remainder=(temporary+'_source_magnitude' if abs(divisor)==(1<<64) else
                                       f'({temporary}_source_magnitude%{abs(divisor)}UL)')
                            out.append(f'const ulong {temporary}_remainder={remainder};')
                            same=f'({temporary}_source_negative=={str(divisor<0).lower()})'
                            modulus=abs(divisor)%(1<<64)
                            out.append(f'const ulong {temporary}_magnitude=({same} || !{temporary}_remainder) ? {temporary}_remainder : {modulus}UL-{temporary}_remainder;')
                            is_negative=divisor<0
                            if is_negative:negative_result=f'{temporary}_magnitude!=0UL'
                            if destination.kind=='i':
                                limit=-int(bounds.min) if is_negative else int(bounds.max)
                                if limit<(1<<64)-1:conditions.append(f'{temporary}_magnitude>{limit}UL')
                            else:
                                if is_negative:conditions.append(f'{temporary}_magnitude!=0UL')
                                if bounds.max<(1<<64)-1:conditions.append(f'{temporary}_magnitude>{int(bounds.max)}UL')
                            out.append(f'const ulong {temporary}='+('0UL-' if is_negative else '')+f'{temporary}_magnitude;')
                        else:
                            # For a wider divisor, Python's exact result is
                            # either x or x+D. Prove its destination interval
                            # using unbounded preparation integers; execution
                            # needs only input comparisons and raw low bits.
                            limits=np.iinfo(source)
                            def outside(low,high):
                                low=max(low,int(limits.min));high=min(high,int(limits.max))
                                if low>high:return 'true'
                                clauses=[];kind='uint' if source.kind=='u' else 'int'
                                if low>limits.min:clauses.append(f'({a})<{_literal(low,kind)}')
                                if high<limits.max:clauses.append(f'({a})>{_literal(high,kind)}')
                                return '('+' || '.join(clauses)+')' if clauses else 'false'
                            opposite=(f'({a})<0' if source.kind=='i' else 'false') if divisor>0 else f'({a})>0'
                            adjusted=outside(int(bounds.min)-divisor,int(bounds.max)-divisor)
                            plain=outside(int(bounds.min),int(bounds.max))
                            if divisor<0:negative_result=f'({opposite}) || '+(f'({a})<0' if source.kind=='i' else 'false')
                            elif source.kind=='i':negative_result=f'!({opposite}) && ({a})<0'
                            conditions.append(f'(({opposite}) ? {adjusted} : {plain})')
                            out.append(f'const ulong {temporary}=ulong({a})+(({opposite}) ? {divisor%(1<<64)}UL : 0UL);')
                    if conditions:
                        from brian2.codegen.generators.cython_generator import get_cpp_dtype
                        # Cython delegates same-width signed-long conversion
                        # directly to CPython; its diagnostic differs from
                        # the checked narrowing-conversion diagnostic.
                        message=('Python int too large to convert to C long'
                            if destination==np.dtype('int64') else
                            'Python int too large to convert to C unsigned long' if destination==np.dtype('uint64') else
                            'value too large to convert to '+get_cpp_dtype(np.dtype(output_dtype).type))
                        code=register_error('OverflowError',message)
                        if destination.kind=='u':
                            negative_code=register_error('OverflowError',"can't convert negative value to "+get_cpp_dtype(np.dtype(output_dtype).type))
                            code=f'(({negative_result}) ? {negative_code}U : {code}U)'
                        else:code=str(code)+'U'
                        expr=f'({" || ".join(conditions)} ? (_eon_integer_error=_eon_integer_error ? _eon_integer_error : {code},0UL) : ulong({temporary}))'
                    else:expr=temporary
                elif function=='PyErr_Occurred':
                    # Brian's Python caller enters with no pending exception;
                    # numeric error producers above retain their native code.
                    expr='ulong(_eon_integer_error)'
                elif function.startswith(('llvm.smax','llvm.smin','llvm.umax','llvm.umin')):
                    signed=function.startswith(('llvm.smax','llvm.smin'));fn='max' if 'max' in function else 'min'
                    expr=fn+'('+','.join(cast(av,at,signed) for at,av in args[:2])+')'
                elif function.startswith('llvm.abs'):
                    at,av=args[0];a=cast(av,at,True);expr=f'(({a})<0 ? (0UL-(ulong)({a})) : (ulong)({a}))'
                elif function in ('llvm.sin.f32','llvm.cos.f32','llvm.sin.f64','llvm.cos.f64') and (label,args[0][1],32 if t=='float' else 64) in paired_arguments:
                    width=32 if t=='float' else 64
                    pair_key=(label,args[0][1],width)
                    if pair_key not in paired_outputs:
                        pair='_cpu_sincos_'+str(len(paired_outputs));paired_outputs[pair_key]=pair
                        out.append(f'const _eon_trig{width}_pair {pair}=_eon_trig{width}_sincos_scalar({value(args[0][1],args[0][0])});')
                    used_math.update({'sin_f32','cos_f32'} if width==32 else {'sin','cos'})
                    expr=paired_outputs[pair_key]+('.sine' if '.sin.' in function else '.cosine')
                elif (paired:=hyperbolic_pairs.expression(function,t,args,value,out)) is not None:
                    expr=paired
                elif function.startswith('llvm.powi.') and policy is not None:
                    at,av=args[0];et,exponent=args[1]
                    try:exponent=int(exponent)
                    except ValueError:raise UnsupportedModelError('Dynamic CPU powi needs compiler-runtime lowering')
                    powers[target]=(at,av,exponent)
                    expr=_constant_powi(policy,exponent,value(av,at),
                        'float32' if at=='float' else 'float',operator_policies,used_math)
                elif function.startswith(('llvm.fma.','llvm.fmuladd.')) and re.search(r'\b(contract|fast)\b',raw) and (product:=powi_product(tuple(av for at,av in args),t)) is not None:
                    expr=product
                elif t in ('float','double'):
                    base=function.removeprefix('llvm.').split('.')[0]
                    if base.endswith('f') and base[:-1] in ('exp','log','expm1','log1p','exp2'):base=base[:-1]
                    if base in ('exp','log','expm1','log1p','exp2'):
                        try:immutable=[known_value(av,at) for at,av in args]
                        except KeyError:immutable=None
                        if immutable is not None:
                            from .clang32 import evaluate
                            result=evaluate('call:'+base,'float32' if t=='float' else 'float64',immutable)
                            known[target]=result;expr=_literal(result,'float32' if t=='float' else 'float')
                        elif t=='double' and base in ('exp','log','expm1','log1p'):
                            used_math.add(base)
                            helper={'exp':'d_exponential','log':'d_logarithm'}.get(base,'_eon_stable_'+base)
                            expr=helper+'('+','.join(value(av,at) for at,av in args)+')'
                        elif t=='float' and base in ('exp','log','expm1','log1p'):
                            # Cython can contract a declared double call to its
                            # float overload inside the selected integer CFG.
                            # Use the same raw-width helper as the typed graph.
                            used_math.add(base+'_f32')
                            expr='_eon_basic32_'+base+'('+','.join(value(av,at) for at,av in args)+')'
                        else:raise UnsupportedModelError('Dynamic '+function+' needs its canonical native math helper')
                        decls[name]=_itype(t);out.append(f'{name}={_itype(t)}({expr});');continue
                    # Reuse the canonical typed graph's qualified math and
                    # operator primitives, rather than another math engine.
                    from .clang32 import Node,render
                    kind={'float':'float32','double':'float64'}
                    operands=tuple(Node('argument',kind.get(at,'int'+at[1:]),value=value(av,at))
                                   for at,av in args)
                    node=Node('call:'+function,kind[t],operands,value=(key,label))
                    decls[name]=_itype(t)
                    rendered=render({name:node},used_math,error_name,operator_policies=operator_policies,
                        reference_flags=policy.flags,operator_compiler=policy.compiler+policy.python_flags,
                        function_registry=function_registry)
                    if (cache_declarations is not None and t=='double' and function in ('pow','llvm.pow.f64')
                            and len(args)==2 and all(invariant_integer_float(av) for at,av in args)):
                        cache='_eon_pow_cache_'+key[:20]+'_'+name
                        cache_declarations.append(f'bool {cache}_ready=false; ulong {cache}=0UL;')
                        out.append(f'if(!{cache}_ready){{'+rendered+f'{cache}={name};{cache}_ready=true;}}else {name}={cache};')
                    else:out.append(rendered)
                    continue
                else:raise UnsupportedModelError(f'CPU integer IR call {function} is not lowered; artifact {key}.ll')
            if t is None or expr is None:
                raise UnsupportedModelError(f'CPU integer IR opcode {op} is not lowered; artifact {key}.ll')
            decls[name]=_itype(t)
            out.append(f'{name}={_itype(t)}({expr});')
        translated.append((ids[label],out))
    def finish(code):
        return _block([f'// frozen CPU optimized IR: {key}','uint _eon_integer_error=0u;',*code,
            f'if(_eon_integer_error && !{error_name}) {{ {error_name}=true; {error_name}_kind=int(_eon_integer_error); }}'])
    # Metal can lose a scalar SSA value kept live across a later wide boxed
    # call. Explicit private storage preserves it; narrow arithmetic phases
    # keep ordinary registers and their existing optimization opportunities.
    wide_private_workspace=any(int(words)>32 for _,code in translated for line in code
        for words in re.findall(r'ci_py<([0-9]+)>',line))
    lines += [f'{"volatile " if wide_private_workspace and dtype=="ulong" else ""}{dtype} {name};'
              for name,dtype in decls.items()]
    if (len(translated)==1 and translated[0][1][-1:] == ['_cpu_pc=-1; break;']
            and not any('_cpu_pc' in line or '_cpu_predecessor' in line for line in translated[0][1][:-1])):
        # A straight-line numeric block needs no CFG interpreter per update.
        return finish([*lines,*translated[0][1][:-1]])
    edges={ids[label]:{ids[target] for raw in code if raw.startswith(('br ','switch '))
                      for target in re.findall(r'label %([\w.]+)',raw)}
           for label,code in blocks}
    indegree={number:0 for number in edges}
    for destinations in edges.values():
        for destination in destinations:indegree[destination]+=1
    ready=sorted(number for number,count in indegree.items() if count==0);order=[]
    while ready:
        number=ready.pop(0);order.append(number)
        for destination in sorted(edges[number]):
            indegree[destination]-=1
            if indegree[destination]==0:ready.append(destination)
    if (len(order)==len(blocks)
            and all(code and code[-1].endswith('break;') for _,code in translated)):
        # Acyclic Cython error/phi paths can execute once in topological order.
        # Keep the same predecessor and branch decisions, while exposing pure
        # numeric work to the Metal compiler's ordinary loop optimizations.
        lines.append('int _cpu_pc=0, _cpu_predecessor=-1;')
        by_number=dict(translated)
        for number in order:
            code=by_number[number]
            lines += [f'if(_cpu_pc=={number}) {{',*code[:-1],code[-1][:-len('break;')],'}']
        return finish(lines)
    # Integer exponentiation loops consume at most64 exponent bits. This
    # bounded CFG interpreter also covers straight-line/select-only blocks.
    lines += ['int _cpu_pc=0, _cpu_predecessor=-1;',f'for(int _cpu_hop=0; _cpu_pc>=0 && _cpu_hop<{65*(len(blocks)+1)}; ++_cpu_hop) {{','switch(_cpu_pc) {']
    for number,code in translated:lines += [f'case {number}: {{',*code,'}']
    lines += ['}', '}']
    lines.insert(0,'uint _eon_integer_error=0u;')
    lines.append(f'if(_eon_integer_error && !{error_name}) {{ {error_name}=true; {error_name}_kind=int(_eon_integer_error); }}')
    return _block([f'// frozen CPU optimized IR: {key}',*lines])
