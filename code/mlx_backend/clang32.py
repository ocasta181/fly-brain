"""Recover compiled float32 arithmetic before lowering it to raw-bit Metal.

Brian's Cython target lets Clang reassociate across its temporary assignments.
Optimizing individual Brian expressions cannot reproduce that graph. Compile
the complete typed block to SSA, then retain its rounding and contraction
points explicitly. Only immutable SSA nodes execute during preparation.
"""

from __future__ import annotations

import ast
import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
from .compiled_cache import cache_output
import sysconfig
import tempfile

import numpy as np


_CPP_TYPES = {"float32": "float", "float64": "double", "bool": "bool",
              **{kind+str(bits):kind+str(bits)+'_t' for kind in ('int','uint') for bits in (8,16,32,64)}}
_LLVM_TYPES = {"float": "float32", "double": "float64", "i1": "bool",
               "i8": "int8", "i16": "int16", "i32": "int32", "i64": "int64"}
_SCALAR_TYPE_RE = r"float|double|i1|i8|i16|i32|i64"
_TYPE_RE = r"(?:<\d+ x (?:"+_SCALAR_TYPE_RE+r")>|"+_SCALAR_TYPE_RE+r")"
_FLAGS = r"(?:(?:reassoc|nnan|ninf|nsz|arcp|contract|afn|fast|nuw|nsw|exact|samesign|disjoint) )*"


@dataclass(eq=False)
class Node:
    op: str
    dtype: str
    args: tuple = ()
    value: object = None
    contract: bool = False


def _unsupported(message):
    from .compiler import UnsupportedModelError
    raise UnsupportedModelError("Compiled float32 lowering: " + message)


def _cpp_source(inventory, scalar, vector, types, stage, target='cython',prepared_constants=None,function_registry=None):
    from brian2.parsing.rendering import CPPNodeRenderer
    from .symbols import variable_code_name
    cpp_types=dict(_CPP_TYPES)
    if target=='cython':
        from brian2.codegen.generators.cython_generator import get_cpp_dtype
        cpp_types['uint8']=get_cpp_dtype(np.dtype('uint8'))

    class Renderer(CPPNodeRenderer):
        def render_Name(self,node):
            return variable_code_name(node.id) if node.id in types else super().render_Name(node)

        def _integer(self, node):
            if isinstance(node, ast.Constant):return isinstance(node.value,(bool,int))
            if isinstance(node, ast.Name):return np.dtype(types[node.id]).kind in 'biu'
            if isinstance(node, ast.UnaryOp):return self._integer(node.operand)
            if isinstance(node, ast.BinOp):return not isinstance(node.op, ast.Div) and self._integer(node.left) and self._integer(node.right)
            return isinstance(node, ast.Call) and isinstance(node.func,ast.Name) and node.func.id in {'int','sign','timestep'}

        def _cython_double(self, node):
            # Cython declares libc.math functions as returning double even
            # when C++ subsequently selects their float overload. PowNode
            # uses that declared type to insert casts on its OTHER operand.
            # Keep this language-level fact separate from Clang's value DAG.
            if isinstance(node,ast.Constant):return isinstance(node.value,float)
            if isinstance(node,ast.Name):return np.dtype(types[node.id])==np.dtype('float64')
            if isinstance(node,ast.UnaryOp):return not isinstance(node.op,ast.Not) and self._cython_double(node.operand)
            if isinstance(node,ast.BinOp):
                return (isinstance(node.op,ast.Div) and self._integer(node.left) and self._integer(node.right)
                        or self._cython_double(node.left) or self._cython_double(node.right))
            if isinstance(node,ast.IfExp):return self._cython_double(node.body) or self._cython_double(node.orelse)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                name=node.func.id
                if name in {'abs','clip'}:return self._cython_double(node.args[0])
                return (name in {'exp','log','expm1','log1p','exprel','log10','sqrt',
                                 'sin','cos','tan','sinh','cosh','tanh','arcsin','arccos','arctan',
                                 'floor','ceil'} or function_registry is not None
                        and name in function_registry.timed_aliases)
            return False

        def render_BinOp(self, node):
            if (target=='cython' and isinstance(node.op,ast.Pow)
                    and not (self._integer(node.left) and self._integer(node.right))):
                a,b=self.render_node(node.left),self.render_node(node.right)
                left,right=self._cython_double(node.left),self._cython_double(node.right)
                if left or right:
                    if not left:a='((double)('+a+'))'
                    if not right:b='((double)('+b+'))'
                return '_eon_cython_pow('+a+','+b+')'
            if target=='cython' and isinstance(node.op,(ast.Mod,ast.FloorDiv)):
                a,b=self.render_node(node.left),self.render_node(node.right)
                flag=',true' if getattr(self,'scalar_statement',False) else ''
                if isinstance(node.op,ast.Mod):
                    return f'_eon_cython_mod_once((_eon_cython_mod_once({a},{b}{flag})+({b})),{b}{flag})'
                return f'_eon_cython_floordiv({a},{b}{flag})'
            if isinstance(node.op,ast.Div):
                a,b=self.render_node(node.left),self.render_node(node.right)
                if self._integer(node.left) and self._integer(node.right):a='(double)('+a+')'
                if target=='cython':
                    flag=',true' if getattr(self,'scalar_statement',False) else ''
                    return f'_eon_cython_divide({a},{b}{flag})'
                return f'(({a}) / ({b}))'
            return super().render_BinOp(node)

        def render_Call(self, node):
            if function_registry is not None:
                code=function_registry.source_call(self,node)
                if code is not None:return code
            if not isinstance(node.func, ast.Name):
                _unsupported("opaque function call")
            functions = {"int": "int_", "sign": "eon_sign", "clip": "eon_clip",
                         "timestep": "eon_timestep", "arctan": "atan", "arcsin": "asin", "arccos": "acos",
                         "exprel": "_exprel"}
            name = functions.get(node.func.id, node.func.id)
            if target=='cython' and name=='eon_clip' and self._cython_double(node.args[0]):
                # Cython resolves its fused helper from the declared input
                # type, before C++ resolves a nested math overload.
                name+='<double>'
            if target=='cython' and name in ('floor','ceil') and len(node.args)==1:
                return name+'((double)('+self.render_node(node.args[0])+'))'
            return name + "(" + ", ".join(self.render_node(arg) for arg in node.args) + ")"

    renderer = Renderer()
    scalar_names = {stmt.var for stmt in scalar}
    auxiliary_names = {stmt.var for stmt in vector if stmt.op==':='} | {'_cond'}
    state_names = {name for name,var in inventory.items() if var.persistent}
    declared = set()
    assigned = {stmt.var for stmt in scalar+vector if stmt.var in inventory and inventory[stmt.var].persistent}
    if any(stmt.var == "_cond" for stmt in scalar+vector):
        assigned.add("_cond")
    arguments, scalar_lines, vector_lines = [], [], []
    if stage=='reset':
        arguments.append('int32_t *indices')
        vector_lines.append('int64_t _neuron=indices[_lane];')
    else:vector_lines.append('int64_t _neuron=_lane;')
    for name, dtype in sorted(types.items()):
        if name.startswith("_lio_") or name in scalar_names and name not in state_names or name in auxiliary_names or name=='i':
            continue
        dtype = np.dtype(dtype).name
        if name in state_names:
            arguments.append(f"{cpp_types[dtype]} *state_{name}")
            scalar_state = inventory[name].scalar
            (scalar_lines if scalar_state else vector_lines).append(
                f"{cpp_types[dtype]} {variable_code_name(name)}=state_{name}[{'0' if scalar_state else '_neuron'}];")
        elif prepared_constants is not None and name in prepared_constants:
            from .operator_policy import _c_literal
            kind='float32' if dtype=='float32' else 'float'
            scalar_lines.append(f'const {cpp_types[dtype]} {variable_code_name(name)}={_c_literal(prepared_constants[name],kind)};')
        else:
            arguments.append(f"{cpp_types[dtype]} in_{name}")
            scalar_lines.append(f"{cpp_types[dtype]} {variable_code_name(name)}=in_{name};")
        declared.add(name)
    if 'N' not in declared:
        arguments.append('int64_t in_N');scalar_lines.append('int64_t N=in_N;')
    vector_lines.append('int32_t i=(int32_t)_neuron;')
    declared.add('i')
    for name in sorted(assigned):
        if name not in state_names:
            dtype = np.dtype(types[name]).name if name in types else "bool"
            arguments.append(f"{cpp_types[dtype]} *out_{name}")
    for stmt in scalar + vector:
        renderer.scalar_statement=stmt.scalar
        name = stmt.var
        prefix = cpp_types[np.dtype(stmt.dtype).name] + " " if name not in declared else ""
        declared.add(name)
        expression = renderer.render_expr(stmt.expr)
        operation = "=" if stmt.op == ":=" else stmt.op
        if stmt.op=='%=':
            if target=='cython':expression=f'_eon_cython_mod_once({variable_code_name(name)},{expression}'+(',true)' if stmt.scalar else ')')
            else:expression=f'_brian_mod({variable_code_name(name)},{expression})'
            operation='='
        # Preserve the qualified state/reset operation graph. Cython's direct
        # threshold expression avoids an artificial loop-carried undef from
        # expanding a multi-Boolean predicate into non-exhaustive C++ branches.
        branches = (getattr(stmt, "boolean_simplified_expressions", None)
                    if target=='cpp' or stage!='threshold' else None)
        code = f"{variable_code_name(name)} {operation} {expression};"
        if branches:
            parts = []
            for index, (conditions, expression) in enumerate(branches.items()):
                condition = " && ".join(variable_code_name(name) if value else "!"+variable_code_name(name) for name, value in conditions)
                parts.append(f"{'if' if index == 0 else 'else if'}({condition}) {{ {variable_code_name(name)} {operation} {renderer.render_expr(expression)}; }}")
            code = " ".join(parts)
            if prefix:
                code = prefix + variable_code_name(name) + "; " + code
        else:
            code = prefix + code
        variable = inventory.get(name)
        guard = variable.conditional_write if variable is not None and stage=='state' else None
        if guard:
            # Callers remove conditional-write metadata for reset blocks.
            code = f"if({variable_code_name(guard)}) {{ {code} }}"
        (scalar_lines if stmt in scalar else vector_lines).append(code)
    if stage=='threshold':vector_lines.append('if(_cond) eon_threshold_spike(_neuron);')
    for name in sorted(assigned):
        shared=name in state_names and inventory[name].scalar
        (scalar_lines if shared else vector_lines).append(
            f"{('state_' if name in state_names else 'out_')+name}[{'0' if shared else '_neuron'}]={variable_code_name(name)};")
    from .operator_graph import CYTHON_POWER_SUPPORT
    from brian2.codegen.generators.cpp_generator import exprel_code
    return """#include <math.h>
#include <stdint.h>
#include <stdlib.h>
template<typename T> inline int32_t int_(T value){return (int32_t)value;}
template<typename T> inline int32_t eon_sign(T value){return (0<value)-(value<0);}
template<typename T> inline T eon_clip(T value,double low,double high){return value<low?(T)low:value>high?(T)high:value;}
extern "C" void eon_conversion_error();
extern "C" void eon_threshold_spike(int64_t neuron);
inline int64_t eon_timestep(double t,double dt){
    int64_t value=(int64_t)((t+1e-3*dt)/dt);
    // Match Cython's exception check after its integer-returning helper.
    if(value==-1) eon_conversion_error();
    return value;
}
template<typename A,typename B> inline auto _brian_pow(A a,B b)->decltype(pow(a,b)){return pow(a,b);}
"""+CYTHON_POWER_SUPPORT+exprel_code+(function_registry.function_support_code if function_registry is not None else '')+"\nextern \"C\" void eon_block(" + ",".join(arguments) + ") {\n" + "\n".join(scalar_lines) + "\nfor(int64_t _lane=0; _lane<N; ++_lane){\n"+"\n".join(vector_lines)+"\n}\n}\n"


def compiler_identity():
    command=shlex.split(os.environ.get('CXX',sysconfig.get_config_var('CXX') or 'c++'))
    prefix=shlex.split(sysconfig.get_config_var('CFLAGS') or '')
    prefix+=shlex.split(os.environ.get('CFLAGS',''))+shlex.split(os.environ.get('CPPFLAGS',''))
    version=_compiler_version(tuple(command))
    return shlex.join(command),shlex.join(prefix),version.strip()


@lru_cache(maxsize=16)
def _compiler_version(command):
    return subprocess.run([*command,'--version'],check=True,capture_output=True,text=True,timeout=10).stdout


@lru_cache(maxsize=64)
def _ir(source, flags, identity):
    import brian2 as b

    compiler,prefix=shlex.split(identity[0]),shlex.split(identity[1])
    cache = b.prefs.codegen.runtime.cython.cache_dir
    base = Path(cache) if cache is not None else Path(tempfile.gettempdir())
    directory = base / "native-float32-v2"
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(repr((source, identity, flags)).encode()).hexdigest()
    source_path, ir_path = directory / (digest+".cpp"), directory / (digest+".ll")
    with cache_output(ir_path) as pending:
        if pending is not None:
            source_path.write_text(source)
            command = [*compiler, *prefix, *flags, "-fno-discard-value-names", "-S", "-emit-llvm",
                       str(source_path), "-o", str(pending)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if result.returncode:
                _unsupported("Clang preparation failed: " + result.stderr[-2000:])
    result = ir_path.read_text()
    # LLVM leaves sibling sine/cosine intrinsics separate until target
    # instruction selection. Inspect the same source/flags' actual assembly:
    # Apple's paired scalar ABIs have different rounding from standalone calls.
    widths=[width for width in (32,64) if all(f"@llvm.{name}.f{width}(" in result for name in ("sin","cos"))]
    if widths:
        assembly_path=directory/(digest+".s")
        with cache_output(assembly_path) as pending:
            if pending is not None:
                if not source_path.exists():source_path.write_text(source)
                command=[*compiler,*prefix,*flags,"-fno-discard-value-names","-S",
                         str(source_path),"-o",str(pending)]
                prepared=subprocess.run(command,capture_output=True,text=True,timeout=30)
                if prepared.returncode:
                    _unsupported("Clang target preparation failed: "+prepared.stderr[-2000:])
        for width in widths:
            suffix="f" if width==32 else ""
            if re.search(r"\bbl\s+___sincos"+suffix+r"_stret\b",assembly_path.read_text()):
                result+="\n; eon.target-paired-sincos"+suffix+"\n"
    return result


def _literal(token, dtype):
    if token in ("true", "false"):
        return Node("constant", "bool", value=token == "true")
    if dtype.startswith("float"):
        value = np.uint64(int(token, 16)).view(np.float64) if token.startswith("0x") else float(token)
        return Node("constant", dtype, value=np.dtype(dtype).type(value))
    return Node("constant", dtype, value=int(token))


def parse(source, inventory=None):
    """Recover one row's typed DAG from the optimized scalar/SIMD phase."""
    definitions, blocks, outputs, pointers = {}, {}, {}, {}
    pointer_bytes = {}
    selected_indices = set()
    operator_errors=[]
    loop_entries, writes = {}, []
    block = "entry"
    blocks[block] = []
    function = re.search(r"define void @eon_block\((.*)\)[^{]*\{", source)
    if function is None:
        _unsupported("missing prepared function")
    for dtype, name in re.findall(r"("+_TYPE_RE+r")\s+[^,%]*?%(in_[-\w.]+)", function[1]):
        definitions["%"+name] = Node("argument", _LLVM_TYPES[dtype], value=name[3:])
    for name in re.findall(r'ptr\s+[^%]*?%((?:state|out)_[-\w.]+)',function[1]):
        pointers['%'+name]=name.split('_',1)[1]
    pointers['%indices']='__selection'
    pointer_bytes.update((name,Node('constant','int64',value=0)) for name in pointers)

    def dtype_of(token):
        scalar=re.sub(r'<\d+ x (.*?)>',r'\1',token)
        return _LLVM_TYPES[scalar]

    def width(token):
        match=re.match(r'<(\d+) x ',token)
        return int(match[1]) if match else 1

    def make_node(op,dtype,args,**kwargs):
        lanes=max((len(arg) if isinstance(arg,tuple) else 1 for arg in args),default=1)
        if lanes>1:
            return tuple(make_node(op,dtype,tuple(arg[i] if isinstance(arg,tuple) else arg for arg in args),**kwargs) for i in range(lanes))
        if any(arg is None for arg in args):_unsupported('undefined SIMD lane used by '+op)
        if op in ('add','sub','mul','or') and len(args)==2:
            a,b=args
            if b.op=='constant' and b.value==0 and op in ('add','sub','or'):return a
            if a.op=='constant' and a.value==0 and op in ('add','or'):return b
            if op=='sub' and a is b:return Node('constant',dtype,value=0)
            if op=='mul' and b.op=='constant' and b.value==1:return a
            if op=='mul' and a.op=='constant' and a.value==1:return b
        return Node(op,dtype,args,**kwargs)

    def operand(token, dtype):
        token = token.strip()
        if token in ('poison','undef'):return None
        if token=='zeroinitializer':return _literal('0',dtype)
        if token.startswith('splat ('):
            return operand(re.sub(r'^splat \(\S+ (.*)\)$',r'\1',token),dtype)
        if token.startswith('<'):
            return tuple(operand(re.sub(r'^\S+\s+','',value),dtype) for value in token[1:-1].split(', '))
        return definitions[token] if token.startswith("%") else _literal(token, dtype)

    body = source[function.end():].split("\n}", 1)[0]
    # Vectorization/interleaving is part of the optimized CPU program. Retain
    # its pointwise lanes and SIMD-prefix/scalar-tail domains, with every
    # original floating operation/rounding edge. Only the integer loop index
    # is mapped back to the native neuron's logical row.
    raw,raw_blocks={},{};label='entry'
    for original in body.splitlines():
        line=original.strip().split(', !',1)[0]
        match=re.match(r'([-\w.]+):',line)
        if match:label=match[1]
        raw_blocks.setdefault(label,[]).append(line)
        match=re.match(r'(%[-\w.]+) = (.*)',line)
        if match:raw[match[1]]=(label,match[2])
    loops={}
    index_node=Node('argument','int64',value='i')
    selected_loop=bool(re.search(r'ptr [^,]*%indices(?:,|$)',function[1]))
    for name,(label,text) in raw.items():
        phi=re.match(r'phi i64 (.*)',text)
        if not phi:continue
        alternatives=re.findall(r'\[ (.*?), %([-\w.]+) \]',phi[1])
        for back,previous in alternatives:
            increment=raw.get(back,('', ''))[1]
            match=re.match(r'add '+_FLAGS+r'i64 '+re.escape(name)+r', (\d+)$',increment)
            if previous!=label or not match:continue
            stride=int(match[1]);initial,entry=next((value,pred) for value,pred in alternatives if pred!=label)
            branch=next((line for line in raw_blocks[label] if line.startswith('br i1 ')),None)
            if not branch:continue
            condition=re.match(r'br i1 ([^,]+)',branch)[1]
            compare=raw.get(condition,('', ''))[1]
            match=re.match(r'icmp '+_FLAGS+r'eq i64 (.*?), (.*)',compare)
            if not match or back not in match.groups():continue
            limit=match[2] if match[1]==back else match[1]
            if stride & (stride-1) and not selected_loop:_unsupported('non-power-of-two SIMD interleave')
            # Spike-list loops may be unrolled by any factor. Their gathers
            # already map each lane to the selected neuron's logical index;
            # store_value proves that mapping before discarding the ordinal.
            base=(index_node if stride==1 or selected_loop else Node('and','int64',(
                index_node,Node('constant','int64',value=-stride))))
            loops[label]=dict(index=name,stride=stride,initial=initial,entry=entry,limit=limit,base=base)
            break

    def aligned(value,base,offset):
        if offset==0:return value
        replacement=make_node('sub','int64',(index_node,Node('constant','int64',value=offset)))
        memo={}
        def visit(item):
            if item is base:return replacement
            if item in memo:return memo[item]
            result=make_node(item.op,item.dtype,tuple(visit(arg) for arg in item.args),value=item.value,contract=item.contract) if item.args else item
            memo[item]=result
            return result
        return visit(value)

    def affine(item,base):
        if item is base or item.op==base.op=='argument' and item.value==base.value:return 1,0
        if item.op=='constant':return 0,int(item.value)
        if item.op in ('sext','zext','trunc'):return affine(item.args[0],base)
        if item.op=='shl' and item.args[1].op=='constant':
            coefficient,offset=affine(item.args[0],base)
            scale=1<<int(item.args[1].value)
            return coefficient*scale,offset*scale
        if item.op in ('add','sub','mul'):
            a,b=(affine(arg,base) for arg in item.args)
            if item.op=='add':return a[0]+b[0],a[1]+b[1]
            if item.op=='sub':return a[0]-b[0],a[1]-b[1]
            if not a[0]:return a[1]*b[0],a[1]*b[1]
            if not b[0]:return b[1]*a[0],b[1]*a[1]
        _unsupported('non-pointwise SIMD address')

    def store_value(name,value,label,position,element_size):
        loop=loops.get(label)
        if loop is None:
            writes.append((name,value,label,None));return
        def selected(item):
            if item in selected_indices:return item
            return next((found for arg in item.args if (found:=selected(arg)) is not None),None)
        selection=selected(position)
        if selection is not None:
            # An unrolled spike-list gather addresses the selected neuron,
            # not the dense loop's SIMD ordinal. Its scalar lane arithmetic
            # already uses the logical neuron index loaded from that list.
            if affine(position,selection)!=(element_size,0):
                _unsupported('non-pointwise selected state store')
            writes.append((name,value,label,None));return
        coefficient,offset=affine(position,loop['base'])
        if coefficient!=element_size or offset%element_size:_unsupported('unaligned SIMD state store')
        offset//=element_size
        value=aligned(value,loop['base'],offset)
        start=operand(loop['initial'],'int64');end=operand(loop['limit'],'int64')
        domain=Node('and','bool',(Node('icmp:sge','bool',(index_node,start)),Node('icmp:slt','bool',(index_node,end))))
        if loop['stride']>1:
            remainder=Node('sub','int64',(index_node,loop['base']))
            domain=Node('and','bool',(domain,Node('icmp:eq','bool',(remainder,Node('constant','int64',value=offset)))))
        writes.append((name,value,label,domain))
    def split_args(text):
        result=[];begin=depth=0
        for i,char in enumerate(text):
            if char in '<(':depth+=1
            elif char in '>)':depth-=1
            elif char==',' and depth==0:result.append(text[begin:i].strip());begin=i+1
        result.append(text[begin:].strip())
        return result

    delayed = []
    for line in body.splitlines():
        line = line.strip().split(", !", 1)[0]
        if not line or line.startswith((";", "#dbg_")):
            continue
        label = re.match(r"([-\w.]+):", line)
        if label:
            block = label[1]
            blocks[block] = []
            continue
        if line.startswith("br "):
            blocks[block].append(line)
            continue
        error=re.match(r'(?:tail )?call void @eon_operator_(scalar_)?error\(i32(?: noundef)? (.*?)\)',line)
        if error:
            operator_errors.append((bool(error[1]),operand(error[2],'int32'),block))
            continue
        if line.startswith(('tail call void @eon_conversion_error','call void @eon_conversion_error',
                            'tail call void @eon_threshold_spike','call void @eon_threshold_spike')):
            continue
        memset=re.match(r'(?:tail )?call void @llvm.memset[\w.]*\(ptr [^%]*(%[-\w.]+), i8 (-?\d+), i64 ([^,]+),',line)
        if memset:
            name=pointers[memset[1]]
            if name=='_cond':
                value=Node('constant','bool',value=int(memset[2])!=0)
            else:
                variable=(inventory or {}).get(name)
                if variable is None or variable.scalar:_unsupported('unexpected bulk state write: '+line)
                dtype=np.dtype(variable.dtype)
                offset=pointer_bytes[memset[1]]
                count=affine(operand(memset[3],'int64'),Node('argument','int64',value='N'))
                if offset.op!='constant' or offset.value!=0 or count!=(dtype.itemsize,0):
                    _unsupported('noncontiguous bulk state write: '+line)
                raw=bytes([int(memset[2])&255])*dtype.itemsize
                value=Node('constant',dtype.name,value=np.frombuffer(raw,dtype=dtype)[0])
            writes.append((name,value,block,None))
            continue
        store = re.match(r"store ("+_TYPE_RE+r") (.*?), ptr (%[-\w.]+),", line)
        if store:
            dtype=dtype_of(store[1]);value=operand(store[2],dtype)
            size=np.dtype(dtype).itemsize
            lanes=value if isinstance(value,tuple) else (value,)*width(store[1])
            for lane,item in enumerate(lanes):
                position=make_node('add','int64',(pointer_bytes[store[3]],Node('constant','int64',value=lane*size)))
                store_value(pointers[store[3]],item,block,position,size)
            continue
        if line.startswith("ret ") or line=='unreachable':
            continue
        match = re.match(r"(%[-\w.]+) = (.*)", line)
        if not match:
            _unsupported("unexpected SSA: " + line)
        name, text = match.groups()
        address=re.match(r'getelementptr (?:inbounds )?(?:nusw )?(?:nuw )?('+_TYPE_RE+r'), ptr (%[-\w.]+), i64 (.*)',text)
        if address:
            pointers[name]=pointers[address[2]]
            size=np.dtype(dtype_of(address[1])).itemsize*width(address[1])
            pointer_bytes[name]=make_node('add','int64',(pointer_bytes[address[2]],
                make_node('mul','int64',(operand(address[3],'int64'),Node('constant','int64',value=size)))))
            continue
        pointer_integer=re.match(r'ptrtoint ptr (%[-\w.]+) to i64',text)
        if pointer_integer:
            pointer=pointer_integer[1];variable=pointers[pointer]
            field=(inventory or {}).get(variable)
            if field is None or not field.persistent:
                _unsupported('unproven pointer region: '+text)
            count=(Node('constant','int64',value=1) if field.scalar else
                   Node('argument','int64',value='N'))
            extent=make_node('mul','int64',(count,Node('constant','int64',value=np.dtype(field.dtype).itemsize)))
            # These model ArrayVariables have distinct storage. Retain an
            # abstract address until its alias predicate can be proved from
            # complete array extents; never invent concrete CPU addresses.
            definitions[name]=Node('address','uint64',(pointer_bytes[pointer],extent),value=variable)
            continue
        inserted=re.match(r'insertelement ('+_TYPE_RE+r') (.*?), ('+_SCALAR_TYPE_RE+r') (.*?), i\d+ (\d+)',text)
        if inserted:
            count=width(inserted[1]);value=operand(inserted[2],dtype_of(inserted[1]))
            lanes=list(value if isinstance(value,tuple) else (value,)*count)
            lanes[int(inserted[5])]=operand(inserted[4],dtype_of(inserted[3]))
            definitions[name]=tuple(lanes)
            continue
        shuffled=re.match(r'shufflevector ('+_TYPE_RE+r') ([^,]+), '+_TYPE_RE+r' ([^,]+), ('+_TYPE_RE+r') (.*)',text)
        if shuffled:
            count=width(shuffled[1]);left=operand(shuffled[2],dtype_of(shuffled[1]));right=operand(shuffled[3],dtype_of(shuffled[1]))
            combined=(left if isinstance(left,tuple) else (left,)*count)+(right if isinstance(right,tuple) else (right,)*count)
            mask=operand(shuffled[5],'int32');mask=mask if isinstance(mask,tuple) else (mask,)*width(shuffled[4])
            definitions[name]=tuple(combined[int(item.value)] for item in mask)
            continue
        load=re.match(r'load ('+_TYPE_RE+r'), ptr (%[-\w.]+),',text)
        if load:
            variable=pointers[load[2]]
            value=Node('argument',dtype_of(load[1]),value='i' if variable=='__selection' else variable)
            if variable=='__selection':selected_indices.add(value)
            definitions[name]=(value,)*width(load[1]) if width(load[1])>1 else value
            continue
        operation = re.match(r"(fadd|fsub|fmul|fdiv|frem|fneg|add|sub|mul|sdiv|urem|and|or|xor|shl|lshr|ashr) ("+_FLAGS+r")("+_TYPE_RE+r") (.*)", text)
        if operation:
            op, flags, llvm_type, args = operation.groups()
            dtype = dtype_of(llvm_type)
            definitions[name] = make_node(op, dtype, tuple(operand(token, dtype) for token in split_args(args)),
                                     contract="contract" in flags or "fast" in flags)
            continue
        frozen = re.match(r"freeze ("+_TYPE_RE+r") (.*)", text)
        if frozen:
            dtype=dtype_of(frozen[1])
            definitions[name]=make_node('freeze',dtype,(operand(frozen[2],dtype),))
            continue
        conversion = re.match(r"(fptrunc|fpext|sitofp|uitofp|fptosi|zext|sext|trunc) (?:"+_FLAGS+r")("+_TYPE_RE+r") (.*?) to ("+_TYPE_RE+r")", text)
        if conversion:
            op, old, arg, new = conversion.groups()
            definitions[name] = make_node(op, dtype_of(new), (operand(arg, dtype_of(old)),))
            continue
        call = re.match(r"(?:tail )?call (?:"+_FLAGS+r")(?:noundef )?("+_TYPE_RE+r") @([-\w.]+)\((.*?)\)", text)
        if call:
            llvm_type, function, arguments = call.groups()
            args = []
            for item in split_args(arguments):
                arg = re.match(r"("+_TYPE_RE+r") (?:(?:noundef|signext|zeroext) )*(.*)", item)
                if arg is None:
                    _unsupported("call operand: " + item)
                args.append(operand(arg[2], dtype_of(arg[1])))
            function=re.sub(r"\.v\d+(f\d+|i\d+)$",r".\1",function)
            definitions[name] = make_node("call:"+function, dtype_of(llvm_type), tuple(args),value=block)
            continue
        compare = re.match(r"(fcmp|icmp) (?:"+_FLAGS+r")([a-z]+) ("+_TYPE_RE+r") (.*)", text)
        if compare:
            op, predicate, llvm_type, arguments = compare.groups()
            dtype = dtype_of(llvm_type)
            definitions[name] = make_node(op+":"+predicate, "bool", tuple(operand(arg, dtype) for arg in split_args(arguments)))
            continue
        select = re.match(r"select (?:"+_FLAGS+r")i1 (.*?), ("+_TYPE_RE+r") (.*?), \2 (.*)", text)
        if select:
            condition, llvm_type, yes, no = select.groups()
            dtype = dtype_of(llvm_type)
            definitions[name] = make_node("select", dtype, (operand(condition, "bool"), operand(yes, dtype), operand(no, dtype)))
            continue
        phi = re.match(r"phi (?:"+_FLAGS+r")("+_TYPE_RE+r") (.*)", text)
        if phi:
            loop=loops.get(block)
            if loop is not None and name==loop['index']:
                definitions[name]=loop['base'];loop_entries[block]=loop['entry']
                continue
            if loop is not None and width(phi[1])>1:
                alternatives=re.findall(r'\[ (.*?), %([-\w.]+) \]',phi[2])
                initial=next(value for value,pred in alternatives if pred!=block)
                back=next(value for value,pred in alternatives if pred==block)
                increment=raw[back][1]
                advance=re.match(r'add '+_FLAGS+_TYPE_RE+r' '+re.escape(name)+r', (.*)',increment)
                if advance is None:_unsupported('non-affine SIMD induction')
                step=operand(advance[1],dtype_of(phi[1]));step=step[0] if isinstance(step,tuple) else step
                if step.op!='constant' or int(step.value)%loop['stride']:_unsupported('non-integral SIMD induction')
                factor=int(step.value)//loop['stride'];dtype=dtype_of(phi[1])
                first=operand(initial,dtype);first=first[0] if isinstance(first,tuple) else first
                start=operand(loop['initial'],'int64')
                start=make_node('trunc',dtype,(start,)) if dtype!='int64' else start
                base=make_node('trunc',dtype,(loop['base'],)) if dtype!='int64' else loop['base']
                factor_node=Node('constant',dtype,value=factor)
                shift=make_node('sub',dtype,(first,make_node('mul',dtype,(factor_node,start))))
                definitions[name]=tuple(make_node('add',dtype,(make_node('mul',dtype,(factor_node,
                    make_node('add',dtype,(base,Node('constant',dtype,value=lane))))),shift)) for lane in range(width(phi[1])))
                continue
            if phi[1]=='i64' and name.startswith(('%_lane.','%index')):
                definitions[name]=Node('argument','int64',value='i')
                alternatives=re.findall(r'\[ (.*?), %([-\w.]+) \]',phi[2])
                if any(value.startswith(('%inc','%index.next')) for value,_ in alternatives):
                    loop_entries[block]=next(previous for value,previous in alternatives
                                             if not value.startswith(('%inc','%index.next')))
                continue
            node = definitions[name] = Node("phi", dtype_of(phi[1]))
            delayed.append((node, re.findall(r"\[ (.*?), %([-\w.]+) \]", phi[2]), block))
            continue
        _unsupported("unexpected instruction: " + text)
    paths = {'entry':Node('constant','bool',value=True)}
    incoming = {}
    for label, branches in blocks.items():
        for branch in branches:
            conditional = re.match(r"br i1 (.*?), label %([-\w.]+), label %([-\w.]+)", branch)
            if conditional:
                condition = operand(conditional[1], "bool")
                def has_address(node):
                    return node.op=='address' or any(has_address(arg) for arg in node.args)
                if has_address(condition):
                    condition=Node('alias_guard','bool',(condition,),value=label)
                # The scalar graph is invoked only for an existing neuron
                # (and reset only for an existing spike), so its loop is live.
                if (condition.op=='icmp:sgt' and condition.args[0].op=='argument'
                        and condition.args[0].value=='N' and condition.args[1].op=='constant'
                        and condition.args[1].value==0):
                    condition=_literal('true','bool')
                if label in loops and label in (conditional[2],conditional[3]):
                    condition=_literal('false' if conditional[2]==label else 'true','bool')
                edges = [(conditional[2], condition), (conditional[3], Node("xor", "bool", (condition, _literal("true", "bool"))))]
            else:
                target = re.match(r"br label %([-\w.]+)", branch)
                edges = [(target[1], _literal("true", "bool"))]
            for target, condition in edges:
                incoming.setdefault(target, []).append((label, condition))

    def path(label):
        if label not in paths:
            predecessors=incoming[label]
            if label in loop_entries:
                predecessors=[item for item in predecessors if item[0]==loop_entries[label]]
            terms = [Node("and", "bool", (path(previous), condition)) for previous, condition in predecessors]
            result = terms[0]
            for term in terms[1:]:
                result = Node("or", "bool", (result, term))
            paths[label] = result
        return paths[label]
    for node, alternatives, label in delayed:
        values = [(operand(value, node.dtype), previous) for value, previous in alternatives]
        result = values[-1][0]
        for value, previous in reversed(values[:-1]):
            edge = next(condition for predecessor, condition in incoming[label] if predecessor == previous)
            result = Node("select", node.dtype, (Node("and", "bool", (path(previous), edge)), value, result))
        node.op, node.args = result.op, result.args
    for name,value,label,domain in writes:
        fallback=outputs.get(name,Node('constant','bool',value=False) if name=='_cond'
                             else Node('argument',value.dtype,value=name))
        condition=path(label) if domain is None else Node('and','bool',(path(label),domain))
        outputs[name]=Node('select',value.dtype,(condition,value,fallback))
    for index,(scalar,kind,label) in enumerate(operator_errors):
        prefix='__eon_operator_scalar_failure_' if scalar else '__eon_operator_failure_'
        outputs[prefix+str(index)]=Node('operator_error','bool',(path(label),kind))
    for width,suffix in ((32,"f"),(64,"")):
        if "; eon.target-paired-sincos"+suffix+"\n" not in source:continue
        pairs={}
        for node in definitions.values():
            if node.op in {f"call:llvm.sin.f{width}",f"call:llvm.cos.f{width}"}:
                pairs.setdefault((node.value,node.args[0]),{}).setdefault(node.op,[]).append(node)
        for calls in pairs.values():
            if len(calls)!=2:continue
            for op,nodes in calls.items():
                for node in nodes:
                    node.op="call:eon_sincos_"+("sine" if ".sin." in op else "cosine")
    return outputs


_LIBM = ctypes.CDLL(None)


class _Scalar32Pair(ctypes.Structure):
    _fields_=[("sine",ctypes.c_float),("cosine",ctypes.c_float)]


class _Scalar64Pair(ctypes.Structure):
    _fields_=[("sine",ctypes.c_double),("cosine",ctypes.c_double)]


def _paired_scalar(value,dtype):
    single=dtype=="float32"
    function=getattr(_LIBM,"__sincosf_stret" if single else "__sincos_stret")
    function.argtypes=[ctypes.c_float if single else ctypes.c_double]
    function.restype=_Scalar32Pair if single else _Scalar64Pair
    return function(float(value))


def evaluate(op, dtype, args):
    cast = np.dtype(dtype).type
    if op.startswith("call:"):
        name = op.split(":", 1)[1].removeprefix("llvm.").split(".")[0]
        if name in {"eon_sincos_sine","eon_sincos_cosine"}:
            return cast(getattr(_paired_scalar(args[0],dtype),name.removeprefix("eon_sincos_")))
        name = {"fabs": "fabs", "fmuladd": "fma"}.get(name, name)
        if name=='powi':
            function=getattr(_LIBM,'__powisf2' if dtype=='float32' else '__powidf2')
            ctype=ctypes.c_float if dtype=='float32' else ctypes.c_double
            function.restype,function.argtypes=ctype,[ctype,ctypes.c_int]
            return cast(function(float(args[0]),int(args[1])))
        if name.endswith("f") and name[:-1] in {"exp","log","expm1","log1p","log10","sinh","cosh","tanh","atan","asin","acos","pow","fmod","floor","ceil","sin","cos","tan"}: name=name[:-1]
        if name not in {"exp", "exp2", "expm1", "log", "log10", "log1p", "sqrt", "sin", "cos", "tan", "sinh", "cosh", "tanh", "fabs", "floor", "ceil", "fma", "pow", "fmod", "atan", "asin", "acos"}:
            _unsupported("invariant intrinsic " + name)
        function = getattr(_LIBM, name + ("f" if dtype == "float32" else ""))
        ctype = ctypes.c_float if dtype == "float32" else ctypes.c_double
        function.restype, function.argtypes = ctype, [ctype]*len(args)
        return cast(function(*[float(a) for a in args]))
    if op in {'zext','sext','trunc'}:
        old=np.asarray(args[0]).dtype
        bits=1 if old.kind=='b' else old.itemsize*8
        value=int(args[0])
        if op=='sext' and bits==1:value=-value
        if op=='zext':value &= (1<<bits)-1
        if dtype!='bool':
            width=np.dtype(dtype).itemsize*8
            value=(value+(1<<(width-1)))%(1<<width)-(1<<(width-1))
        return cast(value)
    if op in {"fptrunc", "fpext", "sitofp", "uitofp", "fptosi"}:
        if op=='sitofp' and isinstance(args[0],(bool,np.bool_)):
            return cast(-int(args[0]))
        if op=='uitofp':
            old=np.asarray(args[0]).dtype
            return cast(int(args[0])&((1<<(1 if old.kind=='b' else old.itemsize*8))-1))
        return cast(args[0])
    if op == "select":
        return cast(args[1] if args[0] else args[2])
    if op=='operator_error':return bool(args[0])
    if op == "freeze":
        return cast(args[0])
    if op == "fneg":
        return cast(-args[0])
    if op=='urem':
        bits=np.dtype(dtype).itemsize*8
        a,b=(int(value)&((1<<bits)-1) for value in args)
        value=a%b
        if np.dtype(dtype).kind=='i' and value>=1<<(bits-1):value-=1<<bits
        return cast(value)
    if op.startswith(("icmp:", "fcmp:")):
        predicate = op.split(":")[1]
        if op.startswith('icmp:') and predicate.startswith('u'):
            bits=np.asarray(args[0]).dtype.itemsize*8
            args=[int(value)&((1<<bits)-1) for value in args]
        nan = bool(np.isnan(args[0]) or np.isnan(args[1]))
        comparisons = {"eq": lambda a,b:a==b, "ne":lambda a,b:a!=b, "gt":lambda a,b:a>b,
                       "ge":lambda a,b:a>=b, "lt":lambda a,b:a<b, "le":lambda a,b:a<=b}
        if predicate == "ord":return not nan
        if predicate == "uno":return nan
        base = predicate[-2:]
        result = comparisons[base](*args)
        return bool(result or nan) if predicate.startswith("u") and op.startswith("fcmp") else bool(result and not nan)
    operations = {"fadd":lambda a,b:a+b, "fsub":lambda a,b:a-b, "fmul":lambda a,b:a*b, "fdiv":lambda a,b:a/b,
                  "frem":lambda a,b:np.fmod(a,b),
                  "add":lambda a,b:int(a)+int(b), "sub":lambda a,b:int(a)-int(b), "mul":lambda a,b:int(a)*int(b),
                  "and":lambda a,b:int(a)&int(b), "or":lambda a,b:int(a)|int(b), "xor":lambda a,b:int(a)^int(b)}
    if op not in operations:
        _unsupported("invariant operation " + op)
    with np.errstate(all="ignore"):
        return cast(operations[op](*args))


def specialize(outputs, values, replacements=None, *, contract=True):
    """Contract once, then evaluate only nodes independent of dynamic state."""
    replacements = replacements or {}
    cache = {}
    uses, seen = {}, set()
    def count(node):
        uses[node]=uses.get(node,0)+1
        if node not in seen:
            seen.add(node)
            for arg in node.args:count(arg)
    for node in outputs.values():count(node)
    def dependencies(node):
        if node.op=='argument':return {node.value}
        return set().union(*(dependencies(arg) for arg in node.args)) if node.args else set()
    def visit(node):
        if node in replacements:
            return replacements[node]
        if node in cache:
            return cache[node]
        if node.op == "argument":
            result = Node("constant", node.dtype, value=np.dtype(node.dtype).type(values[node.value])) if node.value in values else node
        elif node.op == "constant":
            result = node
        else:
            # Contract before recursive folding, preserving CPU fma rounding.
            op, arguments = node.op, node.args
            if contract and node.contract and op in {"fadd", "fsub"}:
                index = next((i for i,arg in enumerate(arguments) if arg.op == "fmul" and arg.contract and uses.get(arg)==1), None)
                if index is not None:
                    product, other = arguments[index], arguments[1-index]
                    a,b = product.args
                    if op == "fsub":
                        if index == 0:other=Node("fneg",node.dtype,(other,))
                        else:a=Node("fneg",node.dtype,(a,))
                    op, arguments = "call:llvm.fma", (a,b,other)
            args = tuple(visit(arg) for arg in arguments)
            if op=='address':
                result=Node(op,node.dtype,args,node.value,node.contract)
            elif op=='alias_guard':
                result=args[0] if args[0].op=='constant' else Node(op,node.dtype,args,node.value,node.contract)
            elif op=='sub' and args[0].op==args[1].op=='address' and args[0].value==args[1].value:
                result=visit(Node('sub',node.dtype,(args[0].args[0],args[1].args[0])))
            elif (op=='icmp:ult' and args[0].op=='sub'
                    and args[0].args[0].op==args[0].args[1].op=='address'
                    and args[1].op=='constant'):
                left,right=args[0].args
                bounds=(*left.args,*right.args)
                if all(arg.op=='constant' for arg in bounds):
                    loffset,lsize,roffset,rsize=(int(arg.value) for arg in bounds)
                    delta=loffset-roffset;limit=int(args[1].value)
                    # For disjoint complete arrays, startL-startR is either
                    # <=-sizeL or >=sizeR. Both allocation orders therefore
                    # fail this unsigned short-distance conflict check.
                    proved=(0<=loffset<lsize and 0<=roffset<rsize
                            and -lsize+delta<0 and rsize+delta>=limit and 0<limit<2**63)
                else:proved=False
                result=Node('constant','bool',value=False) if proved else Node(op,node.dtype,args,node.value,node.contract)
            elif op=='operator_error':
                result=(Node('constant','bool',value=False) if args[0].op=='constant' and not args[0].value
                        else Node(op,node.dtype,args,node.value,node.contract))
            elif op.startswith('call:_eon_operator_timed_'):
                result=Node(op,node.dtype,args,node.value,node.contract)
            elif all(arg.op == "constant" for arg in args):
                result = Node("constant", node.dtype, value=evaluate(op,node.dtype,[arg.value for arg in args]))
            elif op == "select" and args[0].op == "constant":
                result = args[1] if args[0].value else args[2]
            elif op in {'and','or'} and node.dtype=='bool' and any(arg.op=='constant' for arg in args):
                fixed=next(arg for arg in args if arg.op=='constant')
                other=args[1] if fixed is args[0] else args[0]
                result=other if bool(fixed.value)==(op=='and') else fixed
            else:
                result = Node(op,node.dtype,args,node.value,node.contract)
                # Reconverged exception checks execute the same successor
                # on both paths. Remove their tautological path predicates
                # so clock roots remain shared with the dynamics graph.
                def complements(a,b):
                    return ((a.op=='xor' and a.args[0] is b and a.args[1].op=='constant' and bool(a.args[1].value))
                            or (b.op=='xor' and b.args[0] is a and b.args[1].op=='constant' and bool(b.args[1].value)))
                if op=='or' and node.dtype=='bool':
                    a,b=args
                    if complements(a,b):result=Node('constant','bool',value=True)
                    elif a.op=='and' and b.op=='and' and a.args[0] is b.args[0] and complements(a.args[1],b.args[1]):
                        result=a.args[0]
                if op.startswith('call:') and dependencies(result)=={'not_refractory'}:
                    choices=[specialize({'value':result},{'not_refractory':value})['value'] for value in (False,True)]
                    if all(choice.op=='constant' for choice in choices):
                        result=Node('select',node.dtype,(Node('argument','bool',value='not_refractory'),choices[1],choices[0]))
        cache[node] = result
        return result
    return {name:visit(node) for name,node in outputs.items()}


def _resolve_alias_guards(outputs,neurons):
    """Keep conservative short-array branches only when arithmetic agrees.

    A guard sized for an unrolled main loop can precede a shorter epilogue.
    Distinct small arrays may be closer than that conservative distance.
    Check both paths at each logical row; never invent CPU pointer addresses.
    """
    def guard(nodes):
        seen=set()
        def visit(node):
            if node in seen:return None
            seen.add(node)
            if node.op=='alias_guard':return node
            return next((found for arg in node.args if (found:=visit(arg)) is not None),None)
        return next((found for node in nodes.values() if (found:=visit(node)) is not None),None)
    if guard(outputs) is None:return outputs
    # The original DAG still carries N, even when the separately lowered DAG
    # has its model constants. Bind the immutable array extent before deciding
    # whether a conservative short-array branch needs row-wise equivalence.
    # Keep the existing CPU contraction graph; this is not a new FMA pass.
    if neurons is not None:
        outputs=specialize(outputs,{'N':int(neurons)},contract=False)
    if guard(outputs) is None:return outputs
    if neurons is None or not 0<int(neurons)<=256:
        _unsupported('unproved alias branch requires bounded row-wise equivalence')
    def signature(node):
        if node.op=='constant':value=np.asarray(node.value,dtype=node.dtype).tobytes()
        else:value=None if node.op.startswith('call:') else node.value
        return node.op,node.dtype,value,node.contract,tuple(signature(arg) for arg in node.args)
    def canonical(nodes):
        memo={}
        def visit(node):
            if node in memo:return memo[node]
            args=tuple(visit(arg) for arg in node.args)
            result=Node(node.op,node.dtype,args,node.value,node.contract) if args else node
            if node.op=='select':
                condition,yes,no=args
                # SIMD stores and a scalar tail can reconverge under the
                # same exception predicate. Remove only redundant decisions;
                # retain each numerical operation and contraction marker.
                if yes.op=='select' and signature(yes.args[0])==signature(condition):yes=yes.args[1]
                if no.op=='select' and signature(no.args[0])==signature(condition):no=no.args[2]
                result=(yes if signature(yes)==signature(no) else
                        Node('select',node.dtype,(condition,yes,no),node.value,node.contract))
            memo[node]=result
            return result
        return {name:visit(node) for name,node in nodes.items()}
    def resolve(nodes):
        choice=guard(nodes)
        if choice is None:return canonical(nodes)
        alternatives=[resolve(specialize(nodes,{},
            {choice:Node('constant','bool',value=flag)},contract=False)) for flag in (False,True)]
        if any(signature(alternatives[0][name])!=signature(alternatives[1][name]) for name in nodes):
            _unsupported('CPU pointer-alias branches have different typed arithmetic')
        return alternatives[0]
    rows=[resolve(specialize(outputs,{'N':int(neurons),'i':row},contract=False))
          for row in range(int(neurons))]
    result={}
    for name in outputs:
        value=rows[-1][name]
        for row in range(int(neurons)-2,-1,-1):
            if signature(rows[row][name])!=signature(value):
                predicate=Node('icmp:eq','bool',(Node('argument','int64',value='i'),Node('constant','int64',value=row)))
                value=Node('select',value.dtype,(predicate,rows[row][name],value))
        result[name]=value
    return result


def render(outputs, used_math=None, zero_division_error_name='_eon_operator_zero_error',*,
           operator_policies=None,reference_flags=(),operator_compiler=(),function_registry=None,code_names=None,
           safe_float32_div=False):
    from .compiler import _literal, _kind, _Renderer, _Expression

    lines, names, pairs = [], {}, {}
    from .hyperbolic_pair import SsaPairs
    hyperbolic_pairs = SsaPairs(outputs, used_math) if used_math is not None else None
    def visit(node):
        if node in names:return names[node]
        if node.op == "argument":return (code_names or {}).get(node.value,node.value)
        if node.op == "constant":return _literal(node.value,_kind(node.dtype))
        paired = hyperbolic_pairs.expression(node, visit, lines) if hyperbolic_pairs is not None else None
        if paired is not None:
            names[node] = paired
            return paired
        if operator_policies is not None:
            from .operator_graph import literal_powi_product
            frozen=literal_powi_product(node,reference_flags,operator_compiler)
            if frozen is not None:
                policy,inputs=frozen
                operator_policies[policy.name]=policy
                used_math.add('operator_remainder_float')
                if any(arg.dtype=='float32' for arg in inputs):used_math.add('operator_remainder_float32')
                if policy.library_power:used_math.add('operator_pow_'+policy.input_kinds[0])
                if policy.library_powi:used_math.add('operator_powi_'+policy.input_kinds[0])
                return policy.name+'('+','.join(visit(arg) for arg in inputs)+')'
        args = [visit(arg) for arg in node.args]
        op, dtype = node.op, node.dtype
        kind = _kind(dtype)
        prefix = "f" if dtype == "float32" else "d"
        if op=='freeze':
            code=args[0]
        elif op=='frem':
            used_math.add('operator_remainder_'+kind)
            code=prefix+'_operator_fmod('+','.join(args)+')'
        elif op in {"fadd","fsub","fmul","fdiv","fneg"}:
            helper = prefix+"_"+op[1:]
            if safe_float32_div and op=='fdiv' and dtype=='float64' and used_math is not None:
                used_math.add('_float32_safe_div')
                helper='_eon_float32_safe_div'
            code = helper+"("+", ".join(args)+")"
        elif op.startswith("call:"):
            from .operator_graph import render_intrinsic
            operator_code=render_intrinsic(op.split(':',1)[1],dtype,args,used_math)
            if function_registry is not None and op.split(':',1)[1] in function_registry.timed_intrinsics:
                operator_code=function_registry.intrinsic(op.split(':',1)[1],dtype,args)
            function=op.split(":")[1].removeprefix("llvm.").split(".")[0]
            base = function[:-1] if function.endswith("f") and function[:-1] in {"log10","sinh","cosh","tanh","atan","asin","acos","sin","cos","tan"} else function
            if operator_code is not None:
                code=operator_code
            elif base in {"eon_sincos_sine","eon_sincos_cosine"}:
                if used_math is None:_unsupported("missing math dependency collector")
                width=32 if dtype=="float32" else 64
                used_math.update({"sin_f32","cos_f32"} if width==32 else {"sin","cos"})
                key=(node.value,node.args[0],width)
                if key not in pairs:
                    pair=f"_clang_pair_{len(pairs)}"
                    lines.append(f"const _eon_trig{width}_pair {pair} = _eon_trig{width}_sincos_scalar({args[0]});")
                    pairs[key]=pair
                code=pairs[key]+"."+base.removeprefix("eon_sincos_")
            elif base in {'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'log10', 'sinh', 'cosh', 'tanh'}:
                if used_math is None:
                    missing = ('inverse math' if base in ('asin', 'acos') else
                               'arctangent' if base == 'atan' else 'math')
                    _unsupported('missing ' + missing + ' dependency collector')
                from .unary_math_policy import unary_helper
                name = {'asin': 'arcsin', 'acos': 'arccos', 'atan': 'arctan'}.get(base, base)
                single = dtype == 'float32'
                used_math.add(name + '_f32' if single else name)
                function = unary_helper(name, single, 'cython')
                code = function + '(' + ', '.join(args) + ')'
            else:
                if function not in {"fma","fmuladd","fabs","sqrt"}:
                    _unsupported("state-dependent intrinsic "+function)
                function={"fmuladd":"fma","fabs":"abs"}.get(function,function)
                code=("clang_d_fma" if dtype=="float64" and function=="fma" else prefix+"_"+function)+"("+", ".join(args)+")"
        elif op in {"fptrunc","fpext","sitofp","uitofp","fptosi","zext","sext","trunc"}:
            old=node.args[0].dtype
            argument=args[0]
            if op in ('sitofp','sext') and old!='bool' and np.dtype(old).itemsize<8:
                argument=f"({'char' if np.dtype(old).itemsize==1 else 'short' if np.dtype(old).itemsize==2 else 'int'})({argument})"
            value=_Expression(argument,_kind(old))
            code=_Renderer.convert(value,kind).code
            if op=='uitofp':
                if used_math is None:_unsupported('missing unsigned conversion dependency collector')
                used_math.add('operator_uint_float32')
                old=node.args[0].dtype
                mask=(1<<(1 if old=='bool' else np.dtype(old).itemsize*8))-1
                code=f"{prefix}_operator_from_uint((ulong({args[0]}) & {mask}ul))" if kind=='float32' else f"d_from_uint((ulong({args[0]}) & {mask}ul))"
            if op=='sitofp' and node.args[0].dtype=='bool':code=f'{prefix}_from_int(-long({args[0]}))'
            if op=='sext' and node.args[0].dtype=='bool':code=f'-int({args[0]})'
            if op=='zext':
                old=node.args[0].dtype
                mask=(1<<(1 if old=='bool' else np.dtype(old).itemsize*8))-1
                code=f'(ulong({args[0]}) & {mask}ul)'
            if kind=="int":code=f"({'int' if np.dtype(dtype).itemsize<=4 else 'long'})({code})"
        elif op=="select":code=f"({args[0]} ? {args[1]} : {args[2]})"
        elif op.startswith("fcmp:"):
            predicate=op.split(":")[1];p="f" if node.args[0].dtype=="float32" else "d"
            ordered=f"(!d_is_nan({args[0] if p=='d' else 'f_to_d('+args[0]+')'}) && !d_is_nan({args[1] if p=='d' else 'f_to_d('+args[1]+')'}))"
            if predicate in {"ord","uno"}:code=ordered if predicate=="ord" else f"(!{ordered})"
            else:
                relation=predicate[-2:];call=f"{p}_{'eq' if relation=='ne' else relation}({args[0]}, {args[1]})"
                if relation=="ne":call=f"(!{call})"
                code=f"({ordered} && {call})" if predicate.startswith("o") else f"(!{ordered} || {call})"
        else:
            symbols={"add":"+","sub":"-","mul":"*","sdiv":"/","urem":"%","and":"&","or":"|","xor":"^","shl":"<<","lshr":">>","ashr":">>"}
            if op.startswith("icmp:"):
                symbol={"eq":"==","ne":"!=","gt":">","ge":">=","lt":"<","le":"<="}[op[-2:]]
                if op.split(':')[1].startswith('u'):
                    unsigned='uint' if np.dtype(node.args[0].dtype).itemsize<=4 else 'ulong'
                    args=[f'({unsigned})({arg})' for arg in args]
            else:symbol=symbols.get(op)
            if symbol is None:_unsupported("dynamic operation "+op)
            code=f"({args[0]} {symbol} {args[1]})"
            if op in {'add','sub','mul','shl','lshr','urem'} and dtype!='bool':
                signed='int' if np.dtype(dtype).itemsize<=4 else 'long'
                unsigned='uint' if signed=='int' else 'ulong'
                code=f'as_type<{signed}>({unsigned}({args[0]}) {symbol} {unsigned}({args[1]}))'
        name=f"_clang_{len(names)}"
        metal={"float":"ulong","float32":"uint","bool":"bool","int":"int" if np.dtype(dtype).itemsize<=4 else "long"}[kind]
        lines.append(f"const {metal} {name} = {code};")
        names[node]=name
        return name
    # SSA outputs all read the incoming state. A bare argument (or an inline
    # policy call) must not read a value already replaced by an earlier store.
    write_names={(code_names or {}).get(name,name) for name in outputs
        if not name.startswith(('__eon_operator_failure_','__eon_operator_scalar_failure_'))}
    captured=0
    def output_value(node,target=None):
        nonlocal captured
        value=visit(node)
        if set(re.findall(r'[A-Za-z_]\w*',value)) & (write_names-{target}):
            kind=_kind(node.dtype)
            metal={"float":"ulong","float32":"uint","bool":"bool",
                   "int":"int" if np.dtype(node.dtype).itemsize<=4 else "long"}[kind]
            local=f'_clang_output_{captured}'
            captured+=1
            lines.append(f'const {metal} {local} = {value};')
            return local
        return value
    assignments=[]
    for name,node in outputs.items():
        if name.startswith(('__eon_operator_failure_','__eon_operator_scalar_failure_')):
            if node.op=='constant' and not node.value:continue
            condition,kind=node.args
            predicate=output_value(condition)
            error_kind=output_value(kind)
            used_math.add('operator_zero_division')
            assignments.append(f'if({predicate}) {{ if(!{zero_division_error_name}) '
                f'{zero_division_error_name}_kind={error_kind}; {zero_division_error_name}=true; }}')
        else:
            target=(code_names or {}).get(name,name)
            assignments.append(f"{target} = {output_value(node,target)};")
    return "{\n"+"\n".join("    "+line for line in lines+assignments)+"\n}"


def lower(model, inventory, scalar, vector, types, values, stage='state',*,function_registry=None):
    if len(scalar+vector)==1 and (scalar+vector)[0].var=='_cond' and (scalar+vector)[0].expr in {'True','False'}:
        outputs={'_cond':Node('constant','bool',value=(scalar+vector)[0].expr=='True')}
        return outputs,outputs
    types=types.copy()
    for stmt in scalar+vector:types[stmt.var]=stmt.dtype
    target=model.reference_codegen_target or model.float32_codegen
    prepared_constants=({name:values[name] for name,var in inventory.items()
        if name in values and var.scalar and var.constant and var.read_only and not var.persistent
        # CPP loads dt from its clock array. Specialize its value only after
        # compiling the graph: embedding it here changes cancellation/FMA.
        and name != 'dt'
        and np.dtype(var.dtype).kind=='f'} if target=='cpp' else None)
    source=_cpp_source(inventory,scalar,vector,types,stage,target,prepared_constants,function_registry)
    outputs=parse(_ir(source,tuple(model.float32_compile_args),model.float32_compiler_identity),inventory)
    outputs=specialize(outputs,{})
    lowered=specialize(outputs,values)
    return _resolve_alias_guards(outputs,values.get('N')), _resolve_alias_guards(lowered,values.get('N'))
