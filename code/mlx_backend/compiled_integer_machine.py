"""Preparation-only proof of constant target registers on LLVM poison edges.

Undefined C++ conversions have no portable numeric rule. Only a concrete,
constant physical register that the selected compiler stores on the matching
poison edge is admitted here. No compiled function or CPU timestep is executed.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from .compiled_cache import cache_output, cache_lock, atomic_write
import numpy as np

from .compiler import UnsupportedModelError


def floating_add_reassociations(policy, cpp, ir, cache, key):
    """Recover final-target addition trees whose LLVM order is not final.

    AArch64's machine combiner can balance reassociable additions after LLVM
    IR emission. Match complete leaf multisets and source-located libm calls;
    never infer a floating result from a different numerical expression.
    """
    from collections import Counter
    import sysconfig
    import brian2 as b

    body = re.search(r'define[^\n]*integer_block[^\n]*\{\n(.*?)\n\}', ir, re.S)
    if body is None:
        return {}
    code = body[1]
    additions = {}
    for target, kind, left, right in re.findall(
            r'(%[\w.]+) = fadd [^\n]*?\b(float|double) (%[\w.]+), (%[\w.]+)', code):
        additions[target] = (kind, left, right)
    uses = Counter(re.findall(r'%[\w.]+', '\n'.join(
        line.split(' = ', 1)[-1] for line in code.splitlines()
        if not line.lstrip().startswith('#dbg_'))))

    def llvm_tree(token, kind):
        item = additions.get(token)
        if item is not None and item[0] == kind and uses[token] == 1:
            return ('add', kind, llvm_tree(item[1], kind), llvm_tree(item[2], kind))
        return ('leaf', kind, token)

    def leaves(tree):
        return leaves(tree[2]) + leaves(tree[3]) if tree[0] == 'add' else [tree]

    roots = {}
    for token, (kind, left, right) in additions.items():
        tree = ('add', kind, llvm_tree(left, kind), llvm_tree(right, kind))
        if len(leaves(tree)) >= 4:
            roots[token] = tree
    if not roots:
        return {}

    locations = {name: text for name, text in re.findall(r'^!(\d+) = (.*)$', ir, re.M)}

    def source_site(number):
        seen = set()
        while number not in seen:
            seen.add(number)
            text = locations.get(number, '')
            parent = re.search(r'inlinedAt: !(\d+)', text)
            if parent:
                number = parent[1]
                continue
            line = re.search(r'line: (\d+)', text)
            column = re.search(r'column: (\d+)', text)
            return (int(line[1]), int(column[1])) if line and column else None
        return None

    calls = {}
    for line in code.splitlines():
        match = re.search(r'(%[\w.]+) = .*?call .*?\b(float|double) @([\w.]+)\(', line)
        debug = re.search(r'!dbg !(\d+)', line)
        if not match or not debug:
            continue
        token, kind, function = match.groups()
        name = function.removeprefix('llvm.').split('.')[0]
        if name.endswith('f') and kind == 'float':
            name = name[:-1]
        site = source_site(debug[1])
        if site is not None:
            calls.setdefault((name, kind, site), []).append(token)

    assembly = cache / (key + '.additions.s')
    with cache_output(assembly) as pending:
        if pending is not None:
            result = subprocess.run([
                *policy.compiler, *policy.python_flags, *policy.flags, *policy.compiler_options,
                '-I' + sysconfig.get_path('include'), '-I' + np.get_include(),
                '-I' + str(Path(b.__file__).parent / 'synapses'),
                '-fno-discard-value-names', '-S', str(cpp), '-o', str(pending),
            ], capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise UnsupportedModelError('CPU floating addition target inspection failed: ' + result.stderr)
    text = assembly.read_text()
    function = re.search(r'^\S*integer_block\S*:[^\n]*\n(.*?)\.cfi_endproc', text, re.M | re.S)
    if function is None:
        return {}
    registers, candidates = {}, []
    site = None

    def register(name):
        match = re.fullmatch(r'[vds](\d+)', name)
        return int(match[1]) if match else None

    def read(name, kind):
        tree = registers.get(register(name))
        return tree if tree is not None and tree[1] == kind else None

    source_name = re.escape(cpp.name)
    for raw in function[1].splitlines():
        if raw.lstrip().startswith('.loc'):
            original = re.findall(source_name + r':(\d+):(\d+)', raw)
            direct = re.match(r'\s*\.loc\s+1\s+(\d+)\s+(\d+)', raw)
            site = tuple(map(int, original[-1])) if original else tuple(map(int, direct.groups())) if direct else None
            continue
        if re.match(r'^LBB\w*:', raw):
            registers.clear()
        line = raw.split(';', 1)[0].strip()
        if not line or line.startswith('.') or line.endswith(':'):
            continue
        opcode, _, rest = line.partition('\t')
        if not rest:
            opcode, _, rest = line.partition(' ')
        args = [part.strip() for part in rest.split(',')]
        if opcode == 'bl':
            name = args[0].removeprefix('_')
            kind = 'float' if name.endswith('f') else 'double'
            name = name[:-1] if kind == 'float' else name
            matches = calls.get((name, kind, site), ())
            registers = {number: tree for number, tree in registers.items() if 8 <= number <= 15}
            if len(matches) == 1:
                registers[0] = ('leaf', kind, matches[0])
            continue
        destination = register(args[0]) if args else None
        if destination is None:
            continue
        if opcode in ('mov.16b', 'fmov') and len(args) == 2 and register(args[1]) is not None:
            tree = registers.get(register(args[1]))
        elif opcode == 'fadd' and len(args) == 3:
            kind = 'float' if args[0].startswith('s') else 'double'
            left, right = read(args[1], kind), read(args[2], kind)
            tree = ('add', kind, left, right) if left is not None and right is not None else None
            if tree is not None and len(leaves(tree)) >= 4:
                candidates.append(tree)
        else:
            tree = None
        if tree is None:
            registers.pop(destination, None)
        else:
            registers[destination] = tree

    recovered = {}
    for token, tree in roots.items():
        wanted = Counter(leaves(tree))
        matches = {candidate for candidate in candidates if Counter(leaves(candidate)) == wanted}
        if len(matches) == 1:
            recovered[token] = matches.pop()
    return recovered


def _instructions(assembly):
    body=assembly.split('_integer_block:',1)[1].split('.cfi_endproc',1)[0]
    lines=[];labels={}
    for raw in body.splitlines():
        label=re.match(r'^(L\w+):',raw)
        if label:labels[label[1]]=len(lines)
        if 'implicit-def: $' in raw:
            lines.append(('implicit',raw.split('implicit-def: $',1)[1].strip()))
            continue
        code=raw.split(';',1)[0].strip()
        if not code or code.startswith('.') or label:continue
        opcode,_,arguments=code.partition('\t')
        if not arguments:opcode,_,arguments=code.partition(' ')
        lines.append((opcode,[item.strip() for item in arguments.split(',')]))
    return lines,labels


def _write(op,args,constants):
    """Track exact constant register bits, conservatively forgetting others."""
    if not args:return
    register=args[0]
    if not re.fullmatch(r'[wxds]\d+',register):return
    key=('f' if register[0] in 'ds' else 'x')+register[1:]
    width=32 if register[0] in 'ws' else 64
    value=None
    if op in ('mov','movn','movk') and len(args)>1 and args[1].startswith('#'):
        immediate=int(args[1][1:],0)
        shift=int(args[2].split('#')[1]) if len(args)>2 else 0
        if op=='mov':value=immediate
        elif op=='movn':value=~(immediate<<shift)
        elif key in constants:value=(constants[key]&~(65535<<shift))|(immediate<<shift)
    elif op in ('mov','fmov') and len(args)>1:
        source=args[1]
        skey=('f' if source[0] in 'ds' else 'x')+source[1:]
        value=0 if source in ('xzr','wzr') else constants.get(skey)
    if value is None:constants.pop(key,None)
    else:constants[key]=value&((1<<width)-1)


def poison_constants(policy,cpp,ir,inputs,outputs,types,cache,key,*,original_cpp=None):
    match=re.search(r'define[^\n]*@integer_block\(([^\n]*)\)[^\n]*\{\n(.*?)\n\}',ir,re.S)
    if match is None or not re.search(r'\b(?:poison|undef)\b',match[2]):return {}
    body=match[2]
    edges=[]
    for phi,kind,pairs in re.findall(r'(%[\w.]+) = phi (i\d+) (.*)',body):
        for operand,predecessor in re.findall(r'\[\s*(.*?),\s*%([\w.]+)\s*\]',pairs):
            if operand not in ('poison','undef'):continue
            store=re.search(r'store '+kind+' '+re.escape(phi)+r', ptr (%[\w.]+)',body)
            if store is None:raise UnsupportedModelError('Undefined compiled integer value is not a direct scalar store')
            formal=re.findall(r'(%[\w.]+)\s*(?:,|$)',match[1])
            try:output=outputs[formal.index(store[1])-len(inputs)]
            except (ValueError,IndexError):raise UnsupportedModelError('Undefined compiled integer output ABI is unresolved')
            edges.append((phi,predecessor,kind,output))
    if len(edges)!=1 or len(re.findall(r'\b(?:poison|undef)\b',body))!=1:
        raise UnsupportedModelError('Multiple undefined compiled integer values need distinct target-register proofs')
    assembly=cache/(key+'.poison.s')
    with cache_output(assembly) as pending:
        if pending is not None:
            result=subprocess.run([*policy.compiler,*policy.flags,*policy.compiler_options,
                '-fno-discard-value-names','-S',str(cpp),'-o',str(pending)],
                capture_output=True,text=True,timeout=30)
            if result.returncode:raise UnsupportedModelError('Undefined integer target inspection failed: '+result.stderr)
    text=assembly.read_text();instructions,labels=_instructions(text)
    # AArch64 classifies integer/pointer and floating ABI registers separately.
    integer=0;pointers={}
    for name in inputs:
        if np.dtype(types[name]).kind!='f':integer+=1
    for name in outputs:
        if integer>=8:raise UnsupportedModelError('Undefined integer stack-output ABI needs target-register proof')
        pointers['x'+str(integer)]=name;integer+=1
    phi,predecessor,kind,output=edges[0]
    constants={};proofs=[];pointer_writes=set()
    for index,(opcode,args) in enumerate(instructions):
        if opcode=='implicit':
            if not re.fullmatch(r'[wx]\d+',args):continue
            register='x'+args[1:]
            if register not in constants:continue
            candidate=dict(constants);position=index+1;visited=set()
            for _ in range(128):
                if position in visited or position>=len(instructions):break
                visited.add(position)
                op,operands=instructions[position];position+=1
                if op=='b':
                    if operands[0] not in labels:break
                    position=labels[operands[0]];continue
                if op.startswith('b.') or op in ('bl','blr','cbz','cbnz','tbz','tbnz','ret','retab'):break
                if op.startswith('str'):
                    destination=re.fullmatch(r'\[(x\d+)\]',','.join(operands[1:]))
                    source=operands[0]
                    if destination and source[0] in 'wx' and ('x'+source[1:])==register:
                        base=destination[1]
                        if base not in pointer_writes and pointers.get(base)==output and register in candidate:
                            proofs.append((index,register,candidate[register]))
                            break
                    continue
                if op=='implicit':break
                _write(op,operands,candidate)
            continue
        # Only constants established within a physical basic block may justify
        # an implicit definition; no speculative merge of predecessor values.
        if opcode=='b' or opcode.startswith('b.') or opcode in ('cbz','cbnz','tbz','tbnz','bl','blr'):
            constants={}
        elif opcode not in ('str','strb','strh','stp','cmp','cmn','tst','ret','retab'):
            if args and re.fullmatch(r'[wx]\d+',args[0]) and opcode not in ('nop',):
                pointer_writes.add('x'+args[0][1:])
            _write(opcode,args,constants)
    if len(proofs)!=1:raise UnsupportedModelError('No unique constant physical-register proof for compiled integer poison edge')
    index,register,value=proofs[0]
    bits=int(kind[1:]);value&=(1<<bits)-1
    if original_cpp is None:raise UnsupportedModelError('Undefined integer stores need original CPP loop inspection')
    original_assembly,original_metadata=_original_cpp_assembly(*original_cpp,policy,cache,key)
    original_values=_original_constant_stores(original_assembly,original_metadata,output)
    if any((observed&((1<<bits)-1))!=value for observed in original_values):
        raise UnsupportedModelError('Pointwise and actual CPP undefined-register stores differ')
    metadata=dict(original=original_metadata,original_values=original_values,source_sha256=sha256(cpp.read_bytes()).hexdigest(),assembly_sha256=sha256(text.encode()).hexdigest(),
        phi=phi,predecessor=predecessor,output=output,kind=kind,register=register,value=value,
        implicit_instruction=index,policy=policy.identity)
    atomic_write(cache/(key+'.poison.json'),json.dumps(metadata,indent=2)+'\n')
    return {(phi,predecessor):value}


def _original_cpp_assembly(group,runner,abstract_code,policy,cache,key):
    """Render Brian's actual full CPP loop, with no code-object execution."""
    import brian2 as b
    from brian2.codegen.codeobject import create_runner_codeobj
    from brian2.core.functions import Function
    from brian2.core.variables import ArrayVariable
    from brian2.devices.cpp_standalone.codeobject import CPPStandaloneCodeObject
    from brian2.devices.cpp_standalone.device import CPPStandaloneDevice
    from brian2.devices.device import get_device
    from brian2.codegen.generators.cpp_generator import c_data_type
    key+='_actual_v2'
    metadata=cache/(key+'.original.json');cpp=cache/(key+'.original.cpp');assembly=cpp.with_suffix('.s')
    with cache_lock(metadata):
        if not metadata.exists():
            neurons=policy.vector_length or 1
            saved=[]
            for name,var in group.variables.items():
                if isinstance(var,ArrayVariable) and not var.scalar:
                    saved.append((var,'size',var.size));var.size=neurons+(name=='_spikespace')
            saved.append((group.variables['N'],'value',group.variables['N'].value));group.variables['N'].value=neurons
            try:
                obj=create_runner_codeobj(group=group,code=abstract_code,user_code=runner.user_code,
                    template_name=runner.template,run_namespace={},name='eon_integer_original_cpp*',
                    check_units=runner.check_units,additional_variables=getattr(runner,'variables',None),
                    needed_variables=runner.needed_variables,template_kwds=dict(runner.template_kwds,
                        user_headers=list(b.prefs['codegen.cpp.headers']),profiled=False),
                    override_conditional_write=runner.override_conditional_write,codeobj_class=CPPStandaloneCodeObject)
                class Writer:
                    def __init__(self):self.files={}
                    def write(self,name,content):self.files[name]=str(content)
                device=CPPStandaloneDevice();device.code_objects={obj.name:obj};writer=Writer()
                device.generate_codeobj_source(writer)
                source=writer.files['code_objects/'+obj.name+'.cpp']
                declarations={};symbols={}
                for name,var in obj.variables.items():
                    if isinstance(var,ArrayVariable):
                        symbol=get_device().get_array_name(var)
                        declarations[symbol]=f'extern {c_data_type(var.dtype)}* {symbol};'
                        symbols[symbol]=name
                    elif isinstance(var,Function):
                        for symbol,array in (var.implementations[CPPStandaloneCodeObject].get_namespace(group) or {}).items():
                            declarations[symbol]=f'extern {c_data_type(np.asarray(array).dtype)}* {symbol};'
                objects=cache/(key+'.original.objects.h')
                objects.write_text('#include <cstdint>\nnamespace brian {\n'+'\n'.join(declarations.values())+'\n}\n')
                source=re.sub(r'^#include "code_objects/[^\n]+\n','',source,flags=re.M)
                source=source.replace('#include "objects.h"','#include "'+str(objects)+'"')
                stdint=__import__('pathlib').Path(b.__file__).resolve().parent/'synapses/stdint_compat.h'
                source=source.replace('#include "brianlib/stdint_compat.h"','#include "'+str(stdint)+'"')
                cpp.write_text(source)
                data=dict(symbols=symbols,function='_run_'+obj.name,
                    source_sha256=sha256(source.encode()).hexdigest(),header_sha256=sha256(objects.read_bytes()).hexdigest())
                atomic_write(metadata,json.dumps(data,indent=2)+'\n')
            finally:
                for var,name,value in reversed(saved):setattr(var,name,value)
    data=json.loads(metadata.read_text())
    with cache_output(assembly) as pending:
        if pending is not None:
            include=str(__import__('pathlib').Path(b.__file__).resolve().parent/'devices/cpp_standalone')
            result=subprocess.run([*policy.compiler,*policy.flags,*policy.compiler_options,'-I'+include,
                '-fno-discard-value-names','-S',str(cpp),'-o',str(pending)],capture_output=True,text=True,timeout=30)
            if result.returncode:raise UnsupportedModelError('Original CPP undefined-value inspection failed: '+result.stderr)
    return assembly.read_text(),data


def _original_constant_stores(assembly,data,output):
    """Prove every undef-register store to this actual loop output is constant."""
    label=re.search(r'^([^\s:]*'+re.escape(data['function'])+r'[^\s:]*):',assembly,re.M)
    if label is None:raise UnsupportedModelError('Original CPP loop symbol was not emitted')
    # Reuse the scalar instruction parser after selecting only the original
    # Brian loop; unrolled/remainder paths keep their own physical registers.
    selected=assembly[label.end():].split('.cfi_endproc',1)[0]
    instructions,labels=_instructions('_integer_block:\n'+selected+'\n.cfi_endproc')
    pointers={};addresses={};constants={};values=[]
    def store_pointer(operands):
        match=re.match(r'\[(x\d+)(?:,|\])',','.join(operands[1:]))
        return pointers.get(match[1]) if match else None
    for index,(op,args) in enumerate(instructions):
        if op=='implicit':
            if not re.fullmatch(r'[wx]\d+',args):continue
            register='x'+args[1:]
            if register not in constants:raise UnsupportedModelError('Original CPP undefined register has no constant origin')
            candidate=dict(constants);position=index+1;visited=set();found=False
            for _ in range(128):
                if position in visited or position>=len(instructions):break
                visited.add(position);opcode,operands=instructions[position];position+=1
                if opcode=='b':
                    if operands[0] not in labels:break
                    position=labels[operands[0]];continue
                if opcode.startswith('b.') or opcode in ('bl','blr','cbz','cbnz','tbz','tbnz','ret','retab'):break
                if opcode.startswith('str'):
                    source=operands[0]
                    if source[0] in 'wx' and ('x'+source[1:])==register and store_pointer(operands)==output:
                        if register not in candidate:break
                        values.append(candidate[register]);found=True;break
                    continue
                if opcode=='implicit':break
                _write(opcode,operands,candidate)
            if not found:raise UnsupportedModelError('Original CPP undefined register store is not proven')
            continue
        if op=='adrp':
            symbol=args[1].split('@')[0]
            name=next((data['symbols'][symbol_name] for symbol_name in sorted(data['symbols'],key=len,reverse=True) if symbol_name in symbol),None)
            if name is not None:addresses[args[0]]=name
            else:addresses.pop(args[0],None)
            pointers.pop(args[0],None)
        elif op=='ldr' and args[0].startswith('x'):
            destination=args[0];operand=','.join(args[1:]);base=re.match(r'\[(x\d+)',operand)
            if base and base[1] in addresses:
                if '@GOTPAGEOFF' in operand:addresses[destination]=addresses[base[1]]
                else:
                    pointers[destination]=addresses[base[1]];addresses.pop(destination,None)
            else:pointers.pop(destination,None);addresses.pop(destination,None)
        elif op not in ('str','strb','strh','stp','cmp','cmn','tst','b','ret','retab') and not op.startswith('b.'):
            if args and re.fullmatch(r'[wx]\d+',args[0]):
                pointers.pop('x'+args[0][1:],None);addresses.pop('x'+args[0][1:],None)
        if op=='b' or op.startswith('b.') or op in ('cbz','cbnz','tbz','tbnz','bl','blr'):constants={}
        elif op not in ('str','strb','strh','stp','cmp','cmn','tst','ret','retab'):_write(op,args,constants)
    if not values:raise UnsupportedModelError('Original CPP loop has no verified constant undefined store')
    return values
