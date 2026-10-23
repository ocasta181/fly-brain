"""Capture the reference compiler's constant floating-power operation order.

Brian supplies typed statements. A constant exponent can subsequently become
an observable multiplication chain or sqrt/select/reciprocal sequence. This
small preparation step reads the actual compiler's scalar operator recipe;
all values and timesteps still execute through integer-backed Metal helpers.
"""

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sysconfig
import tempfile

import numpy as np


def compiled_integer_operand_dtype(node,types,*,target='cpp',unsigned_char=False):
    """C integer type metadata at a floating conversion boundary.

    This does not lower or evaluate integer arithmetic. The integer owner
    retains that execution graph; floating consumers must retain its sign.
    """
    import ast
    def promote(dtype):
        return np.dtype('int32') if dtype.kind=='b' or dtype.itemsize<4 else dtype
    if isinstance(node,ast.Name):
        dtype=np.dtype(types[node.id])
        if target=='cython' and dtype==np.dtype('uint8'):
            from brian2.codegen.generators.cython_generator import get_cpp_dtype
            if get_cpp_dtype(dtype)=='char' and not unsigned_char:dtype=np.dtype('int8')
        return dtype if dtype.kind in 'biu' else None
    if isinstance(node,ast.Constant) and isinstance(node.value,(bool,int)):
        value=node.value
        return np.dtype('int32' if -(1<<31)<=value<(1<<31) else
                        'int64' if -(1<<63)<=value<(1<<63) else 'uint64')
    if (target=='cython' and isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
            and node.func.id=='int' and len(node.args)==1 and not node.keywords):
        # Brian's actual stdint_compat.h int_ template returns C int.
        return np.dtype('int32')
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
        dtype=compiled_integer_operand_dtype(node.operand,types,target=target,unsigned_char=unsigned_char)
        return promote(dtype) if dtype is not None else None
    if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult)):
        a=compiled_integer_operand_dtype(node.left,types,target=target,unsigned_char=unsigned_char)
        z=compiled_integer_operand_dtype(node.right,types,target=target,unsigned_char=unsigned_char)
        if a is None or z is None:return None
        a,z=promote(a),promote(z)
        if a.kind==z.kind:return max((a,z),key=lambda item:item.itemsize)
        unsigned,signed=(a,z) if a.kind=='u' else (z,a)
        return unsigned if unsigned.itemsize>=signed.itemsize else signed
    return None


@dataclass(frozen=True)
class ConstantPowerPolicy:
    name: str
    header: str
    source: str
    assembly: str
    compiler: tuple[str, ...]
    flags: tuple[str, ...]
    input_kind: str
    result_kind: str
    library_power: bool
    library_powi: bool


def compiler_command():
    return tuple(shlex.split(os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "clang++"))


def _compile_assembly(source, compiler, flags, *, prefix, filename):
    """Read a scalar compiler recipe without executing the generated code."""
    with tempfile.TemporaryDirectory(prefix=prefix) as directory:
        path = Path(directory)
        cpp, assembly = path / (filename + '.cpp'), path / (filename + '.s')
        cpp.write_text(source)
        subprocess.run([*compiler, *flags, '-S', str(cpp), '-o', str(assembly)],
            check=True, capture_output=True, text=True, timeout=15)
        return assembly.read_text()


def _raw(value,kind):
    if kind=="float32":
        return f"{int(np.float32(value).view(np.uint32))}u"
    return f"{struct.unpack('Q',struct.pack('d',float(value)))[0]}ul"


def _width(register):
    return "float32" if register.startswith("s") else "float"


def _translate(assembly,input_kind,result_kind,name,*,arguments=None):
    """Translate only a scalar constant-power function, never network code."""
    from .compiler import UnsupportedModelError
    pools={}
    label=None
    for raw in assembly.splitlines():
        line=raw.split(";",1)[0].strip()
        if line.endswith(":"):
            label=line[:-1]
        elif label is not None and line.startswith((".quad", ".long")):
            directive,value=line.split(None,1)
            width=64 if directive==".quad" else 32
            try:constant=int(value,0)
            except ValueError:
                # Debug sections contain symbolic address expressions. They
                # are not numeric literal pools; any actual load referring to
                # one still fails the unresolved-literal check below.
                continue
            pools.setdefault(label,[]).append((constant,width))
    block=assembly.split("_eon_power:",1)[1].split(".cfi_endproc",1)[0]
    parameters=(("argument",input_kind),) if arguments is None else arguments
    registers={number:(argument,kind) for number,(argument,kind) in enumerate(parameters)}
    addresses={}
    integers={}
    lines=[]
    flags=None
    library_power=library_powi=False

    def read(register):
        kind=_width(register)
        expression,stored=registers[int(register[1:])]
        if kind!=stored:
            expression=(f"(uint)({expression})" if kind=="float32" else f"(ulong)({expression})")
        return expression,kind

    def write(register,expression,kind=None):
        kind=kind or _width(register)
        variable=f"_eon_power_{len(lines)}"
        lines.append(f"const {'uint' if kind=='float32' else 'ulong'} {variable} = {expression};")
        registers[int(register[1:])]=(variable,kind)

    def condition(code):
        a,b,kind=flags
        prefix="f" if kind=="float32" else "d"
        eq=f"{prefix}_eq({a},{b})"
        less=f"{prefix}_lt({a},{b})"
        greater=f"{prefix}_gt({a},{b})"
        unordered=(f"((({a}&0x7fffffffu)>0x7f800000u)||(({b}&0x7fffffffu)>0x7f800000u))"
                   if kind=="float32" else f"(d_is_nan({a})||d_is_nan({b}))")
        return {"eq":eq,"ne":f"(!{eq})","mi":less,"pl":f"(!{less})",
                "gt":greater,"ge":f"({eq}||{greater})","lt":f"({less}||{unordered})",
                "le":f"(!{greater})","vs":unordered,"vc":f"(!{unordered})"}[code]

    for raw_line in block.splitlines():
        line=raw_line.split(";",1)[0].strip()
        if not line or line.startswith(".") or line.endswith(":"):
            continue
        opcode,_,arguments=line.partition("\t")
        if not arguments:
            opcode,_,arguments=line.partition(" ")
        operands=[part.strip() for part in arguments.split(",")]
        if opcode in ("ret","retab"):
            break
        if opcode=="adrp":
            addresses[operands[0]]=operands[1].split("@")[0]
            continue
        if opcode=="ldr" and operands[0].startswith(("d","s")) and "[sp" not in arguments and "[x29" not in arguments:
            match=re.search(r"([A-Za-z_][A-Za-z0-9_]*)@PAGEOFF",arguments)
            if match:
                symbol=match.group(1)
            elif len(operands)==2 and operands[1] in pools:
                symbol=operands[1]
            else:
                base=re.search(r"\[(x\d+)",arguments)
                symbol=addresses.get(base.group(1)) if base else None
            if symbol not in pools:
                raise UnsupportedModelError(f"Constant operator recipe has an unresolved literal load {line}")
            value=0;offset=0
            for part,width in pools[symbol]:
                value|=(part&((1<<width)-1))<<offset;offset+=width
            width=32 if _width(operands[0])=="float32" else 64
            write(operands[0],str(value&((1<<width)-1))+("u" if width==32 else "ul"))
            continue
        if opcode in ("stp","ldp","str","ldr","sub","add","pacibsp","autibsp"):
            # A non-tail library call saves/restores only frame/link registers.
            # A floating load/store is not silently treated as frame metadata.
            if opcode in ("pacibsp","autibsp") or operands[0]=="sp" or "[sp" in arguments or "[x29" in arguments:
                continue
            raise UnsupportedModelError(f"Constant power recipe has an unresolved {line}")
        if opcode in ("mov","movk","movn"):
            if operands[0] in ("x29","sp"):
                continue
            destination=operands[0]
            mask=(1<< (32 if destination.startswith("w") else 64))-1
            key=int(destination[1:])
            immediate=int(operands[1][1:],0)
            shift=int(re.search(r"#(\d+)",operands[2]).group(1)) if len(operands)>2 else 0
            if opcode=="movk":
                integers[key]=((integers[key]&~(65535<<shift))|((immediate&65535)<<shift))&mask
            else:
                integers[key]=((~(immediate<<shift)) if opcode=="movn" else (immediate<<shift))&mask
            continue
        if opcode=="fmov":
            destination,source=operands
            if source.startswith("#"):
                expression=_raw(float(source[1:]),_width(destination))
            elif source in ("xzr","wzr"):
                expression="0u" if _width(destination)=="float32" else "0ul"
            elif source.startswith(("x","w")):
                bits=integers[int(source[1:])]
                width=32 if _width(destination)=="float32" else 64
                expression=f"{bits&((1<<width)-1)}"+("u" if width==32 else "ul")
            else:
                expression=read(source)[0]
            write(destination,expression)
            continue
        if opcode.startswith(("movi","mvni")):
            value=int(operands[1][1:],0)
            shift=int(re.search(r"#(\d+)",operands[2]).group(1)) if len(operands)>2 else 0
            value<<=shift
            if len(operands)>2 and operands[2].startswith("msl"):
                value|=(1<<shift)-1
            lane={"b":8,"h":16,"s":32,"d":64}[opcode[-1]]
            mask=(1<<lane)-1
            value=((~value) if opcode.startswith("mvni") else value)&mask
            if lane<32:
                value=sum(value<<offset for offset in range(0,32,lane))
            if lane<=32:
                write("s"+operands[0][1:],f"{value}u")
            else:
                write("d"+operands[0][1:],f"{value}ul")
            continue
        if opcode=="fcvt":
            destination,source=operands
            expression,kind=read(source)
            function="f_to_d" if _width(destination)=="float" else "f_from_d"
            write(destination,f"{function}({expression})")
            continue
        if opcode in ("fmul","fdiv","fadd","fsub"):
            destination,left,right=operands
            a,kind=read(left)
            b,_=read(right)
            function=("f" if kind=="float32" else "d")+"_"+opcode[1:]
            write(destination,f"{function}({a},{b})")
            continue
        if opcode in ('fmadd','fmsub','fnmadd','fnmsub'):
            destination,left,right,addend=operands
            a,kind=read(left);b,_=read(right);c,_=read(addend)
            prefix='f' if kind=='float32' else 'd'
            # ARM's FMSUB computes addend-product. FNMADD negates both
            # product and addend; FNMSUB negates only the addend.
            if opcode in ('fmsub','fnmadd'):a=f'{prefix}_neg({a})'
            if opcode in ('fnmadd','fnmsub'):c=f'{prefix}_neg({c})'
            write(destination,f'{prefix}_operator_fma({a},{b},{c})')
            continue
        if opcode in ("fsqrt","fabs","fneg"):
            destination,source=operands
            argument,kind=read(source)
            if opcode=="fsqrt" and kind=="float32":
                expression=f"f_from_d(d_sqrt(f_to_d({argument})))"
            else:
                expression=f"{'f' if kind=='float32' else 'd'}_{opcode[1:]}({argument})"
            write(destination,expression)
            continue
        if opcode=="fcmp":
            left,right=operands
            a,kind=read(left)
            b=_raw(float(right[1:]),kind) if right.startswith("#") else read(right)[0]
            flags=(a,b,kind)
            continue
        if opcode=="fcsel":
            destination,left,right,code=operands
            write(destination,f"({condition(code)} ? {read(left)[0]} : {read(right)[0]})")
            continue
        if opcode in ("b","bl") and operands[0] in ("_pow","_powf","___powidf2","___powisf2"):
            kind="float32" if operands[0] in ("_powf","___powisf2") else "float"
            a=read("s0" if kind=="float32" else "d0")[0]
            prefix="f" if kind=="float32" else "d"
            if operands[0] in ("_pow","_powf"):
                b=read("s1" if kind=="float32" else "d1")[0]
                expression=f"{prefix}_operator_pow({a},{b})"
                library_power=True
            else:
                exponent=integers[0]&0xffffffff
                if exponent&(1<<31): exponent-=1<<32
                expression=f"{prefix}_operator_powi({a},{exponent})"
                library_powi=True
            write("s0" if kind=="float32" else "d0",expression)
            if opcode=="b":
                break
            continue
        raise UnsupportedModelError(f"Constant power recipe has an unresolved instruction: {line}")
    result=read("s0" if result_kind=="float32" else "d0")[0]
    declaration=','.join(f"{'uint' if kind=='float32' else 'ulong'} {argument}" for argument,kind in parameters)
    header=(f"inline {'uint' if result_kind=='float32' else 'ulong'} {name}"
            f"({declaration}) {{\n"
            +"\n".join(lines)+f"\nreturn {result};\n}}\n")
    return header,library_power,library_powi


@lru_cache(maxsize=256)
def constant_power_policy(input_kind,result_kind,exponent,flags,compiler=None):
    """Freeze the complete scalar compiler recipe for one exponent and width."""
    compiler=tuple(compiler or compiler_command())
    argument_type="float" if input_kind=="float32" else "double"
    result_type="float" if result_kind=="float32" else "double"
    value=float(np.float32(exponent)) if result_kind=="float32" else float(exponent)
    literal=_c_literal(value,result_kind)
    argument="x" if result_kind==input_kind else f"({result_type})x"
    function="powf" if result_kind=="float32" else "pow"
    source=f'#include <cmath>\nextern "C" {result_type} eon_power({argument_type} x){{return {function}({argument},{literal});}}\n'
    name="_eon_constant_power_"+sha256(repr((source,tuple(flags),compiler)).encode()).hexdigest()[:20]
    assembly=_compile_assembly(source,compiler,flags,prefix='eon-floating-power-',filename='power')
    header,library_power,library_powi=_translate(assembly,input_kind,result_kind,name)
    return ConstantPowerPolicy(name,header,source,assembly,compiler,tuple(flags),input_kind,
                               result_kind,library_power,library_powi)


@lru_cache(maxsize=256)
def constant_cpp_operator_policy(expression,constants,result_kind,flags,compiler=None):
    """Freeze CPP's compile-time folding of an immutable scalar operator.

    Standalone Brian writes namespace Constants as C++ constant declarations;
    Cython loads them at runtime. Clang can therefore fold CPP modulo before
    its usual runtime FMA contraction. This reads the same narrow scalar
    assembly recipe as constant powers and executes no input or timestep.
    """
    from brian2.codegen.generators.cpp_generator import CPPCodeGenerator
    compiler=tuple(compiler or compiler_command())
    declarations='\n'.join(f'const {ctype} {name} = {literal};' for name,ctype,literal in constants)
    c_result='float' if result_kind=='float32' else 'double'
    source=('#include <cmath>\n#include <cstdint>\n'+CPPCodeGenerator.universal_support_code+
        f'\nextern "C" {c_result} eon_power(){{\n{declarations}\nreturn {expression};}}\n')
    assembly=_compile_assembly(source,compiler,flags,prefix='eon-constant-cpp-operator-',filename='operator')
    name='_eon_constant_cpp_operator_'+sha256(repr((source,assembly,tuple(flags),compiler)).encode()).hexdigest()[:20]
    header,power,powi=_translate(assembly,'float',result_kind,name,arguments=())
    return ConstantPowerPolicy(name,header,source,assembly,compiler,tuple(flags),'float',result_kind,power,powi)


@dataclass(frozen=True)
class IntegerPowerPolicy:
    name: str
    header: str
    source: str
    assembly: str
    compiler: tuple[str,...]
    flags: tuple[str,...]
    input_kind: str
    result_kind: str
    exponent_dtype: str
    library_power: bool
    library_powi: bool


@lru_cache(maxsize=64)
def integer_power_policy(input_kind,result_kind,exponent_dtype,flags,compiler=None):
    """Observe whether the actual compiler retains the integer exponent.

    Fast Clang can replace powf(x,float(integer)) by __powisf2(x,integer),
    preserving parity that the source's rounded float cast would discard.
    This preparation query executes no input values or simulation timesteps.
    """
    from .compiler import UnsupportedModelError
    compiler=tuple(compiler or compiler_command())
    c_input='float' if input_kind=='float32' else 'double'
    c_result='float' if result_kind=='float32' else 'double'
    c_integer='int' if exponent_dtype=='int32' else 'long'
    call='powf' if result_kind=='float32' else 'pow'
    source=(f'#include <cmath>\nextern "C" {c_result} eon_power_integer({c_input} x,{c_integer} y)'
            f'{{return {call}(({c_result})x,({c_result})y);}}\n')
    assembly=_compile_assembly(source,compiler,flags,prefix='eon-integer-power-',filename='power')
    calls=re.findall(r'^\s*b[l]?\s+(_[A-Za-z0-9_]+)',assembly,re.MULTILINE)
    if len(calls)!=1 or calls[0] not in ('_pow','_powf','___powidf2','___powisf2'):
        raise UnsupportedModelError('Unresolved default-compiler dynamic integer-power recipe: '+repr(calls))
    powi=calls[0].startswith('___powi')
    actual_kind='float32' if calls[0] in ('_powf','___powisf2') else 'float'
    if actual_kind!=result_kind:
        raise UnsupportedModelError('Unexpected dynamic integer-power library result width')
    name='_eon_integer_power_'+sha256(repr((source,assembly,tuple(flags),compiler)).encode()).hexdigest()[:20]
    input_type='uint' if input_kind=='float32' else 'ulong'
    output_type='uint' if result_kind=='float32' else 'ulong'
    metal_integer='int' if exponent_dtype=='int32' else 'long'
    argument='x' if input_kind==result_kind else ('f_to_d(x)' if result_kind=='float' else 'f_from_d(x)')
    prefix='f' if result_kind=='float32' else 'd'
    exponent='int(y)' if powi else f'{prefix}_from_int(long(y))'
    helper=f'{prefix}_operator_powi' if powi else f'{prefix}_operator_pow'
    header=f'inline {output_type} {name}({input_type} x,{metal_integer} y){{return {helper}({argument},{exponent});}}\n'
    return IntegerPowerPolicy(name,header,source,assembly,compiler,tuple(flags),input_kind,result_kind,
                              exponent_dtype,not powi,powi)


@dataclass(frozen=True)
class ConstantBinaryPolicy:
    name: str
    header: str
    source: str
    assembly: str
    compiler: tuple[str,...]
    flags: tuple[str,...]
    input_kind: str
    result_kind: str
    operator: str
    constant: float
    constant_left: bool
    library_power: bool=False
    library_powi: bool=False


def _c_literal(value,kind):
    single=kind=='float32';value=float(np.float32(value)) if single else float(value)
    if np.isnan(value):return '__builtin_nanf("")' if single else '__builtin_nan("")'
    if np.isinf(value):return ('-' if value<0 else '')+('__builtin_inff()' if single else '__builtin_inf()')
    return value.hex()+('f' if single else '')


@lru_cache(maxsize=256)
def constant_binary_policy(input_kind,result_kind,operator,constant,constant_left,flags,compiler=None):
    """Freeze literal arithmetic instructions surrounding a floating operator."""
    compiler=tuple(compiler or compiler_command())
    c_input='float' if input_kind=='float32' else 'double'
    c_result='float' if result_kind=='float32' else 'double'
    variable='x' if input_kind==result_kind else f'({c_result})x'
    literal=_c_literal(constant,result_kind)
    expression=f'{literal}{operator}({variable})' if constant_left else f'({variable}){operator}{literal}'
    source=f'#include <cmath>\nextern "C" {c_result} eon_power({c_input} x){{return {expression};}}\n'
    assembly=_compile_assembly(source,compiler,flags,prefix='eon-constant-arithmetic-',filename='arithmetic')
    name='_eon_constant_arithmetic_'+sha256(repr((source,assembly,tuple(flags),compiler)).encode()).hexdigest()[:20]
    header,power,powi=_translate(assembly,input_kind,result_kind,name)
    assert not power and not powi
    return ConstantBinaryPolicy(name,header,source,assembly,compiler,tuple(flags),input_kind,result_kind,
                                operator,float(constant),constant_left)


@dataclass(frozen=True)
class ProductAddPolicy:
    name: str
    header: str
    source: str
    assembly: str
    compiler: tuple[str,...]
    flags: tuple[str,...]
    input_kinds: tuple[str,...]
    result_kind: str


@lru_cache(maxsize=128)
def product_add_policy(input_kinds,result_kind,operation,product_left,flags,compiler=None):
    """Capture one enclosing product/addition, not a second model compiler.

    The new operator's already-rounded result is a typed operand. This small
    recipe records the actual target's contraction and sign instructions;
    whole-stage reassociation remains the shared optimized-graph interface.
    """
    compiler=tuple(compiler or compiler_command())
    names=('a','b','c')
    types=['float' if kind=='float32' else 'double' for kind in input_kinds]
    expression=f'(a*b){operation}c' if product_left else f'c{operation}(a*b)'
    c_result='float' if result_kind=='float32' else 'double'
    source=('#include <cmath>\nextern "C" '+c_result+' eon_power('+','.join(
        f'{kind} {name}' for name,kind in zip(names,types))+'){return '+expression+';}\n')
    assembly=_compile_assembly(source,compiler,flags,prefix='eon-product-add-',filename='context')
    name='_eon_product_add_'+sha256(repr((source,assembly,tuple(flags),compiler)).encode()).hexdigest()[:20]
    header,_,_=_translate(assembly,input_kinds[0],result_kind,name,arguments=tuple(zip(names,input_kinds)))
    return ProductAddPolicy(name,header,source,assembly,compiler,tuple(flags),tuple(input_kinds),result_kind)


def render_numpy_operator(node,left_code,left_kind,right_code,right_kind,result_kind,*,
                          values=None,right_scalar=None,original_right_code=None,original_right_kind=None):
    """Reuse the authoritative operator policy from a width-aware renderer.

    Codes are raw IEEE uint32/uint64 words or integer values; kinds are
    ``float32``, ``float`` (binary64), ``int`` or ``bool``. The caller owns
    NumPy promotion and supplies the chosen floating result kind. ``values``
    contains immutable preparation scalars, never per-neuron data.

    ``right_scalar`` distinguishes ndarray ** scalar shortcuts from an array
    exponent, including a scalar whose value changes at runtime. If operand
    rounding has already changed a scalar, pass its original code and kind
    separately: only shortcut predicates use that original value. Return the
    emitted raw-word expression and compiler used_math dependency tags. This
    adapter delegates to _Renderer; it is not another operator engine.
    """
    from .compiler import _Renderer,_Expression
    if result_kind not in ('float','float32'):
        raise ValueError('A floating NumPy operator needs an explicit result width')
    if (original_right_code is None)!=(original_right_kind is None):
        raise ValueError('Original scalar code and kind must be supplied together')
    original=(_Expression(original_right_code,original_right_kind)
              if original_right_code is not None else None)
    used=set()
    renderer=_Renderer({},dict(values or {}),used_math=used,reference_target='numpy')
    result=renderer.floating_operator(node,_Expression(left_code,left_kind),
        _Expression(right_code,right_kind),result_kind=result_kind,numpy_exponent_scalar=right_scalar,
        numpy_original_exponent=original)
    return result.code,tuple(sorted(used))
