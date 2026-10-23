"""Actual default CPU compiler constant-power recipes versus native Metal."""
import ctypes
from pathlib import Path
import subprocess

import numpy as np
import pytest
from native_execution import require_metal

from mlx_backend import binary32,binary64,math64,operator32,operator64
from mlx_backend.operator_policy import constant_power_policy
from test_mlx_floating_operators import power_cases

pytestmark = pytest.mark.metal


@pytest.mark.parametrize("input_kind,result_kind",[("float","float"),("float32","float32"),("float32","float")])
def test_actual_compiler_constant_power_order(input_kind,result_kind,tmp_path):
    from brian2.codegen.cpp_prefs import get_compiler_and_args
    mx = require_metal('Metal required')
    flags=tuple(get_compiler_and_args()[1])
    exponents=(-32.,-2.5,-.5,0.,.5,1.,3.,7.,1/64,.1,float(2**31-1),float(2**52))
    policies=[constant_power_policy(input_kind,result_kind,value,flags) for value in exponents]
    input_dtype=np.float32 if input_kind=="float32" else np.float64
    result_dtype=np.float32 if result_kind=="float32" else np.float64
    raw_dtype=np.uint32 if input_kind=="float32" else np.uint64
    c_input="float" if input_kind=="float32" else "double"
    c_result="float" if result_kind=="float32" else "double"
    raw_input="uint32_t" if input_kind=="float32" else "uint64_t"
    raw_result="uint32_t" if result_kind=="float32" else "uint64_t"
    a,_=power_cases(input_dtype)
    a=np.ascontiguousarray(np.unique(a.view(raw_dtype)))
    source='''#include <cmath>
#include <cstdint>
#include <cstring>
template<class To,class From>To bits(From a){To out;std::memcpy(&out,&a,sizeof(out));return out;}
'''
    for number,policy in enumerate(policies):
        function=policy.source.replace('extern "C"','extern "C" __attribute__((noinline))').replace("eon_power",f"reference_{number}")
        source+=function
        source+=(f'extern "C" void run_{number}(const {raw_input} *x,uint64_t *out,int n){{'
                 f'for(int j=0;j<n;++j)out[j]=bits<{raw_result}>(reference_{number}(bits<{c_input}>(x[j])));}}\n')
    (tmp_path/"reference.cpp").write_text(source)
    subprocess.run(["clang++",*flags,"-shared","-fPIC",str(tmp_path/"reference.cpp"),
                    "-o",str(tmp_path/"reference.so")],check=True,capture_output=True,text=True,timeout=30)
    cpu=ctypes.CDLL(str(tmp_path/"reference.so"))
    expected=np.empty((len(policies),len(a)),np.uint64)
    pointer=np.ctypeslib.ndpointer(dtype=raw_dtype,flags="C_CONTIGUOUS")
    result_pointer=np.ctypeslib.ndpointer(dtype=np.uint64,flags="C_CONTIGUOUS")
    for number in range(len(policies)):
        function=getattr(cpu,f"run_{number}")
        function.argtypes=[pointer,result_pointer,ctypes.c_int]
        function.restype=None
        function(a,expected[number],len(a))
    header=(binary64.MSL_HEADER+math64.MSL_MATH_HEADER+binary32.MSL_HEADER
            +operator64.MSL_POWER_HEADER+operator32.MSL_POWER_HEADER
            +operator64.MSL_INTEGER_EXPONENT_HEADER+operator32.MSL_INTEGER_EXPONENT_HEADER
            +"\n".join(policy.header for policy in policies))
    branches="\n".join(f"{'if' if number==0 else 'else if'}(operation=={number}u) out[i]={policy.name}"
        f"({'uint' if input_kind=='float32' else 'ulong'}(a[column]));" for number,policy in enumerate(policies))
    kernel=mx.fast.metal_kernel(name="constant_operator_policy_"+input_kind+"_"+result_kind,
        input_names=["a"],output_names=["out"],header=header,
        source="uint i=thread_position_in_grid.x;if(i<COUNT){uint operation=i/COLUMNS,column=i%COLUMNS;"+branches+"}",
        compile_options={"math_mode":"safe"})
    observed,=kernel(inputs=[mx.array(a.astype(np.uint64))],template=[("COUNT",expected.size),("COLUMNS",len(a))],
        grid=(expected.size,1,1),threadgroup=(128,1,1),output_shapes=[expected.shape],output_dtypes=[mx.uint64])
    mx.eval(observed)
    actual=np.asarray(observed)
    mismatch=np.argwhere(actual!=expected)
    if len(mismatch):
        operation,column=(int(v) for v in mismatch[0])
        (tmp_path/"policy.s").write_text(policies[operation].assembly)
        pytest.fail(f"{input_kind}->{result_kind} constant power{exponents[operation]} "
            f"input={int(a[column]):#x}: Metal={int(actual[operation,column]):#x}, "
            f"CPU={int(expected[operation,column]):#x};{len(mismatch)}different cells")


@pytest.mark.parametrize('input_kind,result_kind',[('float','float'),('float32','float32'),('float32','float')])
@pytest.mark.parametrize('integer_dtype',['int32','int64'])
def test_actual_compiler_dynamic_integer_power(input_kind,result_kind,integer_dtype,tmp_path):
    from brian2.codegen.cpp_prefs import get_compiler_and_args
    from mlx_backend.operator_policy import integer_power_policy
    mx=pytest.importorskip('mlx.core')
    flags=tuple(get_compiler_and_args()[1])
    policy=integer_power_policy(input_kind,result_kind,integer_dtype,flags)
    dtype=np.float32 if input_kind=='float32' else np.float64
    rawtype=np.uint32 if dtype==np.float32 else np.uint64
    a=np.ascontiguousarray(power_cases(dtype)[0].view(rawtype))
    values=[-2**31,-2**24-1,-3,-2,-1,0,1,2,3,2**24+1,2**31-1]
    if integer_dtype=='int64':values += [-2**63,2**63-1,-2**53-1,2**53+1,2**32+1]
    b=np.resize(np.array(values,integer_dtype),len(a)).astype(np.int64)
    c_input='float' if input_kind=='float32' else 'double'
    c_result='float' if result_kind=='float32' else 'double'
    c_raw='uint32_t' if input_kind=='float32' else 'uint64_t'
    c_output='uint32_t' if result_kind=='float32' else 'uint64_t'
    source='''#include <cmath>
#include <cstdint>
#include <cstring>
template<class To,class From>To bits(From x){To out;std::memcpy(&out,&x,sizeof(out));return out;}
'''+policy.source.replace('extern "C"','extern "C" __attribute__((noinline))')
    source+=f'''extern "C" void run(const {c_raw} *a,const int64_t *b,uint64_t *out,int n){{
      for(int i=0;i<n;++i)out[i]=bits<{c_output}>(eon_power_integer(bits<{c_input}>(a[i]),b[i]));}}
'''
    (tmp_path/'reference.cpp').write_text(source)
    subprocess.run(['clang++',*flags,'-shared','-fPIC',str(tmp_path/'reference.cpp'),'-o',str(tmp_path/'reference.so')],
        check=True,capture_output=True,text=True,timeout=30)
    cpu=ctypes.CDLL(str(tmp_path/'reference.so')).run
    cpu.argtypes=[np.ctypeslib.ndpointer(dtype=rawtype,flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS'),ctypes.c_int]
    cpu.restype=None;expected=np.empty(len(a),np.uint64);cpu(a,b,expected,len(a))
    header=(binary64.MSL_HEADER+math64.MSL_MATH_HEADER+binary32.MSL_HEADER+
        operator64.MSL_POWER_HEADER+operator32.MSL_POWER_HEADER+
        operator64.MSL_INTEGER_EXPONENT_HEADER+operator32.MSL_INTEGER_EXPONENT_HEADER+policy.header)
    kernel=mx.fast.metal_kernel(name='dynamic_integer_power_'+input_kind+'_'+result_kind+'_'+integer_dtype,
        input_names=['a','b'],output_names=['out'],header=header,
        source=f'uint i=thread_position_in_grid.x;if(i<COUNT)out[i]={policy.name}({"uint" if input_kind=="float32" else "ulong"}(a[i]),b[i]);',
        compile_options={'math_mode':'safe'})
    observed,=kernel(inputs=[mx.array(a),mx.array(b)],template=[('COUNT',len(a))],grid=(len(a),1,1),
        threadgroup=(128,1,1),output_shapes=[expected.shape],output_dtypes=[mx.uint64])
    mx.eval(observed);observed=np.asarray(observed)
    mismatch=np.flatnonzero(observed!=expected)
    if len(mismatch):
        i=int(mismatch[0]);pytest.fail(f'{input_kind}->{result_kind} exponent {integer_dtype}: '
            f'x={int(a[i]):#x}, y={int(b[i])}, native={int(observed[i]):#x}, CPU={int(expected[i]):#x}; '
            f'{len(mismatch)} differences')


@pytest.mark.parametrize('kinds',[('float','float','float'),('float32','float32','float32'),
                                  ('float32','float32','float'),('float','float32','float32')])
def test_actual_compiler_product_add_sub_order(kinds,tmp_path):
    from brian2.codegen.cpp_prefs import get_compiler_and_args
    from mlx_backend.operator_policy import product_add_policy
    mx=pytest.importorskip('mlx.core');flags=tuple(get_compiler_and_args()[1])
    result_kind='float' if 'float' in kinds else 'float32'
    recipes=[product_add_policy(kinds,result_kind,op,left,flags) for op,left in
             (('+',True),('+',False),('-',True),('-',False))]
    rng=np.random.default_rng(19213)
    arrays=[]
    for number,kind in enumerate(kinds):
        dtype=np.float32 if kind=='float32' else np.float64
        rawtype=np.uint32 if kind=='float32' else np.uint64
        values=rng.integers(0,1<<(32 if kind=='float32' else 64),1024,dtype=rawtype)
        special=np.array([0.,-0.,1.,-1.,np.inf,-np.inf,np.nan,np.finfo(dtype).tiny,
            np.nextafter(dtype(0),dtype(1))],dtype).view(rawtype)
        values[:len(special)]=np.roll(special,number)
        arrays.append(np.ascontiguousarray(values))
    c_input=['float' if kind=='float32' else 'double' for kind in kinds]
    c_raw=['uint32_t' if kind=='float32' else 'uint64_t' for kind in kinds]
    output_raw='uint32_t' if result_kind=='float32' else 'uint64_t'
    source='''#include <cmath>
#include <cstdint>
#include <cstring>
template<class To,class From>To bits(From x){To out;std::memcpy(&out,&x,sizeof(out));return out;}
'''
    for number,recipe in enumerate(recipes):
        source+=recipe.source.replace('extern "C"','extern "C" __attribute__((noinline))').replace('eon_power',f'reference_{number}')
        source+=f'extern "C" void run_{number}(const {c_raw[0]} *a,const {c_raw[1]} *b,const {c_raw[2]} *c,uint64_t *out,int n){{'
        source+=f'for(int i=0;i<n;++i)out[i]=bits<{output_raw}>(reference_{number}('+','.join(
            f'bits<{kind}>({name}[i])' for kind,name in zip(c_input,('a','b','c')))+'));}\n'
    (tmp_path/'reference.cpp').write_text(source)
    subprocess.run(['clang++',*flags,'-shared','-fPIC',str(tmp_path/'reference.cpp'),'-o',str(tmp_path/'reference.so')],
        check=True,capture_output=True,text=True,timeout=30)
    cpu=ctypes.CDLL(str(tmp_path/'reference.so'));expected=np.empty((4,len(arrays[0])),np.uint64)
    for number in range(4):
        function=getattr(cpu,f'run_{number}');function.restype=None
        function.argtypes=[*[np.ctypeslib.ndpointer(dtype=a.dtype,flags='C_CONTIGUOUS') for a in arrays],
            np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS'),ctypes.c_int]
        function(*arrays,expected[number],len(arrays[0]))
    header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+binary32.MSL_HEADER+operator64.MSL_REMAINDER_HEADER+operator32.MSL_REMAINDER_HEADER
    header+='\n'.join(recipe.header for recipe in recipes)
    cases='\n'.join(f'{"if" if number==0 else "else if"}(operation=={number}u)out[i]={recipe.name}(a[column],b[column],c[column]);'
        for number,recipe in enumerate(recipes))
    kernel=mx.fast.metal_kernel(name='product_context_'+ '_'.join(kinds),input_names=['a','b','c'],output_names=['out'],
        header=header,source='uint i=thread_position_in_grid.x;if(i<COUNT){uint operation=i/COLUMNS,column=i%COLUMNS;'+cases+'}',
        compile_options={'math_mode':'safe'})
    result,=kernel(inputs=[mx.array(a) for a in arrays],template=[('COUNT',expected.size),('COLUMNS',expected.shape[1])],
        grid=(expected.size,1,1),threadgroup=(128,1,1),output_shapes=[expected.shape],output_dtypes=[mx.uint64])
    mx.eval(result);np.testing.assert_array_equal(np.asarray(result),expected)


@pytest.mark.parametrize('left_kind,right_kind,result_kind,right_scalar,large',[
    ('float32','float32','float32',True,False),('float32','float32','float32',False,False),
    ('float','float','float',True,False),('float','float','float',False,False),
    ('float32','float','float',True,True),('float32','float','float32',True,False),
    ('float32','float','float',False,False)])
def test_shared_numpy_operator_adapter_preserves_width_and_scalar_policy(
        left_kind,right_kind,result_kind,right_scalar,large):
    import ast
    from mlx_backend.operator_policy import render_numpy_operator
    mx=pytest.importorskip('mlx.core')
    dtype=np.float32 if left_kind=='float32' else np.float64
    right_dtype=np.float32 if right_kind=='float32' else np.float64
    result_dtype=np.float32 if result_kind=='float32' else np.float64
    rawtype=np.uint32 if dtype==np.float32 else np.uint64
    rhs_rawtype=np.uint32 if right_dtype==np.float32 else np.uint64
    nan_raw=[0x7fc00123,0xff800456] if dtype==np.float32 else [0x7ff8000000000123,0xfff0000000000456]
    base=np.r_[np.array([0.,-0.,.3,-.3,3.,-3.,np.finfo(dtype).tiny,
        np.nextafter(dtype(0),dtype(1)),np.finfo(dtype).max,np.inf,-np.inf],dtype),
        np.array(nan_raw,rawtype).view(dtype)]
    scalars=np.array([1e100,-1e100] if large else [-1.,0.,.5,1.,2.,.3,1.00000001],right_dtype)
    operations=[('**',lambda a,z:a**z),('%',lambda a,z:a%z),('//',lambda a,z:a//z)]
    expected=[]
    with np.errstate(all='ignore'):
        for _,operation in operations:
            rows=[]
            for scalar in scalars:
                right=scalar if right_scalar else np.full(base.shape,scalar,right_dtype)
                value=operation(base,right)
                assert value.dtype==result_dtype
                rows.append(value.view(np.uint32 if result_kind=='float32' else np.uint64).astype(np.uint64))
            expected.append(rows)
    expected=np.array(expected,np.uint64)
    bodies=[];dependencies=set()
    for number,(symbol,_) in enumerate(operations):
        code,used=render_numpy_operator(ast.parse('base'+symbol+'scalar',mode='eval').body,
            'base[row]',left_kind,'scalars[column]',right_kind,result_kind,right_scalar=right_scalar,values=None if right_scalar else {"scalar":scalars})
        dependencies.update(used)
        bodies.append(f'if(operation=={number}u)out[gid]=ulong({code});')
    assert dependencies
    header=(binary64.MSL_HEADER+math64.MSL_MATH_HEADER+binary32.MSL_HEADER+operator64.MSL_POWER_HEADER+
        operator32.MSL_POWER_HEADER+operator64.MSL_REMAINDER_HEADER+operator32.MSL_REMAINDER_HEADER)
    kernel=mx.fast.metal_kernel(name='shared_numpy_operator_'+left_kind+'_'+right_kind+'_'+result_kind+'_'+str(int(right_scalar))+'_'+str(int(large)),
        input_names=['base','scalars'],output_names=['out'],header=header,
        source='uint gid=thread_position_in_grid.x;if(gid<COUNT){uint row=gid%ROWS,column=(gid/ROWS)%COLS,operation=gid/(ROWS*COLS);'+''.join(bodies)+'}',
        compile_options={'math_mode':'safe'})
    observed,=kernel(inputs=[mx.array(base.view(rawtype)),mx.array(scalars.view(rhs_rawtype))],
        template=[('COUNT',expected.size),('ROWS',len(base)),('COLS',len(scalars))],
        grid=(expected.size,1,1),threadgroup=(128,1,1),output_shapes=[expected.shape],output_dtypes=[mx.uint64])
    mx.eval(observed);np.testing.assert_array_equal(np.asarray(observed),expected)


def test_shared_numpy_operator_adapter_uses_original_half_scalar_predicate():
    import ast
    from mlx_backend.operator_policy import render_numpy_operator
    mx=pytest.importorskip('mlx.core')
    half_words=np.arange(65536,dtype=np.uint16)
    base=half_words.view(np.float16)
    scalars=np.array([2.,2.0001,1.,1.0001,.5,.5001,-1.,-1.0001],np.float64)
    with np.errstate(all='ignore'):
        base32=base.astype(np.float32)
        rounded=scalars.astype(np.float16).astype(np.float32)
        expected=np.stack([(base**scalar).view(np.uint16) for scalar in scalars])
    code,used=render_numpy_operator(ast.parse('base**scalar',mode='eval').body,
        'base[row]','float32','rounded[column]','float32','float32',right_scalar=True,
        original_right_code='original[column]',original_right_kind='float')
    assert 'd_eq(original[column]' in code
    assert 'f_operator_pow(base[row],rounded[column])' in code
    assert 'operator_pow_float32' in used
    header=(binary64.MSL_HEADER+math64.MSL_MATH_HEADER+binary32.MSL_HEADER+
        operator64.MSL_POWER_HEADER+operator32.MSL_POWER_HEADER)
    kernel=mx.fast.metal_kernel(name='shared_numpy_operator_half_original_scalar',
        input_names=['base','rounded','original'],output_names=['out'],header=header,
        source='uint gid=thread_position_in_grid.x;if(gid<COUNT){uint row=gid%65536u,column=gid/65536u;out[gid]='+code+';}',
        compile_options={'math_mode':'safe'})
    observed,=kernel(inputs=[mx.array(base32.view(np.uint32)),mx.array(rounded.view(np.uint32)),
        mx.array(scalars.view(np.uint64))],template=[('COUNT',expected.size)],
        grid=(expected.size,1,1),threadgroup=(128,1,1),output_shapes=[expected.shape],output_dtypes=[mx.uint32])
    mx.eval(observed)
    with np.errstate(all='ignore'):
        actual=np.asarray(observed).view(np.float32).astype(np.float16).view(np.uint16)
    np.testing.assert_array_equal(actual,expected)
