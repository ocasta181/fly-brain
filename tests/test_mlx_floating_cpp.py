"""Actual Brian CPP standalone sources versus typed native operator lowering."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

import numpy as np
import pytest
from native_execution import require_metal

from mlx_backend import binary32,binary64,math64
from mlx_backend.compiler import compile_model
from mlx_backend.model import ModelAnalysis,ModelVariable,resolve_model
from original_api_reference import cpu_runtime,load_original_model

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.metal


@pytest.mark.parametrize('dtype,case',[('float64','dynamic'),('float32','dynamic'),('float64','mixed'),('float32','mixed'),
                                    ('float64','context'),('float32','context'),('float32','integer-base'),('float64','constant'),
                                    ('float32','all-float32-context'),('float32','uint64-base'),
                                    ('float32','all-float32-uint64-base'),('float64','two-product'),
                                    ('float64','fixed-base'),('float32','all-float32-fixed-base'),
                                    ('float64','timed'),('float32','all-float32-timed')],
                         ids=['float64','float32','mixed-float64','mixed-float32','context-float64','context-float32','integer-base-float32','constant-float64','all-float32-context','uint64-base-float32','all-float32-uint64-base','two-product-float64','fixed-base-float64','all-float32-fixed-base','timed-float64','all-float32-timed'])
def test_actual_cpp_dynamic_operator_boundary_matrix(dtype,case,tmp_path):
    mx = require_metal('Metal required')
    directory=ROOT/'executions/floating-operators/cpp-boundary'/(dtype+'-'+case)
    # The standalone process is the oracle. No native/backend modules are
    # imported by its reference driver and no reference values are synthesized.
    all_float32=case.startswith('all-float32-')
    reference_case=('timed' if case.endswith('timed') else
                    'fixed-base' if case.endswith('fixed-base') else
                    'integer-base' if case.endswith('uint64-base') else
                    'context' if all_float32 else case)
    reference=subprocess.run([sys.executable,str(ROOT/'tests/probe_floating_cpp_reference.py'),
        '--dtype',dtype,'--case',reference_case,'--output',str(directory),
        *(['--integer-dtype','uint64'] if case.endswith('uint64-base') else []),
        *(['--all-float32'] if all_float32 else [])],capture_output=True,text=True,timeout=55)
    assert reference.returncode==0,reference.stdout+'\n'+reference.stderr
    metadata=json.loads((directory/'metadata.json').read_text())
    assert metadata['target']=='CPPStandaloneCodeObject'
    params=dict(load_original_model().default_params)
    params.update(eqs=metadata['eqs'],eq_th=metadata['eq_th'],eq_rst=metadata['eq_rst'],t_rfc=0*__import__('brian2').ms)
    params.update(metadata.get('operator_namespace',{}))
    if 'timed_array' in metadata:
        import brian2 as b
        array=metadata['timed_array']
        params['drive']=b.TimedArray(np.asarray(array['values']),dt=array['dt']*b.second)
    for name,dtype_name in metadata.get('operator_namespace_dtypes',{}).items():
        params[name]=np.dtype(dtype_name).type(params[name])
    variables={name:ModelVariable(**{**row,'dimensions':tuple(row['dimensions']),'flags':tuple(row['flags'])})
        for name,row in metadata['variables'].items()}
    analysis=ModelAnalysis(*metadata['abstract'],MappingProxyType(variables),metadata['brian_version'],metadata['temporary_dtype'],'cpp')
    with cpu_runtime():
        model=resolve_model(params,state_dtype=metadata['state_dtype'],cpu_target='cpp')
        model=replace(model,analysis=analysis,reference_codegen_target='cpp',
            reference_compile_args=tuple(metadata['flags']),
            signature=sha256((model.signature+repr(analysis)+repr(metadata['flags'])+'cpp').encode()).hexdigest())
        if 'timed_array' in metadata:
            policy=model.integer_policy.execution(metadata['neurons'],1)
            model=replace(model,integer_policy=policy,
                signature=sha256((model.signature+repr(policy)).encode()).hexdigest())
        if all_float32:
            from mlx_backend.clang32 import compiler_identity
            model=replace(model,float32_codegen='cpp',float32_compile_args=tuple(metadata['flags']),
                float32_compiler_identity=compiler_identity())
        program=compile_model(model)
    if case=='all-float32-fixed-base':
        assert 'f_operator_exp2(' in program.reset_vector_code
    names=tuple(name for name,value in program.variables.items() if value.persistent)
    initial=np.load(directory/'initial.npz');expected=np.load(directory/'final.npz')
    def raw(values):
        if values.dtype==np.float64:return values.view(np.uint64)
        if values.dtype==np.float32:return values.view(np.uint32).astype(np.uint64)
        return values.astype(np.uint64)
    data=np.stack([raw(initial[name]) for name in names])
    loads='\n'.join(f'{program.variables[name].metal_type} {program.variables[name].code_name} = ({program.variables[name].metal_type})raw[{slot}ul*N+i];'
        for slot,name in enumerate(names))
    stores='\n'.join(f'out[{slot}ul*N+i]=ulong({program.variables[name].code_name});' for slot,name in enumerate(names))
    source=f'''uint row=thread_position_in_grid.x;if(row<N){{
      const int i=int(row);const ulong t=0ul,dt={binary64.literal(model.dt_s)};const long t_in_timesteps=0;
      {loads}
      {program.scalar_code}
      {program.state_code}
      bool _cond=false;{program.threshold_code}
      if(_cond){{lastspike={'f_from_d(t)' if all_float32 else 't'};not_refractory=false;}}
      {program.reset_scalar_code}
      if(_cond){{{program.reset_vector_code}}}
      {stores}
      out[{len(names)}ul*N+i]=ulong(_cond);
    }}'''
    (directory/'native.metal').write_text(source+'\n'+program.header_code)
    arrays=program.input_arrays
    kernel=mx.fast.metal_kernel(name='actual_cpp_floating_boundary_'+dtype+'_'+case.replace('-','_'),input_names=['raw',*arrays],output_names=['out'],
        header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+binary32.MSL_HEADER+program.header_code,
        source=source,compile_options={'math_mode':'safe'})
    observed,=kernel(inputs=[mx.array(data.reshape(-1)),
        *(mx.array(array.values.view(np.uint64).reshape(-1)) for array in arrays.values())],template=[('N',metadata['neurons'])],
        grid=(metadata['neurons'],1,1),threadgroup=(128,1,1),
        output_shapes=[(len(names)+1,metadata['neurons'])],output_dtypes=[mx.uint64])
    mx.eval(observed);observed=np.asarray(observed)
    for row,name in enumerate(names):
        expected_raw=raw(expected[name]);mismatch=np.flatnonzero(observed[row]!=expected_raw)
        if len(mismatch):
            index=int(mismatch[0])
            pytest.fail(f'CPP {dtype} {name}[{index}]: native={int(observed[row,index]):#x}, CPU={int(expected_raw[index]):#x}; '
                f'x={int(raw(initial["x"])[index]):#x}, y={int(raw(initial["y"])[index]):#x}; {len(mismatch)} differences')
    np.testing.assert_array_equal(np.flatnonzero(observed[-1]),expected['spike_i'])
    assert metadata['clock_timestep']==1
