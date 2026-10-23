"""Actual NumPy integer wrapping before the four floating-function loops."""
import uuid

import numpy as np
import pytest
from native_execution import require_metal


@pytest.mark.metal
@pytest.mark.parametrize('dtype',['int8','uint8','int16','uint16'])
@pytest.mark.parametrize('operation',['-ARG','ARG+1'])
def test_actual_brian_wrapped_integer_hyperbolic_arguments(monkeypatch,tmp_path,dtype,operation):
    import brian2 as b
    from mlx_backend.numpy_half import Renderer,half_precision
    from mlx_backend import binary64,binary32,math64,binary16
    from mlx_backend.hyperbolic32 import metal_hyperbolic32_header
    mx = require_metal('Metal required')
    b.BrianLogger.log_level_error()
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float64)
    monkeypatch.setitem(b.prefs,'core.default_integer_dtype',np.dtype(dtype).type)
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',False)
    symbol='hyper_operand_'+uuid.uuid4().hex[:8]
    functions=('log10','sinh','cosh','tanh')
    info=np.iinfo(dtype)
    points=[info.min,info.min+1,-1,0,1,info.max-1,info.max]
    inputs=np.array(sorted({x for x in points if info.min<=x<=info.max}),dtype=dtype)
    argument=operation.replace('ARG',symbol)
    expressions=[function+'('+argument+')+.1' for function in functions]
    group=b.NeuronGroup(len(inputs),symbol+':integer\n'+'\n'.join('s'+str(j)+':1' for j in range(4)),
        threshold='True',reset='\n'.join('s'+str(j)+'='+expr for j,expr in enumerate(expressions)),dt=.1*b.ms)
    setattr(group,symbol,inputs)
    b.Network(group).run(.1*b.ms,namespace={})
    expected=np.stack([np.asarray(getattr(group,'s'+str(j)+'_')[:]).copy() for j in range(4)])
    (tmp_path/'actual-brian-reset.txt').write_text(str(group.resetter['spike'].codeobj.code))
    np.save(tmp_path/'actual-brian-output.npy',expected)
    registry={name:(('tanh_scalar' if name=='tanh' else name)+'_f32',
        '_eon_hyper32_'+name+('_scalar' if name=='tanh' else '')) for name in functions}
    renderer=Renderer({symbol:np.dtype(dtype)},{},set(),set(),'reset',half_ufuncs=registry)
    operand=renderer.expression(argument)
    assert operand.integer_bits==info.bits
    assert operand.integer_unsigned==(np.dtype(dtype).kind=='u')
    calls=[renderer.expression(function+'('+argument+')') for function in functions]
    assert all(call.narrow is True and half_precision(call)==(info.bits==8) for call in calls)
    expressions=[renderer.expression(expr).code for expr in expressions]
    header=binary64.MSL_HEADER+binary32.MSL_HEADER+math64.MSL_MATH_HEADER
    header+=binary16.MSL_HEADER+binary16.MSL_F32_HEADER
    header+=metal_hyperbolic32_header({'log10','sinh','cosh','tanh_scalar','tanh_numpy'},'numpy')
    source='uint i=thread_position_in_grid.x;if(i<N){long '+symbol+'=values[i];'
    source+=''.join('out['+str(j)+'*N+i]='+expr+';' for j,expr in enumerate(expressions))+'}'
    (tmp_path/'native-expression.metal').write_text(source)
    kernel=mx.fast.metal_kernel(name='integer_argument_'+dtype+'_'+('neg' if operation[0]=='-' else 'add'),
        input_names=['values'],output_names=['out'],header=header,source=source,
        compile_options={'math_mode':'safe'})
    (actual,)=kernel(inputs=[mx.array(inputs)],template=[('N',len(inputs))],grid=(len(inputs),1,1),
        threadgroup=(32,1,1),output_shapes=[expected.shape],output_dtypes=[mx.uint64])
    mx.eval(actual)
    np.testing.assert_array_equal(np.asarray(actual),expected.view(np.uint64))
