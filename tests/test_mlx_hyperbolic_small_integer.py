"""Actual NumPy integer8/16 function loops and native typed components."""
import uuid

import numpy as np
import pytest


@pytest.mark.metal
@pytest.mark.parametrize('dtype',['int8','int16','uint8','uint16'])
@pytest.mark.parametrize('function',['log10','sinh','cosh','tanh'])
def test_actual_brian_small_integer_hyperbolic_loop(monkeypatch,tmp_path,dtype,function):
    import brian2 as b
    from mlx_backend.numpy_half import Renderer,half_precision
    from mlx_backend import binary64,binary32,math64,binary16
    from mlx_backend.hyperbolic32 import metal_hyperbolic32_header
    mx=pytest.importorskip('mlx.core')
    b.BrianLogger.log_level_error()
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float64)
    monkeypatch.setitem(b.prefs,'core.default_integer_dtype',np.dtype(dtype).type)
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',False)
    symbol='hyper_int_'+uuid.uuid4().hex[:8]
    expression=function+'('+symbol+')+.1'
    info=np.iinfo(dtype)
    if np.dtype(dtype).kind=='u':
        # Unsigned selection is a distinct dtype rule; use its signed-boundary
        # and largest-value controls without repeating the full signed screen.
        points=[0,1,2,3,31,127,128,255]
        if info.bits==16:points += [256,257,32767,32768,65534,65535]
        inputs=np.array(points,dtype=dtype)
    else:
        inputs=np.arange(info.min,info.max+1,dtype=np.int64).astype(dtype)
    neurons=b.NeuronGroup(len(inputs),symbol+':integer\nsample:1',
        threshold='True',reset='sample='+expression,dt=.1*b.ms)
    setattr(neurons,symbol,inputs)
    assert neurons.variables[symbol].dtype==np.dtype(dtype)
    b.Network(neurons).run(.1*b.ms,namespace={})
    expected=np.asarray(neurons.sample_[:]).copy()
    (tmp_path/'actual-brian-reset.txt').write_text(str(neurons.resetter['spike'].codeobj.code))
    np.save(tmp_path/'actual-brian-output.npy',expected)
    scalar_name='_eon_hyper32_'+function+('_scalar' if function=='tanh' else '')
    dependency='tanh_scalar_f32' if function=='tanh' else function+'_f32'
    renderer=Renderer({symbol:np.dtype(dtype)},{},set(),set(),'reset',
        half_ufuncs={function:(dependency,scalar_name)})
    call=renderer.expression(function+'('+symbol+')')
    assert call.narrow is True and half_precision(call)==(np.dtype(dtype).itemsize==1)
    result=renderer.expression(expression)
    helpers={function} if function!='tanh' else {'tanh_scalar','tanh_numpy'}
    header=binary64.MSL_HEADER+binary32.MSL_HEADER+math64.MSL_MATH_HEADER
    header+=binary16.MSL_HEADER+binary16.MSL_F32_HEADER+metal_hyperbolic32_header(helpers,'numpy')
    source='uint i=thread_position_in_grid.x;if(i<N){long '+symbol+'=values[i];out[i]='+result.code+';}'
    (tmp_path/'native-expression.metal').write_text(source)
    kernel=mx.fast.metal_kernel(name='small_integer_hyperbolic_'+function+'_'+dtype,
        input_names=['values'],output_names=['out'],header=header,source=source,
        compile_options={'math_mode':'safe'})
    (actual,)=kernel(inputs=[mx.array(inputs)],template=[('N',len(inputs))],grid=(len(inputs),1,1),
        threadgroup=(128,1,1),output_shapes=[inputs.shape],output_dtypes=[mx.uint64])
    mx.eval(actual)
    np.testing.assert_array_equal(np.asarray(actual),expected.view(np.uint64))
