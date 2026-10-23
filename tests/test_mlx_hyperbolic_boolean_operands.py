"""Actual Boolean/integer NumPy ufunc operands before owned math calls."""
import uuid

import numpy as np
import pytest
from native_execution import require_metal


@pytest.mark.metal
@pytest.mark.parametrize('argument',[
    'ARG+FLAG','FLAG+OTHER','FLAG*OTHER','(FLAG*OTHER)+ARG'])
def test_actual_brian_boolean_hyperbolic_operands(monkeypatch,tmp_path,argument):
    import brian2 as b
    from mlx_backend.numpy_half import Renderer
    from mlx_backend import binary64,binary32,math64,binary16
    from mlx_backend.hyperbolic32 import metal_hyperbolic32_header
    mx = require_metal('Metal required')
    b.BrianLogger.log_level_error()
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float64)
    monkeypatch.setitem(b.prefs,'core.default_integer_dtype',np.uint8)
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',False)
    prefix='hyper_bool_operand_'+uuid.uuid4().hex[:8]
    names={name:prefix+'_'+name.lower() for name in ('ARG','FLAG','OTHER')}
    expression=argument
    for name,replacement in names.items():expression=expression.replace(name,replacement)
    values=np.array([0,1,255,128],np.uint8)
    flags=np.array([False,False,True,True])
    others=np.array([False,True,False,True])
    functions=('log10','sinh','cosh','tanh')
    equations=names['ARG']+':integer\n'+names['FLAG']+':boolean\n'+names['OTHER']+':boolean\n'
    equations+='\n'.join('s'+str(j)+':1' for j in range(4))
    expressions=[name+'('+expression+')+.1' for name in functions]
    group=b.NeuronGroup(4,equations,threshold='True',
        reset='\n'.join('s'+str(j)+'='+expr for j,expr in enumerate(expressions)),dt=.1*b.ms)
    for name,value in [('ARG',values),('FLAG',flags),('OTHER',others)]:setattr(group,names[name],value)
    b.Network(group).run(.1*b.ms,namespace={})
    expected=np.stack([np.asarray(getattr(group,'s'+str(j)+'_')[:]).copy() for j in range(4)])
    (tmp_path/'actual-brian-reset.txt').write_text(str(group.resetter['spike'].codeobj.code))
    np.save(tmp_path/'actual-brian-output.npy',expected)
    registry={name:(('tanh_scalar' if name=='tanh' else name)+'_f32',
        '_eon_hyper32_'+name+('_scalar' if name=='tanh' else '')) for name in functions}
    renderer=Renderer({names['ARG']:np.dtype('uint8'),names['FLAG']:np.dtype('bool'),
        names['OTHER']:np.dtype('bool')},{},set(),set(),'reset',half_ufuncs=registry)
    codes=[renderer.expression(expr).code for expr in expressions]
    source=('uint i=thread_position_in_grid.x;if(i<N){long '+names['ARG']+'=values[i];bool '+
        names['FLAG']+'=flags[i];bool '+names['OTHER']+'=other_flags[i];')
    source+=''.join('out['+str(j)+'*N+i]='+code+';' for j,code in enumerate(codes))+'}'
    (tmp_path/'native-expression.metal').write_text(source)
    header=binary64.MSL_HEADER+binary32.MSL_HEADER+math64.MSL_MATH_HEADER
    header+=binary16.MSL_HEADER+binary16.MSL_F32_HEADER
    header+=metal_hyperbolic32_header({'log10','sinh','cosh','tanh_scalar','tanh_numpy'},'numpy')
    kernel=mx.fast.metal_kernel(name='hyper_bool_operands',input_names=['values','flags','other_flags'],
        output_names=['out'],header=header,source=source,compile_options={'math_mode':'safe'})
    (actual,)=kernel(inputs=[mx.array(values),mx.array(flags),mx.array(others)],template=[('N',4)],
        grid=(4,1,1),threadgroup=(32,1,1),output_shapes=[expected.shape],output_dtypes=[mx.uint64])
    mx.eval(actual)
    np.save(tmp_path/'native-output-raw64.npy',np.asarray(actual))
    np.testing.assert_array_equal(np.asarray(actual),expected.view(np.uint64))
    # The dtype itself affects a following integer expression and its ufunc
    # loop, including when multiplication happens to yield the same0/1values.
    actual_argument=eval(argument,{'__builtins__':{}},{'ARG':values,'FLAG':flags,'OTHER':others})
    rendered=renderer.expression(expression)
    assert rendered.kind==('bool' if actual_argument.dtype==np.dtype(bool) else 'int')
    assert rendered.integer_bits==actual_argument.dtype.itemsize*8
    assert rendered.integer_unsigned==(actual_argument.dtype.kind=='u')
