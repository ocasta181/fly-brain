"""Isolated status/result checks for the already supported NumPy math loops."""
import numpy as np
import pytest
from numpy_callback import RestoreCallback

from native_execution import require_metal
mx = require_metal('Metal required', allow_module_level=True)
pytestmark=pytest.mark.metal


@pytest.mark.parametrize('function',['exp','expm1','log','log10','log1p','sqrt',
    'sin','cos','tan','sinh','cosh','tanh','arcsin','arccos','arctan'])
def test_native_unary_graph_flags_match_independent_numpy(function):
    from mlx_backend.binary64 import MSL_HEADER
    from mlx_backend.math64 import MSL_MATH_HEADER
    from mlx_backend.numpy_error_policy import metal_header as binary_status
    from mlx_backend.numpy_math_status import original_header,metal_header
    functions=(function,)
    small=np.nextafter(np.float64(0),np.float64(1))
    tiny=np.finfo(np.float64).tiny
    cases={
        'exp':[-np.inf,-1000.,-745.,-710.,-.02,-.005,0.,709.,710.,np.inf,np.nan],
        'expm1':[-1000.,-tiny,-small,0.,small,tiny,709.,710.,np.inf,np.nan],
        'log':[0.,-0.,-small,-1.,small,tiny,np.inf,np.nan],
        'log10':[0.,-small,-1.,small,tiny,np.inf,np.nan],
        'log1p':[-2.,-1.,-tiny,-small,0.,small,tiny,np.inf,np.nan],
        'sqrt':[-np.inf,-1.,-small,-0.,0.,small,tiny,np.inf,np.nan],
        'sin':[-np.inf,-small,0.,small,tiny,np.inf,np.nan],
        'cos':[-np.inf,-small,0.,small,tiny,np.inf,np.nan],
        'tan':[-np.inf,-small,0.,small,tiny,np.inf,np.nan],
        'sinh':[-1000.,-small,0.,small,tiny,710.,1000.,np.inf,np.nan],
        'cosh':[-1000.,-small,0.,small,tiny,710.,1000.,np.inf,np.nan],
        'tanh':[-np.inf,-small,0.,small,tiny,np.inf,np.nan],
        'arcsin':[-np.inf,-2.,-1.,-small,0.,small,tiny,1.,2.,np.inf,np.nan],
        'arccos':[-np.inf,-2.,-1.,-small,0.,small,tiny,1.,2.,np.inf,np.nan],
        'arctan':[-np.inf,-small,0.,small,tiny,np.inf,np.nan],
    }
    rows=[]
    with RestoreCallback(np.geterrcall()):
        for name,values in ((function,cases[function]),):
            for value in values:
                calls=[]
                np.seterrcall(lambda text,flags:calls.append(flags))
                with np.errstate(all='call'):
                    result=getattr(np,name)(np.array([value],np.float64))[0]
                rows.append((name,value,result,0 if not calls else calls[-1]))
    source=('uint i=thread_position_in_grid.x, flags=0u; ulong x=values[i],result=0ul;'
            'switch(operations[i]){'
            + ''.join(f'case {index}:result=_eon_numpy_{name}_status(x,flags);break;'
                      for index,name in enumerate(functions))
            + '} outputs[i]=result; statuses[i]=flags;')
    kernel=mx.fast.metal_kernel(name='numpy_policy_unary_graph_flags',
        input_names=['values','operations'],output_names=['outputs','statuses'],
        header=MSL_HEADER+MSL_MATH_HEADER+original_header(functions)
               +binary_status()+metal_header(functions),
        source=source,compile_options={'math_mode':'safe'})
    outputs,statuses=kernel(inputs=[
        mx.array(np.array([row[1] for row in rows],np.float64).view(np.uint64)),
        mx.array([functions.index(row[0]) for row in rows],mx.uint32)],
        grid=(len(rows),1,1),threadgroup=(min(len(rows),128),1,1),
        output_shapes=[(len(rows),),(len(rows),)],output_dtypes=[mx.uint64,mx.uint32])
    actual=np.asarray(statuses)
    failures=[(row[0],float(row[1]).hex(),row[3],int(got))
              for row,got in zip(rows,actual) if row[3]!=got]
    assert failures==[]
    np.testing.assert_array_equal(np.asarray(outputs),
                                 np.array([row[2] for row in rows],np.float64).view(np.uint64))
