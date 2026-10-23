"""Actual NumPy complex ufunc bits/status against one native operator graph."""
import os
import json
import operator
from pathlib import Path

import numpy as np
import pytest
from numpy_callback import RestoreCallback

from native_execution import require_metal
mx = require_metal('Metal required', allow_module_level=True)

pytestmark = pytest.mark.metal


def _dependencies():
    # Development may use explicitly frozen, ignored private dependency files.
    # Integrated runs resolve the same modules from the normal backend package.
    import mlx_backend
    overlay = os.environ.get('EON_COMPLEX_STATUS_DEPENDENCY_ROOT')
    if overlay:
        root = str(Path(overlay).resolve())
        if root not in mlx_backend.__path__:
            mlx_backend.__path__.append(root)
    from mlx_backend.binary64 import MSL_HEADER
    from mlx_backend.binary32 import MSL_HEADER as SINGLE_HEADER
    from mlx_backend.math64 import MSL_MATH_HEADER
    from mlx_backend.complex_math64 import MSL_BASE_HEADER
    from mlx_backend.numpy_error_policy import metal_header
    return MSL_HEADER + SINGLE_HEADER + MSL_MATH_HEADER + MSL_BASE_HEADER + metal_header()


def _cases(dtype):
    f = np.finfo(dtype)
    with np.errstate(all='ignore'):
        small = np.nextafter(dtype(0), dtype(1))
    return [((1.25,-2.5),(.75,3.125)), ((0.,-0.),(-0.,0.)),
            ((f.max,f.max),(f.max,-f.max)), ((f.max,0.),(2.,0.)),
            ((f.tiny,f.tiny),(.5,.5)), ((small,small),(.5,0.)),
            ((small,small),(small,small)), ((f.tiny,1.),(1.,small)),
            ((1.,0.),(f.max,f.tiny)), ((1.,-2.),(f.tiny,small)),
            ((1.,1.),(0.,0.)), ((0.,1.),(0.,-0.)), ((0.,0.),(0.,0.)),
            ((np.inf,0.),(0.,0.)), ((np.inf,1.),(-np.inf,2.)),
            ((np.inf,np.inf),(np.inf,np.inf)), ((np.nan,1.),(2.,3.)),
            ((1.,2.),(np.nan,3.)), ((1.,2.),(3.,np.nan)),
            ((-f.tiny,small),(f.tiny,-small))]


def _compare(op,width,cases=None,array_loop=True,receipt=None):
    from mlx_backend.numpy_complex_error_status import metal_complex_error_header
    header = _dependencies() + metal_complex_error_header((op,))
    dtype = np.float64 if width == 64 else np.float32
    complex_dtype = np.complex128 if width == 64 else np.complex64
    cases = _cases(dtype) if cases is None else cases
    left = np.asarray([case[0] for case in cases],dtype)
    right = np.asarray([case[1] for case in cases],dtype)
    function = (np.add,np.subtract,np.multiply,np.divide)[op]
    flags, expected = [], []
    with RestoreCallback(np.geterrcall()):
        for a,b in zip(left,right):
            calls = []
            np.seterrcall(lambda message,status: calls.append(status))
            with np.errstate(all='call'):
                if array_loop:
                    result = function(a.view(complex_dtype),b.view(complex_dtype))
                else:
                    scalar_operator = (operator.add,operator.sub,operator.mul,operator.truediv)[op]
                    result = scalar_operator(a.view(complex_dtype)[0],b.view(complex_dtype)[0])
            flags.append(0 if not calls else calls[-1])
            expected.append(np.asarray(result).reshape(1).view(dtype).astype(np.float64).view(np.uint64))
    names = ('add','sub','mul','div')
    helper = ('_eon_complex_multiply_numpy' if op==2 and array_loop else
              '_eon_complex_' + names[op]) + ('32' if width==32 else '')
    kernel = mx.fast.metal_kernel(name='complex_numpy_status_'+names[op]+'_'+str(width),
        input_names=['left','right'],output_names=['values','status'],
        header=header,source=f'''
        uint i=thread_position_in_grid.x;
        _eon_complex a={{left[2*i],left[2*i+1]}},b={{right[2*i],right[2*i+1]}};
        _eon_complex value={helper}(a,b);
        values[2*i]=value.real;values[2*i+1]=value.imag;
        status[i]=_eon_numpy_complex_binary_status({op}u,a,b,value,{width}u,{str(array_loop).lower()});
        ''',compile_options={'math_mode':'safe'})
    with np.errstate(all='ignore'):
        packed = [mx.array(data.astype(np.float64).view(np.uint64).reshape(-1))
                  for data in (left,right)]
    result,status = kernel(inputs=packed,grid=(len(cases),1,1),
        threadgroup=(len(cases),1,1),output_shapes=[(len(cases),2),(len(cases),)],
        output_dtypes=[mx.uint64,mx.uint32])
    # Explicit evaluation keeps compiler diagnostics in the Python exception
    # channel instead of throwing through NumPy's buffer-protocol conversion.
    mx.eval(result,status)
    observed = np.asarray(status)
    if receipt is not None:
        receipt.write_text(json.dumps(dict(width=width,array_loop=array_loop,
            left=packed[0].tolist(),right=packed[1].tolist(),
            native_raw=np.asarray(result).tolist(),numpy_raw=np.asarray(expected).tolist(),
            native_status=observed.tolist(),numpy_status=flags),indent=2)+'\n')
    differing = np.flatnonzero(observed != flags)
    assert not differing.size, (width,names[op],int(differing[0]) if differing.size else None,
        observed.tolist(),flags)
    np.testing.assert_array_equal(np.asarray(result),np.asarray(expected),
                                  err_msg='Actual NumPy complex raw component bits')


@pytest.mark.parametrize('width',[64,32])
@pytest.mark.parametrize('op',[0,1,2,3],ids=['add','subtract','multiply','divide'])
def test_complex_component_graph_matches_numpy_bits_and_status(op,width):
    _compare(op,width)


@pytest.mark.parametrize('array_loop',[True,False],ids=['array','scalar'])
@pytest.mark.parametrize('width',[64,32])
def test_complex_multiply_contraction_witnesses(width,array_loop,tmp_path):
    dtype=np.float64 if width==64 else np.float32
    cases=_cases(dtype)
    _compare(2,width,[cases[2],cases[7]],array_loop,
             tmp_path/'raw-results-and-action-masks.json')
