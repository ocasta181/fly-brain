"""One selected complex ufunc per shader, with independent NumPy status."""
import numpy as np
import pytest
from numpy_callback import RestoreCallback
from native_execution import require_metal


CASES = {
    'exp': [(0.,0.), (1.,.25), (1500.,0.), (-1500.,0.), (0.,np.inf)],
    'log': [(0.,0.), (1.,0.), (.25,.5)],
    'expm1': [(0.,0.), (1.,.25), (1500.,0.), (-1500.,0.)],
    'log1p': [(-1.,0.), (0.,0.), (.25,.5)],
    'log10': [(0.,0.), (1.,0.), (.25,.5)],
    'sinh': [(0.,0.), (1500.,.25), (.25,.5)],
    'cosh': [(0.,0.), (1500.,.25), (.25,.5)],
    'tanh': [(0.,0.), (0.,np.inf), (.25,.5)],
    'sin': [(0.,0.), (.25,1500.), (.25,.5)],
    'cos': [(0.,0.), (.25,1500.), (.25,.5)],
    'tan': [(0.,0.), (np.inf,0.), (.25,.5)],
    'arctan': [(0.,1.), (0.,0.), (.25,.5)],
    'arcsin': [(0.,0.), (1.,0.), (.25,.5)],
    'arccos': [(0.,0.), (1.,0.), (.25,.5)],
    'sqrt': [(0.,0.), (-1.,0.), (.25,.5)],
    'abs': [(np.finfo(np.float64).max,np.finfo(np.float64).max),
            (np.finfo(np.float32).max,np.finfo(np.float32).max)],
    'sign': [(0.,0.), (np.nan,0.), (.25,.5)],
}


def test_complex_abs_scalar_cpu_dispatch():
    # A scalar argument and a zero-dimensional array still use the ufunc's
    # selected loop. Do not infer fallback status from argument shape.
    maximum=np.finfo(np.float64).max
    with RestoreCallback(np.geterrcall()):
        for value in (np.complex128(complex(maximum,maximum)),
                      np.array(complex(maximum,maximum),np.complex128)):
            calls=[]
            np.seterrcall(lambda text,flags:calls.append(flags))
            with np.errstate(all='call'):
                result=np.abs(value)
            assert result==np.inf
            assert calls==[]


@pytest.mark.metal
@pytest.mark.parametrize('function',tuple(CASES))
def test_selected_complex_unary_status(function):
    mx = require_metal('Metal required')
    from mlx_backend import binary64,binary32,binary16,math64
    from mlx_backend.numpy_error_policy import metal_header as binary_status
    from mlx_backend.numpy_complex_unary_status import (
        FUNCTIONS,original_header,metal_complex_unary_header)
    rows=[]
    with RestoreCallback(np.geterrcall()):
        for width in (64,32):
            for real,imag in CASES[function]:
                with np.errstate(all='ignore'):
                    value=np.array([complex(real,imag)],np.complex128 if width==64 else np.complex64)
                calls=[]
                np.seterrcall(lambda text,flags:calls.append(flags))
                with np.errstate(all='call'):
                    result=getattr(np,function)(value)[0]
                pair=np.array([value.real[0],value.imag[0]],np.float64).view(np.uint64)
                expected=np.array([result.real,result.imag if function!='abs' else 0.],np.float64).view(np.uint64)
                rows.append((width,pair,expected,0 if not calls else calls[-1]))
    root='_eon_complex_'+('abs_numpy' if function=='abs' else function)
    single=root+'32' if function!='sign' else root
    if function=='abs':
        calculation=f'_eon_complex result={{widths[i]==32u?{single}(input):{root}(input),0ul}};'
    else:
        calculation=f'_eon_complex result=widths[i]==32u?{single}(input):{root}(input);if(widths[i]==32u)result=_eon_complex_round32(result);'
    kernel=mx.fast.metal_kernel(name='complex_unary_status_'+function,
        input_names=['raw','widths'],output_names=['out','status'],
        header=(binary64.MSL_HEADER+binary32.MSL_HEADER+binary16.MSL_HEADER
                +binary16.MSL_F32_HEADER+math64.MSL_MATH_HEADER+original_header()
                +binary_status()+metal_complex_unary_header((function,))),
        source=('uint i=thread_position_in_grid.x;_eon_complex input={raw[2*i],raw[2*i+1]};'
                +calculation+f'status[i]=_eon_numpy_complex_unary_status({FUNCTIONS.index(function)}u,input,result,widths[i]);'
                'out[2*i]=result.real;out[2*i+1]=result.imag;'),
        compile_options={'math_mode':'safe'})
    output,status=kernel(inputs=[mx.array(np.stack([r[1] for r in rows]).reshape(-1)),
                                mx.array([r[0] for r in rows],mx.uint32)],
        grid=(len(rows),1,1),threadgroup=(min(32,len(rows)),1,1),
        output_shapes=[(len(rows),2),(len(rows),)],output_dtypes=[mx.uint64,mx.uint32])
    mx.eval(output,status)
    actual=np.asarray(output)
    flags=np.asarray(status)
    for index,(width,pair,expected,wanted) in enumerate(rows):
        assert int(flags[index])==wanted,(function,width,pair.tolist(),wanted,int(flags[index]))
        np.testing.assert_array_equal(actual[index],expected,
            err_msg=str((function,width,pair.tolist())))


@pytest.mark.metal
def test_complex_power_status_keeps_scalar_internal_graph():
    mx = require_metal('Metal required')
    from mlx_backend import binary64,binary32,binary16,math64
    from mlx_backend.numpy_error_policy import metal_header as binary_status
    from mlx_backend.numpy_complex_unary_status import original_header,metal_complex_unary_header
    rows=[]
    with RestoreCallback(np.geterrcall()):
        for width in (64,32):
            for a,b in ((0j,-1+0j),(0j,0j),(.25+.5j,.3+0j),(2+0j,2+0j)):
                dtype=np.complex128 if width==64 else np.complex64
                a=np.array([a],dtype);b=np.array([b],dtype)
                calls=[]
                np.seterrcall(lambda text,flags:calls.append(flags))
                with np.errstate(all='call'):
                    result=np.power(a,b)[0]
                raw=np.array([a.real[0],a.imag[0],b.real[0],b.imag[0]],np.float64).view(np.uint64)
                expected=np.array([result.real,result.imag],np.float64).view(np.uint64)
                rows.append((width,raw,expected,0 if not calls else calls[-1]))
    kernel=mx.fast.metal_kernel(name='complex_power_status_scalar_graph',
        input_names=['raw','widths'],output_names=['out','status'],
        header=(binary64.MSL_HEADER+binary32.MSL_HEADER+binary16.MSL_HEADER
                +binary16.MSL_F32_HEADER+math64.MSL_MATH_HEADER+original_header()
                +binary_status()+metal_complex_unary_header((),power=True)),
        source=('uint i=thread_position_in_grid.x;'
                '_eon_complex a={raw[4*i],raw[4*i+1]},b={raw[4*i+2],raw[4*i+3]};'
                '_eon_complex r=widths[i]==32u?_eon_complex_power32(a,b):_eon_complex_power(a,b);'
                'if(widths[i]==32u)r=_eon_complex_round32(r);'
                'status[i]=_eon_numpy_complex_power_status(a,b,r,widths[i]);'
                'out[2*i]=r.real;out[2*i+1]=r.imag;'),
        compile_options={'math_mode':'safe'})
    result,status=kernel(inputs=[mx.array(np.stack([r[1] for r in rows]).reshape(-1)),
                                mx.array([r[0] for r in rows],mx.uint32)],
        grid=(len(rows),1,1),threadgroup=(min(32,len(rows)),1,1),
        output_shapes=[(len(rows),2),(len(rows),)],output_dtypes=[mx.uint64,mx.uint32])
    mx.eval(result,status)
    actual=np.asarray(result);flags=np.asarray(status)
    for index,(width,raw,expected,wanted) in enumerate(rows):
        assert int(flags[index])==wanted,(width,raw.tolist(),wanted,int(flags[index]))
        np.testing.assert_array_equal(actual[index],expected,err_msg=str((width,raw.tolist())))
