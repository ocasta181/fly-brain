"""Small inverse-family branch checks with exact actual NumPy references."""
import numpy as np
import pytest


def _branch_bits(function,single):
    mx=pytest.importorskip('mlx.core')
    from mlx_backend import binary64,binary32,math64,basic_math32
    from mlx_backend.arctangent64 import metal_arctangent_header
    from mlx_backend.trigonometry64 import metal_trigonometry_header
    from mlx_backend.trigonometry32 import metal_trigonometry32_header
    from mlx_backend.stable_math64 import metal_stable_math_header,_STABLE_DIV_HEADER
    from mlx_backend.transcendentals import metal_math_header
    from mlx_backend.complex_math64 import metal_complex_header
    from mlx_backend.complex_log_math64 import metal_complex_log_header
    from mlx_backend.complex_asinh_math32 import metal_complex_asinh_header
    from mlx_backend.complex_inverse_math64 import metal_complex_inverse_header
    pairs=np.array([
        (0.,0.),(-0.,0.),(0.,-0.),(-0.,-0.),
        (1.,0.),(1.,-0.),(-1.,0.),(-1.,-0.),(0.,1.),(0.,-1.),
        (.25,.5),(-.25,.5),(.25,-.5),(-.25,-.5),
        (1.,.25),(-1.,.25),(1.,-.25),(-1.,-.25),
        (2.**-28,2.**-29),(2.**-13,2.**-14),
        (2.**28,.25),(.25,2.**28),(2.**63,2.**62),
        (np.finfo(np.float64).max,1.),(1.,np.finfo(np.float64).max),
        (np.inf,0.),(-np.inf,0.),(0.,np.inf),(np.inf,np.inf),
        (np.nan,0.),(0.,np.nan),(np.nan,1.),(1.,np.nan),
    ],np.float64)
    with np.errstate(all='ignore'):
        if single:pairs=pairs.astype(np.float32).astype(np.float64)
        values=pairs.astype(np.float32 if single else np.float64).view(
            np.complex64 if single else np.complex128).reshape(-1)
        expected=getattr(np,function)(values)
        expected=np.column_stack((expected.real,expected.imag)).astype(np.float64).view(np.uint64)
    raw=pairs.view(np.uint64)
    header=(binary64.MSL_HEADER+binary32.MSL_HEADER+math64.MSL_MATH_HEADER
        +basic_math32.MSL_HEADER+metal_math_header({'log','exp'})
        +metal_stable_math_header({'log1p','expm1'})
        +metal_trigonometry_header({'sin','cos'})+metal_arctangent_header({'arctan'})
        +metal_trigonometry32_header({'sin','cos','tan'}).replace(_STABLE_DIV_HEADER,'')
        +metal_complex_header({'_complex','_complex_trig','_complex_basic','_complex_extended'})
        +metal_complex_log_header()+metal_complex_asinh_header()+metal_complex_inverse_header())
    call=f'_eon_complex_{function}{"32" if single else ""}({{raw[2*i],raw[2*i+1]}})'
    if single:call=f'_eon_complex_round32({call})'
    kernel=mx.fast.metal_kernel(name='complex_inverse_'+function+str(int(single)),
        input_names=['raw'],output_names=['out'],header=header,
        source=f'''uint i=thread_position_in_grid.x;if(i>={len(raw)}u)return;
        _eon_complex result={call};
        out[2*i]=result.real;out[2*i+1]=result.imag;''')
    out=kernel(inputs=[mx.array(raw.reshape(-1))],grid=(len(raw),1,1),
        threadgroup=(32,1,1),output_shapes=[expected.shape],output_dtypes=[mx.uint64])[0]
    mx.eval(out)
    actual=np.asarray(out)
    differences=np.argwhere(actual!=expected)
    assert not len(differences),dict(function=function,single=single,
        mismatches=len(differences),first=[dict(input=pairs[i].tolist(),component=int(j),
            actual=hex(int(actual[i,j])),expected=hex(int(expected[i,j]))) for i,j in differences[:4]])


@pytest.mark.metal
@pytest.mark.parametrize('single',[False,True])
@pytest.mark.parametrize('function',['arcsin','arccos','arctan'])
def test_complex_inverse_branch_bits(function,single):
    _branch_bits(function,single)


@pytest.mark.metal
@pytest.mark.parametrize('single',[False,True])
@pytest.mark.parametrize('function',['sin','cos','sinh','cosh','exp','sqrt','expm1','tanh','tan'])
def test_complex_elementary_branch_bits(function,single):
    _branch_bits(function,single)
