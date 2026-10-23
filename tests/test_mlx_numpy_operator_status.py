"""One selected native operator graph per tiny independent NumPy oracle."""
import numpy as np
import pytest
from numpy_callback import RestoreCallback

from native_execution import require_metal
mx = require_metal('Metal required', allow_module_level=True)
pytestmark=pytest.mark.metal


@pytest.mark.parametrize('name',['power','remainder','floor_divide'])
def test_operator_graph_status_matches_actual_numpy_actions(name):
    from mlx_backend.binary64 import MSL_HEADER
    from mlx_backend.math64 import MSL_MATH_HEADER
    from mlx_backend.numpy_error_policy import metal_header as checked_header
    from mlx_backend.numpy_operator_status import metal_header
    from mlx_backend.operator64 import MSL_POWER_HEADER,MSL_REMAINDER_HEADER
    small=np.nextafter(np.float64(0),np.float64(1));tiny=np.finfo(np.float64).tiny
    snan=np.uint64(0x7ff0000000000001).view(np.float64)
    cases=([(0.,-1.),(-1.,.5),(.5,1074.),(.5,1075.),(2.,1024.),
            (small,1.),(small,2.),(tiny,1.5),(-2.,np.inf),(snan,0.),(1.,snan),(.3,2.),(.3,2.1)]
           if name=='power' else
           [(1.,0.),(0.,0.),(np.inf,1.),(np.inf,np.inf),(1.,np.inf),
            (small,2.),(-small,2.),(tiny,small),(snan,1.),(np.nan,0.)])
    rows=[];previous=np.geterrcall()
    with RestoreCallback(previous):
        for a,b in cases:
            calls=[]
            np.seterrcall(lambda text,flags:calls.append(flags))
            with np.errstate(all='call'):
                result=getattr(np,name)(np.array([a]),np.array([b]))[0]
            rows.append((result,0 if not calls else calls[-1]))
    op,root={'power':(6,'d_operator_pow'),'remainder':(4,'d_operator_mod_numpy'),
             'floor_divide':(5,'d_operator_floordiv_numpy')}[name]
    kernel=mx.fast.metal_kernel(name='numpy_policy_operator_'+name,
        input_names=['left','right'],output_names=['results','flags'],
        header=MSL_HEADER+MSL_MATH_HEADER
            +(MSL_POWER_HEADER if name=='power' else MSL_REMAINDER_HEADER)
            +checked_header()+metal_header((name,)),
        source=f'''uint i=thread_position_in_grid.x;ulong a=left[i],b=right[i];
            results[i]={root}(a,b);
            flags[i]=_eon_numpy_operator64_status({op}u,a,b,false,b);''',
        compile_options={'math_mode':'safe'})
    values,flags=kernel(inputs=[
        mx.array(np.array([c[0] for c in cases],np.float64).view(np.uint64)),
        mx.array(np.array([c[1] for c in cases],np.float64).view(np.uint64))],
        grid=(len(cases),1,1),threadgroup=(len(cases),1,1),
        output_shapes=[(len(cases),),(len(cases),)],output_dtypes=[mx.uint64,mx.uint32])
    differences=[(float(a).hex(),float(b).hex(),row[1],int(got))
                 for (a,b),row,got in zip(cases,rows,np.asarray(flags)) if row[1]!=got]
    assert differences==[]
    np.testing.assert_array_equal(np.asarray(values),
                                 np.array([row[0] for row in rows],np.float64).view(np.uint64))


@pytest.mark.parametrize('name',['power','remainder','floor_divide'])
@pytest.mark.parametrize('width',[32,16])
def test_narrow_operator_graph_and_half_cast_match_numpy(name,width):
    from mlx_backend import binary64,binary32,binary16,math64,operator64,operator32
    from mlx_backend.numpy_error_policy import metal_header as checked_header
    from mlx_backend.numpy_operator_status import metal_narrow_header
    dtype=np.float32 if width==32 else np.float16
    word=np.uint32 if width==32 else np.uint16
    small=np.array(1,dtype=word).view(dtype)[()]
    tiny=np.finfo(dtype).tiny
    snan=np.array(0x7f800001 if width==32 else 0x7c01,dtype=word).view(dtype)[()]
    last=149 if width==32 else 24
    cases=([(0.,-1.),(-1.,.5),(.5,last),(.5,last+1),(2.,128 if width==32 else 16),
            (small,1.),(small,2.),(tiny,1.5),(-2.,np.inf),(snan,0.),(1.,snan)]
           if name=='power' else
           [(1.,0.),(0.,0.),(np.inf,1.),(np.inf,np.inf),(1.,np.inf),
            (small,2.),(-small,2.),(tiny,small),(snan,1.),(np.nan,0.)])
    left=np.array([a for a,b in cases],dtype)
    right=np.array([b for a,b in cases],dtype)
    expected=[];previous=np.geterrcall()
    with RestoreCallback(previous):
        for a,b in zip(left,right):
            calls=[];np.seterrcall(lambda text,flags:calls.append(flags))
            with np.errstate(all='call'):
                result=getattr(np,name)(np.array([a],dtype),np.array([b],dtype))[0]
            expected.append((int(result.view(word)),0 if not calls else calls[-1]))
    op,root={'power':(6,'f_operator_pow'),'remainder':(4,'f_operator_mod_numpy'),
             'floor_divide':(5,'f_operator_floordiv_numpy')}[name]
    originals=(operator32.MSL_POWER_HEADER if name=='power' else
               operator64.MSL_REMAINDER_HEADER+operator32.MSL_REMAINDER_HEADER)
    operands=('uint a=left[i],b=right[i];' if width==32 else
              'uint a=h_d_to_f(h_to_d(left[i])),b=h_d_to_f(h_to_d(right[i]));')
    result=f'{root}(a,b)' if width==32 else f'h_from_d(h_f32_to_d({root}(a,b)))'
    status=(f'_eon_numpy_operator32_status({op}u,a,b,false,0ul)' if width==32 else
            f'_eon_numpy_operator16_status({op}u,ushort(left[i]),ushort(right[i]),false,0ul)')
    kernel=mx.fast.metal_kernel(name=f'numpy_policy_operator_{name}_{width}',
        input_names=['left','right'],output_names=['results','flags'],
        header=binary64.MSL_HEADER+binary32.MSL_HEADER+binary16.MSL_HEADER
            +binary16.MSL_F32_HEADER+math64.MSL_MATH_HEADER+originals
            +checked_header()+metal_narrow_header((name,)),
        source=f'''uint i=thread_position_in_grid.x;{operands}
            results[i]={result};flags[i]={status};''',compile_options={'math_mode':'safe'})
    values,flags=kernel(inputs=[mx.array(left.view(word)),mx.array(right.view(word))],
        grid=(len(cases),1,1),threadgroup=(len(cases),1,1),output_shapes=[(len(cases),)]*2,
        output_dtypes=[mx.uint32,mx.uint32])
    observed=list(zip(map(int,np.asarray(values)),map(int,np.asarray(flags))))
    assert observed==expected,[(i,expected[i],got) for i,got in enumerate(observed)
                               if got!=expected[i]]


@pytest.mark.parametrize('width',[64,32,16])
def test_scalar_power_status_uses_original_exponent_before_narrowing(width):
    from mlx_backend import binary64,binary32,binary16,math64,operator64,operator32
    from mlx_backend.numpy_error_policy import metal_header as checked_header
    from mlx_backend.numpy_operator_status import metal_header,metal_narrow_header
    dtype={64:np.float64,32:np.float32,16:np.float16}[width]
    word={64:np.uint64,32:np.uint32,16:np.uint16}[width]
    raw_snan={64:0x7ff0000000000001,32:0x7f800001,16:0x7c01}[width]
    snan=np.array(raw_snan,word).view(dtype)[()]
    small=np.array(1,word).view(dtype)[()]
    cases=[(snan,b) for b in (0.,1.,2.,-1.,.5,np.nextafter(1.,2.))]
    cases +=[(-1.,.5),(small,2.),(small,-1.)]
    left=np.array([a for a,b in cases],dtype)
    original=np.array([b for a,b in cases],np.float64)
    expected=[];previous=np.geterrcall()
    with RestoreCallback(previous):
        for a,b in zip(left,original):
            calls=[];np.seterrcall(lambda text,flags:calls.append(flags))
            with np.errstate(all='call'):
                np.array([a],dtype)**float(b)
            expected.append(0 if not calls else calls[-1])
    common=(binary64.MSL_HEADER+binary32.MSL_HEADER+binary16.MSL_HEADER
            +binary16.MSL_F32_HEADER+math64.MSL_MATH_HEADER)
    if width==64:
        headers=operator64.MSL_POWER_HEADER+checked_header()+metal_header(('power',))
        call='_eon_numpy_operator64_status(6u,left[i],original[i],true,original[i])'
    else:
        headers=operator32.MSL_POWER_HEADER+checked_header()+metal_narrow_header(('power',))
        call=('_eon_numpy_operator32_status(6u,uint(left[i]),f_from_d(original[i]),true,original[i])'
              if width==32 else
              '_eon_numpy_operator16_status(6u,ushort(left[i]),ushort(h_from_d(original[i])),true,original[i])')
    kernel=mx.fast.metal_kernel(name='numpy_policy_scalar_power_'+str(width),
        input_names=['left','original'],output_names=['flags'],header=common+headers,
        source=f'uint i=thread_position_in_grid.x;flags[i]={call};',
        compile_options={'math_mode':'safe'})
    flags=kernel(inputs=[mx.array(left.view(word).astype(np.uint64)),
                        mx.array(original.view(np.uint64))],
        grid=(len(cases),1,1),threadgroup=(len(cases),1,1),
        output_shapes=[(len(cases),)],output_dtypes=[mx.uint32])[0]
    observed=list(map(int,np.asarray(flags)))
    assert observed==expected,[(i,expected[i],got) for i,got in enumerate(observed)
                               if got!=expected[i]]
