"""Actual CPython arithmetic oracle for dynamic native integer/float carriers."""
from types import SimpleNamespace
import numpy as np
import pytest
import mlx.core as mx
from mlx_backend.compiled_python_dynamic import render


def _tree(case):
    x=('abs',('argument',0,'int64'));y=('abs',('argument',1,'int64'))
    p=('argument',2,'int64');q=('argument',3,'int64')
    a=('pow_dynamic',x,p);b=('pow_dynamic',y,q)
    return {'difference':('sub',a,b),'ratio':('truediv',a,b),
            'modulo':('mod',a,b),'floor':('floordiv',a,b),
            'nested':('pow_dynamic',a,q)}[case]


@pytest.mark.metal
@pytest.mark.parametrize('limbs',[2,16])
@pytest.mark.parametrize('destination',['float64','int64','bool'])
@pytest.mark.parametrize('case',['difference','ratio','modulo','floor','nested'])
def test_dynamic_integer_float_values_and_precision_requests(case,limbs,destination):
    from mlx_backend.binary64 import MSL_HEADER as BINARY
    from mlx_backend.math64 import MSL_MATH_HEADER
    from mlx_backend.compiled_python_integer import MSL_HEADER as INTEGER
    from mlx_backend.compiled_precision import MSL_HEADER as CHECKED,ERROR_NAME
    from mlx_backend.operator64 import MSL_POWER_HEADER,MSL_REMAINDER_HEADER
    catalog=[]
    def register(name,message):
        item=(name,message)
        if item not in catalog:catalog.append(item)
        return catalog.index(item)+1
    used=set()
    source,result=render(_tree(case),['x[i]','y[i]','p[i]','q[i]'],destination,
        SimpleNamespace(unsigned_char=False),'probe',register,used_math=used,limbs=limbs)
    # Both operand powers remain observable through a result or explicit status.
    data=[(2,3,3,2),(2,3,63,32),(2,3,120,70),(2,3,200,120),
          (2,3,-1,2),(2,3,-7,-3),(2,3,0,-2),(2,3,3,-1)]
    if case=='nested':data=[(2,3,p,q) for p,q in [(3,2),(63,2),(120,2),(200,2),(-1,2),(-7,-3),(0,-2),(3,-1)]]
    def actual(x,y,p,q):
        a=abs(x)**p;b=abs(y)**q
        return {'difference':lambda:a-b,'ratio':lambda:a/b,'modulo':lambda:a%b,
                'floor':lambda:a//b,'nested':lambda:a**q}[case]()
    results=[actual(*row) for row in data]
    expected=[];store_overflow=[]
    for expected_result in results:
        if destination=='float64':expected.append(int(np.asarray(float(expected_result),np.float64).view(np.uint64)));store_overflow.append(False)
        elif destination=='bool':expected.append(int(bool(expected_result)));store_overflow.append(False)
        else:
            value=int(expected_result);expected.append(value&((1<<64)-1));store_overflow.append(not -(1<<63)<=value<(1<<63))
    expected=np.array(expected,np.uint64)
    header=BINARY+MSL_MATH_HEADER+INTEGER+CHECKED+MSL_POWER_HEADER+MSL_REMAINDER_HEADER
    kernel=mx.fast.metal_kernel(name='ci_dynamic_'+case,header=header,
        input_names=['x','y','p','q'],output_names=['out','status'],
        source='uint i=thread_position_in_grid.x;uint _eon_integer_error=0u;\n'+'\n'.join(source)+
               f'\nout[i]={result};status[i]=_eon_integer_error;',compile_options={'math_mode':'safe'})
    arrays=[mx.array(np.array(column,np.int64)) for column in zip(*data)]
    out,status=kernel(inputs=arrays,grid=(len(data),1,1),threadgroup=(len(data),1,1),
        output_shapes=[(len(data),)]*2,output_dtypes=[mx.uint64,mx.uint32])
    out,status=map(np.asarray,(out,status));precision=next(i+1 for i,item in enumerate(catalog) if item[0]==ERROR_NAME)
    for row,kind in enumerate(status.tolist()):
        if kind==precision:
            assert limbs==2,(case,destination,row,catalog,status)
        elif store_overflow[row]:assert catalog[kind-1][0]=='OverflowError' and kind!=0
        else:assert kind==0,(case,limbs,destination,row,catalog,status)
    if limbs==2:assert np.any(status==precision)
    np.testing.assert_array_equal(out[status==0],expected[status==0])
