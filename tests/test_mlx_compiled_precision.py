"""Independent Python integer oracle for checked native temporary capacity."""
import numpy as np
import pytest
import mlx.core as mx
from mlx_backend.compiled_python_integer import MSL_HEADER
from mlx_backend.compiled_precision import MSL_HEADER as CHECKED_HEADER


@pytest.mark.metal
@pytest.mark.parametrize('limbs',[2,8])
def test_checked_native_integer_capacity(limbs):
    rng=np.random.default_rng(9204+limbs)
    bits=32*limbs;mask=(1<<bits)-1
    a=[0,1,-1,mask,-mask,1<<(bits-1),-(1<<(bits-1)),2,-2]
    b=[0,mask,-mask,1,-1,1<<(bits-1),-(1<<(bits-1)),17,18]
    for i in range(32):
        x=int.from_bytes(rng.bytes(4*limbs),'little');y=int.from_bytes(rng.bytes(4*limbs),'little')
        a.append(x if i&1 else -x);b.append(y if i&2 else -y)
    exponents=[0,1,2,3,5,7,13,31,63]*5
    exponents=exponents[:len(a)]
    words=lambda xs:np.array([[(abs(x)>>(32*j))&0xffffffff for j in range(limbs)] for x in xs],np.uint32).reshape(-1)
    kernel=mx.fast.metal_kernel(name='ci_checked_capacity',header=MSL_HEADER+CHECKED_HEADER,
        input_names=['left','right','signs','powers'],output_names=['values','negative','exhausted'],source=f'''
        uint row=thread_position_in_grid.x;
        ci_py<{limbs}> a=ci_py_zero<{limbs}>(),b=ci_py_zero<{limbs}>();
        for(uint j=0;j<{limbs}u;j++){{a.limb[j]=left[row*{limbs}u+j];b.limb[j]=right[row*{limbs}u+j];}}
        a.negative=signs[row*2u]!=0u;b.negative=signs[row*2u+1u]!=0u;
        ci_py<{limbs}> e=ci_py_from<{limbs}>(ulong(powers[row]),false);
        bool flags[4]={{false,false,false,false}};
        ci_py<{limbs}> results[4];
        results[0]=ci_precision_add(a,b,flags[0]);results[1]=ci_precision_sub(a,b,flags[1]);
        results[2]=ci_precision_mul(a,b,flags[2]);results[3]=ci_precision_pow(a,e,flags[3]);
        for(uint op=0;op<4u;op++){{
            exhausted[row*4u+op]=flags[op];negative[row*4u+op]=results[op].negative;
            for(uint j=0;j<{limbs}u;j++)values[(row*4u+op)*{limbs}u+j]=results[op].limb[j];
        }}''')
    output=kernel(inputs=[mx.array(words(a)),mx.array(words(b)),
        mx.array(np.array([[x<0,y<0] for x,y in zip(a,b)],np.uint32).reshape(-1)),
        mx.array(np.array(exponents,np.uint32))],grid=(len(a),1,1),threadgroup=(min(64,len(a)),1,1),
        output_shapes=[(len(a),4,limbs),(len(a),4),(len(a),4)],output_dtypes=[mx.uint32,mx.bool_,mx.bool_])
    values,negative,exhausted=map(np.asarray,output)
    for row,(x,y,e) in enumerate(zip(a,b,exponents)):
        expected=[x+y,x-y,x*y,x**e]
        for op,result in enumerate(expected):
            assert bool(exhausted[row,op])==(abs(result)>mask),(limbs,row,op,x,y,e)
            if not exhausted[row,op]:
                magnitude=sum(int(word)<<(32*j) for j,word in enumerate(values[row,op]))
                actual=-magnitude if negative[row,op] else magnitude
                assert actual==result,(limbs,row,op,x,y,e,actual,result)
