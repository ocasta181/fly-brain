"""Independent CPython arithmetic screens for the typed native limb helpers."""
import numpy as np
import pytest
from native_execution import require_metal

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('words',[3,6,33,36])
def test_wide_divmod_uses_value_width_without_losing_floor_semantics(words):
    mx = require_metal('Metal required')
    from mlx_backend.compiled_python_integer import MSL_HEADER
    rng=np.random.default_rng(20261019)
    limit=(1<<(32*words))-1
    pairs=[(0,1),(1,limit),(limit,1),(limit,limit),(limit,limit-1)]
    for bit in sorted({bit for bit in (0,31,32,33,63,64,511,32*words-1) if bit<32*words}):
        pairs.extend(((limit,1<<bit),(1<<bit,1<<bit),((1<<bit)-1,1<<bit)))
    for bits in sorted({bits for bits in (1,31,32,33,63,64,65,127,511,1023,32*words-1,32*words) if bits<=32*words}):
        for divisor_bits in (1,max(1,bits//2),bits):
            a=(1<<(bits-1))|int.from_bytes(rng.bytes((bits+7)//8),'little')&((1<<(bits-1))-1)
            d=(1<<(divisor_bits-1))|int.from_bytes(rng.bytes((divisor_bits+7)//8),'little')&((1<<(divisor_bits-1))-1)
            pairs.append((a,d))
            pairs.extend(((max(0,a-d),d),(min(limit,a+d),d)))
    signed=[(sa*a,sb*d) for a,d in pairs for sa,sb in ((1,1),(-1,1),(1,-1),(-1,-1))]
    def packed(values):
        return np.array([[(abs(value)>>(32*j))&0xffffffff for j in range(words)] for value in values],np.uint32)
    aa=packed([a for a,d in signed]);bb=packed([d for a,d in signed])
    signs=np.array([[a<0,d<0] for a,d in signed],np.uint32)
    expected=[]
    for a,d in signed:
        q,r=divmod(a,d)
        expected.append([*[(abs(q)>>(32*j))&0xffffffff for j in range(words)],int(q<0),
                         *[(abs(r)>>(32*j))&0xffffffff for j in range(words)],int(r<0)])
    kernel=mx.fast.metal_kernel(name='compiled_python_wide_divmod',
        input_names=['aa','bb','signs'],output_names=['out'],header=MSL_HEADER,
        source='''uint i=thread_position_in_grid.x;if(i>=COUNT)return;
        ci_py<WORDS> a=ci_py_zero<WORDS>(),b=ci_py_zero<WORDS>();
        for(uint j=0;j<WORDS;j++){a.limb[j]=aa[WORDS*i+j];b.limb[j]=bb[WORDS*i+j];}
        a.negative=signs[2*i]!=0u;b.negative=signs[2*i+1]!=0u;
        ci_py<WORDS> q,r;ci_py_divmod(a,b,q,r);
        for(uint j=0;j<WORDS;j++){out[(2*WORDS+2)*i+j]=q.limb[j];out[(2*WORDS+2)*i+WORDS+1+j]=r.limb[j];}
        out[(2*WORDS+2)*i+WORDS]=uint(q.negative);out[(2*WORDS+2)*i+2*WORDS+1]=uint(r.negative);''',
        compile_options={'math_mode':'safe'})
    (actual,)=kernel(inputs=[mx.array(aa.reshape(-1)),mx.array(bb.reshape(-1)),mx.array(signs.reshape(-1))],
        template=[('COUNT',len(signed)),('WORDS',words)],grid=(len(signed),1,1),threadgroup=(128,1,1),
        output_shapes=[(len(signed),2*words+2)],output_dtypes=[mx.uint32])
    mx.eval(actual)
    np.testing.assert_array_equal(np.asarray(actual),np.array(expected,np.uint32))


def test_boxed_true_division_integer_result_matches_cpython_on_metal():
    mx = require_metal('Metal required')
    from mlx_backend.compiled_python_integer import MSL_HEADER
    rng=np.random.default_rng(20261018)
    aa=rng.integers(0,1<<64,4096,dtype=np.uint64)
    bb=rng.integers(1,1<<64,4096,dtype=np.uint64)
    bb[:1024]=rng.choice(np.array([1,2,3,5,7,11,13,17],np.uint64),1024)
    pairs=list(zip(map(int,aa),map(int,bb)))
    maximum=(1<<64)-1
    for power in range(1,64):
        for divisor in (1,2,3,5,7):
            for offset in (-2,-1,0,1,2):
                numerator=((1<<power)+1)*divisor+offset
                if 0<=numerator<=maximum:pairs.append((numerator,divisor))
        pairs.extend([((1<<power)-1,1<<power),((1<<power)-2,1<<power)])
    aa=np.array([a for a,b in pairs],np.uint64);bb=np.array([b for a,b in pairs],np.uint64)
    flags=np.arange(len(pairs),dtype=np.uint32)%4
    expected=[]
    for (a,b),flag in zip(pairs,map(int,flags)):
        value=int((-a if flag&1 else a)/(-b if flag&2 else b))
        magnitude=abs(value)
        expected.append((magnitude&maximum,magnitude>>64,int(value<0)))
    kernel=mx.fast.metal_kernel(name='compiled_python_true_division_integer',
        input_names=['aa','bb','flags'],output_names=['out'],header=MSL_HEADER,
        source='''uint i=thread_position_in_grid.x;if(i>=COUNT)return;
        ci_py<4> a=ci_py_from<4>(aa[i],(flags[i]&1u)!=0u);
        ci_py<4> b=ci_py_from<4>(bb[i],(flags[i]&2u)!=0u);
        ci_py<4> q=ci_py_truediv_integer(a,b);
        out[3*i]=ulong(q.limb[0])|(ulong(q.limb[1])<<32);
        out[3*i+1]=ulong(q.limb[2])|(ulong(q.limb[3])<<32);
        out[3*i+2]=ulong(q.negative);''',compile_options={'math_mode':'safe'})
    (actual,)=kernel(inputs=[mx.array(aa),mx.array(bb),mx.array(flags)],template=[('COUNT',len(pairs))],
        grid=(len(pairs),1,1),threadgroup=(128,1,1),output_shapes=[(len(pairs),3)],output_dtypes=[mx.uint64])
    mx.eval(actual)
    np.testing.assert_array_equal(np.asarray(actual),np.array(expected,np.uint64))


def test_boxed_multi_limb_double_conversion_matches_cpython_on_metal():
    mx = require_metal('Metal required')
    from mlx_backend.compiled_python_integer import MSL_HEADER
    rng=np.random.default_rng(20261020)
    magnitudes=[0,1,(1<<53)-1,(1<<53)+1,(1<<64)-1,1<<64]
    for bits in (54,64,65,80,128,512,971,1023,1024):
        shift=bits-53;half=1<<(shift-1)
        for significand in ((1<<52),(1<<52)+1,(1<<53)-1):
            magnitudes.extend((significand<<shift)+half+delta for delta in (-1,0,1))
    for bits in (65,80,127,200,511,970,1023,1024,1040,1080):
        for _ in range(20):
            magnitudes.append((1<<(bits-1))|(int.from_bytes(rng.bytes((bits+7)//8),'little')&((1<<(bits-1))-1)))
    boundary=(1<<1024)-(1<<970)
    magnitudes.extend(boundary+offset for offset in (-2,-1,0,1,2))
    values=[sign*value for value in magnitudes for sign in (1,-1)]
    limbs=np.array([[(abs(value)>>(32*j))&0xffffffff for j in range(34)] for value in values],np.uint32)
    signs=np.array([value<0 for value in values],np.uint32)
    expected=[]
    for value in values:
        try:
            converted=float(value)
            expected.append((int(np.array(converted,np.float64).view(np.uint64)),0))
        except OverflowError:
            # Infinity is an internal error marker, never an accepted CPU value.
            expected.append((0x7ff0000000000000|((1<<63) if value<0 else 0),1))
    kernel=mx.fast.metal_kernel(name='compiled_python_multi_limb_double',
        input_names=['limbs','signs'],output_names=['out'],header=MSL_HEADER,
        source='''uint i=thread_position_in_grid.x;if(i>=COUNT)return;
        ci_py<34> value=ci_py_zero<34>();for(uint j=0;j<34;j++)value.limb[j]=limbs[34*i+j];
        value.negative=signs[i]!=0u;ulong raw=ci_py_to_double(value);
        out[2*i]=raw;out[2*i+1]=ulong((raw&0x7ff0000000000000UL)==0x7ff0000000000000UL);''',
        compile_options={'math_mode':'safe'})
    (actual,)=kernel(inputs=[mx.array(limbs.reshape(-1)),mx.array(signs)],template=[('COUNT',len(values))],
        grid=(len(values),1,1),threadgroup=(128,1,1),output_shapes=[(len(values),2)],output_dtypes=[mx.uint64])
    mx.eval(actual)
    np.testing.assert_array_equal(np.asarray(actual),np.array(expected,np.uint64))
