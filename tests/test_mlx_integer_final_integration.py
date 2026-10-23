"""Accepted integer component integrations against the actual Brian CPU API."""
import brian2 as b
import numpy as np
import pytest
from original_api_protocol import check_original_api_runs, observe_run
from test_mlx_compiled_integer_adapter import _typed_fixture,_forbid_cpu_steps,private_compiler_cache


def check(reset,initial,monkeypatch,*,record=True,threshold='True',durations=(.1,.2)):
    types={name:value.dtype for name,value in initial.items()}
    def exercise(fixture, mode):
        first=observe_run(fixture,durations[0],raises=b.BrianObjectException,optional=True)
        yield first
        if first.error is None:
            yield observe_run(fixture,durations[1])
    check_original_api_runs(lambda mode:_typed_fixture(types,reset,initial,threshold=threshold,
        record=record,default_reset=False),exercise,monkeypatch,guard=_forbid_cpu_steps,
        target='cython',integer_dtype=np.int64)


def seeds():
    return dict(x=np.array([2**53+1,3,0,1,-1,-1923,2575,2,3,3,-5,7],np.int64),
        y=np.array([1,100,0,64,2048,-1,-2,1023,7,-3,4,2],np.int64),
        z=np.array([1,3,1,-1,0,2575,-1923,2,2,-5,7,-2],np.int64),
        u=np.array([1,-100,128,2048,0,-2,-1,-1074,6,2,-3,5],np.int64),
        seen=np.zeros(12))


@pytest.mark.metal
@pytest.mark.parametrize('operation',['+','*'])
def test_distinct_nonnegative_integer_powers(operation,monkeypatch):
    check('seen=abs(x)**y'+operation+'abs(z)**u',seeds(),monkeypatch)


@pytest.mark.metal
@pytest.mark.parametrize('base',['abs(x)','-abs(x)'])
@pytest.mark.parametrize('dtype',[np.float64,np.int64,np.bool_])
def test_common_base_difference_preserves_huge_cancellation(base,dtype,monkeypatch):
    initial=dict(x=np.array([-2**63,2,3,2,0,0,1,-1,1923,-2575,2,3],np.int64),
        y=np.array([2,4096,2048,62,0,7,2**63-1,2**63-2,-1,-2,1023,-3],np.int64),
        z=np.array([2,4096,2048,61,7,0,2**63-2,2**63-1,-2,-1,1023,-17],np.int64),
        seen=np.zeros(12,dtype))
    check(f'seen=({base})**y-({base})**z',initial,monkeypatch,record=dtype!=np.bool_)


@pytest.mark.metal
@pytest.mark.parametrize('op',['%','//','/'])
def test_contextual_integer_power_division(op,monkeypatch):
    initial=dict(x=np.array([2,-3,0,1,-1,1923,2575,2,3,-3,5,7],np.int64),
        y=np.array([61,17,0,2**63-1,2**63-2,-1,-2,120,3,-3,0,2],np.int64),
        z=np.array([3,-17,7,-2,19,3,-3,2**62,7,-11,1,2],np.int64),seen=np.zeros(12))
    check('seen=abs(x)**y'+op+'z',initial,monkeypatch)


@pytest.mark.metal
def test_composed_division_preserves_earlier_result_and_continuation(monkeypatch):
    # An earlier remainder stays live across two wide numeric calls. This
    # directly affected model fraction exposed a whole-SIMD-group zero result.
    n=32768
    initial=dict(x=np.resize(np.array([2,-3,0,1,-1,1923,2575,2,3,-3,5,7],np.int64),n),
        y=np.resize(np.array([61,17,0,2**63-1,2**63-2,-1,-2,62,3,-3,0,2],np.int64),n),
        z=np.resize(np.array([3,-17,7,-2,19,3,-3,2**62,7,-11,1,2],np.int64),n),
        f=np.zeros(n),q=np.zeros(n),r=np.zeros(n))
    check('f=abs(x)**y%z;q=abs(x)**y//z;r=abs(x)**y/z',initial,monkeypatch,
          record=False,durations=(1.6,.1))


@pytest.mark.metal
@pytest.mark.parametrize('reset,x,y',[
    ('seen=abs(x)**y+abs(y)**x',2,1024),
    ('seen=abs(x)**y-abs(x)**x',0,-1),
    ('seen=abs(x)**y*abs(y)**(-x)',2,1024),
])
def test_narrow_proofs_preserve_error_type_message_and_prefix(reset,x,y,monkeypatch):
    initial=dict(x=np.array([1,x,3],np.int64),y=np.array([1,y,2],np.int64),seen=np.zeros(3))
    check(reset,initial,monkeypatch)


def test_state_aliases_distinguish_member_roles():
    from mlx_backend.configured import _state_source
    assert _state_source('negative=big.negative; limb=big.limb[0]; f(negative);',
        {'negative':'user_negative','limb':'user_limb'})==(
        'user_negative=big.negative; user_limb=big.limb[0]; f(user_negative);')


@pytest.mark.metal
def test_inferred_vector_local_keeps_scope_and_mask(monkeypatch):
    initial=dict(x=np.array([-2**63,-2**63+1,-2**40-1,-2**31-1,-1,0,1,2**31-1,2**32+1,2**40+1,2**63-1],np.int64),
        count=np.full(11,17,np.int64),flag=np.zeros(11,bool))
    check('tmp=int(x);flag=tmp<0;count+=4294967297+int(flag);x=(x%1000003)*4099+8589934595',
          initial,monkeypatch,threshold='i%3!=1')


@pytest.mark.metal
def test_actual_cpp_inferred_vector_local_keeps_scope(tmp_path):
    from test_mlx_compiled_integer_adapter import _actual_cpp_component
    _actual_cpp_component(tmp_path,'vector_tmp')


@pytest.mark.metal
@pytest.mark.parametrize('mode',['power_invariant','power_mutable','power_constant','power_float32'])
def test_actual_cpp_cached_and_mutable_power_match_original(mode,tmp_path):
    from test_mlx_compiled_integer_adapter import _actual_cpp_component
    _actual_cpp_component(tmp_path,mode)


@pytest.mark.metal
@pytest.mark.parametrize('limbs',[3,33])
def test_bounded_power_and_sparse_multiplication_match_exact_integers(limbs):
    import mlx.core as mx
    from mlx_backend.compiled_python_integer import MSL_HEADER
    rng=np.random.default_rng(20261018)
    bases=[0,1,-1,2,-2,3,-3,2**53+1,-2**63]+list(map(int,rng.integers(-2**63,2**63-1,1015,dtype=np.int64)))
    exponents=np.resize(np.array([0,1,2,3,7,31,63,64,100,1023],np.uint64),len(bases))
    exponents[:3]=2**63-1
    limit=(1<<(32*limbs))-1
    expected=[]
    for base,exponent in zip(bases,map(int,exponents)):
        power=base**exponent
        magnitude=min(abs(power),limit)
        expected.append([(magnitude>>(32*j))&0xffffffff for j in range(limbs)]+[int(power<0)])
    kernel=mx.fast.metal_kernel(name='canonical_bounded_integer_power',input_names=['bases','signs','exponents'],output_names=['out'],header=MSL_HEADER,
        source='''uint i=thread_position_in_grid.x;if(i>=COUNT)return;
        auto base=ci_py_from<LIMBS>(bases[i],signs[i]!=0u);
        auto exponent=ci_py_from<LIMBS>(exponents[i],false);
        auto value=ci_py_pow_bounded(base,exponent);
        for(uint j=0;j<LIMBS;j++)out[i*(LIMBS+1)+j]=value.limb[j];
        out[i*(LIMBS+1)+LIMBS]=uint(value.negative);''',compile_options={'math_mode':'safe'})
    actual,=kernel(inputs=[mx.array(np.array(list(map(abs,bases)),np.uint64)),mx.array(np.array([v<0 for v in bases],np.uint32)),mx.array(exponents)],
        template=[('COUNT',len(bases)),('LIMBS',limbs)],grid=(len(bases),1,1),threadgroup=(128,1,1),output_shapes=[(len(bases),limbs+1)],output_dtypes=[mx.uint32])
    mx.eval(actual);np.testing.assert_array_equal(np.asarray(actual),np.array(expected,np.uint32))
