"""Original CPU integer ufuncs inside live random expression segments."""
import numpy as np
import pytest

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('dtype,zero', [(np.int32,False),(np.int64,False),(np.int32,True)])
def test_original_integer_segments(tmp_path, monkeypatch, dtype, zero, target='numpy',nested=False,zero_branch=False,integer_only=False):
    import brian2 as b
    from original_api_reference import build_original_api_fixture
    from original_api_protocol import check_original_api_runs, observe_run

    def build(mode):
        fixture=build_original_api_fixture(tmp_path/str(mode=='native'), parameters=dict(
            eqs='v:volt\ng:volt\nrfc:second\na:integer\nden:integer\nrem:integer\nquo:integer\ns=rand():1 (shared, constant over dt)\nx=poisson(2+(a%3)+s)'+('*int(i>1)'if zero_branch else'')+':1 (constant over dt)',
            eq_th='rand()<1',
            eq_rst=('rem=(poisson(2)+a)%den;quo=(a%3+2147483647)//den;'if nested else'rem=a%den;quo=a//den;')+'a=rem+poisson(2+x);'+(''if integer_only else'v=v_rst+(quo+randn())*mV;g=0*mV'),
            t_rfc=0*b.ms,r_poi=3700*b.Hz,r_poi2=7200*b.Hz,t_dly=.2*b.ms),
            exc=(0,0),exc2=(2,),silenced=(),
            initial_state={'a':[np.iinfo(dtype).min,np.iinfo(dtype).max,7 if np.dtype(dtype).kind=='u'else -7,7],
                'den':([np.iinfo(dtype).max,0 if zero else 3,3,2]if np.dtype(dtype).kind=='u'else [-3 if target=='cython'and dtype==np.int64 else -1,0 if zero else 3,-3,2])})
        return fixture

    def exercise(fixture, mode):
        for duration in (.2,0.,.3):
            yield observe_run(fixture,duration,rng=True)
    check_original_api_runs(build,exercise,monkeypatch,seed=941,target=target,integer_dtype=dtype)


@pytest.mark.parametrize('dtype,nested,zero_branch',[(np.int32,False,False),(np.int32,True,False),(np.int32,False,True),(np.int64,False,False)])
def test_original_cython_integer_segments(tmp_path,monkeypatch,dtype,nested,zero_branch):
    test_original_integer_segments(tmp_path,monkeypatch,dtype,False,target='cython',nested=nested,zero_branch=zero_branch)


@pytest.mark.parametrize('operation,overflow', [('%',False),('//',False),('//',True)])
def test_original_cython_integer_fault_and_repair(tmp_path,monkeypatch,operation,overflow):
    import brian2 as b
    from original_api_reference import build_original_api_fixture
    from original_api_protocol import check_original_api_runs, observe_run

    def build(mode):
        fixture=build_original_api_fixture(tmp_path/str(mode=='native'),parameters=dict(
            eqs='v:volt\ng:volt\nrfc:second\na:integer\nden:integer\nrem:integer',
            eq_th='rand()<1',
            eq_rst=f'rem=a{operation}den;a=poisson(2);v=v_rst+randn()*mV;g=0*mV',
            t_rfc=0*b.ms,r_poi=3700*b.Hz,r_poi2=7200*b.Hz,t_dly=.2*b.ms),
            silenced=(),initial_state={'a':[-7,7,-4,4],'den':[-3,2,3,-2]})
        return fixture

    def exercise(fixture, mode):
        yield observe_run(fixture,.2,rng=True)
        fixture.neurons.den[1]=-1 if overflow else 0
        if overflow:fixture.neurons.a[1]=np.iinfo(np.int64).min
        yield observe_run(fixture,.3,raises=b.BrianObjectException,rng=True)
        fixture.neurons.den=[-3,2,3,-2]
        yield observe_run(fixture,.3,rng=True)
    check_original_api_runs(build,exercise,monkeypatch,seed=941,target='cython',
        integer_dtype=np.int64 if overflow else np.int32)


@pytest.mark.parametrize('target,dtype',[('numpy',np.uint8),('numpy',np.uint64),('cython',np.uint64)])
def test_original_unsigned_integer_segments(tmp_path,monkeypatch,target,dtype):
    test_original_integer_segments(tmp_path,monkeypatch,dtype,False,target=target)


def test_original_cython_integer_only_reset(tmp_path,monkeypatch):
    test_original_integer_segments(tmp_path,monkeypatch,np.int32,False,
        target='cython',nested=True,integer_only=True)
