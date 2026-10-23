"""Float32 storage and float64 stock RNG consumers, independently on CPU."""
import numpy as np
import pytest

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('target',['cpp','cython','numpy'])
def test_random_threshold_sequential_float32_reset_matches_stock(target,tmp_path):
    _float32_case(target,tmp_path)


def _float32_case(target,tmp_path,*,reset='a=randn();v=v_rst+a*mV;g=0*mV',extra=''):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome,Stimulus
    from mlx_backend.configured import prepare,initial_state,host_state,run_prepared,advance,evaluate_state
    from mlx_backend.expression_runtime import snapshots
    from test_mlx_cpp_expression_rng import cpu_reference as cpp
    from test_mlx_cython_expression_rng import cpu_reference as cython,assert_snapshot
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\na:1'+extra,eq_th='rand()<.5',
        eq_rst=reset,t_rfc=0*b.ms),
        state_dtype='float32',dt_ms=.1,random_target=target)
    n=7;steps=5
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((1,steps,0),np.uint8),(),(),(2,),81,0,None,.1)
    p=prepare(graph,inputs,model);names=tuple(host_state(initial_state(p),p))
    if target=='cpp':
        spikes,expected,final=cpp(model,tmp_path/'cpp-default',names,steps=steps,neurons=n,seed=81,normal_used=True)
        start=np.random.RandomState(81).get_state()
    else:
        start,spikes,expected,final,error=cython(model,tmp_path/(target+'-default'),names,steps=steps,
            neurons=n,seed=81,codegen_target=target)
        assert error is None
        if target=='numpy':start=start['numpy_state'];final=final['numpy_state']
    run,state=run_prepared(p,inputs,return_state=True,random_states=[start])
    np.testing.assert_array_equal(run.spikes.neurons,spikes[0]);np.testing.assert_array_equal(run.spikes.steps,spikes[1])
    for name,value in host_state(state,p).items():
        assert value.dtype==expected[name].dtype
        np.testing.assert_array_equal(value.view(np.uint8),expected[name].view(np.uint8),err_msg=name)
    snapshot=snapshots(state.random)[0]
    if target=='cython':assert_snapshot(snapshot,final)
    elif target=='cpp':
        future=np.random.RandomState();future.set_state(snapshot)
        np.testing.assert_array_equal(future.randint(0,2**32,624,dtype=np.uint32),final[0]);assert snapshot[3:]==final[1:]
    else:np.testing.assert_array_equal(snapshot[1],final[1]);assert snapshot[2:]==final[2:]
    _,off=run_prepared(p,inputs,record=False,return_state=True,random_states=[start])
    partial=initial_state(p,random_states=[start])
    for stop in(1,3,5):
        partial,_=advance(p,partial,mx.zeros((1,),mx.uint8),steps,stop-partial.step)
        evaluate_state(partial)
    for other in(off,partial):
        for name,value in host_state(state,p).items():np.testing.assert_array_equal(host_state(other,p)[name],value,err_msg=name)
        actual=snapshots(other.random)[0]
        if target=='cython':assert_snapshot(actual,final)
        elif target=='cpp':
            future=np.random.RandomState();future.set_state(actual)
            np.testing.assert_array_equal(future.randint(0,2**32,624,dtype=np.uint32),final[0]);assert actual[3:]==final[1:]
        else:np.testing.assert_array_equal(actual[1],final[1]);assert actual[2:]==final[2:]


def test_numpy_float32_locals_survive_later_draws_and_inplace_updates(tmp_path):
    _float32_case('numpy',tmp_path,extra='\nb:1',
        reset='a=randn();b=randn()+a;a+=.13;b+=a;v=v_rst+b*mV;g=0*mV')


def test_cpp_float32_nested_poisson_temporaries_and_continuation(tmp_path):
    _float32_case('cpp',tmp_path,extra='\nb:1',
        reset='a=poisson(2+rand());b=poisson(a+1)+randn();a+=.13;b+=a;v=v_rst+b*mV;g=0*mV')
