"""Actual Brian references for fixed draws on the current packed runtime."""
import numpy as np
import pytest

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('target',['cpp','cython'])
@pytest.mark.parametrize('function',['rand','randn'])
def test_fixed_uniform_threshold_uses_canonical_packed_state_and_continues(target,function,tmp_path):
    _fixed_threshold_case(target,function,tmp_path)


def _fixed_threshold_case(target,function,tmp_path,*,boxed=False):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    from mlx_backend.models import Connectome,Stimulus
    from mlx_backend.configured import prepare,initial_state,run_prepared,host_state,advance,evaluate_state,_PackedValues
    from mlx_backend.expression_runtime import snapshots
    from test_mlx_cpp_expression_rng import cpu_reference as cpp_reference
    from test_mlx_cython_expression_rng import cpu_reference as cython_reference,assert_snapshot
    n=257;steps=43;ids=(2,7);seeds=(81,827)
    threshold='rand()<.01'if function=='rand'else'randn()>1.7'
    params=dict(eq_th=threshold,eq_rst='v=v_rst+.3*mV;g=0*mV',t_rfc=2.2*b.ms)
    if boxed:
        params.update(eqs=DEFAULT_EQUATIONS+'\nx:integer\ny:integer\nz:integer\nf:1\nq:1\nr:1',
            eq_th='rand()<.5',eq_rst='x=2+i%2;y=61-i%5;z=3+i%4;'
            'f=abs(x)**y%z;q=abs(x)**y//z;r=abs(x)**y/z;v=v_rst+.3*mV;g=0*mV')
    model=resolve_model(params,dt_ms=.1,random_target=target)
    graph=Connectome(np.arange(n),np.array([0,1,2,0],np.int32),np.array([2,2,3,4],np.int32),
        np.array([3,-2,7,-1]),np.zeros(4))
    inputs=Stimulus(np.zeros((2,steps,0),np.uint8),(),(),ids,81,0,None,.1)
    p=prepare(graph,inputs,model);initial=initial_state(p,2)
    assert isinstance(initial.values,_PackedValues)
    names=tuple(host_state(initial,p));expected=[];starts=[]
    for seed in seeds:
        if target=='cpp':
            events,values,final=cpp_reference(model,tmp_path/f'cpp-default-{seed}',names,steps=steps,
                neurons=n,seed=seed,connectome=graph,normal_used=function=='randn')
            start=np.random.RandomState(seed).get_state()
        else:
            start,events,values,final,error=cython_reference(model,tmp_path/f'cython-default-{seed}',names,
                steps=steps,neurons=n,seed=seed,connectome=graph)
            assert error is None
        expected.append((events,values,final));starts.append(start)
    run,state=run_prepared(p,inputs,random_states=starts,return_state=True,chunk_steps=19)
    assert isinstance(state.values,_PackedValues)
    for row,(events,values,final)in enumerate(expected):
        selected=run.spikes.trials==ids[row]
        np.testing.assert_array_equal(run.spikes.neurons[selected],events[0])
        np.testing.assert_array_equal(run.spikes.steps[selected],events[1])
        for name,value in host_state(state,p).items():np.testing.assert_array_equal(value[row:row+1],values[name],err_msg=name)
        snapshot=snapshots(state.random)[row]
        if target=='cython':assert_snapshot(snapshot,final)
        else:
            future=np.random.RandomState();future.set_state(snapshot)
            np.testing.assert_array_equal(future.randint(0,2**32,624,dtype=np.uint32),final[0]);assert snapshot[3:]==final[1:]
    _,off=run_prepared(p,inputs,random_states=starts,return_state=True,record=False,chunk_steps=19)
    partial=initial_state(p,2,random_states=starts);events=mx.zeros((1,),mx.uint8)
    for stop in(1,7,19,43):
        while partial.step<stop:
            partial,_=advance(p,partial,events,steps,min(stop-partial.step,p.window-partial.step%p.window))
        evaluate_state(partial)
    for other in(off,partial):
        assert isinstance(other.values,_PackedValues)
        assert other.program_identity==state.program_identity
        for name,value in host_state(state,p).items():np.testing.assert_array_equal(host_state(other,p)[name],value,err_msg=name)
        for actual,expected_snapshot in zip(snapshots(other.random),snapshots(state.random)):
            if target=='cython':assert_snapshot(actual,expected_snapshot)
            else:np.testing.assert_array_equal(actual[1],expected_snapshot[1]);assert actual[2:]==expected_snapshot[2:]


def test_fixed_rng_threshold_composes_with_boxed_integer_resets(tmp_path):
    _fixed_threshold_case('cython','rand',tmp_path,boxed=True)


@pytest.mark.parametrize('function',['rand','randn'])
def test_fixed_numpy_stream_matches_independent_brian(function):
    import brian2 as b
    from configured_reference import reference as run_reference
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome,Stimulus
    from mlx_backend.configured import prepare,run_prepared,host_state
    from mlx_backend.expression_runtime import snapshots
    n=33;steps=17;seeds=(81,827);ids=(2,7)
    model=resolve_model(dict(eq_th='rand()<.1'if function=='rand'else'randn()>1.3',
        eq_rst='v=v_rst+.3*mV;g=0*mV',t_rfc=2.2*b.ms),dt_ms=.1,random_target='numpy')
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((2,steps,0),np.uint8),(),(),ids,81,0,None,.1)
    ambient=np.random.get_state();expected=[]
    try:
        for seed,trial in zip(seeds,ids):
            b.seed(seed)
            single=Stimulus(np.zeros((1,steps,0),np.uint8),(),(),(trial,),81,0,None,.1)
            reference=run_reference(graph,single,model)
            expected.append((reference,np.random.get_state()))
    finally:
        np.random.set_state(ambient)
    p=prepare(graph,inputs,model)
    result,state=run_prepared(p,inputs,return_state=True,
        random_states=[np.random.RandomState(seed).get_state()for seed in seeds])
    for row,(reference,final)in enumerate(expected):
        selected=result.spikes.trials==ids[row]
        np.testing.assert_array_equal(result.spikes.neurons[selected],reference.spikes.neurons)
        np.testing.assert_array_equal(result.spikes.steps[selected],reference.spikes.steps)
        for name,value in host_state(state,p).items():
            np.testing.assert_array_equal(value[row:row+1],reference.state[name],err_msg=name)
        snapshot=snapshots(state.random)[row]
        np.testing.assert_array_equal(snapshot[1],final[1]);assert snapshot[2:]==final[2:]


@pytest.mark.parametrize('delay',[0,3])
def test_general_numpy_random_phases_keep_canonical_arrivals_and_streams(delay):
    import brian2 as b
    from configured_reference import reference
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome,Stimulus
    from mlx_backend.configured import prepare,run_prepared,host_state,_PackedValues
    from mlx_backend.expression_runtime import snapshots
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\na:1\nb:1',
        eq_th='(i<2) and ((rand()<.35) or (rand()<.7))',
        eq_rst='a=rand();b=randn()+a;v=v_rst+a*mV;g=0*mV',
        t_dly=delay*.1*b.ms),dt_ms=.1,random_target='numpy')
    graph=Connectome(np.arange(3),np.array([0,1,1,2],np.int32),np.array([2,2,0,1],np.int32),
        np.array([2,-3,7,11]),np.zeros(4))
    events=np.zeros((2,7,1),np.uint8);events[:,[0,2,5],0]=1
    inputs=Stimulus(events,(2,),(0.,),(2,7),81,0,None,.1)
    starts=[np.random.RandomState(seed).get_state()for seed in(81,827)]
    cpu=reference(graph,inputs,model,random_states=starts)
    p=prepare(graph,inputs,model)
    native,state=run_prepared(p,inputs,return_state=True,random_states=starts)
    assert isinstance(state.values,_PackedValues)
    for key in('trials','neurons','steps'):
        np.testing.assert_array_equal(getattr(native.spikes,key),getattr(cpu.spikes,key))
    for name,value in host_state(state,p).items():
        np.testing.assert_array_equal(value,cpu.state[name],err_msg=name)
    for actual,expected in zip(snapshots(state.random),cpu.random_states):
        np.testing.assert_array_equal(actual[1],expected[1]);assert actual[2:]==expected[2:]


@pytest.mark.parametrize('target',['cpp','cython'])
def test_general_compiled_random_phases_follow_actual_stock_stream_order(target,tmp_path):
    import brian2 as b
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome,Stimulus
    from mlx_backend.configured import prepare,initial_state,run_prepared,host_state,_PackedValues
    from mlx_backend.expression_runtime import snapshots
    from test_mlx_cpp_expression_rng import cpu_reference as cpp_reference
    from test_mlx_cython_expression_rng import cpu_reference as cython_reference,assert_snapshot
    n=3;steps=7;ids=(2,7);seeds=(81,827)
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\na:1\nb:1',
        eq_th='(i<2) and ((rand()<.35) or (rand()<.7))',
        eq_rst='a=rand();b=randn()+a;v=v_rst+a*mV;g=0*mV',t_dly=0*b.ms),
        dt_ms=.1,random_target=target)
    graph=Connectome(np.arange(n),np.array([0,1,1,2],np.int32),np.array([2,2,0,1],np.int32),
        np.array([2,-3,7,11]),np.zeros(4))
    inputs=Stimulus(np.zeros((2,steps,0),np.uint8),(),(),ids,81,0,None,.1)
    p=prepare(graph,inputs,model);names=tuple(host_state(initial_state(p,2),p))
    expected=[];starts=[]
    for seed in seeds:
        if target=='cpp':
            events,values,final=cpp_reference(model,tmp_path/f'cpp-default-{seed}',names,steps=steps,
                neurons=n,seed=seed,connectome=graph,normal_used=True)
            start=np.random.RandomState(seed).get_state()
        else:
            start,events,values,final,error=cython_reference(model,tmp_path/f'cython-default-{seed}',names,
                steps=steps,neurons=n,seed=seed,connectome=graph)
            assert error is None
        expected.append((events,values,final));starts.append(start)
    run,state=run_prepared(p,inputs,return_state=True,random_states=starts)
    assert isinstance(state.values,_PackedValues)
    for row,(events,values,final)in enumerate(expected):
        selected=run.spikes.trials==ids[row]
        np.testing.assert_array_equal(run.spikes.neurons[selected],events[0])
        np.testing.assert_array_equal(run.spikes.steps[selected],events[1])
        for name,value in host_state(state,p).items():np.testing.assert_array_equal(value[row:row+1],values[name],err_msg=name)
        snapshot=snapshots(state.random)[row]
        if target=='cython':assert_snapshot(snapshot,final)
        else:
            future=np.random.RandomState();future.set_state(snapshot)
            np.testing.assert_array_equal(future.randint(0,2**32,624,dtype=np.uint32),final[0]);assert snapshot[3:]==final[1:]
