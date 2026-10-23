"""Independent Brian checks of fused pure phases before selected random reset."""
import numpy as np
import pytest

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('target',['cpp','cython'])
@pytest.mark.parametrize('delay',[0,2,18])
def test_fused_prefix_keeps_delayed_arrivals_refractory_resets_and_trials(target,delay,tmp_path,neurons=257,threshold='rand()<.13',steps=37,independent=False):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome,Stimulus
    from mlx_backend.configured import prepare,initial_state,run_prepared,advance,evaluate_state,host_state
    from mlx_backend.expression_runtime import snapshots
    from mlx_backend.expression_general import _prefix
    from test_mlx_cpp_expression_rng import cpu_reference as cpp_reference
    from test_mlx_cython_expression_rng import cpu_reference as cython_reference,assert_snapshot
    n=neurons;ids=(2,7);seeds=(81,827)
    model=resolve_model(dict(eq_th=threshold,eq_rst='v=v_rst+randn()*mV;g=0*mV',
        t_rfc=.2*b.ms,t_dly=delay*.1*b.ms),dt_ms=.1,random_target=target)
    graph=(Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
        if independent else Connectome(np.arange(n),np.array([0,1,2,0],np.int32),np.array([2,2,3,4],np.int32),
        np.array([3,-2,7,-1]),np.zeros(4)))
    stimulus=Stimulus(np.zeros((2,steps,0),np.uint8),(),(),ids,81,0,None,.1)
    p=prepare(graph,stimulus,model)
    assert _prefix(p.program)is not None
    from mlx_backend.expression_single_call import selected_primitive
    assert selected_primitive(p.program,'reset')=='randn'
    names=tuple(host_state(initial_state(p,2),p));expected=[];starts=[]
    for seed in seeds:
        directory=tmp_path/f'{target}-default-{seed}'
        if target=='cpp':
            events,values,final=cpp_reference(model,directory,names,steps=steps,normal_used=threshold!='rand()<0',
                neurons=n,seed=seed,connectome=None if independent else graph)
            start=np.random.RandomState(seed).get_state()
        else:
            start,events,values,final,error=cython_reference(model,directory,names,steps=steps,
                neurons=n,seed=seed,connectome=None if independent else graph)
            assert error is None
        starts.append(start);expected.append((events,values,final))
    result,state=run_prepared(p,stimulus,return_state=True,random_states=starts,chunk_steps=19)
    for row,(events,values,final)in enumerate(expected):
        selected=result.spikes.trials==ids[row]
        np.testing.assert_array_equal(result.spikes.neurons[selected],events[0])
        np.testing.assert_array_equal(result.spikes.steps[selected],events[1])
        for name,value in host_state(state,p).items():
            np.testing.assert_array_equal(value[row:row+1],values[name],err_msg=name)
        snapshot=snapshots(state.random)[row]
        if target=='cython':assert_snapshot(snapshot,final)
        else:
            future=np.random.RandomState();future.set_state(snapshot)
            np.testing.assert_array_equal(future.randint(0,2**32,624,dtype=np.uint32),final[0])
            assert snapshot[3:]==final[1:]
    _,off=run_prepared(p,stimulus,return_state=True,random_states=starts,record=False,chunk_steps=19)
    partial=initial_state(p,2,random_states=starts);events=mx.zeros((1,),mx.uint8)
    for stop in sorted({min(steps,stop)for stop in(1,7,19,steps)}):
        while partial.step<stop:
            partial,_=advance(p,partial,events,steps,min(stop-partial.step,p.window-partial.step%p.window))
        evaluate_state(partial)
    for other in(off,partial):
        for name,value in host_state(state,p).items():
            np.testing.assert_array_equal(host_state(other,p)[name],value,err_msg=name)
        for row,(actual,expected_snapshot)in enumerate(zip(snapshots(other.random),snapshots(state.random))):
            if target=='cython':assert_snapshot(actual,expected_snapshot)
            else:
                np.testing.assert_array_equal(actual[1],expected_snapshot[1]);assert actual[2:]==expected_snapshot[2:]


@pytest.mark.parametrize('threshold',['rand()<0','rand()<1','rand()<.5'],ids=['none','all','mixed'])
def test_cpp_fused_selection_keeps_stable_ranks_across_warps_blocks_and_trials(threshold,tmp_path):
    test_fused_prefix_keeps_delayed_arrivals_refractory_resets_and_trials('cpp',2,tmp_path,
        neurons=512,threshold=threshold,steps=3)


@pytest.mark.parametrize('threshold',['rand()<0','rand()<.5','rand()<1'],ids=['none','mixed','all'])
def test_canonical_threshold_block_ranks_preserve_reset_stream(threshold,tmp_path):
    test_fused_prefix_keeps_delayed_arrivals_refractory_resets_and_trials('cpp',2,tmp_path,
        neurons=512,steps=5,threshold=threshold,independent=True)


@pytest.mark.parametrize('force_tail',[False,True],ids=['sparse','native-tail'])
def test_normal_speculation_budget_preserves_complete_model_and_stream(force_tail,tmp_path,monkeypatch):
    from mlx_backend import expression_single_call
    assert expression_single_call._normal_candidate_budget('rand()<.001',65536)<1024
    if force_tail:
        monkeypatch.setattr(expression_single_call,'_normal_candidate_budget',lambda *_:1)
    test_fused_prefix_keeps_delayed_arrivals_refractory_resets_and_trials('cpp',0 if force_tail else 2,tmp_path,
        neurons=65536,threshold='rand()<.5'if force_tail else'rand()<.001',steps=3)


@pytest.mark.parametrize('target',['cpp','cython'])
def test_fused_prefix_freezes_failed_trial_before_pure_phases(target,tmp_path,neurons=3):
    from dataclasses import replace
    import mlx.core as mx
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome,Stimulus
    from mlx_backend.configured import prepare,initial_state,host_state,advance
    from mlx_backend.expression_runtime import snapshots,arrays
    from test_mlx_cpp_expression_rng import cpu_reference as cpp_reference
    from test_mlx_cython_expression_rng import cpu_reference as cython_reference,assert_snapshot
    n=neurons;steps=3
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second',eq_th='rand()<.5',
        eq_rst='v=v_rst+randn()*mV;g=0*mV'),dt_ms=.1,random_target=target)
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    stimulus=Stimulus(np.zeros((2,steps,0),np.uint8),(),(),(2,7),81,0,None,.1)
    p=prepare(graph,stimulus,model);names=tuple(host_state(initial_state(p,2),p))
    if target=='cpp':
        events,expected,final=cpp_reference(model,tmp_path/'cpp-default',names,steps=steps,
            normal_used=True,neurons=n,seed=827)
        starts=[np.random.RandomState(seed).get_state()for seed in(81,827)]
    else:
        start,events,expected,final,error=cython_reference(model,tmp_path/'cython-default',names,
            steps=steps,neurons=n,seed=827)
        assert error is None
        starts=[dict(start,numpy_state=np.random.RandomState(81).get_state()),start]
    state=initial_state(p,2,random_states=starts);before=host_state(state,p)
    first=snapshots(state.random)[0]
    state=replace(state,random=replace(state.random,errors=mx.array([2,0],mx.uint32)))
    state,bits=advance(p,state,mx.zeros((1,),mx.uint8),steps,steps)
    mx.eval(*state.values.values(),state.release,bits,*arrays(state.random))
    np.testing.assert_array_equal(np.asarray(bits)[:n],0)
    masks=np.asarray(bits)[n:];actual=sorted((k,i)for i,mask in enumerate(masks)for k in range(steps)if mask&(1<<k))
    assert actual==sorted(zip(events[1],events[0]))
    for name,value in host_state(state,p).items():
        np.testing.assert_array_equal(value[:1],before[name][:1],err_msg=name)
        np.testing.assert_array_equal(value[1:],expected[name],err_msg=name)
    frozen,live=snapshots(state.random)
    if target=='cython':
        assert_snapshot(frozen,first);assert_snapshot(live,final)
    else:
        np.testing.assert_array_equal(frozen[1],first[1]);assert frozen[2:]==first[2:]
        future=np.random.RandomState();future.set_state(live)
        np.testing.assert_array_equal(future.randint(0,2**32,624,dtype=np.uint32),final[0]);assert live[3:]==final[1:]


@pytest.mark.parametrize('target',['cpp','cython'])
def test_fused_selection_freezes_whole_failed_trial_groups(target,tmp_path):
    test_fused_prefix_freezes_failed_trial_before_pure_phases(target,tmp_path,neurons=512)


def test_cython_fused_selection_keeps_physical_buffers_and_stable_ranks(tmp_path):
    test_fused_prefix_keeps_delayed_arrivals_refractory_resets_and_trials('cython',2,tmp_path,
        neurons=512,threshold='rand()<.5',steps=3)


@pytest.mark.parametrize('delay',[0,18])
def test_cpp_uniform_prefix_view_exhaustion_matches_native_tail_and_continuation(delay,tmp_path,monkeypatch):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend import expression_rng_prefix
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome,Stimulus
    from mlx_backend.configured import prepare,initial_state,host_state,run_prepared,advance,evaluate_state
    from mlx_backend.expression_runtime import snapshots
    from test_mlx_cpp_expression_rng import cpu_reference
    original=expression_rng_prefix.prefix_pool
    monkeypatch.setattr(expression_rng_prefix,'prefix_pool',
        lambda state,capacity,**kwargs:original(state,19,checkpoint=7,parts=3))
    n=65536;steps=3
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second',eq_th='rand()<.001',
        eq_rst='v=v_rst+randn()*mV;g=0*mV',t_rfc=0*b.ms,t_dly=delay*.1*b.ms),dt_ms=.1,random_target='cpp')
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    stimulus=Stimulus(np.zeros((1,steps,0),np.uint8),(),(),(0,),81,0,None,.1)
    p=prepare(graph,stimulus,model);names=tuple(host_state(initial_state(p),p))
    events,expected,final=cpu_reference(model,tmp_path/'cpp-default',names,steps=steps,normal_used=True,neurons=n)
    starts=[np.random.RandomState(81).get_state()]
    result,state=run_prepared(p,stimulus,return_state=True,random_states=starts)
    np.testing.assert_array_equal(result.spikes.neurons,events[0]);np.testing.assert_array_equal(result.spikes.steps,events[1])
    for name,value in host_state(state,p).items():np.testing.assert_array_equal(value,expected[name],err_msg=name)
    snapshot=snapshots(state.random)[0];future=np.random.RandomState();future.set_state(snapshot)
    np.testing.assert_array_equal(future.randint(0,2**32,624,dtype=np.uint32),final[0]);assert snapshot[3:]==final[1:]
    assert state.random.pool.data.capacity==21
    assert int(np.asarray(state.random.pool.data.cursor)[0])==21
    partial=initial_state(p,random_states=starts);device_events=mx.zeros((1,),mx.uint8)
    for stop in(1,3):
        while partial.step<stop:
            partial,_=advance(p,partial,device_events,steps,min(stop-partial.step,p.window-partial.step%p.window))
        evaluate_state(partial)
    for name,value in host_state(partial,p).items():np.testing.assert_array_equal(value,expected[name],err_msg=name)
    checkpoint=snapshots(partial.random)[0]
    np.testing.assert_array_equal(checkpoint[1],snapshot[1]);assert checkpoint[2:]==snapshot[2:]


def test_cpp_uniform_prefix_view_keeps_failed_trial_input_mask(tmp_path):
    test_fused_prefix_freezes_failed_trial_before_pure_phases('cpp',tmp_path,neurons=65536)
