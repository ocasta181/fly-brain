"""Independent CPU validation of reusable relative-window expression graphs."""
import numpy as np
import pytest


@pytest.mark.metal
def test_compiled_mixed_windows_keep_absolute_clock_events_fresh_seeds_and_continuation(tmp_path):
    import gc
    import weakref
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.model import resolve_model,DEFAULT_EQUATIONS
    from mlx_backend.models import Connectome,Stimulus
    from mlx_backend.configured import prepare,run_prepared,host_state,initial_state,advance,evaluate_state
    from mlx_backend.expression_runtime import snapshots
    from mlx_backend.expression_general import _prefix
    from mlx_backend.expression_window import _PLANS,_OWNERS,_MAX_PLANS
    from test_mlx_cpp_expression_rng import cpu_reference
    n=65536;steps=79;seeds=(81,827);ids=(2,7)
    model=resolve_model(dict(eqs=DEFAULT_EQUATIONS+'\nstamp:second',eq_th='rand()<(.001+int(i==0))',
        eq_rst='stamp=t;v=v_rst+randn()*mV;g=0*mV',t_rfc=.2*b.ms),dt_ms=.1,random_target='cpp')
    graph=Connectome(np.arange(n),np.array([0,0,1,2],np.int32),np.array([n-1,n-2,n-1,n-3],np.int32),
        np.array([3,-2,7,-1]),np.zeros(4))
    events=np.zeros((2,steps,2),np.uint8)
    events[:,[2,18,38,62,78],0]=1;events[:,[1,20,37,63],1]=1
    stimulus=Stimulus(events,(n-1,n-2),(0.,0.),ids,81,0,None,.1,(.25,-.5))
    p=prepare(graph,stimulus,model);assert _prefix(p.program)is not None
    from mlx_backend.expression_single_call import selected_primitive
    assert selected_primitive(p.program,'reset')=='randn'
    names=tuple(host_state(initial_state(p,2),p))
    expected=[cpu_reference(model,tmp_path/f'cpp-default-{seed}',names,steps=steps,
        normal_used=True,neurons=n,seed=seed,connectome=graph,stimulus=stimulus)for seed in seeds]
    starts=[np.random.RandomState(seed).get_state()for seed in seeds]
    def check(state,order):
        for row,reference_row in enumerate(order):
            _,values,final=expected[reference_row]
            for name,value in host_state(state,p).items():
                np.testing.assert_array_equal(value[row:row+1],values[name],err_msg=name)
            snapshot=snapshots(state.random)[row];future=np.random.RandomState();future.set_state(snapshot)
            np.testing.assert_array_equal(future.randint(0,2**32,624,dtype=np.uint32),final[0])
            assert snapshot[3:]==final[1:]
    result,state=run_prepared(p,stimulus,return_state=True,random_states=starts,chunk_steps=36)
    check(state,(0,1))
    for row,(spikes,_,_)in enumerate(expected):
        selected=result.spikes.trials==ids[row]
        np.testing.assert_array_equal(result.spikes.neurons[selected],spikes[0])
        np.testing.assert_array_equal(result.spikes.steps[selected],spikes[1])
    count=len([key for key in _PLANS if key[0]==id(p)])
    # Startup, repeated complete windows, and the final seven steps.
    assert count==3
    _,off=run_prepared(p,stimulus,return_state=True,random_states=starts[::-1],record=False,chunk_steps=36)
    check(off,(1,0));assert len([key for key in _PLANS if key[0]==id(p)])==count
    partial=initial_state(p,2,random_states=starts);device_events=mx.array(events.reshape(-1))
    for stop in(1,17,19,37,43,79):
        while partial.step<stop:
            partial,_=advance(p,partial,device_events,steps,min(stop-partial.step,p.window-partial.step%p.window))
        evaluate_state(partial)
    check(partial,(0,1))
    assert len(_PLANS)<=_MAX_PLANS
    identity=id(p);reference=weakref.ref(p)
    del p;gc.collect()
    assert reference()is None
    assert identity not in _OWNERS
    assert not any(key[0]==identity for key in _PLANS)


@pytest.mark.metal
def test_compiled_numpy_mixed_windows_keep_target_normal_cache_and_trial_order():
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome,Stimulus
    from mlx_backend.configured import prepare,run_prepared,host_state
    from mlx_backend.expression_runtime import snapshots
    from configured_reference import reference
    n=65536;steps=19
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\na:1',eq_th='rand()<.001',
        eq_rst='a+=randn();v=v_rst;g=0*mV'),dt_ms=.1,random_target='numpy')
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    stimulus=Stimulus(np.zeros((2,steps,0),np.uint8),(),(),(2,7),81,0,None,.1)
    starts=[np.random.RandomState(seed).get_state()for seed in(81,827)]
    expected=reference(graph,stimulus,model,random_states=starts)
    p=prepare(graph,stimulus,model)
    result,state=run_prepared(p,stimulus,return_state=True,random_states=starts,chunk_steps=18)
    for name in('trials','neurons','steps'):
        np.testing.assert_array_equal(getattr(result.spikes,name),getattr(expected.spikes,name))
    for name,value in host_state(state,p).items():np.testing.assert_array_equal(value,expected.state[name],err_msg=name)
    for actual,final in zip(snapshots(state.random),expected.random_states):
        np.testing.assert_array_equal(actual[1],final[1]);assert actual[2:]==final[2:]
