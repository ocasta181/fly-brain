"""Exact Cython physical RNG state through compiled TimedArray windows."""
import numpy as np
import pytest


@pytest.mark.metal
def test_cython_timed_windows_continuation_recording_and_trial_order(tmp_path):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome, Stimulus
    from mlx_backend.configured import prepare, run_prepared, host_state, initial_state, advance, evaluate_state
    from mlx_backend.expression_runtime import snapshots
    from mlx_backend.expression_window import _PLANS
    from test_mlx_cython_expression_rng import cpu_reference, assert_snapshot
    n=65536;steps=23
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\na:1',eq_th='rand()<.002',
        eq_rst='a=randn()+drive(t+i*dt)+offset(t-i*dt);v=v_rst+a*mV;g=0*mV',
        drive=b.TimedArray([0.,.5,-.25,.75],dt=.2*b.ms),
        offset=b.TimedArray([.125,-.5,1.],dt=.1*b.ms),t_rfc=0*b.ms),
        dt_ms=.1,random_target='cython')
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((2,steps,0),np.uint8),(),(),(2,7),81,3,None,.1)
    p=prepare(graph,inputs,model)
    names=tuple(host_state(initial_state(p,2),p))
    expected=[cpu_reference(model,tmp_path/f'cpu-{seed}',names,seed=seed,neurons=n,steps=steps)
              for seed in (81,827)]
    assert all(row[-1] is None for row in expected)
    starts=[row[0] for row in expected]
    def check(state,order):
        actual=host_state(state,p);random=snapshots(state.random)
        for row,index in enumerate(order):
            for name,value in actual.items():
                want=expected[index][2][name]
                assert value[row:row+1].tobytes()==want.tobytes(),name
            assert_snapshot(random[row],expected[index][3])
    result,state=run_prepared(p,inputs,return_state=True,random_states=starts,chunk_steps=19)
    check(state,(0,1))
    assert any(key[0]==id(p) for key in _PLANS)
    for row,trial in enumerate(inputs.trial_indices):
        selected=result.spikes.trials==trial
        np.testing.assert_array_equal(result.spikes.neurons[selected],expected[row][1][0])
        np.testing.assert_array_equal(result.spikes.steps[selected],expected[row][1][1])
    _,off=run_prepared(p,inputs,return_state=True,random_states=starts[::-1],record=False,chunk_steps=7)
    check(off,(1,0))
    partial=initial_state(p,2,random_states=starts);events=mx.array(inputs.events.reshape(-1))
    for stop in (1,7,19,23):
        while partial.step<stop:
            partial,_=advance(p,partial,events,steps,min(stop-partial.step,p.window-partial.step%p.window))
        evaluate_state(partial)
    check(partial,(0,1))


@pytest.mark.metal
def test_shared_window_namespace_arrays_are_explicit_graph_arguments():
    from dataclasses import replace
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome, Stimulus
    from mlx_backend.configured import prepare, initial_state, host_state, evaluate_state
    from mlx_backend.expression_runtime import snapshots
    from mlx_backend.expression_window import advance, _PLANS
    from mlx_backend.expression_general import _advance_steps
    from mlx_backend.expression_poisson_input import ExpressionPoisson
    from test_mlx_cython_expression_rng import assert_snapshot
    n=65536;steps=4
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\na:1',eq_th='rand()<.002',
        eq_rst='a=randn()+drive(t+i*dt);v=v_rst;g=0*mV',
        drive=b.TimedArray([0.,.5,-.25,.75],dt=.2*b.ms),t_rfc=0*b.ms),dt_ms=.1,random_target='cython')
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((1,steps,0),np.uint8),(),(),(0,),81,3,None,.1)
    p=prepare(graph,inputs,model)
    # Equal prepared metadata can share one graph, but its namespace pointers
    # must remain runtime inputs. Compare against the original per-step graph.
    alternate=replace(p,input_arrays=tuple(mx.array(np.full(a.shape,2.,np.float64).view(np.uint64))
                                          for a in p.input_arrays))
    poisson=ExpressionPoisson([],b.Clock(dt=.1*b.ms),'cython')
    events=mx.array([],mx.uint8);times=mx.array((np.arange(steps)*.0001).view(np.uint64))
    options=dict(event_begin=0,event_steps=steps,event_times=times,native_poisson=poisson)
    count=None
    equivalent=replace(p,program=replace(p.program))
    other_model=resolve_model(dict(eqs=model.eqs,eq_th='rand()<.003',eq_rst=model.eq_rst,
        drive=b.TimedArray([1.,-.5,.25,-.75],dt=.2*b.ms),t_rfc=0*b.ms),dt_ms=.1,random_target='cython')
    other=prepare(graph,inputs,other_model)
    for prepared,extra_plan in ((p,0),(equivalent,0),(alternate,0),(p,0),(other,1),(p,1)):
        start=initial_state(prepared,random_states=[np.random.RandomState(81).get_state()])
        expected,bits=_advance_steps(prepared,start,events,steps,steps,**options)
        actual,got=advance(prepared,start,events,steps,steps,**options)
        evaluate_state(actual);evaluate_state(expected)
        for name,value in host_state(actual,prepared).items():
            assert value.tobytes()==host_state(expected,prepared)[name].tobytes(),name
        np.testing.assert_array_equal(np.asarray(got),np.asarray(bits))
        assert_snapshot(snapshots(actual.random)[0],snapshots(expected.random)[0])
        size=len(_PLANS)
        if count is None:count=size
        else:assert size==count+extra_plan


@pytest.mark.metal
@pytest.mark.parametrize("n",[4,256])
def test_physical_prefix_asymmetric_normal_refills_and_exhaustion(tmp_path,monkeypatch,n):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome,Stimulus
    from mlx_backend.configured import prepare,initial_state,host_state,evaluate_state
    from mlx_backend.expression_runtime import snapshots
    from mlx_backend.expression_general import _advance_steps
    import mlx_backend.expression_window as window
    import mlx_backend.expression_rng_prefix as prefix_module
    from test_mlx_cython_expression_rng import cpu_reference,assert_snapshot
    monkeypatch.setattr(window,'_MIN_NEURONS',1)
    steps=7
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\na:1',eq_th='rand()<(.2+int(i==0))',
        eq_rst='a=randn()+drive(t+i*dt);v=v_rst;g=0*mV',
        drive=b.TimedArray([0.,.5,-.25,.75],dt=.2*b.ms),t_rfc=0*b.ms,t_dly=.3*b.ms),dt_ms=.1,random_target='cython')
    graph=Connectome(np.arange(n),np.array([0,0,1],np.int32),np.array([1,2,3],np.int32),np.array([3.,-2.,7.]),np.zeros(3))
    inputs=Stimulus(np.zeros((2,steps,0),np.uint8),(),(),(2,7),81,3,None,.1)
    p=prepare(graph,inputs,model);names=tuple(host_state(initial_state(p,2),p))
    expected=[cpu_reference(model,tmp_path/f'cpu-{seed}',names,seed=seed,neurons=n,steps=steps,
        normal_prefix=None if row==0 else np.linspace(-1,1,100),connectome=graph)for row,seed in enumerate((81,827))]
    assert all(row[-1] is None for row in expected)
    starts=[row[0]for row in expected]
    assert starts[0]['normal_index']==0 and starts[1]['normal_index']==1
    events=mx.array([],mx.uint8);times=mx.array((np.arange(steps)*.0001).view(np.uint64))
    from mlx_backend.expression_poisson_input import ExpressionPoisson
    import mlx_backend.expression_poisson_input as input_module
    kernel=input_module._cython_prefix_kernel;fused_routes=[];block_routes=[]
    def tracked_kernel(pending_normal=False,normal_blocks=False):
        fused_routes.append(pending_normal);block_routes.append(normal_blocks)
        return kernel(pending_normal,normal_blocks)if normal_blocks else kernel(pending_normal)
    monkeypatch.setattr(input_module,'_cython_prefix_kernel',tracked_kernel)
    import mlx_backend.configured as configured_module
    neuron_kernel=configured_module._neuron_kernel;rank_routes=[];due_routes=[];clock_routes=[]
    def tracked_neuron(*args,**kwargs):
        rank_routes.append(kwargs.get('expression_rank_flags',False))
        due_routes.append(kwargs.get('expression_due_fusion',False))
        clock_routes.append(kwargs.get('expression_full_clock',False))
        return neuron_kernel(*args,**kwargs)
    monkeypatch.setattr(configured_module,'_neuron_kernel',tracked_neuron)
    poisson=ExpressionPoisson((),b.Clock(dt=.1*b.ms),'cython')
    original_pool=prefix_module.prefix_pool
    for exhausted in (False,True):
        if exhausted:
            monkeypatch.setattr(prefix_module,'prefix_pool',lambda state,capacity,**kwargs:original_pool(state,2,checkpoint=1))
        state=initial_state(p,2,random_states=starts)
        plain=initial_state(p,2,random_states=starts)
        checkpoints=[];emissions=[]
        for step in range(steps):
            options=dict(event_begin=0,event_steps=steps,event_times=times,native_poisson=poisson)
            state,bits=window.advance(p,state,events,steps,1,**options)
            plain,_=_advance_steps(p,plain,events,steps,1,**options)
            evaluate_state(state);evaluate_state(plain)
            checkpoints.append(state);emissions.append(np.asarray(bits))
            for name,value in host_state(state,p).items():assert value.tobytes()==host_state(plain,p)[name].tobytes(),(step,name)
            for got,want in zip(snapshots(state.random),snapshots(plain.random)):assert_snapshot(got,want)
        for row in range(2):
            for name,value in host_state(state,p).items():assert value[row:row+1].tobytes()==expected[row][2][name].tobytes(),name
            assert_snapshot(snapshots(state.random)[row],expected[row][3])

        piped=initial_state(p,2,random_states=starts)
        for stop in (3,7):
            begin=piped.step
            piped,bits=window.advance(p,piped,events,steps,stop-begin,**options)
            evaluate_state(piped);expected_state=checkpoints[stop-1]
            for name,value in host_state(piped,p).items():assert value.tobytes()==host_state(expected_state,p)[name].tobytes(),(stop,name)
            for got,want in zip(snapshots(piped.random),snapshots(expected_state.random)):assert_snapshot(got,want)
            np.testing.assert_array_equal(np.asarray(bits),sum(emissions[k]<<(k-begin)for k in range(begin,stop)))
            for q in piped.due:np.testing.assert_array_equal(np.asarray(piped.due[q]),np.asarray(expected_state.due[q]))
            for q in piped.history:np.testing.assert_array_equal(np.asarray(piped.history[q]),np.asarray(expected_state.history[q]))

    assert any(clock_routes),"The original-clock pipeline must execute"
    assert 'history' in due_routes,"The packed history and due fusion must execute"
    assert any(fused_routes),"The physical-normal fusion must execute"
    if n==256:
        assert any(rank_routes),"The flagged local-rank path must execute"
        assert any(block_routes),"The fused normal-count scan must execute"


@pytest.mark.metal
@pytest.mark.parametrize("steps,compiled",[(11,True),(67,True),(67,False)])
def test_wide_cython_submission_crosses_signed_delay_boundaries(tmp_path,monkeypatch,steps,compiled):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome,Stimulus
    from mlx_backend.configured import prepare,initial_state,advance,host_state,evaluate_state
    from mlx_backend.expression_runtime import snapshots
    import mlx_backend.expression_window as window
    from test_mlx_cython_expression_rng import cpu_reference,assert_snapshot
    monkeypatch.setattr(window,'_MIN_NEURONS',1)
    if not compiled:monkeypatch.setattr(window,'advance',lambda *args,**kwargs:None)
    n=4
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\na:1',eq_th='rand()<(.2+int(i==0))',
        eq_rst='a=randn()+drive(t+i*dt);v=v_rst;g=0*mV',
        drive=b.TimedArray([0.,.5,-.25,.75],dt=.2*b.ms),t_rfc=0*b.ms,t_dly=.3*b.ms),dt_ms=.1,random_target='cython')
    sources=np.array([0,0,1,2],np.int32);destinations=np.array([1,2,3,3],np.int32)
    graph=Connectome(np.arange(n),sources,destinations,np.array([3.,-2.,7.,-1.]),np.zeros(4))
    inputs=Stimulus(np.zeros((1,steps,0),np.uint8),(),(),(0,),81,3,None,.1)
    p=prepare(graph,inputs,model);assert p.window==3
    names=tuple(host_state(initial_state(p),p))
    start,spikes,expected,final,error=cpu_reference(model,tmp_path,names,neurons=n,steps=steps,connectome=graph)
    assert error is None
    events=mx.array([],mx.uint8)
    wide=initial_state(p,random_states=[start]);plain=initial_state(p,random_states=[start])
    neurons=[];emissions=[]
    from mlx_backend.expression_poisson_input import ExpressionPoisson
    poisson=ExpressionPoisson((),b.Clock(dt=.1*b.ms),'cython')if steps>32 else None
    for stop in ((1,64,67)if steps>32 else(1,8,11)):
        begin=wide.step
        wide,bits=advance(p,wide,events,steps,stop-begin,native_poisson=poisson)
        while plain.step<stop:plain,_=advance(p,plain,events,steps,1,native_poisson=poisson)
        evaluate_state(wide);evaluate_state(plain)
        for name,value in host_state(wide,p).items():assert value.tobytes()==host_state(plain,p)[name].tobytes(),(stop,name)
        assert_snapshot(snapshots(wide.random)[0],snapshots(plain.random)[0])
        for q in wide.history:np.testing.assert_array_equal(np.asarray(wide.history[q]),np.asarray(plain.history[q]))
        for q in wide.due:np.testing.assert_array_equal(np.asarray(wide.due[q]),np.asarray(plain.due[q]))
        packed=np.asarray(bits).reshape(-1,n)
        for bit in range(stop-begin):
            rows=np.flatnonzero(packed[bit//32]&(1<<(bit%32)));neurons.extend(rows);emissions.extend([begin+bit]*len(rows))
    np.testing.assert_array_equal(neurons,spikes[0]);np.testing.assert_array_equal(emissions,spikes[1])
    for name,value in host_state(wide,p).items():assert value.tobytes()==expected[name].tobytes(),name
    assert_snapshot(snapshots(wide.random)[0],final)


def test_pending_namespace_arithmetic_stays_inside_guards():
    from mlx_backend.expression_single_call import _namespace_arithmetic
    call = 'd_to_int32_sat(expr_fma(t,scale,offset))'
    source = f'{{draw=compiled_normal(rng);if(!error){{a={call}+{call};}}}}'
    result = _namespace_arithmetic(source)
    assert result.index('compiled_normal(rng)') < result.index('if(!error){') < result.index('const auto')
    assert result.count('expr_fma(') == 1
    # Separate writes must not share a stale value across a changed operand.
    separate = f'{{a={call};t=next_t;}}if(selected){{b={call};}}'
    assert _namespace_arithmetic(separate) == separate
    for unsafe in ('compiled_normal(rng)', 'array[i]', 'checked(t,error)',
                   'as_type<Unknown>(t)', 'condition?t:other'):
        expression = f'expr_fma({unsafe},scale,offset)'
        guarded = f'if(selected){{a={expression}+{expression};}}'
        assert _namespace_arithmetic(guarded) == guarded
