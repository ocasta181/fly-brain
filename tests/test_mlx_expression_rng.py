"""Actual Brian CPU versus Metal expression-RNG state and event identity."""
from state_assertions import assert_spikes_equal
from dataclasses import replace
import numpy as np
import pytest
from configured_reference import reference
from mlx_backend.model import resolve_model
from mlx_backend.models import Connectome,Stimulus


@pytest.fixture(autouse=True)
def numpy_reference_target(tmp_path):
    import brian2 as b
    previous=b.prefs.codegen.target
    cache=b.prefs.codegen.runtime.cython.cache_dir
    b.prefs.codegen.target='numpy'
    b.prefs.codegen.runtime.cython.cache_dir=str(tmp_path/'cython-default')
    try:yield
    finally:
        b.prefs.codegen.target=previous
        b.prefs.codegen.runtime.cython.cache_dir=cache


def graph(n=3):
    return Connectome(np.arange(n),np.array([0,1,2,0],np.int32),
        np.array([1,2,0,2],np.int32),np.array([23.,-17.,11.,7.]),np.zeros(4))


def inputs(steps=43,trials=2):
    events=np.zeros((trials,steps,2),np.uint8)
    events[:,::7,0]=1;events[:,3::11,1]=1
    return Stimulus(events,(0,2),(0.,0.),tuple(range(trials)),81,3,None,.1)


def snapshots(trials=2):return tuple(np.random.RandomState(81+row).get_state()for row in range(trials))


def assert_streams(actual,expected):
    words=np.asarray(actual.words).reshape(len(expected),624)
    for row,e in enumerate(expected):
        np.testing.assert_array_equal(words[row],e[1])
        assert np.asarray(actual.position)[row]==e[2]
        assert np.asarray(actual.has_gauss)[row]==e[3]
        assert np.asarray(actual.gauss).view(np.float64)[row]==e[4]


@pytest.mark.metal
@pytest.mark.parametrize('case', ['threshold','reset','normal','poisson','subexpression','shared','equation','scalar_vector'])
def test_random_model_spikes_states_and_streams_match_independent_brian(case):
    from mlx_backend.configured import prepare,run_prepared,host_state
    base=resolve_model().eqs
    params={
        'threshold':{'eq_th':'(i != 1) and ((rand()<0.25) or (v>v_th))'},
        'reset':{'eqs':base+'a:1\nb:1\n','eq_th':'(rand()<0.35) or (v>v_th)',
                 'eq_rst':'a=rand(); b=rand()+a; v=v_rst+a*mV; g=0*mV'},
        'normal':{'eqs':base+'a:1\nb:1\n','eq_th':'randn()>0.8',
                  'eq_rst':'a=randn(); b=rand(); v=v_rst+a*mV; g=0*mV'},
        'poisson':{'eqs':base+'a:integer\nb:1\n','eq_th':'poisson(1+i)>2',
                   'eq_rst':'a=poisson(10+rand()*3); b=rand(); v=v_rst; g=0*mV'},
        'subexpression':{'eqs':base+'x=rand():1 (constant over dt)\na:1\n',
                         'eq_th':'x<0.3','eq_rst':'a=x;v=v_rst;g=0*mV'},
        'scalar_vector':{'eqs':base+'s=rand():1 (shared, constant over dt)\nx=randn()+s:1 (constant over dt)\na:1\n',
                         'eq_th':'x<0.3','eq_rst':'a=x;v=v_rst;g=0*mV'},
        'equation':{'eqs':base.replace('(v_0 - v + g)', '(v_0 - v + g + x*mV)')+'x=randn():1 (constant over dt)\na:1\n',
                    'eq_th':'(v>v_th) or (rand()<0.2)', 'eq_rst':'a=x;v=v_rst;g=0*mV'},
        'shared':{'eqs':base+'x=rand():1 (shared, constant over dt)\na:1\n',
                  'eq_th':'x<0.3','eq_rst':'a=x;v=v_rst;g=0*mV'},
    }[case]
    model=resolve_model(params,dt_ms=.1);stimulus=inputs();streams=snapshots()
    cpu=reference(graph(),stimulus,model,random_states=streams)
    p=prepare(graph(),stimulus,model)
    actual,state=run_prepared(p,stimulus,return_state=True,random_states=streams,chunk_steps=19)
    assert_spikes_equal(actual.spikes, cpu.spikes)
    for name,array in host_state(state,p).items():np.testing.assert_array_equal(array,cpu.state[name],err_msg=name)
    assert_streams(state.random.stream,cpu.random_states)


def test_shared_reset_rejection_matches_cpu():
    from mlx_backend.compiler import compile_model
    model=resolve_model({'eqs':resolve_model().eqs+'s:1 (shared)\n',
                         'eq_th':'False','eq_rst':'s=rand();v=v_rst;g=0*mV'},dt_ms=.1)
    with pytest.raises(Exception) as caught:
        reference(graph(),inputs(1),model,random_states=snapshots())
    assert isinstance(caught.value.__cause__,SyntaxError)
    assert 'Writing to scalar variable s not allowed' in str(caught.value.__cause__)
    with pytest.raises(SyntaxError,match='Writing to scalar variable s not allowed'):
        compile_model(model)


@pytest.mark.metal
@pytest.mark.parametrize('delay',[0,2,37])
def test_recording_off_and_chunk_continuation_preserve_rng_and_trial_identity(delay):
    import mlx.core as mx
    from mlx_backend.configured import prepare,run_prepared,initial_state,advance,evaluate_state,host_state
    import brian2 as b
    model=resolve_model({'eq_th':'rand()<0.3','eq_rst':'v=v_rst+randn()*mV;g=0*mV',
                         't_dly':delay*.1*b.ms},dt_ms=.1)
    stimulus=replace(inputs(83),trial_indices=(2**53+1,2**53+3))
    streams=snapshots();p=prepare(graph(),stimulus,model)
    full,state=run_prepared(p,stimulus,return_state=True,random_states=streams,chunk_steps=83)
    off,unrecorded=run_prepared(p,stimulus,return_state=True,record=False,random_states=streams,chunk_steps=19)
    assert off.spikes.steps.size==0
    for name,array in host_state(state,p).items():np.testing.assert_array_equal(array,host_state(unrecorded,p)[name])
    cpu=reference(graph(),stimulus,model,random_states=streams)
    assert_streams(unrecorded.random.stream,cpu.random_states)
    assert_spikes_equal(full.spikes, cpu.spikes)
    for name,array in host_state(state,p).items():np.testing.assert_array_equal(array,cpu.state[name],err_msg=name)
    partial=initial_state(p,2,random_states=streams)
    events=mx.array(stimulus.events.reshape(-1))
    for stop in (1,13,19,37,83):
        while partial.step<stop:
            partial,_=advance(p,partial,events,83,min(stop-partial.step,p.window-partial.step%p.window))
        evaluate_state(partial)
    for name,array in host_state(state,p).items():np.testing.assert_array_equal(array,host_state(partial,p)[name])
    np.testing.assert_array_equal(np.asarray(partial.random.stream.words),np.asarray(state.random.stream.words))


@pytest.mark.metal
def test_default_seed_keys_are_stable_across_batch_order_and_fresh_trials():
    from mlx_backend.configured import prepare,run_prepared,host_state
    model=resolve_model({'eq_th':'rand()<0.35','eq_rst':'v=v_rst+randn()*mV;g=0*mV'},dt_ms=.1)
    stimulus=replace(inputs(7),trial_indices=(2**53+1,3))
    p=prepare(graph(),stimulus,model)
    batch,state=run_prepared(p,stimulus,return_state=True)
    for row,trial in enumerate(stimulus.trial_indices):
        single=replace(stimulus,events=stimulus.events[row:row+1],trial_indices=(trial,))
        expected,one=run_prepared(p,single,return_state=True)
        selected=batch.spikes.trials==trial
        np.testing.assert_array_equal(batch.spikes.steps[selected],expected.spikes.steps)
        np.testing.assert_array_equal(batch.spikes.neurons[selected],expected.spikes.neurons)
        for name,value in host_state(state,p).items():
            np.testing.assert_array_equal(value[row],host_state(one,p)[name][0])
    assert not np.array_equal(np.asarray(state.random.stream.words).reshape(2,624)[0],
                              np.asarray(state.random.stream.words).reshape(2,624)[1])


@pytest.mark.metal
@pytest.mark.parametrize('function',['rand()','randn()','poisson(3)','poisson(17)'])
def test_fixed_draw_windows_preserve_partial_boundaries_and_normal_cache(function):
    import mlx.core as mx
    from mlx_backend.configured import prepare,run_prepared,initial_state,advance,evaluate_state,host_state
    model=resolve_model({'eq_th':f'{function}<0.3'},dt_ms=.1)
    stimulus=inputs(43);streams=snapshots();p=prepare(graph(),stimulus,model)
    cpu=reference(graph(),stimulus,model,random_states=streams)
    full,state=run_prepared(p,stimulus,return_state=True,random_states=streams)
    assert_spikes_equal(full.spikes, cpu.spikes)
    assert_streams(state.random.stream,cpu.random_states)
    partial=initial_state(p,2,random_states=streams)
    events=mx.array(stimulus.events.reshape(-1))
    for stop in (1,7,13,18,21,43):
        while partial.step<stop:
            partial,_=advance(p,partial,events,43,min(stop-partial.step,p.window-partial.step%p.window))
        evaluate_state(partial)
    for name,array in host_state(partial,p).items():
        np.testing.assert_array_equal(array,cpu.state[name],err_msg=name)
    assert_streams(partial.random.stream,cpu.random_states)


@pytest.mark.metal
@pytest.mark.parametrize('trials',[1,2])
@pytest.mark.parametrize('function',['rand()','randn()','poisson(0)','poisson(3)','poisson(17)'])
def test_uniform_lookahead_renewal_and_extended_horizon_match_cpu(monkeypatch,trials,function):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend import expression_runtime as runtime
    from mlx_backend.configured import prepare,initial_state,advance,evaluate_state,host_state
    monkeypatch.setattr(runtime,'POOL_MAX_FRAMES',5)
    model=resolve_model({'eq_th':f'{function}<0.3','t_dly':.2*b.ms},dt_ms=.1)
    stimulus=inputs(83,trials);p=prepare(graph(),stimulus,model)
    streams=[]
    for row in range(trials):
        cpu=np.random.RandomState(81+row);cpu.randn(1)
        snapshot=list(cpu.get_state());snapshot[2]=0 if row==0 else 623;streams.append(tuple(snapshot))
    expected=reference(graph(),stimulus,model,random_states=streams)
    partial=initial_state(p,trials,random_states=streams)
    events=mx.array(stimulus.events.reshape(-1));observed=[]
    for horizon in (43,83):
        while partial.step<horizon:
            begin=partial.step;length=min(p.window-begin%p.window,horizon-begin)
            partial,bits=advance(p,partial,events,horizon,length,event_steps=83)
            evaluate_state(partial);words=np.asarray(bits)
            for bit in range(length):
                for row in np.flatnonzero(words&(1<<bit)):
                    observed.append((stimulus.trial_indices[row//p.neurons],row%p.neurons,begin+bit))
    assert sorted(observed,key=lambda row:(row[0],row[2],row[1]))==list(zip(
        expected.spikes.trials,expected.spikes.neurons,expected.spikes.steps))
    for name,array in host_state(partial,p).items():
        np.testing.assert_array_equal(array,expected.state[name],err_msg=name)
    assert_streams(partial.random.stream,expected.random_states)
