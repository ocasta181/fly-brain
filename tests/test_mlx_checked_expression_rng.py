"""Actual Cython draw checkpoints across native integer replay frontiers."""
from dataclasses import replace
import numpy as np
import pytest

pytestmark = pytest.mark.metal


def fixture(threshold='rand()<.5',reset=None):
    import brian2 as b
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome,Stimulus
    reset=reset or ('x=2+i%2;y=61-i%5;z=3+i%2;'
        'f=abs(x)**y%z;q=abs(x)**y-abs(z)**y;r=abs(x)**y/z;v=v_rst;g=0*mV')
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\nx:integer\ny:integer\nz:integer\nf:1\nq:1\nr:1',
        eq_th=threshold,eq_rst=reset,t_rfc=0*b.ms),dt_ms=.1,random_target='cython')
    graph=Connectome(np.arange(5),np.array([0,1,1],np.int32),np.array([2,2,0],np.int32),
        np.array([2,-3,7]),np.zeros(3))
    inputs=Stimulus(np.zeros((2,4,0),np.uint8),(),(),(2,7),81,0,None,.1)
    return model,graph,inputs


@pytest.mark.parametrize('threshold',['rand()<.5','randn()>-100'],ids=['uniform','normal'])
def test_checked_random_frontier_retries_restore_physical_streams(tmp_path,threshold):
    import mlx.core as mx
    from mlx_backend.configured import prepare,initial_state,host_state,advance,run_prepared
    from mlx_backend.compiled_precision_frontier import PrecisionFrontier,rebind
    from mlx_backend.expression_runtime import snapshots
    from test_mlx_cython_expression_rng import cpu_reference,assert_snapshot
    model,graph,inputs=fixture(threshold)
    p=prepare(graph,inputs,model)
    assert p.program.random_program.checked and p.program.native_precision_limbs
    names=tuple(host_state(initial_state(p,2),p))
    references=[cpu_reference(model,tmp_path/f'cpu-default-{seed}',names,seed=seed,
        neurons=5,steps=4,connectome=graph)for seed in(81,827)]
    assert all(row[-1]is None for row in references)
    starts=[row[0]for row in references]
    # Deliberately undersize only the native temporary specialization. The
    # actual reference language, CPU flags and persistent dtypes are unchanged.
    small=rebind(p,2)
    state=initial_state(small,2,random_states=starts)
    frontier=PrecisionFrontier(small,state)
    events=mx.zeros((1,),mx.uint8)
    frontier.submit(lambda pp,ss,origin:advance(pp,ss,events,4,4,precision_epoch_begin=origin),4)
    result=frontier.evaluate()
    assert frontier.retries>0 and result.error_kind is None
    actual=host_state(result.state,result.prepared)
    words=np.asarray(result.windows[0][2]).reshape(2,5)
    for row,(_,spikes,values,final,_)in enumerate(references):
        expected_words=np.zeros(5,np.uint32)
        for neuron,step in zip(*spikes):expected_words[neuron]|=np.uint32(1<<step)
        np.testing.assert_array_equal(words[row],expected_words)
        for name,value in actual.items():np.testing.assert_array_equal(value[row:row+1],values[name],err_msg=name)
        assert_snapshot(snapshots(result.state.random)[row],final)
    # Ordinary recording and recording-off exercise the same frontier without
    # exposing the deliberately small test specialization to model identity.
    run,complete=run_prepared(p,inputs,return_state=True,random_states=starts)
    _,off=run_prepared(p,inputs,return_state=True,random_states=starts,record=False)
    for other in(complete,off):
        for name,value in actual.items():np.testing.assert_array_equal(host_state(other,p)[name],value,err_msg=name)
        for row,reference in enumerate(references):assert_snapshot(snapshots(other.random)[row],reference[3])
    for row,reference in enumerate(references):
        selected=run.spikes.trials==inputs.trial_indices[row]
        np.testing.assert_array_equal(run.spikes.neurons[selected],reference[1][0])
        np.testing.assert_array_equal(run.spikes.steps[selected],reference[1][1])


def test_checked_random_error_commits_first_row_prefix_and_other_trial(tmp_path):
    import mlx.core as mx
    from mlx_backend.configured import prepare,initial_state,host_state,advance,evaluate_state
    from mlx_backend.expression_runtime import snapshots
    from test_mlx_cython_expression_rng import cpu_reference,assert_snapshot
    model,graph,inputs=fixture('rand()<1',
        'x=i-1+2*int(q>0);y=-1;f=abs(x)**y;r=42;v=v_rst;g=0*mV')
    p=prepare(graph,inputs,model)
    names=tuple(host_state(initial_state(p,2),p))
    references=[cpu_reference(model,tmp_path/f'cpu-default-{row}',names,seed=seed,
        neurons=5,steps=4,connectome=graph,initial_values={'q':float(row)})
        for row,seed in enumerate((81,827))]
    cause=references[0][-1]
    assert isinstance(cause,ZeroDivisionError)and references[1][-1]is None
    state=initial_state(p,2,random_states=[row[0]for row in references])
    values=dict(state.values);values['q']=mx.array(np.repeat([0.,1.],5).view(np.uint64))
    state=replace(state,values=values)
    state,bits=advance(p,state,mx.zeros((1,),mx.uint8),4,4)
    with pytest.raises(type(cause))as error:evaluate_state(state)
    assert str(error.value)==str(cause)
    # Read failed-state diagnostics after independently checking its exception.
    actual=host_state(replace(state,check_index_error=False),p)
    words=np.asarray(bits).reshape(2,5)
    for row,(_,spikes,expected,final,_)in enumerate(references):
        for name,value in actual.items():np.testing.assert_array_equal(value[row:row+1],expected[name],err_msg=name)
        expected_words=np.zeros(5,np.uint32)
        for neuron,step in zip(*spikes):expected_words[neuron]|=np.uint32(1<<step)
        np.testing.assert_array_equal(words[row],expected_words)
        assert_snapshot(snapshots(state.random)[row],final)


def test_state_error_before_random_threshold_consumes_no_draws(tmp_path):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.model import resolve_model
    from mlx_backend.configured import prepare,initial_state,host_state,advance,evaluate_state
    from mlx_backend.expression_runtime import snapshots
    from test_mlx_cython_expression_rng import cpu_reference,assert_snapshot
    _,graph,inputs=fixture()
    model=resolve_model(dict(eqs='dv/dt=(g/den-v)/ms:volt\ng:volt\nrfc:second\nden:integer',
        eq_th='randn()>-100',eq_rst='v=v_rst;g=0*mV',t_rfc=0*b.ms),
        dt_ms=.1,random_target='cython')
    p=prepare(graph,inputs,model)
    names=tuple(host_state(initial_state(p,2),p))
    denominator=np.array([[1,0,2,3,4],[1,2,3,4,5]],np.int32)
    references=[cpu_reference(model,tmp_path/f'cpu-default-{row}',names,seed=seed,
        neurons=5,steps=4,connectome=graph,initial_values={'g':b.mV,'den':denominator[row]})
        for row,seed in enumerate((81,827))]
    cause=references[0][-1]
    assert isinstance(cause,ZeroDivisionError)and references[1][-1]is None
    state=initial_state(p,2,random_states=[row[0]for row in references])
    values=dict(state.values)
    values['den']=mx.array(denominator.reshape(-1))
    values['g']=mx.array(np.full(10,.001).view(np.uint64))
    state=replace(state,values=values)
    state,bits=advance(p,state,mx.zeros((1,),mx.uint8),4,4)
    with pytest.raises(type(cause))as error:evaluate_state(state)
    assert str(error.value)==str(cause)
    actual=host_state(replace(state,check_index_error=False),p)
    words=np.asarray(bits).reshape(2,5)
    for row,(_,spikes,expected,final,_)in enumerate(references):
        for name,value in actual.items():np.testing.assert_array_equal(value[row:row+1],expected[name],err_msg=name)
        expected_words=np.zeros(5,np.uint32)
        for neuron,step in zip(*spikes):expected_words[neuron]|=np.uint32(1<<step)
        np.testing.assert_array_equal(words[row],expected_words)
        assert_snapshot(snapshots(state.random)[row],final)


def test_failed_trial_does_not_prevent_other_trials_precision_retry(tmp_path):
    import mlx.core as mx
    from mlx_backend.configured import prepare,initial_state,host_state,advance
    from mlx_backend.compiled_precision_frontier import PrecisionFrontier,rebind
    from mlx_backend.runtime_errors import model_exception
    from mlx_backend.expression_runtime import snapshots
    from test_mlx_cython_expression_rng import cpu_reference,assert_snapshot
    model,graph,inputs=fixture('rand()<1',
        'x=2;y=120;z=3;f=1/int(q>0);r=abs(x)**y-abs(z)**y;v=v_rst;g=0*mV')
    p=prepare(graph,inputs,model)
    names=tuple(host_state(initial_state(p,2),p))
    references=[cpu_reference(model,tmp_path/f'cpu-default-{row}',names,seed=seed,
        neurons=5,steps=4,connectome=graph,initial_values={'q':float(row)})
        for row,seed in enumerate((81,827))]
    cause=references[0][-1]
    assert isinstance(cause,ZeroDivisionError)and references[1][-1]is None
    small=rebind(p,2);state=initial_state(small,2,random_states=[row[0]for row in references])
    values=dict(state.values);values['q']=mx.array(np.repeat([0.,1.],5).view(np.uint64))
    state=replace(state,values=values)
    frontier=PrecisionFrontier(small,state)
    frontier.submit(lambda pp,ss,origin:advance(pp,ss,mx.zeros((1,),mx.uint8),4,4,
        precision_epoch_begin=origin),4)
    result=frontier.evaluate()
    assert frontier.retries>0 and result.error_kind is not None
    error=model_exception(result.prepared.program.runtime_errors,result.error_kind)
    assert type(error)is type(cause)and str(error)==str(cause)
    actual=host_state(replace(result.state,check_index_error=False),result.prepared)
    words=np.asarray(result.windows[0][2]).reshape(2,5)
    for row,(_,spikes,expected,final,_)in enumerate(references):
        for name,value in actual.items():np.testing.assert_array_equal(value[row:row+1],expected[name],err_msg=name)
        expected_words=np.zeros(5,np.uint32)
        for neuron,step in zip(*spikes):expected_words[neuron]|=np.uint32(1<<step)
        np.testing.assert_array_equal(words[row],expected_words)
        assert_snapshot(snapshots(result.state.random)[row],final)
