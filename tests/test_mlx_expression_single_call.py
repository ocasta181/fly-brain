"""Conditional native compaction versus independently executed stock Brian."""
import numpy as np
import pytest
from mlx_backend.model import resolve_model
from mlx_backend.models import Connectome,Stimulus


@pytest.mark.metal
@pytest.mark.parametrize('target',['cpp','cython'])
@pytest.mark.parametrize('kind',['uniform','normal'])
def test_conditional_single_call_preserves_rows_buffers_and_trial_streams(target,kind,tmp_path):
    import brian2 as b
    from mlx_backend.configured import prepare,initial_state,host_state,run_prepared
    from mlx_backend.expression_runtime import snapshots
    from mlx_backend.expression_single_call import primitive
    from test_mlx_cpp_expression_rng import cpu_reference as cpp
    from test_mlx_cython_expression_rng import cpu_reference as cython,assert_snapshot
    threshold=('(i<673) and (rand()<.8)'if kind=='uniform'else'(i>=673) or (randn()>-.8)')
    reset=('a+=1;v=v_rst+randn()*mV;g=0*mV'if kind=='uniform'else'a=rand();v=v_rst+a*mV;g=0*mV')
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\na:1',eq_th=threshold,eq_rst=reset,
        t_rfc=0*b.ms),dt_ms=.1,random_target=target)
    n=1009;steps=43
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((2,steps,0),np.uint8),(),(),(7,2),81,3,None,.1)
    p=prepare(graph,inputs,model);assert primitive(p.program,'threshold')==('rand'if kind=='uniform'else'randn')
    names=tuple(host_state(initial_state(p,2),p));references=[];starts=[]
    for trial,seed in enumerate((81,827)):
        directory=tmp_path/f'trial{trial}'
        if target=='cpp':
            spikes,expected,final=cpp(model,directory/'cpp-default',names,steps=steps,normal_used=True,neurons=n,seed=seed)
            start=np.random.RandomState(seed).get_state()
        else:
            start,spikes,expected,final,error=cython(model,directory,names,steps=steps,neurons=n,seed=seed)
            assert error is None
        starts.append(start);references.append((spikes,expected,final))
    result,state=run_prepared(p,inputs,return_state=True,random_states=starts,chunk_steps=19)
    _,quiet=run_prepared(p,inputs,return_state=True,random_states=starts,chunk_steps=7,record=False)
    for trial,(spikes,expected,final)in enumerate(references):
        selected=result.spikes.trials==inputs.trial_indices[trial]
        np.testing.assert_array_equal(result.spikes.neurons[selected],spikes[0])
        np.testing.assert_array_equal(result.spikes.steps[selected],spikes[1])
        for run in(state,quiet):
            for name,value in host_state(run,p).items():np.testing.assert_array_equal(value[trial:trial+1],expected[name],err_msg=name)
            got=snapshots(run.random)[trial]
            if target=='cython':assert_snapshot(got,final)
            else:
                copy=np.random.RandomState();copy.set_state(got)
                np.testing.assert_array_equal(copy.randint(0,2**32,624,dtype=np.uint32),final[0]);assert got[3:]==final[1:]
