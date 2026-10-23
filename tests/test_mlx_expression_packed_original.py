"""Original API packed arrivals and compiled graph mutable inputs."""
from native_execution import forbid_cpu_steps
from contextlib import nullcontext
import gc
import numpy as np
import pytest


@pytest.mark.metal
def test_original_packed_arrivals_and_mutable_compiled_cache(tmp_path,monkeypatch):
    import brian2 as b
    import pandas as pd
    import original_api_reference as reference
    from mlx_backend.brian_device import original_api
    from mlx_backend import expression_window,expression_numpy_packed
    n=65536
    spiking=np.flatnonzero(np.random.RandomState(875).rand(n)<.0001)
    assert len(spiking)>=3
    sources=spiking[[0,1,2,0]]
    def writer(directory):
        directory.mkdir(parents=True,exist_ok=True)
        nodes=directory/'nodes.csv';edges=directory/'edges.parquet'
        pd.DataFrame({'label':np.arange(n)},index=pd.Index(np.arange(n),name='flywire_id')).to_csv(nodes)
        pd.DataFrame({'Presynaptic_Index':sources,'Postsynaptic_Index':[1,1,3,4],
            'Excitatory x Connectivity':[1.,-.25,2.,-.5]}).to_parquet(edges)
        return nodes,edges
    monkeypatch.setattr(reference,'write_tiny_dataset',writer)
    observed={'hits':[],'markers':0};case_id=[None]
    original_window=expression_window.advance;original_packed=expression_numpy_packed.advance
    def window(*args,**kwargs):
        before=dict(expression_window._PLANS)
        result=original_window(*args,**kwargs)
        if (result is not None and before
                and expression_window._PLANS.keys()==before.keys()
                and all(expression_window._PLANS[key] is value for key,value in before.items())):
            observed['hits'].append(case_id[0])
        return result
    def packed(*args,**kwargs):
        if kwargs.get('marker_shift',args[8]if len(args)>8 else None)is not None:observed['markers']+=1
        return original_packed(*args,**kwargs)
    monkeypatch.setattr(expression_window,'advance',window)
    monkeypatch.setattr(expression_numpy_packed,'advance',packed)
    references=[]
    def capture(fixture):
        arrays={}
        for label,owner,names in (
            ('state',fixture.neurons,('v','g','rfc','lastspike','not_refractory')),
            ('monitor',fixture.monitor,('i','t','count','N')),
            ('synapse',fixture.synapses,('w',)),
            ('clock',fixture.neurons.clock,('t','timestep'))):
            for name in names:arrays[label+'.'+name]=np.array(owner.variables[name].get_value(),copy=True)
        arrays['spikes']=np.array(fixture.neurons.spikes,copy=True)
        arrays['network_t']=np.array(fixture.network.t_)
        return arrays,reference.capture_rng_state()
    for native in(False,True):
        if native:expression_window._PLANS.clear()
        for case in range(3):
            case_id[0]=case
            with reference.cpu_runtime(seed=875,dt_ms=.1),monkeypatch.context()as patches:
                with original_api(workers=False)if native else nullcontext():
                    fixture=reference.build_original_api_fixture(tmp_path/f'{native}-{case}',silenced=(),parameters=dict(
                        eqs='v:volt\ng:volt\nrfc:second',eq_th='rand()<.0001',
                        eq_rst='v=v_rst+randn()*mV;g=0*mV',t_rfc=0*b.ms,t_dly=.2*b.ms,
                        r_poi=(3700 if case<2 else 1100)*b.Hz,r_poi2=7200*b.Hz))
                    fixture.synapses.w*=1 if case==0 else 2
                    if native:
                        forbid_cpu_steps(fixture.network, patches)
                    states=[]
                    for index,duration in enumerate((.4,0.,.4,.4)):
                        fixture.network.run(duration*b.ms,namespace={})
                        states.append(capture(fixture))
                        if index==0:
                            assert np.any(states[-1][0]['state.g'][[1,3,4]]!=0),'No nonzero delayed arrivals'
                            fixture.synapses.w*=-.5
                            fixture.inputs[0].active=False
                        if index==2:fixture.inputs[0].active=True
                    if native:
                        for (actual,rng),(expected,expected_rng)in zip(states,references[case]):
                            for name,value in actual.items():
                                other=expected[name]
                                assert(value.dtype,value.shape)==(other.dtype,other.shape),name
                                assert value.tobytes()==other.tobytes(),(case,name)
                            reference.assert_rng_state_equal(rng,expected_rng)
                    else:references.append(states)
                    del fixture
            gc.collect()
    assert 1 in observed['hits'],'No same-model mutable-weight cache reuse'
    assert observed['markers']>0,'No fused packed arrival marker path'
