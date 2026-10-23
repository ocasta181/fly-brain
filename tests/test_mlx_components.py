"""Topology-fused Metal execution against original Brian and pending state."""
from state_assertions import assert_spikes_equal
from native_execution import forbid_call
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import mlx_engine

from test_mlx_dataset_workflows import COLUMNS, original, metal, assert_spikes
from test_mlx_dataset_preparation_reuse import assert_state
from mlx_backend import configured
from mlx_backend.inputs import load_connectome
from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
from mlx_backend.components import advance_components

pytestmark = pytest.mark.metal


def component_files(path, sizes):
    n=sum(sizes)
    rng=np.random.default_rng(813)
    order=rng.permutation(n)
    inverse=np.argsort(order)
    sources, destinations, weights=[],[],[]
    first=0
    for size in sizes:
        for row in range(first,first+size):
            sources.extend([row,row])
            destinations.extend([first+(row-first+1)%size,first+(row-first+1)%size])
            weights.extend([280.125,-4.25])
        first+=size
    ids=np.uint64(2**63)+np.arange(n,dtype=np.uint64)*7+5
    paths=path/'neurons.csv',path/'edges.parquet'
    pd.DataFrame(index=ids[order]).to_csv(paths[0])
    pd.DataFrame(dict(zip(COLUMNS,[inverse[sources].astype(np.int32),
        inverse[destinations].astype(np.int32),np.asarray(weights,np.float32)]))).to_parquet(paths[1])
    return paths


def compare_pending(actual,wanted,p):
    assert_state(configured.host_state(actual,p),configured.host_state(wanted,p))
    np.testing.assert_array_equal(np.asarray(actual.release),np.asarray(wanted.release))
    size=actual.values['v'].size
    for field in ('history','due'):
        a,w=getattr(actual,field),getattr(wanted,field)
        for key in a.keys()|w.keys():
            np.testing.assert_array_equal(np.asarray(a.get(key,np.zeros(size,np.uint32))),
                                          np.asarray(w.get(key,np.zeros(size,np.uint32))))


@pytest.mark.parametrize('delay,custom_reset,large_component',[
    (1,False,False),(2,False,False),(18,False,False),(33,False,False),(65,False,False),
    (128,False,False),(129,False,False),(33,'arrival',False),
    (33,'names',False),(18,False,True),
])
def test_components_keep_original_state_recording_and_partial_history(
    tmp_path,original,metal,delay,custom_reset,large_component,
):
    b,eon=original
    paths=component_files(tmp_path,[33,2] if large_component else [1,2,3,5,17,13])
    graph=load_connectome(*paths)
    n=graph.neuron_ids.size
    first=list(range(n//2))+[0]
    second=list(range(n//2,n))+[0]
    spec=mlx_engine.experiment_spec(dict(key='components',
        neu_exc=graph.neuron_ids[first].tolist(),neu_exc2=graph.neuron_ids[second].tolist(),
        neu_slnc=[int(graph.neuron_ids[-1])],stim_rate=10000.,stim_rate2=0.))
    stimuli,silenced=mlx_engine.stimuli(graph,spec,173,2,82)
    stimulus=mlx_engine.combine_trials(stimuli)
    params=dict(eon.default_params,t_dly=delay*.1*b.ms,r_poi=10000*b.Hz,r_poi2=0*b.Hz)
    if custom_reset:
        extra = ('arrival_bits : 1\ncounter : integer\n' if custom_reset=='arrival'
                 else 'arrival_bits : 1\ncounter : integer\nvalid : boolean\nend : 1\n')
        reset = ('counter+=1; arrival_bits+=1' if custom_reset=='arrival'
                 else 'counter+=1; arrival_bits+=1; valid=not valid; end+=1')
        params.update(eqs=DEFAULT_EQUATIONS+extra,
            eq_rst='v=v_rst; g=0*mV; rfc=0.3*ms; '+reset)
    model=resolve_model(params)
    p=configured.prepare(graph,stimulus,model,silenced,component=True)
    assert (p.components is None)==(large_component or delay>128)
    expected,states=[],[]
    names=[v.name for v in configured._persistent(p.program)]
    for trial in stimulus.trial_indices:
        neurons,synapses,monitor=eon.create_model(*paths,params)
        pois,neurons=eon.poi(neurons,first,second,params)
        eon.silence(silenced,synapses)
        b.Network(neurons,synapses,monitor,*pois).run(17.3*b.ms,namespace={})
        steps=np.rint(np.asarray(monitor.t_[:])/.0001).astype(np.int64)
        expected.extend(zip(np.full(steps.size,trial),steps,np.asarray(monitor.i[:])))
        states.append({name:np.asarray(getattr(neurons,name+'_')[:]).copy() for name in names})
    wanted={name:np.stack([state[name] for state in states]) for name in names}
    recorded,_=configured.run_prepared(p,stimulus,return_state=True,chunk_steps=45,block_windows=3)
    assert_spikes(recorded,expected)
    result,state=configured.run_prepared(p,stimulus,record=False,return_state=True,chunk_steps=45)
    assert result.spikes.steps.size==0
    assert (result.timings['component_passes']>0)==(p.components is not None)
    assert_state(configured.host_state(state,p),wanted)
    _,baseline=configured.run_prepared(replace(p,components=None),stimulus,
        record=False,return_state=True,chunk_steps=45)
    compare_pending(state,baseline,p)
    public=mlx_engine.execute(graph,stimulus,silenced,model=model,record=False)
    assert (public.timings['component_passes']>0)==(p.components is not None)
    if p.components is not None:
        # Continue at arbitrary sub-window boundaries, including pending events
        # older than one word. All trials retain the same physical row mapping.
        state=configured.initial_state(p,2)
        events=metal.array(stimulus.events.reshape(-1))
        widths=(7,1,11,3,13)
        iteration=0
        while state.step<173:
            end=min(173,state.step+widths[iteration%len(widths)])
            state=advance_components(p,state,events,173,end)
            configured.evaluate_state(state)
            iteration+=1
        compare_pending(state,baseline,p)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_packed_component_keeps_timed_namespace_random_replay_and_storage_width(
    metal,monkeypatch,dtype,
):
    import brian2 as b
    from configured_reference import reference
    from test_mlx_configured import graph,inputs
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',dtype)
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    drive=b.TimedArray(np.array([.125,-.25,.5,0.,.75]),dt=.2*b.ms)
    model=resolve_model({'eqs':DEFAULT_EQUATIONS.replace('v_0 - v + g',
        'v_0 - v + g + drive(t)*mV'),'drive':drive})
    connectome=graph()
    stimulus=inputs(model,targets=(0,1,2),steps=77,trials=2)
    stimulus=replace(stimulus,events=np.random.default_rng(472).binomial(1,.125,stimulus.events.shape).astype(np.uint8))
    expected=reference(connectome,stimulus,model)
    prepared=configured.prepare(connectome,stimulus,model,component=True)
    assert prepared.components is not None
    assert prepared.program.input_arrays
    for record in (True,False):
        result,state=configured.run_prepared(prepared,stimulus,record=record,
            return_state=True,chunk_steps=45)
        if record:
            assert_spikes_equal(result.spikes, expected.spikes)
        else:
            assert result.timings['component_passes']>0 and result.spikes.steps.size==0
        assert_state(configured.host_state(state,prepared),expected.state)
        if not record:
            _,ordinary=configured.run_prepared(replace(prepared,components=None),stimulus,
                record=False,return_state=True,chunk_steps=45)
            compare_pending(state,ordinary,prepared)


def test_float16_component_keeps_combined_typed_fields_and_both_recording_modes(
    tmp_path,original,metal,monkeypatch,
):
    import sys
    create=component_files
    def half_files(path,sizes):
        paths=create(path,sizes)
        edges=pd.read_parquet(paths[1])
        edges[COLUMNS[2]]=edges[COLUMNS[2]].astype(np.float16)
        edges.to_parquet(paths[1])
        return paths
    monkeypatch.setattr(sys.modules[__name__],'component_files',half_files)
    test_components_keep_original_state_recording_and_partial_history(
        tmp_path,original,metal,33,'names',False)


def test_scan_arrivals_survives_collective_reset_split(metal,monkeypatch):
    import brian2 as b
    from configured_reference import reference
    from test_mlx_configured import graph,inputs
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'core.default_integer_dtype',np.int8)
    model=resolve_model({'eqs':DEFAULT_EQUATIONS+'counter : integer\n',
                         'eq_rst':'counter=g/mV;v=v_rst;g=0*mV'})
    connectome=graph()
    stimulus=inputs(model,targets=(0,1,2),steps=57,trials=2)
    expected=reference(connectome,stimulus,model)
    prepared=configured.prepare(connectome,stimulus,model,component=True)
    assert prepared.scan_arrivals and prepared.program.collective_reset
    assert prepared.components is None
    for recording in (True,False):
        result,state=configured.run_prepared(prepared,stimulus,record=recording,
            return_state=True,chunk_steps=29)
        assert_state(configured.host_state(state,prepared),expected.state)
        if recording:
            assert_spikes_equal(result.spikes, expected.spikes)


def test_component_preserves_synapses_before_groups(metal,monkeypatch):
    import brian2 as b
    from configured_reference import reference
    from test_mlx_configured import graph,inputs
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'core.network.default_schedule',
        ['start','synapses','groups','thresholds','resets','end'])
    model=resolve_model()
    connectome=graph()
    stimulus=inputs(model,targets=(0,1,2),steps=63,trials=2)
    expected=reference(connectome,stimulus,model)
    prepared=configured.prepare(connectome,stimulus,model,component=True)
    assert prepared.components is not None
    for recording in (True,False):
        result,state=configured.run_prepared(prepared,stimulus,record=recording,
            return_state=True,chunk_steps=29)
        assert_state(configured.host_state(state,prepared),expected.state)
        if recording:
            assert_spikes_equal(result.spikes, expected.spikes)
        else:
            _,baseline=configured.run_prepared(replace(prepared,components=None),stimulus,
                record=False,return_state=True,chunk_steps=29)
            compare_pending(state,baseline,prepared)


def test_actual_cpp_component_state_recording_and_continuation(tmp_path,original,metal,monkeypatch):
    b,eon=original
    monkeypatch.setitem(b.prefs,'devices.cpp_standalone.extra_make_args_unix',['-j2'])
    paths=component_files(tmp_path,[2,3])
    graph=load_connectome(*paths)
    n=graph.neuron_ids.size
    first,second=list(range(n//2)),list(range(n//2,n))
    spec=mlx_engine.experiment_spec(dict(key='cpp-components',
        neu_exc=graph.neuron_ids[first].tolist(),neu_exc2=graph.neuron_ids[second].tolist(),
        neu_slnc=[int(graph.neuron_ids[-1])],stim_rate=10000.,stim_rate2=0.))
    trials,silenced=mlx_engine.stimuli(graph,spec,73,1,82)
    stimulus=trials[0]
    params=dict(eon.default_params,r_poi=10000*b.Hz,r_poi2=0*b.Hz)
    b.set_device('cpp_standalone',build_on_run=False)
    device=b.get_device()
    device.reinit();device.activate(build_on_run=False)
    try:
        neurons,synapses,monitor=eon.create_model(*paths,params)
        pois,neurons=eon.poi(neurons,first,second,params)
        eon.silence(silenced,synapses)
        b.Network(neurons,synapses,monitor,*pois).run(7.3*b.ms,namespace={})
        device.build(directory=str(tmp_path/'cpp-default-flags'),run=True,with_output=False)
        names=('v','g','rfc','lastspike','not_refractory')
        wanted={name:np.asarray(getattr(neurons,name+'_')[:])[None].copy() for name in names}
        wanted_i=np.asarray(monitor.i[:]).copy()
        wanted_steps=np.rint(np.asarray(monitor.t_[:])/.0001).astype(np.int64)
    finally:
        b.set_device('runtime')
    forbid_call(b.Network, 'run', 'Native component executed a CPU timestep', monkeypatch)
    model=resolve_model(cpu_target='cpp')
    prepared=configured.prepare(graph,stimulus,model,silenced,component=True)
    assert prepared.components is not None and prepared.program.refractory_policy=='cpp_fast'
    for recording in (True,False):
        result,state=configured.run_prepared(prepared,stimulus,record=recording,
            return_state=True,chunk_steps=29)
        assert_state(configured.host_state(state,prepared),wanted)
        if recording:
            np.testing.assert_array_equal(result.spikes.neurons,wanted_i)
            np.testing.assert_array_equal(result.spikes.steps,wanted_steps)
        else:assert result.timings['component_passes']>0 and result.spikes.steps.size==0
    state=configured.initial_state(prepared,1)
    events=metal.array(stimulus.events.reshape(-1))
    for end in (1,17,18,37,73):
        state=advance_components(prepared,state,events,73,end)
        configured.evaluate_state(state)
    assert_state(configured.host_state(state,prepared),wanted)
