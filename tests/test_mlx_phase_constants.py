"""Each Brian runner owns its hoisted constants, including repeated _lio names."""
import numpy as np
import pytest
from state_assertions import assert_raw_state

from test_mlx_dataset_workflows import files, inputs, original, metal, IDS, assert_spikes
from mlx_backend import configured
from mlx_backend.inputs import load_connectome
from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('record',[False,True])
def test_phase_local_constants_preserve_actual_cpu_states_and_continuation(tmp_path,original,metal,monkeypatch,record,target):
    b,eon=original
    monkeypatch.setitem(b.prefs, 'codegen.target', target)
    monkeypatch.setitem(b.prefs, 'codegen.runtime.cython.cache_dir', str(tmp_path/'cython-default'))
    paths=files(tmp_path)
    graph=load_connectome(*paths)
    stimulus,silenced=inputs(graph,IDS[0],steps=73,trials=2)
    params=dict(eon.default_params,eqs=DEFAULT_EQUATIONS+'valid : boolean\nend : 1\n',
                eq_rst='v=v_rst; g=0*mV; rfc=0.3*ms; valid=not valid; end+=1',
                r_poi=10000*b.Hz,r_poi2=0*b.Hz)
    prepared=configured.prepare(graph,stimulus,resolve_model(params),silenced)
    names=[v.name for v in configured._persistent(prepared.program)]
    expected,states=[],[]
    for trial in stimulus.trial_indices:
        neurons,synapses,monitor=eon.create_model(*paths,params)
        pois,neurons=eon.poi(neurons,[0,0],[3],params)
        b.Network(neurons,synapses,monitor,*pois).run(7.3*b.ms,namespace={})
        steps=np.rint(np.asarray(monitor.t_[:])/.0001).astype(np.int64)
        expected.extend(zip(np.full(steps.size,trial),steps,np.asarray(monitor.i[:])))
        states.append({name:np.asarray(getattr(neurons,name+'_')[:]).copy() for name in names})
    result,state=configured.run_prepared(prepared,stimulus,record=record,return_state=True,chunk_steps=23)
    if record:assert_spikes(result,expected)
    else:assert result.spikes.steps.size==0
    def check(state):
        actual=configured.host_state(state,prepared)
        assert_raw_state(actual,{name:np.stack([s[name] for s in states]) for name in names})
    check(state)
    continued=configured.initial_state(prepared,2)
    events=metal.array(stimulus.events.reshape(-1))
    while continued.step<73:
        width=min(7,73-continued.step,prepared.window-continued.step%prepared.window)
        continued,_=configured.advance(prepared,continued,events,73,width)
        configured.evaluate_state(continued)
    check(continued)
