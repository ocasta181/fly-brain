"""Independent Brian and native dense recording with live float32 math."""
from dataclasses import replace
from pathlib import Path
import hashlib
import numpy as np
import pytest


@pytest.mark.metal
def test_dense_live_math_trials_and_continuation(tmp_path,monkeypatch):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    from mlx_backend.models import Connectome,Stimulus
    from test_mlx_float32 import assert_bits
    root=Path(__file__).resolve().parents[1]
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.target','cython')
    monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',
        str(root/'executions/cython-basic32-dense-default'))
    b.start_scope();n=16401;steps=32;ids=(41,3)
    model=resolve_model(dict(eqs=DEFAULT_EQUATIONS+'x:1\nsample:1\n',
        eq_th='True',eq_rst='x+=.000001;sample=exprel(x)+.1',t_rfc=0*b.second))
    empty=np.empty(0,np.int32)
    graph=Connectome(np.arange(n),empty,empty,empty,np.empty(0))
    events=np.empty((2,steps,0),np.uint8)
    stimulus=Stimulus(events,(),(),ids,20261018,0,
        hashlib.sha256(events.tobytes()).hexdigest(),model.dt_ms)
    prepared=configured.prepare(graph,stimulus,model)
    initial=np.random.default_rng(20261018).uniform(.01,1,(2,n)).astype(np.float32)
    original=configured.initial_state
    def seed(p,trials=1):
        state=original(p,trials);values=dict(state.values)
        values['x']=mx.array(initial.reshape(-1).view(np.uint32))
        state=replace(state,values=configured._pack_values(p.program,values))
        configured.evaluate_state(state);return state
    monkeypatch.setattr(configured,'initial_state',seed)
    names=tuple(v.name for v in prepared.program.variables.values() if v.persistent)
    cpus=[];expected=[]
    for trial in range(2):
        clock=b.Clock(dt=model.dt_s*b.second)
        group=b.NeuronGroup(n,model.eqs,method='linear',threshold=model.eq_th,
            reset=model.eq_rst,refractory='rfc',namespace=model.brian_namespace(),clock=clock)
        group.v=model.v_0*b.volt;group.g=0*b.volt;group.rfc=0*b.second
        group.x=initial[trial];spikes=b.SpikeMonitor(group);network=b.Network(group,spikes)
        for length in (0,17,0,15):network.run(length*clock.dt,namespace={})
        cpus.append(group)
        expected.append((ids[trial],np.asarray(spikes.i[:],np.int64).copy(),
            np.rint(np.asarray(spikes.t_[:])/model.dt_s).astype(np.int64)))
    expected.sort(key=lambda part:part[0])
    for record in (False,True):
        result,state=configured.run_prepared(prepared,stimulus,record=record,return_state=True)
        for name,actual in configured.host_state(state,prepared).items():
            assert_bits(actual,np.stack([np.asarray(g.variables[name].get_value()) for g in cpus]),name)
        for index,actual in enumerate((result.spikes.trials,result.spikes.neurons,result.spikes.steps)):
            wanted=(np.concatenate([np.full(len(neurons),trial,np.int64) if index==0
                else neurons if index==1 else times for trial,neurons,times in expected])
                if record else np.empty(0,np.int64))
            np.testing.assert_array_equal(actual,wanted)
