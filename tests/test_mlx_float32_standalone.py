"""Independent application-target CPU coverage; all builds stay test-local."""

from pathlib import Path

import numpy as np
import pytest
from native_execution import require_metal


@pytest.mark.metal
def test_float32_standalone_phases_delays_and_trials(monkeypatch, tmp_path):
    b = pytest.importorskip('brian2')
    mx = require_metal('Metal is required')
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
    from test_mlx_configured import graph, inputs
    from test_mlx_float32 import assert_bits

    monkeypatch.setitem(b.prefs, 'core.default_float_dtype', np.float32)
    monkeypatch.setitem(b.prefs, 'codegen.target', 'cython')
    monkeypatch.setitem(b.prefs, 'codegen.runtime.cython.cache_dir', str(
        Path(__file__).resolve().parents[1] / 'executions/cython-default-flags'))
    cases = []
    c = graph()
    for delay in (0, 1, 18, 33):
        dt = .13 if delay == 33 else .1
        parameters = dict(t_dly=delay*dt*b.ms, t_rfc=.29995*b.ms,
            eqs=DEFAULT_EQUATIONS+'a : 1\ncounter : integer\nflag : boolean\n',
            eq_th='(v > v_th) or (a > .5)',
            eq_rst='v=v_rst;g=0*mV;a=16777216;a+=1;a-=16777216;counter+=1;flag=not flag')
        if delay == 33:
            parameters.update(tau=7*b.ms,t_mbr=24*b.ms,w_syn=.387*b.mV,v_0=-51.25*b.mV)
        model = resolve_model(parameters, dt_ms=dt)
        stimulus = inputs(model, steps=83, trials=3, amplitudes=(60.125,-2.75,72.5))
        cases.append((model,stimulus,configured.prepare(c,stimulus,model),[]))

    b.set_device('cpp_standalone', build_on_run=False)
    try:
        for model,stimulus,prepared,trials in cases:
            for trial,trial_id in enumerate(stimulus.trial_indices):
                clock = b.Clock(dt=model.dt_s*b.second,name='float32_standalone_clock*')
                namespace = model.brian_namespace()
                for channel,amplitude in enumerate(stimulus.amplitudes_mv):
                    namespace[f'events_{channel}'] = b.TimedArray(
                        stimulus.events[trial,:,channel],dt=clock.dt)
                    namespace[f'amplitude_{channel}'] = amplitude*b.mV
                neurons = b.NeuronGroup(3,model.eqs,method='linear',threshold=model.eq_th,
                    reset=model.eq_rst,refractory='rfc',namespace=namespace,clock=clock,
                    name='float32_standalone_neurons*')
                neurons.v=model.v_0*b.volt
                neurons.g=0*b.volt
                neurons.rfc=model.t_rfc*b.second
                edges=b.Synapses(neurons,neurons,'w : volt',on_pre='g_post += w',
                    delay=model.t_dly*b.second,clock=clock,name='float32_standalone_edges*')
                edges.connect(i=c.sources,j=c.destinations)
                edges.w=c.counts*model.w_syn*b.volt
                runners=[]
                for channel,target in enumerate(stimulus.targets):
                    subgroup=neurons[target:target+1]
                    subgroup.rfc=0*b.second
                    runners.append(subgroup.run_regularly(
                        f'v += events_{channel}(t)*amplitude_{channel}',when='synapses',order=channel))
                spikes=b.SpikeMonitor(neurons)
                names=tuple(v.name for v in prepared.program.variables.values() if v.persistent)
                monitor=b.StateMonitor(neurons,names,record=True,when='end')
                network=b.Network(neurons,edges,*runners,spikes,monitor)
                network.run(stimulus.events.shape[1]*clock.dt,namespace={})
                trials.append((neurons,spikes,monitor,trial_id))
        build=tmp_path/'cpp-default-flags'
        b.device.build(directory=str(build),run=True,with_output=False)
        for model,stimulus,prepared,trials in cases:
            native,state=configured.run_prepared(prepared,stimulus,return_state=True)
            expected=[]
            for neurons,spikes,monitor,trial_id in trials:
                expected.extend(zip(np.full(len(spikes.i),trial_id),np.asarray(spikes.i),
                    np.rint(np.asarray(spikes.t_)/model.dt_s).astype(np.int64)))
            assert list(zip(native.spikes.trials,native.spikes.neurons,native.spikes.steps))==expected
            for name,value in configured.host_state(state,prepared).items():
                assert_bits(value,np.stack([np.asarray(getattr(neurons,name+'_')[:])
                            for neurons,_,_,_ in trials]),name)
            quiet,quiet_state=configured.run_prepared(prepared,stimulus,record=False,return_state=True)
            assert not quiet.spikes.steps.size
            for name in state.values:
                np.testing.assert_array_equal(np.asarray(state.values[name]),np.asarray(quiet_state.values[name]))
            # Locate the first differing phase/step, rather than accepting
            # only a matching final state after a long refractory interval.
            partial=configured.initial_state(prepared,len(trials))
            events=mx.array(stimulus.events.reshape(-1))
            for step in range(83):
                partial,_=configured.advance(prepared,partial,events,83,length=1)
                configured.evaluate_state(partial)
                for name,value in configured.host_state(partial,prepared).items():
                    expected_state=np.stack([np.asarray(getattr(monitor,name+'_'))[:,step]
                                             for _,_,monitor,_ in trials])
                    assert_bits(value,expected_state,f'{model.delay_steps} step {step} {name}')
    finally:
        b.device.reinit()
        b.set_device('runtime')
