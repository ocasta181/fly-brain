"""Actual CPP standalone versus actual native component, one bounded build."""
from dataclasses import replace
from pathlib import Path
import hashlib,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython','cpp'])
def test_basic_math_phases_trials_and_continuation(monkeypatch,tmp_path,target):
 import brian2 as b
 import mlx.core as mx
 from mlx_backend import configured
 from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
 from mlx_backend.models import Connectome,Stimulus
 from test_mlx_float32 import assert_bits
 monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
 monkeypatch.setitem(b.prefs,'codegen.target','numpy' if target=='cpp' else target)
 monkeypatch.setitem(b.prefs,'devices.cpp_standalone.extra_make_args_unix',['-j2'])
 monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(ROOT/('executions/cython-basic32-targets-default-'+target)))
 funcs=('exp','log','expm1','log1p','exprel');n=8;steps=35;trial_ids=(41,3)
 params=dict(eqs=DEFAULT_EQUATIONS+'phase:1\ncut:1\ndz/dt=(-z+log1p(phase))/second:1\n'+''.join('s_'+f+':1\n' for f in funcs),
   eq_th='expm1(phase)+.1 >= cut',eq_rst='v=v_rst;phase+=.0001;'+ ';'.join('s_'+f+'='+f+'(phase)+.1' for f in funcs),t_rfc=.29995*b.ms)
 model=resolve_model(params,cpu_target=target)
 assert model.reference_codegen_target==model.float32_codegen==target
 empty=np.empty(0,np.int32);graph=Connectome(np.arange(n),empty,empty,empty,np.empty(0))
 events=np.empty((2,steps,0),np.uint8);stimulus=Stimulus(events,(),(),trial_ids,20261018,0,hashlib.sha256(events.tobytes()).hexdigest(),model.dt_ms)
 prepared=configured.prepare(graph,stimulus,model)
 initial=np.array([.01,.1,.2,.3,.7,1.1,2.1,3.3],np.float32)
 phases=np.stack([initial,initial[::-1]])
 cuts=np.expm1(phases)+np.float32(.1);cuts[:,1::2]=np.nextafter(cuts[:,1::2],np.float32(np.inf))
 original=configured.initial_state
 def seed(p,trials=1):
  state=original(p,trials);values=dict(state.values)
  for name,value in [('phase',phases),('cut',cuts)]:values[name]=mx.array(value.reshape(-1).view(np.uint32))
  state=replace(state,values=configured._pack_values(p.program,values));configured.evaluate_state(state);return state
 monkeypatch.setattr(configured,'initial_state',seed)
 names=tuple(v.name for v in prepared.program.variables.values() if v.persistent)
 b.set_device('cpp_standalone',build_on_run=False) if target=='cpp' else b.set_device('runtime')
 cpus=[]
 try:
  for trial in range(2):
   clock=b.Clock(dt=model.dt_s*b.second,name='basic_math32_'+target+'_clock_'+str(trial))
   group=b.NeuronGroup(n,model.eqs,method='linear',threshold=model.eq_th,reset=model.eq_rst,refractory='rfc',namespace=model.brian_namespace(),clock=clock)
   group.v=model.v_0*b.volt;group.g=0*b.volt;group.rfc=model.t_rfc*b.second;group.phase=phases[trial];group.cut=cuts[trial]
   spikes=b.SpikeMonitor(group);states=b.StateMonitor(group,names,record=True,when='end');network=b.Network(group,spikes,states)
   for length in (1,0,2,32):network.run(length*clock.dt,namespace={})
   cpus.append((group,spikes,states))
  if target=='cpp':b.device.build(directory=str(tmp_path/'cpp-default-flags'),run=True,with_output=False)
  expected=[]
  for trial,(group,spikes,states) in enumerate(cpus):
   expected.extend(zip(np.full(len(spikes.i),trial_ids[trial]),np.asarray(spikes.i),np.rint(np.asarray(spikes.t_)/model.dt_s).astype(np.int64)))
  expected.sort(key=lambda x:(x[0],x[2],x[1]))
  for record in (False,True):
   result,state=configured.run_prepared(prepared,stimulus,record=record,return_state=True,chunk_steps=32,block_windows=1)
   assert list(zip(result.spikes.trials,result.spikes.neurons,result.spikes.steps))==(expected if record else [])
   for name,actual in configured.host_state(state,prepared).items():assert_bits(actual,np.stack([np.asarray(g.variables[name].get_value()) for g,_,_ in cpus]),name)
  partial=seed(prepared,2);uploaded=mx.array(events.reshape(-1) if events.size else np.zeros(1,np.uint8))
  for step in range(steps):
   partial,_=configured.advance(prepared,partial,uploaded,steps,length=1)
   configured.evaluate_state(partial)
   for name,value in configured.host_state(partial,prepared).items():
    assert_bits(value,np.stack([np.asarray(getattr(m,name+'_'))[:,step] for _,_,m in cpus]),f'{step} {name}')
 finally:
  if target=='cpp':b.device.reinit()
  b.set_device('runtime')
