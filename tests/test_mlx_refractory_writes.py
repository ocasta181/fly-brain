"""Actual selected CPU arithmetic for independent refractory write guards."""
from state_assertions import assert_recorded_spikes_equal
from native_execution import forbid_call, require_metal
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

b=pytest.importorskip('brian2')
mx = require_metal('Metal is required', allow_module_level=True)
from configured_reference import reference
from test_mlx_configured import graph,inputs
from mlx_backend import configured
from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model

pytestmark=pytest.mark.metal


def equations(guard_v,guard_g):
    lines=DEFAULT_EQUATIONS.splitlines()
    for index,guard in enumerate((guard_v,guard_g)):
        if not guard:lines[index]=lines[index].replace(' (unless refractory)','')
    return '\n'.join(lines)+'\n'


def assert_raw(actual,expected,label):
    actual,expected=np.asarray(actual),np.asarray(expected)
    assert actual.shape==expected.shape,label
    assert actual.dtype==expected.dtype,label
    np.testing.assert_array_equal(actual.view(np.dtype('u'+str(actual.dtype.itemsize))),
        expected.view(np.dtype('u'+str(expected.dtype.itemsize))),err_msg=label)


@pytest.mark.parametrize('target,guard_v,guard_g',[
    ('cython',False,False),('cython',False,True),('cython',True,False),
    ('cython',True,True),('numpy',False,False),('cython_strict',False,False),
    ('cython_float32',False,False),
])
def test_actual_cpu_refractory_guards_states_spikes_recording_and_continuation(monkeypatch,target,guard_v,guard_g):
    monkeypatch.setitem(b.prefs,'codegen.target','cython' if target.startswith('cython') else target)
    cache=Path(__file__).resolve().parents[1]/'executions/refractory-write/cython-cache'/target
    monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(cache))
    if target=='cython_float32':
        monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    if target=='cython_strict':
        monkeypatch.setitem(b.prefs,'codegen.cpp.extra_compile_args',['-O3','-fno-fast-math','-ffp-contract=off'])
    model=resolve_model({'eqs':equations(guard_v,guard_g),'eq_th':'(v > v_th) or False','t_dly':0*b.ms,'t_rfc':3.5*b.ms})
    c,s=graph(),inputs(model,steps=81,trials=2)
    expected=reference(c,s,model,record_state=True)
    if target=='cython' and not (guard_v and guard_g):
        assert model.integer_policy is not None
        saved_flags=model.integer_policy.flags
        # Preparation must retain the resolved target even if ambient flags
        # change after the independent reference has been constructed.
        monkeypatch.setitem(b.prefs,'codegen.cpp.extra_compile_args',['-O0','-ffp-contract=off'])
        assert model.integer_policy.flags==saved_flags
    prepared=configured.prepare(c,s,model)
    assert prepared.model.reference_codegen_target==('cython' if target.startswith('cython') else target)
    if target.startswith('cython') and model.state_dtype=='float64' and not (guard_v and guard_g):
        assert prepared.program.refractory_policy=='cython_fast'
    forbid_call(b.Network, 'run', 'CPU timestep during native run', monkeypatch)
    staged=mx.array(s.events.reshape(-1));state=configured.initial_state(prepared,2)
    fired=[]
    for step in range(s.events.shape[1]):
        state,bits=configured.advance(prepared,state,staged,s.events.shape[1],1)
        configured.evaluate_state(state)
        for name,actual in configured.host_state(state,prepared).items():
            assert_raw(actual,expected.traces[name][:,step],f'{name} end step{step}')
        for trial,row in np.argwhere(np.asarray(bits).reshape(2,3)&1):
            fired.append((s.trial_indices[trial],int(row),step))
    actual_spikes=list(zip(expected.spikes.trials,expected.spikes.neurons,expected.spikes.steps))
    assert sorted(fired)==sorted(actual_spikes)
    for record in (False,True):
        result,continued=configured.run_prepared(prepared,s,record=record,return_state=True,chunk_steps=19,block_windows=2)
        for name,value in configured.host_state(continued,prepared).items():
            assert_raw(value,expected.state[name],f'final {name}, record={record}')
        assert_recorded_spikes_equal(result, expected, record)
        assert result.input_sha256==tuple(sha256(row.tobytes()).hexdigest() for row in s.events)


def test_cpp_guard_variants_use_actual_standalone_arithmetic(tmp_path,monkeypatch):
    """One small standalone build independently covers all changed guards."""
    from dataclasses import replace
    from mlx_backend.models import Connectome
    from brian2.devices.device import reset_device
    monkeypatch.setitem(b.prefs,'devices.cpp_standalone.extra_make_args_unix',['-j2'])
    names=('v','g','rfc','lastspike','not_refractory')
    initial_v=np.array([-.043,-.046,-.049])
    initial_g=np.array([.003,.03,-.002])
    objects=[];references=[]
    b.set_device('cpp_standalone',directory=str(tmp_path/'cpp'),build_on_run=True)
    try:
        for index,(guard_v,guard_g) in enumerate(((False,False),(False,True),(True,False))):
            model=resolve_model({'eqs':equations(guard_v,guard_g),'eq_th':'v>v_th or t<dt',
                'eq_rst':'v=v_rst; g=3*mV','t_rfc':.35*b.ms,'t_dly':0*b.ms})
            clock=b.Clock(dt=model.dt_s*b.second)
            group=b.NeuronGroup(3,model.eqs,method='linear',threshold=model.eq_th,
                reset=model.eq_rst,refractory='rfc',namespace=model.brian_namespace(),clock=clock,
                name=f'refractory_write_cpp_{index}')
            group.v=initial_v*b.volt;group.g=initial_g*b.volt;group.rfc=model.t_rfc*b.second
            monitor=b.StateMonitor(group,names,record=True,when='end')
            spikes=b.SpikeMonitor(group)
            objects.extend((group,monitor,spikes));references.append((model,monitor,spikes))
        b.Network(*objects).run(39*model.dt_s*b.second,namespace={})
        saved=[(model,{name:np.asarray(getattr(monitor,name+'_')).T.copy() for name in names},
                np.asarray(spikes.i[:]).copy(),np.rint(np.asarray(spikes.t_)/model.dt_s).astype(np.int64))
               for model,monitor,spikes in references]
    finally:reset_device()
    c=Connectome(np.arange(3),np.array([],np.int32),np.array([],np.int32),np.array([],float),np.array([],float))
    forbid_call(b.Network, 'run', 'CPU timestep during native run', monkeypatch)
    for model,traces,spike_rows,spike_steps in saved:
        s=inputs(model,targets=(),steps=39,trials=1)
        p=configured.prepare(c,s,model)
        state=configured.initial_state(p)
        values=dict(state.values)
        values.update(v=mx.array(initial_v.view(np.uint64)),g=mx.array(initial_g.view(np.uint64)))
        state=replace(state,values=values)
        fired=[]
        for step in range(39):
            state,bits=configured.advance(p,state,mx.zeros((1,),mx.uint8),39,1)
            configured.evaluate_state(state)
            for name,value in configured.host_state(state,p).items():
                assert_raw(value[0],traces[name][step],f'CPP {name},step{step}')
            fired.extend((int(row),step) for row in np.flatnonzero(np.asarray(bits)&1))
        assert sorted(fired)==sorted(zip(spike_rows,spike_steps))


@pytest.mark.parametrize('record',[False,True])
def test_explicit_cython_refractory_writes_compose_with_precision_retry(record,tmp_path,monkeypatch):
    """Saved Cython target, unguarded state and checked reset share replay."""
    from mlx_backend.model import DEFAULT_RESET
    # The explicit target must beat ambient preferences during both capture
    # and capacity-specialized recompilation.
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'core.default_integer_dtype',np.int64)
    monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(tmp_path/'cpu'))
    model=resolve_model(dict(
        eqs=equations(False,False)+'x:integer\ny:integer\np:integer\nq:integer\nseen:1\n',
        eq_th='True',t_rfc=.35*b.ms,t_dly=0*b.ms,
        eq_rst=DEFAULT_RESET+';g=30*mV;x=2;y=3;p+=1;q+=1;seen+=abs(x)**(p+60)-abs(y)**(q+38)',
    ),cpu_target='cython')
    assert model.reference_codegen_target==model.integer_policy.target=='cython'
    c,s=graph(),inputs(model,steps=8,trials=1)
    expected=reference(c,s,model,cpu_target='cython')
    compile_=configured._compiled_program;capacities=[]
    def small_capacity(model,*args,**kwargs):
        capacity=kwargs.pop('native_precision_limbs',2);capacities.append(capacity)
        return compile_(model,*args,native_precision_limbs=capacity,**kwargs)
    monkeypatch.setattr(configured,'_compiled_program',small_capacity)
    prepared=configured.prepare(c,s,model,component=True)
    assert prepared.program.native_precision_limbs==2
    assert 'state' in prepared.program.compiled_integer_stages
    forbid_call(b.Network, 'run', 'CPU timestep during native retry', monkeypatch)
    result,state=configured.run_prepared(prepared,s,record=record,return_state=True,
        chunk_steps=3,block_windows=2)
    assert state.step==8 and state.native_precision_limbs>=4 and max(capacities)>=4
    for name,value in configured.host_state(state,prepared).items():
        assert_raw(value,expected.state[name],f'retry final {name}, record={record}')
    assert_recorded_spikes_equal(result, expected, record)
