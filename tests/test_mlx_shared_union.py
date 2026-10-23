"""Shared state on the canonical packing/arithmetic/schedule implementation."""
from state_assertions import assert_array_bits_equal as bits_equal, assert_spikes_equal
from hashlib import sha256
from dataclasses import replace

import numpy as np
import pytest

mx=pytest.importorskip('mlx.core')
b=pytest.importorskip('brian2')
if not mx.metal.is_available():pytest.skip('Metal required',allow_module_level=True)

from mlx_backend import configured
from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
from mlx_backend.models import Connectome,Stimulus
from configured_reference import reference

pytestmark=pytest.mark.metal


def case(target,shared_voltage=False,*,steps=5,trials=2,dtype='float64'):
    equations=DEFAULT_EQUATIONS
    reset='v=v_rst+drive;g=0*mV;seen=held'
    if shared_voltage:
        equations=equations.replace('dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)','v:volt(shared)')
        reset='g=0*mV;seen=held'
    else:equations=equations.replace('v_0 - v + g','v_0 - v + g + drive')
    equations+='\nheld=v/mV+t/ms:1(shared,constant over dt)\ndrive=held*mV:volt(shared)\nseen:1\n' if shared_voltage else '\nheld=0.001*t/ms:1(shared,constant over dt)\ndrive=held*mV:volt(shared)\nseen:1\n'
    model=resolve_model(dict(eqs=equations,eq_th='v>v_th+drive' if not shared_voltage else 'v>v_th',eq_rst=reset),cpu_target=target,state_dtype=dtype)
    empty=np.empty(0,np.int32);weights=np.empty(0,np.float64)
    graph=Connectome(np.arange(4),empty,empty.copy(),weights,weights.copy())
    events=np.zeros((trials,steps,3),np.uint8)
    if steps:
        events[:,0,:]=1
        if trials>1:events[1,0,1:]=0
        if steps>2:events[0,2,0]=1
    stimulus=Stimulus(events,(0,0,2),(30.,-3.,5.),tuple(range(8,8+trials)),123,0,
        sha256(events.tobytes()).hexdigest(),model.dt_ms)
    return graph,stimulus,model




@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('shared_voltage',[False,True])
@pytest.mark.parametrize('dtype',['float64','float32'])
def test_shared_phase_smoke_matches_independent_cpu(tmp_path,monkeypatch,target,shared_voltage,dtype):
    monkeypatch.setitem(b.prefs,"core.default_float_dtype",np.dtype(dtype).type)
    monkeypatch.setitem(b.prefs,"codegen.target",target)
    b.prefs.codegen.runtime.cython.cache_dir=str(tmp_path/'brian-default')
    graph,stimulus,model=case(target,shared_voltage,dtype=dtype)
    expected=reference(graph,stimulus,model,record_state=True)
    np.savez(tmp_path/'cpu.npz',**expected.traces,spike_trials=expected.spikes.trials,
             spike_neurons=expected.spikes.neurons,spike_steps=expected.spikes.steps)
    prepared=configured.prepare(graph,stimulus,model)
    state=configured.initial_state(prepared,2)
    assert all(state.values[v.name].size==2 for v in prepared.program.variables.values() if v.scalar and v.persistent)
    if shared_voltage:assert not prepared.program.scalar_parallel
    events=mx.array(stimulus.events.reshape(-1));spikes=[];states={name:[] for name in expected.traces}
    for step in range(stimulus.events.shape[1]):
        state,fired=configured.advance(prepared,state,events,stimulus.events.shape[1],1)
        configured.evaluate_state(state)
        for name,actual in configured.host_state(state,prepared).items():
            states[name].append(actual.copy())
            np.savez(tmp_path/"native-prefix.npz",**{key:np.stack(value,axis=1) for key,value in states.items() if value})
            bits_equal(actual,expected.traces[name][:,step],name)
        for row in np.flatnonzero(np.asarray(fired)):
            spikes.append((stimulus.trial_indices[row//4],row%4,step))
    observed=sorted(spikes,key=lambda row:(row[0],row[2],row[1]))
    assert observed==list(zip(expected.spikes.trials,expected.spikes.neurons,expected.spikes.steps))
    np.savez(tmp_path/'native.npz',**{name:np.stack(value,axis=1) for name,value in states.items()})


@pytest.mark.parametrize('target',['numpy','cython'])
def test_shared_recording_chunks_and_pending_delays_match_cpu(tmp_path,monkeypatch,target):
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    b.prefs.codegen.runtime.cython.cache_dir=str(tmp_path/'brian-default')
    graph,stimulus,model=case(target,True,steps=35)
    graph=replace(graph,sources=np.array([0,2,0],np.int32),destinations=np.array([1,3,2],np.int32),
        counts=np.array([.25,-.125,.5]),weights_mv=np.zeros(3))
    expected=reference(graph,stimulus,model,record_state=True)
    prepared=configured.prepare(graph,stimulus,model)
    actual,recorded=configured.run_prepared(prepared,stimulus,record=True,return_state=True,
        chunk_steps=17,block_windows=1)
    assert_spikes_equal(actual.spikes, expected.spikes)
    quiet,unrecorded=configured.run_prepared(prepared,stimulus,record=False,return_state=True,
        chunk_steps=17,block_windows=1)
    assert not quiet.spikes.steps.size
    for name,values in configured.host_state(recorded,prepared).items():
        bits_equal(values,expected.state[name],name)
        bits_equal(configured.host_state(unrecorded,prepared)[name],expected.state[name],name)
    state=configured.initial_state(prepared,2)
    events=mx.array(stimulus.events.reshape(-1));spikes=[]
    for begin,end in ((0,1),(1,17),(17,35)):
        while state.step<end:
            offset=state.step;length=min(end-offset,prepared.window-offset%prepared.window)
            state,word=configured.advance(prepared,state,events,end,length,event_steps=35)
            configured.evaluate_state(state)
            for bit in range(length):
                for row in np.flatnonzero(np.asarray(word)&(1<<bit)):
                    spikes.append((stimulus.trial_indices[row//4],row%4,offset+bit))
    assert sorted(spikes,key=lambda row:(row[0],row[2],row[1]))==list(zip(expected.spikes.trials,expected.spikes.neurons,expected.spikes.steps))
    for name,values in configured.host_state(state,prepared).items():bits_equal(values,expected.state[name],name)


@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('voltage',[False,True])
@pytest.mark.parametrize('dtype',['float64','float32'])
def test_actual_brian_api_shared_state_rng_and_continuation(tmp_path,monkeypatch,target,voltage,dtype):
    from original_api_reference import cpu_runtime,run_original_api,capture_rng_state,assert_rng_state_equal,assert_snapshot_equal
    from mlx_backend.brian_device import original_api
    equations=DEFAULT_EQUATIONS
    reset='v=v_rst+drive;g=0*mV'
    if voltage:
        equations=equations.replace('dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)','v:volt(shared)')
        reset='g=0*mV'
    else:equations=equations.replace('v_0 - v + g','v_0 - v + g + drive')
    equations+='\nheld=0.001*t/ms:1(shared,constant over dt)\ndrive=held*mV:volt(shared)\n'
    options=dict(parameters=dict(eqs=equations,eq_rst=reset,
        r_poi=3700*b.Hz,r_poi2=7200*b.Hz,t_dly=.3*b.ms))
    with cpu_runtime(seed=875):
        b.prefs.codegen.target=target
        b.prefs.core.default_float_dtype=np.dtype(dtype).type
        expected=run_original_api(tmp_path/'cpu',[.15,.15],**options)
        assert expected.neurons.variables['v'].dtype==np.dtype(dtype)
        expected_rng=capture_rng_state();next_draw=np.random.random(10)
    with cpu_runtime(seed=875):
        b.prefs.codegen.target=target
        b.prefs.core.default_float_dtype=np.dtype(dtype).type
        with original_api(workers=False) as device:
            actual=run_original_api(tmp_path/'metal',[.15,.15],**options)
            assert actual.neurons.variables['v'].dtype==np.dtype(dtype)
            assert device.native_runs==2
            if not voltage:assert device.last_timings['prepass_and_dependencies_s']==0
            for left,right in zip(actual.snapshots,expected.snapshots):assert_snapshot_equal(left,right)
            assert_rng_state_equal(capture_rng_state(),expected_rng)
            np.testing.assert_array_equal(np.random.random(10),next_draw)


@pytest.mark.parametrize('delay',[0,33])
@pytest.mark.parametrize('shared',[False,True])
def test_vector_before_start_runs_once_across_delivery_split(tmp_path,monkeypatch,delay,shared):
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    graph,stimulus,model=case('numpy',True,steps=5)
    equations=(model.eqs if shared else DEFAULT_EQUATIONS+'\nheld=t/ms:1(constant over dt)\nseen:1\n')+'''
dx/dt=(y-x)/second:1
dy/dt=(held-y)/(2*second):1
frozen=x+y:1(constant over dt)
'''
    model=resolve_model(dict(eqs=equations,eq_th='True',eq_rst='g=0*mV;seen=frozen',t_dly=delay*.1*b.ms),cpu_target='numpy')
    graph=replace(graph,sources=np.array([0],np.int32),destinations=np.array([1],np.int32),
        counts=np.array([.25]),weights_mv=np.zeros(1))
    expected=reference(graph,stimulus,model,record_state=True)
    prepared=configured.prepare(graph,stimulus,model)
    state=configured.initial_state(prepared,2);events=mx.array(stimulus.events.reshape(-1))
    for step in range(5):
        state,_=configured.advance(prepared,state,events,5,1)
        configured.evaluate_state(state)
        for name,actual in configured.host_state(state,prepared).items():bits_equal(actual,expected.traces[name][:,step],name)


def test_shared_cython_division_error_after_exact_valid_prefix(tmp_path,monkeypatch):
    monkeypatch.setitem(b.prefs,'codegen.target','cython')
    b.prefs.codegen.runtime.cython.cache_dir=str(tmp_path/'brian-default')
    graph,stimulus,_=case('cython',False,steps=3,trials=1)
    equations=DEFAULT_EQUATIONS+'''
denom=2-int(timestep(t,dt)):1(shared,constant over dt)
held=1/denom:1(shared,constant over dt)
'''
    model=resolve_model(dict(eqs=equations,eq_th='v>v_th+held*mV'),cpu_target='cython')
    prefix=replace(stimulus,events=stimulus.events[:,:2])
    expected=reference(graph,prefix,model,record_state=True)
    with pytest.raises(b.BrianObjectException) as error:reference(graph,stimulus,model,record_state=True)
    assert isinstance(error.value.__cause__,ZeroDivisionError)
    prepared=configured.prepare(graph,stimulus,model);state=configured.initial_state(prepared,1)
    events=mx.array(stimulus.events.reshape(-1))
    state,_=configured.advance(prepared,state,events,3,2)
    configured.evaluate_state(state)
    for name,actual in configured.host_state(state,prepared).items():bits_equal(actual,expected.state[name],name)
    state,_=configured.advance(prepared,state,events,3,1)
    with pytest.raises(ZeroDivisionError,match='float division'):configured.evaluate_state(state)


def test_before_start_error_retains_first_step_and_row(tmp_path,monkeypatch):
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    graph,stimulus,_=case('numpy',False,steps=4,trials=1)
    equations=DEFAULT_EQUATIONS+'''
bad_scalar=2**(-int(t>=.3*ms)):integer(shared,constant over dt)
bad_vector=2**(-int((i==3) and (t>=.2*ms))):integer(constant over dt)
'''
    model=resolve_model(dict(eqs=equations,eq_th='False'),cpu_target='numpy')
    with pytest.raises(b.BrianObjectException) as error:reference(graph,stimulus,model,record_state=True)
    assert isinstance(error.value.__cause__,ValueError)
    prepared=configured.prepare(graph,stimulus,model)
    state,_=configured.advance(prepared,configured.initial_state(prepared,1),mx.array(stimulus.events.reshape(-1)),4,4)
    with pytest.raises(ValueError,match='negative integer powers'):configured.evaluate_state(state)
    from mlx_backend.runtime_errors import first_error
    row,_=first_error(state.release,prepared.program.runtime_error_order)
    assert row==3


@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('with_inputs',[False,True])
def test_actual_shared_storage_preserves_target_accepted_dtypes(tmp_path,target,with_inputs):
    from original_api_reference import cpu_runtime,capture_rng_state,assert_rng_state_equal
    from mlx_backend.brian_device import original_api
    fields={'s8':np.int8,'u8':np.uint8,'s16':np.int16,'u16':np.uint16,
        's32':np.int32,'u32':np.uint32,'s64':np.int64,'u64':np.uint64,
        'f32':np.float32,'f64':np.float64,'flag':np.bool_}
    offsets={'s8':127,'u8':255,'s16':32767,'u16':65535,
        's32':2147483645,'u32':4294967293,'s64':2**40,'u64':2**40}
    if target=='cython':
        from brian2.codegen.generators.cython_generator import cpp_dtype
        fields={name:dtype for name,dtype in fields.items() if np.dtype(dtype).name in cpp_dtype}
    equations=DEFAULT_EQUATIONS+'\ncopy:1\n'
    for name,dtype in fields.items():
        if name=='flag':expression,kind='t>0*ms','boolean'
        elif np.dtype(dtype).kind=='f':expression,kind='1+t/ms','1'
        else:expression,kind=f'{offsets[name]}+int(timestep(t,dt))','integer'
        equations+=f'{name}={expression}:{kind}(shared,constant over dt)\n'

    def run(native):
        with cpu_runtime(seed=123):
            b.prefs.codegen.target=target
            b.prefs.codegen.runtime.cython.cache_dir=str(tmp_path/'brian-default')
            import contextlib
            context=original_api(workers=False) if native else contextlib.nullcontext()
            with context as device:
                namespace=resolve_model().brian_namespace()
                namespace['eqs']=equations
                clock=b.Clock(dt=.1*b.ms)
                neurons=b.NeuronGroup(4,equations,threshold='t<.1*ms' if with_inputs else 'True',reset='v=v_rst;g=0*mV;copy=f32+s32+u8',
                    refractory='rfc',method='linear',namespace=namespace,clock=clock,
                    dtype={'v':np.float64,'g':np.float64,'rfc':np.float64,'copy':np.float64,**fields})
                neurons.v=-52*b.mV;neurons.g=0*b.mV;neurons.rfc=0*b.ms
                synapses=b.Synapses(neurons,neurons,'w:volt',on_pre='g_post+=w',delay=1.8*b.ms,clock=clock)
                synapses.connect(condition=False)
                spikes=b.SpikeMonitor(neurons)
                inputs=([b.PoissonInput(neurons[:1],'v',N=1,rate=3700*b.Hz,weight=.125*b.mV)] if with_inputs else [])
                network=b.Network(neurons,synapses,spikes,*inputs);snapshots=[]
                for _ in range(3):
                    network.run(.1*b.ms,namespace={})
                    snapshots.append({name:neurons.variables[name].get_value().copy()
                        for name in (*fields,'v','g','rfc','copy','lastspike','not_refractory')})
                for name,dtype in fields.items():
                    assert neurons.variables[name].dtype==np.dtype(dtype)
                    assert neurons.variables[name].get_value().shape==(1,)
                if native:
                    assert device.native_runs==3
                    if with_inputs:assert device.last_timings['prepass_and_dependencies_s']==0
                rng=capture_rng_state();draw=np.random.random(10)
                return snapshots,np.asarray(spikes.i[:]).copy(),np.asarray(spikes.t_).copy(),rng,draw
    expected=run(False);actual=run(True)
    for left,right in zip(actual[0],expected[0]):
        for name,value in left.items():bits_equal(value,right[name],name)
    np.testing.assert_array_equal(actual[1],expected[1]);bits_equal(actual[2],expected[2],'spike time')
    assert_rng_state_equal(actual[3],expected[3]);np.testing.assert_array_equal(actual[4],expected[4])


@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('dtype',['float64','float32'])
def test_vector_before_start_without_shared_fields_actual_api(tmp_path,target,dtype):
    from original_api_reference import build_original_api_fixture
    from original_api_protocol import check_original_api_runs,observe_run
    def build(mode):
        fixture=build_original_api_fixture(tmp_path/str(mode=='native'),parameters=dict(
            eqs=DEFAULT_EQUATIONS+'\nheld=t/ms+i/10:1(constant over dt)\n',
            eq_th='True',eq_rst='v=v_rst;g=held*mV',t_rfc=0*b.ms),
            exc=(),exc2=(),silenced=())
        assert not fixture.neurons.variables['held'].scalar
        assert fixture.neurons.variables['v'].dtype==np.dtype(dtype)
        return fixture
    def exercise(fixture,mode):
        for duration in (0.,.15,.15):
            yield observe_run(fixture,duration)
    check_original_api_runs(build,exercise,None,guard=lambda fixture,patch:None,
        target=target,float_dtype=np.dtype(dtype).type)


def test_empty_scalar_trace_with_only_index_dependent_vector_update(tmp_path,monkeypatch):
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    graph,stimulus,_=case('numpy',False,steps=3,trials=1)
    model=resolve_model(dict(eqs=DEFAULT_EQUATIONS+'\nheld=i:integer(constant over dt)\n',
        eq_th='True',eq_rst='v=v_rst;g=held*mV'),cpu_target='numpy')
    expected=reference(graph,stimulus,model,record_state=True)
    prepared=configured.prepare(graph,stimulus,model)
    assert not any(v.scalar and v.persistent for v in prepared.program.variables.values())
    state=configured.initial_state(prepared,1)
    events=mx.array(stimulus.events.reshape(-1))
    for step in range(3):
        state,_=configured.advance(prepared,state,events,3,1)
        configured.evaluate_state(state)
        for name,value in configured.host_state(state,prepared).items():bits_equal(value,expected.traces[name][:,step],name)


@pytest.mark.parametrize('neurons',[5,17])
def test_compiled_before_start_simd_prefix_and_scalar_tail(tmp_path,monkeypatch,neurons):
    monkeypatch.setitem(b.prefs,'codegen.target','cython')
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    b.prefs.codegen.runtime.cython.cache_dir=str(tmp_path/'brian-default')
    graph,stimulus,_=case('cython',False,steps=3,trials=1,dtype='float32')
    graph=replace(graph,neuron_ids=np.arange(neurons))
    model=resolve_model(dict(eqs=DEFAULT_EQUATIONS+'\nheld=t/ms+i/10:1(constant over dt)\n',
        eq_th='True',eq_rst='v=v_rst;g=held*mV',t_rfc=0*b.ms),cpu_target='cython',state_dtype='float32')
    expected=reference(graph,stimulus,model,record_state=True)
    prepared=configured.prepare(graph,stimulus,model)
    state=configured.initial_state(prepared,1);events=mx.array(stimulus.events.reshape(-1))
    for step in range(3):
        state,_=configured.advance(prepared,state,events,3,1)
        configured.evaluate_state(state)
        for name,value in configured.host_state(state,prepared).items():bits_equal(value,expected.traces[name][:,step],name)


@pytest.mark.parametrize('target',['numpy','cython'])
def test_before_start_failure_matches_actual_brian_state_rollback(tmp_path,target):
    from contextlib import nullcontext
    from mlx_backend.brian_device import original_api
    from original_api_reference import cpu_runtime,build_original_api_fixture,assert_snapshot_equal
    if target=='numpy':
        extra='scalar=t/ms:1(shared,constant over dt)\nheld=(1+t/ms)*(2**(-int((i==3) and (t>=.2*ms)))):1(constant over dt)\n'
        expected_error=ValueError
    else:
        extra='scalar=t/ms:1(shared,constant over dt)\ndenom=2-int(timestep(t,dt)):1(shared,constant over dt)\nheld=1/denom:1(shared,constant over dt)\n'
        expected_error=ZeroDivisionError
    results=[]
    for native in (False,True):
        with cpu_runtime():
            b.prefs.codegen.target=target
            b.prefs.codegen.runtime.cython.cache_dir=str(tmp_path/'brian-default')
            with original_api(workers=False) if native else nullcontext():
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=dict(
                    eqs=DEFAULT_EQUATIONS+'\n'+extra,eq_th='True',eq_rst='v=v_rst;g=0*mV',t_rfc=0*b.ms),
                    exc=(),exc2=(),silenced=())
                with pytest.raises(b.BrianObjectException) as error:fixture.run_ms(.4)
                assert isinstance(error.value.__cause__,expected_error)
                results.append(fixture.capture())
    assert_snapshot_equal(results[1],results[0])


@pytest.mark.parametrize('expression',['0','int(i>=2)'])
def test_compiled_vector_before_start_zero_and_comparison(tmp_path,monkeypatch,expression):
    monkeypatch.setitem(b.prefs,'codegen.target','cython')
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    b.prefs.codegen.runtime.cython.cache_dir=str(tmp_path/'brian-default')
    graph,stimulus,_=case('cython',False,steps=3,trials=1,dtype='float32')
    model=resolve_model(dict(eqs=DEFAULT_EQUATIONS+f'\nheld={expression}:1(constant over dt)\n',
        eq_th='True',eq_rst='v=v_rst;g=held*mV',t_rfc=0*b.ms),cpu_target='cython',state_dtype='float32')
    expected=reference(graph,stimulus,model,record_state=True)
    prepared=configured.prepare(graph,stimulus,model)
    state=configured.initial_state(prepared,1);events=mx.array(stimulus.events.reshape(-1))
    for step in range(3):
        state,_=configured.advance(prepared,state,events,3,1)
        configured.evaluate_state(state)
        for name,value in configured.host_state(state,prepared).items():bits_equal(value,expected.traces[name][:,step],name)


@pytest.mark.parametrize('target,activated',[('numpy',False),('cython',True)])
def test_shared_defaults_refractory_and_neuron_count_identity(tmp_path,monkeypatch,target,activated):
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    b.prefs.codegen.runtime.cython.cache_dir=str(tmp_path/'brian-default')
    graph,stimulus,_=case(target,False,steps=3)
    if not activated:
        events=np.zeros((2,3,0),np.uint8)
        stimulus=replace(stimulus,events=events,targets=(),rates_hz=(),sha256=sha256(events.tobytes()).hexdigest())
    equations=DEFAULT_EQUATIONS.replace('rfc : second','rfc : second (shared)')+'''
base:1(shared,constant)
count:integer(shared)
flag:boolean(shared)
ticks=timestep(t,dt)+N+count+int(flag)+int(base):integer(shared,constant over dt)
'''
    model=resolve_model(dict(eqs=equations,base=100.,count=91,flag=True,
        eq_th='True',eq_rst='v=v_rst+ticks*mV;g=0*mV'),cpu_target=target)
    first=None
    for neurons in (4,7):
        current=replace(graph,neuron_ids=np.arange(neurons))
        expected=reference(current,stimulus,model,record_state=True)
        prepared=configured.prepare(current,stimulus,model)
        state=configured.initial_state(prepared,2)
        if first is None:first=state
        else:
            with pytest.raises(ValueError,match='different model or neuron count'):
                configured.advance(prepared,first,mx.array(stimulus.events.reshape(-1)),3,1)
        events=mx.array(stimulus.events.reshape(-1) if activated else np.zeros(1,np.uint8))
        for step in range(3):
            state,_=configured.advance(prepared,state,events,3,1)
            configured.evaluate_state(state)
            for name,value in configured.host_state(state,prepared).items():bits_equal(value,expected.traces[name][:,step],name)
        for name in ('base','count','flag'):
            assert np.all(np.asarray(state.values[name])==0)
            assert state.values[name].shape==(2,)
        np.testing.assert_array_equal(np.asarray(state.values['ticks']),[neurons+2]*2)
        np.testing.assert_array_equal(np.asarray(state.values['rfc']).view(np.float64),
            [0. if activated else model.t_rfc]*2)


def test_shared_synaptic_target_preserves_actual_brian_rejection(monkeypatch):
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    graph,stimulus,_=case('numpy',False,steps=1,trials=1)
    equations=DEFAULT_EQUATIONS.replace('dg/dt = -g / tau : volt (unless refractory)','g:volt(shared)')
    model=resolve_model(dict(eqs=equations,eq_rst='v=v_rst'),cpu_target='numpy')
    with pytest.raises(b.BrianObjectException) as error:reference(graph,stimulus,model)
    assert 'Writing to scalar variable g_post' in str(error.value.__cause__)
    with pytest.raises(SyntaxError,match='Writing to scalar variable g_post'):
        configured.prepare(graph,stimulus,model)


@pytest.mark.parametrize('target,kind',[
    ('numpy','vector32'),('cython','vector32'),
    ('numpy','shared_rfc'),('cython','shared_rfc'),('numpy','schedule'),
    ('cython','q'),('cython','row')])
def test_fused_before_start_inputs_and_shared_refractory(tmp_path,target,kind):
    from contextlib import nullcontext
    from mlx_backend.brian_device import original_api
    from original_api_reference import cpu_runtime,build_original_api_fixture,assert_snapshot_equal,capture_rng_state,assert_rng_state_equal
    equations=DEFAULT_EQUATIONS.replace('v_0 - v + g','v_0 - v + g + drive')
    scalar='' if kind in ('vector32','q','row') else 'shared,'
    equations+=f'\nheld=0.001*t/ms:1({scalar}constant over dt)\ndrive=held*mV:volt\n'
    if kind in ('q','row'):equations=equations.replace('held',kind)
    if kind=='shared_rfc':equations=equations.replace('rfc : second','rfc : second (shared)')
    expected=[]
    for native in (False,True):
        with cpu_runtime(seed=9021):
            b.prefs.codegen.target=target
            b.prefs.core.default_float_dtype=np.float32 if kind=='vector32' else np.float64
            b.prefs.codegen.runtime.cython.cache_dir=str(tmp_path/'brian-default')
            with original_api(workers=False) if native else nullcontext() as device:
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=dict(
                    eqs=equations,eq_th='v>v_th+drive',eq_rst='v=v_rst+drive;g=0*mV',
                    r_poi=3700*b.Hz,r_poi2=7200*b.Hz,t_dly=.3*b.ms))
                if kind=='schedule':fixture.network.schedule=['start','resets','groups','thresholds','synapses','end']
                for index,duration in enumerate((.15,.25)):
                    current=fixture.run_ms(duration)
                    if native:assert_snapshot_equal(current,expected[index])
                    else:expected.append(current)
                if native:
                    assert_rng_state_equal(capture_rng_state(),rng)
                    np.testing.assert_array_equal(np.random.random(10),draw)
                    if kind!='schedule':assert device.last_timings['prepass_and_dependencies_s']==0
                else:rng=capture_rng_state();draw=np.random.random(10)


@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('dtype',['float32','float64'])
def test_shared_nonfinite_payload_and_signed_zero_transport(tmp_path,target,dtype):
    from contextlib import nullcontext
    from mlx_backend.brian_device import original_api
    from original_api_reference import cpu_runtime,build_original_api_fixture,assert_snapshot_equal
    word=np.dtype('uint'+dtype[5:])
    payload=0x7fc01234 if dtype=='float32' else 0x7ff8000000001234
    initial={'nan':np.array(payload,dtype=word).view(dtype)[()],
             'inf':np.dtype(dtype).type(np.inf),
             'zero':np.dtype(dtype).type(-0.)}
    equations=DEFAULT_EQUATIONS
    for name in initial:
        equations+=f'\nx_{name}:1(shared,constant)\nheld_{name}=x_{name}:1(shared,constant over dt)\nseen_{name}:1\n'
    expected=[]
    for native in (False,True):
        with cpu_runtime():
            b.prefs.codegen.target=target
            b.prefs.core.default_float_dtype=np.dtype(dtype).type
            b.prefs.codegen.runtime.cython.cache_dir=str(tmp_path/'brian-default')
            with original_api(workers=False) if native else nullcontext():
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=dict(eqs=equations,
                    eq_th='True',eq_rst='v=v_rst;g=0*mV;'+';'.join(f'seen_{name}=held_{name}' for name in initial),
                    t_rfc=0*b.ms),exc=(),exc2=(),silenced=(),
                    initial_state={'x_'+name:value for name,value in initial.items()})
                for index in range(2):
                    current=fixture.run_ms(.1)
                    if native:assert_snapshot_equal(current,expected[index])
                    else:expected.append(current)
                    for name,value in initial.items():
                        expected_bits=np.array(value,dtype=dtype).view(word)[()]
                        for prefix in ('x_','held_','seen_'):
                            assert np.all(current.state[prefix+name].view(word)==expected_bits)


@pytest.mark.parametrize('dtype',[np.int8,np.int16,np.uint16,np.uint32])
def test_cython_shared_dtype_rejection_matches_actual_brian(tmp_path,dtype):
    from contextlib import nullcontext
    from mlx_backend.brian_device import original_api
    from original_api_reference import cpu_runtime
    for native in (False,True):
        with cpu_runtime():
            b.prefs.codegen.target='cython'
            b.prefs.codegen.runtime.cython.cache_dir=str(tmp_path/'brian-default')
            with original_api(workers=False) if native else nullcontext():
                equations=DEFAULT_EQUATIONS+'\nnarrow=1:integer(shared,constant over dt)\n'
                clock=b.Clock(dt=.1*b.ms)
                neurons=b.NeuronGroup(2,equations,threshold='True',reset='v=v_rst;g=0*mV',
                    refractory='rfc',method='linear',namespace=resolve_model().brian_namespace(),
                    clock=clock,dtype={'narrow':dtype})
                neurons.v=-52*b.mV;neurons.g=0*b.mV;neurons.rfc=0*b.ms
                synapses=b.Synapses(neurons,neurons,'w:volt',on_pre='g_post+=w',delay=1.8*b.ms,clock=clock)
                synapses.connect(condition=False);spikes=b.SpikeMonitor(neurons)
                with pytest.raises(b.BrianObjectException) as error:b.Network(neurons,synapses,spikes).run(.1*b.ms)
                assert isinstance(error.value.__cause__,KeyError)
                assert error.value.__cause__.args==(np.dtype(dtype).name,)
