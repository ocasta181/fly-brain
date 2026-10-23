"""Independent Brian Cython model scheduling, buffers and portable checkpoints."""
import ctypes
from dataclasses import replace
import numpy as np
import pytest
from state_assertions import CompiledRngComparison
from mlx_backend.model import resolve_model
from mlx_backend.expression_fixed import FixedRandomProgram
from mlx_backend.models import Connectome,Stimulus

pytestmark = pytest.mark.metal


def capture_buffers(device):
    result={}
    for name,attribute in(('uniform','rand'),('normal','randn')):
        pointer=int(getattr(device,attribute+'_buffer')[0])
        result[name]=(np.ctypeslib.as_array((ctypes.c_double*20000).from_address(pointer)).copy()
                      if pointer else np.zeros(20000,np.float64))
        result[name+'_index']=int(getattr(device,attribute+'_buffer_index')[0])
    return result


def cpu_reference(model,directory,names,*,seed=81,neurons=3,steps=43,uniform_prefix=None,normal_prefix=None,initial_values=None,connectome=None,codegen_target='cython'):
    import brian2 as b
    preferences={name:b.prefs[name]for name in('codegen.target','codegen.runtime.cython.cache_dir','codegen.runtime.cython.delete_source_files')}
    ambient=np.random.get_state()
    try:
        b.prefs.codegen.target=codegen_target;b.prefs.codegen.runtime.cython.cache_dir=str(directory/'cython-default')
        b.prefs.codegen.runtime.cython.delete_source_files=False
        group=b.NeuronGroup(neurons,model.eqs,method='linear',threshold=model.eq_th,reset=model.eq_rst,
            refractory='rfc',namespace=model.brian_namespace(),clock=b.Clock(dt=model.dt_s*b.second),
            name='cython_rng_neurons*',dtype={eq.varname:np.dtype(model.integer_dtype if eq.var_type=='integer'else
                bool if eq.var_type=='boolean'else model.state_dtype)for eq in b.Equations(model.eqs).values()})
        group.v=model.v_0*b.volt;group.g=0;group.rfc=model.t_rfc*b.second
        for name,value in(initial_values or{}).items():setattr(group,name,value)
        monitor=b.SpikeMonitor(group);b.seed(seed)
        objects=[group,monitor]
        if connectome is not None:
            synapses=b.Synapses(group,group,'w:volt',on_pre='g_post+=w',clock=group.clock)
            synapses.connect(i=connectome.sources,j=connectome.destinations)
            synapses.w=connectome.counts*model.w_syn*b.volt
            synapses.delay=model.delay_steps*model.dt_s*b.second
            objects.append(synapses)
        if uniform_prefix is not None:
            # Force a real Brian allocation, then set an accepted observable
            # buffer prefix without replacing the compiled Poisson function.
            warm=b.NeuronGroup(1,'a:1',clock=b.Clock(dt=model.dt_s*b.second))
            runner=warm.run_regularly('a=rand()',when='start');b.Network(warm,runner).run(model.dt_s*b.second)
            device=b.get_device();pointer=int(device.rand_buffer[0])
            array=np.ctypeslib.as_array((ctypes.c_double*20000).from_address(pointer))
            array[:len(uniform_prefix)]=uniform_prefix;device.rand_buffer_index[0]=0 if len(uniform_prefix)==20000 else 1
            # Index1 bypasses a refill; values start there.
            array[1:1+len(uniform_prefix)]=uniform_prefix
        if normal_prefix is not None:
            warm=b.NeuronGroup(1,'a:1',clock=b.Clock(dt=model.dt_s*b.second))
            runner=warm.run_regularly('a=randn()',when='start');b.Network(warm,runner).run(model.dt_s*b.second)
            device=b.get_device();pointer=int(device.randn_buffer[0])
            array=np.ctypeslib.as_array((ctypes.c_double*20000).from_address(pointer))
            array[1:1+len(normal_prefix)]=normal_prefix;device.randn_buffer_index[0]=1
        initial=dict(capture_buffers(b.get_device()),numpy_state=np.random.get_state())
        error=None
        try:b.Network(*objects).run(steps*model.dt_s*b.second,namespace={})
        except Exception as exc:
            error=exc
            while error.__cause__ is not None:error=error.__cause__
        states={name:np.broadcast_to(np.asarray(getattr(group,name+'_')[:]),(1,neurons)).copy()for name in names}
        spikes=(np.asarray(monitor.i[:],np.int64),np.rint(np.asarray(monitor.t_)/model.dt_s).astype(np.int64))
        final=dict(capture_buffers(b.get_device()),numpy_state=np.random.get_state())
        return initial,spikes,states,final,error
    finally:
        np.random.set_state(ambient)
        for name,value in preferences.items():b.prefs[name]=value


def assert_snapshot(actual,expected):
    a=actual['numpy_state'];b=expected['numpy_state']
    np.testing.assert_array_equal(a[1],b[1]);assert a[2:]==b[2:]
    for name in('uniform','normal'):
        np.testing.assert_array_equal(actual[name],expected[name]);assert actual[name+'_index']==expected[name+'_index']


@pytest.mark.parametrize('case',['uniform','normal','scalar_vector','poisson','negative_poisson','scaled_reset','two_products','three_products','constant_sum','scaled_accumulate'])
def test_cython_model_phases_buffers_and_partial_continuation(case,tmp_path):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.configured import prepare,run_prepared,host_state,initial_state,advance,evaluate_state
    from mlx_backend.expression_runtime import snapshots,initial
    equations='v:volt\ng:volt\nrfc:second\na:1\nb:1\n'
    settings={
        'uniform':dict(eq_th='(i<2) and ((rand()<.5) or (rand()<.2))',eq_rst='a=rand();b=rand()-a;v=v_rst;g=0*mV'),
        'normal':dict(eq_th='randn()>.4',eq_rst='a=randn();b=rand()+a;v=v_rst;g=0*mV'),
        'poisson':dict(eq_th='poisson(1+i)>2',eq_rst='a=poisson(10+rand()*3);b=rand();v=v_rst;g=0*mV'),
        'negative_poisson':dict(eq_th='poisson(-1)>0',eq_rst='v=v_rst;g=0*mV'),
        'scaled_reset':dict(eq_th='True',eq_rst='a=randn()*0.1+0.2;b=rand()*0.3-0.4;v=v_rst;g=0*mV'),
        'two_products':dict(eq_th='True',eq_rst='a=.13*randn()+.21*rand();b=.37*randn()-.43*rand();v=v_rst;g=0*mV'),
        'three_products':dict(eq_th='True',eq_rst='a=.13*randn()+.21*rand()+.37*poisson(3);v=v_rst;g=0*mV'),
        'constant_sum':dict(eq_th='True',eq_rst='a=.13+randn()+rand()+poisson(3);v=v_rst;g=0*mV'),
        'scaled_accumulate':dict(eq_th='True',eq_rst='a+=.13*randn();b-=.37*rand();v=v_rst;g=0*mV'),
        'scalar_vector':dict(eq_th='x>.4',eq_rst='a=x;b=rand();v=v_rst;g=0*mV')}
    if case=='scalar_vector':equations+='s=rand():1 (shared,constant over dt)\nx=randn()+s:1 (constant over dt)\n'
    model=resolve_model(dict(eqs=equations,t_rfc=(2.2 if case=='scaled_reset'else 0)*b.ms,**settings[case]),dt_ms=.1,random_target='cython')
    n=257 if case in('two_products','three_products','constant_sum','scaled_accumulate')else 3
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((1,43,0),np.uint8),(),(),(0,),81,3,None,.1)
    p=prepare(graph,inputs,model);names=tuple(host_state(initial_state(p),p))
    comparison=CompiledRngComparison(
        cpu_reference(model,tmp_path,names,neurons=n),'cython',81)
    result,state=run_prepared(p,inputs,return_state=True,random_states=[comparison.start],chunk_steps=19)
    comparison.check(result,state,p,host_state,snapshots,assert_snapshot)

    partial=initial_state(p,random_states=[comparison.start]);events=mx.array(inputs.events.reshape(-1))
    for stop in(1,7,13,19,43):
        while partial.step<stop:partial,_=advance(p,partial,events,43,min(stop-partial.step,p.window-partial.step%p.window))
        evaluate_state(partial)
        if stop==13:
            restored=initial(p,1,snapshots(partial.random))
            partial=replace(partial,random=replace(restored,temporaries=partial.random.temporaries))
    for name,values in host_state(partial,p).items():np.testing.assert_array_equal(values,comparison.expected[name],err_msg=name)
    assert_snapshot(snapshots(partial.random)[0],comparison.final)

def test_cython_poisson_variable_rates_int32_edges_and_mixed_buffers(tmp_path):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.configured import prepare,initial_state,advance,evaluate_state,host_state
    from mlx_backend.expression_runtime import snapshots
    rates=np.tile(np.array([0.,-0.,1e-12,.3,1.,3.,np.nextafter(10.,0),10.,np.nextafter(10.,np.inf),
        17.,100.,1e8,1e16,-1.,-np.inf,np.nan,np.inf]),3)
    n=len(rates)
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\na:1\nb:1\nlam:1',eq_th='True',
        eq_rst='a=poisson(lam);b=randn();v=v_rst;g=0*mV',t_rfc=0*b.ms),dt_ms=.1,random_target='cython')
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((1,1,0),np.uint8),(),(),(0,),81,3,None,.1)
    p=prepare(graph,inputs,model);names=tuple(host_state(initial_state(p),p))
    start,_,expected,final,error=cpu_reference(model,tmp_path,names,neurons=n,steps=1,initial_values={'lam':rates})
    assert error is None
    state=initial_state(p,random_states=[start]);values=dict(state.values);values['lam']=mx.array(rates.view(np.uint64))
    state=replace(state,values=values);state,_=advance(p,state,mx.array(inputs.events.reshape(-1)),1,1)
    evaluate_state(state)
    for name,value in host_state(state,p).items():np.testing.assert_array_equal(value,expected[name],err_msg=name)
    assert_snapshot(snapshots(state.random)[0],final)



@pytest.mark.parametrize('function',['rand','randn','poisson(1)','poisson(3)','poisson(0)','poisson(-1)','poisson(17)'])
def test_cython_fixed_buffers_batched_recording_off_and_continuation(function,tmp_path):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.configured import prepare,initial_state,advance,evaluate_state,host_state,run_prepared
    from mlx_backend.expression_runtime import snapshots,initial
    n=1009;steps=43
    condition=({'rand':'rand()<.001','randn':'randn()>3.05'}.get(function)
        or function+'>5')
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second',eq_th=condition,
        eq_rst='v=v_rst;g=0*mV',t_rfc=0*b.ms),dt_ms=.1,random_target='cython')
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((2,steps,0),np.uint8),(),(),(7,2),81,3,None,.1)
    p=prepare(graph,inputs,model);names=tuple(host_state(initial_state(p,2),p))
    from mlx_backend.expression_runtime import _fixed_draw
    assert isinstance(p.program.random_program,FixedRandomProgram)or _fixed_draw(p.program)is not None
    references=[cpu_reference(model,tmp_path/f'trial{trial}',names,seed=seed,neurons=n,steps=steps)
                for trial,seed in enumerate((81,827))]
    assert all(row[-1]is None for row in references)
    starts=[row[0]for row in references]
    result,state=run_prepared(p,inputs,return_state=True,random_states=starts,chunk_steps=19)
    for trial,(_,spikes,expected,final,_)in enumerate(references):
        selected=result.spikes.trials==inputs.trial_indices[trial]
        np.testing.assert_array_equal(result.spikes.neurons[selected],spikes[0])
        np.testing.assert_array_equal(result.spikes.steps[selected],spikes[1])
        for name,value in host_state(state,p).items():np.testing.assert_array_equal(value[trial:trial+1],expected[name],err_msg=name)
        assert_snapshot(snapshots(state.random)[trial],final)
    _,quiet=run_prepared(p,inputs,return_state=True,random_states=starts,chunk_steps=7,record=False)
    for trial,final in enumerate(snapshots(quiet.random)):assert_snapshot(final,references[trial][3])
    partial=initial_state(p,2,random_states=starts);events=mx.array(inputs.events.reshape(-1))
    for stop in(1,7,13,19,43):
        while partial.step<stop:partial,_=advance(p,partial,events,steps,min(stop-partial.step,p.window-partial.step%p.window))
        evaluate_state(partial)
        if stop==13:
            restored=initial(p,2,snapshots(partial.random))
            partial=replace(partial,random=replace(restored,temporaries=partial.random.temporaries))
    for trial,final in enumerate(snapshots(partial.random)):assert_snapshot(final,references[trial][3])


def test_cython_poisson_zero_division_preserves_consumed_buffer_prefix(tmp_path):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.configured import prepare,initial_state,advance,evaluate_state
    from mlx_backend.expression_runtime import snapshots
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\na:1\nx=rand():1 (constant over dt)',eq_th='poisson(17)>0',
        eq_rst='a=rand();v=v_rst;g=0*mV',t_rfc=0*b.ms),dt_ms=.1,random_target='cython')
    graph=Connectome(np.arange(1),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((1,3,0),np.uint8),(),(),(0,),81,3,None,.1)
    p=prepare(graph,inputs,model)
    start,_,expected,final,error=cpu_reference(model,tmp_path,('a','x'),neurons=1,steps=1,uniform_prefix=[.25,0.,.5])
    assert isinstance(error,ZeroDivisionError)
    state=initial_state(p,random_states=[start]);state,_=advance(p,state,mx.array(inputs.events.reshape(-1)),3,3)
    with pytest.raises(ZeroDivisionError,match='float division'):evaluate_state(state)
    assert_snapshot(snapshots(state.random)[0],final)
    from mlx_backend.configured import host_state
    actual=host_state(state,p)
    for name in expected:np.testing.assert_array_equal(actual[name],expected[name])


def test_cython_pooled_poisson_error_replays_partial_model_writes_on_device(tmp_path):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.configured import prepare,initial_state,advance,evaluate_state,host_state
    from mlx_backend.expression_runtime import snapshots,_fixed_draw
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second',eq_th='poisson(17)>0',
        eq_rst='v=v_rst;g=0*mV',t_rfc=0*b.ms),dt_ms=.1,random_target='cython')
    graph=Connectome(np.arange(3),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((1,3,0),np.uint8),(),(),(0,),81,3,None,.1)
    p=prepare(graph,inputs,model);assert _fixed_draw(p.program)is not None
    names=tuple(host_state(initial_state(p),p))
    # Step0 succeeds for all cells; step1 succeeds for cell0 then raises in
    # cell1. CPU threshold bookkeeping is partial; no step1 reset/monitor runs.
    prefix=[.5,0.]*4+[0.,.5]
    start,_,expected,final,error=cpu_reference(model,tmp_path,names,neurons=3,steps=3,uniform_prefix=prefix)
    assert isinstance(error,ZeroDivisionError)
    state=initial_state(p,random_states=[start])
    state,bits=advance(p,state,mx.array(inputs.events.reshape(-1)),3,3)
    with pytest.raises(ZeroDivisionError):evaluate_state(state)
    for name,value in host_state(state,p).items():np.testing.assert_array_equal(value,expected[name],err_msg=name)
    np.testing.assert_array_equal(np.asarray(bits),[1,1,1])
    assert_snapshot(snapshots(state.random)[0],final)


def test_cython_pooled_error_replay_keeps_delayed_arrivals_and_other_trial(tmp_path):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.configured import prepare,initial_state,advance,evaluate_state,host_state
    from mlx_backend.expression_runtime import snapshots,_fixed_draw
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second',eq_th='poisson(17)>0',
        eq_rst='v=v_rst',t_rfc=0*b.ms,t_dly=.2*b.ms),dt_ms=.1,random_target='cython')
    graph=Connectome(np.arange(3),np.array([0],np.int32),np.array([1],np.int32),np.array([2.]),np.zeros(1))
    inputs=Stimulus(np.zeros((2,4,0),np.uint8),(),(),(7,2),81,3,None,.1)
    p=prepare(graph,inputs,model);assert p.window==2 and _fixed_draw(p.program)is not None
    names=tuple(host_state(initial_state(p,2),p))
    # The final window must replay step2, including the delayed step0 arrival,
    # before preserving the partial threshold writes from the step3 exception.
    prefix=[.5,0.]*10+[0.,.5]
    references=[cpu_reference(model,tmp_path/'failed',names,steps=4,uniform_prefix=prefix,connectome=graph),
        cpu_reference(model,tmp_path/'healthy',names,steps=4,seed=827,connectome=graph)]
    assert isinstance(references[0][-1],ZeroDivisionError) and references[1][-1]is None
    state=initial_state(p,2,random_states=[row[0]for row in references]);events=mx.array(inputs.events.reshape(-1))
    state,first=advance(p,state,events,4,2);evaluate_state(state)
    state,last=advance(p,state,events,4,2)
    with pytest.raises(ZeroDivisionError):evaluate_state(state)
    bits=(np.asarray(first)|(np.asarray(last)<<2)).reshape(2,3)
    for trial,(_,spikes,expected,final,_)in enumerate(references):
        for name,value in host_state(state,p).items():np.testing.assert_array_equal(value[trial:trial+1],expected[name],err_msg=name)
        expected_bits=np.zeros(3,np.uint32)
        for neuron,step in zip(*spikes):expected_bits[neuron]|=np.uint32(1)<<np.uint32(step)
        np.testing.assert_array_equal(bits[trial],expected_bits)
        assert_snapshot(snapshots(state.random)[trial],final)
    assert host_state(state,p)['g'][0,1]>0


@pytest.mark.parametrize('case',['sequential','numerator','early','argument','augmented'])
def test_cython_reset_division_error_preserves_draw_and_write_prefix(tmp_path,case):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.configured import prepare,initial_state,advance,evaluate_state,host_state
    from mlx_backend.expression_runtime import snapshots
    reset={'sequential':'a=rand();b=1/den','numerator':'a=rand()/den;b=randn()',
        'early':'a=1/den+rand();b=randn()','argument':'a=poisson(1/den);b=rand()',
        'augmented':'a=rand();a/=den;b=randn()'}[case]
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\na:1\nb:1\nden:1',
        eq_th='True',eq_rst=reset+';v=v_rst;g=0*mV',t_rfc=0*b.ms),
        dt_ms=.1,random_target='cython')
    n=3;steps=4
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((2,steps,0),np.uint8),(),(),(7,2),81,3,None,.1)
    p=prepare(graph,inputs,model);names=tuple(host_state(initial_state(p,2),p))
    den=np.array([[1.,0.,2.],[1.,2.,3.]])
    references=[cpu_reference(model,tmp_path/f'trial{trial}',names,steps=steps,seed=seed,
        initial_values={'den':den[trial]})for trial,seed in enumerate((81,827))]
    assert isinstance(references[0][-1],ZeroDivisionError)and references[1][-1]is None
    state=initial_state(p,2,random_states=[r[0]for r in references]);values=dict(state.values)
    values['den']=mx.array(den.reshape(-1).view(np.uint64));state=replace(state,values=values)
    state,bits=advance(p,state,mx.array(inputs.events.reshape(-1)),steps,steps)
    with pytest.raises(ZeroDivisionError):evaluate_state(state)
    for trial,(_,spikes,expected,final,_)in enumerate(references):
        for name,value in host_state(state,p).items():np.testing.assert_array_equal(value[trial:trial+1],expected[name],err_msg=name)
        expected_bits=np.zeros(n,np.uint32)
        for cell,step in zip(*spikes):expected_bits[cell]|=np.uint32(1)<<np.uint32(step)
        np.testing.assert_array_equal(np.asarray(bits).reshape(2,n)[trial],expected_bits)
        assert_snapshot(snapshots(state.random)[trial],final)


@pytest.mark.parametrize('phase',['threshold','subexpression','shared_error','shared_commit'])
def test_cython_checked_division_phase_and_scalar_commit_order(phase,tmp_path):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.configured import prepare,initial_state,advance,evaluate_state,host_state
    from mlx_backend.expression_runtime import snapshots
    eqs='v:volt\ng:volt\nrfc:second\na:1\n'
    shared=phase=='shared_error'
    eqs+='den:1'+(' (shared)'if shared else'')+'\n'
    threshold='True';reset='v=v_rst;g=0*mV'
    if phase=='threshold':threshold='rand()/den>.2';reset='a=randn();'+reset
    elif phase=='subexpression':
        eqs+='x=rand()/den:1 (constant over dt)\n';threshold='randn()>0';reset='a=x;'+reset
    else:
        eqs+=('s=rand()/den:1 (shared,constant over dt)\nx=randn()+s:1 (constant over dt)\n'if shared else
              's=rand():1 (shared,constant over dt)\nx=randn()/den+s:1 (constant over dt)\n')
        reset='a=x;'+reset
    model=resolve_model(dict(eqs=eqs,eq_th=threshold,eq_rst=reset,t_rfc=0*b.ms),
        dt_ms=.1,random_target='cython')
    n=3;steps=4
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((2,steps,0),np.uint8),(),(),(7,2),81,3,None,.1)
    p=prepare(graph,inputs,model);names=tuple(host_state(initial_state(p,2),p))
    den=np.array([[0.],[2.]])if shared else np.array([[1.,0.,2.],[1.,2.,3.]])
    references=[cpu_reference(model,tmp_path/f'trial{trial}',names,steps=steps,seed=seed,
        initial_values={'den':den[trial]})for trial,seed in enumerate((81,827))]
    assert isinstance(references[0][-1],ZeroDivisionError)and references[1][-1]is None
    state=initial_state(p,2,random_states=[r[0]for r in references]);values=dict(state.values)
    values['den']=mx.array(den.reshape(-1).view(np.uint64));state=replace(state,values=values)
    state,bits=advance(p,state,mx.array(inputs.events.reshape(-1)),steps,steps)
    with pytest.raises(ZeroDivisionError):evaluate_state(state)
    for trial,(_,spikes,expected,final,_)in enumerate(references):
        for name,value in host_state(state,p).items():np.testing.assert_array_equal(value[trial:trial+1],expected[name],err_msg=name)
        expected_bits=np.zeros(n,np.uint32)
        for cell,step in zip(*spikes):expected_bits[cell]|=np.uint32(1)<<np.uint32(step)
        np.testing.assert_array_equal(np.asarray(bits).reshape(2,n)[trial],expected_bits)
        assert_snapshot(snapshots(state.random)[trial],final)
