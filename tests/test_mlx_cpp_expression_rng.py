"""Actual Brian standalone model phases versus native compiled RNG ordering."""
from pathlib import Path
import numpy as np
import pytest
from state_assertions import CompiledRngComparison
from mlx_backend.model import resolve_model
from mlx_backend.expression_fixed import FixedRandomProgram
from mlx_backend.models import Connectome,Stimulus


def cpu_reference(model,directory,state_names,steps=43,normal_used=False,neurons=3,seed=81,connectome=None,stimulus=None,trial=0):
    import brian2 as b
    from brian2.devices.device import reset_device
    from brian2.codegen.cpp_prefs import get_compiler_and_args
    cache=b.prefs.codegen.runtime.cython.cache_dir
    make_args=b.prefs.devices.cpp_standalone.extra_make_args_unix
    b.prefs.codegen.runtime.cython.cache_dir=str(directory/'cython-default')
    b.prefs.devices.cpp_standalone.extra_make_args_unix=['-j2']
    b.set_device('cpp_standalone',build_on_run=False)
    try:
        device=b.get_device();device.reinit();device.activate(build_on_run=False)
        clock=b.Clock(dt=model.dt_s*b.second)
        group=b.NeuronGroup(neurons,model.eqs,method='linear',threshold=model.eq_th,reset=model.eq_rst,
            refractory='rfc',namespace=model.brian_namespace(),clock=clock,name='cpp_rng_neurons',
            dtype={eq.varname:np.dtype(model.integer_dtype if eq.var_type=='integer'else
                bool if eq.var_type=='boolean'else model.state_dtype)for eq in b.Equations(model.eqs).values()})
        group.v=model.v_0*b.volt;group.g=0;group.rfc=model.t_rfc*b.second
        monitor=b.SpikeMonitor(group);b.seed(seed)
        objects=[group,monitor]
        if connectome is not None:
            synapses=b.Synapses(group,group,'w:volt',on_pre='g_post+=w',clock=clock)
            synapses.connect(i=connectome.sources,j=connectome.destinations)
            synapses.w=connectome.counts*model.w_syn*b.volt
            synapses.delay=model.delay_steps*model.dt_s*b.second
            objects.append(synapses)
        if stimulus is not None and stimulus.targets:
            group.rfc[list(stimulus.targets)]=0*b.second
            times,channels=np.nonzero(stimulus.events[trial])
            inputs=b.SpikeGeneratorGroup(len(stimulus.targets),channels,times*model.dt_s*b.second,clock=clock)
            delivery=b.Synapses(inputs,group,'amplitude:volt',on_pre='v_post+=amplitude',clock=clock)
            delivery.connect(i=np.arange(len(stimulus.targets)),j=stimulus.targets)
            amplitudes=(np.full(len(stimulus.targets),model.poisson_weight_v)if stimulus.amplitudes_mv is None
                else np.asarray(stimulus.amplitudes_mv)*.001)
            delivery.amplitude=amplitudes*b.volt
            objects.extend((inputs,delivery))
        b.Network(*objects).run(steps*clock.dt,namespace={})
        device.insert_code('main','{std::ofstream out(brian::results_dir+"rng-state.txt");'
            f'brian::_random_generators[0].dump(out,{"true"if normal_used else"false"});'+'}')
        device.build(directory=str(directory),compile=False,run=False,with_output=False)
        header=directory/'objects.h';text=header.read_text()
        text='#include <fstream>\n#include <iomanip>\n'+text.replace('public:',r'''public:
        void dump(std::ostream& out,bool initialized) {
            auto future=gen;for(int j=0;j<624;++j)out<<future()<<' ';
            out<<has_stored_gauss<<' '<<std::hexfloat<<(initialized?stored_gauss:0.0)<<'\n';
        }
        ''',1)
        header.write_text(text)
        compiler,_=get_compiler_and_args();device.compile_source(str(directory),compiler,False,False)
        device.run(directory=str(directory),with_output=False)
        states={name:np.broadcast_to(np.asarray(getattr(group,name+'_')[:]),(1,neurons)).copy()for name in state_names}
        spikes=(np.asarray(monitor.i[:],np.int64),np.rint(np.asarray(monitor.t_)/model.dt_s).astype(np.int64))
        words=(directory/'results/rng-state.txt').read_text().split();assert len(words)==626
        return spikes,states,(np.asarray(words[:624],np.uint32),int(words[624]),float.fromhex(words[625]))
    finally:
        reset_device()
        b.prefs.codegen.runtime.cython.cache_dir=cache
        b.prefs.devices.cpp_standalone.extra_make_args_unix=make_args


@pytest.mark.metal
@pytest.mark.parametrize('case',['uniform','normal','scalar_vector','poisson','negative_poisson','scaled_reset','two_products','three_products','constant_sum','scaled_accumulate'])
def test_cpp_model_phases_masks_sequential_resets_and_continuation(case,tmp_path):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.configured import prepare,run_prepared,host_state,initial_state,advance,evaluate_state
    equations='v:volt\ng:volt\nrfc:second\na:1\nb:1\n'
    settings={
        'uniform':dict(eq_th='(i<2) and ((rand()<.5) or (rand()<.2))',
            eq_rst='a=rand();b=rand()-a;v=v_rst;g=0*mV'),
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
    params=dict(eqs=equations,t_rfc=(2.2 if case=='scaled_reset'else 0)*b.ms,**settings[case])
    model=resolve_model(params,dt_ms=.1,random_target='cpp')
    assert model.signature!=resolve_model(params,dt_ms=.1,random_target='numpy').signature
    n=257 if case in('two_products','three_products','constant_sum','scaled_accumulate')else 3
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((1,43,0),np.uint8),(),(),(0,),81,3,None,.1)
    p=prepare(graph,inputs,model);streams=[np.random.RandomState(81).get_state()]
    result,state=run_prepared(p,inputs,return_state=True,random_states=streams,chunk_steps=19)
    names=tuple(host_state(state,p))
    spikes,expected,rng=cpu_reference(model,tmp_path/'cpp-default',names,normal_used=case in('normal','scalar_vector','scaled_reset','two_products','three_products','constant_sum','scaled_accumulate'),neurons=n)
    np.testing.assert_array_equal(result.spikes.neurons,spikes[0]);np.testing.assert_array_equal(result.spikes.steps,spikes[1])
    for name,values in host_state(state,p).items():np.testing.assert_array_equal(values,expected[name],err_msg=name)
    copy=np.random.RandomState();snapshot=('MT19937',np.asarray(state.random.stream.words),
        int(np.asarray(state.random.stream.position)[0]),int(np.asarray(state.random.stream.has_gauss)[0]),
        float(np.asarray(state.random.stream.gauss).view(np.float64)[0]));copy.set_state(snapshot)
    np.testing.assert_array_equal(copy.randint(0,2**32,624,dtype=np.uint32),rng[0]);assert snapshot[3:]==rng[1:]
    partial=initial_state(p,1,random_states=streams);events=mx.array(inputs.events.reshape(-1))
    for stop in(1,7,13,19,43):
        while partial.step<stop:
            partial,_=advance(p,partial,events,43,min(stop-partial.step,p.window-partial.step%p.window))
        evaluate_state(partial)
    for name,values in host_state(partial,p).items():np.testing.assert_array_equal(values,expected[name],err_msg=name)
    for name in('words','position','has_gauss','gauss'):
        np.testing.assert_array_equal(np.asarray(getattr(partial.random.stream,name)),np.asarray(getattr(state.random.stream,name)))


@pytest.mark.metal
@pytest.mark.parametrize('function',['rand','randn','poisson(17)'])
def test_cpp_fused_random_threshold_preserves_refractory_draws(function,tmp_path):
    import brian2 as b
    from mlx_backend.configured import prepare,run_prepared,host_state,initial_state
    from mlx_backend.expression_runtime import _fixed_draw,snapshots
    n=1009;steps=43
    condition={'rand':'rand()<.01','randn':'randn()>2.3','poisson(17)':'poisson(17)>23'}[function]
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second',eq_th=condition,
        eq_rst='v=v_rst;g=0*mV',t_rfc=2.2*b.ms),dt_ms=.1,random_target='cpp')
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((1,steps,0),np.uint8),(),(),(0,),81,3,None,.1)
    p=prepare(graph,inputs,model);assert isinstance(p.program.random_program,FixedRandomProgram)or _fixed_draw(p.program)is not None
    names=tuple(host_state(initial_state(p),p))
    comparison=CompiledRngComparison(cpu_reference(model,tmp_path/'cpp-default',names,steps=steps,
        normal_used=function=='randn',neurons=n),'cpp',81)
    result,state=run_prepared(p,inputs,return_state=True,
        random_states=[comparison.start],chunk_steps=19)
    comparison.check(result,state,p,host_state,snapshots,None)
