"""New packed-RNG ABI names must not collide with Brian model variables."""
import numpy as np
import pytest
from mlx_backend.model import resolve_model
from mlx_backend.expression_fixed import FixedRandomProgram
from mlx_backend.models import Connectome,Stimulus
from state_assertions import CompiledRngComparison

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('target',['cpp','cython'])
@pytest.mark.parametrize('mode',['fused','ordered'])
def test_packed_rng_and_prefetch_names_are_private(target,mode,tmp_path):
    import brian2 as b
    from mlx_backend.configured import prepare,initial_state,host_state,run_prepared
    from mlx_backend.expression_runtime import snapshots,_fixed_draw
    from test_mlx_cpp_expression_rng import cpu_reference as cpp
    from test_mlx_cython_expression_rng import cpu_reference as cython,assert_snapshot
    names=tuple('extra'+str(i)for i in range(30))+('p_values','q_values','packet','buffers','selected','random_draws')
    reset='p_values+=1;q_values=p_values+2;packet=3;buffers=4;selected=5;random_draws=6;v=v_rst;g=0*mV'
    if mode=='ordered':reset+=';packet=randn()'
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\n'+'\n'.join(n+':1'for n in names),
        eq_th='rand()<.5',eq_rst=reset,t_rfc=0*b.ms),dt_ms=.1,random_target=target)
    graph=Connectome(np.arange(3),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((1,7,0),np.uint8),(),(),(0,),81,3,None,.1)
    p=prepare(graph,inputs,model);assert (isinstance(p.program.random_program,FixedRandomProgram)or _fixed_draw(p.program)is not None)==(mode=='fused')
    state_names=tuple(host_state(initial_state(p),p))
    if target=='cpp':
        comparison = CompiledRngComparison(
            cpp(model,tmp_path/'cpp-default',state_names,steps=7,normal_used=mode=='ordered'), target, 81)
    else:
        comparison = CompiledRngComparison(cython(model,tmp_path,state_names,steps=7), target, 81)
    result,state=run_prepared(p,inputs,return_state=True,random_states=[comparison.start])
    comparison.check(result, state, p, host_state, snapshots, assert_snapshot)


@pytest.mark.parametrize('target',['cpp','cython'])
@pytest.mark.parametrize('neurons',[1,3,17])
def test_compiled_random_forcing_in_linear_equation_matches_brian(target,neurons,tmp_path):
    import brian2 as b
    from mlx_backend.configured import prepare,initial_state,host_state,run_prepared
    from mlx_backend.expression_runtime import snapshots
    from test_mlx_cpp_expression_rng import cpu_reference as cpp
    from test_mlx_cython_expression_rng import cpu_reference as cython,assert_snapshot
    eqs=resolve_model().eqs.replace('(v_0 - v + g)','(v_0 - v + g + x*mV)')+'x=randn():1 (constant over dt)\n'
    model=resolve_model(dict(eqs=eqs,eq_th='(v>v_th) or (rand()<.02)',
        eq_rst='v=v_rst+randn()*mV;g=0*mV',t_rfc=1*b.ms),dt_ms=.1,random_target=target)
    n=neurons;steps=43
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((1,steps,0),np.uint8),(),(),(0,),81,3,None,.1)
    p=prepare(graph,inputs,model);names=tuple(host_state(initial_state(p),p))
    if target=='cpp':
        comparison = CompiledRngComparison(
            cpp(model,tmp_path/'cpp-default',names,steps=steps,normal_used=True,neurons=n), target, 81)
    else:
        comparison = CompiledRngComparison(cython(model,tmp_path,names,steps=steps,neurons=n), target, 81)
    result,state=run_prepared(p,inputs,return_state=True,random_states=[comparison.start],chunk_steps=19)
    comparison.check(result, state, p, host_state, snapshots, assert_snapshot)


@pytest.mark.parametrize('target',['cpp','cython'])
@pytest.mark.parametrize('neurons',[3,256,257])
def test_random_forcing_and_index_in_unguarded_linear_equation(target,neurons,tmp_path):
    from mlx_backend.configured import prepare,initial_state,host_state,run_prepared
    from mlx_backend.expression_runtime import snapshots
    from test_mlx_cpp_expression_rng import cpu_reference as cpp
    from test_mlx_cython_expression_rng import cpu_reference as cython,assert_snapshot
    eqs=('dv/dt=(-v+x*mV+i*mV)/(10*ms):volt\n'
         'dg/dt=-g/(5*ms):volt\nrfc:second\nx=rand():1 (constant over dt)')
    model=resolve_model(dict(eqs=eqs,eq_th='rand()<.02',eq_rst='v=v_rst;g=0*mV'),
        dt_ms=.1,random_target=target)
    n=neurons;steps=13
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((1,steps,0),np.uint8),(),(),(0,),81,3,None,.1)
    p=prepare(graph,inputs,model);names=tuple(host_state(initial_state(p),p))
    if target=='cpp':
        comparison = CompiledRngComparison(
            cpp(model,tmp_path/'cpp-default',names,steps=steps,neurons=n), target, 81)
    else:
        comparison = CompiledRngComparison(cython(model,tmp_path,names,steps=steps,neurons=n), target, 81)
    result,state=run_prepared(p,inputs,return_state=True,random_states=[comparison.start],chunk_steps=7)
    comparison.check(result, state, p, host_state, snapshots, assert_snapshot)


def test_cython_equation_error_keeps_model_and_rng_prefix_on_device(tmp_path):
    from dataclasses import replace
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.configured import prepare,initial_state,host_state,advance,evaluate_state
    from mlx_backend.expression_runtime import snapshots,_fixed_draw
    from test_mlx_cython_expression_rng import cpu_reference,assert_snapshot
    eqs=('dv/dt=(-v+x*mV*ms/tau_i)/(10*ms):volt\ndg/dt=-g/(5*ms):volt\n'
         'rfc:second\ntau_i:second\nx=randn():1 (constant over dt)')
    model=resolve_model(dict(eqs=eqs,eq_th='rand()<.8',eq_rst='v=v_rst;g=0*mV'),
        dt_ms=.1,random_target='cython')
    n=3;steps=4
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((2,steps,0),np.uint8),(),(),(7,2),81,3,None,.1)
    p=prepare(graph,inputs,model);assert _fixed_draw(p.program)is None
    rates=np.array([[.01,0.,.02],[.01,.02,.03]])
    names=tuple(host_state(initial_state(p,2),p));references=[]
    for trial,seed in enumerate((81,827)):
        references.append(cpu_reference(model,tmp_path/f'trial{trial}',names,neurons=n,steps=steps,seed=seed,
            initial_values={'tau_i':rates[trial]*b.second}))
    assert isinstance(references[0][-1],ZeroDivisionError)and references[1][-1]is None
    state=initial_state(p,2,random_states=[r[0]for r in references]);values=dict(state.values)
    values['tau_i']=mx.array(rates.reshape(-1).view(np.uint64));state=replace(state,values=values)
    state,bits=advance(p,state,mx.array(inputs.events.reshape(-1)),steps,steps)
    with pytest.raises(ZeroDivisionError):evaluate_state(state)
    for trial,(_,spikes,expected,final,_)in enumerate(references):
        for name,value in host_state(state,p).items():np.testing.assert_array_equal(value[trial:trial+1],expected[name],err_msg=name)
        expected_bits=np.zeros(n,np.uint32)
        for cell,step in zip(*spikes):expected_bits[cell]|=np.uint32(1)<<np.uint32(step)
        np.testing.assert_array_equal(np.asarray(bits).reshape(2,n)[trial],expected_bits)
        assert_snapshot(snapshots(state.random)[trial],final)


@pytest.mark.parametrize('target',['cpp','cython'])
@pytest.mark.parametrize('neurons,variable_rfc',[(17,False),(256,False),(256,True)])
def test_profiled_full_lif_keeps_fused_rng_windows_exact(target,neurons,variable_rfc,tmp_path):
    import brian2 as b
    from mlx_backend.configured import prepare,initial_state,host_state,run_prepared
    from mlx_backend.expression_runtime import snapshots,_fixed_draw
    from test_mlx_cpp_expression_rng import cpu_reference as cpp
    from test_mlx_cython_expression_rng import cpu_reference as cython,assert_snapshot
    reset='v=v_rst+.3*mV;g=0*mV'+(';rfc=(i+1)*.1*ms'if variable_rfc else'')
    model=resolve_model(dict(eq_th='rand()<.3',eq_rst=reset,t_rfc=2.2*b.ms),
        dt_ms=.1,random_target=target)
    n=neurons;steps=43
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((1,steps,0),np.uint8),(),(),(0,),81,3,None,.1)
    p=prepare(graph,inputs,model);assert isinstance(p.program.random_program,FixedRandomProgram)or _fixed_draw(p.program)is not None
    names=tuple(host_state(initial_state(p),p))
    if target=='cpp':
        comparison = CompiledRngComparison(
            cpp(model,tmp_path/'cpp-default',names,steps=steps,neurons=n), target, 81)
    else:
        comparison = CompiledRngComparison(cython(model,tmp_path,names,steps=steps,neurons=n), target, 81)
    for chunk in(7,19):
        result,state=run_prepared(p,inputs,return_state=True,random_states=[comparison.start],chunk_steps=chunk)
        comparison.check(result, state, p, host_state, snapshots, assert_snapshot)
