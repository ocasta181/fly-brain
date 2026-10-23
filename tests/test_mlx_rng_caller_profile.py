"""Independent cancellation-sensitive random arithmetic in each model phase."""
import numpy as np
import pytest
from mlx_backend.model import resolve_model
from mlx_backend.models import Connectome,Stimulus
from state_assertions import CompiledRngComparison

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('target',['cpp','cython'])
@pytest.mark.parametrize('phase',['threshold','subexpression','shared','shared_vector'])
def test_random_caller_cancellation_in_model_phases(target,phase,tmp_path,neurons=3,expression='.13+randn()+rand()+poisson(3)',controlled=False):
    import brian2 as b
    from mlx_backend.configured import prepare,initial_state,host_state,run_prepared
    from mlx_backend.expression_runtime import snapshots
    from test_mlx_cpp_expression_rng import cpu_reference as cpp_reference
    from test_mlx_cython_expression_rng import cpu_reference as cython_reference,assert_snapshot
    equations='v:volt\ng:volt\nrfc:second\na:1'
    if phase=='shared_vector':equations+='\ns='+expression+':1 (shared,constant over dt)'
    if phase not in('threshold','conditional_threshold','conditional_arms','reset'):equations+='\nx='+expression+('+s'if phase=='shared_vector'else'')+':1 ('+('shared,'if phase=='shared'else'')+'constant over dt)'
    threshold=('((i<2) and (('+expression+')>-.5)) or ((i>=2) and (('+expression+')>.1))'if phase=='conditional_arms'else
        '(i<2) and (('+expression+')>-.5)'if phase=='conditional_threshold'else
        '('+expression+')>2'if phase=='threshold'else'True'if phase=='reset'else'x>2')
    model=resolve_model(dict(eqs=equations,eq_th=threshold,
        eq_rst=('a='+expression if phase=='reset'else'a+=1')+';v=v_rst;g=0*mV',t_rfc=0*b.ms),dt_ms=.1,random_target=target)
    n=neurons;steps=9
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((1,steps,0),np.uint8),(),(),(0,),81,3,None,.1)
    p=prepare(graph,inputs,model);names=tuple(host_state(initial_state(p),p))
    if target=='cpp':
        comparison = CompiledRngComparison(
            cpp_reference(model,tmp_path/'cpp-default',names,steps=steps,neurons=n,normal_used='randn'in expression), target, 81)
    else:
        prefix=dict(normal_prefix=np.full(19999,-1.),uniform_prefix=np.full(19999,.125))if controlled else{}
        comparison = CompiledRngComparison(
            cython_reference(model,tmp_path,names,steps=steps,neurons=n,**prefix), target, 81)
    result,state=run_prepared(p,inputs,return_state=True,random_states=[comparison.start],chunk_steps=4)
    comparison.check(result, state, p, host_state, snapshots, assert_snapshot)


@pytest.mark.parametrize('target',['cpp','cython'])
def test_shared_and_vector_caller_arithmetic_in_actual_neuron_loop(target,tmp_path):
    test_random_caller_cancellation_in_model_phases(target,'shared_vector',tmp_path,neurons=257)


@pytest.mark.parametrize('target',['cpp','cython'])
@pytest.mark.parametrize('phase',['reset','subexpression'])
@pytest.mark.parametrize('expression',['.13+rand()','.13+rand()+.37'],ids=['bias','regrouped'])
def test_single_uniform_bias_preserves_actual_caller_arithmetic(target,phase,expression,tmp_path):
    test_random_caller_cancellation_in_model_phases(target,phase,tmp_path,
        neurons=257,expression=expression)


@pytest.mark.parametrize('target',['cpp','cython'])
def test_conditional_compound_threshold_keeps_scope_and_undrawn_rows_false(target,tmp_path):
    test_random_caller_cancellation_in_model_phases(target,'conditional_threshold',tmp_path,neurons=257)


@pytest.mark.parametrize('target',['cpp','cython'])
def test_mutually_exclusive_compound_threshold_arms_keep_draw_identity(target,tmp_path):
    test_random_caller_cancellation_in_model_phases(target,'conditional_arms',tmp_path,neurons=257)


def test_cython_mutual_arms_accept_valid_minus_one_normals(tmp_path):
    test_random_caller_cancellation_in_model_phases('cython','conditional_arms',tmp_path,neurons=257,controlled=True)


def test_cython_valid_minus_one_normal_is_not_a_call_error(tmp_path):
    import brian2 as b
    from mlx_backend.configured import prepare,initial_state,host_state,run_prepared
    from mlx_backend.expression_runtime import snapshots
    from test_mlx_cython_expression_rng import cpu_reference,assert_snapshot
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\na:1',eq_th='True',t_rfc=0*b.ms,
        eq_rst='a=.13+randn()+rand()+poisson(3);v=v_rst;g=0*mV'),dt_ms=.1,random_target='cython')
    n=3;steps=4
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((1,steps,0),np.uint8),(),(),(0,),81,3,None,.1)
    p=prepare(graph,inputs,model);names=tuple(host_state(initial_state(p),p))
    comparison = CompiledRngComparison(
        cpu_reference(model,tmp_path,names,steps=steps,neurons=n,
            normal_prefix=np.full(n*steps,-1.),uniform_prefix=np.full(3*n*steps,.125)), 'cython', 81)
    result,state=run_prepared(p,inputs,return_state=True,random_states=[comparison.start],chunk_steps=3)
    comparison.check(result, state, p, host_state, snapshots, assert_snapshot)


@pytest.mark.parametrize('expression,limit',[
    ('.13+randn()+rand()+poisson(3)','1e16'),
    ('.13*randn()+.21*rand()+.37*poisson(3)','1300000000000000.25'),
    ('.13+randn()+rand()+poisson(3)','9999999999999998.0')],ids=['sum-boundary','products-boundary','sum-taken'])
def test_cython_cancellation_sensitive_predicate_controls_later_uniform_draw(tmp_path,expression,limit):
    import brian2 as b
    from mlx_backend.configured import prepare,initial_state,host_state,run_prepared
    from mlx_backend.expression_runtime import snapshots
    from test_mlx_cython_expression_rng import cpu_reference,assert_snapshot
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\na:1',
        eq_th=f'({expression}>{limit}) and (rand()<.5)',
        eq_rst='a+=1;v=v_rst;g=0*mV',t_rfc=0*b.ms),dt_ms=.1,random_target='cython')
    graph=Connectome(np.arange(1),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    stimulus=Stimulus(np.zeros((1,1,0),np.uint8),(),(),(0,),81,0,None,.1)
    p=prepare(graph,stimulus,model);names=tuple(host_state(initial_state(p),p))
    comparison = CompiledRngComparison(
        cpu_reference(model,tmp_path,names,steps=1,neurons=1,
            normal_prefix=np.full(8,1e16),uniform_prefix=np.full(16,.125)), 'cython', 81)
    assert comparison.final['uniform_index']-comparison.start['uniform_index']==(4 if limit.endswith('998.0')else 3)
    # Keep the actual native predicate for diagnosis of a first differing
    # guard; the independent CPU observation remains the assertion oracle.
    (tmp_path/'native-threshold.metal').write_text(p.program.random_program.compiled_stages['threshold'].vector)
    (tmp_path/'reference-consumption.txt').write_text(str(comparison.final['uniform_index']-comparison.start['uniform_index']))
    result,state=run_prepared(p,stimulus,return_state=True,random_states=[comparison.start])
    comparison.check(result, state, p, host_state, snapshots, assert_snapshot)


@pytest.mark.parametrize('target',['cpp','cython'])
def test_repeated_prepared_random_trials_use_their_own_seed(target,tmp_path):
    import brian2 as b
    from mlx_backend.configured import prepare,initial_state,host_state,run_prepared
    from mlx_backend.expression_runtime import snapshots
    from test_mlx_cpp_expression_rng import cpu_reference as cpp_reference
    from test_mlx_cython_expression_rng import cpu_reference as cython_reference,assert_snapshot
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\na:1',eq_th='rand()<.5',t_rfc=0*b.ms,
        eq_rst='a=randn();v=v_rst;g=0*mV'),dt_ms=.1,random_target=target)
    n=3;steps=9
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((1,steps,0),np.uint8),(),(),(0,),81,3,None,.1)
    p=prepare(graph,inputs,model);names=tuple(host_state(initial_state(p),p));previous=None
    for seed in(81,827):
        directory=tmp_path/str(seed);directory.mkdir()
        if target=='cpp':
            comparison = CompiledRngComparison(
                cpp_reference(model,directory/'cpp-default',names,steps=steps,neurons=n,normal_used=True,seed=seed), target, seed)
        else:
            comparison = CompiledRngComparison(
                cython_reference(model,directory,names,steps=steps,neurons=n,seed=seed), target, seed)
        result,state=run_prepared(p,inputs,return_state=True,random_states=[comparison.start])
        comparison.check(result, state, p, host_state, snapshots, assert_snapshot)
        identity=np.column_stack((comparison.spikes[0],comparison.spikes[1]))
        if previous is not None:assert not np.array_equal(previous,identity)
        previous=identity
