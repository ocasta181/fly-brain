"""Independent CPU checks of shared raw-prefix consumption and exhaustion."""
import numpy as np
import pytest
from mlx_backend.expression_rng import from_numpy_states,to_numpy_states,seeded
from test_mlx_cpp_rng import cpp_normal_reference,assert_state

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('position',[0,1,623,624])
@pytest.mark.parametrize('candidate_count',[None,1])
def test_mixed_prefix_masks_cached_pairs_exhaustion_and_later_draws(position,candidate_count):
    import mlx.core as mx
    from mlx_backend.expression_rng_prefix import prefix_pool,prefix_uniforms,prefix_normals
    cpus=[np.random.RandomState(81+row)for row in range(4)]
    for row,cpu in enumerate(cpus):
        if row%2:cpu.randn(1)
        snapshot=list(cpu.get_state());snapshot[2]=position;cpu.set_state(tuple(snapshot))
    state=from_numpy_states([cpu.get_state()for cpu in cpus])
    cache=prefix_pool(state,19,checkpoint=7,parts=3)
    phases=[('randn',[0,1,2,7]),('rand',[0,2,5,31]),('randn',[1,0,17,3]),
            ('rand',[7,1,0,5]),('randn',[0,7,1,0]),('rand',[0,0,0,0])]
    for function,counts in phases:
        counts=np.asarray(counts,np.uint32);maximum=max(1,int(counts.max()))
        if function=='rand':values,state,cache=prefix_uniforms(state,cache,maximum,mx.array(counts))
        else:values,state,cache=prefix_normals(state,cache,maximum,mx.array(counts),candidate_count=candidate_count)
        actual=np.asarray(values).view(np.float64)
        for row,(cpu,count,got)in enumerate(zip(cpus,counts,to_numpy_states(state))):
            expected=getattr(cpu,function)(count)
            np.testing.assert_array_equal(actual[row,:count],expected)
            np.testing.assert_array_equal(actual[row,count:],0)
            snapshot=cpu.get_state()
            np.testing.assert_array_equal(got[1],snapshot[1]);assert got[2:]==snapshot[2:]
    assert np.all(np.asarray(cache.cursor)<=cache.capacity)


@pytest.mark.parametrize('candidate_count',[None,1])
def test_cpp_prefix_values_full_future_stream_cache_and_fallback(cpp_normal_reference,candidate_count):
    from mlx_backend.expression_rng_prefix import prefix_pool,prefix_normals
    values,states,_=cpp_normal_reference;state=seeded([81])
    cache=prefix_pool(state,257,checkpoint=17,parts=3);begin=0
    for count,expected in zip((0,1,2,7,512,3575),states):
        actual,state,cache=prefix_normals(state,cache,count,target='cpp',candidate_count=candidate_count)
        np.testing.assert_array_equal(np.asarray(actual).reshape(-1).view(np.float64),values[begin:begin+count])
        assert_state(state,expected);begin+=count


def test_compiled_mixed_prefix_graph_accepts_fresh_states_without_capturing_seed():
    import mlx.core as mx
    from mlx_backend.expression_rng import MT19937State
    from mlx_backend.expression_rng_prefix import prefix_pool,prefix_uniforms,prefix_normals
    def operation(words,position,cached,saved):
        state=MT19937State(words,position,2,cached,saved);cache=prefix_pool(state,7,checkpoint=7)
        x,state,cache=prefix_uniforms(state,cache,17,mx.array([9,3],mx.uint32))
        y,state,cache=prefix_normals(state,cache,17,mx.array([7,5],mx.uint32))
        return x,y,state.words,state.position,state.has_gauss,state.gauss
    compiled=mx.compile(operation)
    for seeds in((81,827),(89,171)):
        cpus=[np.random.RandomState(seed)for seed in seeds];cpus[1].randn(1)
        initial=from_numpy_states([cpu.get_state()for cpu in cpus])
        x,y,words,position,cached,saved=compiled(initial.words,initial.position,initial.has_gauss,initial.gauss)
        state=MT19937State(words,position,2,cached,saved)
        x=np.asarray(x).view(np.float64);y=np.asarray(y).view(np.float64)
        for row,(cpu,nx,ny,got)in enumerate(zip(cpus,(9,3),(7,5),to_numpy_states(state))):
            np.testing.assert_array_equal(x[row,:nx],cpu.rand(nx));np.testing.assert_array_equal(x[row,nx:],0)
            np.testing.assert_array_equal(y[row,:ny],cpu.randn(ny));np.testing.assert_array_equal(y[row,ny:],0)
            expected=cpu.get_state();np.testing.assert_array_equal(got[1],expected[1]);assert got[2:]==expected[2:]


@pytest.mark.parametrize('conditional',[False,True])
def test_mixed_prefix_runtime_independent_trials_recordoff_and_partial_continuation(conditional,tmp_path):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome,Stimulus
    from mlx_backend.configured import prepare,initial_state,run_prepared,host_state,advance,evaluate_state
    from mlx_backend.expression_runtime import MixedLookahead,snapshots
    from test_mlx_cpp_expression_rng import cpu_reference
    n=65536;steps=5;seeds=(81,827);ids=(2,7)
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\na:1',t_rfc=0*b.ms,
        eq_th='(i<32768) and (rand()<.002)'if conditional else'rand()<.0001',
        eq_rst='a+=randn();v=v_rst;g=0*mV'),dt_ms=.1,random_target='cpp')
    graph=Connectome(np.arange(n),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((2,steps,0),np.uint8),(),(),ids,81,3,None,.1)
    p=prepare(graph,inputs,model);names=tuple(host_state(initial_state(p,2),p))
    expected=[cpu_reference(model,tmp_path/f'cpp-default-{seed}',names,steps=steps,
        normal_used=True,neurons=n,seed=seed)for seed in seeds]
    starts=[np.random.RandomState(seed).get_state()for seed in seeds]
    result,state=run_prepared(p,inputs,return_state=True,random_states=starts,chunk_steps=3)
    assert isinstance(state.random.pool,MixedLookahead)
    assert state.random.pool.next_step==steps
    for row,(events,values,final)in enumerate(expected):
        selected=result.spikes.trials==ids[row]
        np.testing.assert_array_equal(result.spikes.neurons[selected],events[0])
        np.testing.assert_array_equal(result.spikes.steps[selected],events[1])
        for name,value in host_state(state,p).items():np.testing.assert_array_equal(value[row:row+1],values[name],err_msg=name)
        future=np.random.RandomState();snapshot=snapshots(state.random)[row];future.set_state(snapshot)
        np.testing.assert_array_equal(future.randint(0,2**32,624,dtype=np.uint32),final[0])
        assert snapshot[3:]==final[1:]
    _,off=run_prepared(p,inputs,return_state=True,random_states=starts,record=False,chunk_steps=3)
    partial=initial_state(p,2,random_states=starts);events=mx.zeros((1,),mx.uint8)
    for stop in(1,3,5):
        partial,_=advance(p,partial,events,steps,stop-partial.step)
        evaluate_state(partial)
    for other in(off,partial):
        for name,value in host_state(state,p).items():np.testing.assert_array_equal(host_state(other,p)[name],value,err_msg=name)
        for name in('words','position','has_gauss','gauss'):
            np.testing.assert_array_equal(np.asarray(getattr(other.random.stream,name)),np.asarray(getattr(state.random.stream,name)))
