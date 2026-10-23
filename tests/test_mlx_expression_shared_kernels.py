"""Independent NumPy checks of shared-stream fast-path boundaries."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

pytestmark = pytest.mark.metal


def assert_stream(stream,cpus):
    from mlx_backend.expression_rng import to_numpy_states
    for actual,cpu in zip(to_numpy_states(stream),cpus):
        expected=cpu.get_state()
        assert actual[1].tobytes()==expected[1].tobytes()
        assert actual[2:4]==expected[2:4]
        assert np.float64(actual[4]).tobytes()==np.float64(expected[4]).tobytes()


@pytest.mark.parametrize('position',[0,1,623,624])
def test_sparse_normal_consumption_preserves_cached_pairs_twists_and_prefix(position):
    import mlx.core as mx
    from mlx_backend.expression_rng import from_numpy_states
    from mlx_backend.expression_rng_prefix import prefix_pool,prefix_uniforms
    from mlx_backend.expression_runtime import RandomState,MixedLookahead
    from mlx_backend.expression_sparse_normal import draw
    cpus=[np.random.RandomState(81+i)for i in range(4)]
    for i,cpu in enumerate(cpus):
        if i%2:cpu.randn(1)
        state=list(cpu.get_state());state[2]=position;cpu.set_state(tuple(state))
    stream=from_numpy_states([cpu.get_state()for cpu in cpus])
    prefix=prefix_pool(stream,19,checkpoint=7)
    random=RandomState(stream,{},mx.zeros(4,mx.uint32),MixedLookahead(prefix,None,0,1))
    for counts in([0,1,2,7],[11,0,65,4],[1,3,0,1]):
        counts=np.array(counts,np.uint32);maximum=int(counts.max())
        values,stream,pool=draw(random,maximum,mx.array(counts),target='numpy')
        actual=np.asarray(values).view(np.float64)
        for row,(cpu,count)in enumerate(zip(cpus,counts)):
            assert actual[row,:count].tobytes()==cpu.randn(count).tobytes()
        assert_stream(stream,cpus)
        values,stream,prefix=prefix_uniforms(stream,pool.data,3)
        actual=np.asarray(values).view(np.float64)
        for row,cpu in enumerate(cpus):assert actual[row].tobytes()==cpu.rand(3).tobytes()
        assert_stream(stream,cpus)
        random=replace(random,stream=stream,pool=replace(pool,data=prefix))


@pytest.mark.parametrize('position',[1,623,624])
def test_combined_uniform_original_input_prefix_exhaustion_and_future_stream(position):
    import mlx.core as mx
    from mlx_backend.expression_rng import from_numpy_states
    from mlx_backend.expression_rng_prefix import prefix_pool,_merge,_raw,prefix_uniforms
    from mlx_backend.expression_runtime import RandomState,MixedLookahead
    from mlx_backend.expression_poisson_input import ExpressionPoisson
    cpus=[np.random.RandomState(87+i)for i in range(3)]
    for i,cpu in enumerate(cpus):
        if i%2:cpu.randn(1)
        state=list(cpu.get_state());state[2]=position;cpu.set_state(tuple(state))
    stream=from_numpy_states([cpu.get_state()for cpu in cpus])
    prefix=prefix_pool(stream,7,checkpoint=7)
    random=RandomState(stream,{},mx.zeros(3,mx.uint32),MixedLookahead(prefix,None,0,1))
    inputs=[SimpleNamespace(_group=SimpleNamespace(start=t),rate=p,active=True)for t,p in((0,.37),(0,.72),(3,0.))]
    adapter=ExpressionPoisson(inputs,SimpleNamespace(dt=1.),'numpy')
    prepared=SimpleNamespace(neurons=19,program=SimpleNamespace(variables={'v':SimpleNamespace(conditional_write=None)}))
    for counts in([0,4,19],[19,1,0]):
        original=random.pool.data
        events,random,view=adapter.draw(prepared,SimpleNamespace(values={}),random,mx.array([True,False,True]),uniform_counts=mx.array(counts,mx.uint32))
        values=_merge(_raw(original,19),view.tail,view.done,view.counts,19)
        actual=np.asarray(values).view(np.float64);events=np.asarray(events).reshape(3,3)
        for row,(cpu,count)in enumerate(zip(cpus,counts)):
            assert actual[row,:count].tobytes()==cpu.rand(count).tobytes()
            assert not actual[row,count:].any()
            expected=[cpu.binomial(1,p)for p in(.37,.72,0.)]if row!=1 else[2,2,2]
            np.testing.assert_array_equal(events[row],expected)
        assert_stream(random.stream,cpus)
    values,stream,prefix=prefix_uniforms(random.stream,random.pool.data,624)
    actual=np.asarray(values).view(np.float64)
    for row,cpu in enumerate(cpus):assert actual[row].tobytes()==cpu.rand(624).tobytes()
    assert_stream(stream,cpus)


@pytest.mark.parametrize('function',['rand','randn'])
def test_packed_numpy_reset_matches_vector_raw_temporaries_and_sparse_markers(function):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome,Stimulus
    from mlx_backend.configured import prepare,initial_state,_pack_values
    from mlx_backend.expression_runtime import _stage
    from mlx_backend.expression_numpy_packed import selected,advance
    n=256;trials=2
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\na:integer',eq_th='rand()<.4',
        eq_rst=f'a+=1;v=v_rst+{function}()*mV;g=0*mV',t_rfc=0*b.ms),dt_ms=.1,random_target='numpy')
    graph=Connectome(np.arange(n),np.array([0,3,3],np.int32),np.array([1,1,255],np.int32),np.array([1.,-2.,3.]),np.zeros(3))
    inputs=Stimulus(np.zeros((trials,1,0),np.uint8),(),(),(0,1),81,3,None,.1)
    p=prepare(graph,inputs,model);state=initial_state(p,trials)
    assert selected(p.program,'reset') is not None
    chosen=np.zeros((trials,n),bool);chosen[0,[0,3,17]]=True
    values=dict(state.values)
    raw=np.asarray(values['v']).copy();raw[8]=0x8000000000000000;raw[n+9]=0x7ff8000000001234
    values['v']=mx.array(raw);packed=_pack_values(p.program,values)
    random=replace(state.random,temporaries={**state.random.temporaries,'_cond':mx.array(chosen.reshape(-1))})
    vector_values={**packed,**random.temporaries}
    times=mx.array([0],mx.uint64);params=mx.array([0,0,1],mx.int64)
    reference=_stage(p,'reset',vector_values,random,trials,times,params)
    actual,stored,markers=advance(p,'reset',packed,mx.array(chosen.reshape(-1)),random,times,params,None,7)
    for name,value in stored.items():assert np.asarray(value).tobytes()==np.asarray(vector_values[name]).tobytes(),name
    for name,value in actual.temporaries.items():assert np.asarray(value).tobytes()==np.asarray(vector_values[name]).tobytes(),name
    for name in('words','position','has_gauss','gauss'):assert np.asarray(getattr(actual.stream,name)).tobytes()==np.asarray(getattr(reference.stream,name)).tobytes(),name
    expected=np.zeros((trials,n),np.uint32);expected[0,[1,255]]=1<<7
    assert np.asarray(markers).tobytes()==expected.tobytes()
