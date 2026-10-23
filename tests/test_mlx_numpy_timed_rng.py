"""NumPy eager checked calls retain original Brian state and random prefix."""
import numpy as np
import pytest


@pytest.mark.metal
@pytest.mark.parametrize('phase,first,shared',[
    ('threshold_single',True,False),('threshold_single',False,False),
    ('threshold',True,False),('threshold',False,False),
    ('reset',True,False),('reset',False,False),
    ('reset_empty_shared',True,True),('subexpression',True,False),
    ('reset_empty',True,False),('reset',True,True),
    ('subexpression',True,True),('subexpression',False,True),
])
def test_checked_call_order_and_repair(tmp_path,monkeypatch,phase,first,shared,size=4,packed_extra=None,invalid_column=8):
    import brian2 as b
    from original_api_reference import build_original_api_fixture
    from original_api_protocol import check_original_api_runs, observe_run
    if size>4:
        import original_api_reference as reference
        monkeypatch.setattr(reference, "write_tiny_dataset", reference.synthetic_dataset(size))

    def build(mode):
        expression='offset(t,a)+rand()+offset(t,b)+randn()'
        if phase in ('threshold_single','threshold_sparse'):expression='offset(t,a)+rand()+offset(t,b)'
        equations='v:volt\ng:volt\nrfc:second\na:integer'+(' (shared)'if shared else'')+'\nb:integer'+(' (shared)'if shared else'')
        if phase=='subexpression':equations+='\ns='+expression+':1 ('+('shared, 'if shared else'')+'constant over dt)'
        params=dict(eqs=equations,eq_th=expression+'>0'if phase.startswith('threshold')else'rand()<0'if phase.startswith('reset_empty')else'rand()<.7',
            eq_rst='v=v_rst+('+(expression if phase.startswith('reset')else'randn()')+')*mV;g=0*mV',
            offset=b.TimedArray([[0.,1.],[.25,.5]],dt=.1*b.ms),
            r_poi=3700*b.Hz,r_poi2=7200*b.Hz,t_rfc=0*b.ms,t_dly=.2*b.ms)
        if phase=='threshold_sparse':params['eq_th']='rand()<(.0001+.0001*(offset(t,a)+offset(t,b)))'
        if packed_extra is not None:
            params['eqs']+='\nrem:integer\nquo:integer'+('\nextra:1'if packed_extra else'')
            params['eq_rst']=('extra=a+1;'if packed_extra else'')+'rem=a%(-3);quo=a//(-3);a=rem+1;'+params['eq_rst']
        fixture=build_original_api_fixture(tmp_path/str(mode=='native'),parameters=params,exc=(0,0),exc2=(2,),silenced=(),
            initial_state={'a':0 if shared else np.resize([0,-1,0,-2],size),'b':0})
        return fixture

    def exercise(fixture, mode):
        yield observe_run(fixture,.2,rng=True)
        fixture.neurons.a=invalid_column if first else 0
        fixture.neurons.b=9
        if phase in ('threshold_single','threshold_sparse') and first:
            fixture.neurons.a=0;fixture.neurons.a[-1]=invalid_column
            fixture.neurons.b=0;fixture.neurons.b[0]=9
        if phase=='reset_empty':
            yield observe_run(fixture,.3,rng=True)
        else:
            failed=observe_run(fixture,.3,raises=b.BrianObjectException,rng=True)
            assert type(failed.error.__cause__)is IndexError
            yield failed
        fixture.neurons.a=0;fixture.neurons.b=-1
        yield observe_run(fixture,.3,rng=True)
    check_original_api_runs(build,exercise,monkeypatch,seed=731)


@pytest.mark.metal
@pytest.mark.parametrize('size',[65536,65537])
def test_checked_window_fault_and_repaired_cache(tmp_path,monkeypatch,size):
    test_checked_call_order_and_repair(tmp_path,monkeypatch,'threshold_single',True,False,size=size)


@pytest.mark.metal
@pytest.mark.parametrize('extra',[False,True])
def test_packed_reset_live_bindings_and_limit_fallback(tmp_path,monkeypatch,extra):
    from mlx_backend import expression_numpy_packed as packed
    original=packed._kernel
    calls=[]
    def kernel(*args,**kwargs):
        operation=original(*args,**kwargs)
        def dispatch(*args,**kwargs):
            calls.append(len(kwargs['inputs'])+len(kwargs['output_shapes']))
            assert calls[-1]<=31
            return operation(*args,**kwargs)
        return dispatch
    monkeypatch.setattr(packed,'_kernel',kernel)
    test_checked_call_order_and_repair(tmp_path,monkeypatch,'threshold_single',True,False,
        size=256,packed_extra=extra)
    assert bool(calls) is (not extra)


@pytest.mark.metal
def test_negative_invalid_column_keeps_original_message(tmp_path,monkeypatch):
    test_checked_call_order_and_repair(tmp_path,monkeypatch,'threshold_single',True,False,invalid_column=-3)


def test_normal_budget_uses_frozen_namespace_and_safe_unknown_fallback():
    from types import SimpleNamespace
    from mlx_backend.expression_numpy_packed import _normal_budget
    class Program:pass
    def program(values):
        p=Program();p.input_arrays={'encoded':SimpleNamespace(source_name='offset',shape=(1,2),values=np.array([values]))}
        return p
    small=program([0.,1.]);large=program([0.,1000.]);bad=program([0.,np.nan])
    expression='rand()<(.0001+.0001*offset(t,a))'
    assert _normal_budget(small,expression,131072)==256
    assert _normal_budget(large,expression,131072)>256
    assert _normal_budget(small,expression,131072)==256
    assert _normal_budget(bad,expression,131072)is None
    assert _normal_budget(small,'rand()<v',131072)is None
    assert _normal_budget(small,'rand()<1/offset(t,a)',131072)is None


@pytest.mark.metal
def test_sparse_budget_does_not_cap_dense_counts_after_prefix_refill():
    from dataclasses import replace
    import mlx.core as mx
    from mlx_backend.expression_rng import from_numpy_states
    from mlx_backend.expression_rng_prefix import prefix_pool,prefix_uniforms
    from mlx_backend.expression_runtime import RandomState,MixedLookahead
    from mlx_backend.expression_sparse_normal import draw
    from test_mlx_expression_shared_kernels import assert_stream
    cpus=[np.random.RandomState(81+i)for i in range(3)]
    cpus[1].randn(1)
    stream=from_numpy_states([cpu.get_state()for cpu in cpus])
    prefix=prefix_pool(stream,7,checkpoint=1)
    random=RandomState(stream,{},mx.zeros(3,mx.uint32),MixedLookahead(prefix,None,0,1))
    for counts in ([0,513,777],[1025,1,0]):
        counts=np.array(counts,np.uint32)
        values,stream,pool=draw(random,1025,mx.array(counts),target='numpy')
        actual=np.asarray(values).view(np.float64)
        for row,(cpu,count)in enumerate(zip(cpus,counts)):
            assert actual[row,:count].tobytes()==cpu.randn(count).tobytes()
        assert_stream(stream,cpus)
        values,stream,prefix=prefix_uniforms(stream,pool.data,9)
        actual=np.asarray(values).view(np.float64)
        for row,cpu in enumerate(cpus):assert actual[row].tobytes()==cpu.rand(9).tobytes()
        assert_stream(stream,cpus)
        random=replace(random,stream=stream,pool=replace(pool,data=prefix))


@pytest.mark.metal
def test_sparse_checked_fault_replay_and_repair(tmp_path,monkeypatch):
    from mlx_backend import expression_sparse_normal as sparse
    original=sparse.draw;calls=[]
    def draw(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(sparse,'draw',draw)
    test_checked_call_order_and_repair(tmp_path,monkeypatch,'threshold_sparse',True,False,size=65536,packed_extra=False)
    assert calls
