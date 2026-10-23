"""Original input phases must retain deterministic reset and RNG semantics."""

import numpy as np
import pytest


def test_shared_input_compilation_cache_is_separate():
    import brian2 as b
    from original_api_reference import cpu_runtime
    from mlx_backend.configured import _compiled_program
    from mlx_backend.expression_fixed import FixedRandomProgram
    from mlx_backend.model import resolve_model
    with cpu_runtime():
        model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second',eq_th='rand()<.7',
            eq_rst='v=v_rst;g=0*mV',unused=b.TimedArray([[0.,1.]],dt=.1*b.ms)))
        fixed=_compiled_program(model,4)
        shared=_compiled_program(model,4,shared_expression_inputs=True)
        assert isinstance(fixed.random_program,FixedRandomProgram)
        assert not isinstance(shared.random_program,FixedRandomProgram)
        assert not fixed.shared_expression_inputs and shared.shared_expression_inputs
        assert _compiled_program(model,4) is fixed
        assert _compiled_program(model,4,shared_expression_inputs=True) is shared


@pytest.mark.metal
@pytest.mark.parametrize('target,conditional,unused,used',[
    ('numpy',False,False,False),('numpy',False,True,False),
    ('numpy',True,True,False),('cython',False,True,False),
    ('cython',True,False,False),('numpy',False,False,True),
])
def test_original_inputs_and_deterministic_reset(tmp_path,monkeypatch,target,conditional,unused,used,
                                                size=4,normal=False,record=True,empty=False,zero=False):
    import brian2 as b
    from original_api_reference import build_original_api_fixture
    from original_api_protocol import check_original_api_runs, observe_run
    if size>4:
        import original_api_reference as reference
        monkeypatch.setattr(reference, "write_tiny_dataset", reference.synthetic_dataset(size))

    def build(mode):
        equations=('dv/dt=(v_0-v)/t_mbr:volt (unless refractory)'if conditional else'v:volt')
        params=dict(eqs=equations+'\ng:volt\nrfc:second',eq_th='rand()<.7',
            eq_rst='v=v_rst;g=0*mV',t_rfc=.2*b.ms,t_dly=.2*b.ms,
            r_poi=3700*b.Hz,r_poi2=7200*b.Hz)
        if unused:params['unused']=b.TimedArray([[0.,1.],[.25,.5]],dt=.1*b.ms)
        if used:
            params['offset']=b.TimedArray([[0.,.2],[.25,.5]],dt=.1*b.ms)
            params['eq_th']='rand()<(.3+offset(t,i%2))'
        if normal:params['eq_th']='randn()>.2'
        if zero:params.update(r_poi=0*b.Hz,r_poi2=0*b.Hz)
        fixture=build_original_api_fixture(tmp_path/str(mode=='native'),parameters=params,
            exc=()if empty else(0,0),exc2=()if empty else(2,),silenced=(),initial_state={'rfc':np.full(size,.0002)*b.second})
        if not record:
            fixture.network.remove(fixture.monitor)
            fixture.monitor=b.SpikeMonitor(fixture.neurons,record=False)
            fixture.network.add(fixture.monitor)
        return fixture

    def exercise(fixture, mode):
        for duration in (.2,.5):yield observe_run(fixture,duration,rng=True)
    check_original_api_runs(build,exercise,monkeypatch,seed=731,target=target)


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
def test_shared_input_window_continuation(tmp_path,monkeypatch,target):
    from mlx_backend import expression_window
    packet=expression_window._packet;calls=[]
    def capture(state):
        calls.append(state.step)
        return packet(state)
    monkeypatch.setattr(expression_window,'_packet',capture)
    test_original_inputs_and_deterministic_reset(tmp_path,monkeypatch,target,True,True,False,size=65536)
    assert calls


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
def test_shared_normal_threshold_inputs(tmp_path,monkeypatch,target):
    test_original_inputs_and_deterministic_reset(tmp_path,monkeypatch,target,True,False,False,normal=True)


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('conditional',[False,True])
def test_shared_normal_window_continuation(tmp_path,monkeypatch,target,conditional):
    from mlx_backend import expression_window
    packet=expression_window._packet;calls=[]
    def capture(state):
        calls.append(state.step)
        return packet(state)
    monkeypatch.setattr(expression_window,'_packet',capture)
    test_original_inputs_and_deterministic_reset(tmp_path,monkeypatch,target,conditional,False,False,
        size=65536,normal=True)
    assert calls


@pytest.mark.metal
@pytest.mark.parametrize('capacity,checkpoint',[(31,8),(31,1)])
def test_shared_normal_prefix_odd_empty_and_exhaustion(capacity,checkpoint):
    import mlx.core as mx
    from mlx_backend.expression_rng import from_numpy_states
    from mlx_backend.expression_rng_prefix import prefix_pool,prefix_normals,prefix_uniforms
    from mlx_backend.expression_normal_prefix import prepare
    from test_mlx_expression_shared_kernels import assert_stream
    cpus=[np.random.RandomState(seed)for seed in (31,32,33)]
    cpus[0].standard_normal()  # An existing cached pair must be consumed first.
    stream=from_numpy_states([cpu.get_state()for cpu in cpus])
    cache=prepare(prefix_pool(stream,capacity,checkpoint=checkpoint))
    assert (cache.data.normal is None)==bool(cache.capacity%2)
    for normal,counts in ((True,[7,0,9]),(False,[3,2,1]),(True,[5,11,0]),
                          (True,[0,1,1]),(True,[25,25,25]),(True,[3,3,3])):
        maximum=max(counts)
        actual,stream,cache=(prefix_normals if normal else prefix_uniforms)(
            stream,cache,maximum,mx.array(counts,mx.uint32))
        actual=np.asarray(actual).view(np.float64)
        for row,(cpu,count)in enumerate(zip(cpus,counts)):
            expected=cpu.standard_normal(count)if normal else cpu.random_sample(count)
            assert actual[row,:count].tobytes()==expected.tobytes()
            assert not actual[row,count:].any()
        assert_stream(stream,cpus)
    empty,unchanged,_=prefix_normals(stream,cache,0)
    assert empty.shape==(3,0) and unchanged is stream


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('mode',['counts','empty','zero'])
def test_shared_normal_input_contracts(tmp_path,monkeypatch,target,mode):
    test_original_inputs_and_deterministic_reset(tmp_path,monkeypatch,target,True,False,False,
        size=65536 if mode=='counts' else 4,normal=True,
        record=mode!='counts',empty=mode=='empty',zero=mode=='zero')


@pytest.mark.metal
def test_normal_recording_high_bits_and_empty():
    import mlx.core as mx
    from mlx_backend.expression_recording import collect
    blocks=[np.array([0,1,0x80000000,0xffffffff],np.uint32),np.array([3,0,2,1],np.uint32)]
    starts=[2**32+7,2**32+39];lengths=[32,2]
    indices,steps=collect([mx.array(block)for block in blocks],starts,lengths)
    expected=[(row,start+bit)for block,start,length in zip(blocks,starts,lengths)
        for row,word in enumerate(block)for bit in range(length)if int(word)&(1<<bit)]
    assert list(zip(indices.tolist(),steps.tolist()))==sorted(expected,key=lambda event:(event[1],event[0]))
    indices,steps=collect([mx.zeros((4,),mx.uint32)],[0],[3])
    assert indices.shape==steps.shape==(0,)


@pytest.mark.metal
@pytest.mark.parametrize('capacity',[8,128])
def test_normal_recording_bounded_chunks(tmp_path,monkeypatch,capacity):
    import mlx.core as mx
    from mlx_backend import expression_recording as recording
    monkeypatch.setattr(recording,'MAX_EVENTS',capacity)
    blocks=[np.array([0,1,0x80000000,0xffffffff],np.uint32),np.array([3,0,2,1],np.uint32)]
    starts=[2**32+7,2**32+39];lengths=[32,2]
    indices,steps=recording.collect([mx.array(block)for block in blocks],starts,lengths)
    expected=[(row,start+bit)for block,start,length in zip(blocks,starts,lengths)
        for row,word in enumerate(block)for bit in range(length)if int(word)&(1<<bit)]
    assert sorted(zip(indices.tolist(),steps.tolist()))==sorted(expected)
    indices,steps=recording.collect([mx.zeros((4,),mx.uint32)],[0],[3])
    assert indices.shape==steps.shape==(0,)
