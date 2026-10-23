"""Actual Brian Cython refills versus portable native buffer/MT state."""
import ctypes
import numpy as np
import pytest
from mlx_backend.expression_rng import from_numpy_states,to_numpy_states
from mlx_backend.expression_cython import from_buffer_snapshots,buffer_snapshots,buffered_draws


@pytest.mark.metal
@pytest.mark.parametrize('sparse',[False,True])
def test_cython_uniform_normal_refills_interleaving_and_portable_resume(tmp_path,sparse):
    import brian2 as b
    import mlx.core as mx
    preferences={name:b.prefs[name]for name in('codegen.target','codegen.runtime.cython.cache_dir')}
    ambient=np.random.get_state()
    def capture():
        device=b.get_device();result={}
        for name,attribute in(('uniform','rand'),('normal','randn')):
            pointer=int(getattr(device,attribute+'_buffer')[0])
            result[name]=(np.ctypeslib.as_array((ctypes.c_double*20000).from_address(pointer)).copy()
                          if pointer else np.zeros(20000,np.float64))
            result[name+'_index']=int(getattr(device,attribute+'_buffer_index')[0])
        return result
    try:
        b.prefs.codegen.target='cython'
        b.prefs.codegen.runtime.cython.cache_dir=str(tmp_path/'cython-default')
        b.seed(81);np.random.randn(1)  # Ambient Gaussian cache predates Brian's buffers.
        initial=np.random.get_state();initial_buffers=capture();reference=[]
        sequence=[('rand',1),('randn',1),('rand',19999),('randn',20000),('rand',20001),
                  ('randn',1),('rand',0),('randn',0),('rand',65537),('randn',65537)]
        for function,count in sequence:
            if count:
                group=b.NeuronGroup(count,'a:1',clock=b.Clock(dt=.1*b.ms),name='cython_buffer_reference*')
                runner=group.run_regularly('a='+function+'()',when='start')
                b.Network(group,runner).run(.1*b.ms,namespace={})
                values=np.asarray(group.a_[:]).copy();del group,runner
            else:values=np.zeros(0,np.float64)
            reference.append((values,np.random.get_state(),capture()))
    finally:
        np.random.set_state(ambient)
        for name,value in preferences.items():b.prefs[name]=value
    state=from_numpy_states([initial]);buffers=from_buffer_snapshots([initial_buffers])
    for step,((function,count),(expected,want,cache))in enumerate(zip(sequence,reference)):
        # Spare capacity and zero-count calls exercise device selection without
        # consuming padded values or triggering an observable extra refill.
        actual,state,buffers=buffered_draws(state,buffers,function,count+7,mx.array([count],mx.uint32),sparse=sparse)
        output=np.asarray(actual).reshape(-1).view(np.float64)
        np.testing.assert_array_equal(output[:count],expected);np.testing.assert_array_equal(output[count:],0)
        got=to_numpy_states(state)[0];np.testing.assert_array_equal(got[1],want[1]);assert got[2:]==want[2:]
        snapshot=buffer_snapshots(buffers)[0]
        for name in('uniform','normal'):
            np.testing.assert_array_equal(snapshot[name],cache[name]);assert snapshot[name+'_index']==cache[name+'_index']
        if step==3:
            state=from_numpy_states([got]);buffers=from_buffer_snapshots([snapshot])


@pytest.mark.metal
def test_cython_poisson_native_refills_values_and_portable_resume(tmp_path):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend.expression_cython import poisson
    from test_mlx_cython_expression_rng import capture_buffers,assert_snapshot
    preferences={name:b.prefs[name]for name in('codegen.target','codegen.runtime.cython.cache_dir')}
    ambient=np.random.get_state()
    rates=np.resize(np.array([0.,-1.,.3,1.,3.,np.nextafter(10.,0),10.,17.,100.,1e16,np.nan,np.inf,-np.inf]),65537)
    try:
        b.prefs.codegen.target='cython';b.prefs.codegen.runtime.cython.cache_dir=str(tmp_path/'cython-default')
        b.seed(81);np.random.randn(1)
        group=b.NeuronGroup(len(rates),'lam:1\na:integer',clock=b.Clock(dt=.1*b.ms))
        group.lam=rates;runner=group.run_regularly('a=poisson(lam)',when='start')
        start=dict(capture_buffers(b.get_device()),numpy_state=np.random.get_state())
        b.Network(group,runner).run(.1*b.ms,namespace={})
        expected=np.asarray(group.a_[:]).copy()
        final=dict(capture_buffers(b.get_device()),numpy_state=np.random.get_state())
    finally:
        np.random.set_state(ambient)
        for name,value in preferences.items():b.prefs[name]=value
    state=from_numpy_states([start['numpy_state']]);buffers=from_buffer_snapshots([start]);begin=0
    for n in(0,1,20000,45536):
        pad=np.r_[rates[begin:begin+n],np.ones(7)]
        values,state,buffers,errors=poisson(state,buffers,mx.array(pad[None,:].view(np.uint64)),mx.array([n],mx.uint32))
        actual=np.asarray(values).reshape(-1)
        np.testing.assert_array_equal(np.asarray(errors),0)
        np.testing.assert_array_equal(actual[:n],expected[begin:begin+n]);np.testing.assert_array_equal(actual[n:],0)
        begin+=n
        snapshot=dict(buffer_snapshots(buffers)[0],numpy_state=to_numpy_states(state)[0])
        if not n:assert_snapshot(snapshot,start)
        if begin==20001:
            state=from_numpy_states([snapshot['numpy_state']]);buffers=from_buffer_snapshots([snapshot])
    assert_snapshot(dict(buffer_snapshots(buffers)[0],numpy_state=to_numpy_states(state)[0]),final)
