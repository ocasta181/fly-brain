"""Actual unsigned caller conversion must retain the high bit."""
from state_assertions import assert_state_bytes_equal
import numpy as np
import pytest

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_actual_cython_unsigned64_conversion_after_draw(dtype,tmp_path,derived=False,constant=False,bound=None):
    import brian2 as b
    import mlx.core as mx
    from dataclasses import replace
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome,Stimulus
    from mlx_backend.configured import prepare,initial_state,advance,evaluate_state,host_state
    from mlx_backend.expression_runtime import snapshots
    from test_mlx_cython_expression_rng import cpu_reference,assert_snapshot
    previous=b.prefs.core.default_integer_dtype
    try:
        b.prefs.core.default_integer_dtype=np.uint64
        model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\nu:integer\na:1\nb:1',
            eq_th='rand()<1',eq_rst=('u=0-i-1;'if derived else'')+('a=bound+rand();b=bound;'if constant else'a=u+rand();b=u;')+'v=v_rst;g=0*mV',
            bound=np.uint64(2**63+2**39+1)if bound is None else bound,t_rfc=0*b.ms),
            state_dtype=dtype,dt_ms=.1,random_target='cython')
    finally:b.prefs.core.default_integer_dtype=previous
    names=('v','g','rfc','u','a','b','lastspike','not_refractory')
    unsigned=np.array([2**63-1,2**63+1,2**64-1],np.uint64)
    start,events,expected,final,error=cpu_reference(model,tmp_path,names,steps=1,neurons=3,initial_values={'u':unsigned})
    assert error is None
    assert np.all(expected['a']>0)and np.all(expected['b']>0)
    graph=Connectome(np.arange(3),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    inputs=Stimulus(np.zeros((1,1,0),np.uint8),(),(),(0,),81,0,None,.1)
    p=prepare(graph,inputs,model)
    state=initial_state(p,random_states=[start]);values=dict(state.values);values['u']=mx.array(unsigned)
    state=replace(state,values=values);state,bits=advance(p,state,mx.zeros((1,),mx.uint8),1,1)
    evaluate_state(state)
    np.testing.assert_array_equal(np.flatnonzero(np.asarray(bits)),events[0]);assert np.all(events[1]==0)
    assert_state_bytes_equal(host_state(state,p),expected)
    assert_snapshot(snapshots(state.random)[0],final)


def test_actual_cython_complement_and_sign_extend_before_unsigned_draw_consumer(tmp_path):
    test_actual_cython_unsigned64_conversion_after_draw('float64',tmp_path,derived=True)


def test_actual_cython_constant_unsigned_above_float32_midpoint(tmp_path):
    test_actual_cython_unsigned64_conversion_after_draw('float32',tmp_path,constant=True)


def test_actual_cython_constant_signed_above_float32_midpoint(tmp_path):
    test_actual_cython_unsigned64_conversion_after_draw('float32',tmp_path,constant=True,
        bound=np.int64(2**62+2**38+1))
