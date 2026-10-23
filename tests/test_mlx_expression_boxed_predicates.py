"""Actual Cython boxed comparisons must govern subsequent RNG consumption."""
from state_assertions import assert_state_bytes_equal
import numpy as np
import pytest
from mlx_backend.model import resolve_model
from mlx_backend.models import Connectome,Stimulus

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('operator,limit,consumes',[
    ('>','9999999999999998',True),
    ('<','10000000000000001',True),
    ('>','9999999999999999',True),
    ('<','9999999999999999',False),
    pytest.param('<',str(2**80+1),True,id='large-integer-init'),
    pytest.param('>',str(-2**80-1),True,id='large-negative-init'),
    pytest.param('<',str(2**1024),True,id='large-overflow-init'),
])
def test_boxed_comparison_controls_following_draw(operator,limit,consumes,tmp_path,reverse=False):
    import brian2 as b
    from mlx_backend.configured import prepare,initial_state,host_state,run_prepared
    from mlx_backend.expression_runtime import snapshots
    from test_mlx_cython_expression_rng import cpu_reference,assert_snapshot
    producer='(.13+randn()+rand()+poisson(3))'
    comparison=f'{limit}{dict(zip(("<",">"),(">","<")))[operator]}{producer}'if reverse else f'{producer}{operator}{limit}'
    model=resolve_model(dict(eqs='v:volt\ng:volt\nrfc:second\na:1',
        eq_th=f'({comparison}) and (rand()<.5)',
        eq_rst='a+=1;v=v_rst;g=0*mV',t_rfc=0*b.ms),dt_ms=.1,random_target='cython')
    graph=Connectome(np.arange(1),np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0),np.zeros(0))
    stimulus=Stimulus(np.zeros((1,1,0),np.uint8),(),(),(0,),81,0,None,.1)
    names=('v','g','rfc','a','lastspike','not_refractory')
    start,events,expected,final,error=cpu_reference(model,tmp_path,names,steps=1,neurons=1,
        normal_prefix=np.full(8,1e16),uniform_prefix=np.full(16,.125))
    assert error is None
    assert final['uniform_index']-start['uniform_index']==3+int(consumes)
    prepared=prepare(graph,stimulus,model)
    for record in(True,False):
        result,state=run_prepared(prepared,stimulus,return_state=True,random_states=[start],record=record)
        if record:
            np.testing.assert_array_equal(result.spikes.neurons,events[0])
            np.testing.assert_array_equal(result.spikes.steps,events[1])
        assert_state_bytes_equal(host_state(state,prepared),expected)
        assert_snapshot(snapshots(state.random)[0],final)


def test_exact_integer_comparison_helper_all_relations_and_boundaries():
    import operator
    import mlx.core as mx
    from mlx_backend.binary64 import MSL_HEADER
    from mlx_backend.expression_profile_boxed import float_integer_compare
    integers=[0,1,-1,2**53-1,2**53+1,-2**53+1,-2**53-1,
        10**16-1,10**16+1,-10**16-1,2**1024,-2**1024]
    values=np.array([-np.inf,-np.finfo(float).max,-1e16,np.nextafter(-1e16,-np.inf),
        np.nextafter(-1e16,np.inf),-1.,np.nextafter(-1.,0.),-0.,0.,np.nextafter(0.,1.),
        np.nextafter(1.,0.),1.,2**53,np.nextafter(float(2**53),np.inf),
        1e16,np.nextafter(1e16,-np.inf),np.nextafter(1e16,np.inf),np.finfo(float).max,np.inf,np.nan],np.float64)
    operators=(operator.lt,operator.le,operator.eq,operator.ne,operator.gt,operator.ge)
    predicates=[float_integer_compare('raw',integer,op)for integer in integers for op in range(6)]
    code='\n'.join(f'output[row*{len(predicates)}+{i}]=uchar({value});'for i,value in enumerate(predicates))
    kernel=mx.fast.metal_kernel(name='exact_boxed_predicate_domain',input_names=['input'],output_names=['output'],
        header=MSL_HEADER,source=f'uint row=thread_position_in_grid.x;if(row>={len(values)})return;ulong raw=input[row];'+code,
        compile_options={'math_mode':'safe'})
    actual=kernel(inputs=[mx.array(values.view(np.uint64))],grid=(len(values),1,1),threadgroup=(len(values),1,1),
        output_shapes=[(len(values),len(predicates))],output_dtypes=[mx.uint8])[0]
    expected=np.array([[operation(float(value),integer)for integer in integers for operation in operators]for value in values],np.uint8)
    np.testing.assert_array_equal(np.asarray(actual),expected)


@pytest.mark.parametrize('limit',['10000000000000001',str(2**80+1)])
def test_original_reversed_boxed_predicate(limit,tmp_path):
    test_boxed_comparison_controls_following_draw('<',limit,True,tmp_path,reverse=True)
