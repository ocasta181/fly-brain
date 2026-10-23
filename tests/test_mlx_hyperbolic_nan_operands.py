"""Actual Brian NaN priority through ordinary arithmetic and owned ufuncs."""
from native_execution import forbid_call, require_metal
from dataclasses import replace
import uuid

import numpy as np
import pytest

from test_mlx_hyperbolic_half import assert_raw_state


@pytest.mark.metal
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_actual_numpy_hyperbolic_nan_arithmetic(monkeypatch,tmp_path,dtype):
    import brian2 as b
    from component_cpu_reference import HyperbolicComponentCpu,component_inputs
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    mx = require_metal('Metal required')
    b.BrianLogger.log_level_error()
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',dtype)
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',False)
    prefix='hyper_nan_operand_'+uuid.uuid4().hex[:8]
    left,right=prefix+'_left',prefix+'_right'
    if dtype==np.float32:
        a=np.array([0x7fc00111,0x7f800222,0x7fc00333],np.uint32).view(dtype)
        z=np.array([0xff800444,0xffc00555,0xffc00666],np.uint32).view(dtype)
    else:
        a=np.array([0x7ff8000000000111,0x7ff0000000000222,0x7ff8000000000333],np.uint64).view(dtype)
        z=np.array([0xfff0000000000444,0xfff8000000000555,0xfff8000000000666],np.uint64).view(dtype)
    expressions={prefix+'_'+function+'_'+name:function+'('+left+symbol+right+')'
        for function in ('log10','sinh','cosh','tanh')
        for name,symbol in [('add','+'),('sub','-'),('mul','*'),('div','/')]}
    model=resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\n'+left+':1\n'+right+':1\n'+
        ''.join(name+':1\n' for name in expressions),'eq_th':'True','t_rfc':0*b.ms,
        'eq_rst':'v=v_rst;g=0*mV;'+ ';'.join(name+'='+expr for name,expr in expressions.items())})
    cpu=HyperbolicComponentCpu(3,2,model,'numpy',tmp_path/'numpy')
    setattr(cpu.neurons,left,a);setattr(cpu.neurons,right,z)
    cpu.network.store('initial')
    _,_,expected=cpu.run()
    graph,stimulus=component_inputs(3,2,model)
    prepared=configured.prepare(graph,stimulus,model)
    (tmp_path/'actual-brian-reset.txt').write_text(str(cpu.neurons.resetter['spike'].codeobj.code))
    (tmp_path/'native-reset.metal').write_text(prepared.program.reset_code)
    original_initial=configured.initial_state
    def initial(*args,**kwargs):
        state=original_initial(*args,**kwargs)
        words=np.uint32 if dtype==np.float32 else np.uint64
        return replace(state,values={**state.values,left:mx.array(a.view(words)),right:mx.array(z.view(words))})
    monkeypatch.setattr(configured,'initial_state',initial)
    forbid_call(b.Network, 'run', 'Native math ran a CPU timestep', monkeypatch)
    _,state=configured.run_prepared(prepared,stimulus,return_state=True,record=False)
    assert_raw_state(configured.host_state(state,prepared),expected)
    resumed=initial(prepared)
    for _ in range(2):
        resumed,_=configured.advance(prepared,resumed,mx.zeros((1,),mx.uint8),2,1)
    configured.evaluate_state(resumed)
    assert_raw_state(configured.host_state(resumed,prepared),expected)
