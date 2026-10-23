"""Owned function registrations composed with the qualified shared half core."""
from native_execution import forbid_call, require_metal
from dataclasses import replace
import uuid

import numpy as np
import pytest

from test_mlx_hyperbolic_half import assert_raw_state


@pytest.mark.metal
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_actual_numpy_hyperbolic_half_sqrt_floor_clip(monkeypatch,tmp_path,dtype):
    import brian2 as b
    from component_cpu_reference import HyperbolicComponentCpu,component_inputs
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    mx = require_metal('Metal required')
    b.BrianLogger.log_level_error()
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',dtype)
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',False)
    prefix='hyper_core_'+uuid.uuid4().hex[:8]
    flag=prefix+'_flag'
    inputs={'root':'sqrt('+flag+')','floor':'floor(sinh('+flag+'))+1',
            'clip':'clip(-sinh('+flag+'),lower_zero,upper_zero)'}
    expressions={prefix+'_'+function+'_'+kind:function+'('+argument+')'
        for function in ('log10','sinh','cosh','tanh') for kind,argument in inputs.items()}
    model=resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\n'+flag+':boolean\n'+
        ''.join(name+':1\n' for name in expressions),'eq_th':'True','t_rfc':0*b.ms,
        'eq_rst':'v=v_rst;g=0*mV;'+ ';'.join(name+'='+expr for name,expr in expressions.items()),
        'lower_zero':np.float64(-0.),'upper_zero':np.float64(0.)})
    flags=np.array([False,True,True,False])
    cpu=HyperbolicComponentCpu(4,2,model,'numpy',tmp_path/'numpy')
    setattr(cpu.neurons,flag,flags)
    cpu.network.store('initial')
    _,_,expected=cpu.run()
    graph,stimulus=component_inputs(4,2,model)
    prepared=configured.prepare(graph,stimulus,model)
    (tmp_path/'actual-brian-reset.txt').write_text(str(cpu.neurons.resetter['spike'].codeobj.code))
    (tmp_path/'native-reset.metal').write_text(prepared.program.reset_code)
    original_initial=configured.initial_state
    def initial(*args,**kwargs):
        state=original_initial(*args,**kwargs)
        return replace(state,values={**state.values,flag:mx.array(flags)})
    monkeypatch.setattr(configured,'initial_state',initial)
    forbid_call(b.Network, 'run', 'Native math ran a CPU timestep', monkeypatch)
    for recording in (False,True):
        result,state=configured.run_prepared(prepared,stimulus,return_state=True,record=recording)
        assert len(result.spikes.neurons)==(8 if recording else 0)
        assert_raw_state(configured.host_state(state,prepared),expected)
    resumed=initial(prepared)
    for _ in range(2):
        resumed,_=configured.advance(prepared,resumed,mx.zeros((1,),mx.uint8),2,1)
    configured.evaluate_state(resumed)
    assert_raw_state(configured.host_state(resumed,prepared),expected)
