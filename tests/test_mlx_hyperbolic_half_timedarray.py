"""Own half-function policies composed with eager, masked TimedArray reads."""
from native_execution import forbid_call
from dataclasses import replace
import uuid

import numpy as np
import pytest

from test_mlx_hyperbolic_half import assert_raw_state


@pytest.mark.metal
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('invalid_selected',[False,True])
def test_actual_numpy_half_functions_with_masked_timedarray(monkeypatch,tmp_path,dtype,invalid_selected):
    import brian2 as b
    from component_cpu_reference import HyperbolicComponentCpu,component_inputs
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    mx=pytest.importorskip('mlx.core')
    b.BrianLogger.log_level_error()
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',dtype)
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',False)
    prefix='hyper_half_timed_'+uuid.uuid4().hex[:8]
    flag=prefix+'_flag';column=prefix+'_column'
    fields={fn:prefix+'_'+fn for fn in ('log10','sinh','cosh','tanh','arctan')}
    expressions={'log10':'log10(1+sinh('+flag+'))',
        'sinh':'sinh('+flag+')','cosh':'cosh('+flag+')','tanh':'tanh(sinh('+flag+'))',
        'arctan':'arctan(log10(1+sinh('+flag+')))'}
    model=resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\n'+flag+':boolean\n'+column+':integer\n'+
        ''.join(name+':1\n' for name in fields.values()),
        'eq_th':'i<2','t_rfc':0*b.ms,
        'eq_rst':'v=v_rst;g=0*mV;'+
            ';'.join(name+'='+expressions[fn]+'+drive(t,'+column+')' for fn,name in fields.items()),
        'drive':b.TimedArray(np.arange(6.).reshape(3,2),dt=.1*b.ms)})
    flags=np.array([False,True,True,False])
    columns=np.array([-3 if invalid_selected else -1,0,2,-3],np.int32)
    cpu=HyperbolicComponentCpu(4,2,model,'numpy',tmp_path/'numpy')
    setattr(cpu.neurons,flag,flags);setattr(cpu.neurons,column,columns)
    cpu.network.store('initial')
    if invalid_selected:
        with pytest.raises(b.BrianObjectException) as error:cpu.run()
        assert isinstance(error.value.__cause__,IndexError)
    else:
        _,_,expected=cpu.run()
    graph,stimulus=component_inputs(4,2,model)
    prepared=configured.prepare(graph,stimulus,model)
    assert '_eon_timed_result_' in prepared.program.reset_code
    (tmp_path/'cpu-reset.txt').write_text(str(cpu.neurons.resetter['spike'].codeobj.code))
    (tmp_path/'native-reset.metal').write_text(prepared.program.reset_code)
    original_initial=configured.initial_state
    def initial(*args,**kwargs):
        state=original_initial(*args,**kwargs)
        values=dict(state.values);values[flag]=mx.array(flags);values[column]=mx.array(columns)
        return replace(state,values=values)
    monkeypatch.setattr(configured,'initial_state',initial)
    forbid_call(b.Network, 'run', 'Native math ran a CPU timestep', monkeypatch)
    if invalid_selected:
        with pytest.raises(IndexError,match='TimedArray column'):
            configured.run_prepared(prepared,stimulus,record=False)
        return
    for recording in (False,True):
        result,state=configured.run_prepared(prepared,stimulus,return_state=True,record=recording)
        assert len(result.spikes.neurons)==(4 if recording else 0)
        assert_raw_state(configured.host_state(state,prepared),expected)
    resumed=initial(prepared)
    for _ in range(2):
        resumed,_=configured.advance(prepared,resumed,mx.zeros((1,),mx.uint8),2,1)
    configured.evaluate_state(resumed)
    assert_raw_state(configured.host_state(resumed,prepared),expected)
