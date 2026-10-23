"""Actual mixed integer arrays select the enclosing NumPy floating ufunc loop."""
from native_execution import forbid_call, require_metal
from dataclasses import replace
import json
import uuid

import numpy as np
import pytest

from test_mlx_hyperbolic_half import assert_raw_state


@pytest.mark.metal
@pytest.mark.parametrize('integer_dtype,dtype,input_kind',[
    (np.int8,np.float64,'half'),(np.uint8,np.float32,'half'),
    (np.int16,np.float64,'half'),(np.uint16,np.float32,'half'),
    (np.int8,np.float32,'state'),(np.uint16,np.float32,'state'),
    (np.int32,np.float64,'half'),(np.int64,np.float64,'state'),
    (np.int8,np.float64,'clock'),(np.int16,np.float32,'clock'),
    (np.int8,np.float32,'wide_clock')])
def test_actual_numpy_hyperbolic_half_plus_integer(monkeypatch,tmp_path,integer_dtype,dtype,input_kind):
    import brian2 as b
    from component_cpu_reference import HyperbolicComponentCpu,component_inputs
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    mx = require_metal('Metal required')
    b.BrianLogger.log_level_error()
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',dtype)
    monkeypatch.setitem(b.prefs,'core.default_integer_dtype',integer_dtype)
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',False)
    prefix='hyper_mixed_integer_'+uuid.uuid4().hex[:8]
    flag,counter,source=prefix+'_flag',prefix+'_counter',prefix+'_source'
    gate=prefix+'_clock_gate'
    argument={'half':'sinh('+flag+')','state':source,
              'clock':'sinh('+gate+')','wide_clock':'t/ms'}[input_kind]
    expressions={prefix+'_'+function:function+'('+argument+'+'+counter+')'
                 for function in ('log10','sinh','cosh','tanh')}
    shared=(gate+'=t<dt:boolean (shared)\n') if input_kind=='clock' else ''
    model=resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\n'+flag+':boolean\n'+counter+':integer\n'+source+':1\n'+shared+
        ''.join(name+':1\n' for name in expressions),'eq_th':'True','t_rfc':0*b.ms,
        'eq_rst':'v=v_rst;g=0*mV;'+ ';'.join(name+'='+expr for name,expr in expressions.items())})
    flags=np.array([False,True,True,False]);counters=np.array([0,1,2,3],integer_dtype)
    sources=np.array([.125,.1,1.625,.3],dtype)
    cpu=HyperbolicComponentCpu(4,2,model,'numpy',tmp_path/'numpy')
    assert cpu.neurons.variables[counter].dtype==np.dtype(integer_dtype)
    setattr(cpu.neurons,flag,flags);setattr(cpu.neurons,counter,counters)
    setattr(cpu.neurons,source,sources)
    cpu.network.store('initial');_,_,expected=cpu.run()
    (tmp_path/'actual-brian-reset.txt').write_text(str(cpu.neurons.resetter['spike'].codeobj.code))
    np.savez(tmp_path/'actual-brian-state.npz',**expected)
    inputs={'half':np.sinh(flags),'state':sources,'clock':np.sinh(np.bool_(False)),
            'wide_clock':np.float64(model.dt_ms)}[input_kind]
    selected=inputs+counters
    (tmp_path/'numpy-loop.json').write_text(json.dumps({'integer_dtype':str(np.dtype(integer_dtype)),
        'input_kind':input_kind,'inner_dtype':str(inputs.dtype),'sum_dtype':str(selected.dtype)},indent=2)+'\n')
    graph,stimulus=component_inputs(4,2,model)
    prepared=configured.prepare(graph,stimulus,model)
    (tmp_path/'native-reset.metal').write_text(prepared.program.reset_code)
    original_initial=configured.initial_state
    def initial(*args,**kwargs):
        state=original_initial(*args,**kwargs)
        words=np.uint32 if dtype==np.float32 else np.uint64
        return replace(state,values={**state.values,flag:mx.array(flags),counter:mx.array(counters),
            source:mx.array(sources.view(words))})
    monkeypatch.setattr(configured,'initial_state',initial)
    forbid_call(b.Network, 'run', 'Native math ran a CPU timestep', monkeypatch)
    for recording in (False,True):
        result,state=configured.run_prepared(prepared,stimulus,return_state=True,record=recording)
        assert len(result.spikes.neurons)==(8 if recording else 0)
        assert_raw_state(configured.host_state(state,prepared),expected)
    state=initial(prepared)
    for _ in range(2):state,_=configured.advance(prepared,state,mx.zeros((1,),mx.uint8),2,1)
    configured.evaluate_state(state)
    assert_raw_state(configured.host_state(state,prepared),expected)
