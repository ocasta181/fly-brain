"""NumPy function/copy carriers preserve bits until a real conversion occurs."""
from state_assertions import assert_raw_state as raw_state_equal
from dataclasses import replace
import uuid

import numpy as np
import pytest




@pytest.mark.metal
def test_numpy_float32_function_copy_and_arithmetic_nan_policies(monkeypatch,tmp_path):
    import brian2 as b
    from component_cpu_reference import HyperbolicComponentCpu,component_inputs
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    mx=pytest.importorskip('mlx.core')
    b.BrianLogger.log_level_error()
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',False)
    prefix='raw_carrier_'+uuid.uuid4().hex[:8]
    x=prefix+'_x';copy=prefix+'_copy';addition=prefix+'_add'
    fields={fn:prefix+'_'+fn for fn in ('log10','sinh','cosh','tanh')}
    model=resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\n'+
        ''.join(name+':1\n' for name in (x,copy,addition,*fields.values())),
        'eq_th':'True','t_rfc':0*b.ms,'eq_rst':'v=v_rst;g=0*mV;'+
        ';'.join([copy+'='+x,addition+'='+x+'+0',*(name+'='+fn+'('+x+')' for fn,name in fields.items())])})
    raw=np.array([0x7f800001,0xff800001,0x7fc12345,1,0x80000001,0x80000000,0x3f800000],np.uint32)
    cpu=HyperbolicComponentCpu(len(raw),1,model,'numpy',tmp_path/'numpy')
    setattr(cpu.neurons,x,raw.view(np.float32));cpu.network.store('initial')
    _,_,expected=cpu.run()
    assert expected[copy].reshape(-1).view(np.uint32)[0]==raw[0]
    assert expected[fields['log10']].reshape(-1).view(np.uint32)[0]==raw[0]
    assert expected[addition].reshape(-1).view(np.uint32)[0]==(raw[0]|0x400000)
    graph,stimulus=component_inputs(len(raw),1,model)
    prepared=configured.prepare(graph,stimulus,model)
    state=configured.initial_state(prepared);values=dict(state.values);values[x]=mx.array(raw)
    state=replace(state,values=values)
    state,_=configured.advance(prepared,state,mx.zeros((1,),mx.uint8),1,1)
    configured.evaluate_state(state)
    raw_state_equal(configured.host_state(state,prepared),expected)
    (tmp_path/'cpu-reset.txt').write_text(str(cpu.neurons.resetter['spike'].codeobj.code))
    (tmp_path/'native-reset.metal').write_text(prepared.program.reset_code)


@pytest.mark.metal
def test_numpy_float32_state_has_real_float64_store_conversion(monkeypatch,tmp_path):
    import brian2 as b
    from component_cpu_reference import HyperbolicComponentCpu,component_inputs
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    mx=pytest.importorskip('mlx.core')
    b.BrianLogger.log_level_error()
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',False)
    prefix='raw_widen_'+uuid.uuid4().hex[:8]
    signal=prefix+'_signal';x=prefix+'_x';y=prefix+'_y'
    model=resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\n'+signal+':second\n'+x+':1\n'+y+':1\n',
        'eq_th':'True','t_rfc':0*b.ms,
        'eq_rst':'v=v_rst;g=0*mV;'+y+'=log10('+x+');lastspike='+signal})
    # Brian keeps its internal lastspike array at clock precision, even when
    # the model's custom signal array has float32 storage.
    assert model.analysis.variables['lastspike'].dtype=='float64'
    raw=np.array([0x7f800001,0xff800001,0x7fc12345],np.uint32)
    cpu=HyperbolicComponentCpu(3,1,model,'numpy',tmp_path/'numpy')
    setattr(cpu.neurons,x,raw.view(np.float32))
    setattr(cpu.neurons,signal+'_',raw.view(np.float32))
    cpu.network.store('initial');_,_,expected=cpu.run()
    graph,stimulus=component_inputs(3,1,model)
    prepared=configured.prepare(graph,stimulus,model)
    state=configured.initial_state(prepared);values=dict(state.values)
    values[x]=mx.array(raw);values[signal]=mx.array(raw)
    state=replace(state,values=values)
    state,_=configured.advance(prepared,state,mx.zeros((1,),mx.uint8),1,1)
    configured.evaluate_state(state)
    raw_state_equal(configured.host_state(state,prepared),expected)
    (tmp_path/'cpu-reset.txt').write_text(str(cpu.neurons.resetter['spike'].codeobj.code))
    (tmp_path/'native-reset.metal').write_text(prepared.program.reset_code)
