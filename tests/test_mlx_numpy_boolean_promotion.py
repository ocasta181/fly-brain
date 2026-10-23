"""Actual NumPy Boolean-array promotion against the native renderer."""
from state_assertions import assert_raw_state
import numpy as np
import pytest

@pytest.mark.metal
def test_numpy_boolean_array_does_not_narrow_float64_function_argument(monkeypatch,tmp_path):
    """Actual NumPy bool+scalar promotion is distinct from float32-array promotion."""
    import uuid
    import brian2 as b
    from dataclasses import replace
    from component_cpu_reference import HyperbolicComponentCpu,component_inputs
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    mx=pytest.importorskip('mlx.core')
    b.BrianLogger.log_level_error()
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',False)
    # Brian's make_statements cache does not include the global optimiser
    # preference. Fresh source names isolate this policy from other fixtures.
    prefix='hyper_bool_'+uuid.uuid4().hex[:8]
    flag=prefix+'_flag'
    fields={fn:prefix+'_'+fn for fn in ('log10','sinh','cosh','tanh')}
    model=resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\n'+flag+':boolean\n'+
        ''.join(name+':1\n' for name in fields.values()),'eq_th':'True','t_rfc':0*b.ms,
        'eq_rst':'v=v_rst;g=0*mV;'+(';'.join(name+'='+fn+'('+flag+'+0.1)'
                                                  for fn,name in fields.items()))})
    cpu=HyperbolicComponentCpu(3,1,model,'numpy',tmp_path/'numpy')
    flags=np.array([False,True,False])
    setattr(cpu.neurons,flag,flags)
    cpu.network.store('initial')
    widths=[]
    before=cpu.network.before_run
    def observe(namespace):
        before(namespace)
        for obj in cpu.network.sorted_objects:
            for code in obj.code_objects:
                for fn in fields:
                    if fn not in code.namespace:continue
                    original=code.namespace[fn]
                    def wrapped(arg,_original=original,**kwargs):
                        result=_original(arg,**kwargs)
                        widths.append((np.asarray(arg).dtype,np.asarray(result).dtype))
                        return result
                    code.namespace[fn]=wrapped
    cpu.network.before_run=observe
    _,_,expected=cpu.run()
    assert widths==[(np.dtype('float64'),np.dtype('float64'))]*4
    graph,stimulus=component_inputs(3,1,model)
    prepared=configured.prepare(graph,stimulus,model)
    state=configured.initial_state(prepared)
    values=dict(state.values);values[flag]=mx.array(flags)
    state=replace(state,values=values)
    state,_=configured.advance(prepared,state,mx.zeros((1,),mx.uint8),1,1)
    configured.evaluate_state(state)
    actual=configured.host_state(state,prepared)
    assert_raw_state(actual,expected)
    (tmp_path/'cpu-reset.txt').write_text(str(cpu.neurons.resetter['spike'].codeobj.code))
    (tmp_path/'native-reset.metal').write_text(prepared.program.reset_code)
