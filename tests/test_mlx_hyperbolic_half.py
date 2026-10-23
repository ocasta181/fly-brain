"""Actual Brian NumPy half temporaries composed with hyperbolic functions."""
from state_assertions import assert_raw_state
from dataclasses import replace
import uuid

import numpy as np
import pytest




@pytest.mark.metal
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('optimise',[False,True])
def test_actual_numpy_boolean_hyperbolic_temporaries(monkeypatch,tmp_path,dtype,optimise):
    import brian2 as b
    from component_cpu_reference import HyperbolicComponentCpu,component_inputs
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    mx=pytest.importorskip('mlx.core')
    b.BrianLogger.log_level_error()
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',dtype)
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',optimise)
    # make_statements caches do not include the global optimiser preference.
    prefix='hyper_half_'+uuid.uuid4().hex[:8]
    flag=prefix+'_flag'
    fields={fn:prefix+'_'+fn for fn in ('log10','sinh','cosh','tanh')}
    nested={fn:prefix+'_nested_'+fn for fn in fields}
    mixed=prefix+'_mixed'
    cutoff=repr(np.array(0x39a9d80b,np.uint32).view(np.float32).item())
    def call(name,argument):
        expression=name+'('+argument+')'
        # Brian's default Boolean optimiser emits unbound bare `-inf` for
        # log10(False). Clip only this diagnostic's final range so the actual
        # CPU reference can run; primitive half-zero/Inf cases stay covered.
        return 'clip('+expression+',-16,16)' if name=='log10' else expression
    reset=['v=v_rst','g=0*mV']
    reset += [name+'='+call(fn,flag)+'+0.1' for fn,name in fields.items()]
    reset += [name+'='+call(fn,'sinh('+flag+')')+'+0.1' for fn,name in nested.items()]
    reset += [mixed+'=tanh(sinh('+flag+'))+tanh(x+'+cutoff+')',flag+'=not '+flag]
    model=resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\nx:1\n'+flag+':boolean\n'+
        ''.join(name+':1\n' for name in (*fields.values(),*nested.values(),mixed)),
        'eq_th':'(tanh('+flag+')+0.1)+x > 0.86145 or timestep(t,dt)==0',
        't_rfc':0*b.ms,'eq_rst':';'.join(reset)})
    assert model.reference_optimise is optimise
    cpu=HyperbolicComponentCpu(3,3,model,'numpy',tmp_path/'numpy')
    flags=np.array([False,True,False])
    setattr(cpu.neurons,flag,flags)
    monitor=b.SpikeMonitor(cpu.neurons)
    cpu.network.add(monitor);cpu.network.store('initial')
    _,_,expected=cpu.run()
    expected_neurons=np.asarray(monitor.i[:])
    expected_steps=np.rint(np.asarray(monitor.t_[:])/model.dt_s).astype(np.int64)
    # Native preparation must use the policy saved with the actual model.
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',not optimise)
    graph,stimulus=component_inputs(3,3,model)
    prepared=configured.prepare(graph,stimulus,model)
    assert b.prefs.codegen.loop_invariant_optimisations is (not optimise)
    (tmp_path/'cpu-reset.txt').write_text(str(cpu.neurons.resetter['spike'].codeobj.code))
    (tmp_path/'native-reset.metal').write_text(prepared.program.reset_code)
    (tmp_path/'cpu-threshold.txt').write_text(str(cpu.neurons.thresholder['spike'].codeobj.code))
    (tmp_path/'native-threshold.metal').write_text(prepared.program.threshold_code)
    original_initial=configured.initial_state
    def initial(*args,**kwargs):
        state=original_initial(*args,**kwargs)
        values=dict(state.values);values[flag]=mx.array(flags)
        return replace(state,values=values)
    monkeypatch.setattr(configured,'initial_state',initial)
    result,state=configured.run_prepared(prepared,stimulus,return_state=True)
    np.testing.assert_array_equal(result.spikes.neurons,expected_neurons)
    np.testing.assert_array_equal(result.spikes.steps,expected_steps)
    assert_raw_state(configured.host_state(state,prepared),expected)
    silent,silent_state=configured.run_prepared(prepared,stimulus,return_state=True,record=False)
    assert len(silent.spikes.neurons)==0
    assert_raw_state(configured.host_state(silent_state,prepared),expected)
    resumed=initial(prepared)
    for length in (1,2):
        resumed,_=configured.advance(prepared,resumed,mx.zeros((1,),mx.uint8),3,length)
    configured.evaluate_state(resumed)
    assert_raw_state(configured.host_state(resumed,prepared),expected)
    if not optimise:
        assert 'h_round' in prepared.program.reset_code
        assert '_eon_hyper32_tanh_scalar' in prepared.program.reset_code
        ordinary='_eon_hyper32_tanh_numpy' if dtype==np.float32 else '_eon_hyper_tanh_numpy'
        assert ordinary in prepared.program.reset_code
