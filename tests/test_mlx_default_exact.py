"""Default dispatch must retain Brian's physical precision at threshold."""
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
b = pytest.importorskip("brian2")
if not mx.metal.is_available():
    pytest.skip("Metal is required", allow_module_level=True)

from mlx_backend import configured, runner
from mlx_backend.model import resolve_model
from mlx_backend.models import Connectome, Stimulus

pytestmark = pytest.mark.metal


@pytest.mark.parametrize("target,record,supplied", [
    ("cython", True, False), ("cython", False, True),
    ("numpy", True, True), ("numpy", False, False),
])
def test_default_dispatch_preserves_near_threshold_physical_state(
    monkeypatch, tmp_path, target, record, supplied,
):
    monkeypatch.setitem(b.prefs, "codegen.target", target)
    monkeypatch.setitem(b.prefs, "codegen.runtime.cython.cache_dir", str(tmp_path/"cython"))
    model = resolve_model()
    # Retained CPU state immediately before sugar:t1:n1's first different
    # threshold crossing (neuron111697, step3595). The rounded fixed window
    # delayed that spike despite identical supplied Poisson events.
    voltage, synaptic = -0.04501350729210029, 0.009792217317009334
    clock = b.Clock(dt=model.dt_s*b.second)
    neurons = b.NeuronGroup(1, model.eqs, method="linear", threshold=model.eq_th,
        reset=model.eq_rst, refractory="rfc", namespace=model.brian_namespace(), clock=clock)
    neurons.v=voltage*b.volt; neurons.g=synaptic*b.volt
    neurons.rfc=model.t_rfc*b.second
    monitor=b.SpikeMonitor(neurons)
    b.Network(neurons,monitor).run(clock.dt)
    assert np.asarray(monitor.i).tolist() == [0]
    expected={name:np.asarray(getattr(neurons,name+"_")[:]).copy()
              for name in ("v","g","lastspike","not_refractory","rfc")}

    original_initial=configured.initial_state
    original_run=configured.run_prepared
    observed={}
    def seeded(prepared,trials=1,**kwargs):
        state=original_initial(prepared,trials,**kwargs)
        values=dict(state.values)
        for name,value in (("v",voltage),("g",synaptic)):
            values[name]=mx.array(np.array([value],np.float64).view(np.uint64))
        return replace(state,values=configured._pack_values(prepared.program,values))
    def observe(prepared,*args,**kwargs):
        result,state=original_run(prepared,*args,**kwargs,return_state=True)
        observed["target"]=prepared.model.reference_codegen_target
        observed["state"]=configured.host_state(state,prepared)
        return result
    monkeypatch.setattr(configured,"initial_state",seeded)
    monkeypatch.setattr(configured,"run_prepared",observe)
    graph=Connectome(np.array([111697]),np.empty(0,np.int32),np.empty(0,np.int32),
                     np.empty(0,np.int32),np.empty(0,np.float64))
    events=np.empty((1,1,0),np.uint8)
    stimulus=Stimulus(events,(),(),(0,),0,0,sha256(events.tobytes()).hexdigest())
    result=runner.run(graph,stimulus,(),precision="0",record=record,
                      model=model if supplied else None)
    assert observed["target"] == target
    assert result.spikes.steps.tolist() == ([0] if record else [])
    for name,value in expected.items():
        np.testing.assert_array_equal(observed["state"][name].reshape(-1).view(np.uint8),
                                      value.view(np.uint8),err_msg=name)


def test_default_target_binding_preserves_rates_and_separates_compiler_policy(monkeypatch):
    monkeypatch.setitem(b.prefs,"codegen.target","cython")
    model=resolve_model({"r_poi":123*b.Hz,"r_poi2":17*b.Hz})
    actual=runner._exact_default_model(model)
    assert actual.r_poi == 123 and actual.r_poi2 == 17
    assert actual.source_namespace is model.source_namespace
    assert actual.reference_codegen_target == "cython"
    assert not actual.state_specialization
    assert actual.signature != model.signature
    assert runner._exact_default_model(model) is actual
    uncacheable=runner._exact_default_model(replace(model,signature=None))
    assert uncacheable.signature is None
    assert uncacheable.reference_codegen_target == "cython"
    flags=list(actual.reference_compile_args)+["-ffp-contract=off"]
    monkeypatch.setitem(b.prefs,"codegen.cpp.extra_compile_args",flags)
    changed=runner._exact_default_model(model)
    assert changed.signature != actual.signature
    assert changed.reference_compile_args == tuple(flags)
