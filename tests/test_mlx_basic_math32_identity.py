from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.metal
def test_wrapper_shape_and_ambient_preferences_are_snapshotted(monkeypatch,tmp_path):
 import brian2 as b
 from brian2.core.functions import DEFAULT_FUNCTIONS
 from mlx_backend import configured
 from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
 from trigonometry_reference import original_reference
 from component_cpu_reference import component_inputs
 monkeypatch.setitem(b.prefs,'codegen.target','numpy')
 monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
 monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',False)
 implementations=DEFAULT_FUNCTIONS['expm1'].implementations
 monkeypatch.setattr(implementations,'_implementations',dict(implementations._implementations))
 implementations.add_numpy_implementation(DEFAULT_FUNCTIONS['expm1'].pyfunc,discard_units=False)
 reset='sample=expm1(basic_scalar);phase=sample;sample+=.1;sample=phase+.1'
 params=dict(eqs=DEFAULT_EQUATIONS+'sample:1\nphase:1\n',eq_th='True',eq_rst='v=v_rst;'+reset,t_rfc=0*b.ms,basic_scalar=np.float32(.1))
 options=dict(state_inputs=True,dtype='float32',optimise=False,namespace_values={'basic_scalar':np.float32(.1)},result_expression=reset.split('=',1)[1],run_steps=(0,1,2))
 first=resolve_model(params)
 expected_first,_=original_reference('numpy','expm1',np.zeros(4,np.float32),tmp_path/'first-cpu',**options)
 implementations.add_implementation('numpy',code=np.expm1)
 second=resolve_model(params)
 expected_second,_=original_reference('numpy','expm1',np.zeros(4,np.float32),tmp_path/'second-cpu',**options)
 assert dict(first.numpy_basic_wrappers)['expm1']
 assert not dict(second.numpy_basic_wrappers)['expm1']
 assert first.signature!=second.signature
 assert not np.array_equal(expected_first.view(np.uint32),expected_second.view(np.uint32))
 monkeypatch.setitem(b.prefs,'codegen.target','cython')
 monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float64)
 monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(ROOT/'executions/cython-identity-default'))
 for model,expected in ((first,expected_first),(second,expected_second)):
  graph,stimulus=component_inputs(4,3,model)
  prepared=configured.prepare(graph,stimulus,model)
  result,state=configured.run_prepared(prepared,stimulus,record=False,return_state=True)
  np.testing.assert_array_equal(configured.host_state(state,prepared)['sample'][0].view(np.uint32),expected.view(np.uint32))
  assert b.prefs.core.default_float_dtype==np.float64 and b.prefs.codegen.target=='cython'


def test_active_cpp_target_overrides_runtime_codegen_preference(monkeypatch):
 import brian2 as b
 from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
 monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float32)
 monkeypatch.setitem(b.prefs,'codegen.target','numpy')
 params=dict(eqs=DEFAULT_EQUATIONS+'sample:1\n',eq_th='True',eq_rst='v=v_rst;sample=expm1(sample)')
 b.set_device('cpp_standalone',build_on_run=False)
 try:
  implicit=resolve_model(params)
  explicit=resolve_model(params,cpu_target='cpp')
  assert implicit.reference_codegen_target==implicit.float32_codegen=='cpp'
  assert implicit.signature==explicit.signature
 finally:
  b.device.reinit();b.set_device('runtime')
