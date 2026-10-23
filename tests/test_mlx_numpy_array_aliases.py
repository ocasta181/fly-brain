"""Actual Brian reset-local alias, in-place and rebinding semantics."""
import numpy as np
import pytest
from native_execution import require_metal
from component_execution import assert_numpy_component_modes


@pytest.mark.metal
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('reset',[
    'x=f;y=x+1;f+=1',
    'x=f;f=(f+1)*2;y=x',
    'x=f;y=x;y+=2',
    'x=f;y=x+0;x+=2',
    'tmp=f;x=tmp;tmp+=2;y=tmp',
])
def test_original_numpy_array_aliases(monkeypatch,tmp_path,dtype,reset):
    import brian2 as b
    from mlx_backend.model import resolve_model
    mx = require_metal('Metal required')
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',dtype)
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',False)
    model=resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\nx:1\ny:1\nf:1',
        'eq_th':'True','eq_rst':reset,'t_rfc':0*b.ms})
    assert model.reference_codegen_target=='numpy'
    assert model.integer_policy is None
    values=np.array([.1,.3,-.5,3.,-2.],dtype)
    assert_numpy_component_modes(monkeypatch,tmp_path,mx,model,{'f':values},
        prefix='array_alias',cpu_error='Native alias used CPU timesteps')
