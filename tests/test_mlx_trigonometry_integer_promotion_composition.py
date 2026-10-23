"""Forward ufunc result width after transient half plus integer-array promotion."""
import numpy as np
import pytest
from native_execution import require_metal
from component_execution import assert_numpy_component_modes


@pytest.mark.metal
@pytest.mark.parametrize('integer_dtype',[np.int8,np.int16])
def test_actual_forward_half_integer_array_promotion(monkeypatch,tmp_path,integer_dtype):
    import brian2 as b
    from mlx_backend.model import resolve_model
    mx = require_metal('Metal required')
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float64)
    monkeypatch.setitem(b.prefs,'core.default_integer_dtype',integer_dtype)
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',False)
    model=resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\nflag:boolean\nrhs:integer\nsample:1',
        'eq_th':'True','eq_rst':'sample=cos(sin(flag)+rhs)','t_rfc':0*b.ms})
    flags=np.array([False,True,False,True,True])
    rhs=np.array([-2,-1,0,1,2],integer_dtype)
    assert_numpy_component_modes(monkeypatch,tmp_path,mx,model,{'flag':flags,'rhs':rhs},
        prefix='forward_integer_promotion',cpu_error='Native forward promotion used CPU timesteps')
