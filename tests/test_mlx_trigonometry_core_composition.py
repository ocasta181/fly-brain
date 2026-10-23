"""Independent forward-ufunc composition with the shared alias/precision core."""
import numpy as np
import pytest
from native_execution import require_metal
from component_execution import assert_numpy_component_modes


@pytest.mark.metal
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('reset',[
    'x=sin(f);y=x;x+=cos(f);z=y',
    'x=f;tmp=sin(x);x=cos(x);y=tmp;f+=1;z=tmp',
    'tmp=sin(flag);y=tmp;tmp+=cos(flag);z=y;x=tan(tmp)',
])
def test_original_numpy_forward_array_alias_precision(monkeypatch,tmp_path,dtype,reset):
    import brian2 as b
    from mlx_backend.model import resolve_model
    mx = require_metal('Metal required')
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',dtype)
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',False)
    model=resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\nx:1\ny:1\nz:1\nf:1\nflag:boolean',
        'eq_th':'True','eq_rst':reset,'t_rfc':0*b.ms})
    values=np.array([.1,.3,-.5,3.,-2.],dtype)
    flags=np.array([False,True,False,True,True])
    assert_numpy_component_modes(monkeypatch,tmp_path,mx,model,{'f':values,'flag':flags},
        prefix='forward_alias',cpu_error='Native forward composition used CPU timesteps')
