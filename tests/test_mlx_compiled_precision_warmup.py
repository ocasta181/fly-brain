"""A preparation probe cannot invent an error absent with the actual inputs."""
from dataclasses import replace
import brian2 as b
import numpy as np
import pytest
from mlx_backend import configured
from mlx_backend.model import DEFAULT_EQUATIONS,DEFAULT_RESET,resolve_model
from component_cpu_reference import component_inputs
from configured_reference import reference
from test_mlx_compiled_precision_configured import check


@pytest.mark.metal
@pytest.mark.parametrize('event',[0,1])
@pytest.mark.parametrize('checked',[False,True])
def test_checked_warmup_cannot_raise_from_synthetic_zero_inputs(event,checked,monkeypatch,tmp_path):
    monkeypatch.setitem(b.prefs,'codegen.target','cython')
    monkeypatch.setitem(b.prefs,'core.default_integer_dtype',np.int64)
    monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(tmp_path/'cpu'))
    reset=(('seen=(abs(x)**p-abs(y)**q)/int(v>v_0)' if checked else 'seen=1/int(v>v_0);x+=1')
           +';'+DEFAULT_RESET+';p+=1')
    model=resolve_model(dict(eqs=DEFAULT_EQUATIONS.replace('(unless refractory)','')+'x:integer\ny:integer\np:integer\nq:integer\nseen:1\n',
        eq_th='True',eq_rst=reset,v_0=-50*b.mV,v_rst=-50*b.mV,t_rfc=0*b.ms),cpu_target='cython')
    graph,stimulus=component_inputs(2,4,model)
    stimulus=replace(stimulus,events=np.full((1,4,2),event,np.uint8),targets=(0,1),rates_hz=(1.,1.),
        amplitudes_mv=(1.,1.),sha256=None)
    if not event:
        with pytest.raises(b.BrianObjectException) as failure:reference(graph,stimulus,model,cpu_target='cython')
        cause=failure.value.__cause__;assert isinstance(cause,ZeroDivisionError)
        prepared=configured.prepare(graph,stimulus,model)
        with pytest.raises(type(cause)) as failure:configured.run_prepared(prepared,stimulus)
        assert str(failure.value)==str(cause)
        return
    expected=reference(graph,stimulus,model,cpu_target='cython')
    prepared=configured.prepare(graph,stimulus,model)
    assert bool(prepared.program.native_precision_limbs)==checked
    result,state=configured.run_prepared(prepared,stimulus,return_state=True)
    check(configured.host_state(state,prepared),expected.state)
    for name in ('trials','neurons','steps'):
        np.testing.assert_array_equal(getattr(result.spikes,name),getattr(expected.spikes,name))
