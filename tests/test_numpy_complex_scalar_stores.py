"""Actual NumPy object provenance before real/integer destination stores."""
from native_execution import forbid_model_cpu_steps
from contextlib import nullcontext

import numpy as np
import pytest

pytestmark = pytest.mark.metal


def _exercise(tmp_path,monkeypatch,*,dtype,destination,spikes,expression,optimise=False,alias=False,alias_python=False,integer_dtype=np.int8,alias_rebind=False,temporary='tmp'):
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from mlx_backend.model import DEFAULT_EQUATIONS
    from original_api_reference import (build_original_api_fixture,cpu_runtime,
        assert_snapshot_equal,capture_rng_state,assert_rng_state_equal)
    expected=[];rng=[];cause=None;repaired_expected=None
    for native in (False,True):
        with cpu_runtime():
            b.prefs.codegen.target='numpy';b.prefs.core.default_integer_dtype=integer_dtype
            b.prefs.core.default_float_dtype=dtype
            monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',optimise)
            assign=(('tmp=(-1.)**.3;alias=tmp;x=alias' if alias_python else 'tmp=(-1.)**.3;alias=tmp;x=sin(alias)*100') if alias else 'x='+expression+('+0.0' if optimise else ''))
            if alias_rebind:assign='tmp=(-1.)**.3;x=sin(tmp)*100;tmp=i+1;x=tmp'
            assign=assign.replace('tmp',temporary)
            reset='v=v_rst;a_before=17;'+assign+';z_after=19'
            params=dict(eqs=DEFAULT_EQUATIONS+f'x:{destination}\na_before:integer\nz_after:integer\n',
                eq_th=str(spikes),eq_rst=reset,t_rfc=0*b.ms,r_poi=4900*b.Hz,r_poi2=3200*b.Hz)
            with original_api(workers=False) if native else nullcontext():
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=params,
                    initial_state=dict(x=[7]*4,a_before=[8]*4,z_after=[9]*4),
                    exc=(0,0),exc2=(2,),silenced=())
                assert fixture.neurons.variables['v'].dtype == np.dtype(dtype)
                assert fixture.neurons.variables['x'].dtype == np.dtype(
                    integer_dtype if destination=='integer' else bool if destination=='boolean' else dtype)
                if native:
                    forbid_model_cpu_steps(fixture, monkeypatch, 'Native complex scalar used CPU timestep', patch_preparation=True)
                # The optimizer's rejection precedes even a zero-duration run.
                calls=(.1,) if optimise else (0.,.15,.15)
                error=None;observed=[]
                for duration in calls:
                    try:observed.append(fixture.run_ms(duration))
                    except b.BrianObjectException as exc:
                        error=(type(exc.__cause__),str(exc.__cause__))
                        observed.append(fixture.capture());break
                if alias_rebind:
                    assert error is not None
                    assert error[0] is SyntaxError
                elif optimise or destination!='boolean' and (alias_python or expression=='(-1.)**.3' and not alias):
                    assert error is not None
                    assert error[0] is TypeError
                else:assert error is None
                if native:
                    assert error==cause
                    for actual,wanted in zip(observed,expected):assert_snapshot_equal(actual,wanted)
                    assert len(observed)==len(expected)
                    assert_rng_state_equal(capture_rng_state(),rng[0])
                else:
                    cause=error;expected=observed;rng.append(capture_rng_state())
                    if fixture.neurons.resetter['spike'].codeobj is not None:
                        (tmp_path/'actual-reset.py').write_text(str(fixture.neurons.resetter['spike'].codeobj.code.run))
                if error is not None and not optimise and not alias_rebind:
                    fixture.neurons.event_codes['spike']='v=v_rst;a_before=17;x=3.5;z_after=19'
                    repaired=fixture.run_ms(.15)
                    if native:
                        assert_snapshot_equal(repaired,repaired_expected);assert_rng_state_equal(capture_rng_state(),rng[1])
                    else:repaired_expected=repaired;rng.append(capture_rng_state())


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('destination',['integer','1'])
@pytest.mark.parametrize('spikes',[False,True])
def test_python_complex_scalar_store_error_and_repaired_continuation(tmp_path,monkeypatch,dtype,destination,spikes):
    _exercise(tmp_path,monkeypatch,dtype=dtype,destination=destination,spikes=spikes,expression='(-1.)**.3')


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('destination',['integer','1'])
def test_numpy_complex_scalar_projects_at_actual_destination_store(tmp_path,monkeypatch,dtype,destination):
    _exercise(tmp_path,monkeypatch,dtype=dtype,destination=destination,spikes=True,expression='sin((-1.)**.3)*100')


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('temporary',['tmp','thread'])
def test_immutable_complex_scalar_alias_keeps_original_object(tmp_path,monkeypatch,dtype,temporary):
    _exercise(tmp_path,monkeypatch,dtype=dtype,destination='1',spikes=True,expression='alias',alias=True,temporary=temporary)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('expression',['(-1.)**.3','sin((-1.)**.3)*100'])
def test_default_optimizer_rejects_complex_before_execution(tmp_path,monkeypatch,dtype,expression):
    _exercise(tmp_path,monkeypatch,dtype=dtype,destination='integer',spikes=True,expression=expression,optimise=True)


@pytest.mark.parametrize('integer_dtype',[np.int8,np.uint64])
def test_numpy_complex_scalar_large_real_part_uses_actual_integer_cast(tmp_path,monkeypatch,integer_dtype):
    _exercise(tmp_path,monkeypatch,dtype=np.float64,destination='integer',spikes=True,
              expression='sin((-1.)**.3)*1e100',integer_dtype=integer_dtype)


@pytest.mark.parametrize('destination',['integer','1'])
def test_python_complex_alias_retains_store_error(tmp_path,monkeypatch,destination):
    _exercise(tmp_path,monkeypatch,dtype=np.float64,destination=destination,spikes=True,
              expression='alias',alias=True,alias_python=True)


@pytest.mark.parametrize('expression',[
    '((-1.)**.5)-6.123233995736766e-17',
    'sin((-1.)**.5)-9.44864380126377e-17'])
def test_complex_scalar_boolean_store_observes_imaginary_component(tmp_path,monkeypatch,expression):
    _exercise(tmp_path,monkeypatch,dtype=np.float64,destination='boolean',spikes=True,expression=expression)


def test_reference_rejects_scalar_alias_rebound_after_vector_write(tmp_path,monkeypatch):
    _exercise(tmp_path,monkeypatch,dtype=np.float64,destination='1',spikes=True,
              expression='alias-rebind',alias_rebind=True)
