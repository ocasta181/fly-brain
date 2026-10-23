"""Actual Eon array casts, warning sites/filter behavior and failure prefixes."""
from contextlib import nullcontext
import warnings

import numpy as np
import pytest


@pytest.mark.metal
@pytest.mark.parametrize('selected',[True,False])
@pytest.mark.parametrize('mode',['always','default','error'])
def test_original_complex_cast_warning_boundaries(tmp_path,mode,selected):
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from original_api_reference import (
        assert_rng_state_equal,assert_snapshot_equal,build_original_api_fixture,
        capture_rng_state,cpu_runtime,load_original_model,
    )
    pytest.importorskip('mlx.core')
    expected=None
    for native in (False,True):
        with cpu_runtime(target='numpy'):
            b.prefs.codegen.loop_invariant_optimisations=False
            original=load_original_model()
            parameters=dict(original.default_params)
            parameters.update(eqs=parameters['eqs']+'\nu:1\na:1\nbb:1\nr:1\n',
                eq_th=str(selected),t_rfc=0*b.ms,w_syn=0*b.mV,
                eq_rst=parameters['eq_rst']+';r=rand();z=((-1.)**.3)*u;a=z;bb=z')
            with original_api(workers=False) if native else nullcontext():
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=parameters,
                    exc=(),exc2=(),silenced=(),initial_state={'u':np.zeros(4)})
                failure=None
                with warnings.catch_warnings(record=True) as captured,np.errstate(all='ignore'):
                    warnings.simplefilter('ignore')
                    warnings.simplefilter(mode,np.ComplexWarning)
                    try:fixture.run_ms(.2)
                    except Exception as error:
                        while error.__cause__ is not None:error=error.__cause__
                        failure=(type(error).__name__,str(error))
                events=[(type(w.message).__name__,str(w.message),w.filename,w.lineno) for w in captured]
                actual=fixture.capture();rng=capture_rng_state()
                if native:
                    assert failure==expected_failure
                    assert events==expected_events
                    assert_snapshot_equal(actual,expected)
                    assert_rng_state_equal(rng,expected_rng)
                else:
                    expected,expected_rng,expected_events,expected_failure=actual,rng,events,failure
                    if mode=='error':assert failure[0]=='ComplexWarning'
                    else:assert len(events)==(4 if mode=='always' else 2)


@pytest.mark.metal
@pytest.mark.parametrize('function',['int(z)','timestep(z*ms,dt)'])
@pytest.mark.parametrize('mode',['always','default','error'])
def test_original_complex_function_cast_warning(tmp_path,function,mode):
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from original_api_reference import (
        assert_rng_state_equal,assert_snapshot_equal,build_original_api_fixture,
        capture_rng_state,cpu_runtime,load_original_model,
    )
    pytest.importorskip('mlx.core')
    for native in (False,True):
        with cpu_runtime(target='numpy'):
            b.prefs.codegen.loop_invariant_optimisations=False
            original=load_original_model()
            parameters=dict(original.default_params)
            parameters.update(eqs=parameters['eqs']+'\nu:1\nx:1\nr:1\n',eq_th='True',
                t_rfc=0*b.ms,w_syn=0*b.mV,
                eq_rst=parameters['eq_rst']+';r=rand();z=((-1.)**.3)*u;x='+function)
            with original_api(workers=False) if native else nullcontext():
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=parameters,
                    exc=(),exc2=(),silenced=(),initial_state={'u':np.zeros(4)})
                failure=None
                with warnings.catch_warnings(record=True) as captured,np.errstate(all='ignore'):
                    warnings.simplefilter('ignore')
                    warnings.simplefilter(mode,np.ComplexWarning)
                    try:fixture.run_ms(.2)
                    except Exception as error:
                        while error.__cause__ is not None:error=error.__cause__
                        failure=(type(error).__name__,str(error))
                events=[(type(w.message).__name__,str(w.message),w.filename,w.lineno) for w in captured]
                actual=fixture.capture();rng=capture_rng_state()
                if native:
                    assert failure==expected_failure
                    assert events==expected_events
                    assert_snapshot_equal(actual,expected)
                    assert_rng_state_equal(rng,expected_rng)
                else:
                    expected,expected_rng,expected_events,expected_failure=actual,rng,events,failure
                    if mode=='error':assert failure[0]=='ComplexWarning'
                    else:assert len(events)==(2 if mode=='always' else 1)
