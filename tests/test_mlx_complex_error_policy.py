"""Complex ufunc floating policy at the original Eon execution boundary."""
from native_execution import forbid_cpu_code_objects
from contextlib import nullcontext
import warnings
import numpy as np
import pytest

pytestmark = pytest.mark.metal


@pytest.fixture(autouse=True)
def _evaluate_native_arrays_before_buffer_export(monkeypatch):
    # A deferred Metal compile exception cannot cross NumPy's buffer callback.
    # Evaluate first so pytest retains the original native diagnostic.
    import mlx.core as mx
    original=np.asarray
    def checked(value,*args,**kwargs):
        if isinstance(value,mx.array):mx.eval(value)
        return original(value,*args,**kwargs)
    monkeypatch.setattr(np,'asarray',checked)


def test_complex_exp_overflow_policy_prefix(tmp_path):
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from original_api_reference import (
        assert_rng_state_equal,assert_snapshot_equal,build_original_api_fixture,
        capture_rng_state,cpu_runtime,load_original_model,
    )
    pytest.importorskip('mlx.core')
    expected=None
    for native in (False,True):
        with cpu_runtime():
            b.prefs.codegen.target='numpy'
            b.prefs.codegen.loop_invariant_optimisations=False
            original=load_original_model()
            parameters=dict(original.default_params)
            parameters.update(eqs=parameters['eqs']+'\nu:1\nx:1\n',eq_th='True',
                t_rfc=0*b.ms,w_syn=0*b.mV,
                eq_rst=parameters['eq_rst']+';z=((-1.)**.3)*u;x=exp(z)')
            with original_api(workers=False) if native else nullcontext():
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=parameters,
                    exc=(),exc2=(),silenced=(),initial_state={'u':np.full(4,1500.)})
                failure=None
                with warnings.catch_warnings(),np.errstate(over='raise',divide='ignore',under='ignore',invalid='ignore'):
                    warnings.simplefilter('ignore',np.ComplexWarning)
                    try:fixture.run_ms(.1)
                    except Exception as error:
                        while error.__cause__ is not None:error=error.__cause__
                        failure=(type(error).__name__,str(error))
                actual=fixture.capture();rng=capture_rng_state()
                if native:
                    assert failure==expected_failure
                    assert_snapshot_equal(actual,expected)
                    assert_rng_state_equal(rng,expected_rng)
                else:
                    expected,expected_rng,expected_failure=actual,rng,failure
                    assert failure==('FloatingPointError','overflow encountered in exp')


def _case(tmp_path,b,expression,*,selected=True,u=1.):
    from original_api_reference import build_original_api_fixture,load_original_model
    original=load_original_model()
    parameters=dict(original.default_params)
    parameters.update(eqs=parameters['eqs']+'\nu:1\nx:1\nr:1\n',eq_th=str(selected),
        t_rfc=0*b.ms,w_syn=0*b.mV,
        eq_rst=parameters['eq_rst']+';r=rand();z=((-1.)**.3)*u;x='+expression)
    return build_original_api_fixture(tmp_path,parameters=parameters,exc=(),exc2=(),silenced=(),
        initial_state={'u':np.full(4,u)})


@pytest.mark.parametrize('function',['int(z)','timestep(z*ms,dt)'])
def test_complex_integer_warning_precedes_invalid_policy(tmp_path,function):
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from original_api_reference import (cpu_runtime,capture_rng_state,assert_rng_state_equal,
        assert_snapshot_equal)
    for native in (False,True):
        with cpu_runtime():
            b.prefs.codegen.target='numpy'
            b.prefs.codegen.loop_invariant_optimisations=False
            with original_api(workers=False) if native else nullcontext():
                fixture=_case(tmp_path/str(native),b,function,u=1e30)
                failure=None
                with warnings.catch_warnings(record=True) as captured,np.errstate(
                        invalid='raise',over='ignore',divide='ignore',under='ignore'):
                    warnings.simplefilter('always',np.ComplexWarning)
                    try:fixture.run_ms(.1)
                    except Exception as error:
                        while error.__cause__ is not None:error=error.__cause__
                        failure=(type(error).__name__,str(error))
                actual=fixture.capture();rng=capture_rng_state()
                events=[(type(w.message).__name__,str(w.message),w.filename,w.lineno) for w in captured]
                if native:
                    if failure!=expected_failure:pytest.fail(repr(failure))
                    assert events==expected_events
                    assert_snapshot_equal(actual,expected)
                    assert_rng_state_equal(rng,expected_rng)
                else:
                    expected,expected_rng,expected_failure,expected_events=actual,rng,failure,events
                    assert failure==('FloatingPointError','invalid value encountered in cast')
                    assert len(events)==1 and events[0][0]=='ComplexWarning'


def test_complex_scalar_conversion_runs_on_empty_reset(tmp_path,monkeypatch):
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from original_api_reference import (cpu_runtime,capture_rng_state,assert_rng_state_equal,
        assert_snapshot_equal)
    for native in (False,True):
        with cpu_runtime():
            b.prefs.codegen.target='numpy'
            b.prefs.codegen.loop_invariant_optimisations=False
            with original_api(workers=False) if native else nullcontext():
                fixture=_case(tmp_path/str(native),b,'int(((-1.)**.3)*(t/ms))',selected=False)
                if native:
                    forbid_cpu_code_objects(monkeypatch, 'Native empty complex conversion used a CPU timestep')
                failure=None
                with warnings.catch_warnings(record=True) as captured,np.errstate(all='ignore'):
                    warnings.simplefilter('error',np.ComplexWarning)
                    try:fixture.run_ms(.1)
                    except Exception as error:
                        while error.__cause__ is not None:error=error.__cause__
                        failure=(type(error).__name__,str(error))
                actual=fixture.capture();rng=capture_rng_state()
                assert not captured
                if native:
                    assert failure==expected_failure
                    assert_snapshot_equal(actual,expected)
                    assert_rng_state_equal(rng,expected_rng)
                else:
                    expected,expected_rng,expected_failure=actual,rng,failure
                    assert failure==('TypeError',
                        "int() argument must be a string, a bytes-like object or a real number, not 'complex'")


def test_complex_divide_raise_and_native_continuation(tmp_path):
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from original_api_reference import (cpu_runtime,capture_rng_state,assert_rng_state_equal,
        assert_snapshot_equal)
    for native in (False,True):
        with cpu_runtime():
            b.prefs.codegen.target='numpy'
            b.prefs.codegen.loop_invariant_optimisations=False
            with original_api(workers=False) if native else nullcontext():
                fixture=_case(tmp_path/str(native),b,'z/(u-u)')
                failure=None
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',np.ComplexWarning)
                    with np.errstate(divide='raise',over='ignore',under='ignore',invalid='ignore'):
                        try:fixture.run_ms(.1)
                        except Exception as error:
                            while error.__cause__ is not None:error=error.__cause__
                            failure=(type(error).__name__,str(error))
                    failed=fixture.capture();failed_rng=capture_rng_state()
                    with np.errstate(all='ignore'):repaired=fixture.run_ms(.1)
                rng=capture_rng_state()
                if native:
                    assert failure==expected_failure
                    assert_snapshot_equal(failed,expected_failed)
                    assert_rng_state_equal(failed_rng,expected_failed_rng)
                    assert_snapshot_equal(repaired,expected_repaired)
                    assert_rng_state_equal(rng,expected_rng)
                else:
                    expected_failure,expected_failed,expected_failed_rng=failure,failed,failed_rng
                    expected_repaired,expected_rng=repaired,rng
                    assert failure==('FloatingPointError','divide by zero encountered in divide')


@pytest.mark.parametrize('mode',['always','error'])
def test_complex128_float32_store_warning_precedes_overflow(tmp_path,monkeypatch,mode):
    """Independent Eon cast ordering, including warning-as-error before store."""
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from original_api_reference import (
        assert_rng_state_equal,assert_snapshot_equal,build_original_api_fixture,
        capture_rng_state,cpu_runtime,load_original_model,
    )
    pytest.importorskip('mlx.core')
    for native in (False,True):
        with cpu_runtime():
            b.prefs.codegen.target='numpy'
            b.prefs.codegen.loop_invariant_optimisations=False
            b.prefs.core.default_float_dtype=np.float32
            original=load_original_model()
            parameters=dict(original.default_params)
            # The integer-index array anchors complex128 with float32 state.
            parameters.update(eqs=parameters['eqs']+'\nx:1\nr:1\n',eq_th='True',
                t_rfc=0*b.ms,w_syn=0*b.mV,
                eq_rst=parameters['eq_rst']+';r=rand();z=((-1.)**.3)*(i+1)*1e100;x=z')
            with original_api(workers=False) if native else nullcontext():
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=parameters,
                    exc=(),exc2=(),silenced=(),initial_state={'x':[7.]*4})
                if native:
                    forbid_cpu_code_objects(monkeypatch, 'Native complex persistent cast used a CPU timestep')
                failure=None
                with warnings.catch_warnings(record=True) as captured,np.errstate(
                        over='raise',divide='ignore',under='ignore',invalid='ignore'):
                    warnings.simplefilter(mode,np.ComplexWarning)
                    try:fixture.run_ms(.1)
                    except Exception as error:
                        while error.__cause__ is not None:error=error.__cause__
                        failure=(type(error).__name__,str(error))
                actual=fixture.capture();rng=capture_rng_state()
                events=[(type(w.message).__name__,str(w.message),w.filename,w.lineno) for w in captured]
                if native:
                    assert failure==expected_failure
                    assert events==expected_events
                    assert_snapshot_equal(actual,expected)
                    assert_rng_state_equal(rng,expected_rng)
                else:
                    expected,expected_rng,expected_failure,expected_events=actual,rng,failure,events
                    if mode=='error':
                        assert failure[0]=='ComplexWarning'
                        assert events==[]
                        np.testing.assert_array_equal(actual.state['x'],[7.]*4)
                    else:
                        assert failure==('FloatingPointError','overflow encountered in cast')
                        assert len(events)==1 and events[0][0]=='ComplexWarning'
