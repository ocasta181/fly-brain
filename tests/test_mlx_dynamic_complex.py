"""Dynamic complex NumPy intermediates through the unchanged Eon factory."""
from native_execution import forbid_model_cpu_steps, forbid_cpu_code_objects
from contextlib import nullcontext

import numpy as np
import pytest

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('optimise', [False, True])
@pytest.mark.parametrize('reset', [
    'x=sin(((-1.)**.3)*(i+1))*100',
    'z=((-1.)**.3)*(i+1);alias=z;x=sin(alias)*100',
])
def test_original_dynamic_complex_integer_destination(tmp_path, monkeypatch, optimise, reset):
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from original_api_reference import (
        assert_rng_state_equal, assert_snapshot_equal, build_original_api_fixture,
        capture_rng_state, cpu_runtime, load_original_model,
    )

    pytest.importorskip('mlx.core')
    expected = None
    for native in (False, True):
        with cpu_runtime(target='numpy'):
            b.prefs.codegen.loop_invariant_optimisations = optimise
            original = load_original_model()
            parameters = dict(original.default_params)
            parameters.update(eqs=parameters['eqs']+'\nx:integer\n', eq_th='True',
                t_rfc=0*b.ms, w_syn=0*b.mV,
                eq_rst=parameters['eq_rst']+';'+reset)
            with original_api(workers=False) if native else nullcontext() as device:
                fixture = build_original_api_fixture(tmp_path/str(native),
                    parameters=parameters, exc=(), exc2=(), silenced=())
                if native:
                    forbid_model_cpu_steps(fixture, monkeypatch, 'CPU timestep in native complex execution', patch_preparation=True)
                with np.errstate(all='ignore'):
                    snapshots = [fixture.run_ms(value) for value in (0., .1, .1)]
                state = capture_rng_state()
                if native:
                    for actual, wanted in zip(snapshots, expected):
                        assert_snapshot_equal(actual, wanted)
                    assert_rng_state_equal(state, expected_rng)
                    assert device.native_steps == 2
                else:
                    expected, expected_rng = snapshots, state
                    np.testing.assert_array_equal(snapshots[1].state['x'], [74,241,560,905])


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_complex_functions_and_component_stores(tmp_path, dtype):
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from original_api_reference import (
        assert_snapshot_equal, build_original_api_fixture, cpu_runtime,
        load_original_model,
    )

    pytest.importorskip('mlx.core')
    fields = ('xs', 'xc', 'xsh', 'xch', 'xa', 'xm', 'xd')
    expected = None
    for native in (False, True):
        with cpu_runtime(target='numpy'):
            b.prefs.codegen.loop_invariant_optimisations = False
            b.prefs.core.default_float_dtype = dtype
            original = load_original_model()
            parameters = dict(original.default_params)
            parameters.update(eqs=parameters['eqs']+'\n'+
                '\n'.join(name+':1' for name in fields)+'\nu:1\nflag:boolean\n',
                eq_th='True', t_rfc=0*b.ms, w_syn=0*b.mV,
                eq_rst=parameters['eq_rst']+
                ';z=((-1.)**.3)*(u+1);alias=z;'
                'xs=sin(alias);xc=cos(z);xsh=sinh(z);xch=cosh(z);'
                'xa=z+(i+1);xm=z*z;xd=z/(i+1);flag=(z!=0)')
            with original_api(workers=False) if native else nullcontext():
                fixture = build_original_api_fixture(tmp_path/str(native),
                    parameters=parameters, exc=(), exc2=(), silenced=(),
                    initial_state={'u': np.array([0., .25, -.125, .5], dtype)})
                with np.errstate(all='ignore'):
                    actual = fixture.run_ms(.1)
                if native:
                    assert_snapshot_equal(actual, expected)
                else:
                    expected = actual


def _check_complex_expressions(tmp_path, dtype, expressions):
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from original_api_reference import (
        assert_snapshot_equal,build_original_api_fixture,cpu_runtime,load_original_model,
    )
    pytest.importorskip('mlx.core')
    expected = None
    for native in (False,True):
        with cpu_runtime(target='numpy'):
            b.prefs.codegen.loop_invariant_optimisations=False
            b.prefs.core.default_float_dtype=dtype
            original=load_original_model()
            parameters=dict(original.default_params)
            parameters.update(eqs=parameters['eqs']+'\nu:1\n'+
                '\n'.join(name+':1' for name in expressions),eq_th='True',
                t_rfc=0*b.ms,w_syn=0*b.mV,
                eq_rst=parameters['eq_rst']+';z=((-1.)**.3)*(u+1);'+
                ';'.join(name+'='+expression for name,expression in expressions.items()))
            with original_api(workers=False) if native else nullcontext():
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=parameters,
                    exc=(),exc2=(),silenced=(),initial_state={'u':np.array([0.,.25,-.125,.5],dtype)})
                with np.errstate(all='ignore'):
                    actual=fixture.run_ms(.1)
                if native:assert_snapshot_equal(actual,expected)
                else:expected=actual


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_complex_elementary_functions_and_conversions(tmp_path,dtype):
    expressions = dict(ce='exp(z)',cr='sqrt(z)',ca='abs(z)',cs='sign(z)',
        ci='int(z)',cc='clip(z,-1,1)',ct='timestep(z*ms,dt)',cp='z**2')
    _check_complex_expressions(tmp_path, dtype, expressions)


@pytest.mark.parametrize('dtype', [np.float64,np.float32])
def test_complex_expm1_exprel_and_tangents(tmp_path,dtype):
    expressions=dict(em='expm1(z)',er='exprel(z)',th='tanh(z)',tt='tan(z)')
    _check_complex_expressions(tmp_path, dtype, expressions)


def test_complex_local_survives_random_segments(tmp_path):
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
            parameters.update(eqs=parameters['eqs']+'\nu:1\nx:1\ny:1\n',eq_th='True',
                t_rfc=0*b.ms,w_syn=0*b.mV,
                eq_rst=parameters['eq_rst']+';z=((-1.)**.3)*(u+1);'
                    'r=rand();x=sin(z+r);r2=randn();y=cos(z+r2)')
            with original_api(workers=False) if native else nullcontext() as device:
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=parameters,
                    exc=(),exc2=(),silenced=(),initial_state={'u':np.array([0.,.25,-.125,.5])})
                with np.errstate(all='ignore'):
                    actual=[fixture.run_ms(.1),fixture.run_ms(.1)]
                state=capture_rng_state()
                if native:
                    for got,wanted in zip(actual,expected):assert_snapshot_equal(got,wanted)
                    assert_rng_state_equal(state,expected_rng)
                    assert device.native_steps==2
                else:expected,expected_rng=actual,state


@pytest.mark.parametrize('dtype', [np.float64,np.float32])
def test_complex_logarithms_and_fractional_power(tmp_path,dtype):
    expressions=dict(cl='log(z)',cl1='log1p(z)',cl10='log10(z)',cp='z**.3',
                     cpc='sin(z**(.3+u*.1))')
    _check_complex_expressions(tmp_path, dtype, expressions)


@pytest.mark.parametrize('dtype', [np.float64,np.float32])
def test_complex_inverse_functions(tmp_path,dtype):
    expressions=dict(cas='arcsin(z)',cac='arccos(z)',cat='arctan(z)',nested='sin(arccos(z))')
    _check_complex_expressions(tmp_path, dtype, expressions)


@pytest.mark.parametrize('function',['floor','ceil'])
@pytest.mark.parametrize('random_prefix,selected',[
    (False,True),(True,True),(False,False),
],ids=['selected-no-rng','selected-rng','empty'])
def test_complex_rejected_ufunc_preserves_failure_prefix(tmp_path,monkeypatch,function,random_prefix,selected):
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
            parameters.update(eqs=parameters['eqs']+'\nx:1\ny:1\n',eq_th=str(selected),
                t_rfc=0*b.ms,w_syn=0*b.mV,
                eq_rst=parameters['eq_rst']+';x='+('rand()' if random_prefix else 'i+1')+
                    ';z=((-1.)**.3)*(i+1);y='+function+'(z);x=99')
            with original_api(workers=False) if native else nullcontext():
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=parameters,
                    exc=(),exc2=(),silenced=())
                fixture.run_ms(0.)
                if native:
                    forbid_cpu_code_objects(monkeypatch, 'Native complex ufunc rejection used a CPU timestep')
                with pytest.raises(Exception) as caught:
                    fixture.run_ms(.1)
                error=caught.value
                while error.__cause__ is not None:error=error.__cause__
                observed=(type(error).__name__,str(error))
                assert observed[0]=='TypeError'
                actual=fixture.capture();state=capture_rng_state()
                if native:
                    assert observed==expected_error
                    assert_snapshot_equal(actual,expected)
                    assert_rng_state_equal(state,expected_rng)
                else:expected,expected_rng,expected_error=actual,state,observed
