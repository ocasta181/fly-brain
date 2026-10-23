"""Timestep-derived Python complex provenance through the original Eon API.

These pending cases are deliberately separate from NumPy complex qualification.
"""
from native_execution import forbid_cpu_code_objects
from contextlib import nullcontext
import warnings

import numpy as np
import pytest


CASES={
    'division':('r=rand();z=((-1.)**.3)*timestep(t,dt);x=sin(z/2)',.2,None),
    'power':('r=rand();z=((-1.)**.3)*timestep(t,dt);x=sin(z**.3)',.2,None),
    'store':('r=rand();z=((-1.)**.3)*timestep(t,dt);x=z',.1,
        ('TypeError',"float() argument must be a string or a real number, not 'complex'")),
    'zero_division':('z=((-1.)**.3)*timestep(t,dt);q=1/z;r=rand();x=sin(q)',.1,
        ('ZeroDivisionError','complex division by zero')),
    'zero_power':('z=((-1.)**.3)*timestep(t,dt);q=z**-1;r=rand();x=sin(q)',.1,
        ('ZeroDivisionError','0.0 to a negative or complex power')),
}


@pytest.mark.metal
@pytest.mark.parametrize('case',tuple(CASES))
def test_original_timestep_python_complex(tmp_path,monkeypatch,case):
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from original_api_reference import (assert_rng_state_equal,assert_snapshot_equal,
        build_original_api_fixture,capture_rng_state,cpu_runtime,load_original_model)
    pytest.importorskip('mlx.core')
    code,duration,required=CASES[case]
    for native in (False,True):
        with cpu_runtime(target='numpy'):
            b.prefs.codegen.loop_invariant_optimisations=False
            original=load_original_model()
            parameters=dict(original.default_params)
            parameters.update(eqs=parameters['eqs']+'\nx:1\nr:1\n',eq_th='True',
                t_rfc=0*b.ms,w_syn=0*b.mV,eq_rst=parameters['eq_rst']+';'+code)
            with original_api(workers=False) if native else nullcontext():
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=parameters,
                    exc=(),exc2=(),silenced=())
                if native:
                    forbid_cpu_code_objects(monkeypatch, 'Native Python complex expression used a CPU timestep')
                failure=None
                with warnings.catch_warnings(),np.errstate(all='ignore'):
                    warnings.simplefilter('ignore',np.ComplexWarning)
                    try:fixture.run_ms(duration)
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
                    assert failure==required
