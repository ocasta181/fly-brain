"""A constant-per-step random subexpression drives a custom linear state."""

import numpy as np
import pytest


@pytest.mark.metal
@pytest.mark.parametrize('target', ['numpy', 'cython'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64], ids=['float32', 'float64'])
def test_original_random_subexpression_drives_linear_state(tmp_path, monkeypatch, target, dtype):
    import brian2 as b
    from original_api_reference import build_original_api_fixture, capture_rng_state
    from original_api_protocol import RunObservation, check_original_api_runs

    random_values=[]
    def build(mode):
        fixture = build_original_api_fixture(
            tmp_path / str(mode=='native'), parameters=dict(
                eqs='v:volt\ng:volt\nrfc:second\nu=rand():1 (constant over dt)\n'
                    'dx/dt=-(x+u)/(.7*ms):1',
                eq_th='False', eq_rst='v=v_rst', t_rfc=0*b.ms,
            ), initial_state={'x': [-.75, -.125, .125, .75]},
            exc=(), exc2=(), silenced=(),
        )
        return fixture

    def exercise(fixture, mode):
        for index, duration in enumerate((0., .1, .2)):
            fixture.run_ms(duration)
            snapshot=fixture.capture()
            random_value=np.array(fixture.neurons.variables['u'].get_value(),copy=True)
            yield RunObservation(snapshot,rng=capture_rng_state())
            if mode=='cpu':random_values.append(random_value)
            else:
                assert random_value.dtype == random_values[index].dtype
                assert random_value.tobytes() == random_values[index].tobytes()
    check_original_api_runs(build,exercise,monkeypatch,seed=963,target=target,float_dtype=dtype)
