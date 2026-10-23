"""Original Poisson literal weights preserve accepted IEEE arithmetic."""
import numpy as np
import pytest

from test_mlx_brian_device import _run_pair

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('target,scale,rate,dtype', [
    ('numpy', np.inf, 0, np.float64),
    ('cython', np.inf, 0, np.float64),
    ('numpy', -np.inf, 10000, np.float64),
    ('cython', -np.inf, 10000, np.float64),
    ('numpy', np.nan, 3700, np.float64),
    ('cython', np.nan, 3700, np.float64),
    ('numpy', np.inf, 0, np.float32),
    ('cython', np.inf, 0, np.float32),
])
def test_original_nonfinite_input_matches_cpu(tmp_path, target, scale, rate, dtype):
    import brian2 as b
    parameters = {"f_poi": scale, "r_poi": rate*b.Hz, "r_poi2": 0*b.Hz}
    if np.isnan(scale):
        # Brian has no built-in nan constant: the literal must resolve in
        # the actual target namespace for this to be an accepted CPU case.
        parameters['nan'] = np.nan
    _run_pair(tmp_path, [.15, .15], target=target, state_dtype=dtype,
              parameters=parameters,
              exc=(0,), exc2=())


def test_nonfinite_input_prepass_matches_cpu(tmp_path, monkeypatch):
    import brian2 as b
    from mlx_backend.brian_poisson import NativePoisson

    monkeypatch.setattr(NativePoisson, 'can_fuse', lambda self, prepared: False)
    _run_pair(tmp_path, [.15, .15], target='numpy',
              parameters={"f_poi": np.inf, "r_poi": 0*b.Hz, "r_poi2": 0*b.Hz},
              exc=(0,), exc2=())


@pytest.mark.parametrize('target', ['numpy', 'cython'])
def test_nan_weight_spelling_uses_actual_brian_namespace(tmp_path, target):
    import brian2 as b

    _run_pair(tmp_path, [.15, .15], target=target,
              parameters={"f_poi": np.nan, "nan": 2.,
                          "r_poi": 10000*b.Hz, "r_poi2": 0*b.Hz},
              exc=(0,), exc2=())


@pytest.mark.parametrize('rate', [-100, 12000])
def test_original_cython_outside_unit_probability_matches_cpu(tmp_path, rate):
    import brian2 as b

    _run_pair(tmp_path, [.15, .15], target='cython',
              parameters={"r_poi": rate*b.Hz, "r_poi2": 3700*b.Hz},
              exc=(0,), exc2=(2,))
