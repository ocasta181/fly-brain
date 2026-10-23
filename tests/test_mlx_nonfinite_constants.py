"""Actual CPU acceptance and IEEE behavior of original model constants."""

import numpy as np
import pytest

from test_mlx_brian_device import _run_pair


@pytest.mark.metal
@pytest.mark.parametrize("target,constant", [
    ("numpy", "v_0"), ("cython", "w_syn"),
    ("numpy", "v_rst"), ("cython", "v_rst"),
])
def test_infinite_original_resting_and_synaptic_constants(tmp_path, target, constant):
    import brian2 as b

    initial = ({"v": np.array([-44., -49., -48., -50.])*b.mV,
                "g": np.array([3., -2., 4., 1.])*b.mV} if constant == "v_rst" else None)
    _run_pair(tmp_path, [.15, 0., .35, 2.1], target=target,
              parameters={constant: np.inf*b.mV, "r_poi": 3700*b.Hz,
                          "r_poi2": 1400*b.Hz}, initial_state=initial)


@pytest.mark.metal
@pytest.mark.parametrize("target,constant,dtype", [
    ("numpy", "w_syn", np.float64), ("cython", "v_0", np.float64),
    ("numpy", "v_rst", np.float32), ("cython", "w_syn", np.float32),
])
def test_negative_infinite_voltage_constants_and_float32(tmp_path, target, constant, dtype):
    import brian2 as b

    initial = ({"v": np.array([-44., -49., -48., -50.])*b.mV,
                "g": np.array([3., -2., 4., 1.])*b.mV} if constant == "v_rst" else None)
    _run_pair(tmp_path, [.15, 0., .35, 2.1], target=target, state_dtype=dtype,
              parameters={constant: -np.inf*b.mV, "r_poi": 3700*b.Hz,
                          "r_poi2": 1400*b.Hz}, initial_state=initial)
