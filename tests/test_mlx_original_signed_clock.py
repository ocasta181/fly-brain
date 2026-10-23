"""Actual original Eon signed clocks and preserved queued continuation."""
import numpy as np
import pytest

b = pytest.importorskip("brian2")
from native_execution import require_metal
mx = require_metal('Metal is required', allow_module_level=True)

from mlx_backend.brian_device import original_api
from original_api_reference import (
    assert_rng_state_equal, assert_snapshot_equal, capture_rng_state, cpu_runtime, original_runtime,
    load_original_model,
)
from test_mlx_original_empty_clock import _run_original

pytestmark = pytest.mark.metal


def _unconditional_equations():
    # Keep real arriving conductance observable after a threshold event.
    return load_original_model().default_params["eqs"].replace(" (unless refractory)", "")


@pytest.mark.parametrize("delay_ms", [1.8, -.2, np.inf])
def test_original_negative_dt_exact_arrivals_and_continuation(tmp_path, delay_ms):
    params = {"eqs": _unconditional_equations(), "eq_rst": "v=v_rst",
              "v_0": -44*b.mV, "v_rst": -44*b.mV, "t_rfc": 20000*b.second,
              "t_dly": delay_ms*b.ms}
    with cpu_runtime(seed=173, dt_ms=-.1):
        expected, _ = _run_original(tmp_path/"cpu", duration_ms=-.2,
                                     parameters=params, edge_count=1.)
        wanted = [expected.capture(), expected.run_ms(-.2)]
        assert wanted[-1].spike_i.tolist() == [0, 0, 0, 0]
        if np.isfinite(delay_ms):
            assert wanted[0].state["g"][0] == 0 if delay_ms < 0 else wanted[0].state["g"][0] > 0
            assert wanted[-1].state["g"][0] > 0
        expected_rng = capture_rng_state()
    with original_runtime(original_api, seed=173, dt_ms=-.1) as device:
        actual, _ = _run_original(tmp_path/"native", duration_ms=-.2,
                                   parameters=params, edge_count=1.)
        observed = [actual.capture(), actual.run_ms(-.2)]
        for actual_state, expected_state in zip(observed, wanted):
            assert_snapshot_equal(actual_state, expected_state)
        assert_rng_state_equal(capture_rng_state(), expected_rng)
        assert device.native_steps == 4 and device.native_runs == 2
        assert device._last_run_completed_fraction == 1.


def test_original_negative_absolute_iteration_matches_cpu(tmp_path):
    params = {"eqs": _unconditional_equations(), "v_0": -44*b.mV}
    with cpu_runtime(seed=173, dt_ms=.1):
        expected, _ = _run_original(tmp_path/"cpu", duration_ms=-.2, parameters=params)
        wanted = expected.run_ms(.2)
        assert wanted.spike_i.tolist() == [0]
        assert wanted.spike_t_s.tolist() == [-.0002]
        expected_rng = capture_rng_state()
    with original_runtime(original_api, seed=173, dt_ms=.1) as device:
        actual, _ = _run_original(tmp_path/"native", duration_ms=-.2, parameters=params)
        assert_snapshot_equal(actual.run_ms(.2), wanted)
        assert_rng_state_equal(capture_rng_state(), expected_rng)
        assert device.native_steps == 2 and device.native_runs == 2


def test_original_rewind_keeps_live_delayed_arrival(tmp_path):
    params = {"eqs": _unconditional_equations(), "eq_rst": "v=v_rst",
              "v_0": -44*b.mV, "t_dly": .2*b.ms}
    def sequence(directory):
        fixture, _ = _run_original(directory, parameters=params, edge_count=1.)
        snapshots = [fixture.capture(), fixture.run_ms(-.2), fixture.run_ms(.3)]
        return fixture, snapshots
    with cpu_runtime(seed=173, dt_ms=.1):
        _, wanted = sequence(tmp_path/"cpu")
        assert wanted[0].state["g"][0] == wanted[1].state["g"][0] == 0
        assert wanted[2].state["g"][0] > 0
        expected_rng = capture_rng_state()
    with original_runtime(original_api, seed=173, dt_ms=.1) as device:
        _, observed = sequence(tmp_path/"native")
        for actual_state, expected_state in zip(observed, wanted):
            assert_snapshot_equal(actual_state, expected_state)
        assert_rng_state_equal(capture_rng_state(), expected_rng)
        assert device.native_steps == 5 and device.native_runs == 3
