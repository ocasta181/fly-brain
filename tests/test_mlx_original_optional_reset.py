"""Original Eon explicit None reset and inert optional metadata parity."""
import numpy as np
import pytest

b = pytest.importorskip("brian2")
from native_execution import require_metal
mx = require_metal('Metal is required', allow_module_level=True)

from mlx_backend.brian_device import original_api
from original_api_reference import (
    assert_rng_state_equal, assert_snapshot_equal, build_original_api_fixture,
    capture_rng_state, cpu_runtime, original_runtime,
)
from test_mlx_original_empty_clock import _run_original

pytestmark = pytest.mark.metal


@pytest.mark.parametrize("target,dtype", [("numpy", np.float64), ("numpy", np.float32),
                                         ("cython", np.float64)])
def test_original_none_reset_preserves_spikes_state_and_continuation(
        tmp_path, monkeypatch, target, dtype):
    cache = tmp_path/"cython-cache"
    if target == "cython":
        monkeypatch.setitem(b.prefs, "codegen.runtime.cython.cache_dir", str(cache))
    params = {"eq_rst": None, "v_0": -44*b.mV}
    with cpu_runtime(seed=173, dt_ms=.1):
        b.prefs.codegen.target = target
        b.prefs.core.default_float_dtype = dtype
        expected, _ = _run_original(tmp_path/"cpu", parameters=params)
        wanted = [expected.capture(), expected.run_ms(2.3)]
        assert not expected.neurons.resetter
        assert wanted[-1].spike_i.tolist() == [0, 0]
        assert wanted[0].state["v"][0] > -.045
        expected_rng = capture_rng_state()
        if target == "cython":
            assert expected.neurons.state_updater.codeobj.class_name == "cython"
            assert any(cache.rglob("*.so"))
    with cpu_runtime(seed=173, dt_ms=.1):
        b.prefs.codegen.target = target
        b.prefs.core.default_float_dtype = dtype
        with original_api(workers=False) as device:
            actual, _ = _run_original(tmp_path/"native", parameters=params)
            observed = [actual.capture(), actual.run_ms(2.3)]
            assert not actual.neurons.resetter
            assert actual.neurons.namespace["eq_rst"] is None
            for actual_state, expected_state in zip(observed, wanted):
                assert_snapshot_equal(actual_state, expected_state)
            assert_rng_state_equal(capture_rng_state(), expected_rng)
            assert device.native_steps == 25 and device.native_runs == 2


def test_original_unused_metadata_keeps_real_namespace_and_cpu_results(tmp_path):
    def not_called(value):
        raise AssertionError("An unused annotation was executed")
    metadata = {"tag": None, "annotation": object(),
                "unused_function": b.Function(not_called, arg_units=[b.volt], return_unit=b.volt)}
    params = {**metadata, "v_0": -44*b.mV}
    with cpu_runtime(seed=173, dt_ms=.1):
        expected, _ = _run_original(tmp_path/"cpu", parameters=params)
        wanted = expected.capture()
        expected_rng = capture_rng_state()
    with original_runtime(original_api, seed=173, dt_ms=.1) as device:
        actual, _ = _run_original(tmp_path/"native", parameters=params)
        for name, value in metadata.items():
            assert actual.neurons.namespace[name] is value
        assert_snapshot_equal(actual.capture(), wanted)
        assert_rng_state_equal(capture_rng_state(), expected_rng)
        assert device.native_steps == 2


def test_original_referenced_none_still_raises_cpu_preparation_error(tmp_path):
    params = {"tag": None, "eq_th": "tag > 0"}
    def run(directory):
        with pytest.raises(b.BrianObjectException) as caught:
            _run_original(directory, parameters=params)
        cause = caught.value.__cause__
        return type(cause), str(cause)
    with cpu_runtime(seed=173, dt_ms=.1):
        expected = run(tmp_path/"cpu")
    with original_runtime(original_api, seed=173, dt_ms=.1) as device:
        assert run(tmp_path/"native") == expected
        assert device.native_steps == device.native_runs == 0


def test_original_run_namespace_keeps_reset_and_threshold_dependencies(tmp_path):
    params = {"v_0": -44*b.mV, "eq_th": "(v>threshold_value)",
              "eq_rst": "v=reset_value"}
    namespace = {"threshold_value": -45*b.mV, "reset_value": -51*b.mV}
    def run(directory):
        fixture = build_original_api_fixture(directory, parameters=params,
                                             exc=(), exc2=(), silenced=())
        fixture.network.run(.2*b.ms, namespace=namespace)
        return fixture.capture()
    with cpu_runtime(seed=173, dt_ms=.1):
        wanted = run(tmp_path/"cpu")
        assert wanted.spike_i.tolist() == [0, 1, 2, 3]
        expected_rng = capture_rng_state()
    with original_runtime(original_api, seed=173, dt_ms=.1) as device:
        assert_snapshot_equal(run(tmp_path/"native"), wanted)
        assert_rng_state_equal(capture_rng_state(), expected_rng)
        assert device.native_steps == 2 and device.native_runs == 1
