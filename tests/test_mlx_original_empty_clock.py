"""Actual original Eon zero-timestep clock/horizon boundary semantics."""
import numpy as np
import pandas as pd
import pytest

b = pytest.importorskip("brian2")
from native_execution import require_metal
mx = require_metal('Metal is required', allow_module_level=True)

from mlx_backend.brian_device import original_api
from original_api_reference import (
    OriginalAPIFixture, assert_rng_state_equal, assert_snapshot_equal,
    capture_rng_state, cpu_runtime, original_runtime, load_original_model,
)

pytestmark = pytest.mark.metal


def _run_original(directory, *, dt_ms=.1, duration_ms=.2, parameters=None, edge_count=0.):
    directory.mkdir()
    comp, con = directory/"neurons.csv", directory/"edges.parquet"
    pd.DataFrame(index=["one"]).to_csv(comp)
    pd.DataFrame({"Presynaptic_Index": [0], "Postsynaptic_Index": [0],
                  "Excitatory x Connectivity": [edge_count]}).to_parquet(con)
    original = load_original_model()
    networks = []
    def capture(*args, **kwargs):
        network = b.Network(*args, **kwargs)
        networks.append(network)
        return network
    original.Network = capture
    params = {**original.default_params, "t_run": duration_ms*b.ms, **(parameters or {})}
    result = original.run_trial([], [], [], comp, con, params)
    network = networks[0]
    group = next(obj for obj in network.objects if type(obj) is b.NeuronGroup)
    synapses = next(obj for obj in network.objects if type(obj) is b.Synapses)
    monitor = next(obj for obj in network.objects if type(obj) is b.SpikeMonitor)
    fixture = OriginalAPIFixture(original, params, (comp, con), group, synapses,
                                 monitor, [], network)
    return fixture, result


@pytest.mark.parametrize("dt_ms,duration_ms", [(.1, np.nan), (np.nan, .2), (np.inf, .2)])
def test_original_nonfinite_empty_clocks_match_cpu(tmp_path, monkeypatch, dt_ms, duration_ms):
    _assert_empty_clock_pair(tmp_path, monkeypatch, dt_ms, duration_ms)


@pytest.mark.parametrize("dt_ms,duration_ms", [(.1, -.2), (-.1, .2), (.1, -np.inf)])
def test_original_negative_empty_clocks_match_cpu(tmp_path, monkeypatch, dt_ms, duration_ms):
    _assert_empty_clock_pair(tmp_path, monkeypatch, dt_ms, duration_ms)


def _assert_empty_clock_pair(tmp_path, monkeypatch, dt_ms, duration_ms):
    with cpu_runtime(seed=173, dt_ms=dt_ms):
        expected, wanted = _run_original(tmp_path/"cpu", dt_ms=dt_ms, duration_ms=duration_ms)
        snapshot = expected.capture()
        expected_rng = capture_rng_state()
    with original_runtime(original_api, seed=173, dt_ms=dt_ms) as device:
        from brian2.codegen.runtime.numpy_rt.numpy_rt import NumpyCodeObject
        original_run = NumpyCodeObject.run
        def forbid_timestep(codeobj, *args, **kwargs):
            if codeobj.template_name in {"stateupdate", "threshold", "reset", "synapses",
                    "synapses_push_spikes", "spikemonitor", "poissoninput"}:
                raise AssertionError("Empty clock executed a CPU numerical timestep")
            return original_run(codeobj, *args, **kwargs)
        monkeypatch.setattr(NumpyCodeObject, "run", forbid_timestep)
        actual, observed = _run_original(tmp_path/"native", dt_ms=dt_ms, duration_ms=duration_ms)
        assert wanted == observed == {}
        assert_snapshot_equal(actual.capture(), snapshot)
        assert_rng_state_equal(capture_rng_state(), expected_rng)
        assert device.native_runs == 1 and device.native_steps == 0
        assert device.last_timings["execution_s"] == 0.


@pytest.mark.parametrize("dt_ms,duration_ms", [(.1, np.nan), (np.nan, .2)])
def test_empty_clocks_still_run_actual_brian_unit_validation(tmp_path, dt_ms, duration_ms):
    def run(directory):
        with pytest.raises(b.BrianObjectException) as caught:
            _run_original(directory, dt_ms=dt_ms, duration_ms=duration_ms,
                          parameters={"eq_rst": "v = 1*second"})
        cause = caught.value.__cause__
        return type(cause), str(cause)
    with cpu_runtime(seed=173, dt_ms=dt_ms):
        expected_type, expected_message = run(tmp_path/"cpu")
    with original_runtime(original_api, seed=173, dt_ms=dt_ms) as device:
        actual_type, actual_message = run(tmp_path/"native")
        assert (actual_type, actual_message) == (expected_type, expected_message)
        assert device.native_runs == device.native_steps == 0


def test_original_zero_timestep_matches_cpu_clock_error(tmp_path):
    with cpu_runtime(seed=173, dt_ms=0.):
        with pytest.raises(ZeroDivisionError, match="float division by zero"):
            _run_original(tmp_path/"cpu")
    with original_runtime(original_api, seed=173, dt_ms=0.) as device:
        with pytest.raises(ZeroDivisionError, match="float division by zero"):
            _run_original(tmp_path/"native")
        assert device.native_runs == device.native_steps == 0
