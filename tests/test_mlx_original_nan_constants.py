"""Original Eon accepted NaN constants keep exact native CPU behavior."""
import numpy as np
import pandas as pd
import pytest

b = pytest.importorskip("brian2")
from native_execution import require_metal
mx = require_metal('Metal is required', allow_module_level=True)

from mlx_backend.brian_device import original_api
from original_api_reference import (
    OriginalAPIFixture, assert_rng_state_equal, assert_snapshot_equal,
    capture_rng_state, cpu_runtime, load_original_model,
)

pytestmark = pytest.mark.metal


def _fixture(directory, name, *, value=np.nan, delay=0.):
    directory.mkdir()
    comp, con = directory / "neurons.csv", directory / "edges.parquet"
    pd.DataFrame(index=pd.Index(["source", "target"], name="flywire_id")).to_csv(comp)
    pd.DataFrame({"Presynaptic_Index": [0], "Postsynaptic_Index": [1],
                  "Excitatory x Connectivity": [1.]}).to_parquet(con)
    original = load_original_model()
    unit = b.mV if name in ("v_0", "v_rst", "w_syn") else b.ms
    parameters = {**original.default_params, name: value * unit, "t_dly": delay*b.ms}
    neurons, synapses, monitor = original.create_model(comp, con, parameters)
    inputs, neurons = original.poi(neurons, [], [], parameters)
    synapses = original.silence([], synapses)
    if name in ("v_rst", "w_syn"):
        # Exercise the reset/arrival value, rather than accepting a dormant NaN.
        neurons.v = np.array([-44., -52.])*b.mV
    return OriginalAPIFixture(original, parameters, (comp, con), neurons, synapses,
        monitor, inputs, b.Network(neurons, synapses, monitor, *inputs))


_NAMES = ("v_0", "v_rst", "w_syn", "t_mbr", "tau", "t_rfc")
_CASES = [pytest.param(name, target, dtype, id=f"{target}-{np.dtype(dtype).name}-{name}")
    for target, dtype in (("numpy", np.float64), ("numpy", np.float32), ("cython", np.float64))
    for name in _NAMES]
_CASES += [pytest.param(name, "cython", np.float32, id=f"cython-float32-{name}")
    for name in ("v_0", "t_rfc")]


@pytest.fixture(scope="module")
def private_cython_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("nan-constant-cython")


@pytest.mark.parametrize("name,target,dtype", _CASES)
def test_original_nan_constants_match_actual_cpu(
        tmp_path, monkeypatch, private_cython_cache, name, target, dtype):
    if target == "cython":
        monkeypatch.setitem(b.prefs, "codegen.runtime.cython.cache_dir", str(private_cython_cache))
    with cpu_runtime(seed=173, dt_ms=.1):
        b.prefs.codegen.target = target
        b.prefs.core.default_float_dtype = dtype
        expected = _fixture(tmp_path / "cpu", name)
        expected.run_ms(.1)
        expected.run_ms(.1)
        expected_rng = capture_rng_state()
        if target == "cython":
            assert expected.neurons.state_updater.codeobj.class_name == "cython"
            assert any(private_cython_cache.rglob("*.so"))
    with cpu_runtime(seed=173, dt_ms=.1):
        b.prefs.codegen.target = target
        b.prefs.core.default_float_dtype = dtype
        with original_api(workers=False) as device:
            actual = _fixture(tmp_path / "native", name)
            actual.run_ms(.1)
            actual.run_ms(.1)
            assert device.native_steps == 2 and device.native_runs == 2
            for actual_state, expected_state in zip(actual.snapshots, expected.snapshots):
                assert_snapshot_equal(actual_state, expected_state)
            assert_rng_state_equal(capture_rng_state(), expected_rng)


@pytest.mark.parametrize("target,dtype", [("numpy", np.float64), ("numpy", np.float32),
                                         ("cython", np.float64), ("cython", np.float32)])
def test_original_nan_delay_delivers_real_arrival(
        tmp_path, monkeypatch, private_cython_cache, target, dtype):
    _assert_original_delay(tmp_path, monkeypatch, private_cython_cache, target, dtype, np.nan)


@pytest.mark.parametrize("target,dtype,delay", [("numpy", np.float64, -.025),
    ("numpy", np.float32, -.2), ("cython", np.float64, -.2), ("cython", np.float32, -.025),
    pytest.param("numpy", np.float64, -np.inf, id="numpy64-negative-infinity")])
def test_original_negative_delay_delivers_real_arrival(
        tmp_path, monkeypatch, private_cython_cache, target, dtype, delay):
    _assert_original_delay(tmp_path, monkeypatch, private_cython_cache, target, dtype, delay)


def _assert_original_delay(tmp_path, monkeypatch, private_cython_cache, target, dtype, delay):
    if target == "cython":
        monkeypatch.setitem(b.prefs, "codegen.runtime.cython.cache_dir", str(private_cython_cache))
    with cpu_runtime(seed=173, dt_ms=.1):
        b.prefs.codegen.target = target
        b.prefs.core.default_float_dtype = dtype
        expected = _fixture(tmp_path / "cpu", "v_rst", value=-52., delay=delay)
        expected.run_ms(.1)
        expected.run_ms(.1)
        expected_rng = capture_rng_state()
        assert expected.snapshots[0].spike_i.tolist() == [0]
        assert expected.snapshots[0].state["g"][1] > 0
        if target == "cython":
            assert expected.neurons.state_updater.codeobj.class_name == "cython"
            assert any(private_cython_cache.rglob("*.so"))
    with cpu_runtime(seed=173, dt_ms=.1):
        b.prefs.codegen.target = target
        b.prefs.core.default_float_dtype = dtype
        with original_api(workers=False) as device:
            actual = _fixture(tmp_path / "native", "v_rst", value=-52., delay=delay)
            actual.run_ms(.1)
            actual.run_ms(.1)
            assert device.native_steps == 2 and device.native_runs == 2
            assert actual.synapses.delay_[:].tobytes() == expected.synapses.delay_[:].tobytes()
            for actual_state, expected_state in zip(actual.snapshots, expected.snapshots):
                assert_snapshot_equal(actual_state, expected_state)
            assert_rng_state_equal(capture_rng_state(), expected_rng)


def test_nan_delay_retains_raw_cache_identity_and_infinite_rejection():
    from mlx_backend.model import resolve_model
    nan_model = resolve_model({"t_dly": np.nan*b.ms})
    zero_model = resolve_model({"t_dly": 0*b.ms})
    assert nan_model.delay_steps == zero_model.delay_steps == 0
    assert np.isnan(nan_model.t_dly)
    assert nan_model.signature != zero_model.signature
    negative_model = resolve_model({"t_dly": -.2*b.ms})
    assert negative_model.delay_steps == 0 and negative_model.t_dly == -.0002
    assert negative_model.signature not in (nan_model.signature, zero_model.signature)
    negative_infinite_model = resolve_model({"t_dly": -np.inf*b.ms})
    assert negative_infinite_model.delay_steps == 0 and np.isneginf(negative_infinite_model.t_dly)
    assert negative_infinite_model.signature not in (
        nan_model.signature, zero_model.signature, negative_model.signature)
    with pytest.raises(ValueError, match="t_dly must contain finite real values"):
        resolve_model({"t_dly": np.inf*b.ms})
