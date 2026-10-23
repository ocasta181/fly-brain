"""Actual original objects retain binary32 storage across preparation changes."""

import numpy as np
import pytest

b = pytest.importorskip("brian2")
from native_execution import require_metal
mx = require_metal('Metal is required', allow_module_level=True)

from mlx_backend.brian_device import original_api
from mlx_backend.model import DEFAULT_EQUATIONS
from original_api_reference import (
    assert_rng_state_equal, assert_snapshot_equal, build_original_api_fixture,
    capture_rng_state, cpu_runtime, original_runtime,
)

pytestmark = pytest.mark.metal


@pytest.mark.parametrize("target", ["numpy", "cython"])
@pytest.mark.parametrize("alter_rfc", [False, True])
def test_actual_float32_refractory_release_and_mutated_state(tmp_path, target, alter_rfc):
    """Compare release boundaries and continuation using actual stored rfc bits."""

    expected = []
    parameters = {"v_0": -40*b.mV, "v_rst": -40*b.mV,
                  "r_poi": 0*b.Hz, "t_rfc": 2.2*b.ms}
    for native in (False, True):
        with original_runtime(
                original_api, seed=773, target=target, float_dtype=np.float32, native=native) as device:
            fixture = build_original_api_fixture(tmp_path/str(native), parameters=parameters,
                                                 exc=(1,), exc2=(), silenced=())
            if alter_rfc:
                fixture.neurons.rfc_[0] = np.nextafter(np.float32(.0022), np.float32(np.inf))
            for index, duration in enumerate((2.1, .1, .1, 1.2)):
                observed = fixture.run_ms(duration)
                if native:
                    assert_snapshot_equal(observed, expected[index])
                    session = device._sessions[fixture.network]
                    assert session.fast_refractory == (not alter_rfc)
                    assert (session.prepared.program.refractory_elapsed_threshold is not None) == (not alter_rfc)
                else:
                    expected.append(observed)


@pytest.mark.parametrize("target", ["numpy", "cython"])
@pytest.mark.parametrize("mode", ["fused", "wide", "zero_delay", "none"])
def test_actual_float32_objects_inputs_and_later_float64_temporaries(
        tmp_path, monkeypatch, target, mode):
    from mlx_backend.brian_poisson import NativePoisson

    monkeypatch.setattr(NativePoisson, "MAX_TAPE_DRAWS", 12)
    equations = DEFAULT_EQUATIONS + ("\nfoo : volt\nbar : volt" if mode == "wide" else "")
    parameters = {"eqs": equations, "t_dly": (0 if mode == "zero_delay" else .3)*b.ms,
                  "r_poi": 3700*b.Hz, "r_poi2": 7200*b.Hz}

    def run(directory):
        fixture = build_original_api_fixture(directory, parameters=parameters,
            exc=() if mode == "none" else (1, 1), exc2=() if mode == "none" else (3,), initial_state={
                "v": np.array([-49., -44., -48., -50.])*b.mV,
                "g": np.array([1., -3., 4., 2.])*b.mV})
        for index, duration in enumerate((0., .15, .2, .25, .1)):
            if index == 2:
                b.prefs.core.default_float_dtype = np.float64
                if fixture.inputs:
                    fixture.inputs[1].active = False
            elif index == 3 and fixture.inputs:
                fixture.inputs[1].active = True
            fixture.run_ms(duration)
        return fixture

    with cpu_runtime(seed=1733, target=target, float_dtype=np.float32):
        expected = run(tmp_path/"cpu")
        expected_rng = capture_rng_state()
        expected_next = np.random.random(10)
    with original_runtime(original_api, seed=1733, target=target, float_dtype=np.float32) as device:
        actual = run(tmp_path/"metal")
        for observed, reference in zip(actual.snapshots, expected.snapshots):
            assert_snapshot_equal(observed, reference)
        assert_rng_state_equal(capture_rng_state(), expected_rng)
        np.testing.assert_array_equal(np.random.random(10), expected_next)
        model = device._sessions[actual.network].prepared.model
        assert model.state_dtype == "float32"
        assert model.analysis.temporary_dtype == "float64"
        assert model.analysis.variables["lastspike"].dtype == expected.snapshots[-1].state["lastspike"].dtype.name
        assert model.float32_codegen == model.reference_codegen_target == target
        assert actual.snapshots[-1].state["v"].dtype == np.dtype("float32")
        assert b.prefs.core.default_float_dtype is np.float64
        if target == "cython":
            assert model.float32_compile_args == tuple(actual.neurons.state_updater.codeobj.extra_compile_args)


@pytest.mark.parametrize("phases", [("groups","thresholds","synapses","resets"),
                                     ("resets","groups","thresholds","synapses")])
def test_actual_float32_collective_reset_with_native_poisson(tmp_path, monkeypatch, phases):
    import pandas as pd
    from mlx_backend.brian_poisson import NativePoisson
    from original_api_reference import OriginalAPIFixture, load_original_model

    monkeypatch.setattr(NativePoisson, "MAX_TAPE_DRAWS", 12)

    def run(directory):
        directory.mkdir()
        original = load_original_model()
        paths = directory/"neurons.csv", directory/"edges.parquet"
        pd.DataFrame(index=np.arange(17)).to_csv(paths[0])
        pd.DataFrame({"Presynaptic_Index": np.arange(9),
                      "Postsynaptic_Index": np.full(9, 16),
                      "Excitatory x Connectivity": np.linspace(-200.125, 400.25, 9)}).to_parquet(paths[1])
        parameters = {**original.default_params,
            "eqs": DEFAULT_EQUATIONS+"a : 1\nk : integer\n",
            "eq_th": "(t < dt) and (i < 9)",
            "eq_rst": "v=v_rst;g=0*mV;k=big+i;a=k", "big": 2**62+2**38+1,
            "t_rfc": 0*b.ms, "t_dly": .1*b.ms,
            "r_poi": 3700*b.Hz, "r_poi2": 7200*b.Hz}
        neurons, synapses, monitor = original.create_model(*paths, parameters)
        inputs, _ = original.poi(neurons, [1, 1], [16], parameters)
        fixture = OriginalAPIFixture(original, parameters, paths, neurons, synapses,
            monitor, inputs, b.Network(neurons, synapses, monitor, *inputs))
        for duration in ((.2 if phases[0] == "resets" else .1), .15, .2):
            fixture.run_ms(duration)
        return fixture

    with cpu_runtime(seed=1992):
        b.prefs.core.network.default_schedule = ["start", *phases, "end"]
        b.prefs.core.default_float_dtype = np.float32
        b.prefs.core.default_integer_dtype = np.int64
        expected = run(tmp_path/"cpu")
        rng = capture_rng_state()
    with cpu_runtime(seed=1992):
        b.prefs.core.network.default_schedule = ["start", *phases, "end"]
        b.prefs.core.default_float_dtype = np.float32
        b.prefs.core.default_integer_dtype = np.int64
        with original_api(workers=False) as device:
            actual = run(tmp_path/"native")
            for observed, reference in zip(actual.snapshots, expected.snapshots):
                assert_snapshot_equal(observed, reference)
            assert_rng_state_equal(capture_rng_state(), rng)
            prepared = device._sessions[actual.network].prepared
            assert prepared.program.collective_reset and prepared.window == 1
            assert not device._sessions[actual.network].sampler.can_fuse(prepared)
            assert np.unique(expected.snapshots[0].state["a"][:9].view(np.uint32)).size == 2


def test_explicit_state_dtype_keeps_temporary_precision_in_cache_identity():
    from mlx_backend.model import resolve_model

    with cpu_runtime(float_dtype=np.float32):
        first = resolve_model({}, state_dtype=np.float32)
        b.prefs.core.default_float_dtype = np.float64
        second = resolve_model({}, state_dtype=np.float32)
        assert first.state_dtype == second.state_dtype == "float32"
        assert first.analysis.temporary_dtype == "float32"
        assert second.analysis.temporary_dtype == "float64"
        assert first.signature != second.signature
