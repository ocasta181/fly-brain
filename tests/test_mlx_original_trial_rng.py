"""Independent ambient RNG continuity through the unchanged Eon trial API."""

from contextlib import nullcontext

import numpy as np
import pytest

from original_api_reference import (
    assert_rng_state_equal, capture_rng_state, cpu_runtime, load_original_model,
    write_tiny_dataset,
)


def _sequence(paths, target, native):
    import brian2 as b
    from mlx_backend.brian_device import original_api

    snapshots = []
    with cpu_runtime(seed=941):
        b.prefs.codegen.target = target
        original = load_original_model()
        get_spikes = original.get_spk_trn

        def capture(monitor):
            group = monitor.source
            names = [name for name, equation in group.equations.items()
                     if equation.type != "subexpression"]
            names = tuple(dict.fromkeys([*names, "lastspike", "not_refractory"]))
            snapshots.append({
                "state": {name: np.array(group.variables[name].get_value(), copy=True)
                          for name in names},
                "spike_i": np.array(monitor.i[:], copy=True),
                "spike_t": np.asarray(monitor.t[:]).copy(),
                "count": np.array(monitor.count[:], copy=True),
                "clock": np.array(group.clock.variables["timestep"].get_value(), copy=True),
                "rng": capture_rng_state(),
            })
            return get_spikes(monitor)

        # Copy observations at the original return boundary without retaining
        # Brian objects, which would change later automatically assigned names.
        original.get_spk_trn = capture
        with original_api(workers=False) if native else nullcontext() as device:
            trains = []
            interleaved = []
            for duration in [.7, 0., 2.35, .15]:
                parameters = {**original.default_params, "r_poi": 3700*b.Hz,
                              "r_poi2": 1400*b.Hz, "t_run": duration*b.ms}
                result = original.run_trial([0, 0], [2], [3], *paths, parameters)
                trains.append({key: np.asarray(value).copy() for key, value in result.items()})
                # Ordinary caller randomness must remain part of the same
                # ambient NumPy stream between fresh original trial objects.
                interleaved.append(np.random.random(7))
            if native:
                assert device.native_runs == 4
            final_rng = capture_rng_state()
            future = np.random.random(624)
    return snapshots, trains, interleaved, final_rng, future


@pytest.mark.metal
@pytest.mark.parametrize("target", ["numpy", "cython"])
def test_original_trials_preserve_ambient_rng_across_fresh_objects(tmp_path, target):
    paths = write_tiny_dataset(tmp_path)
    expected = _sequence(paths, target, False)
    actual = _sequence(paths, target, True)
    for got, want in zip(actual[0], expected[0]):
        assert got["state"].keys() == want["state"].keys()
        for name in want["state"]:
            assert got["state"][name].dtype == want["state"][name].dtype
            np.testing.assert_array_equal(got["state"][name].view(np.uint8),
                                          want["state"][name].view(np.uint8))
        for name in ("spike_i", "spike_t", "count", "clock"):
            np.testing.assert_array_equal(got[name], want[name])
        assert_rng_state_equal(got["rng"], want["rng"])
    for got, want in zip(actual[1], expected[1]):
        assert got.keys() == want.keys()
        for name in want:
            np.testing.assert_array_equal(got[name], want[name])
    np.testing.assert_array_equal(actual[2], expected[2])
    assert_rng_state_equal(actual[3], expected[3])
    np.testing.assert_array_equal(actual[4], expected[4])
    assert any(s["spike_i"].size for s in expected[0])
    assert not np.array_equal(expected[0][0]["state"]["v"],
                              expected[0][2]["state"]["v"])


def _trial_output(paths, duration):
    import brian2 as b

    original = load_original_model()
    parameters = {**original.default_params, "r_poi": 3700*b.Hz,
                  "r_poi2": 1400*b.Hz, "t_run": duration*b.ms}
    result = original.run_trial([0, 0], [2], [3], *paths, parameters)
    return ({key: np.asarray(value).copy() for key, value in result.items()},
            np.random.random(7), capture_rng_state())


def _reused_worker_trial(paths, phase, rendezvous):
    import os
    import time
    from pathlib import Path
    from brian2.devices.device import get_device
    from mlx_backend.brian_device import MLXDevice

    device = get_device()
    assert type(device) is MLXDevice
    assert device.native_runs == phase
    if phase == 0:
        import brian2 as b
        b.seed(941)
    # Both actual workers execute each phase; a fast worker cannot consume all
    # four tasks and accidentally turn this into another serial-device check.
    directory = Path(rendezvous)
    (directory / f"{phase}-{os.getpid()}").touch()
    deadline = time.monotonic() + 8
    while len(list(directory.glob(f"{phase}-*"))) < 2:
        if time.monotonic() >= deadline:
            raise AssertionError("The second loky worker did not reach this phase")
        time.sleep(.01)
    result = _trial_output(paths, [.7, 2.35][phase])
    assert device.native_runs == phase + 1
    return os.getpid(), phase, result


@pytest.mark.metal
@pytest.mark.parametrize("target", ["numpy", "cython"])
def test_reused_loky_workers_preserve_explicit_seed_and_ambient_rng(tmp_path, target):
    from joblib import Parallel, delayed, parallel_backend
    from joblib.externals.loky import get_reusable_executor
    from mlx_backend.brian_device import original_api

    paths = write_tiny_dataset(tmp_path)
    rendezvous = tmp_path / "rendezvous"
    rendezvous.mkdir()
    with cpu_runtime(seed=941) as b:
        b.prefs.codegen.target = target
        expected = [_trial_output(paths, duration) for duration in [.7, 2.35]]
    with cpu_runtime(seed=941) as b:
        b.prefs.codegen.target = target
        with original_api(workers=True):
            try:
                with parallel_backend("loky", n_jobs=2):
                    actual = Parallel(batch_size=1, pre_dispatch=2)(
                        delayed(_reused_worker_trial)(paths, phase, rendezvous)
                        for phase in [0, 0, 1, 1])
            finally:
                get_reusable_executor().shutdown(wait=True, kill_workers=True)
    assert len({pid for pid, _, _ in actual}) == 2
    for phase in [0, 1]:
        assert {pid for pid, p, _ in actual if p == phase} == {pid for pid, _, _ in actual}
    for _, phase, (trains, caller, rng) in actual:
        expected_trains, expected_caller, expected_rng = expected[phase]
        assert trains.keys() == expected_trains.keys()
        for key in trains:
            np.testing.assert_array_equal(trains[key], expected_trains[key])
        np.testing.assert_array_equal(caller, expected_caller)
        assert_rng_state_equal(rng, expected_rng)
