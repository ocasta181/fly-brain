"""Real original-Python-API execution against independent CPU Brian networks."""
from native_execution import forbid_cpu_steps as guard_cpu_steps, require_metal

from dataclasses import replace

import numpy as np
import pytest

mx = require_metal('Metal is required', allow_module_level=True)
b = pytest.importorskip("brian2")

from brian2.devices.device import get_device
from mlx_backend.brian_device import MLXDevice, UnsupportedBrianNetwork, original_api
from original_api_reference import (
    assert_snapshot_equal, build_original_api_fixture, cpu_runtime, original_runtime, run_original_api,
    assert_rng_state_equal, capture_rng_state,
)

pytestmark = pytest.mark.metal


def _run_pair(tmp_path, durations, *, parameters=None, target="numpy", seed=875,
              mutations=None, exc=(0, 0), exc2=(2,), initial_state=None, dt_ms=.1, state_dtype=np.float64):
    with cpu_runtime(seed=seed, dt_ms=dt_ms, target=target, float_dtype=state_dtype):
        options = dict(parameters=parameters, exc=exc, exc2=exc2, initial_state=initial_state)
        expected = run_original_api(tmp_path / "cpu", durations, mutations=mutations, **options)
        expected_rng = capture_rng_state()
        next_random = np.random.random(10)
    with original_runtime(
            original_api, seed=seed, dt_ms=dt_ms, target=target, float_dtype=state_dtype) as device:
        actual = run_original_api(tmp_path / "metal", durations, mutations=mutations, **options)
        assert device.native_runs == len(durations)
        assert device.native_steps == actual.snapshots[-1].clock_timestep
        for actual_snapshot, expected_snapshot in zip(actual.snapshots, expected.snapshots):
            assert_snapshot_equal(actual_snapshot, expected_snapshot)
        assert_rng_state_equal(capture_rng_state(), expected_rng)
        np.testing.assert_array_equal(np.random.random(10), next_random)
        return actual, device


@pytest.mark.parametrize("durations", [(.3,), (.15, .15), (0., .15, 0., .15, 0.)])
def test_real_original_objects_and_requested_vs_clock_time(tmp_path, durations):
    actual, device = _run_pair(tmp_path, durations,
        parameters={"r_poi": 10000*b.Hz, "r_poi2": 0*b.Hz})
    assert type(actual.neurons) is b.NeuronGroup
    assert type(actual.synapses) is b.Synapses
    assert type(actual.monitor) is b.SpikeMonitor
    assert actual.snapshots[-1].spike_i.size > 0


@pytest.mark.parametrize("target", ["numpy", "cython"])
@pytest.mark.parametrize("threshold", [np.inf, -np.inf, np.nan])
def test_nonfinite_thresholds_preserve_original_ieee_behavior(tmp_path, target, threshold):
    actual, _ = _run_pair(tmp_path, [.15, .05, .3], target=target,
        parameters={"v_th": threshold*b.mV, "r_poi": 10000*b.Hz, "r_poi2": 0*b.Hz},
        exc=(0,), exc2=())
    count = actual.snapshots[-1].spike_i.size
    assert count > 0 if threshold == -np.inf else count == 0


@pytest.mark.parametrize("target,threshold", [("numpy", np.nan), ("cython", -np.inf)])
def test_nonfinite_threshold_float32_promotions_match_original(tmp_path, target, threshold):
    actual, _ = _run_pair(tmp_path, [.15, .05, .3], target=target, state_dtype=np.float32,
        parameters={"v_th": threshold*b.mV, "r_poi": 10000*b.Hz, "r_poi2": 0*b.Hz},
        exc=(0,), exc2=())
    assert actual.snapshots[-1].state["v"].dtype == np.float32


@pytest.mark.parametrize("target,constant,state_dtype", [
    ("numpy", "tau", np.float64), ("cython", "tau", np.float64),
    ("numpy", "t_mbr", np.float64), ("cython", "t_mbr", np.float64),
    ("numpy", "tau", np.float32), ("cython", "t_mbr", np.float32),
])
def test_infinite_decay_constants_match_original(tmp_path, target, constant, state_dtype):
    actual, _ = _run_pair(tmp_path, [.15, .05, 3.0], target=target,
        state_dtype=state_dtype,
        parameters={constant: np.inf*b.ms, "r_poi": 10000*b.Hz, "r_poi2": 0*b.Hz},
        exc=(0,), exc2=(),
        initial_state={"v": np.array([-44., -49., -48., -50.])*b.mV,
                       "g": np.array([3., -2., 4., 1.])*b.mV})
    # The reference's symbolic exact integrator can produce NaN even when
    # the mathematical limiting ODE is finite. _run_pair checks the actual
    # selected CPU program bit for bit, including those nonfinite results.
    assert actual.snapshots[-1].state["v"].dtype == state_dtype


@pytest.mark.parametrize("target,period", [
    ("numpy", np.inf), ("cython", np.inf),
    ("numpy", -np.inf), ("cython", -np.inf),
    pytest.param("cython", .35, id="cython-finite-release-control"),
])
def test_infinite_refractory_periods_preserve_cpu_target_conversion(tmp_path, target, period):
    _run_pair(tmp_path, [.15, .05, 3.0], target=target,
        parameters={"t_rfc": period*b.ms, "r_poi": 10000*b.Hz, "r_poi2": 0*b.Hz},
        exc=(0,), exc2=(),
        initial_state={"v": np.array([-44., -49., -48., -50.])*b.mV,
                       "g": np.array([3., -2., 4., 1.])*b.mV})


@pytest.mark.parametrize("delay_steps", [0, 1, 18, 33, 65])
def test_pending_delays_and_partial_window_continuation(tmp_path, delay_steps):
    _run_pair(tmp_path, [.15, .05, 1.7, 0., 2.35, 8.65], parameters={
        "t_dly": delay_steps*.1*b.ms, "r_poi": 10000*b.Hz, "r_poi2": 0*b.Hz,
    })


@pytest.mark.parametrize("target", ["numpy", "cython"])
@pytest.mark.parametrize("rates", [(3700, 2100), (0, 10000), (10000, 0), (7200, 1300)])
def test_seeded_n1_poisson_matches_state_dependent_rng_and_next_draw(tmp_path, target, rates):
    _run_pair(tmp_path, [0., .15, .15, 1.8, 6.1], target=target, parameters={
        "r_poi": rates[0]*b.Hz, "r_poi2": rates[1]*b.Hz, "t_dly": .3*b.ms,
    })


@pytest.mark.parametrize("target", ["numpy", "cython"])
def test_fused_poisson_nonzero_target_rows_match_independent_cpu(tmp_path, monkeypatch, target):
    from mlx_backend import brian_poisson

    def separate_prepass_was_used(*args, **kwargs):
        raise AssertionError("The small original model must execute its fused native Poisson dispatch")

    monkeypatch.setattr(brian_poisson, "_kernel", separate_prepass_was_used)
    _run_pair(tmp_path, [.15, .15, 1.7, 3.4, .05], target=target,
        exc=(1, 1), exc2=(3,), parameters={
            "t_dly": 3.3*b.ms, "r_poi": 3700*b.Hz, "r_poi2": 7200*b.Hz,
        }, initial_state={"v": np.array([-49., -44., -48., -50.])*b.mV,
                          "g": np.array([1., -3., 4., 2.])*b.mV})


def test_wider_numpy_state_layout_retains_exact_native_prepass(tmp_path, monkeypatch):
    from mlx_backend import brian_poisson
    from original_api_reference import load_original_model

    original_can_fuse = brian_poisson.NativePoisson.can_fuse
    decisions = []
    def observe(self, prepared):
        decision = original_can_fuse(self, prepared)
        decisions.append(decision)
        return decision
    monkeypatch.setattr(brian_poisson.NativePoisson, "can_fuse", observe)
    equations = load_original_model().default_params["eqs"] + "\nfoo : volt\nbar : volt\n"
    _run_pair(tmp_path, [.15, .15, 3.4], target="numpy", parameters={
        "eqs": equations, "r_poi": 3700*b.Hz, "r_poi2": 2100*b.Hz,
    }, initial_state={"foo": np.array([1., -2., 3., -4.])*b.mV,
                      "bar": np.array([4., -3., 2., -1.])*b.mV})
    assert decisions and not any(decisions)


@pytest.mark.parametrize("mode", ["fused", "wide", "zero_delay"])
def test_zero_input_draws_preserve_ieee_addition(tmp_path, mode):
    equations = "v : volt\ng : volt\nrfc : second\n"
    if mode == "wide":
        equations += "foo : volt\nbar : volt\n"
    def inactive(fixture):
        for source in fixture.inputs:
            source.active = False
        fixture.neurons.v = np.full(4, -0.0)*b.volt

    actual, _ = _run_pair(tmp_path, [.1, .2, .1], target="numpy", parameters={
        "eqs": equations, "eq_th": "False", "eq_rst": "v = v_rst",
        "t_dly": (0 if mode == "zero_delay" else .3)*b.ms,
        "r_poi": 0*b.Hz, "r_poi2": 0*b.Hz,
    }, initial_state={"v": np.full(4, -0.0)*b.volt}, mutations={2: inactive})
    # A zero draw still executes v += positive_weight * 0. Active input rows
    # become +0; a row without a channel retains its independently stored -0.
    expected_bits = np.array([0, 1 << 63, 0, 1 << 63], np.uint64)
    np.testing.assert_array_equal(actual.snapshots[1].state["v"].view(np.uint64),
                                  expected_bits)
    np.testing.assert_array_equal(actual.snapshots[2].state["v"].view(np.uint64),
                                  np.full(4, 1 << 63, np.uint64))


@pytest.mark.parametrize("mode", ["fused", "wide", "zero_delay"])
def test_cython_zero_input_draws_preserve_underflow_sign(tmp_path, mode):
    from mlx_backend.model import DEFAULT_EQUATIONS

    equations = DEFAULT_EQUATIONS
    if mode == "wide":
        equations += "\nfoo : volt\nbar : volt\n"
    actual, _ = _run_pair(tmp_path, [.1], target="cython", parameters={
        "eqs": equations, "v_0": -0.0*b.volt, "v_rst": -0.0*b.volt,
        "v_th": 1*b.volt, "t_mbr": (.0001/np.log(10))*b.second,
        "t_dly": (0 if mode == "zero_delay" else .3)*b.ms,
        "r_poi": 0*b.Hz, "r_poi2": 0*b.Hz,
    }, initial_state={"v": np.full(4, -np.nextafter(0., 1.))*b.volt,
                      "g": np.full(4, -0.0)*b.volt})
    # The canonical compiled update underflows to -0. Only rows with an active
    # input execute the subsequent positive-zero addition and become +0.
    np.testing.assert_array_equal(actual.snapshots[0].state["v"].view(np.uint64),
                                  np.array([0, 1 << 63, 0, 1 << 63], np.uint64))


@pytest.mark.parametrize("target", ["numpy", "cython"])
@pytest.mark.parametrize("wide", [False, True])
def test_bounded_input_tapes_preserve_rng_and_partial_windows(tmp_path, monkeypatch, target, wide):
    from mlx_backend import brian_poisson
    from mlx_backend.model import DEFAULT_EQUATIONS

    # Four timesteps per tape cuts across both delay windows and run boundaries.
    # The independent real CPU model sees the same mutations and random stream.
    monkeypatch.setattr(brian_poisson.NativePoisson, "MAX_TAPE_DRAWS", 24)
    prefetch = brian_poisson._UniformTape.prefetch
    counts, decisions = [], []
    can_fuse = brian_poisson.NativePoisson.can_fuse

    def observe_prefetch(self, count):
        counts.append(count)
        return prefetch(self, count)

    def observe_fusion(self, prepared):
        result = can_fuse(self, prepared)
        decisions.append(result)
        return result

    monkeypatch.setattr(brian_poisson._UniformTape, "prefetch", observe_prefetch)
    monkeypatch.setattr(brian_poisson.NativePoisson, "can_fuse", observe_fusion)

    def pause(fixture):
        fixture.inputs[1].active = False
        fixture.monitor.active = False

    def resume(fixture):
        fixture.inputs[1].active = True
        fixture.monitor.active = True

    parameters = {"r_poi": 3700*b.Hz, "r_poi2": 7200*b.Hz, "t_dly": 1.8*b.ms}
    if wide:
        parameters["eqs"] = DEFAULT_EQUATIONS + "\nfoo : volt\nbar : volt"
    actual, device = _run_pair(tmp_path, [0., .15, .2, 1.25, .1, 2.4], target=target,
                               parameters=parameters, mutations={3: pause, 5: resume})
    assert len(counts) > 4 and max(counts) <= 24
    assert decisions and all(value is (not wide) for value in decisions)
    model = device._sessions[actual.network].prepared.model
    assert model.reference_codegen_target == target
    if target == "cython":
        assert model.reference_cpp_compiler == actual.neurons.state_updater.codeobj.compiler
        assert model.reference_compile_args == tuple(actual.neurons.state_updater.codeobj.extra_compile_args)


def test_seeded_cython_empty_initial_eligibility_preserves_unallocated_buffer(tmp_path):
    from brian2.devices.device import runtime_device

    for native in (False, True):
        with cpu_runtime():
            b.prefs.codegen.target = "cython"
            saved_pointer = int(runtime_device.rand_buffer[0])
            saved_index = int(runtime_device.rand_buffer_index[0])
            runtime_device.rand_buffer[0] = 0
            runtime_device.rand_buffer_index[0] = 0
            try:
                from contextlib import nullcontext
                with original_api(workers=False) if native else nullcontext():
                    fixture = build_original_api_fixture(tmp_path / str(native))
                    for source in fixture.inputs:
                        source.active = False
                    before = capture_rng_state()
                    fixture.run_ms(.1)
                    assert_rng_state_equal(capture_rng_state(), before)
            finally:
                runtime_device.rand_buffer[0] = saved_pointer
                runtime_device.rand_buffer_index[0] = saved_index


def test_seeded_cython_tape_preserves_state_across_twenty_thousand_draw_refill(tmp_path):
    # Three eligible N=1 zero-probability channels still consume one uniform
    # each per step. Cross the real Cython 20,000-value refill boundary.
    _run_pair(tmp_path, [.1, 666.7], target="cython",
              parameters={"r_poi": 0*b.Hz, "r_poi2": 0*b.Hz})


@pytest.mark.parametrize("target", ["numpy", "cython"])
def test_actual_state_refractory_lastspike_and_weight_edits_are_imported(tmp_path, target):
    def mutate(fixture):
        fixture.neurons.v = [-44, -48.5, -53.25, -51.125]*b.mV
        fixture.neurons.g = [1.25, -2.5, 3.75, -.125]*b.mV
        fixture.neurons.rfc = [.125, .275, .425, .575]*b.ms
        fixture.neurons.lastspike = [-.1, 0., -.23, -.42]*b.ms
        fixture.neurons.not_refractory = [True, False, True, False]
        fixture.synapses.w[0] = np.nextafter(-1.125, 0)*b.mV
        fixture.synapses.w[1] = .3125*b.mV
        fixture.original.silence([2], fixture.synapses)
    _run_pair(tmp_path, [.2, 0., .7, 3.35], target=target, mutations={1: mutate}, parameters={
        "r_poi": 3700*b.Hz, "r_poi2": 1400*b.Hz, "t_dly": .3*b.ms,
    })


def test_monitor_and_input_active_flags_preserve_real_accumulation(tmp_path):
    def pause(fixture):
        fixture.monitor.active = False
        fixture.inputs[1].active = False
    def resume(fixture):
        fixture.monitor.active = True
        fixture.inputs[1].active = True
    _run_pair(tmp_path, [.7, 1.5, 2.7], mutations={1: pause, 2: resume}, parameters={
        "r_poi": 3700*b.Hz, "r_poi2": 1400*b.Hz,
    })


def test_gpu_count_words_matches_independent_popcount_and_counter_wrap():
    from mlx_backend.window import count_words

    random = np.random.default_rng(71923)
    words = random.integers(0, 2**32, size=(64, 9), dtype=np.uint32)
    words[0, :4] = [0, 0xffffffff, 1, 0x80000000]
    initial = np.array([0, 2**32-1, 2**31-1, 17, 0, 3, 100, 999, 2**32-20], np.uint32)
    expected = np.array([(int(initial[row])+sum(int(word).bit_count() for word in words[:, row]))
                         & 0xffffffff for row in range(words.shape[1])], np.uint32)
    counts = mx.array(initial)
    for block in (words[:1], words[1:31], words[31:]):
        counts = count_words([mx.array(row) for row in block], counts)
    np.testing.assert_array_equal(np.asarray(counts), expected)


@pytest.mark.parametrize("target", ["numpy", "cython"])
@pytest.mark.parametrize("mode", ["counts", "inactive", "mixed"])
@pytest.mark.parametrize("delay_ms", [.1, 3.3])
def test_original_api_unrecorded_counts_and_inactive_monitors(tmp_path, monkeypatch, target, mode, delay_ms):
    from mlx_backend import window

    fixtures = []
    def install(fixture):
        fixture.network.remove(fixture.monitor)
        fixture.monitor = b.SpikeMonitor(fixture.neurons, record=False, name="counts_only_monitor*")
        fixture.monitor.active = mode != "inactive"
        fixture.network.add(fixture.monitor)
        fixture.extra_monitor = None
        if mode == "mixed":
            fixture.extra_monitor = b.SpikeMonitor(fixture.neurons, name="recorded_monitor*")
            fixture.network.add(fixture.extra_monitor)
        fixtures.append(fixture)

    def pause(fixture):
        fixture.monitor.active = False
        if fixture.extra_monitor is not None:
            fixture.extra_monitor.active = False

    def resume(fixture):
        fixture.monitor.active = mode != "inactive"
        if fixture.extra_monitor is not None:
            fixture.extra_monitor.active = True

    if mode != "mixed":
        def forbid_spike_materialization(*args, **kwargs):
            raise AssertionError("Unrecorded runs must not compact/transfer spike records")
        monkeypatch.setattr(window, "collect_words", forbid_spike_materialization)
    actual, _ = _run_pair(tmp_path, [0., .15, .2, 6.7], target=target,
                          mutations={0: install, 2: pause, 3: resume}, parameters={
                              "r_poi": 10000*b.Hz, "r_poi2": 10000*b.Hz,
                              "t_dly": delay_ms*b.ms,
                          })
    assert actual.monitor.num_spikes == (0 if mode == "inactive" else int(actual.monitor.count[:].sum()))
    assert actual.snapshots[-1].spike_i.size == 0
    if mode == "mixed":
        cpu, native = (fixture.extra_monitor for fixture in fixtures)
        for name in ("i", "t", "count", "N"):
            np.testing.assert_array_equal(native.variables[name].get_value(), cpu.variables[name].get_value())


def test_native_execution_never_runs_user_cpu_timestep_codeobjects(tmp_path, monkeypatch):
    with original_runtime(original_api) as device:
        fixture = build_original_api_fixture(tmp_path, parameters={
            "r_poi": 3000*b.Hz, "r_poi2": 700*b.Hz,
        })
        guard_cpu_steps(fixture.network, monkeypatch, 'CPU user timestep code executed', patch_preparation=True)
        fixture.run_ms(1.3)
        assert device.native_steps == 13


def test_poisson_prepass_preserves_many_states_and_dynamic_math(tmp_path):
    from mlx_backend.model import DEFAULT_EQUATIONS, DEFAULT_RESET

    # Per-variable bindings would exceed Metal's argument limit in the RNG
    # eligibility prepass even though the main neuron kernel uses packed state.
    names = [f"observer_{index}" for index in range(30)]
    equations = DEFAULT_EQUATIONS + "\n" + "\n".join(name+" : 1" for name in names)
    equations += "\nspike_counter : integer\nwas_reset : boolean"
    reset = DEFAULT_RESET + "\n" + "\n".join(name+" += .125" for name in names)
    reset += "\nobserver_0 = log(1+observer_0)\nobserver_2 = exp(observer_2)/8"
    reset += "\nobserver_1 += floor(v/mV)\nspike_counter += 1\nwas_reset = True"
    actual, _ = _run_pair(tmp_path, [.2, .35, 3.45], parameters={
        "eqs": equations, "eq_rst": reset,
        "r_poi": 3700*b.Hz, "r_poi2": 1400*b.Hz,
    }, initial_state={name: np.arange(4)/128 for name in names})
    assert actual.snapshots[-1].spike_i.size > 0
    assert np.any(actual.snapshots[-1].state["spike_counter"] > 0)
    assert np.any(actual.snapshots[-1].state["was_reset"])


@pytest.mark.parametrize("target", ["numpy", "cython"])
@pytest.mark.parametrize("dimensions", [1, 2])
def test_original_api_timedarrays_share_prepass_namespace_and_cpu_target(tmp_path, target, dimensions):
    from mlx_backend.model import DEFAULT_EQUATIONS

    drive_values = np.arange(36, dtype=np.float64).reshape(9, 4)/16
    reset_values = np.arange(12, dtype=np.float64).reshape(3, 4)/8
    if dimensions == 1:
        drive_values, reset_values = drive_values[:, 0].copy(), reset_values[:, 0].copy()
    arguments = "t" if dimensions == 1 else "t, i"
    actual, _ = _run_pair(tmp_path, [.15, .2, 1.25, 2.4], target=target, parameters={
        "eqs": DEFAULT_EQUATIONS+"\nobserved : 1",
        "eq_th": f"v > v_th + .01*drive({arguments})*mV",
        "eq_rst": f"v=v_rst; g=0*mV; observed=reset_drive({arguments})",
        "drive": b.TimedArray(drive_values, dt=.15*b.ms),
        "reset_drive": b.TimedArray(reset_values, dt=.2*b.ms),
        "r_poi": 3700*b.Hz, "r_poi2": 1400*b.Hz,
    })
    assert actual.snapshots[-1].spike_i.size > 0
    assert np.any(actual.snapshots[-1].state["observed"] > 0)


@pytest.mark.parametrize("target", ["numpy", "cython"])
@pytest.mark.parametrize("wide", [False, True])
def test_original_api_bounded_tapes_share_timedarray_and_math_headers(tmp_path, monkeypatch, target, wide):
    from mlx_backend.brian_poisson import NativePoisson
    from mlx_backend.model import DEFAULT_EQUATIONS

    monkeypatch.setattr(NativePoisson, "MAX_TAPE_DRAWS", 24)
    equations = DEFAULT_EQUATIONS + "\nobserved : 1"
    if wide:
        equations += "\nfoo : volt\nbar : volt"
    actual, device = _run_pair(tmp_path, [.15, .2, 1.25, 2.4], target=target, parameters={
        "eqs": equations,
        "eq_th": "v > v_th + drive(t)*mV/128",
        "eq_rst": "v=v_rst; g=0*mV; observed=log10(1+observed)"
                  "+sinh(observed/32)+cosh(observed/32)+tanh(observed/32)"
                  "+expm1(observed/128)+log1p(observed/128)+exprel(observed/128)",
        "drive": b.TimedArray(np.arange(9, dtype=np.float64)/64, dt=.15*b.ms),
        "r_poi": 3700*b.Hz, "r_poi2": 1400*b.Hz,
    }, initial_state={"observed": np.array([.125, np.nextafter(.5, 0), .25, 0.])})
    prepared = device._sessions[actual.network].prepared
    assert prepared.program.shared_timed_calls
    assert NativePoisson.can_fuse(device._sessions[actual.network].sampler, prepared) is (not wide)
    assert actual.snapshots[-1].spike_i.size > 0
    assert np.any(actual.snapshots[-1].state["observed"] > 1)


@pytest.mark.parametrize("delay_ms", [0., 1.8])
def test_cython_preparation_keeps_real_source_flags_rng_and_after_run(tmp_path, monkeypatch, delay_ms):
    from brian2.codegen.runtime.cython_rt.cython_rt import CythonCodeObject
    parameters = {"r_poi": 10000*b.Hz, "r_poi2": 10000*b.Hz, "t_dly": delay_ms*b.ms}
    durations = [.15, .15, 4.1]
    with cpu_runtime():
        b.prefs.codegen.target = "cython"
        expected = run_original_api(tmp_path/"cpu", durations, parameters=parameters)
        source = dict(expected.neurons.state_updater.codeobj.code)
        flags = expected.neurons.state_updater.codeobj.extra_compile_args
        expected_rng = capture_rng_state()
    original_compile = CythonCodeObject.compile_block
    raw_rng_compiles = []
    def compile_block(self, block):
        device = get_device()
        assert not (type(device) is MLXDevice and device._preparing_native), \
            "Native preparation must not compile unused CPU timestep blocks"
        if type(device) is MLXDevice and self.owner.name.startswith("mlx_"):
            raw_rng_compiles.append(self.owner.name)
        return original_compile(self, block)
    monkeypatch.setattr(CythonCodeObject, "compile_block", compile_block)
    with cpu_runtime():
        b.prefs.codegen.target = "cython"
        with original_api(workers=False):
            actual = run_original_api(tmp_path/"native", durations, parameters=parameters)
            codeobj = actual.neurons.state_updater.codeobj
            assert type(codeobj) is CythonCodeObject
            assert dict(codeobj.code) == source
            assert codeobj.extra_compile_args == flags
            assert all(value is None for value in codeobj.compiled_code.values())
            assert raw_rng_compiles and all(name.startswith("mlx_") for name in raw_rng_compiles)
            for left,right in zip(actual.snapshots, expected.snapshots):
                assert_snapshot_equal(left,right)
            assert_rng_state_equal(capture_rng_state(),expected_rng)


@pytest.mark.parametrize("unsupported", ["extra_runner", "inactive_reset", "runner_phase", "store", "profile", "report"])
def test_unsupported_contracts_fail_without_running_timesteps(tmp_path, unsupported):
    with original_runtime(original_api) as device:
        fixture = build_original_api_fixture(tmp_path)
        before = fixture.capture()
        if unsupported == "extra_runner":
            fixture.network.add(fixture.neurons.run_regularly("v += 1*mV"))
            run = lambda: fixture.run_ms(.2)
        elif unsupported == "inactive_reset":
            fixture.neurons.resetter["spike"].active = False
            run = lambda: fixture.run_ms(.2)
        elif unsupported == "runner_phase":
            fixture.neurons.state_updater.when = "end"
            run = lambda: fixture.run_ms(.2)
        elif unsupported == "store":
            run = fixture.network.store
        elif unsupported == "profile":
            run = lambda: fixture.network.run(.2*b.ms, profile=True)
        else:
            run = lambda: fixture.network.run(.2*b.ms, report=lambda *args: None)
        with pytest.raises(UnsupportedBrianNetwork):
            run()
        assert_snapshot_equal(fixture.capture(), before)
        assert device.native_steps == 0


@pytest.mark.parametrize("overrides", [
    {"v_0": -61*b.mV, "v_rst": -57*b.mV, "v_th": -43*b.mV,
     "t_mbr": 17*b.ms, "tau": 3*b.ms, "w_syn": .4*b.mV, "f_poi": 123},
    {"t_mbr": 24*b.ms, "tau": 7*b.ms, "t_rfc": 3.15*b.ms},
])
def test_default_cython_arithmetic_policy_preserves_changed_original_constants(tmp_path, overrides):
    _run_pair(tmp_path, [.15, 1.05, 4.7], target="cython", parameters={
        **overrides, "r_poi": 3700*b.Hz, "r_poi2": 1400*b.Hz,
    })


@pytest.mark.parametrize("dt_ms", [.13, .23])
def test_default_cython_refractory_and_clock_policy_matches_nondefault_dt(tmp_path, dt_ms):
    _run_pair(tmp_path, [.15, .15, 1.8, 3.7], target="cython", dt_ms=dt_ms,
              parameters={"r_poi": 3700*b.Hz, "r_poi2": 1400*b.Hz})


@pytest.mark.parametrize("target", ["numpy", "cython"])
def test_actual_group_dtypes_survive_later_global_preference_changes(tmp_path, target):
    def change_defaults(_):
        b.prefs.core.default_float_dtype = np.float32
    _run_pair(tmp_path, [.1, .1, 4.5], target=target, mutations={0: change_defaults},
              parameters={"r_poi": 3700*b.Hz, "r_poi2": 1400*b.Hz},
              initial_state={"g": np.array([1.25, -2.5, .375, 0])*b.mV,
                             "v": np.array([-51.25, -50.75, -52., -53.])*b.mV})


@pytest.mark.parametrize("dimensions", [1, 2])
def test_original_api_timedarray_preserves_later_temporary_dtype(tmp_path, dimensions):
    def change_defaults(_):
        b.prefs.core.default_float_dtype = np.float32
    values = np.arange(36, dtype=np.float64).reshape(9, 4)/128
    if dimensions == 1:
        values = values[:, 0].copy()
    arguments = "t" if dimensions == 1 else "t, i"
    _run_pair(tmp_path, [.1, .1, 4.5], target="cython", mutations={0: change_defaults},
              parameters={"drive": b.TimedArray(values, dt=.15*b.ms),
                          "eq_th": f"v > v_th + drive({arguments})*mV",
                          "r_poi": 3700*b.Hz, "r_poi2": 1400*b.Hz},
              initial_state={"g": np.array([1.25, -2.5, .375, 0])*b.mV,
                             "v": np.array([-51.25, -50.75, -52., -53.])*b.mV})


def _worker_original_trial(paths, parameters):
    from brian2.devices.device import get_device
    from mlx_backend.brian_device import MLXDevice
    from original_api_reference import load_original_model
    import os

    device = get_device()
    assert type(device) is MLXDevice
    before = device.native_runs
    result = load_original_model().run_trial([0, 0], [2], [3], *paths, parameters)
    assert device.native_runs == before+1
    return os.getpid(), {index: np.asarray(times) for index, times in result.items()}


def test_original_run_trial_and_loky_workers_execute_native_device(tmp_path):
    from joblib import Parallel, delayed, parallel_backend
    from joblib.externals.loky import get_reusable_executor
    from original_api_reference import load_original_model, write_tiny_dataset

    with cpu_runtime():
        paths = write_tiny_dataset(tmp_path)
        original = load_original_model()
        params = {**original.default_params, "r_poi": 10000*b.Hz,
                  "r_poi2": 0*b.Hz, "t_run": .7*b.ms}
        expected = original.run_trial([0, 0], [2], [3], *paths, params)
        with original_api(workers=True) as device:
            actual = original.run_trial([0, 0], [2], [3], *paths, params)
            assert device.native_runs == 1
            assert actual.keys() == expected.keys()
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
            try:
                with parallel_backend("loky", n_jobs=2):
                    results = Parallel()(delayed(_worker_original_trial)(paths, params)
                                         for _ in range(2))
                for _, trains in results:
                    assert trains.keys() == expected.keys()
                    for key in expected:
                        np.testing.assert_array_equal(trains[key], np.asarray(expected[key]))
            finally:
                get_reusable_executor().shutdown(wait=True, kill_workers=True)


@pytest.mark.parametrize("workers", [1, 2])
def test_original_run_exp_writes_matching_string_id_parquet_without_source_changes(tmp_path, workers):
    import pandas as pd
    from joblib.externals.loky import get_reusable_executor
    from original_api_reference import NEURON_IDS, load_original_model, write_tiny_dataset

    original = load_original_model()
    paths = write_tiny_dataset(tmp_path)
    cpu_path, native_path = tmp_path / "cpu", tmp_path / "native"
    cpu_path.mkdir()
    native_path.mkdir()
    params = {**original.default_params, "r_poi": 10000*b.Hz,
              "r_poi2": 0*b.Hz, "t_run": .7*b.ms, "n_run": 3}
    with cpu_runtime():
        original.run_exp("native-contract", [NEURON_IDS[0]], cpu_path, *paths,
                         params=params, neu_slnc=[NEURON_IDS[3]], n_proc=1)
        with original_api(workers=True) as device:
            try:
                original.run_exp("native-contract", [NEURON_IDS[0]], native_path, *paths,
                    params=params, neu_slnc=[NEURON_IDS[3]], n_proc=workers)
                if workers == 1:
                    assert device.native_runs == params["n_run"]
                old_runs = device.native_runs
                original.run_exp("native-contract", [NEURON_IDS[0]], native_path, *paths,
                    params=params, n_proc=workers, force_overwrite=False)
                assert device.native_runs == old_runs
                if workers == 1:
                    # The actual default is True, despite the upstream docstring.
                    # Overwrite an invalid old file without an explicit flag.
                    (native_path / "native-contract.parquet").write_bytes(b"stale output")
                    original.run_exp("native-contract", [NEURON_IDS[0]], native_path, *paths,
                        params=params, neu_slnc=[NEURON_IDS[3]], n_proc=workers)
                    assert device.native_runs == old_runs + params["n_run"]
            finally:
                get_reusable_executor().shutdown(wait=True, kill_workers=True)
    expected = pd.read_parquet(cpu_path / "native-contract.parquet")
    actual = pd.read_parquet(native_path / "native-contract.parquet")
    order = ["trial", "t", "flywire_id"]
    pd.testing.assert_frame_equal(actual.sort_values(order).reset_index(drop=True),
                                  expected.sort_values(order).reset_index(drop=True))
    assert set(actual.flywire_id) == {NEURON_IDS[0]}
