"""CPU-only configuration checks against the frozen application's Brian model."""

import importlib.util
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlx_backend.model import TimedArrayValue, resolve_model


@pytest.fixture
def reference(tmp_path, monkeypatch):
    brian = pytest.importorskip("brian2")
    monkeypatch.setitem(brian.prefs, "codegen.target", "numpy")
    path = Path(__file__).resolve().parents[1] / "code/paper-phil-drosophila/model.py"
    spec = importlib.util.spec_from_file_location("eon_model_config_reference", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    completeness, connectivity = tmp_path / "neurons.csv", tmp_path / "edges.parquet"
    pd.DataFrame(index=["first", "second"]).to_csv(completeness)
    pd.DataFrame({
        "Presynaptic_Index": [0], "Postsynaptic_Index": [1],
        "Excitatory x Connectivity": [1.25],
    }).to_parquet(connectivity)
    return brian, original, (completeness, connectivity)


def test_untouched_default_path_does_not_import_brian():
    code = """
import importlib.abc
import sys
class NoBrian(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'brian2' or fullname.startswith('brian2.'):
            raise AssertionError('Default configuration imported Brian')
sys.meta_path.insert(0, NoBrian())
from mlx_backend.model import resolve_model
m = resolve_model()
assert m.default_specialization and m.analysis is None
assert m.delay_steps == 18 and m.refractory_steps == 22
assert not any(name.startswith('brian2') for name in sys.modules)
"""
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-c", code], cwd=root / "code", check=True,
        capture_output=True, text=True,
    )


def test_original_defaults_have_identical_values_units_and_identity(reference):
    brian, original, paths = reference
    resolved = resolve_model(original.default_params, dt_ms=.1)
    default = resolve_model(dt_ms=.1)
    neurons, synapses, _ = original.create_model(*paths, original.default_params)
    assert resolved.signature == default.signature
    assert resolved.default_specialization
    for name in ("v_0", "v_rst", "v_th", "t_mbr", "tau", "t_rfc", "t_dly",
                 "w_syn", "r_poi", "r_poi2", "f_poi"):
        assert getattr(resolved, name) == float(original.default_params[name])
    assert resolved.v_0 == neurons.v_[0]
    assert resolved.t_rfc == neurons.rfc_[0]
    assert resolved.t_dly == float(synapses.delay_[:])
    assert synapses.w_[0] == 1.25 * resolved.w_syn
    reconstructed = resolved.brian_namespace()
    for name, value in original.default_params.items():
        if name not in ("eqs", "eq_th", "eq_rst"):
            assert brian.get_dimensions(reconstructed[name]) == brian.get_dimensions(value)
            assert float(reconstructed[name]) == float(value)
    assert "w" not in resolved.analysis.variables  # reset's w is a local temporary


def test_changed_constants_match_original_initial_state_synapses_and_inputs(reference):
    brian, original, paths = reference
    parameters = dict(original.default_params)
    parameters.update(
        v_0=-61*brian.mV, v_rst=-57*brian.mV, v_th=-43*brian.mV,
        t_mbr=17*brian.ms, tau=3*brian.ms, t_rfc=.35*brian.ms,
        t_dly=.25*brian.ms, w_syn=.4*brian.mV,
        r_poi=73*brian.Hz, r_poi2=29*brian.Hz, f_poi=123,
    )
    model = resolve_model(parameters)
    neurons, synapses, _ = original.create_model(*paths, parameters)
    inputs, _ = original.poi(neurons, [0], [1], parameters)
    assert neurons.v_[0] == model.v_0
    assert synapses.w_[0] == 1.25 * model.w_syn
    assert float(synapses.delay_[:]) == model.t_dly
    assert [float(channel.rate) for channel in inputs] == [model.r_poi, model.r_poi2]
    assert [float(channel.weight) for channel in inputs] == [model.poisson_weight_v] * 2
    assert not model.default_specialization


def test_stimulus_rates_do_not_disqualify_the_unchanged_state_kernel(reference):
    brian, _, _ = reference
    base = resolve_model()
    rates = resolve_model({"r_poi": 10000 * brian.Hz, "r_poi2": 29 * brian.Hz})
    assert rates.state_specialization and not rates.default_specialization
    assert rates.signature != base.signature
    assert not resolve_model({"f_poi": 251}).state_specialization
    assert not resolve_model({"w_syn": .3*brian.mV}).state_specialization
    assert not resolve_model({"t_dly": 1*brian.ms}).state_specialization
    assert not resolve_model({"eq_th": "v >= v_th"}).state_specialization


def test_ambient_clock_and_dtype_match_original_group(reference):
    brian, original, paths = reference
    previous_dt = brian.defaultclock.dt
    previous_dtype = brian.prefs.core.default_float_dtype
    try:
        brian.defaultclock.dt = .23 * brian.ms
        brian.prefs.core.default_float_dtype = np.float32
        neurons, _, _ = original.create_model(*paths, original.default_params)
        resolved = resolve_model()
        assert resolved.dt_s == neurons.clock.dt_
        assert resolved.state_dtype == np.dtype(neurons.variables["v"].dtype).name
        assert resolved.state_dtype == "float32" and not resolved.default_specialization
        explicit = resolve_model({"dt": .7 * brian.ms}, dt_ms=.05)
        assert explicit.dt_s == float(.05 * brian.ms)
        assert explicit.source_namespace["dt"] == float(.7 * brian.ms)
    finally:
        brian.defaultclock.dt = previous_dt
        brian.prefs.core.default_float_dtype = previous_dtype


def test_other_ambient_model_preferences_are_part_of_identity(reference, monkeypatch):
    brian, original, paths = reference
    baseline = resolve_model()
    monkeypatch.setitem(brian.prefs, "core.default_integer_dtype", np.int64)
    params = {"eqs": baseline.eqs + "count : integer\n"}
    integer_model = resolve_model(params)
    neurons, _, _ = original.create_model(*paths, {**original.default_params, **params})
    assert integer_model.integer_dtype == np.dtype(neurons.variables["count"].dtype).name == "int64"
    assert integer_model.analysis.variables["count"].dtype == "int64"
    monkeypatch.setitem(brian.prefs, "legacy.refractory_timing", True)
    legacy = resolve_model()
    assert legacy.signature != baseline.signature and not legacy.default_specialization
    # The original Brian 2.8 string-duration path is broken; the native model
    # locally supplies its missing assignment with the Quantity-path predicate.
    with pytest.raises(ValueError, match="Parsing the statement failed"):
        neurons.state_updater.update_abstract_code(run_namespace={})
    analysed = resolve_model({})
    assert analysed.signature == legacy.signature
    assert analysed.analysis.state_abstract.startswith("not_refractory = (t - lastspike) > rfc\n")
    with pytest.raises(ValueError, match="no clock-independent step count"):
        _ = legacy.refractory_steps
    reordered = ["start", "groups", "thresholds", "resets", "synapses", "end"]
    monkeypatch.setitem(brian.prefs, "core.network.default_schedule", reordered)
    scheduled = resolve_model()
    assert scheduled.schedule == tuple(brian.Network(neurons).schedule) == tuple(reordered)
    assert scheduled.signature != legacy.signature


@pytest.mark.parametrize("dt_ms", [.05, .1, .2, .3])
def test_horizons_match_independent_brian_clock(reference, dt_ms):
    brian, _, _ = reference
    clock = brian.Clock(dt=dt_ms * brian.ms)
    resolved = resolve_model(dt_ms=dt_ms)
    durations = [0., .5 * clock.dt_, clock.dt_, 1.5 * clock.dt_, .0012345, 1000.]
    durations += [2 * clock.dt_ + delta * clock.dt_ for delta in [-2e-4, -.5e-4, .5e-4, 2e-4]]
    for duration in durations:
        assert resolved.computed_steps(duration) == clock._calc_timestep(duration)


@pytest.mark.parametrize("delay_ms", [0., .049, .05, .15, .25, 1.8, 9.999])
def test_delay_matches_actual_compiled_brian_queue(reference, delay_ms):
    brian, _, _ = reference
    queue_module = pytest.importorskip("brian2.synapses.cythonspikequeue")
    resolved = resolve_model({"t_dly": delay_ms * brian.ms}, dt_ms=.1)
    queue = queue_module.SpikeQueue(0, 1)
    queue.prepare(np.array([float(delay_ms * brian.ms)]), resolved.dt_s,
                  np.array([0], dtype=np.int32))
    queue.push(np.array([0], dtype=np.int32))
    deliveries = []
    for step in range(110):
        deliveries.extend([step] * len(queue.peek()))
        queue.advance()
    assert deliveries == [resolved.delay_steps]


@pytest.mark.parametrize("refractory_ms", [0., .0998, .09995, .15, 2.2, 2.29995])
def test_refractory_quantization_matches_brian_timestep(reference, refractory_ms):
    brian, _, _ = reference
    from brian2.core.functions import timestep

    resolved = resolve_model({"t_rfc": refractory_ms * brian.ms}, dt_ms=.1)
    assert resolved.refractory_steps == timestep(refractory_ms * brian.ms, .1 * brian.ms)


def test_custom_expressions_match_original_brian_analysis(reference):
    brian, original, paths = reference
    params = dict(original.default_params)
    params.update(
        eqs=original.default_params["eqs"].replace("v_0 - v + g", "v_0 - v + gain*g - a")
        + "da/dt = -a/(2*tau) : volt\ncount : integer\nflag : boolean\n",
        gain=.5,
        eq_th="(v > v_th + abs(g)*0.01) and (i != 1)",
        eq_rst="v = v_rst; g *= 0.25; a += 1*mV; count += 1; flag = not flag",
    )
    neurons, _, _ = original.create_model(*paths, params)
    neurons.equations.check_units(neurons, run_namespace={})
    expected = []
    for runner in (neurons.state_updater, neurons.thresholder["spike"], neurons.resetter["spike"]):
        runner.update_abstract_code(run_namespace={})
        expected.append(runner.abstract_code)
    actual = resolve_model(params)
    assert [actual.analysis.state_abstract, actual.analysis.threshold_abstract,
            actual.analysis.reset_abstract] == expected
    assert actual.analysis.variables["count"].dtype == np.dtype(neurons.variables["count"].dtype).name
    assert actual.analysis.variables["flag"].dtype == "bool"
    assert actual.analysis.variables["v"].conditional_write == "not_refractory"
    assert actual.analysis.variables["a"].conditional_write is None
    assert actual.eqs == params["eqs"] and actual.eq_rst == params["eq_rst"]
    assert not actual.default_specialization


def test_equations_objects_accepted_by_upstream_are_snapshotted(reference):
    brian, original, paths = reference
    equations = brian.Equations(original.default_params["eqs"] + "extra : 1\n")
    neurons, _, _ = original.create_model(*paths, {**original.default_params, "eqs": equations})
    resolved = resolve_model({"eqs": equations})
    assert "extra" in neurons.equations and "extra" in resolved.analysis.variables
    assert isinstance(resolved.source_namespace["eqs"], str)


@pytest.mark.parametrize("name,unit", [
    ("v_0", "ms"), ("v_rst", "ms"), ("v_th", "ms"), ("t_mbr", "mV"),
    ("tau", "mV"), ("t_rfc", "mV"), ("t_dly", "mV"), ("w_syn", "ms"),
    ("r_poi", "ms"), ("r_poi2", "mV"), ("f_poi", "mV"),
])
def test_dimensional_parameters_use_brian_unit_validation(reference, name, unit):
    brian, _, _ = reference
    with pytest.raises(brian.DimensionMismatchError):
        resolve_model({name: 2 * getattr(brian, unit)})


@pytest.mark.parametrize("overrides", [
    {"eq_th": "v > 1*ms"}, {"eq_rst": "v = 2*ms"},
    {"eqs": "dv/dt = -v/(1*mV) : volt\ng : volt\nrfc : second"},
])
def test_custom_expression_dimensions_are_checked(reference, overrides):
    brian, _, _ = reference
    with pytest.raises(brian.DimensionMismatchError):
        resolve_model(overrides)


def test_namespace_snapshots_and_cache_identity_cover_model_inputs(reference):
    brian, _, _ = reference
    caller = np.array([1., 2., 3.]) * brian.mV
    params = {"extra_values": caller, "gain": .5}
    first = resolve_model(params)
    original_identity = first.signature
    caller[0] = 8 * brian.mV
    params["gain"] = .75
    assert first.source_namespace["extra_values"][0] == .001
    assert first.signature == original_identity
    assert first.signature != resolve_model(params).signature
    with pytest.raises(ValueError):
        first.source_namespace["extra_values"].setflags(write=True)
    with pytest.raises(TypeError):
        first.source_namespace["gain"] = 1
    with pytest.raises(FrozenInstanceError):
        first.v_0 = 0
    assert resolve_model({"w_syn": 1 * brian.mV}).signature == resolve_model({"w_syn": .001 * brian.volt}).signature
    base = resolve_model()
    variants = [
        {"v_0": -51*brian.mV}, {"v_rst": -51*brian.mV}, {"v_th": -44*brian.mV},
        {"t_mbr": 19*brian.ms}, {"tau": 4*brian.ms}, {"t_rfc": 2*brian.ms},
        {"t_dly": 1*brian.ms}, {"w_syn": .3*brian.mV}, {"r_poi": 151*brian.Hz},
        {"r_poi2": 1*brian.Hz}, {"f_poi": 251}, {"eq_th": "v >= v_th"},
        {"eq_rst": "v = v_rst; g = 1*mV"}, {"eqs": base.eqs + "x : 1\n"},
    ]
    identities = {base.signature, resolve_model(dt_ms=.2).signature}
    identities.update(resolve_model(params).signature for params in variants)
    assert len(identities) == len(variants) + 2


def test_token_boundaries_cannot_collide_in_cache_identity(reference):
    # Stripping all whitespace would merge these distinct valid thresholds.
    namespace = {"flag": True, "other": True, "flagandother": False}
    first = resolve_model({**namespace, "eq_th": "flag and other"})
    second = resolve_model({**namespace, "eq_th": "flagandother"})
    assert first.signature != second.signature


def test_timedarray_snapshot_retains_values_dimensions_and_interval(reference):
    brian, _, _ = reference
    drive = brian.TimedArray(np.array([0., 1., 2.]) * brian.mV, dt=1 * brian.ms)
    eqs = resolve_model().eqs.replace("v_0 - v + g", "v_0 - v + g + drive(t)")
    resolved = resolve_model({"drive": drive, "eqs": eqs})
    assert isinstance(resolved.source_namespace["drive"], TimedArrayValue)
    assert resolved.cacheable and not resolved.pending_namespace
    drive.values[0] = .1
    reconstructed = resolved.brian_namespace()["drive"]
    assert reconstructed(0 * brian.ms) == 0 * brian.mV
    assert reconstructed(1 * brian.ms) == 1 * brian.mV
    assert reconstructed.dt == .001
    reconstructed.values[0] = .2
    assert resolved.brian_namespace()["drive"](0 * brian.ms) == 0 * brian.mV
    assert resolved.signature != resolve_model({"drive": drive, "eqs": eqs}).signature


def test_callbacks_are_preserved_as_pending_without_execution_or_cache_claim(reference):
    brian, _, _ = reference
    def callback(value):
        raise AssertionError("Model resolution executed a host callback")
    function = brian.Function(callback, arg_units=[brian.volt], return_unit=brian.volt)
    resolved = resolve_model({"custom_abs": function, "eq_th": "custom_abs(v) > v_th"})
    assert resolved.source_namespace["custom_abs"] is function
    assert resolved.pending_namespace == ("custom_abs",)
    assert resolved.signature is None and not resolved.cacheable
    assert resolved.analysis is None and not resolved.default_specialization
    with pytest.raises(TypeError, match="cannot be cached"):
        hash(resolved)


def test_duration_quantities_keep_their_units(reference):
    brian, _, _ = reference
    assert resolve_model(dt_ms=.1).computed_steps(.15 * brian.ms) == 2
    with pytest.raises(brian.DimensionMismatchError):
        resolve_model().computed_steps(1 * brian.Hz)


@pytest.mark.parametrize("duration", [-1., float("nan"), float("inf")])
def test_invalid_horizons_fail_before_execution(duration):
    with pytest.raises(ValueError, match="Duration"):
        resolve_model().computed_steps(duration)


@pytest.mark.parametrize("dt_ms", [0., -1., float("nan"), float("inf")])
def test_invalid_clock_steps_fail_before_execution(dt_ms):
    with pytest.raises(ValueError, match="Timestep"):
        resolve_model(dt_ms=dt_ms)
