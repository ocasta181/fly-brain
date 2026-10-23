"""Source-frozen Eon interactions against unchanged upstream CPU objects.

The 18 configurations were frozen before backend/test-body inspection, at
executions/confidence-holdouts-20261021/frozen-design.json, SHA256
b844921d1b51cc6de35120d74290bdb7d5bd55bc2df85d2d9fdfd707a57fc45e.
These are engineering holdouts, not samples from a compatibility distribution.
"""
from native_execution import forbid_cpu_steps, require_metal

from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@dataclass(frozen=True)
class Case:
    name: str
    exc: tuple = ()
    exc2: tuple = ()
    silenced: tuple = ()
    parameters: dict = field(default_factory=dict)
    steps: tuple = (16,)
    dtype: type = np.float64
    identities: tuple | None = None
    edges: tuple | None = None
    record: bool = True
    error: bool = False
    preflight_error: bool = False


CASES = (
    Case("H01-empty-selection", steps=(8,)),
    Case("H02-zero-rate-duplicates", (3, 3), (3,), parameters={
        "r_poi_Hz": 0., "r_poi2_Hz": 0.}, steps=(8,)),
    Case("H03-second-channel-only", exc2=(1, 4), parameters={"r_poi2_Hz": 8000.}),
    Case("H04-overlap-duplicate-stochastic", (4, 1, 4), (1, 4, 2)),
    Case("H05-excited-and-silenced", (0, 2), (2,), (0, 2), {
        "r_poi_Hz": 8000., "r_poi2_Hz": 8000.}),
    Case("H06-repeated-edges-silence", (0, 3), (0,), (0, 0, 4),
         {"t_dly_ms": 0.}, edges=((0, 1, 2.5), (0, 1, -1.25), (0, 2, 0.),
         (3, 1, 7.75), (3, 1, -2.25), (4, 2, 3.))),
    Case("H07-alternate-large-identities", (2, 0), (1,), (3,),
         identities=(18446744073709551603, 9223372036854775811, 17, 720575940000000013),
         edges=((2, 0, 2.125), (0, 3, -3.75), (1, 2, 5.5), (3, 1, .125))),
    Case("H08-delay-below-half", (0,), (4,), parameters={
        "t_dly_ms": .187499999, "r_poi_Hz": 8000.}),
    Case("H09-delay-half", (0,), (4,), parameters={
        "t_dly_ms": .1875, "r_poi_Hz": 8000.}),
    Case("H10-refractory-not-integral", (0, 4), parameters={
        "r_poi_Hz": 8000., "t_rfc_ms": .312499999, "w_syn_mV": 2., "f_poi": 8.}),
    Case("H11-configured-threshold-reset", (1, 2), (1,), parameters={
        "eq_th": "v >= v_th", "eq_rst": "v = v_rst + 0.125*mV; g = -0.5*g + 0.25*mV",
        "v_0_mV": -47.5}),
    Case("H12-configured-coupled-state", (0, 3), (3,), (4,), {
        "eqs": "dv/dt = (v_0 - v + 0.75*g) / t_mbr : volt (unless refractory)\n"
               "dg/dt = -g / tau + 0.375*mV/ms : volt (unless refractory)\n"
               "rfc : second\ncount : integer",
        "eq_rst": "v = v_rst; g = -0.25*g; count += 1", "t_dly_ms": .25}),
    Case("H13-float32-physical-state", (1, 4), (1, 2), (0,), {
        "v_0_mV": -50.125, "v_rst_mV": -51.375, "tau_ms": .8125,
        "w_syn_mV": .21875, "f_poi": 32., "t_dly_ms": 0.}, dtype=np.float32),
    Case("H14-rng-threshold-reset", (3, 0, 3), (0, 2), (1,), {
        "eq_th": "v > v_th + 0.125*mV*rand()",
        "eq_rst": "v = v_rst + 0.25*mV*rand(); g = 0*mV"}),
    Case("H15-split-continuation", (4, 0, 4), (0, 3), (2,), steps=(3, 2, 7)),
    Case("H16-recording-disabled", (1, 3, 1), (3,), (4,), record=False),
    Case("H17-reset-error-prefix", (0,), parameters={
        "eqs": "dv/dt = 0*volt/second : volt\ndg/dt = 0*volt/second : volt\nrfc : second",
        "eq_th": "v > v_th", "eq_rst": "v = v_rst; g = g/(int(t/dt)-3)",
        "r_poi_Hz": 8000.}, steps=(8,), error=True, preflight_error=True),
    Case("H18-output-trains-dataframe", (2, 4), (2,), (1,), {
        "r_poi_Hz": 8000., "r_poi2_Hz": 0.}, steps=(8,)),
    # Post-freeze correction: Brian's linear updater rejects H17's all-zero
    # differential system during preparation, before the intended reset fault.
    Case("H17r-post-freeze-reset-error-prefix", (0,), parameters={
        "eq_th": "v > v_th", "eq_rst": "v = v_rst; g = g/(int(t/dt)-3)",
        "r_poi_Hz": 8000.}, steps=(8,), error=True),
)


IDENTITIES = (720575940000000057, 720575940000000003, 720575940000000091,
              720575940000000024, 720575940000000011)
EDGES = ((0, 1, 3.5), (0, 2, -2.25), (1, 3, 1.125), (2, 4, 4.5),
         (3, 0, 2.75), (4, 2, -.625))
PARAMETERS = dict(v_0_mV=-51.25, v_rst_mV=-52.75, v_th_mV=-47.5,
                  t_mbr_ms=1.75, tau_ms=.625, t_rfc_ms=.3125, t_dly_ms=.1875,
                  w_syn_mV=.375, f_poi=24., r_poi_Hz=3000., r_poi2_Hz=5000.)
DT_MS = .125


def _build(directory, case, b):
    from original_api_reference import load_original_model

    original = load_original_model()
    # The oracle must keep the application's exact upstream source.
    source = Path(original.__file__).read_bytes()
    assert sha256(source).hexdigest() == "6e2cb5aab4ddf08cde2ca05cc49e87dd252c0358afe0dad17314211837b1f539"
    directory.mkdir(parents=True, exist_ok=True)
    paths = directory/"neurons.csv", directory/"edges.parquet"
    identities = case.identities or IDENTITIES
    pd.DataFrame({"label": [f"holdout-{i}" for i in range(len(identities))]},
                 index=pd.Index(identities, name="flywire_id")).to_csv(paths[0])
    edges = case.edges or EDGES
    pd.DataFrame({"Presynaptic_Index": np.array([e[0] for e in edges], np.int64),
                  "Postsynaptic_Index": np.array([e[1] for e in edges], np.int64),
                  "Excitatory x Connectivity": [e[2] for e in edges]}).to_parquet(paths[1])
    parameters = dict(original.default_params)
    for name, value in {**PARAMETERS, **case.parameters}.items():
        if name.endswith("_mV"):
            name, value = name[:-3], value*b.mV
        elif name.endswith("_ms"):
            name, value = name[:-3], value*b.ms
        elif name.endswith("_Hz"):
            name, value = name[:-3], value*b.Hz
        parameters[name] = value
    neurons, synapses, monitor = original.create_model(*paths, parameters)
    inputs, returned = original.poi(neurons, list(case.exc), list(case.exc2), parameters)
    assert returned is neurons and len(inputs) == len(case.exc)+len(case.exc2)
    assert original.silence(list(case.silenced), synapses) is synapses
    objects = [neurons, synapses, *inputs]
    if case.record:
        objects.append(monitor)
    return original, identities, neurons, synapses, monitor if case.record else None, b.Network(*objects)


def _snapshot(neurons, synapses, monitor, network):
    def array(owner, name):
        return np.array(owner.variables[name].get_value(), copy=True)

    names = [name for name, equation in neurons.equations.items()
             if equation.type != "subexpression"]
    names = tuple(dict.fromkeys([*names, "lastspike", "not_refractory"]))
    result = {"state."+name: array(neurons, name) for name in names}
    result["current_spikes"] = np.array(neurons.spikes, copy=True)
    result.update({"synapses."+name: array(synapses, name)
                   for name in ("_synaptic_pre", "_synaptic_post", "w")})
    result["delay"] = array(synapses.pre, "delay")
    for name in ("t", "timestep", "dt"):
        result["clock."+name] = array(neurons.clock, name)
    result["network.t"] = np.asarray(network.t_)
    if monitor is not None:
        result.update({"monitor."+name: array(monitor, name) for name in ("i", "t", "count", "N")})
    return result


def _exact(actual, expected):
    assert actual.keys() == expected.keys()
    for name, wanted in expected.items():
        observed = actual[name]
        assert observed.dtype == wanted.dtype and observed.shape == wanted.shape, name
        if observed.tobytes() != wanted.tobytes():
            unequal = np.flatnonzero(observed.reshape(-1).view(np.uint8) !=
                                     wanted.reshape(-1).view(np.uint8))
            index = int(unequal[0])//wanted.dtype.itemsize
            actual_item, expected_item = observed.reshape(-1)[index], wanted.reshape(-1)[index]
            raise AssertionError(f"{name}[{index}]: {actual_item!r} ({actual_item.tobytes().hex()}) != "
                                 f"{expected_item!r} ({expected_item.tobytes().hex()})")


def _forbid_cpu_steps(network, patch):
    forbid_cpu_steps(network, patch, 'A CPU timestep executed in the native holdout', patch_preparation=True)


def _run(directory, case, monkeypatch, *, native, whole=False):
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from original_api_reference import capture_rng_state, cpu_runtime

    snapshots, rng_states, errors = [], [], []
    with cpu_runtime(dt_ms=DT_MS, seed=71429):
        b.prefs.core.default_float_dtype = case.dtype
        with original_api(workers=False) if native else nullcontext() as device:
            original, identities, neurons, synapses, monitor, network = _build(directory, case, b)
            if native:
                _forbid_cpu_steps(network, monkeypatch)
            durations = (sum(case.steps),) if whole else case.steps
            for steps in durations:
                with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
                    if case.error:
                        with pytest.raises(b.BrianObjectException) as error:
                            network.run(steps*DT_MS*b.ms, namespace={})
                        cause = error.value.__cause__
                        errors.append((type(cause).__name__, str(cause)))
                    else:
                        network.run(steps*DT_MS*b.ms, namespace={})
                snapshots.append(_snapshot(neurons, synapses, monitor, network))
                rng_states.append(capture_rng_state())
            trains = {} if monitor is None else {
                int(i): np.array(t, copy=True) for i, t in original.get_spk_trn(monitor).items()}
            frame = original.construct_dataframe([{}, trains], case.name,
                                                 dict(enumerate(identities)))
            counters = None
            if native:
                if case.preflight_error:
                    assert device.native_steps == 0
                else:
                    assert device.native_steps > 0
                assert device.native_runs == (0 if case.error else len(durations))
                if not case.preflight_error:
                    assert device.last_timings["execution_s"] > 0
                    assert device._sessions[network].prepared.model.reference_codegen_target == "numpy"
                counters = {"native_runs": device.native_runs, "native_steps": device.native_steps}
            return snapshots, rng_states, errors, trains, frame, counters


@pytest.mark.metal
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_eon_independent_holdout(tmp_path, monkeypatch, case, record_property):
    mx = require_metal('Native Apple Metal is required')
    from original_api_reference import assert_rng_state_equal

    expected = _run(tmp_path/"cpu", case, monkeypatch, native=False)
    if case.error:
        if case.preflight_error:
            assert expected[0][-1]["clock.timestep"][0] == 0
        else:
            assert expected[2][0][0] == "FloatingPointError", expected[2]
            assert expected[0][-1]["clock.timestep"][0] == 3
    actual = _run(tmp_path/"native", case, monkeypatch, native=True)
    for name, value in actual[5].items():
        record_property(name, value)
    if actual[2]:
        record_property("error_type", actual[2][0][0])
        record_property("error_message", actual[2][0][1])
        record_property("failure_clock_tick", int(actual[0][-1]["clock.timestep"][0]))
        record_property("prefix_spikes", int(actual[0][-1]["monitor.N"][0]))
    for observed, wanted in zip(actual[0], expected[0]):
        _exact(observed, wanted)
    for observed, wanted in zip(actual[1], expected[1]):
        assert_rng_state_equal(observed, wanted)
    assert actual[2] == expected[2]
    _exact(actual[3], expected[3])
    pd.testing.assert_frame_equal(actual[4], expected[4], check_exact=True)
    if case.name == "H11-configured-threshold-reset":
        assert expected[0][0]["monitor.t"].size and expected[0][0]["monitor.t"][0] == 0
    if case.name == "H15-split-continuation":
        whole = _run(tmp_path/"cpu-uninterrupted", case, monkeypatch, native=False, whole=True)
        _exact(expected[0][-1], whole[0][-1])
        assert_rng_state_equal(expected[1][-1], whole[1][-1])
    if case.name == "H16-recording-disabled":
        recorded = _run(tmp_path/"cpu-recorded", replace(case, record=True), monkeypatch, native=False)
        state = {name: value for name, value in recorded[0][-1].items()
                 if not name.startswith("monitor.")}
        _exact(expected[0][-1], state)
        assert_rng_state_equal(expected[1][-1], recorded[1][-1])
    if case.name == "H18-output-trains-dataframe":
        assert expected[3]
        assert set(expected[4]["trial"]) == {1}
        assert expected[4]["exp_name"].eq(case.name).all()
        assert all(times.size for times in expected[3].values())


def test_numpy_division_action_names_match_scalar_dispatch():
    """Use actual NumPy dispatch, independently of the native dtype rule."""
    from mlx_backend.numpy32 import Expression
    from mlx_backend.numpy_error_policy import binary_action_name

    constructors = (int, float, np.int32, np.int64, np.float32, np.float64,
                    lambda n: np.array(n, np.int32), lambda n: np.array(n, np.float64),
                    lambda n: np.array([n], np.float64))

    def expression(value):
        dtype = np.asarray(value).dtype
        return Expression("unused", "float" if dtype.kind == "f" else "int",
            np.ndim(value) == 0, dtype == np.dtype("float32"),
            integer_bits=dtype.itemsize*8, integer_unsigned=dtype.kind == "u",
            python_scalar=type(value) in (int, float), array_object=isinstance(value, np.ndarray))

    for make_left in constructors:
        for make_right in constructors:
            left, right = make_left(1), make_right(0)
            if type(left) in (int, float) and type(right) in (int, float):
                continue  # Python exceptions do not use NumPy policy delivery.
            with np.errstate(divide="raise"):
                with pytest.raises(FloatingPointError) as error:
                    left/right
            expected_name = str(error.value).split(" encountered in ")[1]
            assert binary_action_name("divide", expression(left), expression(right)) == expected_name, (
                type(left), type(right))
