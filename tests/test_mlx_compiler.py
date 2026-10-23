"""Host checks of typed emission, independent of MLX and Metal availability."""

import shutil
import struct
import subprocess

import numpy as np
import pytest

from mlx_backend.binary64 import literal
from mlx_backend.compiler import UnsupportedModelError, compile_model
from mlx_backend.symbols import variable_code_name
from mlx_backend.model import resolve_model


@pytest.fixture(scope="module")
def brian():
    module = pytest.importorskip("brian2")
    module.BrianLogger.log_level_error()
    previous = module.prefs.codegen.target
    module.prefs.codegen.target = "numpy"
    yield module
    module.prefs.codegen.target = previous


def _bits(value):
    return struct.unpack("Q", struct.pack("d", float(value)))[0]


# Run emitted source through native double operators to check the compiler's
# type conversions, guards and statement order. Software arithmetic itself is
# covered separately by the native binary64 tests.
_HOST_HELPERS = r"""
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
using ulong = unsigned long;
std::uint32_t mulhi(std::uint32_t a, std::uint32_t b) {
    return std::uint32_t((std::uint64_t(a) * std::uint64_t(b)) >> 32);
}
double fp(ulong bits) { double d; std::memcpy(&d, &bits, 8); return d; }
ulong raw(double d) { ulong bits; std::memcpy(&bits, &d, 8); return bits; }
ulong d_add(ulong a, ulong b) { return raw(fp(a)+fp(b)); }
ulong d_sub(ulong a, ulong b) { return raw(fp(a)-fp(b)); }
ulong d_mul(ulong a, ulong b) { return raw(fp(a)*fp(b)); }
ulong d_div(ulong a, ulong b) { return raw(fp(a)/fp(b)); }
ulong d_neg(ulong a) { return raw(-fp(a)); }
ulong d_abs(ulong a) { return raw(std::abs(fp(a))); }
ulong d_sqrt(ulong a) { return raw(std::sqrt(fp(a))); }
ulong d_fma(ulong a, ulong b, ulong c) { return raw(std::fma(fp(a),fp(b),fp(c))); }
ulong d_from_int(long a) { return raw(static_cast<double>(a)); }
long d_to_int(ulong a) { return static_cast<long>(fp(a)); }
bool d_gt(ulong a, ulong b) { return fp(a)>fp(b); }
bool d_ge(ulong a, ulong b) { return fp(a)>=fp(b); }
bool d_lt(ulong a, ulong b) { return fp(a)<fp(b); }
bool d_le(ulong a, ulong b) { return fp(a)<=fp(b); }
bool d_eq(ulong a, ulong b) { return fp(a)==fp(b); }
using std::min;
using std::max;
"""


def _run_source(tmp_path, program, body, header=""):
    # Fixture bodies use public Brian names; bind their data locals to the
    # compiler's emitted identifiers before composing the host driver.
    from brian2.utils.stringtools import word_substitute
    body = word_substitute(body, {name: var.code_name for name, var in program.variables.items()
                                  if var.persistent})
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("Host C++ compiler unavailable")
    source = tmp_path / "emitted.cpp"
    binary = tmp_path / "emitted"
    arrays = "\n".join(f"const ulong {name}[] = {{" +
                       ", ".join(literal(value) for value in array.values.flat) + "};"
                       for name, array in program.input_arrays.items())
    error = f"bool {program.index_error_name} = false;" if program.index_error_name else ""
    source.write_text(_HOST_HELPERS + header + "\n#define device\n#define thread\n" + program.header_code
                      + "\nint main() {\n" + arrays + error + program.scalar_code + "\n" + body + "\n}\n")
    result = subprocess.run([compiler, "-std=c++17", "-ffp-contract=off", "-O0", str(source), "-o", str(binary)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return subprocess.check_output([str(binary)], text=True).split()


def test_default_inventory_and_no_host_simulation(brian, monkeypatch):
    def no_run(*args, **kwargs):
        raise AssertionError("Compiler advanced a Brian network")
    monkeypatch.setattr(brian.Network, "run", no_run)
    program = compile_model(resolve_model({}))
    assert set(name for name, var in program.variables.items() if var.persistent) == {
        "v", "g", "rfc", "lastspike", "not_refractory",
    }
    assert program.variables["v"].metal_type == "ulong"
    assert program.variables["not_refractory"].metal_type == "bool"
    assert program.variables["rfc"].conditional_write is None
    assert program.variables["g"].conditional_write == "not_refractory"
    assert "w" not in program.variables
    assert f"const long {variable_code_name('w')} = 0L;" in program.reset_code
    assert "_cond =" in program.threshold_code
    assert "bool _cond" not in program.threshold_code
    assert "lastspike =" not in program.threshold_code
    assert "not_refractory =" not in program.threshold_code
    assert "exp(" not in program.state_code
    assert "float" not in program.state_code
    assert "double" not in program.state_code
    assert not any(f" {name} =" in program.scalar_code for name in ("t", "dt", "i", "N"))
    assert hash(program) == object.__hash__(program)


@pytest.mark.parametrize("refractory", [False, True])
def test_default_emitted_stage_matches_brian_optimized_order(tmp_path, brian, refractory):
    model = resolve_model({})
    program = compile_model(model)
    initial_v, initial_g = np.float64(-.047), np.float64(.03)
    t = np.float64(.0001 if refractory else .003)
    body = f"""
    ulong v = {literal(initial_v)}, g = {literal(initial_g)}, rfc = {literal(model.t_rfc)};
    ulong t = {literal(t)}, lastspike = {literal(0.)};
    bool not_refractory = false;
    {program.state_code}
    std::cout << v << ' ' << g << ' ' << not_refractory;
    """
    values = program.scalar_values
    if refractory:
        expected_v, expected_g = initial_v, initial_g
    else:
        expected_g = values["_lio_state_2"] * initial_g
        expected_v = values["_lio_state_8"] + (
            values["_lio_state_9"] * initial_g + values["_lio_state_10"] * initial_v)
    assert _run_source(tmp_path, program, body) == [str(_bits(expected_v)), str(_bits(expected_g)), str(int(not refractory))]


def test_extra_state_integer_boolean_reset_and_independent_guard(tmp_path, brian):
    baseline = resolve_model()
    model = resolve_model({
        "eqs": baseline.eqs.replace("v_0 - v + g", "v_0 - v + g - a")
        + "da/dt = -a/(2*tau) : volt\ncount : integer\nflag : boolean\n",
        "eq_th": "(v > v_th + abs(g)*0.01) and (i != 1)",
        "eq_rst": "v = v_rst; g *= .25; a += 1*mV; count += int(flag); flag = not flag",
    })
    program = compile_model(model)
    assert program.variables["a"].conditional_write is None
    assert program.variables["count"].metal_type == "int"
    assert program.variables["flag"].metal_type == "bool"
    assert "if (not_refractory) {\n        a =" not in program.state_code
    assert "d_abs(g)" in program.threshold_code
    body = f"""
    ulong v = {literal(-.04)}, g = {literal(.002)}, a = {literal(.003)};
    int count = 2, i = 0;
    bool flag = true, not_refractory = true, _cond = false;
    {program.threshold_code}
    {program.reset_code}
    std::cout << _cond << ' ' << v << ' ' << g << ' ' << a << ' ' << count << ' ' << flag;
    """
    assert _run_source(tmp_path, program, body) == [
        "1", str(_bits(model.v_rst)), str(_bits(.002 * .25)), str(_bits(.003 + .001)), "3", "0",
    ]


def test_float64_adversarial_threshold_survives_reset(tmp_path, brian):
    baseline = resolve_model()
    program = compile_model(resolve_model({
        "eqs": baseline.eqs + "a : 1\n", "epsilon": 1e-9,
        "eq_th": "(t < dt) or (a > 1)", "eq_rst": "v = v_rst; g = 0*mV; a = 1 + epsilon",
    }))
    assert program.variables["a"].metal_type == "ulong"
    exact_a = np.float64(1) + np.float64(1e-9)
    assert literal(exact_a) in program.reset_code
    body = f"""
    ulong v = {literal(0.)}, g = {literal(0.)}, a = {literal(0.)};
    const ulong dt = {literal(.0001)}, t = {literal(.0002)};
    bool not_refractory = true, _cond = false;
    {program.reset_code}
    {program.threshold_code}
    std::cout << a << ' ' << _cond;
    """
    assert _run_source(tmp_path, program, body) == [str(_bits(exact_a)), "1"]
    assert np.float32(exact_a) == np.float32(1)  # the original narrowing counterexample


def test_scalar_coefficients_preserve_numpy_operation_order(brian):
    model = resolve_model({"t_mbr": 17*brian.ms, "tau": 3*brian.ms, "v_0": -61*brian.mV}, dt_ms=.23)
    values = compile_model(model).scalar_values
    dt, tau, t_mbr = map(np.float64, (model.dt_s, model.tau, model.t_mbr))
    expected = ((tau / (t_mbr - tau) * (-np.exp(dt / t_mbr) + np.exp(dt / tau)))
                * np.exp(-dt / t_mbr)) * np.exp(-dt / tau)
    assert _bits(values["_lio_state_9"]) == _bits(expected)


@pytest.mark.parametrize("function", ["sin", "expm1", "log1p"])
def test_dynamic_unimplemented_math_is_explicit(brian, function, monkeypatch):
    # These formerly missing functions now have qualified native providers.
    def no_run(*args, **kwargs):
        raise AssertionError("Compiler advanced a Brian network")
    monkeypatch.setattr(brian.Network, "run", no_run)
    model = resolve_model({"eq_th": f"{function}(g/mV) > 0"})
    program = compile_model(model)
    helper = {"sin": "d_sine", "expm1": "_eon_stable_expm1",
              "log1p": "_eon_stable_log1p"}[function]
    assert helper + "(" in program.threshold_code
    assert "g" in program.threshold_code and "_cond =" in program.threshold_code
    assert "inline ulong " + helper + "(" in program.header_code


def test_per_neuron_exponential_and_callback_are_not_silently_narrowed(brian):
    baseline = resolve_model()
    model = resolve_model({"eqs": baseline.eqs.replace("/ t_mbr", "/ tau_v") + "tau_v : second (constant)\n"})
    program = compile_model(model)
    assert "d_exponential(" in program.state_code
    assert program.variables["tau_v"].persistent
    assert not program.variables["tau_v"].scalar
    def never_call(value):
        raise AssertionError("A namespace callback ran during preparation")
    callback = brian.Function(never_call, arg_units=[brian.volt], return_unit=brian.volt)
    model = resolve_model({"callback": callback, "eq_th": "callback(v) > v_th"})
    with pytest.raises(UnsupportedModelError, match="callbacks remain unresolved: callback"):
        compile_model(model)


def test_float32_inventory_preserves_binary64_clock(brian, monkeypatch):
    monkeypatch.setitem(brian.prefs, "core.default_float_dtype", np.float32)
    monkeypatch.setitem(brian.prefs, "codegen.target", "numpy")
    program = compile_model(resolve_model({}))
    assert program.variables["v"].metal_type == "uint"
    assert program.variables["rfc"].metal_type == "uint"
    assert program.variables["lastspike"].metal_type == "ulong"
    assert program.variables["t"].dtype == "float64"


def test_preparation_uses_saved_brian_dtypes_and_abstract_stages(brian, monkeypatch):
    baseline = resolve_model()
    model = resolve_model({"eqs": baseline.eqs + "count : integer\n",
                           "eq_rst": "v = v_rst; count += 1"})
    original_dtype = model.analysis.variables["count"].dtype
    monkeypatch.setitem(brian.prefs, "core.default_integer_dtype", np.int64)
    monkeypatch.setitem(brian.prefs, "legacy.refractory_timing", True)
    monkeypatch.setitem(brian.prefs, "codegen.loop_invariant_optimisations", False)
    program = compile_model(model)
    assert program.variables["count"].dtype == original_dtype
    assert "d_to_int" in program.refractory_code
    assert "exp(" not in program.dynamics_code


def test_default_without_analysis_preserves_snapshotted_refractory_preference(brian, monkeypatch):
    model = resolve_model()
    assert model.analysis is None and not model.legacy_refractory_timing
    reference = compile_model(model)
    monkeypatch.setitem(brian.prefs, "legacy.refractory_timing", True)
    monkeypatch.setitem(brian.prefs, "codegen.loop_invariant_optimisations", False)
    actual = compile_model(model)
    assert actual.state_code == reference.state_code
    assert actual.refractory_elapsed_threshold == reference.refractory_elapsed_threshold
    assert brian.prefs.legacy.refractory_timing is True


@pytest.mark.parametrize("analyse", [False, True])
def test_legacy_predicate_uses_snapshot_and_bypasses_release_cache(brian, monkeypatch, analyse):
    monkeypatch.setitem(brian.prefs, "legacy.refractory_timing", True)
    model = resolve_model({} if analyse else None)
    reference = compile_model(model)
    assert "d_gt(d_sub(t, lastspike), rfc)" in reference.refractory_code
    assert "d_to_int" not in reference.refractory_code
    assert reference.refractory_elapsed_threshold is None
    monkeypatch.setitem(brian.prefs, "legacy.refractory_timing", False)
    actual = compile_model(model)
    assert actual.state_code == reference.state_code
    assert actual.refractory_elapsed_threshold is None
    assert not brian.prefs.legacy.refractory_timing


@pytest.mark.parametrize("refractory", [False, True])
def test_split_stage_has_identical_emitted_effect(tmp_path, brian, refractory):
    model = resolve_model({})
    program = compile_model(model)
    assert "not_refractory =" in program.refractory_code
    assert "not_refractory =" not in program.dynamics_code
    body = f"""
    ulong v = {literal(-.047)}, g = {literal(.03)}, rfc = {literal(model.t_rfc)};
    ulong t = {literal(.0001 if refractory else .003)}, lastspike = {literal(0.)};
    bool not_refractory = false;
    {program.refractory_code}
    {program.dynamics_code}
    ulong split_v = v, split_g = g;
    v = {literal(-.047)}; g = {literal(.03)}; not_refractory = false;
    {program.state_code}
    std::cout << (v == split_v) << ' ' << (g == split_g);
    """
    assert _run_source(tmp_path, program, body) == ["1", "1"]


def test_proven_equilibrium_is_bit_exact_for_emitted_dynamics(tmp_path, brian):
    model = resolve_model({})
    program = compile_model(model)
    assert set(program.equilibrium) == {"v", "g"}
    assert _bits(program.equilibrium["v"]) == _bits(model.v_0)
    assert _bits(program.equilibrium["g"]) == _bits(0.)
    body = f"""
    ulong v = {literal(model.v_0)}, g = {literal(0.)};
    bool not_refractory = true;
    for(int step=0; step<100; ++step) {program.dynamics_code}
    std::cout << v << ' ' << g;
    """
    assert _run_source(tmp_path, program, body) == [str(_bits(model.v_0)), str(_bits(0.))]


def test_equilibrium_rejects_rounding_drift_live_extra_state_and_refractory_mutation(brian):
    baseline = resolve_model()
    # These finite constants make the optimized rest update drift by one ulp.
    drifting = resolve_model({"v_0": -.48211931267997826*brian.volt,
                              "t_mbr": (1 / 4.540443915911708)*brian.second}, dt_ms=1000)
    assert compile_model(drifting).equilibrium is None
    assert compile_model(resolve_model({"eqs": baseline.eqs + "da/dt = -a/(10*ms) : 1\n"})).equilibrium is None
    for reset in ("rfc = 0*ms", "lastspike = t", "not_refractory = True"):
        program = compile_model(resolve_model({"eq_rst": reset}))
        assert program.equilibrium is None
        assert program.refractory_elapsed_threshold is None


@pytest.mark.parametrize("dt_s,rfc_s", [
    (.0001, .0022), (.0003, .00229995), (.00023, .0000998),
    (.0001, .00009995), (.0001, .00015), (.0001, 0.), (.0001, -.0001),
    (1e-12, 1e-8), (1., 10000.), (np.finfo(np.float64).tiny, np.finfo(np.float64).tiny * 4),
])
def test_refractory_boundary_matches_exact_rounded_brian_predicate(brian, dt_s, rfc_s):
    from brian2.core.functions import timestep

    model = resolve_model({"t_rfc": rfc_s*brian.second}, dt_ms=dt_s*brian.second)
    program = compile_model(model)
    boundary = program.refractory_elapsed_threshold
    assert boundary is not None
    refractory_steps = timestep(np.float64(model.t_rfc), np.float64(model.dt_s))
    boundary_bits = _bits(boundary)
    for difference in (-10, -2, -1, 0, 1, 2, 10):
        if boundary_bits + difference < 0:
            continue
        elapsed = struct.unpack("d", struct.pack("Q", boundary_bits + difference))[0]
        actual = timestep(np.float64(elapsed), np.float64(model.dt_s)) >= refractory_steps
        assert actual == (elapsed >= boundary)
    random = np.random.default_rng(42)
    for elapsed in random.uniform(0., max(1e-200, 2*abs(rfc_s)), 30):
        quotient = (np.float64(elapsed) + np.float64(.001)*np.float64(dt_s)) / np.float64(dt_s)
        if not np.isfinite(quotient) or quotient >= 2**63:
            continue
        actual = timestep(np.float64(elapsed), np.float64(dt_s)) >= refractory_steps
        assert actual == (elapsed >= boundary)


@pytest.mark.parametrize("dt_s,rfc_s,spike_step", [
    (.0001, .0022, 0), (.0003, .0022, 123), (.0001, 0., 17),
    (.0001, .00229995, 2**40 + 1), (.0003, .0022, 2**50 + 1),
    (.0001, .0022, 2**62 + 1), (.0001, .0022, 2**63 - 5),
])
def test_cached_integer_release_inverts_actual_global_clock(brian, dt_s, rfc_s, spike_step):
    """Check the proposed release cache against independent Brian predicates.

    Integer binary search does not replace elapsed time with step subtraction:
    its predicate evaluates the same rounded global clock/subtraction as the
    kernel. The high bound is an exclusive run horizon/sentinel, never an
    executed step if no candidate has become non-refractory by then.
    """
    from brian2.core.functions import timestep

    model = resolve_model({"t_rfc": rfc_s*brian.second}, dt_ms=dt_s*brian.second)
    boundary = compile_model(model).refractory_elapsed_threshold
    dt = np.float64(model.dt_s)
    spike_time = np.float64(spike_step) * dt
    horizon = 2**63 - 1

    def elapsed_at(candidate):
        return np.float64(candidate) * dt - spike_time

    low, high = spike_step, horizon
    while low < high:
        middle = low + (high - low) // 2
        if elapsed_at(middle) >= boundary:
            high = middle
        else:
            low = middle + 1
    release = low
    candidates = set(range(spike_step, min(spike_step + 40, horizon)))
    candidates.update(candidate for candidate in (release-1, release, release+1) if spike_step <= candidate < horizon)
    refractory_steps = timestep(np.float64(model.t_rfc), dt)
    for candidate in candidates:
        actual = timestep(elapsed_at(candidate), dt) >= refractory_steps
        assert actual == (candidate >= release)
    if rfc_s == 0:
        assert release == spike_step


def _brian_timed_function(brian, values, interval, dt):
    array = brian.TimedArray(values, dt=interval*brian.second)
    owner = brian.NeuronGroup(1, "v : 1", dt=dt*brian.second)
    owner._network = "timedarray-compiler-reference-only"
    return array, array.implementations["numpy"].get_code(owner)


@pytest.mark.parametrize("ratio", [.125, 1., 10., 2.3])
def test_timedarray_1d_emission_matches_brian_numpy_rounding_and_clamping(tmp_path, brian, ratio):
    from brian2.input.timedarray import _find_K

    dt = .0001
    interval = dt*ratio
    values = np.array([1.+1e-9, -2.5, 3.75, -4.25, 5.125, 6.5])
    timed, expected_function = _brian_timed_function(brian, values, interval, dt)
    model = resolve_model({"drive": timed, "eqs": resolve_model().eqs+"sample : 1\n",
                           "eq_rst": "sample = drive(t)"}, dt_ms=dt*brian.second)
    program = compile_model(model)
    assert len(program.input_arrays) == 1
    assert program.index_error_name is None
    epsilon = interval / _find_K(dt, interval)
    boundaries = np.array([epsilon*(index+.5) for index in range(-3, 65)])
    times = np.r_[np.nextafter(boundaries, -np.inf), boundaries,
                  np.nextafter(boundaries, np.inf), -1., 0., 1.]
    # Invalid casts differ by CPU architecture. The production target follows
    # Apple silicon NumPy; retain portable finite-range host checks elsewhere.
    with np.errstate(invalid="ignore"):
        if np.int64(np.float64(np.inf)) == np.iinfo(np.int64).max:
            times = np.r_[times, np.inf, -np.inf, np.nan, 2.**64]
    evaluations = "\n".join(f"t={literal(t)}; {program.reset_code} std::cout << sample << ' ';"
                             for t in times)
    with np.errstate(invalid="ignore", over="ignore"):
        expected = [str(_bits(expected_function(np.float64(t)))) for t in times]
    actual = _run_source(tmp_path, program, "ulong t=0, sample=0;\n" + evaluations)
    assert actual == expected


def test_timedarray_2d_negative_columns_and_error_flag_match_numpy(tmp_path, brian):
    values = np.array([[1.+1e-9, -2., 3.], [4., 5., -6.], [7., -8., 9.]])
    timed, expected_function = _brian_timed_function(brian, values, .0003, .0001)
    model = resolve_model({"drive": timed, "eqs": resolve_model().eqs+"sample : 1\ncolumn : integer\n",
                           "eq_rst": "sample = drive(t, column)"})
    program = compile_model(model)
    flag = program.index_error_name
    assert flag is not None
    times, columns = np.meshgrid(np.array([-.01, 0., .000299, .0003, .01]), np.arange(-4, 5))
    evaluations, expected = [], []
    previous_value = np.float64(0.)
    for time, column in zip(times.flat, columns.flat):
        evaluations.append(f"t={literal(time)}; column={column}; {flag}=false; {program.reset_code} "
                           f"std::cout << {flag} << ' ' << sample << ' ';")
        try:
            value = expected_function(time, column)
        except IndexError:
            # A failed NumPy RHS leaves the destination's previous value.
            expected.extend(["1", str(_bits(previous_value))])
        else:
            previous_value = value
            expected.extend(["0", str(_bits(value))])
    assert _run_source(tmp_path, program, "ulong t=0, sample=0; int column=0;\n" +
                       "\n".join(evaluations)) == expected


@pytest.mark.parametrize("dimensions", [1, 2])
def test_invariant_timedarray_calls_fold_without_runtime_inputs(tmp_path, brian, dimensions):
    values = np.arange(6.).reshape((6,) if dimensions == 1 else (3, 2)) + 1e-9
    timed, expected_function = _brian_timed_function(brian, values, .0001, .0001)
    expression = "drive(2*dt)" if dimensions == 1 else "drive(2*dt, -1)"
    model = resolve_model({"drive": timed, "eqs": resolve_model().eqs+"sample : 1\n",
                           "eq_rst": "sample = "+expression})
    program = compile_model(model)
    assert not program.input_arrays and program.header_code == "" and program.index_error_name is None
    expected = expected_function(.0002) if dimensions == 1 else expected_function(.0002, -1)
    assert _run_source(tmp_path, program, "ulong sample=0;" + program.reset_code +
                       "std::cout << sample;") == [str(_bits(expected))]


def test_timedarray_snapshot_and_multiple_arrays_are_bound_independently(brian):
    a = brian.TimedArray([1., 2.], dt=.1*brian.ms)
    z = brian.TimedArray([[3., 4.], [5., 6.]], dt=.2*brian.ms)
    model = resolve_model({"a": a, "z": z, "eq_th": "a(t) > z(t, i)"})
    a.values[:] = 100.
    z.values[:] = 200.
    program = compile_model(model)
    arrays = {array.source_name: array for array in program.input_arrays.values()}
    np.testing.assert_array_equal(arrays["a"].values, [1., 2.])
    np.testing.assert_array_equal(arrays["z"].values, [[3., 4.], [5., 6.]])
    assert all(not array.values.flags.writeable and array.values.flags.c_contiguous for array in arrays.values())


def test_namespace_ndarrays_are_rejected_by_brian_before_native_lowering(brian):
    for values in (np.array([1.]), np.array([1., 2.]), np.array([[1., 2.]])):
        for expression in ("array > 0", "array[i] > 0"):
            with pytest.raises(KeyError, match="not a scalar value"):
                resolve_model({"array": values, "eq_th": expression})


@pytest.mark.parametrize("column", ["0.0", "v/mV"])
@pytest.mark.parametrize("target", ["numpy", "cython"])
def test_timedarray_float_columns_are_explicitly_rejected(brian, column, target, monkeypatch):
    monkeypatch.setitem(brian.prefs, "codegen.target", target)
    timed = brian.TimedArray([[1., 2.]], dt=.1*brian.ms)
    model = resolve_model({"drive": timed, "eq_th": f"drive(t, {column}) > 0"})
    with pytest.raises(UnsupportedModelError, match="integer column index"):
        compile_model(model)


def test_unreferenced_timedarray_does_not_add_runtime_data_or_validation(brian):
    model = resolve_model({"unused": brian.TimedArray([], dt=.1*brian.ms)})
    program = compile_model(model)
    assert not program.input_arrays and not program.header_code and program.index_error_name is None


def test_timedarray_target_snapshot_and_cache_identity_follow_brian_device(brian, monkeypatch, tmp_path):
    from brian2.devices.device import get_device

    timed = brian.TimedArray([0., 1.], dt=brian.defaultclock.dt/8)
    params = {"drive": timed, "eq_th": "drive(t+dt/16)>.5"}
    numpy_model = resolve_model(params)
    numpy_header = compile_model(numpy_model).header_code
    monkeypatch.setitem(brian.prefs, "codegen.target", "cython")
    compiled_model = resolve_model(params)
    assert numpy_model.reference_codegen_target == "numpy"
    assert compiled_model.reference_codegen_target == "cython"
    assert numpy_model.signature != compiled_model.signature
    assert compile_model(numpy_model).header_code == numpy_header
    assert compile_model(compiled_model).header_code != numpy_header
    monkeypatch.setitem(brian.prefs, "codegen.cpp.extra_compile_args", ["-O2", "-fno-fast-math", "-ffp-contract=off"])
    strict_model = resolve_model(params)
    assert strict_model.reference_compile_args == ("-O2", "-fno-fast-math", "-ffp-contract=off")
    assert strict_model.signature != compiled_model.signature
    assert compile_model(strict_model).header_code != compile_model(compiled_model).header_code
    monkeypatch.setitem(brian.prefs, "codegen.target", "auto")
    monkeypatch.setitem(brian.prefs, "codegen.runtime.cython.cache_dir", str(tmp_path / "cython-policy"))
    actual_class = get_device().code_object_class()
    auto_model = resolve_model(params)
    assert auto_model.reference_codegen_target == actual_class.class_name


@pytest.mark.parametrize("target,expected", [("numpy", 0.), ("cython", 1.)])
def test_invariant_timedarray_lookup_uses_snapshotted_target(tmp_path, brian, monkeypatch, target, expected):
    monkeypatch.setitem(brian.prefs, "codegen.target", target)
    model = resolve_model({"drive": brian.TimedArray([0., 1.], dt=brian.defaultclock.dt/8),
                           "eqs": resolve_model().eqs+"sample : 1\n", "eq_rst": "sample=drive(dt/16)"})
    program = compile_model(model)
    assert not program.input_arrays
    assert _run_source(tmp_path, program, "ulong sample=0;"+program.reset_code+
                       "std::cout << sample;") == [str(_bits(expected))]


@pytest.mark.parametrize("interval", [.0000125, .0001, .00023000000000000007])
def test_compiled_timedarray_emission_matches_actual_brian_cpp_source(tmp_path, brian, monkeypatch, interval):
    from brian2.input.timedarray import _find_K

    monkeypatch.setitem(brian.prefs, "codegen.target", "cython")
    # This emitted-source check intentionally compiles the reference without
    # unsafe reassociation; actual default optimized targets are checked on Metal.
    monkeypatch.setitem(brian.prefs, "codegen.cpp.extra_compile_args", ["-O2", "-fno-fast-math", "-ffp-contract=off"])
    values = np.arange(6.)+1e-9
    timed = brian.TimedArray(values, dt=interval*brian.second)
    owner = brian.NeuronGroup(1, "v : 1", dt=.1*brian.ms)
    owner._network = "timedarray-compiled-source-reference-only"
    # This is Brian's actual installed compiled lookup, not a second copy of
    # the native index algorithm or the NumPy reference implementation.
    reference_code = timed.implementations["cpp"].get_code(owner)
    header = ("\nconst double reference_values[] = {" +
              ", ".join(float(value).hex() for value in values) + "};\n" +
              f"const double* _namespace{timed.name}_values = reference_values;\n" + reference_code)
    model = resolve_model({"drive": timed, "eqs": resolve_model().eqs+"sample : 1\n",
                           "eq_rst": "sample=drive(t)"})
    program = compile_model(model)
    epsilon = float(f"{interval:.18f}") / _find_K(.0001, interval)
    boundaries = epsilon*(np.arange(-3, 68)+.5)
    times = np.r_[np.nextafter(boundaries, -np.inf), boundaries, np.nextafter(boundaries, np.inf), -1., 1.]
    body = "ulong t=0, sample=0;\n" + "\n".join(
        f"t={literal(time)}; {program.reset_code} std::cout << (sample==raw({timed.name}(fp(t)))) << ' ';"
        for time in times)
    assert _run_source(tmp_path, program, body, header) == ["1"]*len(times)
