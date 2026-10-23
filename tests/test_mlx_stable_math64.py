"""Exact pinned-Brian stable functions, with independent CPU and Metal checks.

The accepted target is Brian/NumPy's result, including its observed rounding;
these transcendental functions are not claimed to be correctly rounded.
"""
from state_assertions import assert_spikes_equal

import ast
import ctypes
from decimal import Decimal, localcontext
import math
from pathlib import Path
import subprocess

import numpy as np
import pytest
from native_execution import require_metal as _require_metal
from pinned_apple_math import require_apple_arm64_reference

from mlx_backend import binary64, math64, stable_math64


FUNCTIONS = ("expm1", "log1p", "exprel")


def stable_cases():
    """Branch, scaling and every table-cell boundary, plus prior counterexamples."""
    centers = [0., -0., 1., -1., .1, -.1, 717., -38., 700., -700.,
               2**-53, -2**-53, 2**-5, -2**-5, 2**-27, -2**-27,
               1e-16, -1e-16, -.5, -.25, .5, 1e300,
               float.fromhex("-0x1.30fc100000000p+5"),
               float.fromhex("0x1.62e42fefa39f0p+9"),
               float.fromhex("0x1.68a1bac7c16ecp+5")]
    centers += [(scale*128+j)*math.log(2)/128
                for scale in (-55, -3, -2, -1, 0, 1, 51, 52, 53, 1022, 1023)
                for j in (0, 1, 63, 64, 126, 127)]
    centers += [sign*(j+.5)*math.log(2)/128 for sign in (-1, 1) for j in range(128)]
    centers += [(j+.5)/128 for j in range(128)]
    centers += [math.ldexp(1., exponent)-1 for exponent in range(-53, 1024, 71)]
    centers += [(1+j/128)*2**k-1 for k in (-1, 0, 1, 52, 53, 1022)
                for j in (0, 1, 63, 64, 65, 126, 127)]
    # The pinned CPU differs from correctly-rounded CORE-MATH on these inputs.
    centers += [float.fromhex(x) for x in (
        "0x1.7319dc68d57b0p+8", "0x1.c3e619a55f1b8p+7", "0x1.d043f85db2424p+8",
        "0x1.408529a2654e0p+7", "0x1.6dcd9c563adbcp+7", "0x1.07aae349f027fp+35",
        "-0x1.820297ddecf5dp-29", "-0x1.65ff93a90c823p-28", "0x1.aa0f860acb971p-28",
        "0x1.d494e7a1fdb50p-11", "-0x1.ea73faa94f842p-1", "0x1.4a3ca877d75d6p-1",
        "0x1.7581761665f7ap-1", "-0x1.992121d1e380cp-1", "0x1.ae5823bb2b20ep-1")]
    values = np.array([y for x in centers for y in
                       (math.nextafter(x, -math.inf), x, math.nextafter(x, math.inf))])
    special = np.array([0, 1, 0x8000000000000000, 0x8000000000000001,
        0x7ff0000000000000, 0xfff0000000000000, 0x7ff0000000000001,
        0xfff0000000000001, 0x7ff8123456789abc, 0xfff8000000000001], np.uint64)
    return np.unique(np.concatenate((values.view(np.uint64), special,
        (np.arange(1, 4096+65, dtype=np.float64)/16384).view(np.uint64))))


def reference_bits(name, raw):
    require_apple_arm64_reference()
    from brian2.core.functions import DEFAULT_FUNCTIONS
    with np.errstate(all="ignore"):
        return np.asarray(DEFAULT_FUNCTIONS[name].pyfunc(raw.view(np.float64)), np.float64).view(np.uint64)


def assert_bits(actual, expected, raw, name):
    different = np.flatnonzero(actual != expected)
    if different.size:
        i = different[0]
        pytest.fail(f"{name} input={int(raw[i]):#018x}: actual={int(actual[i]):#018x}, "
                    f"Brian={int(expected[i]):#018x}; {different.size}/{len(raw)} differ")


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    module = ast.parse(Path(__file__).with_name("test_mlx_math64.py").read_text())
    prefix = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "HOST_PREFIX" for t in n.targets))
    out = tmp_path_factory.mktemp("stable-math-host")
    source, library = out/"math.cpp", out/"math.so"
    wrappers = '\nextern "C" {\n' + "\n".join(
        f"void run_{name}(const ulong *x, ulong *y, int count) "
        f"{{for(int i=0;i<count;++i)y[i]=d_{name}(x[i]);}}" for name in FUNCTIONS) + r"""
void run_fma(const ulong *a,const ulong *b,const ulong *c,ulong *y,int count) {
    for(int i=0;i<count;++i)y[i]=stable_fma(a[i],b[i],c[i]);
}
void run_div(const ulong *a,const ulong *b,ulong *y,int count) {
    for(int i=0;i<count;++i)y[i]=stable_div(a[i],b[i]);
}
void reference_div(const ulong *a,const ulong *b,ulong *y,int count) {
    for(int i=0;i<count;++i)y[i]=as_type<ulong>(as_type<double>(a[i])/as_type<double>(b[i]));
}
void reference_fma(const ulong *a,const ulong *b,const ulong *c,ulong *y,int count) {
    for(int i=0;i<count;++i)y[i]=as_type<ulong>(std::fma(
        as_type<double>(a[i]),as_type<double>(b[i]),as_type<double>(c[i])));
}
}
"""
    source.write_text(prefix + "\n#define constant static const\n" + binary64.MSL_HEADER
        + math64.MSL_MATH_HEADER + stable_math64.MSL_STABLE_MATH_HEADER + wrappers)
    subprocess.run(["clang++", "-std=c++17", "-shared", "-fPIC", "-O2", "-fno-fast-math",
                    "-ffp-contract=off", str(source), "-o", str(library)],
                   check=True, capture_output=True, text=True, timeout=30)
    result = ctypes.CDLL(str(library))
    pointer = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
    for name in FUNCTIONS:
        function = getattr(result, "run_"+name)
        function.argtypes, function.restype = [pointer, pointer, ctypes.c_int], None
    for name in ("run_fma", "reference_fma"):
        function = getattr(result, name)
        function.argtypes, function.restype = [pointer, pointer, pointer, pointer, ctypes.c_int], None
    for name in ("run_div", "reference_div"):
        function = getattr(result, name)
        function.argtypes, function.restype = [pointer, pointer, pointer, ctypes.c_int], None
    return result


def division_cases():
    raw = fma_cases()[:, :2]
    cases = []
    for cell in (0, 1, 2, 63, 127, 128, 4095, 65535, 2**22-1, 2**23-1):
        for offset in (127, 128, 129):
            high = 0x80000000+cell*256+offset
            if high > 0xffffffff:
                continue
            for low in (0, 1, 2**20, 2**21-1):
                mantissa = ((high<<21)|low)&((1<<52)-1)
                for exponent in (1, 1023, 2046):
                    denominator = (exponent<<52)|mantissa
                    for numerator in (1, 0x10000000000000, 0x3fefffffffffffff,
                            0x3ff0000000000000, 0x3ff0000000000001, 0x7fefffffffffffff):
                        cases.append((numerator, denominator))
    return np.ascontiguousarray(np.concatenate((raw, np.array(cases, np.uint64))))


def test_host_safe_division_matches_independent_hardware(host):
    raw = division_cases()
    columns = [np.ascontiguousarray(raw[:, j]) for j in range(2)]
    actual, expected = np.empty(len(raw), np.uint64), np.empty(len(raw), np.uint64)
    host.run_div(*columns, actual, len(raw))
    host.reference_div(*columns, expected, len(raw))
    valid = ~np.isnan(expected.view(np.float64))
    np.testing.assert_array_equal(actual[valid], expected[valid])
    assert np.all(np.isnan(actual.view(np.float64)[~valid]))


@pytest.mark.metal
def test_actual_metal_safe_division_matches_independent_hardware(host):
    mx = require_metal()
    raw = division_cases()
    columns = [np.ascontiguousarray(raw[:, j]) for j in range(2)]
    expected = np.empty(len(raw), np.uint64)
    host.reference_div(*columns, expected, len(raw))
    kernel = mx.fast.metal_kernel(name="stable_div_direct", input_names=["a", "b"],
        output_names=["out"], source="uint i=thread_position_in_grid.x;if(i<COUNT)out[i]=stable_div(a[i],b[i]);",
        header=binary64.MSL_HEADER+stable_math64._STABLE_DIV_HEADER,
        compile_options={"math_mode": "safe"})
    output, = kernel(inputs=[mx.array(v) for v in columns], template=[("COUNT", len(raw))],
        grid=(len(raw), 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[expected.shape], output_dtypes=[mx.uint64])
    mx.eval(output)
    actual = np.asarray(output)
    valid = ~np.isnan(expected.view(np.float64))
    np.testing.assert_array_equal(actual[valid], expected[valid])
    assert np.all(np.isnan(actual.view(np.float64)[~valid]))


def fma_cases():
    from test_mlx_math64 import fma_cases as existing_fma_cases
    # Reuse every-exponent cancellation/tie/overflow coverage and add product
    # cancellation with all106product bits populated, not just powers of two.
    raw = existing_fma_cases()
    rng = np.random.default_rng(20261017)
    extra = rng.integers(0, np.iinfo(np.uint64).max, (8192, 3), dtype=np.uint64)
    extra &= np.uint64(0xffefffffffffffff)
    near = extra[:4096].copy()
    with np.errstate(all="ignore"):
        product = near[:, 0].view(np.float64)*near[:, 1].view(np.float64)
    finite = np.isfinite(product)
    near = near[finite]
    near[:, 2] = (-product[finite]).view(np.uint64)
    return np.ascontiguousarray(np.concatenate((raw, extra, near)))


def test_host_narrow_exact_fma_matches_independent_hardware(host):
    raw = fma_cases()
    columns = [np.ascontiguousarray(raw[:, j]) for j in range(3)]
    actual, expected = np.empty(len(raw), np.uint64), np.empty(len(raw), np.uint64)
    host.run_fma(*columns, actual, len(raw))
    host.reference_fma(*columns, expected, len(raw))
    finite = ~np.isnan(expected.view(np.float64))
    different = np.flatnonzero((actual != expected) & finite)
    if different.size:
        at = different[0]
        pytest.fail(f"FMA input={tuple(hex(int(v)) for v in raw[at])}: "
                    f"actual={int(actual[at]):#018x}, hardware={int(expected[at]):#018x}")
    assert np.all(np.isnan(actual.view(np.float64)[~finite]))


@pytest.mark.metal
def test_actual_metal_narrow_fma_matches_independent_hardware(host):
    mx = require_metal()
    raw = fma_cases()
    columns = [np.ascontiguousarray(raw[:, j]) for j in range(3)]
    expected = np.empty(len(raw), np.uint64)
    host.reference_fma(*columns, expected, len(raw))
    kernel = mx.fast.metal_kernel(name="stable_fma_direct", input_names=["a", "b", "c"],
        output_names=["out"], source="uint i=thread_position_in_grid.x;if(i<COUNT)out[i]=stable_fma(a[i],b[i],c[i]);",
        header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+stable_math64._STABLE_FMA_HEADER,
        compile_options={"math_mode": "safe"})
    output, = kernel(inputs=[mx.array(v) for v in columns], template=[("COUNT", len(raw))],
        grid=(len(raw), 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[expected.shape], output_dtypes=[mx.uint64])
    mx.eval(output)
    actual = np.asarray(output)
    finite = ~np.isnan(expected.view(np.float64))
    np.testing.assert_array_equal(actual[finite], expected[finite])
    assert np.all(np.isnan(actual.view(np.float64)[~finite]))


def test_headers_include_only_called_functions_and_dependencies():
    assert stable_math64.metal_stable_math_header(()) == ""
    log = stable_math64.metal_stable_math_header({"log1p"})
    assert "inline ulong d_log1p(" in log
    assert "d_expm1(" not in log and "d_exprel(" not in log
    rel = stable_math64.metal_stable_math_header({"exprel"})
    assert "inline ulong d_exprel(" in rel and "inline ulong d_expm1(" in rel
    assert "d_log1p(" not in rel


def test_host_matches_independent_brian_bits_including_prior_counterexamples(host):
    raw = stable_cases()
    for name in FUNCTIONS:
        actual = np.empty_like(raw)
        getattr(host, "run_"+name)(raw, actual, len(raw))
        assert_bits(actual, reference_bits(name, raw), raw, name)


@pytest.mark.parametrize("name,x,rounded", [
    ("expm1", "0x1.7319dc68d57b0p+8", 0x6164e70734f49901),
    ("log1p", "0x1.07aae349f027fp+35", 0x40384a2757e23ac5),
])
def test_high_precision_diagnostic_has_unambiguous_rounding(name, x, rounded):
    with localcontext() as context:
        context.prec = 200
        argument = Decimal.from_float(float.fromhex(x))
        exact = argument.exp()-1 if name == "expm1" else (1+argument).ln()
        low = np.float64(context.next_minus(exact)).view(np.uint64).item()
        high = np.float64(context.next_plus(exact)).view(np.uint64).item()
    assert low == high == rounded


@pytest.mark.parametrize("name,x,rounded", [
    ("expm1", "0x1.7319dc68d57b0p+8", 0x6164e70734f49901),
    ("log1p", "0x1.07aae349f027fp+35", 0x40384a2757e23ac5),
])
def test_pinned_cpu_deviates_from_high_precision_diagnostic(name,x,rounded):
    require_apple_arm64_reference()
    raw = np.array([float.fromhex(x)]).view(np.uint64)
    # These are documented one-ULP CPU deviations, not acceptance tolerances.
    assert reference_bits(name, raw)[0] != rounded


@pytest.mark.metal
def test_actual_metal_matches_independent_brian_bits():
    mx = _require_metal('Metal is required')
    raw = stable_cases()
    kernel = mx.fast.metal_kernel(name="stable_math_direct", input_names=["input"],
        output_names=["out_"+name for name in FUNCTIONS],
        source="uint i=thread_position_in_grid.x;if(i<COUNT){" + "".join(
            f"out_{name}[i]=d_{name}(input[i]);" for name in FUNCTIONS) + "}",
        header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+stable_math64.MSL_STABLE_MATH_HEADER,
        compile_options={"math_mode": "safe"})
    outputs = kernel(inputs=[mx.array(raw)], template=[("COUNT", len(raw))],
        grid=(len(raw), 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[raw.shape]*3, output_dtypes=[mx.uint64]*3)
    mx.eval(*outputs)
    for name, actual in zip(FUNCTIONS, outputs):
        assert_bits(np.asarray(actual), reference_bits(name, raw), raw, name)


def assert_state(actual, expected):
    assert actual.keys() == expected.keys()
    for name, right in expected.items():
        left = actual[name]
        assert left.dtype == right.dtype, name
        if right.dtype == np.float64:
            a, b = left.view(np.uint64), right.view(np.uint64)
            different = np.argwhere(a != b)
            if different.size:
                at = tuple(different[0])
                pytest.fail(f"{name}{at}: actual={int(a[at]):#018x}, Brian={int(b[at]):#018x}")
        else:
            np.testing.assert_array_equal(left, right, err_msg=name)


def require_metal():
    mx = _require_metal('Metal is required')
    return mx


def test_compiler_folds_invariants_without_adding_unused_native_headers():
    from mlx_backend.compiler import compile_model
    from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
    model = resolve_model({"eqs": DEFAULT_EQUATIONS+"out:1\n",
                           "eq_rst": "v=v_rst;g=0*mV;out=exprel(0)"})
    program = compile_model(model)
    assert program.header_code == ""
    assert "d_exprel(" not in program.reset_code


@pytest.mark.metal
def test_dynamic_stable_functions_in_ode_threshold_reset_continuation_and_no_record(monkeypatch):
    mx = require_metal()
    b = pytest.importorskip("brian2")
    from configured_reference import reference
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
    from test_mlx_configured import graph, inputs
    monkeypatch.setitem(b.prefs, "codegen.target", "numpy")
    model = resolve_model({
        "eqs": DEFAULT_EQUATIONS+"da/dt=-a/tau+expm1((i+1)/32)/second+log(i+1)/(64*second):1\n"
            "dc/dt=-c*exprel((i+1)/32)/tau:1\n"
            "dqlog/dt=-qlog/tau+log1p((i+1)/32)/second:1\nz:1\ny:1\nphase:integer\n",
        "eq_th": "(t<dt) or ((log1p(z)>0) and (exprel(y)>0))",
        "eq_rst": "v=v_rst;g=0*mV;c+=1;z=log1p(exp((i+phase+1)/32));y=exprel((i-1+phase)/16);phase+=1",
        "t_rfc": 0*b.ms,
    })
    c, s = graph(), inputs(model, targets=(), steps=53, trials=2)
    expected = reference(c, s, model)
    p = configured.prepare(c, s, model)
    assert all("_eon_stable_"+name+"(" in p.program.dynamics_code for name in FUNCTIONS)
    assert "_eon_stable_log1p(" in p.program.threshold_code and "_eon_stable_exprel(" in p.program.reset_code
    assert "d_exponential(" in p.program.reset_code and "d_logarithm(" in p.program.dynamics_code
    for recording in (True, False):
        actual, state = configured.run_prepared(p, s, record=recording, return_state=True,
                                               chunk_steps=18, block_windows=1)
        if recording:
            assert_spikes_equal(actual.spikes, expected.spikes)
        else:
            assert not actual.spikes.steps.size
        assert_state(configured.host_state(state, p), expected.state)
    events = mx.array(s.events.reshape(-1) if s.events.size else np.zeros(1, np.uint8))
    state = configured.initial_state(p, 2)
    recorded, calls = [], 0
    widths = (7, 1, 11, 3)
    while state.step < 53:
        begin = state.step
        length = min(widths[calls % len(widths)], p.window-begin % p.window, 53-begin)
        state, words = configured.advance(p, state, events, 53, length)
        configured.evaluate_state(state)
        raw = np.asarray(words)
        for bit in range(length):
            for row in np.flatnonzero(raw & (1 << bit)):
                recorded.append((s.trial_indices[row//p.neurons], row % p.neurons, begin+bit))
        calls += 1
    assert sorted(recorded, key=lambda row: (row[0], row[2], row[1])) == list(zip(
        expected.spikes.trials, expected.spikes.neurons, expected.spikes.steps))
    assert_state(configured.host_state(state, p), expected.state)


@pytest.mark.metal
@pytest.mark.parametrize("name,argument", [("expm1", 1.),
    ("log1p", float.fromhex("0x1.68a1bac7c16ecp+5")), ("exprel", 1.)])
@pytest.mark.parametrize("comparison", [">", ">="])
def test_exact_stable_function_threshold_boundaries(name, argument, comparison, monkeypatch):
    require_metal()
    b = pytest.importorskip("brian2")
    from configured_reference import reference
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
    from test_mlx_configured import graph, inputs
    monkeypatch.setitem(b.prefs, "codegen.target", "numpy")
    raw = np.array([argument]).view(np.uint64)
    boundary = reference_bits(name, raw).view(np.float64)[0].item()
    model = resolve_model({"eqs": DEFAULT_EQUATIONS+"a:1\n",
        "eq_th": f"(t<dt) or ({name}(a){comparison}boundary)",
        "eq_rst": "v=v_rst;g=0*mV;a=argument", "argument": argument,
        "boundary": boundary, "t_rfc": 0*b.ms})
    c, s = graph(), inputs(model, targets=(), steps=7, trials=1)
    expected = reference(c, s, model)
    p = configured.prepare(c, s, model)
    actual, state = configured.run_prepared(p, s, return_state=True, chunk_steps=3)
    assert actual.spikes.steps.size == (3 if comparison == ">" else 21)
    assert_spikes_equal(actual.spikes, expected.spikes)
    assert_state(configured.host_state(state, p), expected.state)
    for other in set(FUNCTIONS)-{name, "expm1" if name == "exprel" else name}:
        assert f"inline ulong d_{other}(" not in p.program.header_code


@pytest.mark.metal
def test_default_optimized_cython_function_calls_match_native(monkeypatch, tmp_path):
    require_metal()
    b = pytest.importorskip("brian2")
    from hashlib import sha256
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
    from mlx_backend.models import Connectome, Stimulus
    cache = Path(__file__).resolve().parents[1]/"executions/mlx-feature-parity/math64-cython"/tmp_path.name
    monkeypatch.setitem(b.prefs, "codegen.target", "cython")
    monkeypatch.setitem(b.prefs, "codegen.runtime.cython.cache_dir", str(cache))
    model = resolve_model({"eqs": DEFAULT_EQUATIONS+"a:1\nb:1\nc:1\nphase:integer\nchecksum:integer\n",
        "eq_th": "True", "t_rfc": 0*b.ms,
        "eq_rst": "v=v_rst;g=0*mV;a=expm1((i-1+phase)/32);b=log1p((i+phase+1)/32);c=exprel((i-1+phase)/32);phase+=1;checksum+=int(a*1024)+int(b*1024)+int(c*1024)"})
    clock = b.Clock(dt=model.dt_s*b.second)
    neurons = b.NeuronGroup(3, model.eqs, method="linear", threshold=model.eq_th,
        reset=model.eq_rst, refractory="rfc", namespace=model.brian_namespace(), clock=clock,
        dtype=np.dtype(model.state_dtype))
    neurons.v = model.v_0*b.volt
    neurons.g = 0*b.volt
    neurons.rfc = 0*b.second
    monitor = b.SpikeMonitor(neurons)
    network = b.Network(neurons, monitor)
    network.schedule = list(model.schedule)
    network.run(7*model.dt_s*b.second, namespace={})
    assert {code.__class__.__name__ for obj in network.sorted_objects for code in obj.code_objects} == {"CythonCodeObject"}
    empty = np.array([], np.int32)
    graph = Connectome(np.arange(3), empty, empty, np.array([], np.float64), np.array([], np.float64))
    events = np.zeros((1, 7, 0), np.uint8)
    stimulus = Stimulus(events, (), (), (0,), 0, 0, sha256(events.tobytes()).hexdigest(), model.dt_ms)
    prepared = configured.prepare(graph, stimulus, model)
    actual, state = configured.run_prepared(prepared, stimulus, return_state=True)
    np.testing.assert_array_equal(actual.spikes.neurons, np.asarray(monitor.i[:]))
    np.testing.assert_array_equal(actual.spikes.steps, np.rint(np.asarray(monitor.t_[:])/model.dt_s).astype(np.int64))
    actual_state = configured.host_state(state, prepared)
    expected_state = {name: np.asarray(getattr(neurons, name+"_")[:]).copy().reshape(1, -1)
                      for name in actual_state}
    assert_state(actual_state, expected_state)


@pytest.mark.metal
def test_valid_state_names_do_not_shadow_stable_math_helpers(monkeypatch):
    require_metal()
    b = pytest.importorskip("brian2")
    from configured_reference import reference
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
    from test_mlx_configured import graph, inputs
    monkeypatch.setitem(b.prefs, "codegen.target", "numpy")
    model = resolve_model({"eqs": DEFAULT_EQUATIONS+"d_expm1:1\nd_log1p:1\nd_exprel:1\n",
        "eq_th": "True", "t_rfc": 0*b.ms,
        "eq_rst": "v=v_rst;g=0*mV;d_expm1=expm1((i+1)/32);d_log1p=log1p((i+1)/32);d_exprel=exprel((i+1)/32)"})
    c, s = graph(), inputs(model, targets=(), steps=7, trials=1)
    expected = reference(c, s, model)
    prepared = configured.prepare(c, s, model)
    _, state = configured.run_prepared(prepared, s, return_state=True)
    assert_state(configured.host_state(state, prepared), expected.state)


@pytest.mark.metal
def test_independent_component_benchmark_matches_numpy_and_default_cython(monkeypatch, tmp_path):
    require_metal()
    from mlx_backend import configured
    from component_cpu_reference import (ComponentCpu, component_inputs,
                                         stable_model_configuration as model_configuration)
    model = model_configuration()
    graph, stimulus = component_inputs(4097, 7, model)
    cache = Path(__file__).resolve().parents[1]/"executions/mlx-feature-parity/math64-cython"/tmp_path.name
    numpy = ComponentCpu(4097, 7, model, "numpy", cache)
    _, _, expected = numpy.run()
    cython = ComponentCpu(4097, 7, model, "cython", cache)
    assert cython.code_object_types == ["CythonCodeObject"]
    _, _, compiled = cython.run()
    assert_state(compiled, expected)
    prepared = configured.prepare(graph, stimulus, model)
    assert not prepared.compact_idle
    assert all("_eon_stable_"+name in prepared.program.reset_code for name in FUNCTIONS)
    assert "_eon_stable_exprel_cached(" in prepared.program.reset_code
    for record in (False, True):
        result, state = configured.run_prepared(prepared, stimulus, return_state=True, record=record)
        assert result.spikes.steps.size == (4097*7 if record else 0)
        assert_state(configured.host_state(state, prepared), expected)
    assert expected["activity_expm1"][0, 0] == expected["activity_expm1"][0, 4096]
    assert np.all(expected["phase"] == 7)
    assert np.count_nonzero(expected["activity_expm1"]) == 4097




@pytest.mark.metal
@pytest.mark.parametrize("mutation", ["", "x+=1/32", "a=0", "x=expm1(x)"])
def test_exprel_reuses_only_unchanged_expm1_values(monkeypatch, mutation):
    require_metal()
    b = pytest.importorskip("brian2")
    from configured_reference import reference
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
    from test_mlx_configured import graph, inputs
    monkeypatch.setitem(b.prefs, "codegen.target", "numpy")
    reset = "v=v_rst;g=0*mV;x=(i+phase+1)/32;a=expm1(x);"
    reset += (mutation+";" if mutation else "")+"y=exprel(x);phase+=1"
    model = resolve_model({"eqs": DEFAULT_EQUATIONS+"x:1\na:1\ny:1\nphase:integer\n",
        "eq_th": "True", "t_rfc": 0*b.ms, "eq_rst": reset})
    c, s = graph(), inputs(model, targets=(), steps=13, trials=1)
    expected = reference(c, s, model)
    p = configured.prepare(c, s, model)
    assert ("_eon_stable_exprel_cached(" in p.program.reset_code) == (mutation == "")
    actual, state = configured.run_prepared(p, s, return_state=True, chunk_steps=7)
    np.testing.assert_array_equal(actual.spikes.neurons, expected.spikes.neurons)
    np.testing.assert_array_equal(actual.spikes.steps, expected.spikes.steps)
    assert_state(configured.host_state(state, p), expected.state)

