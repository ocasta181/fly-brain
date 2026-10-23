"""Exact reference math, including native threshold/reset and ODE execution."""
from state_assertions import assert_spikes_equal

import ctypes
import platform
import shutil
import subprocess
import sys

import numpy as np
import pytest
from native_execution import require_metal

from mlx_backend import binary64, math64
from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
from mlx_backend.transcendentals import metal_math_header


def cases(name):
    rng = np.random.RandomState(20261017)
    if name == "exp":
        values = np.r_[rng.uniform(-746., 710., 8192),
                       -np.geomspace(1e-320, 1., 256), np.geomspace(1e-320, 1., 256),
                       0., -0., np.inf, -np.inf, np.nan]
        for x in (-745.25, -745.1332191019411, -708.3964185322641, 709.782712893384):
            values = np.r_[values, np.nextafter(x, -np.inf), x, np.nextafter(x, np.inf)]
        return values
    raw = rng.randint(0, 2**32, size=(8192, 2), dtype=np.uint64)
    raw = ((raw[:, 0] << 32) | raw[:, 1]) & 0x7fffffffffffffff
    return np.r_[raw.view(np.float64), np.arange(1, 65, dtype=np.uint64).view(np.float64),
                 np.geomspace(1e-320, 1e-300, 64), 1., np.nextafter(1., 0.),
                 np.nextafter(1., np.inf), 0., -0., -1., np.inf, -np.inf, np.nan]


def assert_reference_bits(actual, expected):
    # Floating exceptions/NaN payloads are not simulation state semantics.
    # Every non-NaN bit, including signed zero and subnormals, must match.
    mask = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[mask], expected.view(np.uint64)[mask])
    assert np.all(np.isnan(actual.view(np.float64)[~mask]))


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    if sys.platform != "darwin" or platform.machine().lower() not in ("arm64", "aarch64"):
        pytest.skip("The frozen reference libm is Apple arm64")
    compiler = shutil.which("clang++")
    if compiler is None or ctypes.sizeof(ctypes.c_long) != 8:
        pytest.skip("Host checks require Clang and 64-bit long")
    from test_mlx_math64 import HOST_PREFIX
    directory = tmp_path_factory.mktemp("transcendental-host")
    source, library = directory / "math.cpp", directory / "math.so"
    source.write_text(HOST_PREFIX + "\n#define constant static const\n" + binary64.MSL_HEADER
                     + math64.MSL_MATH_HEADER + metal_math_header({"exp", "log"}) + r'''
extern "C" {
ulong exp_bits(ulong x) { return d_exponential(x); }
ulong log_bits(ulong x) { return d_logarithm(x); }
}
''')
    subprocess.run([compiler, "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
                    "-shared", "-fPIC", str(source), "-o", str(library)],
                   check=True, capture_output=True, text=True, timeout=25)
    lib = ctypes.CDLL(str(library))
    for name in ("exp", "log"):
        f = getattr(lib, name+"_bits")
        f.argtypes, f.restype = [ctypes.c_uint64], ctypes.c_uint64
    return lib


@pytest.mark.parametrize("name", ["exp", "log"])
def test_host_helpers_match_numpy_full_range(host, name):
    values = cases(name)
    with np.errstate(all="ignore"):
        expected = getattr(np, name)(values)
    actual = np.array([getattr(host, name+"_bits")(int(x))
                       for x in values.view(np.uint64)], dtype=np.uint64)
    assert_reference_bits(actual, expected)


def metal():
    mx = require_metal('Metal is required')
    return mx


@pytest.mark.metal
@pytest.mark.parametrize("name", ["exp", "log"])
def test_metal_helpers_match_numpy_full_range(name):
    mx = metal()
    values = cases(name)
    function = "d_exponential" if name == "exp" else "d_logarithm"
    header = binary64.MSL_HEADER + math64.MSL_MATH_HEADER + metal_math_header({name})
    kernel = mx.fast.metal_kernel(name="math64_"+name, input_names=["values"], output_names=["out"],
        header=header, source=f"uint i=thread_position_in_grid.x; if(i<SIZE) out[i]={function}(values[i]);",
        compile_options={"math_mode": "safe"})
    (out,) = kernel(inputs=[mx.array(values.view(np.uint64))], template=[("SIZE", values.size)],
                    grid=(values.size, 1, 1), threadgroup=(128, 1, 1),
                    output_shapes=[values.shape], output_dtypes=[mx.uint64])
    mx.eval(out)
    with np.errstate(all="ignore"):
        expected = getattr(np, name)(values)
    assert_reference_bits(np.asarray(out), expected)


@pytest.mark.metal
def test_dynamic_exp_log_odes_thresholds_resets_and_no_record_match_brian(monkeypatch):
    metal()
    b = pytest.importorskip("brian2")
    from configured_reference import reference
    from mlx_backend import configured
    from test_mlx_configured import graph, inputs

    monkeypatch.setitem(b.prefs, "codegen.target", "numpy")
    model = resolve_model({
        "eqs": DEFAULT_EQUATIONS + "da/dt=-a/tau+log(i+1)/second:1\nz:1\ny:1\n",
        "eq_th": "(t<dt) or ((log(1+z)>0) and (exp(y)>=1))",
        "eq_rst": "v=v_rst;g=0*mV;z=exp((i-1)/64+t/(32*dt));y=log(1+z)",
        "t_rfc": 0*b.ms,
    })
    c, s = graph(), inputs(model, targets=(), steps=13, trials=2)
    expected = reference(c, s, model)
    p = configured.prepare(c, s, model)
    assert "d_logarithm(" in p.program.state_code
    assert "d_exponential(" in p.program.threshold_code
    for recording in (True, False):
        actual, state = configured.run_prepared(p, s, record=recording, return_state=True, chunk_steps=3)
        if recording:
            assert_spikes_equal(actual.spikes, expected.spikes)
        else:
            assert not actual.spikes.steps.size
        for name, expected_values in expected.state.items():
            values = configured.host_state(state, p)[name]
            if values.dtype == np.float64:
                assert_reference_bits(values.ravel().view(np.uint64), expected_values.ravel())
            else:
                np.testing.assert_array_equal(values, expected_values, err_msg=name)


@pytest.mark.metal
@pytest.mark.parametrize("name,argument", [("exp", 1.), ("log", 1.+1e-9)])
@pytest.mark.parametrize("comparison", [">", ">="])
def test_exact_math_threshold_boundary_matches_cpu(name, argument, comparison, monkeypatch):
    metal()
    b = pytest.importorskip("brian2")
    from configured_reference import reference
    from mlx_backend import configured
    from test_mlx_configured import graph, inputs

    monkeypatch.setitem(b.prefs, "codegen.target", "numpy")
    model = resolve_model({"eqs": DEFAULT_EQUATIONS+"a:1\n",
        "eq_th": f"(t<dt) or ({name}(a){comparison}boundary)",
        "eq_rst": "v=v_rst;g=0*mV;a=argument", "argument": argument,
        "boundary": getattr(np, name)(np.float64(argument)), "t_rfc": 0*b.ms})
    c, s = graph(), inputs(model, targets=(), steps=7, trials=1)
    expected = reference(c, s, model)
    p = configured.prepare(c, s, model)
    actual, state = configured.run_prepared(p, s, return_state=True, chunk_steps=3)
    assert actual.spikes.steps.size == (3 if comparison == ">" else 21)
    assert_spikes_equal(actual.spikes, expected.spikes)
    assert_reference_bits(configured.host_state(state, p)["a"].ravel().view(np.uint64),
                          expected.state["a"].ravel())
    other = "log" if name == "exp" else "exp"
    assert f"_eon_{other}_segments" not in p.program.header_code
