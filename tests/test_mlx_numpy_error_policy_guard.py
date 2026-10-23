"""NumPy policy continuation and actual code-object target controls."""
from pathlib import Path

import numpy as np
import pytest

b = pytest.importorskip("brian2")
from native_execution import require_metal
mx = require_metal('Metal required', allow_module_level=True)

from brian2.groups.group import CodeRunner
from brian2.codegen.runtime.cython_rt.cython_rt import CythonCodeObject
from mlx_backend.brian_device import original_api
from mlx_backend.model import DEFAULT_EQUATIONS, DEFAULT_RESET
from original_api_reference import (
    assert_rng_state_equal, assert_snapshot_equal, build_original_api_fixture,
    capture_rng_state, cpu_runtime, original_runtime,
)

pytestmark = pytest.mark.metal


@pytest.fixture(autouse=True)
def private_cython_cache(monkeypatch):
    monkeypatch.setitem(b.prefs, "codegen.runtime.cython.cache_dir", str(
        Path(__file__).resolve().parents[1] / "executions/cython-error-policy"))


def power_options(*, inputs=False):
    options = dict(parameters=dict(
        eqs=DEFAULT_EQUATIONS + "x:1\ny:1\np:1\n", eq_th="True",
        eq_rst=DEFAULT_RESET + ";p=x**y", t_rfc=0*b.ms, w_syn=0*b.mV),
        initial_state=dict(x=[.3, 0., 3., 2.], y=[2., -1., 2., 2.]))
    if not inputs:
        options.update(exc=(), exc2=(), silenced=())
    return options


@pytest.mark.parametrize("policy", ["default", "ignore"])
def test_numpy_actions_preserve_zero_and_partial_runs(tmp_path, policy):
    options = power_options(inputs=True)
    actions = dict(divide="warn", over="warn", under="ignore", invalid="warn")
    if policy == "ignore":
        actions = dict(all="ignore")
    with cpu_runtime():
        cpu = build_original_api_fixture(tmp_path / "cpu", **options)
        with np.errstate(**actions):
            expected = [cpu.run_ms(duration) for duration in (0., .15, .15)]
        expected_rng = capture_rng_state()
    with original_runtime(original_api) as device:
        fixture = build_original_api_fixture(tmp_path / "native", **options)
        with np.errstate(**actions):
            assert_snapshot_equal(fixture.run_ms(0.), expected[0])
            assert_snapshot_equal(fixture.run_ms(.15), expected[1])
            assert_snapshot_equal(fixture.run_ms(.15), expected[2])
        assert_rng_state_equal(capture_rng_state(), expected_rng)
        assert device.native_runs == 3
        assert device.native_steps == expected[-1].clock_timestep


def set_runner_target(fixture, target):
    for obj in fixture.network.sorted_objects:
        if isinstance(obj, CodeRunner):
            obj.codeobj_class = target


def test_explicit_cython_runners_allow_ambient_numpy_policy(tmp_path):
    options = power_options()
    with cpu_runtime():
        cpu = build_original_api_fixture(tmp_path / "cpu", **options)
        set_runner_target(cpu, CythonCodeObject)
        with np.errstate(all="raise"):
            expected = cpu.run_ms(.1)
        assert cpu.neurons.resetter["spike"].codeobj.class_name == "cython"
        assert cpu.neurons.resetter["spike"].codeobj.compiled_code["run"] is not None
    with original_runtime(original_api) as device:
        fixture = build_original_api_fixture(tmp_path / "native", **options)
        set_runner_target(fixture, CythonCodeObject)
        assert b.prefs.codegen.target == "numpy"
        with np.errstate(all="raise"):
            observed = fixture.run_ms(.1)
        for obj in fixture.network.sorted_objects:
            if isinstance(obj, CodeRunner):
                assert obj.codeobj.class_name == "cython"
        assert_snapshot_equal(observed, expected)
        assert device.native_steps == 1
