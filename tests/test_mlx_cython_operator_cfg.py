"""Actual Cython exception exits and repaired native operator continuation."""
from native_execution import forbid_cpu_steps as guard_cpu_steps
from pathlib import Path

import brian2 as b
import pytest

from mlx_backend.model import DEFAULT_EQUATIONS, DEFAULT_RESET
from original_api_protocol import check_original_api_runs, observe_run
from original_api_reference import build_original_api_fixture


@pytest.mark.metal
@pytest.mark.parametrize("role", ["scalar", "vector", "hoisted_division", "scalar_then_vector"])
@pytest.mark.parametrize("spikes", [False, True])
def test_cython_known_zero_operator_exit_and_namespace_repair(tmp_path, monkeypatch, role, spikes):
    monkeypatch.setitem(b.prefs, "codegen.runtime.cython.cache_dir", str(
        Path(__file__).resolve().parents[1] / "executions/cython-floating-operators"))
    reset = {"scalar": "q=1./zero;p=x**y", "vector": "p=x**y;q=x%zero",
             "hoisted_division": "p=x**y;q=x/zero",
             "scalar_then_vector": "q=1./zero;p=x**y;r=x%y"}[role]
    params = dict(eqs=DEFAULT_EQUATIONS + "x:1\ny:1\np:1\nq:1\n" +
                  ("r:1\n" if role == "scalar_then_vector" else ""),
        eq_th="True" if spikes else "False", eq_rst=DEFAULT_RESET + ";" + reset,
        zero=0., t_rfc=0 * b.ms)
    options = dict(parameters=params, exc=(), exc2=(), silenced=(),
        initial_state=dict(x=[.3, 3., 1.3, .1], y=[.1, .3, .7, 2.1],
                           p=[13., 14., 15., 16.], q=[17., 18., 19., 20.]))
    if role == "scalar_then_vector":
        options["initial_state"]["r"] = [21., 22., 23., 24.]
    fails = role != "vector" or spikes

    def exercise(fixture, mode):
        if fails:
            first=observe_run(fixture,.1,raises=b.BrianObjectException)
            assert isinstance(first.error.__cause__, ZeroDivisionError)
        else:
            first=observe_run(fixture,.1)
        yield first
        fixture.neurons.namespace["zero"] = 1.
        yield observe_run(fixture,.15)

    def guard(fixture, patch):
        guard_cpu_steps(fixture.network, patch, 'Native Cython CFG executed a CPU timestep', patch_preparation=True)
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/mode,**options),
        exercise,monkeypatch,guard=guard,target='cython')
