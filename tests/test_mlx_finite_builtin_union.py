"""Finite dynamic built-ins coexist through the original Eon object API."""
import json
import subprocess

import numpy as np
import pytest


@pytest.mark.metal
@pytest.mark.parametrize("local", [False, True], ids=["delayed", "local"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("target", ["numpy", "cython"])
def test_original_finite_builtin_union(tmp_path, monkeypatch, target, dtype, local):
    import brian2 as b
    from original_api_protocol import check_original_api_runs, observe_run
    from original_api_reference import build_original_api_fixture
    local_dispatch = []
    if local:
        import original_api_reference
        from mlx_backend import configured

        prepare = configured.prepare

        def observed_prepare(*args, **kwargs):
            prepared = prepare(*args, **kwargs)
            local_dispatch.append((prepared.local_synapses, prepared.program.collective_reset))
            return prepared

        monkeypatch.setattr(original_api_reference, "write_tiny_dataset", original_api_reference.synthetic_dataset(4, labelled=False, neuron_file="neurons.csv",
            edges=((0, 0, .25), (0, 0, -.25), (2, 2, 0.), (3, 3, -0.))))
        monkeypatch.setattr(configured, "prepare", observed_prepare)

    functions = (
        "cos(x)", "sin(x)", "tan(x)", "arccos(x)", "arcsin(x)", "arctan(x)",
        "cosh(x)", "sinh(x)", "tanh(x)", "exp(x)", "expm1(x)", "exprel(x)",
        "log(abs(x)+1)", "log10(abs(x)+1)", "log1p(x)", "sqrt(abs(x))",
        "ceil(x)", "floor(x)", "abs(x)", "sign(x)", "clip(x,-.5,.5)",
        "int(x)", "timestep(t,.1*ms)",
    )
    equations = "v:volt\ng:volt\nrfc:second\nx:1" + "".join(
        f"\nf{index}:1" for index in range(len(functions)))
    reset = "v=v_rst;g=0*mV;" + ";".join(
        f"f{index}={expression}" for index, expression in enumerate(functions)) + ";x=-x"
    def build(mode):
        return build_original_api_fixture(
            tmp_path/str(mode == 'native'), parameters=dict(
                eqs=equations, eq_th="True", eq_rst=reset,
                t_rfc=0*b.ms, t_dly=(0 if local else .2)*b.ms),
            initial_state={"x": [-.75, -.125, .125, .75]},
            exc=(), exc2=(), silenced=(),
        )
    def exercise(fixture, mode):
        snapshots = [observe_run(fixture,duration,rng=True) for duration in (0.,.2,.3)]
        codeobj = fixture.neurons.resetter["spike"].codeobj
        (tmp_path / ("native-reset.txt" if mode == 'native' else "cpu-reset.txt")).write_text(str(codeobj.code))
        if mode == 'cpu' and target == "cython":
            module = codeobj.compiled_code["run"].__file__
            (tmp_path / "cpu-module.json").write_text(json.dumps({"module": module}))
            (tmp_path / "cpu-imports.txt").write_text(subprocess.check_output(["nm", "-u", module], text=True))
        yield from snapshots
    check_original_api_runs(build,exercise,monkeypatch,seed=953,target=target,float_dtype=dtype)
    if local:
        # NumPy's binary64 int(x) keeps its rank-dependent collective reset.
        # Other tested targets can use the fused local path for this block.
        expected_dispatch = (False, True) if target == "numpy" and dtype == np.float64 else (True, False)
        assert local_dispatch and all(item == expected_dispatch for item in local_dispatch)
