"""Complex lowering preserves the existing real and immutable controls."""

import numpy as np
import pytest


@pytest.mark.parametrize("optimise", [False, True])
@pytest.mark.parametrize("reset", [
    "x=sin(((-1.)**.3)*(i+1))*100",
    "z=(-1.)**.3;x=sin(z*(i+1))*100",
    "x=sin(((-1.)**.3)*(i+1+rand()))*100",
])
def test_typed_numpy_dynamic_complex_keeps_both_components(monkeypatch, optimise, reset):
    import brian2 as b
    from mlx_backend.compiler import compile_model
    from mlx_backend.model import resolve_model

    monkeypatch.setitem(b.prefs, "codegen.target", "numpy")
    monkeypatch.setitem(b.prefs, "codegen.loop_invariant_optimisations", optimise)
    model = resolve_model({"eqs": "v:volt\ng:volt\nrfc:second\nx:integer\n",
                           "eq_th": "True", "t_rfc": 0*b.ms,
                           "eq_rst": "v=v_rst;g=0*mV;"+reset})
    program=compile_model(model,neurons=4)
    source=program.reset_code
    if program.random_program is not None:
        source+='\n'.join(getattr(operation,'source','') for operations in
                          program.random_program.stages.values() for operation in operations)
    assert '_eon_complex_sin(' in source


@pytest.mark.metal
@pytest.mark.parametrize("optimise", [False, True])
def test_original_real_array_power_and_immutable_complex_controls(tmp_path, monkeypatch, optimise):
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from original_api_reference import (
        assert_rng_state_equal, assert_snapshot_equal, build_original_api_fixture,
        capture_rng_state, cpu_runtime, original_runtime, load_original_model,
    )

    pytest.importorskip("mlx.core")
    monkeypatch.setitem(b.prefs, "codegen.loop_invariant_optimisations", optimise)
    original = load_original_model()
    parameters = dict(original.default_params)
    parameters.update(
        eqs=parameters["eqs"]+"\nx:integer\ny:1\np:1\nflag:boolean\n",
        eq_th="True", t_rfc=0*b.ms, w_syn=0*b.mV,
        eq_rst=parameters["eq_rst"]+
        ";x=sin((-1.)**.3)*100;p=y**.3;flag=((-1.)**.3)!=0")
    options = dict(parameters=parameters, exc=(), exc2=(), silenced=(),
                   initial_state={"y": [-1., -8., 1., 8.]})
    with cpu_runtime():
        cpu = build_original_api_fixture(tmp_path/"cpu", **options)
        with np.errstate(invalid="ignore"):
            expected = cpu.run_ms(.1)
        expected_rng = capture_rng_state()
        np.testing.assert_array_equal(expected.state["x"], [74]*4)
        assert np.isnan(expected.state["p"][:2]).all()
    with original_runtime(original_api) as device:
        native = build_original_api_fixture(tmp_path/"native", **options)
        actual = native.run_ms(.1)
        assert_snapshot_equal(actual, expected)
        assert_rng_state_equal(capture_rng_state(), expected_rng)
        assert device.native_runs == device.native_steps == 1
