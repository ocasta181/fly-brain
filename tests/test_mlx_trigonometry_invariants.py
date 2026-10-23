"""Actual original CPU/native scalar trigonometry preparation and identity."""
from trigonometry_checks import OriginalComparison
import numpy as np
import pytest
from native_execution import require_metal
b=pytest.importorskip("brian2")
from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model

def test_nonfinite_numeric_namespace_preserves_payload_identity(monkeypatch):
    monkeypatch.setitem(b.prefs,"codegen.target","numpy")
    parameters={"eqs":DEFAULT_EQUATIONS+"sample : 1\n", "eq_th":"t>=0*ms",
                "eq_rst":"v=v_rst; sample=cos(scalar_phase)"}
    first=np.uint64(0xfff8000000000123).view(np.float64)
    second=np.uint64(0xfff8000000000124).view(np.float64)
    a=resolve_model(dict(parameters,scalar_phase=first))
    z=resolve_model(dict(parameters,scalar_phase=second))
    assert a.signature!=z.signature
    assert np.asarray(a.source_namespace["scalar_phase"]).view(np.uint64)==first.view(np.uint64)
    assert resolve_model(dict(parameters,scalar_phase=float(first))).signature!=resolve_model(
        dict(parameters,scalar_phase=float(second))).signature
    nonfinite = resolve_model(dict(parameters, scalar_phase=first, v_0=np.nan*b.mV))
    assert nonfinite.signature != a.signature
    assert np.isnan(nonfinite.v_0)
    assert np.asarray(nonfinite.source_namespace["v_0"], np.float64).view(np.uint64) == np.asarray(
        np.nan*b.mV, np.float64).view(np.uint64)


def test_forward_optimisation_preference_is_frozen_in_model_identity(monkeypatch):
    monkeypatch.setitem(b.prefs,"codegen.target","numpy")
    parameters={"eqs":DEFAULT_EQUATIONS+"sample : 1\nphase : boolean\n", "eq_th":"t>=0*ms",
                "eq_rst":"v=v_rst; sample=sin(phase)+.1"}
    monkeypatch.setitem(b.prefs,"codegen.loop_invariant_optimisations",True)
    optimised=resolve_model(parameters)
    monkeypatch.setitem(b.prefs,"codegen.loop_invariant_optimisations",False)
    unoptimised=resolve_model(parameters)
    assert optimised.reference_optimise is True and unoptimised.reference_optimise is False
    assert optimised.signature!=unoptimised.signature
    from mlx_backend.model import model_preferences
    # Subsequent ambient preferences must not change the saved target graph.
    monkeypatch.setitem(b.prefs,"codegen.loop_invariant_optimisations",True)
    with model_preferences(unoptimised):
        assert b.prefs.codegen.loop_invariant_optimisations is False
    assert b.prefs.codegen.loop_invariant_optimisations is True


@pytest.mark.metal
@pytest.mark.parametrize("target,expression",[("numpy","scalar_phase"),("cython","scalar_phase"),
    ("numpy","sqrt(scalar_phase)"),("cython","sqrt(scalar_phase)"),("cpp","sqrt(scalar_phase)")])
def test_original_scalar_cosine_negative_nan_folding(tmp_path,target,expression):
    mx = require_metal('Metal required')
    negative_nan=np.uint64(0xfff8000000000123).view(np.float64)
    options=dict(argument_expression=expression,namespace_values={"scalar_phase":
                 negative_nan if expression=="scalar_phase" else np.float64(-1.)},
                 cache_suffix="-scalar-cosine")
    comparison=OriginalComparison(target, 'cos', np.zeros(4), tmp_path, **options)
    # Compare standalone scalar libm results too, using the independently
    # resolved Cython scalar policy for native execution in this API fixture.
    native_target="cython" if target=="cpp" else target
    comparison.check_native(target=native_target, compare_report=False)


@pytest.mark.metal
@pytest.mark.parametrize("function,value",[("floor",256.5),("ceil",255.5)])
def test_original_numpy_integer_rounding_scalar_controls_half_promotion(tmp_path,function,value):
    mx = require_metal('Metal required')
    symbol="phase_scalar_"+function
    options=dict(state_inputs=True,phase_dtype="boolean",phase_symbol=symbol,optimise=False,
        namespace_values={"rounding_value":np.float64(value)},
        result_expression="sin("+symbol+")+"+function+"(rounding_value)")
    inputs=np.array([False,True,True,False])
    comparison=OriginalComparison('numpy', 'sin', inputs, tmp_path, **options)
    comparison.check_native()
