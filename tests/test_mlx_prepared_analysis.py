"""Reuse actual Brian preparation while preserving mutable API contracts."""
import numpy as np
import pytest

from test_mlx_brian_device import _run_pair


@pytest.mark.metal
@pytest.mark.parametrize('target', ['numpy', 'cython'])
def test_prepared_analysis_retains_namespace_mutations(tmp_path, monkeypatch, target):
    import brian2 as b
    import mlx_backend.model as model

    # Public model resolution still prepares its own independent snapshot.
    monkeypatch.setitem(b.prefs, 'codegen.target', target)
    assert model.resolve_model({'v_th': -44*b.mV}).analysis is not None
    def duplicate_analysis(_):
        raise AssertionError('Original API rebuilt a second analysis group')
    monkeypatch.setattr(model, '_analyse', duplicate_analysis)
    def mutate(fixture):
        fixture.neurons.namespace['v_th'] = -49.25*b.mV
        fixture.neurons.namespace['v_0'] = -50.5*b.mV
    _run_pair(tmp_path, [.15, 0., .35, 2.1], target=target, mutations={1: mutate},
              parameters={'r_poi': 3700*b.Hz, 'r_poi2': 1400*b.Hz},
              initial_state={'v': np.array([-44.,-49.,-48.,-50.])*b.mV,
                             'g': np.array([3.,-2.,4.,1.])*b.mV})
