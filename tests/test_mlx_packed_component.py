"""Independent exact CPU/Metal check of the performance component itself."""

from pathlib import Path

import numpy as np
import pytest
from native_execution import require_metal


@pytest.mark.metal
def test_component_retains_binding_limit_trigger_and_matches_cpu(monkeypatch, tmp_path):
    mx = require_metal('Metal is required')
    from mlx_backend import configured
    from component_cpu_reference import (PackedComponentCpu as ComponentCpu, component_inputs,
                                         packed_model_configuration as model_configuration)
    model = model_configuration()
    graph, stimulus = component_inputs(7, 5, model)
    cache = Path(__file__).resolve().parents[1]/"executions/mlx-feature-parity/packed-component-cython"/tmp_path.name
    numpy = ComponentCpu(7, 5, model, "numpy", cache)
    _, _, expected = numpy.run()
    cython = ComponentCpu(7, 5, model, "cython", cache)
    assert cython.code_object_types == ["CythonCodeObject"]
    _, _, compiled = cython.run()
    prepared = configured.prepare(graph, stimulus, model)
    layout = configured._state_layout(prepared.program)
    assert len(layout.names) == 42
    assert len(layout.groups) == 3
    assert set(expected) == set(layout.names)
    for record in (False, True):
        result, state = configured.run_prepared(prepared, stimulus, return_state=True, record=record)
        assert len(state.values.buffers) == 3
        assert result.spikes.steps.size == (7*5 if record else 0)
        actual = configured.host_state(state, prepared)
        for name, values in expected.items():
            raw = lambda x: x.view(np.uint64) if x.dtype == np.float64 else x
            np.testing.assert_array_equal(raw(compiled[name]), raw(values), err_msg="Cython "+name)
            np.testing.assert_array_equal(raw(actual[name]), raw(values), err_msg="Metal "+name)
    assert np.all(expected["count11"] == 60)
    assert np.all(expected["flag11"])
    assert np.all(expected["extra11"] == 5*12/1024+1.)
