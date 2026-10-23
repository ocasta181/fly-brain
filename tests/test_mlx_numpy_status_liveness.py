"""Error-policy inputs must exist even when their numerical result is unused."""
from dataclasses import replace

import numpy as np
import pytest

b = pytest.importorskip('brian2')

from mlx_backend.compiler import compile_model
from mlx_backend.expression_lowering import Draw, Segment
from mlx_backend.model import DEFAULT_EQUATIONS, DEFAULT_RESET, resolve_model


@pytest.mark.parametrize('reset,value,function', [
    ('p=k', 1e100, 'numpy_status'),
    ('unused=int(k);p=0', float('nan'), 'numpy_python_int'),
])
def test_policy_inputs_have_preceding_native_producers(monkeypatch, reset, value, function):
    monkeypatch.setitem(b.prefs, 'codegen.target', 'numpy')
    monkeypatch.setitem(b.prefs, 'core.default_float_dtype', np.float32)
    model = resolve_model(dict(eqs=DEFAULT_EQUATIONS+'p:1\n', eq_th='True',
        eq_rst=DEFAULT_RESET+';'+reset, k=value))
    model = replace(model, numpy_policy_ir=True,
        numpy_error_actions=tuple((name, 'raise') for name in ('divide', 'over', 'under', 'invalid')))
    operations = compile_model(model).random_program.stages['reset']
    produced = set()
    checked = []
    for operation in operations:
        if isinstance(operation, Segment):
            produced.update(operation.writes)
        elif isinstance(operation, Draw) and operation.function in ('numpy_status', 'numpy_python_int'):
            assert operation.result in produced
            checked.append(operation.function)
    assert function in checked


def test_vector_widening_commits_destination_before_policy_delivery(monkeypatch):
    monkeypatch.setitem(b.prefs, 'codegen.target', 'numpy')
    monkeypatch.setitem(b.prefs, 'core.default_float_dtype', np.float32)
    model = resolve_model(dict(eqs=DEFAULT_EQUATIONS+'x:second\n', eq_th='True',
        eq_rst=DEFAULT_RESET+';lastspike=x'))
    model = replace(model, numpy_policy_ir=True,
        numpy_error_actions=tuple((name, 'raise') for name in ('divide', 'over', 'under', 'invalid')))
    operations = compile_model(model).random_program.stages['reset']
    deliveries = [(index, operation) for index, operation in enumerate(operations)
        if isinstance(operation, Draw) and operation.function=='numpy_status' and not operation.scalar]
    assert len(deliveries)==1
    index, _ = deliveries[0]
    assert isinstance(operations[index-1], Segment)
    assert 'lastspike' in operations[index-1].writes
