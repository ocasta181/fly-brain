"""Independent NumPy component comparisons across native execution modes."""
from dataclasses import replace

import numpy as np

from state_assertions import assert_raw_state
from native_execution import forbid_call


def assert_numpy_component_modes(monkeypatch, tmp_path, mx, model, initial_values,
                                 *, prefix, cpu_error):
    """Compare the same two-step component with recording and chunked resume."""
    import brian2 as b
    from component_cpu_reference import ComponentCpu, component_inputs
    from mlx_backend import configured

    count = len(next(iter(initial_values.values())))
    cpu = ComponentCpu(count, 2, model, 'numpy', tmp_path/'cache', prefix=prefix)
    for name, values in initial_values.items():
        setattr(cpu.neurons, name, values)
    cpu.network.store('initial')
    _, _, expected = cpu.run()
    (tmp_path/'actual-reset.txt').write_text(str(cpu.neurons.resetter['spike'].codeobj.code))

    graph, stimulus = component_inputs(count, 2, model)
    prepared = configured.prepare(graph, stimulus, model)
    initial = configured.initial_state(prepared)
    encoded = {}
    for name, values in initial_values.items():
        if values.dtype.kind == 'f':
            values = values.view(np.uint32 if values.dtype.itemsize == 4 else np.uint64)
        encoded[name] = mx.array(values)
    initial = replace(initial, values={**initial.values, **encoded})
    monkeypatch.setattr(configured, 'initial_state', lambda *args, **kwargs: initial)

    forbid_call(b.Network, 'run', cpu_error, monkeypatch)

    def check(state):
        assert_raw_state(configured.host_state(state, prepared), expected)

    for record in (False, True):
        _, state = configured.run_prepared(prepared, stimulus, record=record, return_state=True)
        check(state)
    resumed = initial
    for _ in range(2):
        resumed, _ = configured.advance(prepared, resumed, mx.zeros((1,), mx.uint8), 2, 1)
    configured.evaluate_state(resumed)
    check(resumed)
