"""Uniform output decoding preserves real Brian monitor/continuation semantics."""
from native_execution import forbid_cpu_steps
from contextlib import nullcontext

import numpy as np
import pytest


@pytest.mark.metal
@pytest.mark.parametrize('capacity', [2, 2**22])
def test_uniform_output_continuation_and_monitors(tmp_path, monkeypatch, capacity):
    import brian2 as b
    import pandas as pd
    import original_api_reference as reference
    from mlx_backend import expression_recording
    from mlx_backend.brian_device import original_api

    monkeypatch.setattr(expression_recording, 'MAX_EVENTS', capacity)
    dataset = reference.write_tiny_dataset
    def local_dataset(directory):
        nodes, edges = dataset(directory)
        frame = pd.read_parquet(edges)
        frame['Postsynaptic_Index'] = frame['Presynaptic_Index']
        frame['Excitatory x Connectivity'] = 0.
        frame.to_parquet(edges)
        return nodes, edges
    monkeypatch.setattr(reference, 'write_tiny_dataset', local_dataset)
    uniform_results = []
    collect = expression_recording.collect_host_words
    def observed(*args):
        result = collect(*args)
        uniform_results.append(result is not None)
        return result
    monkeypatch.setattr(expression_recording, 'collect_host_words', observed)
    expected = None
    for native in (False, True):
        with reference.cpu_runtime(seed=953, target='cython'):
            with original_api(workers=False) if native else nullcontext() as device:
                fixture = reference.build_original_api_fixture(
                    tmp_path/str(native), parameters=dict(
                        eqs=reference.load_original_model().default_params['eqs']+'x:1\n',
                        eq_th='x>.5', eq_rst='v=v_rst', t_rfc=0*b.ms, t_dly=0*b.ms),
                    exc=(), exc2=(), silenced=(), initial_state={'x': 1.})
                counts = b.SpikeMonitor(fixture.neurons, record=False)
                inactive = b.SpikeMonitor(fixture.neurons)
                inactive.active = False
                fixture.network.add(counts, inactive)
                for monitor in (fixture.monitor, counts, inactive):
                    monitor.variables['count'].get_value()[:] = np.iinfo(np.int32).max-1
                counts.variables['N'].get_value()[0] = np.iinfo(np.int32).max-1
                if native:
                    forbid_cpu_steps(fixture.network, monkeypatch)
                snapshots = []
                retained = []
                for index, (values, duration) in enumerate(((1., 0.), (1., .1), (1., 3.3), (1., 205.),
                        ([1., 0., 0., 0.], .2), ([1., 0., 1., 0.], .2), (0., .1), (1., .2))):
                    if index == 2:
                        fixture.neurons.rfc_[0] = .0003
                        fixture.neurons.lastspike_[0] = float(fixture.neurons.clock.variables['t'].get_value()[0])-.0001
                    fixture.neurons.x = values
                    fixture.run_ms(duration)
                    if native:
                        for array, data in retained:
                            assert array.tobytes() == data
                        for name in ('rfc', 'lastspike'):
                            snapshot = device._sessions[fixture.network].host_state[name]
                            assert not np.shares_memory(snapshot, fixture.neurons.variables[name].get_value())
                            if duration:
                                assert not snapshot.flags.writeable
                                retained.append((snapshot, snapshot.tobytes()))
                            else:
                                assert snapshot.flags.owndata
                    extra = {f'{i}:{name}': monitor.variables[name].get_value().copy()
                             for i, monitor in enumerate((counts, inactive))
                             for name in ('count', 'N')}
                    snapshots.append((fixture.capture(), extra))
                if native:
                    for (actual, extra), (wanted, wanted_extra) in zip(snapshots, expected):
                        reference.assert_snapshot_equal(actual, wanted)
                        for key in extra:
                            assert extra[key].tobytes() == wanted_extra[key].tobytes(), key
                else:
                    expected = snapshots
    assert any(uniform_results) == (capacity == 2**22)
    assert not all(uniform_results)
