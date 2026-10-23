"""Exact original-API counts across recording, continuation and replay."""

import numpy as np
import pytest

pytestmark = pytest.mark.metal


def test_valid_word_lengths_and_count_wrap():
    import mlx.core as mx
    from mlx_backend.brian_device import _count_spike_words

    words = [np.array([0xffffffff, 0x80000000, 3, 0], np.uint32),
             np.array([0xffffffff, 0xfffffffe, 0x80000001, 7], np.uint32)]
    previous = np.array([0xfffffff0, 0x7fffffff, 0xffffffff, 2], np.uint32)
    expected = previous.astype(object)
    counts = mx.array(previous)
    for word, length in zip(words, [32, 1]):
        expected += np.array([sum(bool(int(value) & (1 << bit)) for bit in range(length))
                              for value in word], object)
        counts = _count_spike_words([mx.array(word)], [length], counts)
        np.testing.assert_array_equal(np.asarray(counts), np.array([int(v) % 2**32 for v in expected], np.uint32))


@pytest.mark.parametrize('target', ['numpy', 'cython'])
@pytest.mark.parametrize('record', [False, True])
def test_monitor_offsets_activation_and_continuation(tmp_path, monkeypatch, target, record):
    import brian2 as b
    import original_api_reference as reference
    from original_api_protocol import RunObservation, check_original_api_runs

    monitor_values=[]
    monitors=[]
    def build(mode):
        fixture = reference.build_original_api_fixture(tmp_path/str(mode=='native'), parameters=dict(
            eqs='v:volt\ng:volt\nrfc:second', eq_th='randn()>.2',
            eq_rst='v=v_rst;g=0*mV', t_rfc=.2*b.ms, t_dly=.3*b.ms,
            r_poi=3700*b.Hz, r_poi2=7200*b.Hz), silenced=())
        fixture.network.remove(fixture.monitor)
        monitors[:] = [b.SpikeMonitor(fixture.neurons, record=record),
                    b.SpikeMonitor(fixture.neurons, record=False),
                    b.SpikeMonitor(fixture.neurons, record=record)]
        fixture.monitor = monitors[0]
        fixture.network.add(*monitors)
        monitors[2].active = False
        for monitor in monitors:
            monitor.variables['count'].get_value()[:] = np.array([0x7ffffffe, -2, 5, 0], np.int32)
        monitors[1].variables['N'].get_value()[0] = 0x7ffffffe
        return fixture

    def exercise(fixture, mode):
        for index, (duration, active, primary) in enumerate([
                (20.1, False, True), (.2, True, True), (0., True, True),
                (.7, False, False), (.1, True, True)]):
            monitors[0].active = monitors[1].active = primary
            monitors[2].active = active
            fixture.run_ms(duration)
            values=[{name:np.array(monitor.variables[name].get_value(),copy=True)
                     for name in ('i','t','count','N') if name in monitor.variables}
                    for monitor in monitors]
            yield RunObservation(fixture.capture(),rng=reference.capture_rng_state())
            if mode=='cpu':monitor_values.append(values)
            else:
                for current, previous in zip(values,monitor_values[index]):
                    for name in current:
                        assert current[name].dtype == previous[name].dtype
                        assert current[name].tobytes() == previous[name].tobytes(), name
    check_original_api_runs(build,exercise,monkeypatch,seed=875,target=target)


def test_recording_fallback_counts(tmp_path, monkeypatch):
    from mlx_backend import expression_recording
    monkeypatch.setattr(expression_recording, 'collect', lambda *args: None)
    test_monitor_offsets_activation_and_continuation(tmp_path, monkeypatch, 'numpy', True)
