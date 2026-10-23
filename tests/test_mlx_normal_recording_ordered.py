"""Chronological compaction and independent original-API recording contracts."""

import numpy as np
import pytest

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('capacity', [8, 128, 1024])
@pytest.mark.parametrize('density', ['empty', 'sparse', 'dense'])
def test_ordered_decode(monkeypatch, capacity, density):
    import mlx.core as mx
    from mlx_backend import expression_recording as recording
    monkeypatch.setattr(recording, 'MAX_EVENTS', capacity)
    blocks = [np.array([0, 1, 0x80000000, 0xffffffff], np.uint32),
              np.array([0xffffffff, 0, 2, 1], np.uint32)]
    if density == 'empty':
        blocks = [np.zeros_like(block) for block in blocks]
    elif density == 'dense':
        blocks = [np.full_like(block, 0xffffffff) for block in blocks]
    starts, lengths = [2**32+7, 2**32+41], [32, 2]
    rows, steps = recording.collect([mx.array(block) for block in blocks], starts, lengths)
    expected = [(row, start+bit) for block, start, length in zip(blocks, starts, lengths)
                for bit in range(length) for row, word in enumerate(block) if int(word) & (1 << bit)]
    assert rows.dtype == np.int32 and steps.dtype == np.int64
    assert list(zip(rows.tolist(), steps.tolist())) == expected


def test_unsupported_layout_and_chunk_boundaries(monkeypatch):
    import mlx.core as mx
    from mlx_backend import expression_recording as recording
    word = mx.array([1, 2, 3, 0], mx.uint32)
    assert recording.collect([word, word], [0, 1], [2, 2]) is None
    assert recording.collect([word, word], [5, 0], [2, 2]) is None
    monkeypatch.setattr(recording, 'MAX_EVENTS', 3)
    assert recording.collect([word], [0], [2]) is None
    rows = [np.array([1, 3]), np.array([], np.int32), np.array([0, 2])]
    assert recording.ordered_chunks(rows, [np.array([0, 1]), np.array([]), np.array([2, 2])])
    assert not recording.ordered_chunks(rows, [np.array([0, 1]), np.array([]), np.array([1, 2])])


@pytest.mark.parametrize('target', ['numpy', 'cython'])
@pytest.mark.parametrize('record', [False, True])
def test_original_api_normal_continuations(tmp_path, monkeypatch, target, record):
    import brian2 as b
    import original_api_reference as reference
    from mlx_backend import expression_recording as recording
    from original_api_protocol import check_original_api_runs, observe_run
    size = 128

    monkeypatch.setattr(reference, "write_tiny_dataset", reference.synthetic_dataset(size))
    monkeypatch.setattr(recording, 'MAX_EVENTS', size*7)
    params = dict(eqs='v:volt\ng:volt\nrfc:second', eq_th='randn()>.2',
                  eq_rst='v=v_rst;g=0*mV', t_rfc=.2*b.ms, t_dly=.3*b.ms,
                  r_poi=3700*b.Hz, r_poi2=7200*b.Hz)

    def build(mode):
        fixture = reference.build_original_api_fixture(tmp_path/str(mode=='native'), parameters=params, silenced=())
        if not record:
            fixture.network.remove(fixture.monitor)
            fixture.monitor = b.SpikeMonitor(fixture.neurons, record=False)
            fixture.network.add(fixture.monitor)
        return fixture

    def exercise(fixture, mode):
        for duration in (20.1, .4, .7):
            yield observe_run(fixture,duration,rng=True)
    check_original_api_runs(build,exercise,monkeypatch,seed=875,target=target)


@pytest.mark.parametrize('size', [1, 5, 31, 33, 257])
def test_partial_simd_groups(monkeypatch, size):
    import mlx.core as mx
    from mlx_backend import expression_recording as recording
    monkeypatch.setattr(recording, 'MAX_EVENTS', max(32, size*3))
    blocks = [np.arange(size, dtype=np.uint32)*np.uint32(1234567),
              np.full(size, 0x80000003, np.uint32)]
    starts, lengths = [2**32, 2**32+32], [32, 3]
    rows, steps = recording.collect([mx.array(block) for block in blocks], starts, lengths)
    expected = [(row, start+bit) for block, start, length in zip(blocks, starts, lengths)
                for bit in range(length) for row, word in enumerate(block) if int(word) & (1 << bit)]
    assert list(zip(rows.tolist(), steps.tolist())) == expected


def test_original_api_normal_fallback(tmp_path, monkeypatch):
    from mlx_backend import expression_recording as recording
    monkeypatch.setattr(recording, 'collect', lambda *args: None)
    test_original_api_normal_continuations(tmp_path, monkeypatch, 'numpy', True)
