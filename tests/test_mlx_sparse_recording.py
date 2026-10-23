"""Exact recorder export with an independent CPU bit/order oracle."""
from types import SimpleNamespace

import numpy as np
import pytest


def joined(parts):
    return tuple(np.concatenate([part[k] for part in parts]) if parts else
                 np.array([], np.int64) for k in range(3))


def oracle(flags, starts, ids, bits=32):
    rows = []
    _, windows, neurons = flags.shape
    for flat in np.flatnonzero(flags.reshape(-1)):
        trial, offset = divmod(int(flat), windows*neurons)
        window, neuron = divmod(offset, neurons)
        for bit in range(bits):
            if int(flags[trial, window, neuron]) & (1 << bit):
                rows.append((int(ids[trial]), neuron, int(starts[window])+bit))
    rows.sort(key=lambda row: (row[0], row[2], row[1]))
    return tuple(np.array([row[k] for row in rows], np.int64) for k in range(3))


def assert_arrays(actual, expected):
    for got, wanted in zip(actual, expected):
        assert got.dtype == wanted.dtype == np.dtype(np.int64)
        np.testing.assert_array_equal(got, wanted)


@pytest.mark.metal
@pytest.mark.parametrize("bits", [1, 5, 32])
def test_sparse_words_padding_external_ids_and_overflow(bits):
    import mlx.core as mx
    from mlx_backend import configured as c
    ids = np.array([2**40+7, 2**40+2], np.int64)
    flags = np.array([[[0xffffffff, 0x80000001, 0], [0x10001, 0, 0x10]],
                      [[0, 0x20, 0x11], [0x80, 3, 0xffffffff]]], np.uint32)
    starts = [2**63-100, 2**63-50]
    for raw in (flags, np.zeros_like(flags), np.full_like(flags, 0xffffffff)):
        block = [mx.array(raw[:, window, :].reshape(-1)) for window in range(2)]
        expected = oracle(raw, starts, ids, bits)
        assert_arrays(joined(c._collect_sparse_unit_windows(block, starts, 3, ids, bits)), expected)
        overflow = c._collect_sparse_unit_windows(block, starts, 3, ids, bits, capacity=1)
        if len(expected[0]):
            assert overflow is None
            assert_arrays(joined(c._collect_unit_windows(block, starts, 3, ids, bits)), expected)
        else:
            assert overflow == []


@pytest.mark.metal
def test_sparse_full_partial_blocks_and_streaming_export(tmp_path):
    import mlx.core as mx
    import pyarrow.parquet as pq
    import run_mlx
    from mlx_backend import configured as c
    from mlx_backend.export import SpoolSpikes
    from mlx_backend.models import Connectome, SpikeEvents
    neurons, windows = 257, 172
    ids = np.array([2**40+7, 2**40+2, 2**40+11, 2**40+1], np.int64)
    flags = np.zeros((4, windows, neurons), np.uint32)
    starts = list(range(0, windows*18, 18))
    for trial in range(4):
        for window in range(windows):
            flags[trial, window, (trial+window) % neurons] = np.uint32(
                0x20001 if window < windows-1 else 0x201)
    expected = oracle(flags, starts, ids)
    parts, batches = [], []
    for begin, end in ((0, 64), (64, 128), (128, 172)):
        block = [mx.array(flags[:, window, :].reshape(-1)) for window in range(begin, end)]
        batch = c._collect_unit_windows(block, starts[begin:end], neurons, ids, 32, sparse=True)
        parts.extend(batch)
        batches.append(SpikeEvents(*joined(batch)))
    parts.sort(key=lambda part: int(part[0][0]))
    assert_arrays(joined(parts), expected)
    empty = np.array([], np.int32)
    graph = Connectome(np.arange(neurons), empty, empty, empty, np.array([], np.float64))
    path = tmp_path/"spikes.parquet"
    with SpoolSpikes(graph, ids, "sparse-recorder", directory=tmp_path, batch_rows=31) as spool:
        for batch in batches:
            spool.append(batch)
        spool.finish(path)
    wanted = run_mlx.spike_table(graph, [SimpleNamespace(spikes=SpikeEvents(*expected))], "sparse-recorder")
    assert pq.read_table(path).equals(wanted)


def test_recording_retains_large_index_protection():
    from mlx_backend import configured as c

    class LargeTrials:
        def __len__(self):
            return 2**27

    with pytest.raises(OverflowError, match="uint32 event indexing"):
        c._collect_unit_windows([None], [0], 1, LargeTrials(), 32, sparse=True)
    for size in (2**31, 2**32-1, 2**32):
        assert c._collect_sparse_unit_windows([None], [0], size, [0], 32) is None


@pytest.mark.metal
def test_actual_custom_graph_state_spikes_and_spool_match_cpu(monkeypatch, tmp_path):
    import brian2 as b
    import pyarrow.parquet as pq
    import run_mlx
    from configured_reference import reference
    from mlx_backend import configured as c
    from mlx_backend.export import SpoolSpikes
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome, Stimulus
    monkeypatch.setitem(b.prefs, "codegen.target", "numpy")
    model = resolve_model({"eqs": "v:volt\ng:volt\nrfc:second\ncount:integer\n",
                           "eq_th": "(i % 17 == 0) and (timestep(t, dt) % 3 == 0)",
                           "eq_rst": "v=v_rst;g=0*mV;count+=1",
                           "t_rfc": 0*b.ms, "t_dly": 0*b.ms, "w_syn": 0*b.mV})
    graph = Connectome(np.arange(257), np.array([0], np.int32), np.array([1], np.int32),
                       np.array([1], np.int32), np.array([0.], np.float64))
    ids = (2**40+7, 2**40+2, 2**40+11, 2**40+1)
    stimulus = Stimulus(np.zeros((4, 131, 0), np.uint8), (), (), ids, 0, 0, None, model.dt_ms)
    wanted = reference(graph, stimulus, model)
    # The independent helper runs trials in input order; native export uses
    # the public trial/step/neuron order, including nonmonotonic external IDs.
    cpu_order = np.lexsort((wanted.spikes.neurons, wanted.spikes.steps, wanted.spikes.trials))
    expected_spikes = tuple(array[cpu_order] for array in
                            (wanted.spikes.trials, wanted.spikes.neurons, wanted.spikes.steps))
    prepared = c.prepare(graph, stimulus, model)
    assert prepared.window == 1 and not prepared.independent_neurons
    original, calls = c._collect_sparse_unit_windows, []

    def collect(block, *args, **kwargs):
        calls.append(len(block))
        return original(block, *args, **kwargs)

    monkeypatch.setattr(c, "_collect_sparse_unit_windows", collect)
    for streaming in (False, True):
        path = tmp_path/"actual.parquet"
        if streaming:
            with SpoolSpikes(graph, ids, "custom-recorder", directory=tmp_path, batch_rows=7) as sink:
                result, state = c.run_prepared(prepared, stimulus, spike_sink=sink,
                                               return_state=True, chunk_steps=67)
                sink.finish(path)
            expected = run_mlx.spike_table(graph, [SimpleNamespace(spikes=wanted.spikes)], "custom-recorder")
            assert pq.read_table(path).equals(expected)
            assert result.spikes.steps.size == 0
        else:
            result, state = c.run_prepared(prepared, stimulus, return_state=True, chunk_steps=67)
            assert_arrays((result.spikes.trials, result.spikes.neurons, result.spikes.steps),
                          expected_spikes)
        actual = c.host_state(state, prepared)
        assert actual.keys() == wanted.state.keys()
        for name, value in actual.items():
            assert value.dtype == wanted.state[name].dtype
            assert value.tobytes() == wanted.state[name].tobytes(), name
    assert calls == [64, 64, 64, 64]
