"""Streaming exports are checked against the existing whole-table oracle."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from native_execution import require_metal
import run_mlx

from mlx_backend import export
from mlx_backend.export import SpoolSpikes
from mlx_backend.models import Connectome, SpikeEvents


def connectome(identifiers=None):
    if identifiers is None:
        identifiers = np.array([80, 3, 46, 900], dtype=np.int64)
    return Connectome(
        identifiers, np.array([], np.int32), np.array([], np.int32),
        np.array([], np.int32), np.array([], np.float64),
    )


def spikes(rows=()):
    values = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    return SpikeEvents(values[:, 0], values[:, 1], values[:, 2])


def oracle(network, blocks, exp_name="test"):
    return run_mlx.spike_table(
        network, [SimpleNamespace(spikes=block) for block in blocks], exp_name
    )


def assert_export(path, expected):
    actual = pq.read_table(path)
    assert actual.schema == expected.schema
    assert actual.equals(expected)
    metadata = pq.read_metadata(path)
    for group in range(metadata.num_row_groups):
        assert all(
            metadata.row_group(group).column(column).compression == "BROTLI"
            for column in range(metadata.num_columns)
        )


@pytest.mark.metal
@pytest.mark.parametrize("dt_ms", [.13, .029900000000000003])
def test_actual_custom_clock_export_matches_original_brian_times(tmp_path, dt_ms):
    b = pytest.importorskip("brian2")
    mx = require_metal('Metal required')
    import mlx_engine
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Stimulus
    from original_api_reference import build_original_api_fixture, cpu_runtime

    parameters = {"eq_th": "True", "t_rfc": 0*b.ms}
    with cpu_runtime(dt_ms=dt_ms):
        fixture = build_original_api_fixture(tmp_path/"cpu", parameters=parameters,
                                             exc=(), exc2=(), silenced=())
        fixture.run_ms(5*dt_ms)
        clock_dt_s = float(fixture.monitor.clock.dt_)
        times = np.asarray(fixture.monitor.t_[:]).copy()
        rows = np.asarray(fixture.monitor.i[:]).copy()
        model = resolve_model(parameters, dt_ms=dt_ms)
        native_graph = mlx_engine.load_inputs(*fixture.paths)
    stimulus = Stimulus(np.empty((1, 5, 0), np.uint8), (), (), (0,), 0, 0,
                        None, model.dt_ms)
    actual = mlx_engine.execute(native_graph, stimulus, (), model=model)
    assert actual.dt_s == model.dt_s == clock_dt_s
    output = run_mlx.spike_table(native_graph, [actual], "custom-clock")
    expected = np.lexsort((rows, times))
    assert output["t"].to_numpy().tobytes() == times[expected].tobytes()
    assert output["time_ms"].to_numpy().tobytes() == (times[expected]*1000).tobytes()
    np.testing.assert_array_equal(output["neuron_index"].to_numpy(), rows[expected])
    path = tmp_path/"bounded.parquet"
    with SpoolSpikes(native_graph, [0], "custom-clock", dt_s=model.dt_s,
                     batch_rows=3) as spool:
        streamed = mlx_engine.execute(native_graph, stimulus, (), model=model,
                                      spike_sink=spool)
        assert streamed.spikes.steps.size == 0
        assert spool.n_spikes == len(times)
        spool.finish(path)
    assert pq.read_table(path).equals(output)


@pytest.mark.parametrize("identifiers", [
    np.array([80, 3, 46, 900], dtype=np.int64),
    np.array([2**64 - 1, 2**63, 4, 2**63 + 17], dtype=np.uint64),
    np.array(["zebra", "001", "fly-α", "long-identifier"]),
    np.array(["same", "different", "same", "other"], dtype=object),
])
def test_stream_matches_canonical_order_ids_and_parquet(tmp_path, identifiers):
    network = connectome(identifiers)
    # The original inverse mapping contains only the last duplicate row.
    first = 2 if identifiers[0] == identifiers[2] else 0
    blocks = [
        spikes([(19, 2, 1), (2, 1, 0), (7, 3, 2), (2, first, 0), (19, first, 0)]),
        spikes([(2, 2, 4), (7, 3, 2), (19, 1, 6)]),
        spikes([(19, 2, 9), (2, first, 5), (7, 1, 8)]),
    ]
    destination = tmp_path / "nested" / "spikes.parquet"
    with SpoolSpikes(network, [19, 2, 7], "test", directory=tmp_path, batch_rows=2) as spool:
        for block in blocks:
            assert spool.append(block) >= 0
        assert spool.n_spikes == 11
        assert spool.n_active_neurons == np.unique(identifiers).size
        timings = spool.finish(destination)
        assert set(timings) == {"collection_s", "save_s"}
        assert all(value >= 0 for value in timings.values())
        with pytest.raises(RuntimeError, match="finished"):
            spool.append(spikes())
        with pytest.raises(RuntimeError, match="finished"):
            spool.finish(tmp_path / "second.parquet")
    assert_export(destination, oracle(network, blocks))
    assert not list(tmp_path.glob("mlx-spikes-*"))


@pytest.mark.parametrize("identifiers", [
    np.array(["a", "b", "c", "d"]), np.arange(4, dtype=np.uint64),
])
def test_empty_export_keeps_full_schema_and_types(tmp_path, identifiers):
    network = connectome(identifiers)
    destination = tmp_path / "empty.parquet"
    with SpoolSpikes(network, (9, 1), "test", directory=tmp_path) as spool:
        assert spool.append(spikes()) == 0.0
        spool.finish(destination)
        assert spool.n_spikes == spool.n_active_neurons == 0
    assert_export(destination, oracle(network, [spikes()]))


def test_custom_timestep_and_int64_limits(tmp_path):
    trial, step = np.iinfo(np.int64).max, np.iinfo(np.int64).max
    destination = tmp_path / "custom.parquet"
    with SpoolSpikes(connectome(), [trial], "custom", dt_ms=0.25) as spool:
        spool.append(spikes([(trial, 2, step)]))
        spool.finish(destination)
    row = pq.read_table(destination).to_pylist()[0]
    assert row["trial"] == trial
    assert row["t"] == float(step) * 0.00025
    assert row["time_ms"] == row["t"] * 1000


def test_many_chunks_use_bounded_batches_without_losing_trial_order(tmp_path, monkeypatch):
    blocks = [
        spikes([(trial, neuron, step) for trial in [9, 1, 5]
                for step in [begin + 1, begin] for neuron in [3, 1, 0, 2]])
        for begin in range(0, 20, 2)
    ]
    batch_sizes = []
    original = SpoolSpikes._table

    def table(self, records, trial):
        batch_sizes.append(records.size)
        return original(self, records, trial)

    monkeypatch.setattr(SpoolSpikes, "_table", table)
    destination = tmp_path / "many.parquet"
    with SpoolSpikes(connectome(), [9, 1, 5], "test", batch_rows=7) as spool:
        for block in blocks:
            spool.append(block)
        spool.finish(destination)
    assert len(batch_sizes) == 36 and max(batch_sizes) == 7
    assert sum(batch_sizes) == 240
    assert pq.read_metadata(destination).num_row_groups == len(batch_sizes)
    assert_export(destination, oracle(connectome(), blocks))


def test_validation_is_atomic_and_equal_boundary_keys_are_retained(tmp_path):
    first = spikes([(1, 2, 4), (9, 0, 7)])
    destination = tmp_path / "validated.parquet"
    with SpoolSpikes(connectome(), [1, 9], "test") as spool:
        spool.append(first)
        # Trial 1 is valid; trial 9 fails. Neither trial may be appended.
        with pytest.raises(ValueError, match="out of order"):
            spool.append(spikes([(1, 3, 8), (9, 3, 6)]))
        with pytest.raises(ValueError, match="out of order"):
            spool.append(spikes([(1, 1, 4)]))
        with pytest.raises(ValueError, match="outside trial_ids"):
            spool.append(spikes([(1, 3, 9), (3, 0, 10)]))
        assert spool.n_spikes == 2 and spool.n_active_neurons == 2
        equal = spikes([(1, 2, 4), (9, 0, 7)])
        spool.append(equal)
        spool.finish(destination)
    assert_export(destination, oracle(connectome(), [first, equal]))


@pytest.mark.parametrize("block", [
    SpikeEvents(np.array([1.0]), np.array([0]), np.array([0])),
    SpikeEvents(np.array([True]), np.array([0]), np.array([0])),
    SpikeEvents(np.array([[1]]), np.array([0]), np.array([0])),
    SpikeEvents(np.array([1]), np.array([-1]), np.array([0])),
    SpikeEvents(np.array([1]), np.array([4]), np.array([0])),
    SpikeEvents(np.array([1]), np.array([0]), np.array([-1])),
    SpikeEvents(np.array([1]), np.array([0.0]), np.array([0])),
    SpikeEvents(np.array([1]), np.array([0]), np.array([0.0])),
    SpikeEvents(np.array([1]), np.array([0]), np.array([2**63], np.uint64)),
    SpikeEvents(np.array([2**63], np.uint64), np.array([0]), np.array([0])),
    SpikeEvents(np.array([1]), np.array([0, 1]), np.array([0])),
])
def test_invalid_spikes_cannot_modify_spool(tmp_path, block):
    with SpoolSpikes(connectome(), [1], "test", directory=tmp_path) as spool:
        with pytest.raises(ValueError):
            spool.append(block)
        assert spool.n_spikes == spool.n_active_neurons == 0
        assert not list(Path(spool._scratch.name).iterdir())


@pytest.mark.parametrize("kwargs", [
    {"trial_ids": [1, 1]}, {"trial_ids": [-1]}, {"trial_ids": [2**63]},
    {"trial_ids": [True]}, {"trial_ids": [1.0]},
    {"batch_rows": 0}, {"batch_rows": True}, {"batch_rows": 1.0},
    {"dt_ms": 0}, {"dt_ms": np.inf}, {"dt_ms": True},
    {"dt_s": 0}, {"dt_s": np.inf}, {"dt_s": True},
    {"dt_ms": .1, "dt_s": .0001},
])
def test_invalid_configuration_is_rejected_before_scratch_creation(tmp_path, kwargs):
    arguments = {"trial_ids": [1], "exp_name": "test", "directory": tmp_path, **kwargs}
    with pytest.raises(ValueError):
        SpoolSpikes(connectome(), **arguments)
    assert not list(tmp_path.iterdir())


def test_collision_preserves_existing_output_and_allows_fresh_destination(tmp_path):
    existing = tmp_path / "existing.parquet"
    existing.write_bytes(b"existing evidence")
    block = spikes([(1, 0, 0)])
    with SpoolSpikes(connectome(), [1], "test") as spool:
        spool.append(block)
        with pytest.raises(FileExistsError):
            spool.finish(existing)
        assert existing.read_bytes() == b"existing evidence"
        spool.finish(tmp_path / "fresh.parquet")
    assert_export(tmp_path / "fresh.parquet", oracle(connectome(), [block]))


def test_failed_export_removes_its_incomplete_output(tmp_path, monkeypatch):
    destination = tmp_path / "incomplete.parquet"
    with SpoolSpikes(connectome(), [1], "test", directory=tmp_path) as spool:
        spool.append(spikes([(1, 0, 0)]))

        def fail(*args):
            raise RuntimeError("Arrow conversion failed")

        monkeypatch.setattr(spool, "_table", fail)
        with pytest.raises(RuntimeError, match="Arrow conversion failed"):
            spool.finish(destination)
        assert not destination.exists()
    assert not list(tmp_path.iterdir())


def test_context_exception_and_explicit_close_clean_scratch(tmp_path):
    with pytest.raises(RuntimeError, match="cancelled"):
        with SpoolSpikes(connectome(), [1], "test", directory=tmp_path) as spool:
            spool.append(spikes([(1, 0, 0)]))
            assert list(tmp_path.iterdir())
            raise RuntimeError("cancelled")
    assert not list(tmp_path.iterdir())
    spool.close()
    with pytest.raises(RuntimeError, match="closed"):
        spool.append(spikes())


def test_only_actual_io_is_returned_from_append_and_finish(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=0.0, io=0.0)

    def io(seconds):
        clock.now += seconds
        clock.io += seconds

    class TimedFile:
        def __init__(self, raw):
            self.raw = raw

        def __getattr__(self, name):
            return getattr(self.raw, name)

        def write(self, data):
            io(3)
            return self.raw.write(data)

        def readinto(self, data):
            io(7)
            assert len(data) <= 2 * export._RECORD.itemsize
            return self.raw.readinto(data)

        def read(self, size):
            io(11)
            return self.raw.read(size)

        def close(self):
            io(5)
            return self.raw.close()

    original_open, original_sort = Path.open, np.lexsort
    original_table, original_writer = SpoolSpikes._table, pq.ParquetWriter

    def timed_open(path, *args, **kwargs):
        io(2)
        return TimedFile(original_open(path, *args, **kwargs))

    def expensive_sort(*args, **kwargs):
        clock.now += 1000
        return original_sort(*args, **kwargs)

    def expensive_table(self, records, trial):
        clock.now += 31
        return original_table(self, records, trial)

    class TimedWriter:
        def __init__(self, *args, **kwargs):
            io(13)
            self.raw = original_writer(*args, **kwargs)

        def write_table(self, table):
            io(17)
            return self.raw.write_table(table)

        def close(self):
            io(19)
            return self.raw.close()

    block = spikes([(9, 3, 0), (1, 2, 2), (1, 0, 0), (1, 1, 1)])
    destination = tmp_path / "timed.parquet"
    with SpoolSpikes(connectome(), [9, 1], "test", batch_rows=2) as spool:
        monkeypatch.setattr(export, "perf_counter", lambda: clock.now)
        monkeypatch.setattr(Path, "open", timed_open)
        monkeypatch.setattr(export.np, "lexsort", expensive_sort)
        monkeypatch.setattr(SpoolSpikes, "_table", expensive_table)
        monkeypatch.setattr(export.pq, "ParquetWriter", TimedWriter)
        before = clock.io
        assert spool.append(block) == clock.io - before == 20
        before = clock.io
        timings = spool.finish(destination)
        assert timings["save_s"] == clock.io - before
        assert timings["collection_s"] == 3 * 31
    monkeypatch.undo()
    assert_export(destination, oracle(connectome(), [block]))


def test_duplicate_mapping_failure_is_deferred_and_cannot_create_output(tmp_path):
    network = connectome(np.array(['a', 'a', 'b', 'b']))
    blocks = [spikes([(0, 2, 0)]), spikes([(0, 0, 1)])]
    destination = tmp_path/'must-not-exist'/'spikes.parquet'
    with SpoolSpikes(network, [0], 'duplicates', directory=tmp_path) as spool:
        for block in blocks:
            spool.append(block)
        with pytest.raises(KeyError) as failure:
            spool.finish(destination)
        assert failure.value.args == (0,)
        assert not destination.parent.exists()
    with pytest.raises(KeyError) as failure:
        oracle(network, blocks)
    assert failure.value.args == (0,)
