"""Bounded spike export; simulation blocks never accumulate in host memory."""

from operator import index
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .models import SpikeEvents


_INT64_MAX = np.iinfo(np.int64).max
_RECORD = np.dtype([("step", "<i8"), ("neuron", "<i8")])


def _unmapped_identifier_rows(identifiers):
    """Rows omitted by original run_exp's inverse last-occurrence mapping."""
    import pandas as pd

    if pd.Index(identifiers).is_unique:
        return None
    last = {identifier: row for row, identifier in enumerate(identifiers)}
    if len(last) == len(identifiers):
        return None
    unmapped = np.ones(len(identifiers), np.bool_)
    unmapped[list(last.values())] = False
    return unmapped


def _integer_vector(values, name, maximum=_INT64_MAX):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if np.any(values < 0) or np.any(values > maximum):
        raise ValueError(f"{name} must be between 0 and {maximum}")
    return values.astype(np.int64, copy=False)


def _io_call(function, *args, **kwargs):
    started = perf_counter()
    result = function(*args, **kwargs)
    return result, perf_counter() - started


class SpoolSpikes:
    """Spool ordered simulation blocks and stream canonical Brotli Parquet.

    Blocks may be internally unordered. For each trial their sorted
    ``(step, neuron)`` keys must follow the previous block's last key; equal
    keys are retained. Memory is proportional to the neuron/trial identities,
    the current append block, and ``batch_rows``, never the complete recording.

    ``append`` returns only scratch-file open/write/close time. Its validation,
    sorting and partitioning remain in the caller's simulation/collection time.
    ``finish`` reports final Arrow preparation separately from scratch reads
    and Parquet I/O. Scratch files live until ``close`` or context exit.
    Pass ``dt_s=model.dt_s`` to preserve a resolved clock's exact SI value.
    ``dt_ms`` is an alternative for clocks specified in milliseconds.
    """

    def __init__(
        self, connectome, trial_ids, exp_name, *, dt_ms=None, dt_s=None,
        directory=None, batch_rows=65536,
    ):
        if isinstance(batch_rows, (bool, np.bool_)):
            raise ValueError("batch_rows must be a positive integer")
        try:
            batch_rows = index(batch_rows)
        except TypeError as error:
            raise ValueError("batch_rows must be a positive integer") from error
        if batch_rows < 1:
            raise ValueError("batch_rows must be a positive integer")
        if dt_ms is not None and dt_s is not None:
            raise ValueError("Specify either dt_ms or dt_s")
        if dt_s is None:
            dt_ms = 0.1 if dt_ms is None else dt_ms
            if isinstance(dt_ms, (bool, np.bool_)) or not np.isfinite(dt_ms) or dt_ms <= 0:
                raise ValueError("dt_ms must be positive and finite")
            dt_s = float(dt_ms) * 0.001
        if isinstance(dt_s, (bool, np.bool_)) or not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be positive and finite")
        if not isinstance(exp_name, str):
            raise ValueError("exp_name must be a string")
        try:
            trials = tuple(trial_ids)
            if any(isinstance(trial, (bool, np.bool_)) for trial in trials):
                raise ValueError
            trials = tuple(index(trial) for trial in trials)
            if len(set(trials)) != len(trials) or any(
                not 0 <= trial <= _INT64_MAX for trial in trials
            ):
                raise ValueError
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("trial_ids must be unique nonnegative int64 integers") from error

        identifiers = np.asarray(connectome.neuron_ids)
        if identifiers.ndim != 1:
            raise ValueError("Neuron identifiers must be one-dimensional")
        self._identifiers = pa.array(identifiers)
        # Match the existing exporter, which counts unique external identifiers.
        unique_ids, self._identifier_groups = np.unique(identifiers, return_inverse=True)
        self._unmapped_rows = (_unmapped_identifier_rows(identifiers)
                               if unique_ids.size < identifiers.size else None)
        self._unmapped = {}
        self._active = np.zeros(unique_ids.size, dtype=np.bool_)
        self._neurons = identifiers.size
        self._schema = pa.schema([
            ("t", pa.float64()),
            ("time_ms", pa.float64()),
            ("trial", pa.int64()),
            ("neuron_index", pa.int64()),
            ("flywire_id", self._identifiers.type),
            ("exp_name", pa.string()),
        ])
        self._trials = tuple(sorted(trials))
        self._last = {trial: None for trial in trials}
        self._counts = {trial: 0 for trial in trials}
        self._dt_seconds = float(dt_s)
        self._exp_name = exp_name
        self._batch_rows = batch_rows
        self._n_spikes = self._n_active_neurons = 0
        self._closed = self._finished = False
        self._scratch = TemporaryDirectory(prefix="mlx-spikes-", dir=directory)
        self._paths = {
            trial: Path(self._scratch.name) / f"{position}.bin"
            for position, trial in enumerate(self._trials)
        }

    @property
    def n_spikes(self):
        return self._n_spikes

    @property
    def n_active_neurons(self):
        return self._n_active_neurons

    def _check_open(self):
        if self._closed or self._finished:
            raise RuntimeError("Spike spool is closed or already finished")

    def append(self, spikes: SpikeEvents) -> float:
        self._check_open()
        trials = _integer_vector(spikes.trials, "Spike trials")
        neurons = _integer_vector(spikes.neurons, "Spike neurons", self._neurons - 1)
        steps = _integer_vector(spikes.steps, "Spike steps")
        if not trials.size == neurons.size == steps.size:
            raise ValueError("Spike arrays must have equal lengths")
        if not trials.size:
            return 0.0
        if any(int(trial) not in self._last for trial in np.unique(trials)):
            raise ValueError("Spike trial is outside trial_ids")

        order = np.lexsort((neurons, steps, trials))
        trials, neurons, steps = trials[order], neurons[order], steps[order]
        boundaries = np.r_[0, np.flatnonzero(trials[1:] != trials[:-1]) + 1, trials.size]
        partitions = []
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            trial = int(trials[begin])
            first = (int(steps[begin]), int(neurons[begin]))
            last = (int(steps[end - 1]), int(neurons[end - 1]))
            if self._last[trial] is not None and first < self._last[trial]:
                raise ValueError(f"Spike block is out of order for trial {trial}")
            partitions.append((trial, begin, end, last))
        active = np.unique(self._identifier_groups[neurons])
        new_active = int(np.count_nonzero(~self._active[active]))

        io_s = 0.0
        try:
            for trial, begin, end, _ in partitions:
                records = np.empty(end - begin, dtype=_RECORD)
                records["step"], records["neuron"] = steps[begin:end], neurons[begin:end]
                raw = memoryview(records).cast("B")
                destination, elapsed = _io_call(self._paths[trial].open, "ab", buffering=0)
                io_s += elapsed
                try:
                    while raw:
                        written, elapsed = _io_call(destination.write, raw)
                        io_s += elapsed
                        if not written:
                            raise OSError("Scratch spike write made no progress")
                        raw = raw[written:]
                finally:
                    _, elapsed = _io_call(destination.close)
                    io_s += elapsed
        except BaseException:
            self.close()
            raise
        for trial, begin, end, last in partitions:
            self._last[trial] = last
            self._counts[trial] += int(end - begin)
            if self._unmapped_rows is not None:
                rows = neurons[begin:end]
                missing = rows[self._unmapped_rows[rows]]
                if missing.size:
                    previous = self._unmapped.get(trial, int(missing.min()))
                    self._unmapped[trial] = min(previous, int(missing.min()))
        self._active[active] = True
        self._n_active_neurons += new_active
        self._n_spikes += int(trials.size)
        return io_s

    def _table(self, records, trial):
        seconds = records["step"].astype(np.float64) * self._dt_seconds
        neurons = pa.array(records["neuron"], type=pa.int64())
        return pa.table({
            "t": seconds,
            "time_ms": seconds * 1000,
            "trial": np.full(records.size, trial, dtype=np.int64),
            "neuron_index": neurons,
            "flywire_id": self._identifiers.take(neurons),
            "exp_name": [self._exp_name] * records.size,
        }, schema=self._schema)

    def finish(self, destination) -> dict[str, float]:
        self._check_open()
        if self._unmapped:
            # Original export loops trials, then increasing neuron rows, and
            # fails before opening its destination. Keep scratch collection
            # bounded but defer this failure until all spikes are known.
            raise KeyError(self._unmapped[min(self._unmapped)])
        path = Path(destination)
        collection_s = save_s = 0.0
        destination_file = writer = None
        owned = False
        try:
            _, elapsed = _io_call(path.parent.mkdir, parents=True, exist_ok=True)
            save_s += elapsed
            destination_file, elapsed = _io_call(path.open, "xb")
            owned = True
            save_s += elapsed
            writer, elapsed = _io_call(
                pq.ParquetWriter, destination_file, self._schema, compression="brotli"
            )
            save_s += elapsed
            preparing = perf_counter()
            buffer = np.empty(min(self._batch_rows, self._n_spikes), dtype=_RECORD)
            collection_s += perf_counter() - preparing
            for trial in self._trials:
                remaining = self._counts[trial]
                if not remaining:
                    continue
                source, elapsed = _io_call(self._paths[trial].open, "rb", buffering=0)
                save_s += elapsed
                try:
                    while remaining:
                        preparing = perf_counter()
                        size = min(remaining, self._batch_rows)
                        records = buffer[:size]
                        raw = memoryview(records).cast("B")
                        collection_s += perf_counter() - preparing
                        while raw:
                            count, elapsed = _io_call(source.readinto, raw)
                            save_s += elapsed
                            if not count:
                                raise OSError("Truncated scratch spike records")
                            raw = raw[count:]
                        preparing = perf_counter()
                        table = self._table(records, trial)
                        collection_s += perf_counter() - preparing
                        _, elapsed = _io_call(writer.write_table, table)
                        save_s += elapsed
                        remaining -= size
                    extra, elapsed = _io_call(source.read, 1)
                    save_s += elapsed
                    if extra:
                        raise OSError("Unexpected extra scratch spike records")
                finally:
                    _, elapsed = _io_call(source.close)
                    save_s += elapsed
            _, elapsed = _io_call(writer.close)
            writer = None
            save_s += elapsed
            _, elapsed = _io_call(destination_file.close)
            destination_file = None
            save_s += elapsed
        except BaseException:
            try:
                if writer is not None:
                    writer.close()
            finally:
                try:
                    if destination_file is not None:
                        destination_file.close()
                finally:
                    if owned:
                        path.unlink(missing_ok=True)
            raise
        self._finished = True
        return {"collection_s": collection_s, "save_s": save_s}

    def close(self):
        if not self._closed:
            self._closed = True
            self._scratch.cleanup()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *exc):
        self.close()
