from dataclasses import dataclass
from collections import OrderedDict
from threading import Lock
from weakref import ref

import numpy as np
from numpy.typing import NDArray

from .models import Connectome, NeuronIdentifier, NeuronRow


_EVENT_HASHES = OrderedDict()
_EVENT_HASH_LOCK = Lock()
_EVENT_HASH_BYTES = 0
_MAX_EVENT_HASH_BYTES = 64 * 1024 * 1024


def event_input_snapshot(events):
    """Pin bounded static arrays; reuse hashes only after exact byte equality.

    A readonly ndarray can have a mutable alias. Owned arrays therefore take
    a fresh bytes snapshot on every call. Mapped/foreign storage, streams and
    larger arrays retain the existing incremental consumption/fingerprinting.
    """
    if (type(events) is not np.ndarray or events.dtype != np.uint8
            or events.ndim != 3 or not events.flags.c_contiguous):
        return events, None, None
    owner = events
    while type(owner) is np.ndarray:
        owner = owner.base
    cost = events.nbytes + 128*events.shape[0] + 256
    if cost > _MAX_EVENT_HASH_BYTES:
        return events, None, None
    if type(owner) is bytes and len(owner) == events.nbytes:
        key, holder, snapshot = ('bytes', id(owner), events.shape), owner, owner
    elif owner is None:
        key, holder, snapshot = ('array', id(events), events.shape), ref(events), events.tobytes()
    else:
        return events, None, None
    hashes = None
    with _EVENT_HASH_LOCK:
        saved = _EVENT_HASHES.get(key)
        same_owner = saved is not None and (
            saved[0] is owner if key[0] == 'bytes' else saved[0]() is events)
        if same_owner and saved[1] == snapshot:
            snapshot, hashes = saved[1], saved[2]
            _EVENT_HASHES.move_to_end(key)
    frozen = np.frombuffer(snapshot, np.uint8).reshape(events.shape)
    return frozen, hashes, (key, holder, snapshot, cost)


def remember_event_hashes(token, hashes):
    global _EVENT_HASH_BYTES
    if token is None:
        return
    key, holder, snapshot, cost = token
    hashes = tuple(hashes)
    if len(hashes) != key[2][0]:
        raise ValueError('Event fingerprints must match trial rows')
    with _EVENT_HASH_LOCK:
        previous = _EVENT_HASHES.pop(key, None)
        if previous is not None:
            _EVENT_HASH_BYTES -= previous[3]
        # Immutable owners stay alive; mutable arrays use a checked weakref.
        # Bound payload memory and entry count, including zero-horizon arrays.
        _EVENT_HASHES[key] = (holder, snapshot, hashes, cost)
        _EVENT_HASH_BYTES += cost
        while len(_EVENT_HASHES) > 4 or _EVENT_HASH_BYTES > _MAX_EVENT_HASH_BYTES:
            _, removed = _EVENT_HASHES.popitem(last=False)
            _EVENT_HASH_BYTES -= removed[3]


def neuron_indices(
    connectome: Connectome, ids: tuple[NeuronIdentifier, ...]
) -> tuple[NeuronRow, ...]:
    lookup = {
        identifier: NeuronRow(index)
        for index, identifier in enumerate(connectome.neuron_ids)
    }
    missing = set(ids) - lookup.keys()
    if missing:
        raise ValueError(
            f'Experiment neurons absent from data: {sorted(missing, key=str)}'
        )
    return tuple(lookup[identifier] for identifier in ids)


def trial_events(
    rates_hz: tuple[float, ...], uniforms: NDArray[np.float64], *, dt_ms: float = 0.1
) -> NDArray[np.uint8]:
    if (uniforms.dtype, uniforms.ndim) != (np.float64, 2):
        raise ValueError('Stimulus uniforms require a native float64 matrix')
    probabilities = np.asarray(rates_hz, dtype=np.float64) * (dt_ms / 1000)
    return (uniforms < probabilities).astype(np.uint8)


def uniforms(seed: tuple[int, ...], size: int | tuple[int, int]) -> NDArray[np.float64]:
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(seed))))
    return generator.random(size)


@dataclass(frozen=True)
class SeededEvents:
    """Random-access PCG64 events with exactly the existing trial-major stream.

    Each float64 uniform consumes one PCG64 output. Advancing by begin*channels
    reproduces a full-array slice without generating or retaining its prefix.
    """

    steps: int
    rates_hz: tuple[float, ...]
    trial_indices: tuple[int, ...]
    seed: int
    generator_code: int
    dt_ms: float = 0.1

    def __post_init__(self):
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("Event horizon must be a nonnegative integer")
        rates = tuple(float(rate) for rate in self.rates_hz)
        ids = tuple(self.trial_indices)
        if (
            not np.isfinite(self.dt_ms)
            or self.dt_ms <= 0
            or any(not np.isfinite(rate) or rate < 0 for rate in rates)
            or not ids
            or any(
                isinstance(trial, (bool, np.bool_))
                or not isinstance(trial, (int, np.integer))
                or not 0 <= trial <= np.iinfo(np.int64).max
                for trial in ids
            )
            or len(set(ids)) != len(ids)
            or type(self.seed) is not int
            or self.seed < 0
            or type(self.generator_code) is not int
            or self.generator_code < 0
        ):
            raise ValueError("Invalid event rates, timestep, seed or trial identities")
        object.__setattr__(self, "rates_hz", rates)
        object.__setattr__(self, "trial_indices", tuple(int(trial) for trial in ids))

    @property
    def shape(self):
        return len(self.trial_indices), self.steps, len(self.rates_hz)

    @property
    def size(self):
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def dtype(self):
        return np.dtype(np.uint8)

    @property
    def ndim(self):
        return 3

    def chunk(self, begin, end):
        if (
            type(begin) is not int or type(end) is not int
            or not 0 <= begin <= end <= self.steps
        ):
            raise ValueError("Event chunk must lie within its horizon")
        channels = len(self.rates_hz)
        events = np.empty((len(self.trial_indices), end - begin, channels), np.uint8)
        if not channels:
            return events
        for row, trial in enumerate(self.trial_indices):
            generator = np.random.Generator(np.random.PCG64(
                np.random.SeedSequence([self.seed, self.generator_code, trial])
            ))
            generator.bit_generator.advance(begin * channels)
            events[row] = trial_events(
                self.rates_hz, generator.random((end - begin, channels)), dt_ms=self.dt_ms
            )
        return events


@dataclass(frozen=True)
class CombinedEvents:
    """Join trial arrays/sources by staging only their requested time slices."""

    sources: tuple

    def __post_init__(self):
        sources = tuple(self.sources)
        if not sources or any(
            source.ndim != 3
            or source.dtype != np.dtype(np.uint8)
            or source.shape[0] < 1
            or source.shape[1:] != sources[0].shape[1:]
            for source in sources
        ):
            raise ValueError("Combined events require compatible uint8 trial sources")
        object.__setattr__(self, "sources", sources)

    @property
    def shape(self):
        return sum(source.shape[0] for source in self.sources), *self.sources[0].shape[1:]

    @property
    def size(self):
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def dtype(self):
        return np.dtype(np.uint8)

    @property
    def ndim(self):
        return 3

    def chunk(self, begin, end):
        if (
            type(begin) is not int or type(end) is not int
            or not 0 <= begin <= end <= self.shape[1]
        ):
            raise ValueError("Event chunk must lie within its horizon")
        return np.concatenate([
            source.chunk(begin, end) if hasattr(source, "chunk") else source[:, begin:end]
            for source in self.sources
        ], axis=0)
