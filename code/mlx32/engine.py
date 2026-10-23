"""Fixed-model float32 Metal runtime with bounded input and event buffers.

Adapted from kisame ``lif/core.py``, ``engine_fused.py``, ``engine_metal.py``
and ``spike_record.py``, revision e417b33616513ef350b1b1c3cdf2b5b7a1799c8e
(MIT; the complete donor notice accompanies this package). The integer atomic
propagation, fused tick order and bounded event extraction retain that lineage.
"""

from __future__ import annotations

import operator
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .model import Drive, Graph, Model

_I32_MAX = 2**31 - 1
_EVENT_CAP = 262_144


@dataclass(frozen=True)
class RunResult:
    v: np.ndarray
    g: np.ndarray
    refractory: np.ndarray
    delay_ring: np.ndarray
    counts: np.ndarray
    step: int
    spikes: np.ndarray
    timings: dict[str, float]


_PROPAGATE_SOURCE = """
    uint i = thread_position_in_grid.x;
    uint lane = thread_position_in_grid.y;
    uint base = thread_position_in_grid.z * n_neurons[0];
    if (i >= n_neurons[0] || !spike[base + i]) return;
    uint hi = uint(row_end[i]);
    for (uint e = uint(row_ptr[i]) + lane; e < hi; e += 8) {
        atomic_fetch_add_explicit(&contrib[base + uint(dst[e])], contacts[e], memory_order_relaxed);
    }
"""

_STATE_SOURCE = """
    uint i = thread_position_in_grid.x;
    if (i >= n_neurons[0]) return;
    uint trial = thread_position_in_grid.y;
    uint j = trial * n_neurons[0] + i;
    uint stim_base = trial * uint(stim_row_shape[1]);
    int r = max(rfc[j] - 1, 0);
    bool not_ref = r == 0;
    float vi = v[j];
    float gi = g[j];
    if (not_ref) {
        vi = coefficients[0] + (gi * coefficients[1] + vi * coefficients[2]);
        gi = gi * coefficients[3];
    }
    bool sp = not_ref && vi > parameters[0];
    if (not_ref) {
        gi = gi + float(contrib[j]) * parameters[1];
        // A stable channel CSR preserves repeated additions and channel order.
        for (int c = channel_ptr[i]; c < channel_ptr[i + 1]; ++c) {
            if (stim_row[stim_base + uint(channel_order[c])]) vi = vi + parameters[2];
        }
    }
    if (sp) {
        vi = parameters[3];
        gi = 0.0f;
        r = reload[i];
    }
    v_out[j] = vi;
    g_out[j] = gi;
    rfc_out[j] = r;
    counts_out[j] = counts[j] + (sp ? 1 : 0);
    spike_out[j] = sp ? 1 : 0;
"""

_RECORD_SOURCE = """
    uint i = thread_position_in_grid.x;
    uint k = thread_position_in_grid.y;
    uint trial = thread_position_in_grid.z;
    uint trials = uint(spikes_shape[1]);
    uint n = uint(spikes_shape[2]);
    if (i >= n || k >= uint(spikes_shape[0]) || trial >= trials) return;
    if (!spikes[(k * trials + trial) * n + i]) return;
    uint slot = atomic_fetch_add_explicit(&count[0], 1u, memory_order_relaxed);
    if (slot >= cap[0]) return;
    atomic_store_explicit(&events[3 * slot], int(k), memory_order_relaxed);
    atomic_store_explicit(&events[3 * slot + 1], int(trial), memory_order_relaxed);
    atomic_store_explicit(&events[3 * slot + 2], int(i), memory_order_relaxed);
"""


def _integer(value: object, name: str, maximum: int = _I32_MAX) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 0 <= result <= maximum:
        raise ValueError(f"{name} must be from 0 to {maximum}")
    return result


def _row_end(graph: Graph, silenced: object) -> np.ndarray:
    values = np.asarray(silenced)
    ends = graph.row_ptr[1:].copy()
    if values.dtype == np.bool_:
        if values.shape != (graph.n_neurons,):
            raise ValueError("silenced bool mask must have shape (n_neurons,)")
        ends[values] = graph.row_ptr[:-1][values]
    else:
        if values.ndim != 1 or (values.size and values.dtype.kind not in "iu"):
            raise ValueError("silenced must be a bool mask or integral neuron indices")
        if values.size:
            if np.any(values < 0) or np.any(values >= graph.n_neurons):
                raise ValueError("silenced contains an out-of-range neuron index")
            indices = values.astype(np.int32)
            ends[indices] = graph.row_ptr[indices]
    return ends


def _metal():
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("mlx32 requires MLX with an available Metal GPU") from exc
    if not mx.metal.is_available():
        raise RuntimeError("mlx32 requires an available Metal GPU; CPU execution is unsupported")
    return mx


def _kernels(mx):
    propagate = mx.fast.metal_kernel(
        name="eon32_propagate",
        input_names=["spike", "row_ptr", "row_end", "dst", "contacts", "n_neurons"],
        output_names=["contrib"],
        source=_PROPAGATE_SOURCE,
        atomic_outputs=True,
    )
    state = mx.fast.metal_kernel(
        name="eon32_state",
        input_names=["v", "g", "rfc", "counts", "contrib", "stim_row",
                     "channel_ptr", "channel_order", "reload", "n_neurons",
                     "coefficients", "parameters"],
        output_names=["v_out", "g_out", "rfc_out", "counts_out", "spike_out"],
        source=_STATE_SOURCE,
        header="#pragma clang fp contract(off)\n",
        atomic_outputs=False,
    )
    record = mx.fast.metal_kernel(
        name="eon32_spike_events",
        input_names=["spikes", "cap"],
        output_names=["events", "count"],
        source=_RECORD_SOURCE,
        atomic_outputs=True,
    )
    return propagate, state, record


class RecordOverflow(RuntimeError):
    """A GPU batch or the explicitly capped host collection overflowed."""


class _Recorder:
    def __init__(self, mx, kernel, stream, n: int, trials: int, limit: int, sink):
        self.mx, self.kernel, self.stream, self.n = mx, kernel, stream, n
        self.trials = trials
        self.limit, self.sink = limit, sink
        self.cap = _EVENT_CAP if sink is not None else min(_EVENT_CAP, max(1, limit))
        self.cap_array = mx.array([self.cap], dtype=mx.uint32)
        self.pending = deque()
        self.collected: list[np.ndarray] = []
        self.total = 0

    def submit(self, spikes: list, tick: int) -> list:
        events, count = self.kernel(
            inputs=[self.mx.stack(spikes), self.cap_array],
            output_shapes=[(self.cap, 3), (1,)],
            output_dtypes=[self.mx.int32, self.mx.uint32],
            grid=(self.n, len(spikes), self.trials), threadgroup=(256, 1, 1),
            init_value=0, stream=self.stream,
        )
        self.pending.append((tick, len(spikes), events, count))
        return [events, count]

    def drain(self, keep: int = 0) -> None:
        while len(self.pending) > keep:
            tick, length, events, count = self.pending[0]
            size = int(count.item())
            if size > self.cap:
                raise RecordOverflow(
                    f"ticks {tick}:{tick + length} produced {size} events; "
                    f"GPU batch capacity is {self.cap}; reduce chunk_size or trials"
                )
            if self.sink is None and self.total + size > self.limit:
                raise RecordOverflow(f"recorded events exceed max_recorded_events={self.limit}")
            if size:
                batch = np.array(np.asarray(events[:size]), dtype=np.int64, copy=True)
                batch[:, 0] += tick
                batch = batch[np.lexsort((batch[:, 2], batch[:, 1], batch[:, 0]))]
                if self.sink is None:
                    self.collected.append(batch)
                else:
                    self.sink(batch)
            self.total += size
            self.pending.popleft()

    def finish(self) -> np.ndarray:
        self.drain()
        if self.sink is not None or not self.collected:
            return np.empty((0, 3), dtype=np.int64)
        return np.concatenate(self.collected)


def run_batch(
    graph: Graph, steps: int, drive: Drive, *,
    trials: tuple[int, ...] | list[int] = (0,), model: Model = Model(),
    chunk_size: int = 32, record: bool = True,
    round_idx: int = 0, silenced=(), input_tape=None,
    spike_sink: Callable[[np.ndarray], object] | None = None,
    max_recorded_events: int = 1_000_000,
) -> RunResult:
    """Run independent trials together; state snapshots have shape ``(B, N)``.

    ``trials`` contains distinct seed identifiers, in batch-position order.
    Input tapes have shape ``(steps, B, channels)``. Events are int64 rows
    ``(tick, batch_position, neuron)``, sorted in that order. The delay ring
    has shape ``(delay_steps, B, N)`` after ``steps`` completed ticks.

    Timings are seconds. ``runtime_s`` includes input generation/staging and
    event collection. ``simulation_s`` is synchronized kernel scheduling and
    execution, including GPU event extraction; input, collection and snapshots
    are measured separately. ``setup_compile_s`` includes disposable warmup.
    No timing here asserts a CPU speedup or universal Brian float-bit equality.
    """
    setup_start = time.perf_counter()
    steps = _integer(steps, "steps")
    chunk_size = _integer(chunk_size, "chunk_size")
    if chunk_size == 0:
        raise ValueError("chunk_size must be at least 1")
    if not isinstance(trials, (tuple, list)) or not trials:
        raise ValueError("trials must be a nonempty tuple or list of trial identifiers")
    trial_ids = tuple(_integer(trial, "trial", 2**64 - 1) for trial in trials)
    batch = _integer(len(trial_ids), "trial count")
    if len(set(trial_ids)) != batch:
        raise ValueError("trials must contain distinct trial identifiers")
    round_idx = _integer(round_idx, "round_idx", 2**64 - 1)
    limit = _integer(max_recorded_events, "max_recorded_events")
    if not isinstance(record, (bool, np.bool_)):
        raise ValueError("record must be bool")
    if spike_sink is not None and not callable(spike_sink):
        raise ValueError("spike_sink must be callable")
    n, channels = graph.n_neurons, drive.targets.size
    if channels > _I32_MAX:
        raise ValueError("input channel occurrences exceed int32 indexing")
    targets = drive.targets
    if np.any(targets < 0) or np.any(targets >= n):
        raise ValueError("drive target is outside the graph")
    probabilities = drive.rates_hz * model.dt_s
    if np.any(probabilities > 1):
        raise ValueError("each input rate * dt_s must be at most 1")
    batch_steps = min(chunk_size, steps)
    if max(batch * channels, model.delay_steps * batch * n, n + 1) > _I32_MAX:
        raise ValueError("runtime state, delay or channel arrays exceed int32 indexing")
    if batch_steps * batch * max(n, channels) > _I32_MAX:
        raise ValueError("chunk arrays or event counters exceed int32 indexing; reduce chunk_size or trials")
    tape = None
    if input_tape is not None:
        tape = np.asarray(input_tape)
        if tape.dtype not in (np.dtype(np.uint8), np.dtype(np.bool_)):
            raise ValueError("input_tape must have uint8 or bool dtype")
        if tape.shape != (steps, batch, channels):
            raise ValueError("input_tape must have shape (steps, trials, channel_occurrences)")
        if tape.dtype == np.uint8:
            for tick in range(0, steps, chunk_size):
                if np.any(tape[tick:tick + chunk_size] > 1):
                    raise ValueError("input_tape may contain only 0 and 1")
    row_end = _row_end(graph, silenced)
    channel_order = np.argsort(targets, kind="stable").astype(np.int32)
    channel_ptr = np.empty(n + 1, dtype=np.int32)
    channel_ptr[0] = 0
    channel_ptr[1:] = np.cumsum(np.bincount(targets, minlength=n), dtype=np.int64)
    reload = np.full(n, model.refractory_steps, dtype=np.int32)
    reload[targets] = 0
    rngs = [
        np.random.Generator(np.random.PCG64(np.random.SeedSequence([drive.seed, round_idx, trial])))
        for trial in trial_ids
    ]
    mx = _metal()
    stream = mx.new_stream(mx.gpu)
    timings = dict(input_generation_staging_s=0.0, simulation_s=0.0, collection_s=0.0)

    with mx.stream(stream):
        propagate, state_kernel, record_kernel = _kernels(mx)
        ptr, ends, dst, contacts = map(mx.array, (graph.row_ptr, row_end, graph.destinations, graph.contacts))
        cp, co, rl = map(mx.array, (channel_ptr, channel_order, reload))
        count_n = mx.array([n], dtype=mx.uint32)
        coefficients = mx.array(model.coefficients(), dtype=mx.float32)
        parameters = mx.array([model.threshold_v, model.weight_v, model.input_v, model.v_reset_v], dtype=mx.float32)
        state_shape = (batch, n)

        def fresh():
            state = [mx.full(state_shape, model.v_rest_v, dtype=mx.float32),
                     mx.zeros(state_shape, dtype=mx.float32),
                     mx.zeros(state_shape, dtype=mx.int32),
                     mx.zeros(state_shape, dtype=mx.int32)]
            ring = [mx.zeros(state_shape, dtype=mx.uint8) for _ in range(model.delay_steps)]
            return state, ring

        def encode(state, ring, tick, frames, save_spikes):
            spikes = []
            for offset in range(frames.shape[0]):
                slot = (tick + offset) % model.delay_steps
                contrib = propagate(
                    inputs=[ring[slot], ptr, ends, dst, contacts, count_n],
                    output_shapes=[state_shape], output_dtypes=[mx.int32],
                    grid=(n, 8, batch), threadgroup=(256, 1, 1),
                    init_value=0, stream=stream,
                )[0]
                *state, spike = state_kernel(
                    inputs=[*state, contrib, frames[offset], cp, co, rl, count_n,
                            coefficients, parameters],
                    output_shapes=[state_shape] * 5,
                    output_dtypes=[mx.float32, mx.float32, mx.int32, mx.int32, mx.uint8],
                    grid=(n, batch, 1), threadgroup=(256, 1, 1), stream=stream,
                )
                ring[slot] = spike
                if save_spikes:
                    spikes.append(spike)
            return state, ring, spikes

        warmup_start = time.perf_counter()
        if n and steps:
            warm_lengths = {batch_steps, steps % chunk_size} - {0}
            for length in sorted(warm_lengths):
                disposable, disposable_ring = fresh()
                frames = mx.zeros((length, batch, channels), dtype=mx.uint8)
                disposable, disposable_ring, spikes = encode(disposable, disposable_ring, 0, frames, record)
                out = _Recorder(mx, record_kernel, stream, n, batch, limit, spike_sink).submit(spikes, 0) if record else []
                mx.eval(*disposable, *disposable_ring, *out)
                mx.synchronize(stream)
            del disposable, disposable_ring, spikes, out, frames
        timings["compile_warmup_s"] = time.perf_counter() - warmup_start
        state, ring = fresh()
        recorder = _Recorder(mx, record_kernel, stream, n, batch, limit, spike_sink) if record and n else None
        mx.eval(*state, *ring, ptr, ends, dst, contacts, cp, co, rl, count_n, coefficients, parameters)
        mx.synchronize(stream)
        timings["setup_compile_s"] = time.perf_counter() - setup_start

        runtime_start = time.perf_counter()
        if n:
            for tick in range(0, steps, chunk_size):
                length = min(chunk_size, steps - tick)
                started = time.perf_counter()
                if tape is None:
                    host_frames = np.empty((length, batch, channels), dtype=np.uint8)
                    for position, rng in enumerate(rngs):
                        host_frames[:, position, :] = rng.random((length, channels)) < probabilities
                else:
                    host_frames = tape[tick:tick + length]
                frames = mx.array(np.ascontiguousarray(host_frames, dtype=np.uint8))
                mx.eval(frames)
                mx.synchronize(stream)
                timings["input_generation_staging_s"] += time.perf_counter() - started
                started = time.perf_counter()
                state, ring, spikes = encode(state, ring, tick, frames, recorder is not None)
                out = recorder.submit(spikes, tick) if recorder is not None else []
                mx.eval(*state, *ring, *out)
                mx.synchronize(stream)
                timings["simulation_s"] += time.perf_counter() - started
                if recorder is not None:
                    started = time.perf_counter()
                    recorder.drain(keep=1)
                    timings["collection_s"] += time.perf_counter() - started
        started = time.perf_counter()
        events = recorder.finish() if recorder is not None else np.empty((0, 3), dtype=np.int64)
        timings["collection_s"] += time.perf_counter() - started
        timings["runtime_s"] = time.perf_counter() - runtime_start
        started = time.perf_counter()
        v, g, refractory, counts = (np.array(np.asarray(value), copy=True) for value in state)
        delay_ring = np.array(np.asarray(mx.stack(ring)), dtype=np.bool_, copy=True)
        timings["snapshot_s"] = time.perf_counter() - started
    return RunResult(v, g, refractory, delay_ring, counts, steps, events, timings)


def run(
    graph: Graph, steps: int, drive: Drive, *, model: Model = Model(),
    chunk_size: int = 32, record: bool = True, trial: int = 0,
    round_idx: int = 0, silenced=(), input_tape=None,
    spike_sink: Callable[[np.ndarray], object] | None = None,
    max_recorded_events: int = 1_000_000,
) -> RunResult:
    """Run one trial with scalar state, tape and event shapes.

    State arrays have shape ``(N,)``; the delay ring is ``(delay_steps, N)``.
    The input tape is ``(steps, channels)`` and events are ``(tick, neuron)``.
    Phase timing definitions are the same as :func:`run_batch`.
    """
    setup_start = time.perf_counter()
    trial = _integer(trial, "trial", 2**64 - 1)
    if spike_sink is not None and not callable(spike_sink):
        raise ValueError("spike_sink must be callable")
    tape = None
    if input_tape is not None:
        tape = np.asarray(input_tape)
        if tape.ndim != 2:
            raise ValueError("input_tape must have shape (steps, channel_occurrences)")
        tape = tape[:, None, :]

    def collect(events: np.ndarray) -> object:
        return spike_sink(events[:, (0, 2)])

    setup_overhead = time.perf_counter() - setup_start
    result = run_batch(
        graph, steps, drive, model=model, trials=(trial,), chunk_size=chunk_size,
        record=record, round_idx=round_idx, silenced=silenced, input_tape=tape,
        spike_sink=collect if spike_sink is not None else None,
        max_recorded_events=max_recorded_events,
    )
    started = time.perf_counter()
    events = result.spikes[:, (0, 2)]
    collection_overhead = time.perf_counter() - started
    result.timings["setup_compile_s"] += setup_overhead
    result.timings["collection_s"] += collection_overhead
    result.timings["runtime_s"] += collection_overhead
    return RunResult(
        result.v[0], result.g[0], result.refractory[0], result.delay_ring[:, 0],
        result.counts[0], result.step, events, result.timings,
    )
