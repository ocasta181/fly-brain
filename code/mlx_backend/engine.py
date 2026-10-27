"""Fixed Eon LIF runtime using native Metal and external VF64 operations.

Physical values remain absolute-SI binary64 words on the device. Chronological
blocks read completed delay history and add canonical incoming edges separately.

The bounded event extraction lifecycle adapts Kisame76/drosophila-brain-mlx,
revision e417b33616513ef350b1b1c3cdf2b5b7a1799c8e, through mlx32/engine.py
(MIT, copyright 2026 Kiyan Doguc; the complete notice accompanies this package).
"""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .model import Drive, Graph, Model, integer
from .vf64 import shader_header

_I32_MAX = 2**31 - 1
_EVENT_CAP = 16_384
_QUEUE_BYTES = 128 * 1024**2
_QUEUE_BLOCKS = 16
_STATE_GROUP = 128


@dataclass(frozen=True)
class RunResult:
    v: np.ndarray
    g: np.ndarray
    lastspike: np.ndarray
    not_refractory: np.ndarray
    delay_ring: np.ndarray
    counts: np.ndarray
    step: int
    spikes: np.ndarray
    timings: dict[str, float]
    v_trace: np.ndarray | None = None
    g_trace: np.ndarray | None = None
    eligible_trace: np.ndarray | None = None


class RecordOverflow(OverflowError):
    """The complete event result exceeds the requested host collection limit."""


_MARK_SOURCE = """
    uint i = thread_position_in_grid.x;
    uint lane = thread_position_in_grid.y;
    uint trial = thread_position_in_grid.z;
    if (trial >= BATCH) return;
    bool valid = i < N;
    if (COMPACT && lane == 0u) {
        bool active = valid && (v[trial * N + i] != RAWREST || g[trial * N + i] != 0ul
            || channel_ptr[i] != channel_ptr[i + 1]);
        uint a = uint(active), b = uint(valid && !active);
        uint na = simd_sum(a), nb = simd_sum(b);
        uint pa = simd_prefix_exclusive_sum(a), pb = simd_prefix_exclusive_sum(b);
        uint front = 0u, back = 0u;
        if ((i & 31u) == 0u) {
            front = atomic_fetch_add_explicit(&cursor[trial * 2], na, memory_order_relaxed);
            back = atomic_fetch_add_explicit(&cursor[trial * 2 + 1], nb, memory_order_relaxed);
        }
        front = simd_broadcast_first(front);
        back = simd_broadcast_first(back);
        if (valid) {
            uint slot = active ? front + pa : N - 1u - back - pb;
            atomic_store_explicit(&work_order[trial * N + slot], i, memory_order_relaxed);
        }
    }
    if (!valid) return;
    const ulong begin = tick_begin[0];
    uint mask = 0u;
    uint k = begin < DELAY ? uint(min(ulong(STEPS), ulong(DELAY) - begin)) : 0u;
    while (k < STEPS) {
        const uint row = uint((begin + ulong(k)) % DELAY);
        const uint shift = row & 31u;
        const uint length = min(STEPS - k, min(32u - shift, DELAY - row));
        const uint bits = history[((row >> 5u) * BATCH + trial) * N + i] >> shift;
        mask |= (bits & (~0u >> (32u - length))) << k;
        k += length;
    }
    if (lane == 0u)
        atomic_store_explicit(&source_masks[trial * N + i], mask, memory_order_relaxed);
    if (!mask) return;
    for (uint p = out_ptr[i] + lane; p < out_ptr[i + 1]; p += LANES) {
        uint e = out_edge[p];
        atomic_fetch_or_explicit(&edge_active[trial * WORDS + (e >> 5u)],
                                 1u << (e & 31u), memory_order_relaxed);
        atomic_fetch_or_explicit(&arrival_masks[trial * N + edge_post[e]],
                                 mask, memory_order_relaxed);
    }
"""

_STATE_SOURCE = """
    uint job = thread_position_in_grid.x;
    uint trial = thread_position_in_grid.y;
    if (job >= N || trial >= BATCH) return;
    uint i = COMPACT ? work_order[trial * N + job] : job;
    uint j = trial * N + i;
    const uint rn = soft_round_near_even;
    uint flags = 0u, spike_mask = 0u;
    ulong vi = v[j], gi = g[j], last = lastspike[j];
    long count = counts[j], required = 0;
    uint ready = refractory[j];
    bool eligible = ready != 0;
    const uint first_edge = edge_ptr[i], last_edge = edge_ptr[i + 1];
    const uint arrivals = arrival_masks[j];
    const bool idle = FAST_DEFAULT && !TRACE && vi == parameters[3] && gi == 0ul
        && !arrivals && channel_ptr[i] == channel_ptr[i + 1];
    if (!FAST_DEFAULT) {
        const ulong duration = channel_ptr[i] == channel_ptr[i + 1]
            ? parameters[10] : 0ul;
        const ulong value = soft_mul64_status(
            soft_add64_status(duration, parameters[8], rn, flags),
            parameters[9], rn, flags);
        required = as_type<long>(soft_f64_to_int_status(
            value, soft_round_min_mag, false, true, 64, flags));
    }
    for (uint k = idle ? STEPS : 0u; k < STEPS; ++k) {
        const ulong tick = tick_begin[0] + ulong(k);
        ulong t = 0ul;
        if (FAST_DEFAULT) {
            eligible = tick >= ulong(ready);
        } else {
            t = soft_mul64_status(soft_uint_to_f64_status(
                tick, false, rn, flags), parameters[7], rn, flags);
            const ulong elapsed_value = soft_mul64_status(
                soft_sub64_status(soft_add64_status(
                    t, parameters[8], rn, flags), last, rn, flags),
                parameters[9], rn, flags);
            const long elapsed = as_type<long>(soft_f64_to_int_status(
                elapsed_value, soft_round_min_mag, false, true, 64, flags));
            eligible = elapsed >= required;
        }
        // This shortcut is qualified only for the default raw-bit fixed point.
        const bool resting = FAST_DEFAULT && vi == parameters[3] && gi == 0ul;
        if (eligible && !resting) {
            if ((parameters[3] & 0x7ffffffffffffffful) == 0ul) {
                const ulong sum = soft_fma64_status(parameters[0], gi, vi, rn, flags);
                vi = soft_mul64_status(sum, parameters[1], rn, flags);
            } else {
                const ulong coupling = soft_fma64_status(
                    parameters[0], gi, parameters[3] ^ 0x8000000000000000ul,
                    rn, flags);
                const ulong sum = soft_add64_status(coupling, vi, rn, flags);
                vi = soft_fma64_status(sum, parameters[1], parameters[3], rn, flags);
            }
            gi = soft_mul64_status(parameters[2], gi, rn, flags);
        }
        const bool spike = eligible && !resting && soft_less64_status(
            parameters[5], vi, false, false, flags);
        if (spike) {
            if (FAST_DEFAULT) {
                t = soft_mul64_status(soft_uint_to_f64_status(
                    tick, false, rn, flags), parameters[7], rn, flags);
            }
            last = t;
            eligible = false;
            if (FAST_DEFAULT) {
                // Default state encodes an absolute ready tick; targeted rfc is zero.
                ready = uint(tick) + 1u;
                if (channel_ptr[i] == channel_ptr[i + 1]) {
                    // Default recovery is at 22 or 23 ticks within the int32 horizon.
                    ready = uint(tick) + 22u;
                    const ulong recovery_t = soft_mul64_status(soft_uint_to_f64_status(
                        ulong(ready), false, rn, flags), parameters[7], rn, flags);
                    const ulong elapsed_value = soft_mul64_status(
                        soft_sub64_status(soft_add64_status(
                            recovery_t, parameters[8], rn, flags), last, rn, flags),
                        parameters[9], rn, flags);
                    const long elapsed = as_type<long>(soft_f64_to_int_status(
                        elapsed_value, soft_round_min_mag, false, true, 64, flags));
                    ready += uint(elapsed < 22);
                }
            }
        }
        if (eligible) {
            const uint tick_bit = 1u << k;
            if (arrivals & tick_bit) {
                for (uint word = first_edge >> 5u; word < ((last_edge + 31u) >> 5u); ++word) {
                    uint bits = edge_active[trial * WORDS + word];
                    if (word == (first_edge >> 5u)) bits &= ~0u << (first_edge & 31u);
                    if (word == (last_edge >> 5u)) bits &= (1u << (last_edge & 31u)) - 1u;
                    while (bits) {
                        const uint e = (word << 5u) + ctz(bits);
                        bits &= bits - 1u;
                        if (source_masks[trial * N + edge_pre[e]] & tick_bit)
                            gi = soft_add64_status(gi, weights[e], rn, flags);
                    }
                }
            }
            const uint row = (k * BATCH + trial) * CHANNELS;
            for (uint c = channel_ptr[i]; c < channel_ptr[i + 1]; ++c) {
                if (tape[row + channel_order[c]])
                    vi = soft_add64_status(vi, parameters[6], rn, flags);
            }
        }
        if (spike) {
            vi = parameters[4];
            gi = 0ul;
            ++count;
            spike_mask |= 1u << k;
        }
        // TRACE
    }
    if (STEPS == DELAY && DELAY <= 32) {
        // Full-delay blocks start at slot zero; unused high bits remain zero.
        history_out[j] = spike_mask;
    } else {
        for (uint word = 0; word < HISTORY_WORDS; ++word) {
            const uint offset = (word * BATCH + trial) * N + i;
            history_out[offset] = history[offset];
        }
        for (uint k = 0; k < STEPS;) {
            const uint slot = uint((tick_begin[0] + k) % DELAY);
            const uint shift = slot & 31u;
            const uint length = min(STEPS - k, min(32u - shift, DELAY - slot));
            const uint mask = (~0u >> (32u - length)) << shift;
            const uint offset = ((slot >> 5u) * BATCH + trial) * N + i;
            history_out[offset] = (history_out[offset] & ~mask)
                | (((spike_mask >> k) << shift) & mask);
            k += length;
        }
    }
    v_out[j] = vi;
    g_out[j] = gi;
    lastspike_out[j] = last;
    refractory_out[j] = FAST_DEFAULT ? ready : uint(eligible);
    counts_out[j] = count;
    spike_masks[j] = spike_mask;
"""

_RECORD_SOURCE = """
    uint i = thread_position_in_grid.x;
    uint trial = thread_position_in_grid.y;
    if (i >= N || trial >= BATCH) return;
    uint j = trial * N + i;
    if (j < window[0] || j >= window[1]) return;
    uint word = masks[j] >> window[2];
    if (window[3] < 32u) word &= (1u << window[3]) - 1u;
    uint size = popcount(word);
    if (!size) return;
    uint slot = atomic_fetch_add_explicit(&count[0], size, memory_order_relaxed);
    for (uint bit = window[2]; word && slot < cap[0]; ++bit, word >>= 1u) {
        if (!(word & 1u)) continue;
        atomic_store_explicit(&events[3 * slot], int(bit), memory_order_relaxed);
        atomic_store_explicit(&events[3 * slot + 1], int(trial), memory_order_relaxed);
        atomic_store_explicit(&events[3 * slot + 2], int(i), memory_order_relaxed);
        ++slot;
    }
"""


def _metal():
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("the VF64 backend requires MLX with native Metal") from exc
    if not mx.metal.is_available():
        raise RuntimeError("the VF64 backend requires native Metal; CPU execution is unsupported")
    return mx


def _kernels(mx, trace: bool, parameters: np.ndarray):
    mark = mx.fast.metal_kernel(
        name="eon_vf64_mark",
        input_names=["v", "g", "history", "tick_begin", "channel_ptr",
                     "out_ptr", "out_edge", "edge_post"],
        output_names=["source_masks", "edge_active", "arrival_masks", "work_order", "cursor"],
        source=_MARK_SOURCE,
        header=f"constant ulong RAWREST = 0x{int(parameters[3]):016x}ul;\n",
        atomic_outputs=True,
    )
    outputs = ["v_out", "g_out", "lastspike_out", "refractory_out",
               "counts_out", "history_out", "spike_masks"]
    trace_source = ""
    if trace:
        outputs += ["v_trace_out", "g_trace_out", "eligible_trace_out"]
        trace_source = """
            const uint offset = (k * BATCH + trial) * N + i;
            v_trace_out[offset] = vi;
            g_trace_out[offset] = gi;
            eligible_trace_out[offset] = uchar(eligible);
        """
    state = mx.fast.metal_kernel(
        name="eon_vf64_state",
        input_names=["v", "g", "lastspike", "refractory", "counts", "history",
                     "edge_ptr", "edge_pre", "weights", "channel_ptr", "channel_order",
                     "source_masks", "edge_active", "arrival_masks", "work_order",
                     "tape", "tick_begin"],
        output_names=outputs,
        source=_STATE_SOURCE.replace("// TRACE", trace_source).replace("parameters[", "eon_parameters["),
        header=shader_header() + "\nconstant ulong eon_parameters[] = {"
            + ",".join(f"0x{int(word):016x}ul" for word in parameters) + "};\n",
        compile_options={"math_mode": "safe"},
    )
    record = mx.fast.metal_kernel(
        name="eon_vf64_events", input_names=["masks", "cap", "window"],
        output_names=["events", "count"], source=_RECORD_SOURCE, atomic_outputs=True,
    )
    return mark, state, record


class _Recorder:
    def __init__(self, mx, kernel, stream, n: int, batch: int, limit: int,
                 sink: Callable[[np.ndarray], object] | None):
        self.mx, self.kernel, self.stream = mx, kernel, stream
        self.n, self.batch, self.limit, self.sink = n, batch, limit, sink
        self.cap = _EVENT_CAP if sink is not None else min(_EVENT_CAP, max(1, limit))
        self.cap_array = mx.array([self.cap], dtype=mx.uint32)
        self.pending = deque()
        self.collected: list[np.ndarray] = []
        self.total = 0
        self.simulation_s = 0.0

    def extract(self, masks, lo: int, hi: int, first_bit: int, bits: int):
        return self.kernel(
            inputs=[masks, self.cap_array,
                    self.mx.array([lo, hi, first_bit, bits], dtype=self.mx.uint32)],
            template=[("N", self.n), ("BATCH", self.batch)],
            output_shapes=[(self.cap, 3), (1,)],
            output_dtypes=[self.mx.int32, self.mx.uint32],
            grid=(self.n, self.batch, 1), threadgroup=(min(self.n, 128), 1, 1),
            init_value=0, stream=self.stream,
        )

    def submit(self, masks, tick: int, length: int) -> list:
        events, count = self.extract(masks, 0, self.batch * self.n, 0, length)
        self.pending.append((tick, length, masks, events, count))
        return [events, count]

    def _collect(self, events, size: int, tick: int) -> None:
        if not size:
            return
        batch = np.array(np.asarray(events[:size]), dtype=np.int64, copy=True)
        events = None
        batch[:, 0] += tick
        batch = batch[np.lexsort((batch[:, 2], batch[:, 1], batch[:, 0]))]
        if self.sink is None:
            self.collected.append(batch)
        else:
            self.sink(batch)

    def drain(self) -> None:
        try:
            while self.pending:
                tick, length, masks, events, count = self.pending[0]
                size = int(count.item())
                if self.sink is None and self.total + size > self.limit:
                    raise RecordOverflow(f"recorded events exceed max_recorded_events={self.limit}")
                if size <= self.cap:
                    self._collect(events, size, tick)
                else:
                    # Dense blocks retain the state and use bounded, sorted
                    # tick/linear-neuron windows for event extraction only.
                    for bit in range(length):
                        for lo in range(0, self.batch * self.n, self.cap):
                            started = time.perf_counter()
                            bounded = self.extract(masks, lo, min(lo + self.cap, self.batch * self.n), bit, 1)
                            self.mx.eval(*bounded)
                            self.mx.synchronize(self.stream)
                            self.simulation_s += time.perf_counter() - started
                            self._collect(bounded[0], int(bounded[1].item()), tick)
                self.total += size
                self.pending.popleft()
        except BaseException:
            self.pending.clear()
            self.collected.clear()
            masks = events = count = bounded = None
            raise

    def finish(self) -> np.ndarray:
        self.drain()
        if self.sink is not None or not self.collected:
            return np.empty((0, 3), dtype=np.int64)
        return np.concatenate(self.collected)

    def close(self) -> None:
        self.pending.clear()
        self.collected.clear()
        self.cap_array = None


def _silenced_indices(silenced: Sequence[int] | np.ndarray, n: int) -> np.ndarray:
    array = np.asarray(silenced)
    if array.dtype == np.bool_:
        if array.shape != (n,):
            raise ValueError("silenced bool mask must have shape (n_neurons,)")
        return np.flatnonzero(array)
    if array.ndim != 1:
        raise ValueError("silenced must be a sequence of neuron indices or a bool mask")
    return np.array([integer(value, "silenced index", maximum=n - 1)
                     for value in array], dtype=np.int64)


def _validate_clock(steps: int, parameters: np.ndarray) -> None:
    if not steps:
        return
    values = parameters.view(np.float64)
    dt, epsilon, reciprocal, duration = map(float, values[7:11])
    last_time = float(steps - 1) * dt
    elapsed = ((last_time + epsilon) + 1e4) * reciprocal
    required = (duration + epsilon) * reciprocal
    if not all(math.isfinite(value) and 0 <= value < 2**63
               for value in (last_time, elapsed, required)):
        raise ValueError("clock or refractory conversion exceeds finite signed int64 bounds")


def _initial_state(batch: int, n: int, model: Model) -> list[np.ndarray]:
    return [
        np.full((batch, n), model.v_rest_v, dtype=np.float64).view(np.uint64),
        np.zeros((batch, n), dtype=np.uint64),
        np.full((batch, n), -1e4, dtype=np.float64).view(np.uint64),
        np.full((batch, n), 0 if model == Model() else 1, dtype=np.uint32),
        np.zeros((batch, n), dtype=np.int64),
        np.zeros(((model.delay_steps + 31) // 32, batch, n), dtype=np.uint32),
    ]


def _snapshots(state) -> list[np.ndarray]:
    return [np.array(np.asarray(value), copy=True) for value in state]


def _result(snapshots: list[np.ndarray], steps: int, events: np.ndarray,
            timings: dict[str, float], traces: list[np.ndarray] | None,
            model: Model) -> RunResult:
    v, g, last, refractory, counts, packed = snapshots
    eligible = refractory.astype(bool)
    if model == Model():
        eligible = refractory <= steps - 1 if steps else np.ones_like(refractory, dtype=bool)
    ring = np.zeros((model.delay_steps, *counts.shape), dtype=np.uint8)
    if steps and counts.size:
        for slot in range(model.delay_steps):
            ring[slot] = (packed[slot >> 5] >> np.uint32(slot & 31)) & np.uint32(1)
    return RunResult(
        v.view(np.float64), g.view(np.float64), last.view(np.float64),
        eligible, ring, counts, steps, events, timings,
        *(traces if traces is not None else (None, None, None)),
    )


def run_batch(
    graph: Graph, steps: int, drive: Drive, *,
    trials: tuple[int, ...] | list[int] = (0,), model: Model = Model(),
    chunk_size: int = 32, record: bool = True, round_idx: int = 0,
    silenced: Sequence[int] | np.ndarray = (), input_tape: np.ndarray | None = None,
    spike_sink: Callable[[np.ndarray], object] | None = None,
    max_recorded_events: int = 1_000_000, trace: bool = False,
) -> RunResult:
    """Run independent fixed-model trials with absolute-SI state snapshots.

    State is ``(B,N)``, the ring is ``(delay_steps,B,N)`` and supplied input is
    ``(steps,B,channel_occurrences)``. Events are sorted int64 rows containing
    ``(tick,batch_position,neuron)``. A sink receives complete sorted bounded
    batches; returned events are then empty. Without a sink, the complete
    collection is limited by ``max_recorded_events`` or raises RecordOverflow.

    Timings are seconds. Setup includes disposable compilation/warm-up.
    Runtime includes host input, synchronized simulation/event extraction,
    collection/callbacks and snapshots. Subphase timers do not overlap; input
    staging may overlap already submitted device work. Diagnostic traces grow
    with the full horizon only when explicitly requested.
    """
    setup_start = time.perf_counter()
    steps = integer(steps, "steps")
    chunk_size = integer(chunk_size, "chunk_size", minimum=1)
    round_idx = integer(round_idx, "round_idx", maximum=2**64 - 1)
    limit = integer(max_recorded_events, "max_recorded_events")
    if not isinstance(graph, Graph) or not isinstance(drive, Drive) or not isinstance(model, Model):
        raise ValueError("graph, drive and model must be Graph, Drive and Model values")
    if not isinstance(trials, (tuple, list)) or not trials:
        raise ValueError("trials must be a nonempty tuple or list of trial identifiers")
    trial_ids = tuple(integer(value, "trial", maximum=2**64 - 1) for value in trials)
    batch = integer(len(trial_ids), "trial count", minimum=1)
    if len(set(trial_ids)) != batch:
        raise ValueError("trials must contain distinct trial identifiers")
    if not isinstance(record, (bool, np.bool_)) or not isinstance(trace, (bool, np.bool_)):
        raise ValueError("record and trace must be bool")
    if spike_sink is not None and not callable(spike_sink):
        raise ValueError("spike_sink must be callable")
    n, channels, delay = graph.n_neurons, drive.targets.size, model.delay_steps
    if np.any(drive.targets < 0) or np.any(drive.targets >= n):
        raise ValueError("drive target is outside the graph")
    probabilities = drive.rates_hz * model.dt_s
    if np.any(probabilities > 1):
        raise ValueError("each input rate * dt_s must be at most 1")
    span = min(chunk_size, delay, 32)
    block_length = min(span, steps)
    words = max(1, (graph.n_edges + 31) // 32)
    if max(n + 1, graph.n_edges, batch * n * delay, batch * words,
           block_length * batch * max(1, n, channels), batch * n * 3,
           3 * min(_EVENT_CAP, max(1, limit))) > _I32_MAX:
        raise ValueError("state, delay, input or event indexing exceeds int32 bounds; reduce trials or chunk_size")
    tape = None
    if input_tape is not None:
        tape = np.asarray(input_tape)
        if tape.dtype not in (np.dtype(np.uint8), np.dtype(np.bool_)):
            raise ValueError("input_tape must have uint8 or bool dtype")
        if tape.shape != (steps, batch, channels):
            raise ValueError("input_tape must have shape (steps, trials, channel_occurrences)")
        if tape.dtype == np.uint8:
            for tick in range(0, steps, span):
                if np.any(tape[tick:tick + span] > 1):
                    raise ValueError("input_tape may contain only zero and one")
    parameters = model.parameters()
    _validate_clock(steps, parameters)
    silence = _silenced_indices(silenced, n)
    weights = graph.contacts.astype(np.float64) * np.float64(model.weight_v)
    weights[np.isin(graph.edge_pre, silence)] = np.float64(0)
    if not np.all(np.isfinite(weights)):
        raise ValueError("rounded edge weights must be finite")
    channel_order = np.argsort(drive.targets, kind="stable").astype(np.uint32)
    channel_ptr = np.empty(n + 1, dtype=np.uint32)
    channel_ptr[0] = 0
    channel_ptr[1:] = np.cumsum(np.bincount(drive.targets, minlength=n), dtype=np.uint64)
    initial = _initial_state(batch, n, model)
    timings = dict(setup_compile_s=0.0, compile_warmup_s=0.0, runtime_s=0.0,
                   simulation_s=0.0, input_generation_staging_s=0.0,
                   collection_s=0.0, snapshots_s=0.0)
    traces = [np.empty((steps, batch, n), dtype=dtype)
              for dtype in (np.float64, np.float64, np.bool_)] if trace else None
    if not n or not steps:
        timings["setup_compile_s"] = time.perf_counter() - setup_start
        return _result(initial, steps, np.empty((0, 3), dtype=np.int64), timings, traces, model)

    rngs = [np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([drive.seed, round_idx, trial]))) for trial in trial_ids]
    mx = _metal()
    stream = mx.new_stream(mx.gpu)
    state, static = [], []
    recorder = None
    outputs = frames = None
    recorded = []
    disposable = warm_frames = warm_outputs = warm_events = None
    with mx.stream(stream):
        try:
            mark_kernel, state_kernel, record_kernel = _kernels(mx, bool(trace), parameters)
            host_static = [graph.edge_ptr, graph.edge_pre, weights.view(np.uint64),
                           channel_ptr, channel_order, graph.out_ptr, graph.out_edge,
                           graph.edge_post]
            static = [mx.array(value if value.size else np.zeros(1, dtype=value.dtype))
                      for value in host_static]
            (edge_ptr, edge_pre, edge_weights, cp, co,
             out_ptr, out_edge, edge_post) = static
            grid = (n, batch, 1)
            fast_default = model == Model()
            compact = fast_default and not trace
            lanes = 8 if batch == 1 else 4
            history_words = (delay + 31) // 32

            def fresh():
                return [mx.array(value) for value in initial]

            def dispatch(current, tick: int, length: int, block):
                begin = mx.array([tick], dtype=mx.uint64)
                template = [("N", n), ("BATCH", batch), ("STEPS", length), ("DELAY", delay)]
                masks, active, arrivals, work_order, _ = mark_kernel(
                    inputs=[current[0], current[1], current[5], begin, cp,
                            out_ptr, out_edge, edge_post],
                    template=template + [("WORDS", words), ("COMPACT", compact), ("LANES", lanes)],
                    grid=(((n + 127) // 128) * 128, lanes, batch), threadgroup=(128, 1, 1),
                    output_shapes=[(batch, n), (batch, words), (batch, n),
                                   (batch, n) if compact else (1,), (batch, 2)],
                    output_dtypes=[mx.uint32] * 5, init_value=0, stream=stream,
                )
                shapes = [(batch, n)] * 5 + [(history_words, batch, n), (batch, n)]
                dtypes = [mx.uint64, mx.uint64, mx.uint64, mx.uint32,
                          mx.int64, mx.uint32, mx.uint32]
                if trace:
                    shapes += [(length, batch, n)] * 3
                    dtypes += [mx.uint64, mx.uint64, mx.uint8]
                return state_kernel(
                    inputs=[*current, edge_ptr, edge_pre, edge_weights, cp, co,
                            masks, active, arrivals, work_order, block, begin],
                    template=template + [("WORDS", words), ("CHANNELS", max(1, channels)),
                                         ("HISTORY_WORDS", history_words),
                                         ("TRACE", bool(trace)),
                                         ("FAST_DEFAULT", fast_default), ("COMPACT", compact)],
                    grid=grid, threadgroup=(min(n, _STATE_GROUP), 1, 1), output_shapes=shapes,
                    output_dtypes=dtypes, stream=stream,
                )

            def stage(tick: int):
                length = min(span, steps - tick)
                if not channels:
                    host = np.zeros((length, batch, 1), dtype=np.uint8)
                elif tape is not None:
                    host = np.ascontiguousarray(tape[tick:tick + length], dtype=np.uint8)
                else:
                    host = np.empty((length, batch, channels), dtype=np.uint8)
                    for position, rng in enumerate(rngs):
                        host[:, position, :] = rng.random((length, channels)) < probabilities
                return mx.array(host)

            warmup_start = time.perf_counter()
            warm_recorder = _Recorder(mx, record_kernel, stream, n, batch, limit, spike_sink) if record else None
            try:
                for length in sorted({block_length, steps % span} - {0}):
                    disposable = fresh()
                    warm_frames = mx.zeros((length, batch, max(1, channels)), dtype=mx.uint8)
                    warm_outputs = dispatch(disposable, 0, length, warm_frames)
                    warm_events = warm_recorder.submit(warm_outputs[6], 0, length) if warm_recorder else []
                    mx.eval(*warm_outputs, *warm_events)
                    mx.synchronize(stream)
                    if warm_recorder:
                        warm_recorder.pending.clear()
                    del disposable, warm_frames, warm_outputs, warm_events
            finally:
                if warm_recorder:
                    warm_recorder.close()
                disposable = warm_frames = warm_outputs = warm_events = None
            timings["compile_warmup_s"] = time.perf_counter() - warmup_start
            state = fresh()
            recorder = _Recorder(mx, record_kernel, stream, n, batch, limit, spike_sink) if record else None
            block_bytes = (batch * n * (52 + 4 * history_words) + batch * words * 4 + 8 * batch
                           + block_length * batch * max(1, channels)
                           + (recorder.cap * 12 + 4 if recorder else 0))
            queue_depth = 1 if trace else min(_QUEUE_BLOCKS, max(1, _QUEUE_BYTES // block_bytes))
            mx.eval(*state, *static)
            mx.synchronize(stream)
            timings["setup_compile_s"] = time.perf_counter() - setup_start

            runtime_start = time.perf_counter()
            for block_index, tick in enumerate(range(0, steps, span), 1):
                length = min(span, steps - tick)
                started = time.perf_counter()
                frames = stage(tick)
                timings["input_generation_staging_s"] += time.perf_counter() - started
                started = time.perf_counter()
                outputs = dispatch(state, tick, length, frames)
                state = outputs[:6]
                recorded = recorder.submit(outputs[6], tick, length) if recorder else []
                mx.async_eval(*outputs, *recorded)
                timings["simulation_s"] += time.perf_counter() - started
                if block_index % queue_depth and tick + length < steps:
                    continue
                started = time.perf_counter()
                mx.synchronize(stream)
                timings["simulation_s"] += time.perf_counter() - started
                started = time.perf_counter()
                previous_extraction = recorder.simulation_s if recorder else 0.0
                if recorder:
                    recorder.drain()
                if traces is not None:
                    traces[0][tick:tick + length] = np.asarray(outputs[7]).view(np.float64)
                    traces[1][tick:tick + length] = np.asarray(outputs[8]).view(np.float64)
                    traces[2][tick:tick + length] = np.asarray(outputs[9], dtype=bool)
                extraction = recorder.simulation_s - previous_extraction if recorder else 0.0
                timings["simulation_s"] += extraction
                timings["collection_s"] += time.perf_counter() - started - extraction
            started = time.perf_counter()
            events = recorder.finish() if recorder else np.empty((0, 3), dtype=np.int64)
            timings["collection_s"] += time.perf_counter() - started
            started = time.perf_counter()
            snapshots = _snapshots(state)
            result = _result(snapshots, steps, events, timings, traces, model)
            timings["snapshots_s"] = time.perf_counter() - started
            timings["runtime_s"] = time.perf_counter() - runtime_start
            return result
        finally:
            # Only this run's stream and references are drained; other MLX users
            # retain their streams and allocator/cache state.
            failed = sys.exc_info()[0] is not None
            try:
                try:
                    mx.synchronize(stream)
                except Exception:
                    if not failed:
                        raise
            finally:
                if recorder is not None:
                    recorder.close()
                state.clear()
                static.clear()
                outputs = frames = None
                recorded = []
                edge_ptr = edge_pre = edge_weights = cp = co = None
                out_ptr = out_edge = edge_post = None
