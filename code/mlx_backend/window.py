"""Bounded native spike-word collection and canonical event staging."""
from functools import lru_cache
from operator import index

import mlx.core as mx
import numpy as np


@lru_cache(maxsize=1)
def recording_kernel():
    return mx.fast.metal_kernel(
        name="fly_window_record",
        input_names=["words"],
        output_names=["counter", "entries"],
        atomic_outputs=True,
        source="""
        uint i = thread_position_in_grid.x;
        if(i >= WORDS) return;
        uint bits = words[i];
        if(bits == 0) return;
        uint pos = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        if(pos >= CAPACITY) return;
        atomic_store_explicit(&entries[2*pos], i, memory_order_relaxed);
        atomic_store_explicit(&entries[2*pos+1], bits, memory_order_relaxed);
        """,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=1)
def _count_words_kernel():
    return mx.fast.metal_kernel(
        name="fly_count_spike_words",
        input_names=["words", "previous"], output_names=["counts"],
        source="""
        uint row = thread_position_in_grid.x;
        if(row >= ROWS) return;
        uint count = previous[row];
        for(uint window=0; window<WINDOWS; ++window)
            count += popcount(words[window*ROWS+row]);
        counts[row] = count;
        """, compile_options={"math_mode": "safe"},
    )


def count_words(block, previous):
    """Accumulate Brian's uint32 count bits without materializing spike rows."""
    words = mx.stack(block)
    rows = previous.size
    counts, = _count_words_kernel()(
        inputs=[words, previous], template=[("ROWS", rows), ("WINDOWS", len(block))],
        grid=(rows, 1, 1), threadgroup=(min(256, rows), 1, 1),
        output_shapes=[(rows,)], output_dtypes=[mx.uint32],
    )
    # Bound retained graphs/word buffers as well as the count array itself.
    mx.eval(counts)
    return counts


def collect_words(block, *, capacity=65536, allow_split=False):
    """Transfer active words with bounded compaction buffers.

    Try the whole block first. With ``allow_split``, overflow retries device
    slices no larger than capacity, so even dense output is collected exactly.
    """
    words = mx.stack(block)
    capacity = min(words.size, index(capacity))
    if capacity < 1:
        raise ValueError("Recording capacity must be positive")

    def compact(current):
        slots = min(current.size, capacity)
        counter, entries = recording_kernel()(
            inputs=[current],
            template=[("WORDS", current.size), ("CAPACITY", slots)],
            grid=(current.size, 1, 1),
            threadgroup=(min(256, current.size), 1, 1),
            output_shapes=[(1,), (slots, 2)],
            output_dtypes=[mx.uint32, mx.uint32],
            init_value=0,
        )
        return int(counter.item()), entries

    count, entries = compact(words)
    if count <= capacity:
        active = np.asarray(entries[:count])
        positions, masks = active[:, 0].astype(np.int64), active[:, 1]
    else:
        if not allow_split:
            raise OverflowError(
                f"Window recording capacity {capacity} exceeded by {count} active words"
            )
        # The optimistic buffer is incomplete; collect every segment afresh.
        # Each segment has <= capacity words, so its exact active count fits.
        del entries
        flat = words.reshape(-1)
        positions, masks = [], []
        for start in range(0, words.size, capacity):
            count, entries = compact(flat[start : start + capacity])
            if count:
                active = np.asarray(entries[:count])
                positions.append(active[:, 0].astype(np.int64) + start)
                masks.append(active[:, 1])
        positions, masks = np.concatenate(positions), np.concatenate(masks)
    windows, indices = np.divmod(positions, words.shape[1])
    return windows, indices, masks


def _event_chunk(raw, begin, end, trials, channels):
    if isinstance(raw, np.ndarray):
        chunk = np.ascontiguousarray(raw[:, begin:end, :])
    else:
        chunk = raw.chunk(begin, end)
    if (
        not isinstance(chunk, np.ndarray)
        or chunk.dtype != np.uint8
        or chunk.shape != (trials, end - begin, channels)
        or not chunk.flags.c_contiguous
        or chunk.max(initial=0) > 1
    ):
        raise ValueError("Invalid canonical stimulus chunk")
    return chunk


