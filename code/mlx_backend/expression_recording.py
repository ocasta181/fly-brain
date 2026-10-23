"""Bounded chronological device decoding for expression spike words."""
from functools import lru_cache
import mlx.core as mx
import numpy as np

MAX_EVENTS = 2**22


def collect_host_words(block, starts, lengths):
    """Decode uniform or sparse words with one bounded host inspection.

    This is an output-only shortcut: every neuron's emitted word is checked.
    Dense nonuniform or oversized layouts retain the device compactor.
    Uniform counts are scalar; sparse counts retain one uint32 per neuron.
    """
    if not block or len(block) != len(starts) or len(block) != len(lengths):
        return None
    neurons = block[0].size
    if (not neurons or neurons*sum(lengths) > MAX_EVENTS
            or any(word.ndim != 1 or word.size != neurons for word in block)
            or any(length < 1 or length > 32 for length in lengths)
            or any(start < previous+length for start, previous, length
                   in zip(starts[1:], starts, lengths))):
        return None
    times, words = [], []
    uniform = True
    active = 0
    for word, start, length in zip(block, starts, lengths):
        valid = np.asarray(word) & np.uint32((1 << length)-1)
        mask = int(valid[0])
        if not np.all(valid == mask):
            uniform = False
            count = np.count_nonzero(valid)
            if count*4 > neurons:
                return None
        else:
            count = neurons if mask else 0
        words.append(valid)
        active += count
        times.extend(start+bit for bit in range(length) if (mask >> bit) & 1)
    if not uniform:
        if active*4 > neurons*len(words):
            return None
        rows, steps = [], []
        counts = np.zeros(neurons, np.uint32)
        for word, start, length in zip(words, starts, lengths):
            indices = np.flatnonzero(word).astype(np.int32)
            masks = word[indices]
            for bit in range(length):
                emitted = indices[((masks >> bit) & 1) != 0]
                if emitted.size:
                    rows.append(emitted)
                    steps.append(np.full(emitted.size, start+bit, np.int64))
                    counts[emitted] += np.uint32(1)
        return np.concatenate(rows), np.concatenate(steps), counts
    # Keep the repeated axes as views until monitor writeback. A lone uniform
    # chunk can fill Brian's final arrays directly without dense intermediates.
    indices = np.broadcast_to(np.arange(neurons, dtype=np.int32), (len(times), neurons))
    steps = np.broadcast_to(np.asarray(times, np.int64)[:, None], (len(times), neurons))
    return indices, steps, len(times)


@lru_cache(maxsize=128)
def ranked_reset(program):
    """Random token stages retain their own selected-array cast context."""
    from .expression_lowering import RandomProgram, Segment
    from .expression_runtime import _segment_reset_ranks
    random = program.random_program
    return (program.collective_reset or isinstance(random, RandomProgram)
            and any(_segment_reset_ranks(op) for op in random.stages.get('reset', ())
                    if isinstance(op, Segment)))


@lru_cache(maxsize=1)
def _counts():
    return mx.fast.metal_kernel(
        name='expression_normal_spike_counts',
        input_names=['words', 'rows', 'bits'], output_names=['counts'],
        source='''uint j=thread_position_in_grid.x;
        uint flag=0;if(j<SIZE){uint t=j/N;flag=(words[rows[t]*N+j%N]>>bits[t])&1u;}
        uint count=simd_sum(flag);if((j&31u)==0)counts[j/32]=count;''')


@lru_cache(maxsize=1)
def _decode():
    return mx.fast.metal_kernel(
        name='expression_normal_spike_decode', input_names=['words', 'rows', 'bits', 'prefix', 'times'],
        output_names=['neurons', 'steps'], source='''
        uint j=thread_position_in_grid.x;
        uint flag=0;if(j<SIZE){uint t=j/N;flag=(words[rows[t]*N+j%N]>>bits[t])&1u;}
        uint local=simd_prefix_exclusive_sum(flag);
        if(flag){uint tile=j/32;uint out=(tile?prefix[tile-1]:0)+local;
        neurons[out]=int(j%N);steps[out]=times[j/N];}
        ''')


def _collect(block, starts, lengths):
    words = mx.stack(block)
    neurons = words.shape[1]
    capacity = neurons*sum(lengths)
    if capacity > MAX_EVENTS:
        return None
    rows = np.repeat(np.arange(len(lengths), dtype=np.uint32), lengths)
    bits = np.concatenate([np.arange(length, dtype=np.uint32) for length in lengths])
    times = np.repeat(np.asarray(starts, np.int64), lengths) + bits.astype(np.int64)
    device_rows, device_bits = mx.array(rows), mx.array(bits)
    tiles = (capacity+31)//32
    counts, = _counts()(
        inputs=[words, device_rows, device_bits],
        template=[('SIZE', capacity), ('N', neurons)], grid=(tiles*32, 1, 1),
        threadgroup=(256, 1, 1), output_shapes=[(tiles,)], output_dtypes=[mx.uint32])
    prefix = mx.cumsum(counts)
    indices, steps = _decode()(
        inputs=[words, device_rows, device_bits, prefix, mx.array(times)],
        template=[('SIZE', capacity), ('N', neurons)], grid=(tiles*32, 1, 1),
        threadgroup=(256, 1, 1), output_shapes=[(capacity,), (capacity,)],
        output_dtypes=[mx.int32, mx.int64])
    mx.eval(prefix, indices, steps)
    count = int(prefix[-1].item())
    return np.asarray(indices[:count]), np.asarray(steps[:count])


def collect(block, starts, lengths):
    """Return time/neuron ordered events, or None for an unsupported layout.

    Each compaction allocates at most MAX_EVENTS potential events. Disjoint
    increasing windows make concatenated chunks chronological as well.
    """
    if not block:
        return np.empty(0, np.int32), np.empty(0, np.int64)
    if len(block) != len(starts) or len(block) != len(lengths):
        return None
    neurons = block[0].size
    if (not neurons or any(word.ndim != 1 or word.size != neurons for word in block)
            or any(length < 1 or length > 32 for length in lengths)
            or any(start < previous+length for start, previous, length
                   in zip(starts[1:], starts, lengths))):
        return None
    budget = MAX_EVENTS//neurons
    if not budget:
        return None
    if sum(lengths) <= budget:
        return _collect(block, starts, lengths)
    indices, steps, words, origins, sizes = [], [], [], [], []
    used = 0

    def flush():
        nonlocal words, origins, sizes, used
        rows, times = _collect(words, origins, sizes)
        indices.append(rows)
        steps.append(times)
        words, origins, sizes, used = [], [], [], 0

    for word, start, length in zip(block, starts, lengths):
        begin = 0
        while begin < length:
            size = min(length-begin, budget-used)
            part = (word if begin == 0 and size == length else
                    (word >> begin) & mx.array((1 << size)-1, mx.uint32))
            words.append(part)
            origins.append(start+begin)
            sizes.append(size)
            used += size
            begin += size
            if used == budget:
                flush()
    if words:
        flush()
    return np.concatenate(indices), np.concatenate(steps)


def ordered_chunks(indices, steps):
    """Check boundaries of chunks individually certified by collect()."""
    previous = None
    for rows, times in zip(indices, steps):
        if not rows.size:
            continue
        first = (int(times.flat[0]), int(rows.flat[0]))
        if previous is not None and first < previous:
            return False
        previous = (int(times.flat[-1]), int(rows.flat[-1]))
    return True
