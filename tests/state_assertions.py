"""Exact state comparisons shared by independent CPU/native tests."""

from functools import partial

import numpy as np


def assert_array_bits_equal(actual, expected, name=""):
    assert actual.dtype == expected.dtype, name
    if expected.dtype.kind == "f":
        actual = actual.view(f"u{actual.itemsize}")
        expected = expected.view(f"u{expected.itemsize}")
    np.testing.assert_array_equal(actual, expected, err_msg=name)


def assert_array_bytes_equal(actual_values, expected_values, label=""):
    assert actual_values.dtype == expected_values.dtype, label
    assert actual_values.shape == expected_values.shape, label
    if actual_values.tobytes() != expected_values.tobytes():
        actual_bytes = np.ascontiguousarray(actual_values).view(np.uint8).reshape(-1)
        expected_bytes = np.ascontiguousarray(expected_values).view(np.uint8).reshape(-1)
        index = int(np.flatnonzero(actual_bytes != expected_bytes)[0]) // actual_values.dtype.itemsize
        actual_item, expected_item = actual_values.reshape(-1)[index], expected_values.reshape(-1)[index]
        raise AssertionError(f"{label}[{index}]: {actual_item!r} ({actual_item.tobytes().hex()}) "
                             f"!= {expected_item!r} ({expected_item.tobytes().hex()})")


def compare_state(actual, expected, compare=assert_array_bits_equal, *, strict_keys=False):
    if strict_keys:
        assert actual.keys() == expected.keys()
    for name, wanted in expected.items():
        compare(actual[name], wanted, name)


assert_raw_state = compare_state
assert_exact_state = partial(compare_state, strict_keys=True)
assert_state_bytes_equal = partial(compare_state, compare=assert_array_bytes_equal, strict_keys=True)


def assert_spikes_equal(actual, expected):
    """Compare ordered spike fields without coercing their arrays."""
    for name in ("trials", "neurons", "steps"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def assert_recorded_spikes_equal(actual, expected, recording):
    """Compare recorded result spikes, or require empty disabled recording."""
    if recording:
        assert_spikes_equal(actual.spikes, expected.spikes)
    else:
        assert actual.spikes.steps.size == 0


class CompiledRngComparison:
    """Compare native observations with an independent compiled CPU result."""

    def __init__(self, reference, target, seed):
        self.target = target
        if target == 'cpp':
            self.spikes, self.expected, self.final = reference
            self.start = np.random.RandomState(seed).get_state()
        else:
            self.start, self.spikes, self.expected, self.final, error = reference
            assert error is None

    def check(self, result, state, prepared, host_state, snapshots, assert_snapshot):
        np.testing.assert_array_equal(result.spikes.neurons, self.spikes[0])
        np.testing.assert_array_equal(result.spikes.steps, self.spikes[1])
        for name, value in host_state(state, prepared).items():
            np.testing.assert_array_equal(value, self.expected[name], err_msg=name)
        checkpoint = snapshots(state.random)[0]
        if self.target == 'cython':
            assert_snapshot(checkpoint, self.final)
        else:
            future = np.random.RandomState()
            future.set_state(checkpoint)
            np.testing.assert_array_equal(
                future.randint(0, 2**32, 624, dtype=np.uint32), self.final[0])
            assert checkpoint[3:] == self.final[1:]
