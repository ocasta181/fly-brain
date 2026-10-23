"""Replay explicit original-API checkpoints with independent CPU/native fixtures."""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from native_execution import forbid_cpu_steps
from original_api_reference import (
    assert_rng_state_equal, assert_snapshot_equal, capture_rng_state, cpu_runtime,
)


@dataclass(frozen=True)
class RunObservation:
    snapshot: object
    error: Exception | None = None
    rng: object = None


def observe_run(fixture, duration_ms, *, raises=None, optional=False, rng=False):
    """Copy a completed run or its committed prefix, retaining the real error."""
    if raises is not None and not optional:
        with pytest.raises(raises) as caught:
            fixture.run_ms(duration_ms)
        snapshot, error = fixture.capture(), caught.value
    else:
        try:
            snapshot, error = fixture.run_ms(duration_ms), None
        except (raises or ()) as caught:
            snapshot, error = fixture.capture(), caught
    return RunObservation(snapshot, error, capture_rng_state() if rng else None)


def assert_run_equal(actual, expected):
    """Retain exact snapshot and Brian wrapper/cause checks at each checkpoint."""
    assert type(actual.error) is type(expected.error)
    if expected.error is not None:
        assert type(actual.error.__cause__) is type(expected.error.__cause__)
        assert str(actual.error.__cause__) == str(expected.error.__cause__)
    assert_snapshot_equal(actual.snapshot, expected.snapshot)
    if expected.rng is not None:
        assert_rng_state_equal(actual.rng, expected.rng)


def check_original_api_runs(build, exercise, monkeypatch, *, guard=None, **runtime):
    """Compare each yielded observation before continuing the native exercise.

    ``build(mode)`` creates real Brian fixtures on the selected device. The
    ordinary ``exercise(fixture, mode)`` generator owns durations, mutations and local
    assertions. Preferences are scoped by cpu_runtime; cache overrides remain
    with the caller. No native arithmetic participates in the CPU reference.
    """
    expected = []
    for mode in ("cpu", "native"):
        with ExitStack() as stack:
            stack.enter_context(cpu_runtime(**runtime))
            if mode == "native":
                from mlx_backend.brian_device import original_api

                stack.enter_context(original_api(workers=False))
            fixture = build(mode)
            if mode == "native":
                if guard is None:
                    forbid_cpu_steps(fixture.network, monkeypatch)
                else:
                    guard(fixture, monkeypatch)
            count = 0
            for count, observation in enumerate(exercise(fixture, mode), 1):
                if mode == "cpu":
                    expected.append(observation)
                else:
                    assert count <= len(expected), "extra native checkpoint"
                    assert_run_equal(observation, expected[count - 1])
            if mode == "native":
                assert count == len(expected), "missing native checkpoint"


class SingleRunComparison:
    """Run independent real fixtures once, retaining local native assertions."""

    def __init__(self, build, *, duration_ms=.1, **runtime):
        self.build, self.duration_ms, self.runtime = build, duration_ms, dict(runtime)
        with cpu_runtime(**self.runtime):
            self.cpu = build("cpu")
            self.expected = self.cpu.run_ms(duration_ms)

    @contextmanager
    def native(self, *, guard=None):
        from mlx_backend.brian_device import original_api

        with cpu_runtime(**self.runtime):
            with original_api(workers=False) as device:
                fixture = self.build("native")
                if guard is not None:
                    guard(fixture)
                observed = fixture.run_ms(self.duration_ms)
                yield fixture, device
                assert_snapshot_equal(observed, self.expected)

    def check_native(self, *, guard=None):
        with self.native(guard=guard):
            pass


def set_floating_operator_cache(monkeypatch, preferences):
    monkeypatch.setitem(preferences, "codegen.runtime.cython.cache_dir", str(
        Path(__file__).resolve().parents[1] / "executions/cython-floating-operators"))
