"""CPU reference fixtures using the frozen application's actual Brian API.

This module imports no MLX backend code.  ``build_original_api_fixture`` also
works with an explicitly selected device, so the same original-object contract
can be compared with the isolated CPU runtime supplied by ``cpu_runtime``.
Every snapshot copies Brian's real SI storage, including the original monitor.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import ctypes
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from state_assertions import assert_array_bytes_equal


ORIGINAL_MODEL_PATH = (Path(__file__).resolve().parents[1]
                       / "code/paper-phil-drosophila/model.py")
NEURON_IDS = ("root-A", "root-720575940000000001", "root-C", "root-D")
SOURCES = (2, 0, 0, 1, 0, 3, 2)
DESTINATIONS = (3, 1, 1, 2, 3, 1, 1)
COUNTS = (1.25, 30.5, -4.75, -13.25, 2.375, -.625, 8.125)


def load_original_model():
    """Load the unchanged paper module, without importing a backend facade."""
    spec = importlib.util.spec_from_file_location("eon_original_api_reference", ORIGINAL_MODEL_PATH)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    return original


def write_tiny_dataset(directory: Path) -> tuple[Path, Path]:
    """Write raw, unordered, signed, fractional and duplicate application edges."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    completeness = directory / "original-api-neurons.csv"
    connectivity = directory / "original-api-edges.parquet"
    pd.DataFrame({"fixture_label": ("A", "B", "C", "D")},
                 index=pd.Index(NEURON_IDS, name="flywire_id")).to_csv(completeness)
    pd.DataFrame({
        "Presynaptic_Index": SOURCES,
        "Postsynaptic_Index": DESTINATIONS,
        "Excitatory x Connectivity": COUNTS,
    }).to_parquet(connectivity)
    return completeness, connectivity


def synthetic_dataset(size, *, edges=((0, 1, 0.),), labelled=True,
                      neuron_file="nodes.csv"):
    """Build a writer for a synthetic original-API graph, preserving edge order."""
    def write(directory):
        directory.mkdir(parents=True, exist_ok=True)
        nodes, connectivity = directory / neuron_file, directory / "edges.parquet"
        index = np.arange(size)
        neurons = (pd.DataFrame({"label": index}, index=pd.Index(index, name="flywire_id"))
                   if labelled else pd.DataFrame(index=index))
        neurons.to_csv(nodes)
        sources, destinations, counts = zip(*edges)
        pd.DataFrame({"Presynaptic_Index": sources, "Postsynaptic_Index": destinations,
                      "Excitatory x Connectivity": counts}).to_parquet(connectivity)
        return nodes, connectivity
    return write


@contextmanager
def cpu_runtime(*, dt_ms: float = .1, seed: int = 173, target="numpy",
                float_dtype=np.float64, integer_dtype=np.int32):
    """Select Brian's exact CPU-only runtime, restoring preferences and NumPy.

    Select the registered CPU runtime explicitly: an MLXDevice subclass must
    never be mistaken for the CPU oracle by an ``isinstance`` check. Brian's
    Cython random-function namespace holds this registered runtime's buffers.
    """
    import brian2 as b
    from brian2.devices.device import reset_device, runtime_device

    preference_names = (
        "codegen.target", "core.default_float_dtype", "core.default_integer_dtype",
        "legacy.refractory_timing", "core.network.default_schedule",
    )
    preferences = {name: b.prefs[name] for name in preference_names}
    original_dt = b.defaultclock.dt
    random_state = np.random.get_state()
    saved_buffers = {}
    for kind in ("rand", "randn"):
        pointer = int(getattr(runtime_device, kind+"_buffer")[0])
        values = (np.ctypeslib.as_array(ctypes.cast(pointer, ctypes.POINTER(ctypes.c_double)),
                                       shape=(20000,)).copy() if pointer else None)
        saved_buffers[kind] = (int(getattr(runtime_device, kind+"_buffer_index")[0]), values)
    b.set_device("runtime")
    clock_state = {name: b.defaultclock.variables[name].get_value().copy()
                   for name in ("t", "timestep")}
    try:
        b.prefs.codegen.target = target
        b.prefs.core.default_float_dtype = float_dtype
        b.prefs.core.default_integer_dtype = integer_dtype
        b.prefs.legacy.refractory_timing = False
        b.prefs.core.network.default_schedule = [
            "start", "groups", "thresholds", "synapses", "resets", "end",
        ]
        b.defaultclock.dt = dt_ms * b.ms
        for name in clock_state:
            b.defaultclock.variables[name].get_value()[:] = 0
        b.seed(seed)
        yield b
    finally:
        for name, value in preferences.items():
            b.prefs[name] = value
        b.defaultclock.dt = original_dt
        for name, values in clock_state.items():
            b.defaultclock.variables[name].get_value()[:] = values
        for kind, (index, values) in saved_buffers.items():
            pointer = int(getattr(runtime_device, kind+"_buffer")[0])
            if values is not None:
                np.ctypeslib.as_array(ctypes.cast(pointer, ctypes.POINTER(ctypes.c_double)),
                                     shape=(20000,))[:] = values
            elif pointer:
                free = ctypes.CDLL(None).free
                free.argtypes, free.restype = [ctypes.c_void_p], None
                free(pointer)
                getattr(runtime_device, kind+"_buffer")[0] = 0
            getattr(runtime_device, kind+"_buffer_index")[:] = index
        reset_device()
        np.random.set_state(random_state)


@contextmanager
def original_runtime(device_context, *, native=True, **runtime):
    """Scope the CPU reference runtime and optional original-API device together."""
    with cpu_runtime(**runtime):
        with device_context(workers=False) if native else nullcontext() as device:
            yield device


@dataclass(frozen=True)
class OriginalAPISnapshot:
    """Exact copied SI values after one actual ``Network.run`` call."""

    state: dict[str, np.ndarray]
    synaptic_pre: np.ndarray
    synaptic_post: np.ndarray
    weights_v: np.ndarray
    delay_s: np.ndarray
    spike_i: np.ndarray
    spike_t_s: np.ndarray
    spike_count: np.ndarray
    spike_N: np.ndarray
    group_spikes: np.ndarray
    spike_trains_s: dict[int, np.ndarray]
    network_t_s: float
    clock_t_s: float
    clock_timestep: int
    dt_s: float


@dataclass
class OriginalAPIFixture:
    """Real objects returned by original create_model/poi/silence calls."""

    original: object
    parameters: dict
    paths: tuple[Path, Path]
    neurons: object
    synapses: object
    monitor: object
    inputs: list
    network: object
    snapshots: list[OriginalAPISnapshot] = field(default_factory=list)

    def capture(self) -> OriginalAPISnapshot:
        state_names = [name for name, equation in self.neurons.equations.items()
                       if equation.type != "subexpression"]
        state_names = tuple(dict.fromkeys([*state_names, "lastspike", "not_refractory"]))

        def values(owner, name):
            return np.array(owner.variables[name].get_value(), copy=True)

        clock = self.neurons.clock
        return OriginalAPISnapshot(
            state={name: values(self.neurons, name) for name in state_names},
            synaptic_pre=values(self.synapses, "_synaptic_pre"),
            synaptic_post=values(self.synapses, "_synaptic_post"),
            weights_v=values(self.synapses, "w"),
            delay_s=values(self.synapses.pre, "delay"),
            spike_i=(values(self.monitor, "i") if "i" in self.monitor.variables
                     else np.empty(0, np.int32)),
            spike_t_s=(values(self.monitor, "t") if "t" in self.monitor.variables
                       else np.empty(0, np.float64)),
            spike_count=values(self.monitor, "count"),
            spike_N=values(self.monitor, "N"),
            group_spikes=np.array(self.neurons.spikes, copy=True),
            spike_trains_s=({int(index): np.asarray(times).copy()
                             for index, times in self.original.get_spk_trn(self.monitor).items()}
                            if "i" in self.monitor.variables else {}),
            network_t_s=float(self.network.t_),
            clock_t_s=float(clock.variables["t"].get_value()[0]),
            clock_timestep=int(clock.variables["timestep"].get_value()[0]),
            dt_s=float(clock.dt),
        )

    def run_ms(self, duration_ms: float) -> OriginalAPISnapshot:
        """Execute the selected device through the original real Network."""
        import brian2 as b

        self.network.run(duration=duration_ms * b.ms, namespace={})
        snapshot = self.capture()
        self.snapshots.append(snapshot)
        return snapshot


def build_original_api_fixture(
    directory: Path,
    *,
    parameters: Mapping | None = None,
    exc: Sequence[int] = (0, 0),
    exc2: Sequence[int] = (2,),
    silenced: Sequence[int] = (3,),
    initial_state: Mapping[str, object] | None = None,
) -> OriginalAPIFixture:
    """Construct original objects on the caller's already-selected device.

    ``parameters`` override the frozen module's defaults. ``initial_state`` uses
    normal unit-bearing Brian assignments, rather than a backend initializer.
    The helper never selects a native device or substitutes a CPU fallback.
    """
    import brian2 as b

    original = load_original_model()
    params = {**original.default_params, **(parameters or {})}
    paths = write_tiny_dataset(directory)
    neurons, synapses, monitor = original.create_model(*paths, params)
    inputs, returned_neurons = original.poi(neurons, list(exc), list(exc2), params)
    assert returned_neurons is neurons
    assert original.silence(list(silenced), synapses) is synapses
    for name, value in (initial_state or {}).items():
        setattr(neurons, name, value)
    network = b.Network(neurons, synapses, monitor, *inputs)
    return OriginalAPIFixture(original, params, paths, neurons, synapses, monitor,
                              inputs, network)


def run_original_api(
    directory: Path,
    durations_ms: Sequence[float],
    *,
    mutations: Mapping[int, Callable[[OriginalAPIFixture], None]] | None = None,
    **fixture_kwargs,
) -> OriginalAPIFixture:
    """Run a call sequence; mutations are applied before zero-based run indices."""
    fixture = build_original_api_fixture(directory, **fixture_kwargs)
    for index, duration in enumerate(durations_ms):
        if mutations and index in mutations:
            mutations[index](fixture)
        fixture.run_ms(duration)
    return fixture


def assert_snapshot_equal(actual: OriginalAPISnapshot, expected: OriginalAPISnapshot):
    """Assert dtype and bitwise equality, including signed zero and SI times."""
    from dataclasses import fields

    for descriptor in fields(expected):
        name = descriptor.name
        actual_value, expected_value = getattr(actual, name), getattr(expected, name)
        if isinstance(expected_value, dict):
            assert actual_value.keys() == expected_value.keys(), name
            for key in expected_value:
                assert_array_bytes_equal(actual_value[key], expected_value[key], f"{name}.{key}")
        elif isinstance(expected_value, np.ndarray):
            assert_array_bytes_equal(actual_value, expected_value, name)
        elif isinstance(expected_value, float):
            assert np.float64(actual_value).tobytes() == np.float64(expected_value).tobytes(), name
        else:
            assert actual_value == expected_value, name


def capture_rng_state():
    """Copy real NumPy state and Cython buffer contents, excluding addresses."""
    from brian2.devices.device import get_device

    device = get_device()
    state = np.random.get_state()
    result = {"numpy": (state[0], state[1].copy(), *state[2:])}
    for kind in ("rand", "randn"):
        result[kind+"_index"] = int(getattr(device, kind+"_buffer_index")[0])
        pointer = int(getattr(device, kind+"_buffer")[0])
        result[kind+"_values"] = (np.ctypeslib.as_array(
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_double)), shape=(20000,),
        ).copy() if pointer else None)
    return result


def assert_rng_state_equal(actual, expected):
    assert actual["numpy"][0] == expected["numpy"][0]
    np.testing.assert_array_equal(actual["numpy"][1], expected["numpy"][1])
    assert actual["numpy"][2:] == expected["numpy"][2:]
    for kind in ("rand", "randn"):
        assert actual[kind+"_index"] == expected[kind+"_index"], kind
        if expected[kind+"_values"] is None:
            assert actual[kind+"_values"] is None, kind
        else:
            np.testing.assert_array_equal(actual[kind+"_values"], expected[kind+"_values"])
