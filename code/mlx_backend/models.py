from dataclasses import dataclass
from typing import NewType, Protocol

import numpy as np
from numpy.typing import NDArray

FlyWireNeuronId = NewType('FlyWireNeuronId', int)
NeuronIdentifier = int | float | str
NeuronRow = NewType('NeuronRow', int)
TrialId = NewType('TrialId', int)
FlyWireIds64 = NewType('FlyWireIds64', NDArray[np.int64])
NeuronRows32 = NewType('NeuronRows32', NDArray[np.int32])
NeuronRows64 = NewType('NeuronRows64', NDArray[np.int64])
EdgeRows32 = NewType('EdgeRows32', NDArray[np.int32])
PaddedEdgeRows32 = NewType('PaddedEdgeRows32', NDArray[np.int32])
TrialIds64 = NewType('TrialIds64', NDArray[np.int64])


@dataclass(frozen=True)
class InputPin:
    completeness_sha256: str
    connectivity_sha256: str
    neurons: int
    edges: int
    scale_mv: float = 0.275


@dataclass(frozen=True)
class Connectome:
    neuron_ids: NDArray[np.generic]
    sources: NeuronRows32
    destinations: NeuronRows32
    counts: NDArray[np.int32] | NDArray[np.int64] | NDArray[np.uint64] | NDArray[np.float32] | NDArray[np.float64]
    weights_mv: NDArray[np.float64]
    pin: InputPin | None = None
    coefficient_dtype: str | None = None


@dataclass(frozen=True)
class DestinationGrouping:
    edge_ids: EdgeRows32
    inverse: NDArray[np.int32]
    offsets: NDArray[np.int64]


@dataclass(frozen=True)
class DestinationBucket:
    targets: NeuronRows32
    edge_ids: PaddedEdgeRows32
    counts: NDArray[np.float32]
    occupied: NDArray[np.bool_]


ExperimentName = str


@dataclass(frozen=True)
class Experiment:
    name: ExperimentName
    code: int
    generator_code: int
    activated_ids: tuple[NeuronIdentifier, ...]
    rates_hz: tuple[float, ...]
    silenced_ids: tuple[NeuronIdentifier, ...] = ()
    first_class_channels: int | None = None

    @property
    def activation_classes(self) -> tuple[int, ...]:
        channels = len(self.activated_ids)
        first = (
            channels if self.first_class_channels is None else self.first_class_channels
        )
        if not 0 <= first <= channels:
            raise ValueError('Activation class boundary must be within the channels')
        return (1,) * first + (2,) * (channels - first)


@dataclass(frozen=True)
class Stimulus:
    events: "NDArray[np.uint8] | EventSource"
    targets: tuple[NeuronRow, ...]
    rates_hz: tuple[float, ...]
    trial_indices: tuple[TrialId, ...]
    seed: int
    generator_code: int
    sha256: str | None
    dt_ms: float = 0.1
    amplitudes_mv: tuple[float, ...] | None = None


class EventSource(Protocol):
    """A repeatable event array that can be read without materializing it all."""

    dtype: np.dtype
    ndim: int
    shape: tuple[int, int, int]

    def chunk(self, begin: int, end: int) -> NDArray[np.uint8]: ...


@dataclass(frozen=True)
class SpikeEvents:
    trials: TrialIds64
    neurons: NeuronRows64
    steps: NDArray[np.int64]


@dataclass(frozen=True)
class SimulationRun:
    spikes: SpikeEvents
    timings: dict[str, float]
    peak_device_bytes: int
    device_name: str
    input_sha256: tuple[str, ...] | None = None
    dt_s: float = 0.0001
