"""Immutable host preparation for Eon's fixed binary64 LIF model."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

_I32_MAX = 2**31 - 1
_U32_MAX = 2**32 - 1
_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1


def integer(
    value: object, name: str, minimum: int = 0, maximum: int = _I32_MAX
) -> int:
    """Validate an exact integer without accepting booleans or rounding."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be from {minimum} to {maximum}")
    return int(result)


def _integers(
    values: ArrayLike, name: str, minimum: int, maximum: int
) -> NDArray[np.int64]:
    try:
        array = values if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain exact integers") from error
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not array.size:
        return np.empty(0, dtype=np.int64)
    if array.dtype.kind == "O":
        return np.array(
            [integer(value, name, minimum, maximum) for value in array],
            dtype=np.int64,
        )
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers without float conversion")
    if int(array.min()) < minimum or int(array.max()) > maximum:
        raise ValueError(f"{name} must be from {minimum} to {maximum}")
    return array.astype(np.int64, copy=False)


def _snapshot(array: np.ndarray, dtype: DTypeLike) -> np.ndarray:
    # Immutable bytes prevent setflags(write=True) from exposing mutable storage.
    data = np.ascontiguousarray(array, dtype=dtype).tobytes()
    return np.frombuffer(data, dtype=dtype)


def _pointers(indices: NDArray[np.int64], n_neurons: int) -> NDArray[np.uint32]:
    pointers = np.zeros(n_neurons + 1, dtype=np.uint32)
    pointers[1:] = np.cumsum(np.bincount(indices, minlength=n_neurons), dtype=np.int64)
    return pointers


@dataclass(frozen=True)
class Model:
    dt_s: float = 0.0001
    v_rest_v: float = -52 * 0.001
    v_reset_v: float = -52 * 0.001
    threshold_v: float = -0.045
    tau_m_s: float = 0.020
    tau_g_s: float = 0.005
    refractory_steps: int = 22
    delay_steps: int = 18
    weight_v: float = 0.000275
    input_v: float = 0.06875

    def __post_init__(self) -> None:
        for name in (
            "dt_s", "v_rest_v", "v_reset_v", "threshold_v", "tau_m_s", "tau_g_s",
            "weight_v", "input_v",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite real scalar")
            try:
                value = float(value)
            except (ValueError, OverflowError) as error:
                raise ValueError(f"{name} must be finite") from error
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name, minimum in (("refractory_steps", 0), ("delay_steps", 1)):
            object.__setattr__(self, name, integer(getattr(self, name), name, minimum))
        if min(self.dt_s, self.tau_m_s, self.tau_g_s) <= 0:
            raise ValueError("dt_s and both time constants must be positive")
        if self.tau_m_s == self.tau_g_s:
            raise ValueError("the fixed model requires distinct time constants")
        if self.threshold_v <= max(self.v_rest_v, self.v_reset_v):
            raise ValueError("threshold_v must exceed the rest and reset potentials")
        if self.weight_v < 0 or self.input_v < 0:
            raise ValueError("weight_v and input_v must be nonnegative")
        self.parameters()

    def parameters(self) -> NDArray[np.uint64]:
        """Return the eleven raw binary64 words consumed by the native kernel."""
        dt, tau, tm = map(np.float64, (self.dt_s, self.tau_g_s, self.tau_m_s))
        try:
            with np.errstate(all="ignore"):
                tau_reciprocal = np.float64(1.0) / tau
                tm_reciprocal = np.float64(1.0) / tm
                decay_g = np.float64(math.exp(float(-dt * tau_reciprocal)))
                decay_v = np.float64(math.exp(float(-dt * tm_reciprocal)))
                grow_g = np.float64(math.exp(float(dt * tau_reciprocal)))
                grow_v = np.float64(math.exp(float(dt * tm_reciprocal)))
                # Preserve the qualified CPU's separate host rounding points.
                coupling = (decay_g * (tau / (tm - tau))) * (grow_g - grow_v)
                values = np.array([
                    coupling, decay_v, decay_g, self.v_rest_v, self.v_reset_v,
                    self.threshold_v, self.input_v, dt, np.float64(0.001) * dt,
                    np.float64(1.0) / dt, np.float64(self.refractory_steps) * dt,
                ], dtype=np.float64)
        except OverflowError as error:
            raise ValueError("model preparation produced a nonfinite coefficient") from error
        if not np.isfinite(values).all():
            raise ValueError("model preparation produced a nonfinite coefficient")
        return _snapshot(values.view(np.uint64), np.uint64)


@dataclass(frozen=True, eq=False)
class Graph:
    """Incoming CSR, with outgoing entries referring to its canonical edge IDs."""

    neuron_ids: NDArray[np.int64]
    edge_ptr: NDArray[np.uint32]
    edge_pre: NDArray[np.uint32]
    contacts: NDArray[np.int64]
    out_ptr: NDArray[np.uint32] = field(init=False)
    out_edge: NDArray[np.uint32] = field(init=False)
    edge_post: NDArray[np.uint32] = field(init=False)

    def __post_init__(self) -> None:
        ids = _integers(self.neuron_ids, "neuron_ids", _I64_MIN, _I64_MAX)
        n_neurons = integer(ids.size, "neuron count")
        if np.unique(ids).size != n_neurons:
            raise ValueError("neuron_ids must be unique")
        pointers = _integers(self.edge_ptr, "edge_ptr", 0, _U32_MAX)
        sources = _integers(self.edge_pre, "edge_pre", 0, _U32_MAX)
        contacts = _integers(self.contacts, "contacts", _I64_MIN, _I64_MAX)
        n_edges = integer(sources.size, "edge count", maximum=_U32_MAX)
        if contacts.size != n_edges:
            raise ValueError("edge_pre and contacts must have equal lengths")
        if pointers.size != n_neurons + 1:
            raise ValueError("edge_ptr must have shape (n_neurons + 1,)")
        if pointers[0] != 0 or pointers[-1] != n_edges or (np.diff(pointers) < 0).any():
            raise ValueError("edge_ptr must be nondecreasing from zero to the edge count")
        if (sources >= n_neurons).any():
            raise ValueError("edge_pre must contain neuron indices in [0, n_neurons)")
        destinations = np.repeat(
            np.arange(n_neurons, dtype=np.uint32), np.diff(pointers)
        )
        if ((destinations[1:] == destinations[:-1]) & (sources[1:] < sources[:-1])).any():
            raise ValueError("incoming edges must be ordered by source within each neuron")
        outgoing = np.argsort(sources, kind="stable")
        for name, array, dtype in (
            ("neuron_ids", ids, np.int64),
            ("edge_ptr", pointers, np.uint32),
            ("edge_pre", sources, np.uint32),
            ("contacts", contacts, np.int64),
            ("out_ptr", _pointers(sources, n_neurons), np.uint32),
            ("out_edge", outgoing, np.uint32),
            ("edge_post", destinations, np.uint32),
        ):
            object.__setattr__(self, name, _snapshot(array, dtype))

    @property
    def n_neurons(self) -> int:
        return int(self.neuron_ids.size)

    @property
    def n_edges(self) -> int:
        return int(self.edge_pre.size)


def make_graph(
    neuron_ids: ArrayLike, pre: ArrayLike, post: ArrayLike, contacts: ArrayLike
) -> Graph:
    """Order incoming edges by destination, source, then original edge position."""
    ids = _integers(neuron_ids, "neuron_ids", _I64_MIN, _I64_MAX)
    n_neurons = integer(ids.size, "neuron count")
    sources = _integers(pre, "pre", 0, _U32_MAX)
    destinations = _integers(post, "post", 0, _U32_MAX)
    counts = _integers(contacts, "contacts", _I64_MIN, _I64_MAX)
    n_edges = integer(sources.size, "edge count", maximum=_U32_MAX)
    if destinations.size != n_edges or counts.size != n_edges:
        raise ValueError("pre, post and contacts must have equal lengths")
    if (sources >= n_neurons).any() or (destinations >= n_neurons).any():
        raise ValueError("pre and post must contain neuron indices in [0, n_neurons)")
    order = np.lexsort((sources, destinations))
    return Graph(ids, _pointers(destinations, n_neurons), sources[order], counts[order])


@dataclass(frozen=True, eq=False)
class Drive:
    """Ordered input occurrences, including duplicate target neurons."""

    targets: NDArray[np.int32]
    rates_hz: NDArray[np.float64]
    seed: int = 0

    def __post_init__(self) -> None:
        targets = _integers(self.targets, "targets", 0, _I32_MAX)
        integer(targets.size, "channel count")
        try:
            raw_rates = np.asarray(self.rates_hz)
            if np.iscomplexobj(raw_rates):
                raise ValueError("rates_hz must be real")
            rates = np.array(raw_rates, dtype=np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("rates_hz must contain finite nonnegative rates") from error
        if rates.ndim != 1 or rates.size != targets.size:
            raise ValueError("rates_hz must be one-dimensional and match the targets length")
        if not np.isfinite(rates).all() or (rates < 0).any():
            raise ValueError("rates_hz must contain finite nonnegative rates")
        seed = integer(self.seed, "seed", maximum=2**64 - 1)
        object.__setattr__(self, "targets", _snapshot(targets, np.int32))
        object.__setattr__(self, "rates_hz", _snapshot(rates, np.float64))
        object.__setattr__(self, "seed", seed)
