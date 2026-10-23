"""Host-side preparation for the fixed float32 LIF model."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

_I32_MIN = -(2**31)
_I32_MAX = 2**31 - 1
_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1
_F32_MAX = float(np.finfo(np.float32).max)


def _integer(value: int, name: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its supported integer range")
    return int(result)


def _integers(
    values: ArrayLike,
    name: str,
    minimum: int,
    maximum: int,
    *,
    allow_float: bool = False,
) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not array.size:
        return np.empty(0, dtype=np.int64)
    if array.dtype.kind == "O":
        exact = [_integer(value, name, minimum, maximum) for value in array]
        return np.array(exact, dtype=np.int64)
    if array.dtype.kind == "f" and allow_float:
        if not np.isfinite(array).all() or not (array == np.trunc(array)).all():
            raise ValueError(f"{name} must contain finite exact integers, without rounding")
    elif array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers without float conversion")
    if int(array.min()) < minimum or int(array.max()) > maximum:
        raise ValueError(f"{name} is outside its supported integer range")
    return np.array(array, dtype=np.int64, copy=True)


def _incoming_bound(
    destinations: NDArray[np.int64], contacts: NDArray[np.int64], n_neurons: int
) -> int:
    incoming = np.zeros(n_neurons, dtype=np.int64)
    # E <= INT32_MAX and |contact| <= 2**31 keep this entire sum within int64.
    np.add.at(incoming, destinations, np.abs(contacts))
    maximum = int(incoming.max(initial=0))
    if maximum > _I32_MAX:
        raise ValueError("incoming absolute contact sum exceeds safe int32 accumulation")
    return maximum


@dataclass(frozen=True)
class Model:
    dt_s: float = 0.0001
    v_rest_v: float = -0.052
    v_reset_v: float = -0.052
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
                raise ValueError(f"{name} must be a real scalar")
            value = float(value)
            if not math.isfinite(value) or abs(value) > _F32_MAX:
                raise ValueError(f"{name} must be finite and fit float32")
            object.__setattr__(self, name, value)
        for name, minimum in (("refractory_steps", 0), ("delay_steps", 1)):
            value = _integer(getattr(self, name), name, minimum, _I32_MAX)
            object.__setattr__(self, name, value)
        if min(self.dt_s, self.tau_m_s, self.tau_g_s) <= 0:
            raise ValueError("dt_s and both time constants must be positive")
        if self.tau_m_s == self.tau_g_s:
            raise ValueError("the fixed model requires distinct time constants")
        if self.threshold_v <= max(self.v_rest_v, self.v_reset_v):
            raise ValueError("threshold_v must exceed the rest and reset potentials")
        if self.weight_v < 0 or self.input_v < 0:
            raise ValueError("weight_v and input_v must be nonnegative")
        self.coefficients()

    def coefficients(self) -> NDArray[np.float32]:
        decay_v = math.exp(-self.dt_s / self.tau_m_s)
        decay_g = math.exp(-self.dt_s / self.tau_g_s)
        couple = self.tau_g_s / (self.tau_m_s - self.tau_g_s) * (decay_v - decay_g)
        values = (self.v_rest_v * (1 - decay_v), couple, decay_v, decay_g)
        if any(not math.isfinite(value) or abs(value) > _F32_MAX for value in values):
            raise ValueError("model coefficients must be finite and fit float32")
        return np.array(values, dtype=np.float32)


@dataclass(frozen=True, eq=False)
class Graph:
    neuron_ids: NDArray[np.int64]
    row_ptr: NDArray[np.int32]
    destinations: NDArray[np.int32]
    contacts: NDArray[np.int32]
    max_incoming_abs: int = field(init=False)

    def __post_init__(self) -> None:
        ids = _integers(self.neuron_ids, "neuron_ids", _I64_MIN, _I64_MAX)
        n_neurons = _integer(ids.size, "neuron count", 0, _I32_MAX)
        if np.unique(ids).size != n_neurons:
            raise ValueError("neuron_ids must be unique")
        row_ptr = _integers(self.row_ptr, "row_ptr", 0, _I32_MAX)
        destinations = _integers(self.destinations, "destinations", 0, _I32_MAX)
        contacts = _integers(
            self.contacts, "contacts", _I32_MIN, _I32_MAX, allow_float=True
        )
        n_edges = _integer(destinations.size, "edge count", 0, _I32_MAX)
        if contacts.size != n_edges:
            raise ValueError("destinations and contacts must have equal lengths")
        if row_ptr.size != n_neurons + 1:
            raise ValueError("row_ptr must have shape (n_neurons + 1,)")
        if row_ptr[0] != 0 or row_ptr[-1] != n_edges or (np.diff(row_ptr) < 0).any():
            raise ValueError("row_ptr must be nondecreasing from zero to the edge count")
        if (destinations >= n_neurons).any():
            raise ValueError("destinations must be neuron indices in [0, n_neurons)")
        bound = _incoming_bound(destinations, contacts, n_neurons)
        for name, array, dtype in (
            ("neuron_ids", ids, np.int64),
            ("row_ptr", row_ptr, np.int32),
            ("destinations", destinations, np.int32),
            ("contacts", contacts, np.int32),
        ):
            owned = np.array(array, dtype=dtype, copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)
        object.__setattr__(self, "max_incoming_abs", bound)

    @property
    def n_neurons(self) -> int:
        return int(self.neuron_ids.size)


def make_graph(
    neuron_ids: ArrayLike, pre: ArrayLike, post: ArrayLike, contacts: ArrayLike
) -> Graph:
    """Build source-major CSR from neuron indices, preserving parallel edge order."""
    ids = _integers(neuron_ids, "neuron_ids", _I64_MIN, _I64_MAX)
    n_neurons = _integer(ids.size, "neuron count", 0, _I32_MAX)
    sources = _integers(pre, "pre", 0, _I32_MAX)
    destinations = _integers(post, "post", 0, _I32_MAX)
    counts = _integers(contacts, "contacts", _I32_MIN, _I32_MAX, allow_float=True)
    n_edges = _integer(sources.size, "edge count", 0, _I32_MAX)
    if destinations.size != n_edges or counts.size != n_edges:
        raise ValueError("pre, post and contacts must have equal lengths")
    if (sources >= n_neurons).any() or (destinations >= n_neurons).any():
        raise ValueError("pre and post must be neuron indices in [0, n_neurons)")
    order = np.argsort(sources, kind="stable")
    row_ptr = np.zeros(n_neurons + 1, dtype=np.int32)
    row_ptr[1:] = np.cumsum(np.bincount(sources, minlength=n_neurons), dtype=np.int64)
    return Graph(ids, row_ptr, destinations[order], counts[order])


@dataclass(frozen=True, eq=False)
class Drive:
    targets: NDArray[np.int32]
    rates_hz: NDArray[np.float64]
    seed: int = 0

    def __post_init__(self) -> None:
        targets = _integers(self.targets, "targets", 0, _I32_MAX)
        _integer(targets.size, "channel count", 0, _I32_MAX)
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
        seed = _integer(self.seed, "seed", 0)
        targets = np.array(targets, dtype=np.int32, copy=True)
        targets.setflags(write=False)
        rates.setflags(write=False)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "rates_hz", rates)
        object.__setattr__(self, "seed", seed)
