import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.typing import NDArray

from .models import Connectome, InputPin, NeuronRows32


def coefficient_weights_si(connectome: Connectome, scalar) -> NDArray[np.float64]:
    """Match the original DataFrame/quantity multiply before Brian assignment.

    NumPy 1.26 keeps narrow floating coefficient multiplication in its dtype
    when its scalar fits. Widening before this operation changes the weights.
    Retain original integer values as well, including uint64 beyond 2**53.
    Hand-built graphs without source-column metadata keep their binary64 API.
    """
    return _coefficient_product(connectome.counts, connectome.coefficient_dtype, scalar)


def _coefficient_product(counts, source_dtype, scalar):
    dtype = (np.dtype(np.float64) if source_dtype is None else
             np.result_type(np.empty(0, dtype=source_dtype), scalar))
    values = counts.astype(dtype, copy=False)
    return np.multiply(values, np.asarray(scalar, dtype=dtype)).astype(np.float64, copy=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_connectome(
    completeness: Path,
    connectivity: Path,
    pin: InputPin | None = None,
    *,
    scale_mv: float = 0.275,
) -> Connectome:
    """Load the reference's index-based graph, preserving its external IDs.

    A supplied pin verifies an expected dataset. Otherwise the actual files
    establish an immutable identity; accepted production inputs are not a
    fixed whitelist. Only columns consumed by the Brian2 model are required.
    """
    completeness, connectivity = Path(completeness), Path(connectivity)
    hashes = (file_sha256(completeness), file_sha256(connectivity))
    if pin is not None:
        if hashes != (pin.completeness_sha256, pin.connectivity_sha256):
            raise ValueError('Pinned input hash differs')
        scale_mv = pin.scale_mv
    if not np.isfinite(scale_mv):
        raise ValueError('Synaptic scale must be finite')
    neuron_ids = pd.read_csv(completeness, index_col=0).index.to_numpy(copy=True)
    if neuron_ids.size < 1:
        raise ValueError('The connectome must contain neurons')
    columns = [
        'Presynaptic_Index',
        'Postsynaptic_Index',
        'Excitatory x Connectivity',
    ]
    if not set(columns) <= set(pq.read_schema(connectivity).names):
        raise ValueError('Connectivity is missing reference model columns')
    table = pq.read_table(connectivity, columns=columns)
    if table.num_rows < 1:
        raise ValueError('The reference requires at least one connection row')
    if pin is not None and (neuron_ids.size, table.num_rows) != (pin.neurons, pin.edges):
        raise ValueError('Connectome dimensions differ from the pinned input')
    limit = np.iinfo(np.int32).max
    if max(neuron_ids.size, table.num_rows) > limit:
        raise ValueError('Connectome identity arrays must fit int32 indexing')

    def integer_column(name: str) -> NDArray[np.int64]:
        column = table[name]
        values = column.to_numpy()
        if column.null_count or values.dtype.kind not in 'iu':
            raise ValueError(f'Connectivity indices must be non-null integers: {name}')
        if np.any(values < 0) or np.any(values >= neuron_ids.size):
            raise ValueError('Connection index is outside the neuron ordering')
        return np.asarray(values, dtype=np.int64)

    source_indices = integer_column('Presynaptic_Index')
    destination_indices = integer_column('Postsynaptic_Index')
    signed = table['Excitatory x Connectivity']
    values = signed.to_numpy()
    # pandas.read_parquet (used by the reference) exposes nullable numeric
    # coefficients as NaN. Brian accepts these, as well as infinities, and
    # propagates them through synaptic delivery; do not narrow that domain.
    if values.dtype.kind not in 'buif':
        raise ValueError('Signed connection coefficients must be real numbers')
    counts = values
    # Preserve fractional coefficients for general execution. The existing
    # integer-count fast path is applicable only when this conversion is exact.
    numeric = counts.astype(np.float64)
    negative_zero = counts.dtype.kind == 'f' and np.any((counts == 0) & np.signbit(counts))
    narrow_float = values.dtype.kind == 'f' and values.dtype.itemsize < 8
    if (not narrow_float and not negative_zero
            and np.all(np.abs(numeric) <= limit) and np.all(numeric == np.trunc(numeric))):
        counts = counts.astype(np.int32)
    identity = pin or InputPin(*hashes, neuron_ids.size, table.num_rows, scale_mv)
    def immutable(array):
        # Pin the actual consumed bytes; a readonly owning ndarray can be
        # made writable again and would invalidate a retained layout proof.
        return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)

    connectome = Connectome(
        neuron_ids,
        NeuronRows32(immutable(source_indices.astype(np.int32))),
        NeuronRows32(immutable(destination_indices.astype(np.int32))),
        immutable(counts),
        immutable(_coefficient_product(counts, values.dtype.str, scale_mv*.001)*1000
                  if narrow_float else counts.astype(np.float64) * scale_mv),
        identity,
        values.dtype.str,
    )
    for array in (
        connectome.neuron_ids,
        connectome.sources,
        connectome.destinations,
        connectome.counts,
        connectome.weights_mv,
    ):
        array.setflags(write=False)
    return connectome
