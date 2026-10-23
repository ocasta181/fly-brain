"""Exact immutable tables for the finite binary16 domain of NumPy ufuncs.

Preparation enumerates a builtin's 65,536 possible half inputs once. Metal
indexes the resulting raw binary64 carriers; no host function runs per step.
The four tables use at most 2 MiB and share the existing namespace buffer.
"""
from functools import lru_cache

import numpy as np


FUNCTIONS=frozenset(('exp','log','expm1','log1p'))


@lru_cache(maxsize=len(FUNCTIONS))
def values(function):
    if function not in FUNCTIONS:raise ValueError(function)
    inputs=np.arange(65536,dtype=np.uint16).view(np.float16)
    with np.errstate(all='ignore'):
        result=getattr(np,function)(inputs)
    if result.dtype!=np.dtype('float16'):
        raise RuntimeError(f'Unexpected NumPy half {function} output dtype: {result.dtype}')
    # Bytes-backed storage is immutable, including signaling-NaN payloads.
    return np.frombuffer(result.astype(np.float64).tobytes(),dtype=np.float64)
