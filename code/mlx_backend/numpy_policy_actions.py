"""Deliver native status with NumPy's own host-only ufunc error handler.

The pinned NumPy public C API declares ``PyUFunc_handlefperr`` at slot29 in
``numpy/__ufunc_api.h``.  Calling it performs policy delivery only: no NumPy
ufunc, equation, neuron timestep or numerical fallback executes on the CPU.
"""
import ctypes
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _handler():
    if np.__version__ != '1.26.4':
        raise RuntimeError('Native floating policy delivery requires NumPy1.26.4')
    pointer = ctypes.pythonapi.PyCapsule_GetPointer
    pointer.argtypes = (ctypes.py_object, ctypes.c_char_p)
    pointer.restype = ctypes.c_void_p
    api = ctypes.cast(pointer(np.core._multiarray_umath._UFUNC_API,None),
                      ctypes.POINTER(ctypes.c_void_p))
    # PYFUNCTYPE retains the GIL and propagates exceptions set by NumPy or a
    # user callback.  CFUNCTYPE would violate both requirements here.
    prototype = ctypes.PYFUNCTYPE(ctypes.c_int,ctypes.c_int,ctypes.py_object,
                                 ctypes.c_int,ctypes.POINTER(ctypes.c_int))
    return prototype(api[29])


def deliver(name, status):
    """Apply the actual current NumPy policy to a native ufunc status word."""
    status = int(status)
    if not status:
        return
    if status & ~15 or not isinstance(name,str):
        raise ValueError('Invalid native NumPy ufunc floating status')
    policy = np.geterr()
    actions = {'ignore':0,'warn':1,'raise':2,'call':3,'print':4,'log':5}
    mask = sum(actions[policy[kind]] << shift for kind,shift in
               (('divide',0),('over',3),('under',6),('invalid',9)))
    first = ctypes.c_int(1)
    result = _handler()(mask,(name.encode(),np.geterrcall()),status,ctypes.byref(first))
    if result:
        raise RuntimeError('NumPy floating error handler returned without its exception')
