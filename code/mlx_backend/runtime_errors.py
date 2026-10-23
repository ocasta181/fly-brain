"""Decode native model failures without another device buffer."""
import builtins


ERROR_KIND_MASK = 0xffff
NUMPY_ERROR_STEP_LIMIT = 1 << 45
_EXCEPTIONS = {name: getattr(builtins, name) for name in
               ('IndexError', 'ValueError', 'TypeError', 'OverflowError', 'ZeroDivisionError', 'FloatingPointError')}


def model_exception(catalog, kind):
    """Kind zero retains the existing NumPy TimedArray indexing failure."""
    if kind == 0:
        error = IndexError('TimedArray column index is out of bounds')
    else:
        if not 0 < kind <= len(catalog):
            raise RuntimeError(f'Invalid native model error kind {kind}')
        name, message = catalog[kind - 1]
        from .compiled_precision import ERROR_NAME,PrecisionRetry
        if name=='NumpyFloatingStatus':
            from .numpy_error_policy import NumpyFloatingStatus
            error=NumpyFloatingStatus(message)
        else:
            error = PrecisionRetry(message) if name==ERROR_NAME else _EXCEPTIONS[name](message)
    error._mlx_runtime_error_kind = kind
    return error


def is_model_exception(error):
    return hasattr(error, '_mlx_runtime_error_kind')


def first_error(release, order='row'):
    """Select Cython's first row or NumPy's first time/phase/operation."""
    import numpy as np
    values = np.asarray(release)
    rows = np.flatnonzero(values < 0)
    if not rows.size:
        raise ValueError('No native model failure is present')
    row = int(rows[np.argmin(values[rows])] if order == 'operation' else rows[0])
    return row, int(values[row]) & ERROR_KIND_MASK
