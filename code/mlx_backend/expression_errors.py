"""Typed native expression faults at the original Brian replay boundary."""


INTEGER_ERRORS={
    ('ZeroDivisionError','integer division or modulo by zero'):16,
    ('OverflowError','value too large to perform division'):32,
    # The existing IR frontend catalogs both exception/message PHIs before
    # their control-flow correlation selects the reachable pair.
    ('ZeroDivisionError','value too large to perform division'):64,
    ('OverflowError','integer division or modulo by zero'):128,
}

COMPLEX_UFUNC_ERRORS={name:(512<<index,('TypeError',
    f"ufunc '{name}' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''"))
    for index,name in enumerate(('floor','ceil'))}

PYTHON_COMPLEX_ERRORS={
    1:(2048,('ZeroDivisionError','complex division by zero')),
    2:(4096,('ZeroDivisionError','0.0 to a negative or complex power')),
    3:(8192,('OverflowError','complex exponentiation')),
}


def exception(kind,metadata=None):
    if kind & 256:
        if metadata is None:return IndexError("TimedArray column index is out of bounds")
        column,width=map(int,metadata)
        return IndexError(f"index {column} is out of bounds for axis 1 with size {width}")
    for (name,message),code in INTEGER_ERRORS.items():
        if kind & code:
            return {'ZeroDivisionError':ZeroDivisionError,'OverflowError':OverflowError}[name](message)
    for code,(_,message) in COMPLEX_UFUNC_ERRORS.values():
        if kind & code:return TypeError(message)
    for code,(name,message) in PYTHON_COMPLEX_ERRORS.values():
        if kind & code:return {'ZeroDivisionError':ZeroDivisionError,'OverflowError':OverflowError}[name](message)

    if kind & 2:
        return ZeroDivisionError('float division')
    if kind & 4:
        return ValueError('lam value too large')
    if kind & 8:
        return ValueError('lam < 0 or lam is NaN')
    return ValueError('lam < 0 or lam contains NaNs')


class ExpressionRuntimeError(ValueError):
    def __init__(self,kind):
        self.kind=int(kind)
        super().__init__(str(exception(self.kind)))
