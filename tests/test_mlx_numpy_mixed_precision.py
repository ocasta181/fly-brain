"""Shared floating/integer precision choices versus installed NumPy loops."""
import ast
import itertools

import numpy as np

from mlx_backend.numpy_half import Renderer,Expression,half_precision


def test_mixed_floating_integer_selected_loops_match_numpy():
    integer_types=(np.bool_,np.int8,np.uint8,np.int16,np.uint16,np.int32,np.uint32,np.int64,np.uint64)
    operations=((ast.Add(),np.add),(ast.Sub(),np.subtract),(ast.Mult(),np.multiply),(ast.Div(),np.divide))
    for floating,integer,scalar_float,scalar_integer in itertools.product(
            (np.float16,np.float32,np.float64),integer_types,(False,True),(False,True)):
        float_value=floating(1) if scalar_float else np.ones(1,floating)
        integer_value=integer(1) if scalar_integer else np.ones(1,integer)
        left=Expression('_floating','float',scalar_float,floating!=np.float64,
            float_value if scalar_float else None,half=floating==np.float16)
        dtype=np.dtype(integer)
        right=Expression('_integer','bool' if dtype.kind=='b' else 'int',scalar_integer,
            value=integer_value if scalar_integer else None,integer_bits=dtype.itemsize*8,
            integer_unsigned=dtype.kind=='u')
        for operation,ufunc in operations:
            for reverse in (False,True):
                renderer=Renderer({}, {},set(),set(),'reset')
                a,z=(right,left) if reverse else (left,right)
                x,y=(integer_value,float_value) if reverse else (float_value,integer_value)
                expected=ufunc(x,y).dtype
                actual=renderer._binary(a,z,operation)
                context=(floating,integer,scalar_float,scalar_integer,type(operation).__name__,reverse)
                assert actual.narrow is (expected in (np.dtype('float16'),np.dtype('float32'))),context
                assert half_precision(actual) is (expected==np.dtype('float16')),context
