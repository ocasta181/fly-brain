"""Pinned numeric constants shared by the fixed-model window and test oracle."""

import math

import numpy as np

DT_MS = 0.1


DELAY_STEPS = 18

INITIAL_LAST_SPIKE = -100000000


_a = math.exp(-DT_MS / 20)


COEFFICIENTS = tuple(
    float(np.float32(value))
    for value in (
        _a,
        math.exp(-DT_MS / 5),
        _a * -math.expm1(-DT_MS * (1 / 5 - 1 / 20)) / 3,
    )
)


VOLTAGE_INCREMENT = float(np.float32(math.expm1(-DT_MS / 20)))


SYNAPTIC_INCREMENT = float(np.float32(math.expm1(-DT_MS / 5)))
