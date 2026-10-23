"""Actual Brian forward-trig composition with the owned operator policy."""
from trigonometry_checks import check_original_half_operators
import pytest


@pytest.mark.metal
@pytest.mark.parametrize("expression",[
    "sin(sin(phase_ops)**2.)",
    "sin(sin(phase_ops)%.3)",
    "sin(sin(phase_ops)//.3)",
    "sin(sin(phase_ops)**(t/ms+exponent_offset))",
    "sin((sin(phase_ops)+promote32)**.3)",
    "sin((sin(phase_ops)+g/mV+1.)**.3)",
])
def test_original_numpy_half_operators_and_promoted_widths(tmp_path,expression):
    check_original_half_operators(tmp_path,expression)
