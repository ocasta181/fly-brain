"""Registered integer8 ufunc loops against actual original Eon NumPy."""
import pytest
from trigonometry_checks import check_original_small_integer_loop




@pytest.mark.metal
@pytest.mark.parametrize("function",["sin","cos","tan"])
def test_registered_int8_loop_matches_actual_eon(tmp_path,function):
    check_original_small_integer_loop(tmp_path,"int8",function)


@pytest.mark.metal
@pytest.mark.parametrize("integer_dtype",["uint8","uint16"])
@pytest.mark.parametrize("function",["sin","cos","tan"])
def test_registered_unsigned_loop_matches_actual_eon(tmp_path,integer_dtype,function):
    check_original_small_integer_loop(tmp_path,integer_dtype,function)
