"""The host boundary delegates policy delivery, never arithmetic, to NumPy."""
import numpy as np
import pytest
from numpy_callback import RestoreCallback

from mlx_backend.numpy_policy_actions import deliver


@pytest.mark.parametrize('action',['ignore','warn','raise','call','print','log'])
def test_native_status_uses_exact_numpy_action(action,capfd):
    import warnings
    class Log:
        def __init__(self):self.messages=[]
        def write(self,value):self.messages.append(value)
    def observe(native):
        calls=[];log=Log();previous=np.geterrcall()
        np.seterrcall(log if action=='log' else lambda message,flag:calls.append((message,flag)))
        error=None
        with RestoreCallback(previous):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                with np.errstate(all=action):
                    try:
                        if native:deliver('divide',9)
                        else:np.divide(np.array([1.,0.]),np.array([0.,0.]))
                    except Exception as value:error=(type(value),str(value))
            return (calls,log.messages,error,[str(item.message) for item in caught],capfd.readouterr().err)
    expected=observe(False)
    assert observe(True)==expected


def test_native_action_callback_exception_is_original_exception():
    error=RuntimeError('policy callback failed')
    def callback(*args):raise error
    previous=np.geterrcall();np.seterrcall(callback)
    with RestoreCallback(previous):
        with np.errstate(divide='call'):
            with pytest.raises(RuntimeError) as caught:deliver('power',1)
        assert caught.value is error


@pytest.mark.parametrize('action',['call','log'])
def test_missing_native_action_callback_matches_numpy(action):
    previous=np.geterrcall();np.seterrcall(None)
    with RestoreCallback(previous):
        with np.errstate(divide=action):
            with pytest.raises(NameError) as expected:np.divide(np.array([1.]),np.array([0.]))
            with pytest.raises(NameError) as actual:deliver('divide',1)
        assert str(actual.value)==str(expected.value)
