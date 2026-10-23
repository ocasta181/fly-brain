"""Actual original-Eon error callbacks observe the committed native prefix."""
from native_execution import forbid_cpu_code_objects, require_metal
import numpy as np
import pytest
from numpy_callback import RestoreCallback

b = pytest.importorskip('brian2')
mx = require_metal('Metal required', allow_module_level=True)

from mlx_backend.brian_device import original_api
from mlx_backend.model import DEFAULT_EQUATIONS, DEFAULT_RESET
from original_api_reference import build_original_api_fixture, cpu_runtime, original_runtime, assert_snapshot_equal

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('location,action', [('groups','call'), ('resets','call'),
                                           ('resets','log-error'), ('resets','stop')])
def test_callback_sees_prefix_and_mutates_physical_arrays_and_rng(tmp_path, monkeypatch, location, action):
    if location == 'groups':
        equations = DEFAULT_EQUATIONS + 'x:1\ny:1\ndp/dt=(x**y+x-p)/second:1\nq:1\n'
        reset = DEFAULT_RESET + ';q=rand()'
    else:
        equations = DEFAULT_EQUATIONS + 'x:1\ny:1\np:1\nq:1\nr:1\n'
        reset = DEFAULT_RESET + ';p=x**y;q=rand();r=x'
    parameters = dict(eqs=equations, eq_th='True', eq_rst=reset,
                      t_rfc=0*b.ms, w_syn=0*b.mV)
    mutation_case = location == 'resets' and action in ('call','stop')
    if mutation_case:
        parameters['t_dly'] = .1*b.ms
    duration = .3 if mutation_case else .2
    options = dict(parameters=parameters, exc=(), exc2=(), silenced=(),
                   initial_state=dict(x=[.3,0.,3.,2.], y=[2.,-1.,2.,2.], p=10.))
    previous = np.geterrcall()
    sentinel = RuntimeError('Callback exception identity')

    def run(fixture):
        observations = []
        def callback(message, flags):
            observations.append((message, flags, fixture.capture(), np.random.random(3)))
            fixture.neurons.x = [1.,2.,9.,4.]
            fixture.neurons.p = 44.
            if location == 'resets' and action == 'call':
                fixture.synapses.w = 1*b.mV
                fixture.monitor.variables['count'].get_value()[:] += 3
            if action == 'stop':
                fixture.network.stop()
            if action == 'log-error':
                raise sentinel
        class Logger:
            def write(self, message):
                callback(message, None)
        np.seterrcall(Logger() if action == 'log-error' else callback)
        with np.errstate(all='ignore', divide='log' if action == 'log-error' else 'call'):
            if action == 'log-error':
                with pytest.raises(b.BrianObjectException) as caught:
                    fixture.run_ms(duration)
                assert caught.value.__cause__ is sentinel
                result = fixture.capture()
            elif action == 'stop':
                fixture.network.run(duration*b.ms,namespace={})
                # Brian exposes clock.t_'s VariableView after an early stop;
                # normalize only this snapshot scalar, preserving the object.
                clock_view=fixture.network.t_
                fixture.network.t_=float(np.asarray(clock_view).reshape(-1)[0])
                try:
                    result=fixture.capture()
                finally:
                    fixture.network.t_=clock_view
            else:
                result = fixture.run_ms(duration)
        return result, observations, np.random.get_state()

    with RestoreCallback(previous):
        with cpu_runtime():
            cpu = build_original_api_fixture(tmp_path/'cpu', **options)
            expected, observations, rng = run(cpu)
        with original_runtime(original_api):
            native = build_original_api_fixture(tmp_path/'native', **options)
            forbid_cpu_code_objects(monkeypatch, 'Policy callback used a CPU numerical timestep')
            actual, native_observations, native_rng = run(native)
            assert len(native_observations) == len(observations) == 1
            for actual_call, expected_call in zip(native_observations, observations):
                assert actual_call[:2] == expected_call[:2]
                assert_snapshot_equal(actual_call[2], expected_call[2])
                np.testing.assert_array_equal(actual_call[3].view(np.uint64),
                                              expected_call[3].view(np.uint64))
            assert_snapshot_equal(actual, expected)
            assert native_rng[0] == rng[0]
            np.testing.assert_array_equal(native_rng[1], rng[1])
            assert native_rng[2:] == rng[2:]


@pytest.mark.parametrize('queue_case',['delayed','zero-delay','synapses-first'])
def test_callback_exception_preserves_delayed_queue_for_resumed_run(tmp_path, monkeypatch, queue_case):
    parameters=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\np:1\nq:1\n',
        eq_th='True',eq_rst=DEFAULT_RESET+';p=x**y;q=rand()',
        t_rfc=0*b.ms,t_dly=(0. if queue_case=='zero-delay' else .2)*b.ms,w_syn=.5*b.mV)
    options=dict(parameters=parameters,exc=(),exc2=(),silenced=(),
        initial_state=dict(x=[.3,0.,3.,2.],y=[2.,-1.,2.,2.],p=10.))
    previous=np.geterrcall()
    sentinel=RuntimeError('log handler interrupted reset after synapses')
    def run(fixture):
        if queue_case=='synapses-first':
            fixture.network.schedule=['start','groups','synapses','thresholds','resets','end']
        class Logger:
            def write(self,message):
                np.random.random(3)
                raise sentinel
        np.seterrcall(Logger())
        with np.errstate(all='ignore',divide='log'):
            with pytest.raises(b.BrianObjectException) as caught:
                fixture.run_ms(.2)
        assert caught.value.__cause__ is sentinel
        failed=fixture.capture()
        with np.errstate(all='ignore'):
            resumed=fixture.run_ms(.5)
        return failed,resumed,np.random.get_state()
    with RestoreCallback(previous):
        with cpu_runtime():
            expected=run(build_original_api_fixture(tmp_path/'cpu',**options))
        with original_runtime(original_api):
            fixture=build_original_api_fixture(tmp_path/'native',**options)
            forbid_cpu_code_objects(monkeypatch, 'Delayed callback replay used CPU numerical timesteps')
            actual=run(fixture)
            assert_snapshot_equal(actual[0],expected[0])
            assert_snapshot_equal(actual[1],expected[1])
            assert actual[2][0]==expected[2][0]
            np.testing.assert_array_equal(actual[2][1],expected[2][1])
            assert actual[2][2:]==expected[2][2:]
