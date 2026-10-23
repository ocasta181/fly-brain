"""Input exceptions preserve actual completed Brian phases and draw prefixes."""
from native_execution import forbid_model_cpu_steps

import numpy as np
import pytest

pytestmark = pytest.mark.metal


def test_numpy_invalid_probability_preserves_failure_prefix(tmp_path, monkeypatch):
    import brian2 as b
    from original_api_reference import build_original_api_fixture, capture_rng_state
    from original_api_protocol import RunObservation, check_original_api_runs

    messages=[]
    def build(mode):
        fixture = build_original_api_fixture(tmp_path/str(mode=='native'),
            parameters={"r_poi": 10000*b.Hz, "r_poi2": 12000*b.Hz},
            exc=(0,), exc2=(2,), silenced=(),
            initial_state={"v": np.array([-44., -49., -48., -50.])*b.mV,
                           "g": np.array([3., -2., 4., 1.])*b.mV})
        return fixture

    def guard(fixture, patches):
        forbid_model_cpu_steps(fixture, patches, 'CPU model timestep executed')
    def exercise(fixture, mode):
        with pytest.raises(Exception) as caught:fixture.run_ms(.3)
        cause=caught.value.__cause__ or caught.value
        assert isinstance(cause,ValueError)
        yield RunObservation(fixture.capture(),rng=capture_rng_state())
        if mode=='cpu':messages.append(str(cause))
        else:assert str(cause)==messages[0]
    check_original_api_runs(build,exercise,monkeypatch,guard=guard,seed=731)


@pytest.mark.parametrize('rate,first,delay,record,refractory,warm', [
    (12000, False, 1.8, True, False, False),
    (-10, True, 0, True, False, False),
    (np.nan, False, .1, False, False, False),
    (np.inf, True, 1.8, True, True, False),
    (12000, False, 0, False, True, True),
    (12000, False, .2, True, False, True),
    (-10, True, 1.8, False, False, True),
])
def test_numpy_input_error_repair_queue_and_source_transitions(
        tmp_path, monkeypatch, rate, first, delay, record, refractory, warm, schedule=None):
    import brian2 as b
    from original_api_reference import build_original_api_fixture, capture_rng_state
    from original_api_protocol import RunObservation, check_original_api_runs, observe_run

    messages=[];future=[]
    def build(mode):
        fixture=build_original_api_fixture(tmp_path/str(mode=='native'),
            parameters={'r_poi':(rate if first else 3700)*b.Hz,
                        'r_poi2':(5200 if first else rate)*b.Hz,
                        't_dly':delay*b.ms},
            exc=(0,),exc2=(2,),silenced=(),
            initial_state={'v':np.array([-44.,-49.,-48.,-50.])*b.mV,
                           'g':np.array([3.,-2.,4.,1.])*b.mV})
        if schedule is not None:
            fixture.network.schedule=schedule
        if not record:
            fixture.network.remove(fixture.monitor)
            fixture.monitor=b.SpikeMonitor(fixture.neurons,record=False)
            fixture.network.add(fixture.monitor)
        invalid=fixture.inputs[0 if first else 1]
        if refractory:
            fixture.neurons.rfc[int(invalid._group.start)]=10*b.ms
            fixture.neurons.lastspike[int(invalid._group.start)]=0*b.ms
        return fixture

    def guard(fixture, patches):
        forbid_model_cpu_steps(fixture, patches, 'CPU model/input timestep executed', include_inputs=True)
    def exercise(fixture, mode):
        invalid=fixture.inputs[0 if first else 1]
        yield observe_run(fixture,0,rng=True)
        if warm:
            invalid.active=False
            yield observe_run(fixture,.3,rng=True)
            invalid.active=True
        with pytest.raises(Exception) as caught:fixture.run_ms(.4)
        cause=caught.value.__cause__ or caught.value
        assert type(cause) is ValueError
        assert str(cause)=='p < 0, p > 1 or p is NaN'
        assert caught.value._brian_objname==invalid.codeobj.owner.name
        from brian2.core.functions import Function
        function=next(value for value in invalid.variables.values() if isinstance(value,Function))
        message=caught.value._brian_message.replace(invalid.codeobj.name,'CODEOBJECT').replace(function.name,'BINOMIAL')
        if mode=='cpu':messages.append(message)
        else:assert message==messages[0]
        yield RunObservation(fixture.capture(),rng=capture_rng_state())
        invalid.active=False
        for duration in (.1,2.3,.1):yield observe_run(fixture,duration,rng=True)
        values=(np.random.random(9),np.random.normal(size=9))
        if mode=='cpu':future.append(values)
        else:
            for left,right in zip(values,future[0]):np.testing.assert_array_equal(left,right)
    check_original_api_runs(build,exercise,monkeypatch,guard=guard,seed=731)


@pytest.mark.parametrize('phase',['threshold','reset'])
def test_invalid_input_respects_competing_model_error_order(tmp_path,monkeypatch,phase):
    import brian2 as b
    from original_api_reference import build_original_api_fixture, capture_rng_state
    from original_api_protocol import RunObservation, check_original_api_runs

    power='(i+1)**(i-1)'
    def build(mode):
        fixture=build_original_api_fixture(tmp_path/str(mode=='native'),
            parameters={'r_poi':3700*b.Hz,'r_poi2':12000*b.Hz,
                'eq_th':power+'>0' if phase=='threshold' else 'True',
                'eq_rst':'v=v_rst' if phase=='threshold' else 'v=v_rst+'+power+'*mV'},
            exc=(0,),exc2=(2,),silenced=())
        return fixture

    def guard(fixture, patches):
        forbid_model_cpu_steps(fixture, patches, 'CPU timestep executed', include_inputs=True)
    def exercise(fixture, mode):
        with pytest.raises(b.BrianObjectException) as caught:fixture.run_ms(.3)
        cause=caught.value.__cause__
        assert type(cause) is ValueError
        assert str(cause)==('Integers to negative integer powers are not allowed.' if phase=='threshold'
                            else 'p < 0, p > 1 or p is NaN')
        yield RunObservation(fixture.capture(),rng=capture_rng_state())
    check_original_api_runs(build,exercise,monkeypatch,guard=guard,seed=731)


@pytest.mark.parametrize('schedule', [
    ['start','groups','synapses','thresholds','resets','end'],
    ['start','groups','thresholds','resets','synapses','end'],
])
def test_input_failure_custom_phase_queue_continuation(tmp_path,monkeypatch,schedule):
    test_numpy_input_error_repair_queue_and_source_transitions(
        tmp_path,monkeypatch,12000,False,.2,True,False,True,schedule=schedule)
