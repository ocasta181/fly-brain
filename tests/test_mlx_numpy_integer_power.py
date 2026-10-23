"""Actual NumPy integer-power failures retain vector stores and run state."""
from native_execution import forbid_call, forbid_model_cpu_steps
from contextlib import nullcontext
from dataclasses import replace

import brian2 as b
import numpy as np
import pytest

from mlx_backend.brian_device import original_api
from mlx_backend.model import DEFAULT_EQUATIONS, DEFAULT_RESET
from original_api_reference import (build_original_api_fixture, cpu_runtime,
    assert_snapshot_equal, capture_rng_state, assert_rng_state_equal)
from original_api_protocol import check_original_api_runs, observe_run


pytestmark = pytest.mark.metal


@pytest.mark.parametrize('expression,active,record',[
    ('x**y',True,True),('x**y',True,False),('x**-1',True,True),
    ('x**-1',False,False),('x**y',False,True)])
def test_original_numpy_integer_power_failure_and_continuation(
        tmp_path,monkeypatch,expression,active,record,phase='reset'):
    def build(mode):
        parameters=dict(eqs=DEFAULT_EQUATIONS+'x:integer\ny:integer\np:integer\n',
            eq_th='('+expression+')>0' if phase=='threshold' else 'i<2' if active else 'False',
            eq_rst=DEFAULT_RESET+';x+=1;p='+('x' if phase=='threshold' else expression)+';x+=1',
            t_rfc=0*b.ms,r_poi=4900*b.Hz,r_poi2=3200*b.Hz)
        fixture=build_original_api_fixture(tmp_path/str(mode == 'native'),parameters=parameters,
            exc=(2,2),exc2=(3,),silenced=(),
            initial_state=dict(x=[2,-3,7,11],y=[2,-1,0,3],p=[17]*4))
        if not record:
            fixture.network.remove(fixture.monitor)
            fixture.monitor=b.SpikeMonitor(fixture.neurons,record=False)
            fixture.network.add(fixture.monitor)
        return fixture
    def guard(fixture, patch):
        forbid_model_cpu_steps(fixture, patch, 'Native integer power used a CPU timestep', patch_preparation=True)
    def exercise(fixture, mode):
        for index,duration in enumerate((0.,.3,.3)):
            if index==2 and expression=='x**y':fixture.neurons.y[1]=2
            fails=active and (index==1 or index==2 and expression=='x**-1')
            if fails:
                observed=observe_run(fixture,duration,raises=b.BrianObjectException,rng=True)
                assert type(observed.error.__cause__) is ValueError
                assert str(observed.error.__cause__)=='Integers to negative integer powers are not allowed.'
            else:
                observed=observe_run(fixture,duration,rng=True)
            yield observed
            if mode == 'native':
                assert b.get_device()._sessions[fixture.network].prepared.program.runtime_error_type=='model'
    check_original_api_runs(build,exercise,monkeypatch,guard=guard,integer_dtype=np.int64)


@pytest.mark.parametrize('float_dtype',[np.float32,np.float64])
@pytest.mark.parametrize('chunks',[(3,),(1,1,1)])
def test_numpy_error_order_keeps_first_clock_across_native_lanes(
        tmp_path,monkeypatch,float_dtype,chunks):
    """A later low-row index error cannot replace the first clock's power error."""
    import mlx.core as mx
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome,Stimulus
    from mlx_backend.runtime_errors import first_error,model_exception
    with cpu_runtime(target='numpy', integer_dtype=np.int64, float_dtype=float_dtype):
        params=dict(eqs=DEFAULT_EQUATIONS+'x:integer\ny:integer\np:integer\ncolumn:integer\n',
            eq_th='int(drive(t,column))+x**y>0',
            eq_rst=DEFAULT_RESET+';column+=3*int(i==0)',t_rfc=0*b.ms,
            drive=b.TimedArray([[1.,2.],[3.,4.]],dt=.1*b.ms))
        values=dict(x=[2,3,4,5],y=[2,2,-1,2],p=[17]*4,column=[0]*4)
        fixture=build_original_api_fixture(tmp_path,parameters=params,exc=(),exc2=(),silenced=(),initial_state=values)
        with pytest.raises(b.BrianObjectException) as expected:fixture.run_ms(.3)
        assert type(expected.value.__cause__) is ValueError
        graph=Connectome(np.arange(4),np.asarray(fixture.synapses.i[:],np.int32),
            np.asarray(fixture.synapses.j[:],np.int32),np.ones(len(fixture.synapses)),
            np.asarray(fixture.synapses.w[:]/b.mV))
        model=resolve_model(params)
        stimulus=Stimulus(np.zeros((1,3,0),np.uint8),(),(),(0,),0,0,'fault-clock',model.dt_ms)
        prepared=configured.prepare(graph,stimulus,model)
        initial=configured.initial_state(prepared)
        state=replace(initial,values={**initial.values,**{name:mx.array(np.asarray(value,np.int64)) for name,value in values.items()}})
        forbid_call(b.Network, 'run', 'Native error clock used a CPU timestep', monkeypatch)
        for length in chunks:
            state,_=configured.advance(prepared,state,mx.array(stimulus.events.reshape(-1)),3,length)
        mx.eval(state.release)
        releases=np.asarray(state.release)
        # Row0 fails in a later clock and has the earlier operation number.
        row0=model_exception(prepared.program.runtime_errors,int(releases[0])&65535)
        assert isinstance(row0,IndexError)
        assert first_error(releases,prepared.program.runtime_error_order)[0]==2
        with pytest.raises(ValueError,match='Integers to negative integer powers'):
            configured.evaluate_state(state)


@pytest.mark.parametrize('record',[False,True])
def test_original_numpy_integer_power_threshold_failure_and_continuation(tmp_path,monkeypatch,record):
    test_original_numpy_integer_power_failure_and_continuation(
        tmp_path,monkeypatch,'x**y',True,record,phase='threshold')


@pytest.mark.parametrize('phase',['reset','threshold'])
@pytest.mark.parametrize('power_first',[True,False])
@pytest.mark.parametrize('same_lane',[True,False])
@pytest.mark.parametrize('record',[True,False])
def test_original_numpy_competing_vector_error_priority(
        tmp_path,monkeypatch,phase,power_first,same_lane,record,delay_ms=None):
    """A complete earlier vector operation wins over a later failing row."""
    expected=[]
    expression=('x**y+int(drive(t,column))' if power_first else
                'int(drive(t,column))+x**y')
    for native in (False,True):
        with cpu_runtime(target='numpy', integer_dtype=np.int64):
            parameters=dict(eqs=DEFAULT_EQUATIONS+'x:integer\ny:integer\np:integer\ncolumn:integer\n',
                eq_th='('+expression+')>0' if phase=='threshold' else 'True',
                eq_rst=DEFAULT_RESET+';p='+('x' if phase=='threshold' else expression),
                t_rfc=0*b.ms,r_poi=4900*b.Hz,r_poi2=3200*b.Hz,
                drive=b.TimedArray([[1.,2.],[3.,4.]],dt=.1*b.ms))
            if delay_ms is not None:parameters['t_dly']=delay_ms*b.ms
            with original_api(workers=False) if native else nullcontext():
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=parameters,
                    exc=(2,2),exc2=(3,),silenced=(),
                    initial_state=dict(x=[2,3,4,5],y=[-1,2,2,2] if same_lane else [2,2,-1,2],
                                       p=[17]*4,column=[3,0,0,0]))
                if not record:
                    fixture.network.remove(fixture.monitor)
                    fixture.monitor=b.SpikeMonitor(fixture.neurons,record=False)
                    fixture.network.add(fixture.monitor)
                if native:
                    forbid_model_cpu_steps(fixture, monkeypatch, 'Native competing errors used a CPU timestep', patch_preparation=True)
                for index in range(4):
                    if index==2:
                        if power_first:fixture.neurons.y=2
                        else:fixture.neurons.column=0
                    if index==3:
                        fixture.neurons.y=2;fixture.neurons.column=0
                    if index in (1,2):
                        family=ValueError if power_first==(index==1) else IndexError
                        with pytest.raises(b.BrianObjectException) as error:fixture.run_ms(.2)
                        assert isinstance(error.value.__cause__,family)
                        observed=fixture.capture()
                    else:observed=fixture.run_ms(0. if index==0 else .2)
                    rng=capture_rng_state()
                    if native:
                        assert_snapshot_equal(observed,expected[index][0])
                        assert_rng_state_equal(rng,expected[index][1])
                    else:expected.append((observed,rng))


@pytest.mark.parametrize('phase',['reset','threshold'])
def test_original_numpy_zero_delay_competing_errors(tmp_path,monkeypatch,phase):
    test_original_numpy_competing_vector_error_priority(
        tmp_path,monkeypatch,phase,True,False,False,delay_ms=0.)


@pytest.mark.parametrize('phase',['reset','threshold'])
def test_original_numpy_fused_poisson_power_error_and_repair(tmp_path,monkeypatch,phase):
    expected=[]
    expression='(i+1)**(i-offset)'
    for native in (False,True):
        with cpu_runtime(target='numpy'):
            params=dict(eqs=DEFAULT_EQUATIONS,offset=1,
                eq_th=expression+'>0' if phase=='threshold' else 'True',
                eq_rst='v=v_rst' if phase=='threshold' else 'v=v_rst+'+expression+'*mV',
                t_rfc=0*b.ms,r_poi=4900*b.Hz,r_poi2=3200*b.Hz)
            with original_api(workers=False) if native else nullcontext() as device:
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=params,
                    exc=(0,0),exc2=(3,),silenced=())
                if native:
                    forbid_model_cpu_steps(fixture, monkeypatch, 'Native fused error used a CPU timestep', patch_preparation=True)
                for index in range(2):
                    if index==0:
                        with pytest.raises(b.BrianObjectException) as error:fixture.run_ms(.2)
                        assert type(error.value.__cause__) is ValueError
                        observed=fixture.capture()
                    else:
                        fixture.neurons.namespace['offset']=0
                        observed=fixture.run_ms(.2)
                    rng=capture_rng_state()
                    if native:
                        session=device._sessions[fixture.network]
                        assert session.sampler.can_fuse(session.prepared)
                        assert_snapshot_equal(observed,expected[index][0])
                        assert_rng_state_equal(rng,expected[index][1])
                    else:expected.append((observed,rng))


@pytest.mark.parametrize('component',['v','g'])
@pytest.mark.parametrize('optimise',[True,False])
def test_original_numpy_state_index_failure_and_repaired_continuation(
        tmp_path,monkeypatch,component,optimise):
    expected=[]
    for native in (False,True):
        with cpu_runtime(target='numpy'):
            monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',optimise)
            equations=DEFAULT_EQUATIONS.replace('v_0 - v + g','v_0 - v + g + drive(t,column)*mV') if component=='v' else DEFAULT_EQUATIONS.replace('-g / tau','(-g + drive(t,column)*mV) / tau')
            assert 'drive' in equations
            params=dict(eqs=equations+'column:integer\n',eq_th='i<2',eq_rst=DEFAULT_RESET,
                t_rfc=0*b.ms,drive=b.TimedArray([[1.,2.],[3.,4.]],dt=.1*b.ms))
            with original_api(workers=False) if native else nullcontext():
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=params,
                    exc=(),exc2=(),silenced=(),initial_state=dict(column=[0,0,3,0],
                        g=np.array([1.,2.,3.,4.])*b.mV,v=np.array([-51.,-50.,-49.,-48.])*b.mV))
                if native:
                    forbid_model_cpu_steps(fixture, monkeypatch, 'Native state error used a CPU timestep', patch_preparation=True)
                for index in range(2):
                    if index==0:
                        with pytest.raises(b.BrianObjectException) as error:fixture.run_ms(.2)
                        assert isinstance(error.value.__cause__,IndexError)
                        observed=fixture.capture()
                        if not native:
                            (tmp_path/'cpu-state-source.txt').write_text(str(fixture.neurons.state_updater.codeobj.code))
                    else:
                        fixture.neurons.column=0
                        observed=fixture.run_ms(.2)
                    if native:assert_snapshot_equal(observed,expected[index])
                    else:expected.append(observed)
