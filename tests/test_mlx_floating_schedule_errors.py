"""Actual Cython schedule, operator failure, random stream and resumed queue."""
from native_execution import forbid_cpu_steps as guard_cpu_steps
import importlib.util
import os
from pathlib import Path

import brian2 as b
import numpy as np
import pytest

from mlx_backend.model import DEFAULT_EQUATIONS,DEFAULT_RESET
from original_api_reference import build_original_api_fixture

pytestmark=pytest.mark.metal


@pytest.mark.parametrize('delay_ms',[0.,.1],ids=['zero-delay','delayed'])
@pytest.mark.parametrize('phases',[
    ('groups','thresholds','synapses','resets'),
    ('groups','thresholds','resets','synapses'),
    ('groups','resets','thresholds','synapses'),
    ('synapses','groups','thresholds','resets'),
],ids=['default','reset-before-synapses','reset-before-threshold','synapses-first'])
def test_cython_operator_failure_and_resume_follow_actual_schedule(tmp_path,monkeypatch,phases,delay_ms,record=True):
    monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(
        Path(__file__).resolve().parents[1]/'executions/cython-floating-operators'))
    baseline=os.environ.get('MLX_TEST_REPLAY_SOURCE')
    if baseline:
        from mlx_backend import brian_errors
        spec=importlib.util.spec_from_file_location('mlx_backend._baseline_error_replay',baseline)
        module=importlib.util.module_from_spec(spec)
        __import__('sys').modules[spec.name]=module
        spec.loader.exec_module(module)
        monkeypatch.setattr(brian_errors,'replay_failure',module.replay_failure)
    params=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\np:1\n',eq_th='i<2',
        eq_rst=DEFAULT_RESET+';p=x%y',t_rfc=0*b.ms,t_dly=delay_ms*b.ms,
        r_poi=4900*b.Hz,r_poi2=3200*b.Hz)
    options=dict(parameters=params,exc=(2,2),exc2=(3,),silenced=(),initial_state=dict(
        x=[.3,.7,3.,-3.],y=[2.,0.,2.,2.],g=np.array([.01,.02,.03,.04])*b.volt))
    schedule=['start',*phases,'end']

    from original_api_protocol import check_original_api_runs, observe_run
    following=[]
    def build(mode):
        fixture=build_original_api_fixture(tmp_path/mode,**options);fixture.network.schedule=schedule
        if not record:
            fixture.network.remove(fixture.monitor);fixture.monitor=b.SpikeMonitor(fixture.neurons,record=False);fixture.network.add(fixture.monitor)
        return fixture
    def guard(fixture, patches):
        guard_cpu_steps(fixture.network, patches, 'A CPU neuron or synapse timestep executed', runners=lambda : (fixture.neurons.state_updater, fixture.neurons.thresholder['spike'], fixture.neurons.resetter['spike'], *fixture.synapses._pathways), patch_preparation=True)
    def exercise(fixture, mode):
        failed=observe_run(fixture,.3,raises=b.BrianObjectException,rng=True)
        if mode=='native':
            from brian2.devices.device import get_device
            assert get_device()._sessions[fixture.network].prepared.model.schedule==tuple(schedule)
        yield failed
        fixture.neurons.y[1]=3.
        yield observe_run(fixture,.3,rng=True)
        values=np.random.random(10)
        if mode=='cpu':following.append(values)
        else:np.testing.assert_array_equal(values,following[0])
    check_original_api_runs(build,exercise,monkeypatch,guard=guard,target='cython')


def test_operator_failure_and_resume_with_count_only_monitor(tmp_path,monkeypatch):
    test_cython_operator_failure_and_resume_follow_actual_schedule(tmp_path,monkeypatch,
        ('groups','thresholds','synapses','resets'),0.,record=False)


def test_integer_policy_cache_keeps_each_network_schedule(tmp_path,monkeypatch):
    # i<2 activates IntegerPolicy.execution(), which derives a size-specific
    # model identity. Both networks share equations, graph, inputs and sizes,
    # while only their real Brian schedules differ. Retain this order in one
    # test so a selected regression cannot accidentally skip the cache reuse.
    test_cython_operator_failure_and_resume_follow_actual_schedule(
        tmp_path/'synapses-first',monkeypatch,('synapses','groups','thresholds','resets'),0.)
    test_operator_failure_and_resume_with_count_only_monitor(tmp_path/'count-only',monkeypatch)
