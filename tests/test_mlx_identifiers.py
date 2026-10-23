"""Actual accepted Brian names retain host identity across native lowering."""
from native_execution import forbid_cpu_steps, forbid_model_cpu_steps, require_metal
import json

import numpy as np
import pytest


def test_identifier_encoding_is_injective_and_preserves_runtime_abi():
    from mlx_backend.symbols import variable_code_names
    names = ("switch", "case", "constant", "uint", "float2", "weights", "gid",
             "d_add", "_eon_var_737769746368", "x", "x_1", "k", "row", "channel",
             "cursor", "event", "X", "start", "length", "step", "release_step",
             "p_release", "source_bit", "template", "v", "g", "t", "N")
    codes = variable_code_names(names)
    assert len(set(codes.values())) == len(names)
    assert variable_code_names(reversed(names)) == codes
    assert all(codes[name] == name for name in ("v", "g", "t", "N"))
    assert all(codes[name].startswith("_eon_var_") for name in names[:-4])


def identifier_configuration():
    import brian2 as b
    from mlx_backend.model import DEFAULT_EQUATIONS
    equations = DEFAULT_EQUATIONS + "\n" + "\n".join((
        "switch:boolean", "uint:integer", "gid:integer", "float2:1",
        "weights:1", "d_add:1", "eon_next_v:1", "phase:1", "k:integer", "negative:integer", "power_value:integer",
        "template = weights+float2 : 1"))
    return {"eqs": equations, "eq_th": "(i<2) or switch", "t_rfc": 0*b.ms,
        "eq_rst": "v=v_rst;switch=not switch;uint=1-uint;gid+=2;"
                  "float2+=thread;weights=kernel(t,uint);d_add=template+float2;"
                  "eon_next_v=sin(phase)+eon_var_737769746368;phase+=1/32;power_value=k+negative;k+=1",
        "thread": .125, "eon_var_737769746368": .25,
        "kernel": b.TimedArray(np.array([[.5, 1.], [1.5, 2.], [2.5, 3.]]), dt=.1*b.ms),
        "r_poi": 3700*b.Hz, "r_poi2": 7200*b.Hz, "t_dly": .3*b.ms}


@pytest.mark.metal
@pytest.mark.parametrize("target,dtype,record", [
    (target, dtype, record) for target in ("numpy", "cython")
    for dtype in ("float64", "float32") for record in (False, True)])
def test_actual_identifier_aliases_poisson_recording_and_continuation(tmp_path, target, dtype, record):
    b = pytest.importorskip("brian2")
    mx = require_metal('Metal required')
    from mlx_backend.brian_device import original_api
    from original_api_reference import (
        assert_snapshot_equal, assert_rng_state_equal, build_original_api_fixture,
        capture_rng_state, original_runtime)

    expected = []
    expected_rng = None
    for native in (False, True):
        with original_runtime(
                original_api, seed=8841, target=target, float_dtype=np.dtype(dtype).type, native=native) as device:
            fixture = build_original_api_fixture(tmp_path/str(native),
                parameters=identifier_configuration(), exc=(0, 0), exc2=(2,), silenced=(),
                initial_state={"switch": [False, True, True, False],
                    "float2": np.array([-.0, .25, -.5, 1.]),
                    "phase": np.array([-.25, 0., .25, 1.]), "k": [2,3,5,7], "negative": [1]*4})
            if not record:
                fixture.network.remove(fixture.monitor)
                fixture.monitor = b.SpikeMonitor(fixture.neurons, record=False)
                fixture.network.add(fixture.monitor)
            if native:
                forbid_cpu_steps(fixture.network, None, 'Native execution ran an original CPU timestep')
            for index, duration in enumerate((.1, .2, 0., .15)):
                snapshot = fixture.run_ms(duration)
                if native:
                    assert_snapshot_equal(snapshot, expected[index])
                else:
                    expected.append(snapshot)
            for stage, obj in (("state", fixture.neurons.state_updater),
                    ("threshold", fixture.neurons.thresholder["spike"]),
                    ("reset", fixture.neurons.resetter["spike"])):
                (tmp_path/str(native)/(stage+"-source.txt")).write_text(str(obj.codeobj.code))
            if native:
                assert_rng_state_equal(capture_rng_state(), expected_rng)
                session = device._sessions[fixture.network]
                program = session.prepared.program
                names = {name: var.code_name for name, var in program.variables.items()}
                (tmp_path/"symbol-map.json").write_text(json.dumps(names, indent=2)+"\n")
                for name in ("switch", "uint", "gid", "float2", "weights", "d_add"):
                    assert name in snapshot.state and name in program.variables
                    assert names[name].startswith("_eon_var_")
                assert session.prepared.model.state_dtype == dtype
                assert device.native_steps == 5
            else:
                expected_rng = capture_rng_state()


@pytest.mark.metal
@pytest.mark.parametrize("dtype", [np.float32,np.float64])
@pytest.mark.parametrize("spikes", [False,True])
def test_loop_local_name_preserves_numpy_fault_and_repaired_continuation(tmp_path,monkeypatch,dtype,spikes):
    from test_mlx_numpy_integer_stores import test_python_scalar_store_failure_and_repaired_api_continuation
    test_python_scalar_store_failure_and_repaired_api_continuation(tmp_path,monkeypatch,
        float('inf'),spikes,power_fault=True,base_name='k',default_dtype=dtype)


@pytest.mark.metal
@pytest.mark.parametrize('target', ['numpy','cython'])
@pytest.mark.parametrize('record', [False,True])
def test_accepted_type_and_kernel_names_with_target_physical_widths(monkeypatch,target,record):
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from mlx_backend.model import DEFAULT_EQUATIONS
    from original_api_reference import (OriginalAPIFixture,load_original_model,original_runtime,
        assert_snapshot_equal,capture_rng_state,assert_rng_state_equal)
    mx = require_metal('Metal required')
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',False)
    dtypes=dict(zip(('case','constant','device','thread','gid','uint','weights','d_add'),
        map(np.dtype,('int8','uint8','int16','uint16','int32','uint32','int64','uint64'))))
    dtypes.update(float2=np.dtype('float32'),switch=np.dtype('bool'))
    if target=='cython':
        # Actual pinned Cython supports these seven physical dtypes. Preserve
        # every collision-prone name while using its accepted width inventory.
        dtypes.update(case=np.dtype('int32'),device=np.dtype('int64'),
            thread=np.dtype('uint64'),uint=np.dtype('uint64'))
    declarations='float2:1\nswitch:boolean\n'+''.join(name+':integer\n'
        for name in dtypes if name not in ('float2','switch'))
    updates=';'.join(name+'+=1' for name in dtypes if name not in ('float2','switch'))
    expected=[];expected_rng=[]
    for native in (False,True):
        with original_runtime(original_api, seed=9948, target=target, native=native) as device:
            original=load_original_model();params=dict(original.default_params)
            params.update(eqs=DEFAULT_EQUATIONS+declarations,eq_th='True',
                eq_rst='v=v_rst;float2+=.125;switch=not switch;'+updates,
                t_rfc=0*b.ms,r_poi=4900*b.Hz,r_poi2=3200*b.Hz)
            group=b.NeuronGroup(4,params['eqs'],threshold=params['eq_th'],reset=params['eq_rst'],
                refractory='rfc',method='linear',namespace=params,dtype=dict(dtypes))
            group.v=params['v_0'];group.g=0*b.volt;group.rfc=0*b.ms
            group.float2_[:]=np.array([.125,-.25,.5,-1.],np.float32)
            group.switch_[:]=[False,True,True,False]
            for name,dtype in dtypes.items():
                if dtype.kind in 'iu':
                    limits=np.iinfo(dtype)
                    getattr(group,name+'_')[:]=np.array([limits.max-1,limits.min,0,1],dtype)
            syn=b.Synapses(group,group,'w:volt',on_pre='g_post+=w',delay=params['t_dly'])
            syn.connect(i=[0,1],j=[1,2]);syn.w=[.125,-.25]*b.mV
            monitor=b.SpikeMonitor(group,record=record)
            inputs,_=original.poi(group,[0,0],[2],params)
            fixture=OriginalAPIFixture(original,params,(),group,syn,monitor,inputs,
                b.Network(group,syn,monitor,*inputs))
            if native:
                forbid_model_cpu_steps(fixture, monkeypatch, 'A CPU timestep executed', include_inputs=True, patch_preparation=True)
            for index,duration in enumerate((0.,.15,.25)):
                observed=fixture.run_ms(duration);rng=capture_rng_state()
                if native:
                    assert_snapshot_equal(observed,expected[index])
                    assert_rng_state_equal(rng,expected_rng[index])
                    session=device._sessions[fixture.network]
                    assert len(session.state.values.layout.groups)==(3 if target=='cython' else 4)
                else:expected.append(observed);expected_rng.append(rng)
