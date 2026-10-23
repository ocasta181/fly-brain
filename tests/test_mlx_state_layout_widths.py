"""All physical widths share a bounded Metal ABI without changing field bits."""
from native_execution import forbid_cpu_steps as guard_cpu_steps, require_metal
import numpy as np
import pytest


@pytest.mark.metal
def test_all_physical_dtypes_roundtrip_raw_bits_in_four_buffers():
    import mlx.core as mx
    from mlx_backend import configured
    from mlx_backend.compiler import CompiledVariable,_metal_type
    from mlx_backend.state_layout import StateLayout
    dtypes=('float64','float32','bool','int8','uint8','int16','uint16',
            'int32','uint32','int64','uint64')
    variables={('v' if index==0 else f'field{index}'):CompiledVariable(
        'v' if index==0 else f'field{index}',dtype,_metal_type(dtype),False,False,False,None,True)
        for index,dtype in enumerate(dtypes)}
    class Program:pass
    program=Program();program.variables=variables
    layout=StateLayout.from_program(program)
    assert len(layout.groups)==4
    size=256;values={};expected={}
    rng=np.random.default_rng(20261018)
    for name,variable in variables.items():
        dtype=np.dtype(variable.dtype)
        if dtype.kind=='b':array=(np.arange(size)%3==0)
        else:
            rawtype=np.dtype('uint'+str(dtype.itemsize*8))
            raw=rng.integers(0,np.iinfo(rawtype).max,size,dtype=rawtype)
            raw[:4]=np.array([0,1,np.iinfo(rawtype).max,np.iinfo(rawtype).max>>1],rawtype)
            array=raw if dtype.kind=='f' else raw.view(dtype)
        expected[name]=array.copy();values[name]=mx.array(array)
    packed=configured._pack_values(program,values)
    loads,stores=layout.source(program,size='SIZE')
    kernel=mx.fast.metal_kernel(name='raw_mixed_width_state_roundtrip',
        input_names=['_eon_p_'+group.buffer_name for group in layout.groups],
        output_names=['_eon_q_'+group.buffer_name for group in layout.groups],
        source='uint gid=thread_position_in_grid.x;if(gid<SIZE){'+loads+'\n'+stores+'}',
        compile_options={'math_mode':'safe'})
    outputs=kernel(inputs=list(packed.buffers),template=[('SIZE',size)],grid=(size,1,1),
        threadgroup=(128,1,1),output_shapes=list(layout.shapes(size)),
        output_dtypes=[configured._storage_dtype(group.dtype) for group in layout.groups])
    mx.eval(*outputs)
    result=configured._PackedValues(layout,outputs,size)
    for name,value in expected.items():
        actual=np.asarray(result[name]);assert actual.dtype==value.dtype
        np.testing.assert_array_equal(actual,value,err_msg=name)


@pytest.mark.metal
@pytest.mark.parametrize('record',[False,True])
def test_all_physical_dtypes_survive_native_poisson_prepass(monkeypatch,record):
    """Actual Brian objects exercise packed loads inside the Poisson kernel."""
    import brian2 as b
    from mlx_backend.model import DEFAULT_EQUATIONS
    from original_api_reference import OriginalAPIFixture, load_original_model
    from original_api_protocol import check_original_api_runs, observe_run
    mx = require_metal('Metal required')
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',False)
    dtypes={kind+str(bits):np.dtype(kind+str(bits))
            for kind in ('int','uint') for bits in (8,16,32,64)}
    dtypes.update(single=np.dtype('float32'),state_flag=np.dtype('bool'))
    declarations='single:1\nstate_flag:boolean\n'+''.join(name+':integer\n'
        for name in dtypes if name not in ('single','state_flag'))
    updates=';'.join(name+'+=1' for name in dtypes if name not in ('single','state_flag'))

    def build(mode):
        original=load_original_model();params=dict(original.default_params)
        params.update(eqs=DEFAULT_EQUATIONS+declarations,eq_th='True',
            eq_rst='v=v_rst;single+=.125;state_flag=not state_flag;'+updates,
            t_rfc=0*b.ms,r_poi=4900*b.Hz,r_poi2=3200*b.Hz)
        group=b.NeuronGroup(4,params['eqs'],threshold=params['eq_th'],reset=params['eq_rst'],
            refractory='rfc',method='linear',namespace=params,dtype=dict(dtypes))
        group.v=params['v_0'];group.g=0*b.volt;group.rfc=0*b.ms
        group.single_[:]=np.array([.125,-.25,.5,-1.],np.float32)
        group.state_flag_[:]=[False,True,True,False]
        for name,dtype in dtypes.items():
            if dtype.kind in 'iu':
                limits=np.iinfo(dtype)
                getattr(group,name+'_')[:]=np.array([limits.max-1,limits.min,0,1],dtype)
        syn=b.Synapses(group,group,'w:volt',on_pre='g_post+=w',delay=params['t_dly'])
        syn.connect(i=[0,1],j=[1,2]);syn.w=[.125,-.25]*b.mV
        monitor=b.SpikeMonitor(group,record=record)
        inputs,_=original.poi(group,[0,0],[2],params)
        return OriginalAPIFixture(original,params,(),group,syn,monitor,inputs,
            b.Network(group,syn,monitor,*inputs))

    def guard(fixture, patches):
        group,syn,inputs=fixture.neurons,fixture.synapses,fixture.inputs
        guard_cpu_steps(fixture.network, patches, 'A CPU timestep executed', runners=lambda : (group.state_updater, group.thresholder['spike'], group.resetter['spike'], *syn._pathways, *inputs), patch_preparation=True)
    def exercise(fixture, mode):
        for duration in (0.,.15,.25):
            yield observe_run(fixture,duration,rng=True)
            if mode=='native':
                from brian2.devices.device import get_device
                session=get_device()._sessions[fixture.network]
                assert len(session.state.values.layout.groups)==4
    check_original_api_runs(build,exercise,monkeypatch,guard=guard)
