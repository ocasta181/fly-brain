"""Typed host export and native spike/error export contracts."""
import numpy as np
import pytest
from types import SimpleNamespace
mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.metal
from mlx_backend import configured as c
from mlx_backend.state_layout import StateLayout

def test_packed_host_planes_typed_bits_shared_and_mapping():
    variables={}
    expected={}
    for name,dtype,raw in [('f64','float64',np.array([0x8000000000000000,0x7ff8000000000001,0x3ff0000000000000,0],np.uint64)),('f32','float32',np.array([0x80000000,0x7fc00001,0x3f800000,0],np.uint32)),('i8','int8',np.array([-128,-1,0,127],np.int8)),('u8','uint8',np.array([255,128,0,1],np.uint8)),('i16','int16',np.array([-32768,-1,0,32767],np.int16)),('u16','uint16',np.array([65535,32768,0,1],np.uint16)),('i32','int32',np.array([-2147483648,-1,0,2147483647],np.int32)),('i64','int64',np.array([-2**63,-1,0,2**63-1],np.int64)),('flag','bool',np.array([0,1,1,0],np.uint8))]:
        variables[name]=SimpleNamespace(name=name,dtype=dtype,storage_dtype=str(raw.dtype),scalar=False,persistent=True)
        expected[name]=raw.view(dtype).reshape(2,2)
    variables['shared']=SimpleNamespace(name='shared',dtype='float64',storage_dtype='uint64',scalar=True,persistent=True)
    expected['shared']=np.array([[1.,1.],[-0.,-0.]])
    variables['shared_alias']=SimpleNamespace(name='shared_alias',dtype='float64',storage_dtype='uint64',scalar=True,persistent=True)
    expected['shared_alias']=expected['shared']
    program=SimpleNamespace(variables=variables)
    layout=StateLayout.from_program(program,scalar=False)
    buffers=[]
    for group in layout.groups:
        raw=np.concatenate([expected[n].reshape(-1).view(variables[n].storage_dtype).view(group.dtype) for n in group.names])
        buffers.append(mx.array(raw)+mx.array(np.zeros_like(raw)))
    shared=mx.array(np.array([1.,-0.]).view(np.uint64))
    packed=c._PackedValues(layout,buffers,4,{'shared':shared,'shared_alias':shared})
    prepared=SimpleNamespace(program=program,neurons=2)
    for values in [packed,{name:packed[name] for name in packed}]:
        actual=c.host_state(c.State(values,{},{}),prepared)
        assert np.shares_memory(actual['shared'],actual['shared_alias'])
        assert not actual['shared'].flags.writeable
        for name,want in expected.items():
            assert actual[name].dtype==want.dtype
            assert actual[name].shape==want.shape
            assert actual[name].tobytes()==want.tobytes(),name
    with pytest.raises(IndexError):
        c.host_state(c.State(packed,{},{},release=mx.array([-1],mx.int32),check_index_error=True),prepared)

@pytest.mark.parametrize('bits',[1,31,32])
@pytest.mark.parametrize('pattern',['empty','uniform','mixed'])
def test_small_recording_independent_order(bits,pattern):
    n=33;ids=np.array([9,-3,5],np.int64);starts=[7,41,91]
    raw=np.random.default_rng(17).integers(0,2**bits,(3,3,n),dtype=np.uint32)
    if pattern=='empty':raw[:]=0
    if pattern=='uniform':raw[:]=(1<<bits)-1
    raw[-1]&=1
    parts=c._collect_unit_windows([mx.array(a.reshape(-1)) for a in raw],starts,n,ids,bits)
    actual=[tuple(map(int,row)) for part in parts for row in zip(*part)]
    expected=[(int(ids[t]),j,start+k) for t in np.argsort(ids) for w,start in enumerate(starts) for k in range(bits) for j in range(n) if raw[w,t,j]&(1<<k)]
    assert actual==expected

@pytest.mark.parametrize('kind,order,release,exception,message',[
 ('index','row',[0,-1],IndexError,'TimedArray'),
 ('zero_division','row',[-2],ZeroDivisionError,'float division'),
 ('zero_division','row',[-3],ZeroDivisionError,'float divmod'),
 ('model','row',[-65535,-131070],ValueError,'first'),
 ('model','operation',[-65535,-131070],OverflowError,'second'),
])
def test_export_error_selection(kind,order,release,exception,message):
    state=c.State({}, {}, {}, release=mx.array(release,mx.int64),check_index_error=True,
                  runtime_error_type=kind,runtime_error_order=order,
                  runtime_errors=(('ValueError','first'),('OverflowError','second')))
    with pytest.raises(exception,match=message):c._check_index_error(state)
    with pytest.raises(exception,match=message):c.host_state(state,SimpleNamespace(program=SimpleNamespace(variables={}),neurons=1))

@pytest.mark.parametrize('neurons',[65536,65537])
def test_host_and_gpu_export_boundary(neurons):
    ids=np.array([71],np.int64)
    raw=np.zeros(neurons,np.uint32);raw[::101]=1;raw[-1]=1
    parts=c._collect_unit_windows([mx.array(raw)],[19],neurons,ids,1)
    expected=np.flatnonzero(raw)
    assert len(parts)==1
    np.testing.assert_array_equal(parts[0][0],np.full(expected.size,71,np.int64))
    np.testing.assert_array_equal(parts[0][1],expected)
    np.testing.assert_array_equal(parts[0][2],np.full(expected.size,19,np.int64))
    release=np.zeros(neurons,np.int64)
    state=c.State({}, {}, {},release=mx.array(release),check_index_error=True)
    c._check_index_error(state)
    release[-1]=-1
    state=c.State({}, {}, {},release=mx.array(release),check_index_error=True)
    with pytest.raises(IndexError,match='TimedArray'):c._check_index_error(state)
