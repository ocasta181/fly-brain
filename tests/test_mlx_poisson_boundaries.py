"""Shared-draw acceptance boundaries; generator/refill proof lives separately."""
from functools import lru_cache
import ctypes
import numpy as np
import pytest


@lru_cache(maxsize=2)
def math_kernel(target):
    import mlx.core as mx
    from mlx_backend.binary64 import MSL_HEADER
    from mlx_backend.expression_rng_math import NORMAL_MATH_HEADER,POISSON_EXP_HEADER
    from mlx_backend.expression_poisson_compiled import source
    header=source('device const ulong* draws,thread uint* cursor,thread uint* errors',
        'draws[(*cursor)++]',target=target,error='*errors|=2u;',check='if(*errors)return 0;')
    return mx.fast.metal_kernel(name='expression_poisson_boundary_math',
        input_names=['rates','input_draws'],output_names=['values','used','errors'],
        header=MSL_HEADER+NORMAL_MATH_HEADER+POISSON_EXP_HEADER+header,source='''
        uint j=thread_position_in_grid.x;if(j>=COUNT)return;uint cursor=0,error=0;
        values[j]=compiled_poisson(rates[j],input_draws+4*j,&cursor,&error);used[j]=cursor;errors[j]=error;
        ''',compile_options={'math_mode':'safe'})


def native_math(rates,draws,target):
    import mlx.core as mx
    result=math_kernel(target)(inputs=[mx.array(rates.view(np.uint64)),mx.array(draws.view(np.uint64).reshape(-1))],
        template=[('COUNT',len(rates))],grid=(len(rates),1,1),threadgroup=(256,1,1),
        output_shapes=[(len(rates),)]*3,output_dtypes=[mx.int32,mx.uint32,mx.uint32])
    return tuple(np.asarray(a)for a in result)


@pytest.mark.metal
def test_cython_poisson_acceptance_boundaries_match_actual_compiled_code(tmp_path):
    import brian2 as b
    preferences={name:b.prefs[name]for name in('codegen.target','codegen.runtime.cython.cache_dir',
        'codegen.runtime.cython.delete_source_files')}
    ambient=np.random.get_state();cases=[]
    try:
        b.prefs.codegen.target='cython';b.prefs.codegen.runtime.cython.cache_dir=str(tmp_path/'cython-default')
        b.prefs.codegen.runtime.cython.delete_source_files=False
        b.seed(81)
        warm=b.NeuronGroup(1,'a:1',clock=b.Clock(dt=.1*b.ms))
        allocate=warm.run_regularly('a=rand()',when='start');b.Network(warm,allocate).run(.1*b.ms,namespace={})
        group=b.NeuronGroup(1,'lam:1\na:integer',clock=b.Clock(dt=.1*b.ms))
        runner=group.run_regularly('a=poisson(lam)',when='start');b.Network(group,runner).run(.1*b.ms,namespace={})
        device=b.get_device();pointer=int(device.rand_buffer[0])
        buffer=np.ctypeslib.as_array((ctypes.c_double*20000).from_address(pointer))
        rates=group.variables['lam'].get_value();values=group.variables['a'].get_value()
        def sample(lam,u,v):
            rates[0]=lam;buffer[1:5]=(u,v,.5,0.);device.rand_buffer_index[0]=1
            runner.codeobj()  # Actual unchanged generated Cython machine code.
            count=int(device.rand_buffer_index[0])-1;assert count in(2,4)
            return int(values[0]),count
        for lam in(10.,17.,100.,1000.,1e8,1e16):
            for u in(.0005,.01,.02,.035,.065,.935,.965,.98,.99,.9995,np.nextafter(1.,0.)):
                lo,hi=0,(1<<53)-1
                while lo<hi:
                    mid=lo+(hi-lo+1)//2
                    if sample(lam,u,mid/2**53)[1]==2:lo=mid
                    else:hi=mid-1
                for raw in sorted({0,(1<<53)-1,*range(max(0,lo-8),min(1<<53,lo+9))}):
                    v=raw/2**53;k,count=sample(lam,u,v);cases.append((lam,u,v,k,count))
    finally:
        np.random.set_state(ambient)
        for name,value in preferences.items():b.prefs[name]=value
    rates=np.array([r[0]for r in cases]);draws=np.array([[r[1],r[2],.5,0.]for r in cases])
    values,used,errors=native_math(rates,draws,'cython')
    np.savez(tmp_path/'boundary-replay.npz',rates=rates,draws=draws,values=values,used=used,
        cpu_values=np.array([r[3]for r in cases]),cpu_used=np.array([r[4]for r in cases]))
    np.testing.assert_array_equal(errors,0)
    np.testing.assert_array_equal(used,[r[4]for r in cases])
    np.testing.assert_array_equal(values,[r[3]for r in cases])
