"""Actual Brian CPP standalone reference; this process imports no MLX code."""
import argparse
import importlib.util
import json
from pathlib import Path

import brian2 as b
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def inputs(dtype):
    rawtype=np.uint64 if dtype==np.float64 else np.uint32
    info=np.finfo(dtype)
    values=np.array([0.,-0.,1.,-1.,3.,-3.,.3,-.3,np.inf,-np.inf,info.max,-info.max,
        info.tiny,-info.tiny,np.nextafter(dtype(0),dtype(1)),np.nextafter(dtype(1),dtype(0)),
        np.nextafter(dtype(1),dtype(2))],dtype)
    raw=([0x7ff8123456789abc,0xfff8123456789abc,0x7ff0000000000001,0xfff0000000012345]
         if dtype==np.float64 else [0x7fc12345,0xffc12345,0x7f800001,0xff812345])
    values=np.r_[values,np.array(raw,rawtype).view(dtype)]
    x,y=np.meshgrid(values,values)
    rng=np.random.default_rng(781515)
    random=rng.integers(0,np.iinfo(rawtype).max,(2,2048),dtype=rawtype).view(dtype)
    return np.r_[x.ravel(),random[0]],np.r_[y.ravel(),random[1]]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dtype',choices=['float64','float32'],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--case',choices=['dynamic','mixed','context','two-product','fixed-base','integer-base','constant','timed'],default='dynamic')
    parser.add_argument('--all-float32',action='store_true')
    parser.add_argument('--integer-dtype',choices=['int64','uint64'],default='int64');args=parser.parse_args()
    directory=args.output;directory.mkdir(parents=True,exist_ok=True)
    b.set_device('cpp_standalone',directory=str(directory/'cpp'),build_on_run=True)
    b.prefs.devices.cpp_standalone.extra_make_args_unix=['-j2']
    if args.all_float32:b.prefs.core.default_float_dtype=np.float32
    spec=importlib.util.spec_from_file_location('original_floating_cpp',ROOT/'code/paper-phil-drosophila/model.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    params=dict(original.default_params)
    params.update(eqs=params['eqs']+'x:1\ny:1\np:1\nr:1\nq:1\n',eq_th='True',
        eq_rst=params['eq_rst']+';p=x**y;r=x%y;q=x//y',t_rfc=0*b.ms)
    dtype=np.dtype(args.dtype).type;x,y=inputs(dtype)
    types={name:dtype for name in ('x','y','p','r','q')}
    if args.case=='mixed':
        params['eqs']=params['eqs'].replace('y:1','y:integer')
        x=np.array([-1.,33554432.,3.,-.3],dtype)
        y=np.array([16777217,16777217,-2,-3],np.int32)
        types['y']=np.int32
    if args.case=='integer-base':
        params['eqs']=params['eqs'].replace('x:1','x:integer')
        integer=np.dtype(args.integer_dtype)
        values=([2**24+1,2**63+2**39+1,1,2**63-1,0,2**64-1,2**63,3]
            if integer.kind=='u' else
            [2**24+1,-2**24-1,1,-1,0,np.iinfo(np.int64).max,np.iinfo(np.int64).min,3])
        x=np.array(values,integer)
        y=np.array([1.,1.,.5,.3,-2.,.3,.5,-.3],dtype)
        types['x']=integer
        if integer.kind=='u':
            # Brian's _higher_type has no uint64 remainder specialization.
            # Its accepted floating cast retains an actual unsigned operand
            # conversion; direct unsigned power is accepted unchanged.
            params['eq_rst']=original.default_params['eq_rst']+';p=x**y;r=(1.0*x)%y;q=(1.0*x)//y'
    if args.case in ('context','two-product'):
        params['eqs']+='s:1\nz:1\n'
        params['eq_rst']=original.default_params['eq_rst']+';p=s*(x**y)+z;r=s*(x%y)+z;q=s*(x//y)+z'
        x=np.array([.3,3.,1.3,.1],dtype);y=np.array([.1,.3,.7,2.1],dtype)
        types.update(s=dtype,z=dtype)
        if args.case=='two-product':
            params['eqs']+='h:1\n';types['h']=dtype
            params['eq_rst']=params['eq_rst'].replace('+z','+z*h')
    if args.case=='constant':
        params.update(numerator=1.,denominator=.1)
        params['eq_rst']=original.default_params['eq_rst']+';p=numerator**denominator;r=numerator%denominator;q=numerator//denominator'
        x=np.array([.3,3.,1.3,.1],dtype);y=np.array([.1,.3,.7,2.1],dtype)
    if args.case=='fixed-base':
        params['two']=dtype(2)
        params['eq_rst']=original.default_params['eq_rst']+';p=two**y;r=x%y;q=x//y'
        x=np.array([.3,3.,1.3,.1],dtype);y=np.array([.1,.3,.7,2.1],dtype)
    if args.case=='timed':
        params['drive']=b.TimedArray(np.arange(16,dtype=np.float64).reshape(4,4)/10+.3,dt=.1*b.ms)
        params['eq_rst']=original.default_params['eq_rst']+';p=drive(t,i)**y;r=drive(t,i)%y;q=drive(t,i)//y'
        x=np.array([.3,3.,1.3,.1],dtype);y=np.array([.1,.3,.7,2.1],dtype)
    group=b.NeuronGroup(len(x),params['eqs'],threshold=params['eq_th'],reset=params['eq_rst'],
        refractory='rfc',method='linear',namespace=params,dtype=types)
    group.v=params['v_0'];group.g=0*b.mV;group.rfc=0*b.ms;group.x=x;group.y=y
    if args.case in ('context','two-product'):
        group.s=np.array([1.1,1.3,.1,1.7],dtype);group.z=np.array([-.3,-1.,-.1,-.123],dtype)
        if args.case=='two-product':group.h=np.array([-.77,.33,1.2,-1.9],dtype)
    monitor=b.SpikeMonitor(group);network=b.Network(group,monitor)
    initial={name:np.zeros(len(x),np.dtype(variable.dtype)) for name,variable in group.variables.items()
        if name in group.equations and group.equations[name].type!='subexpression'}
    initial.update(v=np.full(len(x),float(params['v_0']),group.variables['v'].dtype),x=x.copy(),y=y.copy(),
        lastspike=np.full(len(x),-1e4,group.variables['lastspike'].dtype),not_refractory=np.ones(len(x),bool))
    if args.case in ('context','two-product'):
        initial.update(s=np.array([1.1,1.3,.1,1.7],dtype),z=np.array([-.3,-1.,-.1,-.123],dtype))
        if args.case=='two-product':initial['h']=np.array([-.77,.33,1.2,-1.9],dtype)
    np.savez(directory/'initial.npz',**initial)
    network.run(.1*b.ms,namespace={})
    variables={}
    for name,variable in group.variables.items():
        if not hasattr(variable,'dtype'):continue
        condition=getattr(variable,'conditional_write',None)
        equation=group.equations[name] if name in group.equations else None
        variables[name]=dict(name=name,dtype=np.dtype(variable.dtype).name,
            dimensions=[float(v) for v in variable.dim._dims],scalar=variable.scalar,constant=variable.constant,
            read_only=variable.read_only,conditional_write=condition.name if condition is not None else None,
            flags=list(equation.flags) if equation is not None else [])
    runners=(group.state_updater,group.thresholder['spike'],group.resetter['spike'])
    metadata=dict(variables=variables,abstract=[runner.abstract_code for runner in runners],brian_version=b.__version__,
        flags=list(__import__('brian2.codegen.cpp_prefs',fromlist=['get_compiler_and_args']).get_compiler_and_args()[1]),target=type(runners[-1].codeobj).__name__,
        dtype=args.dtype,case=args.case,state_dtype=np.dtype(group.variables['v'].dtype).name,
        temporary_dtype=np.dtype(b.prefs.core.default_float_dtype).name,
        eqs=params['eqs'],eq_th=params['eq_th'],eq_rst=params['eq_rst'],dt_s=float(group.clock.dt),neurons=len(x),clock_timestep=int(group.clock.variables['timestep'].get_value()[0]))
    metadata['operator_namespace']={name:np.asarray(params[name]).item() for name in ('numerator','denominator','two') if name in params}
    metadata['operator_namespace_dtypes']={name:str(np.asarray(params[name]).dtype)
        for name in ('two',) if name in params}
    if args.case=='timed':
        metadata['timed_array']=dict(values=params['drive'].values.tolist(),dt=float(params['drive'].dt))
    np.savez(directory/'final.npz',**{name:group.variables[name].get_value().copy() for name in initial},
        spike_i=np.asarray(monitor.i[:]).copy(),spike_t=np.asarray(monitor.t[:]).copy())
    (directory/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (directory/'source.cpp').write_text(str(runners[-1].codeobj.code.cpp_file))
    print(json.dumps(dict(target=metadata['target'],dtype=args.dtype,neurons=len(x),output=str(directory))),flush=True)


if __name__=='__main__':main()
