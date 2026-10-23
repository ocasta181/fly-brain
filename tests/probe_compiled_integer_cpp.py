"""Independent actual Brian CPP standalone integer-width oracle; imports no MLX."""
import argparse
import importlib.util
import json
from pathlib import Path

import brian2 as b
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
WIDTHS=('int8','int16','uint16','int32','uint32','int64','uint64')


def case(mode='widths'):
    spec=importlib.util.spec_from_file_location('integer_original_cpp',ROOT/'code/paper-phil-drosophila/model.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    params=dict(original.default_params)
    if mode=='vector_tmp':
        params['eqs']+='x:integer\ncount:integer\nflag:boolean\n'
        params.update(eq_th='True',t_rfc=0*b.ms)
        params['eq_rst']+=';tmp=int(x);flag=tmp<0;count+=4294967297+int(flag);x=(x%1000003)*4099+8589934595'
        values=np.array([-2**63,-2**63+1,-2**40-1,-2**31-1,-1,0,1,2**31-1,2**32+1,2**40+1,2**63-1],np.int64)
        return params,dict(x=np.dtype('int64'),count=np.dtype('int64'),flag=np.dtype('bool')),dict(
            x=values,count=np.full(values.size,17,np.int64),flag=np.zeros(values.size,bool))
    if mode=='power_float32':b.prefs.core.default_float_dtype=np.float32
    if mode in ('power_invariant','power_mutable','power_constant','power_float32'):
        params['eqs']+='x:integer\ny:integer\nseen:1\n'
        params.update(eq_th='True',t_rfc=0*b.ms)
        code='seen=x**y' if mode!='power_constant' else 'seen=x**(-1099511627777)'
        if mode=='power_mutable':code+=';y=(y%3)+1'
        params['eq_rst']+=';'+code
        return params,dict(x=np.dtype('int64'),y=np.dtype('int64'),seen=np.dtype('float64')),dict(
            x=np.array([-2,-3,0,1,-1,1923,-2575,2**53+3,-2**53-1],np.int64),
            y=np.array([3,-2,0,2**63-1,2**63-2,-1,-2,3,2],np.int64))
    if mode=='shared':
        params['eqs']+='tick=timestep(t,dt)+32766:integer(shared,constant over dt)\nheld=tick+i*int(3+t/ms):integer(constant over dt)\nseen:integer\n'
        params.update(eq_th='True',t_rfc=0*b.ms)
        params['eq_rst']+=';seen+=held+int(3+t/ms)'
        return params,dict(tick=np.dtype('int16'),held=np.dtype('int64'),seen=np.dtype('int64')),dict(
            seen=np.full(9,17,np.int64))
    if mode=='names':
        params['eqs']+='x:integer\nout_x:integer\n'
        params.update(eq_th='True',t_rfc=0*b.ms)
        params['eq_rst']+=';x+=out_x;out_x+=1'
        return params,dict(x=np.dtype('int64'),out_x=np.dtype('int64')),dict(
            x=np.array([-7,0,5,11],np.int64),out_x=np.array([3,11,-9,17],np.int64))
    if mode in ('timed1','timed2'):
        params['eqs']+='counter:integer\ncolumn:integer\nseen:1\n'
        params.update(eq_th='i%3!=1',t_rfc=0*b.ms)
        data=[.25,1.25,-1.5] if mode=='timed1' else [[.25,1.25],[2.25,3.25],[4.25,5.25]]
        params['drive']=b.TimedArray(data,dt=.1*b.ms)
        call='drive(t-dt)' if mode=='timed1' else 'drive(t-dt,column)'
        params['eq_rst']+=';counter+=int('+call+');seen='+call
        return params,dict(counter=np.dtype('int64'),column=np.dtype('int64'),seen=np.dtype('float64')),dict(
            counter=np.zeros(9,np.int64),column=np.array([-1,0,1,3,0,1,2,-1,1],np.int64))
    if mode=='boolean':
        params['eqs']+='x:integer\ny:integer\nflag:boolean\n'
        params.update(eq_th='True',t_rfc=0*b.ms)
        params['eq_rst']+=';flag=x;y=int(flag)'
        return params,dict(x=np.dtype('int64'),y=np.dtype('int64')),dict(
            x=np.array([0,1,2,127,128,255,256,257,-1],np.int64))
    if mode=='constant_casts':
        types={f'y{j}':np.dtype(width) for j,width in enumerate(WIDTHS)}
        params['eqs']+=''.join(name+':integer\n' for name in types)
        params.update(eq_th='True',t_rfc=0*b.ms)
        params['eq_rst']+=';'+';'.join(name+'=1e100' for name in types)
        return params,types,{name:np.full(9,17,dtype) for name,dtype in types.items()}
    if mode=='casts':
        types={name:np.dtype(width) for j,width in enumerate(WIDTHS) for name in (f'from32_{j}',f'from64_{j}')}
        types.update(f32=np.dtype('float32'),f64=np.dtype('float64'))
        params['eqs']+='f32:1\nf64:1\n'+''.join(name+':integer\n' for name in types if name not in ('f32','f64'))
        params.update(eq_th='True',t_rfc=0*b.ms)
        params['eq_rst']+=';'+';'.join(f'from32_{j}=f32;from64_{j}=f64' for j in range(len(WIDTHS)))
        values=[3.75,-3.75,2.**31,2.**32+1,-2.**31-1,2.**63,-2.**63,np.nan,np.inf,-np.inf,2.**64,2.**63-1024]
        return params,types,dict(f32=np.array(values,np.float32),f64=np.array(values,np.float64))
    params['eqs']+=''.join(f'x{j}:integer\ny{j}:integer\n' for j in range(len(WIDTHS)))+'comparison:boolean\nrem:integer\nquo:integer\n'
    params.update(eq_th='i%3!=1',t_rfc=0*b.ms)
    params['eq_rst']+=';'+';'.join(f'y{j}=x{j}+x{j};x{j}+=1' for j in range(len(WIDTHS)))
    signed=WIDTHS.index('int64')
    params['eq_rst']+=f';comparison=(x{signed}+1)>x{signed};rem=x{signed}%-3;quo=x{signed}//-3'
    types={name:np.dtype(dtype) for j,dtype in enumerate(WIDTHS) for name in (f'x{j}',f'y{j}')}
    types.update(rem=np.dtype('int64'),quo=np.dtype('int64'))
    initial={}
    for j,dtype in enumerate(WIDTHS):
        limits=np.iinfo(dtype)
        initial[f'x{j}']=np.array([limits.min,limits.max,0,1,limits.max-1,limits.min+1,3,7,11],dtype)
        initial[f'y{j}']=np.full(9,17,dtype)
    return params,types,initial


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',choices=['vector_tmp','power_invariant','power_mutable','power_constant','power_float32','widths','boolean','casts','constant_casts','timed1','timed2','names','shared'],default='widths')
    parser.add_argument('--unsigned-char',action='store_true')
    args=parser.parse_args()
    if args.unsigned_char:
        from brian2.codegen.cpp_prefs import get_compiler_and_args
        b.prefs.codegen.cpp.extra_compile_args=[*get_compiler_and_args()[1],'-funsigned-char']
    directory=args.output;directory.mkdir(parents=True,exist_ok=True)
    b.set_device('cpp_standalone',directory=str(directory/'cpp'),build_on_run=False)
    b.prefs.devices.cpp_standalone.extra_make_args_unix=['-j2']
    b.prefs.core.default_integer_dtype=np.int64
    b.defaultclock.dt=.1*b.ms
    params,types,initial=case(args.case)
    neurons=len(next(iter(initial.values())))
    group=b.NeuronGroup(neurons,params['eqs'],threshold=params['eq_th'],reset=params['eq_rst'],
        refractory='rfc',method='linear',namespace=params,dtype=types)
    group.v=params['v_0'];group.g=0*b.mV;group.rfc=0*b.ms
    for name,values in initial.items():setattr(group,name,values)
    monitor=b.SpikeMonitor(group);network=b.Network(group,monitor)
    network.run(.1*b.ms,namespace={});network.run(.2*b.ms,namespace={})
    b.device.build(directory=str(directory/'cpp'),compile=True,run=True,debug=False)
    variables={}
    for name,variable in group.variables.items():
        if not hasattr(variable,'dtype'):continue
        condition=getattr(variable,'conditional_write',None)
        equation=group.equations[name] if name in group.equations else None
        variables[name]=dict(name=name,dtype=np.dtype(variable.dtype).name,
            dimensions=[float(x) for x in variable.dim._dims],scalar=variable.scalar,
            constant=variable.constant,read_only=variable.read_only,
            conditional_write=condition.name if condition is not None else None,
            flags=list(equation.flags) if equation is not None else [])
    names=[name for name,eq in group.equations.items() if eq.type!='subexpression']+['lastspike','not_refractory']
    np.savez(directory/'final.npz',**{name:group.variables[name].get_value().copy() for name in names},
        spike_i=np.asarray(monitor.i[:]).copy(),spike_t=np.asarray(monitor.t[:]).copy())
    runners=(group.state_updater,group.thresholder['spike'],group.resetter['spike'])
    metadata=dict(neurons=neurons,variables=variables,abstract=[runner.abstract_code for runner in runners],
        brian_version=b.__version__,temporary_dtype=np.dtype(b.prefs.core.default_float_dtype).name,
        target=type(runners[-1].codeobj).__name__,clock_timestep=int(group.clock.variables['timestep'].get_value()[0]),
        flags=list(__import__('brian2.codegen.cpp_prefs',fromlist=['get_compiler_and_args']).get_compiler_and_args()[1]))
    (directory/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for stage,runner in zip(('state','threshold','reset'),runners):
        (directory/(stage+'.cpp')).write_text(str(runner.codeobj.code.cpp_file))
    print(json.dumps(dict(target=metadata['target'],neurons=neurons,steps=3,output=str(directory))))


if __name__=='__main__':main()
