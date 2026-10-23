"""Isolated, zero-step Brian code generation for compiled arithmetic profiles."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,os,pickle,shlex,subprocess,sys,sysconfig


def main(request_file):
    import numpy as np
    import brian2 as b
    from .model import ResolvedModel
    from brian2.codegen.cpp_prefs import get_compiler_and_args
    from brian2.core.variables import ArrayVariable
    request=pickle.loads(Path(request_file).read_bytes());root=Path(request['directory'])
    # Transport immutable numeric snapshots, not TimedArray's local callables.
    # Reuse the canonical reconstruction so units, dtypes and lookup data agree.
    namespace=ResolvedModel.brian_namespace(SimpleNamespace(
        source_namespace=request['namespace'],namespace_dimensions=request['namespace_dimensions']))
    target=request['target'];b.prefs.codegen.runtime.cython.cache_dir=str(root/'cython-default')
    b.prefs.codegen.runtime.cython.delete_source_files=False
    if target=='cpp':b.set_device('cpp_standalone',build_on_run=False)
    else:b.set_device('runtime');b.prefs.codegen.target='cython'
    for key,value in request['preferences'].items():b.prefs[key]=value
    group=b.NeuronGroup(request['neurons'],request['eqs'],method='linear',threshold=request['threshold'],
        reset=request['reset'],refractory='rfc',namespace=namespace,dtype={n:np.dtype(v)for n,v in request['dtype'].items()},
        clock=b.Clock(dt=request['dt']*b.second,name='expression_profile_clock'),name='expression_profile_group')
    # Code generation only. No CPU timestep, state update or random draw runs.
    b.Network(group).run(0*b.second,namespace={})
    stage=request.get('stage','state')
    runner={'state':group.state_updater,'reset':group.resetter['spike'],
        'threshold':group.thresholder['spike'],'subexpression':group.subexpression_updater}[stage]
    device=b.get_device();codeobj=runner.codeobj
    arrays={};constants={};namespace_arrays={}
    for binding,name in request.get('namespace_inputs',{}).items():
        value=namespace.get(name)
        if isinstance(value,b.TimedArray):
            namespace_arrays['_namespace'+value.name+'_values']=dict(name=binding,
                dtype='float64',shape=value.values.shape)
    for name,variable in codeobj.variables.items():
        if isinstance(variable,ArrayVariable):
            arrays[device.get_array_name(variable)]=dict(name=name,dtype=np.dtype(variable.dtype).name,scalar=variable.scalar)
        elif getattr(variable,'scalar',False)and getattr(variable,'constant',False):
            try:value=np.asarray(variable.get_value())
            except (TypeError,AttributeError):continue
            if value.size!=1 or value.dtype.kind not in'biuf':continue
            scalar=value.reshape(-1)[0]
            constants[name]=dict(kind='fp'if value.dtype.kind=='f'else'int',
                raw=int(np.asarray(float(scalar),np.float64).view(np.uint64))if value.dtype.kind=='f'else int(scalar))
    constants['dt']=dict(kind='fp',raw=int(np.asarray(request['dt'],np.float64).view(np.uint64)))
    constants['N']=dict(kind='int',raw=request['neurons'])
    if target=='cpp':
        build=root/'cpp-default';device.build(directory=str(build),compile=False,run=False,with_output=False)
        source=build/'code_objects'/(codeobj.name+'.cpp');_,flags=get_compiler_and_args();includes=[str(build)]
    else:
        import brian2.synapses
        module=codeobj.compiled_code['run'];source=root/'cython-default'/(module.__name__+'.cpp')
        flags=(shlex.split(sysconfig.get_config_var('CFLAGS')or'')
            +shlex.split(os.environ.get('CFLAGS',''))+shlex.split(os.environ.get('CPPFLAGS',''))+codeobj.extra_compile_args)
        includes=[*codeobj.include_dirs,np.get_include(),sysconfig.get_path('include'),str(Path(brian2.synapses.__file__).parent)]
    output=root/(stage+'.mir')
    defines=['-D'+name+('='+str(value)if value is not None else'')for name,value in b.prefs.codegen.cpp.define_macros]
    command=[*request['cxx'],*flags,*defines,*('-I'+p for p in includes),'-S','-mllvm','-stop-after=machine-combiner',str(source),'-o',str(output)]
    subprocess.run(command,check=True,capture_output=True)
    from .compiled_cache import atomic_write
    atomic_write(root/'metadata.json',json.dumps(dict(target=target,arrays=arrays,constants=constants,
        namespace_arrays=namespace_arrays,
        source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),command=command,
        neurons=request['neurons'],writes=request['writes'],compiler=request['compiler'],
        function=codeobj.name,stage=stage),indent=2))


if __name__=='__main__':main(sys.argv[1])
