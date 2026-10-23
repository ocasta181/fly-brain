"""Reuse stock Clang's arithmetic order without executing a CPU simulation."""
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
import ast,hashlib,json,os,pickle,platform,re,shlex,subprocess,sys,tempfile
import fcntl


@lru_cache(maxsize=8)
def _compiler_identity(command):
    return subprocess.check_output([*command,'--version'],text=True).strip()


def _preferences():
    import brian2 as b
    return tuple((key,tuple(b.prefs[key])if isinstance(b.prefs[key],list)else b.prefs[key])for key in(
        'codegen.cpp.compiler','codegen.cpp.extra_compile_args_gcc','codegen.cpp.extra_compile_args_msvc',
        'codegen.cpp.define_macros','codegen.cpp.include_dirs','codegen.cpp.headers'))


def _compile_environment():
    return {name:os.environ.get(name)for name in('CC','CXX','CFLAGS','CPPFLAGS','SDKROOT','MACOSX_DEPLOYMENT_TARGET')}


def capture(program,model,neurons,stage,cxx,prefs):
    """Capture one unchanged, zero-step reference codeobject, privately."""
    from .compiler import UnsupportedModelError
    import brian2,numpy,Cython
    worker=Path(__file__).with_name('expression_profile_worker.py')
    identity=dict(model=model.signature,neurons=neurons,target=model.random_target,stage=stage,
        cxx=cxx,compiler=_compiler_identity(cxx),preferences=prefs,platform=platform.platform(),
        environment=_compile_environment(),
        worker=hashlib.sha256(worker.read_bytes()).hexdigest(),generation_schema=4,
        versions=dict(brian=brian2.__version__,numpy=numpy.__version__,cython=Cython.__version__))
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    directory=Path(os.environ.get('EON_MLX_EXPRESSION_CACHE',str(Path(tempfile.gettempdir())/'eon-mlx-expression-profiles')))/key
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'profile.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (directory/'metadata.json').exists():
            request=dict(directory=str(directory.resolve()),target=model.random_target,stage=stage,
                neurons=neurons,eqs=model.eqs,threshold=model.eq_th,reset=model.eq_rst,
                namespace=dict(model.source_namespace),namespace_dimensions=dict(model.namespace_dimensions),
                namespace_inputs={name:array.source_name for name,array in program.input_arrays.items()},
                dtype={name:v.dtype for name,v in program.variables.items()},
                dt=model.dt_s,writes=program.random_program.compiled_stages[stage].writes,
                cxx=cxx,compiler=identity['compiler'],preferences=dict(prefs))
            for name,value in tuple(request['preferences'].items()):
                if name!='codegen.cpp.compiler'and isinstance(value,tuple):request['preferences'][name]=list(value)
            request_file=directory/'request.pkl';request_file.write_bytes(pickle.dumps(request))
            environment=dict(os.environ);environment['PYTHONPATH']=str(Path(__file__).resolve().parent.parent)+os.pathsep+environment.get('PYTHONPATH','')
            with (directory/'generation.log').open('w')as log:
                result=subprocess.run([sys.executable,'-m','mlx_backend.expression_profile_worker',str(request_file)],
                    env=environment,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise UnsupportedModelError(f'Compiled arithmetic code generation failed; see {directory}/generation.log')
    metadata=json.loads((directory/'metadata.json').read_text());metadata['identity']=identity
    metadata['reserved_names']=tuple(program.variables)
    return (directory/(stage+'.mir')).read_text(),metadata


def apply(program,model,neurons):
    random=program.random_program
    if random is None or random.target not in('cpp','cython'):return program
    from .expression_lowering import Segment
    written={name for op in random.stages.get('state',())if isinstance(op,Segment)for name in op.writes}
    equation=any(name in program.variables and program.variables[name].persistent
               and program.variables[name].dtype=='float64'for name in written)
    def compound(statements):
        dependent=set()
        for statement in statements:
            node=ast.parse(statement.expr,mode='eval').body
            random_call=any(isinstance(item,ast.Call)and isinstance(item.func,ast.Name)
                and item.func.id in('rand','randn','poisson')for item in ast.walk(node))
            reads={item.id for item in ast.walk(node)if isinstance(item,ast.Name)}
            if statement.op not in('=',':='):reads.add(statement.var)
            if reads&dependent and any(isinstance(item,ast.BinOp)for item in ast.walk(node)):return True
            if random_call and statement.op not in('=',':='):return True
            for part in ast.walk(node):
                if isinstance(part,ast.BinOp)and any(isinstance(item,ast.Call)and isinstance(item.func,ast.Name)
                        and item.func.id in('rand','randn','poisson')for item in ast.walk(part)):return True
            if random_call or reads&dependent:dependent.add(statement.var)
            elif statement.op in('=',':='):dependent.discard(statement.var)
        return False
    callers=[name for name in random.compiled_stages if compound(
        sum(random.statements[name],())if random.statements is not None else getattr(program,name+'_statements',()))]
    if not equation and not callers:return program
    cxx=tuple(shlex.split(os.environ.get('CXX','c++')))
    prefs=_preferences()
    if equation:program=_apply(program,model,neurons,cxx,prefs,tuple(_compile_environment().items()))
    if callers:
        from .expression_profile_caller import lower
        from .compiler import UnsupportedModelError
        compiled=dict(program.random_program.compiled_stages)
        for stage in callers:
            text,metadata=capture(program,model,neurons,stage,cxx,prefs)
            metadata['metal_names']={name:variable.code_name for name,variable in program.variables.items()
                                     if variable.persistent}
            try:compiled[stage]=lower(text,metadata,compiled[stage])
            except Exception as exc:raise UnsupportedModelError('Compiled random caller arithmetic: '+str(exc))from exc
        if any('f_operator_from_uint'in part.scalar+part.vector for part in compiled.values()):
            from .operator32 import MSL_UINT_CAST_HEADER
            program=replace(program,header_code=program.header_code+MSL_UINT_CAST_HEADER)
        from .expression_lowering import Draw,Segment
        stages=dict(program.random_program.stages)
        serial=set(program.random_program.serial_required)
        for name in callers:
            ops=stages[name];lowered=compiled[name]
            # Keep vector generation when a single unconditional leaf is
            # followed by one pure consumer. Only that consumer is replaced.
            draws=list(re.finditer(r'const ulong (\w+)=compiled_(?:rand|normal)\(_compiled_rng\);',lowered.vector))
            vector=False
            if (len(ops)==2 and isinstance(ops[0],Draw)and isinstance(ops[1],Segment)
                    and ops[0].function in('rand','randn')and not ops[0].scalar
                    and ops[0].guard is None and not lowered.scalar and len(draws)==1):
                sample=draws[0]
                source=lowered.vector[:sample.start()]+lowered.vector[sample.end():]
                from brian2.utils.stringtools import word_substitute
                source=word_substitute(source,{sample[1]:program.random_program.variables[ops[0].result].code_name or ops[0].result})
                if '_compiled_rng'not in source:
                    stages[name]=(ops[0],replace(ops[1],source=source))
                    vector=True
            if not vector:serial.add(name)
        program=replace(program,random_program=replace(program.random_program,
            stages=MappingProxyType(stages),compiled_stages=MappingProxyType(compiled),
            serial_required=tuple(sorted(serial))))
    return program


@lru_cache(maxsize=32)
def _apply(program,model,neurons,cxx,prefs,environment):
    from .expression_profile_mir import lower
    from .expression_lowering import Segment
    from .compiler import UnsupportedModelError
    compiler=_compiler_identity(cxx)
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(
        Path(__file__),Path(__file__).with_name('expression_profile_mir.py'),Path(__file__).with_name('expression_profile_worker.py'))}
    import brian2,numpy,Cython
    identity=dict(model=model.signature,neurons=neurons,target=model.random_target,cxx=cxx,
        compiler=compiler,preferences=prefs,platform=platform.platform(),sources=sources,
        environment=_compile_environment(),
        versions=dict(brian=brian2.__version__,numpy=numpy.__version__,cython=Cython.__version__))
    generation=dict(identity,sources={'worker':sources['expression_profile_worker.py']},generation_schema=3)
    key=hashlib.sha256(json.dumps(generation,sort_keys=True).encode()).hexdigest()
    directory=Path(os.environ.get('EON_MLX_EXPRESSION_CACHE',str(Path(tempfile.gettempdir())/'eon-mlx-expression-profiles')))/key
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'profile.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        saved=directory/('profile-'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()+'.json')
        if saved.exists():profile=json.loads(saved.read_text())
        else:
            request=dict(directory=str(directory.resolve()),target=model.random_target,neurons=neurons,
                eqs=model.eqs,threshold=model.eq_th,reset=model.eq_rst,
                namespace=dict(model.source_namespace),namespace_dimensions=dict(model.namespace_dimensions),
                namespace_inputs={name:array.source_name for name,array in program.input_arrays.items()},
                dtype={name:v.dtype for name,v in program.variables.items()},dt=model.dt_s,
                writes=program.random_program.compiled_stages['state'].writes,cxx=cxx,compiler=compiler,
                preferences=dict(prefs))
            # Brian preference list values remain lists, including macro pairs.
            for name,value in tuple(request['preferences'].items()):
                if name!='codegen.cpp.compiler'and isinstance(value,tuple):request['preferences'][name]=list(value)
            request_file=directory/'request.pkl';request_file.write_bytes(pickle.dumps(request))
            environment=dict(os.environ);environment['PYTHONPATH']=str(Path(__file__).resolve().parent.parent)+os.pathsep+environment.get('PYTHONPATH','')
            if not (directory/'metadata.json').exists():
                with (directory/'generation.log').open('w')as log:
                    result=subprocess.run([sys.executable,'-m','mlx_backend.expression_profile_worker',str(request_file.resolve())],
                        env=environment,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:raise UnsupportedModelError(f'Compiled arithmetic code generation failed; see {directory}/generation.log')
            metadata=json.loads((directory/'metadata.json').read_text())
            import numpy as np
            fixed={'g':0}
            for name,value in (('v',model.v_0),('rfc',model.t_rfc)):
                if (name in program.variables and program.variables[name].dtype=='float64'
                        and np.ndim(value)==0 and not isinstance(value,str)):
                    fixed[name]=int(np.asarray(value,np.float64).view(np.uint64))
            try:profile=lower((directory/'state.mir').read_text(),metadata,fixed_values=fixed)
            except Exception as exc:raise UnsupportedModelError(f'Compiled arithmetic lowering failed: {exc}; profile {directory}')from exc
            profile['identity']=identity
            temporary=saved.with_suffix('.tmp');temporary.write_text(json.dumps(profile,indent=2));temporary.replace(saved)
    random=program.random_program;stages=dict(random.stages)
    stages['state']=(Segment(profile['source'],tuple(profile['writes']),False,profile['may_raise']),)
    from .expression_compiled import CompiledStage
    compiled=dict(random.compiled_stages)
    compiled['state']=CompiledStage('',profile['source'],tuple(profile['writes']))
    # The pure state profile owns its ordered arithmetic and error-prefix
    # commit. It does not use the original caller's cooperative draw stage.
    return replace(program,random_program=replace(random,stages=MappingProxyType(stages),
        compiled_stages=MappingProxyType(compiled),serial_required=tuple(s for s in random.serial_required if s!='state')))
