"""Deliver NumPy complex casts at the actual Brian array-store locations."""
from dataclasses import replace
import inspect
import re
import warnings

import numpy as np


MESSAGE='Casting complex values to real discards the imaginary part'


def python_store_error(kind):
    """Python complex scalars raise before NumPy can cast the destination."""
    if kind in ('i','u'):
        raise TypeError("int() argument must be a string, a bytes-like object or a real number, not 'complex'")
    raise TypeError("float() argument must be a string or a real number, not 'complex'")


def ufunc_error(name):
    from .expression_errors import COMPLEX_UFUNC_ERRORS
    raise TypeError(COMPLEX_UFUNC_ERRORS[name][1][1])


def bind(prepared,group):
    """Reuse actual CPU code-object locations and their warning registries."""
    random=prepared.program.random_program
    if random is None or not any(getattr(op,'function',None)=='numpy_complex_cast'
            for operations in getattr(random,'stages',{}).values() for op in operations):
        return prepared
    runners={'state':group.state_updater,'threshold':group.thresholder['spike'],
             'reset':group.resetter.get('spike')}
    sites={}
    for stage,operations in random.stages.items():
        casts=[op for op in operations if getattr(op,'function',None)=='numpy_complex_cast']
        names={op.result for op in casts if op.argument is None}
        if not casts:continue
        runner=runners.get(stage)
        for obj in getattr(runner,'code_objects',()):
            for op in casts:
                if op.argument is None:continue
                function=obj.namespace.get(op.argument)
                code=getattr(function,'__code__',None)
                if code is None:continue
                namespace=function.__globals__
                number=code.co_firstlineno
                if op.argument=='timestep':
                    source,start=inspect.getsourcelines(function)
                    number=next(start+i for i,line in enumerate(source)
                                if 'elapsed_steps = np.array(' in line)
                sites[stage,op.result]=(number,namespace,code.co_filename)
            source=str(obj.code.run).strip().splitlines()
            for name in names:
                pattern=r'^\s*_array_\w+\[[^\]]*\]\s*=\s*'+re.escape(name)+r'\s*$'
                for number,line in enumerate(source,1):
                    if re.match(pattern,line):
                        sites[stage,name]=(number,obj.namespace,'(string)')
    return replace(prepared,program=replace(prepared.program,complex_warning_sites=sites))


def warn_cast(prepared,stage,name):
    sites=prepared.program.complex_warning_sites
    site=None if sites is None else sites.get((stage,name))
    if site is None:
        # Direct configured-engine callers have no original Brian code object.
        warnings.warn(MESSAGE,np.ComplexWarning,stacklevel=2)
        return
    line,namespace,filename=site
    warnings.warn_explicit(MESSAGE,np.ComplexWarning,filename,line,
        module=namespace.get('__name__','<string>'),
        registry=namespace.setdefault('__warningregistry__',{}),module_globals=namespace)
