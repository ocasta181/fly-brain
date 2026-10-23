"""Immutable identity for Brian's target-specific integer execution contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import ast
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sysconfig
import sys

import numpy as np


@lru_cache(maxsize=8)
def _compiler_identity(command: tuple[str, ...], stamp: tuple[int, int]) -> str:
    del stamp
    result = subprocess.run([*command, '--version'], capture_output=True, text=True, check=True)
    return result.stdout.strip()


@dataclass(frozen=True)
class IntegerPolicy:
    target: str
    requested_target: str
    compiler: tuple[str, ...]
    compiler_version: str
    flags: tuple[str, ...]
    python_flags: tuple[str, ...]
    architecture: str
    brian_version: str
    numpy_version: str
    cython_version: str
    numpy_simd: tuple[str, ...]
    vector_length: int | None = None
    trials: int | None = None
    loop_invariant_optimisations: bool = True
    source_identity: tuple[tuple[str, str], ...] = ()
    compiler_configuration: tuple[tuple[str, str], ...] = ()

    @property
    def identity(self):
        return asdict(self)

    @property
    def signature(self):
        return hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()

    @property
    def wraps_signed(self):
        # Python's extension build flags precede Brian's extra flags. Last
        # explicit overflow option wins, just as in the actual compiler call.
        options = self.python_flags + self.flags
        wrapping = False
        for flag in options:
            if flag in ('-fwrapv', '-fno-strict-overflow'):
                wrapping = True
            elif flag in ('-fno-wrapv', '-fstrict-overflow'):
                wrapping = False
        return wrapping

    @property
    def unsigned_char(self):
        result = False  # The frozen Apple arm64 compiler defaults to signed char.
        for flag in self.python_flags + self.flags:
            if flag == '-funsigned-char': result = True
            elif flag == '-fsigned-char': result = False
        return result

    def execution(self, neurons: int, trials: int):
        return replace(self, vector_length=int(neurons), trials=int(trials))

    @property
    def compiler_options(self):
        configuration=dict(self.compiler_configuration)
        options=['-I'+str(path) for path in ast.literal_eval(
            configuration.get('codegen.cpp.include_dirs','[]'))]
        for name,value in ast.literal_eval(configuration.get('codegen.cpp.define_macros','[]')):
            options.append('-D'+name+('='+str(value) if value is not None else ''))
        return tuple(options)


def snapshot_integer_policy(target=None):
    import brian2 as b
    import Cython
    from brian2.devices.device import get_device

    requested = str(target if target is not None else b.prefs.codegen.target)
    device = get_device()
    if target is None and type(device).__name__ == 'CPPStandaloneDevice':
        actual = 'cpp_standalone'
    elif requested == 'auto':
        actual = device.code_object_class().class_name
    else:
        actual = requested
    actual = {'cpp': 'cpp_standalone'}.get(actual, actual)
    if actual not in ('numpy', 'cython', 'cpp_standalone'):
        raise NotImplementedError(f'Integer semantics for Brian target {actual!r} are not lowered')
    command, version, flags, python_flags = (), '', (), ()
    if actual != 'numpy':
        command = tuple(shlex.split(os.environ.get('CXX') or sysconfig.get_config_var('CXX') or 'c++'))
        path = shutil.which(command[0])
        if path is None:
            raise RuntimeError(f'Brian CPU compiler {command[0]!r} is unavailable')
        command = (str(Path(path).resolve()), *command[1:])
        stat = Path(command[0]).stat()
        version = _compiler_identity(command, (stat.st_mtime_ns, stat.st_size))
        configured = b.prefs.codegen.cpp.extra_compile_args
        flags = tuple(configured if configured is not None else b.prefs.codegen.cpp.extra_compile_args_gcc)
        if actual == 'cython':
            python_flags = tuple(shlex.split(sysconfig.get_config_var('CFLAGS') or '')+
                                 shlex.split(os.environ.get('CFLAGS', '')))
    features = np.core._multiarray_umath.__cpu_features__
    policy = IntegerPolicy(actual, requested, command, version, flags, python_flags,
                         platform.machine(), b.__version__, np.__version__, Cython.__version__,
                         tuple(sorted(name for name, enabled in features.items() if enabled)))
    files = {'numpy_binary': Path(np.core._multiarray_umath.__file__)}
    base = Path(b.__file__).resolve().parent
    for name, relative in (
        ('cpp_rules', 'codegen/generators/cpp_generator.py'),
        ('cython_rules', 'codegen/generators/cython_generator.py'),
        ('cython_runtime', 'codegen/runtime/cython_rt/cython_rt.py'),
        ('cython_extension_manager', 'codegen/runtime/cython_rt/extension_manager.py'),
        ('numpy_rules', 'codegen/generators/numpy_generator.py'),
        ('cython_int_overloads', 'codegen/runtime/cython_rt/templates/common_group.pyx')):
        files[name] = base/relative
    files['integer_cast_header'] = base/'synapses/stdint_compat.h'
    sources = tuple((name, _source_digest(str(path), path.stat().st_mtime_ns, path.stat().st_size))
                    for name, path in sorted(files.items()))
    configuration = tuple((name, repr(b.prefs[name])) for name in (
        'codegen.cpp.compiler', 'codegen.cpp.include_dirs', 'codegen.cpp.define_macros',
        'codegen.generators.cpp.restrict_keyword', 'codegen.generators.cpp.flush_denormals'))
    configuration+=(('cpython_integer_runtime',repr((platform.python_implementation(),
        platform.python_version(),sys.int_info.bits_per_digit,sys.int_info.sizeof_digit))),)
    return replace(policy, loop_invariant_optimisations=bool(b.prefs.codegen.loop_invariant_optimisations),
                   source_identity=sources, compiler_configuration=configuration)


@lru_cache(maxsize=16)
def _source_digest(path, modified, size):
    del modified, size
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


