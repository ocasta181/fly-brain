"""Actual selected Cython compiler flags remain model and arithmetic identity."""
from hashlib import sha256
from pathlib import Path
import brian2 as b
import numpy as np
import pytest
from original_api_protocol import check_original_api_runs, observe_run
from test_mlx_compiled_integer_adapter import _typed_fixture,_forbid_cpu_steps


@pytest.mark.metal
@pytest.mark.parametrize('policy',['strict_signed_overflow','strict_float'])
def test_actual_cython_integer_flags_preserve_whole_selected_block(policy,monkeypatch):
    from brian2.codegen.cpp_prefs import get_compiler_and_args
    from brian2.codegen.runtime.cython_rt.extension_manager import cython_extension_manager
    base=list(get_compiler_and_args()[1])
    flags=([*base,'-fno-wrapv','-fstrict-overflow'] if policy=='strict_signed_overflow' else
           [flag for flag in base if flag!='-ffast-math']+['-ffp-contract=off'])
    monkeypatch.setitem(b.prefs,'codegen.cpp.extra_compile_args',flags)
    identity=sha256(repr(flags).encode()).hexdigest()[:16]
    monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(
        Path(__file__).resolve().parents[1]/'executions'/('cython-compiled-integer-flags-'+identity)))
    create_extension=cython_extension_manager.create_extension
    def extension(code,*args,**kwargs):
        return create_extension('# Frozen integer CPU flags: '+repr(flags)+'\n'+code,*args,**kwargs)
    monkeypatch.setattr(cython_extension_manager,'create_extension',extension)
    types=dict(x=np.dtype('int64'),y=np.dtype('int64'),flag=np.dtype('bool'))
    initial=dict(x=np.array([(1<<63)-1,-(1<<63),-3,7],np.int64),
        v=np.array([-.04,-.043,-.05,-.06])*b.volt,g=np.array([.001,.002,.003,.004])*b.volt)
    reset='y=x+x;flag=(x+1)>x'
    def exercise(fixture, mode):
        for duration in (.1,.2):
            yield observe_run(fixture,duration)
        if mode == 'cpu':
            assert tuple(fixture.neurons.resetter['spike'].codeobj.extra_compile_args)==tuple(flags)
    check_original_api_runs(lambda mode:_typed_fixture(types,reset,initial,default_reset=False),
        exercise,monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)
