"""Actual CPU target rounding when int(x) is stored back in float state."""
import json
import subprocess

import numpy as np
import pytest


@pytest.mark.metal
@pytest.mark.parametrize("target", ["numpy", "cython"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
def test_original_int_float_storage(tmp_path, monkeypatch, target, dtype):
    import brian2 as b
    from original_api_protocol import check_original_api_runs, observe_run
    from original_api_reference import build_original_api_fixture

    def build(mode):
        return build_original_api_fixture(
            tmp_path/str(mode == 'native'), parameters=dict(
                eqs="v:volt\ng:volt\nrfc:second\nx:1\na:1",
                eq_th="True", eq_rst="a=int(x);x=-x;v=v_rst;g=0*mV",
                t_rfc=0*b.ms, t_dly=.2*b.ms),
            initial_state={"x": [-.75, -.125, .125, .75]},
            exc=(), exc2=(), silenced=(),
        )
    def exercise(fixture, mode):
        captures = [observe_run(fixture,duration,rng=True) for duration in (.1,.2,0.,.1)]
        codeobj = fixture.neurons.resetter["spike"].codeobj
        (tmp_path / ("native-reset.txt" if mode == 'native' else "cpu-reset.txt")).write_text(str(codeobj.code))
        if mode == 'cpu' and target == "cython":
            module = codeobj.compiled_code["run"].__file__
            (tmp_path / "cpu-module.json").write_text(json.dumps({"module": module}))
            (tmp_path / "cpu-assembly.txt").write_text(subprocess.check_output(["otool", "-tvV", module], text=True))
        yield from captures
    check_original_api_runs(build,exercise,monkeypatch,seed=953,target=target,float_dtype=dtype)


@pytest.mark.metal
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("mode", ["roundtrip", "integer_use", "strict"])
def test_roundtrip_boundaries_and_integer_use(tmp_path, monkeypatch, dtype, mode):
    import brian2 as b
    from brian2.codegen.cpp_prefs import get_compiler_and_args
    from brian2.codegen.runtime.cython_rt.extension_manager import cython_extension_manager
    from original_api_protocol import check_original_api_runs, observe_run
    from original_api_reference import build_original_api_fixture
    if mode == "strict":
        flags = [flag for flag in get_compiler_and_args()[1] if flag != '-ffast-math'] + ['-ffp-contract=off']
        monkeypatch.setitem(b.prefs, 'codegen.cpp.extra_compile_args', flags)
        create = cython_extension_manager.create_extension
        monkeypatch.setattr(cython_extension_manager, 'create_extension',
                            lambda code,*args,**kwargs: create('# strict roundtrip flags\n'+code,*args,**kwargs))
    tiny = np.nextafter(dtype(0), dtype(1), dtype=dtype)
    cases = [
        [-0., 0., -tiny, tiny],
        [np.nextafter(dtype(-1), dtype(0)), -1., np.nextafter(dtype(1), dtype(0)), 1.],
        [dtype(-(2**31)), np.nextafter(dtype(-(2**31)),dtype(-np.inf)), dtype(2**31), np.nextafter(dtype(2**31),dtype(0))],
        [-np.inf, np.inf, np.nan, -np.nan],
        [-.75, -.125, .125, .75],
    ]
    def build(execution_mode):
        return build_original_api_fixture(
            tmp_path/str(execution_mode == 'native'), parameters=dict(
                eqs='v:volt\ng:volt\nrfc:second\nx:1\na:1\nq:integer',
                eq_th='True', eq_rst='a=int(x);'+('q=int(x);' if mode=='integer_use' else '')+'v=v_rst;g=0*mV',
                t_rfc=0*b.ms, t_dly=.2*b.ms), exc=(), exc2=(), silenced=(),
        )
    def exercise(fixture, execution_mode):
        captures=[]
        for values in cases:
            fixture.neurons.x=np.asarray(values,dtype=dtype)
            captures.append(observe_run(fixture,.1,rng=True))
        if execution_mode == 'cpu':
            codeobj=fixture.neurons.resetter['spike'].codeobj
            module=codeobj.compiled_code['run'].__file__
            (tmp_path/'cpu-module.json').write_text(json.dumps({'module':module,'flags':list(codeobj.extra_compile_args)}))
            (tmp_path/'cpu-assembly.txt').write_text(subprocess.check_output(['otool','-tvV',module],text=True))
        yield from captures
    check_original_api_runs(build,exercise,monkeypatch,seed=955,target='cython',float_dtype=dtype)
