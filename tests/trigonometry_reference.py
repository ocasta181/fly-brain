"""Independent original Eon/Brian trigonometric reset execution and probes."""
from native_execution import forbid_cpu_steps as guard_cpu_steps
from hashlib import sha256
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS = ("sin", "cos", "tan", "arcsin", "arccos", "arctan")


def sine_inputs():
    centers = np.array([0., -0., 1., -1., np.pi/4, -np.pi/4, np.pi/2,
                        -np.pi/2, np.pi, -np.pi, 2*np.pi, 2**20, 2**50,
                        2**100, 1e300, np.finfo(float).max], dtype=np.float64)
    values = np.r_[np.nextafter(centers, -np.inf), centers, np.nextafter(centers, np.inf),
                   np.linspace(-np.pi, np.pi, 257), np.geomspace(1e-320, 1e-10, 33),
                   -np.geomspace(1e-320, 1e-10, 33), np.inf, -np.inf]
    payloads = np.array([0x7ff8000000000000, 0x7ff8000000000123, 0xfff8000000000123,
                         0x7ff0000000000001], np.uint64).view(np.float64)
    return np.r_[values, payloads]


def original_reference(target, function, inputs, directory, *, state_inputs=False, dtype=None, phase_dtype=None, runtime_device=None, run_steps=(1,), forbid_cpu_steps=False, threshold_expression=None, cache_suffix="", argument_expression=None, namespace_values=None,result_expression=None,optimise=None,phase_symbol="phase",integer_dtype=None):
    """Call upstream create_model; retain actual codegen target and raw output."""
    import brian2 as b
    from brian2.devices.device import get_device,reset_device
    from brian2.codegen.cpp_prefs import get_compiler_and_args
    import pandas as pd

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    previous_device, previous_target, previous_dt = get_device(), b.prefs.codegen.target, b.defaultclock.dt
    previous_cache = b.prefs.codegen.runtime.cython.cache_dir
    previous_make = b.prefs.devices.cpp_standalone.extra_make_args_unix
    previous_dtype = b.prefs.core.default_float_dtype
    previous_integer_dtype=b.prefs.core.default_integer_dtype
    previous_optimise=b.prefs.codegen.loop_invariant_optimisations
    source = ROOT/"code/paper-phil-drosophila/model.py"
    spec = importlib.util.spec_from_file_location("original_trigonometry", source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    pd.DataFrame(index=np.arange(len(inputs))).to_csv(directory/"neurons.csv")
    pd.DataFrame({"Presynaptic_Index": [0], "Postsynaptic_Index": [0],
                  "Excitatory x Connectivity": [0.]}).to_parquet(directory/"edges.parquet")
    try:
        if target == "cpp":
            b.set_device("cpp_standalone", directory=str(directory/"cpp"), build_on_run=True)
            b.prefs.devices.cpp_standalone.extra_make_args_unix = ["-j2"]
        else:
            b.set_device("runtime" if runtime_device is None else runtime_device)
            b.prefs.codegen.target = target
        if dtype is not None:
            b.prefs.core.default_float_dtype = np.dtype(dtype).type
        if integer_dtype is not None:b.prefs.core.default_integer_dtype=np.dtype(integer_dtype).type
        if optimise is not None:b.prefs.codegen.loop_invariant_optimisations=optimise
        cache_name="cython-trigonometry-default"+("-"+np.dtype(dtype).name if dtype is not None else "")
        b.prefs.codegen.runtime.cython.cache_dir = str(ROOT/"executions"/(cache_name+cache_suffix))
        b.defaultclock.dt = .1*b.ms
        params = dict(original.default_params)
        expression=argument_expression if argument_expression is not None else (phase_symbol if state_inputs else "probe(0*ms,i)")
        phase_declaration=phase_symbol+" : "+(phase_dtype or "1")+"\n" if state_inputs else ""
        params.update(eqs=params["eqs"]+"sample : 1\n"+phase_declaration, eq_th=threshold_expression or "t>=0*ms",
                      eq_rst="v=v_rst; sample="+(result_expression or f"{function}({expression})"), t_rfc=0*b.ms)
        params.update(namespace_values or {})
        if not state_inputs:
            params["probe"]=b.TimedArray(np.asarray(inputs)[None, :],dt=b.defaultclock.dt)
        neurons, synapses, monitor = original.create_model(directory/"neurons.csv", directory/"edges.parquet", params)
        if dtype is not None:
            for name in ("v", "g", "rfc", "sample"):
                assert np.dtype(neurons.variables[name].dtype) == np.dtype(dtype), name
        if state_inputs:
            setattr(neurons,phase_symbol,np.asarray(inputs))
        network = b.Network(neurons, synapses, monitor)
        if forbid_cpu_steps:
            guard_cpu_steps(network, None, 'Native trigonometry executed a user CPU timestep')
        for count in run_steps:
            network.run(count*b.defaultclock.dt, namespace={})
        actual = np.asarray(neurons.sample_[:]).copy()
        codeobj = neurons.resetter["spike"].codeobj
        compiler, compiler_args = get_compiler_and_args()
        (directory/"reset-source.txt").write_text(str(codeobj.code))
        (directory/"threshold-source.txt").write_text(str(neurons.thresholder["spike"].codeobj.code))
        report = {"target_requested": target, "target_resolved": type(codeobj).__name__,
                  "function": function, "inputs": len(inputs),
                  "loop_optimisations":bool(b.prefs.codegen.loop_invariant_optimisations),
                  "input_dtype":np.asarray(inputs).dtype.name,"state_dtype":actual.dtype.name,
                  "input_mode":"state" if state_inputs else "timedarray",
                  "state_bits":{name:sha256(np.asarray(neurons.variables[name].get_value()).tobytes()).hexdigest()
                                for name in (("v","g","rfc","lastspike","not_refractory","sample")
                                             + ((phase_symbol,) if state_inputs else ()))},
                  "input_sha256": sha256(np.asarray(inputs).tobytes()).hexdigest(),
                  "output_sha256": sha256(actual.tobytes()).hexdigest(),
                  "compiler": compiler, "compiler_args": compiler_args,
                  "steps": np.rint(np.asarray(monitor.t_)/float(b.defaultclock.dt)).astype(np.int64).tolist(),
                  "neurons": np.asarray(monitor.i[:]).astype(np.int64).tolist()}
        if target == "cython":
            report["compiled_module"] = getattr(codeobj.compiled_code["run"], "__file__", None)
            if runtime_device is None:
                report["threshold_module"] = neurons.thresholder["spike"].codeobj.compiled_code["run"].__file__
        return actual, report
    finally:
        if target == "cpp":
            get_device().reinit()
        reset_device()
        assert get_device() is previous_device
        b.prefs.codegen.target, b.defaultclock.dt = previous_target, previous_dt
        b.prefs.codegen.runtime.cython.cache_dir = previous_cache
        b.prefs.devices.cpp_standalone.extra_make_args_unix = previous_make
        b.prefs.core.default_float_dtype = previous_dtype
        b.prefs.core.default_integer_dtype=previous_integer_dtype
        b.prefs.codegen.loop_invariant_optimisations=previous_optimise
