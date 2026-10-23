"""Independent Brian CPU references for focused native component regressions."""
from functools import partial
from hashlib import sha256
from time import perf_counter

import numpy as np

def component_inputs(neurons, steps, model):
    from mlx_backend.models import Connectome, Stimulus
    empty = np.array([], np.int32)
    graph = Connectome(np.arange(neurons), empty, empty, np.array([], np.float64),
                       np.array([], np.float64))
    events = np.zeros((1, steps, 0), np.uint8)
    return graph, Stimulus(events, (), (), (0,), 0, 0,
                          sha256(events.tobytes()).hexdigest(), model.dt_ms)


class ComponentCpu:
    """Independent Brian neurons with recording off and default compiler flags."""

    def __init__(self, count, steps, model, target, cache, *, prefix="stable_component"):
        import brian2 as b
        self.b, self.model, self.target, self.cache, self.steps = b, model, target, cache, steps
        self.activate()
        begin = perf_counter()
        clock = b.Clock(dt=model.dt_s*b.second, name=prefix+"_clock")
        dtype=np.dtype(model.state_dtype)
        if (model.state_dtype == "float32" and model.analysis is not None
                and model.analysis.variables["lastspike"].dtype == "float64"):
            dtype={name:np.dtype(var.dtype) for name,var in model.analysis.variables.items()}
        self.neurons = b.NeuronGroup(count, model.eqs, method="linear", threshold=model.eq_th,
            reset=model.eq_rst, refractory="rfc", namespace=model.brian_namespace(),
            clock=clock, dtype=dtype, name=prefix+"_neurons")
        self.neurons.v = model.v_0*b.volt
        self.neurons.g = 0*b.volt
        self.neurons.rfc = 0*b.second
        self.network = b.Network(self.neurons)
        self.network.schedule = list(model.schedule)
        self.network.store("initial")
        self.setup_s = perf_counter()-begin
        begin = perf_counter()
        self.network.run(0*b.second, namespace={})
        self.prepare_codegen_s = perf_counter()-begin
        self.code_object_types = sorted({code.__class__.__name__
            for obj in self.network.sorted_objects for code in obj.code_objects})
        self.names = tuple(name for name, eq in self.neurons.equations.items()
                           if eq.type != "subexpression")+("lastspike", "not_refractory")

    def activate(self):
        b = self.b
        b.prefs.codegen.target = self.target
        b.prefs.core.default_float_dtype = np.dtype(self.model.state_dtype).type
        b.prefs.core.default_integer_dtype = np.dtype(self.model.integer_dtype).type
        b.prefs.legacy.refractory_timing = self.model.legacy_refractory_timing
        b.prefs.codegen.runtime.cython.cache_dir = str(self.cache)

    def run(self):
        begin = perf_counter()
        self.activate()
        restore_begin = perf_counter()
        self.network.restore("initial")
        reset_s = perf_counter()-restore_begin
        simulation_begin = perf_counter()
        self.network.run(self.steps*self.model.dt_s*self.b.second, namespace={})
        simulation_s = perf_counter()-simulation_begin
        total = perf_counter()-begin
        state = {name: np.asarray(getattr(self.neurons, name+"_")[:]).copy().reshape(1, -1)
                 for name in self.names}
        return {"reset_s": reset_s, "simulation_wall_s": simulation_s,
            "warm_total_s": total}, np.empty((0, 2), np.int64), state


PackedComponentCpu = partial(ComponentCpu, prefix="packed_component")
HyperbolicComponentCpu = partial(ComponentCpu, prefix="hyperbolic_component")


def stable_model_configuration():
    """Every reset computes three observable functions with repeated input blocks."""
    import brian2 as b
    from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
    return resolve_model({"eqs": DEFAULT_EQUATIONS+
        "activity_expm1:1\nactivity_log1p:1\nactivity_exprel:1\nphase:integer\n",
        "eq_th": "True", "t_rfc": 0*b.ms,
        "eq_rst": "v=v_rst;g=0*mV"
        ";activity_expm1=expm1((i-int(i/4096)*4096+phase+1)/16384)"
        ";activity_log1p=log1p((i-int(i/4096)*4096+phase+1)/16384)"
        ";activity_exprel=exprel((i-int(i/4096)*4096+phase+1)/16384);phase+=1"})


def _packed_base_configuration():
    import brian2 as b
    from mlx_backend.model import DEFAULT_EQUATIONS, DEFAULT_RESET, resolve_model
    extra = "".join(f"extra{j}:1\ncount{j}:integer\nflag{j}:boolean\n" for j in range(12))
    reset = DEFAULT_RESET + ";" + ";".join(
        f"extra{j}+={j+1}/1024 + int(flag{j})/2;count{j}+={j+1};flag{j}=not flag{j}"
        for j in range(12))
    return resolve_model({"eqs": DEFAULT_EQUATIONS + extra, "eq_rst": reset})


def packed_model_configuration():
    """36 mixed-type per-step observer writes;42 persistent states exceed 31 bindings."""
    import brian2 as b
    from mlx_backend.model import resolve_model
    original = _packed_base_configuration()
    return resolve_model({"eqs": original.eqs+"phase:integer\n", "eq_th": "True",
        "eq_rst": original.eq_rst+";phase+=1", "t_rfc": 0*b.ms})
