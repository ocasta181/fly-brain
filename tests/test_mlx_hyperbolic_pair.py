"""Independent exact checks for same-operand sinh/cosh reuse."""
from state_assertions import assert_recorded_spikes_equal
from native_execution import forbidden_call, require_metal
import numpy as np
import pytest


def test_pair_render_preserves_missing_dependency_collector_error():
    from mlx_backend.clang32 import Node, render
    from mlx_backend.compiler import UnsupportedModelError
    argument = Node("argument", "float64", value="x")
    outputs = {name: Node("call:"+name, "float64", (argument,))
               for name in ("sinh", "cosh")}
    with pytest.raises(UnsupportedModelError, match="missing math dependency collector"):
        render(outputs)
    used = set()
    assert render(outputs, used).count("_eon_hyper_pair(") == 1
    assert used == {"sinh", "cosh", "_hyperbolic_pair"}


def test_pair_plan_keeps_ssa_width_scope_and_mutation_identity():
    from mlx_backend.clang32 import Node
    from mlx_backend.hyperbolic_pair import SsaPairs
    x, y = Node("argument", "float64"), Node("argument", "float64")
    s, c = Node("call:sinh", "float64", (x,)), Node("call:cosh", "float64", (x,))
    used, lines = set(), []
    plan = SsaPairs({"s": s, "c": c}, used)
    assert plan.expression(s, lambda arg: "x", lines) == "_eon_hyper_pair_0.sinh"
    assert plan.expression(c, lambda arg: "x", lines) == "_eon_hyper_pair_0.cosh"
    assert len(lines) == 1
    assert used == {"sinh", "cosh", "_hyperbolic_pair"}
    # Equal-looking syntax is insufficient: an intervening state write
    # supplies a different actual SSA operand, and cannot share the call.
    assert not SsaPairs({"s": s, "c": Node("call:cosh", "float64", (y,))}, set()).nodes
    assert not SsaPairs({"s": s}, set()).nodes
    assert not SsaPairs({"s": Node("call:sinh", "float32", (x,)),
        "c": Node("call:cosh", "float32", (x,))}, set()).nodes


def test_mixed_cfg_pairs_only_same_block_selected_double_operand():
    from mlx_backend.hyperbolic_pair import IrPairs
    s = "%s = tail call reassoc nsz arcp contract afn double @llvm.sinh.f64(double %x), !dbg !2088"
    c = "%c = tail call reassoc nsz arcp contract afn double @llvm.cosh.f64(double %x), !dbg !2089"
    used, lines = set(), []
    plan = IrPairs((s, c), used)
    for function in ("sinh", "cosh"):
        assert plan.expression("llvm."+function+".f64", "double", [("double", "%x")],
            lambda name, dtype: "x", lines) == "_eon_hyper_pair_0."+function
    assert len(lines) == 1
    assert not IrPairs((s,), set()).arguments
    assert not IrPairs((s, c.replace("%x", "%changed")), set()).arguments
    assert not IrPairs((s.replace("double", "float"),
                        c.replace("double", "float")), set()).arguments


def test_statement_plan_respects_writes_branches_and_argument_evaluation():
    import ast
    from types import SimpleNamespace
    from mlx_backend.hyperbolic_pair import StatementPairs
    def statement(var, expr, branches=None):
        return SimpleNamespace(var=var, expr=expr, boolean_simplified_expressions=branches)
    a, c = statement("a", "sinh(x)"), statement("c", "cosh(x)")
    for between in ((), (statement("unrelated", "1"),)):
        plan = StatementPairs([(s,None) for s in (a,*between,c)], set())
        lines=[]
        for s, name in ((a,"sinh"),(c,"cosh")):
            plan.statement=s
            assert plan.expression(name,ast.Name("x",ast.Load()),"x",lines) is not None
        assert len(lines)==1
    for barrier, guard in ((statement("x","x+1"),None),
                           (statement("unrelated","1"),"mask"),
                           (statement("unrelated","1",{():"1"}),None)):
        plan=StatementPairs([(a,None),(barrier,guard),(c,None)],set())
        assert not plan.calls
    assert not StatementPairs([(statement("a","sinh(rand())"),None),
                               (statement("c","cosh(rand())"),None)],set()).calls


@pytest.mark.metal
def test_pair_actual_metal_full_range_matches_independent_brian_bits():
    mx = require_metal('Metal required')
    from mlx_backend import binary64, math64, hyperbolic_math64
    from mlx_backend.clang32 import Node, render
    from mlx_backend.hyperbolic_pair import MSL_HEADER
    from test_mlx_hyperbolic_math64 import cases, expected, assert_bits
    raw = np.unique(np.r_[cases(),
        (np.arange(1, 4224, dtype=np.float64)/512).view(np.uint64)])
    argument = Node("argument", "float64", value="input_value")
    used = set()
    body = render({"output_s": Node("call:sinh", "float64", (argument,)),
                   "output_c": Node("call:cosh", "float64", (argument,))}, used,
                  code_names={"output_s":"s[i]", "output_c":"c[i]"})
    assert body.count("_eon_hyper_pair(") == 1
    assert used == {"sinh", "cosh", "_hyperbolic_pair"}
    header = (binary64.MSL_HEADER + math64.MSL_MATH_HEADER
        + hyperbolic_math64.metal_hyperbolic_header({"sinh", "cosh"}, "cython") + MSL_HEADER)
    kernel = mx.fast.metal_kernel(name="hyperbolic_pair_boundary", input_names=["x"],
        output_names=["s", "c"], header=header,
        source="uint i=thread_position_in_grid.x;if(i<COUNT){ulong input_value=x[i];"+body+"}",
        compile_options={"math_mode": "safe"})
    outputs = kernel(inputs=[mx.array(raw)], template=[("COUNT", len(raw))],
        grid=(len(raw), 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[raw.shape, raw.shape], output_dtypes=[mx.uint64, mx.uint64])
    mx.eval(*outputs)
    for name, output in zip(("sinh", "cosh"), outputs):
        assert_bits(np.asarray(output), expected(name, raw), raw, name)


def _actual_model(monkeypatch, tmp_path, mutate, *, target="cython", dtype=np.float64,
                  integer_phase=True):
    mx = require_metal('Metal required')
    import brian2 as b
    from brian2.codegen.runtime.cython_rt.cython_rt import CythonCodeObject
    from brian2.codegen.runtime.numpy_rt.numpy_rt import NumpyCodeObject
    from component_cpu_reference import HyperbolicComponentCpu, component_inputs
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    monkeypatch.setitem(b.prefs, "codegen.target", target)
    monkeypatch.setitem(b.prefs, "core.default_float_dtype", dtype)
    monkeypatch.setitem(b.prefs, "core.default_integer_dtype", np.int32)
    model = resolve_model({"eqs": "v:volt\ng:volt\nrfc:second\nx:1\na:1\nc:1\nphase:"
                                  + ("integer" if integer_phase else "1") + "\n",
        "eq_th": "True", "t_rfc": 0*b.ms,
        "eq_rst": "v=v_rst;g=0*mV;x=(i-3+phase)/2;a=sinh(x);"
                  + ("x+=1/8;" if mutate else "") + "c=cosh(x);phase+=1"})
    cpu = HyperbolicComponentCpu(7, 5, model, target, tmp_path/"cpu-cache")
    _, _, reference = cpu.run()
    assert ("CythonCodeObject" if target=="cython" else "NumpyCodeObject") in cpu.code_object_types
    (tmp_path/"actual-cpu-reset.txt").write_text(str(cpu.neurons.resetter["spike"].codeobj.code))
    graph, stimulus = component_inputs(7, 5, model)
    prepared = configured.prepare(graph, stimulus, model)
    (tmp_path/"native-reset.metal").write_text(prepared.program.reset_code)
    paired = target=="cython" and dtype==np.float64 and not mutate
    assert ("_eon_hyper_pair(" in prepared.program.reset_code) == paired
    assert ("_eon_hyper_pair(" in prepared.program.header_code) == paired
    if paired and not integer_phase:
        assert "reset" not in prepared.program.compiled_integer_stages

    def equal(state):
        actual = configured.host_state(state, prepared)
        assert actual.keys() == reference.keys()
        for name, values in actual.items():
            np.testing.assert_array_equal(values.view(np.uint8),
                reference[name].view(np.uint8), err_msg=name)

    forbidden = forbidden_call('Native execution ran a CPU timestep')
    monkeypatch.setattr(CythonCodeObject, "run", forbidden)
    monkeypatch.setattr(NumpyCodeObject, "run", forbidden)
    for record in (False, True):
        result, state = configured.run_prepared(prepared, stimulus,
            record=record, return_state=True, chunk_steps=2)
        equal(state)
        assert result.spikes.steps.size == (35 if record else 0)
    state = configured.initial_state(prepared, 1)
    events = mx.zeros((1,), mx.uint8)
    spikes = []
    for length in (1, 2, 2):
        start = state.step
        state, words = configured.advance(prepared, state, events, 5, length)
        configured.evaluate_state(state)
        for bit in range(length):
            for neuron in np.flatnonzero(np.asarray(words) & (1 << bit)):
                spikes.append((int(neuron), start+bit))
    equal(state)
    assert sorted(spikes) == [(neuron, step) for neuron in range(7) for step in range(5)]


@pytest.mark.metal
@pytest.mark.parametrize("mutate", [False, True])
def test_actual_cython_pair_and_mutation_recording_continuation(monkeypatch, tmp_path, mutate):
    _actual_model(monkeypatch, tmp_path, mutate)


@pytest.mark.metal
@pytest.mark.parametrize("target,dtype,integer_phase", [
    ("cython", np.float64, False),
    ("numpy", np.float64, True),
    ("cython", np.float32, True)])
def test_pair_target_width_and_ordinary_binary64_controls(monkeypatch,tmp_path,target,dtype,integer_phase):
    _actual_model(monkeypatch, tmp_path, False, target=target, dtype=dtype,
                  integer_phase=integer_phase)


@pytest.mark.metal
def test_actual_cython_shared_pair_clock_and_continuation(monkeypatch, tmp_path):
    mx = require_metal('Metal required')
    import brian2 as b
    from brian2.codegen.runtime.cython_rt.cython_rt import CythonCodeObject
    from brian2.codegen.runtime.numpy_rt.numpy_rt import NumpyCodeObject
    from component_cpu_reference import component_inputs
    from configured_reference import reference
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    monkeypatch.setitem(b.prefs, "codegen.target", "cython")
    monkeypatch.setitem(b.prefs, "core.default_float_dtype", np.float64)
    monkeypatch.setitem(b.prefs, "core.default_integer_dtype", np.int32)
    monkeypatch.setitem(b.prefs, "codegen.runtime.cython.cache_dir", str(tmp_path/"cpu-cache"))
    model = resolve_model({"eqs": """
v:volt
g:volt
rfc:second
phase=timestep(t,dt)+1:integer(shared,constant over dt)
shared_x=phase/8:1(shared,constant over dt)
shared_s=sinh(shared_x):1(shared,constant over dt)
shared_c=cosh(shared_x):1(shared,constant over dt)
x=(i+phase+shared_s+shared_c)/8:1(constant over dt)
a=sinh(x):1(constant over dt)
c=cosh(x):1(constant over dt)
seen_s:1
seen_c:1
seen_a:1
seen_b:1
""", "eq_th": "True", "t_rfc": 0*b.ms,
        "eq_rst": "v=v_rst;g=0*mV;seen_s=shared_s;seen_c=shared_c;seen_a=a;seen_b=c"})
    graph, stimulus = component_inputs(7, 5, model)
    expected = reference(graph, stimulus, model, record_state=True)
    prepared = configured.prepare(graph, stimulus, model)
    scalar_stage = prepared.program.scalar_stages["subexpression"]
    assert "_eon_hyper_pair(" in scalar_stage.scalar_code
    assert "_eon_hyper_pair(" in scalar_stage.vector_code
    (tmp_path/"native-shared-scalar.metal").write_text(scalar_stage.scalar_code)
    (tmp_path/"native-shared-vector.metal").write_text(scalar_stage.vector_code)

    def equal(state, index):
        actual = configured.host_state(state, prepared)
        assert actual.keys() == expected.traces.keys()
        for name, values in actual.items():
            np.testing.assert_array_equal(np.ascontiguousarray(values).view(np.uint8),
                np.ascontiguousarray(expected.traces[name][:, index]).view(np.uint8), err_msg=name)

    forbidden = forbidden_call('Native execution ran a CPU timestep')
    monkeypatch.setattr(CythonCodeObject, "run", forbidden)
    monkeypatch.setattr(NumpyCodeObject, "run", forbidden)
    for record in (False, True):
        result, state = configured.run_prepared(prepared, stimulus,
            record=record, return_state=True, chunk_steps=2)
        assert state.step == 5
        equal(state, 4)
        assert_recorded_spikes_equal(result, expected, record)
    state = configured.initial_state(prepared, 1)
    events = mx.zeros((1,), mx.uint8)
    for length in (1, 2, 2):
        state, _ = configured.advance(prepared, state, events, 5, length)
        configured.evaluate_state(state)
        equal(state, state.step-1)
