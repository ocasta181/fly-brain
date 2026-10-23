"""Eager immutable-call evaluation in the reusable NumPy half renderer."""
import numpy as np
import pytest
from native_execution import require_metal


@pytest.mark.metal
def test_half_renderer_keeps_eager_timedarray_prelude_inside_write_mask():
    from types import SimpleNamespace
    mx = require_metal('Metal required')
    from mlx_backend.numpy_half import Renderer
    from mlx_backend.compiler import _TimedArray,CompiledInputArray,_TIMED_ARRAY_HEADER
    from mlx_backend.binary64 import MSL_HEADER
    values=np.arange(6,dtype=np.float64).reshape(3,2)
    array=CompiledInputArray("drive","float64","ulong",values.shape,values)
    timed=_TimedArray("values","timed_lookup",array,np.float64(.0001),1,"numpy","source",
                      np.float64(1e4),np.float64(0.))
    renderer=Renderer({"flag":np.bool_,"sample":np.float64,"t":np.float64,
                       "column":np.int32,"gate":np.bool_},
                      {"drive":timed},{"sample"},{"t"},"reset",index_error_name="error",
                      half_ufuncs={"sin":("sin_f32","unused")})
    first=renderer.statement(SimpleNamespace(var="sample",expr="drive(t,0)",op="=",dtype=np.float64))
    statement=SimpleNamespace(var="sample",expr="sin(flag)+drive(t,column)",op="=",dtype=np.float64)
    emitted=renderer.statement(statement,conditional_write="gate")
    assert emitted.index("const ulong _eon_timed_result_")>emitted.index("if (gate)")
    columns=np.array([-1,0,2,-3,2,-3,1,-2],np.int32)
    gates=np.array([True,True,False,False,True,True,False,False])
    source="uint i=thread_position_in_grid.x;if(i<N){bool flag=true,gate=gates[i],error=false;ulong t=0,sample=0;long column=columns[i];"
    source+="\n".join(renderer.prologue)+first+emitted+"out[i]=error;}"
    kernel=mx.fast.metal_kernel(name="half_timed_prelude",input_names=["values","columns","gates"],output_names=["out"],
        header=MSL_HEADER+_TIMED_ARRAY_HEADER+timed.header(),source=source,compile_options={"math_mode":"safe"})
    (actual,)=kernel(inputs=[mx.array(values.ravel().view(np.uint64)),mx.array(columns),mx.array(gates)],
        template=[("N",len(columns))],grid=(len(columns),1,1),threadgroup=(32,1,1),
        output_shapes=[columns.shape],output_dtypes=[mx.bool_])
    mx.eval(actual);np.testing.assert_array_equal(np.asarray(actual),[False,False,False,False,True,True,False,False])


@pytest.mark.parametrize("invalid_selected",[False,True])
@pytest.mark.parametrize("mode",["direct","euler"])
def test_actual_brian_numpy_mask_and_unconditional_auxiliary(tmp_path,invalid_selected,mode):
    b=pytest.importorskip("brian2")
    from brian2.devices.device import get_device,reset_device
    previous=get_device();target=b.prefs.codegen.target;optimise=b.prefs.codegen.loop_invariant_optimisations
    try:
        b.set_device("runtime")
        b.prefs.codegen.target="numpy";b.prefs.codegen.loop_invariant_optimisations=False
        drive=b.TimedArray(np.arange(6.).reshape(3,2),dt=.1*b.ms)
        derivative="(sin(flag)+drive(t,column))" if mode=="euler" else "0"
        group=b.NeuronGroup(4,"dx/dt="+derivative+"/second : 1 (unless refractory)\n"
            "flag : boolean\ncolumn : integer\n",threshold="False",reset="",refractory=1*b.ms,
            method="euler",dt=.1*b.ms,namespace={"drive":drive})
        group.flag=True;group.column=[-3 if invalid_selected else -1,0,2,-3]
        group.lastspike=np.array([-1.,-1.,0.,0.])*b.second
        runner=group.run_regularly("x=sin(flag)+drive(t,column)",when="groups",order=1) if mode=="direct" else group.state_updater
        network=b.Network(group)
        if invalid_selected or mode=="euler":
            with pytest.raises(b.BrianObjectException) as error:network.run(.1*b.ms)
            assert isinstance(error.value.__cause__,IndexError)
        else:
            network.run(.1*b.ms)
            assert np.all(np.asarray(group.x[:])[:2]>0)
            np.testing.assert_array_equal(group.x[:][2:],[0.,0.])
        source=str(runner.codeobj.code)
        (tmp_path/"actual-numpy-source.txt").write_text(source)
        assert ("column[not_refractory]" in source)==(mode=="direct")
    finally:
        reset_device();assert get_device() is previous
        b.prefs.codegen.target=target;b.prefs.codegen.loop_invariant_optimisations=optimise


