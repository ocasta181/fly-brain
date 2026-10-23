"""Frozen Boolean dtype identity must preserve the original Cython frontend."""
import copy
import brian2 as b
import numpy as np
import pytest
from original_api_protocol import check_original_api_runs, observe_run
from original_api_reference import cpu_runtime
from test_mlx_compiled_integer_adapter import (_typed_fixture,_forbid_cpu_steps,
                                               private_compiler_cache)


def test_frozen_boolean_dtype_retains_brian_frontend_acceptance():
    from brian2.parsing.bast import brian_dtype_from_dtype
    from brian2.codegen.statements import Statement
    from mlx_backend.compiled_integer_frontend import compile_block,NoCode
    from mlx_backend.integer_policy import snapshot_integer_policy
    frozen=copy.deepcopy(np.dtype('bool'))
    assert frozen is not np.dtype('bool')
    with pytest.raises(TypeError,match='Unknown dtype'):brian_dtype_from_dtype(frozen)
    statement=Statement('toggle_flag','=','not toggle_flag','',frozen,scalar=False)
    statement.used_boolean_variables=['toggle_flag']
    statement.boolean_simplified_expressions={
        (('toggle_flag',False),):'True',(('toggle_flag',True),):'False'}
    with cpu_runtime():
        compiled=compile_block([statement],{'toggle_flag':frozen},snapshot_integer_policy('cython'),{'toggle_flag'})
    assert not isinstance(compiled,NoCode)
    assert statement.dtype is frozen
    assert 'toggle_flag' in compiled[1] and 'toggle_flag' in compiled[2]


@pytest.mark.metal
@pytest.mark.parametrize('width',[32,64])
def test_actual_cython_boolean_integer_float_component_after_dtype_normalization(width,monkeypatch):
    types=dict(toggle_flag=np.dtype('bool'),number=np.dtype('int32'),seen=np.dtype('float'+str(width)))
    initial=dict(toggle_flag=np.array([False,True,False,True]),number=np.array([1,3,7,11],np.int32))
    expression='toggle_flag=not toggle_flag;number+=int(toggle_flag)+1;seen+=0.125+0.5*int(toggle_flag)'
    def exercise(fixture, mode):
        for duration in (.1,.2):
            yield observe_run(fixture,duration)
    check_original_api_runs(lambda mode:_typed_fixture(types,expression,initial),exercise,
        monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)
