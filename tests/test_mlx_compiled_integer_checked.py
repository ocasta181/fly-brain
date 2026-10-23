"""Checked Python-integer helper boundaries in genuine compiled Brian blocks."""
import brian2 as b
import numpy as np
import pytest
from original_api_protocol import check_original_api_runs, observe_run
from test_mlx_compiled_integer_adapter import (_typed_fixture,_forbid_cpu_steps,
                                               private_compiler_cache)

CASES=[
    ('y=x//den',[9,1,-7,2],[2,0,3,1],ZeroDivisionError),
    ('y=x//den',[9,-(1<<63),-7,2],[2,-1,3,1],OverflowError),
    ('y=x%1180591620717411303424',[5,-1,0,2],[1]*4,OverflowError),
    ('y=x%-1180591620717411303424',[-5,1,0,-2],[1]*4,OverflowError),
    ('x+=4294967297',[0,(1<<63)-1,17,2],[1]*4,OverflowError),
    ('x*=4294967297',[1,1<<40,0,2],[1]*4,OverflowError),
]


@pytest.mark.metal
@pytest.mark.parametrize('expression,x,den,exception',CASES)
def test_actual_cython_checked_operation_prefix_and_repair(expression,x,den,exception,monkeypatch):
    types={name:np.dtype('int64') for name in ('x','den','y')}
    initial=dict(x=np.array(x,np.int64),den=np.array(den,np.int64),y=np.full(4,17,np.int64))
    def exercise(fixture, mode):
        failed=observe_run(fixture,.1,raises=b.BrianObjectException)
        if mode == 'cpu':
            assert isinstance(failed.error.__cause__,exception)
        yield failed
        # Repair every input before re-running the failed clock, preserving the
        # already committed prefix rather than starting another fresh trial.
        fixture.neurons.x[:]=0;fixture.neurons.den[:]=3
        yield observe_run(fixture,.2)
    check_original_api_runs(lambda mode:_typed_fixture(types,expression,initial),exercise,
        monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)


@pytest.mark.metal
@pytest.mark.parametrize('delta,x',[
    (-4294967297,[(1<<63),0,7,9]),
    (4294967297,[0,(1<<64)-1,7,9]),
])
def test_actual_cython_checked_unsigned_conversion_message_and_prefix(delta,x,monkeypatch):
    types=dict(x=np.dtype('uint64'))
    initial=dict(x=np.array(x,np.uint64));expression=f'x+=({delta})'
    def exercise(fixture, mode):
        failed=observe_run(fixture,.1,raises=b.BrianObjectException)
        if mode == 'cpu':
            assert isinstance(failed.error.__cause__,OverflowError)
        yield failed
        # No further failure is possible after repairing the failing row.
        fixture.neurons.x[:]=(1<<63)
        yield observe_run(fixture,.1)
    check_original_api_runs(lambda mode:_typed_fixture(types,expression,initial),exercise,
        monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)
