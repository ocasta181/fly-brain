"""Boundary checks for conservative, read-only threshold guard proofs."""
import ast
import numpy as np
from mlx_backend.expression_guard import expression,exclusive,stable


def mask(text,kind='int32',identities=None):
    return expression(ast.parse(text,mode='eval').body,{'i':np.dtype(kind)},
                      {}if identities is None else identities,{'i'})


def test_integer_complements_do_not_assume_floating_total_order():
    assert exclusive(mask('i<2'),mask('i>=2'))
    assert stable(mask('i<2'))and stable(mask('i>=2'))
    assert not exclusive(mask('i<2','float64'),mask('i>=2','float64'))
    assert not stable(mask('i<2','float64'))


def test_model_guard_can_make_earlier_random_predicate_irrelevant():
    assert stable(mask('(i>=2) and not((i<2) and (rand()>.5))'))
    assert not stable(mask('(i<3) and not((i<2) and (rand()>.5))'))


def test_distinct_random_calls_never_share_one_propositional_atom():
    identities={}
    for _ in range(32):
        a=mask('rand()>.5',identities=identities)
        b=mask('not(rand()>.5)',identities=identities)
        assert not exclusive(a,b)
    assert len(identities)==64
    node=ast.parse('rand()>.5',mode='eval').body
    identities={};a=expression(node,{},identities)
    b=expression(ast.UnaryOp(op=ast.Not(),operand=node),{},identities)
    assert exclusive(a,b)
    assert not stable(a)
