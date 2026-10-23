"""Propositional proofs for read-only threshold call guards.

This metadata never evaluates a model or substitutes arithmetic. Integer
comparison complements and Boolean structure are exact; other predicates
remain unconstrained atoms. Distinct random calls always get distinct atoms.
"""
import ast
from functools import lru_cache
import numpy as np


def expression(node,types,random_ids,inputs=()):
    def key(part):
        if isinstance(part,ast.Call)and isinstance(part.func,ast.Name)and part.func.id in('rand','randn','poisson'):
            return ('random',random_ids.setdefault(part,len(random_ids)))
        if isinstance(part,ast.AST):return (type(part).__name__,tuple((name,key(value))for name,value in ast.iter_fields(part)))
        if isinstance(part,list):return tuple(key(value)for value in part)
        return part
    def integer(part):
        if isinstance(part,ast.Constant):return isinstance(part.value,(int,bool))
        if isinstance(part,ast.Name):return part.id in inputs and part.id in types and np.dtype(types[part.id]).kind in'biu'
        return False
    def logical(part):
        if isinstance(part,ast.Constant):return ('true',)if bool(part.value)else('false',)
        if isinstance(part,ast.UnaryOp)and isinstance(part.op,ast.Not):return ('not',logical(part.operand))
        if isinstance(part,ast.BoolOp):return ('and'if isinstance(part.op,ast.And)else'or',tuple(logical(x)for x in part.values))
        if isinstance(part,ast.Name)and part.id in inputs:return ('atom',('input',key(part)))
        if isinstance(part,ast.Compare)and len(part.ops)==1:
            a,b=part.left,part.comparators[0];op=type(part.ops[0])
            if integer(a)and integer(b)and op in(ast.Lt,ast.GtE,ast.LtE,ast.Gt):
                name='integer-lt'if op in(ast.Lt,ast.GtE)else'integer-le'
                atom=('atom',(name,key(a),key(b)))
                return ('not',atom)if op in(ast.GtE,ast.Gt)else atom
            if integer(a)and integer(b)and op in(ast.Eq,ast.NotEq):
                atom=('atom',('eq',key(a),key(b)))
                return ('not',atom)if op is ast.NotEq else atom
        return ('atom',('opaque',key(part)))
    return logical(node)


@lru_cache(maxsize=512)
def _logical(value):
    from sympy import And,Or,Not,Symbol,true,false
    op=value[0]
    if op=='true':return true
    if op=='false':return false
    if op=='atom':return Symbol(repr(value[1]))
    if op=='not':return Not(_logical(value[1]))
    return (And if op=='and'else Or)(*(_logical(x)for x in value[1]))


@lru_cache(maxsize=256)
def exclusive(a,b):
    from sympy import And
    from sympy.logic.inference import satisfiable
    return satisfiable(And(_logical(a),_logical(b)))is False


@lru_cache(maxsize=256)
def stable(value):
    """Prove that guard truth does not depend on any opaque FP predicate."""
    from sympy import Symbol,Xor,true,false
    from sympy.logic.inference import satisfiable
    def opaque(part):
        if part[0]=='atom':return{part[1]}if part[1][0]=='opaque'else set()
        if part[0]=='not':return opaque(part[1])
        if part[0]in('and','or'):return set().union(*(opaque(x)for x in part[1]))
        return set()
    result=_logical(value)
    for atom in opaque(value):
        symbol=Symbol(repr(atom))
        if satisfiable(Xor(result.subs(symbol,true),result.subs(symbol,false)))is not False:return False
    return True
