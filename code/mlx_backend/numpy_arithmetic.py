"""Ordinary NumPy arithmetic over the shared binary32/binary64 carriers.

The shared width renderer owns promotion and half handling. This adapter only
distinguishes an already-binary32 representation from a genuine numeric cast,
then reuses the qualified raw-word arithmetic primitives.
"""
import ast


def _choose(predicate,yes,no):
    if predicate is True:return yes
    if predicate is False:return no
    return f'({predicate} ? {yes} : {no})'


def render_numpy_arithmetic(operation,left_code,right_code,result_narrow,*,
                            left_narrow=False,right_narrow=False,left_scalar=False,right_scalar=False,
                            loop_count=None,loop_overlap=None):
    """Return a binary64 carrier expression and shared dependency tags.

    Codes carry exact binary64 words; a true operand-narrow predicate means
    the word represents an already-binary32 value, including an untouched
    signaling NaN. A true result-narrow predicate selects NumPy's float32
    arithmetic loop. Predicates may be static booleans or emitted expressions.

    Recovering a stored float32 operand must not perform a numeric cast. By
    contrast, promotion from an actual float32 operand to float64, or narrowing
    an actual float64 scalar to float32, is a real NumPy cast and quiets sNaNs.
    The existing shared h_* helpers encode representation, not another engine.
    Half arithmetic/promotion is handled by its registered owner before here.

    A right scalar in add/multiply needs the actual NumPy loop count and
    conservative overlap-check result. Its SIMD path is scalar-first above
    eight float32 or four float64 elements; its scalar fallback is left-first.
    The overlap check includes abutting buffers, so dtype/count alone cannot
    infer it. Missing context is deliberately unresolved, not a guessed rule.
    """
    names={ast.Add:'add',ast.Sub:'sub',ast.Mult:'mul',ast.Div:'div',
           '+':'add','-':'sub','*':'mul','/':'div'}
    name=names[operation if isinstance(operation,str) else type(operation)]
    def operands(code,narrow):
        raw32=_choose(narrow,f'h_d_to_f({code})',f'f_from_d({code})')
        wide=_choose(narrow,f'f_to_d(h_d_to_f({code}))',code)
        return raw32,wide
    a32,a64=operands(left_code,left_narrow)
    b32,b64=operands(right_code,right_narrow)
    def arithmetic(prefix,a,b,threshold):
        ordinary=f'{prefix}_{name}({a},{b})'
        if name not in ('add','mul') or not right_scalar or left_scalar:return ordinary
        if loop_count is None or loop_overlap is None:
            raise ValueError('NumPy scalar NaN priority needs actual loop count and overlap context')
        reverse=_choose(loop_overlap,False,f'(({loop_count})>{threshold}L)')
        return _choose(reverse,f'{prefix}_{name}({b},{a})',ordinary)
    result32=f'h_f32_to_d({arithmetic("f",a32,b32,8)})'
    result64=arithmetic('d',a64,b64,4)
    dependencies=('_half_f32',) if any(p is not False for p in
        (result_narrow,left_narrow,right_narrow)) else ()
    return _choose(result_narrow,result32,result64),dependencies
