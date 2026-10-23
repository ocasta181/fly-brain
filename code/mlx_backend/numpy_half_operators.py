"""Width adapter for the authoritative NumPy binary-operator policy.

The registered callback has operator_policy.render_numpy_operator's interface.
This module owns only the half/float32/float64 promotion and raw carriers; the
callback owns power shortcuts, remainder semantics, and its dependency tags.
"""
from .numpy32 import _choose,_and


def render_registered_operator(renderer,node,left,right,policy):
    half_renderer=hasattr(renderer,'_half_promotion')
    if half_renderer:
        from .numpy_half import Expression,half_precision
    else:
        from .numpy32 import Expression
        half_precision=lambda expression:False
    narrow=renderer._promotion((left,right))
    half=renderer._half_promotion((left,right)) if half_renderer else False

    def emit(single,half_loop=False):
        kind="float32" if single else "float"
        if half_loop:
            args=[f"h_d_to_f({renderer._precision_operand(arg,True,True)})"
                  for arg in (left,right)]
        else:
            args=[]
            for arg in (left,right):
                code=renderer.convert(arg,kind).code
                if not single and arg.kind=='float':
                    # The raw64 carrier is not a numeric cast. A genuine
                    # binary32-to-binary64 promotion quiets signaling NaNs;
                    # half carriers retain their separately audited cast rule.
                    half_arg=half_precision(arg)
                    nonhalf=(False if half_arg is True else True if half_arg is False else f'!({half_arg})')
                    widen=_and([arg.narrow,nonhalf])
                    if widen is not False:
                        renderer.used_math.add('_half_f32')
                        code=_choose(widen,f'f_to_d(h_d_to_f({code}))',code)
                args.append(code)
        from .numpy_error_policy import numpy_array_exponent
        code,dependencies=policy(node,args[0],kind,args[1],kind,kind,
            values=renderer.values,right_scalar=numpy_array_exponent(left,right),
            original_right_code=renderer.convert(right,"float").code if right.scalar else None,
            original_right_kind="float" if right.scalar else None)
        renderer.used_math.update(dependencies)
        if single:
            renderer.used_math.add("_half_f32")
            code=f"h_f32_to_d({code})"
        return renderer._half_round(code) if half_loop else code

    if half is True:
        code=emit(True,True)
    else:
        regular=(_choose(narrow,emit(True),emit(False)) if isinstance(narrow,str)
                 else emit(narrow))
        code=_choose(half,emit(True,True),regular) if isinstance(half,str) else regular
    kwargs={'half':half} if half_renderer else {}
    return Expression(code,"float",left.scalar and right.scalar,narrow,**kwargs)
