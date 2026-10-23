"""Floating status from the pinned native libm arithmetic graph.

The result helpers already preserve the selected Apple/NumPy operation order.
This module threads status through that same graph, including operations whose
rounded value looks harmless. Integer polynomial shortcuts keep their qualified
result, while the shared exact arithmetic reports the status of the instruction
they replace. No host libm or model arithmetic executes here.
"""
from functools import lru_cache
import re


ROOTS = {
    'exp': 'd_exponential', 'log': 'd_logarithm',
    'expm1': '_eon_stable_expm1', 'log1p': '_eon_stable_log1p',
    'sin': 'd_sine', 'cos': 'd_cosine_numpy', 'tan': 'd_tangent',
    'sinh': '_eon_hyper_sinh', 'cosh': '_eon_hyper_cosh',
    'tanh': '_eon_hyper_tanh_numpy', 'log10': '_eon_hyper_log10',
    'arcsin': 'd_inverse_arcsin', 'arccos': 'd_inverse_arccos',
    'arctan': '_eon_arctan64', 'sqrt': 'd_sqrt',
    'floor':'d_floor','ceil':'d_ceil','abs':'d_abs','sign':'d_sign',
}
FUNCTIONS = tuple(ROOTS)

# Each shortcut represents exactly one rounded CPU instruction. Keep its
# independently qualified result; derive flags from the shared exact operation,
# rather than from the shortcut's packed result or a tiny-result heuristic.
_LEAVES = {
    'stable_fma': ('fma', 3), 'hyper_fma_normal': ('fma', 3),
    'hyper_fma_any': ('fma', 3), '_eon_asincos64_fma_normal': ('fma', 3),
    '_eon_asincos64_fma_compact': ('fma', 3),
    '_eon_trig_product_fma32': ('fma', 3), '_eon_trig_horner32': ('fma', 3),
    '_eon_trig_multiply': ('mul', 2), '_eon_asincos64_mul': ('mul', 2),
    'hyper_mul_normal': ('mul', 2), 'hyper_mul': ('mul', 2),
    '_eon_operator_mul': ('mul',2), '_eon_operator_polynomial_fma': ('fma',3),
    '_eon_operator_fma': ('fma',3), '_eon_operator_div': ('div',2),
    '_eon_trig_add32': ('add', 2),
    '_eon_trig_sub32': ('sub', 2), 'stable_div': ('div', 2),
}
_CORE = frozenset(('d_finish', 'd_add', 'd_sub', 'd_mul', 'd_div'))
_SIGNATURE = re.compile(
    r'(?:template\s*<[^{};]+>\s*)?\binline\s+\w+\s+(\w+)\s*\(')


def original_header(names):
    """Original declarations/tables required by the selected status graph."""
    from .transcendentals import metal_math_header
    from .stable_math64 import metal_stable_math_header
    from .trigonometry64 import metal_trigonometry_header
    from .hyperbolic_math64 import metal_hyperbolic_header
    from .inverse_trigonometry64 import metal_inverse_trigonometry_header
    from .arctangent64 import metal_arctangent_header
    names = set(names)
    return (metal_math_header(names & {'exp', 'log'})
            + metal_stable_math_header(names & {'expm1', 'log1p'})
            + metal_trigonometry_header(names & {'sin', 'cos', 'tan'})
            + metal_hyperbolic_header(names, 'numpy')
            + metal_inverse_trigonometry_header(names)
            + metal_arctangent_header(names))


def _close(source, start, opening='(', closing=')'):
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == opening) - (source[end] == closing)
        end += 1
    return end


def _definitions(source):
    # Comments can contain parentheses/braces and are irrelevant to source
    # identity: the originals, with their notices, remain the authoritative code.
    source = re.sub(r'/\*.*?\*/|//[^\n]*', '', source, flags=re.S)
    result = {}
    for match in _SIGNATURE.finditer(source):
        end = _close(source, match.end()-1)
        begin = end
        while source[begin].isspace():
            begin += 1
        if source[begin] != '{':
            continue
        finish = _close(source, begin, '{', '}')
        result[match.group(1)] = source[match.start():finish]
    return result


def _calls(source, names):
    pattern = r'\b(' + '|'.join(map(re.escape, sorted(names, key=len, reverse=True)))
    return re.finditer(pattern + r')(?:\s*<[^;{}()]+>)?\s*\(', source)


def _dependencies(definitions, pending, leaves, primitives):
    needed = set()
    while pending:
        name = pending.pop()
        if name in needed or name in primitives:
            continue
        needed.add(name)
        if name not in leaves:
            pending.extend(match.group(1) for match in _calls(definitions[name], definitions)
                           if match.group(1) != name)
    return needed


def _thread(source, names):
    edits = []
    for match in _calls(source, names):
        end = _close(source, match.end()-1)
        suffix = source[end:].lstrip()
        declaration = suffix.startswith('{')
        edits.append((end-1, ', thread uint& _status' if declaration else ', _status'))
        edits.append((match.start(), '_eon_status_'))
    for position, value in sorted(edits, reverse=True):
        source = source[:position] + value + source[position:]
    return source


def _restore_effects(source, name):
    # The qualified numerical helpers deliberately omit instructions whose
    # result rounds back to the input. Their flags are observable under seterr.
    if name == 'd_fma':
        begin=source.index('{')+1
        source=(source[:begin]
                + 'if(_eon_status_snan(a)||_eon_status_snan(b)||_eon_status_snan(c)) _status|=8u;'
                + 'if(!d_is_nan(a)&&!d_is_nan(b)&&!d_is_nan(c) && '
                  '((d_abs(a)==D_INF&&d_is_zero(b)) || '
                  '(d_abs(b)==D_INF&&d_is_zero(a)) || '
                  '((d_abs(a)==D_INF||d_abs(b)==D_INF)&&d_abs(c)==D_INF&&((a^b^c)>>63)))) _status|=8u;'
                + source[begin:])
    if name == 'd_exponential':
        source = source.replace('if(d_le(x,0xc0874a0000000000ul)) return 0ul;',
            'if(d_le(x,0xc0874a0000000000ul)) {'
            'if(d_exp(x)!=2047ul) _status|=4u; return 0ul;}')
    if name == 'd_inverse_arcsin':
        # libsystem_m asin+0xfc: below minimum normal, x-x*x is one FMSUB;
        # above it the discarded (1+minnormal) multiply is exact at this width.
        source = source.replace('if(a<=0x3e57137449123ef5ul) return x;',
            'if(a<=0x3e57137449123ef5ul) {'
            'if(a<0x0010000000000000ul) '
            '(void)_eon_status_d_fma(d_neg(x),x,x,_status); return x;}')
    if name == '_eon_arctan64':
        # atan+0x7c4: subnormal nonzero x uses fma(x,minnormal,x).
        source = source.replace('if(x<=0x3e4d12ed0af1a27ful) return input;',
            'if(x<=0x3e4d12ed0af1a27ful) {'
            'if(x && x<0x0010000000000000ul) '
            '(void)_eon_status_d_fma(input,0x0010000000000000ul,input,_status);'
            'return input;}')
    if name=='d_operator_pow':
        source=source.replace('if(ay<one || (ay<0x4340000000000000ul && d_floor(ay)!=ay)) return D_QNAN;',
            'if(ay<one || (ay<0x4340000000000000ul && d_floor(ay)!=ay)) '
            '{_status|=8u;return D_QNAN;}')
        source=source.replace('return sign|(((ax>one) != bool(y>>63)) ? D_INF:0ul);',
            '_status|=((ax>one)!=bool(y>>63))?2u:4u;'
            'return sign|(((ax>one) != bool(y>>63)) ? D_INF:0ul);')
        source=source.replace('if(d_gt(product,0x4100040000000000ul)) return sign|D_INF;',
            'if(d_gt(product,0x4100040000000000ul)) {_status|=2u;return sign|D_INF;}')
        source=source.replace('if(d_lt(product,0xc100cc0000000000ul)) return sign;',
            'if(d_lt(product,0xc100cc0000000000ul)) {_status|=4u;return sign;}')
    if name=='d_operator_fmod':
        begin=source.index('{')+1
        source=(source[:begin]+'if(_eon_status_snan(x)||_eon_status_snan(y)) _status|=8u;'
                +source[begin:])
    return source


def graph_header(source,roots):
    """Thread the existing selected graph; operator families reuse this too."""
    from .math64 import MSL_MATH_HEADER
    definitions = _definitions(MSL_MATH_HEADER + source)
    pending = [name for name in roots if name in definitions]
    needed = _dependencies(definitions, pending, _LEAVES, _CORE)
    # All graphs can call a fused instruction, including restored tiny paths.
    needed.add('d_fma')
    ordered = []
    visiting = set()
    def emit(name):
        if name in visiting:
            return
        visiting.add(name)
        source = definitions[name]
        if name in _LEAVES:
            operation, count = _LEAVES[name]
            arguments = ','.join('abc'[:count])
            parameters = ','.join('ulong '+arg for arg in 'abc'[:count])
            source = (f'inline ulong _eon_status_{name}({parameters}, thread uint& _status) {{'
                      f'(void)_eon_status_d_{operation}({arguments},_status);'
                      f'return {name}({arguments});}}')
        else:
            for call in _calls(source, needed):
                if call.group(1) != name:
                    emit(call.group(1))
            source = _restore_effects(source, name)
            source = _thread(source, needed | _CORE)
        macro='EON_STATUS_FUNCTION_'+name.upper()
        ordered.append(f'#ifndef {macro}\n#define {macro}\n{source}\n#endif')
    emit('d_fma')
    for name in sorted(needed):
        emit(name)
    return '\n'.join(ordered).replace('inline ulong _eon_status_d_fma(',
        '__attribute__((noinline)) ulong _eon_status_d_fma(')


@lru_cache(maxsize=64)
def metal_header(functions):
    """Emit only the selected status graph, after its original math headers."""
    names=frozenset(functions)
    ordered=[graph_header(original_header(names),tuple(ROOTS[name] for name in names
                                                    if name not in ('abs','sign')))]
    for name in sorted(names):
        root = ROOTS[name]
        call = f'{root}(x)' if name in ('sqrt','abs') else f'_eon_status_{root}(x,_status)'
        if name=='sign':
            call='(d_gt(x,0ul)?0x3ff0000000000000ul:d_lt(x,0ul)?0xbff0000000000000ul:d_is_zero(x)?0ul:x)'
        # Domain branches return bit constants in the numerical implementation;
        # report the FP side effects of the actual CPU entry path explicitly.
        before = '' if name in ('tanh','abs','sign') else 'if(_eon_status_snan(x)) _status|=8u;'
        if name in ('log', 'log10'):
            before += ('if(d_is_zero(x)) _status|=1u;'
                       'else if((x>>63) && !d_is_nan(x)) _status|=8u;')
        if name == 'log1p':
            before += ('if(d_eq(x,0xbff0000000000000ul)) _status|=1u;'
                       'else if(d_lt(x,0xbff0000000000000ul)) _status|=8u;')
        if name in ('arcsin', 'arccos'):
            before += 'if(d_abs(x)>0x3ff0000000000000ul && !d_is_nan(x)) _status|=8u;'
        if name in ('sin', 'cos', 'tan'):
            before += 'if(d_abs(x)==D_INF) _status|=8u;'
        if name == 'sqrt':
            before += 'if((x>>63) && !d_is_zero(x) && !d_is_nan(x)) _status|=8u;'
        ordered.append(f'inline ulong _eon_numpy_{name}_status(ulong x,thread uint& _status) {{'
                       + before + f'return {call};}}')
    return '\n'.join(ordered) + '\n'
