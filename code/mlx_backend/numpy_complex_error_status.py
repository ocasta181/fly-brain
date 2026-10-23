"""NumPy complex arithmetic flags from the existing component operation graph.

The value implementation remains ``complex_math64``. These private copies only
thread status through the real operations its selected branch actually executes;
the caller's already computed complex value is never replaced. In particular,
division observes the ratio, reciprocal, and numerator intermediates too.
"""
from functools import lru_cache
import re

from .numpy_math_status import _close


_OPERATORS = ('add', 'sub', 'mul', 'div')


def _function(source, name):
    match = re.search(r'\binline\s+_eon_complex\s+' + re.escape(name) + r'\s*\(', source)
    if match is None:
        raise ValueError('Missing qualified complex arithmetic helper: ' + name)
    start = source.index('{', match.end())
    return source[match.start():_close(source, start, '{', '}')]


def _checked(source, name):
    # Append the same thread-local status reference to every actual arithmetic
    # call, including nested calls, without changing source branch/evaluation
    # order. The originals remain the authoritative numerical implementation.
    insertions = []
    names = r'(?:[df]_(?:add|sub|mul|div|fma)|' + re.escape(name) + ')'
    for match in re.finditer(r'\b(' + names + r')\s*\(', source):
        position = _close(source, match.end()-1)
        declaration = match.group(1) == name
        insertions.append((position-1, ', thread uint& status' if declaration else ', status'))
    for position, value in sorted(insertions, reverse=True):
        source = source[:position] + value + source[position:]
    source = re.sub(r'\b([df])_(add|sub|mul|div|fma)\b', r'_eon_complex_status_\1_\2', source)
    return re.sub(r'\b' + re.escape(name) + r'\b', '_eon_status' + name, source)


@lru_cache(maxsize=16)
def metal_complex_error_header(operators=(0, 1, 2, 3)):
    """Emit component flag graphs; an isolated shader may select one operator.

    Requires the ordinary binary32/binary64, complex base, and real floating
    status headers first. Width32 still uses the raw64 component carrier.
    The emitted status ABI's array_loop argument selects NumPy's fused ndarray
    multiplication; scalar arithmetic retains its separate component products.
    """
    from .complex_math64 import MSL_BASE_HEADER
    operators = tuple(operators)
    if any(type(op) is not int or op not in range(4) for op in operators):
        raise ValueError('Complex status supports add/subtract/multiply/divide')
    source = []
    if 2 in operators:
        from .binary16 import MSL_HEADER as HALF_HEADER
        from .numpy_math_status import graph_header
        from .numpy_narrow_status import metal_header as narrow_header
        source.extend((HALF_HEADER, graph_header('', ('d_fma',)), narrow_header(()), '''
inline ulong _eon_complex_status_d_fma(ulong a,ulong b,ulong c,thread uint& status) {
    return _eon_status_d_fma(a,b,c,status);
}
inline uint _eon_complex_status_f_fma(uint a,uint b,uint c,thread uint& status) {
    return _eon_status_f_fma(a,b,c,status);
}
'''))
    for op, name in enumerate(_OPERATORS):
        source.append(f'''
inline ulong _eon_complex_status_d_{name}(ulong a,ulong b,thread uint& status) {{
    return _eon_status_d_{name}(a,b,status);
}}
inline uint _eon_complex_status_f_{name}(uint a,uint b,thread uint& status) {{
    uint result=f_{name}(a,b);
    ulong left=f_to_d(a),right=f_to_d(b),rounded=f_to_d(result);
    uint wide_status=0u;
    _eon_status_d_{name}(left,right,wide_status);
    status |= _eon_numpy_binary_status({op}u,left,right,rounded,32u) | (wide_status&9u);
    if(((a&0x7f800000u)==0x7f800000u && (a&0x7fffffu) && !(a&0x400000u)) ||
       ((b&0x7f800000u)==0x7f800000u && (b&0x7fffffu) && !(b&0x400000u))) status|=8u;
    return result;
}}
''')
    cases = []
    for op in operators:
        name = '_eon_complex_' + _OPERATORS[op]
        roots = (name, '_eon_complex_multiply_numpy') if op == 2 else (name,)
        for root in roots:
            for suffix in ('', '32'):
                source.append(_checked(_function(MSL_BASE_HEADER, root + suffix), root + suffix))
        array_name = '_eon_complex_multiply_numpy' if op == 2 else name
        cases.append(f'''if(op=={op}u && !array_loop) {{
            if(width==32u)_eon_status{name}32(a,b,status);
            else _eon_status{name}(a,b,status);
        }} else if(op=={op}u) {{
            if(width==32u)_eon_status{array_name}32(a,b,status);
            else _eon_status{array_name}(a,b,status);
        }}''')
    source.append('''
inline uint _eon_numpy_complex_binary_status(uint op,_eon_complex a,_eon_complex b,
                                            _eon_complex result,uint width,bool array_loop=true) {
    uint status=0u;
    ''' + '\n'.join(cases) + '''
    return status;
}
''')
    return '\n'.join(source)
