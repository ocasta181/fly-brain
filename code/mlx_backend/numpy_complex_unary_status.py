"""Floating status from the existing scalar complex operation graphs.

The numerical providers remain authoritative.  Selected immutable copies
thread status through their actual scalar operations, including binary32
rounding boundaries.  Internal cpow products retain their scalar graph;
NumPy's ndarray binary kernels have separate status providers.
"""
from functools import lru_cache
import re


FUNCTIONS = ('exp', 'log', 'expm1', 'log1p', 'log10', 'sinh', 'cosh',
             'tanh', 'sin', 'cos', 'tan', 'arctan', 'arcsin', 'arccos',
             'sqrt', 'abs', 'sign')
_PREFIX = '_eon_cu_status_'
_PRIMITIVES = {'d_add', 'd_sub', 'd_mul', 'd_div', 'd_fma', 'd_sqrt',
               'f_add', 'f_sub', 'f_mul', 'f_div', 'f_fma', 'f_sqrt',
               'f_from_d'}


def original_header():
    """Original providers, after binary64/32 and math64 declarations."""
    from .basic_math32 import MSL_HEADER as basic
    from .transcendentals import metal_math_header
    from .stable_math64 import metal_stable_math_header, _STABLE_DIV_HEADER
    from .trigonometry64 import metal_trigonometry_header
    from .trigonometry32 import metal_trigonometry32_header
    from .arctangent64 import metal_arctangent_header
    from .complex_math64 import metal_complex_header
    from .complex_log_math64 import metal_complex_log_header
    from .complex_asinh_math32 import metal_complex_asinh_header
    from .complex_inverse_math64 import metal_complex_inverse_header
    from .complex_power_math64 import metal_complex_power_header
    return (basic + metal_math_header({'exp', 'log'})
            + metal_stable_math_header({'expm1', 'log1p'})
            + metal_trigonometry_header({'sin', 'cos', 'tan'})
            + metal_arctangent_header({'arctan'})
            + metal_trigonometry32_header({'sin', 'cos', 'tan'}).replace(_STABLE_DIV_HEADER, '')
            + metal_complex_header({'_complex', '_complex_basic', '_complex_trig', '_complex_extended'})
            + metal_complex_log_header() + metal_complex_asinh_header()
            + metal_complex_inverse_header() + metal_complex_power_header())


def _expand_inverse_macros(source):
    from .numpy_math_status import _close
    macros = {'CADD': 'add', 'CSUB': 'sub', 'CMUL': 'mul', 'CDIV': 'div',
              'CSQRT': 'sqrt', 'CFMA': 'fma'}
    # Definitions are outside the extracted function bodies. Expand uses in
    # those bodies before discovering the actual dependency graph.
    source = re.sub(r'^#(?:define|undef) C\w+[^\n]*', '', source, flags=re.M)
    # Rewrite from right to left, balancing again after each nested edit.
    for match in reversed(list(re.finditer(r'\b(' + '|'.join(macros) + r')\(', source))):
        end = _close(source, match.end()-1)
        source = (source[:match.start()] + '_eon_cinv_' + macros[match.group(1)]
                  + source[match.end()-1:end-1] + ',single)' + source[end:])
    return source


def _thread_first(source, names):
    from .numpy_math_status import _calls
    edits = []
    for match in _calls(source, names):
        # Status precedes optional `single=false`; appending a non-default
        # parameter would make these otherwise valid declarations illegal.
        declaration = bool(re.search(r'\binline\s+\w+\s*$', source[:match.start()]))
        separator = '' if source[match.end():].lstrip().startswith(')') else ','
        edits.append((match.end(), ('thread uint& _status' if declaration else '_status')+separator))
        edits.append((match.start(), _PREFIX))
    # A nested call can start exactly where its parent's first argument is
    # inserted. Prefix that callee first, then prepend the parent's status.
    for position, value in sorted(edits, key=lambda edit: (edit[0], edit[1]==_PREFIX), reverse=True):
        source = source[:position] + value + source[position:]
    return source


def _entry_effects(source, name):
    """Restore observable entry operations omitted by numeric shortcuts."""
    from .numpy_math_status import _restore_effects
    source = _restore_effects(source, name)
    before = ''
    if name in ('d_exponential', 'd_logarithm', '_eon_stable_expm1',
                '_eon_stable_log1p', 'd_sine', 'd_cosine', 'd_tangent',
                '_eon_trig64_sincos_scalar'):
        before += 'if(_eon_status_snan(x))_status|=8u;'
    if name == 'd_logarithm':
        before += 'if(d_is_zero(x))_status|=1u;else if((x>>63)&&!d_is_nan(x))_status|=8u;'
    if name == '_eon_stable_log1p':
        before += 'if(d_eq(x,0xbff0000000000000ul))_status|=1u;else if(d_lt(x,0xbff0000000000000ul))_status|=8u;'
    if name in ('d_sine', 'd_cosine', 'd_tangent', '_eon_trig64_sincos_scalar'):
        before += 'if(d_abs(x)==D_INF)_status|=8u;'
    if name.startswith('_eon_basic32_') and name.rsplit('_', 1)[-1] in ('exp','log','expm1','log1p'):
        before += 'uint _a=bits&0x7fffffffu;if(_eon_narrow_snan32(bits))_status|=8u;'
        if name.endswith('_log'):
            before += 'if(!_a)_status|=1u;else if((bits>>31)&&_a<=0x7f800000u)_status|=8u;'
        if name.endswith('_log1p'):
            before += 'if(bits==0xbf800000u)_status|=1u;else if((bits>>31)&&_a>0x3f800000u&&_a<=0x7f800000u)_status|=8u;'
        if name.endswith('_exp'):
            # The original expf branch directly returns its rounded constant.
            before += 'if(_a>=0x43000000u&&_a<0x7f800000u)_status|=(bits>>31)?4u:2u;'
        if name.endswith('_expm1'):
            before += 'if(!(bits>>31)&&_a>=_eon_basic32_p_expm1f_limits[0]&&_a<0x7f800000u)_status|=2u;'
    if name in ('_eon_complex_power', '_eon_complex_power32'):
        # NumPy's zero-base, nonpositive exponent entry executes inf-inf.
        before += 'if(_eon_complex_truth(b)&&!_eon_complex_truth(a)&&!d_gt(b.real,0ul))_status|=8u;'
    if before:
        begin = source.index('{')+1
        source = source[:begin]+before+source[begin:]
    return source


def _primitive_header(*, bounded=False):
    rows = []
    for width, kind in (('d','ulong'), ('f','uint')):
        for name, count in (('add',2), ('sub',2), ('mul',2), ('div',2), ('fma',3)):
            arguments = ','.join('abc'[:count])
            parameters = ','.join(kind+' '+arg for arg in 'abc'[:count])
            rows.append(f'inline {kind} {_PREFIX}{width}_{name}(thread uint& _status,{parameters}){{'
                        f'return _eon_status_{width}_{name}({arguments},_status);}}')
    rows.append(f'inline uint {_PREFIX}f_from_d(thread uint& _status,ulong x){{return _eon_status_f_from_d(x,_status);}}')
    for width, kind, sign, magnitude, infinity in (
            ('d','ulong','63','d_abs(x)','D_INF'),
            ('f','uint','31','(x&0x7fffffffu)','0x7f800000u')):
        snan = '_eon_status_snan' if width=='d' else '_eon_narrow_snan32'
        rows.append(f'inline {kind} {_PREFIX}{width}_sqrt(thread uint& _status,{kind} x){{'
                    f'if({snan}(x)||((x>>{sign})&&{magnitude}&&{magnitude}<={infinity}))_status|=8u;'
                    f'return {width}_sqrt(x);}}')
    source = '\n'.join(rows)
    # Inverse/power join enough elementary operations that expanding the
    # entire exact rounding body at every use overwhelms Metal compilation.
    # Keep the same checked instructions behind a call boundary, as the
    # shared real status helper already does for its exact FMA accumulator.
    if bounded:
        source = source.replace('inline ', '__attribute__((noinline)) ')
    return source


@lru_cache(maxsize=64)
def metal_complex_unary_header(functions=FUNCTIONS, *, power=False):
    """Emit selected status graphs after their original numerical providers.

    Requires numpy_error_policy.metal_header() and binary16's declarations.
    Operation IDs are the stable positions in FUNCTIONS, independent of the
    selected subset. `result` is retained in the ABI for renderer symmetry;
    flags come from the operation graph, never its final magnitude alone.
    """
    from .math64 import MSL_MATH_HEADER
    from .numpy_math_status import _definitions, _calls, _LEAVES, _dependencies, graph_header
    from .numpy_narrow_status import metal_header as narrow_header
    names = frozenset(functions)
    unsupported = names-set(FUNCTIONS)
    if unsupported:
        raise ValueError('Unsupported complex status functions: '+repr(sorted(unsupported)))
    source = _expand_inverse_macros(MSL_MATH_HEADER+original_header())
    definitions = _definitions(source)
    roots = {name: '_eon_complex_'+('abs_numpy' if name=='abs' else name) for name in names}
    pending = [root+suffix for name,root in roots.items() for suffix in ('','32')
               if name!='abs' and root+suffix in definitions]
    pending.append('_eon_complex_round32')
    if power:
        pending += ['_eon_complex_power', '_eon_complex_power32']
    leaves = dict(_LEAVES)
    leaves['basic32_fma_normal'] = ('fma',3)
    needed = _dependencies(definitions, pending, leaves, _PRIMITIVES)
    emitted = set()
    bounded = power or bool(names&{'arcsin','arccos'})
    rows = [graph_header('', ()), narrow_header(()), _primitive_header(bounded=bounded)]
    def emit(name):
        if name in emitted:
            return
        emitted.add(name)
        body = definitions[name]
        if name in leaves:
            operation,count = leaves[name]
            arguments = ','.join('abc'[:count])
            parameters = ','.join('ulong '+arg for arg in 'abc'[:count])
            rows.append(f'inline ulong {_PREFIX}{name}(thread uint& _status,{parameters}){{'
                        f'(void)_eon_status_d_{operation}({arguments},_status);return {name}({arguments});}}')
            return
        body = _entry_effects(body, name)
        for call in _calls(body, needed):
            if call.group(1)!=name:
                emit(call.group(1))
        body = _thread_first(body, needed|_PRIMITIVES)
        if bounded and name in ('_eon_complex_power', '_eon_complex_power32',
                                '_eon_complex_power_recover', '_eon_complex_asinh',
                                '_eon_complex_arcsin', '_eon_complex_arcsin32',
                                '_eon_complex_arccos', '_eon_complex_arccos32'):
            # The scalar power closure joins both log/exp families and its
            # exceptional recovery. Preserve these call boundaries rather
            # than expanding that graph into each width-selected call site.
            body = body.replace('inline _eon_complex ',
                                '__attribute__((noinline)) _eon_complex ', 1)
        rows.append(body)
    for name in sorted(needed):
        emit(name)
    cases = []
    for name in sorted(names):
        root = roots[name]
        call = f'{_PREFIX}{root}(_status,input)'
        narrow = f'{_PREFIX}{root}32(_status,input)' if root+'32' in definitions else call
        after = '' if name=='abs' else f'if(width==32u)(void){_PREFIX}_eon_complex_round32(_status,value);'
        if name=='abs':
            # NumPy1.26.4 loops_unary_complex.dispatch.c.src:119-121
            # clears status after the selected SIMD array absolute loop,
            # including partial vectors. Only this outer ufunc clears flags;
            # nested scalar hypot/absolute operations retain their graph.
            statement = ''
        else:
            statement = f'_eon_complex value=width==32u?{narrow}:{call};'+after
        cases.append(f'case {FUNCTIONS.index(name)}u:{{{statement}break;}}')
    rows.append('inline uint _eon_numpy_complex_unary_status(uint op,_eon_complex input,'
                '_eon_complex result,uint width){uint _status=0u;switch(op){'
                + ''.join(cases)+'}return _status;}')
    if power:
        rows.append('inline uint _eon_numpy_complex_power_status(_eon_complex a,_eon_complex b,'
                    '_eon_complex result,uint width){uint _status=0u;_eon_complex value=width==32u?'
                    +_PREFIX+'_eon_complex_power32(_status,a,b):'+_PREFIX+'_eon_complex_power(_status,a,b);'
                    +f'if(width==32u)(void){_PREFIX}_eon_complex_round32(_status,value);return _status;}}')
    return '\n'.join(rows)
