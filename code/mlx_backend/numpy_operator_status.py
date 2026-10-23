"""NumPy floating operator flags from the selected qualified native graph."""
from functools import lru_cache


@lru_cache(maxsize=8)
def metal_header(functions):
    from .operator64 import MSL_POWER_HEADER,MSL_REMAINDER_HEADER
    from .numpy_math_status import graph_header
    functions=frozenset(functions)
    roots={'power':'d_operator_pow','remainder':'d_operator_mod_numpy',
           'floor_divide':'d_operator_floordiv_numpy'}
    source=(MSL_POWER_HEADER if 'power' in functions else '')
    source+=(MSL_REMAINDER_HEADER if functions&{'remainder','floor_divide'} else '')
    header=graph_header(source,tuple(roots[name] for name in functions))
    body=r'''
inline uint _eon_numpy_operator64_status(uint op,ulong a,ulong b,bool scalar_exponent,ulong original_exponent) {
    uint status=0u;
'''
    if 'remainder' in functions:
        body+='if(op==4u) (void)_eon_status_d_operator_mod_numpy(a,b,status);\n'
    if 'floor_divide' in functions:
        body+='if(op==5u) (void)_eon_status_d_operator_floordiv_numpy(a,b,status);\n'
    if 'power' in functions:
        body+=r'''
    if(op==6u) {
        // ndarray ** scalar uses NumPy's scalar-value shortcuts before pow.
        // The predicate observes the original scalar before loop narrowing.
        if(scalar_exponent && d_eq(original_exponent,0x3ff0000000000000ul)) {}
        else if(scalar_exponent && d_eq(original_exponent,0x4000000000000000ul))
            (void)_eon_status_d_mul(a,a,status);
        else if(scalar_exponent && d_eq(original_exponent,0xbff0000000000000ul))
            (void)_eon_status_d_div(0x3ff0000000000000ul,a,status);
        else if(scalar_exponent && d_eq(original_exponent,0x3fe0000000000000ul)) {
            if((a>>63)&&!d_is_zero(a)&&!d_is_nan(a)) status|=8u;
            if(_eon_status_snan(a)) status|=8u;
        } else (void)_eon_status_d_operator_pow(a,b,status);
    }
'''
    return header+body+r'''
    return status;
}
'''


@lru_cache(maxsize=8)
def metal_narrow_header(functions):
    """Follow the selected scalar32 graphs and the half loop's final cast.

    Include original operator64/operator32 headers and binary16's F32 header
    first. Shared checked narrow primitives retain instruction widths; powf's
    internal binary64 polynomial stays checked at binary64 precision.
    """
    from .operator32 import MSL_POWER_HEADER, MSL_REMAINDER_HEADER
    from .operator64 import MSL_REMAINDER_HEADER as remainder64
    from .numpy_math_status import graph_header, _thread
    from .numpy_narrow_status import metal_header as checked_narrow_header
    functions=frozenset(functions)
    roots={'power':'f_operator_pow','remainder':'f_operator_mod_numpy',
           'floor_divide':'f_operator_floordiv_numpy'}
    source=MSL_POWER_HEADER if 'power' in functions else ''
    if functions&{'remainder','floor_divide'}:
        source+=remainder64+MSL_REMAINDER_HEADER
        # The numerical integer-reduction shortcut still quiets sNaNs; the
        # actual fmodf entry raises invalid before returning that payload.
        source=source.replace('inline uint f_operator_fmod(uint x,uint y) {',
            'inline uint f_operator_fmod(uint x,uint y) {'
            'if(_eon_narrow_snan32(x)||_eon_narrow_snan32(y)) _status|=8u;')
    if 'power' in functions:
        # powf compares x with one before the zero-exponent shortcut. Unlike
        # binary64 pow, that first comparison signals for sNaN**0 as well.
        source=source.replace('inline uint f_operator_pow(uint x,uint y) {',
            'inline uint f_operator_pow(uint x,uint y) {'
            'if(_eon_narrow_snan32(x)) _status|=8u;')
        # The selected powf domain branch returns a constant instead of
        # materializing the invalid instruction from the installed routine.
        source=source.replace('if(!integral) return 0x7fc00000u;',
            'if(!integral) {_status|=8u;return 0x7fc00000u;}')
    source=_thread(source,{'f_from_d','f_add','f_sub','f_mul','f_div','f_fma'})
    header=graph_header(source,tuple(roots[name] for name in functions))
    body=r'''
inline uint _eon_numpy_operator32_result(uint op,uint a,uint b,bool scalar_exponent,
                                        ulong original_exponent,thread uint& status) {
'''
    if 'remainder' in functions:
        body+='if(op==4u)return _eon_status_f_operator_mod_numpy(a,b,status);\n'
    if 'floor_divide' in functions:
        body+='if(op==5u)return _eon_status_f_operator_floordiv_numpy(a,b,status);\n'
    if 'power' in functions:
        body+=r'''
    if(op==6u) {
        if(scalar_exponent && d_is_zero(original_exponent))return 0x3f800000u;
        if(scalar_exponent && d_eq(original_exponent,0x3ff0000000000000ul))return a;
        if(scalar_exponent && d_eq(original_exponent,0x4000000000000000ul))
            return _eon_status_f_mul(a,a,status);
        if(scalar_exponent && d_eq(original_exponent,0xbff0000000000000ul))
            return _eon_status_f_div(0x3f800000u,a,status);
        if(scalar_exponent && d_eq(original_exponent,0x3fe0000000000000ul)) {
            uint absolute=a&0x7fffffffu;
            if((a>>31)&&absolute&&absolute<=0x7f800000u)status|=8u;
            if(_eon_narrow_snan32(a))status|=8u;
            return f_sqrt(a);
        }
        return _eon_status_f_operator_pow(a,b,status);
    }
'''
    return checked_narrow_header(())+header+body+r'''
    return 0u;
}
inline uint _eon_numpy_operator32_status(uint op,uint a,uint b,bool scalar_exponent,ulong original_exponent) {
    uint status=0u;
    (void)_eon_numpy_operator32_result(op,a,b,scalar_exponent,original_exponent,status);
    return status;
}
inline uint _eon_numpy_operator16_status(uint op,ushort a,ushort b,bool scalar_exponent,ulong original_exponent) {
    uint status=0u;
    uint value=_eon_numpy_operator32_result(op,h_d_to_f(h_to_d(a)),h_d_to_f(h_to_d(b)),
                                           scalar_exponent,original_exponent,status);
    return status|_eon_numpy_cast_status(h_f32_to_d(value),16u);
}
'''
