"""Floating status for the existing NumPy1.26.4 ARM64 narrow math graphs.

Numerical values stay with the qualified basic_math32, trigonometry32 and
hyperbolic32 providers (libsystem_m UUID83b173cd7f3239018e67c0b2676be015).
Status follows their actual CPU widths, not the two-float implementation
shortcuts used to accelerate some native values. NumPy half loops widen to
scalar32 then explicitly narrow: loops_umath_fp.dispatch.c.src:165-168.
NumPy32 sin/cos use the separate NEON graph, including discarded tiny products.
"""
from functools import lru_cache
import re


ROOTS = {
    **{name: '_eon_basic32_'+name for name in ('exp','log','expm1','log1p')},
    'sqrt':'f_sqrt', 'arcsin':'f_inverse_arcsin', 'arccos':'f_inverse_arccos',
    'arctan':'_eon_arctan32',
    **{name:'_eon_hyper32_'+name for name in ('log10','sinh','cosh')},
    'tanh':'_eon_hyper32_tanh_numpy',
    **{name:'_eon_trig32_'+helper+'_numpy' for name,helper in
       (('sin','sine'),('cos','cosine'),('tan','tangent'))},
}
FUNCTIONS = tuple(ROOTS)+('abs','sign','floor','ceil')


def _guard(source,name):
    macro='EON_NUMPY_NARROW_'+re.sub(r'\W','_',name).upper()+'_INCLUDED'
    return '#ifndef '+macro+'\n#define '+macro+'\n'+source+'\n#endif\n'


def original_header(functions):
    """Original numeric providers required before this module's status header."""
    from .basic_math32 import metal_basic_math_header
    from .trigonometry32 import metal_trigonometry32_header
    from .hyperbolic32 import metal_hyperbolic32_header
    from .inverse_trigonometry32 import metal_inverse_trigonometry32_header
    from .arctangent32 import metal_arctangent32_header
    names=set(functions)
    return (metal_basic_math_header(names)
            +metal_trigonometry32_header(names&{'sin','cos','tan'})
            +metal_hyperbolic32_header(names)
            +metal_inverse_trigonometry32_header(names)
            +metal_arctangent32_header(names))


_PRIMITIVES = r'''
#ifndef EON_NUMPY_NARROW_STATUS_INCLUDED
#define EON_NUMPY_NARROW_STATUS_INCLUDED
inline bool _eon_narrow_snan32(uint x) {
    return (x&0x7fffffffu)>0x7f800000u && !(x&0x400000u);
}
inline uint _eon_numpy_cast_status(ulong input,uint target_width) {
    if(target_width==64u)return 0u;
    if(d_exp(input)==2047ul)
        return target_width==32u && d_is_nan(input) && !(input&(1ul<<51)) ? 8u:0u;
    ulong rounded=target_width==16u?h_to_d(h_from_d(input)):f_to_d(f_from_d(input));
    if(d_abs(rounded)==D_INF)return 2u;
    ulong minimum=target_width==16u?0x3f10000000000000ul:0x3810000000000000ul;
    // The reference conversion detects tininess before rounding, including
    // a just-subnormal value whose rounded representation is minimum normal.
    return d_abs(input)<minimum && !d_eq(input,rounded) ? 4u:0u;
}
inline uint _eon_numpy_compare_status(ulong a,ulong b,uint operation) {
    // The selected NumPy comparison loops clear FP status, including sNaNs.
    return 0u;
}
inline uint _eon_numpy_int32_cast_status(ulong input,long position=0L,long length=0L,uint source_bits=64u) {
    // Match n_to_int32's actual ndarray prefix/tail and source-dtype policy.
    // Python float int() exceptions are selected separately by provenance.
    if(d_exp(input)==2047ul)return 8u;
    if(source_bits==64u && position<(length/8L)*8L)
        return d_ge(input,0x43e0000000000000ul)||d_lt(input,0xc3e0000000000000ul)?8u:0u;
    long integer=d_to_int(input);
    return integer>2147483647L||integer<(-2147483647L-1L)?8u:0u;
}
inline uint _eon_numpy_integer_instruction_status(ulong input,uint instruction_bits,bool unsigned_instruction) {
    if(d_exp(input)==2047ul)return 8u;
    if(unsigned_instruction) {
        // FCVTZU truncates first: negative fractions greater than -1 map
        // validly to zero. Subsequent integer narrowing raises no FP status.
        ulong upper=instruction_bits==64u?0x43f0000000000000ul:0x41f0000000000000ul;
        return d_le(input,0xbff0000000000000ul)||d_ge(input,upper)?8u:0u;
    }
    if(instruction_bits==64u)
        return d_ge(input,0x43e0000000000000ul)||d_lt(input,0xc3e0000000000000ul)?8u:0u;
    long integer=d_to_int(input);
    return integer>2147483647L||integer<(-2147483647L-1L)?8u:0u;
}
inline uint _eon_numpy_integer_cast_status(ulong input,uint destination_bits,bool unsigned_destination,
        long position,long length,uint source_bits,bool complex_source=false) {
    // Match numpy32.finish and NumPy1.26.4's actual ARM64 array cast loops.
    // ComplexWarning/projection and Python direct-store exceptions belong
    // to the caller. source_bits describes the projected real component.
    bool unsigned_instruction=unsigned_destination&&destination_bits>=32u;
    uint instruction_bits=destination_bits==64u?64u:32u;
    if(destination_bits<64u) {
        // Complex loops retain a final1..block scalar tail even when length
        // is an exact multiple. Real loops can vectorize the complete span.
        long prefix_length=complex_source?(length>0L?length-1L:0L):length;
        if(source_bits==64u) {
            long block=!complex_source&&destination_bits==16u?32L:8L;
            if(position<(prefix_length/block)*block)instruction_bits=64u;
        } else if(source_bits==32u&&unsigned_destination&&destination_bits==16u) {
            // float32->uint16 uses unsigned32 in its SIMD prefix, then
            // signed32 in the scalar tail. Half sources stay signed32.
            unsigned_instruction=position<(prefix_length/16L)*16L;
        }
    }
    return _eon_numpy_integer_instruction_status(input,instruction_bits,unsigned_instruction);
}
inline uint _eon_status_f_from_d(ulong x,thread uint& flags) {
    flags|=_eon_numpy_cast_status(x,32u);return f_from_d(x);
}
inline uint _eon_status_f_add(uint a,uint b,thread uint& flags) {
    if(_eon_narrow_snan32(a)||_eon_narrow_snan32(b))flags|=8u;
    uint r=f_add(a,b);
    if((a&0x7fffffffu)==0x7f800000u&&(b&0x7fffffffu)==0x7f800000u&&((a^b)>>31))flags|=8u;
    if((a&0x7fffffffu)<0x7f800000u&&(b&0x7fffffffu)<0x7f800000u&&(r&0x7fffffffu)==0x7f800000u)flags|=2u;
    return r;
}
inline uint _eon_status_f_sub(uint a,uint b,thread uint& flags) {
    return _eon_status_f_add(a,b^0x80000000u,flags);
}
inline uint _eon_status_f_mul(uint a,uint b,thread uint& flags) {
    uint aa=a&0x7fffffffu,ab=b&0x7fffffffu;
    if(_eon_narrow_snan32(a)||_eon_narrow_snan32(b))flags|=8u;
    if((!aa&&ab==0x7f800000u)||(!ab&&aa==0x7f800000u))flags|=8u;
    if(aa<0x7f800000u&&ab<0x7f800000u)
        flags|=_eon_numpy_cast_status(d_mul(f_to_d(a),f_to_d(b)),32u);
    return f_mul(a,b);
}
inline uint _eon_status_f_div(uint a,uint b,thread uint& flags) {
    uint aa=a&0x7fffffffu,ab=b&0x7fffffffu;
    if(_eon_narrow_snan32(a)||_eon_narrow_snan32(b))flags|=8u;
    if((!aa&&!ab)||(aa==0x7f800000u&&ab==0x7f800000u))flags|=8u;
    else if(!ab&&aa&&aa<0x7f800000u)flags|=1u;
    if(aa<0x7f800000u&&ab&&ab<0x7f800000u)
        flags|=_eon_numpy_cast_status(d_div(f_to_d(a),f_to_d(b)),32u);
    return f_div(a,b);
}
'''


def _fma_status():
    # Reuse the exact existing128-bit FMA alignment and its pre-round sticky
    # bits. A rounded binary64 comparison would miss an inexact tiny32 FMA.
    from .binary32 import MSL_HEADER
    from .numpy_math_status import _definitions
    source=_definitions(MSL_HEADER)['clang_fma']
    source=source.replace('inline ulong clang_fma(',
        '__attribute__((noinline)) ulong _eon_narrow_fma_exact(')
    source=source.replace('bool single) {','bool single,thread uint& flags) {')
    source=source.replace('ulong(f_from_d(result))','ulong(_eon_status_f_from_d(result,flags))')
    source=source.replace('ulong remainder=significand&7ul;',
        'bool tiny=e==0; ulong remainder=significand&7ul; if(tiny&&remainder)flags|=4u;')
    source=source.replace('if(e>=255) return ulong((sign<<31)|0x7f800000u);',
        'if(e>=255) {flags|=2u;return ulong((sign<<31)|0x7f800000u);}')
    return source+r'''
inline uint _eon_status_f_fma(uint a,uint b,uint c,thread uint& flags) {
    uint aa=a&0x7fffffffu,ab=b&0x7fffffffu,ac=c&0x7fffffffu;
    if(_eon_narrow_snan32(a)||_eon_narrow_snan32(b)||_eon_narrow_snan32(c))flags|=8u;
    if(aa<=0x7f800000u&&ab<=0x7f800000u&&ac<=0x7f800000u&&
       ((!aa&&ab==0x7f800000u)||(!ab&&aa==0x7f800000u)||
        ((aa==0x7f800000u||ab==0x7f800000u)&&ac==0x7f800000u&&((a^b^c)>>31))))flags|=8u;
    (void)_eon_narrow_fma_exact(f_to_d(a),f_to_d(b),f_to_d(c),true,flags);
    return f_fma(a,b,c);
}
#endif // EON_NUMPY_NARROW_STATUS_INCLUDED
'''


def _basic_graph(functions):
    from .basic_math32 import MSL_HEADER
    from .numpy_math_status import _definitions,_calls,_thread
    definitions=_definitions(MSL_HEADER)
    needed={ROOTS[name] for name in functions if name in ('exp','log','expm1','log1p')}
    if 'log' in functions:needed.add('_eon_basic32_log_reduced')
    leaves={'f_from_d','f_add','f_sub','f_mul','f_div','f_fma'}
    ordered=[]
    seen=set()
    def emit(name):
        if name in seen:return
        seen.add(name)
        for call in _calls(definitions[name],needed):
            if call.group(1)!=name:emit(call.group(1))
        ordered.append(_guard(_thread(definitions[name],needed|leaves),name))
    for name in sorted(needed):emit(name)
    return '\n'.join(ordered)


_TRIG_TINY = r'''
inline void _eon_narrow_trig32_tiny(uint x,thread uint& flags) {
    // NumPy NEON executes both polynomials before selecting sine/cosine.
    // Preserve discarded rounded products omitted by the numeric shortcut.
    (void)_eon_status_f_mul(x,as_type<uint>(0x1.45f306p-1f),flags);
    uint square=_eon_status_f_mul(x,x,flags);
    uint sine=_eon_status_f_fma(as_type<uint>(0x1.7d3bbcp-19f),square,as_type<uint>(-0x1.a06bbap-13f),flags);
    sine=_eon_status_f_fma(sine,square,as_type<uint>(0x1.11119ap-7f),flags);
    sine=_eon_status_f_fma(sine,square,as_type<uint>(-0x1.555556p-3f),flags);
    sine=_eon_status_f_fma(sine,square,0u,flags);
    (void)_eon_status_f_fma(sine,x,x,flags);
}
'''


@lru_cache(maxsize=64)
def metal_header(functions):
    """Emit selected32 status helpers and half flag-only hooks.

    Include binary64,binary32,binary16,math64 and original_header(functions)
    first. Flags use1divide,2overflow,4underflow,8invalid. Half numerical
    result LUTs remain owned by the existing renderer.
    """
    names=set(functions)
    unsupported=names-set(FUNCTIONS)
    if unsupported:raise ValueError('Unsupported narrow status functions: '+repr(sorted(unsupported)))
    sources=[_PRIMITIVES,_fma_status(),_basic_graph(names)]
    if names&{'sin','cos'}:sources.append(_guard(_TRIG_TINY,'trig_tiny'))
    for name in sorted(names):
        before='' if name in ('abs','sign','tanh') else 'if(_eon_narrow_snan32(x))flags|=8u;'
        if name in ('log','log10'):
            before+='if(!a)flags|=1u;else if((x>>31)&&a<=0x7f800000u)flags|=8u;'
        if name=='log1p':
            before+='if(x==0xbf800000u)flags|=1u;else if((x>>31)&&a>0x3f800000u&&a<=0x7f800000u)flags|=8u;'
        if name in ('arcsin','arccos'):
            before+='if(a>0x3f800000u&&a<=0x7f800000u)flags|=8u;'
        if name=='sqrt':before+='if((x>>31)&&a&&a<=0x7f800000u)flags|=8u;'
        if name in ('sin','cos','tan'):
            before+='if(a==0x7f800000u)flags|=8u;'
        if name in ('sin','cos'):
            before+='if(a<0x39800000u)_eon_narrow_trig32_tiny(x,flags);'
        if name in ('exp','log','expm1','log1p'):
            call='_eon_status_'+ROOTS[name]+'(x,flags)'
        elif name=='abs':call='(x&0x7fffffffu)'
        elif name=='sign':call='(a>0x7f800000u?x:!a?0u:(x&0x80000000u)|0x3f800000u)'
        elif name in ('floor','ceil'):call='f_from_d(d_'+name+'(f_to_d(x)))'
        else:call=ROOTS[name]+'(x)'
        after=''
        if name in ('exp','expm1','sinh','cosh'):
            after+='if(a<0x7f800000u&&(result&0x7fffffffu)==0x7f800000u)flags|=2u;'
        if name=='exp':
            after+='if(a<0x7f800000u&&(result&0x7fffffffu)<0x800000u)flags|=4u;'
        sources.append(_guard(f'inline uint _eon_numpy_{name}32_status(uint x,thread uint& flags){{'
            f'uint a=x&0x7fffffffu;{before}uint result={call};{after}return result;}}',name+'_32'))
        # All nonzero half inputs are normal32. Therefore the scalar32 and
        # SIMD32 tiny underflow distinction cannot affect a half loop. Tanh's
        # scalar entry additionally signals sNaNs, unlike its NumPy32 loop.
        sources.append(_guard(f'inline uint _eon_numpy_{name}16_flags(ushort raw){{'
            'uint flags=0u;ulong input=h_to_d(uint(raw));'
            + ('if((raw&0x7c00u)==0x7c00u&&(raw&0x1ffu)&&!(raw&0x200u))flags|=8u;' if name=='tanh' else '')
            + f'uint result=_eon_numpy_{name}32_status(h_d_to_f(input),flags);'
            'flags|=_eon_numpy_cast_status(h_f32_to_d(result),16u);return flags;}',name+'_16'))
    return '\n'.join(sources)
