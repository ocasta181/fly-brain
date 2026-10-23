"""Exact floating-operator hooks for the existing compiled typed graph."""

from functools import lru_cache
from dataclasses import dataclass
from hashlib import sha256
import ast


@dataclass(frozen=True)
class LiteralPowiProductPolicy:
    name: str
    header: str
    source: str
    assembly: str
    compiler: tuple
    flags: tuple
    input_kinds: tuple
    result_kind: str
    library_power: bool
    library_powi: bool

CYTHON_POWER_SUPPORT = r'''
#include <type_traits>
template<typename A,typename B>
inline typename std::enable_if<std::is_same<A,double>::value || std::is_same<B,double>::value,double>::type
_eon_cython_pow(A a,B b) { return pow((double)a,(double)b); }
template<typename A,typename B>
inline typename std::enable_if<!std::is_same<A,double>::value && !std::is_same<B,double>::value,float>::type
_eon_cython_pow(A a,B b) { return powf((float)a,(float)b); }
extern "C" __attribute__((noreturn)) void eon_operator_error(int kind);
extern "C" __attribute__((noreturn)) void eon_operator_scalar_error(int kind);
template<typename A,typename B> struct _eon_float_type {
    typedef typename std::conditional<std::is_same<A,double>::value || std::is_same<B,double>::value,double,float>::type type;
};
template<typename A,typename B> inline typename _eon_float_type<A,B>::type
_eon_cython_mod_once(A a,B b,bool scalar=false) {
    typedef typename _eon_float_type<A,B>::type T;
    T x=(T)a,y=(T)b;
    if(y==0) {if(scalar)eon_operator_scalar_error(1);else eon_operator_error(1);}
    T r=sizeof(T)==4 ? (T)fmodf((float)x,(float)y) : (T)fmod((double)x,(double)y);
    r+=((r!=0)&((r<0)^(y<0)))*y;
    return r;
}
template<typename A,typename B> inline double _eon_cython_floordiv(A a,B b,bool scalar=false) {
    if(b==0) {if(scalar)eon_operator_scalar_error(2);else eon_operator_error(2);}
    return floor((double)(a/b));
}
template<typename A,typename B> inline auto _eon_cython_divide(A a,B b,bool scalar=false)->decltype(a/b) {
    if(b==0) {if(scalar)eon_operator_scalar_error(2);else eon_operator_error(2);}
    return a/b;
}
template<typename A,typename B> inline auto _brian_mod(A a,B b)->decltype(a+b) {
    return a-b*floor(1.0*a/b);
}
template<typename A,typename B> inline auto _brian_floordiv(A a,B b)->decltype(a+b) {
    return floor(1.0*a/b);
}
'''


class Binary64Registry:
    """Operator hooks for the shared optimized binary64 arithmetic graph."""
    division_support_code=r'''
inline double _eon_cython_div(double x,double y,bool& error,int32_t& kind,bool scalar=false) {
    if(y==0.) {
        if(scalar) _eon_arithmetic_scalar_error(2);
        else _eon_arithmetic_vector_error(2);
    }
    return x/y;
}
'''
    _support_code=r'''
extern "C" __attribute__((noreturn)) void _eon_arithmetic_scalar_error(int32_t);
extern "C" __attribute__((noreturn)) void _eon_arithmetic_vector_error(int32_t);
inline double _eon_cython_mod_once(double x,double y,bool& error,int32_t& kind,bool scalar=false) {
    if(y==0.) {
        if(scalar) _eon_arithmetic_scalar_error(1);
        else _eon_arithmetic_vector_error(1);
    }
    double r=fmod(x,y);
    r+=((r!=0)&((r<0)^(y<0)))*y;
    return r;
}
'''

    def __init__(self,types,used_math,error_name,policies=None,identity=None,*,values=None,
                 used_arrays=None,index_error_name=None,target='cython'):
        self.types,self.used_math,self.error_name=types,used_math,error_name
        self.policies,self.identity=policies,identity
        self.values=values or {};self.used_arrays=used_arrays
        self.index_error_name=index_error_name;self.target=target
        from .compiler import _TimedArray
        self.timed={name:value for name,value in self.values.items() if isinstance(value,_TimedArray)}
        self.timed_aliases={name:name for name in self.timed}
        self.timed_intrinsics={}
        for index,name in enumerate(sorted(self.timed)):
            symbol=f'_eon_operator_timed_{index}'
            while symbol in self.types or symbol in self.values:
                symbol+='_'
            self.timed_intrinsics[symbol]=name

    @property
    def function_support_code(self):
        prototypes=[]
        for symbol,name in self.timed_intrinsics.items():
            arguments='double' if len(self.timed[name].array.shape)==1 else 'double,int32_t'
            prototypes.append(f'extern "C" __attribute__((const)) double {symbol}({arguments});')
        return '\n'.join(prototypes)

    @property
    def support_code(self):
        return self._support_code+'\n'+self.function_support_code

    def bind_function_aliases(self,aliases):
        for name,alias in aliases.items():
            if name in self.timed:self.timed_aliases[alias]=name

    def source_call(self,renderer,node):
        if not isinstance(node.func,ast.Name):return None
        name=self.timed_aliases.get(node.func.id)
        if name is None:return None
        symbol=next(symbol for symbol,original in self.timed_intrinsics.items() if original==name)
        return symbol+'('+','.join(renderer.render_node(arg) for arg in node.args)+')'

    def _render_function(self,function,argument_types,args):
        from .compiler import _Renderer
        from brian2.utils.stringtools import word_substitute
        names=[]
        for index in range(len(args)):
            name=f'_eon_operator_argument_{index}'
            while name in self.types or name in self.values:
                name+='_'
            names.append(name)
        renderer=_Renderer(dict(zip(names,argument_types)),self.values,used_math=self.used_math,
            used_arrays=self.used_arrays,index_error_name=self.index_error_name,
            reference_target=self.target)
        expression=ast.Call(ast.Name(function,ast.Load()),[ast.Name(name,ast.Load()) for name in names],[])
        code=renderer.expression(expression).code
        return word_substitute(code,dict(zip(names,(f'({value})' for value in args))))

    def late_expression(self,node,fused,emit):
        if fused is None:return None
        # LLVM retains powi until target instruction selection. Capture the
        # existing narrow product/add boundary with the same scalar assembly
        # translator used by float32. Keep its addend as an SSA input: an
        # apparent product there can be reused by other outputs and must
        # retain its original rounding before this contraction.
        for index in (0,1):
            if self.policies is None:break
            power,scale,addend=fused[index],fused[1-index],fused[2]
            if power.op.split('.')[0]!='powi' or power.args[1].op!='constant':continue
            inputs=(power.args[0],scale,addend)
            if any(item.type!='double' for item in inputs):continue
            command,flags,_=self.identity
            policy=literal_powi_product_policy(('float',)*3,'float',
                int(power.args[1].value),tuple(flags),tuple(command))
            self.policies[policy.name]=policy
            self.used_math.add('operator_remainder_float')
            if policy.library_power:self.used_math.add('operator_pow_float')
            if policy.library_powi:self.used_math.add('operator_powi_float')
            return policy.name+'('+','.join(emit(item) for item in inputs)+')'
        # Compiled ARM FMA checks signaling operands, then its addend's quiet
        # NaN before multiplicands. The mathematical three-limb d_fma has a
        # different exceptional operand priority; keep that shared primitive
        # unchanged and reuse the qualified selected-target operator wrapper.
        self.used_math.add('operator_remainder_float')
        return 'd_operator_fma('+','.join(emit(item) for item in fused)+')'

    def source_binary(self,renderer,node):
        if not isinstance(node.op,(ast.Div,ast.Mod,ast.FloorDiv)):return None
        def integer(part):
            if isinstance(part,ast.Constant):return isinstance(part.value,(bool,int))
            if isinstance(part,ast.Name):return __import__('numpy').dtype(self.types[part.id]).kind in 'biu'
            if isinstance(part,ast.UnaryOp):return integer(part.operand)
            if isinstance(part,ast.BinOp):return not isinstance(part.op,ast.Div) and integer(part.left) and integer(part.right)
            return False
        if not isinstance(node.op,ast.Div) and integer(node.left) and integer(node.right):return None
        role='scalar' if renderer.scalar_statement else 'vector'
        renderer.division_roles.add(role)
        a,z=renderer.render_node(node.left),renderer.render_node(node.right)
        error='_eon_'+role+'_error,_eon_'+role+'_error_kind,'+('true' if role=='scalar' else 'false')
        if isinstance(node.op,ast.Div):return f'_eon_cython_div({a},{z},{error})'
        if isinstance(node.op,ast.Mod):
            if getattr(renderer,'augmented_mod',False):
                return f'_eon_cython_mod_once({a},{z},{error})'
            return f'_eon_cython_mod_once((_eon_cython_mod_once({a},{z},{error})+({z})),{z},{error})'
        return f'floor(_eon_cython_div({a},{z},{error}))'

    def intrinsic(self,name,dtype,args):
        if name in self.timed_intrinsics:
            function=self.timed_intrinsics[name]
            types=['float64']+(['int32'] if len(args)==2 else [])
            return self._render_function(function,types,args)
        if name=='frem':name='fmod'
        operator=render_intrinsic(name,dtype,args,self.used_math)
        if operator is not None:return operator
        functions={'asin':'arcsin','acos':'arccos','atan':'arctan',
            **{name:name for name in ('sin','cos','tan','exp','log','expm1','log1p',
                'exprel','log10','sinh','cosh','tanh')}}
        function=functions.get(name.split('.')[0])
        if function is None or dtype!='float64':return None
        # Reuse the canonical target-aware function policy and its dependency
        # collector, with raw SSA values substituted only after rendering.
        return self._render_function(function,['float64']*len(args),args)

    def constant(self,name,dtype,args):
        name=name.split('.')[0]
        if name not in ('frem','fmod','pow','powi','floor','ceil','sin','cos','tan',
                'asin','acos','atan','exp','exp2','log','expm1','log1p','log10',
                'sinh','cosh','tanh'):return None
        import ctypes
        library=ctypes.CDLL(None)
        function=getattr(library,'__powidf2' if name=='powi' else 'fmod' if name=='frem' else name)
        function.restype=ctypes.c_double
        function.argtypes=([ctypes.c_double,ctypes.c_int] if name=='powi' else [ctypes.c_double]*len(args))
        return function(*args)

    def status(self,condition,kind):
        self.used_math.add('operator_zero_division')
        return (f'if({condition}) {{ if(!{self.error_name}) {self.error_name}_kind={kind}; '
                f'{self.error_name}=true; }}')


def render_intrinsic(function, dtype, arguments, used_math):
    """Return a qualified primitive, retaining the graph's typed call boundary."""
    name=function.removeprefix('llvm.').split('.')[0]
    single=dtype=='float32'
    prefix='f' if single else 'd'
    kind='float32' if single else 'float'
    if name in ('fma','fmuladd'):
        # This is an actual compiled-target intrinsic, including widened
        # double arithmetic within a float32 storage block. Its NaN operand
        # order follows ARM FMA, rather than separate multiply/add fallback.
        used_math.add('operator_remainder_'+kind)
        return prefix+'_operator_fma('+','.join(arguments)+')'
    if name in ('exp','expf','log','logf','expm1','expm1f','log1p','log1pf'):
        operation=name.removesuffix('f')
        used_math.add(operation+'_f32' if single else operation)
        helper=('_eon_basic32_'+operation if single else
                {'exp':'d_exponential','log':'d_logarithm'}.get(operation,'_eon_stable_'+operation))
        return helper+'('+','.join(arguments)+')'
    if name in ('pow','powf','powi'):
        operation='powi' if name=='powi' else 'pow'
        used_math.add('operator_'+operation+'_'+kind)
        return prefix+'_operator_'+operation+'('+','.join(arguments)+')'
    if name in ('exp2','exp2f'):
        used_math.add('operator_exp2_'+kind)
        return prefix+'_operator_exp2('+','.join(arguments)+')'
    if name in ('fmod','fmodf'):
        used_math.add('operator_remainder_'+kind)
        return prefix+'_operator_fmod('+','.join(arguments)+')'
    if name in ('floor','floorf','ceil','ceilf'):
        operation=name.removesuffix('f')
        code='d_'+operation+'('+(f'f_to_d({arguments[0]})' if single else arguments[0])+')'
        return f'f_from_d({code})' if single else code
    return None


@lru_cache(maxsize=64)
def literal_powi_product_policy(input_kinds,result_kind,exponent,flags,compiler):
    """Capture late expansion/contraction of a constant powi inside mul/add.

    LLVM retains powi in optimized IR; its target lowering can reassociate the
    expanded multiplications with the surrounding contractable product. Reuse
    the existing narrow scalar assembly recipe instead of treating that powi
    as an already-rounded opaque function result.
    """
    from .operator_policy import _compile_assembly, _translate
    types=['float' if kind=='float32' else 'double' for kind in input_kinds]
    result='float' if result_kind=='float32' else 'double'
    builtin='__builtin_powif' if input_kinds[0]=='float32' else '__builtin_powi'
    source=('#include <math.h>\nextern "C" '+result+' eon_power('+','.join(
        f'{kind} {name}' for name,kind in zip(('x','s','z'),types))+
        f'){{return s*{builtin}(x,{int(exponent)})+z;}}\n')
    assembly=_compile_assembly(source,compiler,flags,prefix='eon-powi-context-',filename='context')
    name='_eon_literal_powi_product_'+sha256(repr((source,assembly,flags,compiler)).encode()).hexdigest()[:20]
    header,power,powi=_translate(assembly,input_kinds[0],result_kind,name,
        arguments=tuple(zip(('x','s','z'),input_kinds)))
    return LiteralPowiProductPolicy(name,header,source,assembly,compiler,flags,input_kinds,result_kind,power,powi)


def literal_powi_product(node,flags,compiler):
    """Return (recipe, ordered graph inputs) for this late-codegen boundary."""
    if not (node.op=='call:llvm.fma' and node.contract):return None
    for index in (0,1):
        power=node.args[index]
        if (power.op.startswith('call:llvm.powi.') and power.args[1].op=='constant'
                and all(arg.dtype in ('float32','float64') for arg in
                    (power.args[0],node.args[1-index],node.args[2]))):
            inputs=(power.args[0],node.args[1-index],node.args[2])
            kinds=tuple('float32' if arg.dtype=='float32' else 'float' for arg in inputs)
            policy=literal_powi_product_policy(kinds,'float32' if node.dtype=='float32' else 'float',
                int(power.args[1].value),tuple(flags),tuple(compiler))
            return policy,inputs
    return None
