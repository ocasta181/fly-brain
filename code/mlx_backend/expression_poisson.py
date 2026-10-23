"""Legacy NumPy Poisson algorithms, lowered through native binary64 helpers.

Adapted from NumPy 1.26.4 distributions.c (BSD-3-Clause); see third_party.
Parameter validation and all draws execute on the device. The caller checks
errors at its ordinary chunk boundary, without per-step synchronization.
"""
from functools import lru_cache
import ast
import numpy as np
from .compiler import _Renderer, _Expression
from .expression_rng import MT19937State, _MT_HEADER


class _MathRenderer(_Renderer):
    def expression(self, source):
        node=ast.parse(source,mode='eval').body if isinstance(source,str) else source
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('log','exp'):
            name={'log':'expr_apple_log','exp':'expr_poisson_exp_negative'}[node.func.id]
            return _Expression(f'{name}({self.convert(self.expression(node.args[0]),"float").code})','float')
        return super().expression(node)


def _source():
    from .binary64 import literal
    renderer=_MathRenderer({name:np.float64 for name in
        'lam prod u v slam loglam a b invalpha vr us candidate x x0 x2 gl gl0'.split()}, {})
    renderer.types.update(k=np.int64,n=np.int64)
    def e(expr):return renderer.convert(renderer.expression(expr),'float').code
    coefficients=[8.333333333333333e-02,-2.777777777777778e-03,
        7.936507936507937e-04,-5.952380952380952e-04,8.417508417508418e-04,
        -1.917526917526918e-03,6.410256410256410e-03,-2.955065359477124e-02,
        1.796443723688307e-01,-1.39243221690590e+00]
    return f'''
    constant ulong poisson_gamma_coeff[10]={{{','.join(literal(x) for x in coefficients)}}};
    inline long poisson_floor(ulong x) {{long k=d_to_int(x);return d_lt(x,d_from_int(k)) ? k-1 : k;}}
    inline ulong poisson_loggam(ulong x) {{
        if(d_eq(x,{literal(1.)})||d_eq(x,{literal(2.)})) return 0ul;
        long n=d_lt(x,{literal(7.)}) ? d_to_int({e('7-x')}) : 0;
        ulong x0={e('x+n')}, x2={e('(1/x0)*(1/x0)')};
        ulong gl0=poisson_gamma_coeff[9];
        for(int j=8;j>=0;--j) gl0=d_add(d_mul(gl0,x2),poisson_gamma_coeff[j]);
        ulong gl={e('gl0/x0+0.5*1.8378770664093453+(x0-0.5)*log(x0)-x0')};
        if(d_lt(x,{literal(7.)})) for(long j=1;j<=n;++j) {{gl=d_sub(gl,{e('log(x0-1)')});x0={e('x0-1')};}}
        return gl;
    }}
    inline long poisson_one(ulong lam,threadgroup uint* mt,thread uint* pos,uint lane) {{
        if(d_eq(lam,0ul)) return 0;
        if(d_lt(lam,{literal(10.)})) {{
            ulong enlam={e('exp(-lam)')},prod={literal(1.)};long k=0;
            while(true) {{ulong u=mt_next_double(mt,pos,lane);prod={e('prod*u')};
                if(d_gt(prod,enlam)) ++k;else return k;}}
        }}
        ulong slam={e('sqrt(lam)')},loglam={e('log(lam)')};
        ulong b={e('0.931+2.53*slam')},a={e('-0.059+0.02483*b')};
        ulong invalpha={e('1.1239+1.1328/(b-3.4)')},vr={e('0.9277-3.6224/(b-2)')};
        while(true) {{
            ulong u=d_sub(mt_next_double(mt,pos,lane),{literal(.5)});
            ulong v=mt_next_double(mt,pos,lane),us={e('0.5-abs(u)')};
            ulong candidate={e('(2*a/us+b)*u+lam+0.43')};long k=poisson_floor(candidate);
            if(d_ge(us,{literal(.07)})&&d_le(v,vr)) return k;
            if(k<0||(d_lt(us,{literal(.013)})&&d_gt(v,us))) continue;
            ulong left={e('log(v)+log(invalpha)-log(a/(us*us)+b)')};
            ulong right=d_sub({e('-lam+k*loglam')},poisson_loggam({e('k+1')}));
            if(d_le(left,right)) return k;
        }}
    }}
    '''


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx
    from .binary64 import MSL_HEADER,literal
    from .expression_rng_math import NORMAL_MATH_HEADER, POISSON_EXP_HEADER
    maximum=float(np.iinfo(np.int64).max)-10*np.sqrt(float(np.iinfo(np.int64).max))
    return mx.fast.metal_kernel(name='fly_expression_mt19937_poisson',
        input_names=['state','position','counts','lambdas','enabled'],
        output_names=['result','next_state','next_position','errors'],
        header=MSL_HEADER+NORMAL_MATH_HEADER+POISSON_EXP_HEADER+_MT_HEADER+_source(),
        source=f'''
        uint lane=thread_position_in_threadgroup.x,trial=threadgroup_position_in_grid.x;
        threadgroup uint mt[624];threadgroup atomic_uint invalid;
        if(lane==0) atomic_store_explicit(&invalid,0u,memory_order_relaxed);
        for(uint j=lane;j<624;j+=256) mt[j]=state[trial*624+j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint count=counts[trial],pos=position[trial];
        uint checks=DETAILED&&SCALAR_CALL ? uint(enabled[trial]) : count;
        for(uint j=lane;j<checks;j+=256) {{
            ulong lam=lambdas[LAMBDA_SCALAR ?0:trial*MAX_COUNT+j];
            uint fault=0;
            if(d_is_nan(lam))fault=DETAILED?(SCALAR_CALL?9u:5u):1u;
            else if(d_gt(lam,{literal(maximum)}))fault=DETAILED?5u:1u;
            else if(d_lt(lam,0ul))fault=DETAILED?(SCALAR_CALL?9u:1u):1u;
            if(fault)atomic_fetch_or_explicit(&invalid,fault,memory_order_relaxed);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint error=atomic_load_explicit(&invalid,memory_order_relaxed);
        if(!error) for(uint j=0;j<count;++j) {{
            long value=poisson_one(lambdas[LAMBDA_SCALAR ?0:trial*MAX_COUNT+j],mt,&pos,lane);
            if(!STATE_ONLY&&lane==0) result[trial*MAX_COUNT+j]=value;
        }}
        for(uint j=lane;j<624;j+=256) next_state[trial*624+j]=mt[j];
        if(lane==0) {{next_position[trial]=pos;errors[trial]=error;}}
        ''',compile_options={'math_mode':'safe'})


def poisson(state:MT19937State,lambdas,counts=None,*,_detailed=False,_scalar=False,_enabled=None):
    """Return (int64 draws, immutable state, per-trial device error flags)."""
    import mlx.core as mx
    if lambdas.ndim!=2 or lambdas.shape[0]!=state.trials or lambdas.dtype!=mx.uint64:
        raise ValueError('Poisson lambdas must contain binary64 bits per trial and draw')
    maximum=lambdas.shape[1]
    if maximum*state.trials>=2**32:raise ValueError('Poisson capacity exceeds safe Metal indexing')
    if counts is None:counts=mx.full((state.trials,),maximum,mx.uint32)
    if counts.shape!=(state.trials,) or counts.dtype!=mx.uint32:
        raise ValueError('Poisson draw counts must be uint32 per stream')
    if not maximum:return mx.zeros((state.trials,0),mx.int64),state,mx.zeros((state.trials,),mx.uint32)
    if _enabled is None:_enabled=mx.ones((state.trials,),mx.bool_)
    with mx.stream(mx.gpu):
        result,words,position,errors=_kernel()(
            inputs=[state.words,state.position,counts,lambdas.reshape(-1),_enabled],
            template=[('MAX_COUNT',maximum),('STATE_ONLY',False),('LAMBDA_SCALAR',False),('DETAILED',_detailed),('SCALAR_CALL',_scalar)],
            grid=(256*state.trials,1,1),threadgroup=(256,1,1),
            output_shapes=[(maximum*state.trials,),(624*state.trials,),(state.trials,),(state.trials,)],
            output_dtypes=[mx.int64,mx.uint32,mx.uint32,mx.uint32],init_value=0)
    return result.reshape(state.trials,maximum),MT19937State(words,position,state.trials,state.has_gauss,state.gauss),errors


def _poisson_states(state,counts,lam):
    """Exact constant-rate prefixes without discarded sample allocations."""
    import mlx.core as mx
    bits=int(np.asarray(float(lam)).view(np.uint64))
    with mx.stream(mx.gpu):
        _,words,position,errors=_kernel()(
            inputs=[state.words,state.position,counts,mx.array([bits],mx.uint64),mx.ones((state.trials,),mx.bool_)],
            template=[('MAX_COUNT',1),('STATE_ONLY',True),('LAMBDA_SCALAR',True),('DETAILED',False),('SCALAR_CALL',False)],
            grid=(256*state.trials,1,1),threadgroup=(256,1,1),
            output_shapes=[(1,),(624*state.trials,),(state.trials,),(state.trials,)],
            output_dtypes=[mx.int64,mx.uint32,mx.uint32,mx.uint32])
    return MT19937State(words,position,state.trials,state.has_gauss,state.gauss),errors
