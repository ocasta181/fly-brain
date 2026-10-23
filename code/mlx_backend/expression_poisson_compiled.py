"""Frozen default-clang Brian CPP Poisson operations and int32 conversion.

The operation order includes default CPU FMA/reassociation, checked against
Brian2.8's actual generator source compiled with its unchanged default flags.
This is separate from NumPy's int64 distribution and parameter validation.
"""
from functools import lru_cache
from .expression_rng import MT19937State,_MT_HEADER


def source(signature='threadgroup uint* mt,thread uint* pos,uint lane',draw='mt_next_double(mt,pos,lane)',
           *,target='cpp',error='',check=''):
    from .binary64 import literal
    coefficients=[8.333333333333333e-02,-2.777777777777778e-03,
        7.936507936507937e-04,-5.952380952380952e-04,8.417508417508418e-04,
        -1.917526917526918e-03,6.410256410256410e-03,-2.955065359477124e-02,
        1.796443723688307e-01,-1.39243221690590e+00]
    correction=('''
        for(int j=0;j<n;++j)gl=d_sub(gl,compiled_log(d_sub(x0,d_from_int(j+1))));
        '''if target=='cython'else'''
        int paired=n&~1;ulong even=0ul;
        for(int j=0;j<paired;j+=2) {
            ulong point=d_sub(x0,d_from_int(j));
            gl=d_sub(gl,compiled_log(d_sub(point,0x3ff0000000000000ul)));
            even=d_sub(even,compiled_log(d_sub(point,0x4000000000000000ul)));
        }
        if(paired)gl=d_add(even,gl);
        if(n&1)gl=d_sub(gl,compiled_log(d_sub(x0,d_from_int(paired+1))));
        ''')
    return f'''
    constant ulong compiled_gamma_coeff[10]={{{','.join(literal(x)for x in coefficients)}}};
    inline ulong compiled_log(ulong x) {{
        if(d_is_zero(x))return 0xfff0000000000000ul;
        if(d_is_nan(x))return x|(1ul<<51);
        if(x>>63)return D_QNAN;
        if(d_exp(x)==2047)return x;
        return expr_apple_log(x);
    }}
    inline int compiled_floor_int32(ulong x) {{
        if(d_is_nan(x))return 0;
        if(d_ge(x,{literal(2147483647.)}))return 2147483647;
        if(d_le(x,{literal(-2147483648.)}))return (-2147483647-1);
        long k=d_to_int(x);return int(d_lt(x,d_from_int(k))?k-1:k);
    }}
    inline ulong compiled_gamma(int k) {{
        if(k<2)return 0ul;
        // Default Cython clang uses frintz(7-x) directly for x0, even when
        // the separate int32 n conversion saturates. At k=INT_MAX, x0=7
        // and wrapped n+1 skips the correction loop (verified machine code).
        ulong x={'k==2147483647?'+literal(7.)+':'if target=='cython'else''}d_from_int(long(uint(k)+1u));
        int n=d_le(x,{literal(7.)})?int(d_to_int(d_sub({literal(7.)},x))):0;
        ulong x0=d_add(x,d_from_int(n)),x2=d_div({literal(1.)},d_mul(x0,x0));
        ulong gl0=compiled_gamma_coeff[9];
        for(int j=8;j>=0;--j)gl0=expr_fma(gl0,x2,compiled_gamma_coeff[j]);
        ulong gl=d_add(expr_fma(d_sub(x0,{literal(.5)}),compiled_log(x0),
            d_sub(0x3fed67f1c864beb4ul,x0)),d_div(gl0,x0));
        {correction}
        return gl;
    }}
    inline int compiled_poisson(ulong lam,{signature}) {{
        {check}
        if(d_eq(lam,0ul))return 0;
        if(!d_ge(lam,{literal(10.)})) {{
            if(d_is_nan(lam)||d_lt(lam,0ul)){{ulong ignored={draw};return 0;}}
            ulong enlam=expr_poisson_exp_negative(d_neg(lam)),prod={literal(1.)};uint count=0;
            while(true){{ulong u={draw};prod=d_mul(prod,u);if(d_gt(prod,enlam))++count;else return as_type<int>(count);}}
        }}
        ulong loglam=compiled_log(lam),product=d_mul({literal(2.53)},d_sqrt(lam));
        ulong b=d_add(product,{literal(.931)}),a=expr_fma(b,{literal(.02483)},{literal(-.059)});
        ulong invalpha=d_add(d_div({literal(1.1328)},d_add(product,{literal(.931-3.4)})),{literal(1.1239)});
        ulong vr=d_add(d_div({literal(-3.6224)},d_add(product,{literal(.931-2)})),{literal(.9277)});
        ulong logalpha=compiled_log(invalpha),twoa=d_add(a,a),lam43=d_add(lam,{literal(.43)});
        while(true){{
            ulong u=d_sub({draw},{literal(.5)}),v={draw},us=d_sub({literal(.5)},d_abs(u));
            {'if(d_is_zero(us)){'+error+'return 0;}'if target=='cython'else''}
            int k=compiled_floor_int32(expr_fma(d_add(d_div(twoa,us),b),u,lam43));
            if(d_ge(us,{literal(.07)})&&d_le(v,vr))return k;
            if(k<0||(d_lt(us,{literal(.013)})&&d_gt(v,us)))continue;
            ulong logv=compiled_log(v),logden=compiled_log(d_add(d_div(a,d_mul(us,us)),b));
            ulong left={'d_add(d_sub(logalpha,logden),logv)'if target=='cython'else'd_sub(d_add(logv,logalpha),logden)'};
            ulong right={'expr_fma(d_from_int(k),loglam,d_neg(d_add(lam,compiled_gamma(k))))'if target=='cython'else'd_sub(d_mul(d_from_int(k),loglam),d_add(lam,compiled_gamma(k)))'};
            if(d_le(left,right))return k;
        }}
    }}
    '''


@lru_cache(maxsize=4)
def _kernel(state_only=False,scalar=False):
    import mlx.core as mx
    from .binary64 import MSL_HEADER
    from .expression_rng_math import NORMAL_MATH_HEADER,POISSON_EXP_HEADER
    return mx.fast.metal_kernel(name='expression_cpp_poisson',
        input_names=['state','position','counts','lambdas'],
        output_names=['values','next_state','next_position'],
        header=MSL_HEADER+NORMAL_MATH_HEADER+POISSON_EXP_HEADER+_MT_HEADER+source(),source=r'''
        uint lane=thread_position_in_threadgroup.x,trial=threadgroup_position_in_grid.x;
        threadgroup uint mt[624];for(uint j=lane;j<624;j+=256)mt[j]=state[trial*624+j];
        uint pos=position[trial];threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<counts[trial];++j){int value=compiled_poisson(lambdas[LAMBDA_SCALAR?0:trial*MAX_COUNT+j],mt,&pos,lane);
            if(!STATE_ONLY&&lane==0)values[trial*MAX_COUNT+j]=value;}
        for(uint j=lane;j<624;j+=256)next_state[trial*624+j]=mt[j];
        if(lane==0)next_position[trial]=pos;
        ''',compile_options={'math_mode':'safe'})


def poisson(state,lambdas,counts=None):
    import mlx.core as mx
    if lambdas.ndim!=2 or lambdas.shape[0]!=state.trials or lambdas.dtype!=mx.uint64:
        raise ValueError('Compiled Poisson rates must contain binary64 bits per trial and draw')
    maximum=lambdas.shape[1]
    if maximum*state.trials>=2**32:raise ValueError('Compiled Poisson capacity exceeds safe Metal indexing')
    if counts is None:counts=mx.full((state.trials,),maximum,mx.uint32)
    if counts.shape!=(state.trials,)or counts.dtype!=mx.uint32:raise ValueError('Invalid compiled Poisson counts')
    if not maximum:return mx.zeros((state.trials,0),mx.int32),state
    values,words,position=_kernel()(inputs=[state.words,state.position,counts,lambdas.reshape(-1)],
        template=[('MAX_COUNT',maximum),('STATE_ONLY',False),('LAMBDA_SCALAR',False)],
        grid=(state.trials*256,1,1),threadgroup=(256,1,1),
        output_shapes=[(maximum*state.trials,),(624*state.trials,),(state.trials,)],
        output_dtypes=[mx.int32,mx.uint32,mx.uint32],init_value=0)
    return values.reshape(state.trials,maximum),MT19937State(words,position,state.trials,state.has_gauss,state.gauss)


def _poisson_states(state,counts,lam):
    import mlx.core as mx
    from .binary64 import literal
    _,words,position=_kernel(True,True)(
        inputs=[state.words,state.position,counts,mx.array([int(literal(lam)[:-2])],mx.uint64)],
        template=[('MAX_COUNT',1),('STATE_ONLY',True),('LAMBDA_SCALAR',True)],
        grid=(state.trials*256,1,1),threadgroup=(256,1,1),
        output_shapes=[(1,),(624*state.trials,),(state.trials,)],
        output_dtypes=[mx.int32,mx.uint32,mx.uint32])
    return MT19937State(words,position,state.trials,state.has_gauss,state.gauss),mx.zeros((state.trials,),mx.uint32)
