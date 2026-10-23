"""Exact polar candidates reused across interleaved shared-input draws."""
from functools import lru_cache
import mlx.core as mx


@lru_cache(maxsize=1)
def _candidates():
    from .binary64 import MSL_HEADER
    from .expression_rng_math import NORMAL_MATH_HEADER
    return mx.fast.metal_kernel(name='expression_shared_normal_candidates',
        input_names=['raw'],output_names=['first','second','radii','valid'],header=MSL_HEADER+NORMAL_MATH_HEADER,
        source=r'''
        uint j=thread_position_in_grid.x;if(j>=B*CAPACITY)return;
        uint trial=j/CAPACITY,local=j%CAPACITY,parity=local/(CAPACITY/2),pair=local%(CAPACITY/2);
        uint offset=2*pair+parity;
        if(offset+1>=CAPACITY){first[j]=0ul;second[j]=0ul;radii[j]=0ul;valid[j]=0u;return;}
        ulong x=d_sub(d_mul(0x4000000000000000ul,raw[trial*CAPACITY+offset]),0x3ff0000000000000ul);
        ulong y=d_sub(d_mul(0x4000000000000000ul,raw[trial*CAPACITY+offset+1]),0x3ff0000000000000ul);
        ulong rr=d_add(d_mul(x,x),d_mul(y,y));
        bool ok=d_lt(rr,0x3ff0000000000000ul)&&!d_eq(rr,0ul);
        valid[j]=uint(ok);first[j]=y;second[j]=x;radii[j]=rr;
        ''',compile_options={'math_mode':'safe'})


def prepare(prefix):
    from dataclasses import replace
    capacity=prefix.capacity;trials=prefix.initial.trials
    if capacity%2:return prefix
    first,second,radii,valid=_candidates()(inputs=[prefix.data.values],
        template=[('B',trials),('CAPACITY',capacity)],grid=(trials*capacity,1,1),threadgroup=(256,1,1),
        output_shapes=[(trials,2,capacity//2)]*4,output_dtypes=[mx.uint64,mx.uint64,mx.uint64,mx.uint32])
    ranks=mx.cumsum(valid,axis=-1).astype(mx.uint64)
    return replace(prefix,data=replace(prefix.data,normal=mx.stack([first,second,radii,ranks])))


@lru_cache(maxsize=1)
def _select():
    from .binary64 import MSL_HEADER
    from .expression_rng_math import NORMAL_MATH_HEADER
    return mx.fast.metal_kernel(name='expression_shared_normal_select',
        input_names=['data','cursor','counts','cached','gauss'],
        output_names=['values','done','consumed','flag','saved'],header=MSL_HEADER+NORMAL_MATH_HEADER,source=r'''
        uint gid=thread_position_in_grid.x;if(gid>=B*PAIRS)return;
        uint trial=gid/PAIRS,pair=gid%PAIRS;
        uint start=cursor[trial]/2,parity=cursor[trial]%2,n=CAPACITY/2;
        ulong base=ulong(trial)*CAPACITY+parity*n,rankbase=3ul*B*CAPACITY+base;
        uint prior=start?uint(data[rankbase+min(start,n)-1]):0;
        uint available=uint(data[rankbase+n-1])-prior;
        uint cache=uint(cached[trial]&&counts[trial]);
        uint remaining=counts[trial]-cache,take=min(remaining,2*available);
        if(pair==0){
            done[trial]=cache+take;
            flag[trial]=counts[trial]?uint(take==remaining)*(remaining%2):cached[trial];
            if(!remaining||take<remaining||remaining%2==0)saved[trial]=counts[trial]?0ul:gauss[trial];
            if(!remaining)consumed[trial]=0;
            if(remaining&&take<remaining)consumed[trial]=CAPACITY-cursor[trial]-(CAPACITY-cursor[trial])%2;
            if(cache)values[trial*MAXIMUM]=gauss[trial];
        }
        uint row=2*pair+cache;
        if(row>=MAXIMUM)return;
        if(2*pair>=take){values[trial*MAXIMUM+row]=0ul;if(row+1<MAXIMUM)values[trial*MAXIMUM+row+1]=0ul;return;}
        uint wanted=prior+pair+1,lo=start,hi=n;
        while(lo<hi){uint mid=lo+(hi-lo)/2;if(data[rankbase+mid]<wanted)lo=mid+1;else hi=mid;}
        ulong rr=data[2ul*B*CAPACITY+base+lo];
        ulong f=d_sqrt(d_div(d_mul(0xc000000000000000ul,expr_apple_log(rr)),rr));
        ulong first=d_mul(f,data[base+lo]),second=d_mul(f,data[ulong(B)*CAPACITY+base+lo]);
        values[trial*MAXIMUM+row]=first;
        if(row+1<MAXIMUM)values[trial*MAXIMUM+row+1]=2*pair+1<take?second:0ul;
        if(2*pair+2>=take){
            if(take==remaining)consumed[trial]=2*lo+parity+2-cursor[trial];
            if(take==remaining&&remaining%2)saved[trial]=second;
        }
        ''',compile_options={'math_mode':'safe'})


def draw(state,cache,maximum,counts):
    from dataclasses import replace
    from .expression_rng import MT19937State,_serial_normals
    from .expression_rng_prefix import _merge
    values,done,consumed,flag,gauss=_select()(inputs=[cache.data.normal,cache.cursor,counts,state.has_gauss,state.gauss],
        template=[('B',state.trials),('CAPACITY',cache.capacity),('MAXIMUM',maximum),('PAIRS',(maximum+1)//2)],
        grid=(state.trials*((maximum+1)//2),1,1),threadgroup=(256,1,1),
        output_shapes=[(state.trials,maximum),*[(state.trials,)]*4],
        output_dtypes=[mx.uint64,mx.uint32,mx.uint32,mx.uint32,mx.uint64])
    cursor=cache.cursor+consumed
    end=MT19937State(state.words,state.position,state.trials,flag,gauss)
    tail,end=_serial_normals(end,maximum,(counts-done).astype(mx.uint32),_clear_unused=False,_prefix=(cache,cursor))
    values=_merge(values,tail,done,counts,maximum)
    cursor=mx.where(done<counts,cache.capacity,cursor).astype(mx.uint32)
    return values,end,replace(cache,cursor=cursor,stream=end)
