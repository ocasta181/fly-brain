"""Exact MT19937 jumps for parallel pieces of one accepted stream.

The degree19937 polynomial of the logical single-word transition was recovered
independently from NumPy1.26.4 uint32 outputs and verified against complete624
word states beyond its degree, including arbitrary accepted words. A full MT
block is624 such transitions. A short untempered sequence turns each jump
polynomial into an XOR convolution. Its output words execute independently;
there is no sequential Horner barrier for every polynomial bit.
"""
from functools import lru_cache
from dataclasses import dataclass
from operator import index
import numpy as np
from .expression_rng import MT19937State

_WORD_POLY=sum(1<<degree for degree in (
    0,1189,1416,1585,1643,1870,2493,2773,3000,3227,3454,3681,3908,4135,
    4362,4753,5661,6337,6569,7129,7477,7525,7583,7752,7979,8206,9505,9901,
    9969,10128,10693,10761,10920,11089,11147,11157,11215,11321,11374,11384,11485,11611,
    11712,11717,11838,11881,11944,11997,12277,12335,12393,12504,12509,12620,12673,12731,
    12736,12789,12905,12958,12963,13137,13185,13190,13243,13301,13412,13528,13533,13639,
    13697,13760,13813,13866,14093,14151,14209,14320,14325,14436,14547,14552,14605,14721,
    14774,14779,14953,15001,15006,15059,15117,15228,15344,15349,15455,15513,15576,15629,
    15682,15909,15967,16025,16136,16141,16252,16363,16368,16421,16537,16590,16595,16817,
    16822,16875,16933,17044,17160,17271,17329,17445,17498,17725,17783,17841,17952,18068,
    18179,18237,18406,18633,18691,18860,19087,19314,19937,
))


def _mod(value):
    while value.bit_length()>=_WORD_POLY.bit_length():
        value^=_WORD_POLY<<(value.bit_length()-_WORD_POLY.bit_length())
    return value


def _multiply(a,b):
    result=0
    while b:
        bit=b&-b;result^=a<<(bit.bit_length()-1);b^=bit
    return _mod(result)


_SQUARE_BYTES=tuple(sum(((byte>>bit)&1)<<(2*bit)for bit in range(8)).to_bytes(2,'little')
                    for byte in range(256))


def _square(a):
    data=a.to_bytes((a.bit_length()+7)//8,'little')
    return _mod(int.from_bytes(b''.join(_SQUARE_BYTES[byte]for byte in data),'little'))


def _power_x(exponent):
    result=1;power=2
    while exponent:
        if exponent&1:result=_multiply(result,power)
        exponent>>=1
        if exponent:power=_square(power)
    return result


@lru_cache(maxsize=32)
def _coefficients(parts,block_stride):
    coefficients=np.zeros((2,parts,624),np.uint32)
    degrees=np.full((2,parts),-1,np.int32)
    factor=_power_x(block_stride*624)
    power=_power_x(block_stride*624-1)
    for part in range(1,parts):
        blocks=part*block_stride
        if part>1:power=_multiply(power,factor)
        for variant in (0,1):
            # Position0 ends a full draw block at624 without twisting; all
            # other positions return to themselves after one full block.
            if blocks<=variant:continue
            q=power
            if variant:
                for _ in range(624):q=((q^_WORD_POLY)>>1)if q&1 else q>>1
            coefficients[variant,part]=np.frombuffer(q.to_bytes(2496,'little'),np.uint32)
            degrees[variant,part]=q.bit_length()-1
    coefficients.flags.writeable=False;degrees.flags.writeable=False
    return coefficients,degrees


@lru_cache(maxsize=1)
def _jump_sequence_kernel():
    import mlx.core as mx
    from .expression_rng import _MT_HEADER
    return mx.fast.metal_kernel(name='expression_mt_jump_sequence',
        input_names=['words'],output_names=['sequence'],header=_MT_HEADER,
        source=r'''
        uint lane=thread_position_in_threadgroup.x,trial=threadgroup_position_in_grid.x;
        threadgroup uint mt[624];
        for(uint j=lane;j<624;j+=256){
            mt[j]=words[trial*624+j];sequence[ulong(trial)*SEQ_WORDS+j]=mt[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint begin=624;begin<SEQ_WORDS;begin+=624){
            mt_twist(mt,lane);
            for(uint j=lane;j<min(624u,uint(SEQ_WORDS)-begin);j+=256)
                sequence[ulong(trial)*SEQ_WORDS+begin+j]=mt[j];
        }
        ''',compile_options={'math_mode':'safe'})


@lru_cache(maxsize=1)
def _jump_kernel():
    import mlx.core as mx
    return mx.fast.metal_kernel(name='expression_mt_word_jump_convolution',
        input_names=['sequence','positions','coefficients','degrees'],
        output_names=['jumped','jumped_positions'],
        source=r'''
        uint index=thread_position_in_grid.x;if(index>=B*PARTS*624)return;
        uint shard=index/624,word=index%624,trial=shard/PARTS,part=shard%PARTS;
        uint position=positions[trial],variant=position==0?1u:0u;
        int degree=degrees[variant*PARTS+part];
        uint value=0;
        if(degree<0)value=sequence[ulong(trial)*SEQ_WORDS+word];
        else{
            // T^k(s)[i] is the untempered sequence at k+i. The existing
            // polynomial acts on T(s), so its coefficient k uses k+i+1.
            for(uint j=0;j<=uint(degree)/32;++j){
                uint bits=coefficients[(variant*PARTS+part)*624+j];
                while(bits){
                    uint bit=31u-clz(bits&(~bits+1u));
                    value^=sequence[ulong(trial)*SEQ_WORDS+word+1+32*j+bit];
                    bits&=bits-1u;
                }
            }
        }
        jumped[index]=value;
        if(word==0)jumped_positions[shard]=(part&&position==0)?624u:position;
                ''',compile_options={'math_mode':'safe'})


def jump_uniform_blocks(state,parts,block_stride):
    """States at offsets part*block_stride*312 uniforms, preserving normal cache.

    Each output row is an exact NumPy snapshot, not a different random stream.
    It can therefore generate one contiguous piece independently on Metal.
    """
    import mlx.core as mx
    parts=index(parts);block_stride=index(block_stride)
    if parts<1 or block_stride<1 or parts*state.trials*624>=2**32:
        raise ValueError('Invalid parallel MT block layout')
    if parts==1:return state
    coefficients,degrees=_coefficients(parts,block_stride)
    with mx.stream(mx.gpu):
        sequence_words=624+19937+1
        sequence=_jump_sequence_kernel()(inputs=[state.words],template=[('SEQ_WORDS',sequence_words)],
            grid=(256*state.trials,1,1),threadgroup=(256,1,1),
            output_shapes=[(sequence_words*state.trials,)],output_dtypes=[mx.uint32])[0]
        words,positions=_jump_kernel()(
            inputs=[sequence,state.position,mx.array(coefficients.reshape(-1)),mx.array(degrees.reshape(-1))],
            template=[('PARTS',parts),('B',state.trials),('SEQ_WORDS',sequence_words)],
            grid=(624*state.trials*parts,1,1),threadgroup=(256,1,1),
            output_shapes=[(624*state.trials*parts,),(state.trials*parts,)],output_dtypes=[mx.uint32,mx.uint32])
        repeat=lambda value:mx.broadcast_to(value.reshape(-1,1),(state.trials,parts)).reshape(-1)
        return MT19937State(words,positions,state.trials*parts,repeat(state.has_gauss),repeat(state.gauss))


def parallel_uniforms(state,maximum,parts=32):
    """Generate contiguous pieces of each existing stream on independent groups."""
    import mlx.core as mx
    from .expression_rng import uniforms
    maximum=index(maximum);parts=index(parts)
    if maximum<0 or parts<1:raise ValueError('Invalid parallel uniform capacity')
    if not maximum:return uniforms(state,0)
    parts=min(parts,(maximum+311)//312)
    stride=(maximum+parts*312-1)//(parts*312)
    span=stride*312;padded=parts*span
    if padded*state.trials>=2**32:return uniforms(state,maximum)
    with mx.stream(mx.gpu):
        starts=jump_uniform_blocks(state,parts,stride)
        counts=mx.maximum(mx.minimum(maximum-mx.arange(parts)*span,span),0).astype(mx.uint32)
        counts=mx.broadcast_to(counts[None,:],(state.trials,parts)).reshape(-1)
        values,ends=uniforms(starts,span,counts)
        last=(maximum-1)//span
        words=ends.words.reshape(state.trials,parts,624)[:,last,:].reshape(-1)
        position=ends.position.reshape(state.trials,parts)[:,last].reshape(-1)
        return (values.reshape(state.trials,padded)[:,:maximum],
                MT19937State(words,position,state.trials,state.has_gauss,state.gauss))


@dataclass(frozen=True)
class UniformPool:
    """Bounded lookahead values plus exact state after every completed frame."""
    values: object
    words: object
    positions: object
    trials: int
    has_gauss: object
    gauss: object
    width: int
    frames: int

    def state_at(self,frame):
        frame=index(frame)
        if not 0<=frame<self.frames:raise IndexError('Uniform checkpoint frame out of range')
        return MT19937State(self.words[:,frame,:].reshape(-1),self.positions[:,frame].reshape(-1),
                            self.trials,self.has_gauss,self.gauss)


def uniform_pool(state,width,frames,*,parts=None,counts=None):
    """Generate on Metal; optional device counts bound each speculative prefix.

    Only completed frames have checkpoints when counts limit a stream. The
    prefix-restoration caller selects only these completed checkpoints.
    """
    import mlx.core as mx
    from .expression_rng import _uniform_kernel
    width=index(width);frames=index(frames);maximum=width*frames
    if width<1 or frames<1 or maximum*state.trials>=2**32 or frames*state.trials*624>=2**32:
        raise ValueError('Invalid uniform pool layout')
    limited=counts is not None
    if counts is None:counts=mx.full((state.trials,),maximum,mx.uint32)
    if counts.shape!=(state.trials,)or counts.dtype!=mx.uint32:
        raise ValueError('Uniform pool counts must be a uint32 vector per stream')
    if parts is None:parts=32 if maximum>=8*1024**2 else 1
    parts=min(index(parts),(maximum+311)//312)
    if parts<1:raise ValueError('Invalid uniform pool partition count')
    stride=(maximum+parts*312-1)//(parts*312)
    span=stride*312 if parts>1 else maximum
    padded=parts*span
    if padded*state.trials>=2**32:parts=1;span=padded=maximum
    with mx.stream(mx.gpu):
        starts=jump_uniform_blocks(state,parts,stride)if parts>1 else state
        counts=mx.minimum(mx.maximum(counts.astype(mx.int64)[:,None]-mx.arange(parts)*span,0),span).astype(mx.uint32).reshape(-1)
        values,_,_,words,positions=_uniform_kernel(True)(
            inputs=[starts.words,starts.position,counts],
            template=[('MAX_COUNT',span),('PARTS',parts),('SNAP_WIDTH',width),('SNAP_STEPS',frames)],
            grid=(320*starts.trials,1,1),threadgroup=(320,1,1),
            output_shapes=[(padded*state.trials,),(624*starts.trials,),(starts.trials,),
                           (624*frames*state.trials,),(frames*state.trials,)],
            output_dtypes=[mx.uint64,mx.uint32,mx.uint32,mx.uint32,mx.uint32],
            **({'init_value':0}if limited else{}))
        values=values.reshape(state.trials,padded)[:,:maximum].reshape(state.trials,frames,width)
        return UniformPool(mx.contiguous(values.transpose(1,0,2)),
            words.reshape(state.trials,frames,624),positions.reshape(state.trials,frames),
            state.trials,state.has_gauss,state.gauss,width,frames)
