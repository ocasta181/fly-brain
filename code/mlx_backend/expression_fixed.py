"""Fixed-size model draws attached to the canonical configured engine.

The compiler supplies the typed threshold and the existing engine still owns
packed state, scheduling, arrivals, errors and recording. This adapter owns
only immutable native MT/buffer state and the prefetched draw input.
"""
import ast
from dataclasses import dataclass,field,replace
from types import MappingProxyType


@dataclass(frozen=True,eq=False)
class FixedRandomProgram:
    function: str
    result: str
    target: str
    variables: object=field(default_factory=lambda:MappingProxyType({}))
    checked: bool=False


def checked_threshold(model,site,*,shared):
    """A nonthrowing complete-population draw can precede checked phases.

    More complex thresholds still need the ordered random-stage lowering:
    their own error may stop draws partway through the population.
    """
    if (not isinstance(site,FixedRandomProgram) or shared or site.target!='cython'
            or model.execution_phases!=('groups','thresholds','synapses','resets')):
        return False
    node=ast.parse(model.eq_th,mode='eval').body
    def literal(value):
        if isinstance(value,ast.UnaryOp)and isinstance(value.op,(ast.UAdd,ast.USub)):
            value=value.operand
        return isinstance(value,ast.Constant)and type(value.value)in(int,float)and abs(value.value)<2**53
    def draw(value):
        return (isinstance(value,ast.Call)and isinstance(value.func,ast.Name)
            and value.func.id==site.function and not value.args and not value.keywords)
    return (isinstance(node,ast.Compare)and len(node.ops)==len(node.comparators)==1
        and isinstance(node.ops[0],(ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.Eq,ast.NotEq))
        and ((draw(node.left)and literal(node.comparators[0]))
             or(literal(node.left)and draw(node.comparators[0]))))


@dataclass(frozen=True)
class GeneralRandomPlan:
    target: str


def plan(model,equations,name):
    from .compiler import UnsupportedModelError
    # Reset is a statement list; only the threshold can use this fixed path.
    other=[model.eq_rst]+[equation.expr.code for equation in equations.values()if equation.expr is not None]
    random_names={'rand','randn','poisson'}
    def calls(node):
        return [part for part in ast.walk(node)if isinstance(part,ast.Call)
            and isinstance(part.func,ast.Name)and part.func.id in random_names]
    threshold=ast.parse(model.eq_th,mode='eval').body;draws=calls(threshold)
    elsewhere=[]
    for source in other:
        # Brian reset statements use semicolons and Python-compatible syntax.
        elsewhere.extend(calls(ast.parse(source,mode='exec')))
    if not draws and not elsewhere:return None
    if len(draws)!=1 or elsewhere or draws[0].func.id not in('rand','randn'):
        return GeneralRandomPlan(model.random_target)
    draw=draws[0]
    if draw.args or draw.keywords:raise ValueError('Invalid built-in random call')
    def unconditional(node,guarded=False):
        if node is draw:return not guarded
        if isinstance(node,ast.BoolOp):
            return any(unconditional(child,guarded or position>0)for position,child in enumerate(node.values))
        if isinstance(node,ast.IfExp):
            return (unconditional(node.test,guarded)or unconditional(node.body,True)
                or unconditional(node.orelse,True))
        if isinstance(node,ast.Compare):
            return any(unconditional(child,guarded or position>1)
                for position,child in enumerate((node.left,*node.comparators)))
        return any(unconditional(child,guarded)for child in ast.iter_child_nodes(node))
    if not unconditional(threshold):
        return GeneralRandomPlan(model.random_target)
    return FixedRandomProgram(draw.func.id,name,model.random_target)


def max_frames(p,trials):
    from .expression_runtime import POOL_VALUE_BYTES,POOL_NORMAL_BYTES,POOL_CHECKPOINT_BYTES,POOL_MAX_FRAMES
    width=p.neurons;site=p.program.random_program
    if not isinstance(site,FixedRandomProgram):return 32
    if site.checked:return 32
    frames=min(POOL_MAX_FRAMES,POOL_VALUE_BYTES//(trials*width*8),
        POOL_CHECKPOINT_BYTES//(trials*624*4),(2**32-1)//(trials*width))
    if site.function=='randn':frames=min(frames,POOL_NORMAL_BYTES//(trials*width*128))
    return max(1,frames)


def advance(p,state,events,steps,length,*,event_begin,event_steps,event_times,timed_calls,
            fused_idle,native_poisson,stop_before_synapses,precision_epoch_begin):
    import mlx.core as mx
    from .configured import advance as model_advance
    from .expression_runtime import uniform_lookahead,cython_lookahead,RandomState
    from .expression_rng import uniforms,normals
    random=state.random;site=p.program.random_program
    if random is None:raise ValueError('Random model requires explicit native stream state')
    if site.checked:
        from .expression_general import _advance_steps
        return _advance_steps(p,state,events,steps,length,event_begin=event_begin,
            event_steps=event_steps,event_times=event_times,timed_calls=timed_calls,
            fused_idle=fused_idle,native_poisson=native_poisson,
            stop_before_synapses=stop_before_synapses,precision_epoch_begin=precision_epoch_begin)
    if native_poisson is not None:
        raise NotImplementedError('Shared native PoissonInput/model-expression stream integration is pending')
    width=p.neurons
    if random.buffers is not None:
        draws,stream,buffers,cache=cython_lookahead(p.program,random.stream,random.buffers,random.pool,
            width,state.step,steps,length,function=site.function,errors=random.errors)
    else:
        draws,stream,cache=uniform_lookahead(p.program,random.stream,random.pool,width,
            state.step,steps,length,function=site.function,target=site.target)
        buffers=None
    if draws is None:
        count=width*length
        if count*random.stream.trials>=2**32:raise OverflowError('Expression draw window exceeds native indexing')
        if random.buffers is not None:
            from .expression_cython import buffered_draws
            packed,stream,buffers=buffered_draws(random.stream,random.buffers,site.function,count)
        elif site.function=='rand':packed,stream=uniforms(random.stream,count)
        else:packed,stream=normals(random.stream,count,target=site.target)
        draws=mx.contiguous(packed.reshape(random.stream.trials,length,width).transpose(1,0,2))
        cache=None
    updated=RandomState(stream,random.temporaries,random.errors,cache,buffers)
    result,bits=model_advance(p,state,events,steps,length,event_begin=event_begin,event_steps=event_steps,
        event_times=event_times,timed_calls=timed_calls,fused_idle=fused_idle,native_poisson=None,
        stop_before_synapses=stop_before_synapses,precision_epoch_begin=precision_epoch_begin,
        _expression_draws=draws)
    return replace(result,random=updated),bits
