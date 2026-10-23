"""Bind native RNG leaves into the existing stock-compiler arithmetic graph.

Only arithmetic is replaced. Ordered draws, error handling and neuron masks
remain in the existing compiled stage. The unchanged reference codeobject,
not a reconstructed expression or a special-case sum rewrite, supplies the
actual contraction and reassociation points.
"""
from dataclasses import replace
from pathlib import Path
import re
import numpy as np
from .expression_profile_mir import Graph,Word,ProfileError
from .binary64 import literal


_DRAW=re.compile(r'const (?:int|ulong) (\w+)=(compiled_rand|compiled_normal|compiled_poisson)\(')
_KINDS={'compiled_rand':'rand','compiled_normal':'randn','compiled_poisson':'poisson'}


def _loops(g):
    predecessors={b:{p for p in g.reachable if b in g.successors[p]}for b in g.reachable}
    entry=min(g.blocks);dom={b:({entry}if b==entry else set(g.reachable))for b in g.reachable}
    changed=True
    while changed:
        changed=False
        for b in g.reachable-{entry}:
            new={b}|set.intersection(*(dom[p]for p in predecessors[b]))
            if new!=dom[b]:dom[b]=new;changed=True
    loops=[]
    for source in g.reachable:
        for header in g.successors[source]:
            if header not in dom[source]:continue
            loop={header};stack=[]if source==header else[source]
            while stack:
                p=stack.pop()
                if p in loop:continue
                loop.add(p);stack.extend(predecessors[p]-loop)
            if loop<g.region:loops.append((header,loop))
    return loops


def _cpp_leaves(g):
    loops=_loops(g);inside=set().union(*(part for _,part in loops))if loops else set()
    point={id(ins):(b,i)for b,items in g.blocks.items()for i,ins in enumerate(items)}
    def ancestors(reg,seen=None):
        seen=set()if seen is None else seen
        if reg in seen:return seen
        seen.add(reg);ins=g.nodes.get(reg)
        if ins:
            for r in ins.refs:ancestors(r,seen)
        return seen
    records=[]
    for block,items in g.blocks.items():
        if block not in g.region:continue
        for i,ins in enumerate(items):
            if ins.op=='BL'and'RandomGenerator5randnEv'in ins.args:
                result=next((x for x in items[i+1:]if x.op=='COPY'and x.refs==('unknown:$d0',)),None)
                if result:records.append(('randn',(result.defs[0],),(block,i)))
            elif ins.op=='BL'and re.search(r'_poisson(?:_mult|_ptrs)?E?di\b',ins.args):
                result=next((x for x in items[i+1:]if x.op=='COPY'and x.refs==('unknown:$w0',)),None)
                if result:records.append(('poisson',(result.defs[0],),(block,i)))
    uniform_groups={};g.caller_uniform_signs={}
    def uncopy(reg):
        seen=set()
        while reg not in seen and reg in g.nodes and g.nodes[reg].op=='COPY'and len(g.nodes[reg].refs)==1:
            seen.add(reg);reg=g.nodes[reg].refs[0]
        return reg
    for reg,ins in g.nodes.items():
        if ins.op not in('FMADDDrrr','FMSUBDrrr','FNMADDDrrr','FNMSUBDrrr')or ins.block in inside or ins.block not in g.region:continue
        scale=g.constant(ins.refs[1])
        if scale is None or scale.raw is None or (scale.raw&((1<<63)-1))!=int(np.asarray(2.**26,np.float64).view(np.uint64)):continue
        # Fast-math can fold a subtracting consumer into the exact53-bit
        # mantissa construction. Bind its sign as well as its magnitude;
        # the later compiler arithmetic remains unchanged.
        high=(-1 if scale.raw>>63 else 1)*(-1 if ins.op in('FMSUBDrrr','FNMADDDrrr')else 1)
        low=-1 if ins.op in('FNMADDDrrr','FNMSUBDrrr')else 1
        if high!=low:continue
        conversions=[g.nodes.get(ins.refs[j])for j in(0,2)]
        if not all(x and x.op.startswith('UCVTFUW')for x in conversions):continue
        deps=set().union(*(ancestors(x.refs[0])for x in conversions))
        producers=[g.nodes[r]for r in deps if r in g.nodes]
        if not any('_random_generators'in x.raw for x in producers):continue
        if sum(x.op.startswith('EORW')for x in producers)<3:continue
        loads=[point[id(x)]for x in producers if x.op.startswith('LDRW')and x.block in g.region-inside]
        if not loads:raise ProfileError('Missing MT word producers for uniform leaf')
        identity=tuple(uncopy(x.refs[0])for x in conversions)
        g.caller_uniform_signs[reg]=high
        if identity in uniform_groups:
            slot=uniform_groups[identity];kind,registers,where=records[slot]
            records[slot]=(kind,registers+(reg,),min(where,min(loads)))
        else:
            uniform_groups[identity]=len(records);records.append(('rand',(reg,),min(loads)))
    for header,loop in loops:
        for ins in g.blocks[header]:
            if ins.op!='PHI'or not ins.defs:continue
            incoming=re.findall(r'(%\d+), %bb\.(\d+)',ins.args)
            initial=[g.constant(r)for r,b in incoming if int(b)not in loop]
            if not initial or not all(v and v.width==32 and v.raw==0xffffffff for v in initial):continue
            increments=[g.nodes[r]for r,b in incoming if int(b)in loop and r in g.nodes]
            if len(increments)!=1:continue
            update=increments[0]
            while update.op=='COPY'and len(update.refs)==1 and update.refs[0]in g.nodes:update=g.nodes[update.refs[0]]
            if update.op!='ADDWri'or update.refs!=(ins.defs[0],)or int(update.args.split(',')[1])!=1:continue
            if not any(x.op.startswith('FCMP')for b in loop for x in g.blocks[b]):continue
            records.append(('poisson',(update.defs[0],),(header,0)))
    return sorted(records,key=lambda r:r[2])


def _cython_leaves(g,text):
    """Follow actual Cython inline-return debug identities to each callsite."""
    scopes=dict(re.findall(r'!(\d+) = distinct !DISubprogram\(name: "([^"\n]+)"',text))
    main={scope for scope,name in scopes.items()if name.startswith('__pyx_pf_')and name.endswith('_main')}
    if len(main)!=1:raise ProfileError('Missing Cython caller source identity')
    start=int(re.search(r'!'+next(iter(main))+r' = distinct !DISubprogram\([^\n]*?line: (\d+)',text)[1])
    calls=[]
    for number,line in enumerate(Path(g.metadata['source']).read_text().splitlines(),1):
        if number<start:continue
        match=re.match(r'\s*__pyx_t_\d+ = .*?__(randn|rand|poisson)\(',line)
        if match:calls.append((number,match[1]))
    if not calls:raise ProfileError('Missing Cython primitive callsites')
    call_kinds=dict(calls);returns={line:set()for line,_ in calls}
    locations={};location_scopes={}
    for identifier,body in re.findall(r'!(\d+) = (?:distinct )?!DILocation\(([^\n]+)\)',text):
        line=re.search(r'\bline: (\d+)',body);parent=re.search(r'\binlinedAt: !(\d+)',body)
        if line:locations[identifier]=(int(line[1]),parent[1]if parent else None)
        scope=re.search(r'\bscope: !(\d+)',body)
        if scope:location_scopes[identifier]=scope[1]
    lexical=dict(re.findall(r'!(\d+) = (?:distinct )?!DILexicalBlock(?:File)?\(scope: !(\d+)',text))
    variables={identifier:(name,scope)for identifier,name,scope in re.findall(
        r'!(\d+) = !DILocalVariable\(name: "([^"\n]+)"[^\n]*?scope: !(\d+)',text)}
    def callsite(raw):
        match=re.search(r'debug-location !(\d+)',raw);location=match[1]if match else None;seen=set()
        while location in locations and location not in seen:
            seen.add(location);line,location=locations[location]
            if line in call_kinds:return line
        return None
    def helper_kind(raw):
        match=re.search(r'debug-location !(\d+)',raw);location=match[1]if match else None;seen=set();result=None
        while location in locations and location not in seen:
            seen.add(location);scope=location_scopes.get(location);parents=set()
            while scope in lexical and scope not in parents:parents.add(scope);scope=lexical[scope]
            function=scopes.get(scope,'')
            kind=next((k for k in('randn','rand','poisson')if re.search(r'__'+k+r'(?:_|$)',function)),None)
            if kind:result=kind
            location=locations[location][1]
        return result
    documents=[d for d in text.split('\n---\n')if'\nbody:'in d and'1main'in d.split('body:',1)[0]and'.cold.'not in d.split('body:',1)[0]]
    if len(documents)!=1:raise ProfileError('Missing one Cython main body')
    g.caller_error_sites={};g.caller_error_kinds={};block=None
    for raw in documents[0].split('body:',1)[1].splitlines():
        label=re.match(r'  bb\.(\d+) ',raw)
        if label:block=int(label[1])
        match=re.search(r'DBG_VALUE ([^,]+), \$noreg, !(\d+), !DIExpression\(\)',raw)
        if not match:continue
        reg,identifier=match.groups();name,scope=variables.get(identifier,('',''))
        if name!='__pyx_r':continue
        function=scopes.get(scope,'');kind=next((k for k in('randn','rand','poisson')if re.search(r'__'+k+r'(?:_|$)',function)),None)
        site=callsite(raw)
        if site in call_kinds and kind==call_kinds[site]:
            if reg in g.nodes:returns[site].add(reg)
            elif reg in('-1','double -1.000000e+00'):g.caller_error_sites[block]=site
        elif site is None and kind:
            value=g.constant(reg)if reg in g.nodes else None
            if reg in('-1','double -1.000000e+00')or(value is not None and value.raw in(0xffffffff,0xbff0000000000000)):
                g.caller_error_kinds[block]=kind
    for block,items in g.blocks.items():
        if block not in g.region:continue
        for i,ins in enumerate(items):
            if ins.op!='BL':continue
            site=callsite(ins.raw)
            if site not in call_kinds:continue
            kind=call_kinds[site]
            if not re.search(r'__'+kind+r'(?:i|di)\b',ins.args):continue
            register='unknown:$w0'if kind=='poisson'else'unknown:$d0'
            result=next((x for x in items[i+1:]if x.op=='COPY'and x.refs==(register,)),None)
            if result:returns[site].add(result.defs[0])
    g.caller_source_sites={id(ins):callsite(ins.raw)for items in g.blocks.values()for ins in items}
    g.caller_helper_kinds={id(ins):helper_kind(ins.raw)for items in g.blocks.values()for ins in items}
    g.caller_buffer_loads={}
    g.caller_zero_return_sites={}
    source=Path(g.metadata['source']).read_text().splitlines()
    for block,items in g.blocks.items():
        if block not in g.region:continue
        for ins in items:
            kind=g.caller_helper_kinds[id(ins)]
            if kind=='poisson' and ins.op=='FCMPDri':
                match=re.search(r'debug-location !(\d+)',ins.raw)
                location=match[1] if match else None
                line=locations.get(location,(0,None))[0]
                fragment=re.sub(r'/\*.*?\*/','', '\n'.join(source[max(0,line-5):line+30]),flags=re.S)
                # Bind only the actual generated sampler's lam==0 return.
                # A user-written zero branch has a different debug scope.
                proven=re.search(r'(__pyx_t_\d+) = \(__pyx_v_lam == 0\.0\);\s*if \(\1\)\s*\{\s*\{?\s*__pyx_r = 0;',fragment)
                if proven:
                    site=callsite(ins.raw)
                    following=items[items.index(ins)+1:]
                    branch=next((part for part in following if part.op=='Bcc'),None)
                    if branch and re.match(r'0, %bb\.(\d+)',branch.args):
                        destination=int(re.match(r'0, %bb\.(\d+)',branch.args)[1])
                        g.caller_zero_return_sites[block,destination]=site
            if kind not in('rand','randn')or not ins.op.startswith('LDRD')or len(ins.defs)!=1:continue
            match=re.search(r'debug-location !(\d+)',ins.raw)
            if not match or match[1]not in locations:continue
            line=locations[match[1]][0]
            if not 1<=line<=len(source)or not re.search(r'__pyx_(?:v_val|r)\s*=\s*\(?__pyx_v_buffer\[',source[line-1]):continue
            site=callsite(ins.raw)
            if site in call_kinds and call_kinds[site]==kind:returns[site].add(ins.defs[0])
            elif site is None:g.caller_buffer_loads.setdefault(kind,set()).add(ins.defs[0])
    return[(kind,tuple(sorted(returns[line])),(line,0))for line,kind in calls]


def _threshold_condition(g,targets=None):
    """Use the existing graph's exact flags at the event-store branches."""
    loops=[part for _,part in _loops(g)]
    regions=[part for part in loops if not any(part<other for other in loops)]
    helpers=set().union(*regions)if regions else set()
    events=targets if targets is not None else {b for b in g.region if any(g.store_name(ins)=='_spikespace'for ins in g.blocks[b])}
    if not events or (targets is None and len(events)!=1):raise ProfileError('Cannot identify threshold destination')
    def conjunction(a,b):
        if a=='false'or b=='false':return'false'
        if a=='true':return b
        if b=='true':return a
        return '('+a+')&&('+b+')'
    def negate(value):return'false'if value=='true'else'true'if value=='false'else'!('+value+')'
    origins_cache={}
    def origins(reg,seen=None):
        # Cython's inlined buffer refill manages Python objects. Their ABI
        # return pointers are sampler internals, never model predicates.
        if isinstance(g.bindings.get(reg),Word):return set()if g.bindings[reg].raw is not None else{'model'}
        if reg in origins_cache:return origins_cache[reg]
        seen=set()if seen is None else seen
        if reg in seen:return{'model'}
        ins=g.nodes.get(reg)
        if ins is None:return{'unknown'}if reg.startswith(('unknown:','$'))and reg not in('$wzr','$xzr')else set()
        if (ins.op=='COPY'and ins.refs==('unknown:$x0',)
                and (getattr(g,'caller_source_sites',{}).get(id(ins))is not None
                     or getattr(g,'caller_helper_kinds',{}).get(id(ins))is not None)):return{'sampler'}
        if g.root(reg)in g.array_info:return{'model'}
        result=set().union(*(origins(r,seen|{reg})for r in ins.refs))if ins.refs else set()
        origins_cache[reg]=result
        return result
    def edge(source,target):
        prefix='true';sequence=[]
        for ins in g.blocks[source]:
            if ins.op.startswith(('FCMP','SUBS','ADDS')):sequence=[ins]
            elif ins.op.startswith(('CCMP','CCMN','FCCMP')):sequence.append(ins)
            if not ins.op.startswith(('Bcc','CBZ','CBNZ','TBZ','TBNZ'))and ins.op!='B':continue
            # Registered sampler internals are opaque leaves. Their existing
            # native implementations own acceptance, refill and errors.
            tagged=sequence[-1]if ins.op=='Bcc'and sequence else ins
            if (source in helpers or source in getattr(g,'caller_boxed_regions',set()) or getattr(g,'caller_source_sites',{}).get(id(tagged))is not None
                    or getattr(g,'caller_helper_kinds',{}).get(id(tagged))is not None):return'true'
            inputs=sequence if ins.op=='Bcc'else[ins]
            dependencies=set().union(*(origins(r)for item in inputs for r in item.refs))
            if dependencies=={'sampler'}:return'true'
            flags=None
            if ins.op=='Bcc':
                for flag in sequence:flags=g.flag_values(flag,g.symbolic,flags)
            condition,destination=g.branch(ins,flags,g.symbolic)
            if condition is None:raise ProfileError('Unknown threshold branch '+ins.raw)
            term='true'if condition is True else'false'if condition is False else condition
            if destination==target:return conjunction(prefix,term)
            prefix=conjunction(prefix,negate(term))
            if condition is True:return'false'
        order=list(g.blocks);index=order.index(source)
        return prefix if index+1<len(order)and order[index+1]==target else'false'
    cache={}
    def path(block,seen):
        if block==g.header:return'true'
        if block in cache:return cache[block]
        if block in seen:raise ProfileError('Unexpected loop in threshold predicate')
        # A rejection/refill loop is an opaque native sampler, but its
        # incoming model guard still decides whether that sampler executes.
        # Collapsing its internal edges must retain those outside prefixes.
        region=next((part for part in regions if block in part),None)
        if region is not None:
            terms=[]
            for previous in g.region-region:
                for target in g.successors[previous]&region:
                    part=edge(previous,target)
                    if part!='false':terms.append(conjunction(path(previous,seen|region),part))
            if not terms:raise ProfileError('Missing sampler entry in threshold predicate')
            value='true'if'true'in terms else'('+'||'.join('('+t+')'for t in dict.fromkeys(terms))+')'
            for member in region:cache[member]=value
            return value
        terms=[]
        for previous in g.region:
            if block not in g.successors[previous]:continue
            part=edge(previous,block)
            if part=='false':continue
            terms.append(conjunction(path(previous,seen|{block}),part))
        if not terms:
            if any(block in g.successors[p]for p in g.region):return 'false'
            raise ProfileError('Missing threshold predecessor')
        cache[block]='true'if'true'in terms else'('+'||'.join('('+t+')'for t in dict.fromkeys(terms))+')'
        return cache[block]
    terms=[path(b,set())for b in sorted(events)]
    condition='true'if 'true'in terms else '('+'||'.join('('+t+')'for t in dict.fromkeys(terms))+')'
    if targets is None and 'not_refractory'in g.array_info:condition=conjunction('('+g.field('not_refractory').code+'!=0L)',condition)
    return condition


def _bind(g,records,reference_draws):
    if [kind for kind,_,_ in records]!=[kind for _,kind in reference_draws]:
        raise ProfileError('Native/reference primitive identity or order differs: '
            +str([kind for kind,_,_ in records])+' versus '+str([kind for _,kind in reference_draws]))
    g.caller_error_words={};g.caller_draw_entries={};g.caller_zero_return_words={}
    for (kind,registers,point),(name,_)in zip(records,reference_draws):
        if g.metadata['target']=='cython':
            blocks={b for b in g.region if any(getattr(g,'caller_source_sites',{}).get(id(ins))==point[0]for ins in g.blocks[b])}
            entries={b for b in blocks if b==g.header or any(b in g.successors[p]for p in g.region-blocks)}
        else:entries={point[0]}
        g.caller_draw_entries[name]=entries
        code='d_mul('+name+','+literal(2.**53)+')'if kind=='rand'and g.metadata['target']=='cpp'else name
        value=Word('int'if kind=='poisson'else'fp',code,width=32 if kind=='poisson'else 64)
        for register in registers:
            g.bindings[register]=(Word(value.kind,'d_neg('+value.code+')',width=value.width)
                if kind=='rand'and getattr(g,'caller_uniform_signs',{}).get(register)==-1 else value)
        for edge,site in getattr(g,'caller_zero_return_sites',{}).items():
            if site==point[0]:g.caller_zero_return_words[edge]=value
        for block,site in getattr(g,'caller_error_sites',{}).items():
            if site==point[0]:g.caller_error_words[block]=value
    g.bindings.update(getattr(g,'caller_buffer_words',{}))
    g.caller_error_words.update(getattr(g,'caller_merged_error_words',{}))
    g._constant_cache.clear();g._symbolic_cache.clear()


def _merged_reference(g,records,draws,stage):
    """Bind one compiler-hoisted sample to provably exclusive native arms."""
    if (g.metadata.get('stage')!='threshold'or len(stage.reference_call_guards)!=len(draws)
            or set(kind for kind,_,_ in records)!=set(kind for _,kind in draws)):return draws
    from .expression_guard import exclusive,stable
    if not all(stable(guard)for guard in stage.reference_call_guards):return draws
    groups={kind:[i for i,(_,candidate)in enumerate(draws)if candidate==kind]for kind,_,_ in records}
    for indices in groups.values():
        for position,a in enumerate(indices):
            for b in indices[position+1:]:
                if not exclusive(stage.reference_call_guards[a],stage.reference_call_guards[b]):return draws
    prefix='_caller_taken_'
    while prefix in stage.vector or any(name.startswith(prefix)for name in g.metadata.get('reserved_names',())):prefix+='_'
    flags={name:prefix+name for name,_ in draws};g.caller_draw_flags=flags
    values={}
    for kind in groups:
        value='0'if kind=='poisson'else'0ul'
        for index in reversed(groups[kind]):
            name=draws[index][0];value='('+flags[name]+'?'+name+':'+value+')'
        values[kind]=value
    g.caller_buffer_words={reg:Word('fp',values[kind])for kind,registers in getattr(g,'caller_buffer_loads',{}).items()for reg in registers}
    g.caller_merged_error_words={block:Word('int'if kind=='poisson'else'fp',values[kind],width=32 if kind=='poisson'else 64)
        for block,kind in getattr(g,'caller_error_kinds',{}).items()if kind in values}
    return[(values[kind],kind)for kind,_,_ in records]


def _rewrite_threshold_draws(g,source):
    """Evaluate each ordered draw at its stock caller control-flow prefix.

    Thresholds are pure before their final predicate store. Drop only their
    temporary boolean scaffolding; draw arguments retain their typed values.
    """
    draws=list(_DRAW.finditer(source))
    if not draws or not any(source[:m.start()].count('{')!=source[:m.start()].count('}')for m in draws):return source
    if getattr(g,'caller_draw_flags',None):return source
    store=re.search(r'(?:if\(!_compiled_rng.error\)\{)?_cond\s*=',source)
    if store is None:raise ProfileError('Missing threshold predicate store')
    prefix=source[:store.start()];suffix=source[store.start():]
    definitions=set(re.findall(r'\b(?:bool|int|ulong) (\w+)\s*=',prefix))
    available=set();lines=[]
    for match in draws:
        end=source.find(';',match.start())
        statement=source[match.start():end+1]
        declaration=re.fullmatch(r'const (int|ulong) (\w+)=(.*);',statement)
        if declaration is None:raise ProfileError('Unknown ordered threshold draw')
        dtype,name,call=declaration.groups()
        if set(re.findall(r'\b\w+\b',call))&definitions-available:
            raise ProfileError('Threshold draw argument requires a boolean temporary')
        entries=getattr(g,'caller_draw_entries',{}).get(name)
        if not entries:raise ProfileError('Missing reference draw control-flow entry')
        guard=_threshold_condition(g,entries)
        dependencies=set(re.findall(r'\b\w+\b',guard))&definitions
        if not dependencies<=available:raise ProfileError('Reference draw guard uses a later draw')
        lines.extend((dtype+' '+name+'=0;', 'if('+guard+'){'+name+'='+call+';}'))
        available.add(name)
    # Every discarded assignment must belong to the temporary boolean graph.
    remainder=re.sub(r'const (?:int|ulong) \w+=compiled_(?:rand|normal|poisson)\([^;]*;','',prefix)
    assigned=set(re.findall(r'(?<![=!<>])\b(\w+)\s*=(?!=)',remainder))
    if not assigned<=definitions:raise ProfileError('Model mutation before threshold predicate')
    return '\n'.join(lines)+'\n'+suffix


def _rewrite(g,source,writes,*,threshold=False,text=None):
    metadata=g.metadata
    originals={}
    for name,info in g.array_info.items():
        if name in g.constants:continue
        if name=='_spikespace':g.initial[name]='i';continue
        variable='_caller_original_'+name
        while variable in metadata.get('reserved_names',g.array_info)or re.search(r'\b'+re.escape(variable)+r'\b',source):variable+='_'
        g.initial[name]=variable
        metal=g.metadata.get('metal_names',{}).get(name,name)
        originals[variable]=('ulong'if info['dtype']=='float64'else'uint'if info['dtype']=='float32'else'long')+' '+variable+'='+metal+';'
    if text is not None:
        from .expression_profile_boxed import bind
        bind(g,text)
    # Checked integer expressions already use the selected Cython frontend.
    # Bind their final integer SSA outputs, retaining the caller's floating
    # graph (including contraction/reassociation) around those exact values.
    integer_outputs={}
    for name in writes:
        info=g.array_info.get(name,{})
        if not info.get('dtype','').startswith(('int','uint')):continue
        metal=metadata.get('metal_names',{}).get(name,name)
        assignments=list(re.finditer(r'(?<!\w)'+re.escape(metal)+r'\s*=\s*([^;]+);',source))
        if assignments and '_rng_integer_'in assignments[-1][1]:
            integer_outputs[name]=metal
    for block in g.region:
        for ins in g.blocks[block]:
            name=g.store_name(ins)
            if name in integer_outputs:
                metal=integer_outputs[name]
                g.bindings[ins.refs[0]]=Word('int',metal,width=np.dtype(g.array_info[name]['dtype']).itemsize*8)
    if integer_outputs:g._constant_cache.clear();g._symbolic_cache.clear()
    sinks={}
    for block in g.region:
        for ins in g.blocks[block]:
            name=g.store_name(ins)
            if name not in writes or name in integer_outputs:continue
            value=g.symbolic(ins.refs[0])
            if value is None:raise ProfileError('Unknown reference caller store '+name)
            sinks.setdefault(name,set()).add(value.code)
    if any(len(values)!=1 for values in sinks.values()):raise ProfileError('Conditional caller stores require distinct arithmetic paths')
    if threshold:
        source=_rewrite_threshold_draws(g,source)
        scoped=any(source[:match.start()].count('{')!=source[:match.start()].count('}')for match in _DRAW.finditer(source))
        if scoped or getattr(g,'caller_draw_flags',None):
            # Keep every draw at its existing conditional execution point.
            # Only its result binding moves outside the lexical branch so
            # the final reference predicate can refer to all possible arms.
            declarations=[]
            pattern=r'\bconst (int|ulong) (\w+)=(compiled_(?:rand|normal|poisson)\()'
            def hoist(match):
                dtype,name,call=match.groups();declarations.append(dtype+' '+name+'=0;')
                flag=getattr(g,'caller_draw_flags',{}).get(name)
                if flag:declarations.append('bool '+flag+'=false;')
                return (flag+'=true; 'if flag else'')+name+'='+call
            source=re.sub(pattern,hoist,source)
            source='\n'.join(declarations)+'\n'+source
        sinks['_cond']={_threshold_condition(g)}
    for name,values in sinks.items():
        expression=next(iter(values))
        metal=g.metadata.get('metal_names',{}).get(name,name)
        pattern=r'(?<!\w)'+re.escape(metal)+r'\s*=\s*[^;]+;'
        assignments=list(re.finditer(pattern,source))
        if not assignments:raise ProfileError('Cannot identify native caller store '+name)
        # The compiled loop writes the final local value to the state array.
        # Earlier native local assignments can feed intervening sampler calls;
        # retain them and replace only the final write with the actual DAG.
        last=assignments[-1]
        for integer in integer_outputs.values():
            if not re.search(r'\b'+re.escape(integer)+r'\b',expression):continue
            integer_assignments=list(re.finditer(r'(?<!\w)'+re.escape(integer)+r'\s*=\s*[^;]+;',source))
            if not integer_assignments or integer_assignments[-1].end()>last.start():
                raise ProfileError('Integer caller output is written after its floating consumer')
        source=source[:last.start()]+metal+'='+expression+';'+source[last.end():]
    if not sinks:
        if not writes or set(writes)<=integer_outputs.keys():return source
        raise ProfileError('No reference caller stores')
    used='\n'.join(expression for values in sinks.values()for expression in values)
    captures=[line for name,line in originals.items()if re.search(r'\b'+re.escape(name)+r'\b',used)]
    return '\n'.join(captures)+'\n'+source


def lower(text,metadata,stage):
    """Return the same ordered stage with its reference arithmetic sinks."""
    if stage.may_raise and '_rng_integer_'not in stage.scalar+stage.vector:return stage
    if _DRAW.search(stage.scalar)and not stage.vector.strip():
        scalar=replace(stage,scalar='',vector=stage.scalar,reference_call_aliases=stage.scalar_reference_call_aliases)
        lowered=lower(text,metadata,scalar)
        return replace(stage,scalar=lowered.vector)
    if stage.scalar and stage.vector.strip():
        # Scalar sampler leaves remain in scope for the vector loop. Give
        # their temporaries distinct names before binding both reference
        # phases; stock Clang may carry scalar arithmetic directly into the
        # vector DAG rather than reloading the shared array.
        prefix='_caller_scalar_draw_'
        while prefix in stage.scalar+stage.vector or any(n.startswith(prefix)for n in metadata.get('reserved_names',())):prefix+='_'
        scalar=re.sub(r'\b_compiled_draw_(\d+)\b',lambda m:prefix+m[1],stage.scalar)
        g=Graph(text,metadata);shared=Graph(text,metadata)
        shared.region=shared.reachable if g.unrolled else shared.reachable-g.region
        if metadata['target']=='cpp':
            records=(_cpp_leaves(g)if g.unrolled else sorted(_cpp_leaves(shared)+_cpp_leaves(g),key=lambda r:r[2]))
        else:
            records=_cython_leaves(g,text)
            scalar_records=_cython_leaves(shared,text)
            if [(kind,point)for kind,_,point in records]!=[(kind,point)for kind,_,point in scalar_records]:
                raise ProfileError('Scalar/vector caller source identities disagree')
            records=[(kind,tuple(sorted(set(a)|set(b))),point)
                for(kind,a,point),(_,b,_)in zip(records,scalar_records)]
        phases=[]
        for source,aliases in((scalar,stage.scalar_reference_call_aliases),(stage.vector,stage.reference_call_aliases)):
            draws=[(name,_KINDS[function])for name,function in _DRAW.findall(source)]
            phases.append([draws[i]for i in aliases]if aliases else draws)
        reference=phases[0]+phases[1]
        if g.unrolled and len(records)==len(phases[0])+metadata['neurons']*len(phases[1]):
            reference=phases[0]+phases[1]*metadata['neurons']
        _bind(g,records,reference);_bind(shared,records,reference)
        scalar_writes={name for name in stage.writes if name in g.array_info and g.array_info[name]['scalar']}
        return replace(stage,scalar=_rewrite(shared,scalar,scalar_writes,text=text),
            vector=_rewrite(g,stage.vector,set(stage.writes)-scalar_writes,
                            threshold=metadata.get('stage')=='threshold',text=text))
    draws=[(name,_KINDS[function])for name,function in _DRAW.findall(stage.vector)]
    if not draws:return stage
    g=Graph(text,metadata)
    records=_cpp_leaves(g)if metadata['target']=='cpp'else _cython_leaves(g,text)
    aliases=stage.reference_call_aliases
    reference=([draws[index]for index in aliases]if aliases and len(records)==len(aliases)else draws)
    if [kind for kind,_,_ in records]!=[kind for _,kind in reference]or getattr(g,'caller_buffer_loads',None):
        reference=_merged_reference(g,records,draws,stage)
    _bind(g,records,reference)
    return replace(stage,vector=_rewrite(g,stage.vector,stage.writes,threshold=metadata.get('stage')=='threshold',text=text))
