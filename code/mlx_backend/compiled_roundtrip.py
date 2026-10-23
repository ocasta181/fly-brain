"""Recover floating roundtrips selected after LLVM IR optimization."""
from __future__ import annotations
import re
import subprocess
from .compiled_cache import cache_output
import sysconfig
from collections import Counter
from .compiler import UnsupportedModelError


def floating_roundtrips(policy, cpp, ir, cache, key):
    """Bind each FRINTZ to its source-located integer-to-float conversion.

    An integer result with additional uses can select FCVTZS/SCVTF instead.
    Inspect the complete compiled block under its original flags, not a
    separate cast probe, and require an unambiguous location and width.
    """
    body = re.search(r'define[^\n]*integer_block[^\n]*\{\n(.*?)\n\}', ir, re.S)
    if not body:
        return {}
    code = body[1]
    casts = dict((name, (kind, operand, width)) for name, kind, operand, width in re.findall(
        r'(%[\w.]+) = fptosi (float|double) (%[\w.]+) to (i\d+)', code))
    candidates = []
    for target, width, operand, kind, debug in re.findall(
            r'(%[\w.]+) = sitofp (i\d+) (%[\w.]+) to (float|double), !dbg !(\d+)', code):
        cast = casts.get(operand)
        if cast is None or cast[0] != kind or cast[2] != width:
            continue
        location = re.search(r'^!' + debug + r' = !DILocation\(line: (\d+), column: (\d+)', ir, re.M)
        if location:
            candidates.append((target, kind, cast[1], tuple(map(int, location.groups()))))
    if not candidates:
        return {}
    assembly = cache / (key + '.roundtrips.s')
    with cache_output(assembly) as pending:
        if pending is not None:
            import brian2 as b
            import numpy as np
            command = [*policy.compiler, *policy.python_flags, *policy.flags, *policy.compiler_options,
                       '-I'+sysconfig.get_path('include'), '-I'+np.get_include(),
                       '-I'+str(__import__('pathlib').Path(b.__file__).parent/'synapses'),
                       '-fno-discard-value-names', '-S', str(cpp), '-o', str(pending)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise UnsupportedModelError('CPU floating roundtrip inspection failed: '+result.stderr)
    function = re.search(r'^\S*integer_block\S*:[^\n]*\n(.*?)\.cfi_endproc', assembly.read_text(), re.M|re.S)
    if not function:
        return {}
    sites = Counter()
    site = None
    for raw in function[1].splitlines():
        if raw.lstrip().startswith('.loc'):
            original = re.findall(re.escape(cpp.name)+r':(\d+):(\d+)', raw)
            direct = re.match(r'\s*\.loc\s+1\s+(\d+)\s+(\d+)', raw)
            site = tuple(map(int, original[-1])) if original else tuple(map(int, direct.groups())) if direct else None
        match = re.match(r'\s*frintz\s+([sd])\d+,\s*\1\d+', raw)
        if match and site:
            sites[('float' if match[1]=='s' else 'double', site)] += 1
    counts = Counter((kind,site) for _,kind,_,site in candidates)
    return {target:(kind,operand) for target,kind,operand,site in candidates
            if sites[(kind,site)] == counts[(kind,site)] == 1}
